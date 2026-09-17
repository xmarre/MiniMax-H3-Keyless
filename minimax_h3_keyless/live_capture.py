from __future__ import annotations

import hashlib
import math
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .activation_capture import PilotActivationCapture
from .capture_io import CaptureBundleProvenance, CaptureBundleWriteResult, write_captured_pilot_bundle
from .contracts import TARGET_MODEL_REVISION, TEACHER_SHA256
from .native_capture_policy import require_plain_native_capture_options
from .pilot_campaign import PILOT_BLOCKS, load_json_manifest, validate_pilot_dataset_manifest
from .pilot_inputs import CANONICAL_STAGE_A_COVERAGE_TAGS
from .stage_a_execution_binding import (
    WORKFLOW_CONTEXT_KEY,
    canonical_stage_a_workflow_prompt_sha256,
    require_manifest_case_workflow_prompt_sha256,
    validate_stage_a_manifest_assets_for_workflow,
)
from .teacher import validate_loaded_native_teacher_model


STAGE_A_TEACHER_ATTACHMENT_KEY = "minimax_h3_keyless_stage_a_teacher_v1"
STAGE_A_CAPTURE_WRAPPER_KEY = "minimax_h3_keyless_stage_a_capture_v1"
_STAGE_A_TEACHER_ATTACHMENT_API = 1
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class StageATeacherAttachment:
    api: int = _STAGE_A_TEACHER_ATTACHMENT_API
    teacher_model_revision: str = TARGET_MODEL_REVISION
    teacher_model_sha256: str = TEACHER_SHA256

    def __post_init__(self) -> None:
        if self.api != _STAGE_A_TEACHER_ATTACHMENT_API:
            raise ValueError("unsupported Stage-A teacher attachment API")
        if self.teacher_model_revision != TARGET_MODEL_REVISION:
            raise ValueError("Stage-A teacher attachment revision is not the pinned BF16 teacher")
        if self.teacher_model_sha256.lower() != TEACHER_SHA256:
            raise ValueError("Stage-A teacher attachment hash is not the pinned BF16 teacher")


@dataclass(frozen=True)
class StageACaptureSpec:
    dataset_manifest_sha256: str
    source_case_id: str
    split: str
    modality_label: str
    target_sigma: float
    output_path: str
    max_capture_bytes: int
    workflow_prompt_sha256: str
    sigma_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        for name, value in (
            ("dataset manifest", self.dataset_manifest_sha256),
            ("workflow prompt", self.workflow_prompt_sha256),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"Stage-A {name} identity must be a SHA-256")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(f"Stage-A {name} identity must be a SHA-256") from exc
        if not self.source_case_id.strip():
            raise ValueError("Stage-A source case_id must be non-empty")
        if self.split not in ("train", "holdout"):
            raise ValueError("Stage-A capture split must be train or holdout")
        if not self.modality_label.strip():
            raise ValueError("Stage-A modality label must be non-empty")
        if not math.isfinite(float(self.target_sigma)) or not 0.0 <= float(self.target_sigma) <= 1.0:
            raise ValueError("Stage-A target sigma must be finite and within [0,1]")
        if self.max_capture_bytes <= 0:
            raise ValueError("Stage-A capture byte budget must be positive")
        tolerance = float(self.sigma_tolerance)
        if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 1e-4:
            raise ValueError("Stage-A sigma tolerance must be finite and within [0,1e-4]")
        if not str(self.output_path).strip():
            raise ValueError("Stage-A capture output path must be non-empty")

    @property
    def receipt_path(self) -> str:
        path = Path(self.output_path)
        return str(path.with_suffix(path.suffix + ".receipt.json"))

    def assert_available(self) -> None:
        occupied = [
            str(path)
            for path in (Path(self.output_path), Path(self.receipt_path))
            if path.exists()
        ]
        if occupied:
            raise FileExistsError(
                "Stage-A capture evidence is immutable; remove neither file and choose a new "
                f"case/output identity instead of overwriting: {occupied}"
            )


@dataclass(frozen=True)
class StageACaptureInstallation:
    spec: StageACaptureSpec
    provenance: CaptureBundleProvenance
    wrapper_key: str = STAGE_A_CAPTURE_WRAPPER_KEY


def mark_pinned_stage_a_teacher(patcher: Any) -> Any:
    """Mark a patcher that was produced by the strict pinned-BF16 Stage-A loader.

    The marker is intentionally insufficient by itself: capture also revalidates the live
    native H3 topology and rejects patcher mutations at runtime. It binds normal Comfy clone
    chains to the strict loader provenance without pretending to make arbitrary object
    mutation cryptographically impossible.
    """
    setter = getattr(patcher, "set_attachments", None)
    if not callable(setter):
        raise RuntimeError("current Comfy ModelPatcher does not expose set_attachments")
    setter(STAGE_A_TEACHER_ATTACHMENT_KEY, StageATeacherAttachment())
    return patcher


def require_pinned_stage_a_teacher(patcher: Any) -> StageATeacherAttachment:
    attachments = getattr(patcher, "attachments", None)
    if not isinstance(attachments, Mapping):
        raise RuntimeError("Stage-A capture requires a Comfy ModelPatcher attachment map")
    marker = attachments.get(STAGE_A_TEACHER_ATTACHMENT_KEY)
    if not isinstance(marker, StageATeacherAttachment):
        raise RuntimeError(
            "Stage-A capture requires a model loaded by MiniMax H3 Stage-A BF16 Teacher Loader"
        )
    return StageATeacherAttachment(
        api=int(marker.api),
        teacher_model_revision=str(marker.teacher_model_revision),
        teacher_model_sha256=str(marker.teacher_model_sha256),
    )


def _plain_mapping(name: str, value: Any) -> None:
    if value:
        raise RuntimeError(f"Stage-A capture requires an unmodified teacher patcher; {name} is non-empty")


def _require_plain_patcher_state(patcher: Any, *, capture_controller: Any | None = None) -> None:
    require_pinned_stage_a_teacher(patcher)
    for name in (
        "patches",
        "object_patches",
        "weight_wrapper_patches",
        "injections",
        "hook_patches",
    ):
        _plain_mapping(name, getattr(patcher, name, {}))
    for name in ("current_hooks", "forced_hooks"):
        if getattr(patcher, name, None) is not None:
            raise RuntimeError(
                f"Stage-A capture requires an unmodified teacher patcher; {name} is active"
            )
    _plain_mapping("callbacks", getattr(patcher, "callbacks", {}))

    wrappers = getattr(patcher, "wrappers", {})
    if capture_controller is None:
        _plain_mapping("wrappers", wrappers)
    else:
        wrapper_type = capture_controller.wrapper_type
        expected = {wrapper_type: {STAGE_A_CAPTURE_WRAPPER_KEY: [capture_controller]}}
        if wrappers != expected:
            raise RuntimeError(
                "Stage-A capture wrapper must be the patcher's only wrapper; another runtime "
                "wrapper was installed before or after capture setup"
            )

    model_options = getattr(patcher, "model_options", {})
    if model_options is None:
        model_options = {}
    if not isinstance(model_options, Mapping):
        raise RuntimeError("Stage-A capture patcher model_options must be a mapping")
    transformer_options = model_options.get("transformer_options", {})
    if transformer_options is None:
        transformer_options = {}
    if not isinstance(transformer_options, Mapping):
        raise RuntimeError("Stage-A capture patcher transformer_options must be a mapping")
    require_plain_native_capture_options(transformer_options)


def _inner_model_from_patcher(patcher: Any) -> Any:
    outer = getattr(patcher, "model", None)
    inner = getattr(outer, "diffusion_model", None)
    if inner is None:
        raise RuntimeError("Stage-A capture requires model.diffusion_model on the MODEL patcher")
    return inner


def _sampling_multiplier_from_patcher(patcher: Any) -> float:
    outer = getattr(patcher, "model", None)
    sampling = getattr(outer, "model_sampling", None)
    multiplier = getattr(sampling, "multiplier", None)
    if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)):
        raise RuntimeError("Stage-A capture requires H3 flow sampling with an explicit multiplier")
    multiplier = float(multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise RuntimeError("Stage-A capture model-sampling multiplier must be finite and positive")
    if multiplier != 1000.0:
        raise RuntimeError(
            f"Stage-A capture expected native H3 flow multiplier 1000, got {multiplier:g}"
        )
    return multiplier


def runtime_video_sigma(timestep: Any, *, multiplier: float = 1000.0) -> float:
    if not torch.is_tensor(timestep):
        timestep = torch.as_tensor(timestep)
    values = timestep.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise RuntimeError("Stage-A capture observed an empty or non-finite H3 timestep")
    first = values[0]
    if not torch.allclose(values, first.expand_as(values), rtol=0.0, atol=1e-4):
        raise RuntimeError(
            "Stage-A capture requires one uniform video sigma per H3 forward; timestep rows differ"
        )
    sigma = float(first.item()) / float(multiplier)
    if not math.isfinite(sigma) or sigma < -1e-7 or sigma > 1.0 + 1e-7:
        raise RuntimeError(f"Stage-A capture derived invalid H3 video sigma {sigma!r}")
    return min(1.0, max(0.0, sigma))


def _wrapper_tree_contains_only(value: Any, target: Any) -> bool:
    if value is None:
        return True
    if value is target:
        return True
    if isinstance(value, Mapping):
        return all(_wrapper_tree_contains_only(item, target) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_wrapper_tree_contains_only(item, target) for item in value)
    return False


def _clean_runtime_options(
    transformer_options: Mapping[str, Any] | None,
    capture_controller: Any,
) -> dict[str, Any]:
    options = dict(transformer_options or {})
    wrappers = options.get("wrappers")
    if wrappers is not None and not _wrapper_tree_contains_only(wrappers, capture_controller):
        raise RuntimeError(
            "Stage-A capture runtime options contain wrapper state beyond the capture wrapper"
        )
    options.pop("wrappers", None)
    require_plain_native_capture_options(options)
    return options


def _require_exclusive_executor(executor: Any, capture_controller: Any) -> None:
    wrappers = getattr(executor, "wrappers", None)
    index = getattr(executor, "idx", None)
    if not isinstance(wrappers, Sequence) or isinstance(wrappers, (str, bytes)):
        raise RuntimeError("current Comfy wrapper executor does not expose an auditable wrapper list")
    if len(wrappers) != 1 or wrappers[0] is not capture_controller or index != 0:
        raise RuntimeError(
            "Stage-A capture must be the sole DIFFUSION_MODEL wrapper and execute first; "
            "patched/wrapped teacher execution is not valid training evidence"
        )


def _sigma_matches(observed: float, target: float, tolerance: float) -> bool:
    return math.isclose(float(observed), float(target), rel_tol=0.0, abs_tol=float(tolerance))


class StageALiveCaptureController:
    """One-shot DIFFUSION_MODEL observer for a fixed dataset case/sigma execution."""

    def __init__(
        self,
        spec: StageACaptureSpec,
        provenance: CaptureBundleProvenance,
    ) -> None:
        if provenance.dataset_manifest_sha256.lower() != spec.dataset_manifest_sha256.lower():
            raise ValueError("capture provenance dataset identity does not match Stage-A spec")
        self.spec = spec
        self.provenance = provenance
        self.wrapper_type: str | None = None
        self.patcher: Any | None = None
        self.inner_model: Any | None = None
        self.sampling_multiplier: float | None = None
        self.matched_forward_count = 0
        self.receipt: CaptureBundleWriteResult | None = None

    @property
    def captured(self) -> bool:
        return self.receipt is not None

    def bind(self, patcher: Any, *, wrapper_type: str) -> None:
        if self.patcher is not None:
            raise RuntimeError("Stage-A capture controller is already bound")
        self.patcher = patcher
        self.wrapper_type = str(wrapper_type)
        self.inner_model = _inner_model_from_patcher(patcher)
        self.sampling_multiplier = _sampling_multiplier_from_patcher(patcher)

    def _require_bound_plain_teacher(self) -> None:
        if self.patcher is None or self.inner_model is None or self.wrapper_type is None:
            raise RuntimeError("Stage-A capture controller is not installed on a MODEL patcher")
        _require_plain_patcher_state(self.patcher, capture_controller=self)
        if _inner_model_from_patcher(self.patcher) is not self.inner_model:
            raise RuntimeError("Stage-A capture MODEL diffusion object changed after installation")
        validate_loaded_native_teacher_model(self.inner_model)

    def __call__(
        self,
        executor: Any,
        x: Any,
        timestep: Any,
        context: Any,
        transformer_options: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._require_bound_plain_teacher()
        assert self.sampling_multiplier is not None
        observed_sigma = runtime_video_sigma(timestep, multiplier=self.sampling_multiplier)
        options = {} if transformer_options is None else transformer_options
        if not _sigma_matches(observed_sigma, self.spec.target_sigma, self.spec.sigma_tolerance):
            return executor(x, timestep, context, options, **kwargs)

        self.matched_forward_count += 1
        if self.matched_forward_count != 1 or self.captured:
            raise RuntimeError(
                "Stage-A target case/sigma executed more than once; refusing to mix CFG/re-entrant "
                "forwards or overwrite existing evidence"
            )
        _require_exclusive_executor(executor, self)
        if getattr(executor, "class_obj", None) is not self.inner_model:
            raise RuntimeError("Stage-A capture wrapper is not executing on the bound native H3 model")
        clean_options = _clean_runtime_options(options, self)
        self.spec.assert_available()

        capture_context = {
            "stage_a_source_case_id": self.spec.source_case_id,
            "stage_a_split": self.spec.split,
            "stage_a_declared_video_sigma": float(self.spec.target_sigma),
            "stage_a_observed_video_sigma": float(observed_sigma),
            "stage_a_sigma_tolerance": float(self.spec.sigma_tolerance),
            WORKFLOW_CONTEXT_KEY: self.spec.workflow_prompt_sha256.lower(),
        }
        with PilotActivationCapture(
            self.inner_model,
            case_id=self.spec.source_case_id,
            sigma=float(self.spec.target_sigma),
            modality_label=self.spec.modality_label,
            max_capture_bytes=int(self.spec.max_capture_bytes),
            block_indices=PILOT_BLOCKS,
            context=capture_context,
            require_plain_native=True,
        ) as capture:
            output = executor(x, timestep, context, clean_options, **kwargs)
        records = capture.records()
        self.spec.assert_available()
        self.receipt = write_captured_pilot_bundle(
            self.spec.output_path,
            records,
            provenance=self.provenance,
            receipt_path=self.spec.receipt_path,
        )
        return output


def install_stage_a_capture_wrapper(patcher: Any, controller: StageALiveCaptureController) -> Any:
    """Install one fail-closed Stage-A H3 DIFFUSION_MODEL wrapper on a patcher clone."""
    _require_plain_patcher_state(patcher)
    inner = _inner_model_from_patcher(patcher)
    validate_loaded_native_teacher_model(inner)

    try:
        import comfy.patcher_extension
    except ImportError as exc:
        raise RuntimeError("ComfyUI is required to install Stage-A live capture") from exc

    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    controller.bind(patcher, wrapper_type=wrapper_type)
    getter = getattr(patcher, "get_all_wrappers", None)
    adder = getattr(patcher, "add_wrapper_with_key", None)
    if not callable(getter) or not callable(adder):
        raise RuntimeError("current Comfy ModelPatcher wrapper API is unavailable")
    if getter(wrapper_type):
        raise RuntimeError("Stage-A capture requires no pre-existing DIFFUSION_MODEL wrappers")
    adder(wrapper_type, STAGE_A_CAPTURE_WRAPPER_KEY, controller)
    _require_plain_patcher_state(patcher, capture_controller=controller)
    return patcher


def discover_clean_git_revision(path: str | Path, *, label: str) -> str:
    path = Path(path).resolve()

    def run(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(path), *args],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"cannot resolve clean git provenance for {label}: {path}") from exc
        return completed.stdout.strip()

    revision = run("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise RuntimeError(f"{label} git revision is not a full 40-hex commit: {revision!r}")
    dirty = run("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError(
            f"{label} has tracked working-tree changes; Stage-A evidence requires a clean revision"
        )
    return revision.lower()


def _manifest_sigma_member(target: float, values: Sequence[Any]) -> bool:
    return any(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isclose(float(target), float(value), rel_tol=0.0, abs_tol=1e-12)
        for value in values
    )


def _case_filename(case_id: str, sigma: float) -> str:
    readable = _SAFE_FILENAME.sub("-", case_id.strip()).strip("._-")[:48] or "case"
    case_digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:12]
    sigma_text = format(float(sigma), ".17g").replace("-", "m").replace("+", "p").replace(".", "d")
    return f"{readable}-{case_digest}.sigma-{sigma_text}.capture.pt"


def resolve_output_subdir(output_root: str | Path, subdir: str) -> Path:
    root = Path(output_root).resolve()
    relative = Path(subdir or "keyless_stage_a")
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError("Stage-A output_subdir must be a clean relative path without '.' or '..'")
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError("Stage-A output_subdir escapes the Comfy output directory")
    return target


def build_stage_a_capture_spec(
    dataset_manifest_path: str | Path,
    *,
    case_id: str,
    target_sigma: float,
    output_root: str | Path,
    workflow_prompt: Mapping[str, Any],
    capture_node_id: str | int,
    asset_path_resolver: Callable[[str], str | Path] | None,
    output_subdir: str = "keyless_stage_a",
    max_capture_mib: int = 8192,
    sigma_tolerance: float = 1e-6,
) -> StageACaptureSpec:
    manifest = load_json_manifest(dataset_manifest_path)
    dataset_sha = validate_pilot_dataset_manifest(
        manifest,
        required_coverage_tags=CANONICAL_STAGE_A_COVERAGE_TAGS,
    )
    cases = manifest["cases"]
    assert isinstance(cases, list)
    selected = [row for row in cases if isinstance(row, dict) and row.get("case_id") == case_id]
    if len(selected) != 1:
        raise ValueError(f"Stage-A dataset must contain exactly one case_id {case_id!r}")
    case = selected[0]
    sigmas = case.get("sigmas")
    assert isinstance(sigmas, list)
    if not _manifest_sigma_member(float(target_sigma), sigmas):
        raise ValueError(
            f"target sigma {target_sigma!r} is not declared for Stage-A case {case_id!r}"
        )
    if isinstance(max_capture_mib, bool) or not isinstance(max_capture_mib, int) or max_capture_mib <= 0:
        raise ValueError("Stage-A max_capture_mib must be a positive integer")

    expected_workflow = require_manifest_case_workflow_prompt_sha256(case, case_id=case_id)
    actual_workflow = canonical_stage_a_workflow_prompt_sha256(
        workflow_prompt,
        capture_node_id=capture_node_id,
    )
    if actual_workflow.lower() != expected_workflow.lower():
        raise ValueError(
            "executed Comfy API prompt does not match the predeclared Stage-A workflow "
            f"identity for case {case_id!r}: expected={expected_workflow}, actual={actual_workflow}"
        )
    validate_stage_a_manifest_assets_for_workflow(
        case,
        workflow_prompt,
        asset_path_resolver=asset_path_resolver,
    )

    output_dir = resolve_output_subdir(output_root, output_subdir)
    output_path = output_dir / _case_filename(case_id, float(target_sigma))
    spec = StageACaptureSpec(
        dataset_manifest_sha256=dataset_sha,
        source_case_id=case_id,
        split=str(case["split"]),
        modality_label=str(case["modality_label"]),
        target_sigma=float(target_sigma),
        output_path=str(output_path),
        max_capture_bytes=int(max_capture_mib) * 1024 * 1024,
        workflow_prompt_sha256=actual_workflow,
        sigma_tolerance=float(sigma_tolerance),
    )
    spec.assert_available()
    return spec


def build_runtime_execution_descriptor(patcher: Any) -> str:
    load_device = getattr(patcher, "load_device", None)
    cuda_runtime = getattr(torch.version, "cuda", None) or "none"
    return (
        "MiniMax H3 Stage-A live full-model capture; "
        f"python={platform.python_version()}; torch={torch.__version__}; "
        f"torch_cuda={cuda_runtime}; load_device={load_device}"
    )
