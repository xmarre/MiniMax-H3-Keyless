from __future__ import annotations

import hashlib
import math
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .live_capture import (
    _clean_runtime_options,
    _inner_model_from_patcher,
    _require_exclusive_executor,
    _sampling_multiplier_from_patcher,
    require_pinned_stage_a_teacher,
    resolve_output_subdir,
    runtime_video_sigma,
)
from .native_capture_policy import require_plain_native_capture_options
from .pilot_campaign import load_json_manifest, validate_pilot_dataset_manifest
from .pilot_inputs import CANONICAL_STAGE_A_COVERAGE_TAGS
from .progressive import ProgressivePrefix, progressive_capture_session, validate_progressive_model_prefix
from .progressive_capture_io import (
    ProgressiveCaptureProvenance,
    ProgressiveCaptureWriteResult,
    write_progressive_capture_bundle,
)
from .progressive_overlay import require_progressive_overlay


PROGRESSIVE_CAPTURE_WRAPPER_KEY = "minimax_h3_keyless_progressive_capture_v1"
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ProgressiveLiveCaptureSpec:
    dataset_manifest_sha256: str
    prefix_identity_sha256: str
    prefix_manifest_sha256: str
    source_case_id: str
    split: str
    modality_label: str
    target_sigma: float
    target_block: int
    output_path: str
    max_capture_bytes: int
    sigma_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "dataset_manifest_sha256",
            "prefix_identity_sha256",
            "prefix_manifest_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"progressive capture {name} must be a SHA-256")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(f"progressive capture {name} must be a SHA-256") from exc
        if not self.source_case_id.strip():
            raise ValueError("progressive capture source case_id must be non-empty")
        if self.split not in ("train", "holdout"):
            raise ValueError("progressive capture split must be train or holdout")
        if not self.modality_label.strip():
            raise ValueError("progressive capture modality label must be non-empty")
        if not math.isfinite(float(self.target_sigma)) or not 0.0 <= float(self.target_sigma) <= 1.0:
            raise ValueError("progressive capture target sigma must be finite and within [0,1]")
        if isinstance(self.target_block, bool) or not isinstance(self.target_block, int):
            raise ValueError("progressive capture target_block must be an integer")
        if not 0 <= self.target_block < 50:
            raise ValueError("progressive capture target_block must be within [0,50)")
        if self.max_capture_bytes <= 0:
            raise ValueError("progressive capture byte budget must be positive")
        tolerance = float(self.sigma_tolerance)
        if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 1e-4:
            raise ValueError(
                "progressive capture sigma tolerance must be finite and within [0,1e-4]"
            )
        if not str(self.output_path).strip():
            raise ValueError("progressive capture output path must be non-empty")

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
                "progressive capture evidence is immutable; choose a new output identity "
                f"instead of overwriting: {occupied}"
            )


def _plain_mapping(name: str, value: Any) -> None:
    if value:
        raise RuntimeError(
            f"progressive live capture requires no extra patcher mutation; {name} is non-empty"
        )


def _require_progressive_capture_patcher_state(
    patcher: Any,
    prefix: ProgressivePrefix,
    prefix_manifest_sha256: str,
    *,
    capture_controller: Any | None = None,
) -> None:
    require_pinned_stage_a_teacher(patcher)
    require_progressive_overlay(
        patcher,
        prefix=prefix,
        prefix_manifest_sha256=prefix_manifest_sha256,
    )
    for name in (
        "patches",
        "weight_wrapper_patches",
        "injections",
        "hook_patches",
    ):
        _plain_mapping(name, getattr(patcher, name, {}))
    for name in ("current_hooks", "forced_hooks"):
        if getattr(patcher, name, None) is not None:
            raise RuntimeError(
                f"progressive live capture requires no active hooks; {name} is active"
            )
    _plain_mapping("callbacks", getattr(patcher, "callbacks", {}))

    wrappers = getattr(patcher, "wrappers", {})
    if capture_controller is None:
        _plain_mapping("wrappers", wrappers)
    else:
        wrapper_type = capture_controller.wrapper_type
        expected = {wrapper_type: {PROGRESSIVE_CAPTURE_WRAPPER_KEY: [capture_controller]}}
        if wrappers != expected:
            raise RuntimeError(
                "progressive live capture wrapper must be the patcher's only wrapper"
            )

    model_options = getattr(patcher, "model_options", {})
    if model_options is None:
        model_options = {}
    if not isinstance(model_options, Mapping):
        raise RuntimeError("progressive capture patcher model_options must be a mapping")
    transformer_options = model_options.get("transformer_options", {})
    if transformer_options is None:
        transformer_options = {}
    if not isinstance(transformer_options, Mapping):
        raise RuntimeError(
            "progressive capture patcher transformer_options must be a mapping"
        )
    require_plain_native_capture_options(transformer_options)


def _sigma_matches(observed: float, target: float, tolerance: float) -> bool:
    return math.isclose(
        float(observed),
        float(target),
        rel_tol=0.0,
        abs_tol=float(tolerance),
    )


def _manifest_sigma_member(target: float, values: Sequence[Any]) -> bool:
    return any(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isclose(float(target), float(value), rel_tol=0.0, abs_tol=1e-12)
        for value in values
    )


def _capture_filename(
    *,
    prefix: ProgressivePrefix,
    case_id: str,
    sigma: float,
) -> str:
    target = prefix.next_block
    if target is None:
        raise ValueError("cannot name a capture for a complete progressive prefix")
    readable = _SAFE_FILENAME.sub("-", case_id.strip()).strip("._-")[:48] or "case"
    case_digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:12]
    sigma_text = (
        format(float(sigma), ".17g")
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "d")
    )
    return (
        f"block-{target:02d}.prefix-{prefix.identity_sha256[:12]}."
        f"{readable}-{case_digest}.sigma-{sigma_text}.capture.pt"
    )


def build_progressive_capture_spec(
    dataset_manifest_path: str | Path,
    prefix: ProgressivePrefix,
    prefix_manifest_sha256: str,
    *,
    case_id: str,
    target_sigma: float,
    output_root: str | Path,
    output_subdir: str = "keyless_progressive",
    max_capture_mib: int = 8192,
    sigma_tolerance: float = 1e-6,
) -> ProgressiveLiveCaptureSpec:
    target = prefix.next_block
    if target is None:
        raise ValueError("progressive prefix is complete; no next block can be captured")
    manifest = load_json_manifest(dataset_manifest_path)
    dataset_sha = validate_pilot_dataset_manifest(
        manifest,
        required_coverage_tags=CANONICAL_STAGE_A_COVERAGE_TAGS,
    )
    if dataset_sha.lower() != prefix.dataset_manifest_sha256.lower():
        raise ValueError(
            "progressive capture dataset manifest differs from the authorized prefix"
        )
    cases = manifest["cases"]
    assert isinstance(cases, list)
    selected = [
        row
        for row in cases
        if isinstance(row, dict) and row.get("case_id") == case_id
    ]
    if len(selected) != 1:
        raise ValueError(
            f"progressive dataset must contain exactly one case_id {case_id!r}"
        )
    case = selected[0]
    sigmas = case.get("sigmas")
    assert isinstance(sigmas, list)
    if not _manifest_sigma_member(float(target_sigma), sigmas):
        raise ValueError(
            f"target sigma {target_sigma!r} is not declared for progressive case {case_id!r}"
        )
    if (
        isinstance(max_capture_mib, bool)
        or not isinstance(max_capture_mib, int)
        or max_capture_mib <= 0
    ):
        raise ValueError("progressive max_capture_mib must be a positive integer")

    output_dir = resolve_output_subdir(output_root, output_subdir)
    output_path = output_dir / _capture_filename(
        prefix=prefix,
        case_id=case_id,
        sigma=float(target_sigma),
    )
    spec = ProgressiveLiveCaptureSpec(
        dataset_manifest_sha256=dataset_sha,
        prefix_identity_sha256=prefix.identity_sha256,
        prefix_manifest_sha256=prefix_manifest_sha256,
        source_case_id=case_id,
        split=str(case["split"]),
        modality_label=str(case["modality_label"]),
        target_sigma=float(target_sigma),
        target_block=target,
        output_path=str(output_path),
        max_capture_bytes=int(max_capture_mib) * 1024 * 1024,
        sigma_tolerance=float(sigma_tolerance),
    )
    spec.assert_available()
    return spec


def build_progressive_execution_descriptor(
    patcher: Any,
    prefix: ProgressivePrefix,
) -> str:
    load_device = getattr(patcher, "load_device", None)
    cuda_runtime = getattr(torch.version, "cuda", None) or "none"
    return (
        "MiniMax H3 progressive full-model live-input capture; "
        f"python={platform.python_version()}; torch={torch.__version__}; "
        f"torch_cuda={cuda_runtime}; load_device={load_device}; "
        f"accepted_prefix={len(prefix.accepted)}; target_block={prefix.next_block}"
    )


class ProgressiveLiveCaptureController:
    """One-shot full-model capture for the next native block after an accepted prefix."""

    def __init__(
        self,
        spec: ProgressiveLiveCaptureSpec,
        prefix: ProgressivePrefix,
        provenance: ProgressiveCaptureProvenance,
    ) -> None:
        if prefix.complete:
            raise ValueError("cannot capture from a complete progressive prefix")
        if spec.prefix_identity_sha256.lower() != prefix.identity_sha256.lower():
            raise ValueError("progressive capture spec does not belong to the supplied prefix")
        if spec.dataset_manifest_sha256.lower() != prefix.dataset_manifest_sha256.lower():
            raise ValueError("progressive capture spec dataset differs from prefix")
        if spec.target_block != prefix.next_block:
            raise ValueError("progressive capture spec targets the wrong next block")
        expected_provenance = {
            "dataset_manifest_sha256": prefix.dataset_manifest_sha256.lower(),
            "gate_manifest_sha256": prefix.gate_manifest_sha256.lower(),
            "stage_a_campaign_sha256": prefix.stage_a_campaign_sha256.lower(),
            "prefix_identity_sha256": prefix.identity_sha256.lower(),
            "target_block": prefix.next_block,
            "code_commit": prefix.code_commit.lower(),
        }
        for name, expected in expected_provenance.items():
            actual = getattr(provenance, name)
            normalized = actual.lower() if isinstance(actual, str) else actual
            if normalized != expected:
                raise ValueError(
                    f"progressive capture provenance {name} does not match prefix"
                )

        self.spec = spec
        self.prefix = prefix
        self.provenance = provenance
        self.wrapper_type: str | None = None
        self.patcher: Any | None = None
        self.inner_model: Any | None = None
        self.sampling_multiplier: float | None = None
        self.matched_forward_count = 0
        self.receipt: ProgressiveCaptureWriteResult | None = None

    @property
    def captured(self) -> bool:
        return self.receipt is not None

    def bind(self, patcher: Any, *, wrapper_type: str) -> None:
        if self.patcher is not None:
            raise RuntimeError("progressive capture controller is already bound")
        self.patcher = patcher
        self.wrapper_type = str(wrapper_type)
        self.inner_model = _inner_model_from_patcher(patcher)
        self.sampling_multiplier = _sampling_multiplier_from_patcher(patcher)

    def _require_bound_overlay(self) -> None:
        if self.patcher is None or self.inner_model is None or self.wrapper_type is None:
            raise RuntimeError(
                "progressive capture controller is not installed on a MODEL patcher"
            )
        _require_progressive_capture_patcher_state(
            self.patcher,
            self.prefix,
            self.spec.prefix_manifest_sha256,
            capture_controller=self,
        )
        if _inner_model_from_patcher(self.patcher) is not self.inner_model:
            raise RuntimeError(
                "progressive capture MODEL diffusion object changed after installation"
            )
        # At DIFFUSION_MODEL wrapper execution time Comfy must already have applied the
        # clone's object patches. This proves the captured target input actually came through
        # every accepted Keyless block rather than the shared native teacher.
        validate_progressive_model_prefix(self.inner_model, self.prefix)

    def __call__(
        self,
        executor: Any,
        x: Any,
        timestep: Any,
        context: Any,
        transformer_options: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._require_bound_overlay()
        assert self.sampling_multiplier is not None
        observed_sigma = runtime_video_sigma(
            timestep,
            multiplier=self.sampling_multiplier,
        )
        options = {} if transformer_options is None else transformer_options
        if not _sigma_matches(
            observed_sigma,
            self.spec.target_sigma,
            self.spec.sigma_tolerance,
        ):
            return executor(x, timestep, context, options, **kwargs)

        self.matched_forward_count += 1
        if self.matched_forward_count != 1 or self.captured:
            raise RuntimeError(
                "progressive target case/sigma executed more than once; refusing to mix "
                "CFG/re-entrant forwards or overwrite existing evidence"
            )
        _require_exclusive_executor(executor, self)
        if getattr(executor, "class_obj", None) is not self.inner_model:
            raise RuntimeError(
                "progressive capture wrapper is not executing on the bound H3 model"
            )
        clean_options = _clean_runtime_options(options, self)
        self.spec.assert_available()

        capture_context = {
            "progressive_source_case_id": self.spec.source_case_id,
            "progressive_split": self.spec.split,
            "progressive_declared_video_sigma": float(self.spec.target_sigma),
            "progressive_observed_video_sigma": float(observed_sigma),
            "progressive_sigma_tolerance": float(self.spec.sigma_tolerance),
            "progressive_prefix_manifest_sha256": self.spec.prefix_manifest_sha256,
        }
        with progressive_capture_session(
            self.inner_model,
            self.prefix,
            case_id=self.spec.source_case_id,
            sigma=float(self.spec.target_sigma),
            modality_label=self.spec.modality_label,
            max_capture_bytes=int(self.spec.max_capture_bytes),
            context=capture_context,
        ) as capture:
            output = executor(x, timestep, context, clean_options, **kwargs)
        records = capture.records()
        if len(records) != 1:
            raise RuntimeError(
                "progressive live capture must produce exactly one next-block record"
            )
        self.spec.assert_available()
        self.receipt = write_progressive_capture_bundle(
            self.spec.output_path,
            records[0],
            provenance=self.provenance,
            receipt_path=self.spec.receipt_path,
        )
        return output


def install_progressive_capture_wrapper(
    patcher: Any,
    controller: ProgressiveLiveCaptureController,
) -> Any:
    """Install the sole DIFFUSION_MODEL wrapper on a validated progressive overlay."""
    _require_progressive_capture_patcher_state(
        patcher,
        controller.prefix,
        controller.spec.prefix_manifest_sha256,
    )
    try:
        import comfy.patcher_extension
    except ImportError as exc:
        raise RuntimeError(
            "ComfyUI is required to install progressive live capture"
        ) from exc

    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    controller.bind(patcher, wrapper_type=wrapper_type)
    getter = getattr(patcher, "get_all_wrappers", None)
    adder = getattr(patcher, "add_wrapper_with_key", None)
    if not callable(getter) or not callable(adder):
        raise RuntimeError("current Comfy ModelPatcher wrapper API is unavailable")
    if getter(wrapper_type):
        raise RuntimeError(
            "progressive live capture requires no pre-existing DIFFUSION_MODEL wrappers"
        )
    adder(wrapper_type, PROGRESSIVE_CAPTURE_WRAPPER_KEY, controller)
    _require_progressive_capture_patcher_state(
        patcher,
        controller.prefix,
        controller.spec.prefix_manifest_sha256,
        capture_controller=controller,
    )
    return patcher
