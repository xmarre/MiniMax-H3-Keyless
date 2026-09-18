from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .activation_capture import PilotActivationCapture
from .capture_io import (
    CaptureBundleProvenance,
    CaptureBundleWriteResult,
    write_captured_pilot_bundle,
)
from .contracts import TEACHER_SHA256
from .live_capture import (
    discover_clean_git_revision,
    require_pinned_stage_a_teacher,
    resolve_output_subdir,
    runtime_video_sigma,
)
from .native_capture_policy import require_plain_native_capture_options
from .pilot_campaign import PILOT_BLOCKS
from .teacher import validate_loaded_native_teacher_model


REAL_H3_REPLAY_CAPTURE_WRAPPER_KEY = "minimax_h3_keyless_real_h3_replay_capture_v1"
REAL_H3_REPLAY_CAPTURE_CLASS_TYPE = "MiniMaxH3RealH3ReplayCapture"
REAL_H3_REPLAY_EVIDENCE_SCHEMA = "minimax_h3_keyless_real_h3_replay_evidence_v1"


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-hex SHA-256 string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a 64-hex SHA-256 string") from exc
    return value.lower()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_real_h3_replay_workflow_prompt_sha256(
    prompt: Mapping[str, Any],
    *,
    capture_node_id: str | int,
) -> str:
    """Hash the executed API prompt for a one-off real-H3 replay capture.

    The target sigma remains part of the semantic identity. Only storage/budget knobs
    that do not alter the H3 forward are normalized.
    """
    if not isinstance(prompt, Mapping) or not prompt:
        raise ValueError("real-H3 replay capture requires a non-empty Comfy API prompt")
    normalized = json.loads(json.dumps(prompt, ensure_ascii=False, allow_nan=False))
    node_id = str(capture_node_id)
    for node in normalized.values():
        if isinstance(node, dict):
            node.pop("_meta", None)
    node = normalized.get(node_id)
    if not isinstance(node, dict):
        raise ValueError(f"real-H3 replay capture node id {node_id!r} is absent from the API prompt")
    if node.get("class_type") != REAL_H3_REPLAY_CAPTURE_CLASS_TYPE:
        raise ValueError(
            f"workflow node {node_id!r} is not {REAL_H3_REPLAY_CAPTURE_CLASS_TYPE!r}"
        )
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("real-H3 replay capture node is missing its input mapping")
    for name in ("output_subdir", "max_capture_mib", "sigma_tolerance"):
        if name in inputs:
            inputs[name] = f"<real-h3-replay:{name}>"
    return _canonical_json_sha256(normalized)


def real_h3_replay_evidence_identity(
    *,
    workflow_prompt_sha256: str,
    target_sigma: float,
) -> str:
    workflow = _require_sha256("workflow_prompt_sha256", workflow_prompt_sha256)
    sigma = float(target_sigma)
    if not math.isfinite(sigma) or not 0.0 <= sigma <= 1.0:
        raise ValueError("real-H3 replay target sigma must be finite and within [0,1]")
    return _canonical_json_sha256(
        {
            "schema": REAL_H3_REPLAY_EVIDENCE_SCHEMA,
            "teacher_model_sha256": TEACHER_SHA256,
            "workflow_prompt_sha256": workflow,
            "target_sigma": sigma,
            "blocks": list(PILOT_BLOCKS),
        }
    )


@dataclass(frozen=True)
class RealH3ReplayCaptureSpec:
    evidence_identity_sha256: str
    workflow_prompt_sha256: str
    target_sigma: float
    output_path: str
    max_capture_bytes: int
    sigma_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        _require_sha256("evidence_identity_sha256", self.evidence_identity_sha256)
        _require_sha256("workflow_prompt_sha256", self.workflow_prompt_sha256)
        sigma = float(self.target_sigma)
        if not math.isfinite(sigma) or not 0.0 <= sigma <= 1.0:
            raise ValueError("real-H3 replay target sigma must be finite and within [0,1]")
        if int(self.max_capture_bytes) <= 0:
            raise ValueError("real-H3 replay capture byte budget must be positive")
        tolerance = float(self.sigma_tolerance)
        if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 1e-4:
            raise ValueError("real-H3 replay sigma tolerance must be finite and within [0,1e-4]")
        if not str(self.output_path).strip():
            raise ValueError("real-H3 replay output path must be non-empty")

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
                "real-H3 replay evidence is immutable; choose a new output identity "
                f"instead of overwriting: {occupied}"
            )


def _plain_mapping(name: str, value: Any) -> None:
    if value:
        raise RuntimeError(
            f"real-H3 replay capture requires an unmodified teacher patcher; {name} is non-empty"
        )


def _inner_model_from_patcher(patcher: Any) -> Any:
    outer = getattr(patcher, "model", None)
    inner = getattr(outer, "diffusion_model", None)
    if inner is None:
        raise RuntimeError("real-H3 replay capture requires model.diffusion_model")
    return inner


def _sampling_multiplier_from_patcher(patcher: Any) -> float:
    outer = getattr(patcher, "model", None)
    sampling = getattr(outer, "model_sampling", None)
    multiplier = getattr(sampling, "multiplier", None)
    if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)):
        raise RuntimeError("real-H3 replay capture requires native H3 flow sampling")
    multiplier = float(multiplier)
    if multiplier != 1000.0:
        raise RuntimeError(
            f"real-H3 replay capture expected H3 flow multiplier 1000, got {multiplier:g}"
        )
    return multiplier


def _require_plain_patcher_state(
    patcher: Any,
    *,
    capture_controller: Any | None = None,
) -> None:
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
                f"real-H3 replay capture requires an unmodified teacher; {name} is active"
            )
    _plain_mapping("callbacks", getattr(patcher, "callbacks", {}))

    wrappers = getattr(patcher, "wrappers", {})
    if capture_controller is None:
        _plain_mapping("wrappers", wrappers)
    else:
        expected = {
            capture_controller.wrapper_type: {
                REAL_H3_REPLAY_CAPTURE_WRAPPER_KEY: [capture_controller]
            }
        }
        if wrappers != expected:
            raise RuntimeError(
                "real-H3 replay capture must be the patcher's only runtime wrapper"
            )

    model_options = getattr(patcher, "model_options", {}) or {}
    if not isinstance(model_options, Mapping):
        raise RuntimeError("real-H3 replay patcher model_options must be a mapping")
    transformer_options = model_options.get("transformer_options", {}) or {}
    if not isinstance(transformer_options, Mapping):
        raise RuntimeError("real-H3 replay transformer_options must be a mapping")
    require_plain_native_capture_options(transformer_options)


def _wrapper_tree_contains_only(value: Any, target: Any) -> bool:
    if value is None or value is target:
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
            "real-H3 replay runtime options contain wrappers beyond the capture wrapper"
        )
    options.pop("wrappers", None)
    require_plain_native_capture_options(options)
    return options


def _require_exclusive_executor(executor: Any, capture_controller: Any) -> None:
    wrappers = getattr(executor, "wrappers", None)
    index = getattr(executor, "idx", None)
    if not isinstance(wrappers, Sequence) or isinstance(wrappers, (str, bytes)):
        raise RuntimeError("current Comfy wrapper executor has no auditable wrapper list")
    if len(wrappers) != 1 or wrappers[0] is not capture_controller or index != 0:
        raise RuntimeError(
            "real-H3 replay capture must be the sole DIFFUSION_MODEL wrapper and execute first"
        )


class RealH3ReplayCaptureController:
    def __init__(
        self,
        spec: RealH3ReplayCaptureSpec,
        provenance: CaptureBundleProvenance,
    ) -> None:
        if provenance.purpose != "real_h3_replay":
            raise ValueError("real-H3 replay provenance has the wrong purpose")
        if (
            provenance.dataset_manifest_sha256.lower()
            != spec.evidence_identity_sha256.lower()
        ):
            raise ValueError("real-H3 replay provenance identity does not match its spec")
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
            raise RuntimeError("real-H3 replay controller is already bound")
        self.patcher = patcher
        self.wrapper_type = str(wrapper_type)
        self.inner_model = _inner_model_from_patcher(patcher)
        self.sampling_multiplier = _sampling_multiplier_from_patcher(patcher)

    def _require_bound_plain_teacher(self) -> None:
        if self.patcher is None or self.inner_model is None or self.wrapper_type is None:
            raise RuntimeError("real-H3 replay controller is not installed on a MODEL patcher")
        _require_plain_patcher_state(self.patcher, capture_controller=self)
        if _inner_model_from_patcher(self.patcher) is not self.inner_model:
            raise RuntimeError("real-H3 replay diffusion object changed after installation")
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
        observed_sigma = runtime_video_sigma(
            timestep,
            multiplier=self.sampling_multiplier,
        )
        options = {} if transformer_options is None else transformer_options
        if not math.isclose(
            observed_sigma,
            float(self.spec.target_sigma),
            rel_tol=0.0,
            abs_tol=float(self.spec.sigma_tolerance),
        ):
            return executor(x, timestep, context, options, **kwargs)

        self.matched_forward_count += 1
        if self.matched_forward_count != 1 or self.captured:
            raise RuntimeError(
                "real-H3 replay target sigma executed more than once; refusing ambiguous evidence"
            )
        _require_exclusive_executor(executor, self)
        if getattr(executor, "class_obj", None) is not self.inner_model:
            raise RuntimeError(
                "real-H3 replay wrapper is not executing on the bound native H3 model"
            )
        clean_options = _clean_runtime_options(options, self)
        self.spec.assert_available()

        capture_context = {
            "real_h3_replay_evidence_identity_sha256": self.spec.evidence_identity_sha256,
            "real_h3_replay_workflow_prompt_sha256": self.spec.workflow_prompt_sha256,
            "real_h3_replay_declared_video_sigma": float(self.spec.target_sigma),
            "real_h3_replay_observed_video_sigma": float(observed_sigma),
            "real_h3_replay_sigma_tolerance": float(self.spec.sigma_tolerance),
        }
        with PilotActivationCapture(
            self.inner_model,
            case_id="real-h3-replay",
            sigma=float(self.spec.target_sigma),
            modality_label="native_h3_replay",
            max_capture_bytes=int(self.spec.max_capture_bytes),
            block_indices=PILOT_BLOCKS,
            context=capture_context,
            require_plain_native=True,
        ) as capture:
            output = executor(x, timestep, context, clean_options, **kwargs)

        self.receipt = write_captured_pilot_bundle(
            self.spec.output_path,
            capture.records(),
            provenance=self.provenance,
            receipt_path=self.spec.receipt_path,
        )
        return output


def install_real_h3_replay_capture_wrapper(
    patcher: Any,
    controller: RealH3ReplayCaptureController,
) -> Any:
    _require_plain_patcher_state(patcher)
    inner = _inner_model_from_patcher(patcher)
    validate_loaded_native_teacher_model(inner)
    try:
        import comfy.patcher_extension
    except ImportError as exc:
        raise RuntimeError(
            "ComfyUI is required to install the real-H3 replay capture"
        ) from exc

    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    controller.bind(patcher, wrapper_type=wrapper_type)
    getter = getattr(patcher, "get_all_wrappers", None)
    adder = getattr(patcher, "add_wrapper_with_key", None)
    if not callable(getter) or not callable(adder):
        raise RuntimeError("current Comfy ModelPatcher wrapper API is unavailable")
    if getter(wrapper_type):
        raise RuntimeError(
            "real-H3 replay capture requires no pre-existing DIFFUSION_MODEL wrappers"
        )
    adder(wrapper_type, REAL_H3_REPLAY_CAPTURE_WRAPPER_KEY, controller)
    _require_plain_patcher_state(patcher, capture_controller=controller)
    return patcher


def build_real_h3_replay_capture_spec(
    *,
    target_sigma: float,
    output_root: str | Path,
    workflow_prompt: Mapping[str, Any],
    capture_node_id: str | int,
    output_subdir: str = "keyless_real_h3_replay",
    max_capture_mib: int = 8192,
    sigma_tolerance: float = 1e-6,
) -> RealH3ReplayCaptureSpec:
    workflow_sha = canonical_real_h3_replay_workflow_prompt_sha256(
        workflow_prompt,
        capture_node_id=capture_node_id,
    )
    evidence_sha = real_h3_replay_evidence_identity(
        workflow_prompt_sha256=workflow_sha,
        target_sigma=float(target_sigma),
    )
    output_dir = resolve_output_subdir(output_root, output_subdir)
    sigma_text = (
        format(float(target_sigma), ".17g")
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "d")
    )
    output_path = output_dir / (
        f"real-h3-replay-{workflow_sha[:12]}.sigma-{sigma_text}.capture.pt"
    )
    spec = RealH3ReplayCaptureSpec(
        evidence_identity_sha256=evidence_sha,
        workflow_prompt_sha256=workflow_sha,
        target_sigma=float(target_sigma),
        output_path=str(output_path),
        max_capture_bytes=int(max_capture_mib) * 1024 * 1024,
        sigma_tolerance=float(sigma_tolerance),
    )
    spec.assert_available()
    return spec


def build_real_h3_replay_execution_descriptor(patcher: Any) -> str:
    load_device = getattr(patcher, "load_device", None)
    cuda_runtime = getattr(torch.version, "cuda", None) or "none"
    return (
        "MiniMax H3 bounded real-H3 Sol replay capture; "
        f"python={platform.python_version()}; torch={torch.__version__}; "
        f"torch_cuda={cuda_runtime}; load_device={load_device}"
    )


def install_from_comfy_node(
    model: Any,
    *,
    target_sigma: float,
    output_subdir: str,
    max_capture_mib: int,
    sigma_tolerance: float,
    prompt: Mapping[str, Any],
    unique_id: str | int,
    plugin_root: str | Path,
    comfy_root: str | Path,
    output_root: str | Path,
) -> Any:
    require_pinned_stage_a_teacher(model)
    cloned = model.clone()
    require_pinned_stage_a_teacher(cloned)
    spec = build_real_h3_replay_capture_spec(
        target_sigma=float(target_sigma),
        output_root=output_root,
        workflow_prompt=prompt,
        capture_node_id=unique_id,
        output_subdir=output_subdir,
        max_capture_mib=int(max_capture_mib),
        sigma_tolerance=float(sigma_tolerance),
    )
    provenance = CaptureBundleProvenance(
        code_commit=discover_clean_git_revision(
            plugin_root,
            label="MiniMax-H3-Keyless",
        ),
        comfy_commit=discover_clean_git_revision(
            comfy_root,
            label="ComfyUI",
        ),
        # Legacy field in the shared capture-bundle format. For this purpose it holds
        # the immutable replay-evidence identity, not a Stage-A training manifest hash.
        dataset_manifest_sha256=spec.evidence_identity_sha256,
        execution_descriptor=build_real_h3_replay_execution_descriptor(cloned),
        purpose="real_h3_replay",
    )
    controller = RealH3ReplayCaptureController(spec, provenance)
    install_real_h3_replay_capture_wrapper(cloned, controller)
    return cloned
