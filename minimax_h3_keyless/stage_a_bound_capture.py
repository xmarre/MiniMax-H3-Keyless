from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from . import live_capture as _live
from .activation_capture import PilotActivationCapture
from .capture_io import CaptureBundleProvenance, CaptureBundleWriteResult, write_captured_pilot_bundle
from .pilot_campaign import PILOT_BLOCKS, load_json_manifest, validate_pilot_dataset_manifest
from .pilot_inputs import CANONICAL_STAGE_A_COVERAGE_TAGS
from .stage_a_execution_binding import (
    WORKFLOW_CONTEXT_KEY,
    canonical_stage_a_workflow_prompt_sha256,
    require_manifest_case_workflow_prompt_sha256,
    validate_stage_a_manifest_assets_for_workflow,
)


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


class StageALiveCaptureController(_live.StageALiveCaptureController):
    """Production Stage-A observer with immutable Comfy-workflow identity binding."""

    spec: StageACaptureSpec

    def __init__(self, spec: StageACaptureSpec, provenance: CaptureBundleProvenance) -> None:
        super().__init__(spec, provenance)  # type: ignore[arg-type]
        self.receipt: CaptureBundleWriteResult | None = None

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
        observed_sigma = _live.runtime_video_sigma(
            timestep, multiplier=self.sampling_multiplier
        )
        options = {} if transformer_options is None else transformer_options
        if not _live._sigma_matches(  # noqa: SLF001 - same-package runtime contract
            observed_sigma, self.spec.target_sigma, self.spec.sigma_tolerance
        ):
            return executor(x, timestep, context, options, **kwargs)

        self.matched_forward_count += 1
        if self.matched_forward_count != 1 or self.captured:
            raise RuntimeError(
                "Stage-A target case/sigma executed more than once; refusing to mix CFG/re-entrant "
                "forwards or overwrite existing evidence"
            )
        _live._require_exclusive_executor(executor, self)  # noqa: SLF001
        if getattr(executor, "class_obj", None) is not self.inner_model:
            raise RuntimeError("Stage-A capture wrapper is not executing on the bound native H3 model")
        clean_options = _live._clean_runtime_options(options, self)  # noqa: SLF001
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
    if not _live._manifest_sigma_member(float(target_sigma), sigmas):  # noqa: SLF001
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

    output_dir = _live.resolve_output_subdir(output_root, output_subdir)
    output_path = output_dir / _live._case_filename(case_id, float(target_sigma))  # noqa: SLF001
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
