from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Mapping, Sequence

from .activation_capture import CapturedPilotCase
from .capture_io import CaptureBundleProvenance, load_captured_pilot_bundle
from .checkpoint import sha256_file
from .pilot_campaign import (
    PILOT_BLOCKS,
    _require_sha256,
    validate_pilot_dataset_manifest,
)


SplitName = Literal["train", "holdout"]


@dataclass(frozen=True)
class CaptureArtifactRef:
    """Immutable reference to one persisted Stage-A live-capture bundle."""

    bundle_path: str
    receipt_path: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not str(self.bundle_path).strip() or not str(self.receipt_path).strip():
            raise ValueError("capture artifact paths must be non-empty")
        _require_sha256("capture receipt SHA-256", self.receipt_sha256)


@dataclass(frozen=True)
class StageACaptureSet:
    """Validated fixed Stage-A replay corpus grouped by block and complete-case split."""

    dataset_manifest_sha256: str
    code_commit: str
    comfy_commit: str
    execution_descriptor: str
    train_by_block: Mapping[int, tuple[CapturedPilotCase, ...]]
    holdout_by_block: Mapping[int, tuple[CapturedPilotCase, ...]]
    artifact_refs: tuple[CaptureArtifactRef, ...]

    def records(self, block_index: int, split: SplitName) -> tuple[CapturedPilotCase, ...]:
        if block_index not in PILOT_BLOCKS:
            raise ValueError(f"Stage-A block must be one of {PILOT_BLOCKS}")
        if split == "train":
            return self.train_by_block[block_index]
        if split == "holdout":
            return self.holdout_by_block[block_index]
        raise ValueError(f"unsupported Stage-A split: {split!r}")

    def cases(self, block_index: int, split: SplitName):
        return tuple(record.case for record in self.records(block_index, split))


def _case_map(dataset_manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    cases = dataset_manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("validated pilot dataset manifest has no case list")
    out: dict[str, Mapping[str, object]] = {}
    for case in cases:
        assert isinstance(case, dict)
        case_id = case["case_id"]
        assert isinstance(case_id, str)
        out[case_id] = case
    return out


def _canonical_sigma(value: float) -> str:
    return format(float(value), ".17g")


def _execution_id(case_id: str, sigma: float) -> str:
    return f"{case_id}::sigma={_canonical_sigma(sigma)}"


def _sigma_member(sigma: float, expected: Sequence[object]) -> bool:
    return any(
        isinstance(value, (int, float))
        and math.isclose(float(sigma), float(value), rel_tol=0.0, abs_tol=1e-12)
        for value in expected
    )


def _validate_bundle_provenance(
    provenance: CaptureBundleProvenance,
    *,
    dataset_manifest_sha256: str,
    common: CaptureBundleProvenance | None,
) -> None:
    if provenance.dataset_manifest_sha256.lower() != dataset_manifest_sha256.lower():
        raise ValueError(
            "capture bundle dataset manifest identity does not match the fixed Stage-A manifest"
        )
    if common is None:
        return
    for name in (
        "code_commit",
        "comfy_commit",
        "execution_descriptor",
        "purpose",
        "teacher_model_revision",
        "teacher_model_sha256",
    ):
        if getattr(provenance, name) != getattr(common, name):
            raise ValueError(
                f"Stage-A capture corpus mixes provenance field {name!r}: "
                f"{getattr(common, name)!r} vs {getattr(provenance, name)!r}"
            )


def _annotate_record(
    record: CapturedPilotCase,
    *,
    source_case_id: str,
    split: SplitName,
    artifact: CaptureArtifactRef,
) -> CapturedPilotCase:
    sigma = record.case.sigma
    if sigma is None:
        raise ValueError(f"Stage-A capture {source_case_id!r} must record sigma")
    context = dict(record.case.context)
    reserved = {
        "stage_a_source_case_id": source_case_id,
        "stage_a_split": split,
        "stage_a_capture_receipt_sha256": artifact.receipt_sha256.lower(),
        "stage_a_capture_bundle_sha256": sha256_file(artifact.bundle_path),
    }
    for key, value in reserved.items():
        if key in context and context[key] != value:
            raise ValueError(
                f"capture context contradicts immutable Stage-A field {key!r}"
            )
        context[key] = value
    case = replace(
        record.case,
        case_id=_execution_id(source_case_id, float(sigma)),
        context=context,
    )
    return replace(record, case=case)


def load_stage_a_capture_set(
    artifacts: Sequence[CaptureArtifactRef],
    dataset_manifest: Mapping[str, object],
    *,
    minimum_cases: int = 16,
    minimum_sigma_strata: int = 8,
    required_coverage_tags: Sequence[str] = (),
) -> StageACaptureSet:
    """Load and bind the fixed Stage-A live-capture corpus to its dataset manifest.

    Production defaults enforce the design's minimum 16 complete cases and eight sigma
    strata. Tests or explicitly identified experiments may raise/lower those validation
    parameters, but the returned dataset identity always hashes the exact manifest.

    Every manifest ``case_id × sigma`` execution must appear exactly once, and each
    execution bundle must contain all prescribed pilot blocks 0/25/49. Replay metric
    IDs append sigma while retaining the complete source case in immutable context;
    this permits multiple sigma strata per complete train/holdout case without leaking
    token/execution rows across the manifest split.
    """
    dataset_sha = validate_pilot_dataset_manifest(
        dataset_manifest,
        minimum_cases=minimum_cases,
        minimum_sigma_strata=minimum_sigma_strata,
        required_coverage_tags=required_coverage_tags,
    )
    if not artifacts:
        raise ValueError("Stage-A capture corpus requires persisted capture artifacts")
    manifest_cases = _case_map(dataset_manifest)

    expected_executions: set[tuple[str, str]] = set()
    for case_id, case in manifest_cases.items():
        sigmas = case["sigmas"]
        assert isinstance(sigmas, list)
        for sigma in sigmas:
            expected_executions.add((case_id, _canonical_sigma(float(sigma))))

    seen_executions: set[tuple[str, str]] = set()
    seen_receipts: set[str] = set()
    train: dict[int, list[CapturedPilotCase]] = {i: [] for i in PILOT_BLOCKS}
    holdout: dict[int, list[CapturedPilotCase]] = {i: [] for i in PILOT_BLOCKS}
    common_provenance: CaptureBundleProvenance | None = None

    for artifact in artifacts:
        receipt_sha = artifact.receipt_sha256.lower()
        if receipt_sha in seen_receipts:
            raise ValueError(f"duplicate Stage-A capture receipt identity: {receipt_sha}")
        seen_receipts.add(receipt_sha)
        records, provenance = load_captured_pilot_bundle(
            artifact.bundle_path,
            receipt_path=artifact.receipt_path,
            expected_receipt_sha256=receipt_sha,
        )
        _validate_bundle_provenance(
            provenance,
            dataset_manifest_sha256=dataset_sha,
            common=common_provenance,
        )
        if common_provenance is None:
            common_provenance = provenance

        indices = tuple(record.block_index for record in records)
        if indices != PILOT_BLOCKS:
            raise ValueError(
                f"canonical Stage-A capture bundle must contain blocks {PILOT_BLOCKS} "
                f"in order, got {indices}"
            )
        source_case_id = records[0].case.case_id
        manifest_case = manifest_cases.get(source_case_id)
        if manifest_case is None:
            raise ValueError(
                f"capture case {source_case_id!r} is absent from the fixed dataset manifest"
            )
        sigma = records[0].case.sigma
        if sigma is None:
            raise ValueError(f"capture case {source_case_id!r} is missing sigma")
        sigmas = manifest_case["sigmas"]
        assert isinstance(sigmas, list)
        if not _sigma_member(float(sigma), sigmas):
            raise ValueError(
                f"capture sigma {sigma!r} for {source_case_id!r} is absent from its manifest strata"
            )
        modality = manifest_case["modality_label"]
        if records[0].case.modality_label != modality:
            raise ValueError(
                f"capture modality for {source_case_id!r} does not match dataset manifest"
            )
        split = manifest_case["split"]
        assert split in ("train", "holdout")
        execution = (source_case_id, _canonical_sigma(float(sigma)))
        if execution in seen_executions:
            raise ValueError(
                f"duplicate Stage-A capture execution for case/sigma {execution}"
            )
        seen_executions.add(execution)

        target = train if split == "train" else holdout
        for record in records:
            target[record.block_index].append(
                _annotate_record(
                    record,
                    source_case_id=source_case_id,
                    split=split,
                    artifact=artifact,
                )
            )

    missing = sorted(expected_executions - seen_executions)
    extra = sorted(seen_executions - expected_executions)
    if missing or extra:
        raise ValueError(
            "Stage-A capture corpus does not exactly cover dataset case/sigma executions; "
            f"missing={missing}, extra={extra}"
        )
    assert common_provenance is not None

    train_out = {i: tuple(train[i]) for i in PILOT_BLOCKS}
    holdout_out = {i: tuple(holdout[i]) for i in PILOT_BLOCKS}
    if any(not train_out[i] or not holdout_out[i] for i in PILOT_BLOCKS):
        raise ValueError("Stage-A capture corpus must provide train and holdout executions for every pilot block")

    return StageACaptureSet(
        dataset_manifest_sha256=dataset_sha,
        code_commit=common_provenance.code_commit,
        comfy_commit=common_provenance.comfy_commit,
        execution_descriptor=common_provenance.execution_descriptor,
        train_by_block=train_out,
        holdout_by_block=holdout_out,
        artifact_refs=tuple(artifacts),
    )
