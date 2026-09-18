from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .activation_capture import CapturedPilotCase
from .capture_io import (
    CAPTURE_RECEIPT_SCHEMA,
    CaptureBundleProvenance,
    load_captured_pilot_bundle,
)
from .checkpoint import sha256_file
from .pilot_campaign import PILOT_BLOCKS, validate_pilot_dataset_manifest
from .pilot_capture_set import CaptureArtifactRef


SplitName = Literal["train", "holdout"]


@dataclass(frozen=True)
class StageALazyCaptureExecution:
    artifact: CaptureArtifactRef
    bundle_sha256: str
    source_case_id: str
    split: SplitName
    sigma: float
    modality_label: str


@dataclass(frozen=True)
class StageALazyCaptureSet:
    """Hash-validated Stage-A corpus whose activation tensors are loaded one block at a time.

    Construction validates every receipt, bundle hash, manifest case/sigma identity and shared
    execution provenance without deserializing activation tensors. ``records()`` revalidates the
    selected bundles and materializes only one requested pilot depth/split. This keeps the active
    CPU activation footprint bounded to one Stage-A block instead of all 0/25/49 captures.
    """

    dataset_manifest_sha256: str
    code_commit: str
    comfy_commit: str
    execution_descriptor: str
    executions: tuple[StageALazyCaptureExecution, ...]
    artifact_refs: tuple[CaptureArtifactRef, ...]
    teacher_model_revision: str
    teacher_model_sha256: str

    def records(self, block_index: int, split: SplitName) -> tuple[CapturedPilotCase, ...]:
        if block_index not in PILOT_BLOCKS:
            raise ValueError(f"Stage-A block must be one of {PILOT_BLOCKS}")
        if split not in ("train", "holdout"):
            raise ValueError(f"unsupported Stage-A split: {split!r}")
        selected = [execution for execution in self.executions if execution.split == split]
        if not selected:
            raise ValueError(f"Stage-A lazy corpus has no {split!r} executions")
        out: list[CapturedPilotCase] = []
        for execution in selected:
            records, provenance = load_captured_pilot_bundle(
                execution.artifact.bundle_path,
                receipt_path=execution.artifact.receipt_path,
                expected_receipt_sha256=execution.artifact.receipt_sha256,
            )
            _validate_loaded_provenance(self, provenance)
            indices = tuple(record.block_index for record in records)
            if indices != PILOT_BLOCKS:
                raise ValueError(
                    f"canonical Stage-A capture bundle must contain blocks {PILOT_BLOCKS} "
                    f"in order, got {indices}"
                )
            source_case_id = records[0].case.case_id
            sigma = records[0].case.sigma
            modality = records[0].case.modality_label
            if source_case_id != execution.source_case_id:
                raise ValueError("Stage-A capture case identity changed after lazy indexing")
            if sigma is None or _canonical_sigma(float(sigma)) != _canonical_sigma(execution.sigma):
                raise ValueError("Stage-A capture sigma identity changed after lazy indexing")
            if modality != execution.modality_label:
                raise ValueError("Stage-A capture modality identity changed after lazy indexing")
            record = next(record for record in records if record.block_index == block_index)
            out.append(_annotate_record(record, execution=execution))
            del records
        return tuple(out)

    def cases(self, block_index: int, split: SplitName):
        return tuple(record.case for record in self.records(block_index, split))


def _canonical_sigma(value: float) -> str:
    return format(float(value), ".17g")


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-hex SHA-256 string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a 64-hex SHA-256 string") from exc
    return value.lower()


def _read_receipt(artifact: CaptureArtifactRef) -> tuple[dict[str, Any], str]:
    receipt_path = Path(artifact.receipt_path)
    actual_receipt_sha = sha256_file(receipt_path)
    if actual_receipt_sha.lower() != artifact.receipt_sha256.lower():
        raise ValueError("Stage-A capture receipt SHA-256 does not match registry identity")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Stage-A capture receipt: {receipt_path}") from exc
    if not isinstance(receipt, dict) or receipt.get("schema") != CAPTURE_RECEIPT_SCHEMA:
        raise ValueError("Stage-A capture receipt has an unsupported schema")

    bundle_path = Path(artifact.bundle_path)
    if receipt.get("bundle_filename") != bundle_path.name:
        raise ValueError("Stage-A capture receipt bundle_filename does not match registry bundle")
    bundle_sha = _require_sha256("Stage-A bundle_sha256", receipt.get("bundle_sha256"))
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Stage-A capture bundle does not exist: {bundle_path}")
    if int(receipt.get("bundle_bytes", -1)) != bundle_path.stat().st_size:
        raise ValueError("Stage-A capture receipt bundle_bytes does not match bundle")
    actual_bundle_sha = sha256_file(bundle_path)
    if actual_bundle_sha.lower() != bundle_sha:
        raise ValueError("Stage-A capture receipt bundle_sha256 does not match bundle")
    return receipt, bundle_sha


def _case_map(dataset_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cases = dataset_manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("validated Stage-A dataset manifest has no case list")
    out: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        assert isinstance(case, dict)
        case_id = case["case_id"]
        assert isinstance(case_id, str)
        out[case_id] = case
    return out


def _sigma_member(sigma: float, expected: Sequence[object]) -> bool:
    return any(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isclose(float(sigma), float(value), rel_tol=0.0, abs_tol=1e-12)
        for value in expected
    )


def _provenance_from_receipt(receipt: Mapping[str, Any]) -> CaptureBundleProvenance:
    raw = receipt.get("provenance")
    if not isinstance(raw, dict):
        raise ValueError("Stage-A capture receipt is missing provenance")
    try:
        return CaptureBundleProvenance(**raw)
    except TypeError as exc:
        raise ValueError("Stage-A capture receipt has invalid provenance fields") from exc


def _same_provenance(left: CaptureBundleProvenance, right: CaptureBundleProvenance) -> bool:
    return (
        left.code_commit == right.code_commit
        and left.comfy_commit == right.comfy_commit
        and left.dataset_manifest_sha256.lower() == right.dataset_manifest_sha256.lower()
        and left.execution_descriptor == right.execution_descriptor
        and left.purpose == right.purpose
        and left.teacher_model_revision == right.teacher_model_revision
        and left.teacher_model_sha256.lower() == right.teacher_model_sha256.lower()
    )


def _validate_loaded_provenance(
    capture_set: StageALazyCaptureSet,
    provenance: CaptureBundleProvenance,
) -> None:
    expected = (
        capture_set.code_commit,
        capture_set.comfy_commit,
        capture_set.dataset_manifest_sha256.lower(),
        capture_set.execution_descriptor,
        capture_set.teacher_model_revision,
        capture_set.teacher_model_sha256.lower(),
    )
    actual = (
        provenance.code_commit,
        provenance.comfy_commit,
        provenance.dataset_manifest_sha256.lower(),
        provenance.execution_descriptor,
        provenance.teacher_model_revision,
        provenance.teacher_model_sha256.lower(),
    )
    if actual != expected:
        raise ValueError("Stage-A lazy capture bundle provenance changed after indexing")


def _annotate_record(
    record: CapturedPilotCase,
    *,
    execution: StageALazyCaptureExecution,
) -> CapturedPilotCase:
    context = dict(record.case.context)
    reserved = {
        "stage_a_source_case_id": execution.source_case_id,
        "stage_a_split": execution.split,
        "stage_a_capture_receipt_sha256": execution.artifact.receipt_sha256.lower(),
        "stage_a_capture_bundle_sha256": execution.bundle_sha256.lower(),
    }
    for key, value in reserved.items():
        if key in context and context[key] != value:
            raise ValueError(f"capture context contradicts immutable Stage-A field {key!r}")
        context[key] = value
    case = replace(
        record.case,
        case_id=f"{execution.source_case_id}::sigma={_canonical_sigma(execution.sigma)}",
        context=context,
    )
    return replace(record, case=case)


def load_stage_a_capture_set_lazy(
    artifacts: Sequence[CaptureArtifactRef],
    dataset_manifest: Mapping[str, Any],
    *,
    minimum_cases: int = 16,
    minimum_sigma_strata: int = 8,
    required_coverage_tags: Sequence[str] = (),
) -> StageALazyCaptureSet:
    """Index a fixed Stage-A corpus without retaining all captured activation tensors."""
    dataset_sha = validate_pilot_dataset_manifest(
        dataset_manifest,
        minimum_cases=minimum_cases,
        minimum_sigma_strata=minimum_sigma_strata,
        required_coverage_tags=required_coverage_tags,
    )
    if not artifacts:
        raise ValueError("Stage-A capture corpus requires persisted capture artifacts")
    cases = _case_map(dataset_manifest)
    expected: set[tuple[str, str]] = set()
    for case_id, case in cases.items():
        sigmas = case["sigmas"]
        assert isinstance(sigmas, list)
        for sigma in sigmas:
            expected.add((case_id, _canonical_sigma(float(sigma))))

    seen: set[tuple[str, str]] = set()
    seen_receipts: set[str] = set()
    executions: list[StageALazyCaptureExecution] = []
    common: CaptureBundleProvenance | None = None
    for artifact in artifacts:
        receipt_sha = artifact.receipt_sha256.lower()
        if receipt_sha in seen_receipts:
            raise ValueError(f"duplicate Stage-A capture receipt identity: {receipt_sha}")
        seen_receipts.add(receipt_sha)
        receipt, bundle_sha = _read_receipt(artifact)
        provenance = _provenance_from_receipt(receipt)
        if provenance.dataset_manifest_sha256.lower() != dataset_sha.lower():
            raise ValueError(
                "capture receipt dataset manifest identity does not match the fixed Stage-A manifest"
            )
        if common is None:
            common = provenance
        elif not _same_provenance(common, provenance):
            raise ValueError("Stage-A capture corpus mixes provenance across receipt artifacts")

        if receipt.get("block_indices") != list(PILOT_BLOCKS):
            raise ValueError(
                f"canonical Stage-A capture receipt must contain blocks {PILOT_BLOCKS} in order"
            )
        source_case_id = receipt.get("case_id")
        if not isinstance(source_case_id, str) or source_case_id not in cases:
            raise ValueError("Stage-A capture receipt case_id is absent from the fixed manifest")
        sigma = receipt.get("sigma")
        if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not math.isfinite(float(sigma)):
            raise ValueError(f"Stage-A capture receipt has invalid sigma for {source_case_id!r}")
        sigma = float(sigma)
        case = cases[source_case_id]
        sigmas = case["sigmas"]
        assert isinstance(sigmas, list)
        if not _sigma_member(sigma, sigmas):
            raise ValueError(
                f"capture sigma {sigma!r} for {source_case_id!r} is absent from its manifest strata"
            )
        modality = receipt.get("modality_label")
        if modality != case["modality_label"]:
            raise ValueError(
                f"capture modality for {source_case_id!r} does not match dataset manifest"
            )
        split = case["split"]
        assert split in ("train", "holdout")
        execution_id = (source_case_id, _canonical_sigma(sigma))
        if execution_id in seen:
            raise ValueError(f"duplicate Stage-A capture execution for case/sigma {execution_id}")
        seen.add(execution_id)
        executions.append(
            StageALazyCaptureExecution(
                artifact=artifact,
                bundle_sha256=bundle_sha,
                source_case_id=source_case_id,
                split=split,
                sigma=sigma,
                modality_label=str(modality),
            )
        )

    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra:
        raise ValueError(
            "Stage-A capture corpus does not exactly cover dataset case/sigma executions; "
            f"missing={missing}, extra={extra}"
        )
    assert common is not None
    split_names = {execution.split for execution in executions}
    if split_names != {"train", "holdout"}:
        raise ValueError("Stage-A capture corpus must provide train and holdout executions")
    return StageALazyCaptureSet(
        dataset_manifest_sha256=dataset_sha,
        code_commit=common.code_commit,
        comfy_commit=common.comfy_commit,
        execution_descriptor=common.execution_descriptor,
        executions=tuple(executions),
        artifact_refs=tuple(artifacts),
        teacher_model_revision=common.teacher_model_revision,
        teacher_model_sha256=common.teacher_model_sha256,
    )
