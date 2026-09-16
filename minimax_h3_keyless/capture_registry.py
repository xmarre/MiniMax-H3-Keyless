from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capture_io import (
    CAPTURE_RECEIPT_SCHEMA,
    CaptureBundleProvenance,
    load_captured_pilot_bundle,
)
from .checkpoint import sha256_file
from .pilot_campaign import (
    PILOT_BLOCKS,
    load_json_manifest,
    validate_pilot_dataset_manifest,
    write_json_atomic,
)
from .pilot_inputs import CANONICAL_STAGE_A_COVERAGE_TAGS, CAPTURE_REGISTRY_SCHEMA


@dataclass(frozen=True)
class StageACaptureRegistryBuildResult:
    registry_path: str
    registry_sha256: str
    dataset_manifest_sha256: str
    artifact_count: int
    code_commit: str
    comfy_commit: str
    execution_descriptor: str


def _canonical_sigma(value: float) -> str:
    return format(float(value), ".17g")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Stage-A capture receipt: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Stage-A capture receipt must decode to an object: {path}")
    return value


def _manifest_case_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cases = manifest.get("cases")
    assert isinstance(cases, list)
    return {str(case["case_id"]): case for case in cases if isinstance(case, dict)}


def _expected_executions(manifest: Mapping[str, Any]) -> set[tuple[str, str]]:
    expected: set[tuple[str, str]] = set()
    for case_id, case in _manifest_case_map(manifest).items():
        sigmas = case["sigmas"]
        assert isinstance(sigmas, list)
        for sigma in sigmas:
            expected.add((case_id, _canonical_sigma(float(sigma))))
    return expected


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


def _relative_path(path: Path, *, registry_parent: Path) -> str:
    return Path(os.path.relpath(path.resolve(), registry_parent.resolve())).as_posix()


def build_stage_a_capture_registry(
    dataset_manifest_path: str | Path,
    receipt_paths: Sequence[str | Path],
    output_path: str | Path,
) -> StageACaptureRegistryBuildResult:
    """Build one immutable Stage-A capture registry after streaming bundle validation.

    Each capture bundle is hash-checked and deserialized one at a time, then released.
    This proves exact case×sigma coverage without retaining the full activation corpus in
    memory merely to construct the registry. The offline pilot runner performs its own
    independent registry/bundle validation before training.
    """
    dataset_manifest = load_json_manifest(dataset_manifest_path)
    dataset_sha = validate_pilot_dataset_manifest(
        dataset_manifest,
        required_coverage_tags=CANONICAL_STAGE_A_COVERAGE_TAGS,
    )
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(
            f"Stage-A capture registry is immutable; choose a new output path: {output_path}"
        )
    if not receipt_paths:
        raise ValueError("Stage-A capture registry requires at least one capture receipt")

    cases = _manifest_case_map(dataset_manifest)
    expected = _expected_executions(dataset_manifest)
    seen: set[tuple[str, str]] = set()
    seen_receipt_hashes: set[str] = set()
    common_provenance: CaptureBundleProvenance | None = None
    validated: list[tuple[str, float, Path, Path, str]] = []

    for raw_receipt_path in sorted((Path(path) for path in receipt_paths), key=lambda p: str(p)):
        receipt_path = raw_receipt_path.resolve()
        receipt_sha = sha256_file(receipt_path)
        if receipt_sha in seen_receipt_hashes:
            raise ValueError(f"duplicate Stage-A capture receipt SHA-256: {receipt_sha}")
        seen_receipt_hashes.add(receipt_sha)
        receipt = _read_json_object(receipt_path)
        if receipt.get("schema") != CAPTURE_RECEIPT_SCHEMA:
            raise ValueError(f"unsupported Stage-A capture receipt schema: {receipt_path}")
        bundle_filename = receipt.get("bundle_filename")
        if not isinstance(bundle_filename, str) or not bundle_filename or Path(bundle_filename).name != bundle_filename:
            raise ValueError(f"capture receipt has unsafe bundle_filename: {receipt_path}")
        bundle_path = receipt_path.parent / bundle_filename
        records, provenance = load_captured_pilot_bundle(
            bundle_path,
            receipt_path=receipt_path,
            expected_receipt_sha256=receipt_sha,
        )
        if provenance.dataset_manifest_sha256.lower() != dataset_sha.lower():
            raise ValueError(
                f"capture receipt dataset identity does not match Stage-A manifest: {receipt_path}"
            )
        if common_provenance is None:
            common_provenance = provenance
        elif not _same_provenance(common_provenance, provenance):
            raise ValueError(
                "Stage-A capture registry may not mix code/Comfy/execution/teacher provenance"
            )
        indices = tuple(record.block_index for record in records)
        if indices != PILOT_BLOCKS:
            raise ValueError(
                f"canonical Stage-A capture must contain blocks {PILOT_BLOCKS} in order, got {indices}"
            )
        case_id = records[0].case.case_id
        sigma = records[0].case.sigma
        if sigma is None or not math.isfinite(float(sigma)):
            raise ValueError(f"Stage-A capture has invalid sigma: {receipt_path}")
        manifest_case = cases.get(case_id)
        if manifest_case is None:
            raise ValueError(f"capture case {case_id!r} is absent from the fixed Stage-A manifest")
        if records[0].case.modality_label != manifest_case["modality_label"]:
            raise ValueError(f"capture modality for {case_id!r} does not match the fixed manifest")
        execution = (case_id, _canonical_sigma(float(sigma)))
        if execution not in expected:
            raise ValueError(
                f"capture case/sigma execution is absent from the fixed Stage-A manifest: {execution}"
            )
        if execution in seen:
            raise ValueError(f"duplicate Stage-A case/sigma capture execution: {execution}")
        seen.add(execution)
        validated.append((case_id, float(sigma), bundle_path.resolve(), receipt_path, receipt_sha))
        del records

    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra:
        raise ValueError(
            "capture receipts do not exactly cover the fixed Stage-A case/sigma corpus; "
            f"missing={missing}, extra={extra}"
        )
    assert common_provenance is not None

    registry_parent = output_path.resolve().parent
    artifacts = []
    for case_id, sigma, bundle_path, receipt_path, receipt_sha in sorted(
        validated,
        key=lambda row: (row[0], row[1], str(row[3])),
    ):
        artifacts.append(
            {
                "bundle_path": _relative_path(bundle_path, registry_parent=registry_parent),
                "receipt_path": _relative_path(receipt_path, registry_parent=registry_parent),
                "receipt_sha256": receipt_sha,
            }
        )
    payload = {
        "schema": CAPTURE_REGISTRY_SCHEMA,
        "dataset_manifest_sha256": dataset_sha,
        "artifacts": artifacts,
    }
    registry_sha = write_json_atomic(output_path, payload)
    return StageACaptureRegistryBuildResult(
        registry_path=str(output_path),
        registry_sha256=registry_sha,
        dataset_manifest_sha256=dataset_sha,
        artifact_count=len(artifacts),
        code_commit=common_provenance.code_commit,
        comfy_commit=common_provenance.comfy_commit,
        execution_descriptor=common_provenance.execution_descriptor,
    )
