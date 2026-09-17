from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .checkpoint import sha256_file
from .contracts import TARGET_MODEL_REVISION, TEACHER_SHA256
from .immutable_io import write_json_no_replace
from .pilot_campaign import _require_sha256, load_json_manifest, validate_pilot_dataset_manifest
from .progressive import ProgressivePrefix
from .progressive_authorization import load_progressive_prefix_manifest
from .progressive_capture_io import (
    PROGRESSIVE_CAPTURE_RECEIPT_SCHEMA,
    ProgressiveCaptureProvenance,
    load_progressive_capture_bundle,
)
from .progressive_capture_set import ProgressiveCaptureArtifactRef


PROGRESSIVE_CAPTURE_REGISTRY_SCHEMA = "minimax_h3_keyless_progressive_capture_registry_v1"


@dataclass(frozen=True)
class ProgressiveCaptureRegistryBuildResult:
    registry_path: str
    registry_sha256: str
    prefix_manifest_sha256: str
    prefix_identity_sha256: str
    dataset_manifest_sha256: str
    artifact_count: int
    target_block: int
    code_commit: str
    comfy_commit: str
    execution_descriptor: str


@dataclass(frozen=True)
class ProgressiveCaptureRegistry:
    registry_path: str
    registry_file_sha256: str
    prefix_manifest_sha256: str
    prefix_identity_sha256: str
    stage_a_campaign_sha256: str
    dataset_manifest_sha256: str
    gate_manifest_sha256: str
    code_commit: str
    comfy_commit: str
    execution_descriptor: str
    target_block: int
    teacher_model_revision: str
    teacher_model_sha256: str
    artifacts: tuple[ProgressiveCaptureArtifactRef, ...]


def _canonical_sigma(value: float) -> str:
    return format(float(value), ".17g")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid progressive capture JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"progressive capture JSON must decode to an object: {path}")
    return value


def _case_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("validated progressive dataset manifest has no case list")
    out: dict[str, Mapping[str, Any]] = {}
    for row in cases:
        assert isinstance(row, dict)
        case_id = row["case_id"]
        assert isinstance(case_id, str)
        out[case_id] = row
    return out


def _expected_executions(manifest: Mapping[str, Any]) -> set[tuple[str, str]]:
    expected: set[tuple[str, str]] = set()
    for case_id, row in _case_map(manifest).items():
        sigmas = row["sigmas"]
        assert isinstance(sigmas, list)
        for sigma in sigmas:
            expected.add((case_id, _canonical_sigma(float(sigma))))
    return expected


def _sigma_member(sigma: float, expected: Sequence[object]) -> bool:
    return any(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isclose(float(sigma), float(value), rel_tol=0.0, abs_tol=1e-12)
        for value in expected
    )


def _same_provenance(
    left: ProgressiveCaptureProvenance,
    right: ProgressiveCaptureProvenance,
) -> bool:
    return (
        left.code_commit == right.code_commit
        and left.comfy_commit == right.comfy_commit
        and left.dataset_manifest_sha256.lower() == right.dataset_manifest_sha256.lower()
        and left.gate_manifest_sha256.lower() == right.gate_manifest_sha256.lower()
        and left.stage_a_campaign_sha256.lower() == right.stage_a_campaign_sha256.lower()
        and left.prefix_identity_sha256.lower() == right.prefix_identity_sha256.lower()
        and left.target_block == right.target_block
        and left.execution_descriptor == right.execution_descriptor
        and left.purpose == right.purpose
        and left.teacher_model_revision == right.teacher_model_revision
        and left.teacher_model_sha256.lower() == right.teacher_model_sha256.lower()
    )


def _validate_provenance(
    provenance: ProgressiveCaptureProvenance,
    *,
    prefix: ProgressivePrefix,
    dataset_sha256: str,
    common: ProgressiveCaptureProvenance | None,
) -> None:
    target = prefix.next_block
    if target is None:
        raise ValueError("cannot build a progressive capture registry for a complete prefix")
    expected = {
        "dataset_manifest_sha256": dataset_sha256.lower(),
        "gate_manifest_sha256": prefix.gate_manifest_sha256.lower(),
        "stage_a_campaign_sha256": prefix.stage_a_campaign_sha256.lower(),
        "prefix_identity_sha256": prefix.identity_sha256.lower(),
    }
    for name, value in expected.items():
        actual = str(getattr(provenance, name)).lower()
        if actual != value:
            raise ValueError(
                f"progressive capture provenance {name!r} does not match the accepted prefix"
            )
    if provenance.target_block != target:
        raise ValueError(
            "progressive capture target_block does not match the next accepted-prefix block"
        )
    if provenance.code_commit.lower() != prefix.code_commit.lower():
        raise ValueError(
            "progressive capture code_commit does not match the fixed sweep revision"
        )
    if common is not None and not _same_provenance(common, provenance):
        raise ValueError(
            "progressive capture registry may not mix code/Comfy/execution/teacher/prefix provenance"
        )


def _relative_path(path: Path, *, registry_parent: Path) -> str:
    return Path(os.path.relpath(path.resolve(), registry_parent.resolve())).as_posix()


def _artifact_from_registry_row(
    row: Any,
    *,
    registry_parent: Path,
    index: int,
) -> ProgressiveCaptureArtifactRef:
    expected_keys = {"bundle_path", "receipt_path", "receipt_sha256"}
    if not isinstance(row, dict) or set(row) != expected_keys:
        raise ValueError(f"progressive capture registry artifact row {index} has an invalid schema")
    bundle_raw = row["bundle_path"]
    receipt_raw = row["receipt_path"]
    if not isinstance(bundle_raw, str) or not bundle_raw.strip():
        raise ValueError(f"progressive capture registry artifact row {index} has invalid bundle_path")
    if not isinstance(receipt_raw, str) or not receipt_raw.strip():
        raise ValueError(f"progressive capture registry artifact row {index} has invalid receipt_path")
    if Path(bundle_raw).is_absolute() or Path(receipt_raw).is_absolute():
        raise ValueError("progressive capture registry artifact paths must be relative")
    return ProgressiveCaptureArtifactRef(
        bundle_path=str((registry_parent / bundle_raw).resolve()),
        receipt_path=str((registry_parent / receipt_raw).resolve()),
        receipt_sha256=row["receipt_sha256"],
    )


def build_progressive_capture_registry(
    dataset_manifest_path: str | Path,
    prefix_manifest_path: str | Path,
    receipt_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    minimum_cases: int = 16,
    minimum_sigma_strata: int = 8,
    required_coverage_tags: Sequence[str] = (),
) -> ProgressiveCaptureRegistryBuildResult:
    """Build one immutable, complete Stage-B capture registry for an accepted prefix.

    Each receipt and bundle is hash-checked and deserialized one at a time, then released.
    The registry is published only after exact fixed-dataset case×sigma coverage and common
    provenance have been proved. This avoids retaining the complete activation corpus merely
    to establish its immutable index.
    """

    dataset = load_json_manifest(dataset_manifest_path)
    dataset_sha = validate_pilot_dataset_manifest(
        dataset,
        minimum_cases=minimum_cases,
        minimum_sigma_strata=minimum_sigma_strata,
        required_coverage_tags=required_coverage_tags,
    )
    prefix, prefix_manifest_sha = load_progressive_prefix_manifest(prefix_manifest_path)
    if prefix.dataset_manifest_sha256.lower() != dataset_sha.lower():
        raise ValueError(
            "progressive prefix dataset manifest identity does not match the supplied dataset"
        )
    target = prefix.next_block
    if target is None:
        raise ValueError("progressive prefix is complete; there is no next-block capture registry")

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(
            f"progressive capture registry is immutable; choose a new output path: {output_path}"
        )
    if not receipt_paths:
        raise ValueError("progressive capture registry requires at least one capture receipt")

    cases = _case_map(dataset)
    expected = _expected_executions(dataset)
    seen: set[tuple[str, str]] = set()
    seen_receipts: set[str] = set()
    common: ProgressiveCaptureProvenance | None = None
    validated: list[tuple[str, float, Path, Path, str]] = []

    for raw_receipt in sorted((Path(path) for path in receipt_paths), key=lambda path: str(path)):
        receipt_path = raw_receipt.resolve()
        receipt_sha = sha256_file(receipt_path)
        if receipt_sha in seen_receipts:
            raise ValueError(f"duplicate progressive capture receipt SHA-256: {receipt_sha}")
        seen_receipts.add(receipt_sha)
        receipt = _read_json_object(receipt_path)
        if receipt.get("schema") != PROGRESSIVE_CAPTURE_RECEIPT_SCHEMA:
            raise ValueError(f"unsupported progressive capture receipt schema: {receipt_path}")
        bundle_filename = receipt.get("bundle_filename")
        if (
            not isinstance(bundle_filename, str)
            or not bundle_filename
            or Path(bundle_filename).name != bundle_filename
        ):
            raise ValueError(f"progressive capture receipt has unsafe bundle_filename: {receipt_path}")
        bundle_path = (receipt_path.parent / bundle_filename).resolve()
        record, provenance = load_progressive_capture_bundle(
            bundle_path,
            receipt_path=receipt_path,
            expected_receipt_sha256=receipt_sha,
        )
        _validate_provenance(
            provenance,
            prefix=prefix,
            dataset_sha256=dataset_sha,
            common=common,
        )
        if common is None:
            common = provenance

        case_id = record.case.case_id
        manifest_case = cases.get(case_id)
        if manifest_case is None:
            raise ValueError(
                f"progressive capture case {case_id!r} is absent from the fixed dataset"
            )
        sigma = record.case.sigma
        if sigma is None or not math.isfinite(float(sigma)):
            raise ValueError(f"progressive capture case {case_id!r} has invalid sigma")
        sigmas = manifest_case["sigmas"]
        assert isinstance(sigmas, list)
        if not _sigma_member(float(sigma), sigmas):
            raise ValueError(
                f"progressive capture sigma {sigma!r} for {case_id!r} is absent from the fixed dataset"
            )
        if record.case.modality_label != manifest_case["modality_label"]:
            raise ValueError(
                f"progressive capture modality for {case_id!r} does not match the fixed dataset"
            )
        context = record.case.context
        if not isinstance(context, Mapping):
            raise ValueError("progressive capture case context must be a mapping")
        if context.get("progressive_source_case_id") != case_id:
            raise ValueError("progressive capture source-case annotation does not match the bundle")
        if context.get("progressive_split") != manifest_case["split"]:
            raise ValueError("progressive capture split annotation does not match the fixed dataset")

        execution = (case_id, _canonical_sigma(float(sigma)))
        if execution in seen:
            raise ValueError(f"duplicate progressive case/sigma capture execution: {execution}")
        seen.add(execution)
        validated.append((case_id, float(sigma), bundle_path, receipt_path, receipt_sha))
        del record

    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra:
        raise ValueError(
            "progressive capture receipts do not exactly cover the fixed dataset corpus; "
            f"missing={missing}, extra={extra}"
        )
    assert common is not None

    registry_parent = output_path.resolve().parent
    artifacts = [
        {
            "bundle_path": _relative_path(bundle_path, registry_parent=registry_parent),
            "receipt_path": _relative_path(receipt_path, registry_parent=registry_parent),
            "receipt_sha256": receipt_sha,
        }
        for _, _, bundle_path, receipt_path, receipt_sha in sorted(
            validated,
            key=lambda row: (row[0], row[1], str(row[3])),
        )
    ]
    payload = {
        "schema": PROGRESSIVE_CAPTURE_REGISTRY_SCHEMA,
        "prefix_manifest_sha256": prefix_manifest_sha,
        "prefix_identity_sha256": prefix.identity_sha256,
        "stage_a_campaign_sha256": prefix.stage_a_campaign_sha256,
        "dataset_manifest_sha256": dataset_sha,
        "gate_manifest_sha256": prefix.gate_manifest_sha256,
        "code_commit": common.code_commit,
        "comfy_commit": common.comfy_commit,
        "execution_descriptor": common.execution_descriptor,
        "target_block": target,
        "teacher_model_revision": common.teacher_model_revision,
        "teacher_model_sha256": common.teacher_model_sha256,
        "artifacts": artifacts,
    }
    try:
        registry_sha = write_json_no_replace(output_path, payload)
    except FileExistsError as exc:
        raise FileExistsError(
            f"progressive capture registry is immutable; choose a new output path: {output_path}"
        ) from exc

    return ProgressiveCaptureRegistryBuildResult(
        registry_path=str(output_path),
        registry_sha256=registry_sha,
        prefix_manifest_sha256=prefix_manifest_sha,
        prefix_identity_sha256=prefix.identity_sha256,
        dataset_manifest_sha256=dataset_sha,
        artifact_count=len(artifacts),
        target_block=target,
        code_commit=common.code_commit,
        comfy_commit=common.comfy_commit,
        execution_descriptor=common.execution_descriptor,
    )


def load_progressive_capture_registry(path: str | Path) -> ProgressiveCaptureRegistry:
    """Load a registry strictly; bundle contents are revalidated by the training loader."""

    path = Path(path)
    value = _read_json_object(path)
    expected_keys = {
        "schema",
        "prefix_manifest_sha256",
        "prefix_identity_sha256",
        "stage_a_campaign_sha256",
        "dataset_manifest_sha256",
        "gate_manifest_sha256",
        "code_commit",
        "comfy_commit",
        "execution_descriptor",
        "target_block",
        "teacher_model_revision",
        "teacher_model_sha256",
        "artifacts",
    }
    if set(value) != expected_keys or value.get("schema") != PROGRESSIVE_CAPTURE_REGISTRY_SCHEMA:
        raise ValueError("progressive capture registry has an incompatible schema")

    hashes = {
        name: _require_sha256(f"progressive capture registry {name}", value[name])
        for name in (
            "prefix_manifest_sha256",
            "prefix_identity_sha256",
            "stage_a_campaign_sha256",
            "dataset_manifest_sha256",
            "gate_manifest_sha256",
            "teacher_model_sha256",
        )
    }
    target = value["target_block"]
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < 50:
        raise ValueError("progressive capture registry target_block must be within [0,50)")
    for name in ("code_commit", "comfy_commit", "execution_descriptor"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise ValueError(f"progressive capture registry {name} must be non-empty")
    if value["teacher_model_revision"] != TARGET_MODEL_REVISION:
        raise ValueError("progressive capture registry teacher revision is not the pinned teacher")
    if hashes["teacher_model_sha256"] != TEACHER_SHA256:
        raise ValueError("progressive capture registry teacher SHA-256 is not the pinned teacher")

    rows = value["artifacts"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("progressive capture registry artifacts must be a non-empty list")
    artifacts = tuple(
        _artifact_from_registry_row(row, registry_parent=path.resolve().parent, index=index)
        for index, row in enumerate(rows)
    )
    receipt_hashes = [artifact.receipt_sha256.lower() for artifact in artifacts]
    if len(receipt_hashes) != len(set(receipt_hashes)):
        raise ValueError("progressive capture registry contains duplicate receipt identities")

    return ProgressiveCaptureRegistry(
        registry_path=str(path),
        registry_file_sha256=sha256_file(path),
        prefix_manifest_sha256=hashes["prefix_manifest_sha256"],
        prefix_identity_sha256=hashes["prefix_identity_sha256"],
        stage_a_campaign_sha256=hashes["stage_a_campaign_sha256"],
        dataset_manifest_sha256=hashes["dataset_manifest_sha256"],
        gate_manifest_sha256=hashes["gate_manifest_sha256"],
        code_commit=value["code_commit"],
        comfy_commit=value["comfy_commit"],
        execution_descriptor=value["execution_descriptor"],
        target_block=target,
        teacher_model_revision=value["teacher_model_revision"],
        teacher_model_sha256=hashes["teacher_model_sha256"],
        artifacts=artifacts,
    )
