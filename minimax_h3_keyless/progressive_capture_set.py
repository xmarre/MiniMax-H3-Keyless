from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, Mapping, Sequence

from .activation_capture import CapturedPilotCase
from .checkpoint import sha256_file
from .pilot_campaign import _require_sha256, validate_pilot_dataset_manifest
from .progressive import ProgressivePrefix
from .progressive_capture_io import (
    ProgressiveCaptureProvenance,
    load_progressive_capture_bundle,
)


SplitName = Literal["train", "holdout"]


@dataclass(frozen=True)
class ProgressiveCaptureArtifactRef:
    """Immutable reference to one Phase-4 live-input capture artifact."""

    bundle_path: str
    receipt_path: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not str(self.bundle_path).strip() or not str(self.receipt_path).strip():
            raise ValueError("progressive capture artifact paths must be non-empty")
        object.__setattr__(
            self,
            "receipt_sha256",
            _require_sha256("progressive capture receipt SHA-256", self.receipt_sha256),
        )


@dataclass(frozen=True)
class ProgressiveBlockCaptureSet:
    """Validated live-input corpus for exactly one progressive target block/prefix."""

    target_block: int
    prefix_identity_sha256: str
    stage_a_campaign_sha256: str
    dataset_manifest_sha256: str
    gate_manifest_sha256: str
    code_commit: str
    comfy_commit: str
    execution_descriptor: str
    train: tuple[CapturedPilotCase, ...]
    holdout: tuple[CapturedPilotCase, ...]
    artifact_refs: tuple[ProgressiveCaptureArtifactRef, ...]

    def records(self, split: SplitName) -> tuple[CapturedPilotCase, ...]:
        if split == "train":
            return self.train
        if split == "holdout":
            return self.holdout
        raise ValueError(f"unsupported progressive split: {split!r}")

    def cases(self, split: SplitName):
        return tuple(record.case for record in self.records(split))


def _case_map(dataset_manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    cases = dataset_manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("validated progressive dataset manifest has no case list")
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
        and not isinstance(value, bool)
        and math.isclose(float(sigma), float(value), rel_tol=0.0, abs_tol=1e-12)
        for value in expected
    )


def _validate_provenance(
    provenance: ProgressiveCaptureProvenance,
    *,
    prefix: ProgressivePrefix,
    dataset_manifest_sha256: str,
    common: ProgressiveCaptureProvenance | None,
) -> None:
    target = prefix.next_block
    if target is None:
        raise ValueError("cannot load progressive captures for a complete prefix")
    expected = {
        "dataset_manifest_sha256": dataset_manifest_sha256.lower(),
        "gate_manifest_sha256": prefix.gate_manifest_sha256.lower(),
        "stage_a_campaign_sha256": prefix.stage_a_campaign_sha256.lower(),
        "prefix_identity_sha256": prefix.identity_sha256.lower(),
    }
    actual = {
        "dataset_manifest_sha256": provenance.dataset_manifest_sha256.lower(),
        "gate_manifest_sha256": provenance.gate_manifest_sha256.lower(),
        "stage_a_campaign_sha256": provenance.stage_a_campaign_sha256.lower(),
        "prefix_identity_sha256": provenance.prefix_identity_sha256.lower(),
    }
    for name, value in expected.items():
        if actual[name] != value:
            raise ValueError(
                f"progressive capture provenance {name!r} does not match the accepted prefix"
            )
    if provenance.target_block != target:
        raise ValueError(
            "progressive capture target_block does not match the next prefix block: "
            f"expected={target}, actual={provenance.target_block}"
        )
    if provenance.code_commit.lower() != prefix.code_commit.lower():
        raise ValueError(
            "progressive capture code_commit does not match the fixed sweep code revision"
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
        "dataset_manifest_sha256",
        "gate_manifest_sha256",
        "stage_a_campaign_sha256",
        "prefix_identity_sha256",
        "target_block",
    ):
        if getattr(provenance, name) != getattr(common, name):
            raise ValueError(
                f"progressive capture corpus mixes provenance field {name!r}: "
                f"{getattr(common, name)!r} vs {getattr(provenance, name)!r}"
            )


def _annotate_record(
    record: CapturedPilotCase,
    *,
    source_case_id: str,
    split: SplitName,
    artifact: ProgressiveCaptureArtifactRef,
) -> CapturedPilotCase:
    sigma = record.case.sigma
    if sigma is None:
        raise ValueError(f"progressive capture {source_case_id!r} must record sigma")
    context = dict(record.case.context)
    reserved = {
        "progressive_source_case_id": source_case_id,
        "progressive_split": split,
        "progressive_capture_receipt_sha256": artifact.receipt_sha256.lower(),
        "progressive_capture_bundle_sha256": sha256_file(artifact.bundle_path),
    }
    for key, value in reserved.items():
        if key in context and context[key] != value:
            raise ValueError(
                f"progressive capture context contradicts immutable field {key!r}"
            )
        context[key] = value
    case = replace(
        record.case,
        case_id=_execution_id(source_case_id, float(sigma)),
        context=context,
    )
    return replace(record, case=case)


def load_progressive_block_capture_set(
    artifacts: Sequence[ProgressiveCaptureArtifactRef],
    dataset_manifest: Mapping[str, object],
    prefix: ProgressivePrefix,
    *,
    minimum_cases: int = 16,
    minimum_sigma_strata: int = 8,
    required_coverage_tags: Sequence[str] = (),
) -> ProgressiveBlockCaptureSet:
    """Bind one full fixed-dataset capture corpus to exactly one accepted prefix.

    Every ``case_id × sigma`` execution in the fixed dataset must appear exactly once.
    Captures from a different accepted prefix, Stage-A campaign, gate policy, source
    revision, target block, or ComfyUI runtime are rejected instead of being mixed.
    """

    dataset_sha = validate_pilot_dataset_manifest(
        dataset_manifest,
        minimum_cases=minimum_cases,
        minimum_sigma_strata=minimum_sigma_strata,
        required_coverage_tags=required_coverage_tags,
    )
    if prefix.dataset_manifest_sha256.lower() != dataset_sha.lower():
        raise ValueError(
            "progressive prefix dataset manifest identity does not match the supplied dataset"
        )
    target_block = prefix.next_block
    if target_block is None:
        raise ValueError("progressive prefix is complete; there is no target block capture set")
    if not artifacts:
        raise ValueError("progressive capture corpus requires persisted capture artifacts")

    manifest_cases = _case_map(dataset_manifest)
    expected_executions: set[tuple[str, str]] = set()
    for case_id, case in manifest_cases.items():
        sigmas = case["sigmas"]
        assert isinstance(sigmas, list)
        for sigma in sigmas:
            expected_executions.add((case_id, _canonical_sigma(float(sigma))))

    train: list[CapturedPilotCase] = []
    holdout: list[CapturedPilotCase] = []
    seen_executions: set[tuple[str, str]] = set()
    seen_receipts: set[str] = set()
    common_provenance: ProgressiveCaptureProvenance | None = None

    for artifact in artifacts:
        receipt_sha = artifact.receipt_sha256.lower()
        if receipt_sha in seen_receipts:
            raise ValueError(f"duplicate progressive capture receipt identity: {receipt_sha}")
        seen_receipts.add(receipt_sha)
        record, provenance = load_progressive_capture_bundle(
            artifact.bundle_path,
            receipt_path=artifact.receipt_path,
            expected_receipt_sha256=receipt_sha,
        )
        _validate_provenance(
            provenance,
            prefix=prefix,
            dataset_manifest_sha256=dataset_sha,
            common=common_provenance,
        )
        if common_provenance is None:
            common_provenance = provenance

        source_case_id = record.case.case_id
        manifest_case = manifest_cases.get(source_case_id)
        if manifest_case is None:
            raise ValueError(
                f"progressive capture case {source_case_id!r} is absent from the fixed dataset"
            )
        sigma = record.case.sigma
        if sigma is None:
            raise ValueError(f"progressive capture case {source_case_id!r} is missing sigma")
        sigmas = manifest_case["sigmas"]
        assert isinstance(sigmas, list)
        if not _sigma_member(float(sigma), sigmas):
            raise ValueError(
                f"progressive capture sigma {sigma!r} for {source_case_id!r} is absent from its manifest strata"
            )
        if record.case.modality_label != manifest_case["modality_label"]:
            raise ValueError(
                f"progressive capture modality for {source_case_id!r} does not match dataset manifest"
            )
        split = manifest_case["split"]
        assert split in ("train", "holdout")
        execution = (source_case_id, _canonical_sigma(float(sigma)))
        if execution in seen_executions:
            raise ValueError(
                f"duplicate progressive capture execution for case/sigma {execution}"
            )
        seen_executions.add(execution)
        annotated = _annotate_record(
            record,
            source_case_id=source_case_id,
            split=split,
            artifact=artifact,
        )
        (train if split == "train" else holdout).append(annotated)

    missing = sorted(expected_executions - seen_executions)
    extra = sorted(seen_executions - expected_executions)
    if missing or extra:
        raise ValueError(
            "progressive capture corpus does not exactly cover dataset case/sigma executions; "
            f"missing={missing}, extra={extra}"
        )
    if not train or not holdout:
        raise ValueError("progressive capture corpus must provide both train and holdout executions")
    assert common_provenance is not None

    return ProgressiveBlockCaptureSet(
        target_block=target_block,
        prefix_identity_sha256=prefix.identity_sha256,
        stage_a_campaign_sha256=prefix.stage_a_campaign_sha256,
        dataset_manifest_sha256=dataset_sha,
        gate_manifest_sha256=prefix.gate_manifest_sha256,
        code_commit=common_provenance.code_commit,
        comfy_commit=common_provenance.comfy_commit,
        execution_descriptor=common_provenance.execution_descriptor,
        train=tuple(train),
        holdout=tuple(holdout),
        artifact_refs=tuple(artifacts),
    )
