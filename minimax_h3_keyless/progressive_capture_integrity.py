from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .activation_capture import CapturedPilotCase
from .checkpoint import sha256_file
from .progressive_capture_io import load_progressive_capture_bundle
from .progressive_capture_set import ProgressiveCaptureSet


def _tensor_equal(name: str, actual: torch.Tensor | None, expected: torch.Tensor | None) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise RuntimeError(f"progressive capture integrity mismatch for {name}")
        return
    if actual.dtype != expected.dtype or tuple(actual.shape) != tuple(expected.shape):
        raise RuntimeError(f"progressive capture integrity signature mismatch for {name}")
    if not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
        raise RuntimeError(f"progressive capture integrity tensor mismatch for {name}")


def _nested_equal(name: str, actual: Any, expected: Any) -> None:
    if torch.is_tensor(actual) or torch.is_tensor(expected):
        if not torch.is_tensor(actual) or not torch.is_tensor(expected):
            raise RuntimeError(f"progressive capture integrity type mismatch for {name}")
        _tensor_equal(name, actual, expected)
        return
    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
            raise RuntimeError(f"progressive capture integrity type mismatch for {name}")
        if set(actual) != set(expected):
            raise RuntimeError(f"progressive capture integrity mapping keys mismatch for {name}")
        for key in actual:
            _nested_equal(f"{name}.{key}", actual[key], expected[key])
        return
    if isinstance(actual, (tuple, list)) or isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or not isinstance(expected, (tuple, list)):
            raise RuntimeError(f"progressive capture integrity type mismatch for {name}")
        if len(actual) != len(expected):
            raise RuntimeError(f"progressive capture integrity length mismatch for {name}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _nested_equal(f"{name}[{index}]", left, right)
        return
    if actual != expected:
        raise RuntimeError(f"progressive capture integrity value mismatch for {name}")


def _sigma_equal(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.0)


def _expected_execution_id(source_case_id: str, sigma: float) -> str:
    return f"{source_case_id}::sigma={format(float(sigma), '.17g')}"


def _compare_annotated_record(
    record: CapturedPilotCase,
    raw: CapturedPilotCase,
    *,
    split: str,
    receipt_sha256: str,
    bundle_sha256: str,
) -> None:
    if record.block_index != raw.block_index:
        raise RuntimeError("progressive capture integrity block_index mismatch")
    if not _sigma_equal(record.case.sigma, raw.case.sigma):
        raise RuntimeError("progressive capture integrity sigma mismatch")
    if record.case.modality_label != raw.case.modality_label:
        raise RuntimeError("progressive capture integrity modality mismatch")
    if raw.case.sigma is None:
        raise RuntimeError("progressive immutable capture is missing sigma")
    expected_id = _expected_execution_id(raw.case.case_id, float(raw.case.sigma))
    if record.case.case_id != expected_id:
        raise RuntimeError("progressive capture integrity execution case_id mismatch")
    if record.captured_bytes != raw.captured_bytes:
        raise RuntimeError("progressive capture integrity captured_bytes mismatch")

    _tensor_equal("x", record.case.x, raw.case.x)
    _tensor_equal("t_emb", record.case.t_emb, raw.case.t_emb)
    _nested_equal("mod_segments", record.case.mod_segments, raw.case.mod_segments)
    _tensor_equal("rope_freqs", record.case.rope_freqs, raw.case.rope_freqs)
    _tensor_equal("position_ids", record.case.position_ids, raw.case.position_ids)
    _tensor_equal("attention_input", record.attention_input, raw.attention_input)
    if record.case.transformer_options != raw.case.transformer_options:
        raise RuntimeError("progressive capture integrity transformer_options mismatch")

    expected_context = dict(raw.case.context)
    expected_context.update(
        {
            "progressive_source_case_id": raw.case.case_id,
            "progressive_split": split,
            "progressive_capture_receipt_sha256": receipt_sha256,
            "progressive_capture_bundle_sha256": bundle_sha256,
        }
    )
    if record.case.context != expected_context:
        raise RuntimeError("progressive capture integrity annotated context mismatch")


def _validate_provenance(captures: ProgressiveCaptureSet, provenance) -> None:
    if provenance.target_block != captures.target_block:
        raise RuntimeError("progressive capture artifact target differs from capture-set target")
    if provenance.prefix_identity_sha256.lower() != captures.prefix_identity_sha256.lower():
        raise RuntimeError("progressive capture artifact prefix differs from capture-set prefix")
    if provenance.stage_a_campaign_sha256.lower() != captures.stage_a_campaign_sha256.lower():
        raise RuntimeError("progressive capture artifact Stage-A identity differs from capture set")
    if provenance.dataset_manifest_sha256.lower() != captures.dataset_manifest_sha256.lower():
        raise RuntimeError("progressive capture artifact dataset identity differs from capture set")
    if provenance.gate_manifest_sha256.lower() != captures.gate_manifest_sha256.lower():
        raise RuntimeError("progressive capture artifact gate identity differs from capture set")
    if provenance.code_commit.lower() != captures.code_commit.lower():
        raise RuntimeError("progressive capture artifact code revision differs from capture set")
    if provenance.comfy_commit != captures.comfy_commit:
        raise RuntimeError("progressive capture artifact Comfy revision differs from capture set")
    if provenance.execution_descriptor != captures.execution_descriptor:
        raise RuntimeError("progressive capture artifact execution descriptor differs from capture set")


def validate_progressive_capture_set_integrity(
    captures: ProgressiveCaptureSet,
) -> None:
    """Rebind every Stage-B record to immutable bytes with bounded activation memory.

    Eager capture sets are ordinary Python data and lazy sets reload records from disk, so
    the persistence/acceptance boundary must independently rehash and compare every record.
    This implementation never builds a list or dictionary containing loaded capture tensors:
    it keeps only artifact references plus the current annotated/raw record pair.
    """

    refs = tuple(captures.artifact_refs)
    expected_records = len(captures.train) + len(captures.holdout)
    if len(refs) != expected_records:
        raise RuntimeError(
            "progressive capture integrity requires exactly one immutable artifact per replay record"
        )
    if not refs:
        raise RuntimeError("progressive capture integrity requires immutable artifact references")

    refs_by_receipt = {}
    for ref in refs:
        receipt_sha = str(ref.receipt_sha256).lower()
        if receipt_sha in refs_by_receipt:
            raise RuntimeError("progressive capture integrity found a duplicate receipt identity")
        if sha256_file(ref.receipt_path).lower() != receipt_sha:
            raise RuntimeError("progressive capture receipt bytes changed after capture-set loading")
        refs_by_receipt[receipt_sha] = ref

    seen: set[str] = set()
    for split in ("train", "holdout"):
        for record in captures.records(split):
            context = record.case.context
            if not isinstance(context, Mapping):
                raise RuntimeError("progressive capture integrity requires mapping context")
            receipt_sha = context.get("progressive_capture_receipt_sha256")
            bundle_sha = context.get("progressive_capture_bundle_sha256")
            if not isinstance(receipt_sha, str) or not isinstance(bundle_sha, str):
                raise RuntimeError("progressive capture record is missing immutable artifact hashes")
            receipt_sha = receipt_sha.lower()
            bundle_sha = bundle_sha.lower()
            if receipt_sha in seen:
                raise RuntimeError("progressive capture integrity maps one artifact to multiple records")
            ref = refs_by_receipt.get(receipt_sha)
            if ref is None:
                raise RuntimeError(
                    "progressive capture record references an artifact outside its capture set"
                )

            raw, provenance = load_progressive_capture_bundle(
                ref.bundle_path,
                receipt_path=ref.receipt_path,
                expected_receipt_sha256=receipt_sha,
            )
            actual_bundle_sha = sha256_file(ref.bundle_path).lower()
            if bundle_sha != actual_bundle_sha:
                raise RuntimeError(
                    "progressive capture record bundle hash does not match immutable bytes"
                )
            _validate_provenance(captures, provenance)
            _compare_annotated_record(
                record,
                raw,
                split=split,
                receipt_sha256=receipt_sha,
                bundle_sha256=actual_bundle_sha,
            )
            seen.add(receipt_sha)
            del raw

    if seen != set(refs_by_receipt):
        raise RuntimeError("progressive capture integrity found unreferenced immutable artifacts")
