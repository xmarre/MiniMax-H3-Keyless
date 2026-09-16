from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from minimax_h3_keyless.activation_capture import CapturedPilotCase
from minimax_h3_keyless.pilot import PilotCase
from minimax_h3_keyless.pilot_campaign import DATASET_SCHEMA, canonical_json_sha256
from minimax_h3_keyless.progressive import PROGRESSIVE_PREFIX_CONTEXT_KEY, ProgressivePrefix
from minimax_h3_keyless.progressive_capture_integrity import (
    validate_progressive_capture_set_integrity,
)
from minimax_h3_keyless.progressive_capture_io import (
    ProgressiveCaptureProvenance,
    write_progressive_capture_bundle,
)
from minimax_h3_keyless.progressive_capture_set import (
    ProgressiveCaptureArtifactRef,
    load_progressive_block_capture_set,
)


def _manifest():
    def row(case_id: str, split: str):
        return {
            "case_id": case_id,
            "split": split,
            "prompt": case_id,
            "seed": 1,
            "schedule": {"name": "fixture"},
            "modality_label": "video",
            "resolution": [64, 64],
            "duration_seconds": 1.0,
            "sigmas": [0.5],
            "coverage_tags": [],
            "assets": [],
        }
    return {"schema": DATASET_SCHEMA, "cases": [row("train", "train"), row("holdout", "holdout")]}


def _prefix(dataset_sha: str):
    return ProgressivePrefix(
        sweep_id="integrity",
        code_commit="d" * 40,
        stage_a_campaign_sha256="a" * 64,
        dataset_manifest_sha256=dataset_sha,
        gate_manifest_sha256="c" * 64,
    )


def _artifact(root: Path, prefix: ProgressivePrefix, case_id: str):
    torch.manual_seed(1000 + len(case_id))
    x = torch.randn(3, 4)
    record = CapturedPilotCase(
        block_index=0,
        case=PilotCase(
            x=x,
            t_emb=torch.zeros(1, 1),
            mod_segments=((0, 3, torch.tensor([0, 1, 2])),),
            rope_freqs=None,
            transformer_options={},
            case_id=case_id,
            sigma=0.5,
            modality_label="video",
            context={PROGRESSIVE_PREFIX_CONTEXT_KEY: prefix.capture_context()},
        ),
        attention_input=x + 0.25,
        captured_bytes=256,
    )
    provenance = ProgressiveCaptureProvenance(
        code_commit=prefix.code_commit,
        comfy_commit="comfy-integrity",
        dataset_manifest_sha256=prefix.dataset_manifest_sha256,
        gate_manifest_sha256=prefix.gate_manifest_sha256,
        stage_a_campaign_sha256=prefix.stage_a_campaign_sha256,
        prefix_identity_sha256=prefix.identity_sha256,
        target_block=0,
        execution_descriptor="integrity-fixture",
    )
    written = write_progressive_capture_bundle(
        root / f"{case_id}.capture.pt",
        record,
        provenance=provenance,
    )
    return ProgressiveCaptureArtifactRef(
        bundle_path=written.bundle_path,
        receipt_path=written.receipt_path,
        receipt_sha256=written.receipt_sha256,
    )


def _capture_set(tmp_path: Path):
    manifest = _manifest()
    prefix = _prefix(canonical_json_sha256(manifest))
    refs = (
        _artifact(tmp_path, prefix, "train"),
        _artifact(tmp_path, prefix, "holdout"),
    )
    captures = load_progressive_block_capture_set(
        refs,
        manifest,
        prefix,
        minimum_cases=2,
        minimum_sigma_strata=1,
    )
    return captures


def test_progressive_capture_integrity_reloads_and_matches_all_immutable_records(tmp_path: Path) -> None:
    captures = _capture_set(tmp_path)
    validate_progressive_capture_set_integrity(captures)


def test_progressive_capture_integrity_rejects_in_memory_tensor_forgery(tmp_path: Path) -> None:
    captures = _capture_set(tmp_path)
    original = captures.train[0]
    forged_case = replace(original.case, x=original.case.x + 1.0)
    forged_record = replace(original, case=forged_case)
    forged = replace(captures, train=(forged_record, *captures.train[1:]))

    with pytest.raises(RuntimeError, match="tensor mismatch for x"):
        validate_progressive_capture_set_integrity(forged)


def test_progressive_capture_integrity_rejects_artifact_record_cardinality_mismatch(tmp_path: Path) -> None:
    captures = _capture_set(tmp_path)
    forged = replace(captures, artifact_refs=captures.artifact_refs[:1])
    with pytest.raises(RuntimeError, match="exactly one immutable artifact"):
        validate_progressive_capture_set_integrity(forged)


def test_progressive_capture_integrity_rejects_receipt_bytes_changed_after_load(tmp_path: Path) -> None:
    captures = _capture_set(tmp_path)
    receipt = Path(captures.artifact_refs[0].receipt_path)
    with receipt.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RuntimeError, match="receipt bytes changed"):
        validate_progressive_capture_set_integrity(captures)
