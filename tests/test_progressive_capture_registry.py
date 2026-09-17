from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from minimax_h3_keyless.activation_capture import CapturedPilotCase
from minimax_h3_keyless.pilot import PilotCase
from minimax_h3_keyless.pilot_campaign import DATASET_SCHEMA, canonical_json_sha256
from minimax_h3_keyless.progressive import (
    PROGRESSIVE_PREFIX_CONTEXT_KEY,
    ProgressiveAcceptedBlock,
    ProgressivePrefix,
)
from minimax_h3_keyless.progressive_authorization import write_progressive_prefix_manifest
from minimax_h3_keyless.progressive_capture_io import (
    ProgressiveCaptureProvenance,
    write_progressive_capture_bundle,
)
from minimax_h3_keyless.progressive_capture_registry import (
    PROGRESSIVE_CAPTURE_REGISTRY_SCHEMA,
    build_progressive_capture_registry,
    load_progressive_capture_registry,
)


def _manifest() -> dict:
    def case(case_id: str, split: str) -> dict:
        return {
            "case_id": case_id,
            "split": split,
            "prompt": f"prompt-{case_id}",
            "seed": 1 if split == "train" else 2,
            "schedule": {"name": "fixture"},
            "modality_label": "video",
            "resolution": [64, 64],
            "duration_seconds": 1.0,
            "sigmas": [0.2, 0.8],
            "coverage_tags": [],
            "assets": [],
        }

    return {
        "schema": DATASET_SCHEMA,
        "cases": [case("train-a", "train"), case("holdout-a", "holdout")],
    }


def _prefix(dataset_sha: str, *, result_byte: str = "2") -> ProgressivePrefix:
    return ProgressivePrefix(
        sweep_id="registry-sweep",
        code_commit="d" * 40,
        stage_a_campaign_sha256="a" * 64,
        dataset_manifest_sha256=dataset_sha,
        gate_manifest_sha256="c" * 64,
        accepted=(
            ProgressiveAcceptedBlock(
                block_index=0,
                final_stage="route",
                checkpoint_sha256="1" * 64,
                result_sha256=result_byte * 64,
            ),
        ),
    )


def _record(prefix: ProgressivePrefix, case_id: str, split: str, sigma: float) -> CapturedPilotCase:
    x = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4) + float(sigma)
    case = PilotCase(
        x=x,
        t_emb=torch.zeros(1, 2, dtype=torch.bfloat16),
        mod_segments=((0, 3, 0),),
        rope_freqs=None,
        transformer_options={},
        case_id=case_id,
        sigma=sigma,
        modality_label="video",
        context={
            PROGRESSIVE_PREFIX_CONTEXT_KEY: prefix.capture_context(),
            "progressive_source_case_id": case_id,
            "progressive_split": split,
        },
    )
    return CapturedPilotCase(
        block_index=1,
        case=case,
        attention_input=x + 0.5,
        captured_bytes=256,
    )


def _write_capture(
    root: Path,
    prefix: ProgressivePrefix,
    case_id: str,
    split: str,
    sigma: float,
):
    provenance = ProgressiveCaptureProvenance(
        code_commit=prefix.code_commit,
        comfy_commit="e" * 40,
        dataset_manifest_sha256=prefix.dataset_manifest_sha256,
        gate_manifest_sha256=prefix.gate_manifest_sha256,
        stage_a_campaign_sha256=prefix.stage_a_campaign_sha256,
        prefix_identity_sha256=prefix.identity_sha256,
        target_block=1,
        execution_descriptor="progressive-registry-fixture",
    )
    return write_progressive_capture_bundle(
        root / f"{case_id}-{sigma}.capture.pt",
        _record(prefix, case_id, split, sigma),
        provenance=provenance,
    )


def _fixture(tmp_path: Path):
    manifest = _manifest()
    dataset_sha = canonical_json_sha256(manifest)
    prefix = _prefix(dataset_sha)
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(manifest), encoding="utf-8")
    prefix_path = tmp_path / "prefix.json"
    prefix_sha = write_progressive_prefix_manifest(
        prefix_path,
        prefix,
        previous_manifest_sha256=None,
    )
    captures = tmp_path / "captures"
    receipts = []
    for case_id, split in (("train-a", "train"), ("holdout-a", "holdout")):
        for sigma in (0.2, 0.8):
            written = _write_capture(captures, prefix, case_id, split, sigma)
            receipts.append(Path(written.receipt_path))
    return manifest, prefix, dataset_path, prefix_path, prefix_sha, receipts


def test_progressive_registry_streams_complete_prefix_bound_corpus(tmp_path: Path) -> None:
    _, prefix, dataset_path, prefix_path, prefix_sha, receipts = _fixture(tmp_path)
    output = tmp_path / "registry" / "block-01.json"

    result = build_progressive_capture_registry(
        dataset_path,
        prefix_path,
        receipts,
        output,
        minimum_cases=2,
        minimum_sigma_strata=2,
    )
    assert result.artifact_count == 4
    assert result.target_block == 1
    assert result.prefix_manifest_sha256 == prefix_sha
    assert result.prefix_identity_sha256 == prefix.identity_sha256

    loaded = load_progressive_capture_registry(output)
    assert loaded.registry_file_sha256 == result.registry_sha256
    assert loaded.prefix_manifest_sha256 == prefix_sha
    assert loaded.prefix_identity_sha256 == prefix.identity_sha256
    assert loaded.target_block == 1
    assert len(loaded.artifacts) == 4
    assert all(Path(row.bundle_path).is_file() for row in loaded.artifacts)
    assert all(Path(row.receipt_path).is_file() for row in loaded.artifacts)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == PROGRESSIVE_CAPTURE_REGISTRY_SCHEMA
    assert all(not Path(row["bundle_path"]).is_absolute() for row in payload["artifacts"])


def test_progressive_registry_rejects_missing_execution_without_publishing(tmp_path: Path) -> None:
    _, _, dataset_path, prefix_path, _, receipts = _fixture(tmp_path)
    output = tmp_path / "registry.json"
    with pytest.raises(ValueError, match="do not exactly cover"):
        build_progressive_capture_registry(
            dataset_path,
            prefix_path,
            receipts[:-1],
            output,
            minimum_cases=2,
            minimum_sigma_strata=2,
        )
    assert not output.exists()


def test_progressive_registry_rejects_capture_from_other_prefix(tmp_path: Path) -> None:
    manifest, prefix, dataset_path, prefix_path, _, receipts = _fixture(tmp_path)
    other = _prefix(canonical_json_sha256(manifest), result_byte="3")
    wrong = _write_capture(tmp_path / "wrong", other, "holdout-a", "holdout", 0.8)
    receipts[-1] = Path(wrong.receipt_path)

    with pytest.raises(ValueError, match="prefix_identity_sha256"):
        build_progressive_capture_registry(
            dataset_path,
            prefix_path,
            receipts,
            tmp_path / "registry.json",
            minimum_cases=2,
            minimum_sigma_strata=2,
        )
    assert prefix.identity_sha256 != other.identity_sha256


def test_progressive_registry_is_no_replace(tmp_path: Path) -> None:
    _, _, dataset_path, prefix_path, _, receipts = _fixture(tmp_path)
    output = tmp_path / "registry.json"
    build_progressive_capture_registry(
        dataset_path,
        prefix_path,
        receipts,
        output,
        minimum_cases=2,
        minimum_sigma_strata=2,
    )
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        build_progressive_capture_registry(
            dataset_path,
            prefix_path,
            receipts,
            output,
            minimum_cases=2,
            minimum_sigma_strata=2,
        )
    assert output.read_bytes() == before
