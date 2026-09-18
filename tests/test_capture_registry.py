from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from minimax_h3_keyless.activation_capture import CapturedPilotCase
from minimax_h3_keyless.capture_io import CaptureBundleProvenance, write_captured_pilot_bundle
from minimax_h3_keyless.capture_registry import build_stage_a_capture_registry
from minimax_h3_keyless.pilot import PilotCase
from minimax_h3_keyless.pilot_campaign import DATASET_SCHEMA, write_json_atomic
from minimax_h3_keyless.pilot_inputs import load_stage_a_capture_registry
from minimax_h3_keyless.stage_a_execution_binding import WORKFLOW_CONTEXT_KEY


WORKFLOW_SHA = "f" * 64


def _manifest() -> dict:
    sigma_values = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]
    tags = ["short", "long", "reference", "audio", "mixed-grid"]
    cases = []
    for index in range(16):
        cases.append(
            {
                "case_id": f"case-{index:02d}",
                "split": "train" if index < 12 else "holdout",
                "prompt": f"prompt {index}",
                "seed": index,
                "schedule": {"name": "fixed"},
                "modality_label": "video_audio" if index % 2 == 0 else "video",
                "resolution": [512, 512],
                "duration_seconds": 4.0,
                "sigmas": [sigma_values[index % len(sigma_values)]],
                "coverage_tags": tags if index == 0 else [],
                "assets": [],
                "workflow_prompt_sha256": WORKFLOW_SHA,
            }
        )
    return {"schema": DATASET_SCHEMA, "cases": cases}


def _records(case_id: str, sigma: float, modality: str, *, workflow_sha: str = WORKFLOW_SHA):
    base = PilotCase(
        x=torch.arange(8, dtype=torch.float32).reshape(2, 4),
        t_emb=torch.ones(1, 2),
        mod_segments=((0, 2, torch.tensor([0, 0], dtype=torch.long)),),
        rope_freqs=torch.ones(1, 2, 1, 1, 2, 2),
        transformer_options={},
        case_id=case_id,
        sigma=sigma,
        modality_label=modality,
        position_ids=torch.zeros(2, 3, dtype=torch.float64),
        context={
            "stage_a_source_case_id": case_id,
            "stage_a_split": "train" if int(case_id.split("-")[-1]) < 12 else "holdout",
            WORKFLOW_CONTEXT_KEY: workflow_sha,
        },
    )
    return tuple(
        CapturedPilotCase(
            block_index=block_index,
            case=replace(base, x=base.x + block_index),
            attention_input=base.x + block_index + 1,
            captured_bytes=128,
        )
        for block_index in (0, 25, 49)
    )


def _provenance(dataset_sha: str) -> CaptureBundleProvenance:
    return CaptureBundleProvenance(
        code_commit="1" * 40,
        comfy_commit="2" * 40,
        dataset_manifest_sha256=dataset_sha,
        execution_descriptor="registry-test",
    )


def _write_corpus(tmp_path: Path):
    manifest = _manifest()
    manifest_path = tmp_path / "dataset.json"
    write_json_atomic(manifest_path, manifest)
    from minimax_h3_keyless.pilot_campaign import validate_pilot_dataset_manifest
    from minimax_h3_keyless.pilot_inputs import CANONICAL_STAGE_A_COVERAGE_TAGS

    dataset_identity = validate_pilot_dataset_manifest(
        manifest,
        required_coverage_tags=CANONICAL_STAGE_A_COVERAGE_TAGS,
    )
    capture_dir = tmp_path / "captures"
    receipts = []
    for case in manifest["cases"]:
        sigma = float(case["sigmas"][0])
        bundle = capture_dir / f"{case['case_id']}.capture.pt"
        written = write_captured_pilot_bundle(
            bundle,
            _records(case["case_id"], sigma, case["modality_label"]),
            provenance=_provenance(dataset_identity),
        )
        receipts.append(Path(written.receipt_path))
    return manifest_path, receipts, dataset_identity


def test_registry_builder_streams_and_emits_loader_compatible_registry(tmp_path: Path) -> None:
    manifest_path, receipts, dataset_identity = _write_corpus(tmp_path)
    output = tmp_path / "registry" / "stage_a_registry.json"
    result = build_stage_a_capture_registry(manifest_path, receipts, output)
    assert result.artifact_count == 16
    assert result.dataset_manifest_sha256 == dataset_identity
    assert result.code_commit == "1" * 40
    assert result.comfy_commit == "2" * 40

    loaded = load_stage_a_capture_registry(output)
    assert loaded.dataset_manifest_sha256 == dataset_identity
    assert len(loaded.artifacts) == 16
    assert all(Path(item.bundle_path).is_absolute() for item in loaded.artifacts)
    assert all(Path(item.receipt_path).is_absolute() for item in loaded.artifacts)


def test_registry_builder_rejects_missing_or_mixed_capture_evidence(tmp_path: Path) -> None:
    manifest_path, receipts, dataset_identity = _write_corpus(tmp_path)
    with pytest.raises(ValueError, match="do not exactly cover"):
        build_stage_a_capture_registry(
            manifest_path,
            receipts[:-1],
            tmp_path / "missing.json",
        )

    last_receipt = receipts[-1]
    last_bundle = last_receipt.with_name(last_receipt.name.removesuffix(".receipt.json"))
    last_bundle.unlink()
    last_receipt.unlink()
    case = _manifest()["cases"][-1]
    written = write_captured_pilot_bundle(
        last_bundle,
        _records(case["case_id"], float(case["sigmas"][0]), case["modality_label"]),
        provenance=CaptureBundleProvenance(
            code_commit="3" * 40,
            comfy_commit="2" * 40,
            dataset_manifest_sha256=dataset_identity,
            execution_descriptor="registry-test",
        ),
    )
    mixed = [*receipts[:-1], Path(written.receipt_path)]
    with pytest.raises(ValueError, match="may not mix"):
        build_stage_a_capture_registry(
            manifest_path,
            mixed,
            tmp_path / "mixed.json",
        )


def test_registry_builder_rejects_capture_from_wrong_workflow(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "dataset.json"
    write_json_atomic(manifest_path, manifest)
    from minimax_h3_keyless.pilot_campaign import validate_pilot_dataset_manifest
    from minimax_h3_keyless.pilot_inputs import CANONICAL_STAGE_A_COVERAGE_TAGS

    dataset_identity = validate_pilot_dataset_manifest(
        manifest,
        required_coverage_tags=CANONICAL_STAGE_A_COVERAGE_TAGS,
    )
    receipts = []
    for case in manifest["cases"]:
        sigma = float(case["sigmas"][0])
        workflow_sha = "e" * 64 if case["case_id"] == "case-05" else WORKFLOW_SHA
        written = write_captured_pilot_bundle(
            tmp_path / "captures" / f"{case['case_id']}.capture.pt",
            _records(
                case["case_id"],
                sigma,
                case["modality_label"],
                workflow_sha=workflow_sha,
            ),
            provenance=_provenance(dataset_identity),
        )
        receipts.append(Path(written.receipt_path))

    with pytest.raises(ValueError, match="workflow identity"):
        build_stage_a_capture_registry(
            manifest_path,
            receipts,
            tmp_path / "wrong-workflow.json",
        )
