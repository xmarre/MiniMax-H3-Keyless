from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from minimax_h3_keyless.activation_capture import CapturedPilotCase
from minimax_h3_keyless.capture_io import (
    CaptureBundleProvenance,
    write_captured_pilot_bundle,
)
from minimax_h3_keyless.pilot import PilotCase
from minimax_h3_keyless.pilot_campaign import DATASET_SCHEMA, canonical_json_sha256
from minimax_h3_keyless.pilot_capture_set import (
    CaptureArtifactRef,
    load_stage_a_capture_set,
)


def _dataset():
    return {
        "schema": DATASET_SCHEMA,
        "cases": [
            {
                "case_id": "train-a",
                "split": "train",
                "prompt": "train prompt",
                "seed": 1,
                "schedule": {"steps": 4},
                "modality_label": "video",
                "resolution": [64, 64],
                "duration_seconds": 1.0,
                "sigmas": [0.0, 1.0],
                "coverage_tags": ["short"],
                "assets": [],
            },
            {
                "case_id": "hold-b",
                "split": "holdout",
                "prompt": "hold prompt",
                "seed": 2,
                "schedule": {"steps": 4},
                "modality_label": "audio-video",
                "resolution": [64, 64],
                "duration_seconds": 1.0,
                "sigmas": [0.0, 1.0],
                "coverage_tags": ["audio"],
                "assets": [],
            },
        ],
    }


def _records(case_id: str, sigma: float, modality: str):
    base = PilotCase(
        x=torch.arange(12, dtype=torch.float32).reshape(3, 4) + sigma,
        t_emb=torch.ones(1, 2),
        mod_segments=((0, 3, 0),),
        rope_freqs=None,
        transformer_options={},
        case_id=case_id,
        sigma=sigma,
        modality_label=modality,
        position_ids=torch.zeros(3, 3),
        context={"capture": "test"},
    )
    rows = []
    for block_index in (0, 25, 49):
        case = replace(base, x=base.x + block_index)
        rows.append(
            CapturedPilotCase(
                block_index=block_index,
                case=case,
                attention_input=case.x + 0.25,
                captured_bytes=1000 + block_index,
            )
        )
    return tuple(rows)


def _write_artifacts(tmp_path: Path, dataset, *, code_commit="code-a"):
    dataset_sha = canonical_json_sha256(dataset)
    refs = []
    for case in dataset["cases"]:
        for sigma in case["sigmas"]:
            path = tmp_path / f"{case['case_id']}-{sigma}.pt"
            provenance = CaptureBundleProvenance(
                code_commit=code_commit,
                comfy_commit="comfy-a",
                dataset_manifest_sha256=dataset_sha,
                execution_descriptor="plain native BF16 test capture",
            )
            result = write_captured_pilot_bundle(
                path,
                _records(case["case_id"], sigma, case["modality_label"]),
                provenance=provenance,
            )
            refs.append(
                CaptureArtifactRef(
                    bundle_path=result.bundle_path,
                    receipt_path=result.receipt_path,
                    receipt_sha256=result.receipt_sha256,
                )
            )
    return refs


def test_capture_set_binds_all_case_sigma_executions_without_split_leakage(tmp_path: Path) -> None:
    dataset = _dataset()
    refs = _write_artifacts(tmp_path, dataset)
    corpus = load_stage_a_capture_set(
        refs,
        dataset,
        minimum_cases=2,
        minimum_sigma_strata=2,
    )
    assert corpus.dataset_manifest_sha256 == canonical_json_sha256(dataset)
    assert corpus.code_commit == "code-a"
    for block_index in (0, 25, 49):
        train = corpus.records(block_index, "train")
        holdout = corpus.records(block_index, "holdout")
        assert len(train) == 2
        assert len(holdout) == 2
        assert {r.case.context["stage_a_source_case_id"] for r in train} == {"train-a"}
        assert {r.case.context["stage_a_source_case_id"] for r in holdout} == {"hold-b"}
        assert {r.case.case_id for r in train} == {
            "train-a::sigma=0",
            "train-a::sigma=1",
        }
        assert len({r.case.case_id for r in holdout}) == 2
        assert all(r.case.context["stage_a_split"] == "train" for r in train)
        assert all(r.case.context["stage_a_split"] == "holdout" for r in holdout)


def test_capture_set_rejects_missing_sigma_execution(tmp_path: Path) -> None:
    dataset = _dataset()
    refs = _write_artifacts(tmp_path, dataset)
    with pytest.raises(ValueError, match="does not exactly cover"):
        load_stage_a_capture_set(
            refs[:-1],
            dataset,
            minimum_cases=2,
            minimum_sigma_strata=2,
        )


def test_capture_set_rejects_duplicate_receipt_and_mixed_provenance(tmp_path: Path) -> None:
    dataset = _dataset()
    refs = _write_artifacts(tmp_path, dataset)
    with pytest.raises(ValueError, match="duplicate Stage-A capture receipt"):
        load_stage_a_capture_set(
            [refs[0], refs[0], *refs[1:]],
            dataset,
            minimum_cases=2,
            minimum_sigma_strata=2,
        )

    mixed_dir = tmp_path / "mixed"
    mixed_dir.mkdir()
    mixed = _write_artifacts(mixed_dir, dataset, code_commit="code-b")
    with pytest.raises(ValueError, match="mixes provenance"):
        load_stage_a_capture_set(
            [refs[0], *mixed[1:]],
            dataset,
            minimum_cases=2,
            minimum_sigma_strata=2,
        )


def test_capture_set_rejects_wrong_manifest_modality_or_unknown_sigma(tmp_path: Path) -> None:
    dataset = _dataset()
    refs = _write_artifacts(tmp_path, dataset)

    bad_modality = _dataset()
    bad_modality["cases"][0]["modality_label"] = "image"
    with pytest.raises(ValueError, match="dataset manifest identity"):
        load_stage_a_capture_set(
            refs,
            bad_modality,
            minimum_cases=2,
            minimum_sigma_strata=2,
        )

    # Construct a fully hash-consistent artifact set where one capture itself uses an
    # undeclared sigma. This must fail on semantic membership, not only provenance hash.
    altered = _dataset()
    altered_sha = canonical_json_sha256(altered)
    path = tmp_path / "unknown-sigma.pt"
    provenance = CaptureBundleProvenance(
        code_commit="code-a",
        comfy_commit="comfy-a",
        dataset_manifest_sha256=altered_sha,
        execution_descriptor="plain native BF16 test capture",
    )
    result = write_captured_pilot_bundle(
        path,
        _records("train-a", 0.25, "video"),
        provenance=provenance,
    )
    bad_ref = CaptureArtifactRef(
        bundle_path=result.bundle_path,
        receipt_path=result.receipt_path,
        receipt_sha256=result.receipt_sha256,
    )
    # All refs must use the same provenance, so generate the remaining canonical files
    # under the same manifest identity and replace only the expected train-a sigma=0 file.
    consistent_dir = tmp_path / "consistent"
    consistent_dir.mkdir()
    consistent = _write_artifacts(consistent_dir, altered)
    with pytest.raises(ValueError, match="absent from its manifest strata"):
        load_stage_a_capture_set(
            [bad_ref, *consistent[1:]],
            altered,
            minimum_cases=2,
            minimum_sigma_strata=2,
        )
