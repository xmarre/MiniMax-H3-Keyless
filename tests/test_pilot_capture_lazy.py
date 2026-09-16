from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

import minimax_h3_keyless.pilot_capture_lazy as lazy_module
from minimax_h3_keyless.activation_capture import CapturedPilotCase
from minimax_h3_keyless.capture_io import CaptureBundleProvenance, write_captured_pilot_bundle
from minimax_h3_keyless.pilot import PilotCase
from minimax_h3_keyless.pilot_campaign import DATASET_SCHEMA, canonical_json_sha256
from minimax_h3_keyless.pilot_capture_lazy import load_stage_a_capture_set_lazy
from minimax_h3_keyless.pilot_capture_set import CaptureArtifactRef


def _dataset():
    return {
        "schema": DATASET_SCHEMA,
        "cases": [
            {
                "case_id": "train-a",
                "split": "train",
                "prompt": "train",
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
                "prompt": "hold",
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
        context={"capture": "lazy-test"},
    )
    return tuple(
        CapturedPilotCase(
            block_index=block_index,
            case=replace(base, x=base.x + block_index),
            attention_input=base.x + block_index + 0.25,
            captured_bytes=1000 + block_index,
        )
        for block_index in (0, 25, 49)
    )


def _artifacts(tmp_path: Path):
    dataset = _dataset()
    dataset_sha = canonical_json_sha256(dataset)
    refs = []
    for case in dataset["cases"]:
        for sigma in case["sigmas"]:
            path = tmp_path / f"{case['case_id']}-{sigma}.pt"
            result = write_captured_pilot_bundle(
                path,
                _records(case["case_id"], float(sigma), case["modality_label"]),
                provenance=CaptureBundleProvenance(
                    code_commit="code-a",
                    comfy_commit="comfy-a",
                    dataset_manifest_sha256=dataset_sha,
                    execution_descriptor="plain native BF16 lazy test capture",
                ),
            )
            refs.append(
                CaptureArtifactRef(
                    bundle_path=result.bundle_path,
                    receipt_path=result.receipt_path,
                    receipt_sha256=result.receipt_sha256,
                )
            )
    return dataset, refs


def test_lazy_capture_index_does_not_deserialize_activation_bundles(tmp_path: Path, monkeypatch) -> None:
    dataset, refs = _artifacts(tmp_path)
    actual_loader = lazy_module.load_captured_pilot_bundle

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("activation bundle deserialized while constructing lazy index")

    monkeypatch.setattr(lazy_module, "load_captured_pilot_bundle", forbidden_loader)
    corpus = load_stage_a_capture_set_lazy(
        refs,
        dataset,
        minimum_cases=2,
        minimum_sigma_strata=2,
    )
    assert len(corpus.executions) == 4
    with pytest.raises(AssertionError, match="deserialized"):
        corpus.records(0, "train")

    monkeypatch.setattr(lazy_module, "load_captured_pilot_bundle", actual_loader)
    train = corpus.records(25, "train")
    holdout = corpus.records(25, "holdout")
    assert len(train) == 2
    assert len(holdout) == 2
    assert {record.block_index for record in train} == {25}
    assert {record.case.context["stage_a_source_case_id"] for record in train} == {"train-a"}
    assert {record.case.context["stage_a_split"] for record in holdout} == {"holdout"}
    assert {record.case.case_id for record in train} == {
        "train-a::sigma=0",
        "train-a::sigma=1",
    }


def test_lazy_capture_records_revalidate_bundle_hash_after_indexing(tmp_path: Path) -> None:
    dataset, refs = _artifacts(tmp_path)
    corpus = load_stage_a_capture_set_lazy(
        refs,
        dataset,
        minimum_cases=2,
        minimum_sigma_strata=2,
    )
    tampered = Path(refs[0].bundle_path)
    with tampered.open("ab") as handle:
        handle.write(b"tamper-after-index")
    with pytest.raises(ValueError, match="bundle_bytes|bundle_sha256"):
        corpus.records(0, "train")


def test_lazy_capture_index_rejects_duplicate_and_missing_execution(tmp_path: Path) -> None:
    dataset, refs = _artifacts(tmp_path)
    with pytest.raises(ValueError, match="duplicate Stage-A capture receipt"):
        load_stage_a_capture_set_lazy(
            [refs[0], refs[0], *refs[1:]],
            dataset,
            minimum_cases=2,
            minimum_sigma_strata=2,
        )
    with pytest.raises(ValueError, match="does not exactly cover"):
        load_stage_a_capture_set_lazy(
            refs[:-1],
            dataset,
            minimum_cases=2,
            minimum_sigma_strata=2,
        )
