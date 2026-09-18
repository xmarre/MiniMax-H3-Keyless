from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import minimax_h3_keyless.progressive_authorization as authorization_module
from minimax_h3_keyless.contracts import TEACHER_SHA256
from minimax_h3_keyless.pilot_campaign import (
    DATASET_SCHEMA,
    GATE_SCHEMA,
    validate_pilot_dataset_manifest,
    validate_pilot_gate_manifest,
)
from minimax_h3_keyless.pilot_gates import StageACampaignGateResult
from minimax_h3_keyless.progressive_authorization import (
    authorize_progressive_sweep,
    load_progressive_prefix_manifest,
    write_progressive_prefix_manifest,
)
from minimax_h3_keyless.stage_a_campaign_result import StageACampaignEvidence


def _dataset():
    tags = ["short", "long", "reference", "audio", "mixed-grid"]
    cases = []
    for index in range(16):
        cases.append(
            {
                "case_id": f"case-{index:02d}",
                "split": "train" if index < 8 else "holdout",
                "prompt": f"prompt {index}",
                "seed": index,
                "schedule": {"name": "fixture"},
                "modality_label": "video",
                "resolution": [64, 64],
                "duration_seconds": 1.0,
                "sigmas": [index % 8 / 7.0],
                "coverage_tags": [tags[index]] if index < len(tags) else [],
                "assets": [],
            }
        )
    return {"schema": DATASET_SCHEMA, "cases": cases}


def _gate():
    return {
        "schema": GATE_SCHEMA,
        "thresholds": {
            "fixture_threshold": 1.0,
            "progressive_fold_atol": 0.002,
            "progressive_fold_rtol": 0.003,
        },
        "calibration_evidence": {"fixture": "authorization"},
    }


def _evidence(dataset_sha: str, gate_sha: str, *, passed: bool = True):
    gate = StageACampaignGateResult(
        passed=passed,
        failures=() if passed else ("fixture failure",),
        block_results={},
    )
    return StageACampaignEvidence(
        path="/evidence/stage-a.json",
        sha256="9" * 64,
        run_id="stage-a-001",
        code_commit="1" * 40,
        final_stage="route",
        experiment_context_sha256="2" * 64,
        teacher_sha256=TEACHER_SHA256,
        dataset_manifest_sha256=dataset_sha,
        gate_manifest_sha256=gate_sha,
        block_evidence={},
        gate=gate,
    )


def test_progressive_authorization_requires_passed_stage_a_and_binds_fixed_inputs(monkeypatch) -> None:
    dataset = _dataset()
    gate = _gate()
    dataset_sha = validate_pilot_dataset_manifest(
        dataset,
        required_coverage_tags=("short", "long", "reference", "audio", "mixed-grid"),
    )
    gate_sha = validate_pilot_gate_manifest(gate)
    evidence = _evidence(dataset_sha, gate_sha)

    def fake_load(path, *, gate_manifest, require_passed):
        assert str(path) == "stage-a.json"
        assert gate_manifest is gate
        assert require_passed is True
        return evidence

    monkeypatch.setattr(authorization_module, "load_stage_a_campaign_evidence", fake_load)
    authorization = authorize_progressive_sweep(
        "stage-a.json",
        dataset_manifest=dataset,
        gate_manifest=gate,
        sweep_id="core50-001",
        code_commit="a" * 40,
    )
    assert authorization.stage_a is evidence
    assert authorization.prefix.accepted_blocks == ()
    assert authorization.prefix.next_block == 0
    assert authorization.prefix.code_commit == "a" * 40
    assert authorization.prefix.stage_a_campaign_sha256 == evidence.sha256
    assert authorization.prefix.dataset_manifest_sha256 == dataset_sha
    assert authorization.prefix.gate_manifest_sha256 == gate_sha


def test_progressive_authorization_rejects_incomplete_stage_b_execution_policy(monkeypatch) -> None:
    dataset = _dataset()
    gate = _gate()
    del gate["thresholds"]["progressive_fold_rtol"]
    called = False

    def forbidden_stage_a(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("incomplete Stage-B policy must fail before Stage-A evidence I/O")

    monkeypatch.setattr(authorization_module, "load_stage_a_campaign_evidence", forbidden_stage_a)
    with pytest.raises(ValueError, match="progressive execution thresholds"):
        authorize_progressive_sweep(
            "stage-a.json",
            dataset_manifest=dataset,
            gate_manifest=gate,
            sweep_id="core50-001",
            code_commit="a" * 40,
        )
    assert called is False


def test_progressive_authorization_rejects_stage_a_dataset_identity_mismatch(monkeypatch) -> None:
    dataset = _dataset()
    gate = _gate()
    gate_sha = validate_pilot_gate_manifest(gate)
    monkeypatch.setattr(
        authorization_module,
        "load_stage_a_campaign_evidence",
        lambda *args, **kwargs: _evidence("f" * 64, gate_sha),
    )
    with pytest.raises(RuntimeError, match="dataset identity"):
        authorize_progressive_sweep(
            "stage-a.json",
            dataset_manifest=dataset,
            gate_manifest=gate,
            sweep_id="core50-001",
            code_commit="a" * 40,
        )


def test_progressive_authorization_requires_full_source_commit(monkeypatch) -> None:
    dataset = _dataset()
    gate = _gate()
    with pytest.raises(ValueError, match="full 40-hex"):
        authorize_progressive_sweep(
            "stage-a.json",
            dataset_manifest=dataset,
            gate_manifest=gate,
            sweep_id="core50-001",
            code_commit="deadbeef",
        )


def test_prefix_manifest_is_immutable_canonical_and_hash_chained(tmp_path: Path, monkeypatch) -> None:
    dataset = _dataset()
    gate = _gate()
    dataset_sha = validate_pilot_dataset_manifest(
        dataset,
        required_coverage_tags=("short", "long", "reference", "audio", "mixed-grid"),
    )
    gate_sha = validate_pilot_gate_manifest(gate)
    evidence = _evidence(dataset_sha, gate_sha)
    monkeypatch.setattr(
        authorization_module,
        "load_stage_a_campaign_evidence",
        lambda *args, **kwargs: evidence,
    )
    prefix = authorize_progressive_sweep(
        "stage-a.json",
        dataset_manifest=dataset,
        gate_manifest=gate,
        sweep_id="core50-001",
        code_commit="b" * 40,
    ).prefix

    initial_path = tmp_path / "core50-001.prefix-00.json"
    initial_sha = write_progressive_prefix_manifest(
        initial_path,
        prefix,
        previous_manifest_sha256=None,
    )
    loaded, loaded_sha = load_progressive_prefix_manifest(initial_path)
    assert loaded == prefix
    assert loaded_sha == initial_sha
    with pytest.raises(FileExistsError, match="already exists"):
        write_progressive_prefix_manifest(
            initial_path,
            prefix,
            previous_manifest_sha256=None,
        )

    advanced = prefix.advance(
        block_index=0,
        final_stage="route",
        checkpoint_sha256="3" * 64,
        result_sha256="4" * 64,
    )
    next_path = tmp_path / "core50-001.prefix-01.json"
    next_sha = write_progressive_prefix_manifest(
        next_path,
        advanced,
        previous_manifest_sha256=initial_sha,
    )
    loaded_advanced, loaded_next_sha = load_progressive_prefix_manifest(
        next_path,
        expected_previous_manifest_sha256=initial_sha,
    )
    assert loaded_advanced == advanced
    assert loaded_next_sha == next_sha

    with pytest.raises(RuntimeError, match="expected prior manifest"):
        load_progressive_prefix_manifest(
            next_path,
            expected_previous_manifest_sha256="5" * 64,
        )


def test_prefix_manifest_detects_identity_tamper(tmp_path: Path) -> None:
    from minimax_h3_keyless.progressive import ProgressivePrefix

    prefix = ProgressivePrefix(
        sweep_id="tamper",
        code_commit="c" * 40,
        stage_a_campaign_sha256="1" * 64,
        dataset_manifest_sha256="2" * 64,
        gate_manifest_sha256="3" * 64,
    )
    path = tmp_path / "prefix.json"
    write_progressive_prefix_manifest(path, prefix, previous_manifest_sha256=None)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["prefix_identity_sha256"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity does not recompute"):
        load_progressive_prefix_manifest(path)
