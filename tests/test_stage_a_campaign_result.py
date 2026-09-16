from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

import minimax_h3_keyless.stage_a_campaign_result as campaign_module
from minimax_h3_keyless.contracts import TEACHER_SHA256
from minimax_h3_keyless.pilot_artifacts import STAGE_A_RESULT_SCHEMA, StageAArtifactReceipt
from minimax_h3_keyless.pilot_campaign import GATE_SCHEMA, validate_pilot_gate_manifest
from minimax_h3_keyless.pilot_completed import CompletedStageABlockEvidence
from minimax_h3_keyless.pilot_gates import StageABlockGateResult, evaluate_stage_a_campaign_gate
from minimax_h3_keyless.stage_a_campaign_result import (
    STAGE_A_CAMPAIGN_RESULT_SCHEMA,
    load_stage_a_campaign_evidence,
)


def _gate_manifest():
    return {
        "schema": GATE_SCHEMA,
        "thresholds": {
            "stage_a_min_case_total_improvement_fraction": 0.0,
            "stage_a_min_mean_total_relative_improvement": 0.0,
            "stage_a_max_mean_attention_normalized_mse": 1.0,
            "stage_a_max_mean_block_normalized_mse": 1.0,
            "stage_a_max_worst_attention_normalized_mse": 1.0,
            "stage_a_max_worst_block_normalized_mse": 1.0,
            "stage_a_min_mean_attention_cosine": -1.0,
            "stage_a_min_mean_block_cosine": -1.0,
            "stage_a_minimum_attention_cosine": -1.0,
            "stage_a_minimum_block_cosine": -1.0,
            "stage_a_max_modality_nmse_ratio_to_best_baseline": 10.0,
            "stage_a_max_gradient_l2_norm": 10.0,
        },
        "calibration_evidence": {"fixture": "campaign"},
    }


def _block_gate(index: int, *, passed: bool = True) -> StageABlockGateResult:
    failures = () if passed else ("synthetic failure",)
    return StageABlockGateResult(
        block_index=index,
        passed=passed,
        failures=failures,
        case_fraction_improved_over_both=1.0 if passed else 0.0,
        mean_total_relative_improvement_vs_identity=0.5 if passed else -0.1,
        mean_total_relative_improvement_vs_least_squares=0.5 if passed else -0.1,
        maximum_modality_attention_nmse_ratio=0.5,
        maximum_modality_block_nmse_ratio=0.5,
        maximum_gradient_l2_norm=1.0,
    )


def _receipt(root: Path, index: int) -> StageAArtifactReceipt:
    stem = f"run.block{index:02d}.route"
    return StageAArtifactReceipt(
        checkpoint_path=str(root / f"{stem}.resume.pt"),
        checkpoint_sha256=(f"{index + 1:02x}" * 32),
        result_path=str(root / f"{stem}.result.json"),
        result_sha256=(f"{index + 2:02x}" * 32),
    )


def _payload(root: Path, gate_manifest, *, failed_block: int | None = None):
    gate_sha = validate_pilot_gate_manifest(gate_manifest)
    block_gates = {
        index: _block_gate(index, passed=index != failed_block) for index in (0, 25, 49)
    }
    receipts = {index: _receipt(root, index) for index in (0, 25, 49)}
    payload = {
        "schema": STAGE_A_CAMPAIGN_RESULT_SCHEMA,
        "stage_a_block_result_schema": STAGE_A_RESULT_SCHEMA,
        "run_id": "run",
        "code_commit": "deadbeef",
        "training_comfy_commit": "comfy-deadbeef",
        "experiment_context_sha256": "c" * 64,
        "teacher_sha256": TEACHER_SHA256,
        "dataset_manifest_sha256": "a" * 64,
        "gate_manifest_sha256": gate_sha,
        "capture_registry_file_sha256": "b" * 64,
        "train_plan_identity_sha256": "d" * 64,
        "train_plan_file_sha256": "e" * 64,
        "capture_code_commit": "capture-code",
        "capture_comfy_commit": "capture-comfy",
        "capture_execution_descriptor": "native-bf16-dense",
        "final_stage": "route",
        "resumed_blocks": [],
        "executed_blocks": [0, 25, 49],
        "block_artifacts": {str(index): asdict(receipts[index]) for index in (0, 25, 49)},
        "gate": asdict(evaluate_stage_a_campaign_gate(block_gates)),
    }
    return payload, block_gates, receipts


def _install_fake_completed(monkeypatch, block_gates, receipts):
    def fake_load(request, *, block_index, final_stage, **kwargs):
        assert request.run_id == "run"
        assert request.code_commit == "deadbeef"
        assert request.experiment_context_sha256 == "c" * 64
        assert final_stage == "route"
        return CompletedStageABlockEvidence(
            block_index=block_index,
            identity=None,  # type: ignore[arg-type]
            stage=final_stage,
            step=1,
            gate=block_gates[block_index],
            artifact=receipts[block_index],
        )

    monkeypatch.setattr(campaign_module, "load_completed_stage_a_block_evidence", fake_load)


def test_campaign_exit_recomputes_all_three_block_gates(tmp_path: Path, monkeypatch) -> None:
    gate_manifest = _gate_manifest()
    payload, block_gates, receipts = _payload(tmp_path, gate_manifest)
    path = tmp_path / "run.campaign.result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _install_fake_completed(monkeypatch, block_gates, receipts)

    evidence = load_stage_a_campaign_evidence(path, gate_manifest=gate_manifest)
    assert evidence.gate.passed is True
    assert set(evidence.block_evidence) == {0, 25, 49}
    assert evidence.final_stage == "route"
    assert evidence.teacher_sha256 == TEACHER_SHA256


def test_campaign_exit_refuses_failed_depth_pilot(tmp_path: Path, monkeypatch) -> None:
    gate_manifest = _gate_manifest()
    payload, block_gates, receipts = _payload(tmp_path, gate_manifest, failed_block=25)
    path = tmp_path / "run.campaign.result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _install_fake_completed(monkeypatch, block_gates, receipts)

    with pytest.raises(RuntimeError, match="core50 sweep is forbidden"):
        load_stage_a_campaign_evidence(path, gate_manifest=gate_manifest, require_passed=True)
    evidence = load_stage_a_campaign_evidence(
        path, gate_manifest=gate_manifest, require_passed=False
    )
    assert evidence.gate.passed is False


def test_campaign_exit_rejects_artifact_hash_claim_mismatch(tmp_path: Path, monkeypatch) -> None:
    gate_manifest = _gate_manifest()
    payload, block_gates, receipts = _payload(tmp_path, gate_manifest)
    payload["block_artifacts"]["25"]["result_sha256"] = "f" * 64
    path = tmp_path / "run.campaign.result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _install_fake_completed(monkeypatch, block_gates, receipts)

    with pytest.raises(RuntimeError, match="result hash is inconsistent"):
        load_stage_a_campaign_evidence(path, gate_manifest=gate_manifest)


def test_campaign_exit_rejects_old_block_result_schema(tmp_path: Path, monkeypatch) -> None:
    gate_manifest = _gate_manifest()
    payload, block_gates, receipts = _payload(tmp_path, gate_manifest)
    payload["stage_a_block_result_schema"] = "minimax_h3_keyless_stage_a_block_result_v1"
    path = tmp_path / "run.campaign.result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _install_fake_completed(monkeypatch, block_gates, receipts)

    with pytest.raises(RuntimeError, match="incompatible block-result schema"):
        load_stage_a_campaign_evidence(path, gate_manifest=gate_manifest)
