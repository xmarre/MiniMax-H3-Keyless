from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from minimax_h3_keyless.attention import KeylessAttentionTrain
from minimax_h3_keyless.pilot import PilotStepReport, set_pilot_block_stage
from minimax_h3_keyless.pilot_artifacts import StageAArtifactRequest, persist_stage_a_block_artifacts
from minimax_h3_keyless.pilot_campaign import (
    GATE_SCHEMA,
    PilotAggregateMetrics,
    PilotCaseMetrics,
    PilotRunIdentity,
    PilotTrainingEvent,
    validate_pilot_gate_manifest,
)
from minimax_h3_keyless.pilot_completed import load_completed_stage_a_block_evidence
from minimax_h3_keyless.pilot_gates import evaluate_stage_a_block_gate, stage_a_policy_from_gate_manifest
from minimax_h3_keyless.pilot_runner import StageAInitializationEvaluation


class TinyStudentBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = KeylessAttentionTrain(4, 2, 2, 1e-5, block_index=25, dtype=torch.float32)
        set_pilot_block_stage(self, "route")


def _metric(total: float, attn: float, block: float) -> PilotAggregateMetrics:
    case = PilotCaseMetrics(
        case_id="hold::sigma=0.5",
        modality_label="video",
        sigma=0.5,
        rows=2,
        total=total,
        attention_normalized_mse=attn,
        block_normalized_mse=block,
        attention_cosine=0.95,
        block_cosine=0.96,
    )
    return PilotAggregateMetrics(
        case_count=1,
        mean_total=total,
        mean_attention_normalized_mse=attn,
        mean_block_normalized_mse=block,
        mean_attention_cosine=0.95,
        mean_block_cosine=0.96,
        worst_attention_normalized_mse=attn,
        worst_block_normalized_mse=block,
        minimum_attention_cosine=0.95,
        minimum_block_cosine=0.96,
        by_modality={
            "video": {
                "case_count": 1,
                "mean_attention_normalized_mse": attn,
                "mean_block_normalized_mse": block,
                "mean_attention_cosine": 0.95,
                "mean_block_cosine": 0.96,
            }
        },
        cases=(case,),
    )


def _gate_manifest():
    return {
        "schema": GATE_SCHEMA,
        "thresholds": {
            "stage_a_min_case_total_improvement_fraction": 0.0,
            "stage_a_min_mean_total_relative_improvement": 0.0,
            "stage_a_max_mean_attention_normalized_mse": 10.0,
            "stage_a_max_mean_block_normalized_mse": 10.0,
            "stage_a_max_worst_attention_normalized_mse": 10.0,
            "stage_a_max_worst_block_normalized_mse": 10.0,
            "stage_a_min_mean_attention_cosine": -1.0,
            "stage_a_min_mean_block_cosine": -1.0,
            "stage_a_minimum_attention_cosine": -1.0,
            "stage_a_minimum_block_cosine": -1.0,
            "stage_a_max_modality_nmse_ratio_to_best_baseline": 10.0,
            "stage_a_max_gradient_l2_norm": 10.0,
        },
        "calibration_evidence": {"fixture": "fixed"},
    }


def _write_completed(tmp_path: Path):
    gate_manifest = _gate_manifest()
    gate_sha = validate_pilot_gate_manifest(gate_manifest)
    identity_metric = _metric(0.6, 0.35, 0.25)
    ls0 = _metric(0.55, 0.30, 0.25)
    selected_metric = _metric(0.50, 0.25, 0.25)
    ls2 = _metric(0.70, 0.45, 0.25)
    candidate = _metric(0.20, 0.10, 0.10)
    evaluations = (
        StageAInitializationEvaluation("identity", 0.0, identity_metric),
        StageAInitializationEvaluation("least_squares", 0.0, ls0),
        StageAInitializationEvaluation("least_squares", 1e-4, selected_metric),
        StageAInitializationEvaluation("least_squares", 1e-2, ls2),
    )
    event = PilotTrainingEvent(
        stage="route",
        epoch=0,
        case_id="train::sigma=0.5",
        report=PilotStepReport(
            total=0.3,
            attention_normalized_mse=0.2,
            block_normalized_mse=0.1,
            attention_cosine=0.9,
            block_cosine=0.9,
            trainable_parameters=12,
            gradient_l2_norm=0.5,
        ),
    )
    gate = evaluate_stage_a_block_gate(
        block_index=25,
        candidate=candidate,
        identity_baseline=identity_metric,
        least_squares_baseline=selected_metric,
        training_events=(event,),
        policy=stage_a_policy_from_gate_manifest(gate_manifest),
    )
    payload = {
        "block_index": 25,
        "replay_reports": [
            {
                "block_index": 25,
                "rows": 2,
                "attention_input_max_abs_error": 0.0,
                "attention_input_mean_abs_error": 0.0,
                "block_output_finite": True,
                "attention_output_finite": True,
            }
        ],
        "initialization_evaluations": [asdict(row) for row in evaluations],
        "selected_route_mode": "least_squares",
        "selected_lambda_relative": 1e-4,
        "identity_baseline": asdict(identity_metric),
        "least_squares_baseline": asdict(selected_metric),
        "candidate": asdict(candidate),
        "training_events": [asdict(event)],
        "gate": asdict(gate),
    }
    student = TinyStudentBlock()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad], lr=1e-3
    )
    context_sha = "c" * 64
    request = StageAArtifactRequest(
        str(tmp_path), "resume-run", "deadbeef", experiment_context_sha256=context_sha
    )
    identity = PilotRunIdentity(
        run_id="resume-run",
        code_commit="deadbeef",
        dataset_manifest_sha256="a" * 64,
        gate_manifest_sha256=gate_sha,
        block_index=25,
        route_mode="least_squares",
        lambda_relative=1e-4,
    )
    receipt = persist_stage_a_block_artifacts(
        request,
        student_block=student,
        optimizer=optimizer,
        identity=identity,
        stage="route",
        step=1,
        result_payload=payload,
    )
    return request, gate_manifest, gate_sha, receipt


def test_completed_block_evidence_recomputes_bound_gate_and_accepts_exact_context(tmp_path: Path) -> None:
    request, gate_manifest, gate_sha, receipt = _write_completed(tmp_path)
    completed = load_completed_stage_a_block_evidence(
        request,
        block_index=25,
        final_stage="route",
        expected_dataset_manifest_sha256="a" * 64,
        expected_gate_manifest_sha256=gate_sha,
        gate_manifest=gate_manifest,
    )
    assert completed is not None
    assert completed.block_index == 25
    assert completed.gate.passed is True
    assert completed.artifact.checkpoint_sha256 == receipt.checkpoint_sha256
    assert completed.artifact.result_sha256 == receipt.result_sha256


def test_completed_block_evidence_rejects_tampered_result_payload(tmp_path: Path) -> None:
    request, gate_manifest, gate_sha, receipt = _write_completed(tmp_path)
    path = Path(receipt.result_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["result"]["candidate"]["mean_total"] = 9.0
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="payload hash"):
        load_completed_stage_a_block_evidence(
            request,
            block_index=25,
            final_stage="route",
            expected_dataset_manifest_sha256="a" * 64,
            expected_gate_manifest_sha256=gate_sha,
            gate_manifest=gate_manifest,
        )


def test_completed_block_evidence_rejects_partial_pair_and_wrong_experiment_context(tmp_path: Path) -> None:
    request, gate_manifest, gate_sha, receipt = _write_completed(tmp_path)
    Path(receipt.result_path).unlink()
    with pytest.raises(RuntimeError, match="partial Stage-A evidence"):
        load_completed_stage_a_block_evidence(
            request,
            block_index=25,
            final_stage="route",
            expected_dataset_manifest_sha256="a" * 64,
            expected_gate_manifest_sha256=gate_sha,
            gate_manifest=gate_manifest,
        )

    other = StageAArtifactRequest(
        str(tmp_path), "other-run", "deadbeef", experiment_context_sha256="d" * 64
    )
    assert load_completed_stage_a_block_evidence(
        other,
        block_index=25,
        final_stage="route",
        expected_dataset_manifest_sha256="a" * 64,
        expected_gate_manifest_sha256=gate_sha,
        gate_manifest=gate_manifest,
    ) is None
