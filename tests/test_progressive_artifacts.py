from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from minimax_h3_keyless.attention import KeylessAttentionTrain
from minimax_h3_keyless.pilot import PilotCase, PilotStepReport, set_pilot_block_stage
from minimax_h3_keyless.pilot_campaign import (
    DATASET_SCHEMA,
    PilotAggregateMetrics,
    PilotCaseMetrics,
    PilotTrainingEvent,
    canonical_json_sha256,
)
from minimax_h3_keyless.pilot_gates import StageABlockGateResult
from minimax_h3_keyless.progressive import PROGRESSIVE_PREFIX_CONTEXT_KEY, ProgressivePrefix
from minimax_h3_keyless.progressive_artifacts import (
    load_progressive_resume_checkpoint,
    persist_progressive_block_artifacts,
    progressive_run_identity,
)
from minimax_h3_keyless.progressive_capture_io import (
    ProgressiveCaptureProvenance,
    write_progressive_capture_bundle,
)
from minimax_h3_keyless.progressive_capture_set import (
    ProgressiveCaptureArtifactRef,
    load_progressive_block_capture_set,
)
from minimax_h3_keyless.progressive_runner import ProgressiveBlockTrainingResult


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


def _prefix(dataset_sha: str) -> ProgressivePrefix:
    return ProgressivePrefix(
        sweep_id="sweep-001",
        code_commit="deadbeef",
        stage_a_campaign_sha256="a" * 64,
        dataset_manifest_sha256=dataset_sha,
        gate_manifest_sha256="c" * 64,
    )


def _capture_artifact(root: Path, prefix: ProgressivePrefix, case_id: str):
    x = torch.randn(3, 4)
    record = __import__(
        "minimax_h3_keyless.activation_capture", fromlist=["CapturedPilotCase"]
    ).CapturedPilotCase(
        block_index=0,
        case=PilotCase(
            x=x,
            t_emb=torch.zeros(1, 1),
            mod_segments=((0, 3, 0),),
            rope_freqs=None,
            transformer_options={},
            case_id=case_id,
            sigma=0.5,
            modality_label="video",
            context={PROGRESSIVE_PREFIX_CONTEXT_KEY: prefix.capture_context()},
        ),
        attention_input=x.clone(),
        captured_bytes=128,
    )
    provenance = ProgressiveCaptureProvenance(
        code_commit=prefix.code_commit,
        comfy_commit="comfy-fixture",
        dataset_manifest_sha256=prefix.dataset_manifest_sha256,
        gate_manifest_sha256=prefix.gate_manifest_sha256,
        stage_a_campaign_sha256=prefix.stage_a_campaign_sha256,
        prefix_identity_sha256=prefix.identity_sha256,
        target_block=0,
        execution_descriptor="progressive-fixture",
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


def _captures(tmp_path: Path, prefix: ProgressivePrefix, manifest):
    refs = (
        _capture_artifact(tmp_path, prefix, "train"),
        _capture_artifact(tmp_path, prefix, "holdout"),
    )
    return load_progressive_block_capture_set(
        refs,
        manifest,
        prefix,
        minimum_cases=2,
        minimum_sigma_strata=1,
    )


def _metrics() -> PilotAggregateMetrics:
    case = PilotCaseMetrics(
        case_id="holdout::sigma=0.5",
        modality_label="video",
        sigma=0.5,
        rows=3,
        total=0.1,
        attention_normalized_mse=0.05,
        block_normalized_mse=0.05,
        attention_cosine=0.99,
        block_cosine=0.99,
    )
    return PilotAggregateMetrics(
        case_count=1,
        mean_total=0.1,
        mean_attention_normalized_mse=0.05,
        mean_block_normalized_mse=0.05,
        mean_attention_cosine=0.99,
        mean_block_cosine=0.99,
        worst_attention_normalized_mse=0.05,
        worst_block_normalized_mse=0.05,
        minimum_attention_cosine=0.99,
        minimum_block_cosine=0.99,
        by_modality={
            "video": {
                "case_count": 1,
                "mean_attention_normalized_mse": 0.05,
                "mean_block_normalized_mse": 0.05,
                "mean_attention_cosine": 0.99,
                "mean_block_cosine": 0.99,
            }
        },
        cases=(case,),
    )


def _result(prefix: ProgressivePrefix):
    student = nn.Module()
    student.attn = KeylessAttentionTrain(4, 2, 2, 1e-5, block_index=0, dtype=torch.float32)
    set_pilot_block_stage(student, "route")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    metrics = _metrics()
    event = PilotTrainingEvent(
        stage="route",
        epoch=0,
        case_id="train::sigma=0.5",
        report=PilotStepReport(
            total=0.2,
            attention_normalized_mse=0.1,
            block_normalized_mse=0.1,
            attention_cosine=0.95,
            block_cosine=0.95,
            trainable_parameters=sum(p.numel() for p in student.parameters() if p.requires_grad),
            gradient_l2_norm=1.0,
        ),
    )
    gate = StageABlockGateResult(
        block_index=0,
        passed=True,
        failures=(),
        case_fraction_improved_over_both=1.0,
        mean_total_relative_improvement_vs_identity=0.5,
        mean_total_relative_improvement_vs_least_squares=0.4,
        maximum_modality_attention_nmse_ratio=0.5,
        maximum_modality_block_nmse_ratio=0.5,
        maximum_gradient_l2_norm=1.0,
    )
    return ProgressiveBlockTrainingResult(
        block_index=0,
        prefix_identity_sha256=prefix.identity_sha256,
        selection_split="train_complete_cases",
        replay_reports=(),
        initialization_evaluations=(),
        selected_route_mode="identity",
        selected_lambda_relative=0.0,
        identity_baseline=metrics,
        least_squares_baseline=metrics,
        least_squares_baseline_lambda_relative=0.0,
        candidate=metrics,
        candidate_attention_diagnostics=(),
        training_events=(event,),
        gate=gate,
        final_stage="route",
        student_block=student,
        optimizer=optimizer,
    )


def test_progressive_artifacts_preserve_resume_state_and_never_overwrite(tmp_path: Path) -> None:
    torch.manual_seed(801)
    manifest = _manifest()
    prefix = _prefix(canonical_json_sha256(manifest))
    captures = _captures(tmp_path / "captures", prefix, manifest)
    result = _result(prefix)
    identity = progressive_run_identity(prefix, captures, result)

    receipt = persist_progressive_block_artifacts(
        tmp_path / "results",
        prefix=prefix,
        captures=captures,
        result=result,
    )
    assert Path(receipt.checkpoint_path).exists()
    assert Path(receipt.result_path).exists()
    assert receipt.capture_set_identity_sha256 == identity.capture_set_identity_sha256

    expected = {
        name: tensor.detach().clone()
        for name, tensor in result.student_block.state_dict().items()
    }
    with torch.no_grad():
        for parameter in result.student_block.parameters():
            parameter.add_(10.0)
    loaded = load_progressive_resume_checkpoint(
        receipt.checkpoint_path,
        student_block=result.student_block,
        optimizer=result.optimizer,
        expected_identity=identity,
        expected_stage="route",
        expected_sha256=receipt.checkpoint_sha256,
        restore_rng=False,
    )
    assert loaded["result_payload_sha256"] == receipt.result_payload_sha256
    for name, tensor in result.student_block.state_dict().items():
        torch.testing.assert_close(tensor, expected[name])

    with pytest.raises(FileExistsError, match="immutable"):
        persist_progressive_block_artifacts(
            tmp_path / "results",
            prefix=prefix,
            captures=captures,
            result=result,
        )


def test_progressive_resume_rejects_checkpoint_tamper_before_deserialization(tmp_path: Path) -> None:
    manifest = _manifest()
    prefix = _prefix(canonical_json_sha256(manifest))
    captures = _captures(tmp_path / "captures", prefix, manifest)
    result = _result(prefix)
    identity = progressive_run_identity(prefix, captures, result)
    receipt = persist_progressive_block_artifacts(
        tmp_path / "results",
        prefix=prefix,
        captures=captures,
        result=result,
    )
    with open(receipt.checkpoint_path, "ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(RuntimeError, match="SHA-256"):
        load_progressive_resume_checkpoint(
            receipt.checkpoint_path,
            student_block=result.student_block,
            optimizer=result.optimizer,
            expected_identity=identity,
            expected_stage="route",
            expected_sha256=receipt.checkpoint_sha256,
            restore_rng=False,
        )
