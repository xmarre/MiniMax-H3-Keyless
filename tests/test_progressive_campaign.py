from __future__ import annotations

from types import SimpleNamespace

import pytest

import minimax_h3_keyless.progressive_campaign as campaign
from minimax_h3_keyless.progressive import ProgressivePrefix
from minimax_h3_keyless.progressive_artifacts import ProgressiveArtifactReceipt
from minimax_h3_keyless.progressive_campaign import (
    ProgressiveBlockRunInputs,
    load_progressive_block_run_inputs,
    run_progressive_block_campaign,
)
from minimax_h3_keyless.progressive_gates import ProgressiveExecutionPolicy


def _prefix() -> ProgressivePrefix:
    return ProgressivePrefix(
        sweep_id="campaign",
        code_commit="e" * 40,
        stage_a_campaign_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        gate_manifest_sha256="c" * 64,
    )


def _artifact() -> ProgressiveArtifactReceipt:
    return ProgressiveArtifactReceipt(
        checkpoint_path="candidate.resume.pt",
        checkpoint_sha256="1" * 64,
        result_path="candidate.result.json",
        result_sha256="2" * 64,
        result_payload_sha256="3" * 64,
        capture_set_identity_sha256="4" * 64,
    )


def _inputs() -> ProgressiveBlockRunInputs:
    prefix = _prefix()
    return ProgressiveBlockRunInputs(
        prefix=prefix,
        prefix_manifest_path="current-prefix.json",
        prefix_manifest_sha256="d" * 64,
        registry=SimpleNamespace(
            registry_path="registry.json",
            registry_file_sha256="f" * 64,
            code_commit=prefix.code_commit,
            comfy_commit="comfy",
            execution_descriptor="fixture",
            artifacts=(),
        ),
        stage_a=SimpleNamespace(),
        dataset_manifest={},
        gate_manifest={},
        train_plan=SimpleNamespace(
            stages=(SimpleNamespace(stage="route"),),
            loss_weights=SimpleNamespace(),
            same_input_atol=0.0,
            same_input_rtol=0.0,
        ),
        execution_policy=ProgressiveExecutionPolicy(fold_atol=0.002, fold_rtol=0.003),
    )


def _install_runtime_fakes(monkeypatch, *, gate_passed: bool):
    model = SimpleNamespace(blocks=[object() for _ in range(50)])
    captures = SimpleNamespace()
    artifact = _artifact()
    calls = []

    monkeypatch.setattr(campaign, "require_progressive_runtime_provenance", lambda inputs: "comfy")
    monkeypatch.setattr(campaign, "load_progressive_training_captures", lambda inputs: captures)
    monkeypatch.setattr(
        campaign,
        "load_pinned_bf16_teacher",
        lambda path: SimpleNamespace(diffusion_model=model),
    )
    monkeypatch.setattr(
        campaign,
        "restore_progressive_model_prefix",
        lambda model_arg, prefix, *, output_dir: calls.append(("restore", output_dir)),
    )
    result = SimpleNamespace(
        block_index=0,
        gate=SimpleNamespace(passed=gate_passed),
    )
    monkeypatch.setattr(
        campaign,
        "run_progressive_block_training",
        lambda *args, **kwargs: result,
    )
    monkeypatch.setattr(
        campaign,
        "persist_progressive_block_artifacts",
        lambda *args, **kwargs: artifact,
    )
    return model, captures, artifact, result, calls


def test_failed_progressive_candidate_is_persisted_but_never_accepted(monkeypatch) -> None:
    _, _, artifact, _, calls = _install_runtime_fakes(monkeypatch, gate_passed=False)

    def forbidden_accept(*args, **kwargs):
        raise AssertionError("failed progressive candidate must not reach acceptance")

    monkeypatch.setattr(campaign, "accept_persisted_progressive_block", forbidden_accept)
    outcome = run_progressive_block_campaign(
        _inputs(),
        teacher_path="teacher.safetensors",
        artifact_dir="artifacts",
        device="cuda:0",
    )

    assert calls == [("restore", "artifacts")]
    assert outcome.gate_passed is False
    assert outcome.artifact is artifact
    assert outcome.accepted is None


def test_passed_progressive_candidate_accepts_the_same_persisted_artifact(monkeypatch) -> None:
    _, captures, artifact, result, _ = _install_runtime_fakes(monkeypatch, gate_passed=True)
    accepted = SimpleNamespace(prefix_manifest_path="next.json")
    seen = {}

    def fake_accept(model, prefix, supplied_captures, supplied_result, supplied_artifact, **kwargs):
        seen.update(
            captures=supplied_captures,
            result=supplied_result,
            artifact=supplied_artifact,
            kwargs=kwargs,
        )
        return accepted

    monkeypatch.setattr(campaign, "accept_persisted_progressive_block", fake_accept)
    inputs = _inputs()
    outcome = run_progressive_block_campaign(
        inputs,
        teacher_path="teacher.safetensors",
        artifact_dir="artifacts",
        device="cuda:0",
    )

    assert outcome.gate_passed is True
    assert outcome.accepted is accepted
    assert seen["captures"] is captures
    assert seen["result"] is result
    assert seen["artifact"] is artifact
    assert seen["kwargs"]["fold_atol"] == 0.002
    assert seen["kwargs"]["fold_rtol"] == 0.003
    assert seen["kwargs"]["current_prefix_manifest_sha256"] == inputs.prefix_manifest_sha256


def test_progressive_input_binding_rejects_train_plan_drift(monkeypatch) -> None:
    prefix = _prefix()
    stage_a = SimpleNamespace(
        sha256=prefix.stage_a_campaign_sha256,
        dataset_manifest_sha256=prefix.dataset_manifest_sha256,
        gate_manifest_sha256=prefix.gate_manifest_sha256,
        train_plan_identity_sha256="1" * 64,
        train_plan_file_sha256="2" * 64,
    )
    plan = SimpleNamespace(
        plan_identity_sha256="9" * 64,
        plan_file_sha256="2" * 64,
    )
    monkeypatch.setattr(campaign, "load_json_manifest", lambda path: {"path": str(path)})
    monkeypatch.setattr(
        campaign,
        "validate_pilot_dataset_manifest",
        lambda manifest, **kwargs: prefix.dataset_manifest_sha256,
    )
    monkeypatch.setattr(
        campaign,
        "progressive_execution_policy_from_gate_manifest",
        lambda manifest: ProgressiveExecutionPolicy(0.0, 0.0),
    )
    monkeypatch.setattr(campaign, "load_stage_a_run_plan", lambda path: plan)
    monkeypatch.setattr(
        campaign,
        "load_progressive_prefix_manifest",
        lambda path: (prefix, "d" * 64),
    )
    monkeypatch.setattr(
        campaign,
        "load_stage_a_campaign_evidence",
        lambda *args, **kwargs: stage_a,
    )

    with pytest.raises(RuntimeError, match="train-plan semantic"):
        load_progressive_block_run_inputs(
            stage_a_result_path="stage-a.json",
            current_prefix_manifest_path="prefix.json",
            capture_registry_path="registry.json",
            dataset_manifest_path="dataset.json",
            gate_manifest_path="gate.json",
            train_plan_path="plan.json",
        )
