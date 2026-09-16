from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import minimax_h3_keyless.progressive_workflow as workflow_module
from minimax_h3_keyless.attention import KeylessAttentionDeploy
from minimax_h3_keyless.progressive import ProgressivePrefix, validate_progressive_model_prefix
from minimax_h3_keyless.progressive_artifacts import ProgressiveArtifactReceipt
from minimax_h3_keyless.progressive_authorization import (
    load_progressive_prefix_manifest,
    write_progressive_prefix_manifest,
)
from minimax_h3_keyless.progressive_workflow import persist_accept_progressive_block


class NativeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(4, 12, bias=False)
        self.q_norm = nn.RMSNorm(2, eps=1e-5)
        self.k_norm = nn.RMSNorm(2, eps=1e-5)
        self.out_proj = nn.Linear(4, 4, bias=False)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = NativeAttention()


class Core50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(50))


def _prefix() -> ProgressivePrefix:
    return ProgressivePrefix(
        sweep_id="workflow",
        code_commit="e" * 40,
        stage_a_campaign_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        gate_manifest_sha256="c" * 64,
    )


def _artifact(tmp_path: Path) -> ProgressiveArtifactReceipt:
    return ProgressiveArtifactReceipt(
        checkpoint_path=str(tmp_path / "candidate.pt"),
        checkpoint_sha256="1" * 64,
        result_path=str(tmp_path / "candidate.json"),
        result_sha256="2" * 64,
        result_payload_sha256="3" * 64,
        capture_set_identity_sha256="4" * 64,
    )


def _install_fakes(monkeypatch, model: Core50, prefix: ProgressivePrefix, artifact):
    calls = []

    def validate_captures(captures):
        calls.append("validate-captures")

    def persist(output_dir, *, prefix, captures, result):
        calls.append("persist")
        return artifact

    def accept(model_arg, prefix_arg, captures, result, receipt, **kwargs):
        calls.append("accept")
        assert model_arg is model
        assert prefix_arg is prefix
        assert receipt is artifact
        model.blocks[0].attn = KeylessAttentionDeploy(
            4,
            2,
            2,
            1e-5,
            block_index=0,
            dtype=torch.float32,
        )
        return prefix.advance(
            block_index=0,
            final_stage="route",
            checkpoint_sha256=artifact.checkpoint_sha256,
            result_sha256=artifact.result_sha256,
        )

    monkeypatch.setattr(workflow_module, "validate_progressive_capture_set_integrity", validate_captures)
    monkeypatch.setattr(workflow_module, "persist_progressive_block_artifacts", persist)
    monkeypatch.setattr(workflow_module, "accept_progressive_block", accept)
    return calls


def test_progressive_step_persists_accepts_then_hash_chains_prefix_manifest(tmp_path: Path, monkeypatch) -> None:
    model = Core50()
    prefix = _prefix()
    current = tmp_path / "workflow.prefix-00.json"
    current_sha = write_progressive_prefix_manifest(current, prefix, previous_manifest_sha256=None)
    artifact = _artifact(tmp_path)
    calls = _install_fakes(monkeypatch, model, prefix, artifact)

    outcome = persist_accept_progressive_block(
        model,
        prefix,
        SimpleNamespace(),
        SimpleNamespace(block_index=0),
        current_prefix_manifest_path=current,
        current_prefix_manifest_sha256=current_sha,
        output_dir=tmp_path,
        gate_manifest={},
        fold_atol=0.0,
        fold_rtol=0.0,
    )

    assert calls == ["validate-captures", "persist", "accept"]
    assert outcome.prefix.accepted_blocks == (0,)
    assert isinstance(model.blocks[0].attn, KeylessAttentionDeploy)
    assert outcome.prefix_manifest_path.endswith("workflow.prefix-01.json")
    loaded, loaded_sha = load_progressive_prefix_manifest(
        outcome.prefix_manifest_path,
        expected_previous_manifest_sha256=current_sha,
    )
    assert loaded == outcome.prefix
    assert loaded_sha == outcome.prefix_manifest_sha256
    validate_progressive_model_prefix(model, outcome.prefix)


def test_progressive_step_rolls_live_model_back_if_prefix_publish_fails(tmp_path: Path, monkeypatch) -> None:
    model = Core50()
    prefix = _prefix()
    original_block = model.blocks[0]
    current = tmp_path / "workflow.prefix-00.json"
    current_sha = write_progressive_prefix_manifest(current, prefix, previous_manifest_sha256=None)
    artifact = _artifact(tmp_path)
    _install_fakes(monkeypatch, model, prefix, artifact)

    def fail_publish(*args, **kwargs):
        raise FileExistsError("simulated concurrent prefix writer")

    monkeypatch.setattr(workflow_module, "write_progressive_prefix_manifest", fail_publish)
    with pytest.raises(FileExistsError, match="concurrent prefix writer"):
        persist_accept_progressive_block(
            model,
            prefix,
            SimpleNamespace(),
            SimpleNamespace(block_index=0),
            current_prefix_manifest_path=current,
            current_prefix_manifest_sha256=current_sha,
            output_dir=tmp_path,
            gate_manifest={},
            fold_atol=0.0,
            fold_rtol=0.0,
        )

    assert model.blocks[0] is original_block
    assert prefix.accepted_blocks == ()
    validate_progressive_model_prefix(model, prefix)


def test_progressive_step_rejects_stale_current_manifest_before_persistence(tmp_path: Path, monkeypatch) -> None:
    model = Core50()
    prefix = _prefix()
    current = tmp_path / "workflow.prefix-00.json"
    write_progressive_prefix_manifest(current, prefix, previous_manifest_sha256=None)
    called = False

    def persist(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("must not persist from a stale prefix")

    monkeypatch.setattr(workflow_module, "persist_progressive_block_artifacts", persist)
    with pytest.raises(RuntimeError, match="SHA-256 changed"):
        persist_accept_progressive_block(
            model,
            prefix,
            SimpleNamespace(),
            SimpleNamespace(block_index=0),
            current_prefix_manifest_path=current,
            current_prefix_manifest_sha256="f" * 64,
            output_dir=tmp_path,
            gate_manifest={},
            fold_atol=0.0,
            fold_rtol=0.0,
        )
    assert called is False
