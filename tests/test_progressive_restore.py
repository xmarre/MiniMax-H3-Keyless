from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import pytest
import torch
import torch.nn as nn

from minimax_h3_keyless.attention import KeylessAttentionDeploy
from minimax_h3_keyless.checkpoint import sha256_file
from minimax_h3_keyless.pilot import build_training_student_block
from minimax_h3_keyless.pilot_campaign import canonical_json_sha256
from minimax_h3_keyless.progressive import ProgressivePrefix, validate_progressive_model_prefix
from minimax_h3_keyless.progressive_acceptance import fold_progressive_training_block
from minimax_h3_keyless.progressive_artifacts import (
    PROGRESSIVE_RESULT_SCHEMA,
    PROGRESSIVE_RESUME_SCHEMA,
    ProgressiveRunIdentity,
)
from minimax_h3_keyless.progressive_restore import restore_progressive_model_prefix


class NativeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = 2
        self.head_dim = 2
        self.qkv_proj = nn.Linear(4, 12, bias=False, dtype=torch.bfloat16)
        self.q_norm = nn.RMSNorm(2, eps=1e-5, dtype=torch.bfloat16)
        self.k_norm = nn.RMSNorm(2, eps=1e-5, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = NativeAttention()
        self.frozen_mlp = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)


class Core50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(50))


def _empty_prefix() -> ProgressivePrefix:
    return ProgressivePrefix(
        sweep_id="restore-test",
        code_commit="e" * 40,
        stage_a_campaign_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        gate_manifest_sha256="c" * 64,
    )


def _append_artifact(
    output_dir: Path,
    model: Core50,
    prefix: ProgressivePrefix,
    *,
    payload_overrides: dict | None = None,
    checkpoint_step: int = 0,
    state_mutator: Callable[[dict[str, torch.Tensor]], None] | None = None,
):
    block_index = prefix.next_block
    assert block_index is not None
    student, _ = build_training_student_block(
        model.blocks[block_index],
        block_index=block_index,
        route_mode="identity",
        lambda_relative=0.0,
    )
    with torch.no_grad():
        student.attn.query_route.weight.add_(block_index * 0.01)

    identity = ProgressiveRunIdentity(
        sweep_id=prefix.sweep_id,
        code_commit=prefix.code_commit,
        stage_a_campaign_sha256=prefix.stage_a_campaign_sha256,
        dataset_manifest_sha256=prefix.dataset_manifest_sha256,
        gate_manifest_sha256=prefix.gate_manifest_sha256,
        prefix_identity_sha256=prefix.identity_sha256,
        capture_set_identity_sha256=f"{block_index + 1:064x}",
        block_index=block_index,
        route_mode="identity",
        lambda_relative=0.0,
    )
    result_payload = {
        "block_index": block_index,
        "prefix_identity_sha256": prefix.identity_sha256,
        "selection_split": "train_complete_cases",
        "selected_route_mode": "identity",
        "selected_lambda_relative": 0.0,
        "least_squares_baseline_lambda_relative": 0.0,
        "gate": {"passed": True, "block_index": block_index},
        "final_stage": "route",
    }
    if payload_overrides:
        result_payload.update(payload_overrides)
    payload_sha = canonical_json_sha256(result_payload)

    stem = f"{prefix.sweep_id}.block{block_index:02d}.route"
    checkpoint_path = output_dir / f"{stem}.resume.pt"
    result_path = output_dir / f"{stem}.result.json"
    student_state = student.state_dict()
    if state_mutator is not None:
        state_mutator(student_state)
    checkpoint = {
        "schema": PROGRESSIVE_RESUME_SCHEMA,
        "identity": asdict(identity),
        "stage": "route",
        "step": checkpoint_step,
        "student_state_dict": student_state,
        "optimizer_state_dict": {},
        "rng_state": {},
        "result_schema": PROGRESSIVE_RESULT_SCHEMA,
        "result_payload_sha256": payload_sha,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)

    result = {
        "schema": PROGRESSIVE_RESULT_SCHEMA,
        "identity": asdict(identity),
        "stage": "route",
        "step": 0,
        "checkpoint_filename": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha,
        "result_payload_sha256": payload_sha,
        "result": result_payload,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_sha = sha256_file(result_path)
    advanced = prefix.advance(
        block_index=block_index,
        final_stage="route",
        checkpoint_sha256=checkpoint_sha,
        result_sha256=result_sha,
    )
    return advanced, student, checkpoint_path, result_path


def test_restore_reconstructs_folded_attention_without_copying_frozen_block(tmp_path: Path) -> None:
    artifact_model = Core50()
    prefix, student, _, _ = _append_artifact(tmp_path, artifact_model, _empty_prefix())
    expected = fold_progressive_training_block(student).attn.qv_proj.weight.detach().clone()

    model = Core50()
    original_block = model.blocks[0]
    original_mlp = model.blocks[0].frozen_mlp
    original_mlp_weight = model.blocks[0].frozen_mlp.weight

    restored = restore_progressive_model_prefix(model, prefix, output_dir=tmp_path)

    assert restored is model
    assert model.blocks[0] is original_block
    assert model.blocks[0].frozen_mlp is original_mlp
    assert model.blocks[0].frozen_mlp.weight is original_mlp_weight
    assert isinstance(model.blocks[0].attn, KeylessAttentionDeploy)
    torch.testing.assert_close(model.blocks[0].attn.qv_proj.weight, expected)
    assert isinstance(model.blocks[1].attn, NativeAttention)
    validate_progressive_model_prefix(model, prefix)


def test_restore_corrupt_later_checkpoint_keeps_failing_and_later_blocks_native(tmp_path: Path) -> None:
    artifact_model = Core50()
    prefix1, _, _, _ = _append_artifact(tmp_path, artifact_model, _empty_prefix())
    prefix2, _, second_checkpoint, _ = _append_artifact(tmp_path, artifact_model, prefix1)
    second_checkpoint.write_bytes(b"corrupted after acceptance")

    model = Core50()
    block1 = model.blocks[1]
    block2 = model.blocks[2]
    with pytest.raises(RuntimeError, match="checkpoint SHA-256"):
        restore_progressive_model_prefix(model, prefix2, output_dir=tmp_path)

    assert isinstance(model.blocks[0].attn, KeylessAttentionDeploy)
    assert model.blocks[1] is block1
    assert model.blocks[2] is block2
    assert isinstance(model.blocks[1].attn, NativeAttention)
    assert isinstance(model.blocks[2].attn, NativeAttention)
    validate_progressive_model_prefix(model, prefix1)


def test_restore_rejects_result_payload_route_mode_inconsistent_with_identity(tmp_path: Path) -> None:
    artifact_model = Core50()
    prefix, _, _, _ = _append_artifact(
        tmp_path,
        artifact_model,
        _empty_prefix(),
        payload_overrides={"selected_route_mode": "least_squares"},
    )
    model = Core50()

    with pytest.raises(RuntimeError, match="route mode differs"):
        restore_progressive_model_prefix(model, prefix, output_dir=tmp_path)

    assert isinstance(model.blocks[0].attn, NativeAttention)
    validate_progressive_model_prefix(model, _empty_prefix())


def test_restore_rejects_missing_train_selection_marker_before_mutation(tmp_path: Path) -> None:
    artifact_model = Core50()
    prefix, _, _, _ = _append_artifact(
        tmp_path,
        artifact_model,
        _empty_prefix(),
        payload_overrides={"selection_split": "holdout"},
    )
    model = Core50()

    with pytest.raises(RuntimeError, match="train-only initialization selection"):
        restore_progressive_model_prefix(model, prefix, output_dir=tmp_path)

    assert isinstance(model.blocks[0].attn, NativeAttention)
    validate_progressive_model_prefix(model, _empty_prefix())


def test_restore_rejects_checkpoint_step_inconsistent_with_result_before_mutation(tmp_path: Path) -> None:
    artifact_model = Core50()
    prefix, _, _, _ = _append_artifact(
        tmp_path,
        artifact_model,
        _empty_prefix(),
        checkpoint_step=1,
    )
    model = Core50()

    with pytest.raises(RuntimeError, match="checkpoint step differs"):
        restore_progressive_model_prefix(model, prefix, output_dir=tmp_path)

    assert isinstance(model.blocks[0].attn, NativeAttention)
    validate_progressive_model_prefix(model, _empty_prefix())


@pytest.mark.parametrize(
    "mutator",
    [
        lambda state: state.pop("attn.route_norm.weight"),
        lambda state: state.__setitem__("attn.unexpected.weight", torch.ones(1)),
        lambda state: state.__setitem__("unexpected.weight", torch.ones(1)),
    ],
)
def test_restore_rejects_noncanonical_checkpoint_state_before_mutation(
    tmp_path: Path,
    mutator,
) -> None:
    artifact_model = Core50()
    prefix, _, _, _ = _append_artifact(
        tmp_path,
        artifact_model,
        _empty_prefix(),
        state_mutator=mutator,
    )
    model = Core50()
    original_block = model.blocks[0]
    original_attention = model.blocks[0].attn
    original_mlp = model.blocks[0].frozen_mlp

    with pytest.raises(RuntimeError, match="checkpoint state keys"):
        restore_progressive_model_prefix(model, prefix, output_dir=tmp_path)

    assert model.blocks[0] is original_block
    assert model.blocks[0].attn is original_attention
    assert model.blocks[0].frozen_mlp is original_mlp
    validate_progressive_model_prefix(model, _empty_prefix())
