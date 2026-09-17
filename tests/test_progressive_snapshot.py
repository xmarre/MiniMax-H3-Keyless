from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

import minimax_h3_keyless.progressive_snapshot as snapshot
from minimax_h3_keyless.attention import KeylessAttentionDeploy
from minimax_h3_keyless.checkpoint import CheckpointValidationError, TensorSignature, sha256_file
from minimax_h3_keyless.contracts import (
    ADALN_BASIS_DIM,
    ADALN_BLOCK_OUT,
    ADALN_CURVE_GRID,
    CORE_BLOCKS,
    HEAD_DIM,
    HIDDEN_SIZE,
    INNER_DIM,
    TOKEN_REFINER_BLOCKS,
)
from minimax_h3_keyless.progressive import ProgressiveAcceptedBlock, ProgressivePrefix
from minimax_h3_keyless.teacher import TEACHER_TENSOR_COUNT


def _prefix(count: int = 2) -> ProgressivePrefix:
    accepted = tuple(
        ProgressiveAcceptedBlock(
            block_index=index,
            final_stage="route",
            checkpoint_sha256=f"{index + 1:064x}",
            result_sha256=f"{index + 101:064x}",
        )
        for index in range(count)
    )
    return ProgressivePrefix(
        sweep_id="snapshot-test",
        code_commit="a" * 40,
        stage_a_campaign_sha256="1" * 64,
        dataset_manifest_sha256="2" * 64,
        gate_manifest_sha256="3" * 64,
        accepted=accepted,
    )


def _signature_fixture(prefix: ProgressivePrefix) -> dict[str, TensorSignature]:
    tensors: dict[str, TensorSignature] = {
        "adaln_t_table": TensorSignature((ADALN_CURVE_GRID, ADALN_BASIS_DIM), "F32")
    }
    boundary = len(prefix.accepted)
    for index in range(CORE_BLOCKS):
        adaln = f"blocks.{index}.adaln_proj.linear."
        tensors[adaln + "weight"] = TensorSignature((ADALN_BLOCK_OUT, ADALN_BASIS_DIM), "BF16")
        tensors[adaln + "bias"] = TensorSignature((ADALN_BLOCK_OUT,), "F32")
        p = f"blocks.{index}.attn."
        if index < boundary:
            tensors[p + "qv_proj.weight"] = TensorSignature((2 * INNER_DIM, HIDDEN_SIZE), "BF16")
            tensors[p + "q_norm.weight"] = TensorSignature((HEAD_DIM,), "BF16")
            tensors[p + "route_norm.weight"] = TensorSignature((HEAD_DIM,), "BF16")
            tensors[p + "out_proj.weight"] = TensorSignature((HIDDEN_SIZE, INNER_DIM), "BF16")
        else:
            tensors[p + "qkv_proj.weight"] = TensorSignature((3 * INNER_DIM, HIDDEN_SIZE), "BF16")
            tensors[p + "q_norm.weight"] = TensorSignature((HEAD_DIM,), "BF16")
            tensors[p + "k_norm.weight"] = TensorSignature((HEAD_DIM,), "BF16")
            tensors[p + "out_proj.weight"] = TensorSignature((HIDDEN_SIZE, INNER_DIM), "BF16")
    for index in range(TOKEN_REFINER_BLOCKS):
        p = f"token_refiner.blocks.{index}.attn."
        tensors[p + "qkv_proj.weight"] = TensorSignature((3 * INNER_DIM, HIDDEN_SIZE), "BF16")
        tensors[p + "q_norm.weight"] = TensorSignature((HEAD_DIM,), "BF16")
        tensors[p + "k_norm.weight"] = TensorSignature((HEAD_DIM,), "BF16")
        tensors[p + "out_proj.weight"] = TensorSignature((HIDDEN_SIZE, INNER_DIM), "BF16")
    filler = 0
    while len(tensors) < TEACHER_TENSOR_COUNT:
        tensors[f"fixture.extra.{filler}"] = TensorSignature((1,), "BF16")
        filler += 1
    assert len(tensors) == TEACHER_TENSOR_COUNT
    return tensors


def test_progressive_snapshot_validator_accepts_exact_mixed_prefix() -> None:
    prefix = _prefix(2)
    snapshot.validate_progressive_snapshot_tensors(_signature_fixture(prefix), prefix)


def test_progressive_snapshot_validator_rejects_native_k_on_accepted_block() -> None:
    prefix = _prefix(2)
    tensors = _signature_fixture(prefix)
    tensors["blocks.1.attn.k_norm.weight"] = TensorSignature((HEAD_DIM,), "BF16")
    tensors.pop(next(key for key in tensors if key.startswith("fixture.extra.")))
    with pytest.raises(CheckpointValidationError, match="still contains native"):
        snapshot.validate_progressive_snapshot_tensors(tensors, prefix)


def test_progressive_snapshot_validator_rejects_qv_on_unaccepted_block() -> None:
    prefix = _prefix(1)
    tensors = _signature_fixture(prefix)
    tensors["blocks.3.attn.qv_proj.weight"] = TensorSignature(
        (2 * INNER_DIM, HIDDEN_SIZE), "BF16"
    )
    tensors.pop(next(key for key in tensors if key.startswith("fixture.extra.")))
    with pytest.raises(CheckpointValidationError, match="contains Keyless"):
        snapshot.validate_progressive_snapshot_tensors(tensors, prefix)


def test_progressive_snapshot_validator_rejects_training_factor() -> None:
    prefix = _prefix(1)
    tensors = _signature_fixture(prefix)
    tensors["blocks.0.attn.query_route.weight"] = TensorSignature((56, 128, 128), "BF16")
    tensors.pop(next(key for key in tensors if key.startswith("fixture.extra.")))
    with pytest.raises(CheckpointValidationError, match="training-only"):
        snapshot.validate_progressive_snapshot_tensors(tensors, prefix)


def test_snapshot_metadata_is_explicitly_noncanonical_and_prefix_bound() -> None:
    prefix = _prefix(3)
    md = snapshot.progressive_snapshot_metadata(
        prefix,
        prefix_manifest_sha256="4" * 64,
        manifest_sha256="5" * 64,
    )
    assert md["architecture"] == snapshot.PROGRESSIVE_SNAPSHOT_ARCHITECTURE
    assert md["canonical_release"] == "false"
    assert md["snapshot_kind"] == "stage_b_mixed_qv_qkv"
    assert md["accepted_blocks"] == "[0,1,2]"
    assert md["prefix_identity_sha256"] == prefix.identity_sha256
    assert md["manifest_sha256"] == "5" * 64


class TinyNativeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = 2
        self.head_dim = 2
        self.qkv_proj = nn.Linear(4, 12, bias=False, dtype=torch.bfloat16)
        self.q_norm = nn.RMSNorm(2, eps=1e-5, dtype=torch.bfloat16)
        self.k_norm = nn.RMSNorm(2, eps=1e-5, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.to_gate_compress = None


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = TinyNativeAttention()
        self.mlp = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)


class TinyProgressiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(TinyBlock() for _ in range(CORE_BLOCKS))


def _mixed_tiny_model(prefix: ProgressivePrefix) -> TinyProgressiveModel:
    model = TinyProgressiveModel()
    snapshot._replace_native_prefix_with_empty_deploy(model, prefix)
    torch.manual_seed(1201)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, dtype=torch.float32).to(parameter.dtype))
    return model


def test_export_receipt_is_immutable_hash_bound_and_noncanonical(tmp_path: Path, monkeypatch) -> None:
    prefix = _prefix(2)
    model = _mixed_tiny_model(prefix)
    monkeypatch.setattr(snapshot, "validate_progressive_snapshot_tensors", lambda tensors, p: None)
    output = tmp_path / "prefix-02.safetensors"
    result = snapshot.export_progressive_snapshot(
        model,
        prefix,
        output,
        prefix_manifest_sha256="4" * 64,
    )
    assert result.artifact_sha256 == sha256_file(output)
    assert result.accepted_blocks == 2
    receipt = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert receipt["schema"] == snapshot.PROGRESSIVE_SNAPSHOT_RECEIPT_SCHEMA
    assert receipt["artifact_sha256"] == result.artifact_sha256
    assert receipt["prefix"] == prefix.identity_payload()
    assert receipt["metadata"]["canonical_release"] == "false"
    assert receipt["manifest_sha256"] == result.manifest_identity_sha256
    with pytest.raises(FileExistsError, match="immutable"):
        snapshot.export_progressive_snapshot(
            model,
            prefix,
            output,
            prefix_manifest_sha256="4" * 64,
        )


def test_snapshot_loader_reconstructs_mixed_topology_and_exact_state(tmp_path: Path, monkeypatch) -> None:
    prefix = _prefix(2)
    source = _mixed_tiny_model(prefix)
    path = tmp_path / "tiny.safetensors"
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in source.state_dict().items()},
        str(path),
        metadata=snapshot.progressive_snapshot_metadata(
            prefix,
            prefix_manifest_sha256="4" * 64,
            manifest_sha256="5" * 64,
        ),
    )
    monkeypatch.setattr(snapshot, "validate_progressive_snapshot_file", lambda *args, **kwargs: ("x", {}))
    target = TinyProgressiveModel()
    snapshot.load_progressive_snapshot_into_native_model(
        target,
        path,
        prefix,
        prefix_manifest_sha256="4" * 64,
    )
    assert isinstance(target.blocks[0].attn, KeylessAttentionDeploy)
    assert isinstance(target.blocks[1].attn, KeylessAttentionDeploy)
    assert isinstance(target.blocks[2].attn, TinyNativeAttention)
    for key, expected in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], expected)
