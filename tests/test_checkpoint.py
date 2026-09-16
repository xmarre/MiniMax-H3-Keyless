from __future__ import annotations

from copy import deepcopy

from minimax_h3_keyless.checkpoint import (
    CheckpointValidationError,
    TensorSignature,
    validate_deploy_checkpoint,
    validate_training_checkpoint,
)
from minimax_h3_keyless.contracts import (
    ADALN_BASIS_DIM,
    ADALN_BLOCK_OUT,
    ADALN_CURVE_GRID,
    CORE_BLOCKS,
    HEAD_DIM,
    HEADS,
    HIDDEN_SIZE,
    INNER_DIM,
    TOKEN_REFINER_BLOCKS,
)
from minimax_h3_keyless.export import canonical_metadata


def _sig(*shape: int) -> TensorSignature:
    return TensorSignature(tuple(shape), "BF16")


def _common() -> dict[str, TensorSignature]:
    tensors = {"adaln_t_table": _sig(ADALN_CURVE_GRID, ADALN_BASIS_DIM)}
    for i in range(CORE_BLOCKS):
        p = f"blocks.{i}.adaln_proj.linear."
        tensors[p + "weight"] = _sig(ADALN_BLOCK_OUT, ADALN_BASIS_DIM)
        tensors[p + "bias"] = _sig(ADALN_BLOCK_OUT)
    for i in range(TOKEN_REFINER_BLOCKS):
        p = f"token_refiner.blocks.{i}.attn."
        tensors[p + "qkv_proj.weight"] = _sig(3 * INNER_DIM, HIDDEN_SIZE)
        tensors[p + "q_norm.weight"] = _sig(HEAD_DIM)
        tensors[p + "k_norm.weight"] = _sig(HEAD_DIM)
        tensors[p + "out_proj.weight"] = _sig(HIDDEN_SIZE, INNER_DIM)
    return tensors


def _deploy() -> dict[str, TensorSignature]:
    tensors = _common()
    for i in range(CORE_BLOCKS):
        p = f"blocks.{i}.attn."
        tensors[p + "qv_proj.weight"] = _sig(2 * INNER_DIM, HIDDEN_SIZE)
        tensors[p + "q_norm.weight"] = _sig(HEAD_DIM)
        tensors[p + "route_norm.weight"] = _sig(HEAD_DIM)
        tensors[p + "out_proj.weight"] = _sig(HIDDEN_SIZE, INNER_DIM)
    return tensors


def _training() -> dict[str, TensorSignature]:
    tensors = _common()
    for i in range(CORE_BLOCKS):
        p = f"blocks.{i}.attn."
        tensors[p + "q_proj.weight"] = _sig(INNER_DIM, HIDDEN_SIZE)
        tensors[p + "query_route.weight"] = _sig(HEADS, HEAD_DIM, HEAD_DIM)
        tensors[p + "v_proj.weight"] = _sig(INNER_DIM, HIDDEN_SIZE)
        tensors[p + "q_norm.weight"] = _sig(HEAD_DIM)
        tensors[p + "route_norm.weight"] = _sig(HEAD_DIM)
        tensors[p + "out_proj.weight"] = _sig(HIDDEN_SIZE, INNER_DIM)
    return tensors


def test_valid_deploy_signature_is_accepted() -> None:
    report = validate_deploy_checkpoint(
        _deploy(), canonical_metadata(training_run="test", export_commit="deadbeef")
    )
    assert report.core_blocks == 50
    assert report.refiner_qkv_blocks == 2
    assert report.qv_shape == (2 * INNER_DIM, HIDDEN_SIZE)


def test_deploy_rejects_core_qkv_mixture() -> None:
    tensors = _deploy()
    tensors["blocks.12.attn.qkv_proj.weight"] = _sig(3 * INNER_DIM, HIDDEN_SIZE)
    try:
        validate_deploy_checkpoint(
            tensors, canonical_metadata(training_run="test", export_commit="deadbeef")
        )
    except CheckpointValidationError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("mixed QKV/QV deploy checkpoint must be rejected")


def test_deploy_rejects_converted_refiner() -> None:
    tensors = _deploy()
    tensors["token_refiner.blocks.0.attn.qv_proj.weight"] = _sig(2 * INNER_DIM, HIDDEN_SIZE)
    try:
        validate_deploy_checkpoint(
            tensors, canonical_metadata(training_run="test", export_commit="deadbeef")
        )
    except CheckpointValidationError as exc:
        assert "native QKV" in str(exc)
    else:
        raise AssertionError("v1 token refiner must remain native QKV")


def test_deploy_rejects_wrong_metadata_identity() -> None:
    metadata = canonical_metadata(training_run="test", export_commit="deadbeef")
    metadata["qv_order"] = "v;q_effective"
    try:
        validate_deploy_checkpoint(_deploy(), metadata)
    except CheckpointValidationError as exc:
        assert "qv_order" in str(exc)
    else:
        raise AssertionError("wrong QV ordering metadata must be rejected")


def test_training_signature_rejects_deploy_projection() -> None:
    tensors = _training()
    tensors["blocks.0.attn.qv_proj.weight"] = _sig(2 * INNER_DIM, HIDDEN_SIZE)
    try:
        validate_training_checkpoint(tensors)
    except CheckpointValidationError as exc:
        assert "mixed deploy/native projection" in str(exc)
    else:
        raise AssertionError("training checkpoint must not contain qv_proj")
