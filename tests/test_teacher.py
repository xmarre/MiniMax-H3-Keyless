from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from minimax_h3_keyless.checkpoint import TensorSignature
from minimax_h3_keyless.contracts import (
    ADALN_BASIS_DIM,
    ADALN_BLOCK_OUT,
    ADALN_CURVE_GRID,
    CORE_BLOCKS,
    HEAD_DIM,
    HIDDEN_SIZE,
    INNER_DIM,
    NORM_EPS,
    TOKEN_REFINER_BLOCKS,
)
from minimax_h3_keyless.teacher import (
    TeacherValidationError,
    validate_loaded_native_teacher_model,
    validate_native_bf16_teacher_signatures,
)


def _sig(shape, dtype="BF16"):
    return TensorSignature(tuple(shape), dtype)


def _native_signatures():
    signatures = {"adaln_t_table": _sig((ADALN_CURVE_GRID, ADALN_BASIS_DIM), "F32")}
    for index in range(CORE_BLOCKS):
        prefix = f"blocks.{index}.attn."
        signatures[prefix + "qkv_proj.weight"] = _sig((3 * INNER_DIM, HIDDEN_SIZE))
        signatures[prefix + "q_norm.weight"] = _sig((HEAD_DIM,))
        signatures[prefix + "k_norm.weight"] = _sig((HEAD_DIM,))
        signatures[prefix + "out_proj.weight"] = _sig((HIDDEN_SIZE, INNER_DIM))
        adaln = f"blocks.{index}.adaln_proj.linear."
        signatures[adaln + "weight"] = _sig((ADALN_BLOCK_OUT, ADALN_BASIS_DIM))
        signatures[adaln + "bias"] = _sig((ADALN_BLOCK_OUT,), "F32")
    for index in range(TOKEN_REFINER_BLOCKS):
        prefix = f"token_refiner.blocks.{index}.attn."
        signatures[prefix + "qkv_proj.weight"] = _sig((3 * INNER_DIM, HIDDEN_SIZE))
        signatures[prefix + "q_norm.weight"] = _sig((HEAD_DIM,))
        signatures[prefix + "k_norm.weight"] = _sig((HEAD_DIM,))
        signatures[prefix + "out_proj.weight"] = _sig((HIDDEN_SIZE, INNER_DIM))
    return signatures


def test_native_teacher_signature_gate_accepts_structural_fixture() -> None:
    report = validate_native_bf16_teacher_signatures(
        _native_signatures(),
        require_exact_tensor_count=False,
    )
    assert report.core_qkv_blocks == 50
    assert report.refiner_qkv_blocks == 2


def test_native_teacher_signature_gate_rejects_keyless_or_quantized_core() -> None:
    keyless = _native_signatures()
    keyless["blocks.0.attn.qv_proj.weight"] = _sig((2 * INNER_DIM, HIDDEN_SIZE))
    with pytest.raises(TeacherValidationError, match="Keyless core tensors"):
        validate_native_bf16_teacher_signatures(keyless, require_exact_tensor_count=False)

    quantized = _native_signatures()
    quantized["blocks.0.attn.qkv_proj.weight_scale"] = _sig((3 * INNER_DIM,), "F32")
    with pytest.raises(TeacherValidationError, match="quantized artifact"):
        validate_native_bf16_teacher_signatures(quantized, require_exact_tensor_count=False)


def _fake_weight(shape, dtype=torch.bfloat16, *, is_meta=False):
    return SimpleNamespace(shape=shape, dtype=dtype, is_meta=is_meta)


def _fake_attention(*, keyless=False, dtype=torch.bfloat16, is_meta=False):
    attention = SimpleNamespace(
        qkv_proj=SimpleNamespace(
            weight=_fake_weight((3 * INNER_DIM, HIDDEN_SIZE), dtype, is_meta=is_meta)
        ),
        q_norm=SimpleNamespace(
            weight=_fake_weight((HEAD_DIM,), dtype),
            eps=NORM_EPS,
        ),
        k_norm=SimpleNamespace(
            weight=_fake_weight((HEAD_DIM,), dtype),
            eps=NORM_EPS,
        ),
        out_proj=SimpleNamespace(
            weight=_fake_weight((HIDDEN_SIZE, INNER_DIM), dtype, is_meta=is_meta)
        ),
    )
    if keyless:
        attention.qv_proj = SimpleNamespace(weight=None)
    return attention


def _fake_model(*, bad_block=None, keyless=False, dtype=torch.bfloat16, is_meta=False):
    blocks = []
    for index in range(CORE_BLOCKS):
        kwargs = {}
        if bad_block == index:
            kwargs = {"keyless": keyless, "dtype": dtype, "is_meta": is_meta}
        blocks.append(SimpleNamespace(attn=_fake_attention(**kwargs)))
    refiners = [SimpleNamespace(attn=_fake_attention()) for _ in range(TOKEN_REFINER_BLOCKS)]
    return SimpleNamespace(
        blocks=blocks,
        token_refiner=SimpleNamespace(blocks=refiners),
    )


def test_loaded_teacher_runtime_gate_rejects_keyless_wrong_dtype_and_meta_weights() -> None:
    validate_loaded_native_teacher_model(_fake_model())
    with pytest.raises(TeacherValidationError, match="already Keyless"):
        validate_loaded_native_teacher_model(_fake_model(bad_block=12, keyless=True))
    with pytest.raises(TeacherValidationError, match="BF16"):
        validate_loaded_native_teacher_model(_fake_model(bad_block=12, dtype=torch.float16))
    with pytest.raises(TeacherValidationError, match="meta"):
        validate_loaded_native_teacher_model(_fake_model(bad_block=12, is_meta=True))
