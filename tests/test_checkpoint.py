from __future__ import annotations

from minimax_h3_keyless.checkpoint import (
    CheckpointValidationError,
    TensorSignature,
    validate_deploy_checkpoint,
    validate_int8_convrot_checkpoint,
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
    QUANTIZATION_RECIPE,
    TEACHER_COMPATIBILITY_MARKER,
    TOKEN_REFINER_BLOCKS,
)
from minimax_h3_keyless.export import canonical_metadata


def _sig(*shape: int, dtype: str = "BF16") -> TensorSignature:
    return TensorSignature(tuple(shape), dtype)


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


def _int8_deploy() -> tuple[dict[str, TensorSignature], dict[str, str]]:
    tensors = _deploy()
    for i in range(CORE_BLOCKS):
        shapes = {
            f"blocks.{i}.attn.qv_proj.weight": (2 * INNER_DIM, HIDDEN_SIZE),
            f"blocks.{i}.attn.out_proj.weight": (HIDDEN_SIZE, INNER_DIM),
            f"blocks.{i}.mlp.fc1.weight": (4 * INNER_DIM, HIDDEN_SIZE),
            f"blocks.{i}.mlp.fc2.weight": (HIDDEN_SIZE, 2 * INNER_DIM),
        }
        for weight_key, shape in shapes.items():
            tensors[weight_key] = _sig(*shape, dtype="I8")
            tensors[weight_key + "_scale"] = _sig(shape[0], 1, dtype="F32")
            descriptor_key = weight_key.removesuffix(".weight") + ".comfy_quant"
            tensors[descriptor_key] = _sig(67, dtype="U8")
    metadata = canonical_metadata(training_run="test", export_commit="deadbeef")
    metadata.update(
        {
            "teacher_compatibility": TEACHER_COMPATIBILITY_MARKER,
            "manifest_sha256": "d" * 64,
            "quantization_source_bf16_sha256": "a" * 64,
            "quantization_source_bf16_manifest_sha256": "b" * 64,
            "quantization_format": "int8_tensorwise",
            "quantization_layer_recipe": QUANTIZATION_RECIPE,
            "quantization_layer_count": "200",
            "quantization_per_channel": "true",
            "quantization_convrot": "true",
            "quantization_convrot_groupsize": "256",
        }
    )
    return tensors, metadata


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


def test_deploy_rejects_wrong_parent_lineage() -> None:
    metadata = canonical_metadata(training_run="test", export_commit="deadbeef")
    metadata["parent_model_sha256"] = "0" * 64
    try:
        validate_deploy_checkpoint(_deploy(), metadata)
    except CheckpointValidationError as exc:
        assert "parent_model_sha256" in str(exc)
    else:
        raise AssertionError("wrong parent model lineage must be rejected")


def test_valid_int8_core50_200_signature_is_accepted() -> None:
    tensors, metadata = _int8_deploy()
    report = validate_int8_convrot_checkpoint(tensors, metadata)
    assert report.core_blocks == 50
    assert len([k for k in tensors if k.endswith(".comfy_quant")]) == 200
    assert len([k for k in tensors if k.endswith(".weight_scale")]) == 200


def test_int8_rejects_missing_bf16_manifest_identity() -> None:
    tensors, metadata = _int8_deploy()
    metadata.pop("quantization_source_bf16_manifest_sha256")
    try:
        validate_int8_convrot_checkpoint(tensors, metadata)
    except CheckpointValidationError as exc:
        assert "quantization_source_bf16_manifest_sha256" in str(exc)
    else:
        raise AssertionError("INT8 artifact without source BF16 manifest identity must be rejected")


def test_int8_rejects_wrong_per_output_row_scale_shape() -> None:
    tensors, metadata = _int8_deploy()
    tensors["blocks.0.attn.qv_proj.weight_scale"] = _sig(2 * INNER_DIM, 2, dtype="F32")
    try:
        validate_int8_convrot_checkpoint(tensors, metadata)
    except CheckpointValidationError as exc:
        assert "per-output-row scale" in str(exc)
    else:
        raise AssertionError("invalid ConvRot scale geometry must be rejected")


def test_int8_rejects_stray_quant_descriptor_outside_200_recipe() -> None:
    tensors, metadata = _int8_deploy()
    tensors["token_refiner.blocks.0.attn.qkv_proj.comfy_quant"] = _sig(67, dtype="U8")
    try:
        validate_int8_convrot_checkpoint(tensors, metadata)
    except CheckpointValidationError as exc:
        assert "exactly the core50/200 recipe" in str(exc)
    else:
        raise AssertionError("stray token-refiner quantization must be rejected")


def test_training_signature_rejects_deploy_projection() -> None:
    tensors = _training()
    tensors["blocks.0.attn.qv_proj.weight"] = _sig(2 * INNER_DIM, HIDDEN_SIZE)
    try:
        validate_training_checkpoint(tensors)
    except CheckpointValidationError as exc:
        assert "mixed deploy/native projection" in str(exc)
    else:
        raise AssertionError("training checkpoint must not contain qv_proj")
