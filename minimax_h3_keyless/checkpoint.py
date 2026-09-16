from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open

from .contracts import (
    ADALN_BASIS_DIM,
    ADALN_BLOCK_OUT,
    ADALN_CURVE_GRID,
    ARCHITECTURE,
    CHECKPOINT_FORMAT_VERSION,
    CORE_BLOCKS,
    HEAD_DIM,
    HEADS,
    HIDDEN_SIZE,
    INNER_DIM,
    QUANTIZATION_RECIPE,
    QV_ORDER,
    ROPE_POLICY,
    TARGET_MODEL_REVISION,
    TEACHER_COMPATIBILITY_MARKER,
    TEACHER_SHA256,
    TOKEN_REFINER_BLOCKS,
)


class CheckpointValidationError(ValueError):
    pass


@dataclass(frozen=True)
class TensorSignature:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ValidationReport:
    architecture: str
    format_version: int
    core_blocks: int
    refiner_qkv_blocks: int
    qv_shape: tuple[int, int]
    pruned_adaln: bool
    forbidden_key_count: int
    tensor_count: int


def _shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, TensorSignature):
        return value.shape
    if hasattr(value, "shape"):
        return tuple(int(x) for x in value.shape)
    if isinstance(value, (tuple, list)):
        return tuple(int(x) for x in value)
    raise TypeError(f"cannot determine shape for {type(value)!r}")


def _dtype_name(value: Any) -> str:
    if isinstance(value, TensorSignature):
        return value.dtype.upper()
    dtype = getattr(value, "dtype", None)
    mapping = {
        torch.bfloat16: "BF16",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.int8: "I8",
        torch.uint8: "U8",
    }
    if dtype in mapping:
        return mapping[dtype]
    if dtype is None:
        raise TypeError(f"cannot determine dtype for {type(value)!r}")
    return str(dtype).replace("torch.", "").upper()


def _require_shape(tensors: Mapping[str, Any], key: str, expected: tuple[int, ...]) -> None:
    if key not in tensors:
        raise CheckpointValidationError(f"missing required tensor: {key}")
    actual = _shape(tensors[key])
    if actual != expected:
        raise CheckpointValidationError(f"{key}: expected shape {expected}, got {actual}")


def _require_dtype(tensors: Mapping[str, Any], key: str, expected: str) -> None:
    if key not in tensors:
        raise CheckpointValidationError(f"missing required tensor: {key}")
    actual = _dtype_name(tensors[key])
    if actual != expected:
        raise CheckpointValidationError(f"{key}: expected dtype {expected}, got {actual}")


def _metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    try:
        return int(metadata[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointValidationError(f"missing/invalid metadata field {key!r}") from exc


def _metadata_sha256(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise CheckpointValidationError(f"missing/invalid SHA-256 metadata field {key!r}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CheckpointValidationError(f"missing/invalid SHA-256 metadata field {key!r}") from exc
    return value.lower()


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("architecture") != ARCHITECTURE:
        raise CheckpointValidationError(
            f"architecture metadata must be {ARCHITECTURE!r}, got {metadata.get('architecture')!r}"
        )
    if _metadata_int(metadata, "checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointValidationError("unsupported checkpoint_format_version")
    if metadata.get("qv_order") != QV_ORDER:
        raise CheckpointValidationError(f"qv_order must be {QV_ORDER!r}")
    if metadata.get("rope_policy") != ROPE_POLICY:
        raise CheckpointValidationError(f"rope_policy must be {ROPE_POLICY!r}")
    for key, expected in {
        "hidden_size": HIDDEN_SIZE,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "inner_dim": INNER_DIM,
        "core_blocks": CORE_BLOCKS,
        "token_refiner_blocks": TOKEN_REFINER_BLOCKS,
    }.items():
        if _metadata_int(metadata, key) != expected:
            raise CheckpointValidationError(f"metadata {key} must be {expected}")
    if metadata.get("token_refiner") != "native_qkv":
        raise CheckpointValidationError("token_refiner metadata must be 'native_qkv'")
    if metadata.get("routing_source") != "value":
        raise CheckpointValidationError("routing_source metadata must be 'value'")
    if metadata.get("retrieval_source") != "raw_projected_value":
        raise CheckpointValidationError("retrieval_source metadata must be 'raw_projected_value'")
    if metadata.get("teacher_model_revision") != TARGET_MODEL_REVISION:
        raise CheckpointValidationError("teacher_model_revision does not match the pinned H3 lineage")
    if metadata.get("teacher_model_sha256", "").lower() != TEACHER_SHA256:
        raise CheckpointValidationError("teacher_model_sha256 does not match the canonical BF16 teacher")
    if metadata.get("parent_model_revision") != TARGET_MODEL_REVISION:
        raise CheckpointValidationError("parent_model_revision does not match the pinned BF16 teacher")
    if metadata.get("parent_model_sha256", "").lower() != TEACHER_SHA256:
        raise CheckpointValidationError("parent_model_sha256 does not match the canonical BF16 teacher")
    for key in ("training_run", "export_commit"):
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            raise CheckpointValidationError(f"missing/invalid metadata field {key!r}")


def _validate_pruned_adaln(tensors: Mapping[str, Any]) -> None:
    _require_shape(tensors, "adaln_t_table", (ADALN_CURVE_GRID, ADALN_BASIS_DIM))
    for i in range(CORE_BLOCKS):
        p = f"blocks.{i}.adaln_proj.linear."
        _require_shape(tensors, p + "weight", (ADALN_BLOCK_OUT, ADALN_BASIS_DIM))
        _require_shape(tensors, p + "bias", (ADALN_BLOCK_OUT,))
    stale = [k for k in tensors if k.startswith("time_embedder.")]
    if stale:
        raise CheckpointValidationError(
            f"canonical pruned AdaLN checkpoint must not restore the native time embedder: {stale[:3]}"
        )


def validate_deploy_checkpoint(
    tensors: Mapping[str, Any], metadata: Mapping[str, Any], *, require_metadata: bool = True,
    require_pruned_adaln: bool = True,
) -> ValidationReport:
    if require_metadata:
        _validate_metadata(metadata)
    if require_pruned_adaln:
        _validate_pruned_adaln(tensors)

    forbidden: list[str] = []
    for i in range(CORE_BLOCKS):
        p = f"blocks.{i}.attn."
        _require_shape(tensors, p + "qv_proj.weight", (2 * INNER_DIM, HIDDEN_SIZE))
        _require_shape(tensors, p + "q_norm.weight", (HEAD_DIM,))
        _require_shape(tensors, p + "route_norm.weight", (HEAD_DIM,))
        _require_shape(tensors, p + "out_proj.weight", (HIDDEN_SIZE, INNER_DIM))
        for suffix in (
            "qkv_proj.weight", "k_norm.weight", "q_proj.weight", "v_proj.weight",
            "query_route.weight",
        ):
            if p + suffix in tensors:
                forbidden.append(p + suffix)
    if forbidden:
        sample = ", ".join(forbidden[:5])
        raise CheckpointValidationError(
            f"deploy checkpoint contains forbidden QKV/training tensors ({len(forbidden)}): {sample}"
        )

    for i in range(TOKEN_REFINER_BLOCKS):
        p = f"token_refiner.blocks.{i}.attn."
        _require_shape(tensors, p + "qkv_proj.weight", (3 * INNER_DIM, HIDDEN_SIZE))
        _require_shape(tensors, p + "q_norm.weight", (HEAD_DIM,))
        _require_shape(tensors, p + "k_norm.weight", (HEAD_DIM,))
        _require_shape(tensors, p + "out_proj.weight", (HIDDEN_SIZE, INNER_DIM))
        if p + "qv_proj.weight" in tensors or p + "route_norm.weight" in tensors:
            raise CheckpointValidationError("token-refiner attention must remain native QKV in v1")

    return ValidationReport(
        architecture=ARCHITECTURE,
        format_version=CHECKPOINT_FORMAT_VERSION,
        core_blocks=CORE_BLOCKS,
        refiner_qkv_blocks=TOKEN_REFINER_BLOCKS,
        qv_shape=(2 * INNER_DIM, HIDDEN_SIZE),
        pruned_adaln=require_pruned_adaln,
        forbidden_key_count=0,
        tensor_count=len(tensors),
    )


def _int8_target_shapes() -> dict[str, tuple[int, int]]:
    shapes: dict[str, tuple[int, int]] = {}
    for i in range(CORE_BLOCKS):
        shapes[f"blocks.{i}.attn.qv_proj.weight"] = (2 * INNER_DIM, HIDDEN_SIZE)
        shapes[f"blocks.{i}.attn.out_proj.weight"] = (HIDDEN_SIZE, INNER_DIM)
        shapes[f"blocks.{i}.mlp.fc1.weight"] = (4 * INNER_DIM, HIDDEN_SIZE)
        shapes[f"blocks.{i}.mlp.fc2.weight"] = (HIDDEN_SIZE, 2 * INNER_DIM)
    return shapes


def validate_int8_convrot_checkpoint(
    tensors: Mapping[str, Any], metadata: Mapping[str, Any]
) -> ValidationReport:
    """Validate the exact native Comfy core50/200 INT8 ConvRot storage contract."""
    report = validate_deploy_checkpoint(tensors, metadata)
    if metadata.get("teacher_compatibility") != TEACHER_COMPATIBILITY_MARKER:
        raise CheckpointValidationError(
            "INT8 ConvRot checkpoint is not derived from a pinned-teacher-compatible BF16 export"
        )
    _metadata_sha256(metadata, "manifest_sha256")
    _metadata_sha256(metadata, "quantization_source_bf16_sha256")
    _metadata_sha256(metadata, "quantization_source_bf16_manifest_sha256")
    if metadata.get("quantization_layer_recipe") != QUANTIZATION_RECIPE:
        raise CheckpointValidationError(
            f"quantization_layer_recipe must be {QUANTIZATION_RECIPE!r}"
        )
    if metadata.get("quantization_format") != "int8_tensorwise":
        raise CheckpointValidationError("quantization_format must be 'int8_tensorwise'")
    if metadata.get("quantization_per_channel") != "true":
        raise CheckpointValidationError("quantization_per_channel must be 'true'")
    if metadata.get("quantization_convrot") != "true":
        raise CheckpointValidationError("quantization_convrot must be 'true'")
    if _metadata_int(metadata, "quantization_convrot_groupsize") != 256:
        raise CheckpointValidationError("quantization_convrot_groupsize must be 256")
    if _metadata_int(metadata, "quantization_layer_count") != CORE_BLOCKS * 4:
        raise CheckpointValidationError("quantization_layer_count must be 200")

    target_shapes = _int8_target_shapes()
    expected_descriptors: set[str] = set()
    expected_scales: set[str] = set()
    for weight_key, shape in target_shapes.items():
        _require_shape(tensors, weight_key, shape)
        _require_dtype(tensors, weight_key, "I8")
        scale_key = weight_key + "_scale"
        descriptor_key = weight_key.removesuffix(".weight") + ".comfy_quant"
        expected_scales.add(scale_key)
        expected_descriptors.add(descriptor_key)
        if scale_key not in tensors:
            raise CheckpointValidationError(f"missing required tensor: {scale_key}")
        scale_shape = _shape(tensors[scale_key])
        if scale_shape != (shape[0], 1):
            raise CheckpointValidationError(
                f"{scale_key}: expected F32 per-output-row scale [{shape[0]},1], got {scale_shape}"
            )
        _require_dtype(tensors, scale_key, "F32")
        if descriptor_key not in tensors:
            raise CheckpointValidationError(f"missing required tensor: {descriptor_key}")
        descriptor_shape = _shape(tensors[descriptor_key])
        if descriptor_shape != (67,):
            raise CheckpointValidationError(
                f"{descriptor_key}: expected canonical 67-byte U8 ConvRot descriptor, got {descriptor_shape}"
            )
        _require_dtype(tensors, descriptor_key, "U8")

    actual_descriptors = {k for k in tensors if k.endswith(".comfy_quant")}
    actual_scales = {k for k in tensors if k.endswith(".weight_scale")}
    if actual_descriptors != expected_descriptors:
        missing = sorted(expected_descriptors - actual_descriptors)
        extra = sorted(actual_descriptors - expected_descriptors)
        raise CheckpointValidationError(
            "native quant descriptor set is not exactly the core50/200 recipe: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    if actual_scales != expected_scales:
        missing = sorted(expected_scales - actual_scales)
        extra = sorted(actual_scales - expected_scales)
        raise CheckpointValidationError(
            "INT8 scale set is not exactly the core50/200 recipe: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return report


def validate_training_checkpoint(tensors: Mapping[str, Any], *, require_pruned_adaln: bool = True) -> None:
    if require_pruned_adaln:
        _validate_pruned_adaln(tensors)
    for i in range(CORE_BLOCKS):
        p = f"blocks.{i}.attn."
        _require_shape(tensors, p + "q_proj.weight", (INNER_DIM, HIDDEN_SIZE))
        _require_shape(tensors, p + "query_route.weight", (HEADS, HEAD_DIM, HEAD_DIM))
        _require_shape(tensors, p + "v_proj.weight", (INNER_DIM, HIDDEN_SIZE))
        _require_shape(tensors, p + "q_norm.weight", (HEAD_DIM,))
        _require_shape(tensors, p + "route_norm.weight", (HEAD_DIM,))
        _require_shape(tensors, p + "out_proj.weight", (HIDDEN_SIZE, INNER_DIM))
        if p + "qkv_proj.weight" in tensors or p + "qv_proj.weight" in tensors:
            raise CheckpointValidationError(
                f"training checkpoint has mixed deploy/native projection at block {i}"
            )


def read_safetensors_signatures(path: str | Path) -> tuple[dict[str, TensorSignature], dict[str, str]]:
    tensors: dict[str, TensorSignature] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = dict(f.metadata() or {})
        for key in f.keys():
            sl = f.get_slice(key)
            tensors[key] = TensorSignature(tuple(sl.get_shape()), str(sl.get_dtype()))
    return tensors, metadata


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def deterministic_manifest(
    *, artifact_path: str, artifact_sha256: str, parent_sha256: str,
    producing_commit: str, command: str, status: str, extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
        "parent_checkpoint_sha256": parent_sha256,
        "producing_commit": producing_commit,
        "command": command,
        "status": status,
    }
    if extra:
        data["extra"] = dict(extra)
    return data
