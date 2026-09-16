from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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
    QV_ORDER,
    ROPE_POLICY,
    TARGET_MODEL_REVISION,
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
    if hasattr(value, "shape"):
        return tuple(int(x) for x in value.shape)
    if isinstance(value, TensorSignature):
        return value.shape
    if isinstance(value, (tuple, list)):
        return tuple(int(x) for x in value)
    raise TypeError(f"cannot determine shape for {type(value)!r}")


def _require_shape(tensors: Mapping[str, Any], key: str, expected: tuple[int, ...]) -> None:
    if key not in tensors:
        raise CheckpointValidationError(f"missing required tensor: {key}")
    actual = _shape(tensors[key])
    if actual != expected:
        raise CheckpointValidationError(f"{key}: expected shape {expected}, got {actual}")


def _metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    try:
        return int(metadata[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointValidationError(f"missing/invalid metadata field {key!r}") from exc


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
