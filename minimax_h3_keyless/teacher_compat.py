from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open

from .checkpoint import TensorSignature, read_safetensors_signatures, sha256_file
from .contracts import (
    CORE_BLOCKS,
    HEAD_DIM,
    HIDDEN_SIZE,
    INNER_DIM,
    TARGET_MODEL_REVISION,
    TEACHER_SHA256,
)


TEACHER_COMPATIBILITY_MARKER = "pinned_exact_copy_v1"


class TeacherCompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class TeacherCompatibilityReport:
    teacher_sha256: str
    teacher_tensor_count: int
    deploy_tensor_count: int
    exact_copy_tensor_count: int
    mutable_core_tensor_count: int


def _signature(value: Any) -> TensorSignature:
    if isinstance(value, TensorSignature):
        return value
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"cannot derive tensor signature from {type(value)!r}")
    dtype = {
        torch.bfloat16: "BF16",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.int8: "I8",
        torch.uint8: "U8",
    }.get(value.dtype, str(value.dtype).replace("torch.", "").upper())
    return TensorSignature(tuple(int(x) for x in value.shape), dtype)


def signatures_from_mapping(tensors: Mapping[str, Any]) -> dict[str, TensorSignature]:
    return {key: _signature(value) for key, value in tensors.items()}


def _teacher_removed_keys(core_blocks: int = CORE_BLOCKS) -> set[str]:
    removed: set[str] = set()
    for i in range(core_blocks):
        p = f"blocks.{i}.attn."
        removed.add(p + "qkv_proj.weight")
        removed.add(p + "k_norm.weight")
    return removed


def _deploy_added_keys(core_blocks: int = CORE_BLOCKS) -> set[str]:
    added: set[str] = set()
    for i in range(core_blocks):
        p = f"blocks.{i}.attn."
        added.add(p + "qv_proj.weight")
        added.add(p + "route_norm.weight")
    return added


def mutable_core_shared_keys(core_blocks: int = CORE_BLOCKS) -> set[str]:
    """Teacher-named tensors that Stage A may eventually unfreeze and therefore need not be byte-identical."""
    keys: set[str] = set()
    for i in range(core_blocks):
        p = f"blocks.{i}.attn."
        keys.add(p + "q_norm.weight")
        keys.add(p + "out_proj.weight")
    return keys


def exact_copy_shared_keys(
    teacher_keys: set[str], *, core_blocks: int = CORE_BLOCKS
) -> set[str]:
    """Shared deploy keys required to remain exactly equal to the pinned BF16 teacher."""
    removed = _teacher_removed_keys(core_blocks)
    mutable = mutable_core_shared_keys(core_blocks)
    return set(teacher_keys).difference(removed).difference(mutable)


def validate_deploy_signatures_against_teacher(
    teacher: Mapping[str, Any],
    deploy: Mapping[str, Any],
    *,
    core_blocks: int = CORE_BLOCKS,
    hidden_size: int = HIDDEN_SIZE,
    inner_dim: int = INNER_DIM,
    head_dim: int = HEAD_DIM,
) -> tuple[set[str], set[str]]:
    """Validate the complete tensor-key/signature transformation from native BF16 teacher to Keyless deploy."""
    teacher_sig = signatures_from_mapping(teacher)
    deploy_sig = signatures_from_mapping(deploy)
    removed = _teacher_removed_keys(core_blocks)
    added = _deploy_added_keys(core_blocks)

    missing_teacher = sorted(removed.difference(teacher_sig))
    if missing_teacher:
        raise TeacherCompatibilityError(
            f"native teacher is missing required QKV/K tensors: {missing_teacher[:5]}"
        )
    expected_keys = set(teacher_sig).difference(removed).union(added)
    if set(deploy_sig) != expected_keys:
        missing = sorted(expected_keys.difference(deploy_sig))
        extra = sorted(set(deploy_sig).difference(expected_keys))
        raise TeacherCompatibilityError(
            "deploy tensor key set is not the exact core50 native->Keyless transformation: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    for key in set(teacher_sig).difference(removed):
        if deploy_sig[key] != teacher_sig[key]:
            raise TeacherCompatibilityError(
                f"shared tensor signature changed for {key}: {teacher_sig[key]} -> {deploy_sig[key]}"
            )

    for i in range(core_blocks):
        p = f"blocks.{i}.attn."
        qkv = teacher_sig[p + "qkv_proj.weight"]
        k_norm = teacher_sig[p + "k_norm.weight"]
        if qkv.shape != (3 * inner_dim, hidden_size) or qkv.dtype != "BF16":
            raise TeacherCompatibilityError(
                f"teacher {p}qkv_proj.weight is not canonical BF16 H3 geometry: {qkv}"
            )
        if k_norm.shape != (head_dim,) or k_norm.dtype != "BF16":
            raise TeacherCompatibilityError(
                f"teacher {p}k_norm.weight is not canonical BF16 H3 geometry: {k_norm}"
            )
        qv = deploy_sig[p + "qv_proj.weight"]
        route = deploy_sig[p + "route_norm.weight"]
        if qv != TensorSignature((2 * inner_dim, hidden_size), "BF16"):
            raise TeacherCompatibilityError(
                f"deploy {p}qv_proj.weight is not canonical BF16 Keyless geometry: {qv}"
            )
        if route != k_norm:
            raise TeacherCompatibilityError(
                f"deploy {p}route_norm.weight signature must replace teacher k_norm exactly"
            )

    exact = exact_copy_shared_keys(set(teacher_sig), core_blocks=core_blocks)
    mutable = mutable_core_shared_keys(core_blocks)
    return exact, mutable


def validate_deploy_mapping_against_teacher(
    teacher_path: str | Path,
    deploy: Mapping[str, torch.Tensor],
    *,
    verify_teacher_sha256: bool = True,
    expected_teacher_sha256: str = TEACHER_SHA256,
) -> TeacherCompatibilityReport:
    """Canonical export gate: full key/signature relation plus exact copied tensor values."""
    teacher_path = Path(teacher_path)
    actual_sha = sha256_file(teacher_path) if verify_teacher_sha256 else ""
    if verify_teacher_sha256 and actual_sha.lower() != expected_teacher_sha256.lower():
        raise TeacherCompatibilityError(
            f"teacher SHA-256 mismatch: expected {expected_teacher_sha256}, got {actual_sha}"
        )
    teacher_sig, _ = read_safetensors_signatures(teacher_path)
    exact, mutable = validate_deploy_signatures_against_teacher(teacher_sig, deploy)

    with safe_open(str(teacher_path), framework="pt", device="cpu") as teacher_file:
        for key in sorted(exact):
            teacher_tensor = teacher_file.get_tensor(key)
            deploy_tensor = deploy[key].detach().cpu()
            if teacher_tensor.dtype != deploy_tensor.dtype or teacher_tensor.shape != deploy_tensor.shape:
                raise TeacherCompatibilityError(f"exact-copy tensor signature changed for {key}")
            if not torch.equal(teacher_tensor, deploy_tensor):
                raise TeacherCompatibilityError(
                    f"copied tensor differs from pinned BF16 teacher: {key}"
                )
            del teacher_tensor, deploy_tensor

    return TeacherCompatibilityReport(
        teacher_sha256=actual_sha or expected_teacher_sha256,
        teacher_tensor_count=len(teacher_sig),
        deploy_tensor_count=len(deploy),
        exact_copy_tensor_count=len(exact),
        mutable_core_tensor_count=len(mutable),
    )


def validate_deploy_artifact_against_teacher(
    teacher_path: str | Path,
    deploy_path: str | Path,
    *,
    verify_teacher_sha256: bool = True,
    expected_teacher_sha256: str = TEACHER_SHA256,
) -> TeacherCompatibilityReport:
    """Post-write form of the canonical full-compatibility gate for existing artifacts."""
    deploy_path = Path(deploy_path)
    deploy_sig, metadata = read_safetensors_signatures(deploy_path)
    teacher_path = Path(teacher_path)
    actual_sha = sha256_file(teacher_path) if verify_teacher_sha256 else ""
    if verify_teacher_sha256 and actual_sha.lower() != expected_teacher_sha256.lower():
        raise TeacherCompatibilityError(
            f"teacher SHA-256 mismatch: expected {expected_teacher_sha256}, got {actual_sha}"
        )
    expected_parent = actual_sha or expected_teacher_sha256
    if metadata.get("parent_model_sha256", "").lower() != expected_parent.lower():
        raise TeacherCompatibilityError(
            "deploy parent_model_sha256 does not identify the teacher used for compatibility validation"
        )
    if metadata.get("parent_model_revision") != TARGET_MODEL_REVISION:
        raise TeacherCompatibilityError(
            "deploy parent_model_revision does not identify the pinned teacher revision"
        )
    if metadata.get("teacher_compatibility") != TEACHER_COMPATIBILITY_MARKER:
        raise TeacherCompatibilityError(
            "deploy artifact is missing the canonical pinned-teacher compatibility marker"
        )

    teacher_sig, _ = read_safetensors_signatures(teacher_path)
    exact, mutable = validate_deploy_signatures_against_teacher(teacher_sig, deploy_sig)
    with safe_open(str(teacher_path), framework="pt", device="cpu") as teacher_file, safe_open(
        str(deploy_path), framework="pt", device="cpu"
    ) as deploy_file:
        for key in sorted(exact):
            teacher_tensor = teacher_file.get_tensor(key)
            deploy_tensor = deploy_file.get_tensor(key)
            if not torch.equal(teacher_tensor, deploy_tensor):
                raise TeacherCompatibilityError(
                    f"copied tensor differs from pinned BF16 teacher: {key}"
                )
            del teacher_tensor, deploy_tensor

    return TeacherCompatibilityReport(
        teacher_sha256=expected_parent,
        teacher_tensor_count=len(teacher_sig),
        deploy_tensor_count=len(deploy_sig),
        exact_copy_tensor_count=len(exact),
        mutable_core_tensor_count=len(mutable),
    )
