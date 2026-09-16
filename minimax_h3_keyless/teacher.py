from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .checkpoint import TensorSignature, read_safetensors_signatures, sha256_file
from .contracts import (
    ADALN_BASIS_DIM,
    ADALN_BLOCK_OUT,
    ADALN_CURVE_GRID,
    CORE_BLOCKS,
    HEAD_DIM,
    HEADS,
    HIDDEN_SIZE,
    INNER_DIM,
    NORM_EPS,
    TEACHER_SHA256,
    TOKEN_REFINER_BLOCKS,
)


TEACHER_TENSOR_COUNT = 534


class TeacherValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeTeacherSignatureReport:
    tensor_count: int
    core_qkv_blocks: int
    refiner_qkv_blocks: int
    teacher_sha256: str | None = None


@dataclass(frozen=True)
class LoadedNativeTeacher:
    patcher: Any
    diffusion_model: Any
    signature_report: NativeTeacherSignatureReport

    @property
    def pilot_blocks(self) -> Mapping[int, Any]:
        return {index: self.diffusion_model.blocks[index] for index in (0, 25, 49)}


def _require_signature(
    signatures: Mapping[str, TensorSignature],
    key: str,
    shape: tuple[int, ...],
    dtype: str,
) -> None:
    value = signatures.get(key)
    expected = TensorSignature(shape, dtype)
    if value != expected:
        raise TeacherValidationError(f"teacher {key} has {value}, expected {expected}")


def validate_native_bf16_teacher_signatures(
    signatures: Mapping[str, TensorSignature],
    *,
    require_exact_tensor_count: bool = True,
) -> NativeTeacherSignatureReport:
    """Validate the fixed native-QKV BF16 teacher topology before Comfy model loading."""
    if require_exact_tensor_count and len(signatures) != TEACHER_TENSOR_COUNT:
        raise TeacherValidationError(
            f"pinned BF16 teacher must contain {TEACHER_TENSOR_COUNT} tensors, got {len(signatures)}"
        )
    forbidden_suffixes = (".qv_proj.weight", ".query_route.weight", ".route_norm.weight")
    forbidden = sorted(
        key
        for key in signatures
        if key.startswith("blocks.") and key.endswith(forbidden_suffixes)
    )
    if forbidden:
        raise TeacherValidationError(
            f"native BF16 teacher contains Keyless core tensors: {forbidden[:5]}"
        )
    quantized = sorted(
        key
        for key in signatures
        if key.endswith(".comfy_quant") or key.endswith(".weight_scale")
    )
    if quantized:
        raise TeacherValidationError(
            f"Stage-A teacher must be BF16, not a quantized artifact: {quantized[:5]}"
        )

    for index in range(CORE_BLOCKS):
        prefix = f"blocks.{index}.attn."
        _require_signature(
            signatures,
            prefix + "qkv_proj.weight",
            (3 * INNER_DIM, HIDDEN_SIZE),
            "BF16",
        )
        _require_signature(signatures, prefix + "q_norm.weight", (HEAD_DIM,), "BF16")
        _require_signature(signatures, prefix + "k_norm.weight", (HEAD_DIM,), "BF16")
        _require_signature(
            signatures,
            prefix + "out_proj.weight",
            (HIDDEN_SIZE, INNER_DIM),
            "BF16",
        )
        adaln = f"blocks.{index}.adaln_proj.linear."
        _require_signature(
            signatures,
            adaln + "weight",
            (ADALN_BLOCK_OUT, ADALN_BASIS_DIM),
            "BF16",
        )
        _require_signature(signatures, adaln + "bias", (ADALN_BLOCK_OUT,), "F32")

    for index in range(TOKEN_REFINER_BLOCKS):
        prefix = f"token_refiner.blocks.{index}.attn."
        _require_signature(
            signatures,
            prefix + "qkv_proj.weight",
            (3 * INNER_DIM, HIDDEN_SIZE),
            "BF16",
        )
        _require_signature(signatures, prefix + "q_norm.weight", (HEAD_DIM,), "BF16")
        _require_signature(signatures, prefix + "k_norm.weight", (HEAD_DIM,), "BF16")
        _require_signature(
            signatures,
            prefix + "out_proj.weight",
            (HIDDEN_SIZE, INNER_DIM),
            "BF16",
        )

    _require_signature(
        signatures,
        "adaln_t_table",
        (ADALN_CURVE_GRID, ADALN_BASIS_DIM),
        "F32",
    )
    return NativeTeacherSignatureReport(
        tensor_count=len(signatures),
        core_qkv_blocks=CORE_BLOCKS,
        refiner_qkv_blocks=TOKEN_REFINER_BLOCKS,
    )


def _shape_dtype(value: Any) -> tuple[tuple[int, ...], torch.dtype | None]:
    shape = tuple(int(x) for x in getattr(value, "shape", ()))
    return shape, getattr(value, "dtype", None)


def validate_loaded_native_teacher_model(diffusion_model: Any) -> None:
    """Fail closed if the live object is no longer the plain native BF16 H3 teacher."""
    blocks = getattr(diffusion_model, "blocks", None)
    token_refiner = getattr(diffusion_model, "token_refiner", None)
    refiner_blocks = getattr(token_refiner, "blocks", None)
    if blocks is None or len(blocks) != CORE_BLOCKS:
        raise TeacherValidationError(f"loaded teacher must expose {CORE_BLOCKS} core blocks")
    if refiner_blocks is None or len(refiner_blocks) != TOKEN_REFINER_BLOCKS:
        raise TeacherValidationError(
            f"loaded teacher must expose {TOKEN_REFINER_BLOCKS} token-refiner blocks"
        )

    def check_attention(attention: Any, label: str) -> None:
        if hasattr(attention, "qv_proj") or hasattr(attention, "query_route"):
            raise TeacherValidationError(f"{label} is already Keyless; native QKV teacher required")
        qkv = getattr(getattr(attention, "qkv_proj", None), "weight", None)
        q_norm = getattr(attention, "q_norm", None)
        k_norm = getattr(attention, "k_norm", None)
        out = getattr(getattr(attention, "out_proj", None), "weight", None)
        if qkv is None or q_norm is None or k_norm is None or out is None:
            raise TeacherValidationError(f"{label} is missing native H3 attention components")
        if _shape_dtype(qkv) != ((3 * INNER_DIM, HIDDEN_SIZE), torch.bfloat16):
            raise TeacherValidationError(f"{label} qkv projection is not canonical BF16 H3")
        if _shape_dtype(q_norm.weight) != ((HEAD_DIM,), torch.bfloat16):
            raise TeacherValidationError(f"{label} q_norm is not canonical BF16 H3")
        if _shape_dtype(k_norm.weight) != ((HEAD_DIM,), torch.bfloat16):
            raise TeacherValidationError(f"{label} k_norm is not canonical BF16 H3")
        if _shape_dtype(out) != ((HIDDEN_SIZE, INNER_DIM), torch.bfloat16):
            raise TeacherValidationError(f"{label} out projection is not canonical BF16 H3")
        if float(getattr(q_norm, "eps", float("nan"))) != NORM_EPS:
            raise TeacherValidationError(f"{label} q_norm epsilon is not {NORM_EPS}")
        if float(getattr(k_norm, "eps", float("nan"))) != NORM_EPS:
            raise TeacherValidationError(f"{label} k_norm epsilon is not {NORM_EPS}")
        if getattr(qkv, "is_meta", False) or getattr(out, "is_meta", False):
            raise TeacherValidationError(f"{label} weights are still on the meta device")

    for index, block in enumerate(blocks):
        check_attention(getattr(block, "attn", None), f"blocks.{index}.attn")
    for index, block in enumerate(refiner_blocks):
        check_attention(
            getattr(block, "attn", None),
            f"token_refiner.blocks.{index}.attn",
        )


def load_pinned_bf16_teacher(
    path: str | Path,
    *,
    model_options: Mapping[str, Any] | None = None,
) -> LoadedNativeTeacher:
    """Load the exact pinned BF16 parent through current ComfyUI for Stage-A work.

    The full-file SHA-256 and native-QKV safetensors signatures are checked before
    deserialization into Comfy. Dynamic/offloaded-on-disk model construction is disabled
    so block-local student initialization never copies from meta/disk placeholder weights.
    The caller remains responsible for normal Comfy model-management placement before a
    full-model capture forward.
    """
    path = Path(path)
    actual_sha = sha256_file(path)
    if actual_sha.lower() != TEACHER_SHA256:
        raise TeacherValidationError(
            f"teacher SHA-256 mismatch: expected {TEACHER_SHA256}, got {actual_sha}"
        )
    signatures, _ = read_safetensors_signatures(path)
    report = validate_native_bf16_teacher_signatures(signatures)

    options = dict(model_options or {})
    requested_dtype = options.get("dtype")
    if requested_dtype not in (None, torch.bfloat16):
        raise TeacherValidationError(
            f"Stage-A teacher compute dtype must remain BF16, got {requested_dtype}"
        )
    if options.get("custom_operations") is not None:
        raise TeacherValidationError(
            "Stage-A teacher loader does not accept custom operations/quantization overrides"
        )
    options["dtype"] = torch.bfloat16
    try:
        import comfy.sd
    except ImportError as exc:
        raise RuntimeError("ComfyUI is required to load the pinned MiniMax-H3 teacher") from exc

    patcher = comfy.sd.load_diffusion_model(
        str(path),
        model_options=options,
        disable_dynamic=True,
    )
    if patcher is None:
        raise TeacherValidationError("ComfyUI did not detect the pinned checkpoint as a diffusion model")
    base_model = getattr(patcher, "model", None)
    diffusion_model = getattr(base_model, "diffusion_model", None)
    if diffusion_model is None:
        raise TeacherValidationError("loaded Comfy model patcher does not expose diffusion_model")
    validate_loaded_native_teacher_model(diffusion_model)
    return LoadedNativeTeacher(
        patcher=patcher,
        diffusion_model=diffusion_model,
        signature_report=NativeTeacherSignatureReport(
            tensor_count=report.tensor_count,
            core_qkv_blocks=report.core_qkv_blocks,
            refiner_qkv_blocks=report.refiner_qkv_blocks,
            teacher_sha256=actual_sha,
        ),
    )
