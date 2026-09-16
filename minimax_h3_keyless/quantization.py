from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .checkpoint import (
    TensorSignature,
    read_safetensors_signatures,
    sha256_file,
    validate_deploy_checkpoint,
    validate_int8_convrot_checkpoint,
)
from .contracts import CORE_BLOCKS, HIDDEN_SIZE, INNER_DIM, QUANTIZATION_RECIPE
from .export import build_export_manifest_body, manifest_identity_sha256


QUANTIZATION_FORMAT = "int8_tensorwise"
CONVROT_GROUPSIZE = 256
QUANTIZED_LINEAR_COUNT = CORE_BLOCKS * 4
OMIT_FROM_QUANTIZED_DERIVATIVE = frozenset({"adaln_basis", "adaln_mean"})


@dataclass(frozen=True)
class Int8ExportResult:
    artifact_path: str
    artifact_sha256: str
    artifact_bytes: int
    source_bf16_sha256: str
    manifest_path: str
    manifest_sha256: str
    manifest_identity_sha256: str
    quantized_linear_count: int
    tensor_count: int


def quantized_linear_weight_keys() -> tuple[str, ...]:
    keys: list[str] = []
    for i in range(CORE_BLOCKS):
        keys.extend(
            (
                f"blocks.{i}.attn.qv_proj.weight",
                f"blocks.{i}.attn.out_proj.weight",
                f"blocks.{i}.mlp.fc1.weight",
                f"blocks.{i}.mlp.fc2.weight",
            )
        )
    return tuple(keys)


def expected_quantized_linear_shapes() -> dict[str, tuple[int, int]]:
    shapes: dict[str, tuple[int, int]] = {}
    for i in range(CORE_BLOCKS):
        shapes[f"blocks.{i}.attn.qv_proj.weight"] = (2 * INNER_DIM, HIDDEN_SIZE)
        shapes[f"blocks.{i}.attn.out_proj.weight"] = (HIDDEN_SIZE, INNER_DIM)
        shapes[f"blocks.{i}.mlp.fc1.weight"] = (4 * INNER_DIM, HIDDEN_SIZE)
        shapes[f"blocks.{i}.mlp.fc2.weight"] = (HIDDEN_SIZE, 2 * INNER_DIM)
    return shapes


def comfy_quant_descriptor(*, group_size: int = CONVROT_GROUPSIZE) -> torch.Tensor:
    config = {
        "convrot": True,
        "convrot_groupsize": int(group_size),
        "format": QUANTIZATION_FORMAT,
    }
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8)


def _load_live_int8_layout():
    try:
        from comfy_kitchen.tensor import TensorWiseINT8Layout
    except ImportError as exc:
        raise RuntimeError(
            "INT8 ConvRot export requires the live comfy-kitchen package. "
            "Run this exporter inside the audited ComfyUI environment."
        ) from exc
    return TensorWiseINT8Layout


def quantize_convrot_weight(
    weight: torch.Tensor,
    *,
    group_size: int = CONVROT_GROUPSIZE,
    layout_cls=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize one accepted BF16 linear with live per-channel ConvRot semantics."""
    if weight.ndim != 2:
        raise ValueError(f"ConvRot linear weight must be rank-2, got {tuple(weight.shape)}")
    if weight.dtype != torch.bfloat16:
        raise ValueError(f"ConvRot source must be BF16, got {weight.dtype}")
    if weight.shape[1] % group_size:
        raise ValueError(
            f"linear input width {weight.shape[1]} is not divisible by ConvRot groupsize {group_size}"
        )
    layout_cls = _load_live_int8_layout() if layout_cls is None else layout_cls
    qdata, params = layout_cls.quantize(
        weight,
        stochastic_rounding=0,
        is_weight=True,
        per_channel=True,
        convrot=True,
        convrot_groupsize=group_size,
    )
    scale = params.scale
    if qdata.dtype != torch.int8 or tuple(qdata.shape) != tuple(weight.shape):
        raise RuntimeError(
            "live TensorWiseINT8Layout returned an unexpected ConvRot storage tensor: "
            f"dtype={qdata.dtype}, shape={tuple(qdata.shape)}"
        )
    if scale.dtype != torch.float32 or tuple(scale.shape) != (weight.shape[0], 1):
        raise RuntimeError(
            "live TensorWiseINT8Layout returned an unexpected per-output-row scale: "
            f"dtype={scale.dtype}, shape={tuple(scale.shape)}"
        )
    if not getattr(params, "convrot", False):
        raise RuntimeError("live TensorWiseINT8Layout did not preserve convrot=True")
    if int(getattr(params, "convrot_groupsize", -1)) != int(group_size):
        raise RuntimeError("live TensorWiseINT8Layout changed the requested ConvRot groupsize")
    return (
        qdata.detach().cpu().contiguous(),
        scale.detach().cpu().contiguous(),
        comfy_quant_descriptor(group_size=group_size),
    )


def _source_preflight(
    signatures: Mapping[str, TensorSignature], metadata: Mapping[str, str]
) -> tuple[str, ...]:
    validate_deploy_checkpoint(signatures, metadata)
    targets = quantized_linear_weight_keys()
    shapes = expected_quantized_linear_shapes()
    if len(targets) != QUANTIZED_LINEAR_COUNT or len(set(targets)) != QUANTIZED_LINEAR_COUNT:
        raise RuntimeError("internal Keyless INT8 recipe does not contain exactly 200 unique linears")
    if any(key.endswith(".comfy_quant") for key in signatures):
        raise RuntimeError(
            "INT8 ConvRot export source must be the accepted folded BF16 artifact, not a quantized artifact"
        )
    for key in targets:
        signature = signatures.get(key)
        if signature is None:
            raise RuntimeError(f"folded BF16 source is missing quantization target {key}")
        if signature.shape != shapes[key]:
            raise RuntimeError(f"{key}: expected {shapes[key]}, got {signature.shape}")
        if signature.dtype != "BF16":
            raise RuntimeError(f"{key}: expected BF16 source weight, got {signature.dtype}")
        if signature.shape[1] % CONVROT_GROUPSIZE:
            raise RuntimeError(f"{key}: input width is incompatible with groupsize {CONVROT_GROUPSIZE}")
    return targets


def _compare_source_and_quantized_signatures(
    source: Mapping[str, TensorSignature],
    quantized: Mapping[str, TensorSignature],
) -> None:
    targets = set(quantized_linear_weight_keys())
    expected_keys = set(source).difference(OMIT_FROM_QUANTIZED_DERIVATIVE)
    for key in targets:
        expected_keys.add(key + "_scale")
        expected_keys.add(key.removesuffix(".weight") + ".comfy_quant")
    if set(quantized) != expected_keys:
        missing = sorted(expected_keys.difference(quantized))
        extra = sorted(set(quantized).difference(expected_keys))
        raise RuntimeError(
            "quantized artifact key set differs from the folded BF16 source/recipe: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    for key, signature in source.items():
        if key in OMIT_FROM_QUANTIZED_DERIVATIVE or key in targets:
            continue
        if quantized[key] != signature:
            raise RuntimeError(
                f"non-quantized tensor {key} changed signature: {signature} -> {quantized[key]}"
            )


def _descriptor_key(weight_key: str) -> str:
    return weight_key.removesuffix(".weight") + ".comfy_quant"


def _validate_descriptor_payloads(path: str | Path) -> None:
    expected = {
        "format": QUANTIZATION_FORMAT,
        "convrot": True,
        "convrot_groupsize": CONVROT_GROUPSIZE,
    }
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for weight_key in quantized_linear_weight_keys():
            descriptor_key = _descriptor_key(weight_key)
            raw = f.get_tensor(descriptor_key)
            try:
                payload = json.loads(bytes(raw.tolist()).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(f"invalid native Comfy quant descriptor {descriptor_key}") from exc
            if payload != expected:
                raise RuntimeError(
                    f"{descriptor_key}: expected native ConvRot descriptor {expected}, got {payload}"
                )


def export_int8_convrot_from_bf16(
    source_path: str | Path,
    output_path: str | Path,
    *,
    export_commit: str,
    quantize_device: str | torch.device = "cuda",
    manifest_path: str | Path | None = None,
    command: str | None = None,
    manifest_extra: Mapping[str, Any] | None = None,
    quantize_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
) -> Int8ExportResult:
    """Stream an accepted folded BF16 Keyless artifact into native Comfy INT8 ConvRot.

    Only one source heavy linear is materialized at a time. The output state remains
    resident until safetensors serialization, avoiding a second full BF16 model copy.
    Production conversion uses the live comfy-kitchen TensorWiseINT8Layout; ``quantize_fn``
    exists only for bounded tests.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    signatures, source_metadata = read_safetensors_signatures(source_path)
    targets = set(_source_preflight(signatures, source_metadata))
    source_sha = sha256_file(source_path)
    device = torch.device(quantize_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA quantization was requested but torch.cuda.is_available() is false")
    quantize_fn = quantize_fn or (lambda weight: quantize_convrot_weight(weight))

    output: dict[str, torch.Tensor] = {}
    with safe_open(str(source_path), framework="pt", device="cpu") as f:
        for key in sorted(f.keys()):
            if key in OMIT_FROM_QUANTIZED_DERIVATIVE:
                continue
            tensor = f.get_tensor(key)
            if key not in targets:
                output[key] = tensor.contiguous()
                continue
            weight = tensor.to(device=device, dtype=torch.bfloat16)
            qdata, scale, descriptor = quantize_fn(weight)
            del weight
            output[key] = qdata.cpu().contiguous()
            output[key + "_scale"] = scale.cpu().contiguous()
            output[_descriptor_key(key)] = descriptor.cpu().contiguous()

    metadata = {k: str(v) for k, v in source_metadata.items() if k != "manifest_sha256"}
    metadata.update(
        {
            "quantization_format": QUANTIZATION_FORMAT,
            "quantization_layer_recipe": QUANTIZATION_RECIPE,
            "quantization_layer_count": str(QUANTIZED_LINEAR_COUNT),
            "quantization_per_channel": "true",
            "quantization_convrot": "true",
            "quantization_convrot_groupsize": str(CONVROT_GROUPSIZE),
            "quantization_source_bf16_sha256": source_sha,
            "export_commit": str(export_commit),
        }
    )
    body_extra = dict(manifest_extra or {})
    body_extra.update(
        {
            "source_bf16_path": source_path.name,
            "source_bf16_sha256": source_sha,
            "quantization_recipe": QUANTIZATION_RECIPE,
            "quantization_device_type": device.type,
            "stochastic_rounding": 0,
        }
    )
    body = build_export_manifest_body(
        output,
        artifact_filename=output_path.name,
        metadata=metadata,
        command=command,
        extra=body_extra,
    )
    identity = manifest_identity_sha256(body)
    metadata["manifest_sha256"] = identity
    validate_int8_convrot_checkpoint(output, metadata)

    save_file(
        {k: v.detach().cpu().contiguous() for k, v in output.items()},
        str(output_path),
        metadata=metadata,
    )
    output_signatures, output_metadata = read_safetensors_signatures(output_path)
    validate_int8_convrot_checkpoint(output_signatures, output_metadata)
    _compare_source_and_quantized_signatures(signatures, output_signatures)
    _validate_descriptor_payloads(output_path)

    artifact_sha = sha256_file(output_path)
    artifact_bytes = output_path.stat().st_size
    receipt = dict(body)
    receipt["manifest_sha256"] = identity
    receipt["artifact_sha256"] = artifact_sha
    receipt["artifact_bytes"] = artifact_bytes
    if manifest_path is None:
        manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path = Path(manifest_path)
    manifest_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return Int8ExportResult(
        artifact_path=str(output_path),
        artifact_sha256=artifact_sha,
        artifact_bytes=artifact_bytes,
        source_bf16_sha256=source_sha,
        manifest_path=str(manifest_path),
        manifest_sha256=sha256_file(manifest_path),
        manifest_identity_sha256=identity,
        quantized_linear_count=QUANTIZED_LINEAR_COUNT,
        tensor_count=len(output),
    )
