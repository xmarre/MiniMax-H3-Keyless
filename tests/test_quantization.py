from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import minimax_h3_keyless.quantization as quant_mod
from minimax_h3_keyless.checkpoint import sha256_file
from minimax_h3_keyless.contracts import (
    CORE_BLOCKS,
    HIDDEN_SIZE,
    INNER_DIM,
    QUANTIZATION_RECIPE,
    TARGET_MODEL_REVISION,
    TEACHER_COMPATIBILITY_MARKER,
    TEACHER_SHA256,
)
from minimax_h3_keyless.export import manifest_identity_sha256
from minimax_h3_keyless.quantization import (
    CONVROT_GROUPSIZE,
    QUANTIZATION_FORMAT,
    QUANTIZED_LINEAR_COUNT,
    comfy_quant_descriptor,
    expected_quantized_linear_shapes,
    quantize_convrot_weight,
    quantized_linear_weight_keys,
)


def test_keyless_convrot_recipe_targets_exactly_four_core_linears_per_block() -> None:
    keys = quantized_linear_weight_keys()
    assert len(keys) == len(set(keys)) == QUANTIZED_LINEAR_COUNT == CORE_BLOCKS * 4 == 200
    assert keys[:4] == (
        "blocks.0.attn.qv_proj.weight",
        "blocks.0.attn.out_proj.weight",
        "blocks.0.mlp.fc1.weight",
        "blocks.0.mlp.fc2.weight",
    )
    assert not any(key.startswith("token_refiner.") for key in keys)
    assert not any("qkv_proj" in key for key in keys)


def test_recipe_shapes_match_keyless_core_geometry_and_groupsize() -> None:
    shapes = expected_quantized_linear_shapes()
    assert shapes["blocks.0.attn.qv_proj.weight"] == (2 * INNER_DIM, HIDDEN_SIZE)
    assert shapes["blocks.0.attn.out_proj.weight"] == (HIDDEN_SIZE, INNER_DIM)
    assert shapes["blocks.0.mlp.fc1.weight"] == (4 * INNER_DIM, HIDDEN_SIZE)
    assert shapes["blocks.0.mlp.fc2.weight"] == (HIDDEN_SIZE, 2 * INNER_DIM)
    assert len(shapes) == 200
    assert all(shape[1] % CONVROT_GROUPSIZE == 0 for shape in shapes.values())


def test_native_descriptor_is_exact_comfy_int8_convrot_contract() -> None:
    descriptor = comfy_quant_descriptor()
    assert descriptor.dtype == torch.uint8
    assert descriptor.shape == (67,)
    payload = json.loads(bytes(descriptor.tolist()).decode("utf-8"))
    assert payload == {
        "format": QUANTIZATION_FORMAT,
        "convrot": True,
        "convrot_groupsize": 256,
    }
    assert QUANTIZATION_RECIPE == "minimax_h3_keyless_core50_200_v1"


def test_quantize_weight_requires_bf16_and_group_aligned_input() -> None:
    class UnusedLayout:
        @classmethod
        def quantize(cls, *args, **kwargs):
            raise AssertionError("layout must not be called")

    with pytest.raises(ValueError, match="BF16"):
        quantize_convrot_weight(torch.zeros(2, 256), layout_cls=UnusedLayout)
    with pytest.raises(ValueError, match="not divisible"):
        quantize_convrot_weight(
            torch.zeros(2, 255, dtype=torch.bfloat16), layout_cls=UnusedLayout
        )


def test_quantize_weight_forwards_exact_live_layout_options() -> None:
    calls = []

    class FakeLayout:
        @classmethod
        def quantize(cls, tensor, **kwargs):
            calls.append(kwargs)
            qdata = torch.zeros_like(tensor, dtype=torch.int8)
            params = SimpleNamespace(
                scale=torch.ones(tensor.shape[0], 1, dtype=torch.float32),
                convrot=True,
                convrot_groupsize=256,
            )
            return qdata, params

    weight = torch.randn(3, 256, dtype=torch.bfloat16)
    qdata, scale, descriptor = quantize_convrot_weight(weight, layout_cls=FakeLayout)
    assert calls == [
        {
            "stochastic_rounding": 0,
            "is_weight": True,
            "per_channel": True,
            "convrot": True,
            "convrot_groupsize": 256,
        }
    ]
    assert qdata.dtype == torch.int8 and qdata.shape == weight.shape
    assert scale.dtype == torch.float32 and scale.shape == (3, 1)
    assert descriptor.dtype == torch.uint8


def _accepted_bf16_metadata(manifest_identity: str = "a" * 64) -> dict[str, str]:
    return {
        "teacher_compatibility": TEACHER_COMPATIBILITY_MARKER,
        "parent_model_sha256": TEACHER_SHA256,
        "parent_model_revision": TARGET_MODEL_REVISION,
        "manifest_sha256": manifest_identity,
    }


def test_int8_source_requires_pinned_teacher_compatibility_metadata() -> None:
    metadata = _accepted_bf16_metadata("b" * 64)
    assert quant_mod._require_accepted_bf16_metadata(metadata) == "b" * 64

    bad = dict(metadata)
    bad.pop("teacher_compatibility")
    with pytest.raises(RuntimeError, match="pinned-teacher-compatible"):
        quant_mod._require_accepted_bf16_metadata(bad)

    bad = dict(metadata)
    bad["parent_model_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="parent_model_sha256"):
        quant_mod._require_accepted_bf16_metadata(bad)

    bad = dict(metadata)
    bad["quantization_format"] = QUANTIZATION_FORMAT
    with pytest.raises(RuntimeError, match="already declares a quantized artifact"):
        quant_mod._require_accepted_bf16_metadata(bad)


def test_bf16_source_receipt_rebinds_manifest_metadata_and_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "student.safetensors"
    artifact.write_bytes(b"bounded-test-artifact")
    metadata = _accepted_bf16_metadata()
    body = {
        "schema": "minimax_h3_keyless_export_manifest_v1",
        "artifact_filename": artifact.name,
        "metadata": dict(sorted({k: v for k, v in metadata.items() if k != "manifest_sha256"}.items())),
        "tensor_count": 0,
        "tensor_payload_bytes": 0,
        "tensors": [],
    }
    identity = manifest_identity_sha256(body)
    metadata["manifest_sha256"] = identity
    receipt = dict(body)
    receipt["manifest_sha256"] = identity
    receipt["artifact_sha256"] = sha256_file(artifact)
    receipt["artifact_bytes"] = artifact.stat().st_size
    receipt_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    verified_path, receipt_sha = quant_mod._verify_source_bf16_receipt(
        artifact,
        source_sha256=sha256_file(artifact),
        source_metadata=metadata,
        manifest_identity=identity,
    )
    assert verified_path == receipt_path
    assert receipt_sha == sha256_file(receipt_path)

    tampered = dict(receipt)
    tampered["tensor_count"] = 1
    receipt_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not reproduce"):
        quant_mod._verify_source_bf16_receipt(
            artifact,
            source_sha256=sha256_file(artifact),
            source_metadata=metadata,
            manifest_identity=identity,
        )
