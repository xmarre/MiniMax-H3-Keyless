from __future__ import annotations

import json

import pytest
import torch

import minimax_h3_keyless.loader as loader_mod
from minimax_h3_keyless.contracts import HEADS, HIDDEN_SIZE
from minimax_h3_keyless.loader import (
    _derive_h3_unet_config,
    _merge_metadata_config,
    _quant_dtype_policy,
    _require_manifest_identity,
    _validate_loaded_checkpoint,
)


_MANIFEST = {"manifest_sha256": "a" * 64}


def _minimal_state() -> dict[str, torch.Tensor]:
    return {
        "blocks.0.mlp.fc1.weight": torch.empty(20, 3),
        "condition_proj.weight": torch.empty(4, 5),
        "rope.inv_freq": torch.empty(16),
        "adaln_t_table": torch.empty(1025, 8),
    }


def test_metadata_config_cannot_override_structural_geometry() -> None:
    metadata = {"config": json.dumps({"num_attention_heads": HEADS - 1})}
    with pytest.raises(RuntimeError, match="contradicts checkpoint structure"):
        _derive_h3_unet_config(_minimal_state(), metadata)


def test_metadata_config_may_add_nonstructural_constructor_fact() -> None:
    cfg = _derive_h3_unet_config(
        _minimal_state(), {"config": json.dumps({"sigma_shift_video": 12.0})}
    )
    assert cfg["hidden_size"] == HIDDEN_SIZE
    assert cfg["num_attention_heads"] == HEADS
    assert cfg["sigma_shift_video"] == 12.0


def test_metadata_config_rejects_non_object_json() -> None:
    with pytest.raises(RuntimeError, match="must decode to an object"):
        _merge_metadata_config({}, {"config": "[]"})


def test_metadata_config_rejects_malformed_json() -> None:
    with pytest.raises(RuntimeError, match="not valid JSON"):
        _merge_metadata_config({}, {"config": "{"})


def test_loader_requires_manifest_identity_before_dispatch() -> None:
    assert _require_manifest_identity(_MANIFEST) == "a" * 64
    with pytest.raises(RuntimeError, match="manifest_sha256"):
        _require_manifest_identity({})
    with pytest.raises(RuntimeError, match="manifest_sha256"):
        _require_manifest_identity({"manifest_sha256": "not-a-hash"})


def test_loader_selects_bf16_validator_only_for_unquantized_storage(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(loader_mod, "validate_deploy_checkpoint", lambda sd, md: calls.append("bf16"))
    monkeypatch.setattr(loader_mod, "validate_int8_convrot_checkpoint", lambda sd, md: calls.append("int8"))
    assert _validate_loaded_checkpoint({"weight": object()}, _MANIFEST) == "bf16"
    assert calls == ["bf16"]


def test_loader_selects_strict_int8_validator_from_native_descriptor(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(loader_mod, "validate_deploy_checkpoint", lambda sd, md: calls.append("bf16"))
    monkeypatch.setattr(loader_mod, "validate_int8_convrot_checkpoint", lambda sd, md: calls.append("int8"))
    assert _validate_loaded_checkpoint({"layer.comfy_quant": object()}, _MANIFEST) == "int8_convrot"
    assert calls == ["int8"]


def test_loader_does_not_ignore_quantization_metadata_without_storage(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(loader_mod, "validate_deploy_checkpoint", lambda sd, md: calls.append("bf16"))
    monkeypatch.setattr(loader_mod, "validate_int8_convrot_checkpoint", lambda sd, md: calls.append("int8"))
    metadata = {**_MANIFEST, "quantization_format": "int8_tensorwise"}
    assert _validate_loaded_checkpoint({}, metadata) == "int8_convrot"
    assert calls == ["int8"]


def test_quantized_storage_dtype_is_not_used_as_compute_dtype_hint() -> None:
    effective, is_quantized = _quant_dtype_policy(torch.int8, {"mixed_ops": True})
    assert effective is None
    assert is_quantized is True


def test_bf16_storage_dtype_remains_compute_dtype_hint() -> None:
    effective, is_quantized = _quant_dtype_policy(torch.bfloat16, None)
    assert effective == torch.bfloat16
    assert is_quantized is False
