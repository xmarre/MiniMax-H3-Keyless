from __future__ import annotations

import json

import pytest
import torch

from minimax_h3_keyless.contracts import HEADS, HIDDEN_SIZE
from minimax_h3_keyless.loader import _derive_h3_unet_config, _merge_metadata_config


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
