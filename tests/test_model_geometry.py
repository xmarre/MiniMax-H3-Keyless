from __future__ import annotations

import pytest

from minimax_h3_keyless.contracts import (
    CORE_BLOCKS,
    HEAD_DIM,
    HEADS,
    HIDDEN_SIZE,
    NORM_EPS,
    TOKEN_REFINER_BLOCKS,
)
from minimax_h3_keyless.model import validate_keyless_geometry


def _valid(**updates):
    values = {
        "hidden": HIDDEN_SIZE,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "core_blocks": CORE_BLOCKS,
        "token_refiner_blocks": TOKEN_REFINER_BLOCKS,
        "norm_epsilon": NORM_EPS,
    }
    values.update(updates)
    return values


def test_canonical_geometry_is_accepted() -> None:
    validate_keyless_geometry(**_valid())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hidden", HIDDEN_SIZE - 1),
        ("heads", HEADS - 1),
        ("head_dim", HEAD_DIM // 2),
        ("core_blocks", CORE_BLOCKS - 1),
        ("token_refiner_blocks", TOKEN_REFINER_BLOCKS - 1),
        ("norm_epsilon", 1e-6),
    ],
)
def test_noncanonical_geometry_fails_before_contract_attachment(field, value) -> None:
    with pytest.raises(RuntimeError, match="geometry mismatch"):
        validate_keyless_geometry(**_valid(**{field: value}))
