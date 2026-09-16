from __future__ import annotations

import pytest
import torch

from minimax_h3_keyless.contracts import RowDomain, RoutingSpecV1, get_keyless_provider, PROVIDER_KEY


def test_row_domain_rejects_mixed_slice_and_indices() -> None:
    with pytest.raises(ValueError, match="both slice bounds and explicit indices"):
        RowDomain(start=0, stop=2, indices=(0, 1))


def test_row_domain_rejects_partial_slice() -> None:
    with pytest.raises(ValueError, match="provided together"):
        RowDomain(start=0)


def test_routing_spec_rejects_nonpositive_epsilon() -> None:
    with pytest.raises(ValueError, match="positive"):
        RoutingSpecV1(api=1, block_index=0, norm_weight=torch.ones(4), norm_epsilon=0.0)


def test_provider_contract_requires_callable_api_one() -> None:
    class NotCallable:
        api = 1

    with pytest.raises(RuntimeError, match="callable"):
        get_keyless_provider({PROVIDER_KEY: NotCallable()})
