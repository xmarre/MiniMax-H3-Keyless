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


def test_value_row_selection_composes_existing_value_and_position_domains() -> None:
    v = torch.arange(6 * 2 * 4, dtype=torch.float32).reshape(6, 2, 4)
    rope = torch.arange(6, dtype=torch.float32).view(1, 6, 1, 1, 1, 1).expand(1, 6, 1, 2, 2, 2)
    measure = torch.arange(6, dtype=torch.float32)
    spec = RoutingSpecV1(
        api=1,
        block_index=3,
        norm_weight=torch.ones(4),
        norm_epsilon=1e-5,
        rope_freqs=rope,
        value_domain=RowDomain(start=10, stop=16, identity="values"),
        routing_position_domain=RowDomain(indices=(100, 101, 102, 103, 104, 105), identity="positions"),
    )

    v1, spec1, measure1 = spec.select_value_rows(v, (1, 3, 5), log_measure=measure)
    assert spec1.value_domain == RowDomain(indices=(11, 13, 15), identity="values")
    assert spec1.routing_position_domain == RowDomain(
        indices=(101, 103, 105), identity="positions"
    )
    torch.testing.assert_close(v1, v.index_select(0, torch.tensor([1, 3, 5])))
    torch.testing.assert_close(measure1, torch.tensor([1.0, 3.0, 5.0]))
    torch.testing.assert_close(spec1.rope_freqs[:, :, 0, 0, 0, 0], torch.tensor([[1.0, 3.0, 5.0]]))

    v2, spec2, measure2 = spec1.select_value_rows(v1, (2, 0), log_measure=measure1)
    assert spec2.value_domain == RowDomain(indices=(15, 11), identity="values")
    assert spec2.routing_position_domain == RowDomain(indices=(105, 101), identity="positions")
    torch.testing.assert_close(v2, v.index_select(0, torch.tensor([5, 1])))
    torch.testing.assert_close(measure2, torch.tensor([5.0, 1.0]))
    torch.testing.assert_close(spec2.rope_freqs[:, :, 0, 0, 0, 0], torch.tensor([[5.0, 1.0]]))


def test_value_row_selection_rejects_stale_domain_lengths() -> None:
    v = torch.zeros(4, 1, 2)
    spec = RoutingSpecV1(
        api=1,
        block_index=0,
        norm_weight=torch.ones(2),
        norm_epsilon=1e-5,
        value_domain=RowDomain(indices=(10, 11, 12)),
    )
    with pytest.raises(ValueError, match="length must match"):
        spec.select_value_rows(v, (0, 1))


def test_value_row_selection_rejects_misaligned_rope_rows_and_oob_indices() -> None:
    v = torch.zeros(4, 1, 2)
    spec = RoutingSpecV1(
        api=1,
        block_index=0,
        norm_weight=torch.ones(2),
        norm_epsilon=1e-5,
        rope_freqs=torch.zeros(1, 3, 1, 1, 2, 2),
    )
    with pytest.raises(ValueError, match="rope_freqs must align"):
        spec.select_value_rows(v, (0, 1))

    spec_no_rope = RoutingSpecV1(
        api=1,
        block_index=0,
        norm_weight=torch.ones(2),
        norm_epsilon=1e-5,
    )
    with pytest.raises(IndexError, match="outside"):
        spec_no_rope.select_value_rows(v, (0, 4))
