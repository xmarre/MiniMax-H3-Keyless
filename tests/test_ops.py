from __future__ import annotations

import torch

from minimax_h3_keyless.contracts import RoutingPreprocessor, RoutingSpecV1
from minimax_h3_keyless.ops import (
    apply_h3_split_half_rope,
    dense_reference_attention,
    materialize_route,
    rms_norm,
    torch_sdpa_attention,
)


def _rope_table(angles: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angles)
    s = torch.sin(angles)
    return torch.stack((c, -s, s, c), dim=-1).reshape(
        1, angles.shape[0], 1, angles.shape[1], 2, 2
    )


def test_rms_norm_matches_explicit_formula() -> None:
    torch.manual_seed(1)
    x = torch.randn(5, 3, 7)
    weight = torch.randn(7)
    eps = 1e-5
    expected = (
        x.float()
        * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
        * weight.float()
    ).to(x.dtype)
    torch.testing.assert_close(rms_norm(x, weight, eps), expected)


def test_split_half_rope_rotates_prefix_and_preserves_tail() -> None:
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 9.0, 10.0]]])
    angles = torch.tensor([[torch.pi / 2, 0.0]])
    rope = _rope_table(angles)
    out = apply_h3_split_half_rope(x, rope, rot_dim=4)
    expected = torch.tensor([[[-3.0, 2.0, 1.0, 4.0, 9.0, 10.0]]])
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(out[..., 4:], x[..., 4:])


def test_materialized_route_never_mutates_retrieval_value() -> None:
    torch.manual_seed(2)
    v = torch.randn(4, 2, 3)
    before = v.clone()
    spec = RoutingSpecV1(
        api=1,
        block_index=0,
        norm_weight=torch.ones(3),
        norm_epsilon=1e-5,
        preprocessors=(RoutingPreprocessor("times-two", lambda route: route * 2.0),),
    )
    route = materialize_route(v, spec)
    torch.testing.assert_close(v, before)
    assert route.data_ptr() != v.data_ptr()
    torch.testing.assert_close(route, rms_norm(v, torch.ones(3), 1e-5) * 2.0)


def test_dense_oracle_matches_torch_sdpa_with_measure_and_mask() -> None:
    torch.manual_seed(3)
    q = torch.randn(3, 2, 4)
    route = torch.randn(5, 2, 4)
    v = torch.randn(5, 2, 4)
    mask = torch.zeros(3, 5)
    mask[0, 4] = -10000.0
    log_measure = torch.log(torch.tensor([1.0, 2.0, 0.5, 1.5, 3.0]))
    scale = 0.37
    expected = dense_reference_attention(
        q, route, v, scale=scale, mask=mask, log_measure=log_measure
    )
    actual = torch_sdpa_attention(
        q, route, v, scale=scale, mask=mask, log_measure=log_measure
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_dense_oracle_matches_torch_sdpa_with_boolean_mask() -> None:
    torch.manual_seed(30)
    q = torch.randn(2, 2, 4)
    route = torch.randn(4, 2, 4)
    v = torch.randn(4, 2, 4)
    mask = torch.tensor([[True, True, False, True], [True, False, True, True]])
    expected = dense_reference_attention(q, route, v, mask=mask)
    actual = torch_sdpa_attention(q, route, v, scale=0.5, mask=mask)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_value_row_selection_keeps_v_rope_and_measure_aligned() -> None:
    torch.manual_seed(4)
    v = torch.randn(5, 2, 4)
    rope = _rope_table(torch.arange(10, dtype=torch.float32).reshape(5, 2) / 10.0)
    measure = torch.arange(5, dtype=torch.float32)
    spec = RoutingSpecV1(
        api=1,
        block_index=7,
        norm_weight=torch.ones(4),
        norm_epsilon=1e-5,
        rope_freqs=rope,
    )
    selected_v, selected_spec, selected_measure = spec.select_value_rows(
        v, (4, 1, 3), log_measure=measure, identity="restricted"
    )
    torch.testing.assert_close(selected_v, v[[4, 1, 3]])
    torch.testing.assert_close(selected_spec.rope_freqs, rope[:, [4, 1, 3]])
    torch.testing.assert_close(selected_measure, measure[[4, 1, 3]])
    assert selected_spec.value_domain == selected_spec.routing_position_domain
    assert selected_spec.value_domain is not None
    assert selected_spec.value_domain.indices == (4, 1, 3)
    assert selected_spec.value_domain.identity == "restricted"
