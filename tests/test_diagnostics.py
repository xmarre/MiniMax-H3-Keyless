from __future__ import annotations

import torch

from minimax_h3_keyless.diagnostics import (
    fixed_value_bilinear_residual,
    regularized_ls_row_route,
    split_qkv_storage_weight,
)


def test_fixed_value_residual_is_zero_when_k_equals_v() -> None:
    torch.manual_seed(21)
    a = torch.randn(7, 3)
    b = torch.randn(7, 3)
    result = fixed_value_bilinear_residual(a, b, b)
    assert result.relative_frobenius < 1e-12
    assert result.value_rank == 3


def test_fixed_value_residual_detects_disjoint_value_subspace() -> None:
    a = torch.eye(4)[:, :2]
    b = torch.eye(4)[:, 2:]
    c = torch.eye(4)[:, :2]
    result = fixed_value_bilinear_residual(a, b, c)
    assert result.teacher_frobenius > 0.0
    assert result.relative_frobenius > 0.99


def test_regularized_ls_route_recovers_exact_linear_relation_at_zero_lambda() -> None:
    torch.manual_seed(22)
    c = torch.randn(8, 3, dtype=torch.float64)
    m = torch.randn(3, 3, dtype=torch.float64)
    b = c @ m
    route, lam = regularized_ls_row_route(b, c, lambda_relative=0.0)
    assert lam == 0.0
    torch.testing.assert_close(c @ route.T, b, atol=1e-10, rtol=1e-10)


def test_split_qkv_storage_weight_returns_mathematical_orientation() -> None:
    qkv = torch.arange(3 * 6 * 5, dtype=torch.float32).reshape(18, 5)
    q, k, v = split_qkv_storage_weight(qkv, heads=2, head_dim=3)
    assert q.shape == k.shape == v.shape == (5, 6)
    torch.testing.assert_close(q, qkv[:6].T)
    torch.testing.assert_close(k, qkv[6:12].T)
    torch.testing.assert_close(v, qkv[12:].T)
