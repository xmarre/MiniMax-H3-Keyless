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


def test_regularized_ls_route_recovers_storage_factor_at_zero_lambda() -> None:
    torch.manual_seed(22)
    c = torch.randn(8, 3, dtype=torch.float64)
    storage_route = torch.randn(3, 3, dtype=torch.float64)
    b = c @ storage_route
    actual, lam = regularized_ls_row_route(b, c, lambda_relative=0.0)
    assert lam == 0.0
    torch.testing.assert_close(actual, storage_route, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(c @ actual, b, atol=1e-10, rtol=1e-10)


def test_ls_storage_factor_reproduces_teacher_raw_bilinear_when_k_is_in_v_span() -> None:
    torch.manual_seed(23)
    hidden, dim = 9, 3
    a = torch.randn(hidden, dim, dtype=torch.float64)
    c = torch.randn(hidden, dim, dtype=torch.float64)
    storage_route = torch.randn(dim, dim, dtype=torch.float64)
    b = c @ storage_route
    recovered, _ = regularized_ls_row_route(b, c, lambda_relative=0.0)
    teacher = a @ b.T
    student = (a @ recovered.T) @ c.T
    torch.testing.assert_close(student, teacher, atol=1e-10, rtol=1e-10)


def test_regularized_storage_factor_matches_design_closed_form() -> None:
    torch.manual_seed(24)
    c = torch.randn(10, 3, dtype=torch.float64)
    b = torch.randn(10, 3, dtype=torch.float64)
    relative = 1e-2
    actual, lam = regularized_ls_row_route(b, c, lambda_relative=relative)
    gram = c.T @ c
    expected = torch.linalg.solve(
        gram + lam * torch.eye(3, dtype=torch.float64),
        c.T @ b,
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    # PerHeadLinear uses q @ W.T, which is the design's B.T C (...) query factor.
    expected_query_factor = b.T @ c @ torch.linalg.inv(
        gram + lam * torch.eye(3, dtype=torch.float64)
    )
    torch.testing.assert_close(actual.T, expected_query_factor, atol=1e-12, rtol=1e-12)


def test_split_qkv_storage_weight_returns_mathematical_orientation() -> None:
    qkv = torch.arange(3 * 6 * 5, dtype=torch.float32).reshape(18, 5)
    q, k, v = split_qkv_storage_weight(qkv, heads=2, head_dim=3)
    assert q.shape == k.shape == v.shape == (5, 6)
    torch.testing.assert_close(q, qkv[:6].T)
    torch.testing.assert_close(k, qkv[6:12].T)
    torch.testing.assert_close(v, qkv[12:].T)
