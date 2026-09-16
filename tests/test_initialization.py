from __future__ import annotations

import torch

from minimax_h3_keyless.attention import KeylessAttentionTrain
from minimax_h3_keyless.initialization import (
    initialize_training_attention_from_native,
    set_pilot_trainable_stage,
)


def _teacher(hidden: int, heads: int, head_dim: int):
    torch.manual_seed(31)
    inner = heads * head_dim
    return {
        "qkv": torch.randn(3 * inner, hidden),
        "q_norm": torch.randn(head_dim),
        "k_norm": torch.randn(head_dim),
        "out": torch.randn(hidden, inner),
    }


def test_identity_initialization_copies_q_v_norms_and_out() -> None:
    hidden, heads, head_dim = 7, 2, 3
    student = KeylessAttentionTrain(hidden, heads, head_dim, 1e-5, dtype=torch.float32)
    teacher = _teacher(hidden, heads, head_dim)
    report = initialize_training_attention_from_native(
        student,
        qkv_weight=teacher["qkv"],
        q_norm_weight=teacher["q_norm"],
        k_norm_weight=teacher["k_norm"],
        out_proj_weight=teacher["out"],
    )
    inner = heads * head_dim
    torch.testing.assert_close(student.q_proj.weight, teacher["qkv"][:inner])
    torch.testing.assert_close(student.v_proj.weight, teacher["qkv"][2 * inner:])
    torch.testing.assert_close(student.q_norm.weight, teacher["q_norm"])
    torch.testing.assert_close(student.route_norm.weight, teacher["k_norm"])
    torch.testing.assert_close(student.out_proj.weight, teacher["out"])
    eye = torch.eye(head_dim).expand(heads, -1, -1)
    torch.testing.assert_close(student.query_route.weight, eye)
    assert report.mode == "identity"
    assert report.lambda_actual is None


def test_least_squares_initialization_recovers_exact_v_to_k_map() -> None:
    torch.manual_seed(32)
    hidden, heads, head_dim = 8, 2, 3
    inner = heads * head_dim
    q = torch.randn(inner, hidden)
    v = torch.randn(inner, hidden)
    expected_route = torch.randn(heads, head_dim, head_dim)
    k_heads = []
    for h in range(heads):
        a, b = h * head_dim, (h + 1) * head_dim
        c_math = v[a:b].T
        k_math = c_math @ expected_route[h].T
        k_heads.append(k_math.T)
    k = torch.cat(k_heads, dim=0)
    qkv = torch.cat((q, k, v), dim=0)
    student = KeylessAttentionTrain(hidden, heads, head_dim, 1e-5, dtype=torch.float32)
    report = initialize_training_attention_from_native(
        student,
        qkv_weight=qkv,
        q_norm_weight=torch.ones(head_dim),
        k_norm_weight=torch.ones(head_dim),
        out_proj_weight=torch.randn(hidden, inner),
        route_mode="least_squares",
        lambda_relative=0.0,
    )
    torch.testing.assert_close(student.query_route.weight, expected_route, atol=1e-5, rtol=1e-5)
    assert report.lambda_actual == (0.0, 0.0)


def test_zero_lambda_least_squares_handles_rank_deficient_value_projection() -> None:
    hidden, heads, head_dim = 5, 1, 3
    inner = heads * head_dim
    q = torch.randn(inner, hidden)
    base = torch.randn(1, hidden)
    v = torch.cat((base, 2 * base, 3 * base), dim=0)
    k = torch.randn(inner, hidden)
    student = KeylessAttentionTrain(hidden, heads, head_dim, 1e-5, dtype=torch.float32)
    initialize_training_attention_from_native(
        student,
        qkv_weight=torch.cat((q, k, v), dim=0),
        q_norm_weight=torch.ones(head_dim),
        k_norm_weight=torch.ones(head_dim),
        out_proj_weight=torch.randn(hidden, inner),
        route_mode="least_squares",
        lambda_relative=0.0,
    )
    assert torch.isfinite(student.query_route.weight).all()


def test_pilot_freeze_schedule_only_exposes_declared_attention_parameters() -> None:
    student = KeylessAttentionTrain(7, 2, 3, 1e-5, dtype=torch.float32, gate_compress=True)
    expected = {
        "route": {"query_route.weight", "route_norm.weight"},
        "query": {"query_route.weight", "route_norm.weight", "q_proj.weight"},
        "value": {"query_route.weight", "route_norm.weight", "q_proj.weight", "v_proj.weight"},
        "norm_out": {
            "query_route.weight",
            "route_norm.weight",
            "q_proj.weight",
            "v_proj.weight",
            "q_norm.weight",
            "out_proj.weight",
        },
    }
    for stage, wanted in expected.items():
        set_pilot_trainable_stage(student, stage)
        actual = {name for name, p in student.named_parameters() if p.requires_grad}
        assert actual == wanted
        assert "to_gate_compress.weight" not in actual
