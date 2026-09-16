from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from minimax_h3_keyless.activation_capture import CapturedPilotCase, PilotCase
from minimax_h3_keyless.activation_metrics import streamed_attention_comparison, tensor_error_metrics
from minimax_h3_keyless.attention import KeylessAttentionTrain
from minimax_h3_keyless.ops import normalized_positioned
from minimax_h3_keyless.pilot_attention_diagnostics import (
    compare_captured_native_keyless_attention,
    deterministic_query_rows,
)


class _NativeAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int) -> None:
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.qkv_proj = nn.Linear(hidden, 3 * inner, bias=False)
        self.q_norm = SimpleNamespace(weight=nn.Parameter(torch.ones(head_dim)), eps=1e-5)
        self.k_norm = SimpleNamespace(weight=nn.Parameter(torch.ones(head_dim)), eps=1e-5)
        self.out_proj = nn.Linear(inner, hidden, bias=False)


def _capture(hidden: torch.Tensor, *, segments: list[list[object]]) -> CapturedPilotCase:
    case = PilotCase(
        case_id="heldout-a",
        hidden=hidden.clone(),
        shift=torch.zeros(hidden.shape[-1]),
        scale=torch.ones(hidden.shape[-1]),
        gate=torch.zeros(hidden.shape[-1]),
        rope_freqs=None,
        sigma=0.5,
        modality_label="mixed",
        context={
            "minimax_h3_keyless_live_capture_v1": {
                "layout": {"seq_len": hidden.shape[0], "segments": segments}
            }
        },
    )
    return CapturedPilotCase(
        case=case,
        attention_input=hidden.clone(),
        attention_output=torch.zeros_like(hidden),
        block_output=torch.zeros_like(hidden),
    )


def _student_from_teacher(teacher: _NativeAttention) -> KeylessAttentionTrain:
    hidden = teacher.qkv_proj.weight.shape[1]
    heads = teacher.heads
    head_dim = teacher.head_dim
    inner = heads * head_dim
    student = KeylessAttentionTrain(hidden, heads, head_dim, 1e-5, dtype=torch.float32)
    with torch.no_grad():
        q, k, v = teacher.qkv_proj.weight.split(inner, dim=0)
        # Make the synthetic teacher exactly representable by the Keyless route.
        v.copy_(k)
        student.q_proj.weight.copy_(q)
        student.v_proj.weight.copy_(v)
        student.query_route.reset_identity()
        student.q_norm.weight.copy_(teacher.q_norm.weight)
        student.route_norm.weight.copy_(teacher.k_norm.weight)
        student.out_proj.weight.copy_(teacher.out_proj.weight)
    return student


def _dense_reference(teacher, student, record, query_rows):
    hidden = record.attention_input
    qkv = teacher.qkv_proj(hidden)
    inner = teacher.heads * teacher.head_dim
    tq, tk, tv = qkv.split(inner, dim=-1)
    tq = tq.view(hidden.shape[0], teacher.heads, teacher.head_dim)
    tk = tk.view_as(tq)
    tv = tv.view_as(tq)
    tq = normalized_positioned(tq, teacher.q_norm.weight, teacher.q_norm.eps, None)
    tk = normalized_positioned(tk, teacher.k_norm.weight, teacher.k_norm.eps, None)

    sq = student.q_proj(hidden).view_as(tq)
    sq = torch.einsum("qhd,hed->qhe", sq, student.query_route.weight)
    sq = normalized_positioned(sq, student.q_norm.weight, student.q_norm.eps, None)
    sv = student.v_proj(hidden).view_as(tq)
    sr = normalized_positioned(sv, student.route_norm.weight, student.route_norm.eps, None)

    ids = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.long)
    direct = streamed_attention_comparison(
        tq[query_rows],
        tk,
        tv,
        sq[query_rows],
        sr,
        sv,
        modality_ids=ids,
        key_chunk_size=hidden.shape[0],
    )
    teacher_flat = direct.teacher_output.reshape(len(query_rows), -1)
    student_flat = direct.student_output.reshape(len(query_rows), -1)
    pre = tensor_error_metrics(teacher_flat, student_flat)
    teacher_post = teacher.out_proj(teacher_flat)
    student_post = student.out_proj(student_flat)
    post = tensor_error_metrics(teacher_post, student_post)
    return direct, pre, post


def test_deterministic_query_rows_cover_each_segment_and_stay_bounded() -> None:
    segments = ((0, 2, "text"), (2, 5, "audio"), (5, 9, "video"))
    rows = deterministic_query_rows(9, segments, max_query_rows=5)
    assert len(rows) <= 5
    assert all(0 <= row < 9 for row in rows)
    for start, stop, _ in segments:
        assert any(start <= row < stop for row in rows)


def test_bounded_attention_diagnostic_matches_dense_streamed_oracle() -> None:
    torch.manual_seed(123)
    hidden, heads, head_dim = 5, 2, 3
    teacher = _NativeAttention(hidden, heads, head_dim)
    with torch.no_grad():
        teacher.qkv_proj.weight.normal_(0.0, 0.2)
        teacher.out_proj.weight.normal_(0.0, 0.2)
        teacher.q_norm.weight.uniform_(0.8, 1.2)
        teacher.k_norm.weight.uniform_(0.8, 1.2)
    student = _student_from_teacher(teacher)
    activations = torch.randn(6, hidden)
    record = _capture(
        activations,
        segments=[[0, 2, "text"], [2, 5, "audio"], [5, 6, "video"]],
    )

    result = compare_captured_native_keyless_attention(
        teacher,
        student,
        record,
        max_query_rows=4,
        head_chunk_size=1,
        key_chunk_size=2,
    )
    query_rows = torch.tensor(result.sampled_query_rows, dtype=torch.long)
    direct, pre, post = _dense_reference(teacher, student, record, query_rows)

    torch.testing.assert_close(
        torch.tensor(result.mean_centered_logit_nrmse),
        direct.centered_logit_nrmse.mean(),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        torch.tensor(result.mean_teacher_to_student_softmax_kl),
        direct.softmax_kl_teacher_student.mean(),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(torch.tensor(result.pre_out_normalized_rmse), pre[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.tensor(result.pre_out_cosine), pre[1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.tensor(result.post_out_normalized_rmse), post[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.tensor(result.post_out_cosine), post[1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        torch.tensor(result.teacher_modality_mass),
        direct.teacher_modality_mass.mean(dim=(0, 1)),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        torch.tensor(result.student_modality_mass),
        direct.student_modality_mass.mean(dim=(0, 1)),
        atol=1e-6,
        rtol=1e-6,
    )
    assert result.modality_kinds == ("text", "audio", "video")


def test_identical_teacher_student_attention_has_zero_error_and_unit_cosine() -> None:
    torch.manual_seed(456)
    teacher = _NativeAttention(4, 2, 2)
    with torch.no_grad():
        teacher.qkv_proj.weight.normal_()
        teacher.out_proj.weight.normal_()
    student = _student_from_teacher(teacher)
    record = _capture(
        torch.randn(5, 4),
        segments=[[0, 2, "text"], [2, 5, "video"]],
    )
    result = compare_captured_native_keyless_attention(
        teacher,
        student,
        record,
        max_query_rows=5,
        head_chunk_size=1,
        key_chunk_size=2,
    )
    assert result.mean_centered_logit_nrmse < 1e-6
    assert abs(result.mean_teacher_to_student_softmax_kl) < 1e-6
    assert result.pre_out_normalized_rmse < 1e-6
    assert abs(result.pre_out_cosine - 1.0) < 1e-6
    assert result.post_out_normalized_rmse < 1e-6
    assert abs(result.post_out_cosine - 1.0) < 1e-6
