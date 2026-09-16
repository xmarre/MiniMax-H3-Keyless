from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from minimax_h3_keyless.attention import KeylessAttentionTrain
from minimax_h3_keyless.contracts import PROVIDER_KEY
from minimax_h3_keyless.initialization import initialize_training_attention_from_native
from minimax_h3_keyless.ops import normalized_positioned, torch_sdpa_attention
from minimax_h3_keyless.pilot import (
    PilotCase,
    PilotLossWeights,
    build_training_student_block,
    pilot_loss,
    pilot_train_step,
    set_pilot_block_stage,
)


class TinyNativeAttention(nn.Module):
    def __init__(self, hidden=4, heads=2, head_dim=2, dtype=torch.float32):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, 3 * inner, bias=False, dtype=dtype)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-5, dtype=dtype)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-5, dtype=dtype)
        self.out_proj = nn.Linear(inner, hidden, bias=False, dtype=dtype)
        self.to_gate_compress = None

    def forward(self, x, rope_freqs=None, transformer_options=None):
        s = x.shape[0]
        inner = self.heads * self.head_dim
        q, k, v = self.qkv_proj(x).split(inner, dim=-1)
        q = q.view(s, self.heads, self.head_dim)
        k = k.view(s, self.heads, self.head_dim)
        v = v.view(s, self.heads, self.head_dim)
        q = normalized_positioned(q, self.q_norm.weight, self.q_norm.eps, rope_freqs)
        k = normalized_positioned(k, self.k_norm.weight, self.k_norm.eps, rope_freqs)
        out = torch_sdpa_attention(
            q, k, v, scale=self.head_dim ** -0.5
        ).reshape(s, inner)
        return self.out_proj(out)


class TinyBlock(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.norm1 = nn.RMSNorm(4, eps=1e-5, dtype=dtype)
        self.attn = TinyNativeAttention(dtype=dtype)
        self.mlp = nn.Linear(4, 4, bias=False, dtype=dtype)
        self.adaln_marker = nn.Parameter(torch.ones(1, dtype=dtype))

    def forward(
        self,
        x,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options=None,
    ):
        # Deliberately mutate the residual input like the live H3 block does.
        h = self.norm1(x)
        x.add_(self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options))
        return x + 0.01 * self.mlp(x) * self.adaln_marker


def _make_pair() -> tuple[TinyBlock, TinyBlock]:
    torch.manual_seed(101)
    teacher = TinyBlock()
    student = copy.deepcopy(teacher)
    native = teacher.attn
    student_attention = KeylessAttentionTrain(4, 2, 2, 1e-5, block_index=0, dtype=torch.float32)
    initialize_training_attention_from_native(
        student_attention,
        qkv_weight=native.qkv_proj.weight,
        q_norm_weight=native.q_norm.weight,
        k_norm_weight=native.k_norm.weight,
        out_proj_weight=native.out_proj.weight,
    )
    student.attn = student_attention
    set_pilot_block_stage(student, "route")
    return teacher, student


def _case() -> PilotCase:
    torch.manual_seed(102)
    return PilotCase(
        x=torch.randn(6, 4),
        t_emb=torch.randn(3, 2),
        mod_segments=((0, 2, 0), (2, 4, 1), (4, 6, 2)),
        rope_freqs=None,
        case_id="tiny",
        sigma=0.5,
        modality_label="mixed",
    )


def test_build_student_block_copies_bf16_teacher_and_freezes_nonattention() -> None:
    torch.manual_seed(103)
    teacher = TinyBlock(dtype=torch.bfloat16)
    student, report = build_training_student_block(teacher, block_index=25)
    assert isinstance(student.attn, KeylessAttentionTrain)
    assert report.mode == "identity"
    inner = teacher.attn.heads * teacher.attn.head_dim
    torch.testing.assert_close(student.attn.q_proj.weight, teacher.attn.qkv_proj.weight[:inner])
    torch.testing.assert_close(student.attn.v_proj.weight, teacher.attn.qkv_proj.weight[2 * inner:])
    trainable = {name for name, p in student.named_parameters() if p.requires_grad}
    assert trainable == {"attn.query_route.weight", "attn.route_norm.weight"}
    assert student.adaln_marker.requires_grad is False
    assert student.mlp.weight.requires_grad is False


def test_block_stage_never_unfreezes_copied_nonattention_weights() -> None:
    _, student = _make_pair()
    for stage in ("route", "query", "value", "norm_out"):
        enabled = set_pilot_block_stage(student, stage)
        actual = {name for name, p in student.named_parameters() if p.requires_grad}
        assert actual == set(enabled)
        assert not student.mlp.weight.requires_grad
        assert not student.adaln_marker.requires_grad


def test_pilot_loss_preserves_case_input_despite_inplace_block_forward() -> None:
    teacher, student = _make_pair()
    case = _case()
    before = case.x.clone()
    losses = pilot_loss(teacher, student, case)
    torch.testing.assert_close(case.x, before)
    assert torch.isfinite(losses.total)
    assert losses.attention_normalized_mse > 0
    assert losses.block_normalized_mse > 0


def test_pilot_train_step_has_finite_connected_stage_gradients_and_updates_route() -> None:
    teacher, student = _make_pair()
    case = _case()
    before = student.attn.query_route.weight.detach().clone()
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=1e-3,
    )
    report = pilot_train_step(
        teacher,
        student,
        case,
        optimizer,
        weights=PilotLossWeights(attention_output=1.0, block_output=1.0),
        max_grad_norm=10.0,
    )
    assert report.total > 0
    assert report.trainable_parameters == student.attn.query_route.weight.numel() + student.attn.route_norm.weight.numel()
    assert report.gradient_l2_norm > 0
    assert not torch.equal(before, student.attn.query_route.weight.detach())


def test_pilot_rejects_external_keyless_provider_before_native_parity() -> None:
    teacher, student = _make_pair()
    case = _case()
    case = PilotCase(
        x=case.x,
        t_emb=case.t_emb,
        mod_segments=case.mod_segments,
        rope_freqs=None,
        transformer_options={PROVIDER_KEY: object()},
        case_id="provider-contaminated",
    )
    with pytest.raises(RuntimeError, match="must not install a Keyless provider"):
        pilot_loss(teacher, student, case)


def test_pilot_loss_weights_reject_empty_objective() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        PilotLossWeights(attention_output=0.0, block_output=0.0)
