from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from minimax_h3_keyless.activation_capture import CapturedPilotCase
from minimax_h3_keyless.attention import KeylessAttentionTrain
from minimax_h3_keyless.initialization import initialize_training_attention_from_native
from minimax_h3_keyless.ops import normalized_positioned, torch_sdpa_attention
from minimax_h3_keyless.pilot import PilotCase, set_pilot_block_stage
from minimax_h3_keyless.pilot_campaign import GATE_SCHEMA, PILOT_LS_LAMBDAS, validate_pilot_gate_manifest
from minimax_h3_keyless.pilot_runner import StageATrainStage
from minimax_h3_keyless.progressive import PROGRESSIVE_PREFIX_CONTEXT_KEY, ProgressivePrefix
from minimax_h3_keyless.progressive_capture_set import ProgressiveBlockCaptureSet
from minimax_h3_keyless.progressive_runner import run_progressive_block_training


LIVE_CAPTURE_KEY = "minimax_h3_keyless_live_capture_v1"


class TinyNativeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = 2
        self.head_dim = 2
        self.qkv_proj = nn.Linear(4, 12, bias=False)
        self.q_norm = nn.RMSNorm(2, eps=1e-5)
        self.k_norm = nn.RMSNorm(2, eps=1e-5)
        self.out_proj = nn.Linear(4, 4, bias=False)
        self.to_gate_compress = None

    def forward(self, x, rope_freqs=None, transformer_options=None):
        q, k, v = self.qkv_proj(x).split(4, dim=-1)
        q = normalized_positioned(q.view(x.shape[0], 2, 2), self.q_norm.weight, 1e-5, rope_freqs)
        k = normalized_positioned(k.view(x.shape[0], 2, 2), self.k_norm.weight, 1e-5, rope_freqs)
        v = v.view(x.shape[0], 2, 2)
        out = torch_sdpa_attention(q, k, v, scale=2 ** -0.5).reshape(x.shape[0], 4)
        return self.out_proj(out)


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(4, eps=1e-5)
        self.attn = TinyNativeAttention()
        self.mlp = nn.Linear(4, 4, bias=False)

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
        h = self.norm1(x)
        x = x + self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
        return x + 0.01 * self.mlp(x)


def _builder(teacher, block_index, route_mode, lambda_relative):
    assert route_mode == "identity"
    assert lambda_relative == 0.0
    student = copy.deepcopy(teacher)
    native = teacher.attn
    attention = KeylessAttentionTrain(
        4,
        2,
        2,
        1e-5,
        block_index=block_index,
        dtype=torch.float32,
    )
    report = initialize_training_attention_from_native(
        attention,
        qkv_weight=native.qkv_proj.weight,
        q_norm_weight=native.q_norm.weight,
        k_norm_weight=native.k_norm.weight,
        out_proj_weight=native.out_proj.weight,
        route_mode="identity",
        lambda_relative=0.0,
    )
    student.attn = attention
    set_pilot_block_stage(student, "route")
    return student, report


def _gate_manifest(max_attention: float = 1e9):
    return {
        "schema": GATE_SCHEMA,
        "thresholds": {
            "stage_a_min_case_total_improvement_fraction": 0.0,
            "stage_a_min_mean_total_relative_improvement": 0.0,
            "stage_a_max_mean_attention_normalized_mse": max_attention,
            "stage_a_max_mean_block_normalized_mse": 1e9,
            "stage_a_max_worst_attention_normalized_mse": 1e9,
            "stage_a_max_worst_block_normalized_mse": 1e9,
            "stage_a_min_mean_attention_cosine": -1.0,
            "stage_a_min_mean_block_cosine": -1.0,
            "stage_a_minimum_attention_cosine": -1.0,
            "stage_a_minimum_block_cosine": -1.0,
            "stage_a_max_modality_nmse_ratio_to_best_baseline": 1e9,
            "stage_a_max_gradient_l2_norm": 1e9,
        },
        "calibration_evidence": {"fixture": "progressive-runner"},
    }


def _prefix(gate_manifest):
    return ProgressivePrefix(
        sweep_id="sweep-001",
        code_commit="deadbeef",
        stage_a_campaign_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        gate_manifest_sha256=validate_pilot_gate_manifest(gate_manifest),
    )


def _record(
    teacher: TinyBlock,
    prefix: ProgressivePrefix,
    *,
    split: str,
    n: int,
    sigma: float,
) -> CapturedPilotCase:
    torch.manual_seed(700 + n + (0 if split == "train" else 100))
    x = torch.randn(5, 4)
    case_id = f"{split}-{n}::sigma={sigma}"
    case = PilotCase(
        x=x,
        t_emb=torch.zeros(1, 1),
        mod_segments=((0, 5, 0),),
        rope_freqs=None,
        transformer_options={},
        case_id=case_id,
        sigma=sigma,
        modality_label="video",
        context={
            "progressive_split": split,
            PROGRESSIVE_PREFIX_CONTEXT_KEY: prefix.capture_context(),
            LIVE_CAPTURE_KEY: {
                "block_index": 0,
                "layout": {"seq_len": 5, "segments": [[0, 5, "video"]]},
            },
        },
    )
    observed = {}

    def hook(module, args, kwargs):
        observed["attention_input"] = args[0].detach().clone()

    handle = teacher.attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        teacher(
            case.x.detach().clone(),
            case.t_emb,
            case.mod_segments,
            case.rope_freqs,
            transformer_options={},
        )
    finally:
        handle.remove()
    return CapturedPilotCase(
        block_index=0,
        case=case,
        attention_input=observed["attention_input"],
        captured_bytes=512,
    )


def _capture_set(teacher: TinyBlock, prefix: ProgressivePrefix) -> ProgressiveBlockCaptureSet:
    train = tuple(
        _record(teacher, prefix, split="train", n=i, sigma=sigma)
        for i, sigma in enumerate((0.2, 0.8))
    )
    holdout = tuple(
        _record(teacher, prefix, split="holdout", n=i, sigma=sigma)
        for i, sigma in enumerate((0.2, 0.8))
    )
    return ProgressiveBlockCaptureSet(
        target_block=0,
        prefix_identity_sha256=prefix.identity_sha256,
        stage_a_campaign_sha256=prefix.stage_a_campaign_sha256,
        dataset_manifest_sha256=prefix.dataset_manifest_sha256,
        gate_manifest_sha256=prefix.gate_manifest_sha256,
        code_commit=prefix.code_commit,
        comfy_commit="comfy-fixture",
        execution_descriptor="progressive-fixture",
        train=train,
        holdout=holdout,
        artifact_refs=(),
    )


def test_progressive_runner_replays_same_input_trains_without_accepting_prefix() -> None:
    torch.manual_seed(701)
    teacher = TinyBlock()
    gate_manifest = _gate_manifest()
    prefix = _prefix(gate_manifest)
    captures = _capture_set(teacher, prefix)

    result = run_progressive_block_training(
        teacher,
        captures,
        prefix,
        device="cpu",
        gate_manifest=gate_manifest,
        train_plan=(StageATrainStage("route", 1, 1e-3, max_grad_norm=100.0),),
        student_builder=_builder,
        require_bf16_teacher=False,
    )

    assert result.block_index == 0
    assert result.prefix_identity_sha256 == prefix.identity_sha256
    assert result.selection_split == "train_complete_cases"
    assert prefix.accepted_blocks == ()
    assert prefix.next_block == 0
    assert isinstance(teacher.attn, TinyNativeAttention)
    assert isinstance(result.student_block.attn, KeylessAttentionTrain)
    assert result.student_block.attn.block_index == 0
    assert len(result.replay_reports) == 4
    assert all(row.attention_input_max_abs_error == 0.0 for row in result.replay_reports)
    assert len(result.initialization_evaluations) == 4
    assert len(result.candidate_attention_diagnostics) == 2
    assert len(result.training_events) == 2
    assert result.final_stage == "route"

    train_ids = tuple(record.case.case_id for record in captures.train)
    holdout_ids = tuple(record.case.case_id for record in captures.holdout)
    assert set(train_ids).isdisjoint(holdout_ids)
    for row in result.initialization_evaluations:
        assert tuple(case.case_id for case in row.metrics.cases) == train_ids
        assert tuple(diag.case_id for diag in row.attention_diagnostics) == train_ids
    assert tuple(case.case_id for case in result.identity_baseline.cases) == holdout_ids
    assert tuple(case.case_id for case in result.least_squares_baseline.cases) == holdout_ids
    assert tuple(case.case_id for case in result.candidate.cases) == holdout_ids
    assert tuple(diag.case_id for diag in result.candidate_attention_diagnostics) == holdout_ids
    assert {event.case_id for event in result.training_events} == set(train_ids)
    assert result.least_squares_baseline_lambda_relative in PILOT_LS_LAMBDAS

    assert any(parameter.requires_grad for parameter in result.student_block.parameters())
    optimized = {
        id(parameter)
        for group in result.optimizer.param_groups
        for parameter in group["params"]
    }
    trainable = {
        id(parameter)
        for parameter in result.student_block.parameters()
        if parameter.requires_grad
    }
    assert optimized == trainable


def test_progressive_runner_rejects_gate_manifest_substitution_before_training() -> None:
    teacher = TinyBlock()
    fixed = _gate_manifest()
    prefix = _prefix(fixed)
    captures = _capture_set(teacher, prefix)
    substituted = _gate_manifest(max_attention=123.0)

    with pytest.raises(RuntimeError, match="supplied progressive gate manifest"):
        run_progressive_block_training(
            teacher,
            captures,
            prefix,
            device="cpu",
            gate_manifest=substituted,
            train_plan=(StageATrainStage("route", 1, 1e-3),),
            student_builder=_builder,
            require_bf16_teacher=False,
        )


def test_progressive_runner_rejects_forged_record_prefix_binding() -> None:
    teacher = TinyBlock()
    gate_manifest = _gate_manifest()
    prefix = _prefix(gate_manifest)
    captures = _capture_set(teacher, prefix)
    bad = captures.train[0]
    bad_context = dict(bad.case.context)
    bad_context[PROGRESSIVE_PREFIX_CONTEXT_KEY] = {
        **bad_context[PROGRESSIVE_PREFIX_CONTEXT_KEY],
        "prefix_identity_sha256": "f" * 64,
    }
    forged = ProgressiveBlockCaptureSet(
        **{
            **captures.__dict__,
            "train": (
                CapturedPilotCase(
                    block_index=bad.block_index,
                    case=PilotCase(**{**bad.case.__dict__, "context": bad_context}),
                    attention_input=bad.attention_input,
                    captured_bytes=bad.captured_bytes,
                ),
                *captures.train[1:],
            ),
        }
    )

    with pytest.raises(RuntimeError, match="prefix context differs"):
        run_progressive_block_training(
            teacher,
            forged,
            prefix,
            device="cpu",
            gate_manifest=gate_manifest,
            train_plan=(StageATrainStage("route", 1, 1e-3),),
            student_builder=_builder,
            require_bf16_teacher=False,
        )


def test_progressive_runner_production_default_rejects_non_bf16_teacher() -> None:
    teacher = TinyBlock()
    gate_manifest = _gate_manifest()
    prefix = _prefix(gate_manifest)
    captures = _capture_set(teacher, prefix)

    with pytest.raises(RuntimeError, match="must be BF16"):
        run_progressive_block_training(
            teacher,
            captures,
            prefix,
            device="cpu",
            gate_manifest=gate_manifest,
            train_plan=(StageATrainStage("route", 1, 1e-3),),
            student_builder=_builder,
        )
