from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import KeylessAttentionTrain
from .contracts import PROVIDER_KEY
from .initialization import (
    PilotTrainStage,
    RouteInitMode,
    RouteInitializationReport,
    initialize_training_attention_from_native,
    set_pilot_trainable_stage,
)


@dataclass(frozen=True)
class PilotCase:
    """One same-input block-local distillation case.

    ``x`` is the input to the DiT block. The post-AdaLN attention input is captured
    from the actual block execution, so the pilot does not reimplement H3's modulation
    path. ``position_ids`` and ``context`` are retained for case provenance/stratification
    but do not replace the live ``rope_freqs`` supplied to the block.
    """

    x: torch.Tensor
    t_emb: torch.Tensor
    mod_segments: Any
    rope_freqs: torch.Tensor | None
    transformer_options: Mapping[str, Any] = field(default_factory=dict)
    case_id: str = ""
    sigma: float | None = None
    modality_label: str | None = None
    position_ids: torch.Tensor | None = None
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PilotLossWeights:
    attention_output: float = 1.0
    block_output: float = 1.0
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.attention_output < 0 or self.block_output < 0:
            raise ValueError("pilot loss weights must be non-negative")
        if self.attention_output == 0 and self.block_output == 0:
            raise ValueError("at least one pilot loss weight must be non-zero")
        if self.epsilon <= 0:
            raise ValueError("pilot normalization epsilon must be positive")


@dataclass
class PilotLossTensors:
    total: torch.Tensor
    attention_normalized_mse: torch.Tensor
    block_normalized_mse: torch.Tensor
    attention_cosine: torch.Tensor
    block_cosine: torch.Tensor
    teacher_attention_output: torch.Tensor
    student_attention_output: torch.Tensor
    teacher_block_output: torch.Tensor
    student_block_output: torch.Tensor


@dataclass(frozen=True)
class PilotStepReport:
    total: float
    attention_normalized_mse: float
    block_normalized_mse: float
    attention_cosine: float
    block_cosine: float
    trainable_parameters: int
    gradient_l2_norm: float


def _normalized_mse(student: torch.Tensor, teacher: torch.Tensor, epsilon: float) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError(
            f"teacher/student shapes differ: {tuple(teacher.shape)} vs {tuple(student.shape)}"
        )
    error = (student.float() - teacher.float()).square().mean()
    teacher_scale = teacher.float().square().mean()
    return error / teacher_scale.clamp_min(float(epsilon))


def _mean_row_cosine(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError("teacher/student shapes must match for cosine")
    return F.cosine_similarity(
        teacher.float().reshape(-1, teacher.shape[-1]),
        student.float().reshape(-1, student.shape[-1]),
        dim=-1,
    ).mean()


def _native_attention_facts(native_attention: nn.Module) -> tuple[int, int, int, float, bool]:
    required = ("qkv_proj", "q_norm", "k_norm", "out_proj", "heads", "head_dim")
    missing = [name for name in required if not hasattr(native_attention, name)]
    if missing:
        raise RuntimeError(f"native H3 attention is missing required attributes: {missing}")
    qkv = native_attention.qkv_proj.weight
    heads = int(native_attention.heads)
    head_dim = int(native_attention.head_dim)
    inner = heads * head_dim
    if qkv.ndim != 2 or qkv.shape[0] != 3 * inner:
        raise RuntimeError(
            "native H3 attention qkv projection has incompatible geometry: "
            f"shape={tuple(qkv.shape)}, heads={heads}, head_dim={head_dim}"
        )
    if getattr(qkv, "is_meta", False):
        raise RuntimeError("cannot initialize a pilot from meta-device teacher weights")
    eps = float(native_attention.q_norm.eps)
    gate = getattr(native_attention, "to_gate_compress", None) is not None
    return int(qkv.shape[1]), heads, head_dim, eps, gate


def build_training_student_block(
    native_block: nn.Module,
    *,
    block_index: int,
    route_mode: RouteInitMode = "identity",
    lambda_relative: float = 0.0,
) -> tuple[nn.Module, RouteInitializationReport]:
    """Deep-copy one native H3 block and install an identity-route training student.

    All copied MLP/AdaLN/norm/non-attention parameters are frozen. This function expects
    the canonical BF16 teacher block, not an INT8/quantized teacher. Least-squares route
    initialization is intentionally not constructed here: it depends on immutable
    captured post-AdaLN train activations and is installed by the Stage-A runner after
    the structural student exists. Calling this helper directly with ``least_squares``
    fails closed rather than falling back to a projection-weight approximation.
    """
    if route_mode != "identity" or lambda_relative != 0.0:
        raise ValueError(
            "build_training_student_block only constructs the identity-route structural student; "
            "Stage-A least-squares initialization must come from captured train activations"
        )
    if not hasattr(native_block, "attn"):
        raise RuntimeError("native H3 block has no attention module")
    native_attention = native_block.attn
    hidden, heads, head_dim, eps, gate = _native_attention_facts(native_attention)
    qkv_weight = native_attention.qkv_proj.weight
    if qkv_weight.dtype != torch.bfloat16:
        raise RuntimeError(
            f"pilot teacher attention must be BF16, got {qkv_weight.dtype}"
        )

    student_block = copy.deepcopy(native_block)
    student_attention = KeylessAttentionTrain(
        hidden,
        heads,
        head_dim,
        eps,
        gate_compress=gate,
        block_index=int(block_index),
        dtype=qkv_weight.dtype,
        device=qkv_weight.device,
        operations=None,
    )
    report = initialize_training_attention_from_native(
        student_attention,
        qkv_weight=qkv_weight.detach(),
        q_norm_weight=native_attention.q_norm.weight.detach(),
        k_norm_weight=native_attention.k_norm.weight.detach(),
        out_proj_weight=native_attention.out_proj.weight.detach(),
        gate_compress_weight=(
            None
            if not gate
            else native_attention.to_gate_compress.weight.detach()
        ),
        route_mode="identity",
        lambda_relative=0.0,
    )
    student_block.attn = student_attention
    set_pilot_block_stage(student_block, "route")
    return student_block, report


def set_pilot_block_stage(block: nn.Module, stage: PilotTrainStage) -> tuple[str, ...]:
    """Freeze the entire copied block, then enable only the design-authorized attention stage."""
    if not isinstance(getattr(block, "attn", None), KeylessAttentionTrain):
        raise RuntimeError("pilot block does not contain KeylessAttentionTrain")
    for parameter in block.parameters():
        parameter.requires_grad_(False)
    enabled = set_pilot_trainable_stage(block.attn, stage)
    return tuple(f"attn.{name}" for name in enabled)


def _run_block_capture_attention(
    block: nn.Module,
    case: PilotCase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(module, args, kwargs):
        if not args:
            raise RuntimeError("attention hook observed no positional hidden input")
        hidden = args[0]
        if not torch.is_tensor(hidden):
            raise RuntimeError("attention hidden input is not a tensor")
        captured["input"] = hidden

    def post_hook(module, args, kwargs, output):
        if not torch.is_tensor(output):
            raise RuntimeError("attention output is not a tensor")
        captured["output"] = output

    pre = block.attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    post = block.attn.register_forward_hook(post_hook, with_kwargs=True)
    try:
        options = dict(case.transformer_options)
        if PROVIDER_KEY in options:
            raise RuntimeError(
                "Stage-A native pilot cases must not install a Keyless provider; "
                "establish materialized native parity first"
            )
        # Native H3 block forward mutates the residual stream in place. A private clone is
        # therefore part of the same-input invariant, not an optimization detail.
        x = case.x.detach().clone()
        t_emb = case.t_emb.detach().clone()
        out = block(
            x,
            t_emb,
            case.mod_segments,
            case.rope_freqs,
            transformer_options=options,
        )
    finally:
        pre.remove()
        post.remove()
    if "input" not in captured or "output" not in captured:
        raise RuntimeError("attention hooks did not observe a complete block execution")
    return out, captured["input"], captured["output"]


def pilot_loss(
    teacher_block: nn.Module,
    student_block: nn.Module,
    case: PilotCase,
    *,
    weights: PilotLossWeights = PilotLossWeights(),
    same_input_atol: float = 0.0,
    same_input_rtol: float = 0.0,
) -> PilotLossTensors:
    """Evaluate the Stage-A same-input objective for one bounded case."""
    teacher_training = teacher_block.training
    teacher_block.eval()
    try:
        with torch.no_grad():
            teacher_block_output, teacher_h, teacher_attention_output = _run_block_capture_attention(
                teacher_block, case
            )
    finally:
        teacher_block.train(teacher_training)

    student_block_output, student_h, student_attention_output = _run_block_capture_attention(
        student_block, case
    )
    torch.testing.assert_close(
        student_h.detach(),
        teacher_h.detach(),
        atol=float(same_input_atol),
        rtol=float(same_input_rtol),
        msg="teacher/student post-AdaLN attention inputs diverged",
    )

    attention_nmse = _normalized_mse(
        student_attention_output, teacher_attention_output, weights.epsilon
    )
    block_nmse = _normalized_mse(student_block_output, teacher_block_output, weights.epsilon)
    total = (
        float(weights.attention_output) * attention_nmse
        + float(weights.block_output) * block_nmse
    )
    return PilotLossTensors(
        total=total,
        attention_normalized_mse=attention_nmse,
        block_normalized_mse=block_nmse,
        attention_cosine=_mean_row_cosine(student_attention_output, teacher_attention_output),
        block_cosine=_mean_row_cosine(student_block_output, teacher_block_output),
        teacher_attention_output=teacher_attention_output,
        student_attention_output=student_attention_output,
        teacher_block_output=teacher_block_output,
        student_block_output=student_block_output,
    )


def _trainable_parameters(module: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, p) for name, p in module.named_parameters() if p.requires_grad]


def pilot_train_step(
    teacher_block: nn.Module,
    student_block: nn.Module,
    case: PilotCase,
    optimizer: torch.optim.Optimizer,
    *,
    weights: PilotLossWeights = PilotLossWeights(),
    max_grad_norm: float | None = None,
) -> PilotStepReport:
    """Run one optimizer step and fail closed on disconnected/non-finite gradients."""
    trainable = _trainable_parameters(student_block)
    if not trainable:
        raise RuntimeError("pilot student has no trainable parameters")
    optimizer.zero_grad(set_to_none=True)
    losses = pilot_loss(teacher_block, student_block, case, weights=weights)
    if not torch.isfinite(losses.total):
        raise RuntimeError(f"non-finite pilot loss for case {case.case_id!r}")
    losses.total.backward()

    disconnected = [name for name, p in trainable if p.grad is None]
    nonfinite = [
        name for name, p in trainable
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    if disconnected or nonfinite:
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError(
            "invalid pilot gradients: "
            f"disconnected={disconnected}, nonfinite={nonfinite}"
        )

    parameters = [p for _, p in trainable]
    grad_sq = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    for parameter in parameters:
        grad_sq += parameter.grad.detach().double().square().sum()
    gradient_l2 = float(torch.sqrt(grad_sq).item())
    if max_grad_norm is not None:
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        torch.nn.utils.clip_grad_norm_(parameters, float(max_grad_norm), error_if_nonfinite=True)
    optimizer.step()

    return PilotStepReport(
        total=float(losses.total.detach().item()),
        attention_normalized_mse=float(losses.attention_normalized_mse.detach().item()),
        block_normalized_mse=float(losses.block_normalized_mse.detach().item()),
        attention_cosine=float(losses.attention_cosine.detach().item()),
        block_cosine=float(losses.block_cosine.detach().item()),
        trainable_parameters=sum(p.numel() for p in parameters),
        gradient_l2_norm=gradient_l2,
    )
