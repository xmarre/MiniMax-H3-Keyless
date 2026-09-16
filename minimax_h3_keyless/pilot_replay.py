from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn

from .activation_capture import CapturedPilotCase
from .pilot import PilotCase, _run_block_capture_attention


@dataclass(frozen=True)
class CapturedReplayReport:
    block_index: int
    rows: int
    attention_input_max_abs_error: float
    attention_input_mean_abs_error: float
    block_output_finite: bool
    attention_output_finite: bool


def _move_nested_tensors(value: Any, device: torch.device, non_blocking: bool) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, tuple):
        return tuple(_move_nested_tensors(v, device, non_blocking) for v in value)
    if isinstance(value, list):
        return [_move_nested_tensors(v, device, non_blocking) for v in value]
    if isinstance(value, dict):
        return {k: _move_nested_tensors(v, device, non_blocking) for k, v in value.items()}
    return value


def pilot_case_to_device(
    case: PilotCase,
    device: str | torch.device,
    *,
    non_blocking: bool = False,
) -> PilotCase:
    """Move replay tensors to one device without changing dtype or provenance metadata.

    Captured Stage-A activations are deliberately stored on CPU. This function is the
    explicit boundary that rematerializes a case on the teacher/student block device.
    Only tensor-bearing execution fields are moved; ``context`` remains audit metadata.
    """
    device = torch.device(device)
    return replace(
        case,
        x=case.x.to(device=device, non_blocking=non_blocking),
        t_emb=case.t_emb.to(device=device, non_blocking=non_blocking),
        mod_segments=_move_nested_tensors(case.mod_segments, device, non_blocking),
        rope_freqs=(
            None
            if case.rope_freqs is None
            else case.rope_freqs.to(device=device, non_blocking=non_blocking)
        ),
        transformer_options=_move_nested_tensors(
            dict(case.transformer_options), device, non_blocking
        ),
        position_ids=(
            None
            if case.position_ids is None
            else case.position_ids.to(device=device, non_blocking=non_blocking)
        ),
    )


def verify_captured_pilot_replay(
    teacher_block: nn.Module,
    captured: CapturedPilotCase,
    *,
    device: str | torch.device,
    same_input_atol: float = 0.0,
    same_input_rtol: float = 0.0,
) -> CapturedReplayReport:
    """Replay one captured case and prove its post-AdaLN execution point is reproducible.

    This gate catches capture/replay drift before training. It intentionally checks the
    actual post-AdaLN attention input observed in the live full-model forward, rather
    than reconstructing modulation from metadata outside the block.
    """
    case = pilot_case_to_device(captured.case, device)
    block_output, replay_attention_input, attention_output = _run_block_capture_attention(
        teacher_block, case
    )
    expected = captured.attention_input.to(device=replay_attention_input.device)
    if replay_attention_input.shape != expected.shape:
        raise RuntimeError(
            "captured/replayed post-AdaLN attention input shape differs: "
            f"captured={tuple(expected.shape)}, replay={tuple(replay_attention_input.shape)}"
        )
    torch.testing.assert_close(
        replay_attention_input.detach(),
        expected,
        atol=float(same_input_atol),
        rtol=float(same_input_rtol),
        msg=(
            f"live/replayed post-AdaLN attention input diverged for "
            f"pilot block {captured.block_index}"
        ),
    )
    if not torch.isfinite(block_output).all():
        raise RuntimeError(f"replayed pilot block {captured.block_index} produced non-finite output")
    if not torch.isfinite(attention_output).all():
        raise RuntimeError(
            f"replayed pilot block {captured.block_index} produced non-finite attention output"
        )
    error = (replay_attention_input.detach().float() - expected.float()).abs()
    return CapturedReplayReport(
        block_index=captured.block_index,
        rows=int(case.x.shape[0]),
        attention_input_max_abs_error=float(error.max().item()) if error.numel() else 0.0,
        attention_input_mean_abs_error=float(error.mean().item()) if error.numel() else 0.0,
        block_output_finite=True,
        attention_output_finite=True,
    )
