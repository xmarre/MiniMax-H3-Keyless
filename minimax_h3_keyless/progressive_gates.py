from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .contracts import CORE_BLOCKS
from .pilot_campaign import PilotAggregateMetrics, PilotTrainingEvent
from .pilot_gates import StageABlockGateResult, StageAGatePolicy, evaluate_stage_a_block_gate


ProgressiveBlockGateResult = StageABlockGateResult


def evaluate_progressive_block_gate(
    *,
    block_index: int,
    candidate: PilotAggregateMetrics,
    identity_baseline: PilotAggregateMetrics,
    least_squares_baseline: PilotAggregateMetrics,
    training_events: Sequence[PilotTrainingEvent],
    policy: StageAGatePolicy,
) -> ProgressiveBlockGateResult:
    """Apply the frozen pilot gate policy to any core50 progressive target block.

    Stage B is specified to inherit the pilot-defined numerical gate rather than invent
    per-layer thresholds. ``evaluate_stage_a_block_gate`` contains the policy arithmetic
    and is deliberately reused unchanged here; its block-index restriction is a Stage-A
    campaign topology check, not part of the numerical policy. The surrogate index is
    discarded and the real progressive block identity is restored in the immutable
    result.
    """

    if isinstance(block_index, bool) or not isinstance(block_index, int):
        raise ValueError("progressive gate block_index must be an integer")
    if not 0 <= block_index < CORE_BLOCKS:
        raise ValueError(
            f"progressive gate block_index must be within [0,{CORE_BLOCKS}), got {block_index}"
        )
    evaluated = evaluate_stage_a_block_gate(
        block_index=0,
        candidate=candidate,
        identity_baseline=identity_baseline,
        least_squares_baseline=least_squares_baseline,
        training_events=training_events,
        policy=policy,
    )
    return replace(evaluated, block_index=block_index)
