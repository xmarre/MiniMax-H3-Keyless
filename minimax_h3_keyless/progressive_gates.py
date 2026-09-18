from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from .contracts import CORE_BLOCKS
from .pilot_campaign import (
    PilotAggregateMetrics,
    PilotTrainingEvent,
    validate_pilot_gate_manifest,
)
from .pilot_gates import StageABlockGateResult, StageAGatePolicy, evaluate_stage_a_block_gate


ProgressiveBlockGateResult = StageABlockGateResult


@dataclass(frozen=True)
class ProgressiveExecutionPolicy:
    """Predeclared Stage-B implementation tolerances from the fixed gate manifest."""

    fold_atol: float
    fold_rtol: float


def progressive_execution_policy_from_gate_manifest(
    manifest: Mapping[str, object],
) -> ProgressiveExecutionPolicy:
    """Load fold/export tolerances without allowing post-result CLI overrides.

    Stage B reuses the Stage-A numerical fit gate, but folding q/R into deployable Q is a
    separate finite-precision implementation gate. Its tolerances therefore live in the
    same immutable gate manifest that is already hash-bound into the progressive prefix.
    """

    validate_pilot_gate_manifest(manifest)
    thresholds = manifest["thresholds"]
    assert isinstance(thresholds, dict)
    required = ("progressive_fold_atol", "progressive_fold_rtol")
    missing = [name for name in required if name not in thresholds]
    if missing:
        raise ValueError(
            f"pilot gate manifest is missing progressive execution thresholds: {missing}"
        )

    values: dict[str, float] = {}
    for name in required:
        raw = thresholds[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{name} must be a finite non-negative number")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")
        values[name] = value
    return ProgressiveExecutionPolicy(
        fold_atol=values["progressive_fold_atol"],
        fold_rtol=values["progressive_fold_rtol"],
    )


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
