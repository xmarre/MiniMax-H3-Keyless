from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import torch.nn as nn

from .activation_capture import PilotActivationCapture
from .attention import KeylessAttentionDeploy, KeylessAttentionTrain
from .contracts import CORE_BLOCKS
from .pilot_campaign import _require_sha256, canonical_json_sha256


PROGRESSIVE_PREFIX_CONTEXT_KEY = "minimax_h3_keyless_progressive_prefix_v1"
_PROGRESSIVE_STAGES = ("route", "query", "value", "norm_out")


@dataclass(frozen=True)
class ProgressiveAcceptedBlock:
    """Immutable evidence identity for one accepted progressive block conversion."""

    block_index: int
    final_stage: str
    checkpoint_sha256: str
    result_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.block_index, bool) or not isinstance(self.block_index, int):
            raise ValueError("progressive block_index must be an integer")
        if not 0 <= self.block_index < CORE_BLOCKS:
            raise ValueError(
                f"progressive block_index must be within [0,{CORE_BLOCKS}), got {self.block_index}"
            )
        if self.final_stage not in _PROGRESSIVE_STAGES:
            raise ValueError(f"unsupported progressive final_stage: {self.final_stage!r}")
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _require_sha256("progressive block checkpoint SHA-256", self.checkpoint_sha256),
        )
        object.__setattr__(
            self,
            "result_sha256",
            _require_sha256("progressive block result SHA-256", self.result_sha256),
        )


@dataclass(frozen=True)
class ProgressivePrefix:
    """Hash-chained identity of the accepted early->late Keyless core prefix.

    A prefix is not a claim that a block is good merely because it is present in memory.
    Every accepted block is identified by immutable checkpoint/result hashes and must be
    added strictly in block order. The Stage-A campaign, fixed dataset and fixed gate
    identities remain part of every prefix digest so live-input captures cannot be reused
    silently across different empirical definitions.
    """

    sweep_id: str
    code_commit: str
    stage_a_campaign_sha256: str
    dataset_manifest_sha256: str
    gate_manifest_sha256: str
    accepted: tuple[ProgressiveAcceptedBlock, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.sweep_id, str) or not self.sweep_id.strip():
            raise ValueError("progressive sweep_id must be non-empty")
        if not isinstance(self.code_commit, str) or not self.code_commit.strip():
            raise ValueError("progressive code_commit must be non-empty")
        for name in (
            "stage_a_campaign_sha256",
            "dataset_manifest_sha256",
            "gate_manifest_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(f"progressive {name}", getattr(self, name)),
            )
        accepted = tuple(self.accepted)
        if len(accepted) > CORE_BLOCKS:
            raise ValueError(f"progressive prefix cannot exceed {CORE_BLOCKS} core blocks")
        expected = tuple(range(len(accepted)))
        actual = tuple(row.block_index for row in accepted)
        if actual != expected:
            raise ValueError(
                "progressive accepted blocks must be the exact contiguous early->late prefix; "
                f"expected={expected}, actual={actual}"
            )
        object.__setattr__(self, "accepted", accepted)

    @property
    def accepted_blocks(self) -> tuple[int, ...]:
        return tuple(row.block_index for row in self.accepted)

    @property
    def next_block(self) -> int | None:
        count = len(self.accepted)
        return None if count == CORE_BLOCKS else count

    @property
    def complete(self) -> bool:
        return len(self.accepted) == CORE_BLOCKS

    def identity_payload(self) -> dict[str, Any]:
        return {
            "api": 1,
            "sweep_id": self.sweep_id,
            "code_commit": self.code_commit.lower(),
            "stage_a_campaign_sha256": self.stage_a_campaign_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "gate_manifest_sha256": self.gate_manifest_sha256,
            "accepted": [asdict(row) for row in self.accepted],
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_json_sha256(self.identity_payload())

    def advance(
        self,
        *,
        block_index: int,
        final_stage: str,
        checkpoint_sha256: str,
        result_sha256: str,
    ) -> "ProgressivePrefix":
        next_block = self.next_block
        if next_block is None:
            raise RuntimeError("progressive prefix already contains all core blocks")
        if block_index != next_block:
            raise ValueError(
                "progressive acceptance must advance exactly one early->late block: "
                f"expected block {next_block}, got {block_index}"
            )
        row = ProgressiveAcceptedBlock(
            block_index=block_index,
            final_stage=final_stage,
            checkpoint_sha256=checkpoint_sha256,
            result_sha256=result_sha256,
        )
        return replace(self, accepted=(*self.accepted, row))

    def capture_context(self) -> dict[str, Any]:
        return {
            "api": 1,
            "prefix_identity_sha256": self.identity_sha256,
            "accepted_blocks": list(self.accepted_blocks),
            "next_block": self.next_block,
            "stage_a_campaign_sha256": self.stage_a_campaign_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "gate_manifest_sha256": self.gate_manifest_sha256,
        }


def _is_keyless_attention(value: Any) -> bool:
    return isinstance(value, (KeylessAttentionTrain, KeylessAttentionDeploy))


def _is_native_qkv_attention(value: Any) -> bool:
    if value is None or _is_keyless_attention(value):
        return False
    required = ("qkv_proj", "q_norm", "k_norm", "out_proj")
    if any(getattr(value, name, None) is None for name in required):
        return False
    if getattr(value, "qv_proj", None) is not None:
        return False
    if getattr(value, "query_route", None) is not None:
        return False
    return True


def validate_progressive_model_prefix(model: nn.Module, prefix: ProgressivePrefix) -> None:
    """Prove that the live model object implements exactly the declared accepted prefix."""

    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) != CORE_BLOCKS:
        raise RuntimeError(
            f"progressive sweep requires exactly {CORE_BLOCKS} core blocks on the live model"
        )
    boundary = len(prefix.accepted)
    for index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        if index < boundary:
            if not _is_keyless_attention(attention):
                raise RuntimeError(
                    f"accepted progressive block {index} is not a Keyless attention module"
                )
            if int(getattr(attention, "block_index", -1)) != index:
                raise RuntimeError(
                    f"accepted progressive block {index} carries the wrong Keyless block_index"
                )
        else:
            if not _is_native_qkv_attention(attention):
                raise RuntimeError(
                    f"unaccepted progressive block {index} is not the native QKV teacher block"
                )


def progressive_capture_session(
    model: nn.Module,
    prefix: ProgressivePrefix,
    *,
    case_id: str,
    sigma: float | None,
    modality_label: str | None,
    max_capture_bytes: int,
    context: dict[str, Any] | None = None,
) -> PilotActivationCapture:
    """Capture the next live student-input block under the exact accepted prefix.

    The returned context manager observes exactly one target block. The target and all
    later blocks must still be native QKV, while every earlier block must already be a
    declared Keyless conversion. The prefix digest is embedded in capture audit context;
    downstream registries must bind to it before using the capture as progressive training
    evidence.
    """

    validate_progressive_model_prefix(model, prefix)
    target = prefix.next_block
    if target is None:
        raise RuntimeError("progressive prefix is complete; there is no next block to capture")
    user_context = dict(context or {})
    if PROGRESSIVE_PREFIX_CONTEXT_KEY in user_context:
        raise ValueError(
            f"caller context may not override {PROGRESSIVE_PREFIX_CONTEXT_KEY!r}"
        )
    user_context[PROGRESSIVE_PREFIX_CONTEXT_KEY] = prefix.capture_context()
    return PilotActivationCapture(
        model,
        case_id=case_id,
        sigma=sigma,
        modality_label=modality_label,
        max_capture_bytes=max_capture_bytes,
        block_indices=(target,),
        context=user_context,
        require_plain_native=True,
    )
