from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from .contracts import PROVIDER_KEY
from .pilot import PilotCase
from .pilot_campaign import PILOT_BLOCKS


_CAPTURE_CONTEXT_KEY = "minimax_h3_keyless_live_capture_v1"


@dataclass(frozen=True)
class CapturedPilotCase:
    """One bounded live-model capture for a Stage-A pilot block.

    ``case`` stores the block-entry tensors needed for deterministic replay.
    ``attention_input`` is the actual post-AdaLN input observed by the native
    attention module during the full-model forward. It is retained separately so
    callers can verify that replay reproduces the live execution point exactly.
    ``captured_bytes`` is the cumulative CPU payload held by the capture session
    when this block completed.
    """

    block_index: int
    case: PilotCase
    attention_input: torch.Tensor
    captured_bytes: int


class _CpuCloneBudget:
    def __init__(self, limit_bytes: int) -> None:
        if limit_bytes <= 0:
            raise ValueError("capture byte budget must be positive")
        self.limit_bytes = int(limit_bytes)
        self.used_bytes = 0
        self._shared_cache: dict[int, torch.Tensor] = {}

    def _clone(self, value: torch.Tensor) -> torch.Tensor:
        nbytes = int(value.numel() * value.element_size())
        if self.used_bytes + nbytes > self.limit_bytes:
            raise RuntimeError(
                "Stage-A live capture would exceed the explicit CPU byte budget: "
                f"used={self.used_bytes}, next={nbytes}, limit={self.limit_bytes}"
            )
        clone = value.detach().to(device="cpu", copy=True)
        self.used_bytes += nbytes
        return clone

    def snapshot_tensor(self, value: torch.Tensor | None) -> torch.Tensor | None:
        """Clone an observation even when the source object was seen before.

        H3 mutates its residual stream in place. Object identity therefore cannot be
        used to deduplicate block-entry or post-AdaLN activation snapshots: the same
        tensor object may represent different numerical states at successive blocks.
        """
        if value is None:
            return None
        return self._clone(value)

    def shared_tensor(self, value: torch.Tensor | None) -> torch.Tensor | None:
        """Clone an immutable-by-contract tensor once and reuse the CPU copy."""
        if value is None:
            return None
        cached = self._shared_cache.get(id(value))
        if cached is not None:
            return cached
        clone = self._clone(value)
        self._shared_cache[id(value)] = clone
        return clone

    def snapshot_object(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return self.snapshot_tensor(value)
        if isinstance(value, tuple):
            return tuple(self.snapshot_object(v) for v in value)
        if isinstance(value, list):
            return [self.snapshot_object(v) for v in value]
        if isinstance(value, dict):
            return {k: self.snapshot_object(v) for k, v in value.items()}
        return value


def _arg(args: tuple[Any, ...], kwargs: Mapping[str, Any], index: int, name: str, default: Any = None) -> Any:
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def _option_summary(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, slice):
        return {"type": "slice", "start": value.start, "stop": value.stop, "step": value.step}
    if torch.is_tensor(value):
        return {
            "type": "tensor",
            "shape": [int(x) for x in value.shape],
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (tuple, list)) and len(value) <= 16:
        return [_option_summary(v) for v in value]
    identity = getattr(value, "identity", None)
    if callable(identity):
        try:
            identity = identity()
        except TypeError:
            identity = None
    semantic_digest = getattr(value, "semantic_digest", None)
    if callable(semantic_digest):
        try:
            semantic_digest = semantic_digest()
        except TypeError:
            semantic_digest = None
    out = {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
    if identity is not None:
        out["identity"] = str(identity)
    if semantic_digest is not None:
        out["semantic_digest"] = str(semantic_digest)
    api = getattr(value, "api", None)
    if isinstance(api, int):
        out["api"] = api
    return out


def _transformer_options_summary(options: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _option_summary(value) for key, value in sorted(options.items(), key=lambda kv: str(kv[0]))}


def _layout_capture_context(layout: Any) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if layout is None:
        return None, {}
    position_ids = getattr(layout, "position_ids", None)
    context: dict[str, Any] = {
        "layout_type": f"{type(layout).__module__}.{type(layout).__qualname__}",
    }
    for name in ("signature", "seq_len", "segments"):
        value = getattr(layout, name, None)
        if value is not None:
            context[name] = _option_summary(value)
    return position_ids if torch.is_tensor(position_ids) else None, context


class PilotActivationCapture:
    """Ephemeral hooks that capture selected native H3 block inputs exactly once.

    The session is intentionally one-forward-only. A selected block executing twice
    is rejected rather than silently mixing sampler/forecast/re-entrant evaluations.
    Mutable activations are snapshotted per observation; only H3 inputs that are
    immutable across the block loop (timestep embedding, RoPE table, layout position
    IDs) are interned. All tensors are detached and copied to CPU under an explicit
    byte budget. Hooks are always removed when the context exits.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        case_id: str,
        sigma: float | None,
        modality_label: str | None,
        max_capture_bytes: int,
        block_indices: Sequence[int] = PILOT_BLOCKS,
        context: Mapping[str, Any] | None = None,
        require_plain_native: bool = True,
    ) -> None:
        if not case_id.strip():
            raise ValueError("live pilot capture requires a stable case_id")
        if sigma is not None and not 0.0 <= float(sigma) <= 1.0:
            raise ValueError("live pilot capture sigma must be within [0,1]")
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise RuntimeError("live pilot capture expects a model with .blocks")
        selected = tuple(int(i) for i in block_indices)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("live pilot capture block indices must be unique and non-empty")
        if any(i < 0 or i >= len(blocks) for i in selected):
            raise IndexError(f"live pilot capture block indices {selected} exceed model depth {len(blocks)}")
        if context and _CAPTURE_CONTEXT_KEY in context:
            raise ValueError(f"caller context may not override {_CAPTURE_CONTEXT_KEY!r}")

        self.model = model
        self.blocks = blocks
        self.case_id = case_id
        self.sigma = sigma
        self.modality_label = modality_label
        self.block_indices = selected
        self.user_context = dict(context or {})
        self.require_plain_native = bool(require_plain_native)
        self.budget = _CpuCloneBudget(max_capture_bytes)
        self._handles: list[Any] = []
        self._pending: dict[int, dict[str, Any]] = {}
        self._records: dict[int, CapturedPilotCase] = {}
        self._active = False

    def __enter__(self) -> "PilotActivationCapture":
        if self._active:
            raise RuntimeError("live pilot capture session is already active")
        self._active = True
        for block_index in self.block_indices:
            block = self.blocks[block_index]
            attention = getattr(block, "attn", None)
            if attention is None:
                self.close()
                raise RuntimeError(f"pilot block {block_index} has no .attn module")
            self._handles.append(
                block.register_forward_pre_hook(self._make_block_pre_hook(block_index), with_kwargs=True)
            )
            self._handles.append(
                attention.register_forward_pre_hook(self._make_attention_pre_hook(block_index), with_kwargs=True)
            )
            self._handles.append(
                block.register_forward_hook(self._make_block_post_hook(block_index), with_kwargs=True)
            )
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = False

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _make_block_pre_hook(self, block_index: int):
        def hook(module, args, kwargs):
            if block_index in self._pending or block_index in self._records:
                raise RuntimeError(
                    f"pilot block {block_index} executed more than once in a one-forward capture"
                )
            x = _arg(args, kwargs, 0, "x")
            t_emb = _arg(args, kwargs, 1, "t_emb")
            mod_segments = _arg(args, kwargs, 2, "mod_segments")
            rope_freqs = _arg(args, kwargs, 3, "rope_freqs")
            options = _arg(args, kwargs, 4, "transformer_options", {})
            attention_override = _arg(args, kwargs, 5, "attention", None)
            if not torch.is_tensor(x) or not torch.is_tensor(t_emb):
                raise RuntimeError("pilot block hook did not receive tensor x/t_emb")
            if options is None:
                options = {}
            if not isinstance(options, Mapping):
                raise RuntimeError("pilot block transformer_options must be a mapping")
            if self.require_plain_native:
                if PROVIDER_KEY in options:
                    raise RuntimeError("Stage-A live capture must not run with a Keyless provider installed")
                if attention_override is not None:
                    raise RuntimeError("Stage-A live capture must not use a DiTBlock attention override")

            layout = options.get("minimax_h3_layout")
            position_ids, layout_context = _layout_capture_context(layout)
            x_cpu = self.budget.snapshot_tensor(x)
            t_cpu = self.budget.shared_tensor(t_emb)
            rope_cpu = self.budget.shared_tensor(rope_freqs) if torch.is_tensor(rope_freqs) else None
            pos_cpu = self.budget.shared_tensor(position_ids) if position_ids is not None else None
            if pos_cpu is not None and pos_cpu.shape[0] != x_cpu.shape[0]:
                raise RuntimeError(
                    f"captured layout position rows {pos_cpu.shape[0]} != block rows {x_cpu.shape[0]}"
                )
            capture_context = {
                "block_index": block_index,
                "transformer_options": _transformer_options_summary(options),
                "layout": layout_context,
            }
            context = dict(self.user_context)
            context[_CAPTURE_CONTEXT_KEY] = capture_context
            self._pending[block_index] = {
                "x": x_cpu,
                "t_emb": t_cpu,
                "mod_segments": self.budget.snapshot_object(mod_segments),
                "rope_freqs": rope_cpu,
                "position_ids": pos_cpu,
                "context": context,
                "attention_input": None,
            }
        return hook

    def _make_attention_pre_hook(self, block_index: int):
        def hook(module, args, kwargs):
            pending = self._pending.get(block_index)
            if pending is None:
                raise RuntimeError(f"attention for pilot block {block_index} executed without block entry")
            if pending["attention_input"] is not None:
                raise RuntimeError(f"attention for pilot block {block_index} executed more than once")
            if not args or not torch.is_tensor(args[0]):
                raise RuntimeError("native attention hook did not receive the post-AdaLN hidden tensor")
            hidden = self.budget.snapshot_tensor(args[0])
            if hidden.shape != pending["x"].shape:
                raise RuntimeError("post-AdaLN attention input shape does not match captured block input")
            pending["attention_input"] = hidden
        return hook

    def _make_block_post_hook(self, block_index: int):
        def hook(module, args, kwargs, output):
            pending = self._pending.pop(block_index, None)
            if pending is None:
                raise RuntimeError(f"pilot block {block_index} completed without a pending capture")
            attention_input = pending.pop("attention_input")
            if attention_input is None:
                raise RuntimeError(f"pilot block {block_index} completed without executing native attention")
            case = PilotCase(
                x=pending["x"],
                t_emb=pending["t_emb"],
                mod_segments=pending["mod_segments"],
                rope_freqs=pending["rope_freqs"],
                transformer_options={},
                case_id=self.case_id,
                sigma=self.sigma,
                modality_label=self.modality_label,
                position_ids=pending["position_ids"],
                context=pending["context"],
            )
            self._records[block_index] = CapturedPilotCase(
                block_index=block_index,
                case=case,
                attention_input=attention_input,
                captured_bytes=self.budget.used_bytes,
            )
        return hook

    def records(self) -> tuple[CapturedPilotCase, ...]:
        missing = [i for i in self.block_indices if i not in self._records]
        if missing:
            raise RuntimeError(f"live pilot capture did not observe selected blocks: {missing}")
        return tuple(self._records[i] for i in self.block_indices)
