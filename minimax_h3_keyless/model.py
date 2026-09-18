from __future__ import annotations

import inspect
from typing import Any

from .attention import KeylessAttentionDeploy, KeylessAttentionTrain
from .contracts import (
    CONTRACT_KEY,
    CORE_BLOCKS,
    HEAD_DIM,
    HEADS,
    HIDDEN_SIZE,
    NORM_EPS,
    TOKEN_REFINER_BLOCKS,
    KeylessContractV1,
)


def validate_keyless_geometry(
    *,
    hidden: int,
    heads: int,
    head_dim: int,
    core_blocks: int,
    token_refiner_blocks: int,
    norm_epsilon: float,
) -> None:
    """Reject constructor geometry that would make the public v1 contract false."""
    expected = {
        "hidden_size": HIDDEN_SIZE,
        "num_attention_heads": HEADS,
        "attention_head_dim": HEAD_DIM,
        "num_layers": CORE_BLOCKS,
        "token_refiner_num_layers": TOKEN_REFINER_BLOCKS,
    }
    actual = {
        "hidden_size": int(hidden),
        "num_attention_heads": int(heads),
        "attention_head_dim": int(head_dim),
        "num_layers": int(core_blocks),
        "token_refiner_num_layers": int(token_refiner_blocks),
    }
    mismatches = [
        f"{name}={actual[name]!r} (expected {value!r})"
        for name, value in expected.items()
        if actual[name] != value
    ]
    if float(norm_epsilon) != NORM_EPS:
        mismatches.append(
            f"qk_norm_eps={float(norm_epsilon)!r} (expected {NORM_EPS!r})"
        )
    if mismatches:
        raise RuntimeError(
            "h3_keyless_core50_v1 geometry mismatch: " + ", ".join(mismatches)
        )


def replace_main_attention_modules(
    model: Any,
    *,
    hidden: int,
    heads: int,
    head_dim: int,
    eps: float,
    gate_compress: bool,
    dtype=None,
    device=None,
    operations=None,
    training: bool = False,
) -> Any:
    """Replace exactly the 50 diffusion-block attentions; token refiners remain native QKV."""
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise RuntimeError("expected MiniMax H3 core50 blocks, got no blocks attribute")
    token_refiner = getattr(model, "token_refiner", None)
    refiner_blocks = getattr(token_refiner, "blocks", None)
    validate_keyless_geometry(
        hidden=hidden,
        heads=heads,
        head_dim=head_dim,
        core_blocks=len(blocks),
        token_refiner_blocks=-1 if refiner_blocks is None else len(refiner_blocks),
        norm_epsilon=eps,
    )
    cls = KeylessAttentionTrain if training else KeylessAttentionDeploy
    for i, block in enumerate(blocks):
        block.attn = cls(
            hidden,
            heads,
            head_dim,
            eps,
            gate_compress=gate_compress,
            block_index=i,
            dtype=dtype,
            device=device,
            operations=operations,
        )
    setattr(model, CONTRACT_KEY, KeylessContractV1())
    return model


def make_comfy_keyless_model_class():
    """Create a narrow subclass around the live Comfy MiniMaxH3Model."""
    try:
        import comfy.ldm.minimax.model as minimax_model
    except ImportError as exc:
        raise RuntimeError("ComfyUI is required to construct KeylessMiniMaxH3Model") from exc

    native_cls = minimax_model.MiniMaxH3Model
    native_sig = inspect.signature(native_cls.__init__)
    required_names = {
        "hidden_size",
        "num_layers",
        "token_refiner_num_layers",
        "num_attention_heads",
        "attention_head_dim",
        "qk_norm_eps",
        "gate_compress",
        "dtype",
        "device",
        "operations",
    }
    missing = required_names.difference(native_sig.parameters)
    if missing:
        raise RuntimeError(f"live MiniMaxH3Model constructor moved required fields: {sorted(missing)}")

    class KeylessMiniMaxH3Model(native_cls):
        def __init__(self, *args, **kwargs):
            bound = native_sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            validate_keyless_geometry(
                hidden=a["hidden_size"],
                heads=a["num_attention_heads"],
                head_dim=a["attention_head_dim"],
                core_blocks=a["num_layers"],
                token_refiner_blocks=a["token_refiner_num_layers"],
                norm_epsilon=a["qk_norm_eps"],
            )
            super().__init__(*args, **kwargs)
            replace_main_attention_modules(
                self,
                hidden=int(a["hidden_size"]),
                heads=int(a["num_attention_heads"]),
                head_dim=int(a["attention_head_dim"]),
                eps=float(a["qk_norm_eps"]),
                gate_compress=bool(a["gate_compress"]),
                dtype=a["dtype"],
                device=a["device"],
                operations=a["operations"],
                training=False,
            )

    KeylessMiniMaxH3Model.__name__ = "KeylessMiniMaxH3Model"
    return KeylessMiniMaxH3Model
