from __future__ import annotations

import inspect
from typing import Any

from .attention import KeylessAttentionDeploy, KeylessAttentionTrain
from .contracts import CONTRACT_KEY, KeylessContractV1


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
    if blocks is None or len(blocks) != 50:
        raise RuntimeError(
            f"expected MiniMax H3 core50 blocks, got {None if blocks is None else len(blocks)}"
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
            if int(a["num_layers"]) != 50:
                raise RuntimeError("h3_keyless_core50_v1 requires exactly 50 diffusion blocks")
            if int(a["token_refiner_num_layers"]) != 2:
                raise RuntimeError("h3_keyless_core50_v1 requires two native-QKV token refiners")
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
