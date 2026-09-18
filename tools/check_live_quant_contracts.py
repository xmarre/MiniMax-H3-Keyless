#!/usr/bin/env python3
"""Validate the live ComfyUI / comfy-kitchen INT8 ConvRot contract used by Keyless.

This is deliberately a narrow implementation-time source/runtime oracle. It does
not establish GPU numerical parity for a production artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _require(path: Path, *needles: str) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise SystemExit(f"{path}: missing required contract fragments: {missing}")


def _check_comfy(comfy_root: Path) -> None:
    ops = comfy_root / "comfy" / "ops.py"
    utils = comfy_root / "comfy" / "utils.py"

    _require(
        ops,
        'layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)',
        'module.quant_format = layer_conf.get("format", None)',
        'elif module.quant_format == "int8_tensorwise":',
        'scale = pop_scale("weight_scale")',
        'if layer_conf.get("convrot", params_conf.get("convrot", False)):',
        'scales["convrot"] = True',
        'scales["convrot_groupsize"] = int(',
        'quant_conf["convrot"] = True',
        'quant_conf["convrot_groupsize"] = getattr(params, "convrot_groupsize", 256)',
    )
    _require(
        utils,
        "def detect_layer_quantization(state_dict, prefix):",
        'k.endswith(".comfy_quant")',
        'return {"mixed_ops": True}',
        "def convert_old_quants(state_dict, model_prefix="", metadata={}):",
    )


def _check_kitchen(kitchen_root: Path) -> None:
    sys.path.insert(0, str(kitchen_root))
    try:
        import comfy_kitchen as ck
        from comfy_kitchen.tensor import TensorWiseINT8Layout
    finally:
        sys.path.pop(0)

    source = kitchen_root / "comfy_kitchen" / "tensor" / "int8.py"
    _require(
        source,
        "class TensorWiseINT8Layout(QuantizedLayout):",
        "per_channel: bool = False",
        "convrot: bool = False",
        "convrot_groupsize: int = 256",
        'if convrot:',
        '"quantize_int8_convrot_weight"',
        'return {',
        '"": qdata,',
        '"_scale": params.scale,',
    )

    weight = torch.randn(3, 256, dtype=torch.bfloat16)
    with ck.registry.use_backend("eager"):
        qdata, params = TensorWiseINT8Layout.quantize(
            weight,
            stochastic_rounding=0,
            is_weight=True,
            per_channel=True,
            convrot=True,
            convrot_groupsize=256,
        )

    if qdata.dtype != torch.int8 or tuple(qdata.shape) != (3, 256):
        raise SystemExit(
            "live TensorWiseINT8Layout returned unexpected ConvRot qdata "
            f"dtype={qdata.dtype}, shape={tuple(qdata.shape)}"
        )
    if params.scale.dtype != torch.float32 or tuple(params.scale.shape) != (3, 1):
        raise SystemExit(
            "live TensorWiseINT8Layout returned unexpected ConvRot scale "
            f"dtype={params.scale.dtype}, shape={tuple(params.scale.shape)}"
        )
    if not params.convrot or int(params.convrot_groupsize) != 256:
        raise SystemExit("live TensorWiseINT8Layout did not retain ConvRot metadata")

    serialized = TensorWiseINT8Layout.state_dict_tensors(qdata, params)
    if set(serialized) != {"", "_scale"}:
        raise SystemExit(
            "live TensorWiseINT8Layout state_dict_tensors contract changed: "
            f"{sorted(serialized)}"
        )
    if serialized[""] is not qdata or serialized["_scale"] is not params.scale:
        raise SystemExit("live TensorWiseINT8Layout state_dict tensor identity changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--kitchen", type=Path, required=True)
    args = parser.parse_args()

    _check_comfy(args.comfy)
    _check_kitchen(args.kitchen)
    print(
        json.dumps(
            {
                "schema": "minimax_h3_keyless_live_quant_contract_v1",
                "comfy": str(args.comfy),
                "comfy_kitchen": str(args.kitchen),
                "convrot_groupsize": 256,
                "storage": {
                    "weight": "I8",
                    "scale_suffix": "_scale",
                    "scale_dtype": "F32",
                    "scale_shape": "[out,1]",
                    "descriptor_suffix": ".comfy_quant",
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
