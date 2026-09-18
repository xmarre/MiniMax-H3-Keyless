# Live INT8 ConvRot loader contract

The canonical Keyless INT8 artifact is exported from an accepted folded BF16
`h3_keyless_core50_v1` checkpoint. The production storage contract is owned by
the live ComfyUI / comfy-kitchen mixed-precision loader, not by filename
convention.

Implementation audit on 2026-09-18 used:

- ComfyUI `master`: `7de99222f06e1b6cddb1868859319010bd7ac878`
- comfy-kitchen `main`: `b2a2972ac68c395bbda8ad9030e8ae1089287815`

At those revisions, `TensorWiseINT8Layout.quantize(..., is_weight=True,
per_channel=True, convrot=True, convrot_groupsize=256)` returns an INT8 tensor
with unchanged matrix shape and a float32 per-output-row scale shaped
`[out_features, 1]`. Its serialized tensor suffixes are the raw weight and
`_scale`.

ComfyUI's mixed-precision Linear loader consumes:

- `<layer>.weight`: INT8 ConvRot storage;
- `<layer>.weight_scale`: float32 scale;
- `<layer>.comfy_quant`: JSON containing
  `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`.

The Keyless exporter intentionally emits that native layer-level contract for
exactly the 200 core-heavy linears in
`minimax_h3_keyless_core50_200_v1`. It does not invent a Keyless-specific
runtime quantization layout.

The hosted source/runtime contract check is a drift alarm only. It exercises the
current comfy-kitchen eager ConvRot quantizer and source-audits ComfyUI's
descriptor loader. It does not validate a CUDA kernel, a full 24+ GB artifact,
decoded audiovisual quality, or BF16-to-INT8 numerical acceptance on the target
GPU.
