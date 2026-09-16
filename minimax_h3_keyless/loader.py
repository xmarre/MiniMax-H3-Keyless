from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .checkpoint import validate_deploy_checkpoint
from .contracts import (
    CONTRACT_KEY,
    HEAD_DIM,
    HEADS,
    HIDDEN_SIZE,
    contract_from_metadata,
)
from .model import make_comfy_keyless_model_class


def _merge_metadata_config(
    derived: Mapping[str, Any], metadata: Mapping[str, str]
) -> dict[str, Any]:
    """Merge optional constructor metadata without letting it rewrite proven structure."""
    cfg = dict(derived)
    config_json = metadata.get("config")
    if not config_json:
        return cfg
    try:
        parsed = json.loads(config_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("MiniMax H3 metadata config is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("MiniMax H3 metadata config must decode to an object")
    for key, value in parsed.items():
        if key in cfg and value != cfg[key]:
            raise RuntimeError(
                "MiniMax H3 metadata config contradicts checkpoint structure: "
                f"{key}={value!r}, derived={cfg[key]!r}"
            )
        cfg[key] = value
    return cfg


def _derive_h3_unet_config(sd: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> dict[str, Any]:
    """Derive current MiniMax-H3 constructor facts from real QV signatures without qkv spoofing."""
    cfg: dict[str, Any] = {
        "image_model": "minimax_h3",
        "num_layers": 50,
        "token_refiner_num_layers": 2,
        "hidden_size": HIDDEN_SIZE,
        "num_attention_heads": HEADS,
        "attention_head_dim": HEAD_DIM,
        "ffn_hidden_size": sd["blocks.0.mlp.fc1.weight"].shape[0] // 2,
        "text_dim": sd["condition_proj.weight"].shape[1],
        "rope_inv_freq_len": sd["rope.inv_freq"].shape[0],
        "gate_compress": "blocks.0.attn.to_gate_compress.weight" in sd,
    }
    if "adaln_t_table" in sd:
        table = sd["adaln_t_table"].shape
        cfg["adaln_curve_grid"] = table[0]
        cfg["time_embed_dim"] = table[1]
    return _merge_metadata_config(cfg, metadata)


def load_keyless_model(path: str | Path, *, model_options: Mapping[str, Any] | None = None):
    """Load a canonical Keyless H3 artifact as an ordinary Comfy ModelPatcher."""
    model_options = dict(model_options or {})
    try:
        import comfy.model_base
        import comfy.model_management
        import comfy.model_patcher
        import comfy.storage
        import comfy.supported_models
        import comfy.utils
    except ImportError as exc:
        raise RuntimeError("ComfyUI is required to load a Keyless H3 model") from exc

    sd, metadata = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
    metadata = dict(metadata or {})
    if model_options.get("custom_operations") is None:
        sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
        metadata = dict(metadata or {})
    validate_deploy_checkpoint(sd, metadata)

    parameters = comfy.utils.calculate_parameters(sd)
    weight_dtype = comfy.utils.weight_dtype(sd)
    load_device = model_options.get("load_device", comfy.model_management.get_torch_device())
    offload_device = model_options.get("offload_device", comfy.model_management.unet_offload_device())
    unet_dtype = model_options.get("dtype") or comfy.model_management.unet_dtype(
        model_params=parameters,
        supported_dtypes=[torch.bfloat16, torch.float16, torch.float32],
        weight_dtype=weight_dtype,
        device=load_device,
    )
    manual_cast_dtype = comfy.model_management.unet_manual_cast(
        unet_dtype, load_device, [torch.bfloat16, torch.float16, torch.float32]
    )

    cfg_dict = _derive_h3_unet_config(sd, metadata)
    config = comfy.supported_models.MiniMaxH3(cfg_dict)
    quant_config = comfy.utils.detect_layer_quantization(sd, "")
    if quant_config is not None:
        config.quant_config = quant_config
    config.set_inference_dtype(unet_dtype, manual_cast_dtype, device=load_device)
    custom_operations = model_options.get("custom_operations")
    if custom_operations is not None:
        config.custom_operations = custom_operations
    if model_options.get("fp8_optimizations"):
        config.optimizations["fp8"] = True

    KeylessDiffusion = make_comfy_keyless_model_class()

    class KeylessBaseModel(comfy.model_base.MiniMaxH3):
        def __init__(self, model_config, device=None):
            comfy.model_base.BaseModel.__init__(
                self,
                model_config,
                comfy.model_base.ModelType.FLOW_AV,
                device=device,
                unet_model=KeylessDiffusion,
            )

    initial_device = comfy.model_management.unet_inital_load_device(parameters, unet_dtype)
    model = KeylessBaseModel(config, device=initial_device)
    Patcher = (
        comfy.model_patcher.ModelPatcher
        if model_options.get("disable_dynamic", False)
        else comfy.model_patcher.CoreModelPatcher
    )
    patcher = Patcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
        fast_disk=comfy.storage.state_dict_fast_disk(sd),
    )
    if not comfy.model_management.is_device_cpu(offload_device):
        model.to(offload_device)
    model.load_model_weights(sd, "", assign=patcher.is_dynamic())

    loaded_contract = contract_from_metadata(metadata)
    existing_contract = getattr(model.diffusion_model, CONTRACT_KEY)
    if existing_contract.architecture != loaded_contract.architecture:
        raise RuntimeError("loaded Keyless model contract does not match checkpoint architecture")
    setattr(model.diffusion_model, CONTRACT_KEY, loaded_contract)
    return patcher
