from __future__ import annotations

import torch

from .minimax_h3_keyless.loader import load_keyless_model


class MiniMaxH3KeylessLoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    {"advanced": True},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders/advanced"
    DESCRIPTION = (
        "Strict loader for h3_keyless_core50_v1 checkpoints. "
        "It rejects native/mixed QKV artifacts instead of synthesizing a K projection."
    )

    def load(self, model_name: str, weight_dtype: str = "default"):
        import folder_paths

        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        return (load_keyless_model(path, model_options=model_options),)


NODE_CLASS_MAPPINGS = {"MiniMaxH3KeylessLoader": MiniMaxH3KeylessLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3KeylessLoader": "MiniMax H3 Keyless Loader"}
