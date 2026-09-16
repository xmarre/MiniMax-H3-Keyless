from __future__ import annotations

from pathlib import Path

import torch

if __package__:
    from .minimax_h3_keyless.capture_io import CaptureBundleProvenance
    from .minimax_h3_keyless.live_capture import (
        StageALiveCaptureController,
        build_runtime_execution_descriptor,
        build_stage_a_capture_spec,
        discover_clean_git_revision,
        install_stage_a_capture_wrapper,
        mark_pinned_stage_a_teacher,
        require_pinned_stage_a_teacher,
    )
    from .minimax_h3_keyless.loader import load_keyless_model
    from .minimax_h3_keyless.teacher import load_pinned_bf16_teacher
else:
    from minimax_h3_keyless.capture_io import CaptureBundleProvenance
    from minimax_h3_keyless.live_capture import (
        StageALiveCaptureController,
        build_runtime_execution_descriptor,
        build_stage_a_capture_spec,
        discover_clean_git_revision,
        install_stage_a_capture_wrapper,
        mark_pinned_stage_a_teacher,
        require_pinned_stage_a_teacher,
    )
    from minimax_h3_keyless.loader import load_keyless_model
    from minimax_h3_keyless.teacher import load_pinned_bf16_teacher


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


class MiniMaxH3StageATeacherLoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {"model_name": (folder_paths.get_filename_list("diffusion_models"),)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders/advanced"
    DESCRIPTION = (
        "Stage-A-only loader for the exact pinned native MiniMax H3 BF16 teacher. "
        "It verifies the full checkpoint SHA-256 and native QKV topology and refuses "
        "quantized, Keyless, mixed, custom-operation or non-BF16 teachers."
    )

    def load(self, model_name: str):
        import folder_paths

        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        loaded = load_pinned_bf16_teacher(path)
        mark_pinned_stage_a_teacher(loaded.patcher)
        return (loaded.patcher,)


class MiniMaxH3StageACapture:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "dataset_manifest_path": ("STRING", {"default": "", "multiline": False}),
                "case_id": ("STRING", {"default": "", "multiline": False}),
                "target_sigma": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.000001},
                ),
                "output_subdir": (
                    "STRING",
                    {"default": "keyless_stage_a", "multiline": False},
                ),
                "max_capture_mib": (
                    "INT",
                    {"default": 8192, "min": 64, "max": 65536, "step": 64},
                ),
            },
            "optional": {
                "sigma_tolerance": (
                    "FLOAT",
                    {"default": 0.000001, "min": 0.0, "max": 0.0001, "step": 0.0000001},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3/Keyless/training"
    DESCRIPTION = (
        "Clone a model loaded by the Stage-A BF16 Teacher Loader and install a one-shot "
        "native H3 capture at one dataset-declared video sigma. The capture is accepted "
        "only when this is the sole DIFFUSION_MODEL wrapper and the teacher remains "
        "unpatched. It writes immutable block 0/25/49 capture evidence under the Comfy "
        "output directory when the sampler reaches the requested sigma."
    )

    def apply(
        self,
        model,
        dataset_manifest_path: str,
        case_id: str,
        target_sigma: float,
        output_subdir: str,
        max_capture_mib: int,
        sigma_tolerance: float = 1e-6,
    ):
        import folder_paths

        require_pinned_stage_a_teacher(model)
        cloned = model.clone()
        require_pinned_stage_a_teacher(cloned)
        spec = build_stage_a_capture_spec(
            dataset_manifest_path,
            case_id=case_id,
            target_sigma=float(target_sigma),
            output_root=folder_paths.get_output_directory(),
            output_subdir=output_subdir,
            max_capture_mib=int(max_capture_mib),
            sigma_tolerance=float(sigma_tolerance),
        )
        plugin_commit = discover_clean_git_revision(
            Path(__file__).resolve().parent,
            label="MiniMax-H3-Keyless",
        )
        comfy_commit = discover_clean_git_revision(
            Path(folder_paths.__file__).resolve().parent,
            label="ComfyUI",
        )
        provenance = CaptureBundleProvenance(
            code_commit=plugin_commit,
            comfy_commit=comfy_commit,
            dataset_manifest_sha256=spec.dataset_manifest_sha256,
            execution_descriptor=build_runtime_execution_descriptor(cloned),
        )
        controller = StageALiveCaptureController(spec, provenance)
        install_stage_a_capture_wrapper(cloned, controller)
        return (cloned,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3KeylessLoader": MiniMaxH3KeylessLoader,
    "MiniMaxH3StageATeacherLoader": MiniMaxH3StageATeacherLoader,
    "MiniMaxH3StageACapture": MiniMaxH3StageACapture,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3KeylessLoader": "MiniMax H3 Keyless Loader",
    "MiniMaxH3StageATeacherLoader": "MiniMax H3 Stage-A BF16 Teacher Loader",
    "MiniMaxH3StageACapture": "MiniMax H3 Stage-A Capture",
}
