from __future__ import annotations

import pytest

from minimax_h3_keyless.native_capture_policy import require_plain_native_capture_options


def test_plain_native_policy_allows_ordinary_h3_execution_metadata() -> None:
    require_plain_native_capture_options(
        {
            "minimax_h3_layout": object(),
            "minimax_h3_sigma_shift_video": 12.0,
            "minimax_h3_sigma_shift_audio": 3.0,
            "block_index": 25,
            "sample_sigmas": None,
            "patches": {},
            "patches_replace": {},
            "wrappers": {},
            "callbacks": {},
        }
    )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("patches", {"attn1_patch": [object()]}, "unpatched native H3 teacher"),
        ("patches_replace", {"dit": {("double_block", 25): object()}}, "unpatched native H3 teacher"),
        ("wrappers", {"diffusion_model": {None: [object()]}}, "unpatched native H3 teacher"),
        ("callbacks", {"on_apply_hooks": {None: [object()]}}, "unpatched native H3 teacher"),
        ("optimized_attention_override", object(), "native optimized attention"),
    ],
)
def test_plain_native_policy_rejects_current_comfy_execution_patch_surfaces(
    key, value, message
) -> None:
    with pytest.raises(RuntimeError, match=message):
        require_plain_native_capture_options({key: value})


def test_plain_native_policy_treats_empty_patch_families_as_inert() -> None:
    require_plain_native_capture_options(
        {
            "patches": {},
            "patches_replace": {},
            "wrappers": {},
            "callbacks": {},
            "optimized_attention_override": None,
        }
    )
