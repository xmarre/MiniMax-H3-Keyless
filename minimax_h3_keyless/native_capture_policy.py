from __future__ import annotations

from typing import Any, Mapping

from .contracts import PROVIDER_KEY


# These option families are the current Comfy mechanisms that can replace or wrap
# the H3 block/attention path before or during a forward. Stage-A native-teacher
# captures must fail closed rather than silently distill a patched teacher.
_FORBIDDEN_NONEMPTY_OPTION_FAMILIES = (
    "patches",
    "patches_replace",
    "wrappers",
    "callbacks",
)
_FORBIDDEN_DIRECT_OPTION_KEYS = (
    "optimized_attention_override",
)


def require_plain_native_capture_options(
    transformer_options: Mapping[str, Any],
    *,
    attention_override: Any = None,
) -> None:
    """Reject current Comfy mechanisms that can alter the native H3 teacher path.

    This is deliberately narrower than rejecting all transformer options: ordinary H3
    execution metadata such as ``minimax_h3_layout``, sigma shifts, ``block_index`` and
    sampler bookkeeping must remain available. The rejected families are executable
    patch/wrapper/override surfaces in the current audited Comfy runtime.
    """
    if PROVIDER_KEY in transformer_options:
        raise RuntimeError("Stage-A live capture must not run with a Keyless provider installed")
    if attention_override is not None:
        raise RuntimeError("Stage-A live capture must not use a DiTBlock attention override")

    for key in _FORBIDDEN_NONEMPTY_OPTION_FAMILIES:
        value = transformer_options.get(key)
        if value:
            raise RuntimeError(
                f"Stage-A live capture requires an unpatched native H3 teacher; "
                f"transformer_options[{key!r}] is non-empty"
            )
    for key in _FORBIDDEN_DIRECT_OPTION_KEYS:
        if transformer_options.get(key) is not None:
            raise RuntimeError(
                f"Stage-A live capture requires native optimized attention; "
                f"transformer_options[{key!r}] is installed"
            )
