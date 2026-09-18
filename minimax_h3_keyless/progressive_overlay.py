from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .attention import KeylessAttentionDeploy
from .live_capture import _inner_model_from_patcher, _require_plain_patcher_state
from .pilot_campaign import _require_sha256
from .progressive import ProgressivePrefix
from .progressive_authorization import load_progressive_prefix_manifest
from .progressive_restore import load_progressive_deploy_attentions
from .teacher import validate_loaded_native_teacher_model


PROGRESSIVE_OVERLAY_ATTACHMENT_KEY = "minimax_h3_keyless_progressive_overlay_v1"
_PROGRESSIVE_OVERLAY_API = 1
_FULL_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class ProgressiveOverlayAttachment:
    api: int
    prefix_identity_sha256: str
    prefix_manifest_sha256: str
    code_commit: str
    accepted_blocks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.api != _PROGRESSIVE_OVERLAY_API:
            raise ValueError("unsupported progressive overlay attachment API")
        object.__setattr__(
            self,
            "prefix_identity_sha256",
            _require_sha256(
                "progressive overlay prefix identity SHA-256",
                self.prefix_identity_sha256,
            ),
        )
        object.__setattr__(
            self,
            "prefix_manifest_sha256",
            _require_sha256(
                "progressive overlay prefix manifest SHA-256",
                self.prefix_manifest_sha256,
            ),
        )
        if not isinstance(self.code_commit, str) or not _FULL_COMMIT.fullmatch(self.code_commit):
            raise ValueError("progressive overlay code_commit must be a full 40-hex revision")
        blocks = tuple(self.accepted_blocks)
        if blocks != tuple(range(len(blocks))):
            raise ValueError("progressive overlay accepted_blocks must be a contiguous prefix")
        object.__setattr__(self, "accepted_blocks", blocks)

    def on_model_patcher_clone(self) -> "ProgressiveOverlayAttachment":
        return self


@dataclass(frozen=True)
class ProgressiveOverlayInstallation:
    patcher: Any
    prefix: ProgressivePrefix
    prefix_manifest_sha256: str
    artifact_dir: str


def _require_current_code_revision(code_commit: str, prefix: ProgressivePrefix) -> str:
    if not isinstance(code_commit, str) or not _FULL_COMMIT.fullmatch(code_commit):
        raise ValueError("progressive overlay current code revision must be a full 40-hex commit")
    normalized = code_commit.lower()
    if normalized != prefix.code_commit.lower():
        raise RuntimeError(
            "progressive overlay code revision differs from the revision that authorized "
            f"the sweep: current={normalized}, authorized={prefix.code_commit.lower()}"
        )
    return normalized


def _expected_object_patches(
    deploy_attentions: tuple[KeylessAttentionDeploy, ...],
) -> dict[str, KeylessAttentionDeploy]:
    return {
        f"diffusion_model.blocks.{index}.attn": attention
        for index, attention in enumerate(deploy_attentions)
    }


def prepare_progressive_overlay(
    patcher: Any,
    prefix_manifest_path: str | Path,
    *,
    code_commit: str,
    artifact_dir: str | Path | None = None,
) -> ProgressiveOverlayInstallation:
    """Clone a pinned native teacher patcher and overlay an accepted Keyless prefix.

    Comfy ModelPatcher clones normally share the same underlying model object. Directly
    assigning accepted attentions into ``clone.model.diffusion_model`` would therefore
    mutate the original teacher as well. This function instead reconstructs each accepted
    deploy attention off to the side and installs it only through the clone's
    ``object_patches`` table. Comfy applies those object patches for that patcher at model
    execution time and restores the shared native object on unpatch.
    """
    _require_plain_patcher_state(patcher)
    inner = _inner_model_from_patcher(patcher)
    validate_loaded_native_teacher_model(inner)

    prefix_manifest_path = Path(prefix_manifest_path)
    prefix, manifest_sha = load_progressive_prefix_manifest(prefix_manifest_path)
    _require_current_code_revision(code_commit, prefix)
    root = (
        Path(artifact_dir)
        if artifact_dir is not None
        else prefix_manifest_path.parent
    )

    deploy_attentions = load_progressive_deploy_attentions(
        inner,
        prefix,
        output_dir=root,
    )
    if len(deploy_attentions) != len(prefix.accepted):
        raise RuntimeError(
            "progressive overlay materializer returned the wrong accepted attention count"
        )
    for index, attention in enumerate(deploy_attentions):
        if not isinstance(attention, KeylessAttentionDeploy):
            raise RuntimeError(
                f"progressive overlay block {index} is not KeylessAttentionDeploy"
            )
        if int(getattr(attention, "block_index", -1)) != index:
            raise RuntimeError(
                f"progressive overlay block {index} carries the wrong block_index"
            )

    clone = patcher.clone()
    _require_plain_patcher_state(clone)
    if _inner_model_from_patcher(clone) is not inner:
        raise RuntimeError(
            "progressive overlay expected a normal Comfy clone sharing the pinned teacher model"
        )
    adder = getattr(clone, "add_object_patch", None)
    setter = getattr(clone, "set_attachments", None)
    if not callable(adder) or not callable(setter):
        raise RuntimeError(
            "current Comfy ModelPatcher lacks object-patch/attachment APIs required "
            "for non-destructive progressive overlays"
        )

    expected_patches = _expected_object_patches(deploy_attentions)
    for path, attention in expected_patches.items():
        adder(path, attention)
    actual_patches = getattr(clone, "object_patches", None)
    if actual_patches != expected_patches:
        raise RuntimeError("progressive overlay object-patch table is not the exact accepted prefix")

    attachment = ProgressiveOverlayAttachment(
        api=_PROGRESSIVE_OVERLAY_API,
        prefix_identity_sha256=prefix.identity_sha256,
        prefix_manifest_sha256=manifest_sha,
        code_commit=prefix.code_commit.lower(),
        accepted_blocks=prefix.accepted_blocks,
    )
    setter(PROGRESSIVE_OVERLAY_ATTACHMENT_KEY, attachment)

    # Prove that constructing the overlay did not mutate the shared native teacher object.
    validate_loaded_native_teacher_model(inner)
    return ProgressiveOverlayInstallation(
        patcher=clone,
        prefix=prefix,
        prefix_manifest_sha256=manifest_sha,
        artifact_dir=str(root),
    )


def require_progressive_overlay(
    patcher: Any,
    *,
    prefix: ProgressivePrefix | None = None,
    prefix_manifest_sha256: str | None = None,
) -> ProgressiveOverlayAttachment:
    """Validate the immutable overlay marker and exact pending object-patch topology."""
    attachments = getattr(patcher, "attachments", None)
    if not isinstance(attachments, dict):
        raise RuntimeError("progressive overlay requires a ModelPatcher attachment map")
    marker = attachments.get(PROGRESSIVE_OVERLAY_ATTACHMENT_KEY)
    if not isinstance(marker, ProgressiveOverlayAttachment):
        raise RuntimeError("MODEL does not carry a MiniMax H3 progressive overlay")
    marker = ProgressiveOverlayAttachment(
        api=marker.api,
        prefix_identity_sha256=marker.prefix_identity_sha256,
        prefix_manifest_sha256=marker.prefix_manifest_sha256,
        code_commit=marker.code_commit,
        accepted_blocks=marker.accepted_blocks,
    )
    if prefix is not None:
        if marker.prefix_identity_sha256 != prefix.identity_sha256:
            raise RuntimeError("progressive overlay marker does not match the requested prefix")
        if marker.accepted_blocks != prefix.accepted_blocks:
            raise RuntimeError("progressive overlay marker accepted blocks differ from prefix")
    if prefix_manifest_sha256 is not None:
        expected = _require_sha256(
            "expected progressive prefix manifest SHA-256",
            prefix_manifest_sha256,
        )
        if marker.prefix_manifest_sha256 != expected:
            raise RuntimeError("progressive overlay marker has the wrong prefix manifest identity")

    patches = getattr(patcher, "object_patches", None)
    if not isinstance(patches, dict):
        raise RuntimeError("progressive overlay patcher has no object-patch table")
    expected_paths = {
        f"diffusion_model.blocks.{index}.attn"
        for index in marker.accepted_blocks
    }
    if set(patches) != expected_paths:
        raise RuntimeError("progressive overlay object-patch paths differ from accepted prefix")
    for index in marker.accepted_blocks:
        path = f"diffusion_model.blocks.{index}.attn"
        attention = patches[path]
        if not isinstance(attention, KeylessAttentionDeploy):
            raise RuntimeError(f"progressive overlay patch {path!r} is not KeylessAttentionDeploy")
        if int(getattr(attention, "block_index", -1)) != index:
            raise RuntimeError(f"progressive overlay patch {path!r} has the wrong block_index")
    return marker
