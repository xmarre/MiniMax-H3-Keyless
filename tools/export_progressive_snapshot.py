#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minimax_h3_keyless.live_capture import discover_clean_git_revision
from minimax_h3_keyless.progressive_authorization import load_progressive_prefix_manifest
from minimax_h3_keyless.progressive_restore import restore_progressive_model_prefix
from minimax_h3_keyless.progressive_snapshot import (
    export_progressive_snapshot,
    validate_progressive_snapshot_file,
)
from minimax_h3_keyless.teacher import load_pinned_bf16_teacher


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export one immutable full-model Stage-B snapshot for periodic testing. "
            "Accepted early blocks are folded QV; all later core blocks and both token "
            "refiners remain native QKV. The artifact is explicitly non-canonical."
        )
    )
    parser.add_argument("--teacher", required=True, help="Exact pinned native BF16 teacher safetensors")
    parser.add_argument("--prefix-manifest", required=True, help="Accepted progressive prefix manifest")
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Directory containing the immutable accepted block checkpoints/results",
    )
    parser.add_argument("--output", required=True, help="Destination .safetensors snapshot path")
    parser.add_argument(
        "--manifest-output",
        help="Optional sidecar path; defaults to <output>.manifest.json",
    )
    args = parser.parse_args()

    prefix, prefix_manifest_sha = load_progressive_prefix_manifest(args.prefix_manifest)
    runtime_commit = discover_clean_git_revision(
        Path(__file__).resolve().parents[1],
        label="MiniMax-H3-Keyless progressive snapshot source",
    )
    if runtime_commit.lower() != prefix.code_commit.lower():
        raise RuntimeError(
            "progressive snapshot source revision differs from the revision that owns the sweep: "
            f"runtime={runtime_commit}, prefix={prefix.code_commit}"
        )

    teacher = load_pinned_bf16_teacher(args.teacher)
    model = teacher.diffusion_model
    restore_progressive_model_prefix(
        model,
        prefix,
        output_dir=args.artifact_dir,
    )

    # Avoid creating a second full GPU-sized copy while safetensors assembles its CPU map.
    model.to(torch.device("cpu"))
    result = export_progressive_snapshot(
        model,
        prefix,
        args.output,
        prefix_manifest_sha256=prefix_manifest_sha,
        manifest_path=args.manifest_output,
    )
    validate_progressive_snapshot_file(
        result.artifact_path,
        prefix,
        prefix_manifest_sha256=prefix_manifest_sha,
        manifest_path=result.manifest_path,
    )

    print(f"Snapshot: {result.artifact_path}")
    print(f"Snapshot SHA-256: {result.artifact_sha256}")
    print(f"Snapshot bytes: {result.artifact_bytes}")
    print(f"Snapshot manifest: {result.manifest_path}")
    print(f"Snapshot manifest file SHA-256: {result.manifest_sha256}")
    print(f"Snapshot manifest identity SHA-256: {result.manifest_identity_sha256}")
    print(f"Accepted blocks: {result.accepted_blocks}")
    print("Canonical release artifact: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
