#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minimax_h3_keyless.live_capture import discover_clean_git_revision
from minimax_h3_keyless.progressive_authorization import load_progressive_prefix_manifest
from minimax_h3_keyless.progressive_snapshot_runtime import load_progressive_snapshot_streaming
from minimax_h3_keyless.teacher import load_pinned_bf16_teacher


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reload a Stage-B mixed QV/QKV snapshot into a fresh pinned native H3 model "
            "using bounded tensor-at-a-time safetensors reads. This proves snapshot "
            "topology/state loading; it is not a denoiser or media parity test."
        )
    )
    parser.add_argument("--teacher", required=True, help="Exact pinned native BF16 teacher safetensors")
    parser.add_argument("--snapshot", required=True, help="Progressive snapshot safetensors")
    parser.add_argument("--snapshot-manifest", required=True, help="Snapshot receipt JSON")
    parser.add_argument("--prefix-manifest", required=True, help="Accepted progressive prefix manifest")
    args = parser.parse_args()

    prefix, prefix_manifest_sha = load_progressive_prefix_manifest(args.prefix_manifest)
    runtime_commit = discover_clean_git_revision(
        Path(__file__).resolve().parents[1],
        label="MiniMax-H3-Keyless progressive snapshot verifier source",
    )
    if runtime_commit.lower() != prefix.code_commit.lower():
        raise RuntimeError(
            "progressive snapshot verifier source revision differs from the sweep revision: "
            f"runtime={runtime_commit}, prefix={prefix.code_commit}"
        )

    teacher = load_pinned_bf16_teacher(args.teacher)
    model = teacher.diffusion_model
    model.to(torch.device("cpu"))
    load_progressive_snapshot_streaming(
        model,
        args.snapshot,
        prefix,
        prefix_manifest_sha256=prefix_manifest_sha,
        manifest_path=args.snapshot_manifest,
    )

    print(f"Snapshot reload validated: {args.snapshot}")
    print(f"Prefix identity SHA-256: {prefix.identity_sha256}")
    print(f"Accepted blocks: {len(prefix.accepted)}")
    print("Reload mode: bounded tensor-at-a-time safetensors streaming")
    print("Scope: structural mixed-topology/state reload only; no numerical/media parity claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
