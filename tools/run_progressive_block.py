#!/usr/bin/env python3
from __future__ import annotations

import argparse

from minimax_h3_keyless.progressive_campaign import (
    load_progressive_block_run_inputs,
    run_progressive_block_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train exactly the next early-to-late MiniMax H3 Keyless core block from a "
            "hash-bound progressive capture registry. Failed numerical candidates remain "
            "immutable evidence but do not advance the accepted prefix."
        )
    )
    parser.add_argument("--teacher", required=True, help="Exact pinned native BF16 teacher safetensors")
    parser.add_argument("--stage-a-result", required=True, help="Passed immutable Stage-A campaign result")
    parser.add_argument("--prefix-manifest", required=True, help="Current accepted progressive prefix manifest")
    parser.add_argument("--capture-registry", required=True, help="Registry for this exact prefix/next block")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--gate-manifest", required=True)
    parser.add_argument("--train-plan", required=True, help="Exact Stage-A-authorized train-plan JSON")
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Progressive artifact directory. It must contain artifacts referenced by the "
            "accepted prefix and receives this block's result/checkpoint and next prefix."
        ),
    )
    parser.add_argument("--device", required=True, help="Explicit torch device, e.g. cuda:0")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue from the existing mutable epoch checkpoint. Without this flag an "
            "existing recovery checkpoint is an error rather than being overwritten."
        ),
    )
    parser.add_argument(
        "--resume-path",
        help=(
            "Optional crash-recovery checkpoint path. By default the campaign uses "
            "<output-dir>/<sweep>.blockNN.training-resume.pt."
        ),
    )
    args = parser.parse_args()

    inputs = load_progressive_block_run_inputs(
        stage_a_result_path=args.stage_a_result,
        current_prefix_manifest_path=args.prefix_manifest,
        capture_registry_path=args.capture_registry,
        dataset_manifest_path=args.dataset_manifest,
        gate_manifest_path=args.gate_manifest,
        train_plan_path=args.train_plan,
    )
    outcome = run_progressive_block_campaign(
        inputs,
        teacher_path=args.teacher,
        artifact_dir=args.output_dir,
        device=args.device,
        resume=args.resume,
        resume_path=args.resume_path,
    )

    print(f"Progressive block: {outcome.block_index}")
    print(f"Numerical gate passed: {outcome.gate_passed}")
    print(f"Candidate checkpoint: {outcome.artifact.checkpoint_path}")
    print(f"Candidate checkpoint SHA-256: {outcome.artifact.checkpoint_sha256}")
    print(f"Candidate result: {outcome.artifact.result_path}")
    print(f"Candidate result SHA-256: {outcome.artifact.result_sha256}")
    if outcome.accepted is None:
        print("Accepted prefix unchanged: candidate failed the frozen numerical gate")
        return 2

    print(f"Next prefix manifest: {outcome.accepted.prefix_manifest_path}")
    print(f"Next prefix manifest SHA-256: {outcome.accepted.prefix_manifest_sha256}")
    print(f"Accepted prefix identity SHA-256: {outcome.accepted.prefix.identity_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
