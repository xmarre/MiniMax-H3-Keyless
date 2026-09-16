#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from minimax_h3_keyless.live_capture import discover_clean_git_revision
from minimax_h3_keyless.pilot_campaign import load_json_manifest
from minimax_h3_keyless.progressive_authorization import (
    authorize_progressive_sweep,
    load_progressive_prefix_manifest,
    write_progressive_prefix_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Authorize the early-to-late Keyless core50 sweep from a passed immutable "
            "Stage-A campaign and write the initial empty accepted-prefix manifest."
        )
    )
    parser.add_argument("--stage-a-result", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--gate-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sweep-id", required=True)
    args = parser.parse_args()

    dataset = load_json_manifest(args.dataset_manifest)
    gate = load_json_manifest(args.gate_manifest)
    code_commit = discover_clean_git_revision(
        Path(__file__).resolve().parents[1],
        label="MiniMax-H3-Keyless progressive sweep source",
    )
    authorization = authorize_progressive_sweep(
        args.stage_a_result,
        dataset_manifest=dataset,
        gate_manifest=gate,
        sweep_id=args.sweep_id,
        code_commit=code_commit,
    )

    output = Path(args.output_dir) / f"{args.sweep_id}.prefix-00.json"
    manifest_sha = write_progressive_prefix_manifest(
        output,
        authorization.prefix,
        previous_manifest_sha256=None,
    )
    loaded, loaded_sha = load_progressive_prefix_manifest(output)
    if loaded != authorization.prefix or loaded_sha.lower() != manifest_sha.lower():
        raise RuntimeError("new progressive authorization manifest changed during validation")

    print(f"Progressive sweep authorized from Stage-A: {authorization.stage_a.path}")
    print(f"Stage-A campaign SHA-256: {authorization.stage_a.sha256}")
    print(f"Initial prefix manifest: {output}")
    print(f"Initial prefix manifest SHA-256: {manifest_sha}")
    print(f"Initial prefix identity SHA-256: {loaded.identity_sha256}")
    print(f"Sweep source revision: {code_commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
