#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from minimax_h3_keyless.pilot_inputs import CANONICAL_STAGE_A_COVERAGE_TAGS
from minimax_h3_keyless.progressive_capture_registry import build_progressive_capture_registry


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an immutable Stage-B capture registry for one accepted progressive prefix. "
            "Every receipt/bundle is hash-checked one at a time and the set must exactly cover "
            "every case×sigma execution in the fixed dataset manifest."
        )
    )
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--prefix-manifest", required=True)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--receipt-glob",
        default="*.capture.pt.receipt.json",
        help="Recursive receipt glob below --capture-dir (default: %(default)s)",
    )
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir)
    if not capture_dir.is_dir():
        raise NotADirectoryError(
            f"progressive capture directory does not exist: {capture_dir}"
        )
    receipts = sorted(capture_dir.rglob(args.receipt_glob))
    if not receipts:
        raise FileNotFoundError(
            f"no progressive capture receipts matched {args.receipt_glob!r} below {capture_dir}"
        )

    result = build_progressive_capture_registry(
        args.dataset_manifest,
        args.prefix_manifest,
        receipts,
        args.output,
        required_coverage_tags=CANONICAL_STAGE_A_COVERAGE_TAGS,
    )
    print(f"Progressive capture registry: {result.registry_path}")
    print(f"Progressive capture registry SHA-256: {result.registry_sha256}")
    print(f"Prefix manifest SHA-256: {result.prefix_manifest_sha256}")
    print(f"Prefix identity SHA-256: {result.prefix_identity_sha256}")
    print(f"Target block: {result.target_block}")
    print(f"Validated capture artifacts: {result.artifact_count}")
    print(f"Capture code commit: {result.code_commit}")
    print(f"Capture Comfy commit: {result.comfy_commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
