#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from minimax_h3_keyless.capture_registry import build_stage_a_capture_registry


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an immutable Stage-A capture registry from live-capture receipts. "
            "Every bundle is hash-checked and validated one at a time, and the receipt set "
            "must exactly cover every case×sigma execution in the fixed dataset manifest."
        )
    )
    parser.add_argument("--dataset-manifest", required=True)
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
        raise NotADirectoryError(f"Stage-A capture directory does not exist: {capture_dir}")
    receipts = sorted(capture_dir.rglob(args.receipt_glob))
    if not receipts:
        raise FileNotFoundError(
            f"no Stage-A capture receipts matched {args.receipt_glob!r} below {capture_dir}"
        )
    result = build_stage_a_capture_registry(
        args.dataset_manifest,
        receipts,
        args.output,
    )
    print(f"Stage-A capture registry: {result.registry_path}")
    print(f"Stage-A capture registry SHA-256: {result.registry_sha256}")
    print(f"Validated capture artifacts: {result.artifact_count}")
    print(f"Capture code commit: {result.code_commit}")
    print(f"Capture Comfy commit: {result.comfy_commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
