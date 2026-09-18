#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from minimax_h3_keyless.progressive_final_export import export_completed_progressive_bf16


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export a complete accepted MiniMax-H3 Keyless Stage-B prefix as the canonical "
            "h3_keyless_core50_v1 BF16 artifact. The output and sidecar are immutable, "
            "pinned-teacher compatibility is revalidated after writing, and incomplete "
            "prefixes are rejected."
        )
    )
    parser.add_argument("--teacher", required=True, help="Exact pinned native BF16 teacher safetensors")
    parser.add_argument("--prefix-manifest", required=True, help="Complete accepted prefix-50 manifest")
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Directory containing immutable accepted block checkpoint/result artifacts",
    )
    parser.add_argument("--output", required=True, help="Canonical BF16 .safetensors destination")
    parser.add_argument(
        "--manifest-output",
        help="Optional canonical export receipt path; defaults to <output>.manifest.json",
    )
    args = parser.parse_args()

    command = (
        f"tools/export_progressive_bf16.py --teacher {args.teacher} "
        f"--prefix-manifest {args.prefix_manifest} --artifact-dir {args.artifact_dir} "
        f"--output {args.output}"
    )
    if args.manifest_output:
        command += f" --manifest-output {args.manifest_output}"

    result = export_completed_progressive_bf16(
        teacher_path=args.teacher,
        prefix_manifest_path=args.prefix_manifest,
        artifact_dir=args.artifact_dir,
        output_path=args.output,
        manifest_path=args.manifest_output,
        command=command,
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
