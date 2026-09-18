#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from minimax_h3_keyless.quantization import export_int8_convrot_from_bf16


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export native ComfyUI INT8 ConvRot from an accepted folded BF16 "
            "h3_keyless_core50_v1 artifact."
        )
    )
    parser.add_argument("source_bf16")
    parser.add_argument("output")
    parser.add_argument("--export-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--manifest")
    parser.add_argument(
        "--source-manifest",
        help="BF16 export receipt; defaults to <source_bf16>.manifest.json",
    )
    args = parser.parse_args()
    command = (
        f"tools/export_int8_convrot.py {args.source_bf16} {args.output} "
        f"--export-commit {args.export_commit} --device {args.device}"
    )
    if args.source_manifest:
        command += f" --source-manifest {args.source_manifest}"
    result = export_int8_convrot_from_bf16(
        args.source_bf16,
        args.output,
        export_commit=args.export_commit,
        quantize_device=args.device,
        manifest_path=args.manifest,
        source_manifest_path=args.source_manifest,
        command=command,
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
