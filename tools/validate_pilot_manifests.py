#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from minimax_h3_keyless.checkpoint import sha256_file
from minimax_h3_keyless.pilot_campaign import (
    load_json_manifest,
    validate_pilot_dataset_manifest,
    validate_pilot_gate_manifest,
    write_json_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and identify the fixed Stage-A Keyless pilot dataset/gate manifests. "
            "This checks experiment contracts only; it does not establish model parity."
        )
    )
    parser.add_argument("dataset_manifest")
    parser.add_argument("gate_manifest")
    parser.add_argument("--minimum-cases", type=int, default=16)
    parser.add_argument("--minimum-sigma-strata", type=int, default=8)
    parser.add_argument(
        "--required-coverage-tag",
        action="append",
        default=[],
        help="Coverage tag that must occur in at least one complete case; repeat as needed.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON validation receipt. Written atomically; stdout is always emitted.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_manifest)
    gate_path = Path(args.gate_manifest)
    dataset = load_json_manifest(dataset_path)
    gates = load_json_manifest(gate_path)
    dataset_identity = validate_pilot_dataset_manifest(
        dataset,
        minimum_cases=args.minimum_cases,
        minimum_sigma_strata=args.minimum_sigma_strata,
        required_coverage_tags=tuple(args.required_coverage_tag),
    )
    gate_identity = validate_pilot_gate_manifest(gates)
    receipt = {
        "schema": "minimax_h3_keyless_pilot_manifest_validation_v1",
        "dataset_manifest": {
            "path": str(dataset_path),
            "file_sha256": sha256_file(dataset_path),
            "canonical_json_sha256": dataset_identity,
            "case_count": len(dataset["cases"]),
        },
        "gate_manifest": {
            "path": str(gate_path),
            "file_sha256": sha256_file(gate_path),
            "canonical_json_sha256": gate_identity,
        },
        "validation_policy": {
            "minimum_cases": args.minimum_cases,
            "minimum_sigma_strata": args.minimum_sigma_strata,
            "required_coverage_tags": list(args.required_coverage_tag),
        },
    }
    if args.output:
        output_sha = write_json_atomic(args.output, receipt)
        receipt["validation_receipt_path"] = str(Path(args.output))
        receipt["validation_receipt_sha256"] = output_sha
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
