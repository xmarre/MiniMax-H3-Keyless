#!/usr/bin/env python3
from __future__ import annotations

import argparse

from minimax_h3_keyless.pilot_campaign import load_json_manifest
from minimax_h3_keyless.stage_a_campaign_result import load_stage_a_campaign_evidence


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate the immutable Stage-A campaign exit from its three bound block "
            "checkpoints/results and predeclared gate policy. Success authorizes Phase 4; "
            "a failed depth pilot is never converted into an authorization by this tool."
        )
    )
    parser.add_argument("--campaign-result", required=True)
    parser.add_argument("--gate-manifest", required=True)
    args = parser.parse_args()

    gate_manifest = load_json_manifest(args.gate_manifest)
    evidence = load_stage_a_campaign_evidence(
        args.campaign_result,
        gate_manifest=gate_manifest,
        require_passed=True,
    )
    print(f"Stage-A campaign result: {evidence.path}")
    print(f"Stage-A campaign SHA-256: {evidence.sha256}")
    print(f"Stage-A campaign gate passed: {evidence.gate.passed}")
    print(f"Authorized next phase: progressive core50 BF16 distillation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
