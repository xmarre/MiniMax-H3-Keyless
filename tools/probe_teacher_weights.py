#!/usr/bin/env python3
from __future__ import annotations

import argparse

from minimax_h3_keyless.contracts import TEACHER_SHA256
from minimax_h3_keyless.probe import run_teacher_probe, write_probe_json


def _indices(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run bounded raw Q/K/V subspace diagnostics on the pinned MiniMax-H3 BF16 teacher. "
            "This is not an activation, generation-quality, or Keyless parity test."
        )
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--layers", type=_indices, default=(0, 25, 49))
    parser.add_argument("--heads", type=_indices, default=(0,))
    parser.add_argument("--expected-sha256", default=TEACHER_SHA256)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--skip-full-sha256",
        action="store_true",
        help="Development-only: record the probe as unverified. Do not use its numbers as canonical evidence.",
    )
    args = parser.parse_args()
    result = run_teacher_probe(
        args.checkpoint,
        layers=args.layers,
        heads=args.heads,
        expected_sha256=args.expected_sha256,
        verify_full_sha256=not args.skip_full_sha256,
    )
    write_probe_json(result, args.output)


if __name__ == "__main__":
    main()
