#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from minimax_h3_keyless.stage_a_execution_binding import (
    canonical_stage_a_workflow_prompt_sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the canonical Stage-A workflow_prompt_sha256 from a saved Comfy API "
            "prompt JSON. UI-only metadata and Stage-A capture bookkeeping inputs are normalized."
        )
    )
    parser.add_argument("prompt_json", help="Saved Comfy API-format prompt JSON")
    parser.add_argument(
        "--capture-node-id",
        help="Node id of MiniMaxH3StageACapture; inferred only when exactly one is present",
    )
    args = parser.parse_args()

    path = Path(args.prompt_json)
    try:
        prompt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Comfy API prompt JSON: {path}") from exc
    if not isinstance(prompt, dict):
        raise ValueError("Comfy API prompt JSON must decode to an object")

    node_id = args.capture_node_id
    if node_id is None:
        candidates = [
            str(key)
            for key, value in prompt.items()
            if isinstance(value, dict) and value.get("class_type") == "MiniMaxH3StageACapture"
        ]
        if len(candidates) != 1:
            raise ValueError(
                "cannot infer Stage-A capture node id; expected exactly one "
                f"MiniMaxH3StageACapture, found {candidates}"
            )
        node_id = candidates[0]

    digest = canonical_stage_a_workflow_prompt_sha256(
        prompt,
        capture_node_id=node_id,
    )
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
