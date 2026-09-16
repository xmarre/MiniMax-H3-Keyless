from __future__ import annotations

from pathlib import Path

import pytest

from minimax_h3_keyless.checkpoint import sha256_file
from minimax_h3_keyless.immutable_io import write_json_no_replace


def test_json_no_replace_is_hash_bound_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    digest = write_json_no_replace(path, {"z": 2, "a": [1, 3]})
    assert digest == sha256_file(path)
    assert path.read_text(encoding="utf-8") == '{\n  "a": [\n    1,\n    3\n  ],\n  "z": 2\n}\n'


def test_json_no_replace_preserves_existing_evidence(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    first = write_json_no_replace(path, {"identity": "first"})
    original = path.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        write_json_no_replace(path, {"identity": "second"})
    assert path.read_bytes() == original
    assert sha256_file(path) == first
    assert not list(tmp_path.glob(".*.tmp"))
