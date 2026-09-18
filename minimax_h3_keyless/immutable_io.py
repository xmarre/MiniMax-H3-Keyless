from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .checkpoint import sha256_file


def write_json_no_replace(path: str | Path, value: Mapping[str, Any]) -> str:
    """Atomically publish deterministic JSON without replacing an existing path.

    The temporary file is created in the destination directory, fsynced, then linked to
    the final name. Hard-link creation is the commit point: it either creates the final
    path or raises ``FileExistsError`` when another writer already owns that immutable
    identity. The temporary link remains until the final SHA-256 has been computed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"immutable JSON output already exists: {path}") from exc
        published = True
        return sha256_file(path)
    except BaseException:
        if published:
            try:
                if path.exists() and temporary.exists() and os.path.samefile(path, temporary):
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
