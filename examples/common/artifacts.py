"""Hash and atomically persist benchmark evidence."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    """Hash a JSON value independently of formatting."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, default=str).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Replace a JSON artifact only after its complete contents reach disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
