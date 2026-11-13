"""Canonical JSON and SHA-256 helpers shared by selection and packaging (BUILD_PLAN Steps 11-12).

One implementation, so a hash computed when a selection is frozen and a hash checked when a
package is verified always use the same byte representation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Deterministic JSON text: sorted keys, no extra spaces, no NaN or infinity."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
