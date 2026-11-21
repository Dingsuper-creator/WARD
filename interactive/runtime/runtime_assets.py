"""Portable configuration and optional clinical vocabulary data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DATA_ROOT: Path | None = None
_VOCABULARY: dict[str, Any] = {}


def configure(data_root: Path, lexicon: Path | None = None) -> None:
    """Configure before importing environment modules; use one process per run."""
    global _DATA_ROOT, _VOCABULARY
    _DATA_ROOT = data_root.resolve()
    path = lexicon if lexicon is not None else _DATA_ROOT / "runtime_lexicon.json"
    _VOCABULARY = {}
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != "ward-runtime-lexicon-1.0":
            raise ValueError("unsupported runtime vocabulary schema")
        _VOCABULARY = data["vocabulary"]
        if not isinstance(_VOCABULARY, dict):
            raise ValueError("vocabulary must be an object")


def vocabulary(key: str, default: Any) -> Any:
    """Read matching terms from data without embedding them in executable source."""
    return _VOCABULARY.get(key, default)


def resolve_asset(value: str, fallback_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (_DATA_ROOT or fallback_root) / path
