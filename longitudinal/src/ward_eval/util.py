from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


def read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def contains_cjk(value: str) -> bool:
    return bool(CJK_RE.search(value))


def normalize_text(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def is_scoreable_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not contains_cjk(value)


def json_path(parts: list[str | int]) -> str:
    result = "$"
    for part in parts:
        if isinstance(part, int):
            result += "[]"
        else:
            result += f".{part}"
    return result


def leaf_paths(value: Any, parts: list[str | int] | None = None):
    current = parts or []
    if isinstance(value, dict):
        if not value:
            yield json_path(current), value
        for key, child in value.items():
            yield from leaf_paths(child, [*current, key])
    elif isinstance(value, list):
        if not value:
            yield json_path(current), value
        for child in value:
            yield from leaf_paths(child, [*current, 0])
    else:
        yield json_path(current), value
