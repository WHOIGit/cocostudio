"""Shared configuration parsing helpers."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_toml(
    path: str | Path | None,
    allowed_keys: set[str],
    error_cls: type[Exception] = ValueError,
) -> dict[str, Any]:
    if path is None:
        return {}

    with Path(path).open("rb") as f:
        values = tomllib.load(f)

    unknown = sorted(set(values) - allowed_keys)
    if unknown:
        raise error_cls(f"Unknown config key(s): {', '.join(unknown)}")
    return values


def namespace_overrides(
    args: Any,
    *,
    drop: tuple[str, ...] = ("cmd", "config"),
) -> dict[str, Any]:
    values = vars(args).copy()
    for key in drop:
        values.pop(key, None)
    return {key: value for key, value in values.items() if value is not None}


def normalize_list(value: Any) -> list[str] | None:
    if value is None:
        return None

    raw = value if isinstance(value, list) else [value]
    items: list[str] = []
    for item in raw:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                items.append(part)
    return items or None


def normalize_int_list(value: Any) -> list[int] | None:
    items = normalize_list(value)
    if items is None:
        return None
    return [int(item) for item in items]


def parse_optional_template(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None if value is False else str(value)

    text = str(value)
    if text.lower() in {"", "false", "none", "null"}:
        return None
    return text

