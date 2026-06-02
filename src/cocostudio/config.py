"""Loader for ``cocostudio`` push/pull TOML configs (``configs/pushpull/``).

A pushpull config is split into shared top-level keys and per-command sections::

    ls_config         = "..."     # shared
    s3_prefix         = "..."     # shared (optional)
    region_id_pattern = "..."     # shared

    [push]
    coco          = "..."
    image_dir     = "..."
    ...

Every config key is the long form of a CLI flag (``map_bbox`` ↔ ``--map-bbox``),
so :func:`load_push_config` returns a flat dict that the CLI layers under any
explicitly-passed flags. A future ``[pull]`` section will follow the same shape.
"""

from __future__ import annotations

import os
import tomllib
from typing import Any, Union

# Keys that live at the top level, shared by push and (eventually) pull.
SHARED_KEYS = ("ls_config", "s3_prefix", "region_id_pattern")

# Keys that live under the [push] section.
PUSH_KEYS = (
    "coco", "image_dir", "model_version", "map_bbox", "map_seg",
    "polygon_preprocess_epsilon", "polygon_preprocess_maxpoints",
    "include", "exclude", "dry_run",
)

# Sections we recognize but do not load here (parsed by their own command).
_KNOWN_SECTIONS = ("push", "pull")


def load_push_config(path: Union[str, os.PathLike]) -> dict[str, Any]:
    """Read a pushpull TOML and return the flat dict for the ``push`` command.

    Merges the shared top-level keys with the ``[push]`` section. Only keys
    actually present in the file appear in the result (no defaults injected —
    the CLI supplies those). Unknown top-level keys or ``[push]`` keys, and a
    non-table ``[push]``, raise ``ValueError`` to catch typos and misplacement.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    push = raw.get("push", {})
    if not isinstance(push, dict):
        raise ValueError(f"{path}: [push] must be a table, got {type(push).__name__}")

    top = {k: v for k, v in raw.items() if k not in _KNOWN_SECTIONS}
    unknown_top = set(top) - set(SHARED_KEYS)
    if unknown_top:
        raise ValueError(
            f"{path}: unknown top-level key(s) {sorted(unknown_top)}. "
            f"Allowed at top level: {list(SHARED_KEYS)}; per-command keys go "
            f"under [push]: {list(PUSH_KEYS)}")

    unknown_push = set(push) - set(PUSH_KEYS)
    if unknown_push:
        raise ValueError(
            f"{path}: unknown [push] key(s) {sorted(unknown_push)}. "
            f"Allowed in [push]: {list(PUSH_KEYS)} (shared keys "
            f"{list(SHARED_KEYS)} belong at the top level)")

    return {**top, **push}
