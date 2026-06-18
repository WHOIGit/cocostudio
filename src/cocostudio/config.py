"""Loader for ``cocostudio`` push/pull TOML configs (``configs/pushpull/``).

A pushpull config is split into shared top-level keys and per-command sections::

    ls_config         = "..."     # shared
    s3_prefix         = "..."     # shared (optional, push only)
    region_id_pattern = "..."     # shared (push only)

    [push]
    coco          = "..."
    image_dir     = "..."
    ...

    [pull]
    out_coco = "..."
    ...

Every config key is the long form of a CLI flag (``map_bbox`` ↔ ``--map-bbox``),
so :func:`load_push_config` / :func:`load_pull_config` return flat dicts that
the CLI layers under any explicitly-passed flags.
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

PULL_KEYS = (
    "push_coco", "out_coco", "view", "completed_only",
    "annotation_policy", "map_bbox", "map_seg", "label_policy",
    "image_datafield", "image_dir", "download_missing_images",
    "ls_cache", "cache_id", "postprocess_function", "dry_run",
)

# Sections we recognize but do not load here (parsed by their own command).
_KNOWN_SECTIONS = ("push", "pull")


def _load_raw(path: Union[str, os.PathLike]) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _validate_top(path: Union[str, os.PathLike], raw: dict[str, Any],
                  *, command: str, command_keys: tuple[str, ...]) -> dict[str, Any]:
    top = {k: v for k, v in raw.items() if k not in _KNOWN_SECTIONS}
    unknown_top = set(top) - set(SHARED_KEYS)
    if unknown_top:
        raise ValueError(
            f"{path}: unknown top-level key(s) {sorted(unknown_top)}. "
            f"Allowed at top level: {list(SHARED_KEYS)}; per-command keys go "
            f"under [{command}]: {list(command_keys)}")
    return top


def _section(path: Union[str, os.PathLike], raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"{path}: [{name}] must be a table, got {type(section).__name__}")
    return section


def load_push_config(path: Union[str, os.PathLike]) -> dict[str, Any]:
    """Read a pushpull TOML and return the flat dict for the ``push`` command.

    Merges the shared top-level keys with the ``[push]`` section. Only keys
    actually present in the file appear in the result (no defaults injected —
    the CLI supplies those). Unknown top-level keys or ``[push]`` keys, and a
    non-table ``[push]``, raise ``ValueError`` to catch typos and misplacement.
    """
    raw = _load_raw(path)
    push = _section(path, raw, "push")
    top = _validate_top(path, raw, command="push", command_keys=PUSH_KEYS)

    unknown_push = set(push) - set(PUSH_KEYS)
    if unknown_push:
        raise ValueError(
            f"{path}: unknown [push] key(s) {sorted(unknown_push)}. "
            f"Allowed in [push]: {list(PUSH_KEYS)} (shared keys "
            f"{list(SHARED_KEYS)} belong at the top level)")

    return {**top, **push}


def load_pull_config(path: Union[str, os.PathLike]) -> dict[str, Any]:
    """Read a pushpull TOML and return the flat dict for the ``pull`` command.

    Merges shared top-level keys with ``[pull]``. For convenience, ``[push].coco``
    becomes ``push_coco`` and ``[push].image_dir`` becomes ``image_dir`` when the
    corresponding pull key is not explicitly set.
    """
    raw = _load_raw(path)
    pull = _section(path, raw, "pull")
    push = _section(path, raw, "push")
    top = _validate_top(path, raw, command="pull", command_keys=PULL_KEYS)

    unknown_pull = set(pull) - set(PULL_KEYS)
    if unknown_pull:
        raise ValueError(
            f"{path}: unknown [pull] key(s) {sorted(unknown_pull)}. "
            f"Allowed in [pull]: {list(PULL_KEYS)} (shared keys "
            f"{list(SHARED_KEYS)} belong at the top level)")

    # Validate the push section when pull relies on it for defaults. This catches
    # typos without requiring pull users to duplicate source paths.
    unknown_push = set(push) - set(PUSH_KEYS)
    if unknown_push:
        raise ValueError(
            f"{path}: unknown [push] key(s) {sorted(unknown_push)}. "
            f"Allowed in [push]: {list(PUSH_KEYS)}")

    out = {**top, **pull}
    if "push_coco" not in out and push.get("coco"):
        out["push_coco"] = push["coco"]
    if "image_dir" not in out and push.get("image_dir"):
        out["image_dir"] = push["image_dir"]
    return out
