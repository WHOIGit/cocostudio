"""Shared types for COCO inspection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class InspectError(Exception):
    pass


@dataclass(frozen=True)
class CocoData:
    path: Path
    raw: dict[str, Any]
    info: dict[str, Any]
    licenses: list[dict[str, Any]]
    images: list[dict[str, Any]]
    categories: list[dict[str, Any]]
    annotations: list[dict[str, Any]]
    categories_by_id: dict[int, str]
    annotations_by_image_id: dict[int, list[dict[str, Any]]]


@dataclass(frozen=True)
class Filters:
    include: list[str] | None
    exclude: list[str] | None
    include_category: list[str] | None
    exclude_category: list[str] | None


@dataclass(frozen=True)
class ImageSummary:
    image: dict[str, Any]
    annotations: list[dict[str, Any]]
    category_counts: Counter[int | None]
    bbox_count: int
    mask_count: int


@dataclass
class SchemaNode:
    name: str
    kind: str | None = None
    total: int | None = None
    count: int | None = None
    count_base: int | None = None
    children: list["SchemaNode"] = field(default_factory=list)
