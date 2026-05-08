"""Segmentation format detection for COCO annotations."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .filters import selected_annotations
from .types import CocoData, Filters


def segmentation_format_summary(coco: CocoData, filters: Filters) -> str:
    counts: Counter[str] = Counter()
    for ann in selected_annotations(coco, filters):
        fmt = segmentation_format(ann.get("segmentation"))
        if fmt is not None:
            counts[fmt] += 1

    if not counts:
        return "None"
    if len(counts) == 1:
        return next(iter(counts))
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def segmentation_format(segmentation: Any) -> str | None:
    if segmentation is None:
        return None
    if isinstance(segmentation, list):
        return "polygon" if segmentation else None
    if isinstance(segmentation, dict):
        if not segmentation:
            return None
        counts = segmentation.get("counts")
        if counts in (None, "", []):
            return None
        if isinstance(counts, list):
            return "rle"
        if isinstance(counts, (str, bytes)):
            return "rle-compressed"
        return "unknown"
    return "unknown" if segmentation else None


def has_mask(ann: dict[str, Any]) -> bool:
    return segmentation_format(ann.get("segmentation")) is not None
