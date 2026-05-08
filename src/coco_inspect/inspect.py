"""Computed facts and table rows for COCO inspection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from .coco import category_id, category_name
from .filters import filtered_annotations, selected_images
from .segfmt import has_mask
from .types import CocoData, Filters, ImageSummary


def build_image_summaries(coco: CocoData, filters: Filters) -> list[ImageSummary]:
    summaries: list[ImageSummary] = []
    category_filter_active = bool(filters.include_category or filters.exclude_category)

    for image in selected_images(coco, filters):
        image_id = int(image["id"])
        anns = filtered_annotations(coco.annotations_by_image_id.get(image_id, []), coco, filters)
        if category_filter_active and not anns:
            continue
        summaries.append(
            ImageSummary(
                image=image,
                annotations=anns,
                category_counts=Counter(category_id(ann) for ann in anns),
                bbox_count=sum(1 for ann in anns if has_bbox(ann)),
                mask_count=sum(1 for ann in anns if has_mask(ann)),
            )
        )
    return summaries


def build_table(
    coco: CocoData,
    summaries: list[ImageSummary],
    filters: Filters,
    args: argparse.Namespace,
    *,
    sum_only: bool = False,
    csv_output: bool = False,
) -> tuple[list[str], list[list[Any]]]:
    meta_fields = [] if args.no_meta else image_metadata_fields(summaries)
    category_columns = [] if args.no_anno or args.no_cats else table_categories(coco, filters)

    headers = ["image_id", "filename"]
    if not args.no_dims:
        headers.extend(["width", "height"])
    headers.extend(meta_fields)
    if not args.no_anno:
        headers.append("annotations")
        if not args.no_anno_types:
            headers.extend(["bbox", "masks"])
        headers.extend(category_header(cat_id, coco) for cat_id in category_columns)

    rows = [
        image_summary_row(summary, headers, meta_fields, category_columns, coco, args)
        for summary in summaries
    ]
    sum_row = summary_row(summaries, headers, category_columns, coco, args)

    if sum_only:
        kept = [idx for idx, value in enumerate(sum_row) if value != ""]
        if csv_output:
            return [headers[idx] for idx in kept], [[sum_row[idx] for idx in kept]]
        rows = [["images", len(summaries)]]
        rows.extend(
            [headers[idx], sum_row[idx]]
            for idx in kept
            if headers[idx] not in {"image_id", "filename"}
        )
        return ["field", "value"], rows

    return headers, rows + [sum_row]


def image_summary_row(
    summary: ImageSummary,
    headers: list[str],
    meta_fields: list[str],
    category_columns: list[int | None],
    coco: CocoData,
    args: argparse.Namespace,
) -> list[Any]:
    image = summary.image
    ann_count = len(summary.annotations)
    values: dict[str, Any] = {
        "image_id": image["id"],
        "filename": image.get("file_name", ""),
        "width": image.get("width", ""),
        "height": image.get("height", ""),
        "annotations": ann_count,
        "bbox": format_presence_count(summary.bbox_count, ann_count),
        "masks": format_presence_count(summary.mask_count, ann_count),
    }
    values.update(flatten_image_metadata(image))
    for cat_id in category_columns:
        values[category_header(cat_id, coco)] = summary.category_counts.get(cat_id, 0)

    return [values.get(header, "") for header in headers]


def summary_row(
    summaries: list[ImageSummary],
    headers: list[str],
    category_columns: list[int | None],
    coco: CocoData,
    args: argparse.Namespace,
) -> list[Any]:
    image_count = len(summaries)
    ann_count = sum(len(summary.annotations) for summary in summaries)
    bbox_count = sum(summary.bbox_count for summary in summaries)
    mask_count = sum(summary.mask_count for summary in summaries)
    total_categories: Counter[int | None] = Counter()
    for summary in summaries:
        total_categories.update(summary.category_counts)

    values: dict[str, Any] = {
        "image_id": "SUM",
        "filename": f"{image_count} images",
        "annotations": ann_count,
        "bbox": format_presence_count(bbox_count, ann_count),
        "masks": format_presence_count(mask_count, ann_count),
    }
    for cat_id in category_columns:
        values[category_header(cat_id, coco)] = total_categories.get(cat_id, 0)

    return [values.get(header, "") for header in headers]


def numeric_summary_values(
    coco: CocoData,
    summaries: list[ImageSummary],
) -> dict[str, int]:
    total_categories: Counter[int | None] = Counter()
    for summary in summaries:
        total_categories.update(summary.category_counts)

    values: dict[str, int] = {
        "images": len(summaries),
        "annotations": sum(len(summary.annotations) for summary in summaries),
        "bbox": sum(summary.bbox_count for summary in summaries),
        "masks": sum(summary.mask_count for summary in summaries),
    }

    for cat_id, label in coco.categories_by_id.items():
        value = total_categories.get(cat_id, 0)
        values[f"category:{cat_id}"] = value
        values[f"category-name:{label.lower()}"] = value
        values[f"category-header:{cat_id}:{label.lower()}"] = value
    return values


def resolve_summary_field(field: str, values: dict[str, int]) -> str | None:
    normalized = field.strip().lower()
    if normalized in values:
        return normalized
    if f"category:{normalized}" in values:
        return f"category:{normalized}"
    if f"category-name:{normalized}" in values:
        return f"category-name:{normalized}"
    if ":" in normalized and f"category-header:{normalized}" in values:
        return f"category-header:{normalized}"
    return None


def summary_field_choices(coco: CocoData) -> list[str]:
    choices = ["images", "annotations", "bbox", "masks"]
    choices.extend(label for _, label in sorted(coco.categories_by_id.items()))
    return choices


def image_metadata_fields(summaries: list[ImageSummary]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for summary in summaries:
        for key in flatten_image_metadata(summary.image):
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def flatten_image_metadata(image: dict[str, Any]) -> dict[str, Any]:
    excluded = {"id", "file_name", "width", "height"}
    values: dict[str, Any] = {}
    for key, value in image.items():
        if key in excluded:
            continue
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                values[f"{key}.{subkey}"] = display_value(subvalue)
        else:
            values[key] = display_value(value)
    return values


def table_categories(coco: CocoData, filters: Filters) -> list[int | None]:
    include = {item.lower() for item in filters.include_category or []}
    exclude = {item.lower() for item in filters.exclude_category or []}
    categories: list[int | None] = []
    for cat_id in sorted(coco.categories_by_id):
        terms = {str(cat_id).lower(), coco.categories_by_id[cat_id].lower()}
        if include and terms.isdisjoint(include):
            continue
        if exclude and not terms.isdisjoint(exclude):
            continue
        categories.append(cat_id)
    return categories


def category_header(cat_id: int | None, coco: CocoData | None) -> str:
    if cat_id is None:
        return "none"
    label = str(cat_id) if coco is None else category_name(cat_id, coco)
    return f"{cat_id}:{label}"


def format_presence_count(count: int, total: int) -> str | int:
    if total == 0:
        return "F"
    if count == total:
        return "T"
    if count == 0:
        return "F"
    return count


def has_bbox(ann: dict[str, Any]) -> bool:
    bbox = ann.get("bbox")
    return isinstance(bbox, list) and len(bbox) == 4


def display_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True)
