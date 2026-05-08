#!/usr/bin/env python3
"""Inspect COCO JSON files from the command line."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import tabulate as tabulate_pkg

from common.config import normalize_list

TABLE_FORMATS = tuple(sorted(tabulate_pkg._table_formats))
SCHEMA_MODES = ("all", "extras")

TOP_LEVEL_STANDARD = ("info", "licenses", "images", "annotations", "categories")
INFO_STANDARD = ("year", "version", "description", "contributor", "url", "date_created")
LICENSE_STANDARD = ("id", "name", "url")
IMAGE_STANDARD = (
    "id",
    "width",
    "height",
    "file_name",
    "license",
    "flickr_url",
    "coco_url",
    "date_captured",
)
CATEGORY_STANDARD = ("id", "name", "supercategory", "keypoints", "skeleton")
ANNOTATION_STANDARD = (
    "id",
    "image_id",
    "category_id",
    "bbox",
    "area",
    "iscrowd",
    "segmentation",
    "keypoints",
    "num_keypoints",
    "caption",
)


class InspectError(ValueError):
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except InspectError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect one COCO JSON file")
    parser.add_argument("coco_json", help="COCO JSON file")
    parser.add_argument(
        "--include",
        nargs="+",
        help=(
            "Include only images matching one of image's id, file_name, "
            "basename, or filename stem."
        ),
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help=(
            "Exclude images matching one of image's id, file_name, "
            "basename, or filename stem."
        ),
    )
    parser.add_argument(
        "--include-category",
        nargs="+",
        help="Include only annotations with these category ids or names",
    )
    parser.add_argument(
        "--exclude-category",
        nargs="+",
        help="Exclude annotations with these category ids or names",
    )
    parser.add_argument(
        "--sum",
        nargs="*",
        metavar="FIELD",
        default=None,
        help=(
            "Only output the SUM row. Optionally provide fields to extract "
            "numeric values: one field prints a raw value; multiple fields "
            "print a field/value table. Fields: images, annotations, bbox, "
            "masks, or category name/id."
        ),
    )
    parser.add_argument(
        "--tablefmt",
        choices=TABLE_FORMATS,
        default="github",
        metavar="TABLEFMT",
        help=(
            "tabulate table format. Default: github. Available: "
            + ", ".join(TABLE_FORMATS)
        ),
    )
    parser.add_argument("-o", "--outfile", help="Write output to this file")
    parser.add_argument("--no-dims", action="store_true", help="Remove width and height columns")
    parser.add_argument("--no-meta", action="store_true", help="Remove extra image metadata columns")
    parser.add_argument("--no-anno", action="store_true", help="Remove annotation count/type/category columns")
    parser.add_argument("--no-anno-types", action="store_true", help="Remove bbox and masks columns")
    parser.add_argument("--no-cats", action="store_true", help="Remove category count columns")
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show the top-level COCO info object as a key/value table.",
    )
    parser.add_argument(
        "--schema",
        nargs="?",
        const="all",
        choices=SCHEMA_MODES,
        help=(
            "Show COCO dict-key structure. Use '--schema extras' to show only "
            "non-standard keys."
        ),
    )
    parser.add_argument(
        "--segfmt",
        action="store_true",
        help=(
            "Report selected annotation segmentation format: polygon, rle, "
            "rle-compressed, None, or JSON counts when mixed."
        ),
    )
    parser.add_argument(
        "--images", "--list",
        action="store_true",
        dest="list_images",
        help='List selected images as "ID: filename" lines, with no header.',
    )
    parser.add_argument(
        "--categories", "--labels",
        action="store_true",
        dest='list_categories',
        help='List selected categories as "ID: label" lines, with no header.',
    )
    return parser


def run(args: argparse.Namespace) -> int:
    coco = load_coco(Path(args.coco_json))
    filters = build_filters(args)

    mode_count = sum(
        bool(value)
        for value in (
            args.sum is not None,
            args.info,
            args.schema,
            args.segfmt,
            args.list_images,
            args.list_categories,
        )
    )
    if mode_count > 1:
        raise InspectError(
            "--sum, --info, --schema, --segfmt, --images/--list, "
            "and --categories/--labels are mutually exclusive"
        )

    if args.info:
        headers, rows = build_info_table(coco)
        write_table(headers, rows, args)
        return 0

    if args.schema:
        lines = render_schema(coco, extras_only=args.schema == "extras")
        write_text_lines(lines, args.outfile)
        return 0

    validate_filters(coco, filters)

    if args.segfmt:
        write_text_lines([segmentation_format_summary(coco, filters)], args.outfile)
        return 0

    if args.list_categories:
        lines = [
            f"{cat_id:02}: {label}"
            for cat_id, label in selected_category_labels(coco, filters)
        ]
        write_text_lines(lines, args.outfile)
        return 0

    summaries = build_image_summaries(coco, filters)

    if args.list_images:
        lines = [
            f"{int(summary.image['id']):02}: {summary.image.get('file_name', '')}"
            for summary in summaries
        ]
        write_text_lines(lines, args.outfile)
        return 0

    if args.sum:
        write_sum_fields(coco, summaries, args.sum, args)
        return 0

    headers, rows = build_table(coco, summaries, filters, args)
    write_table(headers, rows, args)
    return 0


def load_coco(path: Path) -> CocoData:
    try:
        with path.open() as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise InspectError(f"COCO file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise InspectError(f"Invalid JSON in {path}: {e}") from e

    info = data.get("info", {})
    licenses = data.get("licenses", [])
    images = data.get("images")
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    if info is None:
        info = {}
    if not isinstance(info, dict):
        raise InspectError(f"{path}: info must be an object when present")
    if licenses is None:
        licenses = []
    if not isinstance(licenses, list):
        raise InspectError(f"{path}: licenses must be a list when present")
    if not isinstance(images, list):
        raise InspectError(f"{path}: COCO JSON must contain an images list")
    if not isinstance(annotations, list):
        raise InspectError(f"{path}: annotations must be a list")
    if not isinstance(categories, list):
        raise InspectError(f"{path}: categories must be a list")

    seen_image_ids: set[int] = set()
    for image in images:
        if "id" not in image:
            raise InspectError(f"{path}: image missing id")
        image_id = int(image["id"])
        if image_id in seen_image_ids:
            raise InspectError(f"{path}: duplicate image id: {image_id}")
        seen_image_ids.add(image_id)

    categories_by_id = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in categories
        if "id" in category
    }

    annotations_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        if "image_id" not in ann:
            raise InspectError(f"{path}: annotation missing image_id")
        annotations_by_image_id[int(ann["image_id"])].append(ann)
        cat_id = category_id(ann)
        if cat_id is not None and cat_id not in categories_by_id:
            categories_by_id[cat_id] = str(cat_id)

    return CocoData(
        path=path,
        raw=data,
        info=info,
        licenses=licenses,
        images=images,
        categories=categories,
        annotations=annotations,
        categories_by_id=categories_by_id,
        annotations_by_image_id=dict(annotations_by_image_id),
    )


def build_filters(args: argparse.Namespace) -> Filters:
    return Filters(
        include=normalize_list(args.include),
        exclude=normalize_list(args.exclude),
        include_category=normalize_list(args.include_category),
        exclude_category=normalize_list(args.exclude_category),
    )


def validate_filters(coco: CocoData, filters: Filters) -> None:
    if filters.include and filters.exclude:
        overlap = set(filters.include) & set(filters.exclude)
        if overlap:
            raise InspectError(f"include and exclude overlap: {', '.join(sorted(overlap))}")

    if filters.include_category and filters.exclude_category:
        overlap = {item.lower() for item in filters.include_category} & {
            item.lower() for item in filters.exclude_category
        }
        if overlap:
            raise InspectError(
                f"include_category and exclude_category overlap: {', '.join(sorted(overlap))}"
            )

    if filters.include:
        validate_image_filter_terms(coco, filters.include, "include")
    if filters.include_category:
        validate_category_filter_terms(coco, filters.include_category, "include_category")


def validate_image_filter_terms(coco: CocoData, terms: list[str], label: str) -> None:
    missing = []
    for term in terms:
        if not any(term in image_identifiers(image) for image in coco.images):
            missing.append(term)
    if missing:
        raise InspectError(f"{label} value(s) did not match any images: {', '.join(missing)}")


def validate_category_filter_terms(coco: CocoData, terms: list[str], label: str) -> None:
    category_terms = known_category_terms(coco)
    missing = [term for term in terms if term.lower() not in category_terms]
    if missing:
        raise InspectError(f"{label} value(s) did not match any categories: {', '.join(missing)}")


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


def selected_images(coco: CocoData, filters: Filters) -> list[dict[str, Any]]:
    images = []
    include = set(filters.include or [])
    exclude = set(filters.exclude or [])
    for image in coco.images:
        identifiers = image_identifiers(image)
        if include and identifiers.isdisjoint(include):
            continue
        if exclude and not identifiers.isdisjoint(exclude):
            continue
        images.append(image)
    return images


def filtered_annotations(
    annotations: Iterable[dict[str, Any]],
    coco: CocoData,
    filters: Filters,
) -> list[dict[str, Any]]:
    anns = []
    include_category = {item.lower() for item in filters.include_category or []}
    exclude_category = {item.lower() for item in filters.exclude_category or []}

    for ann in annotations:
        terms = annotation_category_terms(ann, coco)
        if include_category and terms.isdisjoint(include_category):
            continue
        if exclude_category and not terms.isdisjoint(exclude_category):
            continue
        anns.append(ann)
    return anns


def selected_annotations(coco: CocoData, filters: Filters) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for image in selected_images(coco, filters):
        image_id = int(image["id"])
        annotations.extend(
            filtered_annotations(
                coco.annotations_by_image_id.get(image_id, []),
                coco,
                filters,
            )
        )
    return annotations


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


def build_table(
    coco: CocoData,
    summaries: list[ImageSummary],
    filters: Filters,
    args: argparse.Namespace,
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

    if args.sum is not None:
        kept = [idx for idx, value in enumerate(sum_row) if value != ""]
        if csv_outfile(args):
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


def write_sum_fields(
    coco: CocoData,
    summaries: list[ImageSummary],
    fields: list[str],
    args: argparse.Namespace,
) -> None:
    values = numeric_summary_values(coco, summaries)
    resolved: list[tuple[str, int]] = []
    for field in fields:
        key = resolve_summary_field(field, values)
        if key is None:
            choices = ", ".join(summary_field_choices(coco))
            raise InspectError(f"Unknown --sum field {field!r}. Choices: {choices}")
        resolved.append((field, values[key]))

    if len(resolved) == 1:
        write_text_lines([str(resolved[0][1])], args.outfile)
    else:
        write_table(
            ["field", "value"],
            [[field, value] for field, value in resolved],
            args,
        )


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


def selected_category_labels(coco: CocoData, filters: Filters) -> list[tuple[int, str]]:
    include = {item.lower() for item in filters.include_category or []}
    exclude = {item.lower() for item in filters.exclude_category or []}
    selected = []
    for cat_id, label in sorted(coco.categories_by_id.items()):
        terms = {str(cat_id).lower(), label.lower()}
        if include and terms.isdisjoint(include):
            continue
        if exclude and not terms.isdisjoint(exclude):
            continue
        selected.append((cat_id, label))
    return selected


def build_info_table(coco: CocoData) -> tuple[list[str], list[list[Any]]]:
    rows = [
        [key, value]
        for key, value in flatten_dict(coco.info).items()
    ]
    if not rows:
        rows = [["info", ""]]
    return ["field", "value"], rows


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in data.items():
        field = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            nested = flatten_dict(value, field)
            if nested:
                values.update(nested)
            else:
                values[field] = "{}"
        else:
            values[field] = display_value(value)
    return values


def render_schema(coco: CocoData, *, extras_only: bool) -> list[str]:
    nodes = build_schema(coco, extras_only=extras_only)
    if not nodes:
        return ["No non-standard schema keys found."] if extras_only else []

    lines: list[str] = []
    for node in nodes:
        lines.extend(render_schema_node(node))
    return lines


def build_schema(coco: CocoData, *, extras_only: bool) -> list[SchemaNode]:
    nodes: list[SchemaNode] = []

    section_specs = [
        ("info", coco.info, "dict", INFO_STANDARD),
        ("licenses", coco.licenses, "list[dict]", LICENSE_STANDARD),
        ("categories", coco.categories, "list[dict]", CATEGORY_STANDARD),
        ("images", coco.images, "list[dict]", IMAGE_STANDARD),
        ("annotations", coco.annotations, "list[dict]", ANNOTATION_STANDARD),
    ]

    for name, value, kind, standard_keys in section_specs:
        if kind == "dict":
            node = schema_from_dict(
                name,
                value if isinstance(value, dict) else {},
                standard_keys,
                extras_only=extras_only,
            )
        else:
            node = schema_from_list(
                name,
                value if isinstance(value, list) else [],
                standard_keys,
                extras_only=extras_only,
            )
        if keep_schema_section(node, extras_only):
            nodes.append(node)

    top_extras = [
        key
        for key in coco.raw
        if key not in set(TOP_LEVEL_STANDARD)
    ]
    for key in top_extras:
        if not extras_only:
            nodes.append(schema_from_value(key, coco.raw[key]))
        else:
            nodes.append(schema_from_value(key, coco.raw[key]))

    return nodes


def schema_from_dict(
    name: str,
    data: dict[str, Any],
    standard_keys: tuple[str, ...] = (),
    *,
    extras_only: bool,
) -> SchemaNode:
    node = SchemaNode(name=name, kind="dict")
    standard = set(standard_keys)
    children: list[SchemaNode] = []
    for key, value in data.items():
        if extras_only and key in standard:
            continue
        children.append(schema_from_value(key, value))
    node.children = order_schema_children(children, standard_keys)
    return node


def schema_from_list(
    name: str,
    data: list[Any],
    standard_keys: tuple[str, ...] = (),
    *,
    extras_only: bool,
) -> SchemaNode:
    node = SchemaNode(name=name, kind="list[dict]", total=len(data))
    records = [item for item in data if isinstance(item, dict)]
    key_counts: Counter[str] = Counter()
    values_by_key: dict[str, list[Any]] = defaultdict(list)
    first_seen: list[str] = []
    seen: set[str] = set()

    for record in records:
        for key, value in record.items():
            if key not in seen:
                seen.add(key)
                first_seen.append(key)
            key_counts[key] += 1
            values_by_key[key].append(value)

    standard = set(standard_keys)
    children: list[SchemaNode] = []
    for key in first_seen:
        if extras_only and key in standard:
            continue
        child = schema_from_values(
            key,
            values_by_key[key],
            count=key_counts[key],
            count_base=len(data),
        )
        children.append(child)
    node.children = order_schema_children(children, standard_keys)
    return node


def schema_from_values(
    name: str,
    values: list[Any],
    *,
    count: int | None,
    count_base: int | None,
) -> SchemaNode:
    dicts = [value for value in values if isinstance(value, dict)]
    list_dicts = [
        item
        for value in values
        if isinstance(value, list)
        for item in value
        if isinstance(item, dict)
    ]

    if dicts:
        node = SchemaNode(name=name, kind="dict", count=count, count_base=count_base)
        node.children = schema_children_from_dicts(dicts)
        return node

    if list_dicts:
        node = SchemaNode(
            name=name,
            kind="list[dict]",
            total=sum(len(value) for value in values if isinstance(value, list)),
            count=count,
            count_base=count_base,
        )
        node.children = schema_children_from_dicts(list_dicts)
        return node

    return SchemaNode(name=name, count=count, count_base=count_base)


def schema_from_value(name: str, value: Any) -> SchemaNode:
    if isinstance(value, dict):
        node = SchemaNode(name=name, kind="dict")
        node.children = order_schema_children(
            [schema_from_value(key, val) for key, val in value.items()],
            (),
        )
        return node
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        node = SchemaNode(name=name, kind="list[dict]", total=len(value))
        node.children = schema_children_from_dicts(value)
        return node
    if isinstance(value, list):
        return SchemaNode(name=name, kind="list", total=len(value))
    return SchemaNode(name=name, kind=type(value).__name__)


def schema_children_from_dicts(dicts: list[dict[str, Any]]) -> list[SchemaNode]:
    key_counts: Counter[str] = Counter()
    values_by_key: dict[str, list[Any]] = defaultdict(list)
    first_seen: list[str] = []
    seen: set[str] = set()

    for data in dicts:
        for key, value in data.items():
            if key not in seen:
                seen.add(key)
                first_seen.append(key)
            key_counts[key] += 1
            values_by_key[key].append(value)

    children = [
        schema_from_values(
            key,
            values_by_key[key],
            count=key_counts[key],
            count_base=len(dicts),
        )
        for key in first_seen
    ]
    return order_schema_children(children, ())


def order_schema_children(
    children: list[SchemaNode],
    standard_keys: tuple[str, ...],
) -> list[SchemaNode]:
    standard_index = {key: index for index, key in enumerate(standard_keys)}
    first_index = {id(child): index for index, child in enumerate(children)}

    standard_children = sorted(
        [child for child in children if child.name in standard_index],
        key=lambda child: standard_index[child.name],
    )
    nonstandard = [child for child in children if child.name not in standard_index]
    nonstandard_plain = [
        child for child in nonstandard
        if not schema_node_is_dict_like(child)
    ]
    nonstandard_dicts = sorted(
        [child for child in nonstandard if schema_node_is_dict_like(child)],
        key=lambda child: (-schema_child_key_count(child), first_index[id(child)]),
    )
    return standard_children + nonstandard_plain + nonstandard_dicts


def schema_node_is_dict_like(node: SchemaNode) -> bool:
    return node.kind in {"dict", "list[dict]"}


def schema_child_key_count(node: SchemaNode) -> int:
    return len(node.children)


def keep_schema_section(node: SchemaNode, extras_only: bool) -> bool:
    if not extras_only:
        return True
    return bool(node.children)


def render_schema_node(node: SchemaNode, depth: int = 0) -> list[str]:
    prefix = "" if depth == 0 else f"{' ' * (depth * 2 - 1)}- "
    lines = [f"{prefix}{schema_node_label(node)}"]
    for child in node.children:
        lines.extend(render_schema_node(child, depth + 1))
    return lines


def schema_node_label(node: SchemaNode) -> str:
    label = node.name
    if node.kind == "dict":
        label += ": DICT"
    elif node.kind == "list[dict]":
        total = 0 if node.total is None else node.total
        label += f": LIST[DICT] ({total})"
    elif node.kind == "list":
        total = 0 if node.total is None else node.total
        label += f": LIST ({total})"

    if node.count is not None and node.count_base is not None and node.count != node.count_base:
        label += f" ({node.count})"
    return label


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


def has_mask(ann: dict[str, Any]) -> bool:
    return segmentation_format(ann.get("segmentation")) is not None


def image_identifiers(image: dict[str, Any]) -> set[str]:
    identifiers = {str(image["id"])}
    file_name = image.get("file_name")
    if file_name:
        path = Path(str(file_name))
        identifiers.update({str(file_name), path.name, path.stem})
    return identifiers


def category_id(ann: dict[str, Any]) -> int | None:
    value = ann.get("category_id")
    if value is None:
        return None
    return int(value)


def category_name(cat_id: int | None, coco: CocoData) -> str:
    if cat_id is None:
        return "none"
    return coco.categories_by_id.get(cat_id, str(cat_id))


def annotation_category_terms(ann: dict[str, Any], coco: CocoData) -> set[str]:
    cat_id = category_id(ann)
    terms = {"" if cat_id is None else str(cat_id).lower()}
    if cat_id is not None:
        terms.add(category_name(cat_id, coco).lower())
    return terms


def known_category_terms(coco: CocoData) -> set[str]:
    terms: set[str] = set()
    for cat_id, name in coco.categories_by_id.items():
        terms.add(str(cat_id).lower())
        terms.add(name.lower())
    return terms


def display_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True)


def write_table(headers: list[str], rows: list[list[Any]], args: argparse.Namespace) -> None:
    outfile = Path(args.outfile) if args.outfile else None
    if csv_outfile(args):
        outfile.parent.mkdir(parents=True, exist_ok=True)
        with outfile.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        return

    text = tabulate_pkg.tabulate(rows, headers=headers, tablefmt=args.tablefmt)
    if outfile:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_text(text + "\n")
    else:
        print(text)


def csv_outfile(args: argparse.Namespace) -> bool:
    return bool(args.outfile) and Path(args.outfile).suffix.lower() == ".csv"


def write_text_lines(lines: list[str], outfile: str | None) -> None:
    text = "\n".join(lines)
    if text:
        text += "\n"
    if outfile:
        path = Path(outfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    raise SystemExit(main())
