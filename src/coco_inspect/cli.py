#!/usr/bin/env python3
"""Inspect COCO JSON files from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import schema, segfmt, views
from .coco import load_coco
from .filters import build_filters, validate_filters
from .types import InspectError


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
        choices=views.TABLE_FORMATS,
        default="github",
        metavar="TABLEFMT",
        help=(
            "tabulate table format. Default: github. Available: "
            + ", ".join(views.TABLE_FORMATS)
        ),
    )
    parser.add_argument("-o", "--outfile", help="Write output to this file")
    parser.add_argument("--no-dims", action="store_true", help="Remove width and height columns")
    parser.add_argument("--no-meta", action="store_true", help="Remove extra image metadata columns")
    parser.add_argument("--no-anno", action="store_true", help="Remove annotation count/type/category columns")
    parser.add_argument("--no-anno-types", action="store_true", help="Remove bbox and masks columns")
    parser.add_argument("--no-cats", action="store_true", help="Remove per-category count columns")
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show the top-level COCO info object as a key/value table.",
    )
    parser.add_argument(
        "--schema",
        nargs="?",
        const="all",
        choices=schema.SCHEMA_MODES,
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
        dest="list_categories",
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
        views.write_info_view(coco, args)
        return 0

    if args.schema:
        views.write_text_lines(
            schema.render_schema(coco, extras_only=args.schema == "extras"),
            args.outfile,
        )
        return 0

    validate_filters(coco, filters)

    if args.segfmt:
        views.write_text_lines([segfmt.segmentation_format_summary(coco, filters)], args.outfile)
        return 0

    if args.list_categories:
        views.write_categories_view(coco, filters, args)
        return 0

    if args.list_images:
        views.write_images_view(coco, filters, args)
        return 0

    if args.sum is not None:
        views.write_sum_view(coco, filters, args)
        return 0

    views.write_default_view(coco, filters, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
