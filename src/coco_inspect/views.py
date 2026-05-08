"""Output views for the COCO inspect command."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import tabulate as tabulate_pkg

from . import inspect as inspect_logic
from .types import CocoData, Filters, ImageSummary, InspectError

TABLE_FORMATS = tuple(sorted(tabulate_pkg._table_formats))


def write_default_view(coco: CocoData, filters: Filters, args: argparse.Namespace) -> None:
    summaries = inspect_logic.build_image_summaries(coco, filters)
    headers, rows = inspect_logic.build_table(coco, summaries, filters, args)
    write_table(headers, rows, args)


def write_sum_view(coco: CocoData, filters: Filters, args: argparse.Namespace) -> None:
    summaries = inspect_logic.build_image_summaries(coco, filters)
    if args.sum:
        write_sum_fields(coco, summaries, args.sum, args)
        return

    headers, rows = inspect_logic.build_table(
        coco,
        summaries,
        filters,
        args,
        sum_only=True,
        csv_output=csv_outfile(args),
    )
    write_table(headers, rows, args)


def write_sum_fields(
    coco: CocoData,
    summaries: list[ImageSummary],
    fields: list[str],
    args: argparse.Namespace,
) -> None:
    values = inspect_logic.numeric_summary_values(coco, summaries)
    resolved: list[tuple[str, int]] = []
    for field in fields:
        key = inspect_logic.resolve_summary_field(field, values)
        if key is None:
            choices = ", ".join(inspect_logic.summary_field_choices(coco))
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


def write_info_view(coco: CocoData, args: argparse.Namespace) -> None:
    headers, rows = build_info_table(coco)
    write_table(headers, rows, args)


def write_images_view(coco: CocoData, filters: Filters, args: argparse.Namespace) -> None:
    summaries = inspect_logic.build_image_summaries(coco, filters)
    lines = [
        f"{int(summary.image['id']):02}: {summary.image.get('file_name', '')}"
        for summary in summaries
    ]
    write_text_lines(lines, args.outfile)


def write_categories_view(coco: CocoData, filters: Filters, args: argparse.Namespace) -> None:
    lines = [
        f"{cat_id:02}: {label}"
        for cat_id, label in selected_category_labels(coco, filters)
    ]
    write_text_lines(lines, args.outfile)


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
            values[field] = inspect_logic.display_value(value)
    return values


def write_table(headers: list[str], rows: list[list[Any]], args: argparse.Namespace) -> None:
    outfile = Path(args.outfile) if args.outfile else None
    if csv_outfile(args):
        if outfile is None:
            raise InspectError("CSV output requires --outfile")
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
