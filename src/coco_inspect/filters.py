"""Image and annotation filtering for COCO inspection."""

from __future__ import annotations

import argparse
from typing import Any, Iterable

from common.config import normalize_list

from .coco import annotation_category_terms, image_identifiers, known_category_terms
from .types import CocoData, Filters, InspectError


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
