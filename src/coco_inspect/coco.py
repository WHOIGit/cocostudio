"""COCO loading and primitive image/category helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .types import CocoData, InspectError


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
