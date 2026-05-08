"""COCO key schema inspection."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .types import CocoData, SchemaNode

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
