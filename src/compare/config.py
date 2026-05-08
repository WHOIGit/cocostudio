"""Configuration parsing for COCO comparison rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.config import (
    load_toml,
    namespace_overrides,
    normalize_int_list,
    normalize_list,
    parse_optional_template,
)


class ConfigError(ValueError):
    pass


MASK_MODES = ("none", "fill", "outline", "both")
INVERT_MASKS = ("none", "a", "b", "both")
IMAGE_INDEX_SIDES = ("a", "b")


DEFAULTS: dict[str, Any] = {
    "coco_a": None,
    "coco_b": None,
    "name_a": "a",
    "name_b": "b",
    "image_root": "data/images",
    "output": "data/compare/rendered/{FFID}.jpg",
    "image_key": "file_name",
    "ffid_field": "file_name",
    "ffid_include": None,
    "ffid_exclude": None,
    "image_index": None,
    "image_index_side": "a",
    "draw_bboxes": True,
    "mask_mode": "none",
    "invert_masks": "none",
    "text": None,
    "color_a": "cyan",
    "color_b": "red",
    "label_color": None,
    "mask_alpha": 0.28,
    "line_width": 3,
    "clobber": False,
    "dry_run": False,
    "provenance": "data/logs/compare_render_{DATETIME}.json",
}

ALLOWED_KEYS = set(DEFAULTS)


@dataclass(frozen=True)
class RenderConfig:
    coco_a: Path
    coco_b: Path
    name_a: str
    name_b: str
    image_root: Path
    output: str
    image_key: str
    ffid_field: str
    ffid_include: list[str] | None
    ffid_exclude: list[str] | None
    image_index: list[int] | None
    image_index_side: str
    draw_bboxes: bool
    mask_mode: str
    invert_masks: str
    text: str | None
    color_a: str
    color_b: str
    label_colors: dict[str, str]
    mask_alpha: float
    line_width: int
    clobber: bool
    dry_run: bool
    provenance: str | None


def build_render_config(args: Any) -> RenderConfig:
    values = DEFAULTS.copy()
    values.update(load_toml(getattr(args, "config", None), ALLOWED_KEYS, ConfigError))
    values.update(namespace_overrides(args))
    return render_config_from_values(values)


def render_config_from_values(values: dict[str, Any]) -> RenderConfig:
    coco_a = values.get("coco_a")
    coco_b = values.get("coco_b")
    if not coco_a:
        raise ConfigError("--coco-a is required unless provided by -c/--config")
    if not coco_b:
        raise ConfigError("--coco-b is required unless provided by -c/--config")

    ffid_include = normalize_list(values.get("ffid_include"))
    ffid_exclude = normalize_list(values.get("ffid_exclude"))
    if ffid_include and ffid_exclude:
        raise ConfigError("ffid_include and ffid_exclude are mutually exclusive")

    image_index_side = str(values.get("image_index_side") or "a").lower()
    if image_index_side not in IMAGE_INDEX_SIDES:
        raise ConfigError(
            f"image_index_side must be one of: {', '.join(IMAGE_INDEX_SIDES)}"
        )

    mask_mode = str(values.get("mask_mode") or "none").lower()
    if mask_mode not in MASK_MODES:
        raise ConfigError(f"mask_mode must be one of: {', '.join(MASK_MODES)}")

    invert_masks = str(values.get("invert_masks") or "none").lower()
    if invert_masks not in INVERT_MASKS:
        raise ConfigError(f"invert_masks must be one of: {', '.join(INVERT_MASKS)}")

    mask_alpha = float(values.get("mask_alpha"))
    if not 0.0 <= mask_alpha <= 1.0:
        raise ConfigError("mask_alpha must be between 0 and 1")

    line_width = int(values.get("line_width"))
    if line_width < 1:
        raise ConfigError("line_width must be >= 1")

    output = str(values.get("output") or "")
    if not output:
        raise ConfigError("--output is required unless provided by -c/--config")

    return RenderConfig(
        coco_a=Path(str(coco_a)),
        coco_b=Path(str(coco_b)),
        name_a=str(values.get("name_a") or "a"),
        name_b=str(values.get("name_b") or "b"),
        image_root=Path(str(values.get("image_root") or "data/images")),
        output=output,
        image_key=str(values.get("image_key") or "file_name"),
        ffid_field=str(values.get("ffid_field") or "file_name"),
        ffid_include=ffid_include,
        ffid_exclude=ffid_exclude,
        image_index=normalize_int_list(values.get("image_index")),
        image_index_side=image_index_side,
        draw_bboxes=bool(values.get("draw_bboxes")),
        mask_mode=mask_mode,
        invert_masks=invert_masks,
        text=parse_optional_template(values.get("text")),
        color_a=str(values.get("color_a") or "cyan"),
        color_b=str(values.get("color_b") or "red"),
        label_colors=parse_label_colors(values.get("label_color")),
        mask_alpha=mask_alpha,
        line_width=line_width,
        clobber=bool(values.get("clobber")),
        dry_run=bool(values.get("dry_run")),
        provenance=parse_optional_template(values.get("provenance")),
    )


def parse_label_colors(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k).lower(): str(v) for k, v in value.items()}

    items = value if isinstance(value, list) else [value]
    parsed: dict[str, str] = {}
    for item in items:
        text = str(item)
        if "=" not in text:
            raise ConfigError("label_color entries must be CATEGORY=COLOR")
        label, color = text.split("=", 1)
        label = label.strip()
        color = color.strip()
        if not label or not color:
            raise ConfigError("label_color entries must be CATEGORY=COLOR")
        parsed[label.lower()] = color
    return parsed

