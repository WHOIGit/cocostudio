"""Render fullframe graphics from two COCO datasets."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter, ImageFont
from pycocotools import mask as mask_utils
from tqdm import tqdm

from .config import ConfigError, RenderConfig

Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class CocoDataset:
    path: Path
    raw: dict[str, Any]
    images_by_id: dict[int, dict[str, Any]]
    image_key_to_image: dict[str, dict[str, Any]]
    anns_by_image_id: dict[int, list[dict[str, Any]]]
    categories_by_id: dict[int, str]


@dataclass(frozen=True)
class ImagePair:
    key: str
    ffid: str
    image_a: dict[str, Any]
    image_b: dict[str, Any]


def run_render(config: RenderConfig) -> int:
    started_at = datetime.now()
    started_monotonic = monotonic()

    coco_a = load_coco(config.coco_a, config.image_key)
    coco_b = load_coco(config.coco_b, config.image_key)

    pairs, map_summary = build_image_pairs(coco_a, coco_b, config)
    pairs = select_pairs(pairs, coco_a, coco_b, config)
    validate_output_template(config.output, pairs)

    if not pairs:
        print("No common images selected for rendering.")
        return 0

    summary = {
        "images_common": map_summary["common"],
        "images_a_only": map_summary["a_only"],
        "images_b_only": map_summary["b_only"],
        "images_selected": len(pairs),
        "images_rendered": 0,
        "images_skipped": 0,
        "annotations_a": 0,
        "annotations_b": 0,
    }

    for pair in tqdm(pairs, desc="Rendering fullframes", unit="image"):
        out_path = render_output_path(config, pair)
        anns_a = coco_a.anns_by_image_id.get(int(pair.image_a["id"]), [])
        anns_b = coco_b.anns_by_image_id.get(int(pair.image_b["id"]), [])
        summary["annotations_a"] += len(anns_a)
        summary["annotations_b"] += len(anns_b)

        if out_path.exists() and not config.clobber:
            summary["images_skipped"] += 1
            continue

        if config.dry_run:
            summary["images_rendered"] += 1
            continue

        render_pair_image(pair, coco_a, coco_b, config, out_path)
        summary["images_rendered"] += 1

    write_provenance(config, started_at, started_monotonic, summary)
    print(
        "Rendered "
        f"{summary['images_rendered']} image(s); skipped {summary['images_skipped']} existing output(s)."
    )
    return 0


def load_coco(path: Path, image_key: str) -> CocoDataset:
    try:
        with path.open() as f:
            raw = json.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"COCO file not found: {path}") from e

    images = raw.get("images")
    if not isinstance(images, list):
        raise ConfigError(f"{path}: COCO JSON must contain an images list")

    images_by_id: dict[int, dict[str, Any]] = {}
    image_key_to_image: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[int] = []
    duplicate_keys: list[str] = []
    missing_keys: list[Any] = []

    for image in images:
        if "id" not in image:
            raise ConfigError(f"{path}: image missing id")
        image_id = int(image["id"])
        if image_id in images_by_id:
            duplicate_ids.append(image_id)
        images_by_id[image_id] = image

        key_value = image.get(image_key)
        if key_value is None:
            missing_keys.append(image_id)
            continue
        key = str(key_value)
        if key in image_key_to_image:
            duplicate_keys.append(key)
        image_key_to_image[key] = image

    if duplicate_ids:
        raise ConfigError(f"{path}: duplicate image id(s): {preview(duplicate_ids)}")
    if missing_keys:
        raise ConfigError(
            f"{path}: image(s) missing image_key {image_key!r}: {preview(missing_keys)}"
        )
    if duplicate_keys:
        raise ConfigError(
            f"{path}: duplicate image_key {image_key!r} value(s): {preview(duplicate_keys)}"
        )

    anns_by_image_id: dict[int, list[dict[str, Any]]] = {}
    for ann in raw.get("annotations", []):
        image_id = int(ann["image_id"])
        anns_by_image_id.setdefault(image_id, []).append(ann)

    categories_by_id = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in raw.get("categories", [])
        if "id" in category
    }

    return CocoDataset(
        path=path,
        raw=raw,
        images_by_id=images_by_id,
        image_key_to_image=image_key_to_image,
        anns_by_image_id=anns_by_image_id,
        categories_by_id=categories_by_id,
    )


def build_image_pairs(
    coco_a: CocoDataset,
    coco_b: CocoDataset,
    config: RenderConfig,
) -> tuple[list[ImagePair], dict[str, int]]:
    keys_a = set(coco_a.image_key_to_image)
    keys_b = set(coco_b.image_key_to_image)
    common = sorted(keys_a & keys_b)

    pairs = [
        ImagePair(
            key=key,
            ffid=derive_ffid(coco_a.image_key_to_image[key], key, config.ffid_field),
            image_a=coco_a.image_key_to_image[key],
            image_b=coco_b.image_key_to_image[key],
        )
        for key in common
    ]
    return pairs, {
        "common": len(common),
        "a_only": len(keys_a - keys_b),
        "b_only": len(keys_b - keys_a),
    }


def select_pairs(
    pairs: list[ImagePair],
    coco_a: CocoDataset,
    coco_b: CocoDataset,
    config: RenderConfig,
) -> list[ImagePair]:
    selected = pairs

    if config.image_index is not None:
        side = coco_a if config.image_index_side == "a" else coco_b
        requested = set(config.image_index)
        missing_ids = sorted(requested - set(side.images_by_id))
        if missing_ids:
            raise ConfigError(
                f"image_index id(s) not present in COCO {config.image_index_side}: "
                f"{preview(missing_ids)}"
            )
        requested_keys = {
            str(side.images_by_id[image_id][config.image_key])
            for image_id in requested
        }
        common_keys = {pair.key for pair in pairs}
        not_common = sorted(requested_keys - common_keys)
        if not_common:
            raise ConfigError(
                "selected image_index value(s) are not common to both COCO files: "
                f"{preview(not_common)}"
            )
        selected = [pair for pair in selected if pair.key in requested_keys]

    if config.ffid_include is not None:
        include = set(config.ffid_include)
        selected = [pair for pair in selected if pair.ffid in include]
    elif config.ffid_exclude is not None:
        exclude = set(config.ffid_exclude)
        selected = [pair for pair in selected if pair.ffid not in exclude]

    return selected


def render_pair_image(
    pair: ImagePair,
    coco_a: CocoDataset,
    coco_b: CocoDataset,
    config: RenderConfig,
    out_path: Path,
) -> None:
    image_path = resolve_image_path(pair, config)
    with Image.open(image_path) as src:
        base = src.convert("RGBA")

    anns_a = coco_a.anns_by_image_id.get(int(pair.image_a["id"]), [])
    anns_b = coco_b.anns_by_image_id.get(int(pair.image_b["id"]), [])

    sources = [
        ("a", config.name_a, coco_a, anns_a, color_rgb(config.color_a)),
        ("b", config.name_b, coco_b, anns_b, color_rgb(config.color_b)),
    ]

    if config.mask_mode != "none":
        for side, _, dataset, anns, color in sources:
            invert = config.invert_masks in {side, "both"}
            for ann in anns:
                ann_color = annotation_color(ann, dataset, color, config)
                draw_annotation_mask(base, ann, ann_color, config, invert)

    draw = ImageDraw.Draw(base)
    font = ImageFont.load_default()
    for _, source_name, dataset, anns, color in sources:
        for ann in anns:
            ann_color = annotation_color(ann, dataset, color, config)
            if config.draw_bboxes:
                draw_bbox(draw, ann, ann_color, config.line_width)
            if config.text:
                draw_annotation_text(draw, font, ann, dataset, source_name, ann_color, config)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        base.convert("RGB").save(out_path)
    else:
        base.save(out_path)


def draw_annotation_mask(
    image: Image.Image,
    ann: dict[str, Any],
    color: tuple[int, int, int],
    config: RenderConfig,
    invert: bool,
) -> None:
    mask = segmentation_mask(ann.get("segmentation"), image.size)
    if mask is None:
        return
    if invert:
        mask = ImageChops.invert(mask)

    if config.mask_mode in {"fill", "both"}:
        alpha = int(round(255 * config.mask_alpha))
        overlay = Image.new("RGBA", image.size, (*color, 0))
        overlay.putalpha(mask.point(lambda p: alpha if p else 0))
        image.alpha_composite(overlay)

    if config.mask_mode in {"outline", "both"}:
        outline = mask.filter(ImageFilter.FIND_EDGES).point(lambda p: 255 if p else 0)
        if config.line_width > 1:
            filter_size = config.line_width
            if filter_size % 2 == 0:
                filter_size += 1
            outline = outline.filter(ImageFilter.MaxFilter(filter_size))
        overlay = Image.new("RGBA", image.size, (*color, 0))
        overlay.putalpha(outline)
        image.alpha_composite(overlay)


def segmentation_mask(segmentation: Any, size: tuple[int, int]) -> Image.Image | None:
    if not segmentation:
        return None

    width, height = size
    if isinstance(segmentation, list):
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        polygons = normalize_polygons(segmentation)
        for polygon in polygons:
            if len(polygon) < 6:
                continue
            points = [
                (float(polygon[i]), float(polygon[i + 1]))
                for i in range(0, len(polygon) - 1, 2)
            ]
            draw.polygon(points, fill=255)
        return mask if mask.getbbox() else None

    if isinstance(segmentation, dict):
        try:
            rle = segmentation
            if isinstance(segmentation.get("counts"), list):
                rle = mask_utils.frPyObjects(segmentation, height, width)
            decoded = mask_utils.decode(rle)
        except Exception as e:
            raise ConfigError(f"Could not decode segmentation RLE: {e}") from e
        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]
        mask_array = (decoded > 0).astype(np.uint8) * 255
        return Image.fromarray(mask_array, mode="L")

    raise ConfigError(f"Unsupported COCO segmentation type: {type(segmentation).__name__}")


def normalize_polygons(segmentation: list[Any]) -> list[list[float]]:
    if not segmentation:
        return []
    if all(isinstance(item, (int, float)) for item in segmentation):
        return [segmentation]
    return [polygon for polygon in segmentation if isinstance(polygon, list)]


def draw_bbox(
    draw: ImageDraw.ImageDraw,
    ann: dict[str, Any],
    color: tuple[int, int, int],
    line_width: int,
) -> None:
    bbox = ann.get("bbox")
    if not bbox or len(bbox) != 4:
        return
    x, y, w, h = [float(v) for v in bbox]
    draw.rectangle([x, y, x + w, y + h], outline=color, width=line_width)


def draw_annotation_text(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    ann: dict[str, Any],
    dataset: CocoDataset,
    source_name: str,
    color: tuple[int, int, int],
    config: RenderConfig,
) -> None:
    text = annotation_text(ann, dataset, source_name, config.text or "")
    if not text:
        return

    bbox = ann.get("bbox") or [0, 0, 0, 0]
    x = float(bbox[0])
    y = max(0.0, float(bbox[1]) - 12)

    text_box = draw.textbbox((x, y), text, font=font)
    padded = (
        text_box[0] - 2,
        text_box[1] - 1,
        text_box[2] + 2,
        text_box[3] + 1,
    )
    draw.rectangle(padded, fill=(0, 0, 0, 180))
    draw.text((x, y), text, fill=(*color, 255), font=font)


def annotation_text(
    ann: dict[str, Any],
    dataset: CocoDataset,
    source_name: str,
    pattern: str,
) -> str:
    values: dict[str, Any] = {}
    for key, value in ann.items():
        values[key] = clean_format_value(value)
        values[key.upper()] = clean_format_value(value)

    category = category_name(ann, dataset)
    obj_id = ann.get("obj_id", "")
    roi_id = ann.get("roi_id", ann.get("roi", ""))
    values.update(
        {
            "SOURCE": source_name,
            "CATEGORY": category,
            "CATEGORY_ID": ann.get("category_id", ""),
            "ROI": roi_id,
            "ROI_ID": roi_id,
            "OBJ": clean_format_value(obj_id),
            "OBJ_ID": clean_format_value(obj_id),
            "ANN_ID": ann.get("id", ""),
        }
    )

    try:
        return pattern.format_map(DefaultFormatDict(values))
    except Exception as e:
        raise ConfigError(f"Could not format --text for annotation {ann.get('id')}: {e}") from e


class MissingFormatValue:
    def __format__(self, format_spec: str) -> str:
        return ""

    def __str__(self) -> str:
        return ""


class DefaultFormatDict(dict):
    def __missing__(self, key: str) -> MissingFormatValue:
        return MissingFormatValue()


def clean_format_value(value: Any) -> Any:
    if value is None:
        return MissingFormatValue()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, (str, int, float)):
        return value
    return str(value)


def annotation_color(
    ann: dict[str, Any],
    dataset: CocoDataset,
    default_color: tuple[int, int, int],
    config: RenderConfig,
) -> tuple[int, int, int]:
    label = category_name(ann, dataset).lower()
    if label in config.label_colors:
        return color_rgb(config.label_colors[label])
    return default_color


def category_name(ann: dict[str, Any], dataset: CocoDataset) -> str:
    category_id = ann.get("category_id")
    if category_id is None:
        return ""
    try:
        return dataset.categories_by_id.get(int(category_id), str(category_id))
    except (TypeError, ValueError):
        return str(category_id)


def color_rgb(value: str) -> tuple[int, int, int]:
    try:
        rgb = ImageColor.getrgb(value)
    except ValueError as e:
        raise ConfigError(f"Invalid color {value!r}: {e}") from e
    if len(rgb) == 4:
        return rgb[:3]
    return rgb


def resolve_image_path(pair: ImagePair, config: RenderConfig) -> Path:
    for image in (pair.image_a, pair.image_b):
        filename = str(image.get("file_name") or pair.key)
        path = Path(filename)
        if not path.is_absolute():
            path = config.image_root / path
        if path.exists():
            return path
    raise ConfigError(
        f"Image file not found for {pair.key!r} under image_root {config.image_root}"
    )


def derive_ffid(image: dict[str, Any], fallback_key: str, ffid_field: str) -> str:
    value = image.get(ffid_field, fallback_key)
    return Path(str(value)).stem


def render_output_path(config: RenderConfig, pair: ImagePair) -> Path:
    file_name_a = str(pair.image_a.get("file_name", pair.key))
    file_name_b = str(pair.image_b.get("file_name", pair.key))
    values = {
        "FFID": pair.ffid,
        "IMAGE_KEY": pair.key,
        "IMAGE_ID_A": pair.image_a["id"],
        "IMAGE_ID_B": pair.image_b["id"],
        "FILE_NAME_A": file_name_a,
        "FILE_NAME_B": file_name_b,
        "STEM_A": Path(file_name_a).stem,
        "STEM_B": Path(file_name_b).stem,
        "NAME_A": config.name_a,
        "NAME_B": config.name_b,
    }
    try:
        return Path(config.output.format_map(DefaultFormatDict(values)))
    except Exception as e:
        raise ConfigError(f"Could not format output template {config.output!r}: {e}") from e


def validate_output_template(output: str, pairs: list[ImagePair]) -> None:
    if len(pairs) <= 1:
        return
    fields = set(re.findall(r"{([^{}:]+)(?::[^{}]+)?}", output))
    variable_fields = {
        "FFID",
        "IMAGE_KEY",
        "IMAGE_ID_A",
        "IMAGE_ID_B",
        "FILE_NAME_A",
        "FILE_NAME_B",
        "STEM_A",
        "STEM_B",
    }
    if not fields & variable_fields:
        raise ConfigError(
            "output must contain a per-image token such as {FFID} when rendering multiple images"
        )


def write_provenance(
    config: RenderConfig,
    started_at: datetime,
    started_monotonic: float,
    summary: dict[str, Any],
) -> None:
    if config.provenance is None or config.dry_run:
        return

    finished_at = datetime.now()
    path = Path(
        config.provenance.format(
            DATETIME=started_at.strftime("%Y%m%d_%H%M%S"),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "elapsed_seconds": round(monotonic() - started_monotonic, 3),
        "config": serializable_config(config),
        "summary": summary,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def serializable_config(config: RenderConfig) -> dict[str, Any]:
    values = asdict(config)
    values["coco_a"] = str(config.coco_a)
    values["coco_b"] = str(config.coco_b)
    values["image_root"] = str(config.image_root)
    return values


def preview(values: list[Any], limit: int = 5) -> str:
    shown = ", ".join(str(value) for value in values[:limit])
    if len(values) > limit:
        shown += f", ... ({len(values)} total)"
    return shown

