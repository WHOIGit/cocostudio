"""Reviewed Label Studio annotation pull utilities."""

from __future__ import annotations

import json
import os
import re
import importlib
import importlib.util
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Union
from urllib.parse import urlparse

from PIL import Image

from labelstudio_tools import TaskManager

from .push import _BBOX_FAMILIES, discover_controls, format_pattern


_CACHE_MODES = frozenset({"never", "once", "clobber"})
_ANNOTATION_POLICIES = frozenset({"error", "skip", "latest", "earliest"})
_LABEL_POLICIES = frozenset({"match_push_coco", "allow_any"})
COCO_FLOAT_PRECISION = 9


def cache_path_for(taskman: TaskManager,
                   config_path: Union[str, os.PathLike],
                   cache_id: str) -> Path:
    """Return ``cocostudio/cache/<hostname>/<project-name>/<cache-id>.json``.

    The cache root is anchored at the nearest ``cocostudio`` ancestor of the
    config file, so commands work from either the repo root or ``cocostudio/``.
    """
    cfg = Path(config_path).resolve()
    root = next((p for p in cfg.parents if p.name == "cocostudio"),
                Path("cocostudio").resolve())
    host = _hostname(taskman.host)
    project = _safe_project_name(getattr(taskman.project, "title", "project"))
    return root / "cache" / host / project / _cache_filename(cache_id)


def _hostname(host: str) -> str:
    parsed = urlparse(host)
    value = parsed.netloc or parsed.path.split("/", 1)[0] or "host"
    return value.replace(":", "_")


def _safe_project_name(title: str) -> str:
    title = str(title).replace(" ", "_")
    return re.sub(r"[\\/]+", "_", title)


def _cache_filename(cache_id: str) -> str:
    return cache_id if Path(cache_id).suffix == ".json" else f"{cache_id}.json"


def load_source_coco(path: Optional[Union[str, os.PathLike]]) -> Optional[dict]:
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def fetch_tasks(taskman: TaskManager, *, view: Optional[Union[int, str]],
                cache_path: Path, ls_cache: str, dry_run: bool = False) -> tuple[list[dict], dict]:
    """Download LS tasks with annotations, honoring the simple cache policy."""
    if ls_cache not in _CACHE_MODES:
        raise ValueError(f"ls_cache must be one of {sorted(_CACHE_MODES)}, got {ls_cache!r}")

    report = {
        "mode": ls_cache,
        "path": str(cache_path),
        "used": False,
        "fetched": False,
        "written": False,
    }
    if ls_cache == "once" and cache_path.exists():
        with open(cache_path) as f:
            report["used"] = True
            return json.load(f), report

    tasks = taskman.get_tasks(
        with_annotations=True,
        view=view,
    )
    report["fetched"] = True

    if ls_cache in {"once", "clobber"} and not dry_run:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(tasks, f, indent=2)
        report["written"] = True
    return tasks, report


def choose_annotation(task: dict, *, completed_only: bool,
                      annotation_policy: str) -> tuple[Optional[dict], str]:
    """Pick one submitted annotation from a task.

    Returns ``(annotation, status)`` where status is one of ``selected``,
    ``empty``, or ``skipped_multiple``.
    """
    if annotation_policy not in _ANNOTATION_POLICIES:
        raise ValueError(
            f"annotation_policy must be one of {sorted(_ANNOTATION_POLICIES)}, "
            f"got {annotation_policy!r}")

    annotations = [
        a for a in (task.get("annotations") or [])
        if not a.get("was_cancelled") and not a.get("cancelled")
    ]
    if not annotations:
        if completed_only:
            return None, "empty"
        return None, "selected"

    if len(annotations) == 1:
        return annotations[0], "selected"

    if annotation_policy == "error":
        raise ValueError(
            f"task {task.get('id')} has multiple submitted annotations "
            f"({len(annotations)}); "
            "set annotation_policy to skip, latest, or earliest")
    if annotation_policy == "skip":
        return None, "skipped_multiple"

    reverse = annotation_policy == "latest"
    return sorted(annotations, key=_annotation_time, reverse=reverse)[0], "selected"


def _annotation_time(annotation: dict) -> str:
    return (
        annotation.get("updated_at")
        or annotation.get("created_at")
        or annotation.get("completed_at")
        or ""
    )


def validate_label_policy(label_policy: str, source_coco: Optional[dict],
                          mapped_controls: Iterable[dict]) -> dict[str, int]:
    """Validate mapped LS control labels before task download.

    ``match_push_coco`` requires an exact label-name match between every mapped
    labeled control and ``source_coco['categories']``.
    """
    if label_policy not in _LABEL_POLICIES:
        raise ValueError(
            f"label_policy must be one of {sorted(_LABEL_POLICIES)}, got {label_policy!r}")

    if source_coco is None:
        if label_policy == "match_push_coco":
            raise ValueError("label_policy=match_push_coco requires push_coco/[push].coco")
        return {}

    category_names = [c["name"] for c in source_coco.get("categories", [])]
    if not category_names:
        raise ValueError("push_coco has no categories")

    by_name = {c["name"]: int(c["id"]) for c in source_coco.get("categories", [])}
    if label_policy != "match_push_coco":
        return by_name

    expected = set(category_names)
    for ctrl in mapped_controls:
        if ctrl["value_key"] is None:
            continue
        actual = set(ctrl["labels"])
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"Label Studio control {ctrl['from_name']!r} labels do not match "
                f"push COCO categories; missing={missing} extra={extra}")
    return by_name


def build_coco_from_tasks(tasks: list[dict], *, source_coco: Optional[dict],
                          bbox_ctrl: Optional[dict], label_policy: str,
                          region_id_pattern: Optional[str] = None,
                          image_datafield: str, image_dir: Optional[Union[str, os.PathLike]],
                          download_missing_images: bool, taskman: TaskManager,
                          completed_only: bool, annotation_policy: str,
                          dry_run: bool = False) -> tuple[dict, dict]:
    """Convert selected Label Studio bbox annotations into COCO."""
    source_images = _source_images_by_file_name(source_coco)
    categories = deepcopy(source_coco.get("categories", [])) if source_coco else []
    category_by_name = {c["name"]: int(c["id"]) for c in categories}
    next_category_id = max(category_by_name.values(), default=0) + 1

    out_images: list[dict] = []
    out_annotations: list[dict] = []
    image_ids_by_name: dict[str, int] = {}
    next_image_id = max((img["id"] for img in source_images.values()), default=0) + 1
    source_region_annotation_ids, source_bbox_annotation_ids, max_source_annotation_id = (
        _source_annotation_ids(source_coco, region_id_pattern=region_id_pattern)
    )
    used_source_annotation_ids: set[int] = set()
    next_annotation_id = max_source_annotation_id + 1
    selected_task_count = 0
    skipped_empty = 0
    skipped_multiple = 0
    image_report = {
        "checked": 0,
        "missing_local": [],
        "downloaded": [],
        "would_download": [],
    }

    for task in tasks:
        annotation, status = choose_annotation(
            task, completed_only=completed_only, annotation_policy=annotation_policy)
        if status == "empty":
            skipped_empty += 1
            continue
        if status == "skipped_multiple":
            skipped_multiple += 1
            continue
        selected_task_count += 1

        image = _image_from_task(
            task,
            annotation=annotation,
            source_images=source_images,
            image_ids_by_name=image_ids_by_name,
            next_image_id=next_image_id,
            image_datafield=image_datafield,
            image_dir=image_dir,
            download_missing_images=download_missing_images,
            taskman=taskman,
            image_report=image_report,
            dry_run=dry_run,
        )
        if image["id"] == next_image_id:
            next_image_id += 1
        if image["file_name"] not in image_ids_by_name:
            image_ids_by_name[image["file_name"]] = image["id"]
            out_images.append(image)

        if annotation is None:
            continue
        if bbox_ctrl is None:
            continue

        for result in _matching_results(annotation, bbox_ctrl):
            label = _result_label(result, bbox_ctrl)
            if label is None:
                raise ValueError(
                    f"task {task.get('id')} result {result.get('id')!r} has no label")
            if label not in category_by_name:
                if label_policy == "match_push_coco":
                    raise ValueError(
                        f"task {task.get('id')} result {result.get('id')!r} "
                        f"uses label {label!r}, not present in push COCO categories")
                category_by_name[label] = next_category_id
                categories.append({"id": next_category_id, "name": label})
                next_category_id += 1
            provisional_annotation = _bbox_result_to_coco_annotation(
                result,
                image=image,
                annotation_id=next_annotation_id,
                category_id=category_by_name[label],
                ls_annotation=annotation,
            )
            source_id = _pop_matching_source_annotation_id(
                source_region_annotation_ids,
                source_bbox_annotation_ids,
                provisional_annotation,
                used_source_annotation_ids,
            )
            if source_id is not None:
                provisional_annotation["id"] = source_id
                used_source_annotation_ids.add(source_id)
            else:
                next_annotation_id += 1
            out_annotations.append(provisional_annotation)

    if image_report["missing_local"] and not dry_run:
        missing = image_report["missing_local"]
        raise FileNotFoundError(
            f"images missing locally and download_missing_images=false: {missing}")

    coco = {
        "info": _output_info(source_coco, taskman),
        "licenses": deepcopy(source_coco.get("licenses", [])) if source_coco else [],
        "categories": categories,
        "images": out_images,
        "annotations": out_annotations,
    }
    report = {
        "tasks": {
            "downloaded": len(tasks),
            "selected": selected_task_count,
            "skipped_empty": skipped_empty,
            "skipped_multiple": skipped_multiple,
        },
        "images": {
            "count": len(out_images),
            **image_report,
        },
        "annotations": {
            "count": len(out_annotations),
        },
        "categories": {
            "count": len(categories),
            "names": [c["name"] for c in categories],
        },
    }
    return coco, report


def _source_annotation_ids(
        source_coco: Optional[dict],
        *,
        region_id_pattern: Optional[str]) -> tuple[
            dict[tuple[int, str], list[int]],
            dict[tuple[int, tuple[float, ...]], list[int]],
            int,
        ]:
    if not source_coco:
        return {}, {}, 0

    source_images_by_id = {
        int(image["id"]): image
        for image in source_coco.get("images", [])
        if image.get("id") is not None
    }
    ids_by_region: dict[tuple[int, str], list[int]] = {}
    ids_by_bbox: dict[tuple[int, tuple[float, ...]], list[int]] = {}
    max_annotation_id = 0
    for annotation in source_coco.get("annotations", []):
        annotation_id = annotation.get("id")
        bbox = annotation.get("bbox")
        image_id = annotation.get("image_id")
        if annotation_id is None:
            continue
        annotation_id = int(annotation_id)
        max_annotation_id = max(max_annotation_id, annotation_id)
        if image_id is None:
            continue
        image_id = int(image_id)

        region_id = _source_region_id(
            annotation, image=source_images_by_id.get(image_id),
            source_coco=source_coco, region_id_pattern=region_id_pattern)
        if region_id is not None:
            ids_by_region.setdefault((image_id, region_id), []).append(annotation_id)

        if isinstance(bbox, list) and len(bbox) == 4:
            key = (image_id, _bbox_key(bbox))
            ids_by_bbox.setdefault(key, []).append(annotation_id)
    return ids_by_region, ids_by_bbox, max_annotation_id


def _source_region_id(annotation: dict, *, image: Optional[dict],
                      source_coco: dict,
                      region_id_pattern: Optional[str]) -> Optional[str]:
    if not region_id_pattern:
        return None
    try:
        return format_pattern(
            region_id_pattern,
            annotation=annotation,
            image=image or {},
            info=source_coco.get("info") or {},
        )
    except KeyError:
        return None


def _pop_matching_source_annotation_id(
        source_region_annotation_ids: dict[tuple[int, str], list[int]],
        source_bbox_annotation_ids: dict[tuple[int, tuple[float, ...]], list[int]],
        annotation: dict,
        used_source_annotation_ids: set[int]) -> Optional[int]:
    image_id = int(annotation["image_id"])
    meta = annotation.get("meta") or {}
    region_id = meta.get("ls_region_id") if isinstance(meta, dict) else None
    candidates = (
        source_region_annotation_ids.get((image_id, region_id))
        if isinstance(region_id, str) else None
    )
    if candidates:
        source_id = _pop_unused_source_id(candidates, used_source_annotation_ids)
        if source_id is not None:
            return source_id

    key = (image_id, _bbox_key(annotation["bbox"]))
    candidates = source_bbox_annotation_ids.get(key)
    if not candidates:
        return None
    return _pop_unused_source_id(candidates, used_source_annotation_ids)


def _pop_unused_source_id(candidates: list[int],
                          used_source_annotation_ids: set[int]) -> Optional[int]:
    while candidates:
        source_id = candidates.pop(0)
        if source_id not in used_source_annotation_ids:
            return source_id
    return None


def _bbox_key(bbox: list) -> tuple[float, ...]:
    return tuple(_round_coco_float(float(value)) for value in bbox)


def _source_images_by_file_name(source_coco: Optional[dict]) -> dict[str, dict]:
    if not source_coco:
        return {}
    return {img["file_name"]: img for img in source_coco.get("images", [])}


def _image_from_task(task: dict, *, annotation: Optional[dict],
                     source_images: dict[str, dict], image_ids_by_name: dict[str, int],
                     next_image_id: int, image_datafield: str,
                     image_dir: Optional[Union[str, os.PathLike]],
                     download_missing_images: bool, taskman: TaskManager,
                     image_report: dict, dry_run: bool) -> dict:
    data = task.get("data") or {}
    file_name = _task_file_name(data, image_datafield)
    _ensure_image_available(
        task,
        file_name=file_name,
        image_datafield=image_datafield,
        image_dir=image_dir,
        download_missing_images=download_missing_images,
        taskman=taskman,
        report=image_report,
        dry_run=dry_run,
    )
    source = source_images.get(file_name)
    if source is not None:
        image = deepcopy(source)
    else:
        width, height = _task_image_size(data, file_name=file_name, image_dir=image_dir)
        image = {
            "id": image_ids_by_name.get(file_name, next_image_id),
            "file_name": file_name,
            "width": width,
            "height": height,
        }

    image["id"] = image_ids_by_name.get(file_name, image["id"])
    meta = deepcopy(image.get("meta") or {})
    if isinstance(data.get("meta"), dict):
        meta.update(data["meta"])
    meta["ls_task_id"] = task.get("id")
    if annotation is not None:
        meta["ls_annotation_id"] = annotation.get("id")
    if meta:
        image["meta"] = meta
    return image


def _task_file_name(data: dict, image_datafield: str) -> str:
    if data.get("file_name"):
        return str(data["file_name"])
    url = data.get(image_datafield)
    if isinstance(url, str) and url:
        return os.path.basename(url.rstrip("/"))
    raise ValueError(
        f"task data has no file_name and no usable {image_datafield!r} datafield")


def _task_image_size(data: dict, *, file_name: str,
                     image_dir: Optional[Union[str, os.PathLike]]) -> tuple[int, int]:
    if data.get("width") and data.get("height"):
        return int(data["width"]), int(data["height"])
    if image_dir:
        path = Path(image_dir) / file_name
        if path.exists():
            return _image_size_from_disk(path)
    raise ValueError(
        f"cannot determine image size for {file_name!r}; provide push_coco, "
        "task data width/height, or a readable image_dir")


def _image_size_from_disk(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            width, height = img.size
    except Exception as exc:
        raise ValueError(f"could not read image dimensions: {path}") from exc
    return int(width), int(height)


def _ensure_image_available(task: dict, *, file_name: str, image_datafield: str,
                            image_dir: Optional[Union[str, os.PathLike]],
                            download_missing_images: bool, taskman: TaskManager,
                            report: dict, dry_run: bool) -> None:
    if not image_dir:
        return
    path = Path(image_dir) / file_name
    report["checked"] += 1
    if path.exists():
        return

    if not download_missing_images:
        report["missing_local"].append(file_name)
        return

    url = (task.get("data") or {}).get(image_datafield)
    if not isinstance(url, str) or not url.startswith("s3://"):
        raise ValueError(
            f"task {task.get('id')} image is missing locally and "
            f"data.{image_datafield} is not an s3:// URL")
    if dry_run:
        report["would_download"].append(file_name)
        return
    taskman.download_s3url(url, str(path), clobber=False)
    report["downloaded"].append(file_name)


def _matching_results(annotation: dict, ctrl: dict) -> list[dict]:
    out = []
    for result in annotation.get("result", []) or []:
        if result.get("from_name") != ctrl["from_name"]:
            continue
        if str(result.get("type", "")).lower() != ctrl["result_type"]:
            continue
        out.append(result)
    return out


def _result_label(result: dict, ctrl: dict) -> Optional[str]:
    key = ctrl["value_key"]
    if key is None:
        return None
    labels = (result.get("value") or {}).get(key) or []
    if not labels:
        return None
    if len(labels) > 1:
        raise ValueError(
            f"result {result.get('id')!r} has multiple labels {labels}; "
            "COCO bbox conversion expects one label")
    return labels[0]


def _bbox_result_to_coco_annotation(result: dict, *, image: dict,
                                    annotation_id: int, category_id: int,
                                    ls_annotation: dict) -> dict:
    value = result.get("value") or {}
    img_w = float(result.get("original_width") or image["width"])
    img_h = float(result.get("original_height") or image["height"])
    x = _round_coco_float(float(value["x"]) / 100.0 * img_w)
    y = _round_coco_float(float(value["y"]) / 100.0 * img_h)
    w = _round_coco_float(float(value["width"]) / 100.0 * img_w)
    h = _round_coco_float(float(value["height"]) / 100.0 * img_h)
    meta = {
        "ls_region_id": result.get("id"),
        "ls_annotation_id": ls_annotation.get("id"),
    }
    if isinstance(result.get("meta"), dict):
        meta.update(result["meta"])

    out = {
        "id": annotation_id,
        "image_id": image["id"],
        "category_id": category_id,
        "bbox": [x, y, w, h],
        "area": _round_coco_float(w * h),
        "iscrowd": 0,
        "meta": {k: v for k, v in meta.items() if v is not None},
    }
    if "score" in result:
        out["score"] = float(result["score"])
    return out


def _round_coco_float(value: float) -> float:
    return round(value, COCO_FLOAT_PRECISION)


def apply_postprocess_function(coco: dict, spec: Optional[str], *,
                               config_path: Union[str, os.PathLike]) -> dict:
    """Apply an optional user hook to the completed COCO dict.

    The hook spec is ``path/to/file.py:function`` or ``module.name:function``.
    Hooks may mutate ``coco`` in place and return ``None``, or return a
    replacement COCO dict.
    """
    if not spec:
        return coco
    func = _load_postprocess_function(spec, config_path=config_path)
    result = func(coco)
    if result is None:
        return coco
    if not isinstance(result, dict):
        raise TypeError(
            f"postprocess_function {spec!r} must return a COCO dict or None, "
            f"got {type(result).__name__}")
    return result


def _load_postprocess_function(spec: str, *,
                               config_path: Union[str, os.PathLike]):
    target, sep, func_name = spec.partition(":")
    if not sep or not target or not func_name:
        raise ValueError(
            "postprocess_function must be path.py:function or module:function")

    module = (_load_module_from_file(target, config_path=config_path)
              if target.endswith(".py") else importlib.import_module(target))
    func = getattr(module, func_name, None)
    if not callable(func):
        raise ValueError(f"postprocess function {func_name!r} not found in {target!r}")
    return func


def _load_module_from_file(target: str, *,
                           config_path: Union[str, os.PathLike]):
    path = _resolve_postprocess_path(target, config_path=config_path)
    module_name = f"_cocostudio_pull_postprocess_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load postprocess module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_postprocess_path(target: str, *,
                              config_path: Union[str, os.PathLike]) -> Path:
    raw = Path(target).expanduser()
    candidates = [raw]
    cfg = Path(config_path).resolve()
    candidates.append(cfg.parent / raw)
    root = next((p for p in cfg.parents if p.name == "cocostudio"), None)
    if root is not None:
        candidates.append(root / raw)
        if raw.parts and raw.parts[0] == root.name:
            candidates.append(root.parent / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"postprocess module not found: {target}")


def _output_info(source_coco: Optional[dict], taskman: TaskManager) -> dict:
    info = deepcopy(source_coco.get("info", {})) if source_coco else {}
    info["cocostudio_pull"] = {
        "project_id": getattr(taskman.project, "id", None),
        "project_title": getattr(taskman.project, "title", None),
        "date_created": datetime.now().isoformat(timespec="seconds"),
    }
    return info


def pull_coco(taskman: TaskManager, *, out_coco: Union[str, os.PathLike],
              config_path: Union[str, os.PathLike], push_coco: Optional[Union[str, os.PathLike]] = None,
              view: Optional[Union[int, str]] = None, completed_only: bool = True,
              annotation_policy: str = "error", map_bbox: Optional[str] = None,
              map_seg: Optional[str] = None, label_policy: str = "match_push_coco",
              region_id_pattern: Optional[str] = None,
              image_datafield: str = "image",
              image_dir: Optional[Union[str, os.PathLike]] = None,
              download_missing_images: bool = False, ls_cache: str = "once",
              cache_id: Optional[str] = None,
              postprocess_function: Optional[str] = None,
              dry_run: bool = False) -> dict:
    """Pull submitted Label Studio bbox annotations into a COCO JSON file."""
    if not map_bbox and not map_seg:
        raise ValueError("provide at least one of map_bbox / map_seg")
    if map_seg:
        raise NotImplementedError(
            "pulling segmentations is not implemented yet; use map_bbox only for v1")

    source_coco = load_source_coco(push_coco)
    controls = discover_controls(taskman)
    bbox_ctrl = (controls.resolve(map_bbox, families=_BBOX_FAMILIES,
                                  what="--map-bbox") if map_bbox else None)
    # Resolving map_seg would happen here once segmentation pull exists.
    validate_label_policy(
        label_policy,
        source_coco,
        [ctrl for ctrl in (bbox_ctrl,) if ctrl is not None],
    )

    cache_id = cache_id or Path(config_path).with_suffix(".json").name
    cache_path = cache_path_for(taskman, config_path, cache_id)
    tasks, cache_report = fetch_tasks(
        taskman, view=view, cache_path=cache_path, ls_cache=ls_cache, dry_run=dry_run)

    coco, report = build_coco_from_tasks(
        tasks,
        source_coco=source_coco,
        bbox_ctrl=bbox_ctrl,
        label_policy=label_policy,
        region_id_pattern=region_id_pattern,
        image_datafield=image_datafield,
        image_dir=image_dir,
        download_missing_images=download_missing_images,
        taskman=taskman,
        completed_only=completed_only,
        annotation_policy=annotation_policy,
        dry_run=dry_run,
    )
    coco = apply_postprocess_function(
        coco, postprocess_function, config_path=config_path)

    out_path = Path(out_coco)
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)

    report.update({
        "cache": cache_report,
        "out_coco": str(out_path),
        "dry_run": dry_run,
    })
    return report
