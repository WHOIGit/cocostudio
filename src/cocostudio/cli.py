"""Console-script CLI for ``cocostudio`` (``cocostudio push`` / ``cocostudio pull``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from labelstudio_tools import TaskManager
from labelstudio_tools.config import find_storage, load_config

from .config import PUSH_KEYS, SHARED_KEYS, load_push_config
from .push import push_coco

# All config-mergeable parameter keys (config + CLI overrides share these names).
_PARAM_KEYS = (*SHARED_KEYS, *PUSH_KEYS)

# Hardcoded defaults. Deliberately omits ls_config / coco / image_dir: those have
# no sensible default, so a missing value is an error rather than a fallback.
_DEFAULTS = {
    "s3_prefix": None,                    # None → derive from ls_config storage
    "region_id_pattern": "{annotation.id}",
    "model_version": None,                # None → derived from COCO info
    "map_bbox": None,
    "map_seg": None,
    "polygon_preprocess_epsilon": None,
    "polygon_preprocess_maxpoints": None,
    "include": None,
    "exclude": None,
    "dry_run": False,
}


def _add_push_args(p: argparse.ArgumentParser) -> None:
    """Push flags. Every override defaults to ``None`` so 'not passed' is detectable
    and the config value (or hardcoded default) shows through."""
    p.add_argument("-c", "--config", required=True,
                   help="Pushpull TOML config (e.g. configs/pushpull/sam2dirty_brush.toml).")
    p.add_argument("--ls-config", default=None,
                   help="Override: LS project TOML.")
    p.add_argument("--coco", default=None, help="Override: COCO JSON file.")
    p.add_argument("--image-dir", default=None,
                   help="Override: local image directory; file_names must match COCO.")
    p.add_argument("--s3-prefix", default=None,
                   help="Override: S3 key prefix. Omit to use the LS project's "
                        "[[storage]].bucket_prefix.")
    p.add_argument("--region-id-pattern", default=None,
                   help="Override: pattern for LS region IDs (default {annotation.id}).")
    p.add_argument("--model-version", default=None,
                   help="Override: LS prediction model_version. {info.*} tokens are "
                        "resolved against COCO info. Default: derived from COCO info.")
    p.add_argument("--map-bbox", default=None, metavar="TAG",
                   help="Send COCO bboxes to this rectangle control (a control name "
                        "or a tag type, e.g. RectangleLabels).")
    p.add_argument("--map-seg", default=None, metavar="TAG",
                   help="Send COCO segmentations to this polygon/brush control (a "
                        "control name or a tag type). Its type selects polygon vs brush.")
    p.add_argument("--polygon-preprocess-epsilon", type=float, default=None,
                   help="Douglas-Peucker simplification of polygon segmentations. "
                        "<1 = fraction of perimeter; >=1 = absolute pixels. Errors if "
                        "--map-seg targets a brush control.")
    p.add_argument("--polygon-preprocess-maxpoints", type=int, default=None, metavar="N",
                   help="Cap vertices per polygon (epsilon binary-searched to honour "
                        "it). 0 disables simplification.")
    p.add_argument("--include", nargs="+", default=None, metavar="VALUE",
                   help="Image id, file_name, or stem(s) to include.")
    p.add_argument("--exclude", nargs="+", default=None, metavar="VALUE",
                   help="Image id, file_name, or stem(s) to exclude.")
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None,
                   help="Plan only — no S3 uploads, no LS task creation "
                        "(--no-dry-run forces a real run over a config dry_run=true).")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cocostudio",
        description="Push COCO predictions to / pull reviewed annotations from Label Studio.")
    sub = p.add_subparsers(dest="command", required=True)
    p_push = sub.add_parser(
        "push", help="Push COCO predictions to a Label Studio project.")
    _add_push_args(p_push)
    p_push.set_defaults(func=push_command)
    p_pull = sub.add_parser("pull", help="(not yet implemented)")
    p_pull.set_defaults(func=pull_command)
    return p


def resolve_simplify(epsilon: float | None,
                     max_points: int | None) -> tuple[float | None, int | None]:
    """Normalize polygon-simplification args into ``(epsilon, max_points)``.

    ``max_points == 0`` is the escape hatch: disables simplification entirely
    (raw polygons). A ``max_points`` cap given without ``epsilon`` falls back
    to ``epsilon=1`` (1px) — the binary search then raises it to honour the cap.
    """
    if max_points == 0:
        return None, None
    if max_points is not None and epsilon is None:
        epsilon = 1.0
    return epsilon, max_points


def _merge_settings(args: argparse.Namespace) -> dict:
    """Layer hardcoded defaults < config file < explicitly-passed CLI flags."""
    config = load_push_config(args.config)
    settings = {**_DEFAULTS, **config}
    for key in _PARAM_KEYS:
        val = getattr(args, key, None)
        if val is not None:                       # None == flag not passed
            settings[key] = val
    return settings


def _resolve_s3_prefix(settings: dict) -> str:
    """Use the configured s3_prefix, else the LS project's storage bucket_prefix."""
    if settings["s3_prefix"] is not None:
        return settings["s3_prefix"]
    storage = find_storage(load_config(settings["ls_config"]))
    prefix = storage.get("bucket_prefix") if storage else None
    if not prefix:
        raise ValueError(
            "no s3_prefix configured and the LS project's [[storage]] entry has "
            "no 'bucket_prefix'; set s3_prefix in the config or pass --s3-prefix")
    return prefix


def _validate(settings: dict) -> None:
    missing = [k for k in ("ls_config", "coco", "image_dir") if not settings.get(k)]
    if missing:
        raise ValueError(f"missing required config/flag(s): {missing}")
    for key in ("ls_config", "coco", "image_dir"):
        if not Path(settings[key]).exists():
            raise ValueError(f"{key} path not found: {settings[key]}")
    if not settings["map_bbox"] and not settings["map_seg"]:
        raise ValueError("provide at least one of map_bbox / map_seg "
                         "(--map-bbox / --map-seg) — nothing to push otherwise")


def _print_report(report: dict) -> None:
    s3 = report["s3"]
    tasks = report["tasks"]
    print()
    print(f"Selected images: {len(report['selected_images'])}")
    print(f"Kinds: {', '.join(report['kinds'])}")
    print(f"Segmentation kind: {report['segmentation_kind']}")
    print(f"Model version: {report['model_version']}")
    print()
    print(f"S3: existed={len(s3['existed'])} uploaded={len(s3['uploaded'])} "
          f"would_upload={len(s3['would_upload'])} missing_local={len(s3['missing_local'])}")
    if s3["missing_local"]:
        print(f"  MISSING LOCAL: {s3['missing_local']}")
    poly = report.get("polygon_simplification")
    if poly and poly.get("polygons"):
        print(f"Polygon simplification: epsilon={poly['epsilon']} "
              f"max_points={poly['max_points']}")
        if "points_before" in poly:
            print(f"  total points: {poly['points_before']} -> "
                  f"{poly['points_after']}  "
                  f"({poly['reduction_pct']:.1f}% reduction)")
            print(f"  per-polygon: avg {poly['avg_before']:.1f} -> "
                  f"{poly['avg_after']:.1f}  "
                  f"(after: min={poly['min_after']} max={poly['max_after']})")
    created = [k for k, v in tasks.items() if v["task_created"] and v["task_id"]]
    skipped = [k for k, v in tasks.items() if v["task_exists"]]
    planned = [k for k, v in tasks.items() if v["task_created"] and not v["task_id"]]
    print(f"Tasks: created={len(created)} existed={len(skipped)} planned={len(planned)}")
    for k, v in tasks.items():
        marker = "EXISTS" if v["task_exists"] else ("CREATED" if v["task_id"] else "PLANNED")
        print(f"  [{marker}] {k}  task_id={v['task_id']}")


def push_command(args: argparse.Namespace) -> int:
    try:
        settings = _merge_settings(args)
        _validate(settings)
        s3_prefix = _resolve_s3_prefix(settings)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    epsilon, max_points = resolve_simplify(
        settings["polygon_preprocess_epsilon"],
        settings["polygon_preprocess_maxpoints"])

    taskman = TaskManager.from_config(settings["ls_config"])
    report = push_coco(
        taskman=taskman,
        coco_path=settings["coco"],
        image_dir=settings["image_dir"],
        s3_prefix=s3_prefix,
        region_id_pattern=settings["region_id_pattern"],
        model_version=settings["model_version"],
        map_bbox=settings["map_bbox"],
        map_seg=settings["map_seg"],
        include=settings["include"],
        exclude=settings["exclude"],
        polygon_epsilon=epsilon,
        polygon_max_points=max_points,
        dry_run=bool(settings["dry_run"]),
    )
    _print_report(report)
    return 0


def pull_command(args: argparse.Namespace) -> int:
    print("cocostudio pull is not yet implemented.", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``cocostudio`` console script."""
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
