"""Console-script CLI for ``cocostudio-push``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from labelstudio_tools import TaskManager

from .push import push_coco


def _default_model_version(coco_path: str) -> str:
    """Best-effort default: ``{info.description}-{info.date_created}`` if present."""
    try:
        with open(coco_path) as f:
            info = json.load(f).get("info", {}) or {}
        desc = info.get("description") or "coco"
        date = info.get("date_created")
        return f"{desc}-{date}" if date else desc
    except Exception:
        return "coco-push"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cocostudio-push",
        description="Push COCO predictions to a Label Studio project.",
    )
    p.add_argument("-c", "--config", required=True,
                   help="LS project TOML config (relative to CWD).")
    p.add_argument("--coco", required=True, help="COCO JSON file.")
    p.add_argument("--image-dir", required=True,
                   help="Local image directory; file_names must match COCO.")
    p.add_argument("--s3-prefix", required=True,
                   help="S3 key prefix (e.g. 'fullframe_halfsize').")
    p.add_argument("--include", nargs="+", metavar="VALUE",
                   help="Image id, file_name, or stem(s) to include.")
    p.add_argument("--exclude", nargs="+", metavar="VALUE",
                   help="Image id, file_name, or stem(s) to exclude.")
    p.add_argument("--region-id-pattern", default="{annotation.id}",
                   help="Pattern for LS region IDs. Default: {annotation.id}")
    p.add_argument("--model-version", default=None,
                   help="LS prediction model_version string. "
                        "Default: derived from COCO info.")
    kinds = p.add_mutually_exclusive_group()
    kinds.add_argument("--bbox", action="store_true",
                       help="Push bbox (rectangle) results only.")
    kinds.add_argument("--mask", action="store_true",
                       help="Push segmentation (brush/polygon) results only.")
    p.add_argument("--epsilon", type=float, default=None,
                   help="Douglas-Peucker simplification of polygon "
                        "segmentations. <1 = fraction of perimeter; "
                        ">=1 = absolute pixels. Omit to push raw polygons.")
    p.add_argument("--max-points", type=int, default=None, metavar="N",
                   help="Cap vertices per polygon (epsilon binary-searched up "
                        "to honour it). 0 disables simplification.")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan only — no S3 uploads, no LS task creation.")
    return p


def resolve_simplify(epsilon: float | None,
                     max_points: int | None) -> tuple[float | None, int | None]:
    """Normalize polygon-simplification CLI args into ``(epsilon, max_points)``.

    ``max_points == 0`` is the escape hatch: disables simplification entirely
    (raw polygons). A ``max_points`` cap given without ``epsilon`` falls back
    to ``epsilon=1`` (1px) — the binary search then raises it to honour the cap.
    """
    if max_points == 0:
        return None, None
    if max_points is not None and epsilon is None:
        epsilon = 1.0
    return epsilon, max_points


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


def push_main(argv: list[str] | None = None) -> int:
    """Entry point for the ``cocostudio-push`` console script."""
    args = _build_parser().parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"error: config file not found: {config_path}", file=sys.stderr)
        return 2

    model_version = args.model_version or _default_model_version(args.coco)
    if args.bbox:
        kinds = ("bbox",)
    elif args.mask:
        kinds = ("mask",)
    else:
        kinds = ("bbox", "mask")

    epsilon, max_points = resolve_simplify(args.epsilon, args.max_points)

    taskman = TaskManager.from_config(str(config_path))
    report = push_coco(
        taskman=taskman,
        coco_path=args.coco,
        image_dir=args.image_dir,
        s3_prefix=args.s3_prefix,
        region_id_pattern=args.region_id_pattern,
        model_version=model_version,
        include=args.include,
        exclude=args.exclude,
        kinds=kinds,
        polygon_epsilon=epsilon,
        polygon_max_points=max_points,
        dry_run=args.dry_run,
    )
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(push_main())
