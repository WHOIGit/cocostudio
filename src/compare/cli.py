#!/usr/bin/env python3
"""Entry point for COCO comparison commands."""

from __future__ import annotations

import argparse
import sys

from .config import ConfigError, build_render_config
from .render import run_render


def _parse_bool(value: str) -> bool:
    text = value.lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value!r}")


def cmd_render(args: argparse.Namespace) -> int:
    try:
        return run_render(build_render_config(args))
    except ConfigError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


def cmd_stub(args: argparse.Namespace) -> int:
    print(f"ERROR: compare {args.cmd} is not implemented yet", file=sys.stderr)
    return 2


def add_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", help="TOML config file")
    parser.add_argument("--coco-a", help="First COCO JSON file")
    parser.add_argument("--coco-b", help="Second COCO JSON file")
    parser.add_argument("--name-a", help="Display name for --coco-a")
    parser.add_argument("--name-b", help="Display name for --coco-b")
    parser.add_argument("--image-root", help="Root directory for COCO image file_name paths")
    parser.add_argument("--output", help="Per-image output template, e.g. data/compare/{FFID}.jpg")
    parser.add_argument(
        "--image-key",
        help="Image field used to map coco-a image IDs to coco-b image IDs. Default: file_name.",
    )
    parser.add_argument(
        "--ffid-field",
        help="A-side image field used to derive {FFID}; Path(value).stem is used. Default: file_name.",
    )

    ffid = parser.add_mutually_exclusive_group()
    ffid.add_argument("--ffid-include", nargs="+", help="Render only these ffids")
    ffid.add_argument("--ffid-exclude", nargs="+", help="Skip these ffids")
    parser.add_argument("--image-index", nargs="+", type=int, help="COCO integer image IDs to render")
    parser.add_argument(
        "--image-index-side",
        choices=("a", "b"),
        help="Which COCO file --image-index applies to. Default: a.",
    )

    parser.add_argument(
        "--draw-bboxes",
        nargs="?",
        const=True,
        default=None,
        type=_parse_bool,
        help="Draw bbox outlines. Bare flag means true.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("none", "fill", "outline", "both"),
        help="How to render COCO segmentation masks. Default: none.",
    )
    parser.add_argument(
        "--invert-masks",
        choices=("none", "a", "b", "both"),
        help="Invert rendered masks for one or both sources. Default: none.",
    )
    parser.add_argument(
        "--text",
        help="Annotation text format. Supports fields like {CATEGORY}, {ROI}, {OBJ:05}, {OBJ_ID}, and annotation properties.",
    )
    parser.add_argument("--color-a", help="Color for --coco-a annotations. Default: cyan.")
    parser.add_argument("--color-b", help="Color for --coco-b annotations. Default: red.")
    parser.add_argument(
        "--label-color",
        action="append",
        help="Override color for a category label, as CATEGORY=COLOR. May be repeated.",
    )
    parser.add_argument("--mask-alpha", type=float, help="Mask fill opacity from 0 to 1")
    parser.add_argument("--line-width", type=int, help="BBox and mask outline width in pixels")
    parser.add_argument("--clobber", action="store_const", const=True, default=None)
    parser.add_argument("--dry-run", action="store_const", const=True, default=None)
    parser.add_argument(
        "--provenance",
        help="Provenance JSON template. Use false/none/null to disable.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare and render COCO annotations")
    sub = parser.add_subparsers(dest="cmd", required=True)

    render = sub.add_parser("render", help="Render all annotations from two COCO files")
    add_render_args(render)
    render.set_defaults(func=cmd_render)

    render_changed = sub.add_parser(
        "render-changed",
        help="Stub for rendering only changed images and annotations",
    )
    render_changed.add_argument("-c", "--config", help="TOML config file")
    render_changed.set_defaults(func=cmd_stub)

    render_diff = sub.add_parser(
        "render-diff",
        help="Stub for rendering explicit annotation/mask diff views",
    )
    render_diff.add_argument("-c", "--config", help="TOML config file")
    render_diff.set_defaults(func=cmd_stub)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

