"""Generic COCO → Label Studio push.

No dataset knowledge lives here: callers pass the COCO dict, image dir, and
pattern strings that name how to derive region IDs and model versions from
the COCO contents. Project-specific glue (paths, naming conventions, etc.)
goes in a sibling package (e.g. ``sam2dirty``).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from pycocotools import mask as mask_utils
from simplification.cutil import simplify_coords
from tqdm import tqdm

from label_studio_sdk.converter.brush import mask2rle
from labelstudio_tools import TaskManager
from labelstudio_tools.utils.s3_tools import s3_object_exists


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_images(coco: dict, include: Optional[Iterable[str]] = None,
                  exclude: Optional[Iterable[str]] = None) -> list[dict]:
    """Filter ``coco['images']`` by include/exclude lists.

    Each filter value is matched against: integer ``image.id``, the full
    ``image.file_name``, the basename, and the stem (no extension). A value
    parseable as int is also compared against ``image.id`` directly.
    """
    images = coco.get('images', [])
    if include is None and exclude is None:
        return list(images)

    def matches(img: dict, values: Iterable[str]) -> bool:
        fn = img['file_name']
        candidates = {str(img['id']), fn, os.path.basename(fn),
                      os.path.splitext(os.path.basename(fn))[0]}
        for v in values:
            if str(v) in candidates:
                return True
        return False

    selected = list(images)
    if include is not None:
        include = list(include)
        selected = [img for img in selected if matches(img, include)]
    if exclude is not None:
        exclude = list(exclude)
        selected = [img for img in selected if not matches(img, exclude)]
    return selected


def annotations_for_image(coco: dict, image_id: int) -> list[dict]:
    return [a for a in coco.get('annotations', []) if a['image_id'] == image_id]


# ---------------------------------------------------------------------------
# Pattern formatting:  "{annotation.meta.roi_id}" → resolves dotted paths
# ---------------------------------------------------------------------------

_PATTERN_RE = re.compile(r'\{([^{}]+)\}')


def format_pattern(pattern: str, **scopes: dict) -> str:
    """Resolve ``{scope.dotted.path}`` tokens against the given scope dicts.

    Each scope is a top-level name; the rest of the dotted path indexes into
    the scope dict. Missing keys raise KeyError with the full path.
    """
    def replace(m: re.Match) -> str:
        path = m.group(1)
        head, *rest = path.split('.')
        if head not in scopes:
            raise KeyError(f"Pattern {pattern!r} references unknown scope "
                           f"{head!r}. Known scopes: {sorted(scopes)}")
        cur: Any = scopes[head]
        for part in rest:
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(f"Pattern {pattern!r}: {head}.{'.'.join(rest)} "
                               f"missing (failed at {part!r})")
            cur = cur[part]
        return str(cur)
    return _PATTERN_RE.sub(replace, pattern)


# ---------------------------------------------------------------------------
# Control discovery
# ---------------------------------------------------------------------------

# The six mappable LS control tags, by lowercased XML tag name. Each maps to
# ``(family, has_labels)``. ``family`` ('rectangle'|'polygon'|'brush') decides
# whether a control is eligible for --map-bbox (rectangle) or --map-seg
# (polygon/brush). ``has_labels`` is True for the ``*Labels`` variants, which
# carry a label list and attach a category name; the bare variants
# (``<Rectangle>``/``<Polygon>``/``<Brush>``) push geometry with no label.
_CONTROL_TAGS = {
    'rectangle':       ('rectangle', False),
    'rectanglelabels': ('rectangle', True),
    'polygon':         ('polygon',   False),
    'polygonlabels':   ('polygon',   True),
    'brush':           ('brush',     False),
    'brushlabels':     ('brush',     True),
}

# Family -> which kinds of COCO geometry it can receive.
_BBOX_FAMILIES = frozenset({'rectangle'})
_SEG_FAMILIES = frozenset({'polygon', 'brush'})


def _make_control(from_name: str, to_name: Optional[str], type_key: str,
                  labels: Optional[Iterable[str]] = None) -> dict:
    """Build a control entry dict from a tag's name/type/labels.

    ``value_key`` is the labels key for ``*Labels`` controls (same string as the
    result type) or ``None`` for the bare, label-less variants.
    """
    family, has_labels = _CONTROL_TAGS[type_key]
    return {
        'from_name': from_name,
        'to_name': to_name,
        'type_key': type_key,
        'family': family,
        'result_type': type_key,
        'value_key': type_key if has_labels else None,
        'labels': set(labels or ()),
    }


class ControlMap:
    """The mappable Label Studio controls discovered in a project's label config.

    Holds a flat list of control entries (see :func:`_make_control`) and resolves
    a ``--map-bbox`` / ``--map-seg`` value to exactly one of them.
    """

    def __init__(self, entries: list[dict]):
        self.entries = list(entries)
        self._by_name = {e['from_name']: e for e in self.entries}

    def __iter__(self):
        return iter(self.entries)

    @property
    def names(self) -> list[str]:
        return [e['from_name'] for e in self.entries]

    def resolve(self, value: str, *, families: frozenset, what: str) -> dict:
        """Resolve a map value to one control restricted to ``families``.

        ``value`` is first matched against control names (``from_name``); failing
        that, it is treated as a tag *type* (one of :data:`_CONTROL_TAGS`) and
        matched by type. A type with exactly one matching control resolves to it;
        more than one is ambiguous (lists the names); none is an error. The
        resolved control's family must be in ``families``.
        """
        # 1) by control name
        ctrl = self._by_name.get(value)
        if ctrl is not None:
            if ctrl['family'] not in families:
                raise ValueError(
                    f"{what} target {value!r} is a {ctrl['type_key']} control, "
                    f"not usable here (need one of: {sorted(families)})")
            return ctrl

        # 2) by tag type
        type_key = value.lower()
        if type_key in _CONTROL_TAGS:
            family = _CONTROL_TAGS[type_key][0]
            if family not in families:
                raise ValueError(
                    f"{what} target type {value!r} is {family}, not usable here "
                    f"(need one of: {sorted(families)})")
            matches = [e for e in self.entries if e['type_key'] == type_key]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    f"{what} target type {value!r} is ambiguous — the project "
                    f"has {len(matches)} {value} controls: "
                    f"{[m['from_name'] for m in matches]}. Use a specific name.")
            raise ValueError(
                f"{what} target type {value!r} has no control in the label "
                f"config. Available controls: {self._eligible(families)}")

        # 3) neither a name nor a known tag type
        raise ValueError(
            f"{what} target {value!r} is neither a control name nor a mappable "
            f"tag type ({sorted(_CONTROL_TAGS)}). "
            f"Available controls: {self._eligible(families)}")

    def _eligible(self, families: frozenset) -> list[str]:
        return [f"{e['from_name']} ({e['type_key']})"
                for e in self.entries if e['family'] in families]


def discover_controls(taskman: TaskManager) -> ControlMap:
    """Inspect a TaskManager's LS project and return its mappable controls.

    Reads ``parsed_label_config`` (a dict of ``{name: {type, to_name, labels, ...}}``)
    and keeps the controls whose XML tag is one of :data:`_CONTROL_TAGS`.
    """
    parsed = taskman.config_asdict()
    entries: list[dict] = []
    for ctrl_name, info in parsed.items():
        type_key = str(info.get('type', '')).lower()
        if type_key not in _CONTROL_TAGS:
            continue
        to_name = info.get('to_name')
        if isinstance(to_name, list):
            to_name = to_name[0]
        entries.append(_make_control(
            ctrl_name, to_name, type_key, info.get('labels', ())))
    return ControlMap(entries)


# ---------------------------------------------------------------------------
# Annotation → LS results
# ---------------------------------------------------------------------------

def _bbox_value(bbox_xywh: tuple, img_w: int, img_h: int) -> dict:
    x, y, w, h = bbox_xywh
    return {
        'x':      x / img_w * 100.0,
        'y':      y / img_h * 100.0,
        'width':  w / img_w * 100.0,
        'height': h / img_h * 100.0,
        'rotation': 0,
    }


# Iterations of the epsilon binary search used to honour ``max_points``.
# Ported, with ``_simplify_polygon``, from ml_backends/SAM2BigImg postprocess.
_EPSILON_SEARCH_STEPS = 12


def _rdp_closed(points: np.ndarray, eps: float) -> list[float]:
    """Ramer-Douglas-Peucker on a closed ring → flat ``[x, y, x, y, ...]`` list.

    ``simplify_coords`` simplifies an open polyline, so the first vertex is
    appended to close the ring before simplification and the duplicated closing
    vertex is dropped from the result.
    """
    closed = np.vstack([points, points[:1]])
    out = np.asarray(simplify_coords(closed, eps))[:-1]
    return [float(c) for pt in out for c in pt]


def _simplify_polygon(ring: list[float], epsilon: float,
                      max_points: Optional[int]) -> list[float]:
    """Douglas-Peucker simplification of a flat ``[x, y, x, y, ...]`` ring.

    Ported from ``ml_backends/SAM2BigImg`` ``postprocess._simplify``: ``epsilon``
    below 1 is a fraction of the ring's perimeter (resolution-independent);
    ``epsilon`` >= 1 is an absolute pixel distance. When ``max_points`` is set
    and the base epsilon overshoots it, a larger epsilon is binary-searched
    (more epsilon → fewer points, monotonic). Never drops below 3 vertices.
    """
    if len(ring) // 2 <= 3:
        return ring
    pts = np.asarray(ring, dtype=np.float64).reshape(-1, 2)
    # Closed ring: include the closing segment back to the first vertex.
    seg = np.diff(pts, axis=0, append=pts[:1])
    perimeter = float(np.linalg.norm(seg, axis=1).sum())
    if perimeter <= 0:
        return ring
    base_eps = epsilon * perimeter if epsilon < 1.0 else float(epsilon)

    approx = _rdp_closed(pts, base_eps)
    if max_points is not None and len(approx) // 2 > max_points:
        lo, hi = base_eps, max(base_eps, perimeter)
        best = approx
        for _ in range(_EPSILON_SEARCH_STEPS):
            mid = (lo + hi) / 2.0
            approx = _rdp_closed(pts, mid)
            if len(approx) // 2 <= max_points:
                best, hi = approx, mid
            else:
                lo = mid
        approx = best
    if len(approx) // 2 < 3:  # over-simplified into a degenerate ring; keep original
        return ring
    return approx


def _polygon_value(segmentation: list, img_w: int, img_h: int, *,
                   epsilon: Optional[float] = None,
                   max_points: Optional[int] = None,
                   stats: Optional[list] = None) -> dict:
    """COCO polygon (list of flat [x,y,x,y,...] rings) → LS polygon points.

    When ``epsilon`` is set the ring is Douglas-Peucker simplified before
    normalization. If ``stats`` is given, ``(n_before, n_after)`` vertex counts
    are appended for the caller's report.
    """
    if not segmentation:
        raise ValueError("empty polygon segmentation")
    ring = segmentation[0]  # LS PolygonLabels only supports one ring per region
    n_before = len(ring) // 2
    if epsilon is not None:
        ring = _simplify_polygon(ring, epsilon, max_points)
    n_after = len(ring) // 2
    if stats is not None:
        stats.append((n_before, n_after))
    pts = [[ring[i] / img_w * 100.0, ring[i + 1] / img_h * 100.0]
           for i in range(0, len(ring), 2)]
    return {'points': pts, 'closed': True}


def coco_segmentation_to_mask(segmentation: Any, img_h: int, img_w: int) -> np.ndarray:
    """Rasterize COCO segmentation (polygon, RLE, or compressed RLE) to a binary mask.

    Returns shape (H, W) uint8 with values in {0, 255}.
    """
    if isinstance(segmentation, list):  # polygons
        rles = mask_utils.frPyObjects(segmentation, img_h, img_w)
        rle = mask_utils.merge(rles)
        m = mask_utils.decode(rle)
    elif isinstance(segmentation, dict):  # RLE / compressed RLE
        seg = segmentation
        if isinstance(seg.get('counts'), list):
            seg = mask_utils.frPyObjects(seg, img_h, img_w)
        m = mask_utils.decode(seg)
    else:
        raise TypeError(f"unsupported segmentation type: {type(segmentation).__name__}")
    return (m > 0).astype(np.uint8) * 255


def _brush_value(segmentation: Any, img_w: int, img_h: int) -> dict:
    mask = coco_segmentation_to_mask(segmentation, img_h, img_w)
    rle = mask2rle(mask)
    return {'format': 'rle', 'rle': rle}


def coco_anno_to_ls_results(annotation: dict, image: dict, *,
                            region_id: str,
                            bbox_ctrl: Optional[dict] = None,
                            seg_ctrl: Optional[dict] = None,
                            score: Optional[float] = None,
                            polygon_epsilon: Optional[float] = None,
                            polygon_max_points: Optional[int] = None,
                            polygon_stats: Optional[list] = None) -> list[dict]:
    """Build LS result dicts for one COCO annotation.

    Emits a rectangle result on ``bbox_ctrl`` (if given and the annotation has a
    bbox) and/or a segmentation result on ``seg_ctrl`` (if given and the
    annotation has segmentation). ``seg_ctrl['family']`` ('polygon' or 'brush')
    selects the segmentation encoding. All result entries share ``region_id``.

    A control with ``value_key=None`` (a bare ``<Rectangle>``/``<Polygon>``/
    ``<Brush>`` tag) carries no label list, so its result omits the category
    name. ``polygon_epsilon`` / ``polygon_max_points`` apply to the polygon path.
    """
    img_w, img_h = image['width'], image['height']
    label = annotation['_category_name']  # injected by build_predictions
    score = score if score is not None else annotation.get('score')

    def shell(ctrl: dict, value: dict) -> dict:
        value = dict(value)
        if ctrl['value_key'] is not None:
            value[ctrl['value_key']] = [label]
        out = {
            'id': region_id,
            'type': ctrl['result_type'],
            'from_name': ctrl['from_name'],
            'to_name': ctrl['to_name'],
            'original_width': img_w,
            'original_height': img_h,
            'image_rotation': 0,
            'value': value,
        }
        if score is not None:
            out['score'] = float(score)
        return out

    results: list[dict] = []
    if bbox_ctrl is not None and 'bbox' in annotation:
        results.append(shell(bbox_ctrl, _bbox_value(annotation['bbox'], img_w, img_h)))

    seg = annotation.get('segmentation')
    if seg_ctrl is not None and seg:
        if seg_ctrl['family'] == 'polygon':
            results.append(shell(seg_ctrl,
                                 _polygon_value(seg, img_w, img_h,
                                                epsilon=polygon_epsilon,
                                                max_points=polygon_max_points,
                                                stats=polygon_stats)))
        else:  # brush
            results.append(shell(seg_ctrl, _brush_value(seg, img_w, img_h)))
    return results


# ---------------------------------------------------------------------------
# S3 image staging
# ---------------------------------------------------------------------------

def ensure_s3_images(s3_client, bucket: str, prefix: str,
                     file_names: Iterable[str], image_dir: Union[str, os.PathLike],
                     *, dry_run: bool = False) -> dict:
    """For each ``file_name``, verify ``s3://{bucket}/{prefix}/{file_name}`` exists.

    Missing objects are uploaded from ``{image_dir}/{file_name}`` (unless
    dry_run). Returns a report dict with lists of names per status.
    """
    image_dir = Path(image_dir)
    prefix = prefix.rstrip('/')
    report = {'existed': [], 'uploaded': [], 'would_upload': [], 'missing_local': []}

    for name in tqdm(list(file_names), desc='checking s3', unit='img'):
        key = f"{prefix}/{name}" if prefix else name
        if s3_object_exists(s3_client, key, bucket=bucket):
            report['existed'].append(name)
            continue
        local = image_dir / name
        if not local.exists():
            report['missing_local'].append(name)
            continue
        if dry_run:
            report['would_upload'].append(name)
            continue
        s3_client.Bucket(bucket).upload_file(str(local), key)
        report['uploaded'].append(name)
    return report


# ---------------------------------------------------------------------------
# Task / prediction builders
# ---------------------------------------------------------------------------

def build_prediction(image: dict, annotations: list[dict], *,
                     region_id_pattern: str, model_version: str,
                     bbox_ctrl: Optional[dict] = None,
                     seg_ctrl: Optional[dict] = None,
                     coco_info: Optional[dict] = None,
                     polygon_epsilon: Optional[float] = None,
                     polygon_max_points: Optional[int] = None,
                     polygon_stats: Optional[list] = None) -> dict:
    """Build a single LS prediction dict for one image's annotations."""
    results: list[dict] = []
    scope_extras = {'image': image, 'info': coco_info or {}}
    for anno in annotations:
        rid = format_pattern(region_id_pattern, annotation=anno, **scope_extras)
        results.extend(coco_anno_to_ls_results(
            anno, image,
            region_id=rid, bbox_ctrl=bbox_ctrl, seg_ctrl=seg_ctrl,
            polygon_epsilon=polygon_epsilon,
            polygon_max_points=polygon_max_points,
            polygon_stats=polygon_stats))
    scores = [r['score'] for r in results if 'score' in r]
    pred = {'model_version': model_version, 'result': results}
    if scores:
        pred['score'] = float(sum(scores) / len(scores))
    return pred


def build_task(image: dict, prediction: dict, s3_url: str, *,
               pk_field: str, extra_data: Optional[dict] = None) -> dict:
    data = {
        pk_field: s3_url,
        'file_name': image['file_name'],
        'width': image['width'],
        'height': image['height'],
    }
    if image.get('meta'):
        data['meta'] = image['meta']
    if extra_data:
        data.update(extra_data)
    return {'data': data, 'predictions': [prediction]}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _coco_categories_by_id(coco: dict) -> dict[int, str]:
    return {c['id']: c['name'] for c in coco.get('categories', [])}


def _annotate_category_names(annotations: list[dict], cat_by_id: dict[int, str]) -> None:
    for a in annotations:
        a['_category_name'] = cat_by_id[a['category_id']]


def _check_categories(coco: dict, controls: list[dict]) -> None:
    """Ensure every COCO category name is a valid label on each *labeled* target.

    Bare (label-less) controls carry no label list, so there is nothing to
    validate against — they are skipped.
    """
    cat_names = {c['name'] for c in coco.get('categories', [])}
    for ctrl in controls:
        if ctrl['value_key'] is None:
            continue
        missing = cat_names - ctrl['labels']
        if missing:
            raise ValueError(
                f"COCO categories {sorted(missing)} are not in "
                f"{ctrl['from_name']!r} labels {sorted(ctrl['labels'])}")


def _resolve_model_version(model_version: Optional[str], coco: dict) -> str:
    """Resolve a ``model_version`` template against the COCO ``info`` scope.

    A falsy ``model_version`` falls back to ``{info.description}-{info.date_created}``
    (best effort). Otherwise the string is treated as a :func:`format_pattern`
    template over the ``info`` scope; a token referencing missing info raises.
    """
    info = coco.get('info') or {}
    if not model_version:
        desc = info.get('description') or 'coco'
        date = info.get('date_created')
        return f"{desc}-{date}" if date else desc
    try:
        return format_pattern(model_version, info=info)
    except KeyError as e:
        raise ValueError(
            f"model_version template {model_version!r} references COCO info "
            f"that is missing: {e}") from e


def _load_coco(coco_path: Union[str, os.PathLike]) -> dict:
    with open(coco_path) as f:
        return json.load(f)


def _summarize_polygon_stats(stats: list, epsilon: Optional[float],
                             max_points: Optional[int]) -> dict:
    """Aggregate ``(n_before, n_after)`` polygon vertex counts into a report."""
    summary = {'epsilon': epsilon, 'max_points': max_points,
               'polygons': len(stats)}
    if not stats:
        return summary
    before = [b for b, _ in stats]
    after = [a for _, a in stats]
    total_before, total_after = sum(before), sum(after)
    summary.update({
        'points_before': total_before,
        'points_after': total_after,
        'reduction_pct': (1.0 - total_after / total_before) * 100.0
                         if total_before else 0.0,
        'avg_before': total_before / len(stats),
        'avg_after': total_after / len(stats),
        'min_after': min(after),
        'max_after': max(after),
    })
    return summary


def push_coco(taskman: TaskManager, coco_path: Union[str, os.PathLike],
              image_dir: Union[str, os.PathLike], *,
              s3_prefix: str,
              region_id_pattern: str = "{annotation.id}",
              model_version: Optional[str] = None,
              map_bbox: Optional[str] = None,
              map_seg: Optional[str] = None,
              include: Optional[Iterable[str]] = None,
              exclude: Optional[Iterable[str]] = None,
              polygon_epsilon: Optional[float] = None,
              polygon_max_points: Optional[int] = None,
              dry_run: bool = False,
              pk_field: Optional[str] = None) -> dict:
    """Generic COCO → Label Studio push.

    1. Resolve ``map_bbox`` / ``map_seg`` to LS controls in the project.
    2. Filter images via include/exclude.
    3. Verify every COCO category name is a valid label on each labeled target.
    4. Stage images to S3 (upload missing).
    5. Build one task per image with one prediction (model_version).
    6. Create tasks via TaskManager (dedups by pk).

    ``map_bbox`` names a rectangle-family control to receive COCO bboxes;
    ``map_seg`` names a polygon- or brush-family control to receive COCO
    segmentations (the family decides polygon vs. brush encoding). Each value
    may be a control name or a tag type (see :meth:`ControlMap.resolve`). At
    least one of the two is required.

    ``polygon_epsilon`` / ``polygon_max_points`` enable Douglas-Peucker
    simplification of polygon segmentations (rejected for a brush ``map_seg``).
    Left as ``None`` the raw COCO polygons are pushed unchanged.
    """
    if not map_bbox and not map_seg:
        raise ValueError(
            "nothing to push: provide at least one of map_bbox / map_seg")

    coco = _load_coco(coco_path)
    cat_by_id = _coco_categories_by_id(coco)

    controls = discover_controls(taskman)
    bbox_ctrl = (controls.resolve(map_bbox, families=_BBOX_FAMILIES,
                                  what='--map-bbox') if map_bbox else None)
    seg_ctrl = (controls.resolve(map_seg, families=_SEG_FAMILIES,
                                 what='--map-seg') if map_seg else None)

    if (seg_ctrl is not None and seg_ctrl['family'] == 'brush'
            and (polygon_epsilon is not None or polygon_max_points is not None)):
        raise ValueError(
            f"polygon preprocessing was requested but --map-seg target "
            f"{seg_ctrl['from_name']!r} is a brush control; brush segmentations "
            f"are rasterized to RLE, not polygon-simplified")

    seg_kind = seg_ctrl['family'] if seg_ctrl is not None else None
    kinds = tuple(k for k, c in (('bbox', bbox_ctrl), ('mask', seg_ctrl))
                  if c is not None)
    _check_categories(coco, [c for c in (bbox_ctrl, seg_ctrl) if c is not None])

    model_version = _resolve_model_version(model_version, coco)

    images = select_images(coco, include=include, exclude=exclude)
    if not images:
        raise ValueError("no images selected after include/exclude filters")

    image_ids = {img['id'] for img in images}
    annos_by_image: dict[int, list[dict]] = {iid: [] for iid in image_ids}
    for a in coco.get('annotations', []):
        if a['image_id'] in annos_by_image:
            annos_by_image[a['image_id']].append(a)
    for annos in annos_by_image.values():
        _annotate_category_names(annos, cat_by_id)

    bucket = taskman.s3_bucket.name if taskman.s3_bucket is not None else None
    if bucket is None:
        raise ValueError("project config has no S3 storage; cannot stage images")

    s3_report = ensure_s3_images(
        taskman.s3, bucket, s3_prefix,
        [img['file_name'] for img in images], image_dir,
        dry_run=dry_run)
    if s3_report['missing_local']:
        raise FileNotFoundError(
            f"images not in S3 and not on disk: {s3_report['missing_local']}")

    pk = pk_field or taskman.task_pk_datafields
    if not pk:
        raise ValueError("no pk_field specified (and TaskManager has none)")
    if not isinstance(pk, str):
        raise ValueError(f"pk_field must be a single string, got {pk!r}")

    s3_prefix_clean = s3_prefix.rstrip('/')
    polygon_stats: list = []
    tasks: list[dict] = []
    for img in images:
        annos = annos_by_image[img['id']]
        pred = build_prediction(
            img, annos,
            region_id_pattern=region_id_pattern,
            model_version=model_version,
            bbox_ctrl=bbox_ctrl,
            seg_ctrl=seg_ctrl,
            coco_info=coco.get('info'),
            polygon_epsilon=polygon_epsilon,
            polygon_max_points=polygon_max_points,
            polygon_stats=polygon_stats)
        key = f"{s3_prefix_clean}/{img['file_name']}" if s3_prefix_clean else img['file_name']
        s3_url = f"s3://{bucket}/{key}"
        tasks.append(build_task(img, pred, s3_url, pk_field=pk))

    task_report, _responses = taskman.create_tasks(tasks, pk_datafields=pk, dry_run=dry_run)

    report = {
        's3': s3_report,
        'tasks': task_report,
        'selected_images': [img['file_name'] for img in images],
        'kinds': tuple(kinds),
        'segmentation_kind': seg_kind,
        'model_version': model_version,
    }
    if seg_kind == 'polygon':
        report['polygon_simplification'] = _summarize_polygon_stats(
            polygon_stats, polygon_epsilon, polygon_max_points)
    return report
