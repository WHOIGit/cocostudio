"""Tests for cocostudio.push — all pure/local, no LS/S3 connections."""

import json
from pathlib import Path
import numpy as np
import pytest

from cocostudio.push import (
    ControlMap,
    _BBOX_FAMILIES,
    _SEG_FAMILIES,
    _bbox_value,
    _brush_value,
    _check_categories,
    _make_control,
    _polygon_value,
    _resolve_model_version,
    build_prediction,
    coco_anno_to_ls_results,
    coco_segmentation_to_mask,
    format_pattern,
    select_images,
    _annotate_category_names,
)

COCO_PATH = Path(__file__).parent.parent / "data/from_cleanup/sam2-coco-small.json"
TEST_FILE_NAME = "101_U1553C_22R_1W_28-30cm_N2of2_Z200x.jpg"
TEST_STEM = "101_U1553C_22R_1W_28-30cm_N2of2_Z200x"


@pytest.fixture(scope="module")
def coco():
    with open(COCO_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def test_image(coco):
    return next(img for img in coco["images"] if img["file_name"] == TEST_FILE_NAME)


@pytest.fixture(scope="module")
def test_annos(coco, test_image):
    annos = [a for a in coco["annotations"] if a["image_id"] == test_image["id"]]
    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    _annotate_category_names(annos, cat_by_id)
    return annos


_LABELS = {"tooth", "denticle", "other", "junk", "unlabeled"}


@pytest.fixture
def brush_ctrl():
    return _make_control("brush", "image", "brushlabels", _LABELS)


# ---------------------------------------------------------------------------
# select_images
# ---------------------------------------------------------------------------

def test_select_images_no_filter(coco):
    imgs = select_images(coco)
    assert len(imgs) == len(coco["images"])


def test_select_images_include_stem(coco):
    imgs = select_images(coco, include=[TEST_STEM])
    assert len(imgs) == 1
    assert imgs[0]["file_name"] == TEST_FILE_NAME


def test_select_images_include_file_name(coco):
    imgs = select_images(coco, include=[TEST_FILE_NAME])
    assert len(imgs) == 1


def test_select_images_include_id(coco, test_image):
    imgs = select_images(coco, include=[str(test_image["id"])])
    assert len(imgs) == 1
    assert imgs[0]["id"] == test_image["id"]


def test_select_images_exclude(coco):
    imgs = select_images(coco, exclude=[TEST_STEM])
    assert all(img["file_name"] != TEST_FILE_NAME for img in imgs)
    assert len(imgs) == len(coco["images"]) - 1


def test_select_images_no_match_returns_empty(coco):
    imgs = select_images(coco, include=["nonexistent"])
    assert imgs == []


# ---------------------------------------------------------------------------
# format_pattern
# ---------------------------------------------------------------------------

def test_format_pattern_simple():
    assert format_pattern("{annotation.id}", annotation={"id": 42}) == "42"


def test_format_pattern_nested():
    anno = {"meta": {"roi_id": "U1553_obj001"}}
    result = format_pattern("roi_{annotation.meta.roi_id}", annotation=anno)
    assert result == "roi_U1553_obj001"


def test_format_pattern_missing_scope():
    with pytest.raises(KeyError, match="unknown scope"):
        format_pattern("{bogus.field}", annotation={})


def test_format_pattern_missing_key():
    with pytest.raises(KeyError):
        format_pattern("{annotation.meta.missing}", annotation={"meta": {}})


def test_format_pattern_no_tokens():
    assert format_pattern("static-id", annotation={}) == "static-id"


# ---------------------------------------------------------------------------
# Coordinate converters
# ---------------------------------------------------------------------------

def test_bbox_value_unit():
    v = _bbox_value((100, 200, 50, 80), 1000, 800)
    assert v["x"] == pytest.approx(10.0)
    assert v["y"] == pytest.approx(25.0)
    assert v["width"] == pytest.approx(5.0)
    assert v["height"] == pytest.approx(10.0)
    assert v["rotation"] == 0


def test_polygon_value_converts_to_percent(test_annos, test_image):
    anno = test_annos[0]
    seg = anno["segmentation"]
    v = _polygon_value(seg, test_image["width"], test_image["height"])
    assert "points" in v
    assert v["closed"] is True
    for pt in v["points"]:
        assert 0.0 <= pt[0] <= 100.0, f"x {pt[0]} out of range"
        assert 0.0 <= pt[1] <= 100.0, f"y {pt[1]} out of range"


def test_polygon_value_point_count(test_annos, test_image):
    anno = test_annos[0]
    ring = anno["segmentation"][0]
    expected_pts = len(ring) // 2
    v = _polygon_value(anno["segmentation"], test_image["width"], test_image["height"])
    assert len(v["points"]) == expected_pts


# ---------------------------------------------------------------------------
# Mask / brush conversion
# ---------------------------------------------------------------------------

def test_coco_segmentation_to_mask_shape(test_annos, test_image):
    anno = test_annos[0]
    mask = coco_segmentation_to_mask(
        anno["segmentation"], test_image["height"], test_image["width"]
    )
    assert mask.shape == (test_image["height"], test_image["width"])
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_coco_segmentation_to_mask_nonempty(test_annos, test_image):
    anno = test_annos[0]
    mask = coco_segmentation_to_mask(
        anno["segmentation"], test_image["height"], test_image["width"]
    )
    assert mask.sum() > 0, "mask should have some foreground pixels"


def test_brush_value_has_rle(test_annos, test_image):
    anno = test_annos[0]
    v = _brush_value(anno["segmentation"], test_image["width"], test_image["height"])
    assert v["format"] == "rle"
    assert isinstance(v["rle"], list)
    assert len(v["rle"]) > 0


# ---------------------------------------------------------------------------
# coco_anno_to_ls_results
# ---------------------------------------------------------------------------

def test_results_brush_only(test_annos, test_image, brush_ctrl):
    anno = test_annos[0]
    results = coco_anno_to_ls_results(
        anno, test_image, region_id="test-id", seg_ctrl=brush_ctrl
    )
    # only brushlabels (no rectangle control mapped)
    assert len(results) == 1
    r = results[0]
    assert r["type"] == "brushlabels"
    assert r["from_name"] == "brush"
    assert r["to_name"] == "image"
    assert r["id"] == "test-id"
    assert r["original_width"] == test_image["width"]
    assert r["original_height"] == test_image["height"]
    assert isinstance(r["value"]["rle"], list)
    assert r["value"]["brushlabels"] == [anno["_category_name"]]


def test_results_region_id_propagated(test_annos, test_image, brush_ctrl):
    anno = test_annos[0]
    results = coco_anno_to_ls_results(
        anno, test_image, region_id="my-unique-id", seg_ctrl=brush_ctrl
    )
    assert all(r["id"] == "my-unique-id" for r in results)


def test_results_score_propagated(test_annos, test_image, brush_ctrl):
    anno = test_annos[0]
    results = coco_anno_to_ls_results(
        anno, test_image, region_id="x", seg_ctrl=brush_ctrl
    )
    # annotation has a score field
    for r in results:
        assert "score" in r
        assert r["score"] == pytest.approx(anno["score"])


def test_results_all_annotations(test_annos, test_image, brush_ctrl):
    for i, anno in enumerate(test_annos):
        results = coco_anno_to_ls_results(
            anno, test_image, region_id=f"r{i}", seg_ctrl=brush_ctrl
        )
        assert len(results) >= 1


def test_results_labelless_omits_category(test_annos, test_image):
    # A bare <Brush> control (value_key=None) pushes geometry with no label.
    bare = _make_control("brush", "image", "brush")
    anno = test_annos[0]
    results = coco_anno_to_ls_results(
        anno, test_image, region_id="x", seg_ctrl=bare
    )
    assert results[0]["type"] == "brush"
    assert "brushlabels" not in results[0]["value"]
    assert "rle" in results[0]["value"]


def test_results_bbox_and_seg(test_annos, test_image, brush_ctrl):
    rect = _make_control("rect", "image", "rectanglelabels", _LABELS)
    anno = test_annos[0]
    anno = {**anno, "bbox": [1.0, 2.0, 3.0, 4.0]}
    results = coco_anno_to_ls_results(
        anno, test_image, region_id="x", bbox_ctrl=rect, seg_ctrl=brush_ctrl
    )
    types = {r["type"] for r in results}
    assert types == {"rectanglelabels", "brushlabels"}


# ---------------------------------------------------------------------------
# build_prediction
# ---------------------------------------------------------------------------

def test_build_prediction_structure(test_annos, test_image, brush_ctrl):
    pred = build_prediction(
        test_image, test_annos,
        region_id_pattern="roi_{annotation.meta.roi_id}",
        model_version="test-model-v1",
        seg_ctrl=brush_ctrl,
    )
    assert pred["model_version"] == "test-model-v1"
    assert isinstance(pred["result"], list)
    assert len(pred["result"]) == len(test_annos)  # one brush result per anno
    assert "score" in pred
    assert 0.0 <= pred["score"] <= 1.0


def test_build_prediction_region_ids_unique(test_annos, test_image, brush_ctrl):
    pred = build_prediction(
        test_image, test_annos,
        region_id_pattern="roi_{annotation.meta.roi_id}",
        model_version="m",
        seg_ctrl=brush_ctrl,
    )
    ids = [r["id"] for r in pred["result"]]
    assert len(ids) == len(set(ids)), "region IDs must be unique across results"


def test_build_prediction_region_id_contains_roi_id(test_annos, test_image, brush_ctrl):
    pred = build_prediction(
        test_image, test_annos,
        region_id_pattern="roi_{annotation.meta.roi_id}",
        model_version="m",
        seg_ctrl=brush_ctrl,
    )
    for r in pred["result"]:
        assert r["id"].startswith("roi_")


# ---------------------------------------------------------------------------
# ControlMap.resolve  (name / type / family resolution)
# ---------------------------------------------------------------------------

@pytest.fixture
def controls():
    # Mirrors the RectangleLabels BRICK project: two rectangle controls + a brush.
    return ControlMap([
        _make_control("rect", "image", "rectanglelabels", _LABELS),
        _make_control("bbox", "image", "rectanglelabels", _LABELS),
        _make_control("brush", "image", "brushlabels", _LABELS),
    ])


def test_resolve_by_name(controls):
    ctrl = controls.resolve("bbox", families=_BBOX_FAMILIES, what="--map-bbox")
    assert ctrl["from_name"] == "bbox"


def test_resolve_by_type_unique(controls):
    ctrl = controls.resolve("BrushLabels", families=_SEG_FAMILIES, what="--map-seg")
    assert ctrl["from_name"] == "brush"


def test_resolve_by_type_ambiguous_lists_names(controls):
    with pytest.raises(ValueError, match="ambiguous"):
        controls.resolve("RectangleLabels", families=_BBOX_FAMILIES, what="--map-bbox")


def test_resolve_wrong_family(controls):
    # brush is a segmentation control; not valid for --map-bbox.
    with pytest.raises(ValueError, match="not usable here"):
        controls.resolve("brush", families=_BBOX_FAMILIES, what="--map-bbox")


def test_resolve_unknown(controls):
    with pytest.raises(ValueError, match="neither a control name nor"):
        controls.resolve("nope", families=_SEG_FAMILIES, what="--map-seg")


def test_resolve_type_no_control(controls):
    no_poly = ControlMap([_make_control("brush", "image", "brushlabels", _LABELS)])
    with pytest.raises(ValueError, match="no control"):
        no_poly.resolve("PolygonLabels", families=_SEG_FAMILIES, what="--map-seg")


# ---------------------------------------------------------------------------
# _check_categories / _resolve_model_version
# ---------------------------------------------------------------------------

def test_check_categories_missing_label():
    coco = {"categories": [{"id": 1, "name": "mystery"}]}
    ctrl = _make_control("brush", "image", "brushlabels", _LABELS)
    with pytest.raises(ValueError, match="mystery"):
        _check_categories(coco, [ctrl])


def test_check_categories_skips_labelless():
    coco = {"categories": [{"id": 1, "name": "mystery"}]}
    bare = _make_control("brush", "image", "brush")  # no labels
    _check_categories(coco, [bare])  # no raise


def test_model_version_template_resolves():
    coco = {"info": {"date_created": "2026-01-02"}}
    assert _resolve_model_version("sam2-{info.date_created}", coco) == "sam2-2026-01-02"


def test_model_version_missing_info_raises():
    with pytest.raises(ValueError, match="missing"):
        _resolve_model_version("sam2-{info.date_created}", {"info": {}})


def test_model_version_default_from_info():
    coco = {"info": {"description": "ds", "date_created": "2026-01-02"}}
    assert _resolve_model_version(None, coco) == "ds-2026-01-02"
