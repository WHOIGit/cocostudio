"""Tests for cocostudio.push — all pure/local, no LS/S3 connections."""

import json
from pathlib import Path
import numpy as np
import pytest

from cocostudio.push import (
    ControlMap,
    _bbox_value,
    _brush_value,
    _polygon_value,
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


@pytest.fixture
def brush_controls():
    return ControlMap({
        "brush": {
            "from_name": "brush",
            "to_name": "image",
            "result_type": "brushlabels",
            "value_key": "brushlabels",
            "labels": {"tooth", "denticle", "other", "junk", "unlabeled"},
        }
    })


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

def test_results_brush_only(test_annos, test_image, brush_controls):
    anno = test_annos[0]
    results = coco_anno_to_ls_results(
        anno, test_image, brush_controls, region_id="test-id"
    )
    # only brushlabels (no rectangle control)
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


def test_results_region_id_propagated(test_annos, test_image, brush_controls):
    anno = test_annos[0]
    results = coco_anno_to_ls_results(
        anno, test_image, brush_controls, region_id="my-unique-id"
    )
    assert all(r["id"] == "my-unique-id" for r in results)


def test_results_score_propagated(test_annos, test_image, brush_controls):
    anno = test_annos[0]
    results = coco_anno_to_ls_results(
        anno, test_image, brush_controls, region_id="x"
    )
    # annotation has a score field
    for r in results:
        assert "score" in r
        assert r["score"] == pytest.approx(anno["score"])


def test_results_all_annotations(test_annos, test_image, brush_controls):
    for i, anno in enumerate(test_annos):
        results = coco_anno_to_ls_results(
            anno, test_image, brush_controls, region_id=f"r{i}"
        )
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# build_prediction
# ---------------------------------------------------------------------------

def test_build_prediction_structure(test_annos, test_image, brush_controls):
    pred = build_prediction(
        test_image, test_annos, brush_controls,
        region_id_pattern="roi_{annotation.meta.roi_id}",
        model_version="test-model-v1",
    )
    assert pred["model_version"] == "test-model-v1"
    assert isinstance(pred["result"], list)
    assert len(pred["result"]) == len(test_annos)  # one brush result per anno
    assert "score" in pred
    assert 0.0 <= pred["score"] <= 1.0


def test_build_prediction_region_ids_unique(test_annos, test_image, brush_controls):
    pred = build_prediction(
        test_image, test_annos, brush_controls,
        region_id_pattern="roi_{annotation.meta.roi_id}",
        model_version="m",
    )
    ids = [r["id"] for r in pred["result"]]
    assert len(ids) == len(set(ids)), "region IDs must be unique across results"


def test_build_prediction_region_id_contains_roi_id(test_annos, test_image, brush_controls):
    pred = build_prediction(
        test_image, test_annos, brush_controls,
        region_id_pattern="roi_{annotation.meta.roi_id}",
        model_version="m",
    )
    for r in pred["result"]:
        assert r["id"].startswith("roi_")
