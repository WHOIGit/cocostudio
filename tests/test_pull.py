"""Pure/local tests for cocostudio.pull."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from cocostudio.pull import (
    _bbox_result_to_coco_annotation,
    apply_postprocess_function,
    build_coco_from_tasks,
    cache_path_for,
    choose_annotation,
    validate_label_policy,
)
from cocostudio.push import _make_control


class FakeTaskManager:
    host = "https://ichthyolith.whoi.edu"
    project = SimpleNamespace(id=9, title="sam2dirty cleanup - RectangleLabels")

    def download_s3url(self, *args, **kwargs):
        raise AssertionError("download_s3url should not be called")


def _source_coco():
    return {
        "info": {"description": "source"},
        "licenses": [],
        "categories": [
            {"id": 1, "name": "tooth", "supercategory": "ichthyolith"},
            {"id": 2, "name": "denticle", "supercategory": "ichthyolith"},
        ],
        "images": [
            {
                "id": 42,
                "file_name": "img.jpg",
                "width": 200,
                "height": 100,
                "meta": {"original_width": 400},
            }
        ],
        "annotations": [],
    }


def _task(label="tooth"):
    return {
        "id": 100,
        "data": {
            "image": "s3://bucket/prefix/img.jpg",
            "file_name": "img.jpg",
            "width": 200,
            "height": 100,
            "meta": {"fullframe_id": "ff-1"},
        },
        "annotations": [
            {
                "id": 7,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-02T00:00:00Z",
                "result": [
                    {
                        "id": "roi_img_obj00001_plane000",
                        "type": "rectanglelabels",
                        "from_name": "bbox",
                        "to_name": "image",
                        "original_width": 200,
                        "original_height": 100,
                        "value": {
                            "x": 10,
                            "y": 20,
                            "width": 30,
                            "height": 40,
                            "rectanglelabels": [label],
                        },
                    }
                ],
            }
        ],
    }


def _bbox_ctrl(labels=("tooth", "denticle")):
    return _make_control("bbox", "image", "rectanglelabels", labels)


def test_cache_path_for_uses_host_project_and_cache_id(tmp_path):
    cfg = tmp_path / "cocostudio" / "configs" / "pushpull" / "cfg.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("")
    path = cache_path_for(FakeTaskManager(), cfg, "cfg")
    assert path == (
        tmp_path
        / "cocostudio"
        / "cache"
        / "ichthyolith.whoi.edu"
        / "sam2dirty_cleanup_-_RectangleLabels"
        / "cfg.json"
    )


def test_choose_annotation_multiple_latest():
    task = {
        "id": 1,
        "annotations": [
            {"id": 1, "updated_at": "2026-01-01T00:00:00Z"},
            {"id": 2, "updated_at": "2026-01-02T00:00:00Z"},
        ],
    }
    anno, status = choose_annotation(
        task, completed_only=True, annotation_policy="latest")
    assert status == "selected"
    assert anno["id"] == 2


def test_choose_annotation_multiple_error():
    task = {"id": 1, "annotations": [{"id": 1}, {"id": 2}]}
    with pytest.raises(ValueError, match="multiple submitted annotations"):
        choose_annotation(task, completed_only=True, annotation_policy="error")


def test_validate_label_policy_requires_exact_push_match():
    with pytest.raises(ValueError, match="do not match"):
        validate_label_policy(
            "match_push_coco",
            _source_coco(),
            [_bbox_ctrl(labels=("tooth", "denticle", "fragment"))],
        )


def test_build_coco_from_tasks_bbox_and_metadata():
    coco, report = build_coco_from_tasks(
        [_task()],
        source_coco=_source_coco(),
        bbox_ctrl=_bbox_ctrl(),
        label_policy="match_push_coco",
        image_datafield="image",
        image_dir=None,
        download_missing_images=False,
        taskman=FakeTaskManager(),
        completed_only=True,
        annotation_policy="error",
    )

    assert report["tasks"]["selected"] == 1
    assert report["annotations"]["count"] == 1
    assert coco["images"][0]["id"] == 42
    assert coco["images"][0]["meta"]["original_width"] == 400
    assert coco["images"][0]["meta"]["fullframe_id"] == "ff-1"
    assert coco["images"][0]["meta"]["ls_task_id"] == 100
    anno = coco["annotations"][0]
    assert anno["id"] == 1
    assert anno["image_id"] == 42
    assert anno["category_id"] == 1
    assert anno["bbox"] == [20.0, 20.0, 60.0, 40.0]
    assert anno["area"] == 2400.0
    assert anno["meta"]["ls_region_id"] == "roi_img_obj00001_plane000"
    assert "roi_id" not in anno["meta"]
    assert "obj_id" not in anno["meta"]


def test_build_coco_reuses_source_annotation_ids_by_image_bbox():
    source_coco = _source_coco()
    source_coco["annotations"] = [
        {"id": 17, "image_id": 42, "category_id": 1, "bbox": [20.0, 20.0, 60.0, 40.0]},
        {"id": 20, "image_id": 42, "category_id": 1, "bbox": [1.0, 2.0, 3.0, 4.0]},
    ]
    task = deepcopy(_task())
    task["annotations"][0]["result"].append({
        "id": "roi_new",
        "type": "rectanglelabels",
        "from_name": "bbox",
        "to_name": "image",
        "original_width": 200,
        "original_height": 100,
        "value": {
            "x": 50,
            "y": 10,
            "width": 5,
            "height": 5,
            "rectanglelabels": ["tooth"],
        },
    })

    coco, _report = build_coco_from_tasks(
        [task],
        source_coco=source_coco,
        bbox_ctrl=_bbox_ctrl(),
        label_policy="match_push_coco",
        image_datafield="image",
        image_dir=None,
        download_missing_images=False,
        taskman=FakeTaskManager(),
        completed_only=True,
        annotation_policy="error",
    )

    assert [anno["id"] for anno in coco["annotations"]] == [17, 21]
    assert coco["annotations"][0]["bbox"] == [20.0, 20.0, 60.0, 40.0]
    assert coco["annotations"][1]["bbox"] == [100.0, 10.0, 10.0, 5.0]


def test_build_coco_prefers_source_annotation_ids_by_region_id():
    source_coco = _source_coco()
    source_coco["annotations"] = [{
        "id": 17,
        "image_id": 42,
        "category_id": 1,
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "meta": {"roi_id": "img_obj00001_plane000"},
    }]

    coco, _report = build_coco_from_tasks(
        [_task()],
        source_coco=source_coco,
        bbox_ctrl=_bbox_ctrl(),
        label_policy="match_push_coco",
        region_id_pattern="roi_{annotation.meta.roi_id}",
        image_datafield="image",
        image_dir=None,
        download_missing_images=False,
        taskman=FakeTaskManager(),
        completed_only=True,
        annotation_policy="error",
    )

    assert coco["annotations"][0]["id"] == 17
    assert coco["annotations"][0]["bbox"] == [20.0, 20.0, 60.0, 40.0]


def test_bbox_conversion_rounds_float_noise():
    result = {
        "id": "roi_1",
        "original_width": 5000,
        "original_height": 1000,
        "value": {
            "x": 87.79,
            "y": 28.7,
            "width": 4.569999999999999,
            "height": 23.299999999999994,
            "rectanglelabels": ["tooth"],
        },
    }

    anno = _bbox_result_to_coco_annotation(
        result,
        image={"id": 1, "width": 5000, "height": 1000},
        annotation_id=2,
        category_id=1,
        ls_annotation={"id": 777},
    )

    assert anno["bbox"] == [4389.5, 287.0, 228.5, 233.0]
    assert anno["area"] == 53240.5


def test_postprocess_function_can_add_project_metadata():
    coco, _report = build_coco_from_tasks(
        [_task()],
        source_coco=_source_coco(),
        bbox_ctrl=_bbox_ctrl(),
        label_policy="match_push_coco",
        image_datafield="image",
        image_dir=None,
        download_missing_images=False,
        taskman=FakeTaskManager(),
        completed_only=True,
        annotation_policy="error",
    )

    apply_postprocess_function(
        coco,
        "configs/pushpull/pull_postprocess_ichthyoliths.py:compute_coco_localization_meta",
        config_path="configs/pushpull/sam2dirty_rect.toml",
    )

    anno = coco["annotations"][0]
    assert anno["meta"]["roi_id"] == "img_obj00001_plane000"
    assert anno["meta"]["obj_id"] == 1


def test_build_coco_allow_any_adds_category():
    coco, _report = build_coco_from_tasks(
        [_task(label="fragment")],
        source_coco=_source_coco(),
        bbox_ctrl=_bbox_ctrl(),
        label_policy="allow_any",
        image_datafield="image",
        image_dir=None,
        download_missing_images=False,
        taskman=FakeTaskManager(),
        completed_only=True,
        annotation_policy="error",
    )
    assert coco["categories"][-1] == {"id": 3, "name": "fragment"}
    assert coco["annotations"][0]["category_id"] == 3
