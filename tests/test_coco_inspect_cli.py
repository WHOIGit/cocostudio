from __future__ import annotations

import json
from pathlib import Path

from coco_inspect import cli


def write_sample_coco(tmp_path: Path) -> Path:
    path = tmp_path / "sample-coco.json"
    path.write_text(
        json.dumps(
            {
                "info": {
                    "description": "test fixture",
                    "version": "1.0",
                    "meta": {"export": {"revision": "abc123"}},
                },
                "licenses": [],
                "categories": [
                    {"id": 1, "name": "tooth", "supercategory": "ichthyolith"},
                    {"id": 2, "name": "denticle", "supercategory": "ichthyolith"},
                    {"id": 3, "name": "other", "supercategory": "ichthyolith"},
                ],
                "images": [
                    {
                        "id": 1,
                        "file_name": "FFID_A.jpg",
                        "width": 640,
                        "height": 480,
                        "location": "alpha",
                        "meta": {"camera": "scope-a"},
                    },
                    {
                        "id": 2,
                        "file_name": "nested/FFID_B.jpg",
                        "width": 800,
                        "height": 600,
                        "note": "reviewed",
                    },
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [50, 80, 120, 90],
                        "area": 10800,
                        "iscrowd": 0,
                        "segmentation": [[50, 80, 170, 80, 170, 170, 50, 170]],
                        "meta": {"roi_id": "roi-1", "obj_id": "obj-1"},
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 2,
                        "bbox": [240, 150, 70, 65],
                        "area": 4550,
                        "iscrowd": 0,
                    },
                    {
                        "id": 3,
                        "image_id": 2,
                        "category_id": 2,
                        "bbox": [100, 120, 95, 110],
                        "area": 10450,
                        "iscrowd": 0,
                        "segmentation": {"counts": [10, 4, 2], "size": [600, 800]},
                    },
                    {
                        "id": 4,
                        "image_id": 2,
                        "category_id": 3,
                        "bbox": [400, 300, 160, 140],
                        "area": 22400,
                        "iscrowd": 0,
                        "segmentation": {"counts": "abc123", "size": [600, 800]},
                    },
                ],
            }
        )
    )
    return path


def run_inspect(coco_path: Path, capsys, *args: str) -> tuple[int, str, str]:
    code = cli.main([str(coco_path), *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_default_table_includes_image_rows_sum_and_category_counts(tmp_path, capsys):
    coco_path = write_sample_coco(tmp_path)

    code, out, err = run_inspect(coco_path, capsys)

    assert code == 0
    assert err == ""
    assert "| image_id" in out
    assert "FFID_A.jpg" in out
    assert "nested/FFID_B.jpg" in out
    assert "SUM" in out
    assert "2 images" in out
    assert "annotations" in out
    assert "bbox" in out
    assert "masks" in out
    assert "1:tooth" in out
    assert "2:denticle" in out
    assert "3:other" in out
    assert "location" in out
    assert "meta.camera" in out
    assert "note" in out


def test_column_options_remove_selected_columns(tmp_path, capsys):
    coco_path = write_sample_coco(tmp_path)

    code, out, err = run_inspect(
        coco_path,
        capsys,
        "--no-dims",
        "--no-meta",
        "--no-anno-types",
        "--no-cats",
    )

    assert code == 0
    assert err == ""
    assert "width" not in out
    assert "height" not in out
    assert "location" not in out
    assert "meta.camera" not in out
    assert "bbox" not in out
    assert "masks" not in out
    assert "1:tooth" not in out
    assert "annotations" in out


def test_sum_modes_raw_single_value_table_and_csv(tmp_path, capsys):
    coco_path = write_sample_coco(tmp_path)

    code, out, err = run_inspect(coco_path, capsys, "--sum")
    assert code == 0
    assert err == ""
    assert "| images      | 2" in out
    assert "| annotations | 4" in out
    assert "| bbox        | T" in out
    assert "| masks       | 3" in out

    code, out, err = run_inspect(coco_path, capsys, "--sum", "images")
    assert code == 0
    assert err == ""
    assert out == "2\n"

    code, out, err = run_inspect(coco_path, capsys, "--sum", "annotations", "bbox", "masks", "denticle")
    assert code == 0
    assert err == ""
    assert "| field       |   value |" in out
    assert "| annotations |       4 |" in out
    assert "| bbox        |       4 |" in out
    assert "| masks       |       3 |" in out
    assert "| denticle    |       2 |" in out

    outfile = tmp_path / "sum-fields.csv"
    code, out, err = run_inspect(
        coco_path,
        capsys,
        "--sum",
        "annotations",
        "bbox",
        "-o",
        str(outfile),
    )
    assert code == 0
    assert out == ""
    assert err == ""
    assert outfile.read_text().splitlines() == [
        "field,value",
        "annotations,4",
        "bbox,4",
    ]


def test_images_categories_and_filters(tmp_path, capsys):
    coco_path = write_sample_coco(tmp_path)

    code, out, err = run_inspect(coco_path, capsys, "--images")
    assert code == 0
    assert err == ""
    assert out.splitlines() == ["01: FFID_A.jpg", "02: nested/FFID_B.jpg"]

    code, out, err = run_inspect(coco_path, capsys, "--labels", "--include-category", "tooth")
    assert code == 0
    assert err == ""
    assert out.splitlines() == ["01: tooth"]

    code, out, err = run_inspect(coco_path, capsys, "--include", "FFID_B", "--sum", "annotations")
    assert code == 0
    assert err == ""
    assert out == "2\n"

    code, out, err = run_inspect(coco_path, capsys, "--include-category", "tooth", "--sum", "images")
    assert code == 0
    assert err == ""
    assert out == "1\n"


def test_info_schema_and_segfmt_views(tmp_path, capsys):
    coco_path = write_sample_coco(tmp_path)

    code, out, err = run_inspect(coco_path, capsys, "--info")
    assert code == 0
    assert err == ""
    assert "meta.export.revision" in out
    assert "abc123" in out

    code, out, err = run_inspect(coco_path, capsys, "--schema", "extras")
    assert code == 0
    assert err == ""
    assert "info: DICT" in out
    assert " - meta: DICT" in out
    assert "   - export: DICT" in out
    assert "     - revision" in out
    assert "images: LIST[DICT] (2)" in out
    assert " - location (1)" in out
    assert " - note (1)" in out
    assert " - meta: DICT (1)" in out
    assert "annotations: LIST[DICT] (4)" in out
    assert " - meta: DICT (1)" in out
    assert "   - roi_id" in out
    assert "   - obj_id" in out

    code, out, err = run_inspect(coco_path, capsys, "--segfmt")
    assert code == 0
    assert err == ""
    assert out == '{"polygon": 1, "rle": 1, "rle-compressed": 1}\n'


def test_cli_reports_expected_user_errors(tmp_path, capsys):
    coco_path = write_sample_coco(tmp_path)

    code, out, err = run_inspect(coco_path, capsys, "--sum", "not-a-field")
    assert code == 2
    assert out == ""
    assert "Unknown --sum field 'not-a-field'" in err

    code, out, err = run_inspect(coco_path, capsys, "--sum", "--images")
    assert code == 2
    assert out == ""
    assert "--sum, --info, --schema, --segfmt" in err
