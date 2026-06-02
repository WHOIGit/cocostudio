"""Tests for the pushpull config loader and CLI config/flag precedence."""

import pytest

from cocostudio.config import load_push_config
from cocostudio import cli


def _write(tmp_path, text):
    p = tmp_path / "c.toml"
    p.write_text(text)
    return str(p)


# ---------------------------------------------------------------------------
# load_push_config
# ---------------------------------------------------------------------------

def test_load_flattens_shared_and_push(tmp_path):
    p = _write(tmp_path, """
ls_config = "x"
region_id_pattern = "r"
[push]
coco = "c"
map_seg = "brush"
""")
    assert load_push_config(p) == {
        "ls_config": "x", "region_id_pattern": "r",
        "coco": "c", "map_seg": "brush",
    }


def test_pull_section_ignored_for_push(tmp_path):
    p = _write(tmp_path, """
ls_config = "x"
[push]
coco = "c"
[pull]
view = 7
""")
    assert load_push_config(p) == {"ls_config": "x", "coco": "c"}


def test_unknown_top_level_key_raises(tmp_path):
    p = _write(tmp_path, 'bogus = 1\n[push]\ncoco = "c"\n')
    with pytest.raises(ValueError, match="unknown top-level"):
        load_push_config(p)


def test_unknown_push_key_raises(tmp_path):
    p = _write(tmp_path, '[push]\nbogus = 1\n')
    with pytest.raises(ValueError, match=r"unknown \[push\]"):
        load_push_config(p)


def test_shared_key_misplaced_in_push_raises(tmp_path):
    # ls_config belongs at the top level, not under [push].
    p = _write(tmp_path, '[push]\nls_config = "x"\n')
    with pytest.raises(ValueError, match=r"unknown \[push\]"):
        load_push_config(p)


def test_push_not_a_table_raises(tmp_path):
    p = _write(tmp_path, 'push = "nope"\n')
    with pytest.raises(ValueError, match="must be a table"):
        load_push_config(p)


# ---------------------------------------------------------------------------
# CLI precedence: defaults < config < explicit flags
# ---------------------------------------------------------------------------

def _settings(tmp_path, config_text, argv_extra):
    cfg = _write(tmp_path, config_text)
    args = cli._build_parser().parse_args(["push", "-c", cfg, *argv_extra])
    return cli._merge_settings(args)


def test_config_value_used_when_no_flag(tmp_path):
    s = _settings(tmp_path, 'ls_config="x"\n[push]\nmap_seg="brush"\n', [])
    assert s["map_seg"] == "brush"


def test_cli_flag_overrides_config(tmp_path):
    s = _settings(tmp_path, 'ls_config="x"\n[push]\nmap_seg="brush"\n',
                  ["--map-seg", "polygon"])
    assert s["map_seg"] == "polygon"


def test_default_used_when_absent(tmp_path):
    s = _settings(tmp_path, 'ls_config="x"\n[push]\nmap_seg="brush"\n', [])
    assert s["region_id_pattern"] == "{annotation.id}"  # hardcoded default


def test_include_replaced_not_merged(tmp_path):
    s = _settings(tmp_path,
                  'ls_config="x"\n[push]\nmap_seg="brush"\ninclude=["a","b"]\n',
                  ["--include", "c"])
    assert s["include"] == ["c"]


def test_dry_run_config_true_then_no_dry_run(tmp_path):
    cfg = 'ls_config="x"\n[push]\nmap_seg="brush"\ndry_run=true\n'
    assert _settings(tmp_path, cfg, [])["dry_run"] is True
    assert _settings(tmp_path, cfg, ["--no-dry-run"])["dry_run"] is False
    assert _settings(tmp_path, cfg, ["--dry-run"])["dry_run"] is True


# ---------------------------------------------------------------------------
# _validate
# ---------------------------------------------------------------------------

def test_validate_requires_a_map(tmp_path):
    coco = tmp_path / "c.json"; coco.write_text("{}")
    img = tmp_path / "imgs"; img.mkdir()
    ls = tmp_path / "ls.toml"; ls.write_text("")
    settings = {
        "ls_config": str(ls), "coco": str(coco), "image_dir": str(img),
        "map_bbox": None, "map_seg": None,
    }
    with pytest.raises(ValueError, match="at least one of map_bbox"):
        cli._validate(settings)


def test_validate_missing_required(tmp_path):
    with pytest.raises(ValueError, match="missing required"):
        cli._validate({"ls_config": None, "coco": None, "image_dir": None,
                       "map_seg": "brush"})


# ---------------------------------------------------------------------------
# resolve_simplify
# ---------------------------------------------------------------------------

def test_resolve_simplify_maxpoints_zero_disables():
    assert cli.resolve_simplify(0.003, 0) == (None, None)


def test_resolve_simplify_maxpoints_without_epsilon():
    assert cli.resolve_simplify(None, 50) == (1.0, 50)


def test_resolve_simplify_passthrough():
    assert cli.resolve_simplify(0.003, 50) == (0.003, 50)
