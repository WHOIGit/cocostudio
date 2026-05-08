# Sibert-2026 Label Studio Cycle FF

Utilities for cycling fullframe COCO annotations through Label Studio review.

This project has three working areas:

- `compare`: compare two COCO JSON files, starting with ROI counts, added/missing ROIs, label distributions, and annotated image renders that show differences for images common to both datasets.
- `push`: upload a COCO file to a configured Label Studio project.
- `pull`: pull reviewed annotations from a configured Label Studio project for downstream export and comparison.

`src/common` holds shared config, COCO, filesystem, and Label Studio helper code as the implementation grows.

## Status

This is an initial scaffold. The directory structure, Python packaging, local virtualenv, and local git repo are initialized first; concrete compare/push/pull commands will be filled in as the workflow settles.

## Layout

```text
labelstudio-cycle-ff/
├── configs/
│   ├── compare.example.toml
│   └── labelstudio.example.json
├── data/
├── src/
│   ├── common/
│   ├── compare/
│   ├── pull/
│   └── push/
├── pyproject.toml
└── README.md
```

`data/` is for local inputs and generated outputs. Its contents are ignored by git except for `.gitkeep`.

## Installation

From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The base package currently depends on:

- `pillow`
- `pycocotools`
- `tqdm`

Label Studio integration will use `labelstudio-tools`.

For the published GitHub install:

```bash
pip install "labelstudio-tools @ git+https://github.com/WHOIGit/labelstudio-tools.git"
```

For local development against the in-progress checkout:

```bash
export PYTHONPATH=/home/sbatchelder/Projects/labelstudio-tools/src:$PYTHONPATH
```

or install it editable into this virtualenv:

```bash
pip install -e /home/sbatchelder/Projects/labelstudio-tools
```

## Configuration

Copy the example config files before putting real paths, project IDs, or tokens into them.

- `configs/compare.example.toml` sketches the two COCO inputs and output locations for compare reports and annotated diff renders.
- `configs/labelstudio.example.json` matches the `labelstudio-tools` config style. Use environment-variable references such as `"$LABELSTUDIO_API_TOKEN"` for secrets.

Local config and secret files should use names like:

- `configs/compare.local.toml`
- `configs/labelstudio.local.json`
- `.env`

These are ignored by git.

## COCO Inspect

The `inspect` command reads one COCO JSON file and prints a tabulated image
summary with annotation totals, bbox/mask availability, and one count column per
category.

```bash
inspect data/example/coco.json
inspect data/example/coco.json -o data/inspect/example-coco.csv
```

### Column Options

Column options only affect the default image summary table and `--sum`.

- `--tablefmt FORMAT` selects the `tabulate` output format. Default: `github`.
- `--no-dims` removes `width` and `height`.
- `--no-meta` removes extra image metadata columns.
- `--no-anno` removes all annotation-derived columns.
- `--no-anno-types` removes only the `bbox` and `masks` columns.
- `--no-cats` removes per-category count columns.

```bash
inspect data/example/coco.json --tablefmt simple
inspect data/example/coco.json --no-dims
inspect data/example/coco.json --no-meta
inspect data/example/coco.json --no-anno
inspect data/example/coco.json --no-anno-types
inspect data/example/coco.json --no-cats
```

### Special Views

Special views do not render the default image summary table. They are mutually
exclusive.

```bash 
inspect data/example/coco.json --sum
inspect data/example/coco.json --sum images
inspect data/example/coco.json --sum annotations bbox masks tooth
inspect data/example/coco.json --info
inspect data/example/coco.json --schema
inspect data/example/coco.json --schema extras
inspect data/example/coco.json --segfmt
inspect data/example/coco.json --images
inspect data/example/coco.json --list
inspect data/example/coco.json --labels
inspect data/example/coco.json --categories
```

`--sum` prints a transposed field/value table for terminal output, with image
count shown as `images`. If `--sum` is written to a `.csv` file, it stays as a
single wide summary row. If one field is provided after `--sum`, only the raw
numeric value is printed. If multiple fields are provided, they are rendered as
a `field` / `value` table. Supported fields are `images`, `annotations`, `bbox`,
`masks`, category IDs and category names.

`--info` prints the top-level COCO `info` object as a key/value table, flattening
nested fields with dot notation.

`--schema` prints the observed COCO key structure. Keys in `images`,
`annotations`, `categories`, and `licenses` are counted across their parent
lists; if a key is not present in every record, their count is shown. 

`--schema extras` shows only keys outside the built-in COCO standard-key lists.

`--segfmt` reports the segmentation format used by selected annotations:
`polygon`, `rle`, or `rle-compressed`. If no selected annotations have masks, it
prints `None`. If multiple segmentation formats are present, it prints a JSON
object with counts for each format. Image and category filters are applied
before computing this value.

`--images` and `--list` are aliases. They print selected images as
`{image_id:02}: {filename}`, one per line, with no header.

`--labels` and `--categories` are aliases. They print selected categories as
`{category_id:02}: {label}`, one per line, with no header. Category filters can
be used to narrow the list.


### Filtering 
Image filters match COCO image ID, `file_name`, basename, or filename stem.
Category filters match category ID or category name.

```bash
inspect data/example/coco.json --include 1 FFID_B FFID_C  # where image_id 1 is FFID_A, for example
inspect data/example/coco.json --exclude FFID_A
inspect data/example/coco.json --include-category tooth
inspect data/example/coco.json --exclude-category junk 5  # where category_id 5 is "unlabeled", for example
```

## Planned Workflow

### Compare

Inputs:

- two COCO JSON files
- image root for rendering shared images

Initial outputs:

- summary of image, ROI, annotation, and category counts
- added/missing images and ROIs
- category distribution comparison
- annotated image renders for images common to both COCO files, highlighting annotation differences

Render all annotations from two COCO files onto each common fullframe:

```bash
python -m compare.cli render \
  -c configs/compare.example.toml \
  --output "data/compare/rendered/{FFID}.jpg"
```

```bash
python -m compare.cli render \
  --coco-a ../cleanup-automorph/data/exports/coco/automorph-coco-tiny.json \
  --coco-b ../cleanup-automorph/data/exports/coco/sam2-coco-tiny.json \
  --name-a automorph \
  --name-b sam2 \
  --image-root ../cleanup-automorph/data/exports/coco/images-quartersizejpg \
  --output 'data/automorph-sam2-coco-tiny-compare-render/{FFID}.jpg' \
  --text "{OBJ:03}.{CATEGORY}" \
  --provenance false \
  --clobber

```

Useful filters:

```bash
python -m compare.cli render -c configs/compare.example.toml --ffid-include FFID_A FFID_B
python -m compare.cli render -c configs/compare.example.toml --image-index 1 2 3
python -m compare.cli render -c configs/compare.example.toml --mask-mode fill --text "{CATEGORY} {OBJ:05}"
```

`render` maps image IDs between both COCO files at startup using `image_key`
(`file_name` by default), then renders image-by-image. `{FFID}` in `--output`
is derived from the A-side `ffid_field` (`file_name` by default).

`render-changed` and `render-diff` are reserved command stubs for the later
changed-only and explicit-difference renderers.

### Push

Inputs:

- COCO JSON file
- Label Studio project config

Planned behavior:

- create Label Studio tasks from COCO images and annotations
- avoid duplicate uploads using a configured primary key
- preserve enough COCO metadata to support later export and comparison

### Pull

Inputs:

- Label Studio project config
- reviewed task filter or view

Planned behavior:

- retrieve reviewed annotations
- save a reproducible raw export
- convert reviewed annotations back into the COCO comparison format

---

## Dev Notes

### Design precedent

This project is a focused refactor of
`labelstudio-tools/projects/ichthyolith_sahi`, minus the YOLO pieces. Cleanup
and label-cycle only. Before adding new command scaffolding, check whether
`labelstudio-tools` has an `initial-project` helper; if not, scope whether it
needs to be added there first.

### Push — sam2BigImg backend and latency

The push command must work with the `sam2BigImg` backend on
`ichthyolith.whoi.edu`. Before coding, define "acceptable" latency (target
page-load time? per-edit response time?). After upload, validate that masks
render correctly in the UI and are fully editable end-to-end before considering
the command done.

### Pull — diff report scope

The diff report should give high-level counts: new / edited / deleted ROIs. It
should **not** produce a per-object-ID listing of every edit — that level of
detail belongs in the annotated fullframe graphics, not the summary.

### Pull — object-ID continuity

When pulling reviewed annotations, preserve COCO annotation IDs across the
cycle using these rules:

1. ROI unchanged → keep original ID.
2. New annotation that pixel-overlaps an existing automorph ROI → reuse that
   ROI's ID. ("Overlap" = IoU above a threshold; threshold to be decided before
   implementation.)
3. Genuinely new annotation → assign next free ID.

The `roi_id` and `obj_id` extra fields in the cleanup-automorph COCO output are the
stable identifiers to carry through.

### Compare / render — one renderer, shared by compare and pull

`compare` and `pull` both need a diff renderer. Implement it once in `compare`
and have `pull` invoke it — do not build two independent diff renderers.

Full output spec for the renderer:

- Annotated fullframe with both datasets' bboxes and masks overlaid.
- Mask-difference view.
- Toggle object-ID labels on/off.
- Extract individual ROIs by ID.
- List new / edited / deleted ROIs; sort by most-different first.
- Per-Z1 mask color defaults: Z1.1 green, Z1.2 blue, Z1.3 purple; Z1.4 not
  shown by default.
- Per-source bbox color defaults: automorph cyan, sam2 red.
