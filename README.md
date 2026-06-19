# cocostudio

Push COCO predictions to Label Studio for human correction, then pull the
refined annotations back into a COCO file for comparison with the original
using `coconut-crab`'s `coco-render-combo`.

## Layout

```text
cocostudio/
├── configs/
│   ├── labelstudio/         # LS project + auth configs (per label type)
│   ├── pushpull/            # cocostudio push/pull configs
│   │   ├── pushpull.example.toml
│   │   ├── pull_postprocess_ichthyoliths.py
│   │   ├── sam2dirty_brush.toml
│   │   ├── sam2dirty_polygon.toml
│   │   └── sam2dirty_rect.toml
│   └── render/              # coco-render and coco-render-combo configs
├── data/
│   ├── from_cleanup -> ../cleanup-automorph/data/exports/coco
│   └── render/
├── src/
│   └── cocostudio/
│       ├── cli.py           # `cocostudio` console script (push/pull)
│       ├── config.py        # pushpull config loader
│       ├── push.py
│       └── pull.py
├── pyproject.toml
├── reinstall_localdev.sh
└── README.md
```

`data/` is for local inputs, symlinks, and generated outputs. Its contents are
ignored by git.

## Installation

Create and activate a virtualenv, then run the local dev reinstall script:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
./reinstall_localdev.sh
```

This installs `coconut-crab`, `labelstudio-tools`, and this package as
editable from their sibling directories. Re-run it any time you need to
resync after changes to those packages' metadata.

## COCO Tools

`coconut-crab` is installed by `reinstall_localdev.sh` and provides the
`coco-inspect`, `coco-render`, and `coco-render-combo` commands.

Inspect a local cleanup export:

```bash
coco-inspect data/from_cleanup/automorph-coco-tiny.json --sum
```

Render the Automorph/SAM2 comparison configured for this project:

```bash
coco-render-combo -c configs/render/rendercombo.automorph-sam2dirty.toml
```

Render the pushed-vs-pulled sam2dirty rectangle review:

```bash
coco-render-combo -c configs/render/rendercombo.sam2dirty_push-pull.toml
```

The render configs expect `data/from_cleanup` to point at the
`cleanup-automorph` COCO export directory and `data/pulled/` to contain pull
outputs.

## Configuration

Copy example config files before putting real paths, project IDs, or tokens into
them.

- `configs/labelstudio/ls_auth.example.toml` shows the local auth/credential
  shape.
- `configs/labelstudio/*/ls_project.*.toml` configure Label Studio projects.
- `configs/pushpull/pushpull.example.toml` documents the push/pull schema.
- `configs/render/rendercombo.*.toml` configure `coco-render-combo` outputs.

Local config and secret files should use names like:

- `configs/ls_auth.toml`
- `configs/ls_auth.ichthyolith.toml`
- `.env`

These are ignored by git.

### Push

`cocostudio push` uploads COCO predictions to a Label Studio project, staging
images to S3 first if needed. It reads a **pushpull config** (`configs/pushpull/`)
that bundles the inputs and the annotation→tag mapping:

```bash
cocostudio push -c configs/pushpull/sam2dirty_brush.toml --dry-run
cocostudio push -c configs/pushpull/sam2dirty_polygon.toml --include 101_U1553C_22R_1W_28-30cm_N2of2_Z200x
```

A pushpull config has shared top-level keys (`ls_config`, `s3_prefix`,
`region_id_pattern`), plus `[push]` and `[pull]` sections. Every key is the long
form of a CLI flag, and any flag passed on the command line overrides the config
value. See `configs/pushpull/pushpull.example.toml` for the annotated schema.

**Annotation → tag mapping.** At least one of `map_bbox` / `map_seg` is required:

- `map_bbox` — a rectangle-family control to receive COCO bounding boxes.
- `map_seg` — a polygon- or brush-family control to receive segmentations; the
  control's type decides whether masks are pushed as polygons or brush strokes.

Each value is either a **control name** (`from_name`) or a **tag type**
(`Rectangle`, `RectangleLabels`, `Polygon`, `PolygonLabels`, `Brush`,
`BrushLabels`). A bare type resolves only if the project has exactly one such
control; otherwise name it explicitly. The bare, label-less variants
(`<Rectangle>`/`<Polygon>`/`<Brush>`) push geometry with no category label.

**Polygon preprocessing** (`polygon_preprocess_epsilon` /
`polygon_preprocess_maxpoints`) Douglas-Peucker–simplifies polygon segmentations
for LS editor performance; it errors if `map_seg` targets a brush control.

Behavior:

- create Label Studio tasks from COCO images and annotations (as predictions)
- avoid duplicate uploads using the project's configured primary key
- preserve enough COCO metadata to support later export and comparison

### Pull

`cocostudio pull` downloads submitted Label Studio annotations and writes a COCO
file for comparison.

```bash
cocostudio pull -c configs/pushpull/sam2dirty_rect.toml --dry-run
cocostudio pull -c configs/pushpull/sam2dirty_rect.toml
```

The pull section can inherit `[push].coco` as `push_coco` and `[push].image_dir`
as `image_dir`, so the output uses the same categories, image IDs, and local
image root as the pushed COCO when available.

Current pull scope:

- bbox-only conversion from a mapped RectangleLabels control (`map_bbox`)
- optional task selection by Label Studio `view`
- `completed_only` filtering for submitted annotations
- multiple-annotation policy: `error`, `skip`, `latest`, or `earliest`
- label validation against the pushed COCO categories (`match_push_coco`)
- optional image download from `data.<image_datafield>` S3 URLs when local
  images are missing
- raw Label Studio task caching with `ls_cache = "never" | "once" | "clobber"`

The raw Label Studio cache is internal, not a first-class product. Cache files
live under:

```text
cache/<hostname>/<project-name>/<cache-id>.json
```

Segmentation pull is intentionally not implemented yet. `map_seg` is accepted
for config symmetry but raises `NotImplementedError`.

#### Pull Postprocessing

Generic pull preserves Label Studio metadata such as `ls_region_id` and
`ls_annotation_id`, plus any `result.meta` fields supplied by Label Studio. It
does not derive project-specific fields like `roi_id` or `obj_id` by default.

Project-specific enrichment belongs in a postprocess hook:

```toml
[pull]
postprocess_function = "configs/pushpull/pull_postprocess_ichthyoliths.py:compute_coco_localization_meta"
```

The function receives the complete COCO dict and may mutate it in place or
return a replacement dict. `pull_postprocess_ichthyoliths.py` derives local
`roi_id` and `obj_id` metadata from pushed Label Studio region IDs like
`roi_..._obj00008_plane000`.

#### Pull Review Render

After pulling, compare pushed and pulled annotations with:

```bash
coco-render-combo -c configs/render/rendercombo.sam2dirty_push-pull.toml
```

That combo config uses `render.sam2dirty.toml` as the formatting base for both
sides, disables masks for a fast bbox comparison, and uses `[style-a]` /
`[style-b]` to force side colors.

## Dev Notes

### Design Precedent

This project is a focused refactor of
`labelstudio-tools/projects/ichthyolith_sahi`, minus the YOLO pieces. Cleanup
and label-cycle only. Before adding new command scaffolding, check whether
`labelstudio-tools` has an `initial-project` helper; if not, scope whether it
needs to be added there first.

### Push - sam2BigImg Backend And Latency

The push command must work with the `sam2BigImg` backend on
`ichthyolith.whoi.edu`. Before coding, define "acceptable" latency: page-load
time or per-edit response time. After upload, validate that masks render
correctly in the UI and are fully editable end-to-end before considering the
command done.

### Pull - Diff Report Scope

The diff report should give high-level counts: new / edited / deleted ROIs. It
should not produce a per-object-ID listing of every edit; that level of detail
belongs in the annotated fullframe graphics, not the summary.

### Pull - Future Object-ID Continuity

Active COCO annotation ID matching is not implemented yet. The current pull
preserves source image IDs when possible and assigns fresh COCO annotation IDs.
Object/ROI-specific metadata can be added with `postprocess_function`, but
generic pull does not enforce object continuity.
