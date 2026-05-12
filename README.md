# cocostudio

Push COCO predictions to Label Studio for human correction, then pull the
refined annotations back into a COCO file for comparison with the original
using `coconut-crab`'s `coco-render-combo`.

## Layout

```text
cocostudio/
├── configs/
│   ├── compare.automorph-sam2.toml
│   ├── ls_auth.example.toml
│   └── ls_project.clean-sam2-automorph.toml
├── data/
│   ├── from_cleanup -> ../cleanup-automorph/data/exports/coco
│   └── render/
├── src/
│   └── cocostudio/
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
`coco-inspect` and `coco-render-combo` commands.

Inspect a local cleanup export:

```bash
coco-inspect data/from_cleanup/automorph-coco-tiny.json --sum
```

Render the Automorph/SAM2 comparison configured for this project:

```bash
coco-render-combo -c configs/compare.automorph-sam2.toml
```

The comparison config expects `data/from_cleanup` to point at the
`cleanup-automorph` COCO export directory.

## Configuration

Copy example config files before putting real paths, project IDs, or tokens into
them.

- `configs/ls_auth.example.toml` shows the local auth/credential shape.
- `configs/ls_project.clean-sam2-automorph.toml` configures the Label Studio
  project.
- `configs/compare.automorph-sam2.toml` configures the `coco-render-combo`
  Automorph-vs-SAM2 output.

Local config and secret files should use names like:

- `configs/ls_auth.toml`
- `configs/ls_auth.ichthyolith.toml`
- `.env`

These are ignored by git.

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

### Pull - Object-ID Continuity

When pulling reviewed annotations, preserve COCO annotation IDs across the cycle
using these rules:

1. ROI unchanged: keep original ID.
2. New annotation that pixel-overlaps an existing automorph ROI: reuse that ROI's ID. "Overlap" means IoU above a threshold; threshold to be decided before implementation.
3. Genuinely new annotation: assign next free ID.

The `roi_id` and `obj_id` extra fields in the cleanup-automorph COCO output are
the stable identifiers to carry through.
