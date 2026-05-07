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

