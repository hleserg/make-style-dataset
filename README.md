# make-style-dataset

> Comic pages → kohya-ready style LoRA dataset pipeline

[![CI](https://github.com/hleserg/make-style-dataset/actions/workflows/ci.yml/badge.svg)](https://github.com/hleserg/make-style-dataset/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)

[Русская версия](README-ru.md)

Turn a pile of comic pages into a [kohya_ss](https://github.com/bmaltais/kohya_ss)-ready
dataset for training a **style** LoRA. The tool runs a linear, file-driven
pipeline: detect and slice panels, mask and inpaint speech bubbles, deduplicate
and size-filter, then caption and lay out the final dataset folder.

> **Not a developer?** Start with the plain-language
> [**step-by-step User Guide**](docs/USER_GUIDE.md) ([RU](docs/USER_GUIDE-ru.md)):
> one setup command (`bash scripts/setup.sh`), then a point-and-click app
> (`make ui`) — drop pages in, press Build, download the dataset.

---

## Pipeline

Each stage reads one workspace folder and writes the next. `00_pages` is the
input drop; stages 1–5 are the dataset pipeline, and an optional stage 6
(`train`) trains the LoRA. All are CLI subcommands:

| # | Stage | Reads → writes | Does |
|---|-------|----------------|------|
| 0 | *(pages)* | → `00_pages/` | Raw comic pages you drop in. |
| 1 | `panels` | `00_pages` → `01_panels` | Detect comic panels and slice each page. |
| 2 | `bubbles` | `01_panels` → `02_masks` | Detect speech bubbles, write removal masks. |
| 3 | `inpaint` | `01_panels`+`02_masks` → `03_inpainted` | Paint the bubbles away. |
| 4 | `clean` | `03_inpainted` → `04_clean` | Drop near-duplicates and too-small panels. |
| 5 | `caption` | `04_clean` → `05_dataset/<N>_<trigger>/` | Caption and lay out the kohya dataset. |
| 6 | `train` *(opt-in)* | `05_dataset/…` → `06_lora/` | Train a style LoRA via kohya sd-scripts (SD 1.5 / SDXL / Flux). |

The runner is **idempotent**: each stage drops a `.stage_complete` marker and is
skipped on re-runs unless `--force`. See the
[workspace layout contract](docs/architecture/WORKSPACE.md) and the
[system overview](docs/architecture/SYSTEM.md).

All stages are implemented. `panels` and `clean` are CPU-only; the model
stages — `bubbles` (YOLOv8-seg + EasyOCR), `inpaint` (ONNX Big-LaMa) and
`caption` (WD14 ViT v3, ONNX) — need the optional **`gpu`** dependency group
(see [GPU stages](#gpu-stages)). `run-all` runs stages 1–5 and prints a summary;
the opt-in `train` stage (`APP_RUN_TRAIN=false` by default) shells out to a
separate [kohya sd-scripts](https://github.com/kohya-ss/sd-scripts) venv — see
[Training a LoRA](#training-a-lora).

## Quickstart

**The easy way** (installs everything, scaffolds the workspace, checks the GPU):

```bash
bash scripts/setup.sh                # one-command setup; --no-gpu to skip the GPU stack
uv run make-style-dataset ui         # the app: drop pages, Build, download .zip  (or: make ui)
#   prefer the terminal? drop pages into workspace/00_pages/ and run:
uv run make-style-dataset run-all    # build the dataset
uv run make-style-dataset train      # optional: train the LoRA -> workspace/06_lora/
```

**The manual way** (developers):

```bash
uv sync --all-extras                 # create .venv + dev tools (CPU stages work now)
uv run make-style-dataset init       # scaffold the workspace + seed .env
uv run make-style-dataset doctor     # check Python / GPU / workspace are ready
make check                           # the Definition-of-Done gate

# To run the model stages (bubbles/inpaint/caption), add the GPU deps:
uv sync --all-extras --group gpu     # torch (cu128) + onnxruntime + ultralytics/easyocr (multi-GB)
uv run make-style-dataset run-all    # run the whole pipeline
```

## Usage

```bash
# Run one stage, or the whole pipeline:
uv run make-style-dataset panels
uv run make-style-dataset run-all
uv run make-style-dataset run-all --help     # lists every stage

# Useful flags:
uv run make-style-dataset run-all --workspace /data/comics   # override workspace root
uv run make-style-dataset clean --force                      # rerun a completed stage
```

Configuration is environment-driven (`APP_` prefix, see `.env.example`):
workspace root, trigger token, kohya repeat count, thresholds (`min_panel_area`,
`dedup_hamming_distance`, `min_side_px`, `target_side`), backend selectors
(`inpaint_backend`, `caption_backend`), and per-stage enable flags (`APP_RUN_*`)
that gate `run-all`.

## GPU stages

The model stages download their weights from Hugging Face (pinned by commit) on
first use and run on the GPU when available:

| Stage | Model | Backend |
|-------|-------|---------|
| `bubbles` | `kitsumed/yolov8m_seg-speech-bubble` + EasyOCR | ultralytics (torch, cu128) |
| `inpaint` | `Carve/LaMa-ONNX` (Big-LaMa) | onnxruntime |
| `caption` | `SmilingWolf/wd-vit-tagger-v3` | onnxruntime |

Install them with `uv sync --all-extras --group gpu`. They live in a PEP 735
**dependency-group** (not an extra), so the default `uv sync --all-extras` —
used by CI and the CPU-only stages — stays lightweight. The CUDA build of torch
(`cu128`, for Blackwell/RTX 50xx) is pinned via `[tool.uv.sources]`; on macOS the
lockfile falls back to the CPU build. onnxruntime falls back to CPU if its CUDA
libs aren't found (the bare pip wheels don't ship cuDNN/`libcublasLt` — a CUDA
base image or a local CUDA/cuDNN install provides them).

**Hardware split:** run `panels`/`clean` (CPU) anywhere; run
`bubbles`/`inpaint`/`caption` on the GPU host. Containerizing the heavy stages
for the GPU machine is a planned follow-up.

## Training a LoRA

The opt-in `train` stage trains a style LoRA from the finished dataset by
shelling out to a local [kohya sd-scripts](https://github.com/kohya-ss/sd-scripts)
clone running in **its own venv** — never installed into this project's, so the
trainer's `torch`/`numpy`/`Pillow` pins can't clash with ours. It supports three
families (`APP_TRAIN_MODEL_TYPE`): **`sd15`**, **`sdxl`** (fits 16 GB cleanly),
and **`flux`** (low-VRAM `--blocks_to_swap` + `--fp8_base`).

Point `APP_TRAIN_SD_SCRIPTS_DIR` / `APP_TRAIN_BASE_MODEL` (and the `APP_TRAIN_*`
knobs in `.env.example`) at your setup, verify with `make-style-dataset doctor`
(it checks the clone, the base model, and that the trainer's torch advertises
your GPU arch), then run `make-style-dataset train` or the app's **Train** step.
Output lands in `06_lora/<trigger>.safetensors`. Full walkthrough:
[User Guide → Training a LoRA](docs/USER_GUIDE.md#training-a-lora-optional).

## Project layout

| Path | Purpose |
|------|---------|
| `src/make_style_dataset/cli.py` | thin argparse shell (subcommand per stage) |
| `src/make_style_dataset/pipeline.py` | ordered stage registry + idempotent runner |
| `src/make_style_dataset/stages/` | one module per stage (`panels`, `bubbles`, `inpaint`, `clean`, `caption`, `train`) |
| `src/make_style_dataset/workspace.py` | typed, single-root directory contract |
| `src/make_style_dataset/config.py` | typed settings via `pydantic-settings` |
| `src/make_style_dataset/observability/` | Sentry init (`send_default_pii=False`) + per-stage tags |
| `tests/` | `unit/` + `integration/`, pytest with ≥90% coverage gate |
| `docs/` | architecture (SYSTEM, WORKSPACE, ADRs), dev standard, PLAYBOOK spec |

## Make targets

```
make setup       # one-command first-time setup (deps + workspace + env check)
make install     # uv sync --all-extras + pre-commit install
make init        # scaffold the workspace folders + seed .env
make doctor      # check this machine is ready (Python, GPU, workspace)
make ui          # launch the local web UI (installs the 'ui' group on first run)
make check       # lint + fmt-check + type + security + tests  (DoD gate)
make run-all     # run the full pipeline
make panels      # run a single stage (also: bubbles/inpaint/caption)
make clean-stage # run the dedup/size-filter stage (`make clean` clears caches)
make test        # full test suite with coverage
make test-fast   # unit tests only, parallel, no coverage
```

## Tooling

- **uv** — environment & dependency management (lockfile committed)
- **ruff** — lint + format
- **pyright** — static type checking (standard mode)
- **pytest** — tests, ≥90% coverage
- **bandit / pip-audit** — security
- **pre-commit** — local gate
- **commitizen** — conventional commits → version bump + changelog
- **Sentry** — error/perf monitoring, privacy-first

## License

MIT — see [LICENSE](LICENSE).
