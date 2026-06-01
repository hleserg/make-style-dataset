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

---

## Pipeline

Six stages, each reading one workspace folder and writing the next. `00_pages`
is the input drop; the other five are CLI subcommands:

| # | Stage | Reads → writes | Does |
|---|-------|----------------|------|
| 0 | *(pages)* | → `00_pages/` | Raw comic pages you drop in. |
| 1 | `panels` | `00_pages` → `01_panels` | Detect comic panels and slice each page. |
| 2 | `bubbles` | `01_panels` → `02_masks` | Detect speech bubbles, write removal masks. |
| 3 | `inpaint` | `01_panels`+`02_masks` → `03_inpainted` | Paint the bubbles away. |
| 4 | `clean` | `03_inpainted` → `04_clean` | Drop near-duplicates and too-small panels. |
| 5 | `caption` | `04_clean` → `05_dataset/<N>_<trigger>/` | Caption and lay out the kohya dataset. |

The runner is **idempotent**: each stage drops a `.stage_complete` marker and is
skipped on re-runs unless `--force`. See the
[workspace layout contract](docs/architecture/WORKSPACE.md) and the
[system overview](docs/architecture/SYSTEM.md).

All six stages are implemented. `panels` and `clean` are CPU-only; the model
stages — `bubbles` (YOLOv8-seg + EasyOCR), `inpaint` (ONNX Big-LaMa) and
`caption` (WD14 ViT v3, ONNX) — need the optional **`gpu`** dependency group
(see [GPU stages](#gpu-stages)). `run-all` prints a summary of how many
artifacts each stage produced.

## Quickstart

```bash
uv sync --all-extras                 # create .venv + dev tools (CPU stages work now)
cp .env.example .env                 # tune workspace/trigger/thresholds (optional)
uv run make-style-dataset --version
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

## Project layout

| Path | Purpose |
|------|---------|
| `src/make_style_dataset/cli.py` | thin argparse shell (subcommand per stage) |
| `src/make_style_dataset/pipeline.py` | ordered stage registry + idempotent runner |
| `src/make_style_dataset/stages/` | one module per stage (`panels`, `bubbles`, `inpaint`, `clean`, `caption`) |
| `src/make_style_dataset/workspace.py` | typed, single-root directory contract |
| `src/make_style_dataset/config.py` | typed settings via `pydantic-settings` |
| `src/make_style_dataset/observability/` | Sentry init (`send_default_pii=False`) + per-stage tags |
| `tests/` | `unit/` + `integration/`, pytest with ≥90% coverage gate |
| `docs/` | architecture (SYSTEM, WORKSPACE, ADRs), dev standard, PLAYBOOK spec |

## Make targets

```
make install     # uv sync --all-extras + pre-commit install
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
