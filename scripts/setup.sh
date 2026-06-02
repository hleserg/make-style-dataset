#!/usr/bin/env bash
# One-command setup for make-style-dataset.
#
# Installs everything (including the multi-GB GPU/ML stack), scaffolds the
# workspace, and runs an environment check. Re-runnable: safe to run again.
#
#   bash scripts/setup.sh           # full install (GPU stack included)
#   bash scripts/setup.sh --no-gpu  # skip the heavy GPU/ML group (CPU stages only)
#
# After it finishes, drop comic pages into workspace/00_pages/ and run
#   uv run make-style-dataset run-all   # build the whole dataset

set -euo pipefail

GPU=1
for arg in "$@"; do
  case "$arg" in
    --no-gpu) GPU=0 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

# Move to the repo root (this script lives in scripts/).
cd "$(dirname "$0")/.."

echo "==> 1/4  Checking for uv (the installer/runner)"
if ! command -v uv >/dev/null 2>&1; then
  echo "    uv not found — installing it from https://astral.sh/uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer drops uv in ~/.local/bin; make it visible for this run.
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "    uv $(uv --version | awk '{print $2}')"

echo "==> 2/4  Installing dependencies (this can take a while the first time)"
if [ "$GPU" -eq 1 ]; then
  echo "    including the GPU/ML stack (several GB) and the web UI ..."
  uv sync --all-extras --group gpu --group ui
else
  echo "    CPU-only (skipping the GPU/ML group), with the web UI ..."
  uv sync --all-extras --group ui
fi

echo "==> 3/4  Scaffolding the workspace"
uv run make-style-dataset init

echo "==> 4/4  Checking this machine is ready"
uv run make-style-dataset doctor || true

cat <<'DONE'

----------------------------------------------------------------------
Setup finished. The easiest way to build a dataset is the app:

  uv run make-style-dataset ui      (or: make ui)

It opens a 3-step wizard in your browser: name your style, drag pages
in, press Build, download the .zip.

Prefer the terminal? Drop pages into workspace/00_pages/ and run
  uv run make-style-dataset run-all

The first real run downloads the AI models (a few GB) and takes a few
minutes — that is normal, it has not frozen.

Full step-by-step guide: docs/USER_GUIDE.md
----------------------------------------------------------------------
DONE
