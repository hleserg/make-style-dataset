#!/usr/bin/env bash
# CLOUD deploy: provision a fresh GPU box (>=24 GB, CUDA 12.x) and train the cmcstyle
# Flux style LoRA via ai-toolkit with the quality-preferred config. Local training is
# untouched — this is cloud-only. Idempotent-ish: re-running reuses the clone/venv/dataset.
#
# On a fresh RunPod / Vast / Lambda box:
#   export HF_TOKEN=hf_xxx                     # REQUIRED: gated FLUX.1-dev (+ private dataset)
#   export DATASET_HF_REPO=youruser/cmcstyle-ds   # OR pre-place images+txt in $DATASET_DIR
#   bash provision_and_train.sh
# Optional overrides: RES=512|768|"512,768,1024"  STEPS=2000  SAVE_EVERY=100  WORK=/workspace
#
# NOTE: not yet run on a real cloud box — validate the first run (esp. the torch/CUDA wheel
# match during pip install). The ai-toolkit repo ships its own docker/ if you prefer a container.
set -uo pipefail

WORK="${WORK:-/workspace}"
AITK_DIR="${AITK_DIR:-$WORK/ai-toolkit}"
DATASET_DIR="${DATASET_DIR:-$WORK/cmcstyle_ds}"
DATASET_HF_REPO="${DATASET_HF_REPO:-}"
OUT_DIR="${OUT_DIR:-$WORK/cmcstyle_out}"
RES="${RES:-768}"
STEPS="${STEPS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-100}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CFG_TEMPLATE="$HERE/cmcstyle_flux_cloud.yaml"

log(){ echo "[$(date '+%F %T')] $*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }

[ -n "${HF_TOKEN:-}" ] || die "set HF_TOKEN (gated FLUX.1-dev + private dataset access)"
[ -f "$CFG_TEMPLATE" ] || die "config template not found next to this script: $CFG_TEMPLATE"
command -v git >/dev/null || die "git not found"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — is this a GPU box?"
mkdir -p "$WORK" "$OUT_DIR"

# --- ai-toolkit (clone + venv + deps) ---
if [ ! -d "$AITK_DIR/.git" ]; then
  log "cloning ai-toolkit -> $AITK_DIR"
  git clone --depth 1 https://github.com/ostris/ai-toolkit "$AITK_DIR" || die "git clone failed"
fi
if [ ! -x "$AITK_DIR/venv/bin/python" ]; then
  log "creating venv + installing requirements (torch must match the box CUDA; CUDA 12.x cloud images work)"
  python3 -m venv "$AITK_DIR/venv"
  "$AITK_DIR/venv/bin/pip" install -U pip wheel || die "pip upgrade failed"
  "$AITK_DIR/venv/bin/pip" install -r "$AITK_DIR/requirements.txt" || die "pip install failed (check torch/CUDA wheel match)"
fi
PY="$AITK_DIR/venv/bin/python"

# --- HF auth (FLUX.1-dev auto-downloads on first run; ~25 GB) ---
export HF_TOKEN
mkdir -p "$HOME/.cache/huggingface"
printf '%s' "$HF_TOKEN" > "$HOME/.cache/huggingface/token"
unset HF_HUB_OFFLINE   # ai-toolkit must run ONLINE (else model_info raises)

# --- dataset (pull from a private HF dataset repo, or use a pre-placed folder) ---
if [ -n "$DATASET_HF_REPO" ] && [ ! -e "$DATASET_DIR/.ready" ]; then
  log "downloading dataset $DATASET_HF_REPO -> $DATASET_DIR"
  "$PY" - "$DATASET_HF_REPO" "$DATASET_DIR" <<'PY' || die "dataset download failed"
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], repo_type="dataset", local_dir=sys.argv[2])
PY
  touch "$DATASET_DIR/.ready"
fi
FIRST_PNG="$(find "$DATASET_DIR" -maxdepth 3 -name '*.png' 2>/dev/null | head -1)"
[ -n "$FIRST_PNG" ] || die "no .png found under $DATASET_DIR (set DATASET_DIR or DATASET_HF_REPO)"
IMG_DIR="$(dirname "$FIRST_PNG")"           # the kohya subfolder (e.g. 10_cmcstyle) or the dir itself
N=$(find "$IMG_DIR" -maxdepth 1 -name '*.png' | wc -l)
log "dataset: $N images in $IMG_DIR"

# --- render config from template ---
CFG="$OUT_DIR/config.yaml"
RES_FMT="$(printf '%s' "$RES" | sed 's/, */, /g')"
sed -e "s#__DATASET_DIR__#$IMG_DIR#g" \
    -e "s#__OUTPUT_DIR__#$OUT_DIR#g" \
    -e "s#__RES__#$RES_FMT#g" \
    -e "s#__STEPS__#$STEPS#g" \
    -e "s#__SAVE_EVERY__#$SAVE_EVERY#g" \
    "$CFG_TEMPLATE" > "$CFG"
log "config -> $CFG (optimizer=adamw8bit, low_vram OFF, res=[$RES_FMT], save_every=$SAVE_EVERY)"

# --- diagnostics + run ---
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 10 \
  > "$OUT_DIR/gpu.csv" 2>&1 &
SAMP=$!; trap 'kill $SAMP 2>/dev/null' EXIT
cd "$AITK_DIR"
log "training... (logs -> $OUT_DIR/train.log)"
"$PY" run.py "$CFG" 2>&1 | tee "$OUT_DIR/train.log"
RC=${PIPESTATUS[0]}
kill $SAMP 2>/dev/null

SFT="$(find "$OUT_DIR" -name '*.safetensors' -printf '%s %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
if [ "$RC" -eq 0 ] && [ -n "$SFT" ] && [ -s "$SFT" ]; then
  log "DONE — LoRA(s) in $OUT_DIR ; newest: $SFT"
  log "Download the .safetensors back, then run the eval grid (scripts/eval_style_lora.py)."
else
  die "training failed or produced no .safetensors (rc=$RC) — see $OUT_DIR/train.log"
fi
