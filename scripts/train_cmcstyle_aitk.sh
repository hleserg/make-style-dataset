#!/usr/bin/env bash
# One-shot cmcstyle Flux style-LoRA launcher (HLE-803).
# Self-contained for an unattended systemd-run launch: checks preconditions
# (dataset in spec + stable, GPU free, models/token present, CUDA usable under
# THIS env), then trains via ai-toolkit with full diagnostics captured.
#   ./launch.sh rehearse   # steps=2 mechanism test + fit-check (run while you watch)
#   ./launch.sh night      # the real run (steps=2000) — scheduled for ~02:00
# Exits 0 on a clean abort (preconditions unmet) so a timer isn't flagged failed;
# the reason is logged loudly in launcher.log + meta.json either way.
set -uo pipefail

MODE="${1:-night}"

# --- self-contained environment (systemd services get a minimal env) ---
export HOME=/root
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"   # WSL libcuda.so + nvidia-smi
unset HF_HUB_OFFLINE                                             # ai-toolkit must run online

AITK=/home/serg/ai-toolkit
PYTHON="$AITK/venv/bin/python"
DATASET=/home/serg/make-style-dataset/workspace/05_dataset/10_cmcstyle
TRIGGER=cmcstyle
BASE=/home/serg/cmcstyle_night
RUNS="$BASE/runs"
HF_CACHE="$HOME/.cache/huggingface"

if [ "$MODE" = "rehearse" ]; then
  STEPS=2;    SAVE_EVERY=2;   TIMEOUT=2400;  STABILITY_MIN=0
else
  STEPS=2000; SAVE_EVERY=250; TIMEOUT=28800; STABILITY_MIN=20
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN="$RUNS/${TS}_${MODE}"
mkdir -p "$RUN/out"
LOG="$RUN/launcher.log"
exec > >(tee -a "$LOG") 2>&1

SAMPLER_PID=""
SAFETENSORS=""; SFT_SIZE=0; SFT_TENSORS=0
START_EPOCH=0; END_EPOCH=0; RC=0; STATUS="started"; REASON=""

log() { echo "[$(date '+%F %T')] $*"; }

write_meta() {
  cat > "$RUN/meta.json" <<JSON
{
  "mode": "$MODE",
  "status": "$STATUS",
  "reason": "$REASON",
  "run_dir": "$RUN",
  "dataset": "$DATASET",
  "images": ${NPNG:-0},
  "steps_requested": $STEPS,
  "gpu_used_mib_at_launch": ${USED:-null},
  "start_epoch": $START_EPOCH,
  "end_epoch": $END_EPOCH,
  "duration_s": $(( END_EPOCH > START_EPOCH ? END_EPOCH - START_EPOCH : 0 )),
  "train_rc": $RC,
  "safetensors": "${SAFETENSORS}",
  "safetensors_bytes": ${SFT_SIZE},
  "safetensors_tensors": ${SFT_TENSORS}
}
JSON
  ln -sfn "$RUN" "$BASE/latest"
  log "meta written; latest -> $RUN ; status=$STATUS ${REASON:+($REASON)}"
}

finalize() { STATUS="$1"; REASON="${2:-}"; [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null; write_meta; }
abort()    { log "ABORT: $1"; finalize "aborted" "$1"; exit 0; }
trap '[ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null' EXIT

log "=== cmcstyle launch  mode=$MODE  steps=$STEPS  run=$RUN ==="

# --- environment snapshot (diagnostics) ---
{
  echo "## date"; date
  echo; echo "## nvidia-smi"; nvidia-smi
  echo; echo "## torch"; "$PYTHON" -c "import torch;print(torch.__version__, 'cuda_avail', torch.cuda.is_available(), torch.cuda.get_arch_list()[-3:])"
  echo; echo "## ai-toolkit"; cat "$AITK/version.py" 2>/dev/null
  echo; echo "## git SHAs"; git -C /home/serg/make-style-dataset rev-parse HEAD 2>/dev/null
  echo; echo "## disk"; df -h "$BASE"
} > "$RUN/env.txt" 2>&1 || true

# --- preconditions ---
log "checking CUDA under this (systemd) environment..."
"$PYTHON" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || abort "torch.cuda.is_available() is False under this environment (WSL/systemd CUDA libs)"

log "checking dataset..."
[ -d "$DATASET" ] || abort "dataset dir missing: $DATASET"
[ -f "$DATASET/.stage_complete" ] || abort "dataset .stage_complete missing (not finished)"
NPNG=$(find "$DATASET" -maxdepth 1 -name '*.png' | wc -l)
NTXT=$(find "$DATASET" -maxdepth 1 -name '*.txt' | wc -l)
[ "$NPNG" -ge 20 ] || abort "too few images: $NPNG (<20)"
[ "$NPNG" -eq "$NTXT" ] || abort "png/txt count mismatch: $NPNG png vs $NTXT txt"
NBAD=$(grep -L "^${TRIGGER}" "$DATASET"/*.txt 2>/dev/null | wc -l)
[ "$NBAD" -eq 0 ] || abort "$NBAD captions do not lead with '$TRIGGER'"
NLEAK=$(grep -liE '\b(comic|manga|lineart|line art|monochrome|greyscale|grayscale|sketch|screentone|halftone|illustration|painterly|watercolou?r)\b' "$DATASET"/*.txt 2>/dev/null | wc -l)
log "dataset: $NPNG images, $NLEAK caption(s) with style-ish words (informational)"
if [ "$STABILITY_MIN" -gt 0 ]; then
  RECENT=$(find "$DATASET" -maxdepth 1 -type f -mmin -"$STABILITY_MIN" | wc -l)
  [ "$RECENT" -eq 0 ] || abort "dataset still being written ($RECENT files changed in the last ${STABILITY_MIN}min)"
fi

log "checking GPU is free..."
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "${USED:-99999}" -lt 2000 ] || abort "GPU busy: ${USED} MiB used (>=2000; another job holds it)"

log "checking models/token..."
[ -d "$HF_CACHE/hub/models--black-forest-labs--FLUX.1-dev" ] || abort "FLUX.1-dev HF cache missing"
[ -f "$HF_CACHE/token" ] || abort "HF token missing at $HF_CACHE/token"

log "preconditions OK: $NPNG imgs, GPU ${USED} MiB used, CUDA ok"

# --- generate the ai-toolkit config (proven recipe; only steps/save/paths vary) ---
CFG="$RUN/config.yaml"
cat > "$CFG" <<YAML
---
job: extension
config:
  name: "cmcstyle_flux"
  process:
    - type: 'sd_trainer'
      training_folder: "$RUN/out"
      device: cuda:0
      trigger_word: "$TRIGGER"
      network:
        type: "lora"
        linear: 32
        linear_alpha: 16
      save:
        dtype: float16
        save_every: $SAVE_EVERY
        max_step_saves_to_keep: 8
        push_to_hub: false
      datasets:
        - folder_path: "$DATASET"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          flip_x: true
          resolution: [ 512 ]
      train:
        batch_size: 1
        steps: $STEPS
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adafactor"
        lr: 1e-4
        dtype: bf16
        disable_sampling: true
        ema_config:
          use_ema: false
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        is_flux: true
        quantize: true
        qtype: qfloat8
        qtype_te: qfloat8
        low_vram: true
meta:
  name: "cmcstyle_flux"
  version: '1.0'
YAML
log "config written: $CFG (steps=$STEPS save_every=$SAVE_EVERY)"

# --- GPU time series (process diagnostics) ---
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv -l 5 > "$RUN/gpu.csv" 2>&1 &
SAMPLER_PID=$!
log "gpu sampler pid=$SAMPLER_PID -> gpu.csv"

# --- train ---
cd "$AITK" || abort "cannot cd to $AITK"
log "launching ai-toolkit (timeout ${TIMEOUT}s)..."
START_EPOCH=$(date +%s)
timeout "$TIMEOUT" "$PYTHON" run.py "$CFG" > "$RUN/train.log" 2>&1
RC=$?
END_EPOCH=$(date +%s)
kill "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""
log "ai-toolkit exited rc=$RC after $((END_EPOCH-START_EPOCH))s (tail of train.log:)"
tail -n 15 "$RUN/train.log" 2>/dev/null | sed 's/^/    /'

# --- assert + record the produced LoRA (catch the 'empty LoRA' failure) ---
SAFETENSORS=$(find "$RUN/out" -name '*.safetensors' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [ -n "$SAFETENSORS" ] && [ -s "$SAFETENSORS" ]; then
  SFT_SIZE=$(stat -c '%s' "$SAFETENSORS")
  SFT_TENSORS=$(python3 - "$SAFETENSORS" <<'PY' 2>/dev/null || echo 0
import sys, struct, json
with open(sys.argv[1], 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n))
print(len([k for k in hdr if k != "__metadata__"]))
PY
)
  log "LoRA: $SAFETENSORS  bytes=$SFT_SIZE  tensors=$SFT_TENSORS"
  if [ "$RC" -eq 0 ] && [ "$SFT_SIZE" -gt 0 ] && [ "${SFT_TENSORS:-0}" -gt 0 ]; then
    finalize "completed" ""
  else
    finalize "suspect" "rc=$RC size=$SFT_SIZE tensors=$SFT_TENSORS (possible empty/failed LoRA)"
  fi
else
  finalize "failed" "no .safetensors produced (rc=$RC; see train.log)"
fi
log "=== done: status=$STATUS ==="
