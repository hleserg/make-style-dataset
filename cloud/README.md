# Cloud training deploy — `cmcstyle` Flux style-LoRA

One-script cloud training with the **quality-preferred** settings, for a rented
GPU. **Local training is unaffected** (this folder is cloud-only; the 16 GB recipe
in `docs/features/style-lora-cmcstyle.md` stays as-is).

## Why cloud (and what changes)

Fitting Flux into our local 16 GB costs mostly **time, not quality**. On a ≥24 GB
cloud GPU the `low_vram` block-streaming penalty disappears (~3–6× faster) and we
flip the few quality/ergonomic knobs that VRAM forced locally:

| Setting | Local (16 GB) | Cloud (this deploy) |
|---|---|---|
| optimizer | adafactor (low-mem) | **AdamW8bit** (quality-preferred) |
| `low_vram` | on (streaming) | **off** (resident → fast) |
| `save_every` | 250 | **100** (finer best-checkpoint pick) |
| resolution | 512 | **768** (`512,768,1024` on ≥48 GB) |
| in-training samples | off (OOM risk) | **on** (eval grid during training) |
| rank/alpha, lr, qfloat8, bf16, flip_x, captions | — | **unchanged** (validated) |

`qfloat8` base quantization is kept (quality-neutral per research; lets it fit
24 GB). Drop `quantize` only on a 48 GB+ card for full bf16.

## What you need

1. A GPU box with **≥24 GB VRAM, CUDA 12.x** (RunPod / Vast / Lambda). RTX 4090 /
   A6000 / L40S are the cheap sweet spot (~$0.3–0.8/hr).
2. An **HF token** with access to the gated `black-forest-labs/FLUX.1-dev`.
3. The **dataset** reachable: either a **private HF dataset repo** (recommended —
   also satisfies the HLE-802 "shared location" contract) or a folder you upload.

## Run

```bash
# on the fresh box:
export HF_TOKEN=hf_xxx
export DATASET_HF_REPO=hleserg/cmcstyle-style-dataset   # already published (private)
bash provision_and_train.sh
# optional: RES="512,768,1024"  STEPS=2000  SAVE_EVERY=100
```

The script: clones ai-toolkit → installs deps → auths HF → downloads FLUX.1-dev
(~25 GB, first run) → pulls the dataset → renders the config → trains with
diagnostics (`gpu.csv`, `train.log`) → asserts a non-empty `.safetensors`.
Then download the checkpoints back and run the eval grid
(`scripts/eval_style_lora.py --lora <ckpt>`).

## Dataset (already published)

The dataset is published to a **private HF dataset repo**:
`hleserg/cmcstyle-style-dataset` (116 img+txt, content-only prose captions). This is
also the HLE-802 "shared location" — the char-agent can pull the same crops as the
style reference. `provision_and_train.sh` pulls it via `DATASET_HF_REPO`.

To re-publish after the dataset changes:

```bash
huggingface-cli upload --repo-type dataset hleserg/cmcstyle-style-dataset \
  /path/to/05_dataset/10_cmcstyle .
```

## Notes / caveats

- **Not yet run on a real cloud box** — validate the first run, especially the
  torch/CUDA wheel match during `pip install -r requirements.txt`. ai-toolkit also
  ships its own `docker/` if you prefer a prebuilt container over the venv path.
- A tight 24 GB card may OOM while sampling at 1024 — set `disable_sampling: true`
  in the rendered config (`$OUT_DIR/config.yaml`) if so.
- Managed trainers (fal.ai / Replicate `ostris/flux-dev-lora-trainer`) are the
  zero-deploy alternative (~$2, minutes) but give less control over rank / captions
  / per-checkpoint eval — this deploy keeps full control.
