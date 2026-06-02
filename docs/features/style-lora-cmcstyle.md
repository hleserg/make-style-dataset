# `cmcstyle` — Flux.1-dev style-LoRA recipe (HLE-803)

Ready-to-fire training recipe and evaluation protocol for the `cmcstyle` **style**
LoRA — the deliverable of [HLE-803](https://linear.app/hleserg/issue/HLE-803)
(parent epic HLE-802). This is the **training half** (question 3): dataset prep
and captioning are owned by the dataset agent (see
[Cross-agent preconditions](#cross-agent-preconditions)).

Goal: a LoRA that applies the comic's *manner* (painterly historical BD —
muted earthy palette, watercolour colouring, fine ink linework) to **arbitrary
content**, without memorising specific characters, panels or layouts.

## Tooling decision: ai-toolkit (local), not kohya

| | Local (this box) | Cloud / 24 GB+ GPU |
|---|---|---|
| Trainer | **ai-toolkit** `/home/serg/ai-toolkit` v0.9.14 | kohya sd-scripts (repo `train` stage) |
| Why | Proven to fit & train Flux on 16 GB | kohya recipe applies directly there |

kohya sd-scripts **cannot train Flux on this 16 GB box**: it loads the whole
fp8 DiT to the GPU and only then offloads swapped blocks, so `blocks_to_swap`
cannot lower the load **peak**, and this kohya has no sub-fp8 training quant. The
5070 Ti is also the live 4K display GPU (no iGPU on the i5-14600KF), so ~1.5 GB
is permanently held by Windows `dwm.exe`; the OOM margin is a few hundred MB and
is **not** WSL-fixable. ai-toolkit quantizes on the **CPU** and streams blocks
(`low_vram`), so the GPU load peak stays low — a real fit-check produced a valid
172 MB Flux LoRA here with no OOM. The repo's kohya `train` stage stays valid for
a one-off cloud/24 GB run.

The hyperparameter recipe below is **backend-agnostic**; only the flag spelling
and the fit-critical knobs (`qtype`/`low_vram`/optimizer) differ.

## The config

ai-toolkit YAML. Fill `<DATASET_DIR>` with the dataset agent's `cmcstyle` folder
(see preconditions). Grounded in the proven fit-check (`/tmp/flux_fitcheck.yaml`)
plus the HLE-803 research synthesis.

```yaml
---
job: extension
config:
  name: "cmcstyle_flux"
  process:
    - type: 'sd_trainer'
      training_folder: "/home/serg/lora_output/aitk/cmcstyle_flux"  # serg-owned output
      device: cuda:0
      trigger_word: "cmcstyle"          # ai-toolkit adds this to any caption lacking it
      network:
        type: "lora"
        linear: 32                      # rank — style needs capacity (HLE-803: 16–32)
        linear_alpha: 16                # alpha/rank = 0.5 → conservative effective LR
      save:
        dtype: float16
        save_every: 250                 # STEP-based → ~8 checkpoints over 2000 steps
        max_step_saves_to_keep: 8
        push_to_hub: false
      datasets:
        - folder_path: "<DATASET_DIR>"  # cmcstyle dataset: images + content-only .txt
          caption_ext: "txt"
          caption_dropout_rate: 0.05    # mild anti-memorisation (ai-toolkit-only lever)
          shuffle_tokens: false         # keep caption order intact
          cache_latents_to_disk: true
          flip_x: true                  # mirror aug → style over composition
          resolution: [ 512 ]           # auto aspect-bucketed; add 768 only after a 768 fit-check
      train:
        batch_size: 1
        steps: 2000                     # 1500–2000; best checkpoint is usually intermediate
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"    # flux-shift objective handled internally
        optimizer: "adafactor"          # near-zero optimiser state → fits 16 GB (proven)
        lr: 1e-4
        dtype: bf16                     # bf16, NOT fp16 (fp16 risks an empty/NaN LoRA)
        disable_sampling: true          # OFF: the sampling 'generate' preset moves the FULL
                                        # transformer (+T5+VAE) to the GPU at once (≠ the streamed
                                        # training step) → unproven OOM risk on 16 GB. Eval post-hoc
                                        # in ComfyUI instead (see Evaluation protocol).
        ema_config:
          use_ema: false                # off to match the proven fit; optional quality knob
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"   # gated diffusers repo (cached + token present)
        is_flux: true
        quantize: true
        qtype: qfloat8                  # NOT qint4 (qint4+low_vram is broken: CUDA-only kernel vs CPU quant)
        qtype_te: qfloat8
        low_vram: true                  # CPU quant + block streaming = the 16 GB fit
      # NOTE: inert while disable_sampling=true. The prompts below ARE the held-out
      # eval set — feed them to the post-hoc ComfyUI grid. Enable in-training sampling
      # only after a short fit-check confirms the 'generate' path fits 16 GB here.
      sample:
        sampler: "flowmatch"
        sample_every: 250               # (only if sampling is enabled)
        width: 512
        height: 512
        guidance_scale: 3.5             # sample at real CFG ~3.5, NOT the training guidance
        sample_steps: 20
        seed: 42
        walk_seed: false                # fixed seed → checkpoints are comparable
        neg: ""
        prompts:                        # held-out FOREIGN subjects + leakage probes ([trigger]→cmcstyle)
          - "[trigger], a red sports car parked on a wet city street at night"
          - "[trigger], an astronaut riding a horse across a desert"
          - "[trigger], a bowl of steaming ramen on a wooden table"
          - "[trigger], a lighthouse on a rocky coast during a storm"
          - "[trigger], a fox sitting in a snowy pine forest"
          - "[trigger], a vintage steam locomotive at a station platform"
          - "[trigger], a portrait of an elderly fisherman, weathered face"   # leakage probe
          - "[trigger], a single robot standing alone in an empty field"      # bubble/border/SFX probe
meta:
  name: "cmcstyle_flux"
  version: '1.0'
```

## Launch (only on the user's signal — needs ~all 16 GB VRAM)

Run **online** (do *not* set `HF_HUB_OFFLINE=1` — ai-toolkit's `model_info`
raises instead of using the cache):

```bash
cd /home/serg/ai-toolkit
venv/bin/python run.py /path/to/cmcstyle_flux.yaml
```

Runtime on this box ≈ **7–9 s/it** under `low_vram` → a 2000-step run ≈ **4–5 h**.
Coordinate GPU time with the other agent (Flux needs ~all 16 GB). Checkpoints land
in `training_folder/cmcstyle_flux/`; sample grids next to them.

## Hyperparameter rationale

| Setting | Value | Why |
|---|---|---|
| rank (`linear`) | **32** | Style needs more capacity than a character LoRA; cheap vs the base. Drop to 16 only if VRAM-tight. |
| alpha (`linear_alpha`) | **16** | alpha/rank = 0.5 → effective LR ≈ 5e-5: the conservative, overfit-resistant regime. Raise to 32 if style under-bakes. |
| `lr` | **1e-4** | Most-cited Flux LoRA LR; in the proven fit-check. Under-bake → 1.5–2e-4; over-bake/bleed → 5e-5. |
| `steps` | **2000** | Style guides cluster 1000–2500; the *best* checkpoint is usually intermediate → pick by eval, not by the last save. |
| `resolution` | **[512]** | Style is low-frequency, good at 512, ~3× faster, less per-panel memorisation. ai-toolkit auto aspect-buckets. 768 only if linework looks soft (needs its own fit-check). |
| `dtype` | **bf16** | Flux is bf16-native; fp16 risks an empty/NaN LoRA. Orthogonal to fp8 weight quant. |
| `optimizer` | **adafactor** | Near-zero optimiser state → fits 16 GB (proven). AdamW8bit is a quality option for the cloud/24 GB path. |
| `noise_scheduler` | **flowmatch** | Selects the flux-shift objective + guidance-distilled training internally. Don't pass training guidance as a flag. |
| `qtype`/`qtype_te` | **qfloat8** | The actual original blocker was qtype, not VRAM: `qint4 + low_vram` throws (CUDA-only int4 kernel vs CPU quant). qfloat8 casts on CPU. |
| `low_vram` | **true** | CPU quant + block streaming keeps the GPU load peak low — mandatory on this display GPU. |
| `flip_x` | **true** | Mirror aug doubles effective data and pushes toward style over memorised composition; comic linework is ~mirror-invariant. |
| `caption_dropout_rate` | **0.05** | Light caption dropout — an extra anti-memorisation nudge (ai-toolkit-only). |
| `save_every` | **250 steps** | Step-based (not epoch-based) → a consistent ~8 checkpoints regardless of dataset size, so the best intermediate one is catchable. |

## Cross-agent preconditions

These are owned by the **dataset/caption agent**, but the `cmcstyle` LoRA
**cannot pass DoD** unless the dataset it consumes satisfies them. Flagged here,
not implemented here.

1. **Trigger = `cmcstyle`, not `comicstyle`.** The repo default (`config.py`) and
   the current dataset use `comicstyle`. The trigger is baked into every caption's
   leading token *and* the dataset folder name, so it must be set
   (`APP_TRIGGER_TOKEN=cmcstyle`) **before** the caption stage runs — renaming the
   folder afterwards yields a silent dud (the LoRA binds to whatever word leads the
   captions). ai-toolkit's `trigger_word` will *add* `cmcstyle` to captions missing
   it, so a clean alternative is **content-only captions with no style prefix at
   all** and let ai-toolkit inject the trigger.
2. **Captions must not name the style.** The current WD14 captioner keeps *all*
   general tags, including style/medium tags (`monochrome`, `greyscale`, `comic`,
   `lineart`, `sketch`, `screentone`, `halftone`, `traditional media`,
   `speech bubble`). Naming the style in words defeats the Style-Locker premise —
   the style must bind to the undescribed trigger, not to common words that appear
   in arbitrary inference prompts. Remedy: a style/medium blocklist + raise the
   threshold 0.35 → ~0.65, **or** a content-only natural-language captioner. This
   is the single highest-impact caption fix and applies to any captioner.
3. **Describe recurring characters (variably) to keep them out of the trigger.**
   The describe-to-exclude principle: anything described binds to the (varying)
   text; anything left undescribed binds to the trigger. So explicitly and
   *variably* describing the recurring people ("a young man with braided hair", "a
   Roman soldier in armour") is exactly what stops the style from memorising them.
   Never give a recurring character a fixed name/token.
4. **Curate for diversity (anti-memorisation lever #1).** Target 20–40 *maximally
   diverse* crops (different characters, scenes, poses, palettes). No single
   recurring subject > ~20–25 % of the set. pHash dedup removes near-identical
   frames but not "same character, different pose" — a per-subject cap is the
   complementary gate.
5. **Keep gutter-bled crops out.** The S1 slicer bleeds neighbour content on
   diagonal/irregular gutters (visible panel-divider lines / two sub-panels in one
   crop). Those inject panel-structure contamination into the very training crops —
   route low fill-ratio crops to `manual_review`.

## Evaluation protocol

**Primary: post-hoc in ComfyUI** on the saved checkpoints (Flux.1-dev inference is
already proven on this box). ai-toolkit's in-training sampling is left **off**: its
`generate` device preset moves the whole transformer (+T5+VAE) to the GPU at once —
unlike the streamed training step — an unproven OOM risk on this 16 GB display GPU.
The `sample.prompts` in the config double as the held-out eval set for the grid. (To
enable in-training previews, first run a short sampling fit-check — see Fallbacks.)

- **Held-out / foreign prompts** — subjects absent from the training crops. PASS =
  the comic style (linework, shading, palette) transfers while **no** specific
  training character/costume/face/scene reappears.
- **Comic leakage probe** (mandatory here) — single-subject prompts must **not**
  come back with panel borders, gutters, multi-panel grids, empty speech bubbles or
  SFX/onomatopoeia text. Unprompted bubbles/borders are the strongest tell that the
  trigger absorbed layout instead of style.
- **Final fixed-seed weight × prompt grid** (for the report) — X = LoRA weight
  `{0.0, 0.5, 0.7, 0.85, 1.0}`, Y = the prompts. The **weight 0.0 (base) column is
  mandatory** as the baseline that proves the style is *added*. Sample at CFG ~3.5.
  Build it in ComfyUI (Flux.1-dev blueprint + a LoRA-strength XY) or SwarmUI XYZ.
- **Pick the checkpoint**: the **earliest** save where the style is locked in but
  prompt-following and subject diversity still survive (style LoRAs peak early; the
  last checkpoint is usually *not* best).
- **Inference weight**: the highest weight that still passes the foreign + leakage
  probes — expect ~0.7–0.85.

## DoD mapping (HLE-803)

| DoD item | How this recipe satisfies it |
|---|---|
| Applies comic style to arbitrary content | Foreign-prompt grid; style binds to the undescribed trigger. |
| Trigger works | `cmcstyle` injected/leading in every caption; probed in every sample prompt. |
| Does **not** drag in specific characters/panels | Diversity curation + describe-to-exclude captions + flip + leakage probes + early-checkpoint picking. |
| Settings documented | This file (config + rationale + frozen values). |
| Dataset format matches the HLE-802 contract | `<dataset>/{img,txt}`, content-only captions, multi-aspect buckets, trigger `cmcstyle`. |

## Fallbacks

- **Empty LoRA** (tiny file / near-zero tensors) → a precision/quant bug: confirm
  `dtype: bf16` and `qtype: qfloat8` (not qint4).
- **Under-baked** (weights present, style weak) → raise `lr` to 1.5–2e-4, or
  `linear_alpha` 16 → 32, or pick a later checkpoint.
- **Over-baked / content bleed** → drop `lr` to 5e-5, lower the inference weight,
  or pick an earlier checkpoint.
- **Linework soft** → add `768` to `resolution` (re-check the fit first).
- **In-training sampling OOMs (~step 250)** → expected if you flip `disable_sampling`
  to `false`: the `generate` preset materialises the full transformer on the GPU.
  Keep `disable_sampling: true` and eval post-hoc in ComfyUI; only enable sampling
  after a short fit-check (≈260 steps to cross one `sample_every` boundary) proves it
  fits — that fit-check is a "train" action, so run it in a user-opened GPU window.

> Heuristic numbers (inference weight 0.7–0.85, per-subject cap 20–25 %, dropout
> 0.05) are starting points to calibrate against the eval grid, not hard constants.
