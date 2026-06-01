# System overview

`make-style-dataset` turns a pile of comic pages into a
[kohya_ss](https://github.com/bmaltais/kohya_ss)-ready dataset for training a
**style** LoRA. It is a linear, file-driven pipeline: each stage reads one
folder under the workspace and writes the next.

## Pipeline stages

The pipeline has six conceptual stages — an input drop plus five transforms,
one per CLI subcommand:

0. **pages** *(input)* — raw pages land in `00_pages/`.
1. **panels** — detect comic panels and slice pages into individual panels (Kumiko + OpenCV).
2. **bubbles** — detect speech bubbles and emit removal masks.
3. **inpaint** — paint the masked bubbles out of each panel.
4. **clean** — drop near-duplicates (perceptual hash) and too-small panels.
5. **caption** — caption survivors and lay out `05_dataset/<N>_<trigger>/`.

`run-all` executes stages 1–5 in order; each stage is idempotent (see
[Workspace contract](WORKSPACE.md)). Stage internals are stubs at S0 and are
filled in by the per-stage issues (HLE-742 …).

## Components

- **cli** — thin argparse shell (`make-style-dataset <stage>` / `run-all`).
- **pipeline** — ordered stage registry + idempotent runner.
- **stages** — one module per transform; metadata + a `run(ctx)` function.
- **workspace** — typed, single-root directory contract.
- **config** — typed settings (paths, trigger token, thresholds, stage flags);
  the only place that reads the environment.
- **observability** — Sentry init (`send_default_pii=False`) + per-stage
  component tags (`stage:panels`, `stage:bubbles`, …).

## Data flow

```mermaid
flowchart LR
    In[00_pages] --> panels --> bubbles --> inpaint --> clean --> caption --> DS[(05_dataset)]
    panels -.tags.-> Sentry
    caption -.tags.-> Sentry
```

See the [Workspace layout contract](WORKSPACE.md) for the full folder map.
