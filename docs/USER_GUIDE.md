# User guide — building a style dataset, step by step

> A plain-language walkthrough for non-developers.
> [Русская версия](USER_GUIDE-ru.md)

This tool turns a pile of comic pages into a clean, ready-to-train **style
dataset** (folders of pictures with matching text descriptions, laid out the
way the [kohya](https://github.com/bmaltais/kohya_ss) LoRA trainer expects).
It slices pages into panels, erases speech bubbles and sound effects, throws out
duplicates and tiny crops, and writes a caption for every picture.

You do not need to understand the code. Follow the steps below.

---

## What you need

- A **PC with an NVIDIA graphics card** (GPU). It works without one, but the
  AI steps become very slow.
- **Windows or Linux.** On Windows, use **WSL** (Ubuntu) — a free Linux
  environment inside Windows. ([How to install WSL](https://learn.microsoft.com/windows/wsl/install) —
  one command, then restart.)
- About **15 GB of free disk space** (the AI models are large).
- An **internet connection** for the first run (to download the models, once).

---

## One-time setup

Do this **once**. Open your terminal (on Windows: the **Ubuntu** app from WSL),
then get a copy of the project and run the setup script:

```bash
# 1. Download the project (or use the green "Code → Download ZIP" button on GitHub)
git clone https://github.com/hleserg/make-style-dataset.git
cd make-style-dataset

# 2. Run the all-in-one setup (installs everything, ~10–20 min the first time)
bash scripts/setup.sh
```

That single command installs the tools, downloads the dependencies, creates
your working folders, and checks that your machine is ready. When it finishes
it prints what to do next.

> **No NVIDIA GPU?** Run `bash scripts/setup.sh --no-gpu` instead. The AI steps
> will run on the CPU (much slower) or you can run them later on a GPU machine.

If you ever want to re-check your machine, run:

```bash
uv run make-style-dataset doctor
```

It prints a checklist — every line should say `[ok]`.

---

## Building a dataset

### The easy way — the app (recommended)

After setup, open the app in your browser:

```bash
uv run make-style-dataset ui      # or:  make ui
```

A 3-step wizard appears:

1. **Name your style** — pick a trigger word (and the kohya repeat count).
2. **Add pages & build** — drag your comic pages in and press **▶ Build dataset**.
   The progress log streams live. *The first run downloads several GB of AI
   models — a step may sit on “running…” for a few minutes; that's normal, it
   hasn't frozen.*
3. **Get your dataset** — browse the result gallery and click **Download .zip**.
   A separate **Manual review** tab shows the tricky pages set aside for you.

That's the whole flow, no terminal needed after setup. Prefer the command line?
The steps below do exactly the same thing.

### Or run it from the command line

#### Step 1 — add your pages

Put your comic page images (`.png`, `.jpg`, …) into the **`workspace/00_pages/`**
folder. The setup created it for you. One image per page; the more pages, the
better (aim for **20+ pages** to end up with enough panels).

#### Step 2 — run the pipeline

```bash
uv run make-style-dataset run-all
```

The **first** run downloads the AI models (a few GB) and takes several minutes.
**This is normal — it has not frozen.** Later runs are much faster.

When it finishes it prints a summary like:

```
Pipeline summary:
  pages (00_pages)        24
  panels (01_panels)      96
  ...
  dataset (10_comicstyle) 83
  manual_review            11
```

#### Step 3 — collect your dataset

Your finished dataset is in **`workspace/05_dataset/10_comicstyle/`** — a folder
of `.png` images, each with a matching `.txt` caption next to it. That whole
folder is what you feed to the kohya LoRA trainer.

(`10` is the repeat count and `comicstyle` is the trigger word — see *Settings*
below to change them.)

---

## The two settings most people change

Open the **`.env`** file (created during setup) in any text editor and change
these two lines if you want:

```ini
APP_TRIGGER_TOKEN=comicstyle   # the word that will "summon" your style in prompts
APP_DATASET_REPEATS=10         # kohya repeat count; names the output folder
```

Save the file and run the pipeline again. Everything else has sensible defaults;
you can ignore it.

---

## About the `manual_review/` folder

**Full automation is not realistic, and that's expected.** Roughly **1 in 7**
pages are tricky — a full-page splash, overlapping panels, or a bubble that
crosses two panels. Instead of guessing and making a mess, the tool sets those
aside in **`workspace/manual_review/`**.

Open that folder and look. You can:

- **Ignore it** — if you already have enough good panels, you're done.
- **Fix a few by hand** — crop the good panels out yourself and drop them into
  `workspace/04_clean/`, then run the pipeline again to caption them.

Getting **80–150 clean panels** is a great style dataset. You usually don't need
to rescue everything.

---

## Troubleshooting

| What you see | What's happening / what to do |
|---|---|
| It looks frozen on the first run | It's downloading the AI models (a few GB). Wait — it hasn't crashed. |
| `doctor` shows `[--] torch / CUDA` | The GPU stack isn't installed, or no NVIDIA GPU is visible. Re-run `bash scripts/setup.sh`, or use a GPU machine. |
| `doctor` shows `[--] pages` | The `workspace/00_pages/` folder is empty — add your page images. |
| Very slow | You're running on CPU. The AI steps want an NVIDIA GPU. |
| `command not found: uv` | Close and reopen your terminal, then try again (setup added `uv` to your path). |
| Want to start over | Delete the `workspace/` folder and run `uv run make-style-dataset init`. |

---

## In short

```bash
bash scripts/setup.sh                  # once
uv run make-style-dataset ui           # the app: drop pages, Build, download .zip
```

…or all from the terminal:

```bash
#   put pages in workspace/00_pages/
uv run make-style-dataset run-all      # each time
#   collect workspace/05_dataset/<N>_<trigger>/
```

For the technical details of each stage, see the [README](../README.md) and
[docs/architecture/SYSTEM.md](architecture/SYSTEM.md).
