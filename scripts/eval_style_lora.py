#!/usr/bin/env python3
"""Eval harness for the ``cmcstyle`` Flux style LoRA (HLE-803).

Renders a fixed-seed **weight × prompt** contact sheet through a running ComfyUI
(Flux.1-dev) so a trained LoRA can be judged in one step:

* held-out / *foreign* prompts (subjects absent from the training crops) prove the
  style transfers to arbitrary content;
* the mandatory ``weight 0.0`` (base, no-LoRA) column proves the style is *added*;
* the last prompts are leakage probes — single subjects that must come back
  WITHOUT specific training characters, panel borders, gutters or speech bubbles.

The harness talks to ComfyUI over its HTTP API (stdlib only) and assembles the grid
with Pillow — no torch/diffusers here (ComfyUI owns the GPU env). It builds a stock
Flux txt2img graph for this box by default; pass ``--workflow`` to override it with
your own API-format workflow (export from ComfyUI via *Save (API Format)*) using the
``%PROMPT%`` / ``%LORA%`` / ``%STRENGTH%`` / ``%SEED%`` placeholders.

Usage (only meaningful once a LoRA exists and ComfyUI is up):

    # start ComfyUI first (separate terminal), then:
    python scripts/eval_style_lora.py --lora cmcstyle.safetensors
    python scripts/eval_style_lora.py --lora cmcstyle-000500.safetensors cmcstyle.safetensors

    # validate the harness now, no GPU/ComfyUI needed (writes graphs + a layout grid):
    python scripts/eval_style_lora.py --lora cmcstyle.safetensors --dry-run

See ``docs/features/style-lora-cmcstyle.md`` for the eval protocol this implements.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --- Defaults grounded in this box + the recipe ----------------------------

#: ComfyUI model filenames on this box (models/{unet,clip,vae}).
DEFAULT_UNET = "flux1-dev-fp8.safetensors"
DEFAULT_CLIP_L = "clip_l.safetensors"
DEFAULT_T5XXL = "t5xxl_fp16.safetensors"
DEFAULT_VAE = "ae.safetensors"

#: LoRA inference weights to sweep. 0.0 (base, no LoRA) is the mandatory baseline.
DEFAULT_WEIGHTS = (0.0, 0.5, 0.7, 0.85, 1.0)

#: Held-out FOREIGN prompts (subjects absent from the comic) + leakage probes.
#: ``%TRIGGER%`` is substituted with the trigger token.
DEFAULT_PROMPTS = (
    "%TRIGGER%, a red sports car parked on a wet city street at night",
    "%TRIGGER%, an astronaut riding a horse across a desert",
    "%TRIGGER%, a bowl of steaming ramen on a wooden table",
    "%TRIGGER%, a lighthouse on a rocky coast during a storm",
    "%TRIGGER%, a fox sitting in a snowy pine forest",
    "%TRIGGER%, a vintage steam locomotive at a station platform",
    "%TRIGGER%, a portrait of an elderly fisherman, weathered face",  # leakage: not the blond youth
    "%TRIGGER%, a single robot standing alone in an empty field",  # leakage: no borders/bubbles/SFX
)

#: Flux sampling defaults (guidance ~3.5 at inference, NOT the training guidance).
DEFAULT_GUIDANCE = 3.5
DEFAULT_STEPS = 20
DEFAULT_SIZE = 1024
DEFAULT_SEED = 42
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"


@dataclass(frozen=True)
class Cell:
    """One grid cell: a prompt row crossed with a LoRA-weight column."""

    row: int
    prompt: str
    weight: float


# --- Pure: build the ComfyUI API-format Flux txt2img graph -----------------


def build_flux_workflow(
    *,
    prompt: str,
    lora_name: str | None,
    strength: float,
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    unet: str = DEFAULT_UNET,
    clip_l: str = DEFAULT_CLIP_L,
    t5xxl: str = DEFAULT_T5XXL,
    vae: str = DEFAULT_VAE,
) -> dict:
    """Return a ComfyUI API-format Flux txt2img graph.

    When ``lora_name`` is ``None`` or ``strength == 0`` the LoRA loader is omitted,
    so the model is the bare base — that is the mandatory ``weight 0.0`` baseline
    column. Uses core ComfyUI node ``class_type``s only.
    """
    graph: dict[str, dict] = {
        "10": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet, "weight_dtype": "default"},
        },
        "11": {
            "class_type": "DualCLIPLoader",
            "inputs": {"clip_name1": clip_l, "clip_name2": t5xxl, "type": "flux"},
        },
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
    }
    model_ref = ["10", 0]
    if lora_name and strength != 0:
        graph["13"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["10", 0], "lora_name": lora_name, "strength_model": strength},
        }
        model_ref = ["13", 0]

    graph["20"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": prompt}}
    graph["21"] = {
        "class_type": "FluxGuidance",
        "inputs": {"conditioning": ["20", 0], "guidance": guidance},
    }
    graph["22"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": ""}}
    graph["30"] = {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    graph["40"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": ["21", 0],
            "negative": ["22", 0],
            "latent_image": ["30", 0],
            "seed": seed,
            "steps": steps,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
    }
    graph["50"] = {"class_type": "VAEDecode", "inputs": {"samples": ["40", 0], "vae": ["12", 0]}}
    graph["60"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["50", 0], "filename_prefix": "cmcstyle_eval"},
    }
    return graph


def apply_template(
    template: dict, *, prompt: str, lora_name: str, strength: float, seed: int
) -> dict:
    """Substitute %PROMPT%/%LORA%/%STRENGTH%/%SEED% placeholders in a user workflow."""
    text = json.dumps(template)
    text = text.replace("%PROMPT%", prompt).replace("%LORA%", lora_name)
    text = text.replace('"%STRENGTH%"', str(strength)).replace("%STRENGTH%", str(strength))
    text = text.replace('"%SEED%"', str(seed)).replace("%SEED%", str(seed))
    return json.loads(text)


def make_cells(prompts: list[str], weights: list[float]) -> list[Cell]:
    """Cross every prompt row with every weight column, in display order."""
    return [Cell(row=r, prompt=p, weight=w) for r, p in enumerate(prompts) for w in weights]


# --- ComfyUI HTTP client ---------------------------------------------------


class ComfyClient:
    """Minimal ComfyUI API client (submit a graph, wait, fetch the output image)."""

    def __init__(self, base_url: str, timeout: float = 600.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._client_id = uuid.uuid4().hex

    def is_up(self) -> bool:
        """True if ComfyUI answers ``/system_stats``."""
        try:
            with urllib.request.urlopen(f"{self._base}/system_stats", timeout=5) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def submit(self, graph: dict) -> str:
        """Queue a graph; return its prompt_id."""
        body = json.dumps({"prompt": graph, "client_id": self._client_id}).encode()
        req = urllib.request.Request(
            f"{self._base}/prompt", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["prompt_id"]

    def wait_image(self, prompt_id: str, poll: float = 1.5) -> bytes:
        """Poll history until the prompt finishes; return the first output image's bytes."""
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            with urllib.request.urlopen(f"{self._base}/history/{prompt_id}", timeout=15) as r:
                history = json.loads(r.read())
            entry = history.get(prompt_id)
            if entry:
                for node_out in entry.get("outputs", {}).values():
                    images = node_out.get("images")
                    if images:
                        return self._download(images[0])
                raise RuntimeError(f"prompt {prompt_id} finished with no image output")
            time.sleep(poll)
        raise TimeoutError(f"prompt {prompt_id} did not finish in {self._timeout:.0f}s")

    def _download(self, image_ref: dict) -> bytes:
        q = urllib.parse.urlencode(
            {
                "filename": image_ref["filename"],
                "subfolder": image_ref.get("subfolder", ""),
                "type": image_ref.get("type", "output"),
            }
        )
        with urllib.request.urlopen(f"{self._base}/view?{q}", timeout=30) as r:
            return r.read()


# --- Grid assembly ---------------------------------------------------------


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if font.getbbox(trial)[2] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def assemble_grid(
    images: dict[tuple[int, float], Image.Image | None],
    prompts: list[str],
    weights: list[float],
    *,
    thumb: int = 320,
    label_w: int = 260,
    header_h: int = 36,
) -> Image.Image:
    """Lay out a labelled contact sheet: rows = prompts, columns = LoRA weights."""
    cols, rows = len(weights), len(prompts)
    grid_w = label_w + cols * thumb
    grid_h = header_h + rows * thumb
    sheet = Image.new("RGB", (grid_w, grid_h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    head_font, body_font = _font(20), _font(15)

    for c, w in enumerate(weights):
        label = "base (w=0.0)" if w == 0 else f"w={w:g}"
        x = label_w + c * thumb
        draw.text((x + 8, 8), label, fill=(20, 20, 20), font=head_font)

    for r, prompt in enumerate(prompts):
        y = header_h + r * thumb
        text = prompt.replace("%TRIGGER%, ", "")
        for i, line in enumerate(_wrap(text, body_font, label_w - 16)):
            draw.text((8, y + 8 + i * 18), line, fill=(20, 20, 20), font=body_font)
        for c, w in enumerate(weights):
            x = label_w + c * thumb
            img = images.get((r, w))
            if img is None:
                draw.rectangle([x + 2, y + 2, x + thumb - 2, y + thumb - 2], outline=(200, 60, 60))
                draw.text(
                    (x + 10, y + thumb // 2), "(no image)", fill=(200, 60, 60), font=body_font
                )
            else:
                fitted = img.copy()
                fitted.thumbnail((thumb - 4, thumb - 4))
                sheet.paste(fitted, (x + 2, y + 2))
    return sheet


# --- Orchestration ---------------------------------------------------------


def run_eval(args: argparse.Namespace) -> int:
    """Render one contact sheet per LoRA file. Returns a process exit code."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = _load_prompts(args)
    resolved_prompts = [p.replace("%TRIGGER%", args.trigger) for p in prompts]
    weights = [float(w) for w in args.weights.split(",")]
    template = json.loads(Path(args.workflow).read_text()) if args.workflow else None

    client = ComfyClient(args.comfy_url)
    if not args.dry_run and not client.is_up():
        print(
            f"ComfyUI is not reachable at {args.comfy_url}. Start it, or use --dry-run.",
            file=sys.stderr,
        )
        return 2

    for lora in args.lora:
        lora_name = Path(lora).name
        stem = Path(lora).stem
        images: dict[tuple[int, float], Image.Image | None] = {}
        for cell in make_cells(resolved_prompts, weights):
            graph = (
                apply_template(
                    template,
                    prompt=cell.prompt,
                    lora_name=lora_name,
                    strength=cell.weight,
                    seed=args.seed,
                )
                if template
                else build_flux_workflow(
                    prompt=cell.prompt,
                    lora_name=lora_name,
                    strength=cell.weight,
                    seed=args.seed,
                    width=args.size,
                    height=args.size,
                    steps=args.steps,
                    guidance=args.guidance,
                )
            )
            if args.dry_run:
                gpath = out_dir / f"graph_{stem}_r{cell.row}_w{cell.weight:g}.json"
                gpath.write_text(json.dumps(graph, indent=2))
                images[(cell.row, cell.weight)] = None
                continue
            try:
                images[(cell.row, cell.weight)] = Image.open(
                    _bytes_io(client.wait_image(client.submit(graph)))
                )
                print(f"[{stem}] row {cell.row} w={cell.weight:g} ok")
            except (RuntimeError, TimeoutError, urllib.error.URLError, OSError) as exc:
                print(f"[{stem}] row {cell.row} w={cell.weight:g} FAILED: {exc}", file=sys.stderr)
                images[(cell.row, cell.weight)] = None

        grid = assemble_grid(images, resolved_prompts, weights)
        grid_path = out_dir / f"grid_{stem}.png"
        grid.save(grid_path)
        print(f"wrote {grid_path}")

    _write_report(out_dir, args, [Path(p).stem for p in args.lora])
    return 0


def _bytes_io(data: bytes):
    import io

    return io.BytesIO(data)


def _load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_file:
        return [ln.strip() for ln in Path(args.prompts_file).read_text().splitlines() if ln.strip()]
    return list(DEFAULT_PROMPTS)


def _write_report(out_dir: Path, args: argparse.Namespace, stems: list[str]) -> None:
    grids = "\n".join(f"### `{s}`\n\n![grid](grid_{s}.png)\n" for s in stems)
    out_dir.joinpath("report.md").write_text(
        f"""# cmcstyle eval report

Trigger `{args.trigger}`, seed {args.seed}, guidance {args.guidance}, weights `{args.weights}`.
Each grid: rows = held-out prompts, columns = LoRA weight (the `w=0.0` column is the base baseline).

## What to check (HLE-803 DoD)
- [ ] **Style transfers** to foreign subjects (compare each row vs its `w=0.0` cell).
- [ ] **No character leakage** — the elderly-fisherman / robot rows do NOT bring back the
      trained blond braided youth, Roman officers, or any specific training face.
- [ ] **No layout leakage** — no panel borders, gutters, multi-panel grids, empty speech
      bubbles or SFX text appear unprompted.
- [ ] **Best weight** (highest that still passes the above; expect ~0.7–0.85): __
- [ ] **Best checkpoint** (earliest with the style locked, prompt still followed): __

## Grids
{grids}
"""
    )
    print(f"wrote {out_dir / 'report.md'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Eval the cmcstyle Flux style LoRA via ComfyUI.")
    p.add_argument(
        "--lora",
        nargs="+",
        required=True,
        help="LoRA filename(s) in ComfyUI/models/loras (one grid each).",
    )
    p.add_argument("--trigger", default="cmcstyle", help="Trigger token substituted into prompts.")
    p.add_argument(
        "--weights",
        default=",".join(f"{w:g}" for w in DEFAULT_WEIGHTS),
        help="Comma-separated LoRA weights.",
    )
    p.add_argument(
        "--prompts-file", help="Optional file: one prompt per line (overrides the defaults)."
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    p.add_argument("--size", type=int, default=DEFAULT_SIZE)
    p.add_argument("--comfy-url", default=DEFAULT_COMFY_URL)
    p.add_argument(
        "--workflow",
        help="API-format workflow JSON; placeholders %%PROMPT%% %%LORA%% %%STRENGTH%% %%SEED%%.",
    )
    p.add_argument("--out", default="eval_out", help="Output directory for grids + report.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build graphs + a layout grid without calling ComfyUI.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    return run_eval(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
