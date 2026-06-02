"""Re-caption a finished dataset with a Gemini VLM (prose), via the proxy.

A **separate** mechanism from the WD14 ``caption`` stage (owned by another agent):
it reads the final ``05_dataset/<N>_<trigger>/*.png`` and rewrites the ``.txt``
sidecars with trigger-first natural-language prose that describes **all variable
content** and **never names the art style** — so for a Flux style LoRA the style
residual binds to the trigger instead of leaking onto style words (which booru
tags like ``monochrome``/``comic``/``lineart`` otherwise cause).

Every Gemini call goes through :mod:`make_style_dataset.proxy` (the box's region
is geo-blocked). The network client is injected via the ``CaptionClient`` protocol
so this module is unit-tested with a stub.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from make_style_dataset.media import image_files

if TYPE_CHECKING:
    from make_style_dataset.proxy import CaptionClient

#: Phrase appended to every prompt: the style trigger owns these, so the caption
#: must not name them (else the style binds to the words, not the trigger).
NO_STYLE = (
    "Do NOT mention the art style, medium, rendering, line, palette, colors, or words like "
    "comic / painterly / ink / illustration / monochrome / lineart / sketch / screentone / "
    "speech bubble — a separate style trigger handles all of that."
)

#: Caption-style presets. 'rich' describes more content (cleaner style residual);
#: 'optimal' is shorter. See the experiment in [[gemini-proxy-and-vlm-caption]].
PROMPT_STYLES = ("optimal", "rich")


def build_prompt(trigger: str, style: str) -> str:
    """Return the VLM instruction for ``style`` ('optimal' | 'rich'), trigger-first."""
    head = (
        "You are captioning ONE comic panel to train a Flux style LoRA. "
        f"Begin exactly with '{trigger}, '."
    )
    if style == "rich":
        return (
            f"{head} Output natural-language prose, 3-6 sentences. Describe EVERYTHING visible "
            "in detail: every character with full physical description (hair, face, expression, "
            "build, age), every garment and accessory, every object and prop, each figure's pose "
            "and gesture, the spatial composition (foreground / midground / background), the "
            "environment, the shot type, camera angle and framing, and the lighting. Be exhaustive "
            "about content. " + NO_STYLE + " Output only the caption text."
        )
    return (
        f"{head} Output ONE line of natural-language prose, 1-3 sentences. Describe the variable "
        "content so the style trigger does not absorb it: who is in frame (rich physical detail), "
        "pose/action, clothing, key objects, background/location, the shot type and camera angle, "
        "and the lighting. " + NO_STYLE + " Output only the caption line."
    )


def normalize_caption(text: str, trigger: str) -> str:
    """Collapse to one line and guarantee a single leading ``'<trigger>, '`` prefix."""
    one_line = " ".join((text or "").split())
    if one_line.lower().startswith(trigger.lower()):
        one_line = one_line[len(trigger) :].lstrip(" ,")
    return f"{trigger}, {one_line}".rstrip()


def _is_transient(error: object) -> bool:
    """True for retryable proxy/Gemini errors (5xx overload, 429 rate limit)."""
    err = str(error)
    return err.startswith("http_5") or err == "http_429" or err in {"empty_response", "URLError"}


@dataclass(frozen=True)
class RecaptionResult:
    """Outcome of a dataset re-caption pass."""

    written: int
    failed: int
    errors: list[str] = field(default_factory=list)


def recaption_dataset(
    dataset_dir: Path,
    *,
    trigger: str,
    model: str,
    style: str,
    client: CaptionClient,
    max_workers: int = 8,
    retries: int = 3,
    backoff: float = 1.5,
    sleep=time.sleep,
) -> RecaptionResult:
    """Rewrite every image's ``.txt`` in ``dataset_dir`` with a VLM prose caption.

    Calls ``client`` (the proxy) concurrently; retries transient errors with
    backoff. A non-empty caption is normalized (trigger-first, single line) and
    written to ``<image>.txt``; failures are counted and the first few reported.
    The images and the run order are untouched, so the kohya layout still holds.
    """
    prompt = build_prompt(trigger, style)
    images = image_files(dataset_dir)

    def caption_one(path: Path) -> tuple[Path, str | None, str | None]:
        data = path.read_bytes()
        result: dict[str, object] = {"error": "not_run"}
        for attempt in range(retries):
            result = client.caption(model, prompt, data)
            if not _is_transient(result.get("error", "")) or "error" not in result:
                break
            sleep(backoff * (attempt + 1))
        text = result.get("caption")
        if result.get("error") or not text:
            return path, None, str(result.get("error") or "empty_caption")
        return path, normalize_caption(str(text), trigger), None

    written = 0
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for path, caption, err in pool.map(caption_one, images):
            if caption is None:
                errors.append(f"{path.name}: {err}")
            else:
                path.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
                written += 1
    return RecaptionResult(written=written, failed=len(errors), errors=errors[:10])
