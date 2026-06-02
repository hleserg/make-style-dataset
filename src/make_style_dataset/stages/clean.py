"""Stage 4 — deduplicate, size-filter and upscale inpainted panels.

Reads inpainted panels from ``03_inpainted`` and writes the survivors to
``04_clean``:

- **Dedup** — a perceptual hash (DCT-based pHash, computed with ``cv2.dct``)
  drops near-duplicate panels (the same shot across adjacent frames), which
  would otherwise over-train the LoRA on one moment. Two panels are duplicates
  when their Hamming distance is below ``dedup_hamming_distance``.
- **Size filter** — panels whose shorter side is below ``min_side_px`` after
  inpainting are routed to ``manual_review`` (too small to upscale cleanly).
- **Upscale** — survivors whose shorter side is below ``target_side`` are
  upscaled to it with Lanczos. Per the issue's rule "clean line matters more
  than resolution — do not fabricate fake details", this is a plain resampler,
  not an AI super-resolver (which would hallucinate texture a style LoRA would
  then learn).
- **Denoise** — an optional, off-by-default JPEG-artifact pass
  (``clean_denoise``): it softens the very line texture a style LoRA must learn,
  so it is only for noisy scans.

This is a CPU stage with no heavy/optional backend: the perceptual hash uses
``cv2.dct`` (OpenCV is already a dependency), so the whole stage — hashing,
dedup policy, size/upscale rules, report — is pure and unit tested, like
``panels.py``. Output names are deterministic so re-runs overwrite rather than
duplicate.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from make_style_dataset.media import image_files, route_to_manual
from make_style_dataset.observability import tag_component
from make_style_dataset.stages.base import Stage, StageContext, StageResult

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np

NAME = "clean"
SUMMARY = "Deduplicate panels and filter out ones that are too small."
COMPONENT = "stage:clean"

#: pHash: edge of the low-frequency block kept (8 -> a 64-bit hash) and the
#: factor by which the image is first downscaled (8 * 4 = 32x32 before the DCT).
PHASH_SIZE = 8
PHASH_HIGHFREQ_FACTOR = 4


# --- Pure perceptual hashing + dedup policy --------------------------------


def phash(image: np.ndarray) -> int:
    """Return a 64-bit DCT perceptual hash of a BGR ``image`` (via ``cv2.dct``).

    Grayscale, downscale to ``PHASH_SIZE * PHASH_HIGHFREQ_FACTOR`` square, take
    the top-left ``PHASH_SIZE`` x ``PHASH_SIZE`` DCT coefficients, and pack the
    "coefficient > median" bits into an int. Perceptually similar images get
    hashes a small Hamming distance apart.
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge = PHASH_SIZE * PHASH_HIGHFREQ_FACTOR
    small = cv2.resize(gray, (edge, edge), interpolation=cv2.INTER_AREA).astype(np.float32)
    coeffs = cv2.dct(small)[:PHASH_SIZE, :PHASH_SIZE]
    median = float(np.median(coeffs))
    value = 0
    for bit in (coeffs > median).flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    """Return the number of differing bits between two hashes."""
    return (a ^ b).bit_count()


def dedup(items: Iterable[tuple[Path, int]], threshold: int) -> tuple[list[Path], list[Path]]:
    """Split ordered ``(path, hash)`` pairs into kept and dropped-as-duplicate.

    Greedy in input order: a panel is kept when its hash is at least
    ``threshold`` bits from every already-kept panel; otherwise it is a
    near-duplicate of an earlier one and dropped. ``threshold`` of 0 keeps
    everything.
    """
    kept: list[Path] = []
    kept_hashes: list[int] = []
    dropped: list[Path] = []
    for path, value in items:
        if all(hamming_distance(value, other) >= threshold for other in kept_hashes):
            kept.append(path)
            kept_hashes.append(value)
        else:
            dropped.append(path)
    return kept, dropped


# --- Pure size / quality transforms ----------------------------------------


def short_side(image: np.ndarray) -> int:
    """Return the length of the image's shorter side in pixels."""
    height, width = image.shape[:2]
    return min(height, width)


def upscale_to(image: np.ndarray, target_short_side: int) -> np.ndarray:
    """Lanczos-upscale so the shorter side reaches ``target_short_side``.

    Returns the image unchanged when it is already large enough. Lanczos keeps
    lines clean without inventing detail (unlike an AI super-resolver).
    """
    import cv2

    height, width = image.shape[:2]
    current = min(height, width)
    if current >= target_short_side:
        return image
    scale = target_short_side / current
    new_size = (round(width * scale), round(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)


def denoise(image: np.ndarray) -> np.ndarray:
    """Apply a mild colour denoise (JPEG-artifact removal) to a BGR image."""
    import cv2

    return cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)


def format_clean_report(*, kept: int, deduped: int, dropped_small: int, upscaled: int) -> str:
    """Render the one-line cleaning summary for the stage log."""
    return (
        f"clean: kept {kept}, deduped {deduped}, dropped-small {dropped_small}, upscaled {upscaled}"
    )


# --- I/O helpers -----------------------------------------------------------


def _decode_bgr(path: Path) -> np.ndarray | None:
    """Decode an image to a BGR array via a byte buffer (OS-agnostic, non-ASCII safe)."""
    import cv2
    import numpy as np

    buffer = np.fromfile(path, dtype=np.uint8)
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _write_png(image_bgr: np.ndarray, path: Path) -> None:
    """Save a BGR uint8 array as PNG via PIL (never ``cv2.imwrite`` on a path)."""
    from PIL import Image

    Image.fromarray(image_bgr[:, :, ::-1], mode="RGB").save(path)


def iter_panels(panels_dir: Path) -> list[Path]:
    """Return panel image files under ``panels_dir`` in stable (sorted) order."""
    return image_files(panels_dir)


# --- Orchestration ---------------------------------------------------------


def run(ctx: StageContext) -> StageResult:
    """Dedup, size-filter and upscale ``03_inpainted`` into ``04_clean``."""
    tag_component(COMPONENT)
    out = ctx.workspace.clean
    out.mkdir(parents=True, exist_ok=True)
    manual_review = ctx.workspace.manual_review
    manual_review.mkdir(parents=True, exist_ok=True)

    decoded: list[tuple[Path, np.ndarray]] = []
    for panel_path in iter_panels(ctx.workspace.inpainted):
        image = _decode_bgr(panel_path)
        if image is not None:
            decoded.append((panel_path, image))

    kept, dropped_dupes = dedup(
        [(path, phash(image)) for path, image in decoded], ctx.settings.dedup_hamming_distance
    )
    images = dict(decoded)

    dropped_small = 0
    upscaled = 0
    written = 0
    for path in kept:
        image = images[path]
        if short_side(image) < ctx.settings.min_side_px:
            route_to_manual(path, manual_review, "too small after inpaint")
            dropped_small += 1
            continue
        if ctx.settings.clean_denoise:
            image = denoise(image)
        if short_side(image) < ctx.settings.target_side:
            image = upscale_to(image, ctx.settings.target_side)
            upscaled += 1
        _write_png(image, out / f"{path.stem}.png")
        written += 1

    print(
        format_clean_report(
            kept=written,
            deduped=len(dropped_dupes),
            dropped_small=dropped_small,
            upscaled=upscaled,
        )
    )
    return StageResult(name=NAME, output_dir=out, produced=written)


STAGE = Stage(
    name=NAME,
    summary=SUMMARY,
    component=COMPONENT,
    flag="run_clean",
    output=lambda ws, _s: ws.clean,
    run=run,
)
