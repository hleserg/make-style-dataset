"""Stage 2 — speech-bubble and SFX/text detection, emitting removal masks.

Reads panels from ``01_panels`` and writes one binary mask per panel to
``02_masks`` (white = pixels to remove later by inpainting). Two detectors feed
the mask: a YOLOv8 **segmentation** model contributes tight bubble polygons, and
EasyOCR's detector contributes boxes for SFX / drawn text that lives outside
balloons. The two are unioned, dilated a little to swallow bubble outlines and
letter strokes, and written 1:1 with the panel (``<panel-stem>.png``) so the
inpaint stage can pair each panel with its mask. Panels with no text get an
empty (all-black) mask and pass straight through.

The heavy backends (ultralytics, easyocr) are imported lazily inside the
adapters that need them, so the pure mask geometry/policy helpers stay
importable and unit-testable without torch installed. Detection is injected via
the :class:`BubbleDetector` and :class:`TextDetector` protocols, so tests swap a
fake and a different backend can be substituted later. This reuses the
``pure-core-lazy-backend`` pattern documented in ``panels.py``.

A panel whose mask covers more than ``max_mask_coverage`` of its area is treated
as a mis-detection (inpainting it would gut the panel) and routed whole to
``manual_review`` instead, mirroring the splash/over-segmentation fallback in
``panels.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from make_style_dataset.observability import tag_component
from make_style_dataset.stages.base import Stage, StageContext, StageResult

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy as np

    from make_style_dataset.config import Settings

NAME = "bubbles"
SUMMARY = "Detect speech bubbles and SFX/text in panels and write removal masks."
COMPONENT = "stage:bubbles"

#: Panel formats we can read from ``01_panels`` (matches the panels stage).
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})

#: Weight file to pull when ``bubble_model`` is a Hugging Face repo id.
BUBBLE_WEIGHTS_FILE = "model.pt"

#: Pinned commit of the default ``bubble_model`` repo. Pinning the revision keeps
#: downloads reproducible and guards against the repo changing under us (a
#: pointer-to-default model would otherwise be fetched at HEAD). Override this
#: alongside ``bubble_model`` when pointing at a different repo.
BUBBLE_WEIGHTS_REVISION = "da4efccf35a15c8a8c2564431a4b7e121d3e0d99"

#: A polygon is a list of integer (x, y) points; a box is (x1, y1, x2, y2).
Point = tuple[int, int]
Polygon = list[Point]
BBox = tuple[int, int, int, int]


class BubbleDetector(Protocol):
    """Segments speech balloons in a panel (YOLOv8-seg's role)."""

    def detect(self, image: np.ndarray) -> list[Polygon]:
        """Return bubble polygons (original-image pixel coords) for ``image`` (BGR)."""
        ...


class TextDetector(Protocol):
    """Detects SFX / drawn-text regions in a panel (OCR's role)."""

    def detect(self, image: np.ndarray) -> list[BBox]:
        """Return axis-aligned text boxes ``(x1, y1, x2, y2)`` for ``image`` (BGR)."""
        ...


# PLAYBOOK-START
# id: pure-core-lazy-backend
# title: Pure policy core behind a lazily-imported heavy backend
# status: draft
# category: testability
# tags: [testing, dependency-injection, coverage]
# Split a stage into pure functions over plain data (mask geometry, coverage
# thresholds, routing decisions) and thin adapters around heavy/optional
# dependencies (here ultralytics + easyocr, both pulling torch) that are
# imported lazily and hidden behind a Protocol. The policy is unit-tested
# without the heavy deps; the adapters are injected, so tests swap fakes.
# Generalizes to any stage wrapping a costly backend. Also used by panels.py.
# PLAYBOOK-END
class YoloBubbleDetector:
    """YOLOv8-seg speech-bubble detector; weights load lazily on first use."""

    def __init__(self, model: str, *, confidence: float, device: str | None = None) -> None:
        self._model_ref = model
        self._confidence = confidence
        self._device = device
        self._yolo: object | None = None

    def _weights(self) -> str:  # pragma: no cover - needs huggingface_hub + network
        """Resolve a local .pt path, downloading from the Hub if ``model`` is a repo id."""
        local = Path(self._model_ref)
        if local.exists():
            return str(local)
        from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]

        return hf_hub_download(
            self._model_ref, filename=BUBBLE_WEIGHTS_FILE, revision=BUBBLE_WEIGHTS_REVISION
        )

    def detect(self, image: np.ndarray) -> list[Polygon]:  # pragma: no cover - needs ultralytics
        """Run YOLOv8-seg and return one polygon per detected bubble instance."""
        from ultralytics import YOLO  # pyright: ignore[reportMissingImports]

        if self._yolo is None:
            self._yolo = YOLO(self._weights())
        results = self._yolo.predict(  # type: ignore[attr-defined]
            image, conf=self._confidence, device=self._device, verbose=False
        )
        polygons: list[Polygon] = []
        for result in results:
            masks = result.masks
            if masks is None:
                continue
            # ``masks.xy`` is already scaled to the original image resolution.
            for poly in masks.xy:
                polygons.append([(int(x), int(y)) for x, y in poly])
        return polygons


class OcrTextDetector:
    """EasyOCR detection-only text/SFX detector; the reader loads lazily once."""

    def __init__(self, languages: Sequence[str], *, gpu: bool = True) -> None:
        self._languages = list(languages)
        self._gpu = gpu
        self._reader: object | None = None

    def detect(self, image: np.ndarray) -> list[BBox]:  # pragma: no cover - needs easyocr
        """Return text/SFX boxes via EasyOCR's CRAFT detector (no recognition)."""
        import easyocr  # pyright: ignore[reportMissingImports]

        if self._reader is None:
            # recognizer=False: detect() only uses the detector, so the heavy
            # recognition model is never downloaded/loaded. gpu falls back to
            # CPU with a warning when CUDA is unavailable.
            self._reader = easyocr.Reader(self._languages, gpu=self._gpu, recognizer=False)

        # detect() returns lists of depth 3 (one inner list per input image), so
        # index [0] for our single image.
        horizontal_list, free_list = self._reader.detect(image)  # type: ignore[attr-defined]
        boxes: list[BBox] = []
        # Horizontal boxes are [x_min, x_max, y_min, y_max] — NOT (x1,y1,x2,y2).
        for x_min, x_max, y_min, y_max in horizontal_list[0]:
            boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))
        # Free (rotated/slanted) boxes are 4 corner points; collapse to a bbox so
        # slanted SFX is not silently dropped.
        for quad in free_list[0]:
            xs = [int(p[0]) for p in quad]
            ys = [int(p[1]) for p in quad]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        return boxes


# --- Pure mask geometry / policy ------------------------------------------


def boxes_to_mask(boxes: Iterable[BBox], width: int, height: int) -> np.ndarray:
    """Rasterize ``(x1, y1, x2, y2)`` boxes onto a binary mask (255 = covered)."""
    import numpy as np

    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        left, right = sorted((int(x1), int(x2)))
        top, bottom = sorted((int(y1), int(y2)))
        left, top = max(left, 0), max(top, 0)
        right, bottom = min(right, width), min(bottom, height)
        if right > left and bottom > top:
            mask[top:bottom, left:right] = 255
    return mask


def polygons_to_mask(polygons: Iterable[Polygon], width: int, height: int) -> np.ndarray:
    """Rasterize filled polygons onto a binary mask (255 = covered)."""
    import cv2
    import numpy as np

    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(list(polygon), dtype=np.int32)
        if points.shape[0] >= 3:
            cv2.fillPoly(mask, [points], 255)
    return mask


def union_masks(masks: Iterable[np.ndarray], width: int, height: int) -> np.ndarray:
    """Combine binary masks with a pixelwise OR (robust to an empty input)."""
    import numpy as np

    out = np.zeros((height, width), dtype=np.uint8)
    for mask in masks:
        out = np.maximum(out, mask)
    return out


def dilate_mask(mask: np.ndarray, dilation_px: int) -> np.ndarray:
    """Grow a binary mask by ``dilation_px`` to swallow outlines/letter strokes."""
    import cv2

    if dilation_px <= 0:
        return mask.copy()
    size = 2 * dilation_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask, kernel, iterations=1)


def mask_coverage_ratio(mask: np.ndarray) -> float:
    """Return the fraction of the mask that is set (0.0 when the mask is empty)."""
    import numpy as np

    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def mask_is_empty(mask: np.ndarray) -> bool:
    """Return ``True`` when no pixel is set (a clean, text-free panel)."""
    import numpy as np

    return int(np.count_nonzero(mask)) == 0


def classify_mask(mask: np.ndarray, *, max_coverage: float) -> str | None:
    """Return a manual-review reason when the mask covers too much, else ``None``.

    A mask covering more than ``max_coverage`` of the panel signals a runaway
    detection: inpainting it would destroy the panel, so a human should look.
    """
    coverage = mask_coverage_ratio(mask)
    if coverage > max_coverage:
        return f"excessive mask coverage ({coverage:.0%} > {max_coverage:.0%})"
    return None


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tint masked pixels red over the panel for a quick visual debug check."""
    import numpy as np

    overlay = image.copy()
    selected = mask > 0
    # image is BGR; paint the masked region pure red (BGR = 0,0,255), blended.
    overlay[selected] = (0.4 * overlay[selected] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    return overlay


# --- Orchestration --------------------------------------------------------


def iter_panels(panels_dir: Path) -> list[Path]:
    """Return panel image files under ``panels_dir`` in stable (sorted) order."""
    if not panels_dir.is_dir():
        return []
    return sorted(
        path
        for path in panels_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _decode_bgr(panel_path: Path) -> np.ndarray | None:
    """Decode a panel to a BGR array via a byte buffer (OS-agnostic, non-ASCII safe)."""
    import cv2
    import numpy as np

    buffer = np.fromfile(panel_path, dtype=np.uint8)
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _write_png(array: np.ndarray, path: Path) -> None:
    """Save a uint8 array as PNG via PIL (never ``cv2.imwrite`` on a path)."""
    from PIL import Image

    mode = "L" if array.ndim == 2 else "RGB"
    pixels = array if array.ndim == 2 else array[:, :, ::-1]  # BGR -> RGB for color
    Image.fromarray(pixels, mode=mode).save(path)


def _route_to_manual(panel_path: Path, manual_review: Path, reason: str) -> None:
    """Copy a panel whole into ``manual_review`` with a reason note."""
    import shutil

    shutil.copy2(panel_path, manual_review / panel_path.name)
    note = manual_review / f"{panel_path.stem}.reason.txt"
    note.write_text(f"{reason}\n", encoding="utf-8")


def mask_panel(
    panel_path: Path,
    *,
    bubble_detector: BubbleDetector,
    text_detector: TextDetector,
    settings: Settings,
    out_dir: Path,
    manual_review: Path,
) -> bool:
    """Build and write one panel's removal mask, or route the panel to manual review.

    Returns ``True`` when a mask was written, ``False`` when the panel was routed
    out (unreadable, or mask coverage above ``max_mask_coverage``). The mask is
    named ``<panel-stem>.png`` so a re-run overwrites rather than duplicates and
    each mask pairs 1:1 with its panel.
    """
    image = _decode_bgr(panel_path)
    if image is None:
        _route_to_manual(panel_path, manual_review, "unreadable panel")
        return False

    height, width = image.shape[:2]
    bubble_mask = polygons_to_mask(bubble_detector.detect(image), width, height)
    text_mask = boxes_to_mask(text_detector.detect(image), width, height)
    mask = dilate_mask(
        union_masks([bubble_mask, text_mask], width, height), settings.mask_dilation_px
    )

    reason = classify_mask(mask, max_coverage=settings.max_mask_coverage)
    if reason is not None:
        _route_to_manual(panel_path, manual_review, reason)
        return False

    _write_png(mask, out_dir / f"{panel_path.stem}.png")
    if settings.bubbles_debug:
        _write_png(overlay_mask(image, mask), manual_review / f"{panel_path.stem}.overlay.png")
    return True


def run(ctx: StageContext) -> StageResult:
    """Write a bubble+text removal mask for every panel in ``01_panels``."""
    tag_component(COMPONENT)
    out = ctx.workspace.masks
    out.mkdir(parents=True, exist_ok=True)
    manual_review = ctx.workspace.manual_review
    manual_review.mkdir(parents=True, exist_ok=True)

    bubble_detector = YoloBubbleDetector(
        ctx.settings.bubble_model, confidence=ctx.settings.bubble_confidence
    )
    languages = [code.strip() for code in ctx.settings.ocr_languages.split(",") if code.strip()]
    text_detector = OcrTextDetector(languages)

    produced = 0
    for panel_path in iter_panels(ctx.workspace.panels):
        if mask_panel(
            panel_path,
            bubble_detector=bubble_detector,
            text_detector=text_detector,
            settings=ctx.settings,
            out_dir=out,
            manual_review=manual_review,
        ):
            produced += 1
    return StageResult(name=NAME, output_dir=out, produced=produced)


STAGE = Stage(
    name=NAME,
    summary=SUMMARY,
    component=COMPONENT,
    flag="run_bubbles",
    output=lambda ws, _s: ws.masks,
    run=run,
)
