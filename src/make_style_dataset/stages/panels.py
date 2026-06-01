"""Stage 1 — panel detection and slicing.

Reads raw pages from ``00_pages`` and writes one image per detected panel to
``01_panels``. Panel coordinates come from a Kumiko-style OpenCV contour
detector (threshold the gutters, take the bounding box of each ink region);
the page is then cropped with Pillow. Frame/gutter pixels are trimmed
(``panel_border``), micro-panels below ``min_panel_area`` are dropped, and
pages that look mis-segmented — a single splash panel covering the page, or
more than ``max_panels`` panels — are copied whole into ``manual_review`` for a
human instead of being auto-sliced.

The heavy backends (OpenCV, Pillow) are imported lazily inside the functions
that need them so the pure geometry/policy helpers stay importable and testable
on their own. The detector is injected via the :class:`PanelDetector` protocol,
so a different backend (e.g. the real Kumiko) can be swapped in later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from make_style_dataset.observability import tag_component
from make_style_dataset.stages.base import Stage, StageContext, StageResult

if TYPE_CHECKING:
    from collections.abc import Iterable

    from make_style_dataset.config import Settings

NAME = "panels"
SUMMARY = "Detect comic panels and slice pages into individual panel images."
COMPONENT = "stage:panels"

#: Page formats Pillow/OpenCV will read from ``00_pages``.
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})

#: Grayscale level above which a pixel counts as gutter/background (Kumiko uses
#: the same idea): anything lighter is gutter, anything darker is panel ink.
GUTTER_THRESHOLD = 220


@dataclass(frozen=True)
class Box:
    """An axis-aligned panel bounding box in page pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        """Pixel area of the box."""
        return self.w * self.h


class PanelDetector(Protocol):
    """Maps a page image to candidate panel boxes (Kumiko's role)."""

    def detect(self, image_path: Path) -> list[Box]:
        """Return raw panel bounding boxes for the page at ``image_path``."""
        ...


# PLAYBOOK-START
# id: pure-core-lazy-backend
# title: Pure policy core behind a lazily-imported heavy backend
# status: draft
# category: testability
# tags: [testing, dependency-injection, coverage]
# Split a stage into pure functions over plain data (geometry, thresholds,
# routing decisions) and a thin adapter around a heavy/optional dependency
# (here OpenCV) that is imported lazily and hidden behind a Protocol. The
# policy is unit-tested without the heavy dep; the adapter is injected, so
# tests swap a fake. Generalizes to any stage wrapping a costly backend.
# PLAYBOOK-END
class ContourPanelDetector:
    """Kumiko-style detector: threshold gutters, take ink-region bounding boxes."""

    def __init__(self, *, gutter_threshold: int = GUTTER_THRESHOLD) -> None:
        self._threshold = gutter_threshold

    def detect(self, image_path: Path) -> list[Box]:
        """Detect panel boxes via OpenCV external-contour bounding rectangles."""
        import cv2
        import numpy as np

        # Decode from a byte buffer rather than cv2.imread(path): the latter
        # fails on non-ASCII paths on Windows, while np.fromfile handles them.
        buffer = np.fromfile(image_path, dtype=np.uint8)
        if buffer.size == 0:
            return []
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:  # unreadable/corrupt page
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self._threshold, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[Box] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append(Box(int(x), int(y), int(w), int(h)))
        return boxes


def shrink_box(box: Box, border: int, page_w: int, page_h: int) -> Box | None:
    """Clamp ``box`` to the page and trim ``border`` px off each side.

    Returns ``None`` if trimming collapses the box to nothing.
    """
    left = max(box.x, 0) + border
    top = max(box.y, 0) + border
    right = min(box.x + box.w, page_w) - border
    bottom = min(box.y + box.h, page_h) - border
    if right <= left or bottom <= top:
        return None
    return Box(left, top, right - left, bottom - top)


def filter_by_area(boxes: Iterable[Box], min_area: int) -> list[Box]:
    """Drop boxes smaller than ``min_area`` (micro-panels and detection noise)."""
    return [box for box in boxes if box.area >= min_area]


def reading_order(boxes: Iterable[Box]) -> list[Box]:
    """Sort boxes top-to-bottom then left-to-right for stable panel indices."""
    return sorted(boxes, key=lambda b: (b.y, b.x))


def classify_page(
    boxes: list[Box],
    page_area: int,
    *,
    max_panels: int,
    splash_ratio: float,
) -> str | None:
    """Return a manual-review reason for a mis-segmented page, else ``None``.

    A page is kicked out when nothing was found, when there are more panels
    than ``max_panels`` (segmentation clearly broke), or when a single panel
    covers at least ``splash_ratio`` of the page (a splash that should not be
    auto-sliced).
    """
    count = len(boxes)
    if count == 0:
        return "no panels detected"
    if count > max_panels:
        return f"too many panels ({count} > {max_panels})"
    if count == 1 and page_area > 0 and boxes[0].area >= splash_ratio * page_area:
        return "splash page (single full-page panel)"
    return None


def panel_boxes(raw: Iterable[Box], page_w: int, page_h: int, settings: Settings) -> list[Box]:
    """Turn raw detector boxes into trimmed, area-filtered, ordered panels."""
    trimmed = [
        shrunk
        for box in raw
        if (shrunk := shrink_box(box, settings.panel_border, page_w, page_h)) is not None
    ]
    return reading_order(filter_by_area(trimmed, settings.min_panel_area))


def iter_pages(pages_dir: Path) -> list[Path]:
    """Return page image files under ``pages_dir`` in stable (sorted) order."""
    if not pages_dir.is_dir():
        return []
    return sorted(
        path
        for path in pages_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _route_to_manual(page_path: Path, manual_review: Path, reason: str) -> None:
    """Copy a mis-segmented page whole into ``manual_review`` with a reason note."""
    import shutil

    shutil.copy2(page_path, manual_review / page_path.name)
    note = manual_review / f"{page_path.stem}.reason.txt"
    note.write_text(f"{reason}\n", encoding="utf-8")


def slice_page(
    page_path: Path,
    *,
    detector: PanelDetector,
    settings: Settings,
    out_dir: Path,
    manual_review: Path,
) -> int:
    """Slice one page into panels, or route it to ``manual_review``.

    Returns the number of panel images written (0 when the page is kicked out).
    Output names are deterministic — ``<page-stem>_<idx>.png`` — so a re-run
    overwrites rather than duplicates, and each panel traces back to its page
    and index.
    """
    from PIL import Image

    with Image.open(page_path) as image:
        page_w, page_h = image.size
        rgb = image.convert("RGB")
        boxes = panel_boxes(detector.detect(page_path), page_w, page_h, settings)
        reason = classify_page(
            boxes,
            page_w * page_h,
            max_panels=settings.max_panels,
            splash_ratio=settings.splash_area_ratio,
        )
        if reason is not None:
            _route_to_manual(page_path, manual_review, reason)
            return 0
        stem = page_path.stem
        for idx, box in enumerate(boxes):
            crop = rgb.crop((box.x, box.y, box.x + box.w, box.y + box.h))
            crop.save(out_dir / f"{stem}_{idx:02d}.png")
        return len(boxes)


def run(ctx: StageContext) -> StageResult:
    """Slice every page in ``00_pages`` into ``01_panels`` (Kumiko + OpenCV)."""
    tag_component(COMPONENT)
    out = ctx.workspace.panels
    out.mkdir(parents=True, exist_ok=True)
    manual_review = ctx.workspace.manual_review
    manual_review.mkdir(parents=True, exist_ok=True)

    detector = ContourPanelDetector()
    produced = 0
    for page_path in iter_pages(ctx.workspace.pages):
        produced += slice_page(
            page_path,
            detector=detector,
            settings=ctx.settings,
            out_dir=out,
            manual_review=manual_review,
        )
    return StageResult(name=NAME, output_dir=out, produced=produced)


STAGE = Stage(
    name=NAME,
    summary=SUMMARY,
    component=COMPONENT,
    flag="run_panels",
    output=lambda ws, _s: ws.panels,
    run=run,
)
