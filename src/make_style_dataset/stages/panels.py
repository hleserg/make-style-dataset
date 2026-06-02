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

from make_style_dataset.media import image_files, route_to_manual
from make_style_dataset.observability import tag_component
from make_style_dataset.stages.base import Stage, StageContext, StageResult

if TYPE_CHECKING:
    from collections.abc import Iterable

    from make_style_dataset.config import Settings

NAME = "panels"
SUMMARY = "Detect comic panels and slice pages into individual panel images."
COMPONENT = "stage:panels"

#: Grayscale level above which a pixel counts as gutter/background (Kumiko uses
#: the same idea): anything lighter is gutter, anything darker is panel ink.
GUTTER_THRESHOLD = 220

#: Recursive gutter-split (X-Y cut) of merged panel boxes. A row/column counts as
#: gutter when at least ``RESPLIT_GUTTER_FRAC`` of it is lighter than the gutter
#: threshold; a gutter band must be >= ``RESPLIT_MIN_GUTTER`` px to cut (thin ink
#: lines don't); split pieces below ``RESPLIT_MIN_SIDE`` px a side are ignored
#: (``filter_by_area`` drops the rest). Splitting only ever subdivides
#: coordinates — it never resizes, so crops keep their exact proportions.
RESPLIT_GUTTER_FRAC = 0.97
RESPLIT_MIN_GUTTER = 8
RESPLIT_MIN_SIDE = 32
RESPLIT_MAX_DEPTH = 4


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

    def __init__(self, *, gutter_threshold: int = GUTTER_THRESHOLD, resplit: bool = True) -> None:
        self._threshold = gutter_threshold
        self._resplit = resplit

    def detect(self, image_path: Path) -> list[Box]:
        """Detect panel boxes via OpenCV external-contour bounding rectangles.

        When ``resplit`` is on, each contour box is then X-Y-cut along any clean
        interior gutter bands, so touching / thin-gutter panels that merged into
        one contour are recovered as separate panels.
        """
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
        if not self._resplit:
            return boxes
        return [piece for box in boxes for piece in split_by_gutters(gray, box, self._threshold)]


def content_spans(is_gutter: list[bool], min_run: int, min_side: int) -> list[tuple[int, int]]:
    """Split an axis into content spans separated by gutter runs of ``>= min_run``.

    ``is_gutter[i]`` marks a fully-gutter row/column. Leading/trailing gutter is
    trimmed; a gutter run shorter than ``min_run`` is a thin ink line, not a cut,
    so the span continues across it. Spans shorter than ``min_side`` are dropped.
    Returns ``[(start, end), ...]`` half-open content ranges.
    """
    spans: list[tuple[int, int]] = []
    n = len(is_gutter)
    i = 0
    while i < n:
        if is_gutter[i]:
            i += 1
            continue
        start = i
        last_content = i
        while i < n:
            if not is_gutter[i]:
                last_content = i
                i += 1
                continue
            run_start = i
            while i < n and is_gutter[i]:
                i += 1
            if i - run_start >= min_run or i >= n:
                break  # a real gutter band (or the end) closes the span
            # else: a short run (thin line) — keep going within the same span
        if last_content + 1 - start >= min_side:
            spans.append((start, last_content + 1))
    return spans


def split_by_gutters(
    gray: object,
    box: Box,
    threshold: int,
    *,
    gutter_frac: float = RESPLIT_GUTTER_FRAC,
    min_run: int = RESPLIT_MIN_GUTTER,
    min_side: int = RESPLIT_MIN_SIDE,
    max_depth: int = RESPLIT_MAX_DEPTH,
) -> list[Box]:
    """Recursively X-Y-cut ``box`` along clean interior gutter bands of ``gray``.

    Pure geometry over the page's grayscale array: a row/column is gutter when
    ``>= gutter_frac`` of it is ``>= threshold`` (light); the box is split along
    gutter bands ``>= min_run`` px, recursing on the pieces (preferring the axis
    with more cuts). A box with no clean interior gutter returns ``[box]``
    unchanged. Only coordinates are subdivided — nothing is resized, so every
    resulting crop keeps its native proportions.
    """
    import numpy as np

    array = np.asarray(gray)

    def rec(x: int, y: int, w: int, h: int, depth: int) -> list[Box]:
        if depth >= max_depth or (h < 2 * min_side and w < 2 * min_side):
            return [Box(x, y, w, h)]
        region = array[y : y + h, x : x + w]
        light = region >= threshold
        row_gutter = (light.mean(axis=1) >= gutter_frac).tolist()
        col_gutter = (light.mean(axis=0) >= gutter_frac).tolist()
        rows = content_spans(row_gutter, min_run, min_side)
        cols = content_spans(col_gutter, min_run, min_side)
        if len(rows) > 1 and len(rows) >= len(cols):
            return [piece for s, e in rows for piece in rec(x, y + s, w, e - s, depth + 1)]
        if len(cols) > 1:
            return [piece for s, e in cols for piece in rec(x + s, y, e - s, h, depth + 1)]
        return [Box(x, y, w, h)]

    return rec(box.x, box.y, box.w, box.h, 0)


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
    return image_files(pages_dir)


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
            route_to_manual(page_path, manual_review, reason)
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

    detector = ContourPanelDetector(resplit=ctx.settings.panel_resplit)
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
