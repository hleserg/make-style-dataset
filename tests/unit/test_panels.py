"""Unit tests for the panel-detection/slicing stage.

These exercise the pure geometry/policy helpers and the slicing orchestration
with a fake detector, so no OpenCV is needed here (see the integration test for
the real contour detector).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from make_style_dataset.config import Settings
from make_style_dataset.stages import panels
from make_style_dataset.stages.base import StageContext
from make_style_dataset.stages.panels import Box
from make_style_dataset.workspace import Workspace


class FakeDetector:
    """Returns a fixed list of boxes regardless of the page (stands in for Kumiko)."""

    def __init__(self, boxes: list[Box]) -> None:
        self._boxes = boxes

    def detect(self, image_path: Path) -> list[Box]:
        del image_path  # path is unused: the fake ignores the page content
        return list(self._boxes)


def _white_page(path: Path, width: int = 400, height: int = 400) -> Path:
    Image.new("RGB", (width, height), "white").save(path)
    return path


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "workspace": tmp_path / "ws",
        "panel_border": 0,
        "min_panel_area": 100,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# --- geometry -------------------------------------------------------------


def test_box_area() -> None:
    assert Box(0, 0, 10, 20).area == 200


def test_shrink_box_trims_each_side() -> None:
    shrunk = panels.shrink_box(Box(10, 10, 100, 100), border=5, page_w=400, page_h=400)
    assert shrunk == Box(15, 15, 90, 90)


def test_shrink_box_clamps_to_page_bounds() -> None:
    # Box runs off the right/bottom edge; it should be clamped to the page.
    shrunk = panels.shrink_box(Box(-10, -10, 500, 500), border=0, page_w=400, page_h=400)
    assert shrunk == Box(0, 0, 400, 400)


def test_shrink_box_collapses_to_none() -> None:
    assert panels.shrink_box(Box(0, 0, 10, 10), border=8, page_w=400, page_h=400) is None


def test_filter_by_area_drops_micro_panels() -> None:
    big, small = Box(0, 0, 100, 100), Box(0, 0, 5, 5)
    assert panels.filter_by_area([big, small], min_area=100) == [big]


def test_reading_order_sorts_top_then_left() -> None:
    a, b, c = Box(200, 10, 50, 50), Box(10, 10, 50, 50), Box(10, 200, 50, 50)
    assert panels.reading_order([a, b, c]) == [b, a, c]


# --- fallback classification ---------------------------------------------


def test_classify_empty_page() -> None:
    assert panels.classify_page([], 1000, max_panels=12, splash_ratio=0.85) is not None


def test_classify_too_many_panels() -> None:
    boxes = [Box(i, 0, 10, 10) for i in range(13)]
    reason = panels.classify_page(boxes, 1_000_000, max_panels=12, splash_ratio=0.85)
    assert reason is not None
    assert "13" in reason


def test_classify_splash_page() -> None:
    boxes = [Box(0, 0, 400, 400)]
    reason = panels.classify_page(boxes, 400 * 400, max_panels=12, splash_ratio=0.85)
    assert reason is not None
    assert "splash" in reason


def test_classify_normal_page_passes() -> None:
    boxes = [Box(0, 0, 100, 100), Box(200, 0, 100, 100)]
    assert panels.classify_page(boxes, 400 * 400, max_panels=12, splash_ratio=0.85) is None


# --- iter_pages -----------------------------------------------------------


def test_iter_pages_filters_and_sorts(tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    _white_page(pages / "b.png")
    _white_page(pages / "a.jpg")
    (pages / "notes.txt").write_text("ignore me", encoding="utf-8")
    found = panels.iter_pages(pages)
    assert [p.name for p in found] == ["a.jpg", "b.png"]


def test_iter_pages_missing_dir(tmp_path: Path) -> None:
    assert panels.iter_pages(tmp_path / "does-not-exist") == []


# --- slice_page -----------------------------------------------------------


def test_slice_page_writes_named_panels(tmp_path: Path) -> None:
    page = _white_page(tmp_path / "page01.png")
    out, manual = tmp_path / "out", tmp_path / "manual"
    out.mkdir()
    manual.mkdir()
    detector = FakeDetector([Box(10, 10, 100, 100), Box(200, 10, 80, 80)])
    written = panels.slice_page(
        page, detector=detector, settings=_settings(tmp_path), out_dir=out, manual_review=manual
    )
    assert written == 2
    assert sorted(p.name for p in out.iterdir()) == ["page01_00.png", "page01_01.png"]
    with Image.open(out / "page01_00.png") as crop:
        assert crop.size == (100, 100)


def test_slice_page_splash_routes_to_manual(tmp_path: Path) -> None:
    page = _white_page(tmp_path / "splash.png")
    out, manual = tmp_path / "out", tmp_path / "manual"
    out.mkdir()
    manual.mkdir()
    detector = FakeDetector([Box(0, 0, 400, 400)])
    written = panels.slice_page(
        page, detector=detector, settings=_settings(tmp_path), out_dir=out, manual_review=manual
    )
    assert written == 0
    assert list(out.iterdir()) == []
    assert (manual / "splash.png").is_file()
    assert "splash" in (manual / "splash.reason.txt").read_text(encoding="utf-8")


def test_slice_page_too_many_routes_to_manual(tmp_path: Path) -> None:
    page = _white_page(tmp_path / "busy.png")
    out, manual = tmp_path / "out", tmp_path / "manual"
    out.mkdir()
    manual.mkdir()
    detector = FakeDetector([Box(i * 12, 0, 10, 10) for i in range(13)])
    written = panels.slice_page(
        page,
        detector=detector,
        settings=_settings(tmp_path, min_panel_area=1),
        out_dir=out,
        manual_review=manual,
    )
    assert written == 0
    assert (manual / "busy.png").is_file()


# --- run() orchestration --------------------------------------------------


def test_run_slices_all_pages(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    _white_page(ws.pages / "p1.png")
    _white_page(ws.pages / "p2.png")
    monkeypatch.setattr(
        panels, "ContourPanelDetector", lambda **_kwargs: FakeDetector([Box(10, 10, 100, 100)])
    )
    ctx = StageContext(workspace=ws, settings=settings)

    result = panels.run(ctx)
    assert result.produced == 2
    assert sorted(p.name for p in ws.panels.iterdir()) == ["p1_00.png", "p2_00.png"]

    # Idempotent re-run: same deterministic names, no duplicates.
    again = panels.run(ctx)
    assert again.produced == 2
    assert len(list(ws.panels.iterdir())) == 2


# --- recursive gutter split (X-Y cut of merged panel boxes) ----------------


def test_content_spans_cuts_on_gutter_band_not_thin_line() -> None:
    # content 0-1, gutter band 2-4 (run 3), content 5-6
    assert panels.content_spans([False, False, True, True, True, False, False], 2, 1) == [
        (0, 2),
        (5, 7),
    ]
    # a single gutter row (run 1) is a thin line, not a cut -> one span
    assert panels.content_spans([False, False, True, False, False], 2, 1) == [(0, 5)]
    # spans shorter than min_side are dropped
    assert panels.content_spans([False, True, True, False, False, False], 2, 2) == [(3, 6)]


def test_split_by_gutters_splits_a_2x2_grid_without_resizing() -> None:
    import numpy as np

    page = np.full((100, 100), 255, dtype=np.uint8)  # all light = gutter
    for y, x in [(0, 0), (0, 60), (60, 0), (60, 60)]:
        page[y : y + 40, x : x + 40] = 0  # four 40x40 dark panels, 20px gutter cross
    boxes = panels.split_by_gutters(page, Box(0, 0, 100, 100), 220, min_side=10, min_run=5)
    assert len(boxes) == 4
    # every piece keeps the native 40x40 proportions — coordinates only, no stretch
    assert all((b.w, b.h) == (40, 40) for b in boxes)
    assert {(b.x, b.y) for b in boxes} == {(0, 0), (60, 0), (0, 60), (60, 60)}


def test_split_by_gutters_keeps_a_single_panel_whole() -> None:
    import numpy as np

    page = np.zeros((50, 50), dtype=np.uint8)  # all dark = one panel, no interior gutter
    assert panels.split_by_gutters(page, Box(0, 0, 50, 50), 220) == [Box(0, 0, 50, 50)]
