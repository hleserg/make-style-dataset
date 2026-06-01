"""Integration tests for the real OpenCV contour panel detector.

We render synthetic "pages" — dark panel rectangles on a white gutter
background, the layout Kumiko's threshold step assumes — at a few complexity
levels (grid, splash, over-segmented) and run the actual ``ContourPanelDetector``
plus the slicing stage end-to-end. No copyrighted comic art is committed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from make_style_dataset.config import Settings
from make_style_dataset.stages import panels
from make_style_dataset.stages.base import StageContext
from make_style_dataset.stages.panels import Box, ContourPanelDetector
from make_style_dataset.workspace import Workspace

PANEL_FILL = (40, 40, 40)  # darker than the gutter threshold -> detected as ink


def _grid_page(path: Path, *, cols: int, rows: int, size: int = 1000, gutter: int = 40) -> Path:
    """Render a white page with a ``cols`` x ``rows`` grid of dark panels."""
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    cell_w = (size - gutter * (cols + 1)) // cols
    cell_h = (size - gutter * (rows + 1)) // rows
    for row in range(rows):
        for col in range(cols):
            x0 = gutter + col * (cell_w + gutter)
            y0 = gutter + row * (cell_h + gutter)
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill=PANEL_FILL)
    image.save(path)
    return path


def _splash_page(path: Path, *, size: int = 1000) -> Path:
    """Render a single near-full-page panel (a splash)."""
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, size - 20, size - 20], fill=PANEL_FILL)
    image.save(path)
    return path


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {"workspace": tmp_path / "ws"}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.mark.integration
def test_detector_finds_grid_panels(tmp_path: Path) -> None:
    page = _grid_page(tmp_path / "grid.png", cols=2, rows=2)
    boxes = ContourPanelDetector().detect(page)
    assert len(boxes) == 4
    assert all(isinstance(b, Box) and b.area > 0 for b in boxes)


@pytest.mark.integration
def test_detector_returns_empty_for_unreadable(tmp_path: Path) -> None:
    bogus = tmp_path / "not-an-image.png"
    bogus.write_text("definitely not a PNG", encoding="utf-8")
    assert ContourPanelDetector().detect(bogus) == []


@pytest.mark.integration
def test_detector_returns_empty_for_zero_byte_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.png"
    empty.touch()
    assert ContourPanelDetector().detect(empty) == []


@pytest.mark.integration
def test_run_slices_grid_and_routes_edge_cases(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    _grid_page(ws.pages / "grid2x2.png", cols=2, rows=2)  # 4 panels -> sliced
    _grid_page(ws.pages / "grid3x2.png", cols=3, rows=2)  # 6 panels -> sliced
    _splash_page(ws.pages / "splash.png")  # splash -> manual_review
    _grid_page(ws.pages / "busy.png", cols=4, rows=4)  # 16 > max_panels -> manual_review

    result = panels.run(StageContext(workspace=ws, settings=settings))

    assert result.produced == 10  # 4 + 6 sliced panels
    sliced = sorted(p.name for p in ws.panels.iterdir())
    assert sliced.count("grid2x2_00.png") == 1
    assert any(name.startswith("grid3x2_") for name in sliced)
    # Splash and over-segmented pages went to manual review, not 01_panels.
    assert (ws.manual_review / "splash.png").is_file()
    assert "splash" in (ws.manual_review / "splash.reason.txt").read_text(encoding="utf-8")
    assert (ws.manual_review / "busy.png").is_file()
    assert not any(name.startswith(("splash_", "busy_")) for name in sliced)
