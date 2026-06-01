"""Integration tests for the bubble/SFX masking stage.

The mask-building pipeline (decode -> rasterize polygons/boxes -> union ->
dilate -> coverage policy -> write) is exercised end-to-end on a realistic
multi-panel workspace, with the heavy YOLOv8/EasyOCR backends replaced by fakes
(cv2/Pillow are real). A separate, ``importorskip``-gated test runs the real
detectors when ultralytics/easyocr happen to be installed; it is excluded from
coverage and skipped in CI, which never installs the GPU dependency group.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from make_style_dataset.config import Settings
from make_style_dataset.stages import bubbles
from make_style_dataset.stages.base import StageContext
from make_style_dataset.stages.bubbles import BBox, Polygon
from make_style_dataset.workspace import Workspace


class FakeBubbleDetector:
    """Returns fixed bubble polygons (a stand-in for YOLOv8-seg)."""

    def __init__(self, polygons: list[Polygon]) -> None:
        self._polygons = polygons

    def detect(self, image: np.ndarray) -> list[Polygon]:
        del image
        return [list(p) for p in self._polygons]


class FakeTextDetector:
    """Returns fixed SFX/text boxes (a stand-in for EasyOCR)."""

    def __init__(self, boxes: list[BBox]) -> None:
        self._boxes = boxes

    def detect(self, image: np.ndarray) -> list[BBox]:
        del image
        return list(self._boxes)


def _panel(path: Path, size: int = 256) -> Path:
    Image.new("RGB", (size, size), "white").save(path)
    return path


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {"workspace": tmp_path / "ws", "mask_dilation_px": 3}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.mark.integration
def test_bubble_sfx_and_clean_panels_end_to_end(tmp_path: Path) -> None:
    """A bubble panel, an SFX panel, and a clean panel each get a 1:1 mask."""
    settings = _settings(tmp_path)
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.panels.mkdir(parents=True, exist_ok=True)
    out, manual = ws.masks, ws.manual_review
    out.mkdir(parents=True, exist_ok=True)

    bubble_polys: list[Polygon] = [[(30, 30), (120, 30), (120, 110), (30, 110)]]
    sfx_boxes: list[BBox] = [(160, 160, 230, 200)]

    cases = {
        "bubble.png": (FakeBubbleDetector(bubble_polys), FakeTextDetector([])),
        "sfx.png": (FakeBubbleDetector([]), FakeTextDetector(sfx_boxes)),
        "clean.png": (FakeBubbleDetector([]), FakeTextDetector([])),
    }
    for name, (bubble, text) in cases.items():
        _panel(ws.panels / name)
        assert bubbles.mask_panel(
            ws.panels / name,
            bubble_detector=bubble,
            text_detector=text,
            settings=settings,
            out_dir=out,
            manual_review=manual,
        )

    with Image.open(out / "bubble.png") as img:
        bubble_mask = np.array(img)
    with Image.open(out / "sfx.png") as img:
        sfx_mask = np.array(img)
    with Image.open(out / "clean.png") as img:
        clean_mask = np.array(img)

    assert bubble_mask[70, 70] == 255 and not bubbles.mask_is_empty(bubble_mask)
    assert sfx_mask[180, 195] == 255 and not bubbles.mask_is_empty(sfx_mask)
    assert bubbles.mask_is_empty(clean_mask)
    # 1:1 contract: one mask per panel, matching basenames.
    assert sorted(p.name for p in out.iterdir() if p.suffix == ".png") == [
        "bubble.png",
        "clean.png",
        "sfx.png",
    ]


@pytest.mark.integration
def test_run_writes_one_mask_per_panel(tmp_path: Path, monkeypatch) -> None:
    """run() over a workspace writes exactly one mask per input panel."""
    settings = _settings(tmp_path)
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.panels.mkdir(parents=True, exist_ok=True)
    for name in ("a.png", "b.png", "c.png"):
        _panel(ws.panels / name)

    monkeypatch.setattr(bubbles, "YoloBubbleDetector", lambda *a, **k: FakeBubbleDetector([]))
    monkeypatch.setattr(bubbles, "OcrTextDetector", lambda *a, **k: FakeTextDetector([]))

    result = bubbles.run(StageContext(workspace=ws, settings=settings))
    assert result.produced == 3
    assert sorted(p.name for p in ws.masks.iterdir()) == ["a.png", "b.png", "c.png"]


@pytest.mark.integration
def test_real_detectors_smoke(tmp_path: Path) -> None:  # pragma: no cover - needs GPU deps
    """Run the real YOLOv8 + EasyOCR adapters if (and only if) they are installed.

    Skipped in CI and the default DoD env, which never install the ``gpu`` group.
    Downloads model weights on first run, so it is excluded from coverage.
    """
    pytest.importorskip("ultralytics")
    pytest.importorskip("easyocr")

    settings = Settings(workspace=tmp_path / "ws")  # type: ignore[arg-type]
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.panels.mkdir(parents=True, exist_ok=True)
    _panel(ws.panels / "real.png")

    result = bubbles.run(StageContext(workspace=ws, settings=settings))
    assert result.produced >= 0
    if result.produced:
        assert (ws.masks / "real.png").is_file()
