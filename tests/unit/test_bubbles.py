"""Unit tests for the bubble/SFX detection + masking stage.

These exercise the pure mask geometry/policy helpers and the masking
orchestration with fake detectors, so no ultralytics/easyocr/torch is needed
here (the heavy adapters are injected). cv2/numpy/PIL are real dependencies and
are used directly.
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
    """Returns fixed polygons regardless of the panel (stands in for YOLOv8-seg)."""

    def __init__(self, polygons: list[Polygon]) -> None:
        self._polygons = polygons

    def detect(self, image: np.ndarray) -> list[Polygon]:
        del image  # the fake ignores the panel content
        return [list(p) for p in self._polygons]


class FakeTextDetector:
    """Returns fixed text boxes regardless of the panel (stands in for EasyOCR)."""

    def __init__(self, boxes: list[BBox]) -> None:
        self._boxes = boxes

    def detect(self, image: np.ndarray) -> list[BBox]:
        del image
        return list(self._boxes)


def _white_panel(path: Path, width: int = 200, height: int = 200) -> Path:
    Image.new("RGB", (width, height), "white").save(path)
    return path


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "workspace": tmp_path / "ws",
        "mask_dilation_px": 0,
        "max_mask_coverage": 0.9,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# --- pure mask geometry ---------------------------------------------------


def test_boxes_to_mask_fills_region() -> None:
    mask = bubbles.boxes_to_mask([(10, 20, 60, 80)], 100, 100)
    assert mask.shape == (100, 100)
    assert mask[50, 30] == 255
    assert mask[0, 0] == 0


def test_boxes_to_mask_normalizes_swapped_corners() -> None:
    # (x1,y1) below-right of (x2,y2): the box should still fill the same area.
    mask = bubbles.boxes_to_mask([(60, 80, 10, 20)], 100, 100)
    assert mask[50, 30] == 255


def test_boxes_to_mask_clamps_and_skips_degenerate() -> None:
    clamped = bubbles.boxes_to_mask([(-10, -10, 40, 40)], 50, 50)
    assert clamped[0, 0] == 255  # clamped into bounds
    degenerate = bubbles.boxes_to_mask([(10, 10, 10, 30)], 50, 50)  # zero width
    assert int(np.count_nonzero(degenerate)) == 0


def test_polygons_to_mask_fills_contour() -> None:
    square: Polygon = [(10, 10), (60, 10), (60, 60), (10, 60)]
    mask = bubbles.polygons_to_mask([square], 100, 100)
    assert mask[30, 30] == 255
    assert mask[5, 5] == 0


def test_polygons_to_mask_skips_under_three_points() -> None:
    mask = bubbles.polygons_to_mask([[(10, 10), (60, 60)]], 100, 100)
    assert int(np.count_nonzero(mask)) == 0


def test_union_masks_combines_via_or() -> None:
    a = bubbles.boxes_to_mask([(0, 0, 10, 10)], 100, 100)
    b = bubbles.boxes_to_mask([(50, 50, 60, 60)], 100, 100)
    union = bubbles.union_masks([a, b], 100, 100)
    assert union[5, 5] == 255
    assert union[55, 55] == 255
    assert union[30, 30] == 0


def test_union_masks_empty_is_all_black() -> None:
    union = bubbles.union_masks([], 40, 30)
    assert union.shape == (30, 40)
    assert int(np.count_nonzero(union)) == 0


def test_dilate_mask_expands_region() -> None:
    mask = bubbles.boxes_to_mask([(40, 40, 60, 60)], 100, 100)
    dilated = bubbles.dilate_mask(mask, 5)
    assert int(np.count_nonzero(dilated)) > int(np.count_nonzero(mask))


def test_dilate_mask_zero_is_identity() -> None:
    mask = bubbles.boxes_to_mask([(40, 40, 60, 60)], 100, 100)
    assert np.array_equal(bubbles.dilate_mask(mask, 0), mask)


def test_mask_coverage_ratio() -> None:
    mask = bubbles.boxes_to_mask([(0, 0, 100, 50)], 100, 100)  # half the panel
    assert bubbles.mask_coverage_ratio(mask) == pytest.approx(0.5, abs=0.001)


def test_mask_coverage_ratio_empty_array() -> None:
    assert bubbles.mask_coverage_ratio(np.zeros((0, 0), dtype=np.uint8)) == 0.0


def test_mask_is_empty() -> None:
    assert bubbles.mask_is_empty(np.zeros((10, 10), dtype=np.uint8))
    assert not bubbles.mask_is_empty(bubbles.boxes_to_mask([(0, 0, 5, 5)], 10, 10))


def test_classify_mask_accepts_small_and_rejects_large() -> None:
    small = bubbles.boxes_to_mask([(0, 0, 10, 10)], 100, 100)
    assert bubbles.classify_mask(small, max_coverage=0.6) is None
    big = bubbles.boxes_to_mask([(0, 0, 100, 100)], 100, 100)
    reason = bubbles.classify_mask(big, max_coverage=0.6)
    assert reason is not None and "coverage" in reason


def test_overlay_mask_tints_only_masked_pixels() -> None:
    image = np.full((20, 20, 3), 200, dtype=np.uint8)  # light gray BGR
    mask = bubbles.boxes_to_mask([(0, 0, 10, 10)], 20, 20)
    overlay = bubbles.overlay_mask(image, mask)
    assert tuple(int(c) for c in overlay[5, 5]) != (200, 200, 200)  # tinted
    assert tuple(int(c) for c in overlay[15, 15]) == (200, 200, 200)  # untouched


# --- decode / iter --------------------------------------------------------


def test_decode_bgr_reads_valid(tmp_path: Path) -> None:
    panel = _white_panel(tmp_path / "p.png", 30, 20)
    image = bubbles._decode_bgr(panel)
    assert image is not None and image.shape == (20, 30, 3)


def test_decode_bgr_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.png"
    empty.touch()
    assert bubbles._decode_bgr(empty) is None


def test_decode_bgr_garbage(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.png"
    bogus.write_text("not an image", encoding="utf-8")
    assert bubbles._decode_bgr(bogus) is None


def test_iter_panels_filters_and_sorts(tmp_path: Path) -> None:
    panels_dir = tmp_path / "panels"
    panels_dir.mkdir()
    _white_panel(panels_dir / "b.png")
    _white_panel(panels_dir / "a.jpg")
    (panels_dir / "notes.txt").write_text("ignore", encoding="utf-8")
    assert [p.name for p in bubbles.iter_panels(panels_dir)] == ["a.jpg", "b.png"]


def test_iter_panels_missing_dir(tmp_path: Path) -> None:
    assert bubbles.iter_panels(tmp_path / "nope") == []


# --- adapter construction (no heavy import) -------------------------------


def test_adapters_construct_without_heavy_deps() -> None:
    yolo = bubbles.YoloBubbleDetector("kitsumed/yolov8m_seg-speech-bubble", confidence=0.5)
    assert yolo._confidence == 0.5
    ocr = bubbles.OcrTextDetector(["en", "ja"], gpu=False)
    assert ocr._languages == ["en", "ja"]


# --- mask_panel orchestration with fakes ----------------------------------


def _dirs(tmp_path: Path) -> tuple[Path, Path]:
    out, manual = tmp_path / "out", tmp_path / "manual"
    out.mkdir()
    manual.mkdir()
    return out, manual


def test_mask_panel_writes_union_of_bubble_and_text(tmp_path: Path) -> None:
    panel = _white_panel(tmp_path / "panel.png")
    out, manual = _dirs(tmp_path)
    bubble = FakeBubbleDetector([[(10, 10), (60, 10), (60, 60), (10, 60)]])
    text = FakeTextDetector([(100, 100, 140, 140)])
    written = bubbles.mask_panel(
        panel,
        bubble_detector=bubble,
        text_detector=text,
        settings=_settings(tmp_path),
        out_dir=out,
        manual_review=manual,
    )
    assert written is True
    with Image.open(out / "panel.png") as img:
        assert img.mode == "L"
        arr = np.array(img)
    assert arr[30, 30] == 255  # bubble polygon
    assert arr[120, 120] == 255  # text box
    assert arr[0, 0] == 0  # background


def test_mask_panel_empty_when_no_detections(tmp_path: Path) -> None:
    panel = _white_panel(tmp_path / "clean.png")
    out, manual = _dirs(tmp_path)
    written = bubbles.mask_panel(
        panel,
        bubble_detector=FakeBubbleDetector([]),
        text_detector=FakeTextDetector([]),
        settings=_settings(tmp_path),
        out_dir=out,
        manual_review=manual,
    )
    assert written is True  # an empty mask is still written (1:1 contract)
    with Image.open(out / "clean.png") as img:
        assert bubbles.mask_is_empty(np.array(img))


def test_mask_panel_routes_excessive_coverage(tmp_path: Path) -> None:
    panel = _white_panel(tmp_path / "busy.png")
    out, manual = _dirs(tmp_path)
    full = FakeBubbleDetector([[(0, 0), (200, 0), (200, 200), (0, 200)]])
    written = bubbles.mask_panel(
        panel,
        bubble_detector=full,
        text_detector=FakeTextDetector([]),
        settings=_settings(tmp_path, max_mask_coverage=0.5),
        out_dir=out,
        manual_review=manual,
    )
    assert written is False
    assert not (out / "busy.png").exists()
    assert (manual / "busy.png").is_file()
    assert "coverage" in (manual / "busy.reason.txt").read_text(encoding="utf-8")


def test_mask_panel_routes_unreadable(tmp_path: Path) -> None:
    bogus = tmp_path / "bad.png"
    bogus.write_text("garbage", encoding="utf-8")
    out, manual = _dirs(tmp_path)
    written = bubbles.mask_panel(
        bogus,
        bubble_detector=FakeBubbleDetector([]),
        text_detector=FakeTextDetector([]),
        settings=_settings(tmp_path),
        out_dir=out,
        manual_review=manual,
    )
    assert written is False
    assert "unreadable" in (manual / "bad.reason.txt").read_text(encoding="utf-8")


def test_mask_panel_writes_debug_overlay(tmp_path: Path) -> None:
    panel = _white_panel(tmp_path / "dbg.png")
    out, manual = _dirs(tmp_path)
    written = bubbles.mask_panel(
        panel,
        bubble_detector=FakeBubbleDetector([[(10, 10), (60, 10), (60, 60), (10, 60)]]),
        text_detector=FakeTextDetector([]),
        settings=_settings(tmp_path, bubbles_debug=True),
        out_dir=out,
        manual_review=manual,
    )
    assert written is True
    overlay = manual / "dbg.overlay.png"
    assert overlay.is_file()
    with Image.open(overlay) as img:
        assert img.mode == "RGB"


# --- run() orchestration --------------------------------------------------


def _patch_fakes(monkeypatch, bubble: FakeBubbleDetector, text: FakeTextDetector) -> None:
    monkeypatch.setattr(bubbles, "YoloBubbleDetector", lambda *a, **k: bubble)
    monkeypatch.setattr(bubbles, "OcrTextDetector", lambda *a, **k: text)


def test_run_masks_all_panels_idempotently(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.panels.mkdir(parents=True, exist_ok=True)
    _white_panel(ws.panels / "p1.png")
    _white_panel(ws.panels / "p2.png")
    _patch_fakes(
        monkeypatch,
        FakeBubbleDetector([[(10, 10), (60, 10), (60, 60), (10, 60)]]),
        FakeTextDetector([]),
    )
    ctx = StageContext(workspace=ws, settings=settings)

    result = bubbles.run(ctx)
    assert result.produced == 2
    assert sorted(p.name for p in ws.masks.iterdir()) == ["p1.png", "p2.png"]

    again = bubbles.run(ctx)
    assert again.produced == 2
    assert len(list(ws.masks.iterdir())) == 2  # deterministic names: no duplicates


def test_run_routes_overcovered_panels(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, max_mask_coverage=0.1)
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.panels.mkdir(parents=True, exist_ok=True)
    _white_panel(ws.panels / "p1.png")
    _patch_fakes(
        monkeypatch,
        FakeBubbleDetector([[(0, 0), (200, 0), (200, 200), (0, 200)]]),
        FakeTextDetector([]),
    )
    result = bubbles.run(StageContext(workspace=ws, settings=settings))
    assert result.produced == 0
    assert list(ws.masks.iterdir()) == []
    assert (ws.manual_review / "p1.png").is_file()
