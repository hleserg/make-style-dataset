"""Unit tests for the inpaint stage.

These exercise the backend factory, the pure ONNX tensor pre/post-processing
and compositing, and the masking orchestration with a fake inpainter — so no
onnxruntime/torch is needed (the heavy session lives behind the Inpainter
protocol). cv2/numpy/PIL are real dependencies and used directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from make_style_dataset.config import Settings
from make_style_dataset.stages import inpaint
from make_style_dataset.stages.base import StageContext
from make_style_dataset.workspace import Workspace


class FakeInpainter:
    """Paints masked pixels a flat BGR color (stands in for LaMa)."""

    def __init__(self, color: tuple[int, int, int] = (0, 0, 255)) -> None:
        self._color = color

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = image.copy()
        result[mask > inpaint.MASK_THRESHOLD] = self._color
        return result


def _white_panel(path: Path, width: int = 64, height: int = 48) -> Path:
    Image.new("RGB", (width, height), "white").save(path)
    return path


def _mask(path: Path, width: int = 64, height: int = 48, *, fill: bool = False) -> Path:
    array = np.full((height, width), 255 if fill else 0, dtype=np.uint8)
    Image.fromarray(array, mode="L").save(path)
    return path


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {"workspace": tmp_path / "ws"}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# --- backend factory ------------------------------------------------------


def test_make_inpainter_returns_lama(tmp_path: Path) -> None:
    inpainter = inpaint.make_inpainter(_settings(tmp_path, inpaint_backend="lama"))
    assert isinstance(inpainter, inpaint.LamaInpainter)


def test_make_inpainter_case_insensitive(tmp_path: Path) -> None:
    inpainter = inpaint.make_inpainter(_settings(tmp_path, inpaint_backend="  LaMa "))
    assert isinstance(inpainter, inpaint.LamaInpainter)


def test_make_inpainter_unknown_backend_raises(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="APP_INPAINT_BACKEND"):
        inpaint.make_inpainter(_settings(tmp_path, inpaint_backend="sd"))


def test_inpaint_backend_default_is_lama(tmp_path: Path) -> None:
    assert _settings(tmp_path).inpaint_backend == "lama"


# --- pure tensor pre/post-processing --------------------------------------


def test_lama_inputs_resizes_to_model_size() -> None:
    image = np.full((48, 64, 3), 200, dtype=np.uint8)  # H=48, W=64 BGR
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    feed = inpaint.lama_inputs(image, mask, (32, 32))
    assert feed["image"].shape == (1, 3, 32, 32)
    assert feed["mask"].shape == (1, 1, 32, 32)
    assert feed["image"].dtype == np.float32
    assert feed["image"].min() >= 0.0
    assert feed["image"].max() <= 1.0
    assert set(np.unique(feed["mask"]).tolist()) <= {0.0, 1.0}  # binarized


def test_lama_inputs_binarizes_gray_mask() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0, 0] = 100  # below threshold -> 0
    mask[1, 1] = 200  # above threshold -> 1
    feed = inpaint.lama_inputs(image, mask, (8, 8))  # identity resize keeps coords
    assert feed["mask"][0, 0, 0, 0] == 0.0
    assert feed["mask"][0, 0, 1, 1] == 1.0


def test_lama_output_clips_and_converts_rgb_to_bgr() -> None:
    # CHW RGB with an out-of-range value in the red channel.
    raw = np.zeros((3, 4, 4), dtype=np.float32)
    raw[0] = 300.0  # R way over 255 -> must clip to 255, not wrap
    out = inpaint.lama_output_to_bgr(raw, width=4, height=4)
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.uint8
    # In BGR, pure red is (0, 0, 255).
    assert tuple(int(c) for c in out[0, 0]) == (0, 0, 255)


def test_lama_output_drops_batch_dim_and_resizes() -> None:
    raw = np.zeros((1, 3, 8, 8), dtype=np.float32)  # NCHW with batch
    out = inpaint.lama_output_to_bgr(raw, width=20, height=16)
    assert out.shape == (16, 20, 3)  # resized to native (H, W)


def test_composite_takes_painted_only_in_mask() -> None:
    original = np.full((10, 10, 3), 255, dtype=np.uint8)  # white
    painted = np.zeros((10, 10, 3), dtype=np.uint8)  # black
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:5, :] = 255  # top half masked
    out = inpaint.composite(original, painted, mask)
    assert np.all(out[0:5, :] == 0)  # masked -> painted (black)
    assert np.all(out[5:, :] == 255)  # unmasked -> original (white, untouched)


def test_mask_is_empty() -> None:
    assert inpaint.mask_is_empty(np.zeros((5, 5), dtype=np.uint8))
    assert not inpaint.mask_is_empty(np.full((5, 5), 255, dtype=np.uint8))


# --- I/O helpers ----------------------------------------------------------


def test_decode_bgr_and_mask(tmp_path: Path) -> None:
    panel = _white_panel(tmp_path / "p.png", 30, 20)
    bgr = inpaint._decode_bgr(panel)
    assert bgr is not None and bgr.shape == (20, 30, 3)
    _mask(tmp_path / "m.png", 30, 20, fill=True)
    gray = inpaint._decode_mask(tmp_path / "m.png")
    assert gray is not None and gray.shape == (20, 30) and gray.ndim == 2


def test_decode_empty_file_returns_none(tmp_path: Path) -> None:
    empty = tmp_path / "empty.png"
    empty.touch()
    assert inpaint._decode_bgr(empty) is None
    assert inpaint._decode_mask(empty) is None


def test_iter_masks_filters_and_sorts(tmp_path: Path) -> None:
    masks = tmp_path / "masks"
    masks.mkdir()
    _mask(masks / "b.png")
    _mask(masks / "a.png")
    (masks / "notes.txt").write_text("x", encoding="utf-8")
    assert [p.name for p in inpaint.iter_masks(masks)] == ["a.png", "b.png"]


def test_iter_masks_missing_dir(tmp_path: Path) -> None:
    assert inpaint.iter_masks(tmp_path / "nope") == []


def test_find_panel_matches_by_stem(tmp_path: Path) -> None:
    panels = tmp_path / "panels"
    panels.mkdir()
    _white_panel(panels / "foo.png")
    assert inpaint._find_panel(panels, "foo") == panels / "foo.png"
    assert inpaint._find_panel(panels, "missing") is None


# --- inpaint_panel orchestration ------------------------------------------


def _dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    panels, masks, out = tmp_path / "panels", tmp_path / "masks", tmp_path / "out"
    for d in (panels, masks, out):
        d.mkdir()
    return panels, masks, out


def test_inpaint_panel_empty_mask_passes_through(tmp_path: Path) -> None:
    panels, masks, out = _dirs(tmp_path)
    _white_panel(panels / "p.png")
    empty = _mask(masks / "p.png", fill=False)
    written = inpaint.inpaint_panel(
        empty, inpainter=FakeInpainter(), panels_dir=panels, out_dir=out
    )
    assert written is True
    with Image.open(out / "p.png") as img:
        assert np.all(np.array(img.convert("RGB")) == 255)  # unchanged white panel


def test_inpaint_panel_nonempty_mask_inpaints(tmp_path: Path) -> None:
    panels, masks, out = _dirs(tmp_path)
    _white_panel(panels / "p.png")
    full = _mask(masks / "p.png", fill=True)
    written = inpaint.inpaint_panel(
        full, inpainter=FakeInpainter(color=(0, 0, 255)), panels_dir=panels, out_dir=out
    )
    assert written is True
    with Image.open(out / "p.png") as img:
        arr = np.array(img.convert("RGB"))
    assert np.all(arr[:, :, 0] == 255) and np.all(arr[:, :, 1] == 0)  # painted red


def test_inpaint_panel_missing_panel_skips(tmp_path: Path) -> None:
    panels, masks, out = _dirs(tmp_path)
    orphan = _mask(masks / "p.png", fill=True)  # no panel for it (S2-routed)
    written = inpaint.inpaint_panel(
        orphan, inpainter=FakeInpainter(), panels_dir=panels, out_dir=out
    )
    assert written is False
    assert not (out / "p.png").exists()


def test_inpaint_panel_unreadable_panel_skips(tmp_path: Path) -> None:
    panels, masks, out = _dirs(tmp_path)
    (panels / "p.png").write_text("garbage", encoding="utf-8")
    full = _mask(masks / "p.png", fill=True)
    assert (
        inpaint.inpaint_panel(full, inpainter=FakeInpainter(), panels_dir=panels, out_dir=out)
        is False
    )


def test_inpaint_panel_output_name_is_png(tmp_path: Path) -> None:
    panels, masks, out = _dirs(tmp_path)
    _white_panel(panels / "foo.jpg")  # panel is a jpg
    empty = _mask(masks / "foo.png")  # mask is png
    written = inpaint.inpaint_panel(
        empty, inpainter=FakeInpainter(), panels_dir=panels, out_dir=out
    )
    assert written is True
    assert (out / "foo.png").exists() and not (out / "foo.jpg").exists()


# --- run() orchestration --------------------------------------------------


def test_run_idempotent(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.panels.mkdir(parents=True, exist_ok=True)
    ws.masks.mkdir(parents=True, exist_ok=True)
    _white_panel(ws.panels / "p1.png")
    _white_panel(ws.panels / "p2.png")
    _mask(ws.masks / "p1.png", fill=True)
    _mask(ws.masks / "p2.png", fill=False)
    monkeypatch.setattr(inpaint, "make_inpainter", lambda _s: FakeInpainter())

    ctx = StageContext(workspace=ws, settings=settings)
    first = inpaint.run(ctx)
    assert first.produced == 2
    assert sorted(p.name for p in ws.inpainted.iterdir()) == ["p1.png", "p2.png"]

    again = inpaint.run(ctx)
    assert again.produced == 2
    assert len(list(ws.inpainted.iterdir())) == 2  # deterministic names, no dupes


def test_run_skips_masks_without_panels(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.panels.mkdir(parents=True, exist_ok=True)
    ws.masks.mkdir(parents=True, exist_ok=True)
    _white_panel(ws.panels / "p1.png")
    _mask(ws.masks / "p1.png", fill=True)
    _mask(ws.masks / "orphan.png", fill=True)  # no matching panel -> skipped
    monkeypatch.setattr(inpaint, "make_inpainter", lambda _s: FakeInpainter())

    result = inpaint.run(StageContext(workspace=ws, settings=settings))
    assert result.produced == 1
    assert [p.name for p in ws.inpainted.iterdir()] == ["p1.png"]
