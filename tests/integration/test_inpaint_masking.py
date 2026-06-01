"""Integration tests for the inpaint stage.

Drives ``run()`` end-to-end over a workspace (mask iteration, panel pairing,
empty-mask pass-through, inpaint + write) with the heavy ONNX backend replaced
by a fake. A separate real-LaMa smoke test is ``importorskip``-gated and
excluded from coverage; it is skipped in CI, which never installs the gpu group.
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
    """Paints masked pixels a flat BGR color (stands in for ONNX LaMa)."""

    def __init__(self, color: tuple[int, int, int] = (0, 0, 255)) -> None:
        self._color = color

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = image.copy()
        result[mask > inpaint.MASK_THRESHOLD] = self._color
        return result


def _panel(path: Path, size: int = 128) -> Path:
    Image.new("RGB", (size, size), "white").save(path)
    return path


def _mask(path: Path, size: int = 128, *, fill: bool = False) -> Path:
    array = np.full((size, size), 255 if fill else 0, dtype=np.uint8)
    Image.fromarray(array, mode="L").save(path)
    return path


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {"workspace": tmp_path / "ws"}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.mark.integration
def test_masked_and_clean_panels_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """A masked panel is inpainted; a clean (empty-mask) panel passes through."""
    settings = _settings(tmp_path)
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.panels.mkdir(parents=True, exist_ok=True)
    ws.masks.mkdir(parents=True, exist_ok=True)

    _panel(ws.panels / "masked.png")
    _mask(ws.masks / "masked.png", fill=True)  # whole panel masked
    _panel(ws.panels / "clean.png")
    _mask(ws.masks / "clean.png", fill=False)  # empty mask -> pass through

    monkeypatch.setattr(inpaint, "make_inpainter", lambda _s: FakeInpainter(color=(0, 0, 255)))
    result = inpaint.run(StageContext(workspace=ws, settings=settings))

    assert result.produced == 2
    assert sorted(p.name for p in ws.inpainted.iterdir()) == ["clean.png", "masked.png"]

    with Image.open(ws.inpainted / "masked.png") as img:
        masked = np.array(img.convert("RGB"))
    assert np.all(masked[:, :, 0] == 255) and np.all(masked[:, :, 1] == 0)  # painted red

    with Image.open(ws.inpainted / "clean.png") as img:
        clean = np.array(img.convert("RGB"))
    assert np.all(clean == 255)  # untouched white panel


@pytest.mark.integration
def test_partial_mask_preserves_unmasked_region(tmp_path: Path, monkeypatch) -> None:
    """Compositing keeps pixels outside the mask byte-identical to the panel."""
    settings = _settings(tmp_path)
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.panels.mkdir(parents=True, exist_ok=True)
    ws.masks.mkdir(parents=True, exist_ok=True)

    # A non-uniform panel so "unchanged" is meaningful.
    panel = np.random.default_rng(0).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(panel, mode="RGB").save(ws.panels / "p.png")
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[:64, :] = 255  # top half masked
    Image.fromarray(mask, mode="L").save(ws.masks / "p.png")

    # Fake inpainter that composites via the real composite() helper, like LaMa does.
    class CompositingFake:
        def inpaint(self, image: np.ndarray, m: np.ndarray) -> np.ndarray:
            painted = np.zeros_like(image)
            return inpaint.composite(image, painted, m)

    monkeypatch.setattr(inpaint, "make_inpainter", lambda _s: CompositingFake())
    inpaint.run(StageContext(workspace=ws, settings=settings))

    with Image.open(ws.inpainted / "p.png") as img:
        out = np.array(img.convert("RGB"))
    assert np.all(out[64:, :] == panel[64:, :])  # unmasked bottom half preserved exactly
    assert np.all(out[:64, :] == 0)  # masked top half repainted


@pytest.mark.integration
def test_real_lama_smoke(tmp_path: Path) -> None:  # pragma: no cover - needs gpu group
    """Run the real ONNX LaMa adapter if (and only if) onnxruntime is installed.

    Skipped in CI and the default DoD env, which never install the gpu group.
    Downloads the LaMa weights on first run, so it is excluded from coverage.
    """
    pytest.importorskip("onnxruntime")

    settings = Settings(workspace=tmp_path / "ws")  # type: ignore[arg-type]
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.panels.mkdir(parents=True, exist_ok=True)
    ws.masks.mkdir(parents=True, exist_ok=True)
    _panel(ws.panels / "real.png")
    _mask(ws.masks / "real.png", fill=False)  # empty mask -> pass-through, no model needed

    result = inpaint.run(StageContext(workspace=ws, settings=settings))
    assert result.produced == 1
    assert (ws.inpainted / "real.png").is_file()
