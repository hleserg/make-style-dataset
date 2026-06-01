"""Integration tests for the caption stage.

Drives ``run()`` end-to-end over a ``04_clean`` workspace (iteration, tagging,
kohya layout) with the heavy WD14 backend replaced by a fake. A separate
real-WD14 smoke test is ``importorskip``-gated and excluded from coverage; it is
skipped in CI, which never installs the gpu group.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from make_style_dataset.config import Settings
from make_style_dataset.stages import caption
from make_style_dataset.stages.base import StageContext
from make_style_dataset.workspace import Workspace


class FakeTagger:
    """Returns fixed content tags (stands in for WD14)."""

    def __init__(self, tags: list[str]) -> None:
        self._tags = tags

    def tag(self, image: np.ndarray) -> list[str]:
        del image
        return list(self._tags)


def _image(path: Path, size: int = 128) -> Path:
    Image.new("RGB", (size, size), "white").save(path)
    return path


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {"workspace": tmp_path / "ws"}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.mark.integration
def test_caption_end_to_end_kohya_layout(tmp_path: Path, monkeypatch) -> None:
    """Clean panels become a kohya 05_dataset/<N>_<trigger>/ of image+caption pairs."""
    settings = _settings(tmp_path, dataset_repeats=15, trigger_token="grnvlstyle")
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.clean.mkdir(parents=True, exist_ok=True)
    _image(ws.clean / "panel_00.png")
    _image(ws.clean / "panel_01.png")

    monkeypatch.setattr(
        caption, "make_tagger", lambda _s: FakeTagger(["1girl", "solo", "outdoors"])
    )
    result = caption.run(StageContext(workspace=ws, settings=settings))

    training = ws.training_dir(15, "grnvlstyle")
    assert training.name == "15_grnvlstyle"  # kohya <repeats>_<trigger>
    assert result.produced == 2
    assert sorted(p.name for p in training.iterdir()) == [
        "panel_00.png",
        "panel_00.txt",
        "panel_01.png",
        "panel_01.txt",
    ]
    # Style Locker: trigger first, then content tags.
    text = (training / "panel_00.txt").read_text(encoding="utf-8")
    assert text == "grnvlstyle, 1girl, solo, outdoors\n"
    # Every image has exactly one caption sidecar.
    pngs = {p.stem for p in training.glob("*.png")}
    txts = {p.stem for p in training.glob("*.txt")}
    assert pngs == txts


@pytest.mark.integration
def test_real_wd14_smoke(tmp_path: Path) -> None:  # pragma: no cover - needs gpu group
    """Run the real WD14 ONNX tagger if (and only if) onnxruntime is installed.

    Skipped in CI and the default DoD env, which never install the gpu group.
    Downloads the WD14 weights + tags CSV on first run, so excluded from coverage.
    """
    pytest.importorskip("onnxruntime")

    settings = Settings(  # type: ignore[arg-type]
        workspace=tmp_path / "ws", dataset_repeats=10, trigger_token="style"
    )
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.clean.mkdir(parents=True, exist_ok=True)
    _image(ws.clean / "real.png")

    result = caption.run(StageContext(workspace=ws, settings=settings))
    assert result.produced == 1
    training = ws.training_dir(10, "style")
    assert (training / "real.png").is_file()
    text = (training / "real.txt").read_text(encoding="utf-8")
    assert text.startswith("style")  # trigger first
