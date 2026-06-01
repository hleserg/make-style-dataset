"""Integration test for the clean stage end-to-end over a realistic set.

Exercises ``run()`` on a mix of duplicate "adjacent frames" and mixed-size
panels through the real cv2 pHash + Lanczos path (no heavy backend).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from make_style_dataset.config import Settings
from make_style_dataset.stages import clean
from make_style_dataset.stages.base import StageContext
from make_style_dataset.workspace import Workspace


def _save(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array[:, :, ::-1], mode="RGB").save(path)  # array is BGR


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {"workspace": tmp_path / "ws"}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.mark.integration
def test_clean_pipeline_end_to_end(tmp_path: Path) -> None:
    settings = _settings(tmp_path, dedup_hamming_distance=6, min_side_px=64, target_side=256)
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.inpainted.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    shot_a = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
    shot_b = rng.integers(0, 256, (120, 120, 3), dtype=np.uint8)  # distinct, needs upscale

    # Two near-identical frames of shot A (a tiny perturbation), plus an exact
    # repeat — only the first survives dedup.
    _save(shot_a, ws.inpainted / "p00.png")
    near = shot_a.copy()
    near[0:2, 0:2] = 0  # 4-pixel tweak -> perceptually identical
    _save(near, ws.inpainted / "p01.png")
    _save(shot_a, ws.inpainted / "p02.png")
    # A distinct, smaller shot that should be kept and upscaled.
    _save(shot_b, ws.inpainted / "p03.png")
    # A too-small panel routed to manual review.
    _save(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8), ws.inpainted / "p04.png")

    result = clean.run(StageContext(workspace=ws, settings=settings))

    survivors = sorted(p.name for p in ws.clean.iterdir())
    assert survivors == ["p00.png", "p03.png"]  # p01/p02 deduped, p04 too small
    assert result.produced == 2
    assert (ws.manual_review / "p04.png").is_file()
    # The distinct small shot was upscaled to the target shorter side.
    with Image.open(ws.clean / "p03.png") as img:
        assert min(img.size) == 256
    # The already-large survivor was left at its native size (no fake upscaling).
    with Image.open(ws.clean / "p00.png") as img:
        assert min(img.size) == 300
