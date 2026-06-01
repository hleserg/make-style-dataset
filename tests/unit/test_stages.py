"""Tests for the individual pipeline stage stubs."""

from __future__ import annotations

from pathlib import Path

import pytest

from make_style_dataset.config import Settings
from make_style_dataset.stages import bubbles, caption, clean, inpaint, panels
from make_style_dataset.stages.base import StageContext
from make_style_dataset.workspace import Workspace


def _context(tmp_path: Path) -> StageContext:
    settings = Settings(workspace=tmp_path / "ws")
    return StageContext(workspace=Workspace(root=settings.workspace), settings=settings)


@pytest.mark.parametrize(
    ("module", "expected_attr"),
    [
        (panels, "panels"),
        (bubbles, "masks"),
        (inpaint, "inpainted"),
        (clean, "clean"),
    ],
)
def test_stage_creates_output_dir(tmp_path: Path, module: object, expected_attr: str) -> None:
    ctx = _context(tmp_path)
    result = module.run(ctx)  # type: ignore[attr-defined]
    expected = getattr(ctx.workspace, expected_attr)
    assert result.output_dir == expected
    assert expected.is_dir()
    assert result.produced == 0
    assert result.skipped is False


def test_caption_creates_training_dir(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    result = caption.run(ctx)
    expected = ctx.workspace.training_dir(ctx.settings.dataset_repeats, ctx.settings.trigger_token)
    assert result.output_dir == expected
    assert expected.is_dir()


def test_stage_metadata_is_consistent() -> None:
    for module in (panels, bubbles, inpaint, clean, caption):
        stage = module.STAGE  # type: ignore[attr-defined]
        assert stage.name == module.NAME  # type: ignore[attr-defined]
        assert stage.component.startswith("stage:")
        assert stage.flag.startswith("run_")
