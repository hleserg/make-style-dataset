"""Tests for make_style_dataset.pipeline."""

from __future__ import annotations

from pathlib import Path

from make_style_dataset.config import Settings
from make_style_dataset.pipeline import (
    DONE_MARKER,
    STAGE_BY_NAME,
    STAGES,
    make_context,
    run_all,
    run_single,
    run_stage,
    stage_names,
)
from make_style_dataset.stages.base import StageContext
from make_style_dataset.workspace import Workspace


def _context(tmp_path: Path, **overrides: object) -> StageContext:
    settings = Settings(workspace=tmp_path / "ws").model_copy(update=overrides)
    return StageContext(workspace=Workspace(root=settings.workspace), settings=settings)


def test_stage_order_and_names() -> None:
    assert stage_names() == ["panels", "bubbles", "inpaint", "clean", "caption"]
    assert list(STAGE_BY_NAME) == stage_names()
    assert len(STAGES) == 5


def test_make_context_creates_base_dirs(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path / "ws")
    ctx = make_context(settings)
    assert ctx.workspace.pages.is_dir()
    assert ctx.workspace.manual_review.is_dir()


def test_run_single_creates_output_and_marker(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    result = run_single("panels", ctx)
    assert result.skipped is False
    assert ctx.workspace.panels.is_dir()
    assert (ctx.workspace.panels / DONE_MARKER).exists()


def test_run_single_is_idempotent(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    run_single("panels", ctx)
    second = run_single("panels", ctx)
    assert second.skipped is True
    assert "force" in second.reason


def test_force_reruns_completed_stage(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    run_single("panels", ctx)
    forced = run_single("panels", ctx, force=True)
    assert forced.skipped is False


def test_run_all_runs_every_enabled_stage(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    results = run_all(ctx)
    assert [r.name for r in results] == stage_names()
    assert all(not r.skipped for r in results)
    for stage in STAGES:
        assert stage.output(ctx.workspace, ctx.settings).is_dir()


def test_run_all_skips_disabled_stage(tmp_path: Path) -> None:
    ctx = _context(tmp_path, run_bubbles=False)
    results = run_all(ctx)
    bubbles_result = next(r for r in results if r.name == "bubbles")
    assert bubbles_result.skipped is True
    assert "run_bubbles" in bubbles_result.reason
    assert not ctx.workspace.masks.exists()


def test_run_stage_accepts_stage_object(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    result = run_stage(STAGES[0], ctx)
    assert result.name == "panels"
