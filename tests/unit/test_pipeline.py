"""Tests for make_style_dataset.pipeline."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from make_style_dataset.config import Settings
from make_style_dataset.pipeline import (
    DONE_MARKER,
    STAGE_BY_NAME,
    STAGES,
    _count_images,
    make_context,
    run_all,
    run_single,
    run_stage,
    stage_names,
    summarize_run,
)
from make_style_dataset.stages.base import StageContext
from make_style_dataset.workspace import Workspace


def _context(tmp_path: Path, **overrides: object) -> StageContext:
    settings = Settings(workspace=tmp_path / "ws").model_copy(update=overrides)
    return StageContext(workspace=Workspace(root=settings.workspace), settings=settings)


def test_stage_order_and_names() -> None:
    assert stage_names() == ["panels", "bubbles", "inpaint", "clean", "caption", "train"]
    assert list(STAGE_BY_NAME) == stage_names()
    assert len(STAGES) == 6


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
    # The dataset stages run on an empty workspace (producing 0); the heavy
    # train stage is gated off by default (run_train=False), so it is skipped
    # and its 06_lora output dir is never created.
    dataset_results = [r for r in results if r.name != "train"]
    assert all(not r.skipped for r in dataset_results)
    for stage in STAGES:
        if stage.name == "train":
            continue
        assert stage.output(ctx.workspace, ctx.settings).is_dir()
    train_result = next(r for r in results if r.name == "train")
    assert train_result.skipped is True
    assert "run_train" in train_result.reason


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


# --- run summary ----------------------------------------------------------


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "white").save(path)


def test_count_images(tmp_path: Path) -> None:
    _png(tmp_path / "a.png")
    _png(tmp_path / "b.jpg")
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")
    assert _count_images(tmp_path) == 2
    assert _count_images(tmp_path / "missing") == 0  # absent dir -> 0


def test_summarize_run_tallies_each_stage(tmp_path: Path) -> None:
    ctx = _context(tmp_path, dataset_repeats=10, trigger_token="mystyle")
    ws = ctx.workspace
    _png(ws.pages / "p1.png")
    _png(ws.pages / "p2.png")
    _png(ws.panels / "p1_00.png")
    _png(ws.manual_review / "p2.png")
    _png(ws.training_dir(10, "mystyle") / "p1_00.png")

    report = summarize_run(ctx)
    assert "Pipeline summary:" in report
    assert "pages (00_pages)" in report and "  2" in report
    assert "dataset (10_mystyle)" in report
    # the dataset folder has exactly one image
    dataset_line = next(line for line in report.splitlines() if "dataset (10_mystyle)" in line)
    assert dataset_line.strip().endswith(" 1")
    manual_line = next(line for line in report.splitlines() if "manual_review" in line)
    assert manual_line.strip().endswith(" 1")
