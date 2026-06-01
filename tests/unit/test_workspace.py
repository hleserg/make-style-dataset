"""Tests for make_style_dataset.workspace."""

from __future__ import annotations

from pathlib import Path

from make_style_dataset.workspace import Workspace


def test_derived_paths() -> None:
    ws = Workspace(root=Path("ws"))
    assert ws.pages == Path("ws/00_pages")
    assert ws.panels == Path("ws/01_panels")
    assert ws.masks == Path("ws/02_masks")
    assert ws.inpainted == Path("ws/03_inpainted")
    assert ws.clean == Path("ws/04_clean")
    assert ws.dataset == Path("ws/05_dataset")
    assert ws.manual_review == Path("ws/manual_review")


def test_training_dir_uses_repeats_and_trigger() -> None:
    ws = Workspace(root=Path("ws"))
    assert ws.training_dir(10, "comicstyle") == Path("ws/05_dataset/10_comicstyle")


def test_ensure_base_creates_input_dirs(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "ws")
    ws.ensure_base()
    assert ws.root.is_dir()
    assert ws.pages.is_dir()
    assert ws.manual_review.is_dir()
    # Idempotent: a second call must not raise.
    ws.ensure_base()
