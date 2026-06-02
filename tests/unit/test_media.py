"""Tests for make_style_dataset.media (shared image-listing + routing helpers)."""

from __future__ import annotations

from pathlib import Path

from make_style_dataset.media import IMAGE_SUFFIXES, image_files, route_to_manual


def _touch(path: Path) -> Path:
    path.write_bytes(b"x")
    return path


def test_image_suffixes_are_lowercase_with_dot() -> None:
    assert ".png" in IMAGE_SUFFIXES
    assert all(s.startswith(".") and s.islower() for s in IMAGE_SUFFIXES)


def test_image_files_missing_dir_is_empty(tmp_path: Path) -> None:
    assert image_files(tmp_path / "absent") == []


def test_image_files_sorts_and_filters(tmp_path: Path) -> None:
    _touch(tmp_path / "b.png")
    _touch(tmp_path / "a.JPG")  # mixed-case suffix still counts
    _touch(tmp_path / "notes.txt")  # non-image ignored
    (tmp_path / "sub").mkdir()  # directories ignored

    assert image_files(tmp_path) == [tmp_path / "a.JPG", tmp_path / "b.png"]


def test_route_to_manual_copies_and_writes_reason(tmp_path: Path) -> None:
    src = _touch(tmp_path / "panel_3.png")
    review = tmp_path / "manual_review"
    review.mkdir()

    route_to_manual(src, review, "too small after inpaint")

    assert (review / "panel_3.png").read_bytes() == b"x"
    note = (review / "panel_3.reason.txt").read_text(encoding="utf-8")
    assert note == "too small after inpaint\n"
