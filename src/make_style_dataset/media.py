"""Shared media helpers used across the pipeline.

Small, dependency-free utilities that several stages (and the UI/onboarding
code) all need: the set of image suffixes the pipeline recognises, listing a
directory's images in a stable order, and routing a file to ``manual_review``
with a reason note. Centralised here so there is a single source of truth
instead of a copy per module.

This module imports nothing heavy (stdlib only), so it is safe to import from
the pure, lazily-backed stage cores without pulling in OpenCV/torch.
"""

from __future__ import annotations

import shutil
from pathlib import Path

#: Image file extensions the pipeline reads/writes (lowercase, with the dot).
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def image_files(directory: Path) -> list[Path]:
    """Return image files directly under ``directory``, sorted by path.

    Returns an empty list when ``directory`` is absent or not a directory, so
    callers can iterate unconditionally. The stable sort makes per-file output
    names (and order-sensitive steps like dedup) deterministic across runs.
    """
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def route_to_manual(source: Path, manual_review: Path, reason: str) -> None:
    """Copy ``source`` whole into ``manual_review`` with a ``<stem>.reason.txt`` note.

    Used by stages to set aside pages/panels a human should inspect rather than
    guessing (mis-segmented pages, over-large masks, too-small crops).
    """
    shutil.copy2(source, manual_review / source.name)
    note = manual_review / f"{source.stem}.reason.txt"
    note.write_text(f"{reason}\n", encoding="utf-8")
