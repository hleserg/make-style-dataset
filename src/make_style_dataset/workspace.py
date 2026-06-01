"""Workspace layout contract for the dataset pipeline.

A single, typed description of where each stage reads and writes. Every path
is derived from one ``root`` so the whole pipeline can be relocated by changing
``APP_WORKSPACE``. See ``docs/architecture/WORKSPACE.md`` for the prose
contract.

Layout::

    <root>/
      00_pages/              raw comic pages (pipeline input)
      01_panels/             individual panels sliced from pages
      02_masks/              speech-bubble masks per panel
      03_inpainted/          panels with bubbles inpainted away
      04_clean/              deduplicated, size-filtered panels
      05_dataset/<N>_<trig>/  kohya-ready images + caption sidecars
      manual_review/         anything kicked out for a human to inspect
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    """Resolved, typed view of the pipeline's on-disk layout.

    # PLAYBOOK-START
    # id: derived-workspace-layout
    # title: Single-root derived directory contract
    # status: draft
    # category: pipeline
    # tags: [filesystem, pipeline, config]
    # Derive every stage directory from one configurable root behind named
    # properties instead of scattering string joins across the codebase.
    # Relocating or sandboxing the whole pipeline becomes a one-line change,
    # and stages depend on an interface, not on literal path strings.
    # PLAYBOOK-END
    """

    root: Path

    @property
    def pages(self) -> Path:
        """Stage 0 input: raw comic pages."""
        return self.root / "00_pages"

    @property
    def panels(self) -> Path:
        """Panels sliced from pages."""
        return self.root / "01_panels"

    @property
    def masks(self) -> Path:
        """Speech-bubble masks per panel."""
        return self.root / "02_masks"

    @property
    def inpainted(self) -> Path:
        """Panels with bubbles inpainted away."""
        return self.root / "03_inpainted"

    @property
    def clean(self) -> Path:
        """Deduplicated, size-filtered panels."""
        return self.root / "04_clean"

    @property
    def dataset(self) -> Path:
        """Root of the kohya-ready dataset output."""
        return self.root / "05_dataset"

    @property
    def manual_review(self) -> Path:
        """Artifacts kicked out for a human to inspect."""
        return self.root / "manual_review"

    def training_dir(self, repeats: int, trigger: str) -> Path:
        """Return the kohya training subfolder named ``<repeats>_<trigger>``."""
        return self.dataset / f"{repeats}_{trigger}"

    def ensure_base(self) -> None:
        """Create the input and review directories that no stage owns."""
        for path in (self.root, self.pages, self.manual_review):
            path.mkdir(parents=True, exist_ok=True)
