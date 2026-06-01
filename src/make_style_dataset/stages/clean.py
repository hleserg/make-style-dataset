"""Stage 4 — dedup and size filtering.

Reads inpainted panels from ``03_inpainted`` and writes the survivors to
``04_clean``: near-duplicates (perceptual-hash distance < ``dedup_hamming_distance``)
are dropped, and panels whose shorter side is below ``min_side_px`` are routed to
``manual_review``. This is a scaffold stub that only creates its output directory.
"""

from __future__ import annotations

from make_style_dataset.observability import tag_component
from make_style_dataset.stages.base import Stage, StageContext, StageResult

NAME = "clean"
SUMMARY = "Deduplicate panels and filter out ones that are too small."
COMPONENT = "stage:clean"


def run(ctx: StageContext) -> StageResult:
    """Dedup/filter ``03_inpainted`` into ``04_clean`` (stub)."""
    tag_component(COMPONENT)
    out = ctx.workspace.clean
    out.mkdir(parents=True, exist_ok=True)
    return StageResult(name=NAME, output_dir=out, produced=0)


STAGE = Stage(
    name=NAME,
    summary=SUMMARY,
    component=COMPONENT,
    flag="run_clean",
    output=lambda ws, _s: ws.clean,
    run=run,
)
