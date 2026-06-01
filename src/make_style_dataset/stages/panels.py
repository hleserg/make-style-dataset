"""Stage 1 — panel detection and slicing.

Reads raw pages from ``00_pages`` and writes one image per detected panel to
``01_panels``. The real implementation (HLE-742) uses Kumiko + OpenCV and drops
panels below ``min_panel_area``. This is a scaffold stub: it only creates its
output directory so the pipeline wires up end-to-end.
"""

from __future__ import annotations

from make_style_dataset.observability import tag_component
from make_style_dataset.stages.base import Stage, StageContext, StageResult

NAME = "panels"
SUMMARY = "Detect comic panels and slice pages into individual panel images."
COMPONENT = "stage:panels"


def run(ctx: StageContext) -> StageResult:
    """Slice ``00_pages`` into ``01_panels`` (stub: creates the output dir)."""
    tag_component(COMPONENT)
    out = ctx.workspace.panels
    out.mkdir(parents=True, exist_ok=True)
    return StageResult(name=NAME, output_dir=out, produced=0)


STAGE = Stage(
    name=NAME,
    summary=SUMMARY,
    component=COMPONENT,
    flag="run_panels",
    output=lambda ws, _s: ws.panels,
    run=run,
)
