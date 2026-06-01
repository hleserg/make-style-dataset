"""Stage 3 — bubble inpainting.

Combines panels (``01_panels``) with their masks (``02_masks``) and writes
bubble-free panels to ``03_inpainted``. The real implementation inpaints the
masked regions; this is a scaffold stub that only creates its output directory.
"""

from __future__ import annotations

from make_style_dataset.observability import tag_component
from make_style_dataset.stages.base import Stage, StageContext, StageResult

NAME = "inpaint"
SUMMARY = "Inpaint masked speech bubbles out of each panel."
COMPONENT = "stage:inpaint"


def run(ctx: StageContext) -> StageResult:
    """Inpaint ``01_panels`` using ``02_masks`` into ``03_inpainted`` (stub)."""
    tag_component(COMPONENT)
    out = ctx.workspace.inpainted
    out.mkdir(parents=True, exist_ok=True)
    return StageResult(name=NAME, output_dir=out, produced=0)


STAGE = Stage(
    name=NAME,
    summary=SUMMARY,
    component=COMPONENT,
    flag="run_inpaint",
    output=lambda ws, _s: ws.inpainted,
    run=run,
)
