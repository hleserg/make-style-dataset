"""Stage 2 — speech-bubble detection.

Reads panels from ``01_panels`` and writes a binary bubble mask per panel to
``02_masks`` (white = bubble to remove). The real implementation detects speech
balloons and text regions; this is a scaffold stub that only creates its output
directory.
"""

from __future__ import annotations

from make_style_dataset.observability import tag_component
from make_style_dataset.stages.base import Stage, StageContext, StageResult

NAME = "bubbles"
SUMMARY = "Detect speech bubbles in panels and write removal masks."
COMPONENT = "stage:bubbles"


def run(ctx: StageContext) -> StageResult:
    """Write bubble masks for ``01_panels`` into ``02_masks`` (stub)."""
    tag_component(COMPONENT)
    out = ctx.workspace.masks
    out.mkdir(parents=True, exist_ok=True)
    return StageResult(name=NAME, output_dir=out, produced=0)


STAGE = Stage(
    name=NAME,
    summary=SUMMARY,
    component=COMPONENT,
    flag="run_bubbles",
    output=lambda ws, _s: ws.masks,
    run=run,
)
