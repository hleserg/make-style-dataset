"""Stage 5 — captioning and kohya packaging.

Reads clean panels from ``04_clean`` and writes the final kohya-ready dataset to
``05_dataset/<repeats>_<trigger>/``: each image gets a ``.txt`` caption sidecar
beginning with the ``trigger_token``. The real implementation resizes to
``target_side`` and generates captions; this is a scaffold stub that only
creates its output directory.
"""

from __future__ import annotations

from pathlib import Path

from make_style_dataset.config import Settings
from make_style_dataset.observability import tag_component
from make_style_dataset.stages.base import Stage, StageContext, StageResult
from make_style_dataset.workspace import Workspace

NAME = "caption"
SUMMARY = "Caption clean panels and lay out the kohya-ready dataset folder."
COMPONENT = "stage:caption"


def _training_dir(ws: Workspace, settings: Settings) -> Path:
    return ws.training_dir(settings.dataset_repeats, settings.trigger_token)


def run(ctx: StageContext) -> StageResult:
    """Caption ``04_clean`` into ``05_dataset/<N>_<trigger>`` (stub)."""
    tag_component(COMPONENT)
    out = _training_dir(ctx.workspace, ctx.settings)
    out.mkdir(parents=True, exist_ok=True)
    return StageResult(name=NAME, output_dir=out, produced=0)


STAGE = Stage(
    name=NAME,
    summary=SUMMARY,
    component=COMPONENT,
    flag="run_caption",
    output=_training_dir,
    run=run,
)
