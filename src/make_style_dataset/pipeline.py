"""Pipeline orchestration: an ordered stage registry and an idempotent runner.

The registry (:data:`STAGES`) is the single source of truth for stage order and
identity; both the CLI and ``run-all`` derive their behaviour from it. The runner
is idempotent: a completed stage drops a marker file in its output directory and
is skipped on re-runs unless ``force`` is set.
"""

from __future__ import annotations

from pathlib import Path

from make_style_dataset.config import Settings
from make_style_dataset.media import image_files
from make_style_dataset.stages import bubbles, caption, clean, inpaint, panels
from make_style_dataset.stages.base import Stage, StageContext, StageResult
from make_style_dataset.workspace import Workspace

#: Stages in execution order. Order is load-bearing: each consumes the previous
#: stage's output directory.
STAGES: tuple[Stage, ...] = (
    panels.STAGE,
    bubbles.STAGE,
    inpaint.STAGE,
    clean.STAGE,
    caption.STAGE,
)

STAGE_BY_NAME: dict[str, Stage] = {stage.name: stage for stage in STAGES}

#: Marker file written into a stage's output dir once it completes.
DONE_MARKER = ".stage_complete"


def stage_names() -> list[str]:
    """Return stage names in execution order (used for CLI help and listings)."""
    return [stage.name for stage in STAGES]


def make_context(settings: Settings) -> StageContext:
    """Build a :class:`StageContext` from settings, ensuring base dirs exist."""
    workspace = Workspace(root=settings.workspace)
    workspace.ensure_base()
    return StageContext(workspace=workspace, settings=settings)


def run_stage(stage: Stage, ctx: StageContext, *, force: bool = False) -> StageResult:
    """Run one stage idempotently.

    # PLAYBOOK-START
    # id: idempotent-stage-marker
    # title: Idempotent stage runner via a completion marker
    # status: draft
    # category: pipeline
    # tags: [idempotency, pipeline, batch]
    # Each stage drops a marker file in its output dir on success and is
    # skipped on re-runs unless forced. Reruns of a partially-finished
    # pipeline become safe and cheap; the same shape works for any resumable
    # batch job, not just this dataset pipeline.
    # PLAYBOOK-END
    """
    out = stage.output(ctx.workspace, ctx.settings)
    marker = out / DONE_MARKER
    if marker.exists() and not force:
        return StageResult(
            name=stage.name,
            output_dir=out,
            skipped=True,
            reason="already complete (use --force to rerun)",
        )

    result = stage.run(ctx)
    marker.write_text(f"{stage.name}\n", encoding="utf-8")
    return result


def run_single(name: str, ctx: StageContext, *, force: bool = False) -> StageResult:
    """Run one stage by name, ignoring its enable flag (explicit invocation)."""
    stage = STAGE_BY_NAME[name]
    return run_stage(stage, ctx, force=force)


def run_all(ctx: StageContext, *, force: bool = False) -> list[StageResult]:
    """Run every enabled stage in order; flag-disabled stages are skipped."""
    results: list[StageResult] = []
    for stage in STAGES:
        if not getattr(ctx.settings, stage.flag):
            results.append(
                StageResult(
                    name=stage.name,
                    output_dir=stage.output(ctx.workspace, ctx.settings),
                    skipped=True,
                    reason=f"disabled by {stage.flag}",
                )
            )
            continue
        results.append(run_stage(stage, ctx, force=force))
    return results


def _count_images(directory: Path) -> int:
    """Count image files directly under ``directory`` (0 when it is absent)."""
    return len(image_files(directory))


def summarize_run(ctx: StageContext) -> str:
    """Render the end-of-run tally of surviving artifacts in each stage folder.

    Reads the workspace after a run, so it reflects the real on-disk result
    (including a resumed/partial run), not just this invocation's stages.
    """
    ws = ctx.workspace
    dataset = ws.training_dir(ctx.settings.dataset_repeats, ctx.settings.trigger_token)
    rows = [
        ("pages (00_pages)", _count_images(ws.pages)),
        ("panels (01_panels)", _count_images(ws.panels)),
        ("masks (02_masks)", _count_images(ws.masks)),
        ("inpainted (03_inpainted)", _count_images(ws.inpainted)),
        ("clean (04_clean)", _count_images(ws.clean)),
        (f"dataset ({dataset.name})", _count_images(dataset)),
        ("manual_review", _count_images(ws.manual_review)),
    ]
    width = max(len(label) for label, _ in rows)
    lines = ["Pipeline summary:"]
    lines += [f"  {label:<{width}}  {count}" for label, count in rows]
    return "\n".join(lines)
