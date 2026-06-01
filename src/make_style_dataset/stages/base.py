"""Shared types for pipeline stages.

A stage is a pure-ish function ``run(ctx) -> StageResult`` plus metadata. Keeping
the metadata next to the function lets ``pipeline.py`` assemble an ordered
registry without importing each stage's internals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from make_style_dataset.config import Settings
from make_style_dataset.workspace import Workspace


@dataclass(frozen=True)
class StageContext:
    """Everything a stage needs to run: where to work and how it's configured."""

    workspace: Workspace
    settings: Settings


@dataclass(frozen=True)
class StageResult:
    """Outcome of running a single stage."""

    name: str
    output_dir: Path
    produced: int = 0
    skipped: bool = False
    reason: str = ""


@dataclass(frozen=True)
class Stage:
    """A pipeline stage: ordered metadata plus its run function.

    ``flag`` names the :class:`~make_style_dataset.config.Settings` boolean that
    gates this stage during ``run-all``. ``output`` resolves the stage's output
    directory from a :class:`~make_style_dataset.workspace.Workspace`.
    """

    name: str
    summary: str
    component: str
    flag: str
    output: Callable[[Workspace, Settings], Path]
    run: Callable[[StageContext], StageResult]
