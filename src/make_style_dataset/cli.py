"""Command-line entry point for the style-dataset pipeline.

Thin argparse shell: one subcommand per pipeline stage plus ``run-all``. All
real logic lives in importable, tested modules (:mod:`make_style_dataset.pipeline`
and :mod:`make_style_dataset.stages`); this file is excluded from coverage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from make_style_dataset import __version__
from make_style_dataset.config import Settings, get_settings
from make_style_dataset.observability import init_sentry
from make_style_dataset.onboarding import (
    format_doctor_report,
    format_init_report,
    gather_checks,
    initialize_workspace,
)
from make_style_dataset.pipeline import STAGES, make_context, run_all, run_single, summarize_run
from make_style_dataset.stages.base import StageContext, StageResult
from make_style_dataset.workspace import Workspace


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", type=Path, help="Override the workspace root directory.")
    parser.add_argument(
        "--force", action="store_true", help="Rerun even if the stage is already complete."
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI: a subcommand per stage plus ``run-all``."""
    parser = argparse.ArgumentParser(
        prog="make-style-dataset",
        description="Comic pages -> kohya-ready style LoRA dataset pipeline.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"make-style-dataset {__version__}"
    )
    sub = parser.add_subparsers(dest="command", metavar="<stage>")

    init_parser = sub.add_parser(
        "init",
        help="Scaffold the workspace folders and seed .env (run this first).",
        description="Create the input/review folders and seed .env from the template.",
    )
    init_parser.add_argument(
        "--workspace", type=Path, help="Override the workspace root directory."
    )

    doctor_parser = sub.add_parser(
        "doctor",
        help="Check that this machine is ready (Python, GPU, workspace).",
        description="Diagnose the environment: interpreter, venv, .env, workspace, GPU stack.",
    )
    doctor_parser.add_argument(
        "--workspace", type=Path, help="Override the workspace root directory."
    )

    ui_parser = sub.add_parser(
        "ui",
        help="Launch the local web UI (needs the 'ui' dependency group).",
        description="Serve the step-by-step wizard in your browser (install: uv sync --group ui).",
    )
    ui_parser.add_argument("--workspace", type=Path, help="Override the workspace root directory.")

    for stage in STAGES:
        stage_parser = sub.add_parser(stage.name, help=stage.summary, description=stage.summary)
        _add_common(stage_parser)

    stage_list = ", ".join(stage.name for stage in STAGES)
    run_all_parser = sub.add_parser(
        "run-all",
        help="Run the whole pipeline in order.",
        description=f"Run every enabled stage in order: {stage_list}.",
        epilog=f"Stages: {stage_list}",
    )
    _add_common(run_all_parser)

    return parser


def _print_results(results: list[StageResult]) -> None:
    for result in results:
        if result.skipped:
            print(f"  - {result.name}: skipped ({result.reason})")
        else:
            print(f"  - {result.name}: ok -> {result.output_dir} ({result.produced} produced)")


def _resolve_settings(args: argparse.Namespace) -> Settings:
    get_settings.cache_clear()
    settings = get_settings()
    if args.workspace is not None:
        settings = settings.model_copy(update={"workspace": args.workspace})
    return settings


def _resolve_context(args: argparse.Namespace) -> StageContext:
    return make_context(_resolve_settings(args))


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.command is None:
        parser.print_help()
        return 0

    init_sentry()

    if args.command in ("doctor", "init"):
        # Build the workspace WITHOUT make_context's ensure_base(): doctor must
        # stay read-only, and init must do (and truthfully report) the creating.
        settings = _resolve_settings(args)
        workspace = Workspace(root=settings.workspace)
        if args.command == "doctor":
            print(format_doctor_report(gather_checks(workspace, settings=settings)))
        else:
            print(format_init_report(initialize_workspace(workspace), workspace))
        return 0

    if args.command == "ui":
        # Lazy import so the CLI (and `make check`) never require Gradio: the
        # heavy UI deps live in the optional 'ui' group, imported only on demand.
        try:
            from make_style_dataset.ui.app import launch_ui
        except ModuleNotFoundError:
            print(
                "The web UI needs the optional 'ui' dependency group. Install it with:\n"
                "  uv sync --group ui\n"
                "then rerun:  make-style-dataset ui   (or just: make ui)",
                file=sys.stderr,
            )
            return 1
        launch_ui(_resolve_context(args))
        return 0

    ctx = _resolve_context(args)
    try:
        if args.command == "run-all":
            print("Running pipeline:")
            _print_results(run_all(ctx, force=args.force))
            print(summarize_run(ctx))
        else:
            print(f"Running stage '{args.command}':")
            _print_results([run_single(args.command, ctx, force=args.force)])
    except (ValueError, RuntimeError) as exc:
        # e.g. `train` with no dataset, or the trainer subprocess failing: show a
        # one-line message instead of a raw traceback.
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
