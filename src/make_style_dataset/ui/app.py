"""Gradio wizard for the style-dataset pipeline (thin view layer).

A three-step wizard for a non-technical user: **1) settings → 2) pages & build
→ 3) result**. All real work lives in :mod:`make_style_dataset.ui.service` and
the pipeline; this module is pure Gradio wiring and is excluded from coverage
(see ``[tool.coverage.run] omit`` in ``pyproject.toml``), exactly like
:mod:`make_style_dataset.cli`.

Gradio is an optional dependency (the ``ui`` group). Nothing imports this module
at package import time, so the rest of the package stays Gradio-free.
"""

# Gradio is an optional dependency (the 'ui' group) and uses dynamic exports
# pyright can't follow statically, so type-checking is disabled file-wide here —
# the same reasoning that makes this module coverage-omit. CI (no 'ui' group)
# would otherwise flag the import as missing; a dev with 'ui' installed would
# otherwise see attribute errors on gr.* — this covers both.
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
from __future__ import annotations

import gradio as gr

from make_style_dataset.pipeline import make_context, summarize_run
from make_style_dataset.stages.base import StageContext
from make_style_dataset.ui.service import (
    build_settings,
    gallery_items,
    run_pipeline_stream,
    save_uploaded_pages,
    zip_training_dir,
)

_FIRST_RUN_NOTE = (
    "ℹ️ **First run downloads several GB of AI models.** A step can sit on "
    "*“running…”* for several minutes the first time — this is normal, it has "
    "not frozen. Later runs are much faster."
)


def build_demo(ctx: StageContext) -> gr.Blocks:
    """Assemble the wizard. ``ctx`` provides the base settings and workspace."""
    settings = ctx.settings

    with gr.Blocks(title="make-style-dataset") as demo:
        gr.Markdown(
            "# make-style-dataset\n"
            "Turn comic pages into a kohya-ready **style** dataset — in three steps."
        )

        # --- Step 1: settings ---------------------------------------------
        with gr.Group(visible=True) as step1:
            gr.Markdown("### Step 1 — name your style")
            trigger_in = gr.Textbox(
                label="Trigger word",
                value=settings.trigger_token,
                info="The word that summons your style in prompts; also names the output folder.",
            )
            repeats_in = gr.Number(
                label="kohya repeats",
                value=settings.dataset_repeats,
                precision=0,
                minimum=1,
                info="Repeat count kohya uses; names the dataset folder '<repeats>_<trigger>'.",
            )
            to_step2 = gr.Button("Next →", variant="primary")

        # --- Step 2: pages & build ----------------------------------------
        with gr.Group(visible=False) as step2:
            gr.Markdown("### Step 2 — add pages and build")
            gr.Markdown(_FIRST_RUN_NOTE)
            files_in = gr.File(
                label="Comic pages (drag & drop, multiple)",
                file_count="multiple",
                file_types=["image"],
            )
            build_btn = gr.Button("▶ Build dataset", variant="primary")
            build_log = gr.Textbox(
                label="Progress",
                lines=12,
                max_lines=12,
                interactive=False,
                autoscroll=True,
            )

        # --- Step 3: result -----------------------------------------------
        with gr.Group(visible=False) as step3:
            gr.Markdown("### Step 3 — your dataset")
            summary_md = gr.Markdown()
            download_file = gr.File(label="Download dataset (.zip)")
            with gr.Tab("Dataset"):
                result_gallery = gr.Gallery(label="Dataset images", columns=4, height="auto")
            with gr.Tab("Manual review"):
                gr.Markdown(
                    "Pages too tricky to auto-slice land here. Crop the good panels "
                    "by hand into `04_clean/` and re-build to caption them — or ignore "
                    "them if you already have enough."
                )
                review_gallery = gr.Gallery(label="Needs a human", columns=4, height="auto")

        def _go_to_step2() -> tuple[object, object]:
            return gr.update(visible=False), gr.update(visible=True)

        to_step2.click(_go_to_step2, outputs=[step1, step2])

        def _build(trigger: str, repeats: float, uploaded: list[str] | None):
            run_settings = build_settings(settings, trigger, repeats)
            run_ctx = make_context(run_settings)
            saved = save_uploaded_pages(uploaded, run_ctx.workspace.pages)

            lines = [f"Saved {saved} new page(s) to 00_pages/.", "Running the pipeline…", ""]
            yield {build_log: "\n".join(lines)}
            for progress in run_pipeline_stream(run_ctx):
                lines.append(progress.line)
                yield {build_log: "\n".join(lines)}

            training = run_ctx.workspace.training_dir(
                run_settings.dataset_repeats, run_settings.trigger_token
            )
            zip_path = zip_training_dir(training, run_ctx.workspace.root / f"{training.name}.zip")
            lines += ["", "Finished — see the Result step below."]
            yield {
                build_log: "\n".join(lines),
                step3: gr.update(visible=True),
                summary_md: f"```\n{summarize_run(run_ctx)}\n```",
                result_gallery: gallery_items(training),
                review_gallery: gallery_items(run_ctx.workspace.manual_review),
                download_file: str(zip_path) if zip_path else None,
            }

        build_btn.click(
            _build,
            inputs=[trigger_in, repeats_in, files_in],
            outputs=[build_log, step3, summary_md, result_gallery, review_gallery, download_file],
            api_name="build",
        )

    return demo


def launch_ui(ctx: StageContext) -> None:
    """Build and serve the wizard on the configured host/port, opening a browser."""
    demo = build_demo(ctx)
    demo.launch(
        server_name=ctx.settings.ui_host,
        server_port=ctx.settings.ui_port,
        inbrowser=True,
    )
