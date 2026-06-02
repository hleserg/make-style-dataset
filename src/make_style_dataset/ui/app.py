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

from pathlib import Path

import gradio as gr

from make_style_dataset.pipeline import make_context, summarize_run
from make_style_dataset.stages import train
from make_style_dataset.stages.base import StageContext
from make_style_dataset.ui.service import (
    build_settings,
    build_train_settings,
    gallery_items,
    lora_files,
    promote_to_clean,
    recaption_training_dir,
    release_gpu_memory,
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
                gr.Markdown(
                    "Captions come from the WD14 tagger. **Re-caption with a VLM** (Gemini Pro) "
                    "for trigger-first prose that describes the content and never names the style "
                    "— cleaner for a Flux style LoRA. Runs via the proxy."
                )
                recaption_btn = gr.Button("📝 Re-caption with VLM (Pro)")
                recaption_status = gr.Markdown()
            with gr.Tab("Manual review"):
                gr.Markdown(
                    "Panels the auto-clean set aside (too small, or a tricky crop). "
                    "**Click the good ones** to select them, then **Send → 04_clean** — "
                    "they're upscaled and folded into the dataset on your next **Build** "
                    "(no file hunting). Re-run Build to caption them."
                )
                review_gallery = gr.Gallery(label="Needs a human", columns=4, height="auto")
                review_names = gr.State([])
                review_select = gr.Dropdown(
                    label="Selected to rescue (click thumbnails above, or pick here)",
                    multiselect=True,
                    choices=[],
                    interactive=True,
                )
                with gr.Row():
                    rescue_btn = gr.Button("✅ Send selected → 04_clean", variant="primary")
                    rescue_all_btn = gr.Button("Send all")
                rescue_status = gr.Markdown()
            to_step4 = gr.Button("Train a LoRA from this dataset →")

        # --- Step 4: train (optional) -------------------------------------
        with gr.Group(visible=False) as step4:
            gr.Markdown(
                "### Step 4 — train a style LoRA (optional)\n"
                "Needs a local kohya **sd-scripts** clone + a base checkpoint. Run "
                "`make-style-dataset doctor` first to check the trainer environment. "
                "Training takes a while — per-step progress prints in the terminal."
            )
            model_type_in = gr.Dropdown(
                ["sd15", "sdxl", "flux"],
                value=settings.train_model_type,
                label="Base-model family",
            )
            base_model_in = gr.Textbox(
                label="Base checkpoint path",
                value=settings.train_base_model,
                info="Local .safetensors (SD1.5/SDXL) or the Flux DiT. Required.",
            )
            with gr.Row():
                dim_in = gr.Number(
                    label="Network dim", value=settings.train_network_dim, precision=0, minimum=1
                )
                alpha_in = gr.Number(
                    label="Network alpha",
                    value=settings.train_network_alpha,
                    precision=0,
                    minimum=1,
                )
                lr_in = gr.Number(label="Learning rate", value=settings.train_learning_rate)
                steps_in = gr.Number(
                    label="Max train steps",
                    value=settings.train_max_train_steps,
                    precision=0,
                    minimum=1,
                )
            train_btn = gr.Button("▶ Train LoRA", variant="primary")
            train_log = gr.Textbox(
                label="Training progress",
                lines=12,
                max_lines=12,
                interactive=False,
                autoscroll=True,
            )
            lora_download = gr.File(label="Download LoRA (.safetensors)")

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

            release_gpu_memory()  # drop model VRAM so a following Flux train gets the whole GPU
            lines.append("Freed GPU memory (model stages no longer hold VRAM).")
            yield {build_log: "\n".join(lines)}

            training = run_ctx.workspace.training_dir(
                run_settings.dataset_repeats, run_settings.trigger_token
            )
            zip_path = zip_training_dir(training, run_ctx.workspace.root / f"{training.name}.zip")
            review = [Path(p).name for p, _ in gallery_items(run_ctx.workspace.manual_review)]
            lines += ["", "Finished — see the Result step below."]
            yield {
                build_log: "\n".join(lines),
                step3: gr.update(visible=True),
                summary_md: f"```\n{summarize_run(run_ctx)}\n```",
                result_gallery: gallery_items(training),
                review_gallery: gallery_items(run_ctx.workspace.manual_review),
                review_names: review,
                review_select: gr.update(choices=review, value=[]),
                download_file: str(zip_path) if zip_path else None,
            }

        build_btn.click(
            _build,
            inputs=[trigger_in, repeats_in, files_in],
            outputs=[
                build_log,
                step3,
                summary_md,
                result_gallery,
                review_gallery,
                review_names,
                review_select,
                download_file,
            ],
            api_name="build",
        )

        def _select_review(names: list[str], current: list[str], evt: gr.SelectData):
            """Add the clicked manual_review thumbnail to the rescue selection."""
            if evt.index is None or evt.index >= len(names):
                return gr.update()
            chosen = list(current or [])
            name = names[evt.index]
            if name not in chosen:
                chosen.append(name)
            return gr.update(value=chosen)

        review_gallery.select(
            _select_review, inputs=[review_names, review_select], outputs=[review_select]
        )

        def _rescue(trigger: str, repeats: float, selected: list[str] | None):
            run_settings = build_settings(settings, trigger, repeats)
            ws = make_context(run_settings).workspace
            count = promote_to_clean(ws, selected or [], run_settings)
            remaining = [Path(p).name for p, _ in gallery_items(ws.manual_review)]
            note = (
                f"✅ Rescued **{count}** panel(s) into `04_clean`. Click **▶ Build dataset** to "
                "caption them — the heavy stages skip, only captioning re-runs."
                if count
                else "Nothing selected — click thumbnails above (or pick names) first."
            )
            return (
                gallery_items(ws.manual_review),
                remaining,
                gr.update(choices=remaining, value=[]),
                note,
            )

        rescue_btn.click(
            _rescue,
            inputs=[trigger_in, repeats_in, review_select],
            outputs=[review_gallery, review_names, review_select, rescue_status],
        )
        rescue_all_btn.click(
            _rescue,
            inputs=[trigger_in, repeats_in, review_names],
            outputs=[review_gallery, review_names, review_select, rescue_status],
        )

        def _recaption(trigger: str, repeats: float):
            run_settings = build_settings(settings, trigger, repeats)
            result = recaption_training_dir(run_settings, model="gemini-2.5-pro", style="rich")
            training = make_context(run_settings).workspace.training_dir(
                run_settings.dataset_repeats, run_settings.trigger_token
            )
            if result.written:
                note = f"✅ Re-captioned **{result.written}** image(s) with Gemini Pro (prose)"
                note += f" — {result.failed} failed." if result.failed else "."
                if result.errors:
                    note += " " + "; ".join(result.errors[:3])
            else:
                note = "⚠️ " + (result.errors[0] if result.errors else "nothing to do.")
            return gallery_items(training), note

        recaption_btn.click(
            _recaption, inputs=[trigger_in, repeats_in], outputs=[result_gallery, recaption_status]
        )

        to_step4.click(lambda: gr.update(visible=True), outputs=[step4])

        def _train(
            trigger: str,
            repeats: float,
            model_type: str,
            base_model: str,
            dim: float,
            alpha: float,
            lr: float,
            steps: float,
        ):
            run_settings = build_train_settings(
                build_settings(settings, trigger, repeats),
                model_type=model_type,
                base_model=base_model,
                network_dim=dim,
                network_alpha=alpha,
                learning_rate=lr,
                max_train_steps=steps,
            )
            run_ctx = make_context(run_settings)

            lines = [
                f"Training a {run_settings.train_model_type} LoRA — this can take a while…",
                "",
            ]
            yield {train_log: "\n".join(lines)}
            for progress in run_pipeline_stream(run_ctx, force=True, stages=(train.STAGE,)):
                lines.append(progress.line)
                yield {train_log: "\n".join(lines)}

            produced = lora_files(run_ctx.workspace.lora)
            final = (
                run_ctx.workspace.lora / f"{train.resolve_output_name(run_settings)}.safetensors"
            )
            download = str(final) if final.is_file() else (str(produced[-1]) if produced else None)
            lines += ["", f"Done — {len(produced)} LoRA file(s) in {run_ctx.workspace.lora}."]
            yield {train_log: "\n".join(lines), lora_download: download}

        train_btn.click(
            _train,
            inputs=[
                trigger_in,
                repeats_in,
                model_type_in,
                base_model_in,
                dim_in,
                alpha_in,
                lr_in,
                steps_in,
            ],
            outputs=[train_log, lora_download],
            api_name="train",
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
