"""Pure, Gradio-free logic behind the local web UI.

Everything the UI *does* — saving uploaded pages, streaming per-stage progress,
assembling image galleries, zipping the finished dataset, deriving the run
settings from the wizard's inputs — lives here so it can be unit-tested without
importing Gradio. :mod:`make_style_dataset.ui.app` is then a thin view layer.
"""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from make_style_dataset.config import Settings
from make_style_dataset.media import IMAGE_SUFFIXES, image_files
from make_style_dataset.pipeline import DONE_MARKER, STAGES, run_stage
from make_style_dataset.stages.base import Stage, StageContext, StageResult
from make_style_dataset.stages.clean import _decode_bgr, _write_png, denoise, upscale_to
from make_style_dataset.vlm_caption import RecaptionResult, recaption_dataset
from make_style_dataset.workspace import Workspace

if TYPE_CHECKING:
    from make_style_dataset.proxy import CaptionClient

#: Glyphs prefixed to each progress line, one per phase.
_PHASE_GLYPH = {"running": "…", "done": "✓", "skipped": "•", "error": "✗"}


def build_settings(base: Settings, trigger: str, repeats: float | int) -> Settings:
    """Return ``base`` with the wizard's trigger/repeats applied and clamped.

    ``model_copy(update=…)`` skips validation, so the clamping (non-empty
    trigger, ``repeats >= 1``) happens here rather than relying on the field
    constraints. A blank trigger falls back to the base value.
    """
    token = (trigger or "").strip() or base.trigger_token
    reps = max(1, int(repeats))
    return base.model_copy(update={"trigger_token": token, "dataset_repeats": reps})


def build_train_settings(
    base: Settings,
    *,
    model_type: str,
    base_model: str,
    network_dim: float | int,
    network_alpha: float | int,
    learning_rate: float,
    max_train_steps: float | int,
) -> Settings:
    """Return ``base`` with the training-step inputs applied, clamped, ``run_train`` on.

    Like :func:`build_settings`, ``model_copy(update=…)`` skips validation, so the
    clamps (dim/alpha/steps ``>= 1``; learning rate ``> 0``, else the base value)
    live here. Blank ``model_type`` falls back to the base family.
    """
    lr = float(learning_rate)
    return base.model_copy(
        update={
            "train_model_type": ((model_type or "").strip().lower() or base.train_model_type),
            "train_base_model": (base_model or "").strip(),
            "train_network_dim": max(1, int(network_dim)),
            "train_network_alpha": max(1, int(network_alpha)),
            "train_learning_rate": lr if lr > 0 else base.train_learning_rate,
            "train_max_train_steps": max(1, int(max_train_steps)),
            "run_train": True,
        }
    )


def lora_files(lora_dir: Path) -> list[Path]:
    """Return trained ``.safetensors`` files in ``lora_dir`` (sorted; empty if absent)."""
    if not lora_dir.is_dir():
        return []
    return sorted(p for p in lora_dir.iterdir() if p.is_file() and p.suffix == ".safetensors")


def save_uploaded_pages(uploaded: Iterable[str | Path] | None, pages_dir: Path) -> int:
    """Copy uploaded image files into ``pages_dir``; return how many were saved.

    Non-image uploads are ignored. ``pages_dir`` is created if missing. Gradio
    hands us temp file paths, so the originals are copied (not moved).
    """
    pages_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for item in uploaded or []:
        src = Path(item)
        if src.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        shutil.copy2(src, pages_dir / src.name)
        saved += 1
    return saved


def gallery_items(directory: Path) -> list[tuple[str, str]]:
    """Return ``(image_path, caption)`` pairs for a stage folder, sorted by name.

    The caption is the sidecar ``<name>.txt`` if present, else the file name.
    A missing directory yields an empty list (the gallery just shows nothing).
    """
    items: list[tuple[str, str]] = []
    for path in image_files(directory):
        sidecar = path.with_suffix(".txt")
        caption = sidecar.read_text(encoding="utf-8").strip() if sidecar.is_file() else path.name
        items.append((str(path), caption))
    return items


def promote_to_clean(workspace: Workspace, names: Iterable[str], settings: Settings) -> int:
    """Rescue hand-picked ``manual_review`` panels into ``04_clean``; return the count.

    For each selected file name (basename only — directory parts are stripped, so a
    malicious gallery value cannot escape ``manual_review``): upscale it to
    ``target_side`` exactly like the clean stage, write it into ``04_clean``, then
    delete the original and its ``<stem>.reason.txt`` from ``manual_review``. When
    anything is promoted, the caption stage's completion marker is removed so the
    next **Build** re-captions the enlarged clean set *without* re-running the clean
    stage (which would regenerate ``04_clean`` and wipe the rescued panels).
    """
    clean_dir = workspace.clean
    clean_dir.mkdir(parents=True, exist_ok=True)
    review = workspace.manual_review
    promoted = 0
    for name in names:
        src = review / Path(name).name
        if not src.is_file() or src.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = _decode_bgr(src)
        if image is None:
            continue
        if settings.clean_denoise:
            image = denoise(image)
        image = upscale_to(image, settings.target_side)
        _write_png(image, clean_dir / f"{src.stem}.png")
        src.unlink()
        src.with_suffix(".reason.txt").unlink(missing_ok=True)
        promoted += 1
    if promoted:
        marker = (
            workspace.training_dir(settings.dataset_repeats, settings.trigger_token) / DONE_MARKER
        )
        marker.unlink(missing_ok=True)
    return promoted


def release_gpu_memory() -> None:
    """Best-effort release of VRAM held by the pipeline's model backends.

    The model stages (YOLO bubble detector, LaMa inpainter, WD14 tagger) load
    weights into the GPU. In the long-running UI process those linger after a
    build; freeing them lets a following LoRA training claim the whole GPU (Flux
    needs nearly all 16 GB). Forces a GC pass so out-of-scope ONNX/torch sessions
    are finalised, then empties torch's CUDA cache. No-op when torch / a GPU is
    absent (CPU-only env or CI).
    """
    import gc

    gc.collect()
    try:
        import torch  # pyright: ignore[reportMissingImports]
    except Exception:  # pragma: no cover - torch is an optional gpu-group dep
        return
    if torch.cuda.is_available():  # pragma: no cover - needs a real GPU present
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def recaption_training_dir(
    settings: Settings,
    *,
    model: str,
    style: str = "rich",
    client: CaptionClient | None = None,
) -> RecaptionResult:
    """Re-caption the assembled dataset's ``.txt`` sidecars via the Gemini proxy.

    Used by the UI 'Re-caption' button (``model='gemini-2.5-pro'``) and testable
    with an injected ``client``. Returns a friendly :class:`RecaptionResult` (no
    raise) when the HF token or the dataset folder is missing, so the UI can show
    the message instead of a traceback.
    """
    if not settings.hf_token and client is None:
        return RecaptionResult(
            0, 0, ["No HF token — put HF_TOKEN (read access to the proxy) in .env."]
        )
    target = Workspace(root=settings.workspace).training_dir(
        settings.dataset_repeats, settings.trigger_token
    )
    if not target.is_dir():
        return RecaptionResult(
            0, 0, [f"No dataset folder yet ({target.name}) — build the dataset first."]
        )
    if client is None:  # pragma: no cover - constructs the real network client
        from make_style_dataset.proxy import GeminiProxyClient

        client = GeminiProxyClient(settings.hf_token)
    return recaption_dataset(
        target,
        trigger=settings.trigger_token,
        model=model,
        style=style,
        client=client,
        max_workers=settings.vlm_concurrency,
    )


def zip_training_dir(dataset_dir: Path, out_path: Path) -> Path | None:
    """Zip the files directly under ``dataset_dir`` into ``out_path``.

    Returns ``out_path`` on success, or ``None`` if the dataset directory is
    absent or empty (nothing to download yet). ``out_path`` should be a stable
    location (e.g. under the workspace), not a tempdir that may be reaped before
    the user clicks download.
    """
    if not dataset_dir.is_dir():
        return None
    files = sorted(p for p in dataset_dir.iterdir() if p.is_file())
    if not files:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    return out_path


@dataclass(frozen=True)
class StageProgress:
    """One progress event emitted while the pipeline runs.

    A stage emits ``running`` before its (possibly long) work and then exactly
    one terminal event: ``done``, ``skipped`` or ``error``. ``fraction`` drives
    a progress bar; :attr:`line` is a ready-to-show log line.
    """

    index: int  # 1-based position of this stage
    total: int  # total stages in the run
    name: str
    phase: str  # "running" | "done" | "skipped" | "error"
    produced: int = 0
    detail: str = ""

    @property
    def fraction(self) -> float:
        """Completed fraction in ``[0, 1]`` (a running stage counts as in-flight)."""
        if self.total <= 0:
            return 1.0
        completed = self.index - 1 + (0.0 if self.phase == "running" else 1.0)
        return completed / self.total

    @property
    def line(self) -> str:
        """Render this event as a single human-readable log line."""
        glyph = _PHASE_GLYPH.get(self.phase, "?")
        head = f"{glyph} [{self.index}/{self.total}] {self.name}"
        if self.phase == "running":
            return f"{head}: running…"
        if self.phase == "done":
            return f"{head}: done ({self.produced} produced)"
        if self.phase == "skipped":
            return f"{head}: skipped ({self.detail})"
        return f"{head}: error — {self.detail}"


# Type of the per-stage runner, injected so tests need neither models nor disk.
Runner = Callable[..., StageResult]


def run_pipeline_stream(
    ctx: StageContext,
    *,
    force: bool = False,
    stages: tuple[Stage, ...] = STAGES,
    runner: Runner = run_stage,
) -> Iterator[StageProgress]:
    """Run the pipeline, yielding a :class:`StageProgress` as each stage advances.

    Mirrors :func:`make_style_dataset.pipeline.run_all`'s flag-gating, but emits
    a ``running`` event before each enabled stage so a non-technical user sees
    live movement instead of a frozen screen. A stage that raises yields an
    ``error`` event and stops the run (later stages depend on its output), rather
    than surfacing a raw traceback.
    """
    total = len(stages)
    for index, stage in enumerate(stages, start=1):
        if not getattr(ctx.settings, stage.flag):
            yield StageProgress(
                index=index,
                total=total,
                name=stage.name,
                phase="skipped",
                detail=f"disabled by {stage.flag}",
            )
            continue
        yield StageProgress(index=index, total=total, name=stage.name, phase="running")
        try:
            result = runner(stage, ctx, force=force)
        except Exception as exc:  # surface a friendly line, not a traceback
            yield StageProgress(
                index=index,
                total=total,
                name=stage.name,
                phase="error",
                detail=str(exc) or exc.__class__.__name__,
            )
            return
        if result.skipped:
            yield StageProgress(
                index=index,
                total=total,
                name=stage.name,
                phase="skipped",
                detail=result.reason,
            )
        else:
            yield StageProgress(
                index=index,
                total=total,
                name=stage.name,
                phase="done",
                produced=result.produced,
            )
