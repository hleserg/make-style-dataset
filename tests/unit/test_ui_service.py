"""Tests for make_style_dataset.ui.service (Gradio-free UI logic)."""

from __future__ import annotations

import zipfile
from pathlib import Path

from make_style_dataset.config import Settings
from make_style_dataset.stages.base import Stage, StageContext, StageResult
from make_style_dataset.ui.service import (
    StageProgress,
    build_settings,
    build_train_settings,
    gallery_items,
    lora_files,
    run_pipeline_stream,
    save_uploaded_pages,
    zip_training_dir,
)
from make_style_dataset.workspace import Workspace


def _png(path: Path) -> Path:
    """Write a 1x1 PNG so suffix-based image checks see a real file."""
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
            "de0000000c4944415408d76360000002000154a24f9f0000000049454e44ae42"
            "6082"
        )
    )
    return path


# --- build_settings --------------------------------------------------------


def test_build_settings_applies_trigger_and_repeats() -> None:
    base = Settings(trigger_token="base", dataset_repeats=10)
    out = build_settings(base, "hero", 7)
    assert out.trigger_token == "hero"
    assert out.dataset_repeats == 7


# --- build_train_settings --------------------------------------------------


def test_build_train_settings_applies_and_enables() -> None:
    out = build_train_settings(
        Settings(),
        model_type="SDXL",
        base_model=" /m.safetensors ",
        network_dim=64,
        network_alpha=32,
        learning_rate=2e-4,
        max_train_steps=2000,
    )
    assert out.train_model_type == "sdxl"  # lower-cased
    assert out.train_base_model == "/m.safetensors"  # stripped
    assert (out.train_network_dim, out.train_network_alpha) == (64, 32)
    assert out.train_learning_rate == 2e-4
    assert out.train_max_train_steps == 2000
    assert out.run_train is True


def test_build_train_settings_clamps_and_falls_back() -> None:
    base = Settings(train_model_type="sd15", train_learning_rate=1e-4)
    out = build_train_settings(
        base,
        model_type="",  # blank -> base family
        base_model="",
        network_dim=0,  # clamped to >= 1
        network_alpha=0,
        learning_rate=0,  # non-positive -> base lr
        max_train_steps=0,  # clamped to >= 1
    )
    assert out.train_model_type == "sd15"
    assert out.train_network_dim == 1 and out.train_network_alpha == 1
    assert out.train_learning_rate == 1e-4
    assert out.train_max_train_steps == 1


# --- lora_files ------------------------------------------------------------


def test_lora_files_lists_safetensors_sorted(tmp_path: Path) -> None:
    lora = tmp_path / "06_lora"
    lora.mkdir()
    (lora / "b.safetensors").write_bytes(b"x")
    (lora / "a.safetensors").write_bytes(b"x")
    (lora / "dataset.toml").write_text("x", encoding="utf-8")  # not a LoRA
    assert [p.name for p in lora_files(lora)] == ["a.safetensors", "b.safetensors"]


def test_lora_files_missing_dir(tmp_path: Path) -> None:
    assert lora_files(tmp_path / "absent") == []


def test_build_settings_blank_trigger_falls_back() -> None:
    base = Settings(trigger_token="base", dataset_repeats=10)
    assert build_settings(base, "   ", 5).trigger_token == "base"


def test_build_settings_clamps_and_floors_repeats() -> None:
    base = Settings(trigger_token="base", dataset_repeats=10)
    assert build_settings(base, "x", 0).dataset_repeats == 1  # clamped to >= 1
    assert build_settings(base, "x", 12.9).dataset_repeats == 12  # float floored


# --- save_uploaded_pages ---------------------------------------------------


def test_save_uploaded_pages_copies_images_only(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    good = _png(src / "page1.png")
    other = _png(src / "page2.JPG")  # mixed case suffix still counts
    bad = src / "notes.txt"
    bad.write_text("nope", encoding="utf-8")
    pages = tmp_path / "ws" / "00_pages"

    saved = save_uploaded_pages([good, other, bad], pages)

    assert saved == 2
    assert (pages / "page1.png").is_file()
    assert (pages / "page2.JPG").is_file()
    assert not (pages / "notes.txt").exists()


def test_save_uploaded_pages_none_creates_dir_and_returns_zero(tmp_path: Path) -> None:
    pages = tmp_path / "ws" / "00_pages"
    assert save_uploaded_pages(None, pages) == 0
    assert pages.is_dir()


# --- gallery_items ---------------------------------------------------------


def test_gallery_items_missing_dir_is_empty(tmp_path: Path) -> None:
    assert gallery_items(tmp_path / "absent") == []


def test_gallery_items_uses_sidecar_caption_or_filename(tmp_path: Path) -> None:
    d = tmp_path / "out"
    d.mkdir()
    _png(d / "b.png")
    (d / "b.txt").write_text("  a caption  \n", encoding="utf-8")
    _png(d / "a.png")  # no sidecar -> falls back to filename
    (d / "ignore.txt").write_text("orphan", encoding="utf-8")
    (d / "sub").mkdir()  # directories are skipped

    items = gallery_items(d)

    assert items == [
        (str(d / "a.png"), "a.png"),
        (str(d / "b.png"), "a caption"),
    ]


# --- zip_training_dir ------------------------------------------------------


def test_zip_training_dir_missing_or_empty_returns_none(tmp_path: Path) -> None:
    assert zip_training_dir(tmp_path / "absent", tmp_path / "out.zip") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert zip_training_dir(empty, tmp_path / "out.zip") is None


def test_zip_training_dir_packs_files(tmp_path: Path) -> None:
    dataset = tmp_path / "10_comicstyle"
    dataset.mkdir()
    _png(dataset / "img.png")
    (dataset / "img.txt").write_text("cap", encoding="utf-8")
    out = tmp_path / "deep" / "dataset.zip"

    result = zip_training_dir(dataset, out)

    assert result == out
    assert out.is_file()
    with zipfile.ZipFile(out) as archive:
        assert sorted(archive.namelist()) == ["img.png", "img.txt"]


# --- StageProgress ---------------------------------------------------------


def test_stage_progress_fraction() -> None:
    assert StageProgress(index=1, total=4, name="p", phase="running").fraction == 0.0
    assert StageProgress(index=1, total=4, name="p", phase="done").fraction == 0.25
    assert StageProgress(index=4, total=4, name="p", phase="done").fraction == 1.0
    assert StageProgress(index=1, total=0, name="p", phase="done").fraction == 1.0


def test_stage_progress_lines() -> None:
    base = {"index": 2, "total": 5, "name": "panels"}
    assert "running…" in StageProgress(**base, phase="running").line
    assert "done (3 produced)" in StageProgress(**base, phase="done", produced=3).line
    assert "skipped (off)" in StageProgress(**base, phase="skipped", detail="off").line
    assert "error — boom" in StageProgress(**base, phase="error", detail="boom").line


# --- run_pipeline_stream ---------------------------------------------------


def _stage(name: str, flag: str) -> Stage:
    return Stage(
        name=name,
        summary=name,
        component=name,
        flag=flag,
        output=lambda ws, st: ws.root / name,
        run=lambda ctx: StageResult(name=name, output_dir=ctx.workspace.root),
    )


def _ctx(tmp_path: Path, **flags: bool) -> StageContext:
    settings = Settings().model_copy(update=flags)
    return StageContext(workspace=Workspace(root=tmp_path), settings=settings)


def test_stream_emits_running_then_done(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, run_panels=True, run_clean=True)
    stages = (_stage("panels", "run_panels"), _stage("clean", "run_clean"))
    produced = {"panels": 3, "clean": 5}

    def runner(stage: Stage, c: StageContext, *, force: bool = False) -> StageResult:
        return StageResult(
            name=stage.name, output_dir=c.workspace.root, produced=produced[stage.name]
        )

    events = list(run_pipeline_stream(ctx, stages=stages, runner=runner))

    assert [(e.name, e.phase) for e in events] == [
        ("panels", "running"),
        ("panels", "done"),
        ("clean", "running"),
        ("clean", "done"),
    ]
    assert events[1].produced == 3
    assert events[3].produced == 5


def test_stream_skips_disabled_stage(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, run_bubbles=False)
    stages = (_stage("bubbles", "run_bubbles"),)

    def runner(
        stage: Stage, c: StageContext, *, force: bool = False
    ) -> StageResult:  # pragma: no cover
        raise AssertionError("disabled stage must not run")

    events = list(run_pipeline_stream(ctx, stages=stages, runner=runner))

    assert len(events) == 1
    assert events[0].phase == "skipped"
    assert "disabled by run_bubbles" in events[0].detail


def test_stream_reports_runner_skip(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, run_panels=True)
    stages = (_stage("panels", "run_panels"),)

    def runner(stage: Stage, c: StageContext, *, force: bool = False) -> StageResult:
        return StageResult(
            name=stage.name, output_dir=c.workspace.root, skipped=True, reason="already complete"
        )

    events = list(run_pipeline_stream(ctx, stages=stages, runner=runner))

    assert [e.phase for e in events] == ["running", "skipped"]
    assert "already complete" in events[1].detail


def test_stream_stops_on_error(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, run_panels=True, run_clean=True)
    stages = (_stage("panels", "run_panels"), _stage("clean", "run_clean"))

    def runner(stage: Stage, c: StageContext, *, force: bool = False) -> StageResult:
        raise RuntimeError("boom")

    events = list(run_pipeline_stream(ctx, stages=stages, runner=runner))

    # first stage runs then errors; the second is never reached
    assert [(e.name, e.phase) for e in events] == [("panels", "running"), ("panels", "error")]
    assert events[1].detail == "boom"


def test_stream_error_falls_back_to_class_name(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, run_panels=True)
    stages = (_stage("panels", "run_panels"),)

    def runner(stage: Stage, c: StageContext, *, force: bool = False) -> StageResult:
        raise RuntimeError()  # empty message

    events = list(run_pipeline_stream(ctx, stages=stages, runner=runner))
    assert events[1].phase == "error"
    assert events[1].detail == "RuntimeError"


# --- promote_to_clean (manual_review rescue) -------------------------------


def test_promote_to_clean_rescues_upscales_and_invalidates_caption(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    from make_style_dataset.pipeline import DONE_MARKER
    from make_style_dataset.ui.service import promote_to_clean

    ws = Workspace(root=tmp_path)
    ws.ensure_base()
    settings = Settings(target_side=16, dataset_repeats=10, trigger_token="comicstyle")
    review = ws.manual_review
    review.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    def _img(name: str) -> Path:
        path = review / name
        Image.new("RGB", (8, 8), (120, 120, 120)).save(path)  # real PNG cv2 can decode
        return path

    keep = _img("panel_00.png")
    (review / "panel_00.reason.txt").write_text("too small after inpaint", encoding="utf-8")
    _img("panel_01.png")  # left un-selected
    (review / "panel_01.reason.txt").write_text("too small after inpaint", encoding="utf-8")

    dataset = ws.training_dir(settings.dataset_repeats, settings.trigger_token)
    dataset.mkdir(parents=True, exist_ok=True)
    marker = dataset / DONE_MARKER
    marker.write_text("caption\n", encoding="utf-8")

    # the "../escape.png" value is sanitised to a basename that doesn't exist -> skipped
    promoted = promote_to_clean(ws, ["panel_00.png", "../escape.png"], settings)

    assert promoted == 1
    out = ws.clean / "panel_00.png"
    assert out.is_file()
    rescued = cv2.imdecode(np.fromfile(out, np.uint8), cv2.IMREAD_COLOR)
    assert min(rescued.shape[:2]) == 16  # upscaled to target_side
    assert not keep.is_file()  # moved out of manual_review
    assert not (review / "panel_00.reason.txt").is_file()
    assert (review / "panel_01.png").is_file()  # un-selected panel stays
    assert not marker.exists()  # caption marker cleared -> next build re-captions


def test_promote_to_clean_no_selection_keeps_caption_marker(tmp_path: Path) -> None:
    from make_style_dataset.pipeline import DONE_MARKER
    from make_style_dataset.ui.service import promote_to_clean

    ws = Workspace(root=tmp_path)
    ws.ensure_base()
    settings = Settings(target_side=16)
    dataset = ws.training_dir(settings.dataset_repeats, settings.trigger_token)
    dataset.mkdir(parents=True, exist_ok=True)
    marker = dataset / DONE_MARKER
    marker.write_text("caption\n", encoding="utf-8")

    assert promote_to_clean(ws, [], settings) == 0
    assert marker.exists()  # nothing promoted -> marker untouched


# --- recaption_training_dir (VLM re-caption via proxy) ---------------------


class _StubCaptionClient:
    def caption(self, model: str, prompt: str, image_bytes: bytes) -> dict:
        return {"caption": "a hero on a hill"}


def test_recaption_training_dir_no_token() -> None:
    from make_style_dataset.ui.service import recaption_training_dir

    result = recaption_training_dir(Settings(hf_token=""), model="gemini-2.5-pro")
    assert (result.written, result.failed) == (0, 0)
    assert "No HF token" in result.errors[0]


def test_recaption_training_dir_no_dataset(tmp_path: Path) -> None:
    from make_style_dataset.ui.service import recaption_training_dir

    settings = Settings(
        hf_token="t", workspace=tmp_path, dataset_repeats=10, trigger_token="cmcstyle"
    )
    result = recaption_training_dir(settings, model="gemini-2.5-pro", client=_StubCaptionClient())
    assert result.written == 0
    assert "build the dataset" in result.errors[0].lower()


def test_recaption_training_dir_writes(tmp_path: Path) -> None:
    from make_style_dataset.ui.service import recaption_training_dir

    settings = Settings(
        hf_token="t", workspace=tmp_path, dataset_repeats=10, trigger_token="cmcstyle"
    )
    dataset = Workspace(root=tmp_path).training_dir(10, "cmcstyle")
    dataset.mkdir(parents=True)
    (dataset / "p.png").write_bytes(b"img")

    result = recaption_training_dir(
        settings, model="gemini-2.5-pro", style="rich", client=_StubCaptionClient()
    )

    assert result.written == 1
    assert (dataset / "p.txt").read_text(encoding="utf-8").strip() == "cmcstyle, a hero on a hill"
