"""Unit tests for the caption stage.

Exercise the backend factory, the pure vocabulary parsing / tag selection /
caption building, and the orchestration with a fake tagger — so no onnxruntime
is needed. cv2/numpy/PIL are real dependencies and used directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from make_style_dataset.config import Settings
from make_style_dataset.stages import caption
from make_style_dataset.stages.base import StageContext
from make_style_dataset.workspace import Workspace


class FakeTagger:
    """Returns a fixed tag list regardless of the image (stands in for WD14)."""

    def __init__(self, tags: list[str] | None = None) -> None:
        self._tags = ["1girl", "solo", "long_hair"] if tags is None else tags

    def tag(self, image: np.ndarray) -> list[str]:
        del image
        return list(self._tags)


def _white_image(path: Path, width: int = 64, height: int = 48) -> Path:
    Image.new("RGB", (width, height), "white").save(path)
    return path


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {"workspace": tmp_path / "ws"}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _tags_csv(path: Path) -> Path:
    """Write a tiny selected_tags.csv (cols tag_id,name,category,count)."""
    path.write_text(
        "tag_id,name,category,count\n"
        "1,general,9,100\n"  # rating -> excluded
        "2,1girl,0,90\n"  # general
        "3,long_hair,0,80\n"  # general
        "4,solo_character_x,4,5\n",  # character -> excluded
        encoding="utf-8",
    )
    return path


# --- backend factory ------------------------------------------------------


def test_make_tagger_returns_wd14(tmp_path: Path) -> None:
    tagger = caption.make_tagger(_settings(tmp_path, caption_backend="wd14"))
    assert isinstance(tagger, caption.Wd14Tagger)


def test_make_tagger_case_insensitive(tmp_path: Path) -> None:
    tagger = caption.make_tagger(_settings(tmp_path, caption_backend=" WD14 "))
    assert isinstance(tagger, caption.Wd14Tagger)


def test_make_tagger_unknown_backend_raises(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="APP_CAPTION_BACKEND"):
        caption.make_tagger(_settings(tmp_path, caption_backend="joycaption"))


def test_caption_defaults(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert settings.caption_backend == "wd14"
    assert settings.caption_threshold == 0.35


# --- pure vocabulary / tag selection --------------------------------------


def test_parse_general_vocabulary_keeps_only_general(tmp_path: Path) -> None:
    names, mask = caption.parse_general_vocabulary(_tags_csv(tmp_path / "tags.csv"))
    assert names == ["1girl", "long_hair"]  # rating + character excluded
    assert mask.tolist() == [False, True, True, False]


def test_select_general_tags_threshold_and_mask() -> None:
    # Full per-row probs (4 rows); mask keeps rows 1 and 2 (the general ones).
    probs = np.array([0.99, 0.80, 0.20, 0.95], dtype=np.float32)
    mask = np.array([False, True, True, False])
    general_names = ["1girl", "long_hair"]
    selected = caption.select_general_tags(probs, mask, general_names, threshold=0.5)
    assert selected == ["1girl"]  # long_hair (0.20) is below threshold


def test_select_general_tags_filters_around_threshold() -> None:
    probs = np.array([0.36, 0.34], dtype=np.float32)
    mask = np.array([True, True])
    selected = caption.select_general_tags(probs, mask, ["keep", "drop"], threshold=0.35)
    assert selected == ["keep"]


# --- caption building -----------------------------------------------------


def test_build_caption_trigger_first() -> None:
    assert caption.build_caption("comicstyle", ["1girl", "solo"]) == "comicstyle, 1girl, solo"


def test_build_caption_dedupes_trigger_case_insensitive() -> None:
    assert caption.build_caption("Comic", ["comic", "1girl"]) == "Comic, 1girl"


def test_build_caption_dedupes_repeated_tags() -> None:
    assert caption.build_caption("style", ["1girl", "1girl", "solo"]) == "style, 1girl, solo"


def test_build_caption_no_tags_is_just_trigger() -> None:
    assert caption.build_caption("style", []) == "style"


# --- wd14 preprocessing ---------------------------------------------------


def test_wd14_input_shape_dtype_and_bgr_preserved() -> None:
    image = np.zeros((30, 60, 3), dtype=np.uint8)
    image[:, :, 0] = 200  # blue channel set (BGR) -> must stay channel 0
    feed = caption.wd14_input(image, 448)
    assert feed.shape == (1, 448, 448, 3)  # NHWC
    assert feed.dtype == np.float32
    assert feed.max() <= 255.0  # not normalized to [0,1]
    # Padding is white (255) on the short axis; center keeps the blue channel.
    assert feed[0, 224, 224, 0] == 200.0


# --- I/O helpers ----------------------------------------------------------


def test_decode_bgr_and_empty(tmp_path: Path) -> None:
    _white_image(tmp_path / "p.png", 20, 10)
    img = caption._decode_bgr(tmp_path / "p.png")
    assert img is not None and img.shape == (10, 20, 3)
    empty = tmp_path / "e.png"
    empty.touch()
    assert caption._decode_bgr(empty) is None


def test_write_caption_appends_newline(tmp_path: Path) -> None:
    caption._write_caption("style, 1girl", tmp_path / "c.txt")
    assert (tmp_path / "c.txt").read_text(encoding="utf-8") == "style, 1girl\n"


def test_iter_images_filters_and_sorts(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    clean.mkdir()
    _white_image(clean / "b.png")
    _white_image(clean / "a.jpg")
    (clean / "notes.txt").write_text("x", encoding="utf-8")
    assert [p.name for p in caption.iter_images(clean)] == ["a.jpg", "b.png"]


def test_iter_images_missing_dir(tmp_path: Path) -> None:
    assert caption.iter_images(tmp_path / "nope") == []


# --- caption_image orchestration ------------------------------------------


def test_caption_image_writes_png_and_txt(tmp_path: Path) -> None:
    _white_image(tmp_path / "panel.png")
    out = tmp_path / "out"
    out.mkdir()
    written = caption.caption_image(
        tmp_path / "panel.png",
        tagger=FakeTagger(["1girl", "solo"]),
        trigger="mystyle",
        out_dir=out,
    )
    assert written is True
    assert (out / "panel.png").is_file()
    assert (out / "panel.txt").read_text(encoding="utf-8") == "mystyle, 1girl, solo\n"


def test_caption_image_unreadable_skips(tmp_path: Path) -> None:
    (tmp_path / "bad.png").write_text("garbage", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    written = caption.caption_image(
        tmp_path / "bad.png", tagger=FakeTagger(), trigger="s", out_dir=out
    )
    assert written is False
    assert not (out / "bad.png").exists()


# --- run() orchestration --------------------------------------------------


def test_run_captions_into_kohya_layout(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, dataset_repeats=20, trigger_token="mystyle")
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.clean.mkdir(parents=True, exist_ok=True)
    _white_image(ws.clean / "p1.png")
    _white_image(ws.clean / "p2.png")
    monkeypatch.setattr(caption, "make_tagger", lambda _s: FakeTagger(["1girl", "solo"]))

    ctx = StageContext(workspace=ws, settings=settings)
    result = caption.run(ctx)
    assert result.produced == 2
    training = ws.training_dir(20, "mystyle")
    assert result.output_dir == training
    assert sorted(p.name for p in training.iterdir()) == [
        "p1.png",
        "p1.txt",
        "p2.png",
        "p2.txt",
    ]
    assert (training / "p1.txt").read_text(encoding="utf-8") == "mystyle, 1girl, solo\n"


def test_run_idempotent(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, dataset_repeats=5, trigger_token="style")
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.clean.mkdir(parents=True, exist_ok=True)
    _white_image(ws.clean / "p.png")
    monkeypatch.setattr(caption, "make_tagger", lambda _s: FakeTagger(["1girl"]))

    ctx = StageContext(workspace=ws, settings=settings)
    assert caption.run(ctx).produced == 1
    assert caption.run(ctx).produced == 1
    training = ws.training_dir(5, "style")
    assert len(list(training.iterdir())) == 2  # png + txt, deterministic names


def test_run_skips_unreadable_image(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, dataset_repeats=10, trigger_token="style")
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.clean.mkdir(parents=True, exist_ok=True)
    _white_image(ws.clean / "ok.png")
    (ws.clean / "bad.png").write_text("garbage", encoding="utf-8")  # undecodable
    monkeypatch.setattr(caption, "make_tagger", lambda _s: FakeTagger(["1girl"]))

    result = caption.run(StageContext(workspace=ws, settings=settings))
    assert result.produced == 1  # only the readable image
    training = ws.training_dir(10, "style")
    assert sorted(p.name for p in training.iterdir()) == ["ok.png", "ok.txt"]
