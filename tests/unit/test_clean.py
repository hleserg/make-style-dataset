"""Unit tests for the clean (dedup / size-filter / upscale) stage.

The whole stage is pure CPU code (pHash via cv2.dct, dedup policy over ints,
Lanczos upscale), so everything here runs against the real cv2/numpy/PIL with no
heavy backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from make_style_dataset.config import Settings
from make_style_dataset.stages import clean
from make_style_dataset.stages.base import StageContext
from make_style_dataset.workspace import Workspace


def _save(array: np.ndarray, path: Path) -> Path:
    Image.fromarray(array[:, :, ::-1], mode="RGB").save(path)  # array is BGR
    return path


def _noise(seed: int, width: int, height: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {"workspace": tmp_path / "ws"}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# --- perceptual hash + hamming --------------------------------------------


def test_phash_is_stable_and_64_bit() -> None:
    image = _noise(0, 80, 60)
    assert clean.phash(image) == clean.phash(image.copy())  # deterministic
    assert 0 <= clean.phash(image) < (1 << 64)


def test_phash_differs_for_different_images() -> None:
    a = clean.phash(_noise(1, 80, 80))
    b = clean.phash(_noise(2, 80, 80))
    assert clean.hamming_distance(a, b) > 6  # unrelated images are far apart


def test_hamming_distance() -> None:
    assert clean.hamming_distance(0b0000, 0b0000) == 0
    assert clean.hamming_distance(0b1011, 0b0001) == 2


# --- dedup policy (tested off the threshold boundary) ----------------------


def test_dedup_drops_near_duplicates() -> None:
    a, b, c = Path("a.png"), Path("b.png"), Path("c.png")
    items = [(a, 0b0), (b, 0b111), (c, 0b1111111111)]  # b is 3 bits from a, c is 10
    kept, dropped = clean.dedup(items, threshold=6)
    assert kept == [a, c]  # b (distance 3 < 6) dropped; c (distance 10) kept
    assert dropped == [b]


def test_dedup_threshold_zero_keeps_all() -> None:
    a, b = Path("a.png"), Path("b.png")
    kept, dropped = clean.dedup([(a, 5), (b, 5)], threshold=0)
    assert kept == [a, b] and dropped == []


# --- size / quality transforms --------------------------------------------


def test_short_side() -> None:
    assert clean.short_side(np.zeros((40, 90, 3), dtype=np.uint8)) == 40


def test_upscale_to_enlarges_short_side() -> None:
    out = clean.upscale_to(np.zeros((100, 200, 3), dtype=np.uint8), 150)
    assert clean.short_side(out) == 150
    assert out.shape[1] == 300  # aspect ratio preserved (200 * 1.5)


def test_upscale_to_noop_when_large_enough() -> None:
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    assert clean.upscale_to(image, 150) is image


def test_denoise_preserves_shape() -> None:
    image = _noise(3, 64, 64)
    out = clean.denoise(image)
    assert out.shape == image.shape and out.dtype == np.uint8


def test_format_clean_report() -> None:
    report = clean.format_clean_report(kept=5, deduped=2, dropped_small=1, upscaled=3)
    assert report == "clean: kept 5, deduped 2, dropped-small 1, upscaled 3"


# --- I/O ------------------------------------------------------------------


def test_decode_bgr_and_empty(tmp_path: Path) -> None:
    _save(_noise(0, 20, 10), tmp_path / "p.png")
    img = clean._decode_bgr(tmp_path / "p.png")
    assert img is not None and img.shape == (10, 20, 3)
    empty = tmp_path / "e.png"
    empty.touch()
    assert clean._decode_bgr(empty) is None


def test_iter_panels_filters_and_sorts(tmp_path: Path) -> None:
    inp = tmp_path / "inp"
    inp.mkdir()
    _save(_noise(0, 10, 10), inp / "b.png")
    _save(_noise(1, 10, 10), inp / "a.png")
    (inp / "notes.txt").write_text("x", encoding="utf-8")
    assert [p.name for p in clean.iter_panels(inp)] == ["a.png", "b.png"]


# --- run() orchestration --------------------------------------------------


def test_run_dedups_filters_and_upscales(tmp_path: Path) -> None:
    settings = _settings(tmp_path, dedup_hamming_distance=6, min_side_px=64, target_side=128)
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.inpainted.mkdir(parents=True, exist_ok=True)

    base = _noise(0, 200, 200)
    _save(base, ws.inpainted / "a.png")  # kept (>= target, no upscale)
    _save(base, ws.inpainted / "b.png")  # identical to a -> deduped
    _save(_noise(1, 100, 100), ws.inpainted / "c.png")  # distinct, < target -> upscaled
    _save(_noise(2, 40, 40), ws.inpainted / "tiny.png")  # < min_side -> manual_review

    result = clean.run(StageContext(workspace=ws, settings=settings))

    assert result.produced == 2  # a + c
    assert sorted(p.name for p in ws.clean.iterdir()) == ["a.png", "c.png"]
    assert "b.png" not in {p.name for p in ws.clean.iterdir()}  # deduped
    # tiny routed out, with a reason note.
    assert (ws.manual_review / "tiny.png").is_file()
    assert "too small" in (ws.manual_review / "tiny.reason.txt").read_text(encoding="utf-8")
    # c was upscaled so its shorter side reached target_side.
    with Image.open(ws.clean / "c.png") as img:
        assert min(img.size) == 128


def test_run_denoise_flag(tmp_path: Path) -> None:
    settings = _settings(tmp_path, min_side_px=16, target_side=16, clean_denoise=True)
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.inpainted.mkdir(parents=True, exist_ok=True)
    _save(_noise(0, 64, 64), ws.inpainted / "p.png")

    result = clean.run(StageContext(workspace=ws, settings=settings))
    assert result.produced == 1
    assert (ws.clean / "p.png").is_file()


def test_run_empty_input(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    result = clean.run(StageContext(workspace=ws, settings=settings))
    assert result.produced == 0
    assert ws.clean.is_dir()


def test_run_skips_unreadable_panel(tmp_path: Path) -> None:
    settings = _settings(tmp_path, min_side_px=16, target_side=16)
    ws = Workspace(root=settings.workspace)
    ws.ensure_base()
    ws.inpainted.mkdir(parents=True, exist_ok=True)
    _save(_noise(0, 64, 64), ws.inpainted / "ok.png")
    (ws.inpainted / "bad.png").write_text("garbage", encoding="utf-8")  # undecodable

    result = clean.run(StageContext(workspace=ws, settings=settings))
    assert result.produced == 1  # only the readable panel
    assert [p.name for p in ws.clean.iterdir()] == ["ok.png"]
