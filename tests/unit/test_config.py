"""Tests for make_style_dataset.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from make_style_dataset.config import Settings, get_settings


def test_defaults() -> None:
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.environment == "development"
    assert settings.debug is False
    assert settings.sentry_dsn == ""


def test_pipeline_defaults() -> None:
    settings = get_settings()
    assert settings.workspace == Path("workspace")
    assert settings.trigger_token == "comicstyle"
    assert settings.dataset_repeats == 10
    assert settings.min_panel_area == 10_000
    assert settings.dedup_hamming_distance == 6
    assert settings.min_side_px == 256
    assert settings.target_side == 1024
    assert settings.run_panels is True
    assert settings.run_caption is True


def test_reads_pipeline_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_WORKSPACE", "/tmp/ws")
    monkeypatch.setenv("APP_TRIGGER_TOKEN", "mystyle")
    monkeypatch.setenv("APP_MIN_SIDE_PX", "256")
    monkeypatch.setenv("APP_RUN_BUBBLES", "false")
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.workspace == Path("/tmp/ws")
    assert settings.trigger_token == "mystyle"
    assert settings.min_side_px == 256
    assert settings.run_bubbles is False


def test_singleton_is_cached() -> None:
    assert get_settings() is get_settings()


def test_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENVIRONMENT", "production")
    monkeypatch.setenv("APP_DEBUG", "true")
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.environment == "production"
    assert settings.debug is True


def test_vlm_caption_defaults_and_hf_token_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_value")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.vlm_model == "gemini-2.5-flash"
    assert settings.vlm_prompt_style == "rich"
    assert settings.vlm_concurrency == 8
    assert settings.hf_token == "hf_test_value"  # read from non-prefixed HF_TOKEN
    get_settings.cache_clear()
