"""Tests for the VLM re-caption mechanism and the Gemini proxy payload parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from make_style_dataset.proxy import GeminiProxyClient, parse_proxy_payload
from make_style_dataset.vlm_caption import (
    NO_STYLE,
    build_prompt,
    normalize_caption,
    recaption_dataset,
)


class StubClient:
    """A CaptionClient that returns queued results in call order."""

    def __init__(self, results: list[dict]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, int]] = []

    def caption(self, model: str, prompt: str, image_bytes: bytes) -> dict:
        self.calls.append((model, len(image_bytes)))
        return self.results.pop(0) if self.results else {"caption": "fallback"}


# --- proxy payload parsing -------------------------------------------------


def test_parse_proxy_payload_double_encoded() -> None:
    inner = json.dumps({"caption": "hi", "in_tokens": 5})
    assert parse_proxy_payload(json.dumps([inner])) == {"caption": "hi", "in_tokens": 5}


def test_parse_proxy_payload_empty_and_bad() -> None:
    assert parse_proxy_payload(None)["error"] == "empty_response"
    assert parse_proxy_payload("")["error"] == "empty_response"
    assert parse_proxy_payload("not json at all")["error"] == "parse_failed"


# --- prompt building -------------------------------------------------------


def test_build_prompt_both_styles_trigger_first_and_no_style() -> None:
    opt = build_prompt("cmcstyle", "optimal")
    rich = build_prompt("cmcstyle", "rich")
    for text in (opt, rich):
        assert "cmcstyle, " in text
        assert NO_STYLE in text
    assert "1-3 sentences" in opt
    assert "3-6 sentences" in rich  # rich asks for more


def test_build_prompt_uses_the_given_trigger() -> None:
    assert "p3r5on, " in build_prompt("p3r5on", "optimal")


# --- caption normalization -------------------------------------------------


def test_normalize_caption_prepends_or_dedupes_trigger() -> None:
    assert normalize_caption("cmcstyle, a man", "cmcstyle") == "cmcstyle, a man"
    assert normalize_caption("CMCSTYLE,  a man", "cmcstyle") == "cmcstyle, a man"
    assert normalize_caption("a woman", "cmcstyle") == "cmcstyle, a woman"
    assert normalize_caption("  multi\nline   text ", "t") == "t, multi line text"
    assert normalize_caption("", "t") == "t,"


# --- batch re-caption ------------------------------------------------------


def test_recaption_writes_normalized_txt(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"img-a")
    (tmp_path / "b.png").write_bytes(b"img-b")
    client = StubClient([{"caption": "cmcstyle, a man"}, {"caption": "a woman"}])

    result = recaption_dataset(
        tmp_path,
        trigger="cmcstyle",
        model="gemini-2.5-flash",
        style="optimal",
        client=client,
        max_workers=1,
    )

    assert (result.written, result.failed) == (2, 0)
    assert (tmp_path / "a.txt").read_text(encoding="utf-8").strip() == "cmcstyle, a man"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8").strip() == "cmcstyle, a woman"
    assert client.calls and client.calls[0][0] == "gemini-2.5-flash"


def test_recaption_counts_non_transient_error_and_skips_write(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"x")
    client = StubClient([{"error": "http_400", "body": "bad request"}])

    result = recaption_dataset(
        tmp_path, trigger="cmcstyle", model="m", style="optimal", client=client, max_workers=1
    )

    assert (result.written, result.failed) == (0, 1)
    assert not (tmp_path / "a.txt").exists()
    assert "a.png" in result.errors[0]
    assert len(client.calls) == 1  # http_400 is not retried


def test_recaption_retries_transient_then_succeeds(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"x")
    client = StubClient([{"error": "http_503"}, {"caption": "a knight"}])
    slept: list[float] = []

    result = recaption_dataset(
        tmp_path,
        trigger="t",
        model="m",
        style="rich",
        client=client,
        max_workers=1,
        sleep=slept.append,
    )

    assert result.written == 1
    assert (tmp_path / "a.txt").read_text(encoding="utf-8").strip() == "t, a knight"
    assert slept  # backed off once before the retry
    assert len(client.calls) == 2


def test_recaption_retries_on_client_exception(tmp_path: Path) -> None:
    """A network exception from the real client must be caught + retried, not crash."""
    (tmp_path / "a.png").write_bytes(b"x")

    class FlakyRaise:
        def __init__(self) -> None:
            self.n = 0

        def caption(self, model: str, prompt: str, image_bytes: bytes) -> dict:
            self.n += 1
            if self.n == 1:
                raise TimeoutError("write operation timed out")
            return {"caption": "a knight"}

    client = FlakyRaise()
    slept: list[float] = []
    result = recaption_dataset(
        tmp_path,
        trigger="t",
        model="m",
        style="rich",
        client=client,
        max_workers=1,
        sleep=slept.append,
    )

    assert result.written == 1
    assert client.n == 2  # raised once, retried, then succeeded
    assert (tmp_path / "a.txt").read_text(encoding="utf-8").strip() == "t, a knight"


def test_gemini_proxy_client_requires_https() -> None:
    with pytest.raises(ValueError, match="https"):
        GeminiProxyClient("t", url="http://insecure")
    GeminiProxyClient("tok")  # default https URL constructs fine
