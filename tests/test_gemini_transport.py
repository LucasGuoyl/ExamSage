from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pypdf import PdfWriter

from exam_predictor.cloud_analyzer import _analyze_one
from exam_predictor.providers import (
    BaseProvider,
    GeminiProvider,
    ModelRouting,
    ProviderError,
    _is_retryable_provider_error,
)


def _provider_without_sdk() -> GeminiProvider:
    provider = object.__new__(GeminiProvider)
    BaseProvider.__init__(
        provider,
        "test-key",
        ModelRouting("fast", "balanced", "reasoning", "embedding"),
    )
    return provider


def _analysis_response(text: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=20,
        ),
    )


def test_gemini_small_pdf_uses_inline_transport(tmp_path: Path):
    source = tmp_path / "lecture.pdf"
    source.write_bytes(b"small transient pdf")
    provider = _provider_without_sdk()
    calls: list[tuple[Path, str, str]] = []

    def inline(path: Path, prompt: str, mime_type: str, model: str):
        calls.append((path, mime_type, model))
        return _analysis_response()

    provider._generate_inline_file = inline
    provider._upload_file = lambda path: pytest.fail("Files API upload was not expected")

    assert provider.analyze_file(source, "analyze") == "ok"
    assert calls == [(source, "application/pdf", "balanced")]


def test_gemini_503_falls_back_to_fast_multimodal_model(tmp_path: Path):
    source = tmp_path / "lecture.pdf"
    source.write_bytes(b"small transient pdf")
    provider = _provider_without_sdk()
    calls: list[str] = []

    class BusyModelError(RuntimeError):
        code = 503

    def inline(path: Path, prompt: str, mime_type: str, model: str):
        calls.append(model)
        if model == "balanced":
            raise BusyModelError("high demand")
        return _analysis_response("fallback succeeded")

    provider._generate_inline_file = inline

    assert provider.analyze_file(source, "analyze") == "fallback succeeded"
    assert calls == ["balanced", "fast"]


def test_gemini_file_error_exposes_stage_file_and_original_error(tmp_path: Path):
    source = tmp_path / "lecture.pdf"
    source.write_bytes(b"small transient pdf")
    provider = _provider_without_sdk()
    original = httpx.RemoteProtocolError("server disconnected")

    def fail_inline(path: Path, prompt: str, mime_type: str, model: str):
        raise original

    provider._generate_inline_file = fail_inline

    with pytest.raises(ProviderError) as caught:
        provider.analyze_file(source, "analyze")

    message = str(caught.value)
    assert "inline document request" in message
    assert "lecture.pdf" in message
    assert "RemoteProtocolError" in message
    assert "RetryError" not in message
    assert caught.value.__cause__ is original


def test_remote_protocol_and_retryable_http_statuses_are_transient():
    assert _is_retryable_provider_error(httpx.RemoteProtocolError("disconnected"))
    assert _is_retryable_provider_error(SimpleNamespace(status_code=429))
    assert _is_retryable_provider_error(SimpleNamespace(status_code=503))
    assert _is_retryable_provider_error(SimpleNamespace(code=500))
    assert not _is_retryable_provider_error(SimpleNamespace(status_code=400))


def test_gemini_pdf_is_split_to_inline_sized_pieces(tmp_path: Path):
    source = tmp_path / "large-lecture.pdf"
    writer = PdfWriter()
    for _ in range(4):
        writer.add_blank_page(width=612, height=792)
    with source.open("wb") as stream:
        writer.write(stream)

    one_page = tmp_path / "one-page.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with one_page.open("wb") as stream:
        writer.write(stream)

    piece_limit = (one_page.stat().st_size + source.stat().st_size) // 2

    class SizedGeminiLikeProvider:
        inline_file_limit_bytes = piece_limit

        def __init__(self):
            self.seen_sizes: list[int] = []

        def analyze_file(self, path: Path, prompt: str) -> str:
            self.seen_sizes.append(path.stat().st_size)
            return json.dumps({
                "document_title": "Lecture",
                "detected_language": "en",
                "course_name": "Mechanics",
                "material_kind": "lecture",
                "sections": [{
                    "locator": "page",
                    "title": "Topic",
                    "text": "Content",
                    "visual_description": "",
                }],
                "exam_questions": [],
                "syllabus_points": [],
                "warnings": [],
            })

    provider = SizedGeminiLikeProvider()
    pieces_dir = tmp_path / "pieces"
    pieces_dir.mkdir()
    document = _analyze_one(provider, source, pieces_dir)

    assert document.course_name == "Mechanics"
    assert len(provider.seen_sizes) > 1
    assert all(size <= piece_limit for size in provider.seen_sizes)
