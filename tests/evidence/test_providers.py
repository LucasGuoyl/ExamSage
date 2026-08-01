from __future__ import annotations

from hashlib import sha256
from io import IOBase
import json
from types import SimpleNamespace

from google import genai
import pytest

from exam_predictor.evidence.policy import EvidencePolicy
from exam_predictor.evidence.providers import (
    ANALYSIS_MAX_OUTPUT_TOKENS,
    AnalyzeSourcePartRequest,
    EvidencePartResult,
    EvidenceProviderError,
    EvidenceRouteIdentity,
    ProviderEvidenceAdapter,
)
from exam_predictor.providers import ModelRouting, ProviderCapabilities


_SECRET = "private-provider-response-sentinel-123456789"
_MODELS = ModelRouting(
    fast="fast-model",
    balanced="balanced-model",
    reasoning="reasoning-model",
    embedding="embedding-model",
)


def _request(
    content: bytes = b"bounded approved source bytes",
    *,
    media_type: str = "application/pdf",
    model_route: str = "balanced",
) -> AnalyzeSourcePartRequest:
    return AnalyzeSourcePartRequest(
        source_part_id="part_1234567890abcdef",
        relative_path="course/lecture.pdf",
        locator="pages 1-2",
        media_type=media_type,
        content_bytes=content,
        content_size_bytes=len(content),
        content_sha256=sha256(content).hexdigest(),
        model_route=model_route,
    )


class _OpenAIResponses:
    def __init__(self, response: object | None = None, error: BaseException | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._response = response or SimpleNamespace(output_text='{"headings": []}')
        self._error = error

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class _Files:
    def __init__(self) -> None:
        self.uploads: list[dict[str, object]] = []
        self.deletes: list[str] = []

    def create(self, **kwargs: object) -> object:
        self.uploads.append(kwargs)
        return SimpleNamespace(id="file_123")

    def upload(self, **kwargs: object) -> object:
        self.uploads.append(kwargs)
        return SimpleNamespace(name="files/provider-upload-123")

    def delete(self, file_id: str | None = None, *, name: str | None = None) -> None:
        self.deletes.append(file_id or name or "")


def _openai_provider(responses: _OpenAIResponses) -> SimpleNamespace:
    return SimpleNamespace(
        name="openai",
        capabilities=ProviderCapabilities(),
        models=_MODELS,
        client=SimpleNamespace(responses=responses, files=_Files()),
        inline_file_limit_bytes=10 * 1024 * 1024,
    )


class _GeminiModels:
    def __init__(self, response: object | None = None, error: BaseException | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._response = response or SimpleNamespace(text='{"headings": []}')
        self._error = error

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


def _gemini_provider(
    models: _GeminiModels,
    *,
    files: _Files | None = None,
    inline_limit: int = 10 * 1024 * 1024,
) -> SimpleNamespace:
    return SimpleNamespace(
        name="gemini",
        capabilities=ProviderCapabilities(),
        models=_MODELS,
        client=SimpleNamespace(models=models, files=files or _Files()),
        _genai=genai,
        inline_file_limit_bytes=inline_limit,
    )


def test_openai_adapter_sends_one_versioned_structured_request_and_exact_model():
    responses = _OpenAIResponses()
    request = _request(model_route="balanced")

    result = ProviderEvidenceAdapter(_openai_provider(responses)).analyze_source_part(request)

    assert isinstance(result, EvidencePartResult)
    assert result.model_id == "balanced-model"
    assert result.source_part_id == request.source_part_id
    assert result.locator == "pages 1-2"
    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call["model"] == "balanced-model"
    assert call["store"] is False
    assert call["max_output_tokens"] == ANALYSIS_MAX_OUTPUT_TOKENS
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["text"]["format"]["strict"] is True
    assert call["text"]["format"]["schema"]["additionalProperties"] is False
    content = call["input"][0]["content"]
    prompt_text = "\n".join(item.get("text", "") for item in content if item["type"] == "input_text")
    assert "source-analysis-v1" in prompt_text
    assert "BEGIN_UNTRUSTED_SOURCE" in prompt_text
    assert "END_UNTRUSTED_SOURCE" in prompt_text
    assert repr(request.content_bytes) not in repr(request)
    assert "content_bytes" not in request.model_dump()
    assert result.raw_output not in repr(result)
    assert "raw_output" not in result.model_dump()


def test_route_identity_is_exact_safe_and_available_before_provider_call():
    responses = _OpenAIResponses()
    adapter = ProviderEvidenceAdapter(_openai_provider(responses))

    identity = adapter.route_identity("balanced")

    assert identity == EvidenceRouteIdentity(
        provider="openai",
        model_route="balanced",
        model_id="balanced-model",
    )
    assert identity.model_dump() == {
        "provider": "openai",
        "model_route": "balanced",
        "model_id": "balanced-model",
    }
    assert responses.calls == []
    assert "client" not in repr(identity).casefold()
    assert "credential" not in repr(identity).casefold()


def test_gemini_adapter_uses_inline_bytes_json_schema_and_fast_route_once():
    models = _GeminiModels()
    request = _request(media_type="image/png", model_route="fast")

    result = ProviderEvidenceAdapter(_gemini_provider(models)).analyze_source_part(request)

    assert result.model_id == "fast-model"
    assert len(models.calls) == 1
    call = models.calls[0]
    assert call["model"] == "fast-model"
    config = call["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema["additionalProperties"] is False
    assert config.max_output_tokens == ANALYSIS_MAX_OUTPUT_TOKENS
    assert len(call["contents"]) == 3
    assert call["contents"][1].inline_data.data == request.content_bytes


@pytest.mark.parametrize("field", ["content_size_bytes", "content_sha256"])
def test_adapter_rechecks_private_bytes_and_makes_zero_calls_on_mismatch(field: str):
    responses = _OpenAIResponses()
    valid = _request()
    values = valid.model_dump()
    values["content_bytes"] = valid.content_bytes
    values[field] = 1 if field == "content_size_bytes" else "0" * 64
    forged = AnalyzeSourcePartRequest.model_construct(**values)

    with pytest.raises(EvidenceProviderError) as caught:
        ProviderEvidenceAdapter(_openai_provider(responses)).analyze_source_part(forged)

    assert caught.value.code == "provider_request_invalid"
    assert caught.value.retryable is False
    assert responses.calls == []


def test_adapter_enforces_policy_max_before_any_provider_call():
    responses = _OpenAIResponses()
    request = _request(b"x" * 1025)
    policy = EvidencePolicy(max_part_bytes=1024)

    with pytest.raises(EvidenceProviderError) as caught:
        ProviderEvidenceAdapter(_openai_provider(responses), policy=policy).analyze_source_part(request)

    assert caught.value.code == "provider_media_unsupported"
    assert responses.calls == []


def test_custom_provider_without_multimodal_capability_is_rejected_without_call():
    responses = _OpenAIResponses()
    provider = _openai_provider(responses)
    provider.name = "custom"
    provider.capabilities = ProviderCapabilities(vision=False, file_understanding=False)

    with pytest.raises(EvidenceProviderError) as caught:
        ProviderEvidenceAdapter(provider).analyze_source_part(_request())

    assert caught.value.code == "provider_media_unsupported"
    assert caught.value.retryable is False
    assert responses.calls == []


class _SecretProviderError(RuntimeError):
    def __init__(self, status_code: int, retry_after: object = None) -> None:
        super().__init__(_SECRET)
        self.status_code = status_code
        self.body = {"private": _SECRET}
        self.request = SimpleNamespace(url=f"https://example.invalid/?api_key={_SECRET}")
        headers = {} if retry_after is None else {"Retry-After": retry_after}
        self.response = SimpleNamespace(status_code=status_code, headers=headers, text=_SECRET)


def test_503_is_one_attempt_safe_retryable_and_has_no_exception_context_leak():
    provider_error = _SecretProviderError(503, "2")
    responses = _OpenAIResponses(error=provider_error)

    with pytest.raises(EvidenceProviderError) as caught:
        ProviderEvidenceAdapter(_openai_provider(responses)).analyze_source_part(_request())

    error = caught.value
    assert len(responses.calls) == 1
    assert error.code == "provider_unavailable"
    assert error.retryable is True
    assert error.retry_after_seconds == 2
    assert error.__cause__ is None
    assert error.__context__ is None
    serialized = json.dumps(error.to_event_payload(), sort_keys=True)
    for visible in (str(error), repr(error), serialized):
        assert _SECRET not in visible
        assert "example.invalid" not in visible


class _ExplosiveAttributesError(RuntimeError):
    @property
    def status_code(self):
        raise RuntimeError(_SECRET)

    @property
    def response(self):
        raise RuntimeError(_SECRET)

    @property
    def code(self):
        raise RuntimeError(_SECRET)

    @property
    def status(self):
        raise RuntimeError(_SECRET)


def _assert_no_raw_exception_state(error: EvidenceProviderError) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    assert not any(isinstance(value, BaseException) for value in vars(error).values())
    visible_state = json.dumps(
        {
            "args": error.args,
            "attributes": vars(error),
            "event": error.to_event_payload(),
        },
        default=repr,
        sort_keys=True,
    )
    assert _SECRET not in visible_state


def test_normalization_is_total_when_exception_properties_raise():
    responses = _OpenAIResponses(error=_ExplosiveAttributesError(_SECRET))

    with pytest.raises(EvidenceProviderError) as caught:
        ProviderEvidenceAdapter(_openai_provider(responses)).analyze_source_part(_request())

    assert caught.value.code == "provider_failed"
    assert caught.value.retryable is False
    assert len(responses.calls) == 1
    _assert_no_raw_exception_state(caught.value)


def test_retry_after_digit_length_is_bounded_before_integer_parsing():
    responses = _OpenAIResponses(error=_SecretProviderError(503, "9" * 5_000))

    with pytest.raises(EvidenceProviderError) as caught:
        ProviderEvidenceAdapter(_openai_provider(responses)).analyze_source_part(_request())

    assert caught.value.code == "provider_unavailable"
    assert caught.value.retryable is True
    assert caught.value.retry_after_seconds is None
    assert len(responses.calls) == 1
    _assert_no_raw_exception_state(caught.value)


def test_http_408_is_one_attempt_retryable_timeout():
    responses = _OpenAIResponses(error=_SecretProviderError(408))

    with pytest.raises(EvidenceProviderError) as caught:
        ProviderEvidenceAdapter(_openai_provider(responses)).analyze_source_part(_request())

    assert caught.value.code == "provider_timeout"
    assert caught.value.retryable is True
    assert len(responses.calls) == 1
    _assert_no_raw_exception_state(caught.value)


@pytest.mark.parametrize("retry_after", ["-1", "+1", " 1", "1.0", "999999", 2, _SECRET])
def test_retry_after_accepts_only_bounded_nonnegative_ascii_integer(retry_after: object):
    responses = _OpenAIResponses(error=_SecretProviderError(429, retry_after))

    with pytest.raises(EvidenceProviderError) as caught:
        ProviderEvidenceAdapter(_openai_provider(responses)).analyze_source_part(_request())

    assert caught.value.code == "provider_rate_limited"
    assert caught.value.retry_after_seconds is None


@pytest.mark.parametrize(
    ("error", "code", "retryable"),
    [
        (TimeoutError(_SECRET), "provider_timeout", True),
        (ConnectionError(_SECRET), "provider_connection_failed", True),
        (_SecretProviderError(401), "provider_credentials_invalid", False),
        (_SecretProviderError(404), "provider_model_unsupported", False),
        (_SecretProviderError(415), "provider_media_unsupported", False),
        (_SecretProviderError(400), "provider_request_invalid", False),
        (_SecretProviderError(418), "provider_failed", False),
    ],
)
def test_errors_map_to_stable_action_codes(error: BaseException, code: str, retryable: bool):
    responses = _OpenAIResponses(error=error)

    with pytest.raises(EvidenceProviderError) as caught:
        ProviderEvidenceAdapter(_openai_provider(responses)).analyze_source_part(_request())

    assert caught.value.code == code
    assert caught.value.retryable is retryable
    assert len(responses.calls) == 1
    assert _SECRET not in repr(caught.value)


def test_gemini_uploaded_bytes_are_deleted_after_terminal_failure():
    files = _Files()
    models = _GeminiModels(error=_SecretProviderError(503))
    provider = _gemini_provider(models, files=files, inline_limit=4)

    with pytest.raises(EvidenceProviderError):
        ProviderEvidenceAdapter(provider).analyze_source_part(_request(b"12345"))

    assert len(files.uploads) == 1
    assert isinstance(files.uploads[0]["file"], IOBase)
    assert files.deletes == ["files/provider-upload-123"]
    assert len(models.calls) == 1
