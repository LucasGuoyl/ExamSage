"""One-attempt, provider-neutral multimodal evidence contracts."""

from __future__ import annotations

import base64
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any, Literal, Protocol

from pydantic import ConfigDict, Field, field_validator, model_validator

from exam_predictor.evidence.models import EvidenceFrozenModel, validate_safe_evidence_text
from exam_predictor.evidence.policy import EvidencePolicy
from exam_predictor.evidence.prompts import (
    SOURCE_ANALYSIS_JSON_SCHEMA,
    source_analysis_prefix,
    source_analysis_suffix,
)
from exam_predictor.providers import BaseProvider
from exam_predictor.workspace.models import normalize_relative_path


ANALYSIS_MAX_OUTPUT_TOKENS = 4096
_DEFAULT_INLINE_FILE_LIMIT_BYTES = 10 * 1024 * 1024
_MAX_RETRY_AFTER_DIGITS = 10
_SAFE_ERROR_MESSAGES = {
    "provider_timeout": "The provider request timed out.",
    "provider_rate_limited": "The provider is rate limiting requests.",
    "provider_unavailable": "The provider is temporarily unavailable.",
    "provider_connection_failed": "The provider connection failed.",
    "provider_credentials_invalid": "The provider credentials are invalid.",
    "provider_model_unsupported": "The selected provider model is unsupported.",
    "provider_media_unsupported": "The provider cannot analyze this prepared source part.",
    "provider_request_invalid": "The provider request is invalid.",
    "provider_failed": "The provider could not analyze this source part.",
}


class _PrivateEvidenceModel(EvidenceFrozenModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )


class AnalyzeSourcePartRequest(_PrivateEvidenceModel):
    """Hash-bound prepared bytes for exactly one provider analysis call."""

    source_part_id: str = Field(min_length=1, max_length=128)
    relative_path: str
    locator: str = Field(min_length=1, max_length=512)
    media_type: str = Field(min_length=3, max_length=128)
    content_bytes: bytes = Field(repr=False, exclude=True)
    content_size_bytes: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_route: Literal["fast", "balanced"] = "balanced"

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return normalize_relative_path(value)

    @field_validator("locator")
    @classmethod
    def validate_locator(cls, value: str) -> str:
        return validate_safe_evidence_text(value)

    @field_validator("media_type")
    @classmethod
    def validate_media_type(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if "/" not in normalized or any(character.isspace() for character in normalized):
            raise ValueError("media type is invalid")
        return normalized

    @model_validator(mode="after")
    def validate_private_bytes(self) -> AnalyzeSourcePartRequest:
        if len(self.content_bytes) != self.content_size_bytes:
            raise ValueError("prepared byte length does not match its declaration")
        if sha256(self.content_bytes).hexdigest() != self.content_sha256:
            raise ValueError("prepared byte hash does not match its declaration")
        return self


class EvidencePartResult(_PrivateEvidenceModel):
    """Raw provider JSON plus the exact route identity used to produce it."""

    source_part_id: str
    locator: str
    provider: Literal["openai", "gemini"]
    model_id: str
    prompt_version: str
    raw_output: str = Field(repr=False, exclude=True)


class EvidenceRouteIdentity(_PrivateEvidenceModel):
    """Safe provider/model identity available before an external call."""

    provider: Literal["openai", "gemini"]
    model_route: Literal["fast", "balanced"]
    model_id: str = Field(min_length=1, max_length=256)

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("model ID must not be empty")
        return validate_safe_evidence_text(normalized)


class EvidenceProviderError(RuntimeError):
    """A normalized provider failure that retains no raw SDK exception state."""

    def __init__(
        self,
        code: str,
        *,
        retryable: bool,
        retry_after_seconds: int | None = None,
    ) -> None:
        if code not in _SAFE_ERROR_MESSAGES:
            raise ValueError("unsupported evidence provider error code")
        self.code = code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        super().__init__(_SAFE_ERROR_MESSAGES[code])

    def __repr__(self) -> str:
        return (
            f"EvidenceProviderError(code={self.code!r}, retryable={self.retryable!r}, "
            f"retry_after_seconds={self.retry_after_seconds!r})"
        )

    def to_event_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "code": self.code,
            "retryable": self.retryable,
        }
        if self.retry_after_seconds is not None:
            payload["retry_after_seconds"] = self.retry_after_seconds
        return payload


class EvidenceProvider(Protocol):
    def analyze_source_part(self, request: AnalyzeSourcePartRequest) -> EvidencePartResult:
        """Analyze one verified prepared source part with one provider attempt."""


class ProviderEvidenceAdapter:
    """Wrap an already-connected provider without creating another client or credential."""

    def __init__(
        self,
        provider: BaseProvider,
        *,
        policy: EvidencePolicy = EvidencePolicy(),
    ) -> None:
        self._provider = provider
        self._policy = policy

    def analyze_source_part(self, request: AnalyzeSourcePartRequest) -> EvidencePartResult:
        self._verify_request(request)
        route = self.route_identity(request.model_route)
        provider_name = route.provider
        model_id = route.model_id

        try:
            if provider_name == "openai":
                raw_output = self._analyze_openai(request, model_id)
            else:
                raw_output = self._analyze_gemini(request, model_id)
        except Exception as provider_exception:
            normalized = _normalize_provider_exception(provider_exception, self._policy)
        else:
            return EvidencePartResult(
                source_part_id=request.source_part_id,
                locator=request.locator,
                provider=provider_name,
                model_id=model_id,
                prompt_version=self._policy.prompt_version,
                raw_output=raw_output,
            )
        raise EvidenceProviderError(
            normalized[0],
            retryable=normalized[1],
            retry_after_seconds=normalized[2],
        ) from None

    def route_identity(self, model_route: str) -> EvidenceRouteIdentity:
        """Resolve a safe immutable route without invoking the provider."""

        provider_name = getattr(self._provider, "name", "")
        capabilities = getattr(self._provider, "capabilities", None)
        if (
            provider_name not in {"openai", "gemini"}
            or capabilities is None
            or not (
                bool(getattr(capabilities, "vision", False))
                or bool(getattr(capabilities, "file_understanding", False))
            )
        ):
            raise EvidenceProviderError("provider_media_unsupported", retryable=False)
        if model_route not in {"fast", "balanced"}:
            raise EvidenceProviderError("provider_model_unsupported", retryable=False)
        model_id = getattr(getattr(self._provider, "models", None), model_route, None)
        if not isinstance(model_id, str) or not model_id.strip():
            raise EvidenceProviderError("provider_model_unsupported", retryable=False)
        return EvidenceRouteIdentity(
            provider=provider_name,
            model_route=model_route,
            model_id=model_id,
        )

    def _verify_request(self, request: AnalyzeSourcePartRequest) -> None:
        content = request.content_bytes
        if (
            not isinstance(content, bytes)
            or len(content) != request.content_size_bytes
            or sha256(content).hexdigest() != request.content_sha256
        ):
            raise EvidenceProviderError("provider_request_invalid", retryable=False)
        if len(content) > self._policy.max_part_bytes:
            raise EvidenceProviderError("provider_media_unsupported", retryable=False)

    def _analyze_openai(self, request: AnalyzeSourcePartRequest, model_id: str) -> str:
        client = self._provider.client
        prefix = source_analysis_prefix(
            locator=request.locator,
            prompt_version=self._policy.prompt_version,
        )
        content: list[dict[str, object]] = [{"type": "input_text", "text": prefix}]
        uploaded_id: str | None = None
        try:
            inline_limit = _inline_limit(self._provider)
            if request.content_size_bytes <= inline_limit:
                encoded = base64.b64encode(request.content_bytes).decode("ascii")
                data_url = f"data:{request.media_type};base64,{encoded}"
                if request.media_type.startswith("image/"):
                    content.append({"type": "input_image", "image_url": data_url})
                else:
                    content.append(
                        {
                            "type": "input_file",
                            "filename": _provider_filename(request),
                            "file_data": data_url,
                        }
                    )
            else:
                uploaded = client.files.create(
                    file=(_provider_filename(request), request.content_bytes, request.media_type),
                    purpose="user_data",
                )
                candidate_id = getattr(uploaded, "id", None)
                if not isinstance(candidate_id, str) or not candidate_id:
                    raise _InvalidProviderResponse
                uploaded_id = candidate_id
                content.append({"type": "input_file", "file_id": uploaded_id})
            content.append({"type": "input_text", "text": source_analysis_suffix()})
            response = client.responses.create(
                model=model_id,
                input=[{"role": "user", "content": content}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "examsage_evidence_part",
                        "schema": SOURCE_ANALYSIS_JSON_SCHEMA,
                        "strict": True,
                    }
                },
                max_output_tokens=ANALYSIS_MAX_OUTPUT_TOKENS,
                store=False,
            )
            raw_output = getattr(response, "output_text", None)
            if not isinstance(raw_output, str):
                raise _InvalidProviderResponse
            return raw_output
        finally:
            if uploaded_id is not None:
                try:
                    client.files.delete(uploaded_id)
                except Exception:
                    pass

    def _analyze_gemini(self, request: AnalyzeSourcePartRequest, model_id: str) -> str:
        client = self._provider.client
        types = self._provider._genai.types
        uploaded: object | None = None
        uploaded_name: str | None = None
        try:
            if request.content_size_bytes <= _inline_limit(self._provider):
                source_part = types.Part.from_bytes(
                    data=request.content_bytes,
                    mime_type=request.media_type,
                )
            else:
                stream = BytesIO(request.content_bytes)
                stream.name = _provider_filename(request)
                try:
                    uploaded = client.files.upload(
                        file=stream,
                        config={"mime_type": request.media_type},
                    )
                finally:
                    stream.close()
                candidate_name = getattr(uploaded, "name", None)
                if not isinstance(candidate_name, str) or not candidate_name:
                    raise _InvalidProviderResponse
                uploaded_name = candidate_name
                source_part = uploaded
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=SOURCE_ANALYSIS_JSON_SCHEMA,
                max_output_tokens=ANALYSIS_MAX_OUTPUT_TOKENS,
            )
            response = client.models.generate_content(
                model=model_id,
                contents=[
                    source_analysis_prefix(
                        locator=request.locator,
                        prompt_version=self._policy.prompt_version,
                    ),
                    source_part,
                    source_analysis_suffix(),
                ],
                config=config,
            )
            raw_output = getattr(response, "text", None)
            if not isinstance(raw_output, str):
                raise _InvalidProviderResponse
            return raw_output
        finally:
            if uploaded_name is not None:
                try:
                    client.files.delete(name=uploaded_name)
                except Exception:
                    pass


class _InvalidProviderResponse(RuntimeError):
    pass


def _inline_limit(provider: object) -> int:
    value = getattr(provider, "inline_file_limit_bytes", _DEFAULT_INLINE_FILE_LIMIT_BYTES)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return _DEFAULT_INLINE_FILE_LIMIT_BYTES


def _provider_filename(request: AnalyzeSourcePartRequest) -> str:
    suffix = PurePosixPath(request.relative_path).suffix.casefold()
    if not suffix or len(suffix) > 16 or not suffix[1:].isalnum():
        suffix = ".bin"
    return f"{request.source_part_id}{suffix}"


def _normalize_provider_exception(
    exception: Exception,
    policy: EvidencePolicy,
) -> tuple[str, bool, int | None]:
    try:
        status = _status_code(exception)
        class_name = type(exception).__name__.casefold()
        symbol = _structured_error_symbol(exception)
        retry_after = _retry_after_seconds(exception, policy)
        if (
            status == 408
            or isinstance(exception, TimeoutError)
            or "timeout" in class_name
            or (symbol == "deadline_exceeded")
        ):
            return "provider_timeout", True, retry_after
        if isinstance(exception, ConnectionError) or any(
            token in class_name for token in ("connection", "protocol", "disconnect", "transport")
        ):
            return "provider_connection_failed", True, retry_after
        if status == 429 or symbol in {"rate_limit_exceeded", "resource_exhausted", "rate_limit_error"}:
            return "provider_rate_limited", True, retry_after
        if status is not None and status >= 500 or symbol in {"unavailable", "internal"}:
            return "provider_unavailable", True, retry_after
        if status in {401, 403} or symbol in {
            "authentication_error",
            "invalid_api_key",
            "permission_denied",
            "unauthenticated",
        }:
            return "provider_credentials_invalid", False, None
        if status == 404 or symbol in {"model_not_found", "not_found", "unsupported_model"}:
            return "provider_model_unsupported", False, None
        if status == 415 or symbol in {"media_unsupported", "unsupported_media_type"}:
            return "provider_media_unsupported", False, None
        if status in {400, 409, 422} or symbol in {
            "bad_request",
            "invalid_argument",
            "unprocessable_entity",
        }:
            return "provider_request_invalid", False, None
    except BaseException:
        pass
    return "provider_failed", False, None


def _status_code(exception: Exception) -> int | None:
    candidates: list[object] = [_safe_getattr(exception, "status_code")]
    response = _safe_getattr(exception, "response")
    if response is not None:
        candidates.append(_safe_getattr(response, "status_code"))
    candidates.append(_safe_getattr(exception, "code"))
    for candidate in candidates:
        if type(candidate) is int:
            return candidate
        if type(candidate) is str and len(candidate) <= 3 and candidate.isascii() and candidate.isdigit():
            return int(candidate)
    return None


def _structured_error_symbol(exception: Exception) -> str | None:
    for name in ("code", "status"):
        value = _safe_getattr(exception, name)
        if type(value) is str and len(value) <= 64 and value.isascii():
            return value.strip().casefold().replace("-", "_")
    return None


def _retry_after_seconds(exception: Exception, policy: EvidencePolicy) -> int | None:
    response = _safe_getattr(exception, "response")
    headers = _safe_getattr(response, "headers")
    if headers is None:
        return None
    get_header = _safe_getattr(headers, "get")
    if not callable(get_header):
        return None
    try:
        value: Any = get_header("Retry-After")
        if value is None:
            value = get_header("retry-after")
    except BaseException:
        return None
    if type(value) is not str or len(value) > _MAX_RETRY_AFTER_DIGITS:
        return None
    if not value.isascii() or not value.isdigit():
        return None
    seconds = int(value)
    if seconds > int(policy.tool_deadline_seconds):
        return None
    return seconds


def _safe_getattr(value: object, name: str) -> object | None:
    try:
        return getattr(value, name, None)
    except BaseException:
        return None
