"""Cloud AI providers used by ExamSage.

The rest of the application talks to this module instead of an SDK directly.
That gives users one simple choice (provider + one key) while keeping every AI
operation -- chat, vision, embeddings and web research -- with that provider.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
)

from .schema import SourceCitation


DEFAULT_PROVIDER_TIMEOUT_SECONDS = 90.0


class ProviderError(RuntimeError):
    """A provider could not complete a requested operation."""


def _provider_status_code(exc: BaseException) -> int | None:
    status_code = getattr(exc, "status_code", getattr(exc, "code", None))
    if status_code is None:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(status_code)
    except (TypeError, ValueError):
        return None


def _is_retryable_provider_error(exc: BaseException) -> bool:
    """Retry transient transport failures and retryable HTTP responses only."""
    try:
        import httpx

        if isinstance(exc, httpx.TransportError):
            return True
    except ImportError:  # pragma: no cover - installed with supported cloud SDKs
        pass

    status_code = _provider_status_code(exc)
    if status_code is None:
        return False
    return status_code == 429 or status_code >= 500


class BudgetExceeded(ProviderError):
    """The user-approved spending ceiling would be exceeded."""


@dataclass(frozen=True)
class ModelRouting:
    """Models assigned to distinct jobs while sharing the same API key."""

    fast: str
    balanced: str
    reasoning: str
    embedding: str


@dataclass(frozen=True)
class ProviderCapabilities:
    chat: bool = True
    vision: bool = True
    file_understanding: bool = True
    embeddings: bool = True
    web_search: bool = True
    citations: bool = True
    ephemeral_requests: bool = True


@dataclass
class UsageLedger:
    """In-memory usage and conservative cost guard for one run."""

    approved_max_usd: float | None = None
    estimated_spend_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    web_searches: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def guard(self, reserved_usd: float, label: str) -> None:
        if reserved_usd < 0:
            raise ValueError("reserved_usd cannot be negative")
        if (
            self.approved_max_usd is not None
            and self.estimated_spend_usd + reserved_usd > self.approved_max_usd
        ):
            raise BudgetExceeded(
                f"'{label}' could exceed the approved ${self.approved_max_usd:.2f} limit. "
                "The run was stopped before making that request."
            )

    def record(
        self,
        label: str,
        estimated_usd: float,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        embedding_tokens: int = 0,
        web_searches: int = 0,
    ) -> None:
        self.estimated_spend_usd += max(0.0, estimated_usd)
        self.input_tokens += max(0, input_tokens)
        self.output_tokens += max(0, output_tokens)
        self.embedding_tokens += max(0, embedding_tokens)
        self.web_searches += max(0, web_searches)
        self.events.append({
            "label": label,
            "estimated_usd": round(max(0.0, estimated_usd), 6),
            "input_tokens": max(0, input_tokens),
            "output_tokens": max(0, output_tokens),
            "embedding_tokens": max(0, embedding_tokens),
            "web_searches": max(0, web_searches),
        })


@dataclass
class WebSearchResponse:
    query: str
    summary: str
    citations: list[SourceCitation] = field(default_factory=list)


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _usage_values(usage: Any) -> tuple[int, int]:
    if not usage:
        return 0, 0
    input_tokens = _attr(usage, "input_tokens", _attr(usage, "prompt_tokens", 0)) or 0
    output_tokens = _attr(usage, "output_tokens", _attr(usage, "completion_tokens", 0)) or 0
    return int(input_tokens), int(output_tokens)


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


class _CompletionsFacade:
    def __init__(self, provider: "BaseProvider"):
        self.provider = provider

    def create(self, **kwargs):
        return self.provider.create_chat_completion(**kwargs)


class _ChatFacade:
    def __init__(self, provider: "BaseProvider"):
        self.completions = _CompletionsFacade(provider)


class ChatClientFacade:
    """Minimal OpenAI-shaped facade used by the existing prediction pipeline."""

    def __init__(self, provider: "BaseProvider"):
        self.chat = _ChatFacade(provider)


class BaseProvider:
    name = "base"
    capabilities = ProviderCapabilities()
    embedding_dimension = 1536

    def __init__(
        self,
        api_key: str,
        models: ModelRouting,
        *,
        approved_max_usd: float | None = None,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("An API key is required.")
        self.api_key = api_key.strip()
        self.models = models
        self.ledger = UsageLedger(approved_max_usd=approved_max_usd)
        self.chat_client = ChatClientFacade(self)

    def set_budget(self, approved_max_usd: float | None) -> None:
        self.ledger.approved_max_usd = approved_max_usd

    def create_chat_completion(self, **kwargs):
        raise NotImplementedError

    def create_chat_completion_once(
        self,
        *,
        timeout_seconds: float | None = None,
        **kwargs,
    ):
        """Make one SDK chat attempt with an optional per-request deadline."""
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def analyze_file(self, path: str | Path, prompt: str) -> str:
        raise NotImplementedError

    def web_search(self, query: str, prompt: str | None = None) -> WebSearchResponse:
        raise NotImplementedError

    def _guard(self, label: str, reserved_usd: float = 0.05) -> None:
        self.ledger.guard(reserved_usd, label)


class OpenAIProvider(BaseProvider):
    name = "openai"
    capabilities = ProviderCapabilities()
    inline_file_limit_bytes = 10 * 1024 * 1024

    def __init__(
        self,
        api_key: str,
        models: ModelRouting,
        *,
        approved_max_usd: float | None = None,
        base_url: str | None = None,
        provider_timeout_seconds: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ):
        super().__init__(api_key, models, approved_max_usd=approved_max_usd)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise ProviderError("Install the 'openai' package to use OpenAI.") from exc
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=provider_timeout_seconds,
            max_retries=0,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=12))
    def create_chat_completion(self, **kwargs):
        return self.create_chat_completion_once(**kwargs)

    def create_chat_completion_once(
        self,
        *,
        timeout_seconds: float | None = None,
        **kwargs,
    ):
        message_chars = sum(len(_json_text(item.get("content", ""))) for item in kwargs.get("messages", []))
        input_guess = max(1, message_chars // 4)
        output_guess = int(kwargs.get("max_completion_tokens", kwargs.get("max_tokens", 2048)) or 2048)
        self._guard(
            "language-model request",
            max(0.02, self._estimate_text_cost(input_guess, output_guess, kwargs.get("model"))),
        )
        # Newer reasoning models use max_completion_tokens and reject sampling.
        request = dict(kwargs)
        if timeout_seconds is not None:
            request["timeout"] = timeout_seconds
        if "max_tokens" in request and str(request.get("model", "")).startswith("gpt-5"):
            request["max_completion_tokens"] = request.pop("max_tokens")
            # GPT-5 reasoning modes do not accept sampling temperature.
            request.pop("temperature", None)
        response = self.client.chat.completions.create(**request)
        input_tokens, output_tokens = _usage_values(_attr(response, "usage"))
        self.ledger.record(
            "language-model request",
            self._estimate_text_cost(input_tokens, output_tokens, kwargs.get("model")),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return response

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=12))
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        token_guess = sum(max(1, len(text) // 4) for text in texts)
        self._guard("embedding request", max(0.001, token_guess / 1_000_000 * 0.02))
        response = self.client.embeddings.create(
            model=self.models.embedding,
            input=list(texts),
        )
        usage = _attr(response, "usage")
        tokens = int(_attr(usage, "total_tokens", _attr(usage, "prompt_tokens", 0)) or 0)
        self.ledger.record(
            "embedding request",
            tokens / 1_000_000 * 0.02,
            embedding_tokens=tokens,
        )
        return [list(row.embedding) for row in response.data]

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=10))
    def analyze_file(self, path: str | Path, prompt: str) -> str:
        path = Path(path)
        file_reserve = max(0.10, path.stat().st_size / (1024 * 1024) * 0.08)
        self._guard(f"analyze {path.name}", file_reserve)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        if mime.startswith("image/"):
            content = [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"},
            ]
        else:
            content = [
                {
                    "type": "input_file",
                    "filename": path.name,
                    "file_data": f"data:{mime};base64,{encoded}",
                },
                {"type": "input_text", "text": prompt},
            ]
        response = self.client.responses.create(
            model=self.models.balanced,
            input=[{"role": "user", "content": content}],
            store=False,
        )
        input_tokens, output_tokens = _usage_values(_attr(response, "usage"))
        self.ledger.record(
            f"analyze {path.name}",
            self._estimate_text_cost(input_tokens, output_tokens, self.models.balanced),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return str(_attr(response, "output_text", "") or "")

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=10))
    def web_search(self, query: str, prompt: str | None = None) -> WebSearchResponse:
        self._guard("web search", 0.08)
        instruction = prompt or (
            "Find trustworthy university-level sources that clarify this topic and provide "
            "related practice patterns. Separate sourced facts from your inference."
        )
        response = self.client.responses.create(
            model=self.models.balanced,
            tools=[{"type": "web_search"}],
            input=f"Research query: {query}\n\nTask: {instruction}",
            store=False,
        )
        citations: list[SourceCitation] = []
        seen: set[str] = set()
        for output in _attr(response, "output", []) or []:
            for content in _attr(output, "content", []) or []:
                for ann in _attr(content, "annotations", []) or []:
                    url = _attr(ann, "url")
                    if url and url not in seen:
                        seen.add(url)
                        citations.append(SourceCitation(
                            title=str(_attr(ann, "title", url)),
                            url=str(url),
                            domain=_domain(str(url)),
                        ))
        input_tokens, output_tokens = _usage_values(_attr(response, "usage"))
        self.ledger.record(
            "web search",
            self._estimate_text_cost(input_tokens, output_tokens, self.models.balanced) + 0.05,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            web_searches=1,
        )
        return WebSearchResponse(
            query=query,
            summary=str(_attr(response, "output_text", "") or ""),
            citations=citations,
        )

    @staticmethod
    def _estimate_text_cost(
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
    ) -> float:
        model = str(model or "")
        if "luna" in model:
            input_rate, output_rate = 1.0, 6.0
        elif "terra" in model:
            input_rate, output_rate = 2.5, 15.0
        else:  # Sol/alias and unknown future models use the conservative tier.
            input_rate, output_rate = 5.0, 30.0
        return input_tokens / 1_000_000 * input_rate + output_tokens / 1_000_000 * output_rate


class GeminiProvider(BaseProvider):
    name = "gemini"
    capabilities = ProviderCapabilities()
    # Keep inline requests comfortably below legacy/base64 request limits.
    # Larger PDFs are page-split by cloud_analyzer; indivisible large files
    # retain the Files API fallback.
    inline_file_limit_bytes = 10 * 1024 * 1024
    _inline_mime_types = frozenset({
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/webp",
    })

    def __init__(
        self,
        api_key: str,
        models: ModelRouting,
        *,
        approved_max_usd: float | None = None,
        provider_timeout_seconds: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ):
        super().__init__(api_key, models, approved_max_usd=approved_max_usd)
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise ProviderError("Install the 'google-genai' package to use Gemini.") from exc
        self._genai = genai
        http_options = genai.types.HttpOptions(
            timeout=int(provider_timeout_seconds * 1000),
            retry_options=genai.types.HttpRetryOptions(attempts=1),
        )
        self.client = genai.Client(api_key=api_key, http_options=http_options)

    @staticmethod
    def _contents_from_messages(messages: list[dict]) -> tuple[str, str]:
        system: list[str] = []
        conversation: list[str] = []
        for message in messages:
            role = str(message.get("role", "user"))
            text = _json_text(message.get("content", ""))
            if role == "system":
                system.append(text)
            else:
                conversation.append(f"{role.upper()}: {text}")
        return "\n\n".join(system), "\n\n".join(conversation)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=12),
        retry=retry_if_exception(_is_retryable_provider_error),
        reraise=True,
    )
    def create_chat_completion(self, **kwargs):
        return self.create_chat_completion_once(**kwargs)

    def create_chat_completion_once(
        self,
        *,
        timeout_seconds: float | None = None,
        **kwargs,
    ):
        message_chars = sum(len(_json_text(item.get("content", ""))) for item in kwargs.get("messages", []))
        input_guess = max(1, message_chars // 4)
        output_guess = int(kwargs.get("max_completion_tokens", kwargs.get("max_tokens", 2048)) or 2048)
        self._guard(
            "language-model request",
            max(0.02, self._estimate_text_cost(input_guess, output_guess, kwargs.get("model"))),
        )
        system, contents = self._contents_from_messages(kwargs.get("messages", []))
        config_kwargs: dict[str, Any] = {}
        if system:
            config_kwargs["system_instruction"] = system
        if kwargs.get("temperature") is not None:
            config_kwargs["temperature"] = kwargs["temperature"]
        token_limit = kwargs.get("max_completion_tokens", kwargs.get("max_tokens"))
        if token_limit:
            config_kwargs["max_output_tokens"] = token_limit
        if timeout_seconds is not None:
            config_kwargs["http_options"] = self._genai.types.HttpOptions(
                timeout=max(1, int(timeout_seconds * 1000)),
                retry_options=self._genai.types.HttpRetryOptions(attempts=1),
            )
        config = self._genai.types.GenerateContentConfig(**config_kwargs)
        response = self.client.models.generate_content(
            model=kwargs.get("model") or self.models.balanced,
            contents=contents,
            config=config,
        )
        usage = _attr(response, "usage_metadata")
        input_tokens = int(_attr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(_attr(usage, "candidates_token_count", 0) or 0)
        self.ledger.record(
            "language-model request",
            self._estimate_text_cost(input_tokens, output_tokens, kwargs.get("model")),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response.text or ""))],
            usage=usage,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=12),
        retry=retry_if_exception(_is_retryable_provider_error),
        reraise=True,
    )
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        token_guess = sum(max(1, len(text) // 4) for text in texts)
        self._guard("embedding request", max(0.001, token_guess / 1_000_000 * 0.20))
        response = self.client.models.embed_content(
            model=self.models.embedding,
            contents=[
                self._genai.types.Content(parts=[self._genai.types.Part(text=text)])
                for text in texts
            ],
            config=self._genai.types.EmbedContentConfig(output_dimensionality=self.embedding_dimension),
        )
        self.ledger.record(
            "embedding request",
            token_guess / 1_000_000 * 0.20,
            embedding_tokens=token_guess,
        )
        return [list(item.values) for item in response.embeddings]

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_random_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable_provider_error),
        reraise=True,
    )
    def _generate_inline_file(
        self,
        path: Path,
        prompt: str,
        mime_type: str,
        model: str,
    ):
        part = self._genai.types.Part.from_bytes(
            data=path.read_bytes(),
            mime_type=mime_type,
        )
        return self.client.models.generate_content(
            model=model,
            contents=[part, prompt],
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(_is_retryable_provider_error),
        reraise=True,
    )
    def _upload_file(self, path: Path):
        return self.client.files.upload(file=str(path))

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_random_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable_provider_error),
        reraise=True,
    )
    def _generate_uploaded_file(self, uploaded: Any, prompt: str, model: str):
        return self.client.models.generate_content(
            model=model,
            contents=[uploaded, prompt],
        )

    @staticmethod
    def _file_error(path: Path, stage: str, exc: BaseException) -> ProviderError:
        detail = str(exc).strip()
        suffix = f": {detail}" if detail else ""
        return ProviderError(
            f"Gemini {stage} failed for '{path.name}' "
            f"({type(exc).__name__}){suffix}"
        )

    def analyze_file(self, path: str | Path, prompt: str) -> str:
        path = Path(path)
        file_reserve = max(0.08, path.stat().st_size / (1024 * 1024) * 0.04)
        self._guard(f"analyze {path.name}", file_reserve)
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        uploaded = None
        stage = "file preparation"
        used_model = self.models.balanced
        fallback_allowed = False
        try:
            try:
                if (
                    path.stat().st_size <= self.inline_file_limit_bytes
                    and mime_type in self._inline_mime_types
                ):
                    stage = "inline document request"
                    fallback_allowed = True
                    response = self._generate_inline_file(
                        path,
                        prompt,
                        mime_type,
                        used_model,
                    )
                else:
                    stage = "Files API upload"
                    uploaded = self._upload_file(path)
                    stage = "uploaded-file analysis"
                    fallback_allowed = True
                    response = self._generate_uploaded_file(uploaded, prompt, used_model)
            except Exception as primary_exc:
                fallback_model = self.models.fast
                if (
                    not fallback_allowed
                    or _provider_status_code(primary_exc) != 503
                    or fallback_model == used_model
                ):
                    raise

                primary_model = used_model
                used_model = fallback_model
                try:
                    if uploaded is None:
                        response = self._generate_inline_file(
                            path,
                            prompt,
                            mime_type,
                            used_model,
                        )
                    else:
                        response = self._generate_uploaded_file(
                            uploaded,
                            prompt,
                            used_model,
                        )
                except Exception as fallback_exc:
                    raise ProviderError(
                        f"Gemini {stage} failed for '{path.name}': primary model "
                        f"'{primary_model}' was unavailable ({type(primary_exc).__name__}); "
                        f"fallback model '{fallback_model}' also failed "
                        f"({type(fallback_exc).__name__}: {fallback_exc})."
                    ) from fallback_exc

                stage = f"{stage} fallback to {fallback_model}"

            usage = _attr(response, "usage_metadata")
            input_tokens = int(_attr(usage, "prompt_token_count", 0) or 0)
            output_tokens = int(_attr(usage, "candidates_token_count", 0) or 0)
            self.ledger.record(
                f"analyze {path.name}",
                self._estimate_text_cost(input_tokens, output_tokens, used_model),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            return response.text or ""
        except ProviderError:
            raise
        except Exception as exc:
            raise self._file_error(path, stage, exc) from exc
        finally:
            # Gemini file uploads otherwise remain available temporarily. Best-
            # effort deletion is part of ExamSage's privacy-by-default policy.
            if uploaded is not None:
                try:
                    self.client.files.delete(name=uploaded.name)
                except Exception:
                    pass

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(_is_retryable_provider_error),
        reraise=True,
    )
    def web_search(self, query: str, prompt: str | None = None) -> WebSearchResponse:
        self._guard("web search", 0.08)
        instruction = prompt or (
            "Research trustworthy university-level explanations and assessment patterns. "
            "Summarize them and avoid copying complete copyrighted exam papers."
        )
        config = self._genai.types.GenerateContentConfig(
            tools=[self._genai.types.Tool(google_search=self._genai.types.GoogleSearch())]
        )
        response = self.client.models.generate_content(
            model=self.models.balanced,
            contents=f"Research query: {query}\n\nTask: {instruction}",
            config=config,
        )
        citations: list[SourceCitation] = []
        seen: set[str] = set()
        for candidate in _attr(response, "candidates", []) or []:
            metadata = _attr(candidate, "grounding_metadata")
            for chunk in _attr(metadata, "grounding_chunks", []) or []:
                web = _attr(chunk, "web")
                url = _attr(web, "uri")
                if url and url not in seen:
                    seen.add(url)
                    citations.append(SourceCitation(
                        title=str(_attr(web, "title", url)),
                        url=str(url),
                        domain=_domain(str(url)),
                    ))
        usage = _attr(response, "usage_metadata")
        input_tokens = int(_attr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(_attr(usage, "candidates_token_count", 0) or 0)
        self.ledger.record(
            "web search",
            self._estimate_text_cost(input_tokens, output_tokens, self.models.balanced) + 0.05,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            web_searches=1,
        )
        return WebSearchResponse(query=query, summary=response.text or "", citations=citations)

    @staticmethod
    def _estimate_text_cost(
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
    ) -> float:
        model = str(model or "")
        if "flash-lite" in model:
            input_rate, output_rate = 0.25, 1.50
        elif "3.5-flash" in model:
            input_rate, output_rate = 1.50, 9.00
        elif "pro" in model:
            input_rate, output_rate = 1.25, 10.00
        else:
            input_rate, output_rate = 2.00, 12.00
        return input_tokens / 1_000_000 * input_rate + output_tokens / 1_000_000 * output_rate


class CustomOpenAIProvider(OpenAIProvider):
    """Experimental OpenAI-compatible endpoint.

    Chat and embeddings are broadly interoperable. File understanding, native
    web search, citations and no-retention flags are provider-specific, so the
    UI must display these reduced guarantees before a run.
    """

    name = "custom"
    capabilities = ProviderCapabilities(
        vision=False,
        file_understanding=False,
        web_search=False,
        citations=False,
        ephemeral_requests=False,
    )

    def analyze_file(self, path: str | Path, prompt: str) -> str:
        raise ProviderError(
            "This custom endpoint has not declared a compatible file/vision API. "
            "Choose OpenAI or Gemini for multimodal course files."
        )

    def web_search(self, query: str, prompt: str | None = None) -> WebSearchResponse:
        raise ProviderError(
            "This custom endpoint has not declared a native web-search API. "
            "Choose OpenAI or Gemini for cited web research."
        )


DEFAULT_MODELS: dict[str, ModelRouting] = {
    "openai": ModelRouting(
        fast="gpt-5.6-luna",
        balanced="gpt-5.6-terra",
        reasoning="gpt-5.6",
        embedding="text-embedding-3-small",
    ),
    "gemini": ModelRouting(
        fast="gemini-3.1-flash-lite",
        balanced="gemini-3.5-flash",
        reasoning="gemini-3.1-pro-preview",
        embedding="gemini-embedding-2",
    ),
    "custom": ModelRouting(
        fast="model",
        balanced="model",
        reasoning="model",
        embedding="text-embedding-3-small",
    ),
}


def _models_from_config(provider_name: str, config: dict[str, Any]) -> ModelRouting:
    defaults = DEFAULT_MODELS[provider_name]
    legacy_model = config.get("model")
    return ModelRouting(
        fast=config.get("fast_model") or legacy_model or defaults.fast,
        balanced=config.get("balanced_model") or legacy_model or defaults.balanced,
        reasoning=config.get("reasoning_model") or legacy_model or defaults.reasoning,
        embedding=config.get("embedding_model") or defaults.embedding,
    )


def create_provider(config: dict[str, Any]) -> BaseProvider:
    """Create one provider from UI/config values; no keys are persisted here."""

    name = str(config.get("provider", "openai")).strip().lower()
    aliases = {"google": "gemini", "openai-compatible": "custom"}
    name = aliases.get(name, name)
    if name not in DEFAULT_MODELS:
        raise ValueError(f"Unsupported provider '{name}'. Choose openai, gemini or custom.")
    key = str(config.get("api_key") or os.getenv("EXAMSAGE_API_KEY") or "")
    models = _models_from_config(name, config)
    approved = config.get("approved_max_usd")
    if name == "openai":
        return OpenAIProvider(key, models, approved_max_usd=approved)
    if name == "gemini":
        return GeminiProvider(key, models, approved_max_usd=approved)
    base_url = str(config.get("base_url") or "").strip()
    if not base_url:
        raise ValueError("A base URL is required for a custom OpenAI-compatible provider.")
    return CustomOpenAIProvider(
        key,
        models,
        base_url=base_url,
        approved_max_usd=approved,
    )


def _domain(url: str) -> str | None:
    try:
        from urllib.parse import urlparse

        return urlparse(url).netloc or None
    except Exception:
        return None
