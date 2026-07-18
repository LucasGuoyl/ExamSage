from __future__ import annotations

from typing import Any, TypeVar
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from .models import (
    AgentEvent,
    ConnectProviderRequest,
    HealthResponse,
    ProviderDescriptor,
    RunSnapshot,
    SubmitMessageRequest,
    SubmitMessageResponse,
)


ModelT = TypeVar("ModelT", bound=BaseModel)
_SAFE_URL_ERROR = "The Agent Worker URL must be http://127.0.0.1:<port>."


class WorkerClientError(RuntimeError):
    """A secret-safe failure while communicating with the local Agent Worker."""


class WorkerClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = self._validated_base_url(base_url)
        if not token.strip():
            raise WorkerClientError("Worker authentication is unavailable.")
        self._token = token
        self._client = httpx.Client(
            headers={"X-ExamSage-Token": token},
            timeout=httpx.Timeout(60.0, connect=10.0),
            transport=transport,
        )

    @staticmethod
    def _validated_base_url(base_url: str) -> str:
        try:
            parsed = urlsplit(base_url)
            port = parsed.port
        except (TypeError, ValueError):
            raise WorkerClientError(_SAFE_URL_ERROR) from None
        valid = (
            parsed.scheme == "http"
            and parsed.hostname == "127.0.0.1"
            and port is not None
            and parsed.username is None
            and parsed.password is None
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
            and parsed.netloc == f"127.0.0.1:{port}"
        )
        if not valid:
            raise WorkerClientError(_SAFE_URL_ERROR)
        return f"http://127.0.0.1:{port}"

    @staticmethod
    def _redact(message: str, values: list[str]) -> str:
        safe = message
        for value in values:
            if value:
                safe = safe.replace(value, "[REDACTED]")
        return safe

    def _request(
        self,
        method: str,
        path: str,
        *,
        redact: list[str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        response: httpx.Response | None = None
        secrets = [self._token, *(redact or [])]
        try:
            response = self._client.request(method, f"{self.base_url}{path}", **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            detail = response.text if response is not None else str(exc)
        except RuntimeError as exc:
            detail = str(exc)
        safe_detail = self._redact(detail, secrets).strip()
        if not safe_detail:
            safe_detail = "The local Worker did not accept the request."
        raise WorkerClientError(f"Agent Worker request failed: {safe_detail}") from None

    @staticmethod
    def _model(response: httpx.Response, model: type[ModelT]) -> ModelT:
        try:
            return model.model_validate(response.json())
        except (TypeError, ValueError, ValidationError):
            raise WorkerClientError("Agent Worker returned an invalid response.") from None

    def health(self) -> HealthResponse:
        return self._model(self._request("GET", "/health"), HealthResponse)

    def connect_provider(self, request: ConnectProviderRequest) -> ProviderDescriptor:
        secret = request.api_key.get_secret_value()
        payload = {
            "profile": request.profile.model_dump(exclude_none=True),
            "api_key": secret,
        }
        response = self._request(
            "POST",
            "/v1/providers/connect",
            json=payload,
            redact=[secret],
        )
        return self._model(response, ProviderDescriptor)

    def submit_message(self, request: SubmitMessageRequest) -> SubmitMessageResponse:
        thread_id = quote(request.thread_id, safe="")
        payload = request.model_dump(exclude={"thread_id"})
        response = self._request(
            "POST",
            f"/v1/threads/{thread_id}/messages",
            json=payload,
        )
        return self._model(response, SubmitMessageResponse)

    def get_run(self, run_id: str) -> RunSnapshot:
        response = self._request("GET", f"/v1/runs/{quote(run_id, safe='')}")
        return self._model(response, RunSnapshot)

    def events_after(self, run_id: str, after: int = 0) -> list[AgentEvent]:
        response = self._request(
            "GET",
            f"/v1/runs/{quote(run_id, safe='')}/events",
            params={"after": after},
        )
        try:
            return TypeAdapter(list[AgentEvent]).validate_python(response.json())
        except (TypeError, ValueError, ValidationError):
            raise WorkerClientError("Agent Worker returned an invalid response.") from None

    def stop(self, run_id: str) -> RunSnapshot:
        response = self._request("POST", f"/v1/runs/{quote(run_id, safe='')}/stop")
        return self._model(response, RunSnapshot)

    def resume(self, run_id: str) -> RunSnapshot:
        response = self._request("POST", f"/v1/runs/{quote(run_id, safe='')}/resume")
        return self._model(response, RunSnapshot)

    def pause_all(self) -> None:
        self._request("POST", "/v1/runtime/pause-all")

    def close(self) -> None:
        self._client.close()
