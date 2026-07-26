from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any, BinaryIO, Protocol, TypeVar
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from .models import (
    AgentEvent,
    ConnectProviderRequest,
    HealthResponse,
    ProviderDescriptor,
    RunSnapshot,
    SavedProviderProfile,
    SubmitMessageRequest,
    SubmitMessageResponse,
)
from exam_predictor.workspace.models import (
    ApprovalRecord,
    EntryInclusionRequest,
    ManifestPage,
    ManifestRevision,
    SourceState,
    WorkspaceDetail,
    WorkspaceEvent,
    WorkspaceJob,
    WorkspaceSummary,
)


ModelT = TypeVar("ModelT", bound=BaseModel)
_SAFE_URL_ERROR = "The Agent Worker URL must be http://127.0.0.1:<port>."


class WorkerClientError(RuntimeError):
    """A secret-safe failure while communicating with the local Agent Worker."""


class SeekableBinaryStream(Protocol):
    """Caller-owned upload stream whose cursor can be restored after use."""

    def read(self, size: int = -1) -> bytes: ...

    def seek(self, offset: int, whence: int = 0) -> int: ...

    def tell(self) -> int: ...


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
            follow_redirects=False,
            transport=transport,
        )

    @staticmethod
    def _validated_base_url(base_url: str) -> str:
        try:
            parsed = urlsplit(base_url)
            port = parsed.port
        except (TypeError, ValueError):
            parsed = None
            port = None
        if parsed is None:
            raise WorkerClientError(_SAFE_URL_ERROR)
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
            parsed = model.model_validate(response.json())
        except (TypeError, ValueError, ValidationError):
            parsed = None
        if parsed is None:
            raise WorkerClientError("Agent Worker returned an invalid response.")
        return parsed

    @staticmethod
    def _models(response: httpx.Response, model: type[ModelT]) -> list[ModelT]:
        try:
            parsed = TypeAdapter(list[model]).validate_python(response.json())
        except (TypeError, ValueError, ValidationError):
            parsed = None
        if parsed is None:
            raise WorkerClientError("Agent Worker returned an invalid response.")
        return parsed

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
            events = TypeAdapter(list[AgentEvent]).validate_python(response.json())
        except (TypeError, ValueError, ValidationError):
            events = None
        if events is None:
            raise WorkerClientError("Agent Worker returned an invalid response.")
        return events

    def stop(self, run_id: str) -> RunSnapshot:
        response = self._request("POST", f"/v1/runs/{quote(run_id, safe='')}/stop")
        return self._model(response, RunSnapshot)

    def resume(self, run_id: str) -> RunSnapshot:
        response = self._request("POST", f"/v1/runs/{quote(run_id, safe='')}/resume")
        return self._model(response, RunSnapshot)

    def pause_all(self) -> None:
        self._request("POST", "/v1/runtime/pause-all")

    def list_workspaces(self) -> list[WorkspaceSummary]:
        return self._models(self._request("GET", "/v1/workspaces"), WorkspaceSummary)

    def get_workspace(self, workspace_id: str) -> WorkspaceDetail:
        response = self._request("GET", f"/v1/workspaces/{quote(workspace_id, safe='')}")
        return self._model(response, WorkspaceDetail)

    def get_manifest(
        self,
        workspace_id: str,
        *,
        state: SourceState | None = None,
        course: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> ManifestPage:
        params: dict[str, str | int] = {"offset": offset, "limit": limit}
        if state is not None:
            params["state"] = state.value
        if course is not None:
            params["course"] = course
        response = self._request(
            "GET",
            f"/v1/workspaces/{quote(workspace_id, safe='')}/manifest",
            params=params,
        )
        return self._model(response, ManifestPage)

    def select_folder(self, idempotency_key: str) -> WorkspaceJob | None:
        response = self._request(
            "POST",
            "/v1/workspaces/select-folder",
            headers={"Idempotency-Key": idempotency_key},
        )
        return None if response.status_code == 204 else self._model(response, WorkspaceJob)

    def upload_directory(
        self,
        display_name: str,
        files: Mapping[str, str | Path | SeekableBinaryStream],
        idempotency_key: str,
    ) -> WorkspaceJob:
        with ExitStack() as stack:
            multipart = []
            for relative_path, path in files.items():
                stream: BinaryIO | SeekableBinaryStream
                if isinstance(path, (str, Path)):
                    stream = stack.enter_context(Path(path).open("rb"))
                else:
                    stream = path
                    try:
                        original_position = stream.tell()
                        stream.seek(0)
                    except (AttributeError, OSError, TypeError, ValueError):
                        raise WorkerClientError(
                            "Caller-owned upload streams must be seekable."
                        ) from None

                    def restore_cursor(
                        upload: SeekableBinaryStream = stream,
                        position: int = original_position,
                    ) -> None:
                        try:
                            upload.seek(position)
                        except (AttributeError, OSError, TypeError, ValueError):
                            raise WorkerClientError(
                                "The caller-owned upload stream cursor could not be restored."
                            ) from None

                    stack.callback(restore_cursor)
                multipart.append(
                    ("files", (relative_path, stream, "application/octet-stream"))
                )
            response = self._request(
                "POST",
                "/v1/workspaces/browser-snapshot",
                data={"display_name": display_name, "idempotency_key": idempotency_key},
                files=multipart,
            )
        return self._model(response, WorkspaceJob)

    def rescan(self, workspace_id: str, idempotency_key: str) -> WorkspaceJob:
        response = self._request(
            "POST",
            f"/v1/workspaces/{quote(workspace_id, safe='')}/rescan",
            headers={"Idempotency-Key": idempotency_key},
        )
        return self._model(response, WorkspaceJob)

    def set_entry_inclusion(
        self,
        workspace_id: str,
        entry_id: str,
        request: EntryInclusionRequest,
    ) -> ManifestRevision:
        response = self._request(
            "PATCH",
            f"/v1/workspaces/{quote(workspace_id, safe='')}/entries/{quote(entry_id, safe='')}",
            json=request.model_dump(),
        )
        return self._model(response, ManifestRevision)

    def approve_workspace(self, workspace_id: str, revision_id: str) -> ApprovalRecord:
        response = self._request(
            "POST",
            f"/v1/workspaces/{quote(workspace_id, safe='')}/approval",
            json={"revision_id": revision_id},
        )
        return self._model(response, ApprovalRecord)

    def delete_workspace(self, workspace_id: str) -> None:
        self._request("DELETE", f"/v1/workspaces/{quote(workspace_id, safe='')}")

    def delete_all_workspaces(self) -> None:
        self._request("DELETE", "/v1/workspaces", json={"confirmation": "DELETE ALL"})

    def get_job(self, job_id: str) -> WorkspaceJob:
        response = self._request("GET", f"/v1/workspace-jobs/{quote(job_id, safe='')}")
        return self._model(response, WorkspaceJob)

    def workspace_events_after(self, job_id: str, after: int = 0) -> list[WorkspaceEvent]:
        response = self._request(
            "GET",
            f"/v1/workspace-jobs/{quote(job_id, safe='')}/events",
            params={"after": after},
        )
        return self._models(response, WorkspaceEvent)

    def list_saved_providers(self) -> list[SavedProviderProfile]:
        return self._models(self._request("GET", "/v1/providers/saved"), SavedProviderProfile)

    def forget_provider_credential(self, profile_id: str) -> None:
        self._request("DELETE", f"/v1/providers/{quote(profile_id, safe='')}/credential")

    def close(self) -> None:
        self._client.close()
