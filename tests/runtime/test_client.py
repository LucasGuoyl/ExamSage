from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path

import httpx
import pytest

from exam_predictor.runtime import client as client_module
from exam_predictor.runtime.client import WorkerClient, WorkerClientError
from exam_predictor.runtime.models import (
    ConnectProviderRequest,
    ProviderProfile,
    RunStatus,
    SubmitMessageRequest,
)
from exam_predictor.workspace.models import (
    EntryInclusionRequest,
    SourceMode,
    SourceState,
)


NOW = datetime.now(timezone.utc).isoformat()
WORKER_TOKEN = "local-worker-token"
PROVIDER_SECRET = "provider-api-secret"


def _exception_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return chain


def _assert_exception_chain_is_secret_safe(error: BaseException) -> None:
    for linked_error in _exception_chain(error):
        message = str(linked_error)
        assert PROVIDER_SECRET not in message
        assert WORKER_TOKEN not in message


def _run_json(status: str) -> dict[str, str]:
    return {
        "run_id": "run/one",
        "thread_id": "course/one",
        "provider_profile_id": "primary",
        "message": "Explain limits.",
        "status": status,
        "created_at": NOW,
        "updated_at": NOW,
    }


def test_client_parses_authenticated_worker_contracts_and_quotes_path_segments():
    seen_raw_paths: list[bytes] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-ExamSage-Token"] == WORKER_TOKEN
        seen_raw_paths.append(request.url.raw_path)
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok", "agent_v2": True})
        if path == "/v1/providers/connect":
            assert request.method == "POST"
            assert request.content.count(PROVIDER_SECRET.encode()) == 1
            return httpx.Response(
                200,
                json={
                    "profile": {"profile_id": "primary", "provider": "gemini"},
                    "capabilities": {"chat": True},
                },
            )
        if path == "/v1/threads/course/one/messages":
            assert request.method == "POST"
            assert request.url.raw_path == b"/v1/threads/course%2Fone/messages"
            return httpx.Response(202, json={"run_id": "run/one", "status": "running"})
        if path == "/v1/runs/run/one/events":
            assert request.url.raw_path.startswith(b"/v1/runs/run%2Fone/events?")
            assert request.url.params["after"] == "7"
            return httpx.Response(
                200,
                json=[
                    {
                        "sequence": 8,
                        "run_id": "run/one",
                        "event_type": "message",
                        "stage": "answer",
                        "message": "Answer",
                        "payload": {},
                        "created_at": NOW,
                    }
                ],
            )
        if path == "/v1/runs/run/one/stop":
            assert request.url.raw_path == b"/v1/runs/run%2Fone/stop"
            return httpx.Response(200, json=_run_json("stopping"))
        if path == "/v1/runs/run/one/resume":
            assert request.url.raw_path == b"/v1/runs/run%2Fone/resume"
            return httpx.Response(200, json=_run_json("running"))
        if path == "/v1/runs/run/one":
            assert request.url.raw_path == b"/v1/runs/run%2Fone"
            return httpx.Response(200, json=_run_json("completed"))
        if path == "/v1/runtime/pause-all":
            return httpx.Response(204)
        raise AssertionError(f"unexpected path {path}")

    client = WorkerClient(
        "http://127.0.0.1:8765/",
        WORKER_TOKEN,
        transport=httpx.MockTransport(respond),
    )
    request = ConnectProviderRequest(
        profile=ProviderProfile(profile_id="primary", provider="gemini"),
        api_key=PROVIDER_SECRET,
    )
    try:
        assert client.health().agent_v2
        assert client.connect_provider(request).capabilities["chat"]
        submitted = client.submit_message(
            SubmitMessageRequest(
                thread_id="course/one",
                provider_profile_id="primary",
                message="Explain limits.",
            )
        )
        assert submitted.run_id == "run/one"
        assert client.events_after("run/one", after=7)[0].sequence == 8
        assert client.get_run("run/one").status is RunStatus.COMPLETED
        assert client.stop("run/one").status is RunStatus.STOPPING
        assert client.resume("run/one").status is RunStatus.RUNNING
        client.pause_all()
    finally:
        client.close()

    assert any(b"course%2Fone" in path for path in seen_raw_paths)
    assert any(b"run%2Fone" in path for path in seen_raw_paths)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://127.0.0.1:8765",
        "http://localhost:8765",
        "http://127.0.0.2:8765",
        "http://[::1]:8765",
        "http://127.0.0.1",
        "http://127.0.0.1:8765/v1",
        "http://127.0.0.1:8765?token=local-worker-token",
        "http://local-worker-token@127.0.0.1:8765",
    ],
)
def test_client_rejects_unsafe_worker_urls_before_transport_use(base_url: str):
    requests = 0

    def should_not_run(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200)

    with pytest.raises(
        WorkerClientError,
        match=r"^The Agent Worker URL must be http://127\.0\.0\.1:<port>\.$",
    ) as captured:
        WorkerClient(
            base_url,
            WORKER_TOKEN,
            transport=httpx.MockTransport(should_not_run),
        )

    assert requests == 0
    assert WORKER_TOKEN not in str(captured.value)


def test_malformed_worker_url_exception_chain_omits_all_secrets():
    base_url = f"http://127.0.0.1:{PROVIDER_SECRET}-{WORKER_TOKEN}"

    with pytest.raises(WorkerClientError) as captured:
        WorkerClient(base_url, WORKER_TOKEN)

    _assert_exception_chain_is_secret_safe(captured.value)


@pytest.mark.parametrize("token", ["", " ", "\t"])
def test_client_rejects_empty_worker_token_without_echoing_it(token: str):
    with pytest.raises(WorkerClientError, match="Worker authentication is unavailable"):
        WorkerClient("http://127.0.0.1:8765", token)


@pytest.mark.parametrize("failure_kind", ["status", "transport", "invalid-json", "invalid-model"])
def test_client_errors_never_expose_worker_or_provider_secrets(failure_kind: str):
    def reject(request: httpx.Request) -> httpx.Response:
        if failure_kind == "transport":
            raise httpx.ConnectError(
                f"{PROVIDER_SECRET} {WORKER_TOKEN} rejected",
                request=request,
            )
        if failure_kind == "invalid-json":
            return httpx.Response(200, text=f"not-json {PROVIDER_SECRET} {WORKER_TOKEN}")
        if failure_kind == "invalid-model":
            return httpx.Response(
                200,
                json={
                    "profile": {
                        "profile_id": WORKER_TOKEN,
                        "provider": PROVIDER_SECRET,
                    },
                    "capabilities": {"chat": True},
                },
            )
        return httpx.Response(400, text=f"{PROVIDER_SECRET} {WORKER_TOKEN} rejected")

    client = WorkerClient(
        "http://127.0.0.1:8765",
        WORKER_TOKEN,
        transport=httpx.MockTransport(reject),
    )
    request = ConnectProviderRequest(
        profile=ProviderProfile(profile_id="primary", provider="gemini"),
        api_key=PROVIDER_SECRET,
    )
    try:
        with pytest.raises(WorkerClientError) as captured:
            client.connect_provider(request)
    finally:
        client.close()

    _assert_exception_chain_is_secret_safe(captured.value)
    assert str(captured.value)


@pytest.mark.parametrize("failure_kind", ["invalid-json", "invalid-model"])
@pytest.mark.parametrize("contract", ["provider", "events"])
def test_invalid_response_exception_chains_omit_all_secrets(
    failure_kind: str,
    contract: str,
):
    def reject(_request: httpx.Request) -> httpx.Response:
        if failure_kind == "invalid-json":
            return httpx.Response(
                200,
                text=f"not-json {PROVIDER_SECRET} {WORKER_TOKEN}",
            )
        if contract == "provider":
            return httpx.Response(
                200,
                json={
                    "profile": {
                        "profile_id": WORKER_TOKEN,
                        "provider": PROVIDER_SECRET,
                    },
                    "capabilities": {"chat": True},
                },
            )
        return httpx.Response(
            200,
            json=[
                {
                    "sequence": PROVIDER_SECRET,
                    "run_id": WORKER_TOKEN,
                    "event_type": "message",
                    "stage": "answer",
                    "message": "unsafe",
                    "payload": {},
                    "created_at": NOW,
                }
            ],
        )

    client = WorkerClient(
        "http://127.0.0.1:8765",
        WORKER_TOKEN,
        transport=httpx.MockTransport(reject),
    )
    provider_request = ConnectProviderRequest(
        profile=ProviderProfile(profile_id="primary", provider="gemini"),
        api_key=PROVIDER_SECRET,
    )
    try:
        with pytest.raises(WorkerClientError) as captured:
            if contract == "provider":
                client.connect_provider(provider_request)
            else:
                client.events_after("run-1")
    finally:
        client.close()

    _assert_exception_chain_is_secret_safe(captured.value)


def test_client_explicitly_refuses_redirects_without_requesting_target(monkeypatch):
    real_client = httpx.Client
    seen_paths: list[str] = []

    def client_with_redirecting_default(*args, **kwargs):
        kwargs.setdefault("follow_redirects", True)
        return real_client(*args, **kwargs)

    def redirect(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/v1/providers/connect":
            return httpx.Response(
                307,
                headers={"Location": "/redirect-target"},
                text=f"{PROVIDER_SECRET} {WORKER_TOKEN}",
            )
        return httpx.Response(
            400,
            text=f"redirected {PROVIDER_SECRET} {WORKER_TOKEN}",
        )

    monkeypatch.setattr(client_module.httpx, "Client", client_with_redirecting_default)
    client = WorkerClient(
        "http://127.0.0.1:8765",
        WORKER_TOKEN,
        transport=httpx.MockTransport(redirect),
    )
    request = ConnectProviderRequest(
        profile=ProviderProfile(profile_id="primary", provider="gemini"),
        api_key=PROVIDER_SECRET,
    )
    try:
        with pytest.raises(WorkerClientError) as captured:
            client.connect_provider(request)
    finally:
        client.close()

    assert seen_paths == ["/v1/providers/connect"]
    _assert_exception_chain_is_secret_safe(captured.value)


def test_closed_client_fails_with_stable_error():
    client = WorkerClient(
        "http://127.0.0.1:8765",
        WORKER_TOKEN,
        transport=httpx.MockTransport(lambda _request: httpx.Response(200)),
    )
    client.close()

    with pytest.raises(WorkerClientError, match="Agent Worker request failed") as captured:
        client.health()

    assert WORKER_TOKEN not in str(captured.value)


def test_client_exposes_typed_workspace_and_saved_provider_operations(tmp_path: Path):
    upload = tmp_path / "notes.pdf"
    upload.write_bytes(b"notes")

    def respond(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/workspaces":
            if request.method == "GET":
                return httpx.Response(200, json=[_workspace_summary_json()])
            assert json.loads(request.content) == {"confirmation": "DELETE ALL"}
            return httpx.Response(204)
        if path == "/v1/workspaces/workspace/one":
            assert request.url.raw_path == b"/v1/workspaces/workspace%2Fone"
            return httpx.Response(200, json=_workspace_detail_json()) if request.method == "GET" else httpx.Response(204)
        if path == "/v1/workspaces/workspace/one/manifest":
            assert dict(request.url.params) == {"state": "pending_approval", "course": "week-1", "offset": "2", "limit": "50"}
            return httpx.Response(200, json={"items": [], "total": 0, "offset": 2, "limit": 50, "counts": {}})
        if path == "/v1/workspaces/select-folder":
            assert request.headers["Idempotency-Key"] == "key-1"
            return httpx.Response(204)
        if path == "/v1/workspaces/browser-snapshot":
            assert request.headers["content-type"].startswith("multipart/form-data")
            assert b'filename="week-1/notes.pdf"' in request.content
            return httpx.Response(202, json=_workspace_job_json())
        if path == "/v1/workspaces/workspace/one/rescan":
            assert request.headers["Idempotency-Key"] == "key-2"
            return httpx.Response(202, json=_workspace_job_json())
        if path == "/v1/workspaces/workspace/one/entries/entry/one":
            assert json.loads(request.content) == {"revision_id": "revision-1", "included": False, "subtree": True}
            return httpx.Response(200, json=_manifest_revision_json())
        if path == "/v1/workspaces/workspace/one/approval":
            return httpx.Response(200, json=_approval_json())
        if path == "/v1/workspace-jobs/job/one":
            return httpx.Response(200, json=_workspace_job_json())
        if path == "/v1/workspace-jobs/job/one/events":
            assert dict(request.url.params) == {"after": "4"}
            return httpx.Response(200, json=[])
        if path == "/v1/providers/saved":
            return httpx.Response(200, json=[])
        if path == "/v1/providers/primary/one/credential":
            return httpx.Response(204)
        raise AssertionError(f"unexpected {request.method} {path}")

    client = WorkerClient("http://127.0.0.1:8765", WORKER_TOKEN, transport=httpx.MockTransport(respond))
    try:
        assert client.list_workspaces()[0].workspace_id == "workspace/one"
        assert client.get_workspace("workspace/one").source_mode is SourceMode.BROWSER_SNAPSHOT
        assert client.get_manifest("workspace/one", state=SourceState.PENDING_APPROVAL, course="week-1", offset=2, limit=50).total == 0
        assert client.select_folder("key-1") is None
        assert client.upload_directory("Course", {"week-1/notes.pdf": upload}, "key-1").job_id == "job/one"
        assert client.rescan("workspace/one", "key-2").workspace_id == "workspace/one"
        assert client.set_entry_inclusion("workspace/one", "entry/one", EntryInclusionRequest(revision_id="revision-1", included=False, subtree=True)).revision_id == "revision-1"
        assert client.approve_workspace("workspace/one", "revision-1").approval_id == "approval-1"
        client.delete_workspace("workspace/one")
        client.delete_all_workspaces()
        assert client.get_job("job/one").job_id == "job/one"
        assert client.workspace_events_after("job/one", after=4) == []
        assert client.list_saved_providers() == []
        client.forget_provider_credential("primary/one")
    finally:
        client.close()

    upload.unlink()


@pytest.mark.parametrize("outcome", ["success", "http-failure", "transport-failure"])
def test_upload_directory_closes_opened_files_for_every_request_outcome(
    monkeypatch, tmp_path: Path, outcome: str
):
    upload = tmp_path / "notes.pdf"
    upload.write_bytes(b"notes")
    opened = []
    real_open = Path.open

    def record_open(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        opened.append(handle)
        return handle

    def respond(_request: httpx.Request) -> httpx.Response:
        if outcome == "transport-failure":
            raise httpx.ConnectError("worker unavailable", request=_request)
        if outcome == "http-failure":
            return httpx.Response(503, text="worker unavailable")
        return httpx.Response(202, json=_workspace_job_json())

    monkeypatch.setattr(Path, "open", record_open)
    client = WorkerClient(
        "http://127.0.0.1:8765",
        WORKER_TOKEN,
        transport=httpx.MockTransport(respond),
    )
    try:
        if outcome == "success":
            assert client.upload_directory(
                "Course", {"week-1/notes.pdf": upload}, "key-1"
            ).job_id == "job/one"
        else:
            with pytest.raises(WorkerClientError, match="Agent Worker request failed"):
                client.upload_directory("Course", {"week-1/notes.pdf": upload}, "key-1")
    finally:
        client.close()

    assert len(opened) == 1
    assert opened[0].closed


def test_upload_directory_accepts_a_caller_owned_binary_stream_without_closing_it():
    upload = BytesIO(b"course notes")

    def respond(request: httpx.Request) -> httpx.Response:
        assert b'filename="week-1/notes.pdf"' in request.content
        assert b"course notes" in request.content
        return httpx.Response(202, json=_workspace_job_json())

    client = WorkerClient(
        "http://127.0.0.1:8765",
        WORKER_TOKEN,
        transport=httpx.MockTransport(respond),
    )
    try:
        assert client.upload_directory(
            "Course", {"week-1/notes.pdf": upload}, "key-1"
        ).job_id == "job/one"
    finally:
        client.close()

    assert not upload.closed


def _workspace_summary_json() -> dict[str, object]:
    return {"workspace_id": "workspace/one", "display_name": "Course", "source_mode": "browser_snapshot", "state": "ready", "counts": {}, "updated_at": NOW}


def _workspace_detail_json() -> dict[str, object]:
    return {**_workspace_summary_json(), "current_draft_revision_id": "revision-1", "current_approved_revision_id": None, "created_at": NOW, "last_scanned_at": None, "last_access_verified_at": None}


def _workspace_job_json() -> dict[str, object]:
    return {"job_id": "job/one", "workspace_id": "workspace/one", "job_kind": "scan", "status": "queued", "idempotency_key": "key-1", "safe_error_code": None, "created_at": NOW, "started_at": None, "finished_at": None}


def _manifest_revision_json() -> dict[str, object]:
    return {"revision_id": "revision-1", "workspace_id": "workspace/one", "parent_revision_id": None, "scan_job_id": None, "policy_version": "workspace-v1", "entries": [], "created_at": NOW}


def _approval_json() -> dict[str, object]:
    return {"approval_id": "approval-1", "workspace_id": "workspace/one", "revision_id": "revision-1", "entries": [], "policy_version": "workspace-v1", "approved_at": NOW}
