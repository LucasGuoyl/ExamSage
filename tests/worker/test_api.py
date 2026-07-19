from __future__ import annotations

import asyncio
import sys
from contextlib import suppress
from types import SimpleNamespace

import httpx
import pytest

from exam_predictor.runtime.coordinator import ProviderProfileInUseError
from exam_predictor.runtime.models import EventType, RunStatus
from exam_predictor.runtime.provider_sessions import ProviderSessionRegistry
from exam_predictor.runtime.store import RuntimeStore
import exam_predictor.worker.api as worker_api
from exam_predictor.worker.api import WorkerSettings, create_worker_app
from exam_predictor.worker.main import main


WORKER_TOKEN = "local-worker-token"
API_KEY = "provider-api-secret"
pytestmark = pytest.mark.anyio


class FakeProvider:
    name = "fake"
    capabilities = SimpleNamespace(chat=True, vision=True, web_search=False)


class FakeRuntime:
    def __init__(self, path, *, provider_factory=None):
        self.store = RuntimeStore(path)
        self.provider_sessions = ProviderSessionRegistry(
            factory=provider_factory or (lambda config: FakeProvider())
        )
        self.lifecycle_calls: list[str] = []

    def start(self):
        self.lifecycle_calls.append("start")

    def connect_provider(self, request):
        return self.provider_sessions.connect(request)

    def submit_message(self, thread_id, provider_profile_id, message):
        self.provider_sessions.get_provider(provider_profile_id)
        run = self.store.create_run(
            thread_id,
            provider_profile_id,
            message,
            RunStatus.RUNNING,
        )
        self.store.append_event(
            run.run_id,
            EventType.STARTED,
            "queue",
            "Agent run started.",
        )
        return run

    def stop(self, run_id):
        run = self.store.get_run(run_id)
        if run.status is not RunStatus.RUNNING:
            raise ValueError("Only a running Agent run can be stopped.")
        self.store.append_event(
            run_id,
            EventType.STOP_REQUESTED,
            "stopping",
            "Stop requested.",
        )
        return self.store.set_status(run_id, RunStatus.STOPPING)

    def resume(self, run_id):
        run = self.store.get_run(run_id)
        if run.status is not RunStatus.PAUSED:
            raise ValueError("Only a paused Agent run can be resumed.")
        self.provider_sessions.get_provider(run.provider_profile_id)
        self.store.set_status(run_id, RunStatus.RUNNING)
        self.store.append_event(
            run_id,
            EventType.RESUMED,
            "queue",
            "Resume requested.",
        )
        return self.store.get_run(run_id)

    def pause_all(self):
        self.lifecycle_calls.append("pause_all")

    def shutdown(self):
        self.lifecycle_calls.append("shutdown")


@pytest.fixture
def runtime(tmp_path):
    return FakeRuntime(tmp_path / "runtime.sqlite3")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client(tmp_path, runtime):
    settings = WorkerSettings(port=8765, token=WORKER_TOKEN, data_dir=tmp_path)
    app = create_worker_app(settings, runtime=runtime)
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as test_client:
            yield test_client


@pytest.fixture
def auth_headers():
    return {"X-ExamSage-Token": WORKER_TOKEN}


async def connect_provider(client, auth_headers):
    return await client.post(
        "/v1/providers/connect",
        headers=auth_headers,
        json={
            "profile": {"profile_id": "primary", "provider": "gemini"},
            "api_key": API_KEY,
        },
    )


class SignalingClock:
    def __init__(self, ready: asyncio.Event):
        self.ready = ready
        self.calls = 0

    def monotonic(self):
        self.calls += 1
        if self.calls >= 2:
            self.ready.set()
            return 15.0
        return 0.0


@pytest.mark.parametrize("token", ["", " ", "\t\n"])
def test_worker_settings_reject_empty_token(tmp_path, token):
    with pytest.raises(ValueError, match="token"):
        WorkerSettings(port=8765, token=token, data_dir=tmp_path)


def test_worker_settings_reject_non_loopback_bind(tmp_path):
    with pytest.raises(ValueError, match="127.0.0.1"):
        WorkerSettings(
            host="0.0.0.0",
            port=8765,
            token=WORKER_TOKEN,
            data_dir=tmp_path,
        )


@pytest.mark.parametrize(
    ("method", "path", "json"),
    [
        (
            "post",
            "/v1/providers/connect",
            {
                "profile": {"profile_id": "primary", "provider": "gemini"},
                "api_key": API_KEY,
            },
        ),
        (
            "post",
            "/v1/threads/course-1/messages",
            {"provider_profile_id": "primary", "message": "Explain limits."},
        ),
        ("get", "/v1/runs/missing", None),
        ("get", "/v1/runs/missing/events", None),
        ("get", "/v1/runs/missing/stream", None),
        ("post", "/v1/runs/missing/stop", None),
        ("post", "/v1/runs/missing/resume", None),
        ("post", "/v1/runtime/pause-all", None),
    ],
)
@pytest.mark.parametrize("headers", [None, {"X-ExamSage-Token": "wrong"}])
async def test_every_v1_route_requires_exact_token(
    client,
    method,
    path,
    json,
    headers,
):
    response = await client.request(method, path, headers=headers, json=json)

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized."}
    assert WORKER_TOKEN not in response.text
    assert API_KEY not in response.text


async def test_health_is_the_only_unauthenticated_route(client):
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "agent_v2": True}


async def test_connect_success_does_not_echo_secrets(client, auth_headers):
    response = await connect_provider(client, auth_headers)

    assert response.status_code == 200
    assert response.json()["profile"]["profile_id"] == "primary"
    assert response.json()["capabilities"]["chat"] is True
    assert API_KEY not in response.text
    assert WORKER_TOKEN not in response.text


async def test_replacing_provider_used_by_active_run_returns_stable_safe_409(
    client,
    auth_headers,
    runtime,
):
    assert (await connect_provider(client, auth_headers)).status_code == 200

    def reject_replacement(_request):
        raise ProviderProfileInUseError(
            f"profile primary currently in use; rejected {API_KEY} {WORKER_TOKEN}"
        )

    runtime.connect_provider = reject_replacement
    response = await client.post(
        "/v1/providers/connect",
        headers=auth_headers,
        json={
            "profile": {"profile_id": "primary", "provider": "gemini"},
            "api_key": API_KEY,
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Provider profile is currently in use by an active run."
    }
    assert API_KEY not in response.text
    assert WORKER_TOKEN not in response.text


async def test_connect_failure_is_redacted_and_has_no_exception_chain(tmp_path):
    def fail(config):
        raise RuntimeError(
            f"provider rejected {config['api_key']}; request marker {WORKER_TOKEN}"
        )

    runtime = FakeRuntime(tmp_path / "runtime.sqlite3", provider_factory=fail)
    settings = WorkerSettings(port=8765, token=WORKER_TOKEN, data_dir=tmp_path)

    app = create_worker_app(settings, runtime=runtime)
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await connect_provider(
                client,
                {"X-ExamSage-Token": WORKER_TOKEN},
            )

    assert response.status_code == 400
    assert response.json() == {"detail": "Provider connection failed."}
    assert API_KEY not in response.text
    assert WORKER_TOKEN not in response.text


async def test_connect_validation_failure_does_not_echo_secrets(
    client,
    auth_headers,
):
    response = await client.post(
        "/v1/providers/connect",
        headers=auth_headers,
        json={"api_key": API_KEY},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request."}
    assert API_KEY not in response.text
    assert WORKER_TOKEN not in response.text


async def test_authentication_precedes_malformed_body_validation(client):
    response = await client.post(
        "/v1/providers/connect",
        json={"api_key": API_KEY},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized."}
    assert API_KEY not in response.text


@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Type": "application/json"},
        {
            "Content-Type": "application/json",
            "X-ExamSage-Token": "wrong",
        },
    ],
)
async def test_malformed_json_is_rejected_at_auth_boundary(client, headers):
    response = await client.post(
        "/v1/providers/connect",
        headers=headers,
        content=b"{",
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized."}
    assert API_KEY not in response.text
    assert WORKER_TOKEN not in response.text


async def test_authenticated_malformed_json_returns_safe_422(client, auth_headers):
    response = await client.post(
        "/v1/providers/connect",
        headers={**auth_headers, "Content-Type": "application/json"},
        content=b"{",
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request."}
    assert API_KEY not in response.text
    assert WORKER_TOKEN not in response.text


async def test_auth_boundary_matches_only_v1_root_and_subpaths(client, auth_headers):
    assert (await client.get("/v1")).status_code == 401
    assert (
        await client.get(
            "/v1/",
            headers={"X-ExamSage-Token": "wrong"},
        )
    ).status_code == 401
    assert (await client.get("/v1", headers=auth_headers)).status_code == 404
    assert (await client.get("/v10")).status_code == 404


async def test_auth_boundary_rejects_duplicate_token_headers(client):
    response = await client.get(
        "/v1/runs/missing",
        headers=[
            ("X-ExamSage-Token", WORKER_TOKEN),
            ("X-ExamSage-Token", WORKER_TOKEN),
        ],
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized."}
    assert WORKER_TOKEN not in response.text
    assert API_KEY not in response.text


async def test_valid_message_run_event_stop_resume_and_pause_all_routes(
    client,
    auth_headers,
    runtime,
):
    assert (await connect_provider(client, auth_headers)).status_code == 200
    submitted = await client.post(
        "/v1/threads/course-1/messages",
        headers=auth_headers,
        json={"provider_profile_id": "primary", "message": "Explain limits."},
    )

    assert submitted.status_code == 202
    run_id = submitted.json()["run_id"]
    run_response = await client.get(f"/v1/runs/{run_id}", headers=auth_headers)
    events_response = await client.get(
        f"/v1/runs/{run_id}/events",
        headers=auth_headers,
    )
    stop_response = await client.post(
        f"/v1/runs/{run_id}/stop",
        headers=auth_headers,
    )
    runtime.store.set_status(run_id, RunStatus.PAUSED)
    runtime.store.append_event(
        run_id,
        EventType.PAUSED,
        "paused",
        "Paused at a safe boundary.",
    )
    resume_response = await client.post(
        f"/v1/runs/{run_id}/resume",
        headers=auth_headers,
    )
    lifecycle_events = await client.get(
        f"/v1/runs/{run_id}/events",
        headers=auth_headers,
    )
    pause_response = await client.post(
        "/v1/runtime/pause-all",
        headers=auth_headers,
    )

    assert run_response.status_code == 200
    assert run_response.json()["run_id"] == run_id
    assert events_response.status_code == 200
    assert [event["event_type"] for event in events_response.json()] == ["started"]
    assert stop_response.status_code == 200
    assert stop_response.json()["status"] == "stopping"
    assert resume_response.status_code == 200
    assert resume_response.json()["status"] == "running"
    assert [event["event_type"] for event in lifecycle_events.json()] == [
        "started",
        "stop_requested",
        "paused",
        "resumed",
    ]
    assert pause_response.status_code == 204
    assert runtime.lifecycle_calls.count("pause_all") == 1


@pytest.mark.parametrize(
    ("thread_id", "body"),
    [
        ("course-1", {"provider_profile_id": "primary", "message": ""}),
        ("course-1", {"provider_profile_id": "primary", "message": "   "}),
        ("course-1", {"provider_profile_id": "   ", "message": "Explain"}),
        ("%20%20", {"provider_profile_id": "primary", "message": "Explain"}),
    ],
)
async def test_message_route_rejects_blank_required_text_with_stable_422(
    client,
    auth_headers,
    thread_id,
    body,
):
    assert (await connect_provider(client, auth_headers)).status_code == 200

    response = await client.post(
        f"/v1/threads/{thread_id}/messages",
        headers=auth_headers,
        json=body,
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request."}
    assert API_KEY not in response.text
    assert WORKER_TOKEN not in response.text


async def test_message_route_uses_shared_required_text_normalization(
    client,
    auth_headers,
    runtime,
):
    assert (await connect_provider(client, auth_headers)).status_code == 200

    response = await client.post(
        "/v1/threads/%20course-1%20/messages",
        headers=auth_headers,
        json={"provider_profile_id": " primary ", "message": " Explain limits. "},
    )

    assert response.status_code == 202
    run = runtime.store.get_run(response.json()["run_id"])
    assert run.thread_id == "course-1"
    assert run.provider_profile_id == "primary"
    assert run.message == "Explain limits."


async def test_message_authentication_precedes_blank_body_validation(client):
    response = await client.post(
        "/v1/threads/course-1/messages",
        json={"provider_profile_id": "", "message": ""},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized."}


async def test_events_after_returns_only_later_durable_sequences(
    client,
    auth_headers,
    runtime,
):
    run = runtime.store.create_run("course-1", "primary", "Explain", RunStatus.RUNNING)
    first = runtime.store.append_event(
        run.run_id,
        EventType.MESSAGE,
        "answer",
        "First",
    )
    second = runtime.store.append_event(
        run.run_id,
        EventType.PROGRESS,
        "answer",
        "Second",
    )

    response = await client.get(
        f"/v1/runs/{run.run_id}/events",
        params={"after": first.sequence},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert [event["sequence"] for event in response.json()] == [second.sequence]


async def test_resume_without_reconnected_provider_returns_stable_409(
    client,
    auth_headers,
    runtime,
):
    run = runtime.store.create_run("course-1", "missing", "Explain", RunStatus.PAUSED)

    response = await client.post(
        f"/v1/runs/{run.run_id}/resume",
        headers=auth_headers,
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Connect provider profile before starting or resuming this run."
    }
    assert "missing" not in response.text
    assert runtime.store.get_run(run.run_id).status is RunStatus.PAUSED


@pytest.mark.parametrize(
    ("operation", "run_status"),
    [
        ("stop", RunStatus.PAUSED),
        ("resume", RunStatus.RUNNING),
    ],
)
async def test_invalid_run_transition_returns_stable_409(
    client,
    auth_headers,
    runtime,
    operation,
    run_status,
):
    run = runtime.store.create_run("course-1", "primary", "Explain", run_status)

    response = await client.post(
        f"/v1/runs/{run.run_id}/{operation}",
        headers=auth_headers,
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Run state conflicts with this operation."}
    assert run.run_id not in response.text


@pytest.mark.parametrize(
    ("method", "suffix"),
    [
        ("get", ""),
        ("get", "/events"),
        ("get", "/stream"),
        ("post", "/stop"),
        ("post", "/resume"),
    ],
)
async def test_missing_run_routes_return_stable_404(
    client,
    auth_headers,
    method,
    suffix,
):
    response = await client.request(
        method,
        f"/v1/runs/private-run-id{suffix}",
        headers=auth_headers,
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Agent run was not found."}
    assert "private-run-id" not in response.text


async def test_sse_respects_after_emits_ordered_frames_and_drains_settled_run(
    client,
    auth_headers,
    runtime,
):
    run = runtime.store.create_run("course-1", "primary", "Explain", RunStatus.RUNNING)
    skipped = runtime.store.append_event(
        run.run_id,
        EventType.STARTED,
        "queue",
        "Started",
    )
    message = runtime.store.append_event(
        run.run_id,
        EventType.MESSAGE,
        "answer",
        "Answer",
    )
    complete = runtime.store.append_event(
        run.run_id,
        EventType.COMPLETED,
        "complete",
        "Done",
    )
    runtime.store.set_status(run.run_id, RunStatus.COMPLETED)

    async with client.stream(
        "GET",
        f"/v1/runs/{run.run_id}/stream",
        params={"after": skipped.sequence},
        headers=auth_headers,
    ) as response:
        body = "".join([chunk async for chunk in response.aiter_text()])

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert f"id: {skipped.sequence}\n" not in body
    assert body.index(f"id: {message.sequence}\n") < body.index(
        f"id: {complete.sequence}\n"
    )
    assert f"event: {message.event_type.value}\n" in body
    assert f"event: {complete.event_type.value}\n" in body
    assert body.count("data: {") == 2
    assert "\n\n" in body
    assert WORKER_TOKEN not in body
    assert API_KEY not in body


@pytest.mark.parametrize(
    ("run_status", "terminal_type"),
    [
        (RunStatus.PAUSED, EventType.PAUSED),
        (RunStatus.FAILED, EventType.FAILED),
    ],
)
async def test_sse_closes_after_status_specific_terminal_event(
    client,
    auth_headers,
    runtime,
    run_status,
    terminal_type,
):
    run = runtime.store.create_run(
        "course-1",
        "primary",
        "Explain",
        RunStatus.RUNNING,
    )
    terminal = runtime.store.append_event(
        run.run_id,
        terminal_type,
        terminal_type.value,
        "Settled",
    )
    runtime.store.set_status(run.run_id, run_status)

    async with client.stream(
        "GET",
        f"/v1/runs/{run.run_id}/stream",
        headers=auth_headers,
    ) as response:
        body = "".join([chunk async for chunk in response.aiter_text()])

    assert response.status_code == 200
    assert f"id: {terminal.sequence}\n" in body
    assert f"event: {terminal_type.value}\n" in body


async def test_sse_waits_for_terminal_event_published_after_settled_status(
    client,
    auth_headers,
    runtime,
    monkeypatch,
):
    run = runtime.store.create_run(
        "course-1",
        "primary",
        "Explain",
        RunStatus.RUNNING,
    )
    runtime.store.set_status(run.run_id, RunStatus.COMPLETED)
    ready = asyncio.Event()
    monkeypatch.setattr(worker_api, "time", SignalingClock(ready))

    async def publish_terminal_event():
        await ready.wait()
        runtime.store.append_event(
            run.run_id,
            EventType.COMPLETED,
            "complete",
            "Done",
        )

    publisher = asyncio.create_task(publish_terminal_event())
    try:
        async with client.stream(
            "GET",
            f"/v1/runs/{run.run_id}/stream",
            headers=auth_headers,
        ) as response:
            body = "".join([chunk async for chunk in response.aiter_text()])
    finally:
        if not publisher.done():
            publisher.cancel()
        with suppress(asyncio.CancelledError):
            await publisher

    assert response.status_code == 200
    assert "event: completed\n" in body
    assert '"message":"Done"' in body


async def test_sse_waits_for_current_pause_epoch_terminal_event(
    client,
    auth_headers,
    runtime,
    monkeypatch,
):
    run = runtime.store.create_run(
        "course-1",
        "primary",
        "Explain",
        RunStatus.RUNNING,
    )
    runtime.store.append_event(
        run.run_id,
        EventType.STARTED,
        "queue",
        "Started",
    )
    first_pause = runtime.store.append_event(
        run.run_id,
        EventType.PAUSED,
        "paused",
        "First pause",
    )
    resumed = runtime.store.append_event(
        run.run_id,
        EventType.RESUMED,
        "queue",
        "Resumed",
    )
    runtime.store.set_status(run.run_id, RunStatus.PAUSED)
    ready = asyncio.Event()
    monkeypatch.setattr(worker_api, "time", SignalingClock(ready))

    async def publish_second_pause():
        await ready.wait()
        runtime.store.append_event(
            run.run_id,
            EventType.PAUSED,
            "paused",
            "Second pause",
        )

    publisher = asyncio.create_task(publish_second_pause())
    try:
        async with client.stream(
            "GET",
            f"/v1/runs/{run.run_id}/stream",
            params={"after": resumed.sequence},
            headers=auth_headers,
        ) as response:
            body = "".join([chunk async for chunk in response.aiter_text()])
    finally:
        if not publisher.done():
            publisher.cancel()
        with suppress(asyncio.CancelledError):
            await publisher

    assert response.status_code == 200
    assert f"id: {first_pause.sequence}\n" not in body
    assert '"message":"Second pause"' in body


async def test_sse_emits_heartbeat_while_waiting_for_live_events(
    client,
    auth_headers,
    runtime,
    monkeypatch,
):
    run = runtime.store.create_run(
        "course-1",
        "primary",
        "Explain",
        RunStatus.RUNNING,
    )
    ready = asyncio.Event()
    monkeypatch.setattr(worker_api, "time", SignalingClock(ready))

    async def complete_run():
        await ready.wait()
        runtime.store.set_status(run.run_id, RunStatus.COMPLETED)
        runtime.store.append_event(
            run.run_id,
            EventType.COMPLETED,
            "complete",
            "Done",
        )

    publisher = asyncio.create_task(complete_run())
    async with client.stream(
        "GET",
        f"/v1/runs/{run.run_id}/stream",
        headers=auth_headers,
    ) as response:
        body = "".join([chunk async for chunk in response.aiter_text()])
    await publisher

    assert response.status_code == 200
    assert ": keep-alive\n\n" in body
    assert "event: completed\n" in body


async def test_lifespan_starts_then_pauses_and_shuts_down_runtime(tmp_path, runtime):
    settings = WorkerSettings(port=8765, token=WORKER_TOKEN, data_dir=tmp_path)
    app = create_worker_app(settings, runtime=runtime)

    async with app.router.lifespan_context(app):
        assert runtime.lifecycle_calls == ["start"]

    assert runtime.lifecycle_calls == ["start", "shutdown"]


def test_cli_rejects_non_loopback_host(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["examsage-worker", "--host", "0.0.0.0", "--port", "8765"],
    )
    monkeypatch.setenv("EXAMSAGE_WORKER_TOKEN", WORKER_TOKEN)

    with pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize("token", [None, "", "  "])
def test_cli_requires_nonempty_environment_token(monkeypatch, token):
    monkeypatch.setattr(sys, "argv", ["examsage-worker", "--port", "8765"])
    if token is None:
        monkeypatch.delenv("EXAMSAGE_WORKER_TOKEN", raising=False)
    else:
        monkeypatch.setenv("EXAMSAGE_WORKER_TOKEN", token)

    with pytest.raises(SystemExit, match="EXAMSAGE_WORKER_TOKEN is required"):
        main()


def test_cli_runs_uvicorn_on_loopback_without_echoing_token(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sys, "argv", ["examsage-worker", "--port", "8765"])
    monkeypatch.setenv("EXAMSAGE_WORKER_TOKEN", WORKER_TOKEN)
    monkeypatch.setenv("EXAMSAGE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("exam_predictor.worker.main.uvicorn.run", lambda *args, **kwargs: calls.append((args, kwargs)))

    main()

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs == {"host": "127.0.0.1", "port": 8765, "log_level": "warning"}
    assert WORKER_TOKEN not in repr(calls)
