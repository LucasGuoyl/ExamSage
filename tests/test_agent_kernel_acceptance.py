from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from exam_predictor.agent import ExamSageAgent
from exam_predictor.runtime.coordinator import RuntimeCoordinator
from exam_predictor.runtime.models import EventType, RunStatus
from exam_predictor.runtime.provider_sessions import ProviderSessionRegistry
from exam_predictor.runtime.store import RuntimeStore
from exam_predictor.worker.api import WorkerSettings, create_worker_app


WORKER_TOKEN = "acceptance-worker-token"
API_KEY = "test-only" + "-secret"
AUTH = {"X-ExamSage-Token": WORKER_TOKEN}
pytestmark = pytest.mark.anyio


class ProviderHarness:
    def __init__(self) -> None:
        self.planner_payloads: list[dict[str, object]] = []
        self.block_started = Event()
        self.release_block = Event()
        self.http_bodies: list[str] = []

    def factory(self, config: dict[str, object]) -> FakeProvider:
        assert config["api_key"] == API_KEY
        return FakeProvider(self)

    def record(self, response: httpx.Response) -> httpx.Response:
        self.http_bodies.append(response.text)
        return response


class FakeProvider:
    name = "fake"
    capabilities = SimpleNamespace(chat=True, vision=True, web_search=True)
    models = SimpleNamespace(fast="fake-fast", balanced="fake-balanced")

    def __init__(self, harness: ProviderHarness) -> None:
        self.harness = harness

    def create_chat_completion(self, **kwargs: object) -> SimpleNamespace:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        if "Choose exactly one ExamSage kernel tool" in messages[0]["content"]:
            payload = json.loads(messages[1]["content"])
            self.harness.planner_payloads.append(payload)
            content = json.dumps(
                {
                    "tool": "tutor_reply",
                    "arguments": {},
                    "reason": "The user requested tutoring.",
                }
            )
        else:
            message = messages[-1]["content"]
            if "Block safely" in message:
                self.harness.block_started.set()
                if not self.harness.release_block.wait(timeout=3):
                    raise TimeoutError("Acceptance harness was not released in time.")
            content = f"Worked answer for: {message}"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def runtime_for(tmp_path: Path, harness: ProviderHarness) -> RuntimeCoordinator:
    return RuntimeCoordinator(
        store=RuntimeStore(tmp_path / "agent-runtime.sqlite3"),
        provider_sessions=ProviderSessionRegistry(factory=harness.factory),
        checkpoints_path=tmp_path / "agent-checkpoints.sqlite3",
    )


async def worker_client(
    settings: WorkerSettings,
    runtime: RuntimeCoordinator,
):
    app = create_worker_app(settings, runtime=runtime)
    transport = httpx.ASGITransport(app=app)
    return app, httpx.AsyncClient(transport=transport, base_url="http://testserver")


async def connect(client: httpx.AsyncClient, harness: ProviderHarness) -> None:
    response = harness.record(
        await client.post(
            "/v1/providers/connect",
            headers=AUTH,
            json={
                "profile": {"profile_id": "primary", "provider": "gemini"},
                "api_key": API_KEY,
            },
        )
    )
    assert response.status_code == 200


async def submit(
    client: httpx.AsyncClient,
    harness: ProviderHarness,
    thread_id: str,
    message: str,
) -> dict[str, str]:
    response = harness.record(
        await client.post(
            f"/v1/threads/{thread_id}/messages",
            headers=AUTH,
            json={"provider_profile_id": "primary", "message": message},
        )
    )
    assert response.status_code == 202
    return response.json()


async def get_run(
    client: httpx.AsyncClient,
    harness: ProviderHarness,
    run_id: str,
) -> dict[str, object]:
    response = harness.record(await client.get(f"/v1/runs/{run_id}", headers=AUTH))
    assert response.status_code == 200
    return response.json()


async def wait_for_status(
    client: httpx.AsyncClient,
    harness: ProviderHarness,
    run_id: str,
    expected: RunStatus,
) -> dict[str, object]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        run = await get_run(client, harness, run_id)
        if run["status"] == expected.value:
            return run
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} did not reach {expected.value}")


async def get_events(
    client: httpx.AsyncClient,
    harness: ProviderHarness,
    run_id: str,
) -> list[dict[str, object]]:
    response = harness.record(
        await client.get(f"/v1/runs/{run_id}/events", headers=AUTH)
    )
    assert response.status_code == 200
    return response.json()


async def test_agent_kernel_vertical_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    harness = ProviderHarness()
    settings = WorkerSettings(port=8765, token=WORKER_TOKEN, data_dir=tmp_path)
    runtime = runtime_for(tmp_path, harness)

    app, client = await worker_client(settings, runtime)
    async with app.router.lifespan_context(app):
        async with client:
            await connect(client, harness)

            first_submission = await submit(client, harness, "calculus", "Explain limits.")
            first = first_submission["run_id"]
            assert first_submission["status"] == RunStatus.RUNNING.value
            await wait_for_status(client, harness, first, RunStatus.COMPLETED)
            first_events = await get_events(client, harness, first)
            assert [event["event_type"] for event in first_events] == [
                EventType.STARTED.value,
                EventType.PROGRESS.value,
                EventType.TOOL_STARTED.value,
                EventType.TOOL_COMPLETED.value,
                EventType.MESSAGE.value,
                EventType.COMPLETED.value,
            ]
            assert [event["sequence"] for event in first_events] == sorted(
                event["sequence"] for event in first_events
            )
            tool_events = [
                event
                for event in first_events
                if event["event_type"] == EventType.TOOL_STARTED.value
            ]
            assert len(tool_events) == 1
            assert tool_events[0]["payload"] == {"tool": "tutor_reply"}
            answer_events = [
                event
                for event in first_events
                if event["event_type"] == EventType.MESSAGE.value
            ]
            assert [event["message"] for event in answer_events] == [
                "Worked answer for: Explain limits."
            ]
            assert len(harness.planner_payloads) == 1

            follow_up_submission = await submit(
                client,
                harness,
                "calculus",
                "Give one example.",
            )
            follow_up = follow_up_submission["run_id"]
            await wait_for_status(client, harness, follow_up, RunStatus.COMPLETED)
            follow_up_history = harness.planner_payloads[-1]["history"]
            assert isinstance(follow_up_history, list)
            assert any(
                item["role"] == "assistant" and "Worked answer" in item["content"]
                for item in follow_up_history
            )

            blocked_submission = await submit(
                client,
                harness,
                "physics",
                "Block safely before composing.",
            )
            blocked = blocked_submission["run_id"]
            assert harness.block_started.wait(timeout=3)
            try:
                queued_submission = await submit(
                    client,
                    harness,
                    "chemistry",
                    "Explain covalent bonds.",
                )
                queued = queued_submission["run_id"]
                assert queued_submission["status"] == RunStatus.QUEUED.value
                assert (await get_run(client, harness, queued))["status"] == RunStatus.QUEUED.value

                stopped = harness.record(
                    await client.post(f"/v1/runs/{blocked}/stop", headers=AUTH)
                )
                assert stopped.status_code == 200
                assert stopped.json()["status"] == RunStatus.STOPPING.value
                assert (await get_run(client, harness, queued))["status"] == RunStatus.QUEUED.value
            finally:
                harness.release_block.set()

            await wait_for_status(client, harness, blocked, RunStatus.PAUSED)
            assert (await get_run(client, harness, queued))["status"] == RunStatus.QUEUED.value

    restarted = runtime_for(tmp_path, harness)
    app, client = await worker_client(settings, restarted)
    async with app.router.lifespan_context(app):
        async with client:
            assert (await get_run(client, harness, blocked))["status"] == RunStatus.PAUSED.value
            assert (await get_run(client, harness, queued))["status"] == RunStatus.QUEUED.value

            missing_provider = harness.record(
                await client.post(f"/v1/runs/{blocked}/resume", headers=AUTH)
            )
            assert missing_provider.status_code == 409
            assert missing_provider.json() == {
                "detail": "Connect provider profile before starting or resuming this run."
            }
            assert restarted.store.get_run(blocked).status is RunStatus.PAUSED

            await connect(client, harness)
            resumed = harness.record(
                await client.post(f"/v1/runs/{blocked}/resume", headers=AUTH)
            )
            assert resumed.status_code == 200
            assert resumed.json()["run_id"] == blocked
            await wait_for_status(client, harness, blocked, RunStatus.COMPLETED)
            await wait_for_status(client, harness, queued, RunStatus.COMPLETED)

            blocked_events = await get_events(client, harness, blocked)
            queued_events = await get_events(client, harness, queued)
            blocked_completed = next(
                event["sequence"]
                for event in blocked_events
                if event["event_type"] == EventType.COMPLETED.value
            )
            queued_started = next(
                event["sequence"]
                for event in queued_events
                if event["event_type"] == EventType.STARTED.value
            )
            assert blocked_completed < queued_started
            assert queued_events[0]["event_type"] == EventType.QUEUED.value
            assert queued_events[-1]["event_type"] == EventType.COMPLETED.value

            unfinished = restarted.store.create_run(
                "restart-recovery",
                "primary",
                "Recover my unfinished metadata.",
                RunStatus.RUNNING,
            )
            restarted.store.append_event(
                unfinished.run_id,
                EventType.STARTED,
                "queue",
                "Agent run started.",
            )

    recovered = runtime_for(tmp_path, harness)
    app, client = await worker_client(settings, recovered)
    async with app.router.lifespan_context(app):
        async with client:
            recovered_run = await get_run(client, harness, unfinished.run_id)
            assert recovered_run["status"] == RunStatus.PAUSED.value
            recovery_events = await get_events(client, harness, unfinished.run_id)
            assert recovery_events[-1]["event_type"] == EventType.PAUSED.value
            assert "Select Resume" in recovery_events[-1]["message"]

    snapshots = [
        runtime.store.get_run(run_id).model_dump(mode="json")
        for run_id in (first, follow_up, blocked, queued, unfinished.run_id)
    ]
    durable_json = json.dumps(
        {
            "runs": snapshots,
            "events": {
                run_id: [event.model_dump(mode="json") for event in runtime.store.list_events(run_id)]
                for run_id in (first, follow_up, blocked, queued, unfinished.run_id)
            },
        },
        ensure_ascii=False,
    )
    assert API_KEY not in durable_json
    assert API_KEY not in "".join(harness.http_bodies)
    assert API_KEY not in caplog.text

    sqlite_files = list(tmp_path.glob("agent-*.sqlite3*"))
    assert tmp_path / "agent-runtime.sqlite3" in sqlite_files
    assert tmp_path / "agent-checkpoints.sqlite3" in sqlite_files
    for sqlite_file in sqlite_files:
        assert API_KEY.encode() not in sqlite_file.read_bytes()

    def forbidden_build(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("legacy build_course must not run in Agent mode")

    monkeypatch.setattr(ExamSageAgent, "build_course", forbidden_build)
    monkeypatch.setenv("EXAMSAGE_AGENT_V2", "1")
    monkeypatch.setenv("EXAMSAGE_WORKER_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("EXAMSAGE_WORKER_TOKEN", WORKER_TOKEN)
    monkeypatch.setenv("EXAMSAGE_DATA_DIR", str(tmp_path / "ui-data"))
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    streamlit_app = AppTest.from_file(str(app_path), default_timeout=60).run()
    assert not streamlit_app.exception
    assert any("Worker unavailable" in item.value for item in streamlit_app.error)
    assert "Estimate cost" not in [button.label for button in streamlit_app.button]
    assert "Build my ExamSage agent" not in [button.label for button in streamlit_app.button]
