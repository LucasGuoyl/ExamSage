from __future__ import annotations

from datetime import datetime, timezone

import pytest
from streamlit.testing.v1 import AppTest

from exam_predictor.runtime.client import WorkerClientError
from exam_predictor.runtime.models import (
    AgentEvent,
    EventType,
    HealthResponse,
    ProviderDescriptor,
    ProviderProfile,
    RunSnapshot,
    RunStatus,
    SubmitMessageResponse,
)
from exam_predictor.ui import agent_view
from exam_predictor.ui.agent_view import AgentViewState, reduce_agent_events


NOW = datetime.now(timezone.utc)
VIEW_SCRIPT = "from exam_predictor.ui.agent_view import render_agent_kernel\nrender_agent_kernel()"


def event(sequence: int, event_type: EventType, message: str) -> AgentEvent:
    return AgentEvent(
        sequence=sequence,
        run_id="run-1",
        event_type=event_type,
        stage=event_type.value,
        message=message,
        created_at=NOW,
    )


def snapshot(status: RunStatus) -> RunSnapshot:
    return RunSnapshot(
        run_id="run-1",
        thread_id="course-1",
        provider_profile_id="primary",
        message="Explain limits.",
        status=status,
        created_at=NOW,
        updated_at=NOW,
    )


class FakeWorkerClient:
    def __init__(self):
        self.status = RunStatus.RUNNING
        self.events: list[AgentEvent] = []
        self.connect_error: Exception | None = None
        self.fail_health_after_connect = False
        self.connected_keys: list[str] = []
        self.after_values: list[int] = []
        self.stop_calls = 0
        self.resume_calls = 0
        self.close_calls = 0

    def health(self) -> HealthResponse:
        if self.fail_health_after_connect and self.connected_keys:
            raise WorkerClientError("local-worker-token disconnected")
        return HealthResponse()

    def connect_provider(self, request) -> ProviderDescriptor:
        self.connected_keys.append(request.api_key.get_secret_value())
        if self.connect_error:
            raise self.connect_error
        return ProviderDescriptor(
            profile=request.profile,
            capabilities={"chat": True},
        )

    def submit_message(self, _request) -> SubmitMessageResponse:
        return SubmitMessageResponse(run_id="run-1", status=RunStatus.RUNNING)

    def events_after(self, _run_id: str, after: int = 0) -> list[AgentEvent]:
        self.after_values.append(after)
        return list(self.events)

    def get_run(self, _run_id: str) -> RunSnapshot:
        return snapshot(self.status)

    def stop(self, _run_id: str) -> RunSnapshot:
        self.stop_calls += 1
        self.status = RunStatus.STOPPING
        return snapshot(self.status)

    def resume(self, _run_id: str) -> RunSnapshot:
        self.resume_calls += 1
        self.status = RunStatus.RUNNING
        return snapshot(self.status)

    def close(self) -> None:
        self.close_calls += 1


def button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


def test_event_reducer_is_idempotent_orders_events_and_commits_latest_answer_once():
    state = reduce_agent_events(
        AgentViewState(),
        [
            event(4, EventType.COMPLETED, "Done"),
            event(2, EventType.TOOL_STARTED, "Running tutor_reply"),
            event(1, EventType.PROGRESS, "Planning"),
            event(3, EventType.MESSAGE, "A limit is the value approached."),
            event(2, EventType.TOOL_STARTED, "duplicate"),
        ],
    )

    assert state.last_sequence == 4
    assert state.answer == "A limit is the value approached."
    assert not state.answer_committed
    assert state.settled
    assert state.activity == ["Planning", "Running tutor_reply", "Done"]

    state.answer_committed = True
    same_state = reduce_agent_events(
        state,
        [event(3, EventType.MESSAGE, "duplicate answer"), event(4, EventType.COMPLETED, "duplicate")],
    )
    assert same_state is state
    assert state.answer == "A limit is the value approached."
    assert state.answer_committed
    assert state.activity == ["Planning", "Running tutor_reply", "Done"]


def test_event_reducer_tracks_pause_failure_and_resume_epochs():
    state = reduce_agent_events(
        AgentViewState(),
        [event(1, EventType.PAUSED, "Paused at a checkpoint")],
    )
    assert state.settled and state.paused and not state.failed

    reduce_agent_events(state, [event(2, EventType.RESUMED, "Resumed")])
    assert not state.settled and not state.paused and not state.failed

    reduce_agent_events(state, [event(3, EventType.FAILED, "Provider failed")])
    assert state.settled and not state.paused and state.failed


@pytest.mark.parametrize("fails", [False, True])
def test_provider_key_is_cleared_after_every_connection_attempt_and_errors_are_safe(
    monkeypatch,
    fails: bool,
):
    fake = FakeWorkerClient()
    if fails:
        fake.connect_error = WorkerClientError(
            "provider-api-secret local-worker-token rejected"
        )
    monkeypatch.setattr(agent_view, "_new_client", lambda: fake)
    monkeypatch.setenv("EXAMSAGE_WORKER_TOKEN", "local-worker-token")

    app = AppTest.from_string(VIEW_SCRIPT).run()
    key_input = next(item for item in app.text_input if item.label == "API key")
    key_input.input("provider-api-secret")
    button(app, "Connect").click()
    app.run()

    key_input = next(item for item in app.text_input if item.label == "API key")
    assert key_input.value == ""
    assert fake.connected_keys == ["provider-api-secret"]
    visible_errors = " ".join(item.value for item in app.error)
    assert "provider-api-secret" not in visible_errors
    assert "local-worker-token" not in visible_errors
    if fails:
        assert "Provider connection failed" in visible_errors
    else:
        assert any("Provider connected" in item.value for item in app.success)


def test_worker_unavailable_renders_safely_without_secret_or_legacy_controls(monkeypatch):
    monkeypatch.setenv("EXAMSAGE_WORKER_TOKEN", "local-worker-token")
    monkeypatch.setattr(
        agent_view,
        "_new_client",
        lambda: (_ for _ in ()).throw(
            WorkerClientError("local-worker-token could not connect")
        ),
    )

    app = AppTest.from_string(VIEW_SCRIPT).run()

    assert not app.exception
    assert app.title[0].value == "🎓 ExamSage"
    errors = " ".join(item.value for item in app.error)
    assert "Worker unavailable" in errors
    assert "local-worker-token" not in errors
    labels = [item.label for item in app.button]
    assert "Estimate cost" not in labels
    assert "Build my ExamSage agent" not in labels
    assert not app.file_uploader


def test_provider_key_clears_even_when_worker_disappears_after_attempt(monkeypatch):
    fake = FakeWorkerClient()
    fake.fail_health_after_connect = True
    monkeypatch.setattr(agent_view, "_new_client", lambda: fake)
    monkeypatch.setenv("EXAMSAGE_WORKER_TOKEN", "local-worker-token")

    app = AppTest.from_string(VIEW_SCRIPT).run()
    next(item for item in app.text_input if item.label == "API key").input(
        "provider-api-secret"
    )
    button(app, "Connect").click()
    app.run()

    assert app.session_state["agent_provider_key"] == ""
    errors = " ".join(item.value for item in app.error)
    assert "Worker unavailable" in errors
    assert "provider-api-secret" not in errors
    assert "local-worker-token" not in errors


def test_run_fragment_commits_answer_once_and_displays_stop_resume_transitions(monkeypatch):
    fake = FakeWorkerClient()
    fake.events = [
        event(2, EventType.MESSAGE, "A limit is the value approached."),
        event(1, EventType.PROGRESS, "Planning"),
    ]
    monkeypatch.setattr(agent_view, "_new_client", lambda: fake)

    app = AppTest.from_string(VIEW_SCRIPT)
    app.session_state["agent_messages"] = []
    app.session_state["agent_active_run_id"] = "run-1"
    app.session_state["agent_provider"] = ProviderDescriptor(
        profile=ProviderProfile(profile_id="primary", provider="gemini"),
        capabilities={"chat": True},
    ).model_dump(mode="json")
    app.run()

    assistant_messages = [
        message
        for message in app.session_state["agent_messages"]
        if message["role"] == "assistant"
    ]
    assert assistant_messages == [
        {"role": "assistant", "content": "A limit is the value approached."}
    ]
    app.run()
    assert len(
        [
            message
            for message in app.session_state["agent_messages"]
            if message["role"] == "assistant"
        ]
    ) == 1
    assert max(fake.after_values) == 2

    stop = button(app, "Stop")
    assert not stop.disabled
    stop.click()
    app.run()
    assert fake.stop_calls == 1
    assert any("stopping" in item.label.lower() for item in app.status)
    assert button(app, "Stop").disabled

    fake.status = RunStatus.PAUSED
    app.run()
    button(app, "Resume").click()
    app.run()
    assert fake.resume_calls == 1
    assert any("running" in item.label.lower() for item in app.status)


def test_submit_requires_provider_then_starts_worker_run(monkeypatch):
    fake = FakeWorkerClient()
    monkeypatch.setattr(agent_view, "_new_client", lambda: fake)
    app = AppTest.from_string(VIEW_SCRIPT).run()

    app.chat_input[0].set_value("Explain limits.")
    app.run()
    assert any("Connect one provider" in item.value for item in app.warning)
    assert "agent_active_run_id" not in app.session_state

    app.session_state["agent_provider"] = {
        "profile": {"profile_id": "primary", "provider": "gemini"},
        "capabilities": {"chat": True},
    }
    app.chat_input[0].set_value("Explain limits.")
    app.run()
    assert app.session_state["agent_active_run_id"] == "run-1"
    assert app.session_state["agent_messages"][-1] == {
        "role": "user",
        "content": "Explain limits.",
    }
