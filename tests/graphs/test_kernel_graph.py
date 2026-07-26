import json
from pathlib import Path
from shutil import copyfile
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from exam_predictor.graphs.kernel import KernelDependencies, build_kernel_graph
from exam_predictor.runtime.control import RunControlRegistry
from exam_predictor.tools.kernel import KernelPlanner, KernelToolRegistry


TEST_CREDENTIAL_SENTINEL = "checkpoint-credential-sentinel"


class FakeProvider:
    name = "fake"
    capabilities = SimpleNamespace(chat=True)
    models = SimpleNamespace(fast="fast", balanced="balanced")

    def __init__(self):
        self.calls = 0
        self.api_key = TEST_CREDENTIAL_SENTINEL
        self.requests: list[dict[str, Any]] = []

    def create_chat_completion(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        content = (
            json.dumps({"tool": "tutor_reply", "arguments": {}, "reason": "Tutor request"})
            if self.calls % 2 == 1
            else "A limit is the value approached by a function."
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class Sessions:
    def __init__(self, provider: FakeProvider | None = None):
        self.provider = provider or FakeProvider()

    def get_provider(self, profile_id: str):
        assert profile_id == "primary"
        return self.provider


def dependencies(
    events: list[dict],
    controls: RunControlRegistry,
    *,
    provider: FakeProvider | None = None,
    on_emit=None,
):
    def emit(run_id, event_type, stage, message, payload=None):
        event = {
            "run_id": run_id,
            "event_type": event_type,
            "stage": stage,
            "message": message,
            "payload": payload or {},
        }
        events.append(event)
        if on_emit is not None:
            on_emit(event)

    return KernelDependencies(
        provider_sessions=Sessions(provider),
        planner=KernelPlanner(),
        tools=KernelToolRegistry(),
        controls=controls,
        emit=emit,
    )


def test_graph_runs_tool_and_persists_follow_up_state(tmp_path: Path):
    events: list[dict] = []
    controls = RunControlRegistry()
    provider = FakeProvider()
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite3")) as saver:
        graph = build_kernel_graph(dependencies(events, controls, provider=provider), saver)
        config = {"configurable": {"thread_id": "calculus"}}
        first = graph.invoke({
            "run_id": "run-1",
            "provider_profile_id": "primary",
            "user_message": "Explain limits.",
            "messages": [{"role": "user", "content": "Explain limits."}],
        }, config)
        result = graph.invoke({
            "run_id": "run-1-follow-up",
            "provider_profile_id": "primary",
            "user_message": "Can you say that another way?",
            "messages": [{"role": "user", "content": "Can you say that another way?"}],
        }, config)
        assert "approached" in first["assistant_message"]
        assert result["selected_tool"] == "tutor_reply"
        assert [message["role"] for message in result["messages"]] == [
            "user", "assistant", "user", "assistant"
        ]
        assert result["messages"][1] == {
            "role": "assistant",
            "content": first["assistant_message"],
        }
        planner_payload = json.loads(provider.requests[2]["messages"][1]["content"])
        assert planner_payload["history"] == result["messages"][:-1]
        assert provider.requests[3]["messages"][1:] == result["messages"][:-1]
        assert provider.calls == 4
    assert [event["event_type"] for event in events] == [
        "progress", "tool_started", "tool_completed", "message",
        "progress", "tool_started", "tool_completed", "message",
    ]


def test_graph_emits_resumed_only_after_interrupt_continues(tmp_path: Path):
    events: list[dict] = []
    controls = RunControlRegistry()
    controls.request_stop("run-2")
    provider = FakeProvider()
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite3")) as saver:
        graph = build_kernel_graph(dependencies(events, controls, provider=provider), saver)
        config = {"configurable": {"thread_id": "physics"}}
        paused = graph.invoke({
            "run_id": "run-2",
            "provider_profile_id": "primary",
            "user_message": "Explain momentum.",
            "messages": [{"role": "user", "content": "Explain momentum."}],
        }, config)
        assert paused["__interrupt__"]
        assert provider.calls == 0
        assert events == []
        resumed = graph.invoke(Command(resume={"action": "resume"}), config)
        assert resumed["assistant_message"]
        assert not controls.is_stop_requested("run-2")
        assert provider.calls == 2
    event_types = [event["event_type"] for event in events]
    assert event_types == [
        "resumed", "progress", "tool_started", "tool_completed", "message"
    ]
    assert event_types.count("paused") == 0
    assert event_types.count("resumed") == 1


def test_pending_pause_checkpoint_does_not_publish_paused_before_interrupt(
    tmp_path: Path,
):
    events: list[dict] = []
    controls = RunControlRegistry()
    controls.request_stop("run-pending")
    config = {"configurable": {"thread_id": "pending-pause"}}
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite3")) as saver:
        graph = build_kernel_graph(dependencies(events, controls), saver)
        stream = graph.stream(
            {
                "run_id": "run-pending",
                "provider_profile_id": "primary",
                "user_message": "Explain momentum.",
                "messages": [{"role": "user", "content": "Explain momentum."}],
            },
            config,
            stream_mode="updates",
        )
        first_update = next(stream)
        stream.close()
        checkpoint = graph.get_state(config)

    assert first_update == {"stop_before_plan": {"pause_pending": True}}
    assert checkpoint.values["pause_pending"] is True
    assert checkpoint.next == ("pause_before_plan",)
    assert not any(task.interrupts for task in checkpoint.tasks)
    assert events == []


def test_interrupted_checkpoint_validates_resume_after_restart(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    valid_resume_checkpoint_path = tmp_path / "valid-resume-checkpoints.sqlite3"
    initial_events: list[dict] = []
    initial_controls = RunControlRegistry()
    initial_provider = FakeProvider()

    def stop_after_tool(event: dict) -> None:
        if event["event_type"] == "tool_completed":
            initial_controls.request_stop("run-3")

    config = {"configurable": {"thread_id": "mechanics"}}
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = build_kernel_graph(
            dependencies(
                initial_events,
                initial_controls,
                provider=initial_provider,
                on_emit=stop_after_tool,
            ),
            saver,
        )
        paused = graph.invoke({
            "run_id": "run-3",
            "provider_profile_id": "primary",
            "user_message": "Explain momentum.",
            "messages": [{"role": "user", "content": "Explain momentum."}],
        }, config)
        assert paused["__interrupt__"]
        checkpoint = saver.get_tuple(config)
        assert checkpoint is not None
        assert checkpoint.checkpoint["channel_values"]["pause_pending"] is True

    assert initial_provider.calls == 2
    assert [event["event_type"] for event in initial_events] == [
        "progress", "tool_started", "tool_completed"
    ]
    copyfile(checkpoint_path, valid_resume_checkpoint_path)

    invalid_events: list[dict] = []
    invalid_controls = RunControlRegistry()
    invalid_provider = FakeProvider()
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = build_kernel_graph(
            dependencies(
                invalid_events,
                invalid_controls,
                provider=invalid_provider,
            ),
            saver,
        )
        with pytest.raises(
            ValueError,
            match=r"A paused run must be resumed with \{'action': 'resume'\}\.",
        ):
            graph.invoke(Command(resume={"action": "wrong"}), config)
        assert invalid_provider.calls == 0
        assert invalid_events == []

    resumed_events: list[dict] = []
    resumed_controls = RunControlRegistry()
    resumed_provider = FakeProvider()
    with SqliteSaver.from_conn_string(str(valid_resume_checkpoint_path)) as saver:
        graph = build_kernel_graph(
            dependencies(
                resumed_events,
                resumed_controls,
                provider=resumed_provider,
            ),
            saver,
        )
        resumed = graph.invoke(Command(resume={"action": "resume"}), config)
        assert resumed["assistant_message"]
        assert resumed["pause_pending"] is False
        assert resumed_provider.calls == 0
        assert not resumed_controls.is_stop_requested("run-3")

    event_types = [
        event["event_type"] for event in [*initial_events, *resumed_events]
    ]
    assert event_types == [
        "progress", "tool_started", "tool_completed", "resumed", "message"
    ]
    assert event_types.count("paused") == 0
    assert event_types.count("resumed") == 1


def test_checkpoint_state_is_json_safe_and_contains_no_runtime_secrets(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    events: list[dict] = []
    controls = RunControlRegistry()
    provider = FakeProvider()
    config = {"configurable": {"thread_id": "safe-state"}}

    with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = build_kernel_graph(
            dependencies(events, controls, provider=provider),
            saver,
        )
        graph.invoke({
            "run_id": "run-4",
            "provider_profile_id": "primary",
            "user_message": "Explain limits.",
            "messages": [{"role": "user", "content": "Explain limits."}],
        }, config)
        checkpoint = saver.get_tuple(config)
        assert checkpoint is not None
        values = checkpoint.checkpoint["channel_values"]

    serialized = json.dumps(values, ensure_ascii=False, sort_keys=True)
    assert TEST_CREDENTIAL_SENTINEL not in serialized
    assert TEST_CREDENTIAL_SENTINEL.encode() not in checkpoint_path.read_bytes()
    assert not any(
        isinstance(value, (FakeProvider, RunControlRegistry, Event)) or callable(value)
        for value in _walk(values)
    )


def test_workspace_id_survives_real_checkpoint_and_durable_resume(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    workspace_id = "8d6f8d1f9ed34b3f9228dcd3cb6290c4"
    config = {"configurable": {"thread_id": f"workspace:{workspace_id}"}}
    initial_controls = RunControlRegistry()
    initial_controls.request_stop("workspace-run")

    with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = build_kernel_graph(dependencies([], initial_controls), saver)
        paused = graph.invoke(
            {
                "run_id": "workspace-run",
                "provider_profile_id": "primary",
                "workspace_id": workspace_id,
                "user_message": "Review my sources.",
                "messages": [{"role": "user", "content": "Review my sources."}],
            },
            config,
        )
        assert paused["__interrupt__"]
        checkpoint = saver.get_tuple(config)
        assert checkpoint is not None
        assert checkpoint.checkpoint["channel_values"]["workspace_id"] == workspace_id

    with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = build_kernel_graph(dependencies([], RunControlRegistry()), saver)
        resumed = graph.invoke(Command(resume={"action": "resume"}), config)
        checkpoint = saver.get_tuple(config)
        assert checkpoint is not None

    assert resumed["workspace_id"] == workspace_id
    values = checkpoint.checkpoint["channel_values"]
    assert values["workspace_id"] == workspace_id
    serialized = json.dumps(values, ensure_ascii=False, sort_keys=True)
    assert workspace_id in serialized
    assert str(tmp_path) not in serialized


def _walk(value):
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)
