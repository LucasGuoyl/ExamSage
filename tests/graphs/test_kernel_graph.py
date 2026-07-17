import json
from pathlib import Path
from types import SimpleNamespace

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from exam_predictor.graphs.kernel import KernelDependencies, build_kernel_graph
from exam_predictor.runtime.control import RunControlRegistry
from exam_predictor.tools.kernel import KernelPlanner, KernelToolRegistry


class FakeProvider:
    name = "fake"
    capabilities = SimpleNamespace(chat=True)
    models = SimpleNamespace(fast="fast", balanced="balanced")

    def __init__(self):
        self.calls = 0

    def create_chat_completion(self, **kwargs):
        self.calls += 1
        content = (
            json.dumps({"tool": "tutor_reply", "arguments": {}, "reason": "Tutor request"})
            if self.calls % 2 == 1
            else "A limit is the value approached by a function."
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class Sessions:
    def __init__(self):
        self.provider = FakeProvider()

    def get_provider(self, profile_id: str):
        assert profile_id == "primary"
        return self.provider


def dependencies(events: list[dict], controls: RunControlRegistry):
    return KernelDependencies(
        provider_sessions=Sessions(),
        planner=KernelPlanner(),
        tools=KernelToolRegistry(),
        controls=controls,
        emit=lambda run_id, event_type, stage, message, payload=None: events.append({
            "run_id": run_id,
            "event_type": event_type,
            "stage": stage,
            "message": message,
            "payload": payload or {},
        }),
    )


def test_graph_runs_tool_and_persists_follow_up_state(tmp_path: Path):
    events: list[dict] = []
    controls = RunControlRegistry()
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite3")) as saver:
        graph = build_kernel_graph(dependencies(events, controls), saver)
        config = {"configurable": {"thread_id": "calculus"}}
        result = graph.invoke({
            "run_id": "run-1",
            "provider_profile_id": "primary",
            "user_message": "Explain limits.",
            "messages": [{"role": "user", "content": "Explain limits."}],
        }, config)
        assert "approached" in result["assistant_message"]
        assert result["selected_tool"] == "tutor_reply"
        assert graph.get_state(config).values["messages"][-1]["role"] == "assistant"
    assert [event["event_type"] for event in events] == [
        "progress", "tool_started", "tool_completed", "message"
    ]


def test_stop_interrupt_requires_explicit_resume(tmp_path: Path):
    events: list[dict] = []
    controls = RunControlRegistry()
    controls.request_stop("run-2")
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite3")) as saver:
        graph = build_kernel_graph(dependencies(events, controls), saver)
        config = {"configurable": {"thread_id": "physics"}}
        paused = graph.invoke({
            "run_id": "run-2",
            "provider_profile_id": "primary",
            "user_message": "Explain momentum.",
            "messages": [{"role": "user", "content": "Explain momentum."}],
        }, config)
        assert paused["__interrupt__"]
        resumed = graph.invoke(Command(resume={"action": "resume"}), config)
        assert resumed["assistant_message"]
        assert not controls.is_stop_requested("run-2")
