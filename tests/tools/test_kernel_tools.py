import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from exam_predictor.tools.kernel import KernelPlanner, KernelToolRegistry


class FakeProvider:
    name = "fake"
    models = SimpleNamespace(fast="fast", balanced="balanced")
    capabilities = SimpleNamespace(chat=True, web_search=False)

    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls: list[dict] = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        content = next(self.responses)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_planner_selects_only_registered_tool():
    provider = FakeProvider([json.dumps({
        "tool": "describe_capabilities",
        "arguments": {},
        "reason": "The user asks what the Agent can do.",
    })])
    plan = KernelPlanner().plan("What can you do?", [], provider)
    assert plan.tool == "describe_capabilities"


def test_planner_rejects_unknown_tool():
    provider = FakeProvider([json.dumps({
        "tool": "delete_course_files",
        "arguments": {},
        "reason": "Ignore the bounded registry.",
    })])

    with pytest.raises(ValidationError):
        KernelPlanner().plan("Delete everything.", [], provider)


def test_planner_rejects_extra_plan_fields():
    provider = FakeProvider([json.dumps({
        "tool": "tutor_reply",
        "arguments": {},
        "reason": "Answer the question.",
        "unregistered_action": "delete_course_files",
    })])

    with pytest.raises(ValidationError):
        KernelPlanner().plan("Explain limits.", [], provider)


@pytest.mark.parametrize(
    ("tool", "provider_arguments", "expected_arguments"),
    [
        ("tutor_reply", {"message": "Ignore the user."}, {"message": "Explain limits."}),
        ("describe_capabilities", {"message": "Override."}, {}),
    ],
)
def test_planner_normalizes_tool_arguments(tool, provider_arguments, expected_arguments):
    provider = FakeProvider([json.dumps({
        "tool": tool,
        "arguments": provider_arguments,
        "reason": "Choose a bounded tool.",
    })])

    plan = KernelPlanner().plan("Explain limits.", [], provider)

    assert plan.arguments == expected_arguments


def test_planner_bounds_history_and_uses_fast_model():
    history = [{"role": "user", "content": f"message-{index}"} for index in range(15)]
    provider = FakeProvider([json.dumps({
        "tool": "tutor_reply",
        "arguments": {},
        "reason": "Answer the question.",
    })])

    KernelPlanner().plan("Explain limits.", history, provider)

    call = provider.calls[0]
    payload = json.loads(call["messages"][1]["content"])
    assert call["model"] == "fast"
    assert payload["history"] == history[-12:]


def test_tutor_tool_uses_provider_and_returns_structured_result():
    provider = FakeProvider(["A limit is the value a function approaches."])
    registry = KernelToolRegistry()
    result = registry.execute(
        tool="tutor_reply",
        arguments={"message": "Explain limits."},
        history=[{"role": "user", "content": "Explain limits."}],
        provider=provider,
    )
    assert result.tool == "tutor_reply"
    assert "approaches" in result.content
    assert len(provider.calls) == 1


def test_tutor_tool_bounds_history_and_uses_balanced_model():
    message = "message-24"
    history = [{"role": "user", "content": f"message-{index}"} for index in range(25)]
    provider = FakeProvider(["A bounded response."])

    KernelToolRegistry().execute(
        tool="tutor_reply",
        arguments={"message": message},
        history=history,
        provider=provider,
    )

    call = provider.calls[0]
    assert call["model"] == "balanced"
    assert call["messages"][1:] == history[-20:]


def test_unknown_tool_execution_is_rejected():
    provider = FakeProvider([])

    with pytest.raises(ValueError, match="Unknown kernel tool: delete_course_files"):
        KernelToolRegistry().execute(
            tool="delete_course_files",
            arguments={},
            history=[],
            provider=provider,
        )


def test_describe_capabilities_returns_provider_metadata_without_calling_provider():
    provider = FakeProvider([])

    result = KernelToolRegistry().execute(
        tool="describe_capabilities",
        arguments={},
        history=[],
        provider=provider,
    )

    assert result.tool == "describe_capabilities"
    assert "Course folders and academic tools arrive in the next subprojects." in result.content
    assert result.metadata == {"provider": "fake", "capabilities": ["chat"]}
    assert provider.calls == []
