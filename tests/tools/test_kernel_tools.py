import json
from types import SimpleNamespace

from exam_predictor.tools.kernel import KernelPlanner, KernelToolRegistry


class FakeProvider:
    name = "fake"
    models = SimpleNamespace(fast="fast", balanced="balanced")

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
