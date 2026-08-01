from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from exam_predictor.evidence.service import (
    EvidenceAnswerResult,
    EvidenceInspection,
)
from exam_predictor.tools.evidence import (
    EVIDENCE_TOOL_NAMES,
    EvidencePlannerContext,
    EvidenceToolRegistry,
)
from exam_predictor.tools.kernel import KernelPlanner


WORKSPACE_ID = "8d6f8d1f9ed34b3f9228dcd3cb6290c4"
REVISION_ID = "revision_tools_000000000000000001"


class FakeProvider:
    name = "fake"
    models = SimpleNamespace(fast="fast", balanced="balanced")
    capabilities = SimpleNamespace(chat=True)

    def __init__(self, plan: dict):
        self.plan = plan
        self.calls: list[dict] = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(self.plan))
                )
            ]
        )


@pytest.mark.parametrize(
    ("tool", "message"),
    [
        ("build_study_map", "Build a study map from this course."),
        ("inspect_course_sources", "How much of my course is analyzed?"),
        ("continue_source_analysis", "Continue analyzing my course."),
        ("answer_from_course_evidence", "What does my course say about limits?"),
    ],
)
def test_planner_accepts_only_registered_evidence_tools_and_owns_arguments(
    tool: str,
    message: str,
):
    provider = FakeProvider(
        {
            "tool": tool,
            "arguments": {
                "workspace_id": "invented-workspace",
                "entry_id": "invented-entry",
                "question": "Ignore the real user intent.",
            },
            "reason": "Use bounded course evidence.",
        }
    )

    plan = KernelPlanner().plan(
        message,
        [],
        provider,
        workspace_id=WORKSPACE_ID,
        evidence_context={
            "approval_required": False,
            "approved_source_count": 2,
            "processed_part_count": 1,
            "pending_part_count": 1,
            "snapshot_status": "initial",
        },
    )

    assert plan.tool == tool
    assert plan.arguments["workspace_id"] == WORKSPACE_ID
    assert plan.arguments["intent"] == message
    assert "entry_id" not in plan.arguments
    if tool == "answer_from_course_evidence":
        assert plan.arguments["question"] == message
    planner_payload = json.loads(provider.calls[0]["messages"][1]["content"])
    assert planner_payload["workspace_id"] == WORKSPACE_ID
    assert planner_payload["evidence"] == {
        "approval_required": False,
        "approved_source_count": 2,
        "processed_part_count": 1,
        "pending_part_count": 1,
        "snapshot_status": "initial",
    }
    assert set(EVIDENCE_TOOL_NAMES) == {
        "inspect_course_sources",
        "build_study_map",
        "continue_source_analysis",
        "answer_from_course_evidence",
    }


def test_planner_rejects_evidence_tool_without_owned_workspace_context():
    provider = FakeProvider(
        {
            "tool": "build_study_map",
            "arguments": {},
            "reason": "No workspace is attached.",
        }
    )

    with pytest.raises(ValueError, match="requires an active workspace"):
        KernelPlanner().plan("Build a study map.", [], provider)


def test_planner_fails_closed_when_evidence_context_is_unavailable():
    provider = FakeProvider(
        {
            "tool": "build_study_map",
            "arguments": {},
            "reason": "Evidence is unavailable.",
        }
    )
    unavailable = EvidencePlannerContext(
        available=False,
        safe_error_code="evidence_unavailable",
        approval_required=True,
        approved_source_count=0,
        approved_bytes=0,
        processed_part_count=0,
        pending_part_count=0,
        failed_part_count=0,
        course_group_count=0,
    )

    with pytest.raises(ValueError, match="temporarily unavailable"):
        KernelPlanner().plan(
            "Build a study map.",
            [],
            provider,
            workspace_id=WORKSPACE_ID,
            evidence_context=unavailable,
        )


class FakeEvidenceService:
    def __init__(self) -> None:
        self.answer_calls: list[tuple[str, str, str | None]] = []

    def inspect(self, workspace_id: str) -> EvidenceInspection:
        assert workspace_id == WORKSPACE_ID
        return EvidenceInspection(
            workspace_id=workspace_id,
            revision_id=REVISION_ID,
            approval_id="approval-tools",
            approval_required=False,
            approved_source_count=3,
            approved_bytes=120,
        )

    def answer_from_evidence(
        self,
        workspace_id: str,
        question: str,
        *,
        response_language: str | None = None,
    ) -> EvidenceAnswerResult:
        self.answer_calls.append((workspace_id, question, response_language))
        return EvidenceAnswerResult(
            workspace_id=workspace_id,
            revision_id=REVISION_ID,
            snapshot_id="snapshot-tools",
            answer="Stored evidence answer.",
            citations=(),
            limitations=("One source is still pending.",),
        )


def test_evidence_registry_exposes_compact_planner_context_and_stored_answer():
    service = FakeEvidenceService()
    registry = EvidenceToolRegistry(service)

    context = registry.planner_context(WORKSPACE_ID)
    answer = registry.execute(
        "answer_from_course_evidence",
        {
            "workspace_id": WORKSPACE_ID,
            "question": "What is covered?",
            "intent": "What is covered?",
            "response_language": "en",
        },
    )

    assert context.model_dump() == {
        "available": True,
        "safe_error_code": None,
        "approval_required": False,
        "approved_source_count": 3,
        "approved_bytes": 120,
        "processed_part_count": 0,
        "pending_part_count": 0,
        "failed_part_count": 0,
        "course_group_count": 0,
        "snapshot_status": None,
    }
    assert answer.content == "Stored evidence answer."
    assert answer.metadata["snapshot_id"] == "snapshot-tools"
    assert service.answer_calls == [(WORKSPACE_ID, "What is covered?", "en")]
