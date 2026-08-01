from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from exam_predictor.evidence.scheduler import SchedulerOutcome
from exam_predictor.evidence.service import (
    EvidenceAnswerResult,
    EvidenceFrontierResult,
    EvidenceInspection,
    EvidenceRunResult,
)


EvidenceToolName = Literal[
    "inspect_course_sources",
    "build_study_map",
    "continue_source_analysis",
    "answer_from_course_evidence",
]
EVIDENCE_TOOL_NAMES: tuple[EvidenceToolName, ...] = (
    "inspect_course_sources",
    "build_study_map",
    "continue_source_analysis",
    "answer_from_course_evidence",
)


class EvidenceServiceProtocol(Protocol):
    def inspect(self, workspace_id: str) -> EvidenceInspection: ...

    def prepare_analysis(
        self,
        workspace_id: str,
        run_id: str | None = None,
    ) -> EvidenceInspection: ...

    def analyze_frontier(
        self,
        workspace_id: str,
        revision_id: str,
        run_id: str,
    ) -> EvidenceFrontierResult: ...

    def publish_frontier(
        self,
        workspace_id: str,
        revision_id: str,
        outcome: SchedulerOutcome,
        *,
        run_id: str,
        response_language: str | None = None,
    ) -> EvidenceRunResult: ...

    def answer_from_evidence(
        self,
        workspace_id: str,
        question: str,
        *,
        response_language: str | None = None,
    ) -> EvidenceAnswerResult: ...


class EvidencePlannerContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    available: bool = True
    safe_error_code: str | None = None
    approval_required: bool
    approved_source_count: int = Field(ge=0)
    approved_bytes: int = Field(ge=0)
    processed_part_count: int = Field(ge=0)
    pending_part_count: int = Field(ge=0)
    failed_part_count: int = Field(ge=0)
    course_group_count: int = Field(ge=0)
    snapshot_status: str | None = None


class EvidenceToolArguments(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str = Field(min_length=1, max_length=128)
    intent: str = Field(min_length=1, max_length=4_000)
    question: str | None = Field(default=None, max_length=4_000)
    response_language: str | None = Field(default=None, max_length=64)


class EvidenceToolOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: EvidenceToolName
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidencePublicationState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    revision_id: str
    status: Literal["complete", "paused"]
    outcome: SchedulerOutcome
    safe_error_code: str | None = None
    snapshot_id: str | None = None
    snapshot_status: str | None = None


class EvidenceToolRegistry:
    """Typed boundary between the Agent graph and the evidence service."""

    def __init__(self, service: EvidenceServiceProtocol) -> None:
        self.service = service

    def planner_context(self, workspace_id: str) -> EvidencePlannerContext:
        return self.context_from_inspection(self.service.inspect(workspace_id))

    @staticmethod
    def context_from_inspection(
        inspection: EvidenceInspection,
    ) -> EvidencePlannerContext:
        coverage = inspection.coverage
        return EvidencePlannerContext(
            approval_required=inspection.approval_required,
            approved_source_count=inspection.approved_source_count,
            approved_bytes=inspection.approved_bytes,
            processed_part_count=(
                0 if coverage is None else coverage.part_processed_count
            ),
            pending_part_count=(0 if coverage is None else coverage.part_pending_count),
            failed_part_count=(0 if coverage is None else coverage.part_failed_count),
            course_group_count=(
                0
                if inspection.snapshot is None
                else len(inspection.snapshot.course_groups)
            ),
            snapshot_status=(
                None
                if inspection.snapshot is None
                else inspection.snapshot.status.value
            ),
        )

    def execute(
        self,
        tool: EvidenceToolName,
        arguments: dict[str, Any],
    ) -> EvidenceToolOutput:
        values = EvidenceToolArguments.model_validate(arguments)
        if tool == "inspect_course_sources":
            return self._inspection_output(tool, self.service.inspect(values.workspace_id))
        if tool == "answer_from_course_evidence":
            question = (values.question or values.intent).strip()
            answer = self.service.answer_from_evidence(
                values.workspace_id,
                question,
                response_language=values.response_language,
            )
            return EvidenceToolOutput(
                tool=tool,
                content=answer.answer,
                metadata={
                    "workspace_id": answer.workspace_id,
                    "revision_id": answer.revision_id,
                    "snapshot_id": answer.snapshot_id,
                    "citations": [
                        citation.model_dump(mode="json")
                        for citation in answer.citations
                    ],
                    "limitations": list(answer.limitations),
                },
            )
        raise ValueError(f"Evidence tool '{tool}' requires the resumable graph.")

    def prepare_analysis(
        self,
        workspace_id: str,
        run_id: str | None = None,
    ) -> EvidenceInspection:
        return self.service.prepare_analysis(workspace_id, run_id)

    def analyze_frontier(
        self,
        workspace_id: str,
        revision_id: str,
        run_id: str,
    ) -> EvidenceFrontierResult:
        return self.service.analyze_frontier(workspace_id, revision_id, run_id)

    def publish_frontier(
        self,
        workspace_id: str,
        revision_id: str,
        outcome: SchedulerOutcome,
        *,
        run_id: str,
        response_language: str | None = None,
    ) -> EvidenceRunResult:
        return self.service.publish_frontier(
            workspace_id,
            revision_id,
            outcome,
            run_id=run_id,
            response_language=response_language,
        )

    def run_output(
        self,
        tool: EvidenceToolName,
        result: EvidencePublicationState,
    ) -> EvidenceToolOutput:
        if result.status == "complete":
            content = "The cited study map is complete."
        elif result.safe_error_code == "provider_timeout":
            content = (
                "The evidence deadline was reached after saved progress. "
                "Start a new analysis run to continue with a fresh deadline."
            )
        else:
            content = "Course evidence was saved and analysis is paused."
        return EvidenceToolOutput(
            tool=tool,
            content=content,
            metadata={
                "workspace_id": result.workspace_id,
                "revision_id": result.revision_id,
                "status": result.status,
                "safe_error_code": result.safe_error_code,
                "processed_part_ids": list(result.outcome.processed_part_ids),
                "failed_part_ids": list(result.outcome.failed_part_ids),
                "pending_part_count": result.outcome.pending_count,
                "snapshot_id": result.snapshot_id,
                "snapshot_status": result.snapshot_status,
            },
        )

    @staticmethod
    def publication_state(result: EvidenceRunResult) -> EvidencePublicationState:
        snapshot = result.snapshot
        return EvidencePublicationState(
            workspace_id=result.workspace_id,
            revision_id=result.revision_id,
            status=result.status,
            outcome=result.outcome,
            safe_error_code=result.safe_error_code,
            snapshot_id=None if snapshot is None else snapshot.snapshot_id,
            snapshot_status=None if snapshot is None else snapshot.status.value,
        )

    @staticmethod
    def _inspection_output(
        tool: EvidenceToolName,
        inspection: EvidenceInspection,
    ) -> EvidenceToolOutput:
        if inspection.approval_required:
            content = "Course source approval is required before analysis."
        else:
            content = (
                f"{inspection.approved_source_count} approved course sources are ready."
            )
        return EvidenceToolOutput(
            tool=tool,
            content=content,
            metadata={
                "workspace_id": inspection.workspace_id,
                "revision_id": inspection.revision_id,
                "approval_id": inspection.approval_id,
                **EvidenceToolRegistry.context_from_inspection(
                    inspection
                ).model_dump(mode="json"),
            },
        )


def is_evidence_tool(tool: str) -> bool:
    return tool in EVIDENCE_TOOL_NAMES
