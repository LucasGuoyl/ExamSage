from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from exam_predictor.evidence.scheduler import SchedulerOutcome, SchedulerStatus
from exam_predictor.evidence.service import (
    EvidenceFrontierResult,
    EvidenceInspection,
    EvidenceRunResult,
)
from exam_predictor.graphs.evidence import (
    EvidenceGraphDependencies,
    build_evidence_graph,
)
from exam_predictor.runtime.control import RunControlRegistry
from exam_predictor.tools.evidence import EvidenceToolRegistry
from exam_predictor.workspace.transmission import SourceAuthorizationError


WORKSPACE_ID = "8d6f8d1f9ed34b3f9228dcd3cb6290c4"
REVISION_ID = "revision_graph_00000000000000001"


class StagedEvidenceService:
    def __init__(
        self,
        controls: RunControlRegistry,
        *,
        stop_after_first_frontier: bool = False,
    ) -> None:
        self.controls = controls
        self.stop_after_first_frontier = stop_after_first_frontier
        self.calls: list[tuple] = []
        self.frontier_count = 0

    def inspect(self, workspace_id: str) -> EvidenceInspection:
        self.calls.append(("inspect", workspace_id))
        return self._inspection(workspace_id)

    def prepare_analysis(self, workspace_id: str) -> EvidenceInspection:
        self.calls.append(("prepare", workspace_id))
        return self._inspection(workspace_id)

    def analyze_frontier(
        self,
        workspace_id: str,
        revision_id: str,
        run_id: str,
    ) -> EvidenceFrontierResult:
        self.frontier_count += 1
        pending = 1 if self.frontier_count == 1 else 0
        outcome = SchedulerOutcome(
            status=SchedulerStatus.COMPLETE,
            processed_part_ids=(f"part-{self.frontier_count}",),
            pending_count=pending,
        )
        self.calls.append(("analyze", workspace_id, revision_id, run_id, pending))
        if self.stop_after_first_frontier and self.frontier_count == 1:
            self.controls.request_stop(run_id)
        return EvidenceFrontierResult(
            workspace_id=workspace_id,
            revision_id=revision_id,
            outcome=outcome,
        )

    def publish_frontier(
        self,
        workspace_id: str,
        revision_id: str,
        outcome: SchedulerOutcome,
        *,
        response_language: str | None = None,
    ) -> EvidenceRunResult:
        self.calls.append(
            (
                "publish",
                workspace_id,
                revision_id,
                outcome.pending_count,
                response_language,
            )
        )
        return EvidenceRunResult(
            workspace_id=workspace_id,
            revision_id=revision_id,
            status="complete" if outcome.pending_count == 0 else "paused",
            outcome=outcome,
        )

    @staticmethod
    def _inspection(workspace_id: str) -> EvidenceInspection:
        return EvidenceInspection(
            workspace_id=workspace_id,
            revision_id=REVISION_ID,
            approval_id="approval-graph",
            approval_required=False,
            approved_source_count=2,
            approved_bytes=200,
        )


def _dependencies(service: StagedEvidenceService, events: list[dict]):
    def emit(run_id, event_type, stage, message, payload=None):
        events.append(
            {
                "run_id": run_id,
                "event_type": event_type,
                "stage": stage,
                "message": message,
                "payload": payload or {},
            }
        )

    return EvidenceGraphDependencies(
        tools=EvidenceToolRegistry(service),
        controls=service.controls,
        emit=emit,
    )


def _initial_state(run_id: str = "run-evidence") -> dict:
    return {
        "run_id": run_id,
        "workspace_id": WORKSPACE_ID,
        "selected_tool": "build_study_map",
        "user_message": "Build my study map.",
        "tool_arguments": {
            "workspace_id": WORKSPACE_ID,
            "intent": "Build my study map.",
            "response_language": "en",
        },
    }


def test_evidence_graph_runs_one_frontier_at_a_time_and_publishes_to_completion(
    tmp_path: Path,
):
    controls = RunControlRegistry()
    service = StagedEvidenceService(controls)
    events: list[dict] = []
    checkpoint_path = tmp_path / "evidence-checkpoints.sqlite3"
    config = {"configurable": {"thread_id": f"workspace:{WORKSPACE_ID}"}}

    with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = build_evidence_graph(_dependencies(service, events), saver)
        result = graph.invoke(_initial_state(), config)
        checkpoint = saver.get_tuple(config)
        assert checkpoint is not None

    assert result["tool_result"]["tool"] == "build_study_map"
    assert result["tool_result"]["metadata"]["status"] == "complete"
    assert [call[0] for call in service.calls] == [
        "prepare",
        "analyze",
        "publish",
        "analyze",
        "publish",
    ]
    assert [call[-1] for call in service.calls if call[0] == "publish"] == [
        "en",
        "en",
    ]
    values = checkpoint.checkpoint["channel_values"]
    assert values["workspace_id"] == WORKSPACE_ID
    assert values["revision_id"] == REVISION_ID
    assert values["run_id"] == "run-evidence"
    serialized = json.dumps(values, ensure_ascii=False, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert [event["event_type"] for event in events] == [
        "tool_started",
        "progress",
        "progress",
        "tool_completed",
    ]


def test_stop_after_validated_frontier_interrupts_before_snapshot_publication(
    tmp_path: Path,
):
    controls = RunControlRegistry()
    service = StagedEvidenceService(controls, stop_after_first_frontier=True)
    events: list[dict] = []
    config = {"configurable": {"thread_id": f"workspace:{WORKSPACE_ID}"}}

    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite3")) as saver:
        graph = build_evidence_graph(_dependencies(service, events), saver)
        paused = graph.invoke(_initial_state("run-stop-after-frontier"), config)
        checkpoint = saver.get_tuple(config)
        assert checkpoint is not None
        assert paused["__interrupt__"]
        assert [call[0] for call in service.calls] == ["prepare", "analyze"]
        values = checkpoint.checkpoint["channel_values"]
        assert values["frontier"]["outcome"]["pending_count"] == 1
        assert values["revision_id"] == REVISION_ID

        resumed = graph.invoke(Command(resume={"action": "resume"}), config)

    assert resumed["tool_result"]["metadata"]["status"] == "complete"
    assert [call[0] for call in service.calls] == [
        "prepare",
        "analyze",
        "publish",
        "analyze",
        "publish",
    ]
    assert not controls.is_stop_requested("run-stop-after-frontier")
    assert [event["event_type"] for event in events].count("resumed") == 1


def test_stop_before_authorization_checkpoints_without_preparing_sources(
    tmp_path: Path,
):
    controls = RunControlRegistry()
    controls.request_stop("run-stop-before-auth")
    service = StagedEvidenceService(controls)
    events: list[dict] = []
    config = {"configurable": {"thread_id": f"workspace:{WORKSPACE_ID}"}}

    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite3")) as saver:
        graph = build_evidence_graph(_dependencies(service, events), saver)
        paused = graph.invoke(_initial_state("run-stop-before-auth"), config)
        assert paused["__interrupt__"]
        assert service.calls == []
        resumed = graph.invoke(Command(resume={"action": "resume"}), config)

    assert resumed["tool_result"]["metadata"]["status"] == "complete"
    assert service.calls[0][0] == "prepare"


def test_empty_approved_source_selection_terminates_before_any_frontier(
    tmp_path: Path,
):
    controls = RunControlRegistry()

    class EmptyService(StagedEvidenceService):
        def prepare_analysis(self, workspace_id: str) -> EvidenceInspection:
            self.calls.append(("prepare", workspace_id))
            return EvidenceInspection(
                workspace_id=workspace_id,
                revision_id=REVISION_ID,
                approval_id="approval-empty",
                approval_required=False,
                approved_source_count=0,
                approved_bytes=0,
            )

    service = EmptyService(controls)
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite3")) as saver:
        graph = build_evidence_graph(_dependencies(service, []), saver)
        with pytest.raises(ValueError, match="Select and approve"):
            graph.invoke(
                _initial_state("run-empty-sources"),
                {"configurable": {"thread_id": f"workspace:{WORKSPACE_ID}"}},
            )

    assert [call[0] for call in service.calls] == ["prepare"]


def test_changed_source_after_analysis_resumes_from_current_revision(
    tmp_path: Path,
):
    controls = RunControlRegistry()
    replacement_revision = "revision_graph_replacement_000000001"

    class ChangingService(StagedEvidenceService):
        def __init__(self, registry: RunControlRegistry) -> None:
            super().__init__(registry)
            self.current_revision = REVISION_ID
            self.change_once = True

        def prepare_analysis(self, workspace_id: str) -> EvidenceInspection:
            self.calls.append(("prepare", workspace_id, self.current_revision))
            return EvidenceInspection(
                workspace_id=workspace_id,
                revision_id=self.current_revision,
                approval_id="approval-current",
                approval_required=False,
                approved_source_count=1,
                approved_bytes=10,
            )

        def analyze_frontier(
            self,
            workspace_id: str,
            revision_id: str,
            run_id: str,
        ) -> EvidenceFrontierResult:
            self.calls.append(("analyze", workspace_id, revision_id, run_id))
            return EvidenceFrontierResult(
                workspace_id=workspace_id,
                revision_id=revision_id,
                outcome=SchedulerOutcome(
                    status=SchedulerStatus.COMPLETE,
                    processed_part_ids=(f"part-{revision_id}",),
                    pending_count=0,
                ),
            )

        def publish_frontier(
            self,
            workspace_id: str,
            revision_id: str,
            outcome: SchedulerOutcome,
            *,
            response_language: str | None = None,
        ) -> EvidenceRunResult:
            self.calls.append(("publish", workspace_id, revision_id))
            if self.change_once:
                self.change_once = False
                self.current_revision = replacement_revision
                raise SourceAuthorizationError(
                    "source_approval_revoked",
                    workspace_id,
                    "entry-changed",
                )
            return EvidenceRunResult(
                workspace_id=workspace_id,
                revision_id=revision_id,
                status="complete",
                outcome=outcome,
            )

    service = ChangingService(controls)
    config = {"configurable": {"thread_id": f"workspace:{WORKSPACE_ID}"}}
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite3")) as saver:
        graph = build_evidence_graph(_dependencies(service, []), saver)
        paused = graph.invoke(_initial_state("run-source-changed"), config)
        assert paused["__interrupt__"][0].value == {
            "kind": "source_changed",
            "run_id": "run-source-changed",
            "code": "source_approval_revoked",
            "entry_id": "entry-changed",
        }

        resumed = graph.invoke(Command(resume={"action": "resume"}), config)
        checkpoint = saver.get_tuple(config)
        assert checkpoint is not None

    assert resumed["tool_result"]["metadata"]["status"] == "complete"
    assert service.calls == [
        ("prepare", WORKSPACE_ID, REVISION_ID),
        ("analyze", WORKSPACE_ID, REVISION_ID, "run-source-changed"),
        ("publish", WORKSPACE_ID, REVISION_ID),
        ("prepare", WORKSPACE_ID, replacement_revision),
        ("analyze", WORKSPACE_ID, replacement_revision, "run-source-changed"),
        ("publish", WORKSPACE_ID, replacement_revision),
    ]
    assert checkpoint.checkpoint["channel_values"]["revision_id"] == (
        replacement_revision
    )
