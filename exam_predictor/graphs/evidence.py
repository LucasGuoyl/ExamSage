from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from exam_predictor.evidence.scheduler import SchedulerStatus
from exam_predictor.evidence.service import EvidenceFrontierResult
from exam_predictor.runtime.control import RunControlRegistry
from exam_predictor.tools.evidence import (
    EvidenceToolArguments,
    EvidencePublicationState,
    EvidenceToolRegistry,
    is_evidence_tool,
)
from exam_predictor.workspace.transmission import SourceAuthorizationError


class EvidenceGraphState(TypedDict, total=False):
    run_id: str
    workspace_id: str
    selected_tool: str
    user_message: str
    tool_arguments: dict[str, Any]
    revision_id: str
    evidence_inspection: dict[str, Any]
    frontier: dict[str, Any]
    evidence_result: dict[str, Any]
    tool_result: dict[str, Any]
    pause_pending: bool
    resumed_from_stop: bool
    authority_error: dict[str, str] | None


EventEmitter = Callable[[str, str, str, str, dict[str, Any] | None], None]


@dataclass(frozen=True)
class EvidenceGraphDependencies:
    tools: EvidenceToolRegistry
    controls: RunControlRegistry
    emit: EventEmitter


def build_evidence_graph(
    deps: EvidenceGraphDependencies,
    checkpointer=None,
):
    def start_tool(state: EvidenceGraphState) -> dict[str, Any]:
        tool = state["selected_tool"]
        if not is_evidence_tool(tool):
            raise ValueError(f"Unknown evidence tool: {tool}")
        arguments = EvidenceToolArguments.model_validate(
            state.get("tool_arguments", {})
        )
        if arguments.workspace_id != state.get("workspace_id"):
            raise ValueError("Evidence tool workspace does not match the run workspace.")
        deps.emit(
            state["run_id"],
            "tool_started",
            "evidence",
            f"Running {tool}.",
            {"tool": tool},
        )
        return {"pause_pending": False, "resumed_from_stop": False}

    def route_kind(state: EvidenceGraphState) -> str:
        if state["selected_tool"] == "inspect_course_sources":
            return "inspect"
        if state["selected_tool"] == "answer_from_course_evidence":
            return "answer"
        return "staged"

    def detect_stop(state: EvidenceGraphState) -> dict[str, Any]:
        if state.get("pause_pending") or not deps.controls.is_stop_requested(
            state["run_id"]
        ):
            return {}
        return {"pause_pending": True}

    def route_after_stop(state: EvidenceGraphState) -> str:
        return "pause" if state.get("pause_pending") else "continue"

    def pause_at(stage: str):
        def pause(state: EvidenceGraphState) -> dict[str, Any]:
            if not state.get("pause_pending"):
                return {}
            run_id = state["run_id"]
            resume = interrupt(
                {"kind": "stopped", "run_id": run_id, "stage": stage}
            )
            if resume != {"action": "resume"}:
                raise ValueError(
                    "A paused evidence run must be resumed with {'action': 'resume'}."
                )
            deps.controls.clear_stop(run_id)
            deps.emit(
                run_id,
                "resumed",
                "evidence",
                "Course evidence analysis resumed from its checkpoint.",
                {"stage": stage},
            )
            return {"pause_pending": False, "resumed_from_stop": True}

        return pause

    def run_immediate(state: EvidenceGraphState) -> dict[str, Any]:
        output = deps.tools.execute(
            state["selected_tool"],  # type: ignore[arg-type]
            state["tool_arguments"],
        )
        deps.emit(
            state["run_id"],
            "tool_completed",
            "evidence",
            f"Completed {state['selected_tool']}.",
            {"tool": state["selected_tool"]},
        )
        return {"tool_result": output.model_dump(mode="json")}

    def prepare(state: EvidenceGraphState) -> dict[str, Any]:
        try:
            inspection = deps.tools.prepare_analysis(state["workspace_id"])
        except SourceAuthorizationError as error:
            return {"authority_error": _authority_error(error)}
        if inspection.approval_required or inspection.revision_id is None:
            raise ValueError("Course source approval is required before analysis.")
        if inspection.approved_source_count == 0:
            raise ValueError("Select and approve at least one course source.")
        return {
            "authority_error": None,
            "revision_id": inspection.revision_id,
            "evidence_inspection": deps.tools.context_from_inspection(
                inspection
            ).model_dump(mode="json"),
        }

    def analyze(state: EvidenceGraphState) -> dict[str, Any]:
        deps.emit(
            state["run_id"],
            "progress",
            "evidence",
            "Analyzing the next bounded course-evidence frontier.",
            {"revision_id": state["revision_id"]},
        )
        try:
            frontier = deps.tools.analyze_frontier(
                state["workspace_id"],
                state["revision_id"],
                state["run_id"],
            )
        except SourceAuthorizationError as error:
            return {"authority_error": _authority_error(error)}
        return {
            "authority_error": None,
            "frontier": frontier.model_dump(mode="json"),
            "resumed_from_stop": False,
        }

    def publish(state: EvidenceGraphState) -> dict[str, Any]:
        frontier = EvidenceFrontierResult.model_validate(state["frontier"])
        arguments = EvidenceToolArguments.model_validate(state["tool_arguments"])
        try:
            result = deps.tools.publish_frontier(
                state["workspace_id"],
                state["revision_id"],
                frontier.outcome,
                response_language=arguments.response_language,
            )
        except SourceAuthorizationError as error:
            return {"authority_error": _authority_error(error)}
        return {
            "authority_error": None,
            "evidence_result": deps.tools.publication_state(result).model_dump(
                mode="json"
            )
        }

    def route_after_publication(state: EvidenceGraphState) -> str:
        if state.get("authority_error") is not None:
            return "source_changed"
        frontier = EvidenceFrontierResult.model_validate(state["frontier"])
        result = EvidencePublicationState.model_validate(state["evidence_result"])
        if result.status == "complete":
            return "complete"
        if state.get("resumed_from_stop") and not frontier.outcome.failed_part_ids:
            return "continue"
        if frontier.outcome.status is SchedulerStatus.PAUSED:
            return "paused"
        return "continue"

    def route_after_authority_step(state: EvidenceGraphState) -> str:
        return (
            "source_changed"
            if state.get("authority_error") is not None
            else "continue"
        )

    def pause_source_changed(state: EvidenceGraphState) -> dict[str, Any]:
        run_id = state["run_id"]
        error = state.get("authority_error") or {
            "code": "source_approval_revoked",
            "entry_id": "",
        }
        resume = interrupt(
            {
                "kind": "source_changed",
                "run_id": run_id,
                "code": error["code"],
                "entry_id": error["entry_id"],
            }
        )
        if resume != {"action": "resume"}:
            raise ValueError(
                "A paused evidence run must be resumed with {'action': 'resume'}."
            )
        deps.controls.clear_stop(run_id)
        deps.emit(
            run_id,
            "resumed",
            "evidence",
            "Course evidence analysis resumed with current approval.",
            {"stage": "source_changed"},
        )
        return {
            "authority_error": None,
            "pause_pending": False,
            "resumed_from_stop": False,
        }

    def pause_evidence(state: EvidenceGraphState) -> dict[str, Any]:
        run_id = state["run_id"]
        resume = interrupt(
            {
                "kind": "evidence_paused",
                "run_id": run_id,
                "revision_id": state["revision_id"],
            }
        )
        if resume != {"action": "resume"}:
            raise ValueError(
                "A paused evidence run must be resumed with {'action': 'resume'}."
            )
        deps.controls.clear_stop(run_id)
        deps.emit(
            run_id,
            "resumed",
            "evidence",
            "Course evidence analysis resumed from its checkpoint.",
            {"stage": "frontier"},
        )
        return {"pause_pending": False, "resumed_from_stop": False}

    def finish(state: EvidenceGraphState) -> dict[str, Any]:
        result = EvidencePublicationState.model_validate(state["evidence_result"])
        output = deps.tools.run_output(
            state["selected_tool"],  # type: ignore[arg-type]
            result,
        )
        deps.emit(
            state["run_id"],
            "tool_completed",
            "evidence",
            f"Completed {state['selected_tool']}.",
            {"tool": state["selected_tool"], "status": result.status},
        )
        return {"tool_result": output.model_dump(mode="json")}

    builder = StateGraph(EvidenceGraphState)
    builder.add_node("start_evidence_tool", start_tool)
    builder.add_node("run_immediate_evidence_tool", run_immediate)
    builder.add_node("stop_before_immediate", detect_stop)
    builder.add_node("pause_before_immediate", pause_at("immediate"))
    builder.add_node("stop_before_authorize", detect_stop)
    builder.add_node("pause_before_authorize", pause_at("authorize"))
    builder.add_node("prepare_course_sources", prepare)
    builder.add_node("stop_before_frontier", detect_stop)
    builder.add_node("pause_before_frontier", pause_at("frontier"))
    builder.add_node("analyze_frontier", analyze)
    builder.add_node("stop_before_publish", detect_stop)
    builder.add_node("pause_before_publish", pause_at("publish"))
    builder.add_node("publish_frontier", publish)
    builder.add_node("pause_evidence_frontier", pause_evidence)
    builder.add_node("pause_source_changed", pause_source_changed)
    builder.add_node("finish_evidence_tool", finish)
    builder.add_edge(START, "start_evidence_tool")
    builder.add_conditional_edges(
        "start_evidence_tool",
        route_kind,
        {
            "inspect": "run_immediate_evidence_tool",
            "answer": "stop_before_immediate",
            "staged": "stop_before_authorize",
        },
    )
    builder.add_conditional_edges(
        "stop_before_immediate",
        route_after_stop,
        {"pause": "pause_before_immediate", "continue": "run_immediate_evidence_tool"},
    )
    builder.add_edge("pause_before_immediate", "run_immediate_evidence_tool")
    builder.add_edge("run_immediate_evidence_tool", END)
    builder.add_conditional_edges(
        "stop_before_authorize",
        route_after_stop,
        {"pause": "pause_before_authorize", "continue": "prepare_course_sources"},
    )
    builder.add_edge("pause_before_authorize", "prepare_course_sources")
    builder.add_conditional_edges(
        "prepare_course_sources",
        route_after_authority_step,
        {
            "continue": "stop_before_frontier",
            "source_changed": "pause_source_changed",
        },
    )
    builder.add_conditional_edges(
        "stop_before_frontier",
        route_after_stop,
        {"pause": "pause_before_frontier", "continue": "analyze_frontier"},
    )
    builder.add_edge("pause_before_frontier", "analyze_frontier")
    builder.add_conditional_edges(
        "analyze_frontier",
        route_after_authority_step,
        {
            "continue": "stop_before_publish",
            "source_changed": "pause_source_changed",
        },
    )
    builder.add_conditional_edges(
        "stop_before_publish",
        route_after_stop,
        {"pause": "pause_before_publish", "continue": "publish_frontier"},
    )
    builder.add_edge("pause_before_publish", "publish_frontier")
    builder.add_conditional_edges(
        "publish_frontier",
        route_after_publication,
        {
            "complete": "finish_evidence_tool",
            "continue": "stop_before_frontier",
            "paused": "pause_evidence_frontier",
            "source_changed": "pause_source_changed",
        },
    )
    builder.add_edge("pause_evidence_frontier", "stop_before_frontier")
    builder.add_edge("pause_source_changed", "stop_before_authorize")
    builder.add_edge("finish_evidence_tool", END)
    return builder.compile(checkpointer=checkpointer)


def _authority_error(error: SourceAuthorizationError) -> dict[str, str]:
    return {"code": error.code, "entry_id": error.entry_id}
