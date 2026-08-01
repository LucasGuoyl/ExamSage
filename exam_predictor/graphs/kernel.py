from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from exam_predictor.graphs.evidence import (
    EvidenceGraphDependencies,
    build_evidence_graph,
)
from exam_predictor.runtime.control import RunControlRegistry
from exam_predictor.runtime.provider_sessions import ProviderSessionRegistry
from exam_predictor.tools.kernel import KernelPlanner, KernelToolRegistry
from exam_predictor.tools.evidence import EvidencePlannerContext, is_evidence_tool


class KernelState(TypedDict, total=False):
    run_id: str
    provider_profile_id: str
    workspace_id: str | None
    user_message: str
    messages: Annotated[list[dict[str, str]], operator.add]
    selected_tool: str
    tool_arguments: dict[str, Any]
    plan_reason: str
    tool_result: dict[str, Any]
    assistant_message: str
    pause_pending: bool
    revision_id: str
    evidence_inspection: dict[str, Any]
    frontier: dict[str, Any]
    evidence_result: dict[str, Any]


EventEmitter = Callable[[str, str, str, str, dict[str, Any] | None], None]


@dataclass(frozen=True)
class KernelDependencies:
    provider_sessions: ProviderSessionRegistry
    planner: KernelPlanner
    tools: KernelToolRegistry
    controls: RunControlRegistry
    emit: EventEmitter


def build_kernel_graph(deps: KernelDependencies, checkpointer):
    def detect_stop(state: KernelState) -> dict[str, Any]:
        run_id = state["run_id"]
        if state.get("pause_pending") or not deps.controls.is_stop_requested(run_id):
            return {}
        return {"pause_pending": True}

    def pause(state: KernelState) -> dict[str, Any]:
        if not state.get("pause_pending"):
            return {}
        run_id = state["run_id"]
        resume = interrupt({"kind": "stopped", "run_id": run_id})
        if resume != {"action": "resume"}:
            raise ValueError("A paused run must be resumed with {'action': 'resume'}.")
        deps.controls.clear_stop(run_id)
        deps.emit(run_id, "resumed", "planning", "The run resumed from its checkpoint.")
        return {"pause_pending": False}

    def route_after_stop(state: KernelState) -> str:
        return "pause" if state.get("pause_pending") else "continue"

    def plan(state: KernelState) -> dict[str, Any]:
        run_id = state["run_id"]
        deps.emit(
            run_id,
            "progress",
            "planning",
            "Choosing the next Agent tool.",
            {
                "evidence_event": "provider_operation_started",
                "operation": "kernel_planner",
                "model_route": "fast",
            },
        )
        provider = deps.provider_sessions.get_provider(state["provider_profile_id"])
        workspace_id = state.get("workspace_id")
        evidence_context = None
        if workspace_id is not None and deps.tools.evidence is not None:
            try:
                evidence_context = deps.tools.evidence.planner_context(workspace_id)
            except Exception:
                evidence_context = EvidencePlannerContext(
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
        plan_result = deps.planner.plan(
            state["user_message"],
            state.get("messages", []),
            provider,
            workspace_id=workspace_id,
            evidence_context=evidence_context,
        )
        return {
            "selected_tool": plan_result.tool,
            "tool_arguments": plan_result.arguments,
            "plan_reason": plan_result.reason,
        }

    def run_tool(state: KernelState) -> dict[str, Any]:
        run_id = state["run_id"]
        tool = state["selected_tool"]
        deps.emit(run_id, "tool_started", "tool", f"Running {tool}.", {"tool": tool})
        provider = deps.provider_sessions.get_provider(state["provider_profile_id"])
        result = deps.tools.execute(
            tool=tool,
            arguments=state.get("tool_arguments", {}),
            history=state.get("messages", []),
            provider=provider,
        )
        deps.emit(run_id, "tool_completed", "tool", f"Completed {tool}.", {"tool": tool})
        return {"tool_result": result.model_dump()}

    def compose(state: KernelState) -> dict[str, Any]:
        answer = str(state["tool_result"]["content"])
        deps.emit(state["run_id"], "message", "answer", answer)
        return {
            "assistant_message": answer,
            "messages": [{"role": "assistant", "content": answer}],
        }

    def route_tool(state: KernelState) -> str:
        if is_evidence_tool(state["selected_tool"]) and deps.tools.evidence is not None:
            return "evidence"
        return "kernel"

    builder = StateGraph(KernelState)
    builder.add_node("stop_before_plan", detect_stop)
    builder.add_node("pause_before_plan", pause)
    builder.add_node("plan", plan)
    builder.add_node("stop_before_tool", detect_stop)
    builder.add_node("pause_before_tool", pause)
    builder.add_node("run_tool", run_tool)
    evidence_routes = {"kernel": "run_tool"}
    if deps.tools.evidence is not None:
        evidence_graph = build_evidence_graph(
            EvidenceGraphDependencies(
                tools=deps.tools.evidence,
                controls=deps.controls,
                emit=deps.emit,
            )
        )
        builder.add_node("run_evidence_tool", evidence_graph)
        evidence_routes["evidence"] = "run_evidence_tool"
    builder.add_node("stop_before_compose", detect_stop)
    builder.add_node("pause_before_compose", pause)
    builder.add_node("compose", compose)
    builder.add_edge(START, "stop_before_plan")
    builder.add_conditional_edges(
        "stop_before_plan",
        route_after_stop,
        {"pause": "pause_before_plan", "continue": "plan"},
    )
    builder.add_edge("pause_before_plan", "plan")
    builder.add_edge("plan", "stop_before_tool")
    builder.add_conditional_edges(
        "stop_before_tool",
        route_after_stop,
        {"pause": "pause_before_tool", "continue": "route_selected_tool"},
    )
    builder.add_node("route_selected_tool", lambda _state: {})
    builder.add_conditional_edges("route_selected_tool", route_tool, evidence_routes)
    builder.add_edge("pause_before_tool", "route_selected_tool")
    builder.add_edge("run_tool", "stop_before_compose")
    if deps.tools.evidence is not None:
        builder.add_edge("run_evidence_tool", "stop_before_compose")
    builder.add_conditional_edges(
        "stop_before_compose",
        route_after_stop,
        {"pause": "pause_before_compose", "continue": "compose"},
    )
    builder.add_edge("pause_before_compose", "compose")
    builder.add_edge("compose", END)
    return builder.compile(checkpointer=checkpointer)
