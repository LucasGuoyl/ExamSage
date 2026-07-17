from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from exam_predictor.runtime.control import RunControlRegistry
from exam_predictor.runtime.provider_sessions import ProviderSessionRegistry
from exam_predictor.tools.kernel import KernelPlanner, KernelToolRegistry


class KernelState(TypedDict, total=False):
    run_id: str
    provider_profile_id: str
    user_message: str
    messages: Annotated[list[dict[str, str]], operator.add]
    selected_tool: str
    tool_arguments: dict[str, Any]
    plan_reason: str
    tool_result: dict[str, Any]
    assistant_message: str


EventEmitter = Callable[[str, str, str, str, dict[str, Any] | None], None]


@dataclass(frozen=True)
class KernelDependencies:
    provider_sessions: ProviderSessionRegistry
    planner: KernelPlanner
    tools: KernelToolRegistry
    controls: RunControlRegistry
    emit: EventEmitter


def build_kernel_graph(deps: KernelDependencies, checkpointer):
    def stop_gate(state: KernelState) -> dict[str, Any]:
        run_id = state["run_id"]
        if not deps.controls.is_stop_requested(run_id):
            return {}
        deps.emit(run_id, "paused", "paused", "The run is paused at a safe boundary.")
        resume = interrupt({"kind": "stopped", "run_id": run_id})
        if resume != {"action": "resume"}:
            raise ValueError("A paused run must be resumed with {'action': 'resume'}.")
        deps.controls.clear_stop(run_id)
        deps.emit(run_id, "resumed", "planning", "The run resumed from its checkpoint.")
        return {}

    def plan(state: KernelState) -> dict[str, Any]:
        run_id = state["run_id"]
        deps.emit(run_id, "progress", "planning", "Choosing the next Agent tool.")
        provider = deps.provider_sessions.get_provider(state["provider_profile_id"])
        plan_result = deps.planner.plan(state["user_message"], state.get("messages", []), provider)
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

    builder = StateGraph(KernelState)
    builder.add_node("stop_before_plan", stop_gate)
    builder.add_node("plan", plan)
    builder.add_node("stop_before_tool", stop_gate)
    builder.add_node("run_tool", run_tool)
    builder.add_node("stop_before_compose", stop_gate)
    builder.add_node("compose", compose)
    builder.add_edge(START, "stop_before_plan")
    builder.add_edge("stop_before_plan", "plan")
    builder.add_edge("plan", "stop_before_tool")
    builder.add_edge("stop_before_tool", "run_tool")
    builder.add_edge("run_tool", "stop_before_compose")
    builder.add_edge("stop_before_compose", "compose")
    builder.add_edge("compose", END)
    return builder.compile(checkpointer=checkpointer)
