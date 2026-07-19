from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field

import streamlit as st

from exam_predictor.runtime.client import WorkerClient, WorkerClientError
from exam_predictor.runtime.models import (
    AgentEvent,
    ConnectProviderRequest,
    EventType,
    ProviderProfile,
    RunStatus,
    SubmitMessageRequest,
)


@dataclass
class AgentViewState:
    last_sequence: int = 0
    activity: list[str] = field(default_factory=list)
    answer: str | None = None
    settled: bool = False
    paused: bool = False
    failed: bool = False
    answer_committed: bool = False


def reduce_agent_events(
    state: AgentViewState,
    events: Iterable[AgentEvent],
) -> AgentViewState:
    for event in sorted(events, key=lambda item: item.sequence):
        if event.sequence <= state.last_sequence:
            continue
        state.last_sequence = event.sequence
        if event.event_type is EventType.MESSAGE:
            state.answer = event.message
            state.answer_committed = False
        elif event.event_type in {
            EventType.PAUSED,
            EventType.COMPLETED,
            EventType.FAILED,
        }:
            state.settled = True
            state.paused = event.event_type is EventType.PAUSED
            state.failed = event.event_type is EventType.FAILED
            state.activity.append(event.message)
        else:
            if event.event_type in {EventType.STARTED, EventType.RESUMED}:
                state.settled = False
                state.paused = False
                state.failed = False
            if event.event_type not in {EventType.QUEUED, EventType.STARTED}:
                state.activity.append(event.message)
    return state


def _safe_error(exc: Exception, *extra_secrets: str) -> str:
    message = str(exc)
    secrets = [os.environ.get("EXAMSAGE_WORKER_TOKEN", ""), *extra_secrets]
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message or "The local Agent Worker request failed."


def _new_client() -> WorkerClient:
    url = os.environ.get("EXAMSAGE_WORKER_URL", "")
    token = os.environ.get("EXAMSAGE_WORKER_TOKEN", "")
    if not url or not token:
        raise WorkerClientError(
            "The local Agent Worker was not started by the ExamSage launcher."
        )
    return WorkerClient(url, token)


def _check_worker(*extra_secrets: str) -> str | None:
    client: WorkerClient | None = None
    try:
        client = _new_client()
        client.health()
    except WorkerClientError as exc:
        return _safe_error(exc, *extra_secrets)
    finally:
        if client is not None:
            client.close()
    return None


@st.fragment(run_every=0.5)
def _render_run_activity(run_id: str) -> None:
    states = st.session_state.setdefault("agent_view_states", {})
    state = states.setdefault(run_id, AgentViewState())
    client: WorkerClient | None = None
    try:
        client = _new_client()
        events = client.events_after(run_id, after=state.last_sequence)
        snapshot = client.get_run(run_id)
        reduce_agent_events(state, events)
        answer_committed = False
        if state.answer and not state.answer_committed:
            st.session_state.agent_messages.append(
                {"role": "assistant", "content": state.answer}
            )
            state.answer_committed = True
            answer_committed = True
        with st.status(
            f"Agent: {snapshot.status.value}",
            expanded=not snapshot.status.is_settled,
        ):
            for item in state.activity[-12:]:
                st.write(item)
        if snapshot.status in {RunStatus.RUNNING, RunStatus.STOPPING}:
            if st.button(
                "Stop",
                key=f"stop-{run_id}",
                disabled=snapshot.status is RunStatus.STOPPING,
            ):
                client.stop(run_id)
                st.rerun()
        elif snapshot.status is RunStatus.PAUSED:
            if st.button("Resume", key=f"resume-{run_id}"):
                client.resume(run_id)
                state.settled = False
                state.paused = False
                state.failed = False
                st.rerun()
        if snapshot.status in {RunStatus.COMPLETED, RunStatus.FAILED}:
            st.session_state.pop("agent_active_run_id", None)
            st.rerun()
        if answer_committed:
            st.rerun()
    except WorkerClientError as exc:
        st.session_state.pop("agent_provider", None)
        st.error(
            f"Worker unavailable: {_safe_error(exc)}. "
            "Restart ExamSage with the launcher."
        )
    finally:
        if client is not None:
            client.close()


def render_agent_kernel() -> None:
    if st.session_state.pop("clear_agent_provider_key", False):
        st.session_state["agent_provider_key"] = ""

    st.title("🎓 ExamSage")
    st.caption(
        "Agent kernel alpha · chat, tool activity, Stop, checkpoints and Resume"
    )

    provider_key = st.session_state.get("agent_provider_key", "")
    unavailable = _check_worker(provider_key)
    if unavailable:
        st.session_state["agent_provider_key"] = ""
        st.session_state.pop("agent_provider", None)
        st.error(
            f"Worker unavailable: {unavailable}. Restart ExamSage with the launcher."
        )
        return

    connect_error = st.session_state.pop("agent_connect_error", None)
    st.session_state.setdefault("agent_messages", [])
    st.session_state.setdefault("agent_thread_id", "default")

    with st.sidebar:
        st.subheader("Provider")
        if connect_error:
            st.error(connect_error)
        provider = st.radio(
            "Provider",
            ["openai", "gemini", "custom"],
            format_func=lambda item: {
                "openai": "OpenAI",
                "gemini": "Google Gemini",
                "custom": "OpenAI-compatible",
            }[item],
        )
        base_url = (
            st.text_input("Compatible base URL") if provider == "custom" else None
        )
        api_key = st.text_input(
            "API key",
            type="password",
            key="agent_provider_key",
        )
        if st.button("Connect", type="primary"):
            st.session_state.pop("agent_provider", None)
            client: WorkerClient | None = None
            try:
                request = ConnectProviderRequest(
                    profile=ProviderProfile(
                        profile_id="primary",
                        provider=provider,
                        base_url=base_url or None,
                    ),
                    api_key=api_key,
                )
                client = _new_client()
                descriptor = client.connect_provider(request)
            except (WorkerClientError, ValueError) as exc:
                safe = _safe_error(exc, api_key)
                st.session_state.agent_connect_error = (
                    f"Provider connection failed: {safe}"
                )
            else:
                st.session_state.agent_provider = descriptor.model_dump(mode="json")
            finally:
                if client is not None:
                    client.close()
            st.session_state.clear_agent_provider_key = True
            st.rerun()
        if "agent_provider" in st.session_state:
            st.success("Provider connected for this session.")

    for message in st.session_state.agent_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    run_id = st.session_state.get("agent_active_run_id")
    if run_id:
        _render_run_activity(run_id)

    prompt = st.chat_input("Message ExamSage…", disabled=bool(run_id))
    if not prompt or run_id:
        return
    if "agent_provider" not in st.session_state:
        st.warning("Connect one provider before sending a message.")
        return

    request = SubmitMessageRequest(
        thread_id=st.session_state.agent_thread_id,
        provider_profile_id="primary",
        message=prompt,
    )
    client = None
    try:
        client = _new_client()
        submitted = client.submit_message(request)
    except WorkerClientError as exc:
        st.error(f"Could not start the Agent run: {_safe_error(exc)}")
    else:
        st.session_state.agent_messages.append(
            {"role": "user", "content": prompt}
        )
        st.session_state.agent_active_run_id = submitted.run_id
        st.rerun()
    finally:
        if client is not None:
            client.close()
