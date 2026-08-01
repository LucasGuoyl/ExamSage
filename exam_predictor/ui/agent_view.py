from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

from exam_predictor.legacy_intake import (
    LegacyIntakeError,
    cleanup_legacy_intake,
    diagnose_legacy_intake,
)
from exam_predictor.runtime.client import WorkerClient, WorkerClientError
from exam_predictor.runtime.models import (
    AgentEvent,
    ConnectProviderRequest,
    EventType,
    ProviderProfile,
    RunStatus,
    SubmitMessageRequest,
)
from exam_predictor.tools.kernel import _response_language
from exam_predictor.ui.evidence_view import render_evidence_panel
from exam_predictor.ui.i18n import (
    SUPPORTED_UI_LANGUAGES,
    get_ui_language,
    set_ui_language,
    text,
)
from exam_predictor.ui.workspace_view import render_workspace_panel


AGENT_ACTIVITY_MAX_POLLS = 12


@dataclass
class AgentViewState:
    last_sequence: int = 0
    activity: list[str] = field(default_factory=list)
    answer: str | None = None
    settled: bool = False
    paused: bool = False
    failed: bool = False
    answer_committed: bool = False
    poll_count: int = 0
    status: RunStatus | None = None
    is_evidence_run: bool = False


def reduce_agent_events(
    state: AgentViewState,
    events: Iterable[AgentEvent],
) -> AgentViewState:
    for event in sorted(events, key=lambda item: item.sequence):
        if event.sequence <= state.last_sequence:
            continue
        state.last_sequence = event.sequence
        if event.stage == "evidence":
            state.is_evidence_run = True
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


def _language() -> str:
    return get_ui_language(st.session_state)


def _safe_error(exc: Exception, *extra_secrets: str) -> str:
    message = str(exc)
    secrets = [os.environ.get("EXAMSAGE_WORKER_TOKEN", ""), *extra_secrets]
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message or text("worker_request_failed", _language())


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


def _render_language_selector() -> None:
    current = _language()
    selected = st.selectbox(
        text("interface_language", current),
        options=list(SUPPORTED_UI_LANGUAGES),
        index=SUPPORTED_UI_LANGUAGES.index(current),
        format_func=lambda value: text(
            "language_zh" if value == "zh-CN" else "language_en",
            current,
        ),
        key="ui_language_selector",
    )
    if selected != current:
        set_ui_language(st.session_state, selected)
        st.rerun()


def _legacy_data_root() -> Path:
    configured = os.environ.get("EXAMSAGE_DATA_DIR", "").strip()
    return Path(configured) if configured else Path.home() / ".examsage"


def _render_legacy_intake() -> None:
    language = _language()
    try:
        summary = diagnose_legacy_intake(_legacy_data_root())
    except LegacyIntakeError:
        return
    if not (
        summary.session_count
        or summary.unknown_entry_count
        or summary.unsafe_session_count
    ):
        return
    with st.expander(text("legacy_upload_title", language)):
        st.caption(
            text(
                "legacy_upload_summary",
                language,
                sessions=summary.session_count,
                files=summary.file_count,
                bytes=summary.total_bytes,
            )
        )
        if summary.unknown_entry_count or summary.unsafe_session_count:
            st.warning(text("legacy_upload_warning", language))
        if summary.session_count and st.button(
            text("legacy_upload_cleanup", language),
            key="cleanup-legacy-intake",
        ):
            active = st.session_state.get("intake_id")
            active_ids = (active,) if isinstance(active, str) and active else ()
            try:
                result = cleanup_legacy_intake(
                    _legacy_data_root(),
                    session_ids=summary.session_ids,
                    active_session_ids=active_ids,
                )
            except LegacyIntakeError:
                st.error(text("legacy_upload_cleanup_failed", language))
            else:
                st.success(
                    text(
                        "legacy_upload_cleanup_success",
                        language,
                        sessions=len(result.deleted_session_ids),
                        bytes=result.deleted_bytes,
                    )
                )


def _render_polling_limit(run_id: str, state: AgentViewState) -> bool:
    if state.poll_count < AGENT_ACTIVITY_MAX_POLLS:
        return False
    language = _language()
    st.info(text("polling_paused", language))
    if st.button(text("refresh_activity", language), key=f"refresh-run-{run_id}"):
        state.poll_count = 0
        st.rerun()
    return True


@st.fragment(run_every=0.5)
def _render_run_activity(run_id: str) -> None:
    states = st.session_state.setdefault("agent_view_states", {})
    state = states.setdefault(run_id, AgentViewState())
    if _render_polling_limit(run_id, state):
        return
    language = _language()
    client: WorkerClient | None = None
    try:
        state.poll_count += 1
        client = _new_client()
        events = client.events_after(run_id, after=state.last_sequence)
        snapshot = client.get_run(run_id)
        state.status = snapshot.status
        reduce_agent_events(state, events)
        answer_committed = False
        if state.answer and not state.answer_committed:
            st.session_state.agent_messages.append(
                {"role": "assistant", "content": state.answer}
            )
            state.answer_committed = True
            answer_committed = True
        with st.status(
            text("agent_status", language, status=snapshot.status.value),
            expanded=not snapshot.status.is_settled,
        ):
            for item in state.activity[-12:]:
                st.write(item)
        if snapshot.status in {RunStatus.RUNNING, RunStatus.STOPPING}:
            if st.button(
                text("stop", language),
                key=f"stop-{run_id}",
                disabled=snapshot.status is RunStatus.STOPPING,
            ):
                client.stop(run_id)
                state.poll_count = 0
                st.rerun()
        elif snapshot.status is RunStatus.PAUSED:
            if st.button(text("resume", language), key=f"resume-{run_id}"):
                client.resume(run_id)
                state.settled = False
                state.paused = False
                state.failed = False
                state.poll_count = 0
                st.rerun()
        if snapshot.status in {RunStatus.COMPLETED, RunStatus.FAILED}:
            st.session_state.pop("agent_active_run_id", None)
            st.rerun()
        if answer_committed:
            st.rerun()
    except WorkerClientError:
        st.session_state.pop("agent_provider", None)
        st.error(
            text(
                "worker_unavailable",
                language,
                detail=text("worker_request_failed", language),
            )
        )
    finally:
        if client is not None:
            client.close()


def _submit_agent_message(
    prompt: str,
    workspace_id: str | None,
    *,
    academic_language: str | None = None,
) -> bool:
    language = _language()
    if st.session_state.get("agent_active_run_id"):
        return False
    if "agent_provider" not in st.session_state:
        st.warning(text("connect_before_message", language))
        return False
    request = SubmitMessageRequest(
        thread_id=st.session_state.get("agent_thread_id", "default"),
        provider_profile_id="primary",
        workspace_id=workspace_id,
        message=prompt,
    )
    client: WorkerClient | None = None
    try:
        client = _new_client()
        submitted = client.submit_message(request)
    except WorkerClientError as exc:
        st.error(
            text("run_start_failed", language, detail=_safe_error(exc))
        )
        return False
    finally:
        if client is not None:
            client.close()
    st.session_state.agent_messages.append({"role": "user", "content": prompt})
    st.session_state.agent_active_run_id = submitted.run_id
    st.session_state.agent_academic_language = (
        academic_language or _response_language(prompt)
    )
    return True


def _retry_message(response_language: str | None) -> tuple[str, str]:
    language = response_language or st.session_state.get(
        "agent_academic_language", "en"
    )
    label = {"zh-CN": "Chinese", "en": "English"}.get(language, language)
    return (
        "Retry or continue analyzing the approved course evidence from the saved "
        f"checkpoint. Preserve the academic response language; respond in {label}.",
        language,
    )


def render_agent_kernel() -> None:
    if st.session_state.pop("clear_agent_provider_key", False):
        st.session_state["agent_provider_key"] = ""

    with st.sidebar:
        _render_language_selector()
    language = _language()
    st.title("🎓 ExamSage")
    st.caption(text("app_caption", language))

    provider_key = st.session_state.get("agent_provider_key", "")
    unavailable = _check_worker(provider_key)
    if unavailable:
        st.session_state["agent_provider_key"] = ""
        st.session_state.pop("agent_provider", None)
        st.error(text("worker_unavailable", language, detail=unavailable))
        return

    with st.sidebar:
        _render_legacy_intake()

    connect_error = st.session_state.pop("agent_connect_error", None)
    st.session_state.setdefault("agent_messages", [])
    st.session_state.setdefault("agent_thread_id", "default")
    workspace_client: WorkerClient | None = None
    try:
        workspace_client = _new_client()
        selected_workspace_id = render_workspace_panel(workspace_client)
    except WorkerClientError:
        selected_workspace_id = None
        st.error(text("workspace_unavailable", language))
    finally:
        if workspace_client is not None:
            workspace_client.close()

    with st.sidebar:
        st.subheader(text("provider", language))
        if connect_error:
            st.error(connect_error)
        provider = st.radio(
            text("provider", language),
            ["openai", "gemini", "custom"],
            format_func=lambda item: {
                "openai": "OpenAI",
                "gemini": "Google Gemini",
                "custom": "OpenAI-compatible",
            }[item],
        )
        base_url = (
            st.text_input(text("compatible_base_url", language))
            if provider == "custom"
            else None
        )
        api_key = st.text_input(
            text("api_key", language),
            type="password",
            key="agent_provider_key",
        )
        if st.button(text("connect", language), type="primary"):
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
                st.session_state.agent_connect_error = text(
                    "provider_connection_failed", language, detail=safe
                )
            else:
                st.session_state.agent_provider = descriptor.model_dump(mode="json")
            finally:
                if client is not None:
                    client.close()
            st.session_state.clear_agent_provider_key = True
            st.rerun()
        if "agent_provider" in st.session_state:
            st.success(text("provider_connected", language))

    for message in st.session_state.agent_messages:
        with st.chat_message(message["role"]):
            st.text(message["content"])

    run_id = st.session_state.get("agent_active_run_id")
    if run_id:
        _render_run_activity(run_id)

    if selected_workspace_id is not None:
        evidence_client: WorkerClient | None = None
        try:
            evidence_client = _new_client()
            view_states = st.session_state.get("agent_view_states", {})
            run_state = view_states.get(run_id) if run_id else None
            run_status = (
                run_state.status
                if isinstance(run_state, AgentViewState)
                and run_state.is_evidence_run
                else None
            )

            def retry_evidence(response_language: str | None) -> None:
                prompt, academic_language = _retry_message(response_language)
                if _submit_agent_message(
                    prompt,
                    selected_workspace_id,
                    academic_language=academic_language,
                ):
                    st.rerun()

            render_evidence_panel(
                evidence_client,
                selected_workspace_id,
                run_status=run_status,
                on_retry=(
                    retry_evidence
                    if not run_id and "agent_provider" in st.session_state
                    else None
                ),
            )
        except WorkerClientError:
            st.error(text("workspace_request_failed", language))
        finally:
            if evidence_client is not None:
                evidence_client.close()

    run_id = st.session_state.get("agent_active_run_id")
    prompt = st.chat_input(text("message_placeholder", language), disabled=bool(run_id))
    if not prompt or run_id:
        return
    if _submit_agent_message(prompt, selected_workspace_id):
        st.rerun()
