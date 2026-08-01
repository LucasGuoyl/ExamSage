from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

import streamlit as st

from exam_predictor.runtime.client import (
    SeekableBinaryStream,
    WorkerClient,
    WorkerClientError,
)
from exam_predictor.ui.i18n import get_ui_language, text
from exam_predictor.workspace.models import (
    EntryInclusionRequest,
    ManifestEntry,
    ManifestPage,
    SourceState,
    WorkspaceDetail,
    WorkspaceJob,
    WorkspaceJobStatus,
    WorkspaceState,
    WorkspaceSummary,
)


WORKSPACE_JOB_MAX_POLLS = 8
WORKSPACE_JOB_MAX_RERUNS = 8
MANIFEST_WORKER_PAGE_SIZE = 500
MANIFEST_DISPLAY_PAGE_SIZE = 100


def _language() -> str:
    return get_ui_language(st.session_state)


def _copy(key: str, **values: object) -> str:
    return text(key, _language(), **values)


_ACTION_REASON_KEYS = {
    "A scan is in progress.": "reason_scan_in_progress",
    "Workspace deletion is in progress.": "reason_deletion_in_progress",
    "A current draft requires review.": "reason_draft_review",
    "No current draft is available.": "reason_no_draft",
    "Review all manifest entries before approval.": "reason_review_all",
    "The manifest needs attention before approval.": "reason_manifest_attention",
    "Include at least one hashable file before approval.": "reason_include_file",
}


def _localized_reason(reason: str) -> str:
    key = _ACTION_REASON_KEYS.get(reason)
    return reason if key is None else _copy(key)


def _state_label(value: SourceState | WorkspaceState) -> str:
    return _copy(f"state_{value.value}")


@dataclass(frozen=True)
class WorkspaceActionState:
    can_rescan: bool
    can_edit_inclusion: bool
    can_approve: bool
    can_delete: bool
    reason: str | None = None


@dataclass(frozen=True)
class SubtreeAuthority:
    entry_id: str
    relative_prefix: str


def manifest_counts(entries: Sequence[ManifestEntry]) -> dict[SourceState, int]:
    """Return every source state with a visible zero-or-greater count."""
    counted = Counter(entry.state for entry in entries)
    return {state: counted[state] for state in SourceState}


def load_complete_manifest(
    client: WorkerClient, workspace_id: str
) -> tuple[ManifestEntry, ...]:
    """Read the finite Worker manifest snapshot in bounded server pages."""
    first = client.get_manifest(workspace_id, offset=0, limit=MANIFEST_WORKER_PAGE_SIZE)
    entries = list(first.items)
    for offset in range(first.limit, first.total, first.limit):
        entries.extend(
            client.get_manifest(
                workspace_id, offset=offset, limit=MANIFEST_WORKER_PAGE_SIZE
            ).items
        )
    return tuple(entries)


def can_delete_all_workspaces(
    workspaces: Sequence[WorkspaceDetail | WorkspaceSummary],
) -> bool:
    """Allow deletion only when no durable workspace operation is active."""
    return all(
        workspace.state
        not in {
            WorkspaceState.SCANNING,
            WorkspaceState.DELETING,
            WorkspaceState.CLEANUP_PENDING,
        }
        for workspace in workspaces
    )


def subtree_authority(
    entries: Sequence[ManifestEntry], entry: ManifestEntry
) -> SubtreeAuthority | None:
    """Return server-expandable authority for the source entry's parent tree."""
    source = entry
    if entry.archive_parent_entry_id is not None:
        source = next(
            (
                candidate
                for candidate in entries
                if candidate.entry_id == entry.archive_parent_entry_id
            ),
            entry,
        )
    prefix = (
        source.relative_path.rstrip("/")
        if source.item_kind == "folder"
        else source.relative_path.rpartition("/")[0]
    )
    folder = next(
        (
            candidate
            for candidate in entries
            if prefix
            and candidate.archive_parent_entry_id is None
            and candidate.item_kind == "folder"
            and candidate.relative_path == prefix
        ),
        None,
    )
    return SubtreeAuthority(
        entry_id=folder.entry_id if folder is not None else source.entry_id,
        relative_prefix=prefix,
    )


class NamedBinaryUpload(Protocol):
    name: str

    def read(self, size: int = -1) -> bytes: ...

    def seek(self, offset: int, whence: int = 0) -> int: ...

    def tell(self) -> int: ...


def upload_directory(
    client: WorkerClient,
    display_name: str,
    uploaded_files: Sequence[NamedBinaryUpload],
    idempotency_key: str,
) -> WorkspaceJob:
    """Forward browser-uploaded streams to Worker without a local write boundary."""
    files: dict[str, SeekableBinaryStream] = {
        str(uploaded.name): uploaded for uploaded in uploaded_files
    }
    return client.upload_directory(display_name, files, idempotency_key)


def action_state(
    workspace: WorkspaceDetail, manifest: ManifestPage
) -> WorkspaceActionState:
    """Derive workspace control enablement exclusively from Worker state."""
    if workspace.state is WorkspaceState.SCANNING:
        return WorkspaceActionState(False, False, False, False, "A scan is in progress.")
    if workspace.state in {WorkspaceState.DELETING, WorkspaceState.CLEANUP_PENDING}:
        return WorkspaceActionState(
            False, False, False, False, "Workspace deletion is in progress."
        )

    can_rescan = True
    can_edit = workspace.state is WorkspaceState.APPROVAL_REQUIRED
    can_delete = True
    if workspace.state is not WorkspaceState.APPROVAL_REQUIRED:
        return WorkspaceActionState(
            can_rescan, can_edit, False, can_delete, "A current draft requires review."
        )
    if workspace.current_draft_revision_id is None:
        return WorkspaceActionState(
            can_rescan, can_edit, False, can_delete, "No current draft is available."
        )
    if manifest.total > len(manifest.items):
        return WorkspaceActionState(
            can_rescan,
            can_edit,
            False,
            can_delete,
            "Review all manifest entries before approval.",
        )
    approval_eligible_states = {
        SourceState.PENDING_APPROVAL,
        SourceState.APPROVED,
    }
    if any(
        entry.included
        and (entry.state not in approval_eligible_states or not entry.sha256)
        for entry in manifest.items
    ):
        return WorkspaceActionState(
            can_rescan,
            can_edit,
            False,
            can_delete,
            "The manifest needs attention before approval.",
        )
    if not any(
        entry.included
        and entry.sha256
        and entry.state in approval_eligible_states
        for entry in manifest.items
    ):
        return WorkspaceActionState(
            can_rescan,
            can_edit,
            False,
            can_delete,
            "Include at least one hashable file before approval.",
        )
    return WorkspaceActionState(can_rescan, can_edit, True, can_delete)


def _safe_workspace_error() -> None:
    st.error(_copy("workspace_request_failed"))


def _format_modified(modified_ns: int | None) -> str:
    if modified_ns is None:
        return "—"
    return datetime.fromtimestamp(modified_ns / 1_000_000_000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def _start_job(job: WorkspaceJob) -> None:
    st.session_state["workspace_job"] = job
    st.session_state["workspace_job_cursor"] = 0
    st.session_state["workspace_job_polls"] = 0
    st.session_state["workspace_job_progress"] = {
        "discovered_count": 0,
        "bytes_hashed": 0,
        "failure_count": 0,
    }
    st.rerun()


def _job_next_action(job: WorkspaceJob) -> str:
    if job.status is WorkspaceJobStatus.SUCCEEDED:
        return _copy("job_review_manifest")
    if job.status is WorkspaceJobStatus.FAILED:
        return _copy("job_review_or_choose")
    if job.status is WorkspaceJobStatus.CANCELLED:
        return _copy("job_start_scan")
    return _copy("job_scanning")


def _render_workspace_job(client: WorkerClient) -> None:
    current = st.session_state.get("workspace_job")
    if not isinstance(current, WorkspaceJob):
        return
    try:
        events = client.workspace_events_after(
            current.job_id, after=int(st.session_state.get("workspace_job_cursor", 0))
        )
        if events:
            st.session_state["workspace_job_cursor"] = max(
                event.sequence for event in events
            )
        progress = dict(st.session_state.get("workspace_job_progress", {}))
        for event in events:
            for key in ("discovered_count", "bytes_hashed", "failure_count"):
                if key in event.payload:
                    progress[key] = event.payload[key]
        st.session_state["workspace_job_progress"] = progress
        current = client.get_job(current.job_id)
        st.session_state["workspace_job"] = current
    except WorkerClientError:
        _safe_workspace_error()
        return

    progress = st.session_state["workspace_job_progress"]
    with st.status(_copy("job_status", status=current.status.value), expanded=True):
        left, middle, right = st.columns(3)
        left.metric(_copy("files_discovered"), progress.get("discovered_count", 0))
        middle.metric(_copy("bytes_hashed"), progress.get("bytes_hashed", 0))
        right.metric(_copy("failures"), progress.get("failure_count", 0))
        st.caption(_copy("next_action", action=_job_next_action(current)))
        for event in events[-8:]:
            st.write(event.message)

    if current.status in {
        WorkspaceJobStatus.SUCCEEDED,
        WorkspaceJobStatus.FAILED,
        WorkspaceJobStatus.CANCELLED,
    }:
        st.session_state.pop("workspace_job", None)
        return
    polls = int(st.session_state.get("workspace_job_polls", 0))
    if polls >= min(WORKSPACE_JOB_MAX_POLLS, WORKSPACE_JOB_MAX_RERUNS):
        st.info(_copy("scan_polling_paused"))
        if st.button(_copy("refresh_workspace_job")):
            st.session_state["workspace_job_polls"] = 0
            st.rerun()
        return
    st.session_state["workspace_job_polls"] = polls + 1
    st.rerun()


def _upload_directory(
    client: WorkerClient, uploaded_files: Sequence[NamedBinaryUpload]
) -> None:
    display_name = st.session_state.get("workspace_upload_name", "").strip()
    if not display_name:
        st.warning(_copy("display_name_required"))
        return
    try:
        _start_job(upload_directory(client, display_name, uploaded_files, str(uuid4())))
    except WorkerClientError:
        _safe_workspace_error()


def _render_credentials(client: WorkerClient) -> None:
    try:
        saved = client.list_saved_providers()
    except WorkerClientError:
        _safe_workspace_error()
        return
    if not saved:
        return
    with st.expander(_copy("saved_provider_status")):
        for provider in saved:
            state = (
                _copy("credential_saved")
                if provider.credential_expected
                else _copy("credential_not_saved")
            )
            if provider.reconnect_required:
                state += _copy("reconnect_required")
            st.caption(
                _copy(
                    "credential_state",
                    provider=provider.profile.provider,
                    state=state,
                )
            )
            st.warning(_copy("forget_key_warning"))
            if st.button(
                _copy("forget_api_key"),
                key=f"forget-provider-{provider.profile.profile_id}",
            ):
                try:
                    client.forget_provider_credential(provider.profile.profile_id)
                except WorkerClientError:
                    _safe_workspace_error()
                else:
                    st.rerun()


def _render_manifest_controls(
    client: WorkerClient,
    workspace: WorkspaceDetail,
    manifest: ManifestPage,
    approval_manifest: ManifestPage,
) -> None:
    state = action_state(workspace, approval_manifest)
    st.subheader(workspace.display_name)
    st.caption(_copy("workspace_state", state=_state_label(workspace.state)))
    if state.reason:
        st.info(_localized_reason(state.reason))
    if st.button(_copy("rescan_workspace"), disabled=not state.can_rescan):
        try:
            _start_job(client.rescan(workspace.workspace_id, str(uuid4())))
        except WorkerClientError:
            _safe_workspace_error()

    count_columns = st.columns(len(SourceState))
    for column, source_state in zip(count_columns, SourceState, strict=True):
        column.metric(
            _state_label(source_state),
            manifest_counts(approval_manifest.items)[source_state],
        )

    rows = [
        {
            _copy("relative_path"): entry.relative_path,
            _copy("category"): entry.format_category or entry.item_kind,
            _copy("bytes"): entry.size_bytes,
            _copy("modified"): _format_modified(entry.modified_ns),
            _copy("state"): _state_label(entry.state),
            _copy("hash"): entry.sha256[:12] if entry.sha256 else "—",
            _copy("group"): entry.proposed_course_group,
            _copy("reason"): entry.safe_message
            or entry.failure_code
            or entry.inclusion_reason
            or _state_label(entry.state),
        }
        for entry in manifest.items
    ]
    st.dataframe(rows, width="stretch", hide_index=True)

    for entry in manifest.items:
        with st.expander(entry.relative_path):
            included = st.checkbox(
                _copy("include_path"),
                value=entry.included,
                key=f"workspace-include-{entry.entry_id}",
                disabled=not state.can_edit_inclusion,
            )
            if included != entry.included:
                try:
                    client.set_entry_inclusion(
                        workspace.workspace_id,
                        entry.entry_id,
                        EntryInclusionRequest(
                            revision_id=workspace.current_draft_revision_id or "",
                            included=included,
                        ),
                    )
                except (ValueError, WorkerClientError):
                    _safe_workspace_error()
                else:
                    st.rerun()
            authority = subtree_authority(approval_manifest.items, entry)
            if authority is not None and st.button(
                _copy(
                    "apply_inclusion",
                    path=authority.relative_prefix or _copy("workspace_root"),
                ),
                key=f"workspace-subtree-{entry.entry_id}",
                disabled=not state.can_edit_inclusion,
            ):
                try:
                    client.set_entry_inclusion(
                        workspace.workspace_id,
                        authority.entry_id,
                        EntryInclusionRequest(
                            revision_id=workspace.current_draft_revision_id or "",
                            included=included,
                            subtree=True,
                        ),
                    )
                except (ValueError, WorkerClientError):
                    _safe_workspace_error()
                else:
                    st.rerun()

    if st.button(_copy("approve_draft"), disabled=not state.can_approve):
        try:
            client.approve_workspace(
                workspace.workspace_id, workspace.current_draft_revision_id or ""
            )
        except WorkerClientError:
            _safe_workspace_error()
        else:
            st.rerun()

    with st.expander(_copy("delete_workspace")):
        confirmation = st.text_input(
            _copy("delete_workspace_confirmation", name=workspace.display_name),
            key=f"delete-workspace-confirmation-{workspace.workspace_id}",
        )
        if st.button(
            _copy("delete_workspace"),
            disabled=not state.can_delete or confirmation != workspace.display_name,
            key=f"delete-workspace-{workspace.workspace_id}",
        ):
            try:
                client.delete_workspace(workspace.workspace_id)
            except WorkerClientError:
                _safe_workspace_error()
            else:
                st.session_state.pop("selected_workspace_id", None)
                st.rerun()


def render_workspace_panel(client: WorkerClient) -> str | None:
    """Render course workspace controls and return the selected public workspace ID."""
    st.header(_copy("workspace_title"))
    if st.button(_copy("choose_folder"), type="primary"):
        try:
            job = client.select_folder(str(uuid4()))
        except WorkerClientError:
            _safe_workspace_error()
        else:
            if job is None:
                st.info(_copy("no_folder_selected"))
            else:
                _start_job(job)

    with st.expander(_copy("upload_fallback")):
        uploaded_files = st.file_uploader(
            _copy("upload_directory"),
            accept_multiple_files="directory",
            max_upload_size=1024,
            key="workspace_directory_upload",
        )
        st.text_input(_copy("directory_display_name"), key="workspace_upload_name")
        if st.button(
            _copy("create_uploaded_workspace"), disabled=not uploaded_files
        ):
            _upload_directory(client, uploaded_files)

    _render_workspace_job(client)
    _render_credentials(client)
    try:
        workspaces = client.list_workspaces()
    except WorkerClientError:
        _safe_workspace_error()
        return None
    if not workspaces:
        st.info(_copy("choose_folder_empty"))
        return None

    workspace_by_id = {item.workspace_id: item for item in workspaces}
    prior = st.session_state.get("selected_workspace_id")
    selected_id = st.selectbox(
        _copy("workspace_select"),
        options=list(workspace_by_id),
        index=list(workspace_by_id).index(prior) if prior in workspace_by_id else 0,
        format_func=lambda workspace_id: workspace_by_id[workspace_id].display_name,
    )
    st.session_state["selected_workspace_id"] = selected_id
    try:
        workspace = client.get_workspace(selected_id)
        all_entries = load_complete_manifest(client, selected_id)
        source_filter = st.selectbox(
            _copy("filter_source_state"),
            options=[None, *SourceState],
            format_func=lambda value: (
                _copy("all_states") if value is None else _state_label(value)
            ),
        )
        courses = sorted({entry.proposed_course_group for entry in all_entries})
        course_filter = st.selectbox(
            _copy("filter_course_group"),
            options=[None, *courses],
            format_func=lambda value: _copy("all_groups") if value is None else value,
        )
        filtered_entries = tuple(
            entry
            for entry in all_entries
            if (source_filter is None or entry.state is source_filter)
            and (
                course_filter is None
                or entry.proposed_course_group == course_filter
            )
        )
        page_count = max(
            1,
            (len(filtered_entries) + MANIFEST_DISPLAY_PAGE_SIZE - 1)
            // MANIFEST_DISPLAY_PAGE_SIZE,
        )
        page_number = st.selectbox(
            _copy("manifest_page"), options=list(range(1, page_count + 1))
        )
        display_offset = (page_number - 1) * MANIFEST_DISPLAY_PAGE_SIZE
        manifest = ManifestPage(
            items=filtered_entries[
                display_offset : display_offset + MANIFEST_DISPLAY_PAGE_SIZE
            ],
            total=len(filtered_entries),
            offset=display_offset,
            limit=MANIFEST_DISPLAY_PAGE_SIZE,
            counts=manifest_counts(filtered_entries),
        )
        approval_manifest = ManifestPage(
            items=all_entries,
            total=len(all_entries),
            offset=0,
            limit=MANIFEST_WORKER_PAGE_SIZE,
            counts=manifest_counts(all_entries),
        )
    except WorkerClientError:
        _safe_workspace_error()
        return selected_id
    _render_manifest_controls(client, workspace, manifest, approval_manifest)

    with st.expander(_copy("delete_all")):
        confirmation = st.text_input(
            _copy("delete_all_confirmation"), key="delete-all-workspaces"
        )
        if st.button(
            _copy("delete_all"),
            disabled=(
                confirmation != "DELETE ALL"
                or not can_delete_all_workspaces(workspaces)
            ),
        ):
            try:
                client.delete_all_workspaces()
            except WorkerClientError:
                _safe_workspace_error()
            else:
                st.session_state.pop("selected_workspace_id", None)
                st.rerun()
    return selected_id
