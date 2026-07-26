from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import BinaryIO, Protocol
from uuid import uuid4

import streamlit as st

from exam_predictor.runtime.client import WorkerClient, WorkerClientError
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


@dataclass(frozen=True)
class WorkspaceActionState:
    can_rescan: bool
    can_edit_inclusion: bool
    can_approve: bool
    can_delete: bool
    reason: str | None = None


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
) -> ManifestEntry | None:
    """Return the folder entry that exactly represents an entry's visible parent."""
    prefix, _, _ = entry.relative_path.rpartition("/")
    if not prefix:
        return None
    return next(
        (
            candidate
            for candidate in entries
            if candidate.item_kind == "folder" and candidate.relative_path == prefix
        ),
        None,
    )


class NamedBinaryUpload(Protocol):
    name: str

    def read(self, size: int = -1) -> bytes: ...

    def seek(self, offset: int, whence: int = 0) -> int: ...


def upload_directory(
    client: WorkerClient,
    display_name: str,
    uploaded_files: Sequence[NamedBinaryUpload],
    idempotency_key: str,
) -> WorkspaceJob:
    """Forward browser-uploaded streams to Worker without a local write boundary."""
    files: dict[str, BinaryIO] = {
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
    if any(
        entry.state
        in {SourceState.CHANGED, SourceState.FAILED, SourceState.REMOVED}
        for entry in manifest.items
    ):
        return WorkspaceActionState(
            can_rescan,
            can_edit,
            False,
            can_delete,
            "The manifest needs attention before approval.",
        )
    if not any(entry.included and entry.sha256 for entry in manifest.items):
        return WorkspaceActionState(
            can_rescan,
            can_edit,
            False,
            can_delete,
            "Include at least one hashable file before approval.",
        )
    return WorkspaceActionState(can_rescan, can_edit, True, can_delete)


def _safe_workspace_error() -> None:
    st.error("The course workspace request could not be completed. Please try again.")


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
        return "Review the updated manifest."
    if job.status is WorkspaceJobStatus.FAILED:
        return "Review the manifest or choose the course folder again."
    if job.status is WorkspaceJobStatus.CANCELLED:
        return "Start a new scan when you are ready."
    return "The local Worker is scanning your course files."


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
    with st.status(f"Course workspace: {current.status.value}", expanded=True):
        left, middle, right = st.columns(3)
        left.metric("Files discovered", progress.get("discovered_count", 0))
        middle.metric("Bytes hashed", progress.get("bytes_hashed", 0))
        right.metric("Failures", progress.get("failure_count", 0))
        st.caption(f"Next action: {_job_next_action(current)}")
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
        st.info("Scan updates are paused. Select Refresh workspace job to poll again.")
        if st.button("Refresh workspace job"):
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
        st.warning("Give the uploaded course directory a display name.")
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
    with st.expander("Saved provider status"):
        for provider in saved:
            state = "saved" if provider.credential_expected else "not saved"
            if provider.reconnect_required:
                state += "; reconnect required"
            st.caption(f"{provider.profile.provider}: credential {state}.")
            st.warning(
                "Forgetting an API key disconnects this provider, but leaves course files and workspaces intact."
            )
            if st.button("Forget API key", key=f"forget-provider-{provider.profile.profile_id}"):
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
    st.caption(f"Workspace state: {workspace.state.value.replace('_', ' ')}")
    if state.reason:
        st.info(state.reason)
    if st.button("Rescan workspace", disabled=not state.can_rescan):
        try:
            _start_job(client.rescan(workspace.workspace_id, str(uuid4())))
        except WorkerClientError:
            _safe_workspace_error()

    count_columns = st.columns(len(SourceState))
    for column, source_state in zip(count_columns, SourceState, strict=True):
        column.metric(
            source_state.value.replace("_", " "),
            manifest_counts(approval_manifest.items)[source_state],
        )

    rows = [
        {
            "Relative path": entry.relative_path,
            "Category": entry.format_category or entry.item_kind,
            "Bytes": entry.size_bytes,
            "Modified": _format_modified(entry.modified_ns),
            "State": entry.state.value,
            "Hash": entry.sha256[:12] if entry.sha256 else "—",
            "Group": entry.proposed_course_group,
            "Reason": entry.safe_message
            or entry.failure_code
            or entry.inclusion_reason
            or entry.state.value.replace("_", " "),
        }
        for entry in manifest.items
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    for entry in manifest.items:
        with st.expander(entry.relative_path):
            included = st.checkbox(
                "Include this path",
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
                f"Apply inclusion to {authority.relative_path}",
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

    if st.button("Approve current draft", disabled=not state.can_approve):
        try:
            client.approve_workspace(
                workspace.workspace_id, workspace.current_draft_revision_id or ""
            )
        except WorkerClientError:
            _safe_workspace_error()
        else:
            st.rerun()

    with st.expander("Delete workspace"):
        confirmation = st.text_input(
            f"Type {workspace.display_name} to delete this workspace",
            key=f"delete-workspace-confirmation-{workspace.workspace_id}",
        )
        if st.button(
            "Delete workspace",
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
    st.header("Course workspace")
    if st.button("Choose course folder", type="primary"):
        try:
            job = client.select_folder(str(uuid4()))
        except WorkerClientError:
            _safe_workspace_error()
        else:
            if job is None:
                st.info("No course folder was selected.")
            else:
                _start_job(job)

    with st.expander("Development fallback: upload a course directory"):
        uploaded_files = st.file_uploader(
            "Upload a course directory",
            accept_multiple_files="directory",
            max_upload_size=1024,
            key="workspace_directory_upload",
        )
        st.text_input("Course directory display name", key="workspace_upload_name")
        if st.button(
            "Create uploaded workspace", disabled=not uploaded_files
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
        st.info("Choose a course folder to create a workspace.")
        return None

    workspace_by_id = {item.workspace_id: item for item in workspaces}
    prior = st.session_state.get("selected_workspace_id")
    selected_id = st.selectbox(
        "Course workspace",
        options=list(workspace_by_id),
        index=list(workspace_by_id).index(prior) if prior in workspace_by_id else 0,
        format_func=lambda workspace_id: workspace_by_id[workspace_id].display_name,
    )
    st.session_state["selected_workspace_id"] = selected_id
    try:
        workspace = client.get_workspace(selected_id)
        all_entries = load_complete_manifest(client, selected_id)
        source_filter = st.selectbox(
            "Filter source state",
            options=[None, *SourceState],
            format_func=lambda value: "All states" if value is None else value.value,
        )
        courses = sorted({entry.proposed_course_group for entry in all_entries})
        course_filter = st.selectbox("Filter course group", options=[None, *courses])
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
            "Manifest page", options=list(range(1, page_count + 1))
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

    with st.expander("Delete all workspaces"):
        confirmation = st.text_input(
            "Type DELETE ALL to delete every workspace", key="delete-all-workspaces"
        )
        if st.button(
            "Delete all workspaces",
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
