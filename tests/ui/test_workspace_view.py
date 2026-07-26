from __future__ import annotations

from io import BytesIO
from datetime import UTC, datetime

from streamlit.testing.v1 import AppTest

from exam_predictor.runtime.models import SavedProviderProfile
from exam_predictor.workspace.models import (
    ManifestEntry,
    ManifestPage,
    SourceMode,
    SourceState,
    WorkspaceDetail,
    WorkspaceState,
)
from exam_predictor.ui.workspace_view import (
    action_state,
    can_delete_all_workspaces,
    load_complete_manifest,
    manifest_counts,
    subtree_authority,
    upload_directory,
)


NOW = datetime(2026, 7, 21, tzinfo=UTC)
CANONICAL_ROOT = "C:/private/course-folder"


def workspace(
    state: WorkspaceState,
    *,
    draft_revision: str | None = "revision-full-1234567890",
    approved_revision: str | None = None,
) -> WorkspaceDetail:
    return WorkspaceDetail(
        workspace_id="workspace-12345678901234567890123456789012",
        display_name="Calculus",
        source_mode=SourceMode.NATIVE_FOLDER,
        state=state,
        counts={},
        updated_at=NOW,
        current_draft_revision_id=draft_revision,
        current_approved_revision_id=approved_revision,
        created_at=NOW,
    )


def entry(
    state: SourceState = SourceState.PENDING_APPROVAL,
    *,
    entry_id: str = "entry-full-1234567890",
    relative_path: str = "week-1/notes.pdf",
    item_kind: str = "file",
    included: bool = True,
    sha256: str | None = "a" * 64,
    reason: str | None = None,
    proposed_course_group: str = "week-1",
) -> ManifestEntry:
    return ManifestEntry(
        entry_id=entry_id,
        workspace_id="workspace-12345678901234567890123456789012",
        relative_path=relative_path,
        item_kind=item_kind,
        format_category="pdf",
        size_bytes=12,
        modified_ns=1,
        sha256=sha256,
        state=state,
        included=included,
        proposed_course_group=proposed_course_group,
        safe_message=reason,
    )


def page(*entries: ManifestEntry) -> ManifestPage:
    return ManifestPage(items=entries, total=len(entries), offset=0, limit=100, counts={})


def test_manifest_counts_includes_every_source_state_and_keeps_attention_entries_visible():
    manifest = page(
        entry(SourceState.CHANGED, reason="Modified since approval"),
        entry(SourceState.FAILED, reason="Unreadable file"),
        entry(SourceState.REMOVED, reason="Removed from folder"),
    )

    counts = manifest_counts(manifest.items)

    assert set(counts) == set(SourceState)
    assert counts[SourceState.CHANGED] == counts[SourceState.FAILED] == counts[SourceState.REMOVED] == 1
    assert all(count >= 0 for count in counts.values())
    assert [item.safe_message for item in manifest.items] == [
        "Modified since approval",
        "Unreadable file",
        "Removed from folder",
    ]


def test_action_state_disables_all_workspace_mutations_while_scanning():
    state = action_state(workspace(WorkspaceState.SCANNING), page(entry()))

    assert not state.can_rescan
    assert not state.can_edit_inclusion
    assert not state.can_approve
    assert not state.can_delete
    assert state.reason is not None and "scan" in state.reason.lower()


def test_action_state_requires_a_current_draft_before_approval():
    state = action_state(
        workspace(WorkspaceState.APPROVAL_REQUIRED, draft_revision=None), page(entry())
    )

    assert not state.can_approve
    assert state.reason is not None and "draft" in state.reason.lower()


def test_action_state_blocks_approval_for_changed_failed_or_removed_entries():
    for attention in (SourceState.CHANGED, SourceState.FAILED, SourceState.REMOVED):
        state = action_state(
            workspace(WorkspaceState.APPROVAL_REQUIRED), page(entry(attention))
        )

        assert not state.can_approve
        assert state.reason is not None and "attention" in state.reason.lower()


def test_action_state_allows_rescan_for_an_approved_workspace_but_not_approval():
    state = action_state(
        workspace(
            WorkspaceState.APPROVED,
            draft_revision="revision-full-1234567890",
            approved_revision="revision-full-1234567890",
        ),
        page(entry(SourceState.APPROVED)),
    )

    assert state.can_rescan
    assert not state.can_approve


class PagingClient:
    def __init__(self) -> None:
        self.offsets: list[int] = []

    def get_manifest(self, _workspace_id: str, *, offset: int = 0, limit: int = 500):
        self.offsets.append(offset)
        if offset == 0:
            return ManifestPage(
                items=tuple(entry(entry_id=f"first-{index}") for index in range(limit)),
                total=501,
                offset=0,
                limit=limit,
                counts={},
            )
        return ManifestPage(
            items=(
                entry(
                    entry_id="last",
                    relative_path="week-2/final.pdf",
                    proposed_course_group="week-2",
                ),
            ),
            total=501,
            offset=offset,
            limit=limit,
            counts={},
        )


def test_load_complete_manifest_reads_every_worker_page_for_review_and_approval():
    client = PagingClient()

    entries = load_complete_manifest(client, "workspace-1")

    assert len(entries) == 501
    assert entries[0].entry_id == "first-0"
    assert entries[-1].entry_id == "last"
    assert client.offsets == [0, 500]


def test_subtree_authority_uses_the_folder_entry_for_the_displayed_parent_prefix():
    folder = entry(entry_id="week-1-folder", relative_path="week-1", item_kind="folder")
    first = entry(entry_id="first-file", relative_path="week-1/notes.pdf")
    second = entry(entry_id="second-file", relative_path="week-1/slides.pdf")

    target = subtree_authority((folder, first, second), first)

    assert target == folder
    assert target.entry_id != first.entry_id


def test_delete_all_is_disabled_when_any_workspace_has_an_active_deletion_or_scan():
    assert can_delete_all_workspaces(
        [workspace(WorkspaceState.APPROVED), workspace(WorkspaceState.SCANNING)]
    ) is False
    assert can_delete_all_workspaces(
        [workspace(WorkspaceState.APPROVED), workspace(WorkspaceState.CLEANUP_PENDING)]
    ) is False
    assert can_delete_all_workspaces([workspace(WorkspaceState.APPROVED)]) is True


class InMemoryUpload(BytesIO):
    def __init__(self, name: str, content: bytes) -> None:
        super().__init__(content)
        self.name = name


class UploadClient:
    def __init__(self) -> None:
        self.files = None

    def upload_directory(self, _display_name: str, files, _idempotency_key: str):
        self.files = files
        return None


def test_upload_directory_forwards_uploaded_streams_without_materializing_escape_names(tmp_path):
    client = UploadClient()
    upload = InMemoryUpload("../outside.txt", b"notes")

    upload_directory(client, "Calculus", [upload], "upload-once")

    assert client.files == {"../outside.txt": upload}
    assert not (tmp_path / "outside.txt").exists()


class WidgetClient:
    def __init__(self, selected: WorkspaceDetail, manifest: ManifestPage) -> None:
        self.selected = selected
        self.manifest = manifest
        self.inclusion_requests = []
        self.approvals = []

    def list_workspaces(self):
        return [self.selected]

    def get_workspace(self, _workspace_id: str):
        return self.selected

    def get_manifest(self, _workspace_id: str, **_kwargs):
        return self.manifest

    def list_saved_providers(self):
        return [
            SavedProviderProfile(
                profile={"profile_id": "primary", "provider": "openai"},
                capabilities={"chat": True},
                credential_expected=True,
                reconnect_required=False,
                updated_at=NOW,
            )
        ]


class MultiPagePanelClient(WidgetClient):
    def get_manifest(self, _workspace_id: str, *, offset: int = 0, limit: int = 500, **_kwargs):
        if offset == 0:
            return ManifestPage(
                items=tuple(
                    entry(
                        entry_id=f"first-{index}",
                        relative_path=f"week-1/notes-{index}.pdf",
                    )
                    for index in range(limit)
                ),
                total=501,
                offset=0,
                limit=limit,
                counts={},
            )
        return ManifestPage(
            items=(
                entry(
                    entry_id="last",
                    relative_path="week-2/final.pdf",
                    proposed_course_group="week-2",
                ),
            ),
            total=501,
            offset=offset,
            limit=limit,
            counts={},
        )


class SubtreePanelClient(WidgetClient):
    def __init__(self) -> None:
        super().__init__(workspace(WorkspaceState.APPROVAL_REQUIRED), page())
        self.calls: list[tuple[str, object]] = []
        self.entries = (
            entry(entry_id="week-1-folder", relative_path="week-1", item_kind="folder", sha256=None),
            entry(entry_id="week-1-notes", relative_path="week-1/notes.pdf"),
            entry(entry_id="week-1-slides", relative_path="week-1/slides.pdf"),
        )

    def get_manifest(self, _workspace_id: str, **_kwargs):
        return ManifestPage(items=self.entries, total=3, offset=0, limit=500, counts={})

    def set_entry_inclusion(self, _workspace_id: str, entry_id: str, request):
        self.calls.append((entry_id, request))
        import streamlit as st

        st.session_state["subtree_call"] = (entry_id, request.subtree)


MULTIPAGE_SCRIPT = """
from exam_predictor.ui.workspace_view import render_workspace_panel
from exam_predictor.workspace.models import WorkspaceState
from tests.ui.test_workspace_view import MultiPagePanelClient, page, workspace
render_workspace_panel(MultiPagePanelClient(workspace(WorkspaceState.APPROVAL_REQUIRED), page()))
"""


SUBTREE_CLIENT = SubtreePanelClient()
SUBTREE_SCRIPT = """
from exam_predictor.ui.workspace_view import render_workspace_panel
from tests.ui.test_workspace_view import SUBTREE_CLIENT
render_workspace_panel(SUBTREE_CLIENT)
"""


WIDGET_SCRIPT = """
from exam_predictor.ui.workspace_view import render_workspace_panel
from exam_predictor.workspace.models import SourceState, WorkspaceState
from tests.ui.test_workspace_view import WidgetClient, entry, page, workspace
render_workspace_panel(WidgetClient(
    workspace(WorkspaceState.SCANNING),
    page(entry(SourceState.FAILED, reason="Unreadable file")),
))
"""


def test_workspace_panel_uses_public_fields_and_shows_action_reason():
    app = AppTest.from_string(WIDGET_SCRIPT).run()

    assert not app.exception
    visible = " ".join(
        [item.value for item in app.markdown]
        + [item.value for item in app.caption]
        + [item.value for item in app.info]
        + [item.value for item in app.warning]
    )
    assert "Choose course folder" in visible or any(
        item.label == "Choose course folder" for item in app.button
    )
    assert "C:/private/course-folder" not in visible
    assert "a" * 64 not in visible
    assert any(item.label == "Approve current draft" and item.disabled for item in app.button)


def test_workspace_panel_exposes_navigation_after_loading_multiple_worker_pages():
    app = AppTest.from_string(MULTIPAGE_SCRIPT).run()

    assert not app.exception
    assert any(item.label == "Manifest page" for item in app.selectbox)
    course_group = next(item for item in app.selectbox if item.label == "Filter course group")
    assert "week-2" in course_group.options
    assert any(
        item.label == "pending approval" and item.value == "501" for item in app.metric
    )
    assert not next(
        item for item in app.button if item.label == "Approve current draft"
    ).disabled


def test_workspace_panel_uses_folder_authority_for_subtree_inclusion():
    app = AppTest.from_string(SUBTREE_SCRIPT).run()

    next(item for item in app.button if item.label == "Apply inclusion to week-1").click()
    app.run()

    assert app.session_state["subtree_call"] == ("week-1-folder", True)


def test_workspace_panel_requires_delete_all_confirmation_and_all_workspaces_to_be_deletable():
    app = AppTest.from_string(WIDGET_SCRIPT).run()

    next(
        item
        for item in app.text_input
        if item.label == "Type DELETE ALL to delete every workspace"
    ).input("DELETE ALL")
    app.run()

    assert next(item for item in app.button if item.label == "Delete all workspaces").disabled
