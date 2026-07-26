from __future__ import annotations

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
    manifest_counts,
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
    included: bool = True,
    sha256: str | None = "a" * 64,
    reason: str | None = None,
) -> ManifestEntry:
    return ManifestEntry(
        entry_id="entry-full-1234567890",
        workspace_id="workspace-12345678901234567890123456789012",
        relative_path="week-1/notes.pdf",
        item_kind="file",
        format_category="pdf",
        size_bytes=12,
        modified_ns=1,
        sha256=sha256,
        state=state,
        included=included,
        proposed_course_group="week-1",
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
