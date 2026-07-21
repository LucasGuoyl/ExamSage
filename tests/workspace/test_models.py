import pytest
from pydantic import ValidationError

from exam_predictor.workspace.models import ArchiveMember, ManifestEntry, SourceState


def test_manifest_entry_rejects_paths_that_can_escape_the_grant():
    for unsafe in ("../secret.pdf", "/etc/passwd", "C:/secret.pdf", "a\\b.pdf", "a\x00b"):
        with pytest.raises(ValidationError):
            ManifestEntry(
                entry_id="entry-1",
                workspace_id="workspace-1",
                relative_path=unsafe,
                item_kind="file",
                format_category="pdf",
                size_bytes=4,
                modified_ns=1,
                sha256="0" * 64,
                state=SourceState.PENDING_APPROVAL,
                included=True,
            )


def test_source_state_and_course_group_are_independent():
    entry = ManifestEntry(
        entry_id="entry-1",
        workspace_id="workspace-1",
        relative_path="Week 1/notes.pdf",
        item_kind="file",
        format_category="pdf",
        size_bytes=4,
        modified_ns=1,
        sha256="0" * 64,
        state=SourceState.PENDING_APPROVAL,
        included=True,
        proposed_course_group="unclassified",
    )
    assert entry.state is SourceState.PENDING_APPROVAL
    assert entry.proposed_course_group == "unclassified"


def test_archive_member_display_path_strips_control_characters_and_is_bounded():
    member = ArchiveMember(
        parent_entry_id="entry-1",
        display_path="notes\x00\x7f.pdf" + "a" * 2_000,
        item_kind="file",
        size_bytes=4,
        compressed_bytes=2,
        state=SourceState.PENDING_APPROVAL,
    )
    assert "\x00" not in member.display_path
    assert "\x7f" not in member.display_path
    assert len(member.display_path) == 1_024
