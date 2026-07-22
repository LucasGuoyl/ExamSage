from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exam_predictor.workspace.models import (
    ManifestEntry,
    ScanProgress,
    ScanResult,
    SourceMode,
    SourceState,
    WorkspaceEvent,
    WorkspaceJob,
    WorkspaceJobStatus,
    WorkspaceRecord,
    WorkspaceState,
)
from exam_predictor.workspace.store import (
    ActiveWorkspaceOperationError,
    InvalidApprovalError,
    ManifestNotFoundError,
    StaleManifestError,
    WorkspaceJobNotFoundError,
    WorkspaceNotFoundError,
    WorkspaceStore,
)


NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
HASH_A = "a" * 64
HASH_B = "b" * 64


def workspace_record(
    tmp_path: Path,
    *,
    workspace_id: str = "workspace-1",
    updated_at: datetime = NOW,
) -> WorkspaceRecord:
    return WorkspaceRecord(
        workspace_id=workspace_id,
        display_name=f"Course {workspace_id}",
        source_mode=SourceMode.NATIVE_FOLDER,
        canonical_root=tmp_path / workspace_id,
        root_device="7",
        root_file_id="11",
        state=WorkspaceState.SCANNING,
        created_at=NOW,
        updated_at=updated_at,
    )


def entry(
    *,
    workspace_id: str = "workspace-1",
    entry_id: str = "entry-1",
    relative_path: str = "Module/notes.pdf",
    sha256: str | None = HASH_A,
    state: SourceState = SourceState.PENDING_APPROVAL,
    included: bool = True,
    archive_parent_entry_id: str | None = None,
    archive_member_path: str | None = None,
) -> ManifestEntry:
    return ManifestEntry(
        entry_id=entry_id,
        workspace_id=workspace_id,
        relative_path=relative_path,
        item_kind="archive_member" if archive_parent_entry_id else "file",
        format_category="pdf",
        size_bytes=10,
        modified_ns=123,
        device_id="7",
        file_id="11",
        sha256=sha256,
        state=state,
        included=included,
        proposed_course_group="Module",
        archive_parent_entry_id=archive_parent_entry_id,
        archive_member_path=archive_member_path,
    )


def job(
    *,
    job_id: str = "job-1",
    workspace_id: str = "workspace-1",
    status: WorkspaceJobStatus = WorkspaceJobStatus.QUEUED,
    idempotency_key: str = "request-1",
) -> WorkspaceJob:
    return WorkspaceJob(
        job_id=job_id,
        workspace_id=workspace_id,
        job_kind="scan",
        status=status,
        idempotency_key=idempotency_key,
        created_at=NOW,
    )


@pytest.fixture
def store(tmp_path: Path):
    value = WorkspaceStore(tmp_path / "workspace.sqlite3")
    try:
        yield value
    finally:
        value.close()


@pytest.fixture
def scanned_workspace(store: WorkspaceStore, tmp_path: Path) -> WorkspaceRecord:
    created = store.create_workspace(workspace_record(tmp_path))
    store.create_job(job(), "request-1")
    store.start_job("job-1")
    store.commit_scan(
        created.workspace_id,
        ScanResult(
            workspace_id=created.workspace_id,
            entries=(entry(), entry(entry_id="entry-2", relative_path="Module/problems.pdf")),
            discovered_count=2,
            bytes_hashed=20,
            failure_count=0,
            completed_at=NOW + timedelta(minutes=1),
        ),
        "job-1",
    )
    return store.get_workspace(created.workspace_id)  # type: ignore[return-value]


def test_schema_pragmas_and_workspace_round_trip(store: WorkspaceStore, tmp_path: Path):
    persisted = store.create_workspace(workspace_record(tmp_path))

    assert store.get_workspace(persisted.workspace_id) == persisted
    assert store.source_root(persisted.workspace_id) == (tmp_path / "workspace-1")
    assert store.get_manifest_entries(persisted.workspace_id) == ()
    with store._lock:
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert store._connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert store._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        tables = {
            row[0]
            for row in store._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {
        "workspaces",
        "manifest_revisions",
        "manifest_entries",
        "approvals",
        "workspace_jobs",
        "workspace_events",
        "cleanup_queue",
    } <= tables


def test_list_workspaces_orders_updates_and_counts_current_draft(
    store: WorkspaceStore, tmp_path: Path
):
    old = store.create_workspace(workspace_record(tmp_path, workspace_id="old"))
    store.create_workspace(
        workspace_record(tmp_path, workspace_id="new", updated_at=NOW + timedelta(hours=1))
    )

    summaries = store.list_workspaces()

    assert [item.workspace_id for item in summaries] == ["new", "old"]
    assert summaries[0].counts == {}
    assert old.canonical_root not in [getattr(item, "canonical_root", None) for item in summaries]


def test_commit_scan_publishes_immutable_revision_and_completion_event(
    store: WorkspaceStore, tmp_path: Path
):
    store.create_workspace(workspace_record(tmp_path))
    store.create_job(job(), "request-1")
    store.start_job("job-1")
    first_entries = (entry(),)
    first = store.commit_scan(
        "workspace-1",
        ScanResult(
            workspace_id="workspace-1",
            entries=first_entries,
            discovered_count=1,
            bytes_hashed=10,
            failure_count=0,
            completed_at=NOW + timedelta(minutes=1),
        ),
        "job-1",
    )

    stored = store.get_workspace("workspace-1")
    completed = store.get_job("job-1")
    events = store.list_job_events("job-1")
    assert stored is not None
    assert stored.state is WorkspaceState.APPROVAL_REQUIRED
    assert stored.current_draft_revision_id == first.revision_id
    assert stored.last_scanned_at == NOW + timedelta(minutes=1)
    assert completed.status is WorkspaceJobStatus.SUCCEEDED
    assert completed.finished_at == NOW + timedelta(minutes=1)
    assert [event.event_type for event in events] == ["started", "approval_required"]

    store.create_job(job(job_id="job-2", idempotency_key="request-2"), "request-2")
    store.start_job("job-2")
    second_entries = (
        entry(state=SourceState.CHANGED, sha256=HASH_B),
        entry(
            entry_id="entry-2",
            relative_path="Module/new.pdf",
            state=SourceState.PENDING_APPROVAL,
        ),
        entry(
            entry_id="entry-3",
            relative_path="Module/old.pdf",
            sha256=None,
            state=SourceState.REMOVED,
            included=False,
        ),
    )
    second = store.commit_scan(
        "workspace-1",
        ScanResult(
            workspace_id="workspace-1",
            entries=second_entries,
            discovered_count=2,
            bytes_hashed=20,
            failure_count=0,
            completed_at=NOW + timedelta(minutes=2),
        ),
        "job-2",
    )

    assert second.parent_revision_id == first.revision_id
    assert store.get_manifest("workspace-1", first.revision_id).entries == first_entries
    assert {
        item.entry_id: item
        for item in store.get_manifest("workspace-1", second.revision_id).entries
    } == {item.entry_id: item for item in second_entries}


def test_set_inclusion_clones_current_revision_and_expands_archive_subtree(
    store: WorkspaceStore, tmp_path: Path
):
    store.create_workspace(workspace_record(tmp_path))
    store.create_job(job(), "request-1")
    store.start_job("job-1")
    parent = entry(entry_id="archive", relative_path="bundle.zip")
    member = entry(
        entry_id="member",
        relative_path="bundle.zip",
        included=True,
        archive_parent_entry_id="archive",
        archive_member_path="inside.pdf",
    )
    original = store.commit_scan(
        "workspace-1",
        ScanResult(
            workspace_id="workspace-1",
            entries=(parent, member),
            discovered_count=2,
            bytes_hashed=10,
            failure_count=0,
            completed_at=NOW,
        ),
        "job-1",
    )

    clone = store.set_inclusion(
        "workspace-1", original.revision_id, ["archive", "member"], False
    )

    assert clone.parent_revision_id == original.revision_id
    assert [item.included for item in clone.entries] == [False, False]
    assert store.get_manifest("workspace-1", original.revision_id).entries[0].included is True
    with pytest.raises(StaleManifestError):
        store.set_inclusion("workspace-1", original.revision_id, ["archive"], True)
    with pytest.raises(ManifestNotFoundError):
        store.set_inclusion("workspace-1", clone.revision_id, ["missing"], True)


def test_approval_is_atomic_and_rejects_a_stale_revision(
    store: WorkspaceStore, scanned_workspace: WorkspaceRecord
):
    first = store.get_manifest(scanned_workspace.workspace_id)
    second = store.set_inclusion(
        scanned_workspace.workspace_id,
        first.revision_id,
        [first.entries[0].entry_id],
        False,
    )
    with pytest.raises(StaleManifestError):
        store.approve(scanned_workspace.workspace_id, first.revision_id, "workspace-v1")
    approval = store.approve(
        scanned_workspace.workspace_id, second.revision_id, "workspace-v1"
    )
    assert approval.revision_id == second.revision_id
    assert all(item.entry_id != first.entries[0].entry_id for item in approval.entries)
    assert store.get_approval(scanned_workspace.workspace_id) == approval
    assert store.get_workspace(scanned_workspace.workspace_id).state is WorkspaceState.APPROVED  # type: ignore[union-attr]
    assert store.get_manifest(scanned_workspace.workspace_id).entries[1].state is SourceState.APPROVED


def test_approval_rejects_unhashable_or_invalid_included_entries(
    store: WorkspaceStore, tmp_path: Path
):
    store.create_workspace(workspace_record(tmp_path))
    store.create_job(job(), "request-1")
    store.start_job("job-1")
    draft = store.commit_scan(
        "workspace-1",
        ScanResult(
            workspace_id="workspace-1",
            entries=(entry(sha256=None, state=SourceState.FAILED),),
            discovered_count=1,
            bytes_hashed=0,
            failure_count=1,
            completed_at=NOW,
        ),
        "job-1",
    )
    with store._transaction() as connection:
        connection.execute(
            "UPDATE manifest_entries SET included = 1 WHERE revision_id = ?",
            (draft.revision_id,),
        )

    with pytest.raises(InvalidApprovalError):
        store.approve("workspace-1", draft.revision_id, "workspace-v1")

    assert store.get_approval("workspace-1") is None
    assert store.get_workspace("workspace-1").state is WorkspaceState.APPROVAL_REQUIRED  # type: ignore[union-attr]


def test_idempotent_jobs_restart_recovery_and_event_cursor(
    tmp_path: Path,
):
    path = tmp_path / "workspace.sqlite3"
    first_store = WorkspaceStore(path)
    first_store.create_workspace(workspace_record(tmp_path))
    original = first_store.create_job(job(), "request-1")
    duplicate = first_store.create_job(
        job(job_id="different", idempotency_key="ignored"), "request-1"
    )
    started = first_store.start_job(original.job_id)
    progress = first_store.append_progress(
        original.job_id,
        ScanProgress(
            discovered_count=2,
            bytes_hashed=10,
            failure_count=0,
            current_relative_path="Module/notes.pdf",
        ),
    )
    first_store.close()

    restarted = WorkspaceStore(path)
    try:
        assert duplicate == original
        assert restarted.get_job(original.job_id) == started
        events = restarted.list_job_events(original.job_id)
        assert [event.sequence for event in events] == sorted(
            event.sequence for event in events
        )
        assert [event.event_type for event in events] == ["started", "scan_progress"]
        assert restarted.list_job_events(original.job_id, progress.sequence) == ()
    finally:
        restarted.close()


def test_progress_payload_is_bounded_relative_and_canonical_json(
    store: WorkspaceStore, tmp_path: Path
):
    store.create_workspace(workspace_record(tmp_path))
    store.create_job(job(), "request-1")
    event = store.append_progress(
        "job-1",
        ScanProgress(
            discovered_count=2,
            bytes_hashed=10,
            failure_count=1,
            current_relative_path="Module/notes.pdf",
        ),
    )
    assert event.payload["current_relative_path"] == "Module/notes.pdf"
    with store._lock:
        raw = store._connection.execute(
            "SELECT payload_json FROM workspace_events WHERE sequence = ?", (event.sequence,)
        ).fetchone()[0]
    assert raw == json.dumps(event.payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))

    for unsafe in ("../secret.pdf", "/etc/passwd", "C:/secret.pdf", "a\\b.pdf"):
        with pytest.raises(ValueError):
            store.append_progress(
                "job-1",
                ScanProgress(
                    discovered_count=2,
                    bytes_hashed=10,
                    failure_count=0,
                    current_relative_path=unsafe,
                ),
            )


def test_update_job_rolls_back_state_when_event_insert_fails(
    store: WorkspaceStore, tmp_path: Path
):
    store.create_workspace(workspace_record(tmp_path))
    queued = store.create_job(job(), "request-1")
    with store._transaction() as connection:
        connection.execute(
            """CREATE TRIGGER reject_failed_event
               BEFORE INSERT ON workspace_events
               WHEN NEW.event_type = 'failed'
               BEGIN SELECT RAISE(ABORT, 'forced event failure'); END"""
        )
    failed = queued.model_copy(
        update={
            "status": WorkspaceJobStatus.FAILED,
            "safe_error_code": "scan_failed",
            "finished_at": NOW + timedelta(minutes=1),
        }
    )
    event_value = WorkspaceEvent(
        sequence=1,
        job_id=queued.job_id,
        event_type="failed",
        message="Scan failed.",
        payload={"safe_error_code": "scan_failed"},
        created_at=NOW + timedelta(minutes=1),
    )

    with pytest.raises(sqlite3.IntegrityError, match="forced event failure"):
        store.update_job(failed, event_value)

    assert store.get_job(queued.job_id) == queued
    assert store.list_job_events(queued.job_id) == ()


def test_fail_job_is_atomic_and_sets_workspace_attention(
    store: WorkspaceStore, tmp_path: Path
):
    store.create_workspace(workspace_record(tmp_path))
    store.create_job(job(), "request-1")
    store.start_job("job-1")

    failed = store.fail_job("job-1", "source_root_unreadable")

    assert failed.status is WorkspaceJobStatus.FAILED
    assert failed.safe_error_code == "source_root_unreadable"
    assert failed.finished_at is not None
    assert store.get_workspace("workspace-1").state is WorkspaceState.NEEDS_ATTENTION  # type: ignore[union-attr]
    assert store.list_job_events("job-1")[-1].event_type == "failed"


def test_mark_changed_clones_draft_and_access_verification_persists(
    store: WorkspaceStore, scanned_workspace: WorkspaceRecord
):
    original = store.get_manifest(scanned_workspace.workspace_id)
    verified_at = NOW + timedelta(hours=2)

    store.mark_entry_changed(
        scanned_workspace.workspace_id, original.entries[0].entry_id, "source_changed"
    )
    store.record_access_verified(scanned_workspace.workspace_id, verified_at)

    changed = store.get_manifest(scanned_workspace.workspace_id)
    assert changed.parent_revision_id == original.revision_id
    assert changed.entries[0].state is SourceState.CHANGED
    assert changed.entries[0].included is False
    assert changed.entries[0].failure_code == "source_changed"
    assert store.get_workspace(scanned_workspace.workspace_id).state is WorkspaceState.NEEDS_ATTENTION  # type: ignore[union-attr]
    assert store.get_workspace(scanned_workspace.workspace_id).last_access_verified_at == verified_at  # type: ignore[union-attr]


def test_deletion_blocks_active_jobs_then_cascades_only_rows(
    store: WorkspaceStore, scanned_workspace: WorkspaceRecord
):
    store.create_job(job(job_id="job-2", idempotency_key="request-2"), "request-2")
    with pytest.raises(ActiveWorkspaceOperationError):
        store.mark_deleting(scanned_workspace.workspace_id)

    store.fail_job("job-2", "cancelled")
    deleting = store.mark_deleting(scanned_workspace.workspace_id)
    assert deleting.state is WorkspaceState.DELETING
    store.delete_workspace_rows(scanned_workspace.workspace_id)
    assert store.get_workspace(scanned_workspace.workspace_id) is None
    assert store.list_job_events("job-1") == ()


def test_deletion_rejects_an_unsettled_scanning_workspace(
    store: WorkspaceStore, tmp_path: Path
):
    store.create_workspace(workspace_record(tmp_path))

    with pytest.raises(ActiveWorkspaceOperationError):
        store.mark_deleting("workspace-1")

    assert store.get_workspace("workspace-1").state is WorkspaceState.SCANNING  # type: ignore[union-attr]


def test_cleanup_queue_rejects_native_or_escaping_paths_and_survives_restart(
    tmp_path: Path,
):
    path = tmp_path / "workspace.sqlite3"
    first = WorkspaceStore(path)
    source = workspace_record(tmp_path)
    first.create_workspace(source)
    for unsafe in (source.canonical_root, Path("../escape"), Path("/absolute"), Path("a/../../b")):
        with pytest.raises(ValueError):
            first.queue_cleanup(source.workspace_id, unsafe, "cleanup_failed")
    first.queue_cleanup(
        source.workspace_id,
        tmp_path / "workspaces/workspace-1/browser-intake",
        "cleanup_failed",
    )
    first.delete_workspace_rows(source.workspace_id)
    first.close()

    restarted = WorkspaceStore(path)
    try:
        records = restarted.list_cleanup()
        assert len(records) == 1
        assert records[0].workspace_id == source.workspace_id
        assert records[0].owned_relative_path == "workspaces/workspace-1/browser-intake"
        assert source.canonical_root.as_posix() not in path.read_bytes().decode(
            "utf-8", errors="ignore"
        ).split("cleanup_failed")[-1]
    finally:
        restarted.close()


def test_missing_workspace_manifest_and_job_errors_are_safe(store: WorkspaceStore):
    assert store.get_workspace("missing") is None
    assert store.get_approval("missing") is None
    with pytest.raises(WorkspaceNotFoundError, match="missing"):
        store.source_root("missing")
    with pytest.raises(WorkspaceNotFoundError, match="missing"):
        store.get_manifest_entries("missing")
    with pytest.raises(WorkspaceNotFoundError, match="missing"):
        store.get_manifest("missing")
    with pytest.raises(WorkspaceJobNotFoundError, match="missing"):
        store.get_job("missing")
    assert store.list_job_events("missing") == ()


def test_close_is_explicit_and_idempotent(tmp_path: Path):
    store = WorkspaceStore(tmp_path / "workspace.sqlite3")
    connection = store._connection

    store.close()
    store.close()

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
