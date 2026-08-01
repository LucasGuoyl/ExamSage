from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier, Event

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
    TransmissionAuthorityReentrancyError,
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
    archive_member_index: int | None = None,
    archive_member_crc32: int | None = None,
    archive_member_compressed_bytes: int | None = None,
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
        archive_member_index=archive_member_index,
        archive_member_crc32=archive_member_crc32,
        archive_member_compressed_bytes=archive_member_compressed_bytes,
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
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 3
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


def test_commit_scan_replay_returns_existing_revision_without_duplicate_event(
    store: WorkspaceStore, tmp_path: Path
):
    store.create_workspace(workspace_record(tmp_path))
    store.create_job(job(), "request-1")
    store.start_job("job-1")
    result = ScanResult(
        workspace_id="workspace-1",
        entries=(entry(),),
        discovered_count=1,
        bytes_hashed=10,
        failure_count=0,
        completed_at=NOW + timedelta(minutes=1),
    )

    committed = store.commit_scan("workspace-1", result, "job-1")
    events = store.list_job_events("job-1")
    replayed = store.commit_scan("workspace-1", result, "job-1")

    assert replayed == committed
    assert store.list_job_events("job-1") == events
    with store._lock:
        count = store._connection.execute(
            "SELECT COUNT(*) FROM manifest_revisions WHERE scan_job_id = 'job-1'"
        ).fetchone()[0]
    assert count == 1


def test_commit_scan_requires_running_and_active_authority_blocks_a_newer_scan(
    store: WorkspaceStore, tmp_path: Path
):
    store.create_workspace(workspace_record(tmp_path))
    older = store.create_job(job(), "request-1")
    queued_result = ScanResult(
        workspace_id="workspace-1",
        entries=(entry(),),
        discovered_count=1,
        bytes_hashed=10,
        failure_count=0,
        completed_at=NOW + timedelta(minutes=3),
    )
    with pytest.raises(ActiveWorkspaceOperationError):
        store.commit_scan("workspace-1", queued_result, older.job_id)

    store.start_job(older.job_id)
    with pytest.raises(ActiveWorkspaceOperationError):
        store.create_job(
            job(
                job_id="job-2",
                idempotency_key="request-2",
            ).model_copy(update={"created_at": NOW + timedelta(minutes=1)}),
            "request-2",
        )

    committed = store.commit_scan("workspace-1", queued_result, older.job_id)

    assert store.get_manifest("workspace-1") == committed
    assert store.get_job(older.job_id).status is WorkspaceJobStatus.SUCCEEDED


def test_reopen_migrates_unique_scan_job_revision_index(tmp_path: Path):
    path = tmp_path / "workspace.sqlite3"
    original = WorkspaceStore(path)
    with original._transaction() as connection:
        connection.execute("DROP INDEX IF EXISTS idx_manifest_revisions_scan_job")
    original.close()

    reopened = WorkspaceStore(path)
    try:
        with reopened._lock:
            indexes = {
                row["name"]: bool(row["unique"])
                for row in reopened._connection.execute(
                    "PRAGMA index_list('manifest_revisions')"
                )
            }
        assert indexes["idx_manifest_revisions_scan_job"] is True
    finally:
        reopened.close()


def test_unique_scan_job_index_migration_repairs_legacy_replay_duplicates(
    tmp_path: Path,
):
    path = tmp_path / "workspace.sqlite3"
    original = WorkspaceStore(path)
    ready = workspace_record(tmp_path).model_copy(update={"state": WorkspaceState.READY})
    original.create_workspace(ready)
    original.create_job(job(status=WorkspaceJobStatus.SUCCEEDED), "request-1")
    with original._transaction() as connection:
        connection.execute("DROP INDEX idx_manifest_revisions_scan_job")
        for revision_id, created_at in (
            ("legacy-first", NOW),
            ("legacy-current", NOW + timedelta(minutes=1)),
        ):
            connection.execute(
                """INSERT INTO manifest_revisions(
                       revision_id, workspace_id, parent_revision_id, scan_job_id,
                       policy_version, created_at
                   ) VALUES (?, ?, NULL, 'job-1', 'workspace-v1', ?)""",
                (revision_id, ready.workspace_id, created_at.isoformat()),
            )
        connection.execute(
            """UPDATE workspaces SET current_draft_revision_id = 'legacy-current'
               WHERE workspace_id = ?""",
            (ready.workspace_id,),
        )
    original.close()

    reopened = WorkspaceStore(path)
    try:
        with reopened._lock:
            rows = reopened._connection.execute(
                """SELECT revision_id FROM manifest_revisions
                   WHERE scan_job_id = 'job-1'"""
            ).fetchall()
        assert [row["revision_id"] for row in rows] == ["legacy-current"]
        assert reopened.get_manifest(ready.workspace_id).revision_id == "legacy-current"
    finally:
        reopened.close()


def test_v2_migration_adds_and_round_trips_exact_archive_member_identity(tmp_path: Path):
    path = tmp_path / "workspace.sqlite3"
    original = WorkspaceStore(path)
    original.close()
    connection = sqlite3.connect(path)
    try:
        connection.execute("ALTER TABLE manifest_entries DROP COLUMN archive_member_index")
        connection.execute("ALTER TABLE manifest_entries DROP COLUMN archive_member_crc32")
        connection.execute(
            "ALTER TABLE manifest_entries DROP COLUMN archive_member_compressed_bytes"
        )
        connection.execute("PRAGMA user_version=2")
        connection.commit()
    finally:
        connection.close()

    migrated = WorkspaceStore(path)
    try:
        migrated.create_workspace(workspace_record(tmp_path))
        migrated.create_job(job(), "request-1")
        migrated.start_job("job-1")
        parent = entry(entry_id="archive", relative_path="bundle.zip")
        member = entry(
            entry_id="member",
            relative_path="bundle.zip",
            archive_parent_entry_id="archive",
            archive_member_path="inside.pdf",
            archive_member_index=2,
            archive_member_crc32=0x1234ABCD,
            archive_member_compressed_bytes=7,
        )
        migrated.commit_scan(
            "workspace-1",
            ScanResult(
                workspace_id="workspace-1",
                entries=(parent, member),
                discovered_count=2,
                bytes_hashed=20,
                failure_count=0,
                completed_at=NOW,
            ),
            "job-1",
        )
    finally:
        migrated.close()

    reopened = WorkspaceStore(path)
    try:
        persisted = next(
            item
            for item in reopened.get_manifest_entries("workspace-1")
            if item.archive_member_path == "inside.pdf"
        )
        assert (
            persisted.archive_member_index,
            persisted.archive_member_crc32,
            persisted.archive_member_compressed_bytes,
        ) == (2, 0x1234ABCD, 7)
    finally:
        reopened.close()


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
    assert [item.state for item in clone.entries] == [
        SourceState.EXCLUDED,
        SourceState.EXCLUDED,
    ]
    assert [item.inclusion_reason for item in clone.entries] == [
        "user_excluded",
        "user_excluded",
    ]
    assert store.get_manifest("workspace-1", original.revision_id).entries[0].included is True
    with pytest.raises(StaleManifestError):
        store.set_inclusion("workspace-1", original.revision_id, ["archive"], True)
    with pytest.raises(ManifestNotFoundError):
        store.set_inclusion("workspace-1", clone.revision_id, ["missing"], True)

    restored = store.set_inclusion(
        "workspace-1", clone.revision_id, ["archive"], True
    )

    assert restored.entries[0].included is True
    assert restored.entries[0].state is SourceState.PENDING_APPROVAL
    assert restored.entries[0].inclusion_reason is None


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


def test_recover_running_jobs_is_restart_safe_ordered_and_idempotent(tmp_path: Path):
    path = tmp_path / "workspace.sqlite3"
    original = WorkspaceStore(path)
    original.create_workspace(workspace_record(tmp_path))
    original.create_workspace(
        workspace_record(tmp_path, workspace_id="workspace-2")
    )
    running_jobs = [
        job(
            job_id=f"job-{index}",
            workspace_id=f"workspace-{index}",
            status=WorkspaceJobStatus.RUNNING,
        ).model_copy(
            update={
                "idempotency_key": f"request-{index}",
                "safe_error_code": "interrupted",
                "started_at": NOW + timedelta(minutes=1),
            }
        )
        for index in (1, 2)
    ]
    for item in running_jobs:
        original.create_job(item, item.idempotency_key)
    settled = job(
        job_id="job-3",
        status=WorkspaceJobStatus.SUCCEEDED,
        idempotency_key="request-3",
    ).model_copy(update={"finished_at": NOW + timedelta(minutes=2)})
    original.create_job(settled, settled.idempotency_key)
    original.close()

    restarted = WorkspaceStore(path)
    try:
        recovered = restarted.recover_running_jobs()

        assert [item.job_id for item in recovered] == ["job-1", "job-2"]
        assert all(item.status is WorkspaceJobStatus.QUEUED for item in recovered)
        assert all(item.started_at is None for item in recovered)
        assert all(item.finished_at is None for item in recovered)
        assert all(item.safe_error_code is None for item in recovered)
        for item in recovered:
            events = restarted.list_job_events(item.job_id)
            assert len(events) == 1
            assert events[0].event_type == "queued"
            assert events[0].message == "Workspace operation queued."
            assert events[0].payload == {}
        assert restarted.get_job(settled.job_id) == settled

        assert restarted.recover_running_jobs() == ()
        assert all(
            len(restarted.list_job_events(item.job_id)) == 1 for item in recovered
        )
    finally:
        restarted.close()


def test_generic_update_does_not_allow_running_to_queued_recovery_transition(
    store: WorkspaceStore, tmp_path: Path
):
    store.create_workspace(workspace_record(tmp_path))
    running = store.create_job(
        job(status=WorkspaceJobStatus.RUNNING).model_copy(
            update={"started_at": NOW + timedelta(minutes=1)}
        ),
        "request-1",
    )
    queued = running.model_copy(
        update={
            "status": WorkspaceJobStatus.QUEUED,
            "started_at": None,
            "finished_at": None,
        }
    )
    event_value = WorkspaceEvent(
        sequence=1,
        job_id=running.job_id,
        event_type="queued",
        message="Workspace operation queued.",
        payload={},
        created_at=NOW + timedelta(minutes=2),
    )

    with pytest.raises(ActiveWorkspaceOperationError):
        store.update_job(queued, event_value)

    assert store.get_job(running.job_id) == running
    assert store.list_job_events(running.job_id) == ()


def test_progress_payload_is_bounded_relative_and_canonical_json(
    store: WorkspaceStore, tmp_path: Path
):
    store.create_workspace(workspace_record(tmp_path))
    store.create_job(job(), "request-1")
    store.start_job("job-1")
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


@pytest.mark.parametrize(
    "status",
    [
        WorkspaceJobStatus.QUEUED,
        WorkspaceJobStatus.SUCCEEDED,
        WorkspaceJobStatus.FAILED,
        WorkspaceJobStatus.CANCELLED,
    ],
)
def test_progress_rejects_non_running_jobs_without_an_event(
    store: WorkspaceStore, tmp_path: Path, status: WorkspaceJobStatus
):
    store.create_workspace(workspace_record(tmp_path))
    value = job(status=status)
    if status in {
        WorkspaceJobStatus.SUCCEEDED,
        WorkspaceJobStatus.FAILED,
        WorkspaceJobStatus.CANCELLED,
    }:
        value = value.model_copy(update={"finished_at": NOW + timedelta(minutes=1)})
    stored = store.create_job(value, "request-1")

    with pytest.raises(ActiveWorkspaceOperationError):
        store.append_progress(
            stored.job_id,
            ScanProgress(
                discovered_count=1,
                bytes_hashed=10,
                failure_count=0,
                current_relative_path="Module/notes.pdf",
            ),
        )

    assert store.get_job(stored.job_id) == stored
    assert store.list_job_events(stored.job_id) == ()


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
        message="Workspace scan failed.",
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


def test_progress_events_have_a_durable_per_job_cap(
    store: WorkspaceStore, tmp_path: Path
):
    store.create_workspace(workspace_record(tmp_path))
    store.create_job(job(), "request-1")
    store.start_job("job-1")

    for index in range(300):
        store.append_progress(
            "job-1",
            ScanProgress(
                discovered_count=index + 1,
                bytes_hashed=index,
                failure_count=0,
                current_relative_path=f"{index}.txt",
            ),
        )

    progress = [
        event
        for event in store.list_job_events("job-1")
        if event.event_type == "scan_progress"
    ]
    assert len(progress) == 256


@pytest.mark.parametrize(
    "terminal_status",
    [
        WorkspaceJobStatus.SUCCEEDED,
        WorkspaceJobStatus.FAILED,
        WorkspaceJobStatus.CANCELLED,
    ],
)
def test_fail_job_rejects_terminal_rewrites(
    store: WorkspaceStore, tmp_path: Path, terminal_status: WorkspaceJobStatus
):
    store.create_workspace(workspace_record(tmp_path))
    terminal = job(status=terminal_status).model_copy(
        update={"finished_at": NOW + timedelta(minutes=1)}
    )
    stored = store.create_job(terminal, "request-1")

    with pytest.raises(ActiveWorkspaceOperationError):
        store.fail_job(stored.job_id, "late_failure")

    assert store.get_job(stored.job_id) == stored
    assert store.list_job_events(stored.job_id) == ()


def test_update_job_rejects_identity_rewrites_and_invalid_transitions(
    store: WorkspaceStore, tmp_path: Path
):
    queued = store.create_workspace(workspace_record(tmp_path))
    store.create_workspace(workspace_record(tmp_path, workspace_id="workspace-2"))
    original = store.create_job(job(workspace_id=queued.workspace_id), "request-1")
    event_value = WorkspaceEvent(
        sequence=1,
        job_id=original.job_id,
        event_type="started",
        message="Workspace scan started.",
        payload={},
        created_at=NOW + timedelta(minutes=1),
    )

    rewritten = original.model_copy(
        update={
            "workspace_id": "workspace-2",
            "status": WorkspaceJobStatus.RUNNING,
            "started_at": NOW + timedelta(minutes=1),
        }
    )
    with pytest.raises(ValueError, match="identity"):
        store.update_job(rewritten, event_value)

    skipped = original.model_copy(
        update={
            "status": WorkspaceJobStatus.SUCCEEDED,
            "finished_at": NOW + timedelta(minutes=1),
        }
    )
    succeeded_event = event_value.model_copy(
        update={
            "event_type": "succeeded",
            "message": "Workspace operation completed.",
        }
    )
    with pytest.raises(ActiveWorkspaceOperationError, match="transition"):
        store.update_job(skipped, succeeded_event)

    assert store.get_job(original.job_id) == original
    assert store.list_job_events(original.job_id) == ()


@pytest.mark.parametrize(
    ("event_type", "message", "payload"),
    [
        ("unknown", "Workspace scan started.", {}),
        ("started", "Provider raised RuntimeError: api_key=secret", {}),
        ("started", "Workspace scan started.", {"unknown": "value"}),
        ("started", "Workspace scan started.", {"api_key": "sk-secret"}),
        (
            "started",
            "Workspace scan started.",
            {"current_relative_path": "C:/Users/private/notes.pdf"},
        ),
    ],
)
def test_update_job_rejects_unsafe_lifecycle_events_without_writes(
    store: WorkspaceStore,
    tmp_path: Path,
    event_type: str,
    message: str,
    payload: dict[str, str],
):
    store.create_workspace(workspace_record(tmp_path))
    original = store.create_job(job(), "request-1")
    running = original.model_copy(
        update={
            "status": WorkspaceJobStatus.RUNNING,
            "started_at": NOW + timedelta(minutes=1),
        }
    )
    unsafe_event = WorkspaceEvent(
        sequence=1,
        job_id=original.job_id,
        event_type=event_type,
        message=message,
        payload=payload,
        created_at=NOW + timedelta(minutes=1),
    )

    with pytest.raises(ValueError):
        store.update_job(running, unsafe_event)

    assert store.get_job(original.job_id) == original
    assert store.list_job_events(original.job_id) == ()


def test_update_job_allows_a_safe_declared_transition_and_event(
    store: WorkspaceStore, tmp_path: Path
):
    store.create_workspace(workspace_record(tmp_path))
    original = store.create_job(job(), "request-1")
    started_at = NOW + timedelta(minutes=1)
    running = original.model_copy(
        update={"status": WorkspaceJobStatus.RUNNING, "started_at": started_at}
    )
    started = WorkspaceEvent(
        sequence=999,
        job_id=original.job_id,
        event_type="started",
        message="Workspace scan started.",
        payload={},
        created_at=started_at,
    )

    store.update_job(running, started)

    assert store.get_job(original.job_id) == running
    persisted = store.list_job_events(original.job_id)
    assert len(persisted) == 1
    assert persisted[0].event_type == "started"


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


def test_deletion_state_blocks_job_creation_and_start_and_guards_row_delete(
    store: WorkspaceStore, tmp_path: Path
):
    ready = workspace_record(tmp_path).model_copy(update={"state": WorkspaceState.READY})
    store.create_workspace(ready)
    with pytest.raises(ActiveWorkspaceOperationError):
        store.delete_workspace_rows(ready.workspace_id)

    deleting = store.mark_deleting(ready.workspace_id)
    with pytest.raises(ActiveWorkspaceOperationError):
        store.create_job(job(), "request-1")

    with store._transaction() as connection:
        connection.execute(
            """INSERT INTO workspace_jobs(
                   job_id, workspace_id, job_kind, status, idempotency_key,
                   safe_error_code, created_at, started_at, finished_at
               ) VALUES ('injected', ?, 'scan', 'queued', 'injected', NULL, ?, NULL, NULL)""",
            (ready.workspace_id, NOW.isoformat()),
        )
    with pytest.raises(ActiveWorkspaceOperationError):
        store.start_job("injected")
    with pytest.raises(ActiveWorkspaceOperationError):
        store.delete_workspace_rows(ready.workspace_id)

    with store._transaction() as connection:
        connection.execute(
            "UPDATE workspace_jobs SET status = 'cancelled' WHERE job_id = 'injected'"
        )
    store.delete_workspace_rows(deleting.workspace_id)
    assert store.get_workspace(deleting.workspace_id) is None


def test_cleanup_pending_state_blocks_job_creation_and_start(
    store: WorkspaceStore, tmp_path: Path
):
    ready = workspace_record(tmp_path).model_copy(update={"state": WorkspaceState.READY})
    store.create_workspace(ready)
    queued = store.create_job(job(), "request-1")
    with store._transaction() as connection:
        connection.execute(
            "UPDATE workspaces SET state = 'cleanup_pending' WHERE workspace_id = ?",
            (ready.workspace_id,),
        )

    with pytest.raises(ActiveWorkspaceOperationError):
        store.create_job(job(job_id="job-2", idempotency_key="request-2"), "request-2")
    with pytest.raises(ActiveWorkspaceOperationError):
        store.start_job(queued.job_id)


def test_mark_deleting_and_create_job_are_serialized_across_connections(tmp_path: Path):
    path = tmp_path / "workspace.sqlite3"
    first = WorkspaceStore(path)
    second = WorkspaceStore(path)
    ready = workspace_record(tmp_path).model_copy(update={"state": WorkspaceState.READY})
    first.create_workspace(ready)
    barrier = Barrier(2)

    def mark() -> str:
        barrier.wait()
        try:
            first.mark_deleting(ready.workspace_id)
        except ActiveWorkspaceOperationError:
            return "blocked"
        return "deleted"

    def create() -> str:
        barrier.wait()
        try:
            second.create_job(job(), "request-1")
        except ActiveWorkspaceOperationError:
            return "blocked"
        return "created"

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = {executor.submit(mark), executor.submit(create)}
            results = {future.result() for future in outcomes}
        assert results in ({"deleted", "blocked"}, {"created", "blocked"})
        workspace = first.get_workspace(ready.workspace_id)
        if "deleted" in results:
            assert workspace is not None and workspace.state is WorkspaceState.DELETING
            assert first.list_job_events("job-1") == ()
        else:
            assert first.get_job("job-1").status is WorkspaceJobStatus.QUEUED
            with pytest.raises(ActiveWorkspaceOperationError):
                first.delete_workspace_rows(ready.workspace_id)
    finally:
        first.close()
        second.close()


def test_cleanup_queue_rejects_native_or_escaping_paths_and_survives_restart(
    tmp_path: Path,
):
    path = tmp_path / "workspace.sqlite3"
    first = WorkspaceStore(path)
    source = workspace_record(tmp_path).model_copy(update={"state": WorkspaceState.READY})
    first.create_workspace(source)
    for unsafe in (source.canonical_root, Path("../escape"), Path("/absolute"), Path("a/../../b")):
        with pytest.raises(ValueError):
            first.queue_cleanup(source.workspace_id, unsafe, "cleanup_failed")
    first.queue_cleanup(
        source.workspace_id,
        tmp_path / "workspaces/workspace-1/browser-intake",
        "cleanup_failed",
    )
    first.mark_deleting(source.workspace_id)
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


def test_cleanup_failure_and_completion_are_atomic_and_completion_is_idempotent(
    store: WorkspaceStore, tmp_path: Path
):
    owned_root = tmp_path / "workspaces" / "workspace-1"
    owned_root.mkdir(parents=True)
    source = workspace_record(tmp_path).model_copy(
        update={
            "source_mode": SourceMode.BROWSER_SNAPSHOT,
            "canonical_root": owned_root / "browser-intake",
            "state": WorkspaceState.READY,
        }
    )
    store.create_workspace(source)
    store.mark_deleting(source.workspace_id)
    store.queue_cleanup(source.workspace_id, owned_root, "cleanup_failed")
    [record] = store.list_cleanup()

    failed = store.fail_cleanup(record.cleanup_id, "cleanup_retry_failed")

    assert failed is not None
    assert failed.attempt_count == 1
    assert failed.safe_error_code == "cleanup_retry_failed"
    workspace = store.get_workspace(source.workspace_id)
    assert workspace is not None
    assert workspace.state is WorkspaceState.CLEANUP_PENDING

    with store._transaction() as connection:
        connection.execute(
            """CREATE TRIGGER reject_cleanup_workspace_delete
               BEFORE DELETE ON workspaces
               BEGIN SELECT RAISE(ABORT, 'forced cleanup failure'); END"""
        )
    with pytest.raises(sqlite3.IntegrityError, match="forced cleanup failure"):
        store.complete_cleanup(record.cleanup_id)
    assert store.get_workspace(source.workspace_id) is not None
    assert store.list_cleanup() == (failed,)
    with store._transaction() as connection:
        connection.execute("DROP TRIGGER reject_cleanup_workspace_delete")

    assert store.complete_cleanup(record.cleanup_id) == source.workspace_id
    assert store.get_workspace(source.workspace_id) is None
    assert store.list_cleanup() == ()
    assert store.complete_cleanup(record.cleanup_id) is None


def test_initial_workspace_and_job_are_atomic_and_durably_idempotent(
    store: WorkspaceStore, tmp_path: Path
):
    first_workspace = workspace_record(tmp_path).model_copy(
        update={"state": WorkspaceState.READY}
    )
    first_job = job().model_copy(update={"job_kind": "initial_scan"})
    second_workspace = workspace_record(
        tmp_path, workspace_id="workspace-2"
    ).model_copy(update={"state": WorkspaceState.READY})
    second_job = job(
        job_id="job-2",
        workspace_id="workspace-2",
        idempotency_key="request-1",
    ).model_copy(update={"job_kind": "initial_scan"})
    request, claimed = store.claim_creation(
        "request-1",
        SourceMode.BROWSER_SNAPSHOT,
        first_workspace.workspace_id,
    )
    assert claimed is True

    created_workspace, created_job, created = store.create_workspace_with_initial_job(
        first_workspace,
        first_job,
        "request-1",
        owned_root_identity=("7", "22"),
    )
    duplicate_workspace, duplicate_job, duplicate_created = (
        store.create_workspace_with_initial_job(
            second_workspace,
            second_job,
            "request-1",
            owned_root_identity=("8", "33"),
        )
    )

    assert created is True
    assert duplicate_created is False
    assert duplicate_workspace == created_workspace
    assert duplicate_job == created_job
    assert [item.workspace_id for item in store.list_workspaces()] == [
        first_workspace.workspace_id
    ]
    assert store.get_owned_root_identity(first_workspace.workspace_id) == ("7", "22")
    assert store.get_creation_job("request-1") == created_job


def test_database_rejects_two_active_scans_for_one_workspace(tmp_path: Path):
    path = tmp_path / "workspace.sqlite3"
    first = WorkspaceStore(path)
    second = WorkspaceStore(path)
    try:
        ready = workspace_record(tmp_path).model_copy(update={"state": WorkspaceState.READY})
        first.create_workspace(ready)
        barrier = Barrier(2)

        def create_active(target: WorkspaceStore, value: WorkspaceJob) -> str:
            barrier.wait()
            try:
                target.create_job(value, value.idempotency_key)
            except ActiveWorkspaceOperationError:
                return "blocked"
            return "created"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = {
                future.result()
                for future in (
                    pool.submit(create_active, first, job()),
                    pool.submit(
                        create_active,
                        second,
                        job(job_id="job-2", idempotency_key="request-2"),
                    ),
                )
            }
        assert outcomes == {"blocked", "created"}
        with first._lock:
            active_count = first._connection.execute(
                """SELECT COUNT(*) FROM workspace_jobs
                   WHERE workspace_id = ? AND status IN ('queued', 'running')""",
                (ready.workspace_id,),
            ).fetchone()[0]
        assert active_count == 1
    finally:
        first.close()
        second.close()


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


def test_transmission_authority_lock_is_shared_across_store_instances(
    store: WorkspaceStore,
    scanned_workspace: WorkspaceRecord,
    tmp_path: Path,
):
    manifest = store.get_manifest(scanned_workspace.workspace_id)
    approval = store.approve(
        scanned_workspace.workspace_id,
        manifest.revision_id,
        manifest.policy_version,
    )
    second = WorkspaceStore(tmp_path / "workspace.sqlite3")
    mutation_started = Event()

    def revoke_from_second_store():
        mutation_started.set()
        return second.set_inclusion(
            scanned_workspace.workspace_id,
            manifest.revision_id,
            (manifest.entries[0].entry_id,),
            False,
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with store.hold_transmission_authority(
                scanned_workspace.workspace_id,
                approval_id=approval.approval_id,
                revision_id=manifest.revision_id,
            ):
                mutation = pool.submit(revoke_from_second_store)
                assert mutation_started.wait(timeout=5)
                assert not mutation.done()
            changed = mutation.result(timeout=5)
        assert changed.revision_id != manifest.revision_id
    finally:
        second.close()


def test_cleanup_state_mutation_waits_for_transmission_authority(
    store: WorkspaceStore,
    scanned_workspace: WorkspaceRecord,
    tmp_path: Path,
):
    manifest = store.get_manifest(scanned_workspace.workspace_id)
    approval = store.approve(
        scanned_workspace.workspace_id,
        manifest.revision_id,
        manifest.policy_version,
    )
    owned_root = tmp_path / "workspaces" / scanned_workspace.workspace_id
    owned_root.mkdir(parents=True)
    cleanup = store.queue_cleanup(
        scanned_workspace.workspace_id,
        owned_root,
        "cleanup_failed",
    )
    second = WorkspaceStore(tmp_path / "workspace.sqlite3")
    cleanup_started = Event()

    def fail_from_second_store():
        cleanup_started.set()
        return second.fail_cleanup(cleanup.cleanup_id, "cleanup_retry_failed")

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with store.hold_transmission_authority(
                scanned_workspace.workspace_id,
                approval_id=approval.approval_id,
                revision_id=manifest.revision_id,
            ):
                failed = pool.submit(fail_from_second_store)
                assert cleanup_started.wait(timeout=5)
                assert not failed.done()
            record = failed.result(timeout=5)
        assert record is not None
        assert record.safe_error_code == "cleanup_retry_failed"
    finally:
        second.close()


@pytest.mark.parametrize("use_hardlink_alias", [False, True])
def test_transmission_authority_lock_is_shared_across_processes(
    store: WorkspaceStore,
    scanned_workspace: WorkspaceRecord,
    tmp_path: Path,
    use_hardlink_alias: bool,
):
    manifest = store.get_manifest(scanned_workspace.workspace_id)
    approval = store.approve(
        scanned_workspace.workspace_id,
        manifest.revision_id,
        manifest.policy_version,
    )
    database_path = tmp_path / "workspace.sqlite3"
    if use_hardlink_alias:
        with store._lock:
            store._connection.execute("PRAGMA wal_checkpoint(FULL)")
        alias = tmp_path / "workspace-alias.sqlite3"
        os.link(database_path, alias)
        database_path = alias
    child_code = """
import sys
from pathlib import Path
from exam_predictor.workspace.store import WorkspaceStore

database_path, workspace_id, revision_id, entry_id = sys.argv[1:]
child = WorkspaceStore(Path(database_path))
try:
    print("ready", flush=True)
    child.set_inclusion(workspace_id, revision_id, (entry_id,), False)
    print("changed", flush=True)
finally:
    child.close()
"""
    process = None
    try:
        with store.hold_transmission_authority(
            scanned_workspace.workspace_id,
            approval_id=approval.approval_id,
            revision_id=manifest.revision_id,
        ):
            process = subprocess.Popen(
                (
                    sys.executable,
                    "-c",
                    child_code,
                    str(database_path),
                    scanned_workspace.workspace_id,
                    manifest.revision_id,
                    manifest.entries[0].entry_id,
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert process.stdout is not None
            assert process.stdout.readline().strip() == "ready"
            assert process.poll() is None
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        assert stdout.strip() == "changed"
        if not use_hardlink_alias:
            assert store.get_workspace(scanned_workspace.workspace_id).state is (
                WorkspaceState.APPROVAL_REQUIRED
            )
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_same_thread_authority_mutation_fails_fast_instead_of_deadlocking(
    store: WorkspaceStore,
    scanned_workspace: WorkspaceRecord,
):
    manifest = store.get_manifest(scanned_workspace.workspace_id)
    approval = store.approve(
        scanned_workspace.workspace_id,
        manifest.revision_id,
        manifest.policy_version,
    )

    with store.hold_transmission_authority(
        scanned_workspace.workspace_id,
        approval_id=approval.approval_id,
        revision_id=manifest.revision_id,
    ):
        with pytest.raises(TransmissionAuthorityReentrancyError):
            store.set_inclusion(
                scanned_workspace.workspace_id,
                manifest.revision_id,
                (manifest.entries[0].entry_id,),
                False,
            )

    assert store.get_workspace(scanned_workspace.workspace_id).state is (
        WorkspaceState.APPROVED
    )
