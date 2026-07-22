from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

from exam_predictor.workspace.models import (
    ApprovalRecord,
    ApprovedEntryHash,
    CleanupRecord,
    ManifestEntry,
    ManifestRevision,
    ScanProgress,
    ScanResult,
    SourceState,
    WorkspaceEvent,
    WorkspaceJob,
    WorkspaceJobStatus,
    WorkspaceRecord,
    WorkspaceState,
    WorkspaceSummary,
    normalize_relative_path,
)
from exam_predictor.workspace.policy import DEFAULT_SCAN_POLICY


class WorkspaceNotFoundError(LookupError):
    pass


class ManifestNotFoundError(LookupError):
    pass


class WorkspaceJobNotFoundError(LookupError):
    pass


class StaleManifestError(RuntimeError):
    pass


class ActiveWorkspaceOperationError(RuntimeError):
    pass


class InvalidApprovalError(ValueError):
    pass


_SCHEMA = (
    """CREATE TABLE workspaces (
      workspace_id TEXT PRIMARY KEY,
      display_name TEXT NOT NULL,
      source_mode TEXT NOT NULL,
      canonical_root TEXT NOT NULL,
      root_device TEXT,
      root_file_id TEXT,
      state TEXT NOT NULL,
      current_draft_revision_id TEXT,
      current_approved_revision_id TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      last_scanned_at TEXT,
      last_access_verified_at TEXT
    )""",
    """CREATE TABLE manifest_revisions (
      revision_id TEXT PRIMARY KEY,
      workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
      parent_revision_id TEXT,
      scan_job_id TEXT,
      policy_version TEXT NOT NULL,
      created_at TEXT NOT NULL
    )""",
    """CREATE TABLE manifest_entries (
      revision_id TEXT NOT NULL REFERENCES manifest_revisions(revision_id) ON DELETE CASCADE,
      entry_id TEXT NOT NULL,
      relative_path TEXT NOT NULL,
      item_kind TEXT NOT NULL,
      format_category TEXT,
      size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
      modified_ns INTEGER,
      device_id TEXT,
      file_id TEXT,
      sha256 TEXT,
      state TEXT NOT NULL,
      included INTEGER NOT NULL CHECK(included IN (0, 1)),
      inclusion_reason TEXT,
      proposed_course_group TEXT NOT NULL,
      failure_code TEXT,
      safe_message TEXT,
      archive_parent_entry_id TEXT,
      archive_member_path TEXT,
      PRIMARY KEY(revision_id, entry_id)
    )""",
    """CREATE TABLE approvals (
      approval_id TEXT PRIMARY KEY,
      workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
      revision_id TEXT NOT NULL REFERENCES manifest_revisions(revision_id) ON DELETE CASCADE,
      approved_entries_json TEXT NOT NULL,
      policy_version TEXT NOT NULL,
      approved_at TEXT NOT NULL
    )""",
    """CREATE TABLE workspace_jobs (
      job_id TEXT PRIMARY KEY,
      workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
      job_kind TEXT NOT NULL,
      status TEXT NOT NULL,
      idempotency_key TEXT NOT NULL,
      safe_error_code TEXT,
      created_at TEXT NOT NULL,
      started_at TEXT,
      finished_at TEXT,
      UNIQUE(workspace_id, job_kind, idempotency_key)
    )""",
    """CREATE TABLE workspace_events (
      sequence INTEGER PRIMARY KEY AUTOINCREMENT,
      job_id TEXT NOT NULL REFERENCES workspace_jobs(job_id) ON DELETE CASCADE,
      event_type TEXT NOT NULL,
      message TEXT NOT NULL,
      payload_json TEXT NOT NULL,
      created_at TEXT NOT NULL
    )""",
    """CREATE TABLE cleanup_queue (
      cleanup_id TEXT PRIMARY KEY,
      workspace_id TEXT NOT NULL,
      owned_relative_path TEXT NOT NULL,
      safe_error_code TEXT NOT NULL,
      attempt_count INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )""",
    """CREATE INDEX idx_manifest_revisions_workspace_created
      ON manifest_revisions(workspace_id, created_at)""",
    """CREATE INDEX idx_manifest_entries_revision_state
      ON manifest_entries(revision_id, state, relative_path)""",
    """CREATE INDEX idx_workspace_events_job_sequence
      ON workspace_events(job_id, sequence)""",
)


class WorkspaceStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            self._connection = connection
            self._migrate()
        except BaseException:
            connection.close()
            raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _migrate(self) -> None:
        with self._transaction() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > 1:
                raise RuntimeError("The workspace database schema is newer than this application.")
            if version == 0:
                for statement in _SCHEMA:
                    connection.execute(statement)
                connection.execute("PRAGMA user_version=1")

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _timestamp(value: datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC).isoformat()

    @staticmethod
    def _canonical_json(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _workspace(row: sqlite3.Row) -> WorkspaceRecord:
        return WorkspaceRecord.model_validate(dict(row))

    @staticmethod
    def _job(row: sqlite3.Row) -> WorkspaceJob:
        return WorkspaceJob.model_validate(dict(row))

    @staticmethod
    def _entry(row: sqlite3.Row, workspace_id: str) -> ManifestEntry:
        item = dict(row)
        item.pop("revision_id", None)
        item["workspace_id"] = workspace_id
        item["included"] = bool(item["included"])
        return ManifestEntry.model_validate(item)

    @staticmethod
    def _event(row: sqlite3.Row) -> WorkspaceEvent:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        return WorkspaceEvent.model_validate(item)

    @staticmethod
    def _approval(row: sqlite3.Row) -> ApprovalRecord:
        item = dict(row)
        raw_entries = json.loads(item.pop("approved_entries_json"))
        item["entries"] = tuple(ApprovedEntryHash.model_validate(entry) for entry in raw_entries)
        return ApprovalRecord.model_validate(item)

    @staticmethod
    def _cleanup(row: sqlite3.Row) -> CleanupRecord:
        return CleanupRecord.model_validate(dict(row))

    @staticmethod
    def _insert_entry(
        connection: sqlite3.Connection, revision_id: str, item: ManifestEntry
    ) -> None:
        connection.execute(
            """INSERT INTO manifest_entries(
                   revision_id, entry_id, relative_path, item_kind, format_category,
                   size_bytes, modified_ns, device_id, file_id, sha256, state, included,
                   inclusion_reason, proposed_course_group, failure_code, safe_message,
                   archive_parent_entry_id, archive_member_path
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                revision_id,
                item.entry_id,
                item.relative_path,
                item.item_kind,
                item.format_category,
                item.size_bytes,
                item.modified_ns,
                item.device_id,
                item.file_id,
                item.sha256,
                item.state.value,
                int(item.included),
                item.inclusion_reason,
                item.proposed_course_group,
                item.failure_code,
                item.safe_message,
                item.archive_parent_entry_id,
                item.archive_member_path,
            ),
        )

    @classmethod
    def _insert_event(
        cls,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        event_type: str,
        message: str,
        payload: object,
        created_at: datetime,
    ) -> WorkspaceEvent:
        cursor = connection.execute(
            """INSERT INTO workspace_events(
                   job_id, event_type, message, payload_json, created_at
               ) VALUES (?, ?, ?, ?, ?)""",
            (
                job_id,
                event_type,
                message,
                cls._canonical_json(payload),
                cls._timestamp(created_at),
            ),
        )
        row = connection.execute(
            "SELECT * FROM workspace_events WHERE sequence = ?", (cursor.lastrowid,)
        ).fetchone()
        if row is None:
            raise RuntimeError("The workspace event could not be read back.")
        return cls._event(row)

    def _workspace_row(
        self, connection: sqlite3.Connection, workspace_id: str
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM workspaces WHERE workspace_id = ?", (workspace_id,)
        ).fetchone()
        if row is None:
            raise WorkspaceNotFoundError(f"Workspace '{workspace_id}' was not found.")
        return row

    def _job_row(self, connection: sqlite3.Connection, job_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM workspace_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise WorkspaceJobNotFoundError(f"Workspace job '{job_id}' was not found.")
        return row

    def _revision(
        self, connection: sqlite3.Connection, workspace_id: str, revision_id: str
    ) -> ManifestRevision:
        row = connection.execute(
            """SELECT * FROM manifest_revisions
               WHERE workspace_id = ? AND revision_id = ?""",
            (workspace_id, revision_id),
        ).fetchone()
        if row is None:
            raise ManifestNotFoundError(
                f"Manifest revision '{revision_id}' was not found for workspace '{workspace_id}'."
            )
        entries = connection.execute(
            """SELECT * FROM manifest_entries
               WHERE revision_id = ?
               ORDER BY relative_path COLLATE NOCASE, relative_path,
                        COALESCE(archive_member_path, ''), entry_id""",
            (revision_id,),
        ).fetchall()
        item = dict(row)
        item["entries"] = tuple(self._entry(entry, workspace_id) for entry in entries)
        return ManifestRevision.model_validate(item)

    def create_workspace(self, workspace: WorkspaceRecord) -> WorkspaceRecord:
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO workspaces(
                       workspace_id, display_name, source_mode, canonical_root,
                       root_device, root_file_id, state, current_draft_revision_id,
                       current_approved_revision_id, created_at, updated_at,
                       last_scanned_at, last_access_verified_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    workspace.workspace_id,
                    workspace.display_name,
                    workspace.source_mode.value,
                    str(workspace.canonical_root),
                    workspace.root_device,
                    workspace.root_file_id,
                    workspace.state.value,
                    workspace.current_draft_revision_id,
                    workspace.current_approved_revision_id,
                    self._timestamp(workspace.created_at),
                    self._timestamp(workspace.updated_at),
                    self._timestamp(workspace.last_scanned_at),
                    self._timestamp(workspace.last_access_verified_at),
                ),
            )
            row = self._workspace_row(connection, workspace.workspace_id)
        return self._workspace(row)

    def list_workspaces(self) -> Sequence[WorkspaceSummary]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM workspaces ORDER BY updated_at DESC, workspace_id ASC"
            ).fetchall()
            summaries: list[WorkspaceSummary] = []
            for row in rows:
                counts = {
                    SourceState(count["state"]): int(count["item_count"])
                    for count in self._connection.execute(
                        """SELECT state, COUNT(*) AS item_count
                           FROM manifest_entries
                           WHERE revision_id = ? GROUP BY state""",
                        (row["current_draft_revision_id"],),
                    ).fetchall()
                }
                summaries.append(
                    WorkspaceSummary(
                        workspace_id=row["workspace_id"],
                        display_name=row["display_name"],
                        source_mode=row["source_mode"],
                        state=row["state"],
                        counts=counts,
                        updated_at=row["updated_at"],
                    )
                )
        return tuple(summaries)

    def get_workspace(self, workspace_id: str) -> WorkspaceRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM workspaces WHERE workspace_id = ?", (workspace_id,)
            ).fetchone()
        return self._workspace(row) if row is not None else None

    def source_root(self, workspace_id: str) -> Path:
        with self._lock:
            row = self._workspace_row(self._connection, workspace_id)
        return Path(row["canonical_root"])

    def get_manifest_entries(self, workspace_id: str) -> Sequence[ManifestEntry]:
        with self._lock:
            workspace = self._workspace_row(self._connection, workspace_id)
            revision_id = workspace["current_draft_revision_id"]
            if revision_id is None:
                return ()
            return self._revision(self._connection, workspace_id, revision_id).entries

    def commit_scan(
        self, workspace_id: str, result: ScanResult, job_id: str
    ) -> ManifestRevision:
        if result.workspace_id != workspace_id:
            raise ValueError("scan result workspace does not match")
        revision_id = uuid4().hex
        with self._transaction() as connection:
            workspace = self._workspace_row(connection, workspace_id)
            job_row = self._job_row(connection, job_id)
            if job_row["workspace_id"] != workspace_id:
                raise WorkspaceJobNotFoundError(
                    f"Workspace job '{job_id}' does not belong to workspace '{workspace_id}'."
                )
            connection.execute(
                """INSERT INTO manifest_revisions(
                       revision_id, workspace_id, parent_revision_id, scan_job_id,
                       policy_version, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    revision_id,
                    workspace_id,
                    workspace["current_draft_revision_id"],
                    job_id,
                    DEFAULT_SCAN_POLICY.policy_version,
                    self._timestamp(result.completed_at),
                ),
            )
            seen: set[str] = set()
            for item in result.entries:
                if item.workspace_id != workspace_id:
                    raise ValueError("manifest entry workspace does not match")
                if item.entry_id in seen:
                    raise ValueError("manifest entry IDs must be unique within a revision")
                seen.add(item.entry_id)
                self._insert_entry(connection, revision_id, item)
            connection.execute(
                """UPDATE workspace_jobs
                   SET status = ?, safe_error_code = NULL, finished_at = ?
                   WHERE job_id = ?""",
                (
                    WorkspaceJobStatus.SUCCEEDED.value,
                    self._timestamp(result.completed_at),
                    job_id,
                ),
            )
            connection.execute(
                """UPDATE workspaces
                   SET state = ?, current_draft_revision_id = ?,
                       last_scanned_at = ?, updated_at = ?
                   WHERE workspace_id = ?""",
                (
                    WorkspaceState.APPROVAL_REQUIRED.value,
                    revision_id,
                    self._timestamp(result.completed_at),
                    self._timestamp(result.completed_at),
                    workspace_id,
                ),
            )
            self._insert_event(
                connection,
                job_id=job_id,
                event_type="approval_required",
                message="Scan complete. Review the selected course files.",
                payload={
                    "bytes_hashed": result.bytes_hashed,
                    "discovered_count": result.discovered_count,
                    "failure_count": result.failure_count,
                    "revision_id": revision_id,
                },
                created_at=result.completed_at,
            )
            revision = self._revision(connection, workspace_id, revision_id)
        return revision

    def get_manifest(
        self, workspace_id: str, revision_id: str | None = None
    ) -> ManifestRevision:
        with self._lock:
            workspace = self._workspace_row(self._connection, workspace_id)
            selected = revision_id or workspace["current_draft_revision_id"]
            if selected is None:
                raise ManifestNotFoundError(
                    f"Workspace '{workspace_id}' does not have a manifest."
                )
            return self._revision(self._connection, workspace_id, selected)

    def set_inclusion(
        self,
        workspace_id: str,
        revision_id: str,
        entry_ids: Sequence[str],
        included: bool,
    ) -> ManifestRevision:
        requested = set(entry_ids)
        new_revision_id = uuid4().hex
        now = self._now()
        with self._transaction() as connection:
            workspace = self._workspace_row(connection, workspace_id)
            if workspace["current_draft_revision_id"] != revision_id:
                raise StaleManifestError(
                    f"Manifest revision '{revision_id}' is no longer current."
                )
            current = self._revision(connection, workspace_id, revision_id)
            known = {item.entry_id for item in current.entries}
            missing = requested - known
            if missing:
                raise ManifestNotFoundError(
                    f"Manifest entries were not found: {', '.join(sorted(missing))}."
                )
            connection.execute(
                """INSERT INTO manifest_revisions(
                       revision_id, workspace_id, parent_revision_id, scan_job_id,
                       policy_version, created_at
                   ) VALUES (?, ?, ?, NULL, ?, ?)""",
                (
                    new_revision_id,
                    workspace_id,
                    revision_id,
                    current.policy_version,
                    self._timestamp(now),
                ),
            )
            for item in current.entries:
                clone = (
                    item.model_copy(update={"included": included})
                    if item.entry_id in requested
                    else item
                )
                self._insert_entry(connection, new_revision_id, clone)
            connection.execute(
                """UPDATE workspaces
                   SET current_draft_revision_id = ?, state = ?, updated_at = ?
                   WHERE workspace_id = ?""",
                (
                    new_revision_id,
                    WorkspaceState.APPROVAL_REQUIRED.value,
                    self._timestamp(now),
                    workspace_id,
                ),
            )
            revision = self._revision(connection, workspace_id, new_revision_id)
        return revision

    def approve(
        self, workspace_id: str, revision_id: str, policy_version: str
    ) -> ApprovalRecord:
        approval_id = uuid4().hex
        now = self._now()
        with self._transaction() as connection:
            workspace = self._workspace_row(connection, workspace_id)
            if workspace["current_draft_revision_id"] != revision_id:
                raise StaleManifestError(
                    f"Manifest revision '{revision_id}' is no longer current."
                )
            revision = self._revision(connection, workspace_id, revision_id)
            if revision.policy_version != policy_version:
                raise InvalidApprovalError("The manifest policy version does not match.")
            selected = [item for item in revision.entries if item.included]
            if any(
                item.sha256 is None
                or item.state not in {SourceState.PENDING_APPROVAL, SourceState.APPROVED}
                for item in selected
            ):
                raise InvalidApprovalError(
                    "Every selected manifest entry must be hashable and approval-eligible."
                )
            approved_entries = sorted(
                (
                    {"entry_id": item.entry_id, "sha256": item.sha256}
                    for item in selected
                ),
                key=lambda item: item["entry_id"],
            )
            connection.execute(
                """INSERT INTO approvals(
                       approval_id, workspace_id, revision_id, approved_entries_json,
                       policy_version, approved_at
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    approval_id,
                    workspace_id,
                    revision_id,
                    self._canonical_json(approved_entries),
                    policy_version,
                    self._timestamp(now),
                ),
            )
            if selected:
                placeholders = ",".join("?" for _ in selected)
                connection.execute(
                    f"""UPDATE manifest_entries SET state = ?
                        WHERE revision_id = ? AND entry_id IN ({placeholders})""",
                    (
                        SourceState.APPROVED.value,
                        revision_id,
                        *(item.entry_id for item in selected),
                    ),
                )
            connection.execute(
                """UPDATE workspaces
                   SET state = ?, current_approved_revision_id = ?, updated_at = ?
                   WHERE workspace_id = ?""",
                (
                    WorkspaceState.APPROVED.value,
                    revision_id,
                    self._timestamp(now),
                    workspace_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError("The approval could not be read back.")
            approval = self._approval(row)
        return approval

    def get_approval(self, workspace_id: str) -> ApprovalRecord | None:
        with self._lock:
            workspace = self._connection.execute(
                "SELECT * FROM workspaces WHERE workspace_id = ?", (workspace_id,)
            ).fetchone()
            if workspace is None or workspace["current_approved_revision_id"] is None:
                return None
            row = self._connection.execute(
                """SELECT * FROM approvals
                   WHERE workspace_id = ? AND revision_id = ?
                   ORDER BY approved_at DESC, rowid DESC LIMIT 1""",
                (workspace_id, workspace["current_approved_revision_id"]),
            ).fetchone()
        return self._approval(row) if row is not None else None

    def create_job(self, job: WorkspaceJob, idempotency_key: str) -> WorkspaceJob:
        with self._transaction() as connection:
            self._workspace_row(connection, job.workspace_id)
            existing = connection.execute(
                """SELECT * FROM workspace_jobs
                   WHERE workspace_id = ? AND job_kind = ? AND idempotency_key = ?""",
                (job.workspace_id, job.job_kind, idempotency_key),
            ).fetchone()
            if existing is not None:
                return self._job(existing)
            connection.execute(
                """INSERT INTO workspace_jobs(
                       job_id, workspace_id, job_kind, status, idempotency_key,
                       safe_error_code, created_at, started_at, finished_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.job_id,
                    job.workspace_id,
                    job.job_kind,
                    job.status.value,
                    idempotency_key,
                    job.safe_error_code,
                    self._timestamp(job.created_at),
                    self._timestamp(job.started_at),
                    self._timestamp(job.finished_at),
                ),
            )
            row = self._job_row(connection, job.job_id)
        return self._job(row)

    def get_job(self, job_id: str) -> WorkspaceJob:
        with self._lock:
            return self._job(self._job_row(self._connection, job_id))

    def start_job(self, job_id: str) -> WorkspaceJob:
        now = self._now()
        with self._transaction() as connection:
            row = self._job_row(connection, job_id)
            status = WorkspaceJobStatus(row["status"])
            if status is WorkspaceJobStatus.RUNNING:
                return self._job(row)
            if status is not WorkspaceJobStatus.QUEUED:
                raise ActiveWorkspaceOperationError(
                    f"Workspace job '{job_id}' cannot be started from '{status.value}'."
                )
            connection.execute(
                """UPDATE workspace_jobs SET status = ?, started_at = ?
                   WHERE job_id = ?""",
                (WorkspaceJobStatus.RUNNING.value, self._timestamp(now), job_id),
            )
            connection.execute(
                """UPDATE workspaces SET state = ?, updated_at = ?
                   WHERE workspace_id = ?""",
                (
                    WorkspaceState.SCANNING.value,
                    self._timestamp(now),
                    row["workspace_id"],
                ),
            )
            self._insert_event(
                connection,
                job_id=job_id,
                event_type="started",
                message="Workspace scan started.",
                payload={},
                created_at=now,
            )
            updated = self._job_row(connection, job_id)
        return self._job(updated)

    def append_progress(self, job_id: str, progress: ScanProgress) -> WorkspaceEvent:
        relative_path = progress.current_relative_path
        if relative_path is not None:
            relative_path = normalize_relative_path(relative_path)
            if len(relative_path) > 1_024:
                raise ValueError("current_relative_path is too long")
        payload = {
            "bytes_hashed": progress.bytes_hashed,
            "current_relative_path": relative_path,
            "discovered_count": progress.discovered_count,
            "failure_count": progress.failure_count,
        }
        with self._transaction() as connection:
            self._job_row(connection, job_id)
            event = self._insert_event(
                connection,
                job_id=job_id,
                event_type="scan_progress",
                message="Scanning course files.",
                payload=payload,
                created_at=self._now(),
            )
        return event

    def fail_job(self, job_id: str, safe_error_code: str) -> WorkspaceJob:
        now = self._now()
        with self._transaction() as connection:
            row = self._job_row(connection, job_id)
            connection.execute(
                """UPDATE workspace_jobs
                   SET status = ?, safe_error_code = ?, finished_at = ?
                   WHERE job_id = ?""",
                (
                    WorkspaceJobStatus.FAILED.value,
                    safe_error_code,
                    self._timestamp(now),
                    job_id,
                ),
            )
            connection.execute(
                """UPDATE workspaces SET state = ?, updated_at = ?
                   WHERE workspace_id = ?""",
                (
                    WorkspaceState.NEEDS_ATTENTION.value,
                    self._timestamp(now),
                    row["workspace_id"],
                ),
            )
            self._insert_event(
                connection,
                job_id=job_id,
                event_type="failed",
                message="Workspace scan failed.",
                payload={"safe_error_code": safe_error_code},
                created_at=now,
            )
            updated = self._job_row(connection, job_id)
        return self._job(updated)

    def update_job(self, job: WorkspaceJob, event: WorkspaceEvent) -> None:
        if event.job_id != job.job_id:
            raise ValueError("event job does not match the updated job")
        with self._transaction() as connection:
            self._job_row(connection, job.job_id)
            connection.execute(
                """UPDATE workspace_jobs
                   SET workspace_id = ?, job_kind = ?, status = ?, idempotency_key = ?,
                       safe_error_code = ?, created_at = ?, started_at = ?, finished_at = ?
                   WHERE job_id = ?""",
                (
                    job.workspace_id,
                    job.job_kind,
                    job.status.value,
                    job.idempotency_key,
                    job.safe_error_code,
                    self._timestamp(job.created_at),
                    self._timestamp(job.started_at),
                    self._timestamp(job.finished_at),
                    job.job_id,
                ),
            )
            self._insert_event(
                connection,
                job_id=event.job_id,
                event_type=event.event_type,
                message=event.message,
                payload=event.payload,
                created_at=event.created_at,
            )

    def list_job_events(
        self, job_id: str, after_sequence: int = 0
    ) -> Sequence[WorkspaceEvent]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM workspace_events
                   WHERE job_id = ? AND sequence > ? ORDER BY sequence ASC""",
                (job_id, after_sequence),
            ).fetchall()
        return tuple(self._event(row) for row in rows)

    def mark_entry_changed(self, workspace_id: str, entry_id: str, code: str) -> None:
        new_revision_id = uuid4().hex
        now = self._now()
        with self._transaction() as connection:
            workspace = self._workspace_row(connection, workspace_id)
            revision_id = workspace["current_draft_revision_id"]
            if revision_id is None:
                raise ManifestNotFoundError(
                    f"Workspace '{workspace_id}' does not have a manifest."
                )
            revision = self._revision(connection, workspace_id, revision_id)
            if entry_id not in {item.entry_id for item in revision.entries}:
                raise ManifestNotFoundError(f"Manifest entry '{entry_id}' was not found.")
            connection.execute(
                """INSERT INTO manifest_revisions(
                       revision_id, workspace_id, parent_revision_id, scan_job_id,
                       policy_version, created_at
                   ) VALUES (?, ?, ?, NULL, ?, ?)""",
                (
                    new_revision_id,
                    workspace_id,
                    revision_id,
                    revision.policy_version,
                    self._timestamp(now),
                ),
            )
            for item in revision.entries:
                clone = item
                if item.entry_id == entry_id:
                    clone = item.model_copy(
                        update={
                            "state": SourceState.CHANGED,
                            "included": False,
                            "inclusion_reason": code,
                            "failure_code": code,
                            "safe_message": "The selected source changed and needs review.",
                        }
                    )
                self._insert_entry(connection, new_revision_id, clone)
            connection.execute(
                """UPDATE workspaces
                   SET state = ?, current_draft_revision_id = ?, updated_at = ?
                   WHERE workspace_id = ?""",
                (
                    WorkspaceState.NEEDS_ATTENTION.value,
                    new_revision_id,
                    self._timestamp(now),
                    workspace_id,
                ),
            )

    def record_access_verified(self, workspace_id: str, verified_at: datetime) -> None:
        with self._transaction() as connection:
            self._workspace_row(connection, workspace_id)
            connection.execute(
                """UPDATE workspaces
                   SET last_access_verified_at = ?, updated_at = ? WHERE workspace_id = ?""",
                (
                    self._timestamp(verified_at),
                    self._timestamp(verified_at),
                    workspace_id,
                ),
            )

    def mark_deleting(self, workspace_id: str) -> WorkspaceRecord:
        now = self._now()
        active = (WorkspaceJobStatus.QUEUED.value, WorkspaceJobStatus.RUNNING.value)
        with self._transaction() as connection:
            workspace = self._workspace_row(connection, workspace_id)
            if workspace["state"] == WorkspaceState.SCANNING.value:
                raise ActiveWorkspaceOperationError(
                    f"Workspace '{workspace_id}' is not settled."
                )
            active_row = connection.execute(
                """SELECT 1 FROM workspace_jobs
                   WHERE workspace_id = ? AND status IN (?, ?) LIMIT 1""",
                (workspace_id, *active),
            ).fetchone()
            if active_row is not None:
                raise ActiveWorkspaceOperationError(
                    f"Workspace '{workspace_id}' has an active operation."
                )
            connection.execute(
                "UPDATE workspaces SET state = ?, updated_at = ? WHERE workspace_id = ?",
                (
                    WorkspaceState.DELETING.value,
                    self._timestamp(now),
                    workspace_id,
                ),
            )
            row = self._workspace_row(connection, workspace_id)
        return self._workspace(row)

    def delete_workspace_rows(self, workspace_id: str) -> None:
        with self._transaction() as connection:
            connection.execute("DELETE FROM workspaces WHERE workspace_id = ?", (workspace_id,))

    def _owned_relative_path(self, owned_path: Path) -> tuple[str, Path]:
        data_directory = self.database_path.parent.absolute()
        candidate = owned_path.absolute() if owned_path.is_absolute() else data_directory / owned_path
        try:
            relative = candidate.relative_to(data_directory)
        except ValueError:
            raise ValueError(
                "cleanup path must remain within the ExamSage data directory"
            ) from None
        raw = relative.as_posix()
        if not raw or "\x00" in raw or "\\" in raw:
            raise ValueError("cleanup path must be relative to the ExamSage data directory")
        value = PurePosixPath(raw)
        if value.is_absolute() or ".." in value.parts or ":" in value.parts[0]:
            raise ValueError("cleanup path must remain within the ExamSage data directory")
        normalized = value.as_posix()
        if normalized in {"", "."}:
            raise ValueError("cleanup path must identify an app-owned path")
        return normalized, candidate

    def queue_cleanup(self, workspace_id: str, owned_path: Path, code: str) -> None:
        relative_path, absolute_path = self._owned_relative_path(owned_path)
        now = self._now()
        with self._transaction() as connection:
            workspace = connection.execute(
                "SELECT source_mode, canonical_root FROM workspaces WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
            if workspace is not None and workspace["source_mode"] == "native_folder":
                native_root = Path(workspace["canonical_root"]).absolute()
                if absolute_path == native_root or native_root in absolute_path.parents:
                    raise ValueError("native source roots cannot be queued for cleanup")
            connection.execute(
                """INSERT INTO cleanup_queue(
                       cleanup_id, workspace_id, owned_relative_path, safe_error_code,
                       attempt_count, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, 0, ?, ?)""",
                (
                    uuid4().hex,
                    workspace_id,
                    relative_path,
                    code,
                    self._timestamp(now),
                    self._timestamp(now),
                ),
            )

    def list_cleanup(self) -> Sequence[CleanupRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM cleanup_queue ORDER BY created_at ASC, rowid ASC"
            ).fetchall()
        return tuple(self._cleanup(row) for row in rows)
