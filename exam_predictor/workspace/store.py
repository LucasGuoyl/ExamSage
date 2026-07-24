from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Callable, Iterator, Sequence
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

_BLOCKED_JOB_WORKSPACE_STATES = {
    WorkspaceState.DELETING,
    WorkspaceState.CLEANUP_PENDING,
}
_TERMINAL_JOB_STATUSES = {
    WorkspaceJobStatus.SUCCEEDED,
    WorkspaceJobStatus.FAILED,
    WorkspaceJobStatus.CANCELLED,
}
_ALLOWED_JOB_TRANSITIONS = {
    WorkspaceJobStatus.QUEUED: {
        WorkspaceJobStatus.RUNNING,
        WorkspaceJobStatus.FAILED,
        WorkspaceJobStatus.CANCELLED,
    },
    WorkspaceJobStatus.RUNNING: {
        WorkspaceJobStatus.SUCCEEDED,
        WorkspaceJobStatus.FAILED,
        WorkspaceJobStatus.CANCELLED,
    },
    WorkspaceJobStatus.SUCCEEDED: set(),
    WorkspaceJobStatus.FAILED: set(),
    WorkspaceJobStatus.CANCELLED: set(),
}
_EVENT_RULES = {
    "queued": ("Workspace operation queued.", frozenset(), frozenset()),
    "started": ("Workspace scan started.", frozenset(), frozenset()),
    "scan_progress": (
        "Scanning course files.",
        frozenset(
            {
                "bytes_hashed",
                "current_relative_path",
                "discovered_count",
                "failure_count",
            }
        ),
        frozenset(
            {
                "bytes_hashed",
                "current_relative_path",
                "discovered_count",
                "failure_count",
            }
        ),
    ),
    "approval_required": (
        "Scan complete. Review the selected course files.",
        frozenset(
            {"bytes_hashed", "discovered_count", "failure_count", "revision_id"}
        ),
        frozenset(
            {"bytes_hashed", "discovered_count", "failure_count", "revision_id"}
        ),
    ),
    "succeeded": (
        "Workspace operation completed.",
        frozenset({"revision_id"}),
        frozenset(),
    ),
    "failed": (
        "Workspace scan failed.",
        frozenset({"safe_error_code"}),
        frozenset({"safe_error_code"}),
    ),
    "cancelled": (
        "Workspace operation cancelled.",
        frozenset({"safe_error_code"}),
        frozenset(),
    ),
}
_STATUS_EVENTS = {
    WorkspaceJobStatus.RUNNING: {"started"},
    WorkspaceJobStatus.SUCCEEDED: {"approval_required", "succeeded"},
    WorkspaceJobStatus.FAILED: {"failed"},
    WorkspaceJobStatus.CANCELLED: {"cancelled"},
}
_SAFE_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,119}$")
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_SECRET_KEY_PARTS = ("api_key", "credential", "password", "secret", "token")


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
            connection.execute(
                """WITH ranked AS (
                       SELECT revisions.rowid AS revision_rowid,
                              ROW_NUMBER() OVER (
                                PARTITION BY revisions.scan_job_id
                                ORDER BY
                                  CASE WHEN EXISTS (
                                    SELECT 1 FROM workspaces
                                    WHERE current_draft_revision_id = revisions.revision_id
                                       OR current_approved_revision_id = revisions.revision_id
                                  ) THEN 0 ELSE 1 END,
                                  revisions.created_at DESC,
                                  revisions.rowid DESC
                              ) AS replay_rank
                       FROM manifest_revisions AS revisions
                       WHERE revisions.scan_job_id IS NOT NULL
                   )
                   UPDATE manifest_revisions SET scan_job_id = NULL
                   WHERE rowid IN (
                     SELECT revision_rowid FROM ranked WHERE replay_rank > 1
                   )"""
            )
            connection.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_manifest_revisions_scan_job
                   ON manifest_revisions(scan_job_id) WHERE scan_job_id IS NOT NULL"""
            )

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
    def _safe_code(value: str) -> str:
        if not _SAFE_CODE_PATTERN.fullmatch(value):
            raise ValueError("safe_error_code must be a bounded stable code")
        return value

    @classmethod
    def _validated_event_payload(
        cls, event_type: str, message: str, payload: object
    ) -> dict[str, str | int | float | bool | None]:
        rule = _EVENT_RULES.get(event_type)
        if rule is None:
            raise ValueError("workspace event type is not allowed")
        expected_message, allowed_keys, required_keys = rule
        if message != expected_message:
            raise ValueError("workspace event message is not allowed")
        if not isinstance(payload, dict):
            raise ValueError("workspace event payload must be an object")
        keys = set(payload)
        if any(part in key.casefold() for key in keys for part in _SECRET_KEY_PARTS):
            raise ValueError("workspace event payload contains a secret-shaped key")
        if not required_keys <= keys or not keys <= allowed_keys:
            raise ValueError("workspace event payload keys are not allowed")
        for key in {"bytes_hashed", "discovered_count", "failure_count"} & keys:
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"workspace event {key} must be a nonnegative integer")
        if "current_relative_path" in keys:
            value = payload["current_relative_path"]
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError("workspace event path must be text")
                normalized = normalize_relative_path(value)
                if normalized != value or len(normalized) > 1_024:
                    raise ValueError("workspace event path must be bounded and normalized")
        if "safe_error_code" in keys:
            value = payload["safe_error_code"]
            if not isinstance(value, str):
                raise ValueError("workspace event safe_error_code must be text")
            cls._safe_code(value)
        if "revision_id" in keys:
            value = payload["revision_id"]
            if not isinstance(value, str) or not _SAFE_ID_PATTERN.fullmatch(value):
                raise ValueError("workspace event revision_id must be a bounded identifier")
        return payload

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
        validated_payload = cls._validated_event_payload(event_type, message, payload)
        cursor = connection.execute(
            """INSERT INTO workspace_events(
                   job_id, event_type, message, payload_json, created_at
               ) VALUES (?, ?, ?, ?, ?)""",
            (
                job_id,
                event_type,
                message,
                cls._canonical_json(validated_payload),
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

    @staticmethod
    def _guard_workspace_accepts_jobs(workspace: sqlite3.Row) -> None:
        state = WorkspaceState(workspace["state"])
        if state in _BLOCKED_JOB_WORKSPACE_STATES:
            raise ActiveWorkspaceOperationError(
                f"Workspace '{workspace['workspace_id']}' cannot accept jobs while {state.value}."
            )

    @staticmethod
    def _guard_job_transition(
        current: WorkspaceJobStatus, requested: WorkspaceJobStatus
    ) -> None:
        if requested not in _ALLOWED_JOB_TRANSITIONS[current]:
            raise ActiveWorkspaceOperationError(
                f"Workspace job transition '{current.value}' to '{requested.value}' is not allowed."
            )

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

    def _clone_revision(
        self,
        connection: sqlite3.Connection,
        source: ManifestRevision,
        revision_id: str,
        created_at: datetime,
        transform: Callable[[ManifestEntry], ManifestEntry],
    ) -> ManifestRevision:
        connection.execute(
            """INSERT INTO manifest_revisions(
                   revision_id, workspace_id, parent_revision_id, scan_job_id,
                   policy_version, created_at
               ) VALUES (?, ?, ?, NULL, ?, ?)""",
            (
                revision_id,
                source.workspace_id,
                source.revision_id,
                source.policy_version,
                self._timestamp(created_at),
            ),
        )
        for item in source.entries:
            self._insert_entry(connection, revision_id, transform(item))
        return self._revision(connection, source.workspace_id, revision_id)

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
        with self._transaction() as connection:
            workspace = self._workspace_row(connection, workspace_id)
            job_row = self._job_row(connection, job_id)
            if job_row["workspace_id"] != workspace_id:
                raise WorkspaceJobNotFoundError(
                    f"Workspace job '{job_id}' does not belong to workspace '{workspace_id}'."
                )
            existing = connection.execute(
                """SELECT revision_id FROM manifest_revisions
                   WHERE workspace_id = ? AND scan_job_id = ?""",
                (workspace_id, job_id),
            ).fetchone()
            if existing is not None:
                return self._revision(connection, workspace_id, existing["revision_id"])
            if WorkspaceJobStatus(job_row["status"]) is not WorkspaceJobStatus.RUNNING:
                raise ActiveWorkspaceOperationError(
                    f"Workspace job '{job_id}' is not running."
                )
            self._guard_workspace_accepts_jobs(workspace)
            job_order = connection.execute(
                "SELECT rowid, created_at FROM workspace_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if job_order is None:
                raise WorkspaceJobNotFoundError(
                    f"Workspace job '{job_id}' was not found."
                )
            newer_commit = connection.execute(
                """SELECT 1
                   FROM manifest_revisions AS revisions
                   JOIN workspace_jobs AS committed
                     ON committed.job_id = revisions.scan_job_id
                   WHERE revisions.workspace_id = ?
                     AND (
                       committed.created_at > ?
                       OR (committed.created_at = ? AND committed.rowid > ?)
                     )
                   LIMIT 1""",
                (
                    workspace_id,
                    job_order["created_at"],
                    job_order["created_at"],
                    job_order["rowid"],
                ),
            ).fetchone()
            if newer_commit is not None:
                raise StaleManifestError(
                    f"Workspace job '{job_id}' completed after a newer scan."
                )
            revision_id = uuid4().hex
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
            self._clone_revision(
                connection,
                current,
                new_revision_id,
                now,
                lambda item: (
                    item.model_copy(update={"included": included})
                    if item.entry_id in requested
                    else item
                ),
            )
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
            workspace = self._workspace_row(connection, job.workspace_id)
            self._guard_workspace_accepts_jobs(workspace)
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

    def recover_running_jobs(self) -> Sequence[WorkspaceJob]:
        recovered: list[WorkspaceJob] = []
        now = self._now()
        with self._transaction() as connection:
            rows = connection.execute(
                """SELECT * FROM workspace_jobs WHERE status = ?
                   ORDER BY created_at ASC, rowid ASC""",
                (WorkspaceJobStatus.RUNNING.value,),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """UPDATE workspace_jobs
                       SET status = ?, safe_error_code = NULL,
                           started_at = NULL, finished_at = NULL
                       WHERE job_id = ? AND status = ?""",
                    (
                        WorkspaceJobStatus.QUEUED.value,
                        row["job_id"],
                        WorkspaceJobStatus.RUNNING.value,
                    ),
                )
                self._insert_event(
                    connection,
                    job_id=row["job_id"],
                    event_type="queued",
                    message="Workspace operation queued.",
                    payload={},
                    created_at=now,
                )
                recovered.append(self._job(self._job_row(connection, row["job_id"])))
        return tuple(recovered)

    def start_job(self, job_id: str) -> WorkspaceJob:
        now = self._now()
        with self._transaction() as connection:
            row = self._job_row(connection, job_id)
            workspace = self._workspace_row(connection, row["workspace_id"])
            self._guard_workspace_accepts_jobs(workspace)
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
            row = self._job_row(connection, job_id)
            if WorkspaceJobStatus(row["status"]) is not WorkspaceJobStatus.RUNNING:
                raise ActiveWorkspaceOperationError(
                    f"Workspace job '{job_id}' is not running."
                )
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
        self._safe_code(safe_error_code)
        now = self._now()
        with self._transaction() as connection:
            row = self._job_row(connection, job_id)
            current_status = WorkspaceJobStatus(row["status"])
            if current_status in _TERMINAL_JOB_STATUSES:
                raise ActiveWorkspaceOperationError(
                    f"Workspace job '{job_id}' is already {current_status.value}."
                )
            self._guard_job_transition(current_status, WorkspaceJobStatus.FAILED)
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
            stored_row = self._job_row(connection, job.job_id)
            stored = self._job(stored_row)
            immutable_identity = (
                "workspace_id",
                "job_kind",
                "idempotency_key",
                "created_at",
            )
            if any(getattr(stored, field) != getattr(job, field) for field in immutable_identity):
                raise ValueError("workspace job identity fields are immutable")
            self._guard_job_transition(stored.status, job.status)
            allowed_events = _STATUS_EVENTS.get(job.status, set())
            if event.event_type not in allowed_events:
                raise ValueError("workspace event does not match the job transition")
            if job.status is WorkspaceJobStatus.RUNNING and job.started_at is None:
                raise ValueError("a running workspace job requires started_at")
            if job.status in _TERMINAL_JOB_STATUSES and job.finished_at is None:
                raise ValueError("a terminal workspace job requires finished_at")
            if job.safe_error_code is not None:
                self._safe_code(job.safe_error_code)
            connection.execute(
                """UPDATE workspace_jobs
                   SET status = ?, safe_error_code = ?, started_at = ?, finished_at = ?
                   WHERE job_id = ?""",
                (
                    job.status.value,
                    job.safe_error_code,
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

            def mark_changed(item: ManifestEntry) -> ManifestEntry:
                if item.entry_id != entry_id:
                    return item
                return item.model_copy(
                    update={
                        "state": SourceState.CHANGED,
                        "included": False,
                        "inclusion_reason": code,
                        "failure_code": code,
                        "safe_message": "The selected source changed and needs review.",
                    }
                )

            self._clone_revision(
                connection,
                revision,
                new_revision_id,
                now,
                mark_changed,
            )
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
            workspace = self._workspace_row(connection, workspace_id)
            if workspace["state"] != WorkspaceState.DELETING.value:
                raise ActiveWorkspaceOperationError(
                    f"Workspace '{workspace_id}' is not marked for deletion."
                )
            active_row = connection.execute(
                """SELECT 1 FROM workspace_jobs
                   WHERE workspace_id = ? AND status IN (?, ?) LIMIT 1""",
                (
                    workspace_id,
                    WorkspaceJobStatus.QUEUED.value,
                    WorkspaceJobStatus.RUNNING.value,
                ),
            ).fetchone()
            if active_row is not None:
                raise ActiveWorkspaceOperationError(
                    f"Workspace '{workspace_id}' has an active operation."
                )
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

    def fail_cleanup(
        self, cleanup_id: str, safe_error_code: str
    ) -> CleanupRecord | None:
        self._safe_code(safe_error_code)
        now = self._now()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM cleanup_queue WHERE cleanup_id = ?", (cleanup_id,)
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """UPDATE cleanup_queue
                   SET safe_error_code = ?, attempt_count = attempt_count + 1,
                       updated_at = ? WHERE cleanup_id = ?""",
                (safe_error_code, self._timestamp(now), cleanup_id),
            )
            connection.execute(
                """UPDATE workspaces SET state = ?, updated_at = ?
                   WHERE workspace_id = ?""",
                (
                    WorkspaceState.CLEANUP_PENDING.value,
                    self._timestamp(now),
                    row["workspace_id"],
                ),
            )
            updated = connection.execute(
                "SELECT * FROM cleanup_queue WHERE cleanup_id = ?", (cleanup_id,)
            ).fetchone()
            if updated is None:
                raise RuntimeError("The cleanup record could not be read back.")
            return self._cleanup(updated)

    def complete_cleanup(self, cleanup_id: str) -> str | None:
        with self._transaction() as connection:
            cleanup = connection.execute(
                "SELECT * FROM cleanup_queue WHERE cleanup_id = ?", (cleanup_id,)
            ).fetchone()
            if cleanup is None:
                return None
            workspace_id = str(cleanup["workspace_id"])
            workspace = connection.execute(
                "SELECT * FROM workspaces WHERE workspace_id = ?", (workspace_id,)
            ).fetchone()
            if workspace is not None:
                state = WorkspaceState(workspace["state"])
                if state not in {
                    WorkspaceState.DELETING,
                    WorkspaceState.CLEANUP_PENDING,
                }:
                    raise ActiveWorkspaceOperationError(
                        f"Workspace '{workspace_id}' is not pending cleanup."
                    )
                active = connection.execute(
                    """SELECT 1 FROM workspace_jobs
                       WHERE workspace_id = ? AND status IN (?, ?) LIMIT 1""",
                    (
                        workspace_id,
                        WorkspaceJobStatus.QUEUED.value,
                        WorkspaceJobStatus.RUNNING.value,
                    ),
                ).fetchone()
                if active is not None:
                    raise ActiveWorkspaceOperationError(
                        f"Workspace '{workspace_id}' has an active operation."
                    )
            connection.execute(
                "DELETE FROM cleanup_queue WHERE cleanup_id = ?", (cleanup_id,)
            )
            if workspace is not None:
                connection.execute(
                    "DELETE FROM workspaces WHERE workspace_id = ?", (workspace_id,)
                )
        return workspace_id
