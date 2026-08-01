from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import wraps
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
    SourceMode,
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


_AUTHORITY_LOCKS_GUARD = threading.Lock()
_AUTHORITY_LOCKS: dict[tuple[tuple[int, int], str], _WorkspaceAuthorityLock] = {}


def _serialize_workspace_authority(method):
    @wraps(method)
    def guarded(self, workspace_id: str, *args, **kwargs):
        with self._workspace_authority_lock(workspace_id):
            return method(self, workspace_id, *args, **kwargs)

    return guarded


def _serialize_job_authority(method):
    @wraps(method)
    def guarded(self, job_id: str, *args, **kwargs):
        workspace_id = self._job_workspace_id(job_id)
        with self._workspace_authority_lock(workspace_id):
            return method(self, job_id, *args, **kwargs)

    return guarded


def _serialize_cleanup_authority(method):
    @wraps(method)
    def guarded(self, cleanup_id: str, *args, **kwargs):
        workspace_id = self._cleanup_workspace_id(cleanup_id)
        if workspace_id is None:
            return method(self, cleanup_id, *args, **kwargs)
        with self._workspace_authority_lock(workspace_id):
            return method(self, cleanup_id, *args, **kwargs)

    return guarded


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


class TransmissionAuthorityRevokedError(RuntimeError):
    pass


class TransmissionAuthorityReentrancyError(RuntimeError):
    pass


class _WorkspaceAuthorityLock:
    """A same-thread fail-fast lock shared by threads and local processes."""

    def __init__(
        self,
        process_key: str,
    ) -> None:
        self._process_key = process_key
        self._thread_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._owner_thread_id: int | None = None
        self._reentrant_depth = 0
        self._handle = None

    def __enter__(self):
        thread_id = threading.get_ident()
        with self._state_lock:
            if self._owner_thread_id == thread_id:
                raise TransmissionAuthorityReentrancyError(
                    "Workspace transmission authority cannot be mutated reentrantly."
                )
        self._thread_lock.acquire()
        with self._state_lock:
            self._owner_thread_id = thread_id
            self._reentrant_depth = 1
        try:
            self._handle = self._acquire_process_lock()
            return self
        except BaseException:
            with self._state_lock:
                self._owner_thread_id = None
                self._reentrant_depth = 0
            self._thread_lock.release()
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        handle = self._handle
        self._handle = None
        try:
            if handle is not None:
                self._release_process_lock(handle)
        finally:
            with self._state_lock:
                self._owner_thread_id = None
                self._reentrant_depth = 0
            self._thread_lock.release()

    @contextmanager
    def reentrant(self) -> Iterator[_WorkspaceAuthorityLock]:
        thread_id = threading.get_ident()
        with self._state_lock:
            nested = self._owner_thread_id == thread_id
            if nested:
                self._reentrant_depth += 1
        if not nested:
            with self:
                yield self
            return
        try:
            yield self
        finally:
            with self._state_lock:
                self._reentrant_depth -= 1

    def _acquire_process_lock(self):
        if os.name == "nt":
            return self._acquire_windows_mutex()
        return self._acquire_posix_file_lock()

    def _acquire_windows_mutex(self):
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = (
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        )
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateMutexW(
            None,
            False,
            f"Local\\ExamSageAuthority-{self._process_key}",
        )
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        result = kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
        if result not in {0x00000000, 0x00000080}:
            kernel32.CloseHandle(handle)
            raise RuntimeError("Workspace authority mutex acquisition failed.")
        return handle

    def _acquire_posix_file_lock(self) -> int:
        import fcntl

        user_id = os.getuid()
        root = Path("/tmp") / f".examsage-authority-{user_id}"
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        root_stat = root.lstat()
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or root_stat.st_uid != user_id
            or stat.S_IMODE(root_stat.st_mode) & 0o077
        ):
            raise RuntimeError("Workspace authority lock directory is unsafe.")
        flags = os.O_CREAT | os.O_RDWR
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        lock_path = root / f"{self._process_key}.lock"
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            lock_stat = os.fstat(descriptor)
            if (
                not stat.S_ISREG(lock_stat.st_mode)
                or lock_stat.st_uid != user_id
                or lock_stat.st_nlink != 1
                or stat.S_IMODE(lock_stat.st_mode) & 0o077
            ):
                raise RuntimeError("Workspace authority lock file is unsafe.")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _release_process_lock(handle) -> None:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.ReleaseMutex.argtypes = (wintypes.HANDLE,)
            kernel32.ReleaseMutex.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            try:
                if not kernel32.ReleaseMutex(handle):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                kernel32.CloseHandle(handle)
        else:
            import fcntl

            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            finally:
                os.close(handle)


def _physical_file_identity(stat_result: os.stat_result) -> tuple[int, int]:
    return int(stat_result.st_dev), int(stat_result.st_ino)


@dataclass(frozen=True)
class CreationRequest:
    idempotency_key: str
    operation_kind: SourceMode
    state: str
    workspace_id: str
    job_id: str | None = None
    safe_error_code: str | None = None


@dataclass(frozen=True)
class TransmissionAuthoritySnapshot:
    workspace: WorkspaceRecord
    approval: ApprovalRecord | None
    revision: ManifestRevision | None


_SCHEMA = (
    """CREATE TABLE workspaces (
      workspace_id TEXT PRIMARY KEY,
      display_name TEXT NOT NULL,
      source_mode TEXT NOT NULL,
      canonical_root TEXT NOT NULL,
      root_device TEXT,
      root_file_id TEXT,
      owned_root_device TEXT,
      owned_root_file_id TEXT,
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
      archive_member_index INTEGER CHECK(archive_member_index IS NULL OR archive_member_index >= 1),
      archive_member_crc32 INTEGER CHECK(archive_member_crc32 IS NULL OR archive_member_crc32 >= 0),
      archive_member_compressed_bytes INTEGER CHECK(archive_member_compressed_bytes IS NULL OR archive_member_compressed_bytes >= 0),
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
      deletion_root_device TEXT,
      deletion_root_file_id TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      CHECK(
        (deletion_root_device IS NULL AND deletion_root_file_id IS NULL) OR
        (deletion_root_device IS NOT NULL AND deletion_root_file_id IS NOT NULL)
      )
    )""",
    """CREATE TABLE workspace_creation_requests (
      idempotency_key TEXT PRIMARY KEY,
      operation_kind TEXT NOT NULL,
      state TEXT NOT NULL,
      workspace_id TEXT NOT NULL,
      job_id TEXT,
      safe_error_code TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )""",
    """CREATE INDEX idx_manifest_revisions_workspace_created
      ON manifest_revisions(workspace_id, created_at)""",
    """CREATE INDEX idx_manifest_entries_revision_state
      ON manifest_entries(revision_id, state, relative_path)""",
    """CREATE INDEX idx_workspace_events_job_sequence
      ON workspace_events(job_id, sequence)""",
    """CREATE UNIQUE INDEX idx_workspace_jobs_one_active
      ON workspace_jobs(workspace_id) WHERE status IN ('queued', 'running')""",
    """CREATE UNIQUE INDEX idx_cleanup_owned_path
      ON cleanup_queue(owned_relative_path)""",
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
            self._authority_scope = _physical_file_identity(
                os.stat(self.database_path)
            )
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

    def _workspace_authority_lock(self, workspace_id: str) -> _WorkspaceAuthorityLock:
        key = (self._authority_scope, workspace_id)
        with _AUTHORITY_LOCKS_GUARD:
            process_key = hashlib.sha256(
                (
                    f"{self._authority_scope[0]}:{self._authority_scope[1]}"
                    f"\0{workspace_id}"
                ).encode("utf-8")
            ).hexdigest()
            return _AUTHORITY_LOCKS.setdefault(
                key,
                _WorkspaceAuthorityLock(process_key),
            )

    @contextmanager
    def hold_workspace_authority(self, workspace_id: str) -> Iterator[None]:
        """Serialize a workspace authority mutation without reading source data."""
        with self._workspace_authority_lock(workspace_id):
            yield

    def _job_workspace_id(self, job_id: str) -> str:
        with self._lock:
            row = self._job_row(self._connection, job_id)
        return str(row["workspace_id"])

    def _cleanup_workspace_id(self, cleanup_id: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT workspace_id FROM cleanup_queue WHERE cleanup_id = ?",
                (cleanup_id,),
            ).fetchone()
        return None if row is None else str(row["workspace_id"])

    def _migrate(self) -> None:
        with self._transaction() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > 3:
                raise RuntimeError("The workspace database schema is newer than this application.")
            if version == 0:
                for statement in _SCHEMA:
                    connection.execute(statement)
                connection.execute("PRAGMA user_version=3")
            elif version == 1:
                connection.execute(
                    "ALTER TABLE workspaces ADD COLUMN owned_root_device TEXT"
                )
                connection.execute(
                    "ALTER TABLE workspaces ADD COLUMN owned_root_file_id TEXT"
                )
                connection.execute(
                    "ALTER TABLE cleanup_queue ADD COLUMN deletion_root_device TEXT"
                )
                connection.execute(
                    "ALTER TABLE cleanup_queue ADD COLUMN deletion_root_file_id TEXT"
                )
                connection.execute(
                    """CREATE TABLE workspace_creation_requests (
                      idempotency_key TEXT PRIMARY KEY,
                      operation_kind TEXT NOT NULL,
                      state TEXT NOT NULL,
                      workspace_id TEXT NOT NULL,
                      job_id TEXT,
                      safe_error_code TEXT,
                      created_at TEXT NOT NULL,
                      updated_at TEXT NOT NULL
                    )"""
                )
                connection.execute("PRAGMA user_version=2")
                version = 2
            if version == 2:
                connection.execute(
                    "ALTER TABLE manifest_entries ADD COLUMN archive_member_index INTEGER"
                )
                connection.execute(
                    "ALTER TABLE manifest_entries ADD COLUMN archive_member_crc32 INTEGER"
                )
                connection.execute(
                    "ALTER TABLE manifest_entries ADD COLUMN archive_member_compressed_bytes INTEGER"
                )
                connection.execute("PRAGMA user_version=3")
            now = self._timestamp(self._now())
            duplicate_active = connection.execute(
                """WITH ranked AS (
                       SELECT rowid,
                              ROW_NUMBER() OVER (
                                PARTITION BY workspace_id
                                ORDER BY created_at ASC, rowid ASC
                              ) AS active_rank
                       FROM workspace_jobs
                       WHERE status IN ('queued', 'running')
                   )
                   SELECT jobs.job_id
                   FROM workspace_jobs AS jobs
                   JOIN ranked ON ranked.rowid = jobs.rowid
                   WHERE ranked.active_rank > 1"""
            ).fetchall()
            for duplicate in duplicate_active:
                job_id = str(duplicate["job_id"])
                connection.execute(
                    """UPDATE workspace_jobs
                       SET status = ?, safe_error_code = ?, finished_at = ?
                       WHERE job_id = ?""",
                    (
                        WorkspaceJobStatus.FAILED.value,
                        "superseded_active_job",
                        now,
                        job_id,
                    ),
                )
                self._insert_event(
                    connection,
                    job_id=job_id,
                    event_type="failed",
                    message="Workspace scan failed.",
                    payload={"safe_error_code": "superseded_active_job"},
                    created_at=self._now(),
                )
            connection.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_workspace_jobs_one_active
                   ON workspace_jobs(workspace_id)
                   WHERE status IN ('queued', 'running')"""
            )
            connection.execute(
                """DELETE FROM cleanup_queue
                   WHERE rowid NOT IN (
                     SELECT MIN(rowid) FROM cleanup_queue GROUP BY owned_relative_path
                   )"""
            )
            connection.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_cleanup_owned_path
                   ON cleanup_queue(owned_relative_path)"""
            )
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
                   archive_parent_entry_id, archive_member_path, archive_member_index,
                   archive_member_crc32, archive_member_compressed_bytes
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                item.archive_member_index,
                item.archive_member_crc32,
                item.archive_member_compressed_bytes,
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

    @staticmethod
    def _creation_request(row: sqlite3.Row) -> CreationRequest:
        return CreationRequest(
            idempotency_key=str(row["idempotency_key"]),
            operation_kind=SourceMode(row["operation_kind"]),
            state=str(row["state"]),
            workspace_id=str(row["workspace_id"]),
            job_id=str(row["job_id"]) if row["job_id"] is not None else None,
            safe_error_code=(
                str(row["safe_error_code"])
                if row["safe_error_code"] is not None
                else None
            ),
        )

    def claim_creation(
        self,
        idempotency_key: str,
        operation_kind: SourceMode,
        workspace_id: str,
    ) -> tuple[CreationRequest, bool]:
        now = self._now()
        with self._transaction() as connection:
            existing = connection.execute(
                """SELECT * FROM workspace_creation_requests
                   WHERE idempotency_key = ?""",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                request = self._creation_request(existing)
                if request.operation_kind is not operation_kind:
                    raise ActiveWorkspaceOperationError(
                        "The creation idempotency key belongs to another operation."
                    )
                return request, False
            connection.execute(
                """INSERT INTO workspace_creation_requests(
                       idempotency_key, operation_kind, state, workspace_id,
                       job_id, safe_error_code, created_at, updated_at
                   ) VALUES (?, ?, 'claimed', ?, NULL, NULL, ?, ?)""",
                (
                    idempotency_key,
                    operation_kind.value,
                    workspace_id,
                    self._timestamp(now),
                    self._timestamp(now),
                ),
            )
            row = connection.execute(
                """SELECT * FROM workspace_creation_requests
                   WHERE idempotency_key = ?""",
                (idempotency_key,),
            ).fetchone()
            if row is None:
                raise RuntimeError("The creation request could not be read back.")
            return self._creation_request(row), True

    def create_workspace_with_initial_job(
        self,
        workspace: WorkspaceRecord,
        job: WorkspaceJob,
        idempotency_key: str,
        *,
        owned_root_identity: tuple[str, str] | None = None,
    ) -> tuple[WorkspaceRecord, WorkspaceJob, bool]:
        if (
            job.workspace_id != workspace.workspace_id
            or job.job_kind != "initial_scan"
            or job.status is not WorkspaceJobStatus.QUEUED
        ):
            raise ValueError("initial workspace job identity is invalid")
        owned_device, owned_file = owned_root_identity or (None, None)
        with self._transaction() as connection:
            request_row = connection.execute(
                """SELECT * FROM workspace_creation_requests
                   WHERE idempotency_key = ?""",
                (idempotency_key,),
            ).fetchone()
            if request_row is None:
                raise ActiveWorkspaceOperationError(
                    "The creation idempotency key was not claimed."
                )
            request = self._creation_request(request_row)
            if request.state == "completed":
                existing_workspace = self._workspace_row(
                    connection, request.workspace_id
                )
                if request.job_id is None:
                    raise RuntimeError("The completed creation request has no job.")
                return (
                    self._workspace(existing_workspace),
                    self._job(self._job_row(connection, request.job_id)),
                    False,
                )
            if request.state != "claimed" or request.workspace_id != workspace.workspace_id:
                raise ActiveWorkspaceOperationError(
                    "The creation request cannot be completed."
                )
            connection.execute(
                """INSERT INTO workspaces(
                       workspace_id, display_name, source_mode, canonical_root,
                       root_device, root_file_id, owned_root_device,
                       owned_root_file_id, state, current_draft_revision_id,
                       current_approved_revision_id, created_at, updated_at,
                       last_scanned_at, last_access_verified_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    workspace.workspace_id,
                    workspace.display_name,
                    workspace.source_mode.value,
                    str(workspace.canonical_root),
                    workspace.root_device,
                    workspace.root_file_id,
                    owned_device,
                    owned_file,
                    workspace.state.value,
                    workspace.current_draft_revision_id,
                    workspace.current_approved_revision_id,
                    self._timestamp(workspace.created_at),
                    self._timestamp(workspace.updated_at),
                    self._timestamp(workspace.last_scanned_at),
                    self._timestamp(workspace.last_access_verified_at),
                ),
            )
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
            connection.execute(
                """UPDATE workspace_creation_requests
                   SET state = 'completed', job_id = ?, updated_at = ?
                   WHERE idempotency_key = ? AND state = 'claimed'""",
                (job.job_id, self._timestamp(self._now()), idempotency_key),
            )
            return (
                self._workspace(self._workspace_row(connection, workspace.workspace_id)),
                self._job(self._job_row(connection, job.job_id)),
                True,
            )

    def get_creation_request(self, idempotency_key: str) -> CreationRequest | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT * FROM workspace_creation_requests
                   WHERE idempotency_key = ?""",
                (idempotency_key,),
            ).fetchone()
        return self._creation_request(row) if row is not None else None

    def list_recoverable_browser_creations(self) -> Sequence[CreationRequest]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM workspace_creation_requests
                   WHERE operation_kind = ?
                     AND state IN ('claimed', 'interrupted', 'failed')
                   ORDER BY created_at ASC, rowid ASC""",
                (SourceMode.BROWSER_SNAPSHOT.value,),
            ).fetchall()
        return tuple(self._creation_request(row) for row in rows)

    def get_creation_job(self, idempotency_key: str) -> WorkspaceJob | None:
        request = self.get_creation_request(idempotency_key)
        if request is None or request.state != "completed" or request.job_id is None:
            return None
        return self.get_job(request.job_id)

    def finish_cancelled_creation(self, idempotency_key: str) -> None:
        with self._transaction() as connection:
            connection.execute(
                """UPDATE workspace_creation_requests
                   SET state = 'cancelled', updated_at = ?
                   WHERE idempotency_key = ? AND state = 'claimed'""",
                (self._timestamp(self._now()), idempotency_key),
            )

    def fail_creation(self, idempotency_key: str, code: str) -> None:
        self._safe_code(code)
        with self._transaction() as connection:
            connection.execute(
                """UPDATE workspace_creation_requests
                   SET state = 'failed', safe_error_code = ?, updated_at = ?
                   WHERE idempotency_key = ? AND state = 'claimed'""",
                (code, self._timestamp(self._now()), idempotency_key),
            )

    def interrupt_claimed_creations(self) -> None:
        with self._transaction() as connection:
            connection.execute(
                """UPDATE workspace_creation_requests
                   SET state = 'interrupted', safe_error_code = 'creation_interrupted',
                       updated_at = ? WHERE state = 'claimed'""",
                (self._timestamp(self._now()),),
            )

    def get_owned_root_identity(
        self, workspace_id: str
    ) -> tuple[str, str] | None:
        with self._lock:
            row = self._workspace_row(self._connection, workspace_id)
            device = row["owned_root_device"]
            file_id = row["owned_root_file_id"]
        if device is None or file_id is None:
            return None
        return str(device), str(file_id)

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

    @_serialize_workspace_authority
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

    def require_current_draft(
        self, workspace_id: str, revision_id: str
    ) -> ManifestRevision:
        with self._lock:
            workspace = self._workspace_row(self._connection, workspace_id)
            if workspace["current_draft_revision_id"] != revision_id:
                raise StaleManifestError(
                    f"Manifest revision '{revision_id}' is no longer current."
                )
            return self._revision(self._connection, workspace_id, revision_id)

    @_serialize_workspace_authority
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

        def update_inclusion(item: ManifestEntry) -> ManifestEntry:
            if item.entry_id not in requested:
                return item
            if not included and item.state in {
                SourceState.PENDING_APPROVAL,
                SourceState.APPROVED,
                SourceState.CHANGED,
            }:
                return item.model_copy(
                    update={
                        "state": SourceState.EXCLUDED,
                        "included": False,
                        "inclusion_reason": "user_excluded",
                        "failure_code": None,
                        "safe_message": None,
                    }
                )
            if included and (
                item.inclusion_reason == "user_excluded"
                or item.state is SourceState.CHANGED
            ):
                return item.model_copy(
                    update={
                        "state": SourceState.PENDING_APPROVAL,
                        "included": True,
                        "inclusion_reason": None,
                        "failure_code": None,
                        "safe_message": None,
                    }
                )
            return item.model_copy(update={"included": included})

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
                update_inclusion,
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

    @_serialize_workspace_authority
    def approve(
        self,
        workspace_id: str,
        revision_id: str,
        policy_version: str,
        *,
        verified_at: datetime | None = None,
    ) -> ApprovalRecord:
        approval_id = uuid4().hex
        now = verified_at or self._now()
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
                   SET state = ?, current_approved_revision_id = ?, updated_at = ?,
                       last_access_verified_at = COALESCE(?, last_access_verified_at)
                   WHERE workspace_id = ?""",
                (
                    WorkspaceState.APPROVED.value,
                    revision_id,
                    self._timestamp(now),
                    self._timestamp(verified_at),
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

    def transmission_authority_snapshot(
        self,
        workspace_id: str,
    ) -> TransmissionAuthoritySnapshot | None:
        """Load workspace state, current approval, and its revision atomically."""
        with self._transaction() as connection:
            return self._transmission_authority_snapshot(connection, workspace_id)

    @contextmanager
    def hold_transmission_authority(
        self,
        workspace_id: str,
        *,
        approval_id: str,
        revision_id: str,
        verified_at: datetime | None = None,
    ) -> Iterator[TransmissionAuthoritySnapshot]:
        """Hold current approval authority stable through a bounded source read."""
        with self._workspace_authority_lock(workspace_id).reentrant():
            with self._transaction() as connection:
                snapshot = self._transmission_authority_snapshot(
                    connection,
                    workspace_id,
                )
                if (
                    snapshot is None
                    or snapshot.approval is None
                    or snapshot.revision is None
                    or snapshot.workspace.state is not WorkspaceState.APPROVED
                    or snapshot.workspace.current_draft_revision_id != revision_id
                    or snapshot.workspace.current_approved_revision_id != revision_id
                    or snapshot.approval.approval_id != approval_id
                    or snapshot.approval.revision_id != revision_id
                    or snapshot.revision.revision_id != revision_id
                ):
                    raise TransmissionAuthorityRevokedError(
                        "Workspace source approval is no longer current."
                    )
                if verified_at is not None:
                    connection.execute(
                        """UPDATE workspaces
                           SET last_access_verified_at = ?, updated_at = ?
                           WHERE workspace_id = ?""",
                        (
                            self._timestamp(verified_at),
                            self._timestamp(verified_at),
                            workspace_id,
                        ),
                    )
            yield snapshot

    def _transmission_authority_snapshot(
        self,
        connection: sqlite3.Connection,
        workspace_id: str,
    ) -> TransmissionAuthoritySnapshot | None:
        workspace_row = connection.execute(
            "SELECT * FROM workspaces WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchone()
        if workspace_row is None:
            return None
        workspace = self._workspace(workspace_row)
        revision_id = workspace.current_approved_revision_id
        if revision_id is None:
            return TransmissionAuthoritySnapshot(workspace, None, None)
        approval_row = connection.execute(
            """SELECT * FROM approvals
               WHERE workspace_id = ? AND revision_id = ?
               ORDER BY approved_at DESC, rowid DESC LIMIT 1""",
            (workspace_id, revision_id),
        ).fetchone()
        if approval_row is None:
            return TransmissionAuthoritySnapshot(workspace, None, None)
        return TransmissionAuthoritySnapshot(
            workspace=workspace,
            approval=self._approval(approval_row),
            revision=self._revision(connection, workspace_id, revision_id),
        )

    def create_job(self, job: WorkspaceJob, idempotency_key: str) -> WorkspaceJob:
        try:
            with self._transaction() as connection:
                workspace = self._workspace_row(connection, job.workspace_id)
                self._guard_workspace_accepts_jobs(workspace)
                existing = connection.execute(
                    """SELECT * FROM workspace_jobs
                       WHERE workspace_id = ? AND job_kind = ?
                         AND idempotency_key = ?""",
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
        except sqlite3.IntegrityError:
            with self._lock:
                active = self._connection.execute(
                    """SELECT * FROM workspace_jobs
                       WHERE workspace_id = ? AND status IN ('queued', 'running')
                       ORDER BY created_at ASC, rowid ASC LIMIT 1""",
                    (job.workspace_id,),
                ).fetchone()
            if (
                active is not None
                and active["job_kind"] == job.job_kind
                and active["idempotency_key"] == idempotency_key
            ):
                return self._job(active)
            raise ActiveWorkspaceOperationError(
                f"Workspace '{job.workspace_id}' has an active operation."
            ) from None

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

    def recover_unfinished_jobs(self) -> Sequence[WorkspaceJob]:
        self.recover_running_jobs()
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM workspace_jobs WHERE status = ?
                   ORDER BY created_at ASC, rowid ASC""",
                (WorkspaceJobStatus.QUEUED.value,),
            ).fetchall()
        return tuple(self._job(row) for row in rows)

    def list_queued_jobs(self) -> Sequence[WorkspaceJob]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM workspace_jobs WHERE status = ?
                   ORDER BY created_at ASC, rowid ASC""",
                (WorkspaceJobStatus.QUEUED.value,),
            ).fetchall()
        return tuple(self._job(row) for row in rows)

    @_serialize_job_authority
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
            progress_count = int(
                connection.execute(
                    """SELECT COUNT(*) FROM workspace_events
                       WHERE job_id = ? AND event_type = 'scan_progress'""",
                    (job_id,),
                ).fetchone()[0]
            )
            if progress_count >= 256:
                latest = connection.execute(
                    """SELECT * FROM workspace_events
                       WHERE job_id = ? AND event_type = 'scan_progress'
                       ORDER BY sequence DESC LIMIT 1""",
                    (job_id,),
                ).fetchone()
                if latest is None:
                    raise RuntimeError("The capped progress event could not be read.")
                return self._event(latest)
            event = self._insert_event(
                connection,
                job_id=job_id,
                event_type="scan_progress",
                message="Scanning course files.",
                payload=payload,
                created_at=self._now(),
            )
        return event

    @_serialize_job_authority
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

    @_serialize_workspace_authority
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

    @_serialize_workspace_authority
    def mark_revision_attention_if_current(
        self,
        workspace_id: str,
        revision_id: str,
        entry_ids: Sequence[str],
        code: str,
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
            revision = self._revision(connection, workspace_id, revision_id)
            known = {item.entry_id for item in revision.entries}
            if not requested <= known:
                raise ManifestNotFoundError("Manifest entries were not found.")

            def mark_changed(item: ManifestEntry) -> ManifestEntry:
                if item.entry_id not in requested:
                    return item
                return item.model_copy(
                    update={
                        "state": SourceState.CHANGED,
                        "included": False,
                        "inclusion_reason": code,
                        "failure_code": code,
                        "safe_message": (
                            "The selected source changed and needs review."
                        ),
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
            return self._revision(connection, workspace_id, new_revision_id)

    @_serialize_workspace_authority
    def mark_entries_changed_latest(
        self,
        workspace_id: str,
        entry_ids: Sequence[str],
        code: str,
    ) -> ManifestRevision:
        """Atomically merge changed entries into the latest current draft."""
        requested = set(entry_ids)
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
            known = {item.entry_id for item in revision.entries}
            if not requested <= known:
                raise ManifestNotFoundError("Manifest entries were not found.")

            def mark_changed(item: ManifestEntry) -> ManifestEntry:
                if item.entry_id not in requested:
                    return item
                return item.model_copy(
                    update={
                        "state": SourceState.CHANGED,
                        "included": False,
                        "inclusion_reason": code,
                        "failure_code": code,
                        "safe_message": (
                            "The selected source changed and needs review."
                        ),
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
            return self._revision(connection, workspace_id, new_revision_id)

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

    @_serialize_workspace_authority
    def mark_deleting(self, workspace_id: str) -> WorkspaceRecord:
        return self._mark_deleting(workspace_id)

    @_serialize_workspace_authority
    def mark_deleting_if_settled(
        self,
        workspace_id: str,
        has_unsettled_runs: Callable[[], bool],
    ) -> WorkspaceRecord:
        if has_unsettled_runs():
            raise ActiveWorkspaceOperationError(
                f"Workspace '{workspace_id}' has unsettled Agent runs."
            )
        return self._mark_deleting(workspace_id)

    def _mark_deleting(self, workspace_id: str) -> WorkspaceRecord:
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

    @_serialize_workspace_authority
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

    def queue_cleanup(
        self,
        workspace_id: str,
        owned_path: Path,
        code: str,
        *,
        deletion_root_identity: tuple[str, str] | None = None,
    ) -> CleanupRecord:
        relative_path, absolute_path = self._owned_relative_path(owned_path)
        deletion_device, deletion_file = deletion_root_identity or (None, None)
        if any(
            value is not None and len(value) > 128
            for value in (deletion_device, deletion_file)
        ):
            raise ValueError("cleanup root identity is too long")
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
            existing = connection.execute(
                "SELECT * FROM cleanup_queue WHERE owned_relative_path = ?",
                (relative_path,),
            ).fetchone()
            if existing is not None:
                existing_identity = (
                    existing["deletion_root_device"],
                    existing["deletion_root_file_id"],
                )
                requested_identity = (deletion_device, deletion_file)
                if existing_identity != requested_identity:
                    raise ValueError("cleanup root identity cannot be replaced")
                return self._cleanup(existing)
            cleanup_id = uuid4().hex
            connection.execute(
                """INSERT INTO cleanup_queue(
                       cleanup_id, workspace_id, owned_relative_path, safe_error_code,
                       attempt_count, deletion_root_device, deletion_root_file_id,
                       created_at, updated_at
                   ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)""",
                (
                    cleanup_id,
                    workspace_id,
                    relative_path,
                    code,
                    deletion_device,
                    deletion_file,
                    self._timestamp(now),
                    self._timestamp(now),
                ),
            )
            row = connection.execute(
                "SELECT * FROM cleanup_queue WHERE cleanup_id = ?", (cleanup_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError("The cleanup record could not be read back.")
            return self._cleanup(row)

    def list_cleanup(self) -> Sequence[CleanupRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM cleanup_queue ORDER BY created_at ASC, rowid ASC"
            ).fetchall()
        return tuple(self._cleanup(row) for row in rows)

    @_serialize_cleanup_authority
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

    @_serialize_cleanup_authority
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
