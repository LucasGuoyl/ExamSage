from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    AgentEvent,
    EventType,
    ProviderConfigurationError,
    ProviderProfile,
    RunSnapshot,
    RunStatus,
    SavedProviderProfile,
    validate_provider_profile,
)


class RuntimeStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
        except BaseException:
            connection.close()
            raise
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with closing(self._connect()) as db, db:
            yield db

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _initialize(self) -> None:
        with self._connection() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                  run_id TEXT PRIMARY KEY,
                  thread_id TEXT NOT NULL,
                  provider_profile_id TEXT NOT NULL,
                  workspace_id TEXT,
                  message TEXT NOT NULL,
                  status TEXT NOT NULL,
                  error TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_agent_runs_thread_status
                  ON agent_runs(thread_id, status, created_at);
                CREATE INDEX IF NOT EXISTS idx_agent_runs_status_created_at
                  ON agent_runs(status, created_at);
                CREATE TABLE IF NOT EXISTS agent_events (
                  sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  stage TEXT NOT NULL,
                  message TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_agent_events_run_sequence
                  ON agent_events(run_id, sequence);
                CREATE TABLE IF NOT EXISTS saved_provider_profiles (
                  profile_id TEXT PRIMARY KEY,
                  profile_json TEXT NOT NULL,
                  capabilities_json TEXT NOT NULL,
                  credential_expected INTEGER NOT NULL CHECK(credential_expected IN (0, 1)),
                  reconnect_required INTEGER NOT NULL CHECK(reconnect_required IN (0, 1)),
                  updated_at TEXT NOT NULL
                );
                """
            )
            columns = {
                str(row["name"])
                for row in db.execute("PRAGMA table_info(agent_runs)")
            }
            if "workspace_id" not in columns:
                db.execute("ALTER TABLE agent_runs ADD COLUMN workspace_id TEXT")
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_runs_workspace_status "
                "ON agent_runs(workspace_id, status, created_at)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_runs_workspace_thread "
                "ON agent_runs(workspace_id, thread_id)"
            )

    @staticmethod
    def _run(row: sqlite3.Row) -> RunSnapshot:
        return RunSnapshot.model_validate(dict(row))

    @staticmethod
    def _event(row: sqlite3.Row) -> AgentEvent:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        return AgentEvent.model_validate(item)

    @staticmethod
    def _saved_provider_profile(row: sqlite3.Row) -> SavedProviderProfile:
        profile = ProviderProfile.model_validate_json(row["profile_json"])
        if profile.profile_id != row["profile_id"]:
            raise ValueError("Saved provider profile identity mismatch.")
        profile = validate_provider_profile(profile)
        capabilities = json.loads(row["capabilities_json"])
        return SavedProviderProfile.model_validate(
            {
                "profile": profile,
                "capabilities": capabilities,
                "credential_expected": bool(row["credential_expected"]),
                "reconnect_required": bool(row["reconnect_required"]),
                "updated_at": row["updated_at"],
            }
        )

    def save_provider_profile(self, profile: SavedProviderProfile) -> None:
        validated = validate_provider_profile(profile.profile)
        profile = profile.model_copy(update={"profile": validated})
        profile_json = validated.model_dump_json()
        capabilities_json = json.dumps(
            profile.capabilities,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._connection() as db:
            db.execute(
                """INSERT INTO saved_provider_profiles(
                       profile_id, profile_json, capabilities_json,
                       credential_expected, reconnect_required, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(profile_id) DO UPDATE SET
                       profile_json = excluded.profile_json,
                       capabilities_json = excluded.capabilities_json,
                       credential_expected = excluded.credential_expected,
                       reconnect_required = excluded.reconnect_required,
                       updated_at = excluded.updated_at""",
                (
                    profile.profile.profile_id,
                    profile_json,
                    capabilities_json,
                    int(profile.credential_expected),
                    int(profile.reconnect_required),
                    profile.updated_at.isoformat(),
                ),
            )

    def list_saved_provider_profiles(self) -> list[SavedProviderProfile]:
        with self._connection() as db:
            rows = db.execute(
                """SELECT profile_id, profile_json, capabilities_json, credential_expected,
                          reconnect_required, updated_at
                   FROM saved_provider_profiles
                   ORDER BY profile_id ASC"""
            ).fetchall()
        profiles: list[SavedProviderProfile] = []
        for row in rows:
            try:
                profiles.append(self._saved_provider_profile(row))
            except (ValueError, TypeError, ProviderConfigurationError):
                continue
        return profiles

    def mark_provider_reconnect_required(self, profile_id: str) -> None:
        with self._connection() as db:
            cursor = db.execute(
                """UPDATE saved_provider_profiles
                   SET credential_expected = 0,
                       reconnect_required = 1,
                       updated_at = ?
                   WHERE profile_id = ?""",
                (self._now(), profile_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Saved provider profile '{profile_id}' was not found.")

    def create_run(
        self,
        thread_id: str,
        provider_profile_id: str,
        message: str,
        status: RunStatus,
        workspace_id: str | None = None,
    ) -> RunSnapshot:
        if (
            workspace_id is not None
            and thread_id != f"workspace:{workspace_id}"
        ):
            raise ValueError(
                "Workspace checkpoint thread does not match its workspace."
            )
        run_id = uuid.uuid4().hex
        now = self._now()
        with self._connection() as db:
            db.execute(
                """INSERT INTO agent_runs(
                       run_id, thread_id, provider_profile_id, workspace_id,
                       message, status, error, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)""",
                (
                    run_id,
                    thread_id,
                    provider_profile_id,
                    workspace_id,
                    message,
                    status.value,
                    now,
                    now,
                ),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> RunSnapshot:
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Agent run '{run_id}' was not found.")
        return self._run(row)

    def active_run(self) -> RunSnapshot | None:
        statuses = (RunStatus.RUNNING.value, RunStatus.STOPPING.value, RunStatus.PAUSED.value)
        with self._connection() as db:
            row = db.execute(
                """SELECT * FROM agent_runs
                   WHERE status IN (?, ?, ?)
                   ORDER BY created_at ASC, rowid ASC LIMIT 1""",
                statuses,
            ).fetchone()
        return self._run(row) if row else None

    def next_queued_run(self) -> RunSnapshot | None:
        with self._connection() as db:
            row = db.execute(
                """SELECT * FROM agent_runs
                   WHERE status = ?
                   ORDER BY created_at ASC, rowid ASC LIMIT 1""",
                (RunStatus.QUEUED.value,),
            ).fetchone()
        return self._run(row) if row else None

    def set_status(
        self,
        run_id: str,
        status: RunStatus,
        error: str | None = None,
    ) -> RunSnapshot:
        with self._connection() as db:
            cursor = db.execute(
                "UPDATE agent_runs SET status = ?, error = ?, updated_at = ? WHERE run_id = ?",
                (status.value, error, self._now(), run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Agent run '{run_id}' was not found.")
        return self.get_run(run_id)

    def append_event(
        self,
        run_id: str,
        event_type: EventType,
        stage: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> AgentEvent:
        payload_json = json.dumps(payload or {}, ensure_ascii=False, allow_nan=False)
        with self._connection() as db:
            cursor = db.execute(
                """INSERT INTO agent_events(
                       run_id, event_type, stage, message, payload_json, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, event_type.value, stage, message, payload_json, self._now()),
            )
            row = db.execute(
                "SELECT * FROM agent_events WHERE sequence = ?", (cursor.lastrowid,)
            ).fetchone()
        if row is None:
            raise RuntimeError("The appended agent event could not be read back.")
        return self._event(row)

    def set_status_and_append_event(
        self,
        run_id: str,
        status: RunStatus,
        event_type: EventType,
        stage: str,
        message: str,
        payload: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> tuple[RunSnapshot, AgentEvent]:
        payload_json = json.dumps(payload or {}, ensure_ascii=False, allow_nan=False)
        now = self._now()
        with self._connection() as db:
            updated = db.execute(
                "UPDATE agent_runs SET status = ?, error = ?, updated_at = ? WHERE run_id = ?",
                (status.value, error, now, run_id),
            )
            if updated.rowcount != 1:
                raise KeyError(f"Agent run '{run_id}' was not found.")
            inserted = db.execute(
                """INSERT INTO agent_events(
                       run_id, event_type, stage, message, payload_json, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, event_type.value, stage, message, payload_json, now),
            )
            run_row = db.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            event_row = db.execute(
                "SELECT * FROM agent_events WHERE sequence = ?", (inserted.lastrowid,)
            ).fetchone()
        if run_row is None or event_row is None:
            raise RuntimeError("The agent run transition could not be read back.")
        return self._run(run_row), self._event(event_row)

    def list_events(self, run_id: str, after: int = 0) -> list[AgentEvent]:
        with self._connection() as db:
            rows = db.execute(
                """SELECT * FROM agent_events
                   WHERE run_id = ? AND sequence > ? ORDER BY sequence ASC""",
                (run_id, after),
            ).fetchall()
        return [self._event(row) for row in rows]

    def list_by_status(self, statuses: set[RunStatus]) -> list[RunSnapshot]:
        if not statuses:
            return []
        values = sorted(status.value for status in statuses)
        placeholders = ",".join("?" for _ in values)
        with self._connection() as db:
            rows = db.execute(
                f"""SELECT * FROM agent_runs
                    WHERE status IN ({placeholders})
                    ORDER BY created_at ASC, rowid ASC""",
                values,
            ).fetchall()
        return [self._run(row) for row in rows]

    def list_for_workspace(
        self,
        workspace_id: str,
        *,
        statuses: set[RunStatus] | None = None,
    ) -> list[RunSnapshot]:
        values = sorted(status.value for status in statuses or ())
        status_clause = ""
        parameters: list[str] = [workspace_id]
        if statuses is not None:
            if not values:
                return []
            placeholders = ",".join("?" for _ in values)
            status_clause = f" AND status IN ({placeholders})"
            parameters.extend(values)
        with self._connection() as db:
            rows = db.execute(
                f"""SELECT * FROM agent_runs
                    WHERE workspace_id = ?{status_clause}
                    ORDER BY created_at ASC, rowid ASC""",
                parameters,
            ).fetchall()
        return [self._run(row) for row in rows]

    def has_unsettled_runs(self, workspace_id: str) -> bool:
        statuses = {
            RunStatus.QUEUED,
            RunStatus.RUNNING,
            RunStatus.STOPPING,
            RunStatus.PAUSED,
        }
        return bool(self.list_for_workspace(workspace_id, statuses=statuses))

    def thread_ids_for_workspace(self, workspace_id: str) -> Sequence[str]:
        """Return distinct checkpoint thread IDs in stable order."""
        with self._connection() as db:
            rows = db.execute(
                """SELECT thread_id
                   FROM agent_runs
                   WHERE workspace_id = ?
                   GROUP BY thread_id
                   ORDER BY MIN(created_at) ASC, MIN(rowid) ASC""",
                (workspace_id,),
            ).fetchall()
        return tuple(str(row["thread_id"]) for row in rows)

    def delete_settled_workspace_runs(self, workspace_id: str) -> None:
        """Raise on an unsettled run, otherwise delete linked run/event rows atomically."""
        settled = (RunStatus.COMPLETED.value, RunStatus.FAILED.value)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            unsettled = db.execute(
                """SELECT 1 FROM agent_runs
                   WHERE workspace_id = ? AND status NOT IN (?, ?)
                   LIMIT 1""",
                (workspace_id, *settled),
            ).fetchone()
            if unsettled is not None:
                raise ValueError("Workspace has unsettled Agent runs.")
            db.execute(
                "DELETE FROM agent_runs WHERE workspace_id = ?",
                (workspace_id,),
            )

    def recover_unfinished(self) -> list[str]:
        unfinished = (RunStatus.RUNNING.value, RunStatus.STOPPING.value)
        lifecycle_events = (
            EventType.QUEUED.value,
            EventType.STARTED.value,
            EventType.STOP_REQUESTED.value,
            EventType.PAUSED.value,
            EventType.RESUMED.value,
            EventType.COMPLETED.value,
            EventType.FAILED.value,
        )
        recovered: list[str] = []
        now = self._now()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                """SELECT runs.run_id,
                          (
                            SELECT events.event_type
                            FROM agent_events AS events
                            WHERE events.run_id = runs.run_id
                              AND events.event_type IN (?, ?, ?, ?, ?, ?, ?)
                            ORDER BY events.sequence DESC
                            LIMIT 1
                          ) AS last_lifecycle_event
                   FROM agent_runs AS runs
                   WHERE runs.status IN (?, ?)
                   ORDER BY runs.created_at ASC, runs.rowid ASC""",
                (*lifecycle_events, *unfinished),
            ).fetchall()
            for row in rows:
                run_id = str(row["run_id"])
                db.execute(
                    """UPDATE agent_runs
                       SET status = ?, error = NULL, updated_at = ?
                       WHERE run_id = ?""",
                    (RunStatus.PAUSED.value, now, run_id),
                )
                if row["last_lifecycle_event"] != EventType.PAUSED.value:
                    db.execute(
                        """INSERT INTO agent_events(
                               run_id, event_type, stage, message, payload_json, created_at
                           ) VALUES (?, ?, ?, ?, '{}', ?)""",
                        (
                            run_id,
                            EventType.PAUSED.value,
                            "paused",
                            "ExamSage stopped before this run completed. Select Resume to continue.",
                            now,
                        ),
                    )
                recovered.append(run_id)
        return recovered
