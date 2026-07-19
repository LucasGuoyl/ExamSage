from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import AgentEvent, EventType, RunSnapshot, RunStatus


class RuntimeStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
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
                """
            )

    @staticmethod
    def _run(row: sqlite3.Row) -> RunSnapshot:
        return RunSnapshot.model_validate(dict(row))

    @staticmethod
    def _event(row: sqlite3.Row) -> AgentEvent:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        return AgentEvent.model_validate(item)

    def create_run(
        self,
        thread_id: str,
        provider_profile_id: str,
        message: str,
        status: RunStatus,
    ) -> RunSnapshot:
        run_id = uuid.uuid4().hex
        now = self._now()
        with self._connection() as db:
            db.execute(
                """INSERT INTO agent_runs(
                       run_id, thread_id, provider_profile_id, message,
                       status, error, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?)""",
                (run_id, thread_id, provider_profile_id, message, status.value, now, now),
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

    def recover_unfinished(self) -> list[str]:
        unfinished = (RunStatus.RUNNING.value, RunStatus.STOPPING.value)
        recovered: list[str] = []
        now = self._now()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                """SELECT run_id FROM agent_runs
                   WHERE status IN (?, ?)
                   ORDER BY created_at ASC, rowid ASC""",
                unfinished,
            ).fetchall()
            for row in rows:
                run_id = str(row["run_id"])
                db.execute(
                    """UPDATE agent_runs
                       SET status = ?, error = NULL, updated_at = ?
                       WHERE run_id = ?""",
                    (RunStatus.PAUSED.value, now, run_id),
                )
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
