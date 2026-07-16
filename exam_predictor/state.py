"""Local-only persistence for courses and conversations.

API keys and raw file contents are intentionally absent from this database.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CourseStore:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "examsage.sqlite3"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS courses (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    workspace_path TEXT NOT NULL,
                    report_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    course_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(course_id) REFERENCES courses(id) ON DELETE CASCADE
                );
            """)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def save_course(
        self,
        name: str,
        provider: str,
        workspace_path: str | Path,
        report: dict[str, Any],
        course_id: str | None = None,
    ) -> str:
        course_id = course_id or uuid.uuid4().hex
        now = self._now()
        with self._connect() as db:
            db.execute(
                """INSERT INTO courses(id, name, provider, workspace_path, report_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     name=excluded.name, provider=excluded.provider,
                     workspace_path=excluded.workspace_path,
                     report_json=excluded.report_json, updated_at=excluded.updated_at""",
                (
                    course_id,
                    name,
                    provider,
                    str(Path(workspace_path).resolve()),
                    json.dumps(report, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return course_id

    def list_courses(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, name, provider, workspace_path, created_at, updated_at FROM courses ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_course(self, course_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM courses WHERE id = ?", (course_id,)).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["report"] = json.loads(item.pop("report_json") or "{}")
        return item

    def add_message(self, course_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant")
        with self._connect() as db:
            db.execute(
                "INSERT INTO messages(course_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (course_id, role, content, self._now()),
            )

    def messages(self, course_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT role, content, created_at FROM (
                       SELECT id, role, content, created_at FROM messages
                       WHERE course_id = ? ORDER BY id DESC LIMIT ?
                   ) ORDER BY id ASC""",
                (course_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_course(self, course_id: str) -> Path | None:
        course = self.get_course(course_id)
        with self._connect() as db:
            db.execute("DELETE FROM courses WHERE id = ?", (course_id,))
        return Path(course["workspace_path"]) if course else None
