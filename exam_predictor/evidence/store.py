from __future__ import annotations

import hashlib
import hmac
import json
import re
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from exam_predictor.evidence.models import (
    CoverageSummary,
    EvidenceUnit,
    PartState,
    SourcePartPlan,
    StudyMapSnapshot,
    validate_safe_evidence_text,
)


_SCHEMA_VERSION = "3"

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS evidence_meta (
      name TEXT PRIMARY KEY,
      value TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS evidence_parts (
      part_id TEXT PRIMARY KEY,
      workspace_id TEXT NOT NULL,
      revision_id TEXT NOT NULL,
      entry_id TEXT NOT NULL,
      source_sha256 TEXT NOT NULL,
      part_sha256 TEXT NOT NULL,
      priority INTEGER NOT NULL CHECK(priority >= 0),
      ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
      state TEXT NOT NULL,
      plan_json TEXT NOT NULL,
      attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
      next_attempt_at TEXT,
      plan_generation INTEGER NOT NULL DEFAULT 1 CHECK(plan_generation >= 1),
      claim_generation INTEGER NOT NULL DEFAULT 0 CHECK(claim_generation >= 0),
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS evidence_attempts (
      attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
      part_id TEXT NOT NULL REFERENCES evidence_parts(part_id) ON DELETE CASCADE,
      attempt_number INTEGER NOT NULL CHECK(attempt_number >= 1),
      route TEXT NOT NULL,
      outcome TEXT NOT NULL,
      safe_error_code TEXT,
      started_at TEXT NOT NULL,
      finished_at TEXT,
      UNIQUE(part_id, attempt_number)
    )""",
    """CREATE TABLE IF NOT EXISTS evidence_units (
      evidence_unit_id TEXT PRIMARY KEY,
      source_part_id TEXT NOT NULL REFERENCES evidence_parts(part_id) ON DELETE CASCADE,
      unit_json TEXT NOT NULL,
      unit_sha256 TEXT NOT NULL,
      published_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS evidence_cache (
      cache_digest TEXT PRIMARY KEY,
      evidence_unit_id TEXT NOT NULL REFERENCES evidence_units(evidence_unit_id) ON DELETE CASCADE,
      unit_sha256 TEXT NOT NULL,
      source_sha256 TEXT NOT NULL,
      plan_generation INTEGER CHECK(plan_generation IS NULL OR plan_generation >= 1),
      claim_generation INTEGER CHECK(claim_generation IS NULL OR claim_generation >= 0),
      created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS study_map_snapshots (
      snapshot_id TEXT PRIMARY KEY,
      workspace_id TEXT NOT NULL,
      revision_id TEXT NOT NULL,
      status TEXT NOT NULL,
      snapshot_json TEXT NOT NULL,
      created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS study_map_dependencies (
      snapshot_id TEXT NOT NULL REFERENCES study_map_snapshots(snapshot_id) ON DELETE CASCADE,
      evidence_unit_id TEXT NOT NULL REFERENCES evidence_units(evidence_unit_id) ON DELETE CASCADE,
      PRIMARY KEY(snapshot_id, evidence_unit_id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_evidence_parts_workspace_revision_state
      ON evidence_parts(workspace_id, revision_id, state, priority)""",
    """CREATE INDEX IF NOT EXISTS idx_evidence_parts_source_sha256
      ON evidence_parts(source_sha256)""",
    """CREATE INDEX IF NOT EXISTS idx_evidence_parts_next_attempt_at
      ON evidence_parts(next_attempt_at)""",
    """CREATE INDEX IF NOT EXISTS idx_evidence_cache_digest
      ON evidence_cache(cache_digest)""",
    """CREATE INDEX IF NOT EXISTS idx_evidence_cache_source_sha256
      ON evidence_cache(source_sha256)""",
    """CREATE INDEX IF NOT EXISTS idx_study_map_snapshots_workspace_revision
      ON study_map_snapshots(workspace_id, revision_id, created_at)""",
)

_REQUIRED_INDEXES = {
    "idx_evidence_parts_workspace_revision_state": (
        ("workspace_id", "revision_id", "state", "priority"),
        """CREATE INDEX idx_evidence_parts_workspace_revision_state
           ON evidence_parts(workspace_id, revision_id, state, priority)""",
    ),
    "idx_evidence_parts_source_sha256": (
        ("source_sha256",),
        """CREATE INDEX idx_evidence_parts_source_sha256
           ON evidence_parts(source_sha256)""",
    ),
    "idx_evidence_parts_next_attempt_at": (
        ("next_attempt_at",),
        """CREATE INDEX idx_evidence_parts_next_attempt_at
           ON evidence_parts(next_attempt_at)""",
    ),
    "idx_evidence_cache_digest": (
        ("cache_digest",),
        "CREATE INDEX idx_evidence_cache_digest ON evidence_cache(cache_digest)",
    ),
    "idx_evidence_cache_source_sha256": (
        ("source_sha256",),
        """CREATE INDEX idx_evidence_cache_source_sha256
           ON evidence_cache(source_sha256)""",
    ),
    "idx_study_map_snapshots_workspace_revision": (
        ("workspace_id", "revision_id", "created_at"),
        """CREATE INDEX idx_study_map_snapshots_workspace_revision
           ON study_map_snapshots(workspace_id, revision_id, created_at)""",
    ),
}


class EvidenceStore:
    def __init__(self, database_path: str | Path) -> None:
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
            self.migrate()
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
                self._connection.commit()
            except BaseException:
                try:
                    self._connection.rollback()
                except BaseException:
                    pass
                raise

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
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _claim_token(
        cls, part_id: str, plan_generation: int, claim_generation: int
    ) -> str:
        identity = f"{part_id}\0{plan_generation}\0{claim_generation}"
        return f"evidence-claim-v1:{cls._digest(identity)}"

    @staticmethod
    def _locator_contains(part_locator: str, citation_locator: str) -> bool:
        part_raw = part_locator.strip()
        citation_raw = citation_locator.strip()
        part_member = re.fullmatch(r"member\s+(.+)", part_raw, re.IGNORECASE)
        if part_member is not None:
            citation_member = re.fullmatch(
                r"member\s+(.+)", citation_raw, re.IGNORECASE
            )
            if citation_member is None:
                return False
            member_name = part_member.group(1)
            citation_body = citation_member.group(1)
            if citation_body == member_name:
                return True
            numeric = re.compile(
                r"^(pages?|slides?|sheets?|rows?)\s+(\d+)"
                r"(?:\s*[-\N{EN DASH}]\s*(\d+))?$",
                re.IGNORECASE,
            )
            for delimiter in (":", ",", ";"):
                prefix = member_name + delimiter
                if citation_body.startswith(prefix):
                    nested = numeric.fullmatch(citation_body[len(prefix) :].strip())
                    if nested is None:
                        return False
                    nested_start = int(nested.group(2))
                    nested_end = int(nested.group(3) or nested.group(2))
                    return nested_start <= nested_end
            return False
        part = part_raw.casefold()
        citation = citation_raw.casefold()
        if part == citation:
            return True
        numeric = re.compile(
            r"^(pages?|slides?|sheets?|rows?)\s+(\d+)"
            r"(?:\s*[-\N{EN DASH}]\s*(\d+))?$"
        )
        part_match = numeric.fullmatch(part)
        citation_match = numeric.fullmatch(citation)
        if part_match is not None and citation_match is not None:
            part_kind = part_match.group(1).removesuffix("s")
            citation_kind = citation_match.group(1).removesuffix("s")
            part_start = int(part_match.group(2))
            part_end = int(part_match.group(3) or part_match.group(2))
            citation_start = int(citation_match.group(2))
            citation_end = int(citation_match.group(3) or citation_match.group(2))
            return (
                part_kind == citation_kind
                and part_start <= citation_start <= citation_end <= part_end
            )
        return False

    @classmethod
    def _validate_serialized_strings(cls, value: object) -> None:
        if isinstance(value, str):
            validate_safe_evidence_text(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                cls._validate_serialized_strings(key)
                cls._validate_serialized_strings(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                cls._validate_serialized_strings(item)

    def migrate(self) -> None:
        with self._transaction() as connection:
            for statement in _SCHEMA:
                connection.execute(statement)
            part_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(evidence_parts)")
            }
            if "plan_generation" not in part_columns:
                connection.execute(
                    """ALTER TABLE evidence_parts ADD COLUMN plan_generation
                       INTEGER NOT NULL DEFAULT 1 CHECK(plan_generation >= 1)"""
                )
            if "claim_generation" not in part_columns:
                connection.execute(
                    """ALTER TABLE evidence_parts ADD COLUMN claim_generation
                       INTEGER NOT NULL DEFAULT 0 CHECK(claim_generation >= 0)"""
                )
            cache_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(evidence_cache)")
            }
            if "plan_generation" not in cache_columns:
                connection.execute(
                    """ALTER TABLE evidence_cache ADD COLUMN plan_generation
                       INTEGER CHECK(plan_generation IS NULL OR plan_generation >= 1)"""
                )
            if "claim_generation" not in cache_columns:
                connection.execute(
                    """ALTER TABLE evidence_cache ADD COLUMN claim_generation
                       INTEGER CHECK(claim_generation IS NULL OR claim_generation >= 0)"""
                )
            for index_name, (expected_columns, statement) in _REQUIRED_INDEXES.items():
                actual_columns = tuple(
                    str(item["name"])
                    for item in connection.execute(
                        f'PRAGMA index_info("{index_name}")'
                    )
                )
                if actual_columns != expected_columns:
                    connection.execute(f'DROP INDEX IF EXISTS "{index_name}"')
                    connection.execute(statement)
            row = connection.execute(
                "SELECT value FROM evidence_meta WHERE name = 'schema_version'"
            ).fetchone()
            if row is not None and int(row["value"]) > int(_SCHEMA_VERSION):
                raise RuntimeError(
                    "The evidence database schema is newer than this application."
                )
            connection.execute(
                """INSERT INTO evidence_meta(name, value) VALUES ('schema_version', ?)
                   ON CONFLICT(name) DO UPDATE SET value = excluded.value""",
                (_SCHEMA_VERSION,),
            )

    @staticmethod
    def _part(row: sqlite3.Row) -> SourcePartPlan:
        value = json.loads(row["plan_json"])
        value["state"] = row["state"]
        return SourcePartPlan.model_validate(value)

    def upsert_part_plans(self, plans: Sequence[SourcePartPlan]) -> None:
        revisions: dict[str, str] = {}
        for plan in plans:
            self._validate_serialized_strings(plan.model_dump(mode="json"))
            if plan.state is PartState.RUNNING:
                raise ValueError(
                    "running parts must be entered via claim_parts or mark_running"
                )
            previous = revisions.setdefault(plan.workspace_id, plan.revision_id)
            if previous != plan.revision_id:
                raise ValueError("one upsert cannot mix revisions for a workspace")
        now = self._timestamp(self._now())
        with self._transaction() as connection:
            for workspace_id, revision_id in revisions.items():
                connection.execute(
                    """INSERT INTO evidence_meta(name, value) VALUES (?, ?)
                       ON CONFLICT(name) DO UPDATE SET value = excluded.value""",
                    (f"current_revision:{self._digest(workspace_id)}", revision_id),
                )
            for plan in plans:
                plan_json = self._canonical_json(plan.model_dump(mode="json"))
                existing = connection.execute(
                    """SELECT workspace_id, revision_id, plan_json
                       FROM evidence_parts WHERE part_id = ?""",
                    (plan.part_id,),
                ).fetchone()
                values = (
                    plan.workspace_id,
                    plan.revision_id,
                    plan.entry_id,
                    plan.source_sha256,
                    plan.part_sha256,
                    plan.priority,
                    plan.ordinal,
                    plan.state.value,
                    plan_json,
                )
                if existing is None:
                    connection.execute(
                        """INSERT INTO evidence_parts(
                               part_id, workspace_id, revision_id, entry_id,
                               source_sha256, part_sha256, priority, ordinal,
                               state, plan_json, attempt_count, next_attempt_at,
                               plan_generation, claim_generation,
                               created_at, updated_at
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 1, 0, ?, ?)""",
                        (plan.part_id, *values, now, now),
                    )
                    continue
                if existing["plan_json"] == plan_json:
                    connection.execute(
                        "UPDATE evidence_parts SET updated_at = ? WHERE part_id = ?",
                        (now, plan.part_id),
                    )
                    continue
                unit_rows = connection.execute(
                    """SELECT evidence_unit_id FROM evidence_units
                       WHERE source_part_id = ?""",
                    (plan.part_id,),
                ).fetchall()
                unit_ids = tuple(str(row["evidence_unit_id"]) for row in unit_rows)
                if unit_ids:
                    placeholders = ",".join("?" for _ in unit_ids)
                    connection.execute(
                        f"""DELETE FROM study_map_snapshots
                            WHERE workspace_id = ? AND revision_id = ?
                              AND snapshot_id IN (
                                SELECT snapshot_id FROM study_map_dependencies
                                WHERE evidence_unit_id IN ({placeholders})
                              )""",
                        (
                            existing["workspace_id"],
                            existing["revision_id"],
                            *unit_ids,
                        ),
                    )
                connection.execute(
                    "DELETE FROM evidence_units WHERE source_part_id = ?",
                    (plan.part_id,),
                )
                connection.execute(
                    "DELETE FROM evidence_attempts WHERE part_id = ?",
                    (plan.part_id,),
                )
                connection.execute(
                    """UPDATE evidence_parts SET
                           workspace_id = ?, revision_id = ?, entry_id = ?,
                           source_sha256 = ?, part_sha256 = ?, priority = ?,
                           ordinal = ?, state = ?, plan_json = ?,
                           attempt_count = 0, next_attempt_at = NULL,
                           plan_generation = plan_generation + 1,
                           created_at = ?, updated_at = ?
                       WHERE part_id = ?""",
                    (*values, now, now, plan.part_id),
                )

    def get_part(self, part_id: str) -> SourcePartPlan:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM evidence_parts WHERE part_id = ?", (part_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Evidence part '{part_id}' was not found.")
        return self._part(row)

    def attempt_count(self, part_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT attempt_count FROM evidence_parts WHERE part_id = ?",
                (part_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Evidence part '{part_id}' was not found.")
        return int(row["attempt_count"])

    def claim_parts(
        self,
        workspace_id: str,
        revision_id: str,
        *,
        limit: int,
        now: datetime,
    ) -> tuple[SourcePartPlan, ...]:
        if limit < 0:
            raise ValueError("limit must not be negative")
        claimed_at = self._timestamp(now)
        if limit == 0:
            return ()
        with self._transaction() as connection:
            current = connection.execute(
                "SELECT value FROM evidence_meta WHERE name = ?",
                (f"current_revision:{self._digest(workspace_id)}",),
            ).fetchone()
            if current is None or current["value"] != revision_id:
                return ()
            rows = connection.execute(
                """SELECT * FROM evidence_parts
                   WHERE workspace_id = ? AND revision_id = ?
                     AND state IN (?, ?, ?, ?)
                     AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                   ORDER BY priority ASC, ordinal ASC, part_id ASC
                   LIMIT ?""",
                (
                    workspace_id,
                    revision_id,
                    PartState.PLANNED.value,
                    PartState.AUTHORIZED.value,
                    PartState.PREPARED.value,
                    PartState.RETRY_WAIT.value,
                    claimed_at,
                    limit,
                ),
            ).fetchall()
            part_ids = tuple(str(row["part_id"]) for row in rows)
            for part_id in part_ids:
                connection.execute(
                    """UPDATE evidence_parts
                       SET state = ?, next_attempt_at = NULL,
                           claim_generation = claim_generation + 1,
                           updated_at = ?
                       WHERE part_id = ?""",
                    (PartState.RUNNING.value, claimed_at, part_id),
                )
            claimed_rows = [
                dict(row) | {"state": PartState.RUNNING.value} for row in rows
            ]
        return tuple(self._part(self._row_from_mapping(row)) for row in claimed_rows)

    @staticmethod
    def _row_from_mapping(value: dict[str, object]) -> sqlite3.Row:
        # _part only relies on mapping access, while this annotation documents
        # that callers pass the same shape as a SQLite row.
        return value  # type: ignore[return-value]

    @staticmethod
    def _attempt_state(value: PartState | str) -> PartState:
        if isinstance(value, PartState):
            return value
        aliases = {
            "success": PartState.PROCESSED,
            "processed": PartState.PROCESSED,
            "retry": PartState.RETRY_WAIT,
            "retry_wait": PartState.RETRY_WAIT,
            "failed": PartState.FAILED,
            "running": PartState.RUNNING,
        }
        try:
            return aliases[value]
        except KeyError:
            raise ValueError("attempt outcome is not supported") from None

    def record_attempt(
        self,
        part_id: str,
        *,
        attempt: int,
        route: str,
        outcome: PartState | str,
        started_at: datetime,
        finished_at: datetime | None = None,
        safe_error_code: str | None = None,
        next_attempt_at: datetime | None = None,
    ) -> SourcePartPlan:
        if attempt < 1:
            raise ValueError("attempt must be at least one")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", route):
            raise ValueError("route must be a bounded stable identifier")
        if safe_error_code is not None and not re.fullmatch(
            r"[a-z][a-z0-9_]{0,119}", safe_error_code
        ):
            raise ValueError("safe_error_code must be a bounded stable code")
        state = self._attempt_state(outcome)
        if state is PartState.RUNNING:
            raise ValueError(
                "running attempts must be entered via claim_parts or mark_running"
            )
        if state is PartState.RETRY_WAIT and next_attempt_at is None:
            raise ValueError("retry attempts require next_attempt_at")
        started = self._timestamp(started_at)
        finished = self._timestamp(finished_at)
        next_attempt = self._timestamp(next_attempt_at)
        updated = finished or started
        with self._transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM evidence_parts WHERE part_id = ?", (part_id,)
            ).fetchone() is None:
                raise KeyError(f"Evidence part '{part_id}' was not found.")
            connection.execute(
                """INSERT INTO evidence_attempts(
                       part_id, attempt_number, route, outcome, safe_error_code,
                       started_at, finished_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    part_id,
                    attempt,
                    route,
                    state.value,
                    safe_error_code,
                    started,
                    finished,
                ),
            )
            connection.execute(
                """UPDATE evidence_parts
                   SET state = ?, attempt_count = MAX(attempt_count, ?),
                       next_attempt_at = ?, updated_at = ?
                   WHERE part_id = ?""",
                (state.value, attempt, next_attempt, updated, part_id),
            )
            row = connection.execute(
                "SELECT * FROM evidence_parts WHERE part_id = ?", (part_id,)
            ).fetchone()
        assert row is not None
        return self._part(row)

    def mark_running(self, part_id: str, *, attempt: int) -> SourcePartPlan:
        if attempt < 1:
            raise ValueError("attempt must be at least one")
        now = self._timestamp(self._now())
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE evidence_parts
                   SET state = ?, attempt_count = MAX(attempt_count, ?),
                       next_attempt_at = NULL,
                       claim_generation = claim_generation + 1,
                       updated_at = ?
                   WHERE part_id = ?""",
                (PartState.RUNNING.value, attempt, now, part_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Evidence part '{part_id}' was not found.")
            row = connection.execute(
                "SELECT * FROM evidence_parts WHERE part_id = ?", (part_id,)
            ).fetchone()
        assert row is not None
        return self._part(row)

    def publication_token(self, part_id: str) -> str:
        with self._lock:
            part = self._connection.execute(
                """SELECT workspace_id, revision_id, state,
                          plan_generation, claim_generation
                   FROM evidence_parts WHERE part_id = ?""",
                (part_id,),
            ).fetchone()
            if part is None:
                raise KeyError(f"Evidence part '{part_id}' was not found.")
            current = self._connection.execute(
                "SELECT value FROM evidence_meta WHERE name = ?",
                (f"current_revision:{self._digest(part['workspace_id'])}",),
            ).fetchone()
        if current is None or current["value"] != part["revision_id"]:
            raise ValueError("evidence part revision is no longer current")
        if part["state"] != PartState.RUNNING.value:
            raise ValueError("publication tokens exist only for a running claim")
        return self._claim_token(
            part_id,
            int(part["plan_generation"]),
            int(part["claim_generation"]),
        )

    @staticmethod
    def _unit(row: sqlite3.Row) -> EvidenceUnit:
        return EvidenceUnit.model_validate_json(row["unit_json"])

    def get_evidence_unit(self, evidence_unit_id: str) -> EvidenceUnit | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT unit_json FROM evidence_units WHERE evidence_unit_id = ?",
                (evidence_unit_id,),
            ).fetchone()
        return None if row is None else self._unit(row)

    def cached_evidence(self, cache_key: str) -> EvidenceUnit | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT units.unit_json
                   FROM evidence_cache AS cache
                   JOIN evidence_units AS units
                     ON units.evidence_unit_id = cache.evidence_unit_id
                   WHERE cache.cache_digest = ?""",
                (self._digest(cache_key),),
            ).fetchone()
        return None if row is None else self._unit(row)

    def get_cached_evidence(self, cache_key: str) -> EvidenceUnit | None:
        return self.cached_evidence(cache_key)

    def publish_evidence(
        self,
        part_id: str,
        unit: EvidenceUnit,
        *,
        cache_key: str,
        claim_token: str,
        completed_at: datetime,
    ) -> bool:
        unit_payload = unit.model_dump(mode="json")
        self._validate_serialized_strings(unit_payload)
        if unit.source_part_id != part_id:
            raise ValueError("evidence unit does not belong to the source part")
        if any(
            citation.evidence_unit_id != unit.evidence_unit_id
            or citation.source_part_id != part_id
            for citation in unit.citations
        ):
            raise ValueError("evidence citations do not match their evidence unit")
        published_at = self._timestamp(completed_at)
        unit_json = self._canonical_json(unit_payload)
        unit_sha256 = self._digest(unit_json)
        cache_digest = self._digest(cache_key)
        with self._transaction() as connection:
            part = connection.execute(
                """SELECT workspace_id, revision_id, source_sha256, state, plan_json,
                          plan_generation, claim_generation
                   FROM evidence_parts WHERE part_id = ?""",
                (part_id,),
            ).fetchone()
            if part is None:
                raise KeyError(f"Evidence part '{part_id}' was not found.")
            current = connection.execute(
                "SELECT value FROM evidence_meta WHERE name = ?",
                (f"current_revision:{self._digest(part['workspace_id'])}",),
            ).fetchone()
            if current is None or current["value"] != part["revision_id"]:
                raise ValueError("evidence part revision is no longer current")
            expected_token = self._claim_token(
                part_id,
                int(part["plan_generation"]),
                int(part["claim_generation"]),
            )
            if not hmac.compare_digest(claim_token, expected_token):
                raise ValueError("claim token does not match the current running claim")
            persisted_plan = SourcePartPlan.model_validate_json(part["plan_json"])
            if any(
                citation.relative_path != persisted_plan.relative_path
                or not self._locator_contains(
                    persisted_plan.locator, citation.locator
                )
                for citation in unit.citations
            ):
                raise ValueError("evidence citation is outside the persisted source part")
            cached = connection.execute(
                """SELECT cache.evidence_unit_id, cache.unit_sha256,
                          cache.source_sha256,
                          cache.plan_generation AS cache_plan_generation,
                          cache.claim_generation AS cache_claim_generation,
                          units.source_part_id,
                          units.unit_sha256 AS persisted_unit_sha256
                   FROM evidence_cache AS cache
                   JOIN evidence_units AS units
                     ON units.evidence_unit_id = cache.evidence_unit_id
                   WHERE cache.cache_digest = ?""",
                (cache_digest,),
            ).fetchone()
            if part["state"] == PartState.PROCESSED.value:
                if (
                    cached is not None
                    and cached["evidence_unit_id"] == unit.evidence_unit_id
                    and cached["unit_sha256"] == unit_sha256
                    and cached["source_sha256"] == part["source_sha256"]
                    and cached["cache_plan_generation"]
                    == part["plan_generation"]
                    and cached["cache_claim_generation"]
                    == part["claim_generation"]
                    and cached["source_part_id"] == part_id
                    and cached["persisted_unit_sha256"] == unit_sha256
                ):
                    return False
                raise ValueError(
                    "processed evidence does not exactly match its running claim"
                )
            if part["state"] != PartState.RUNNING.value:
                raise ValueError("evidence part must still be running before publication")
            if cached is not None:
                raise ValueError("cache identity is already bound to different evidence")
            existing_unit = connection.execute(
                """SELECT source_part_id, unit_sha256 FROM evidence_units
                   WHERE evidence_unit_id = ?""",
                (unit.evidence_unit_id,),
            ).fetchone()
            if existing_unit is None:
                connection.execute(
                    """INSERT INTO evidence_units(
                           evidence_unit_id, source_part_id, unit_json,
                           unit_sha256, published_at
                       ) VALUES (?, ?, ?, ?, ?)""",
                    (
                        unit.evidence_unit_id,
                        part_id,
                        unit_json,
                        unit_sha256,
                        published_at,
                    ),
                )
            elif (
                existing_unit["source_part_id"] != part_id
                or existing_unit["unit_sha256"] != unit_sha256
            ):
                raise ValueError("evidence unit identity is already in use")
            connection.execute(
                """INSERT INTO evidence_cache(
                       cache_digest, evidence_unit_id, unit_sha256,
                       source_sha256, plan_generation, claim_generation,
                       created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    cache_digest,
                    unit.evidence_unit_id,
                    unit_sha256,
                    part["source_sha256"],
                    part["plan_generation"],
                    part["claim_generation"],
                    published_at,
                ),
            )
            connection.execute(
                """UPDATE evidence_parts
                   SET state = ?, next_attempt_at = NULL, updated_at = ?
                   WHERE part_id = ?""",
                (PartState.PROCESSED.value, published_at, part_id),
            )
        return True

    @staticmethod
    def _snapshot(row: sqlite3.Row) -> StudyMapSnapshot:
        return StudyMapSnapshot.model_validate_json(row["snapshot_json"])

    def get_snapshot(self, snapshot_id: str) -> StudyMapSnapshot | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT snapshot_json FROM study_map_snapshots
                   WHERE snapshot_id = ?""",
                (snapshot_id,),
            ).fetchone()
        return None if row is None else self._snapshot(row)

    def save_snapshot(self, snapshot: StudyMapSnapshot) -> bool:
        snapshot_payload = snapshot.model_dump(mode="json")
        self._validate_serialized_strings(snapshot_payload)
        dependency_ids = set(snapshot.evidence_unit_ids)
        node_dependency_ids = {
            evidence_unit_id
            for node in snapshot.nodes
            for evidence_unit_id in node.evidence_unit_ids
        }
        if dependency_ids != node_dependency_ids:
            raise ValueError(
                "snapshot top-level and node evidence IDs must have one dependency closure"
            )
        snapshot_json = self._canonical_json(snapshot_payload)
        created_at = self._timestamp(snapshot.created_at)
        with self._transaction() as connection:
            existing = connection.execute(
                """SELECT snapshot_json FROM study_map_snapshots
                   WHERE snapshot_id = ?""",
                (snapshot.snapshot_id,),
            ).fetchone()
            if existing is not None:
                if existing["snapshot_json"] == snapshot_json:
                    return False
                raise ValueError("snapshot identity is already in use")
            if dependency_ids:
                placeholders = ",".join("?" for _ in dependency_ids)
                dependency_rows = connection.execute(
                    f"""SELECT units.evidence_unit_id,
                               parts.workspace_id, parts.revision_id
                        FROM evidence_units AS units
                        JOIN evidence_parts AS parts
                          ON parts.part_id = units.source_part_id
                        WHERE units.evidence_unit_id IN ({placeholders})""",
                    tuple(sorted(dependency_ids)),
                ).fetchall()
                if len(dependency_rows) != len(dependency_ids):
                    raise sqlite3.IntegrityError(
                        "snapshot evidence dependencies do not exist"
                    )
                if any(
                    row["workspace_id"] != snapshot.workspace_id
                    or row["revision_id"] != snapshot.revision_id
                    for row in dependency_rows
                ):
                    raise ValueError(
                        "snapshot evidence must belong to its workspace and revision"
                    )
            connection.execute(
                """INSERT INTO study_map_snapshots(
                       snapshot_id, workspace_id, revision_id, status,
                       snapshot_json, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    snapshot.snapshot_id,
                    snapshot.workspace_id,
                    snapshot.revision_id,
                    snapshot.status.value,
                    snapshot_json,
                    created_at,
                ),
            )
            for evidence_unit_id in snapshot.evidence_unit_ids:
                connection.execute(
                    """INSERT INTO study_map_dependencies(
                           snapshot_id, evidence_unit_id
                       ) VALUES (?, ?)""",
                    (snapshot.snapshot_id, evidence_unit_id),
                )
        return True

    def coverage(
        self, workspace_id: str, revision_id: str
    ) -> CoverageSummary | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT snapshot_json FROM study_map_snapshots
                   WHERE workspace_id = ? AND revision_id = ?
                   ORDER BY created_at DESC, rowid DESC LIMIT 1""",
                (workspace_id, revision_id),
            ).fetchone()
        if row is None:
            return None
        return self._snapshot(row).coverage

    def invalidate_entry(
        self,
        workspace_id: str,
        revision_id: str,
        entry_id: str,
    ) -> tuple[str, ...]:
        now = self._timestamp(self._now())
        with self._transaction() as connection:
            rows = connection.execute(
                """SELECT part_id FROM evidence_parts
                   WHERE workspace_id = ? AND revision_id = ? AND entry_id = ?
                   ORDER BY part_id ASC""",
                (workspace_id, revision_id, entry_id),
            ).fetchall()
            part_ids = tuple(str(row["part_id"]) for row in rows)
            if not part_ids:
                return ()
            placeholders = ",".join("?" for _ in part_ids)
            unit_rows = connection.execute(
                f"""SELECT evidence_unit_id FROM evidence_units
                    WHERE source_part_id IN ({placeholders})""",
                part_ids,
            ).fetchall()
            unit_ids = tuple(str(row["evidence_unit_id"]) for row in unit_rows)
            if unit_ids:
                unit_placeholders = ",".join("?" for _ in unit_ids)
                connection.execute(
                    f"""DELETE FROM study_map_snapshots
                        WHERE snapshot_id IN (
                          SELECT snapshot_id FROM study_map_dependencies
                          WHERE evidence_unit_id IN ({unit_placeholders})
                        ) AND workspace_id = ? AND revision_id = ?""",
                    (*unit_ids, workspace_id, revision_id),
                )
            connection.execute(
                f"DELETE FROM evidence_units WHERE source_part_id IN ({placeholders})",
                part_ids,
            )
            connection.execute(
                f"""UPDATE evidence_parts
                    SET state = ?, next_attempt_at = NULL, updated_at = ?
                    WHERE part_id IN ({placeholders})""",
                (PartState.INVALIDATED.value, now, *part_ids),
            )
        return part_ids

    def delete_workspace(self, workspace_id: str) -> None:
        with self._transaction() as connection:
            connection.execute(
                "DELETE FROM study_map_snapshots WHERE workspace_id = ?",
                (workspace_id,),
            )
            connection.execute(
                "DELETE FROM evidence_parts WHERE workspace_id = ?", (workspace_id,)
            )
            connection.execute(
                "DELETE FROM evidence_meta WHERE name = ?",
                (f"current_revision:{self._digest(workspace_id)}",),
            )

    def recover_unfinished(self) -> tuple[str, ...]:
        now = self._timestamp(self._now())
        with self._transaction() as connection:
            rows = connection.execute(
                """SELECT part_id FROM evidence_parts
                   WHERE state = ? ORDER BY created_at ASC, part_id ASC""",
                (PartState.RUNNING.value,),
            ).fetchall()
            part_ids = tuple(str(row["part_id"]) for row in rows)
            if part_ids:
                connection.execute(
                    """UPDATE evidence_parts SET state = ?, next_attempt_at = ?, updated_at = ?
                       WHERE state = ?""",
                    (
                        PartState.RETRY_WAIT.value,
                        now,
                        now,
                        PartState.RUNNING.value,
                    ),
                )
        return part_ids
