from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
import json

import pytest

from exam_predictor.evidence.models import (
    CoverageItem,
    CoverageSummary,
    EvidenceCitation,
    EvidenceUnit,
    KnowledgeNode,
    PartState,
    SnapshotStatus,
    SourcePartPlan,
    StudyMapSnapshot,
)
from exam_predictor.evidence.store import EvidenceStore


NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


def part_plan(
    part_id: str = "part-1",
    *,
    workspace_id: str = "workspace-1",
    revision_id: str = "revision-1",
    entry_id: str = "entry-1",
    source_sha256: str = "a" * 64,
    part_sha256: str = "b" * 64,
    state: PartState = PartState.PLANNED,
    locator: str = "pages 1-3",
) -> SourcePartPlan:
    return SourcePartPlan(
        part_id=part_id,
        workspace_id=workspace_id,
        revision_id=revision_id,
        entry_id=entry_id,
        relative_path=f"notes/{entry_id}.pdf",
        source_sha256=source_sha256,
        part_sha256=part_sha256,
        ordinal=0,
        locator=locator,
        media_type="application/pdf",
        size_bytes=1024,
        scheduling_class="document",
        priority=1,
        state=state,
        idempotency_key=f"evidence:{part_id}",
    )


def evidence_unit(
    unit_id: str = "unit-1",
    *,
    part_id: str = "part-1",
    entry_id: str = "entry-1",
    content: str = "The hand-checked evidence content.",
    relative_path: str | None = None,
    locator: str = "pages 1-3",
) -> EvidenceUnit:
    return EvidenceUnit(
        evidence_unit_id=unit_id,
        source_part_id=part_id,
        content=content,
        citations=(
            EvidenceCitation(
                citation_id=f"citation-{unit_id}",
                evidence_unit_id=unit_id,
                source_part_id=part_id,
                relative_path=relative_path or f"notes/{entry_id}.pdf",
                locator=locator,
            ),
        ),
    )


def publish_claim(
    store: EvidenceStore,
    part_id: str,
    unit: EvidenceUnit,
    *,
    cache_key: str,
    completed_at: datetime,
    claim_token: str | None = None,
) -> bool:
    token = claim_token or store.publication_token(part_id)
    return store.publish_evidence(
        part_id,
        unit,
        cache_key=cache_key,
        claim_token=token,
        completed_at=completed_at,
    )


def snapshot(
    snapshot_id: str,
    unit_id: str,
    *,
    workspace_id: str = "workspace-1",
    revision_id: str = "revision-1",
    created_at: datetime = NOW,
    covered_topic: str = "algebra",
) -> StudyMapSnapshot:
    return StudyMapSnapshot(
        snapshot_id=snapshot_id,
        workspace_id=workspace_id,
        revision_id=revision_id,
        status=SnapshotStatus.INITIAL,
        nodes=(
            KnowledgeNode(
                node_id=f"node-{snapshot_id}",
                title=covered_topic,
                focus_score=0.8,
                confidence=0.7,
                evidence_unit_ids=(unit_id,),
            ),
        ),
        coverage=CoverageSummary(
            items=(
                CoverageItem(topic=covered_topic, covered=True),
                CoverageItem(topic="calculus", covered=False),
            ),
            covered_count=1,
            total_count=2,
        ),
        evidence_unit_ids=(unit_id,),
        created_at=created_at,
    )


def sqlite_names_and_columns(path: Path) -> dict[str, tuple[str, ...]]:
    with sqlite3.connect(path) as connection:
        names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        return {
            name: tuple(
                str(row[1])
                for row in connection.execute(f'PRAGMA table_info("{name}")')
            )
            for name in names
        }


def all_columns(names: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(column.casefold() for columns in names.values() for column in columns)


def sqlite_text_cells(path: Path) -> tuple[str, ...]:
    with sqlite3.connect(path) as connection:
        table_names = tuple(
            str(row[0])
            for row in connection.execute(
                """SELECT name FROM sqlite_master
                   WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"""
            )
        )
        return tuple(
            value
            for table_name in table_names
            for row in connection.execute(f'SELECT * FROM "{table_name}"')
            for value in row
            if isinstance(value, str)
        )


def test_migration_is_additive_idempotent_and_never_stores_secret_columns(
    tmp_path: Path,
):
    path = tmp_path / "evidence.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE existing_feature (feature_id TEXT PRIMARY KEY)"
        )
        connection.execute("INSERT INTO existing_feature VALUES ('preserved')")

    store = EvidenceStore(path)
    store.migrate()
    store.migrate()
    store.close()

    names = sqlite_names_and_columns(path)
    assert {
        "evidence_meta",
        "evidence_parts",
        "evidence_attempts",
        "evidence_units",
        "evidence_cache",
        "study_map_snapshots",
        "study_map_dependencies",
    } <= names.keys()
    assert not any(
        "key" in column or "authorization" in column
        for column in all_columns(names)
    )
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT * FROM existing_feature").fetchall() == [
            ("preserved",)
        ]


def test_migration_adds_persistent_generation_columns_to_a_legacy_parts_table(
    tmp_path: Path,
):
    path = tmp_path / "legacy-evidence.sqlite3"
    plan = part_plan()
    plan_json = EvidenceStore._canonical_json(plan.model_dump(mode="json"))
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE evidence_parts (
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
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            """INSERT INTO evidence_parts VALUES (
                   ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?
               )""",
            (
                plan.part_id,
                plan.workspace_id,
                plan.revision_id,
                plan.entry_id,
                plan.source_sha256,
                plan.part_sha256,
                plan.priority,
                plan.ordinal,
                plan.state.value,
                plan_json,
                NOW.isoformat(),
                NOW.isoformat(),
            ),
        )

    store = EvidenceStore(path)
    with store._lock:
        columns = {
            str(row[1])
            for row in store._connection.execute("PRAGMA table_info(evidence_parts)")
        }
        generations = store._connection.execute(
            """SELECT plan_generation, claim_generation
               FROM evidence_parts WHERE part_id = ?""",
            (plan.part_id,),
        ).fetchone()

    assert {"plan_generation", "claim_generation"} <= columns
    assert tuple(generations) == (1, 0)
    assert store.get_part(plan.part_id) == plan
    store.close()


def test_schema_runtime_invariants_are_executable_and_persisted(tmp_path: Path):
    path = tmp_path / "evidence.sqlite3"
    store = EvidenceStore(path)
    plan = part_plan()
    store.upsert_part_plans((plan,))

    with store._lock:
        assert store._connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert store._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        index_rows = store._connection.execute(
            """SELECT name FROM sqlite_master
               WHERE type = 'index'
                 AND (name LIKE 'idx_evidence_%'
                      OR name = 'idx_study_map_snapshots_workspace_revision')"""
        ).fetchall()
        index_names = {str(row[0]) for row in index_rows}
        expected_indexes = {
            "idx_evidence_parts_workspace_revision_state": (
                "workspace_id",
                "revision_id",
                "state",
                "priority",
            ),
            "idx_evidence_parts_source_sha256": ("source_sha256",),
            "idx_evidence_parts_next_attempt_at": ("next_attempt_at",),
            "idx_evidence_cache_digest": ("cache_digest",),
            "idx_evidence_cache_source_sha256": ("source_sha256",),
            "idx_study_map_snapshots_workspace_revision": (
                "workspace_id",
                "revision_id",
                "created_at",
            ),
        }
        assert expected_indexes.keys() <= index_names
        for index_name, expected_columns in expected_indexes.items():
            info_columns = tuple(
                str(row[2])
                for row in store._connection.execute(
                    f'PRAGMA index_info("{index_name}")'
                )
            )
            xinfo_columns = tuple(
                str(row[2])
                for row in store._connection.execute(
                    f'PRAGMA index_xinfo("{index_name}")'
                )
                if row[5] == 1 and row[1] >= 0
            )
            assert info_columns == expected_columns
            assert xinfo_columns == expected_columns
        stored = store._connection.execute(
            """SELECT plan_json, created_at, updated_at
               FROM evidence_parts WHERE part_id = ?""",
            (plan.part_id,),
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            store._connection.execute(
                """INSERT INTO evidence_attempts(
                       part_id, attempt_number, route, outcome,
                       started_at, finished_at
                   ) VALUES ('missing', 1, 'document', 'failed', ?, ?)""",
                (NOW.isoformat(), NOW.isoformat()),
            )

    expected_json = json.dumps(
        plan.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert stored["plan_json"] == expected_json
    for column in ("created_at", "updated_at"):
        parsed = datetime.fromisoformat(stored[column])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)
        assert stored[column].endswith("+00:00")
    store.close()


def test_migration_repairs_a_same_name_index_with_wrong_columns(tmp_path: Path):
    path = tmp_path / "evidence.sqlite3"
    store = EvidenceStore(path)
    store.close()
    with sqlite3.connect(path) as connection:
        connection.execute("DROP INDEX idx_evidence_cache_digest")
        connection.execute(
            """CREATE INDEX idx_evidence_cache_digest
               ON evidence_cache(evidence_unit_id, cache_digest)"""
        )

    restarted = EvidenceStore(path)
    try:
        with restarted._lock:
            columns = tuple(
                str(row[2])
                for row in restarted._connection.execute(
                    'PRAGMA index_info("idx_evidence_cache_digest")'
                )
            )
        assert columns == ("cache_digest",)
    finally:
        restarted.close()


def test_begin_immediate_serializes_claim_after_an_uncommitted_writer(
    tmp_path: Path,
):
    path = tmp_path / "evidence.sqlite3"
    first = EvidenceStore(path)
    second = EvidenceStore(path)
    plan = part_plan()
    first.upsert_part_plans((plan,))
    second._connection.execute("PRAGMA busy_timeout=2000")
    start = Barrier(2)
    first._connection.execute("BEGIN IMMEDIATE")
    first._connection.execute(
        "UPDATE evidence_parts SET state = ? WHERE part_id = ?",
        (PartState.RUNNING.value, plan.part_id),
    )

    def claim() -> tuple[SourcePartPlan, ...]:
        start.wait()
        return second.claim_parts(
            plan.workspace_id, plan.revision_id, limit=1, now=NOW
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(claim)
            start.wait()
            with pytest.raises(FutureTimeoutError):
                future.result(timeout=0.1)
            first._connection.commit()
            assert future.result(timeout=2) == ()
    finally:
        if first._connection.in_transaction:
            first._connection.rollback()
        first.close()
        second.close()


def test_serialized_models_recursively_reject_unsafe_strings_before_writes(
    tmp_path: Path,
):
    path = tmp_path / "evidence.sqlite3"
    store = EvidenceStore(path)
    unsafe_values = (
        "sk-" + "A" * 24,
        "Authorization: Bearer abcdefghijklmnop",
        "https://example.test/file?x-amz-signature=private",
        r"C:\Users\private\notes.pdf",
        "<httpx.Client object at private>",
        "RuntimeError: provider exploded",
    )

    for index, unsafe in enumerate(unsafe_values):
        plan = part_plan(part_id=f"unsafe-plan-{index}").model_copy(
            update={"idempotency_key": unsafe}
        )
        with pytest.raises(ValueError, match="credentials|handles|paths"):
            store.upsert_part_plans((plan,))

    safe_plan = part_plan()
    safe_unit = evidence_unit()
    store.upsert_part_plans((safe_plan,))
    [claimed] = store.claim_parts(
        safe_plan.workspace_id, safe_plan.revision_id, limit=1, now=NOW
    )
    unsafe_unit_id = unsafe_values[1]
    unsafe_unit = safe_unit.model_copy(
        update={
            "evidence_unit_id": unsafe_unit_id,
            "citations": (
                safe_unit.citations[0].model_copy(
                    update={"evidence_unit_id": unsafe_unit_id}
                ),
            ),
        }
    )
    with pytest.raises(ValueError, match="credentials|handles|paths"):
        publish_claim(store,
            safe_plan.part_id,
            unsafe_unit,
            cache_key=claimed.idempotency_key,
            completed_at=NOW,
        )

    publish_claim(store,
        safe_plan.part_id,
        safe_unit,
        cache_key=claimed.idempotency_key,
        completed_at=NOW,
    )
    base = snapshot("safe-snapshot", safe_unit.evidence_unit_id)
    unsafe_snapshots = (
        base.model_copy(update={"snapshot_id": unsafe_values[3]}),
        base.model_copy(
            update={
                "nodes": (
                    base.nodes[0].model_copy(update={"title": unsafe_values[5]}),
                )
            }
        ),
        base.model_copy(
            update={
                "coverage": base.coverage.model_copy(
                    update={
                        "items": (
                            base.coverage.items[0].model_copy(
                                update={"topic": unsafe_values[2]}
                            ),
                            base.coverage.items[1],
                        )
                    }
                )
            }
        ),
    )
    for unsafe_snapshot in unsafe_snapshots:
        with pytest.raises(ValueError, match="credentials|handles|paths"):
            store.save_snapshot(unsafe_snapshot)

    cells = "\n".join(sqlite_text_cells(path))
    assert all(unsafe not in cells for unsafe in unsafe_values)
    store.close()


def test_recovery_pauses_running_parts_without_consuming_attempt(tmp_path: Path):
    path = tmp_path / "evidence.sqlite3"
    store = EvidenceStore(path)
    store.upsert_part_plans((part_plan(),))
    store.mark_running("part-1", attempt=1)
    store.close()

    restarted = EvidenceStore(path)
    assert restarted.recover_unfinished() == ("part-1",)
    part = restarted.get_part("part-1")
    assert part.state is PartState.RETRY_WAIT
    assert restarted.attempt_count("part-1") == 1
    assert restarted.recover_unfinished() == ()
    restarted.close()


def test_claims_are_atomic_ordered_and_reject_a_stale_revision(tmp_path: Path):
    path = tmp_path / "evidence.sqlite3"
    first = EvidenceStore(path)
    second = EvidenceStore(path)
    old = part_plan(part_id="old-part", revision_id="revision-old")
    first.upsert_part_plans((old,))
    current_low = part_plan(
        part_id="current-low",
        revision_id="revision-current",
        entry_id="entry-low",
    ).model_copy(update={"priority": 5})
    current_high = part_plan(
        part_id="current-high",
        revision_id="revision-current",
        entry_id="entry-high",
    ).model_copy(update={"priority": 0})
    first.upsert_part_plans((current_low, current_high))

    assert first.claim_parts("workspace-1", "revision-old", limit=4, now=NOW) == ()
    claimed = first.claim_parts(
        "workspace-1", "revision-current", limit=1, now=NOW
    )
    assert tuple(item.part_id for item in claimed) == ("current-high",)
    assert claimed[0].state is PartState.RUNNING
    [low_claim] = second.claim_parts(
        "workspace-1", "revision-current", limit=4, now=NOW
    )
    assert low_claim.model_copy(
        update={"idempotency_key": current_low.idempotency_key}
    ) == current_low.model_copy(update={"state": PartState.RUNNING})
    assert first.claim_parts(
        "workspace-1", "revision-current", limit=4, now=NOW
    ) == ()
    first.close()
    second.close()


def test_retry_wait_is_claimed_only_when_due(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    store.upsert_part_plans((part_plan(),))
    store.record_attempt(
        "part-1",
        attempt=1,
        route="document",
        outcome=PartState.RETRY_WAIT,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
        next_attempt_at=NOW + timedelta(minutes=1),
        safe_error_code="provider_timeout",
    )

    assert store.claim_parts(
        "workspace-1", "revision-1", limit=1, now=NOW
    ) == ()
    [claimed] = store.claim_parts(
        "workspace-1",
        "revision-1",
        limit=1,
        now=NOW + timedelta(minutes=1),
    )
    assert claimed.state is PartState.RUNNING
    assert store.attempt_count("part-1") == 1
    store.close()


def test_record_attempt_rolls_back_part_state_when_attempt_insert_fails(
    tmp_path: Path,
):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    original = part_plan()
    store.upsert_part_plans((original,))
    with store._transaction() as connection:
        connection.execute(
            """CREATE TRIGGER reject_evidence_attempt
               BEFORE INSERT ON evidence_attempts
               BEGIN SELECT RAISE(ABORT, 'forced attempt failure'); END"""
        )

    try:
        store.record_attempt(
            "part-1",
            attempt=1,
            route="document",
            outcome=PartState.FAILED,
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=1),
            safe_error_code="provider_failed",
        )
    except sqlite3.IntegrityError as error:
        assert "forced attempt failure" in str(error)
    else:
        raise AssertionError("the trigger must reject the attempt")

    assert store.get_part("part-1") == original
    assert store.attempt_count("part-1") == 0
    store.close()


def test_transaction_rolls_back_when_commit_itself_fails(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    real_connection = store._connection

    class CommitFaultConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection
            self.rollback_called = False

        def execute(self, *args: object) -> sqlite3.Cursor:
            return self.connection.execute(*args)

        def commit(self) -> None:
            raise sqlite3.OperationalError("forced commit failure")

        def rollback(self) -> None:
            self.rollback_called = True
            self.connection.rollback()

    fault = CommitFaultConnection(real_connection)
    store._connection = fault  # type: ignore[assignment]
    try:
        with pytest.raises(sqlite3.OperationalError, match="forced commit failure"):
            with store._transaction() as connection:
                connection.execute(
                    "INSERT INTO evidence_meta(name, value) VALUES ('commit-fault', 'x')"
                )
        assert fault.rollback_called is True
        assert real_connection.execute(
            "SELECT value FROM evidence_meta WHERE name = 'commit-fault'"
        ).fetchone() is None
    finally:
        store._connection = real_connection
        store.close()


def test_publication_is_durable_and_processed_claim_cannot_publish_again(
    tmp_path: Path,
):
    path = tmp_path / "evidence.sqlite3"
    store = EvidenceStore(path)
    plan = part_plan()
    store.upsert_part_plans((plan,))
    [claimed] = store.claim_parts(
        plan.workspace_id, plan.revision_id, limit=1, now=NOW
    )
    claim_token = store.publication_token(plan.part_id)
    unit = evidence_unit()

    assert publish_claim(
        store,
        plan.part_id,
        unit,
        cache_key=claimed.idempotency_key,
        claim_token=claim_token,
        completed_at=NOW,
    ) is True
    with pytest.raises(ValueError, match="running"):
        publish_claim(
            store,
            plan.part_id,
            unit,
            cache_key=claimed.idempotency_key,
            claim_token=claim_token,
            completed_at=NOW,
        )
    assert store.get_part(plan.part_id).state is PartState.PROCESSED
    assert store.cached_evidence(claimed.idempotency_key) == unit
    store.close()

    restarted = EvidenceStore(path)
    assert restarted.cached_evidence(claimed.idempotency_key) == unit
    restarted.close()


def test_publication_rejects_a_part_that_was_never_claimed(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    plan = part_plan()
    unit = evidence_unit()
    store.upsert_part_plans((plan,))

    with pytest.raises(ValueError, match="running"):
        publish_claim(store,
            plan.part_id,
            unit,
            cache_key=plan.idempotency_key,
            completed_at=NOW,
        )

    assert store.get_part(plan.part_id) == plan
    assert store.get_evidence_unit(unit.evidence_unit_id) is None
    assert store.cached_evidence(plan.idempotency_key) is None
    store.close()


def test_late_publication_for_a_stale_revision_has_no_side_effects(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    stale = part_plan(revision_id="revision-old")
    store.upsert_part_plans((stale,))
    [running] = store.claim_parts(
        stale.workspace_id, stale.revision_id, limit=1, now=NOW
    )
    current = part_plan(
        part_id="part-current",
        revision_id="revision-current",
        entry_id="entry-current",
    )
    store.upsert_part_plans((current,))

    with pytest.raises(ValueError, match="revision"):
        publish_claim(store,
            stale.part_id,
            evidence_unit(),
            cache_key=running.idempotency_key,
            completed_at=NOW,
        )

    assert store.get_part(stale.part_id) == running.model_copy(
        update={"idempotency_key": stale.idempotency_key}
    )
    assert store.get_evidence_unit("unit-1") is None
    assert store.cached_evidence(running.idempotency_key) is None
    store.close()


def test_late_publication_for_an_invalidated_part_has_no_side_effects(
    tmp_path: Path,
):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    plan = part_plan()
    store.upsert_part_plans((plan,))
    [claimed] = store.claim_parts(
        plan.workspace_id, plan.revision_id, limit=1, now=NOW
    )
    store.invalidate_entry(plan.workspace_id, plan.revision_id, plan.entry_id)

    with pytest.raises(ValueError, match="running"):
        publish_claim(store,
            plan.part_id,
            evidence_unit(),
            cache_key=claimed.idempotency_key,
            completed_at=NOW,
        )

    assert store.get_part(plan.part_id).state is PartState.INVALIDATED
    assert store.get_evidence_unit("unit-1") is None
    store.close()


def test_old_claim_token_cannot_publish_into_a_replanned_source(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    old = part_plan()
    store.upsert_part_plans((old,))
    [old_claim] = store.claim_parts(
        old.workspace_id, old.revision_id, limit=1, now=NOW
    )
    old_token = store.publication_token(old.part_id)
    replanned = old.model_copy(
        update={
            "source_sha256": "c" * 64,
            "part_sha256": "d" * 64,
            "idempotency_key": "evidence:part-1:new-source",
        }
    )
    store.upsert_part_plans((replanned,))
    [running] = store.claim_parts(
        replanned.workspace_id, replanned.revision_id, limit=1, now=NOW
    )

    with pytest.raises(ValueError, match="claim token"):
        publish_claim(
            store,
            replanned.part_id,
            evidence_unit(),
            cache_key=old_claim.idempotency_key,
            claim_token=old_token,
            completed_at=NOW,
        )

    assert store.get_part(replanned.part_id) == running.model_copy(
        update={"idempotency_key": replanned.idempotency_key}
    )
    assert store.get_evidence_unit("unit-1") is None
    assert store.cached_evidence(old_claim.idempotency_key) is None
    store.close()


def test_claim_returns_exact_plan_and_cache_identity_is_independent_of_claim_token(
    tmp_path: Path,
):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    plan = part_plan().model_copy(update={"idempotency_key": "caller-plan-key"})
    store.upsert_part_plans((plan,))
    [claimed] = store.claim_parts(
        plan.workspace_id, plan.revision_id, limit=1, now=NOW
    )
    claim_token = store.publication_token(plan.part_id)
    cache_key = "caller-cache-key-that-is-not-a-claim-token"
    unit = evidence_unit()

    assert claimed == plan.model_copy(update={"state": PartState.RUNNING})
    assert cache_key != claim_token
    assert publish_claim(
        store,
        plan.part_id,
        unit,
        cache_key=cache_key,
        claim_token=claim_token,
        completed_at=NOW,
    ) is True
    assert store.cached_evidence(cache_key) == unit
    store.close()


def test_aba_plan_identity_rotates_persistent_generations(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    caller_key = "caller-reused-key"
    first_a = part_plan(locator="pages 1-3").model_copy(
        update={"idempotency_key": caller_key}
    )
    store.upsert_part_plans((first_a,))
    [claim_a1] = store.claim_parts(
        first_a.workspace_id, first_a.revision_id, limit=1, now=NOW
    )
    token_a1 = store.publication_token(first_a.part_id)
    plan_b = first_a.model_copy(update={"locator": "pages 4-6"})
    store.upsert_part_plans((plan_b,))
    [claim_b] = store.claim_parts(
        plan_b.workspace_id, plan_b.revision_id, limit=1, now=NOW
    )
    token_b = store.publication_token(plan_b.part_id)
    second_a = plan_b.model_copy(update={"locator": "pages 1-3"})
    store.upsert_part_plans((second_a,))
    [claim_a2] = store.claim_parts(
        second_a.workspace_id, second_a.revision_id, limit=1, now=NOW
    )
    token_a2 = store.publication_token(second_a.part_id)

    assert claim_a1.idempotency_key == caller_key
    assert claim_b.idempotency_key == caller_key
    assert claim_a2.idempotency_key == caller_key
    assert len({token_a1, token_b, token_a2}) == 3
    with store._lock:
        generations = store._connection.execute(
            """SELECT plan_generation, claim_generation
               FROM evidence_parts WHERE part_id = ?""",
            (second_a.part_id,),
        ).fetchone()
    assert tuple(generations) == (3, 3)

    unit = evidence_unit(content="Current A worker result")
    with pytest.raises(ValueError, match="claim token"):
        publish_claim(
            store,
            second_a.part_id,
            unit,
            cache_key="caller-cache-a",
            claim_token=token_a1,
            completed_at=NOW,
        )
    assert store.get_part(second_a.part_id) == second_a.model_copy(
        update={"state": PartState.RUNNING}
    )
    assert publish_claim(
        store,
        second_a.part_id,
        unit,
        cache_key="caller-cache-a",
        claim_token=token_a2,
        completed_at=NOW,
    ) is True
    store.close()


def test_recovery_reclaim_rotates_token_across_process_restart(tmp_path: Path):
    path = tmp_path / "evidence.sqlite3"
    store = EvidenceStore(path)
    plan = part_plan()
    store.upsert_part_plans((plan,))
    [first_claim] = store.claim_parts(
        plan.workspace_id, plan.revision_id, limit=1, now=NOW
    )
    first_token = store.publication_token(plan.part_id)
    store.close()

    restarted = EvidenceStore(path)
    restarted.recover_unfinished()
    restarted.upsert_part_plans((plan,))
    [resumed_claim] = restarted.claim_parts(
        plan.workspace_id,
        plan.revision_id,
        limit=1,
        now=datetime.now(UTC) + timedelta(minutes=1),
    )
    resumed_token = restarted.publication_token(plan.part_id)
    restarted.close()

    again = EvidenceStore(path)
    assert again.publication_token(plan.part_id) == resumed_token
    unit = evidence_unit()
    with pytest.raises(ValueError, match="claim token"):
        publish_claim(
            again,
            plan.part_id,
            unit,
            cache_key="restart-cache",
            claim_token=first_token,
            completed_at=NOW + timedelta(minutes=2),
        )
    assert publish_claim(
        again,
        plan.part_id,
        unit,
        cache_key="restart-cache",
        claim_token=resumed_token,
        completed_at=NOW + timedelta(minutes=2),
    ) is True

    assert first_claim.idempotency_key == plan.idempotency_key
    assert resumed_claim.idempotency_key == plan.idempotency_key
    assert resumed_token != first_token
    again.close()


def test_identical_plan_upsert_preserves_processed_state_and_cache(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    plan = part_plan()
    unit = evidence_unit()
    store.upsert_part_plans((plan,))
    [claimed] = store.claim_parts(
        plan.workspace_id, plan.revision_id, limit=1, now=NOW
    )
    publish_claim(store,
        plan.part_id, unit, cache_key=claimed.idempotency_key, completed_at=NOW
    )

    store.upsert_part_plans((plan,))

    assert store.get_part("part-1") == plan.model_copy(
        update={"state": PartState.PROCESSED}
    )
    assert store.cached_evidence(claimed.idempotency_key) == unit
    assert store.claim_parts(
        "workspace-1", "revision-1", limit=1, now=NOW
    ) == ()
    store.close()


@pytest.mark.parametrize(
    "identity_update",
    (
        {"locator": "pages 4-6"},
        {"media_type": "text/plain"},
        {"scheduling_class": "multimodal"},
        {"priority": 9},
        {"idempotency_key": "evidence:part-1:replanned"},
    ),
)
def test_changed_plan_identity_atomically_resets_all_prior_work(
    tmp_path: Path,
    identity_update: dict[str, str | int],
):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    original = part_plan()
    unit = evidence_unit()
    store.upsert_part_plans((original,))
    store.claim_parts(
        original.workspace_id, original.revision_id, limit=1, now=NOW
    )
    store.record_attempt(
        original.part_id,
        attempt=1,
        route="document",
        outcome=PartState.RETRY_WAIT,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
        next_attempt_at=NOW + timedelta(minutes=1),
        safe_error_code="provider_timeout",
    )
    [resumed] = store.claim_parts(
        original.workspace_id,
        original.revision_id,
        limit=1,
        now=NOW + timedelta(minutes=1),
    )
    publish_claim(store,
        original.part_id,
        unit,
        cache_key=resumed.idempotency_key,
        completed_at=NOW + timedelta(minutes=1),
    )
    dependent = snapshot("old-snapshot", unit.evidence_unit_id)
    store.save_snapshot(dependent)
    changed = original.model_copy(update=identity_update)

    store.upsert_part_plans((changed,))

    assert store.get_part(original.part_id) == changed
    assert store.attempt_count(original.part_id) == 0
    assert store.get_evidence_unit(unit.evidence_unit_id) is None
    assert store.cached_evidence(resumed.idempotency_key) is None
    assert store.get_snapshot(dependent.snapshot_id) is None
    with store._lock:
        attempts = store._connection.execute(
            "SELECT COUNT(*) FROM evidence_attempts WHERE part_id = ?",
            (original.part_id,),
        ).fetchone()[0]
        next_attempt = store._connection.execute(
            "SELECT next_attempt_at FROM evidence_parts WHERE part_id = ?",
            (original.part_id,),
        ).fetchone()[0]
    assert attempts == 0
    assert next_attempt is None
    store.close()


def test_processed_claim_token_cannot_republish_changed_content(
    tmp_path: Path,
):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    plan = part_plan()
    original = evidence_unit()
    store.upsert_part_plans((plan,))
    [claimed] = store.claim_parts(
        plan.workspace_id, plan.revision_id, limit=1, now=NOW
    )
    claim_token = store.publication_token(plan.part_id)
    publish_claim(
        store,
        plan.part_id,
        original,
        cache_key=claimed.idempotency_key,
        claim_token=claim_token,
        completed_at=NOW,
    )
    changed = original.model_copy(update={"content": "Changed evidence content."})

    with pytest.raises(ValueError, match="running"):
        publish_claim(
            store,
            plan.part_id,
            changed,
            cache_key=claimed.idempotency_key,
            claim_token=claim_token,
            completed_at=NOW,
        )

    assert store.get_part(plan.part_id).state is PartState.PROCESSED
    assert store.get_evidence_unit(original.evidence_unit_id) == original
    assert store.cached_evidence(claimed.idempotency_key) == original
    store.close()


@pytest.mark.parametrize(
    ("part_locator", "citation_locator"),
    (
        ("pages 1-24", "page 5"),
        ("slides 2-8", "slide 3"),
        ("sheets 1-4", "sheet 2"),
        ("rows 10-20", "rows 12-15"),
        ("member appendix.pdf", "member appendix.pdf: page 2"),
        ("pages 1-3", "pages 1-3"),
    ),
)
def test_publication_accepts_only_verifiable_citation_sublocators(
    tmp_path: Path,
    part_locator: str,
    citation_locator: str,
):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    plan = part_plan(locator=part_locator)
    unit = evidence_unit(locator=citation_locator)
    store.upsert_part_plans((plan,))
    [claimed] = store.claim_parts(
        plan.workspace_id, plan.revision_id, limit=1, now=NOW
    )

    assert publish_claim(store,
        plan.part_id,
        unit,
        cache_key=claimed.idempotency_key,
        completed_at=NOW,
    ) is True
    store.close()


def test_archive_member_locator_keywords_are_insensitive_but_names_are_exact():
    assert EvidenceStore._locator_contains(
        "MEMBER Appendix.pdf", "member Appendix.pdf: PAGE 2"
    )
    assert EvidenceStore._locator_contains(
        "member Appendix.pdf", "member Appendix.pdf"
    )
    assert not EvidenceStore._locator_contains(
        "member Appendix.pdf", "member appendix.pdf: page 2"
    )


def test_publication_rejects_archive_member_case_collision(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    plan = part_plan(locator="member Appendix.pdf")
    store.upsert_part_plans((plan,))
    [claimed] = store.claim_parts(
        plan.workspace_id, plan.revision_id, limit=1, now=NOW
    )
    unit = evidence_unit(locator="member appendix.pdf: page 2")

    with pytest.raises(ValueError, match="citation"):
        publish_claim(store,
            plan.part_id,
            unit,
            cache_key=claimed.idempotency_key,
            completed_at=NOW,
        )

    assert store.get_part(plan.part_id).state is PartState.RUNNING
    assert store.get_evidence_unit(unit.evidence_unit_id) is None
    store.close()


@pytest.mark.parametrize(
    "unit",
    (
        evidence_unit(relative_path="notes/other.pdf"),
        evidence_unit(locator="page 25"),
        evidence_unit(locator="paragraph 2"),
        evidence_unit(locator="member other.pdf: page 2"),
        evidence_unit(locator="member appendix.pdf: paragraph 2"),
    ),
)
def test_publication_rejects_citations_not_bound_to_the_persisted_part(
    tmp_path: Path,
    unit: EvidenceUnit,
):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    plan = part_plan(locator="pages 1-24")
    if unit.citations[0].locator.startswith("member"):
        plan = plan.model_copy(update={"locator": "member appendix.pdf"})
    store.upsert_part_plans((plan,))
    [running] = store.claim_parts(
        plan.workspace_id, plan.revision_id, limit=1, now=NOW
    )

    with pytest.raises(ValueError, match="citation"):
        publish_claim(store,
            plan.part_id,
            unit,
            cache_key=running.idempotency_key,
            completed_at=NOW,
        )

    assert store.get_part(plan.part_id) == running.model_copy(
        update={"idempotency_key": plan.idempotency_key}
    )
    assert store.get_evidence_unit(unit.evidence_unit_id) is None
    store.close()


def test_publication_rolls_back_unit_and_part_when_cache_write_fails(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    plan = part_plan()
    store.upsert_part_plans((plan,))
    [running] = store.claim_parts(
        plan.workspace_id, plan.revision_id, limit=1, now=NOW
    )
    with store._transaction() as connection:
        connection.execute(
            """CREATE TRIGGER reject_evidence_cache
               BEFORE INSERT ON evidence_cache
               BEGIN SELECT RAISE(ABORT, 'forced cache failure'); END"""
        )

    try:
        publish_claim(store,
            plan.part_id,
            evidence_unit(),
            cache_key=running.idempotency_key,
            completed_at=NOW,
        )
    except sqlite3.IntegrityError as error:
        assert "forced cache failure" in str(error)
    else:
        raise AssertionError("the trigger must reject the cache write")

    assert store.get_part("part-1") == running.model_copy(
        update={"idempotency_key": plan.idempotency_key}
    )
    assert store.get_evidence_unit("unit-1") is None
    assert store.cached_evidence(running.idempotency_key) is None
    store.close()


def test_snapshots_persist_exact_coverage_and_missing_dependencies_roll_back(
    tmp_path: Path,
):
    path = tmp_path / "evidence.sqlite3"
    store = EvidenceStore(path)
    plan = part_plan()
    store.upsert_part_plans((plan,))
    [claimed] = store.claim_parts(
        plan.workspace_id, plan.revision_id, limit=1, now=NOW
    )
    publish_claim(store,
        plan.part_id,
        evidence_unit(),
        cache_key=claimed.idempotency_key,
        completed_at=NOW,
    )
    saved = snapshot("snapshot-1", "unit-1")

    assert store.save_snapshot(saved) is True
    assert store.save_snapshot(saved) is False
    assert store.coverage("workspace-1", "revision-1") == saved.coverage
    missing = snapshot("snapshot-missing", "missing-unit")
    try:
        store.save_snapshot(missing)
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("missing evidence dependencies must fail")
    assert store.get_snapshot("snapshot-missing") is None
    store.close()

    restarted = EvidenceStore(path)
    assert restarted.coverage("workspace-1", "revision-1") == saved.coverage
    restarted.close()


def test_snapshot_rejects_cross_workspace_or_revision_evidence(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    plan = part_plan()
    unit = evidence_unit()
    store.upsert_part_plans((plan,))
    [claimed] = store.claim_parts(
        plan.workspace_id, plan.revision_id, limit=1, now=NOW
    )
    publish_claim(store,
        plan.part_id,
        unit,
        cache_key=claimed.idempotency_key,
        completed_at=NOW,
    )

    for invalid in (
        snapshot("cross-workspace", unit.evidence_unit_id, workspace_id="workspace-2"),
        snapshot("cross-revision", unit.evidence_unit_id, revision_id="revision-2"),
    ):
        with pytest.raises(ValueError, match="workspace|revision"):
            store.save_snapshot(invalid)
        assert store.get_snapshot(invalid.snapshot_id) is None
    store.close()


def test_snapshot_requires_exact_top_level_and_node_dependency_closure(
    tmp_path: Path,
):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    first = part_plan()
    second = part_plan(part_id="part-2", entry_id="entry-2")
    first_unit = evidence_unit()
    second_unit = evidence_unit(
        unit_id="unit-2", part_id="part-2", entry_id="entry-2"
    )
    store.upsert_part_plans((first, second))
    claims = {
        item.part_id: item
        for item in store.claim_parts(
            first.workspace_id, first.revision_id, limit=2, now=NOW
        )
    }
    publish_claim(store,
        first.part_id,
        first_unit,
        cache_key=claims[first.part_id].idempotency_key,
        completed_at=NOW,
    )
    publish_claim(store,
        second.part_id,
        second_unit,
        cache_key=claims[second.part_id].idempotency_key,
        completed_at=NOW,
    )
    base = snapshot("mismatched-closure", first_unit.evidence_unit_id)
    mismatched = base.model_copy(
        update={
            "nodes": (
                base.nodes[0].model_copy(
                    update={"evidence_unit_ids": (second_unit.evidence_unit_id,)}
                ),
            )
        }
    )

    with pytest.raises(ValueError, match="dependency closure"):
        store.save_snapshot(mismatched)

    assert store.get_snapshot(mismatched.snapshot_id) is None
    store.close()


def test_invalidation_does_not_delete_a_legacy_cross_workspace_snapshot(
    tmp_path: Path,
):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    plan = part_plan()
    unit = evidence_unit()
    store.upsert_part_plans((plan,))
    [claimed] = store.claim_parts(
        plan.workspace_id, plan.revision_id, limit=1, now=NOW
    )
    publish_claim(store,
        plan.part_id,
        unit,
        cache_key=claimed.idempotency_key,
        completed_at=NOW,
    )
    legacy = snapshot(
        "legacy-cross-workspace",
        unit.evidence_unit_id,
        workspace_id="workspace-2",
        revision_id="revision-2",
    )
    with store._transaction() as connection:
        connection.execute(
            """INSERT INTO study_map_snapshots(
                   snapshot_id, workspace_id, revision_id, status,
                   snapshot_json, created_at
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                legacy.snapshot_id,
                legacy.workspace_id,
                legacy.revision_id,
                legacy.status.value,
                store._canonical_json(legacy.model_dump(mode="json")),
                NOW.isoformat(),
            ),
        )
        connection.execute(
            """INSERT INTO study_map_dependencies(snapshot_id, evidence_unit_id)
               VALUES (?, ?)""",
            (legacy.snapshot_id, unit.evidence_unit_id),
        )

    store.invalidate_entry(plan.workspace_id, plan.revision_id, plan.entry_id)

    assert store.get_snapshot(legacy.snapshot_id) == legacy
    store.close()


def test_changed_entry_invalidation_preserves_unaffected_sibling(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    first_plan = part_plan()
    second_plan = part_plan(
        part_id="part-2",
        entry_id="entry-2",
        source_sha256="c" * 64,
        part_sha256="d" * 64,
    )
    store.upsert_part_plans((first_plan, second_plan))
    first_unit = evidence_unit()
    second_unit = evidence_unit(
        unit_id="unit-2", part_id="part-2", entry_id="entry-2"
    )
    claims = {
        item.part_id: item
        for item in store.claim_parts(
            first_plan.workspace_id, first_plan.revision_id, limit=2, now=NOW
        )
    }
    publish_claim(store,
        first_plan.part_id,
        first_unit,
        cache_key=claims[first_plan.part_id].idempotency_key,
        completed_at=NOW,
    )
    publish_claim(store,
        second_plan.part_id,
        second_unit,
        cache_key=claims[second_plan.part_id].idempotency_key,
        completed_at=NOW,
    )
    dependent = snapshot("snapshot-dependent", "unit-1")
    sibling = snapshot(
        "snapshot-sibling",
        "unit-2",
        created_at=NOW + timedelta(seconds=1),
        covered_topic="geometry",
    )
    store.save_snapshot(dependent)
    store.save_snapshot(sibling)

    assert store.invalidate_entry("workspace-1", "revision-1", "entry-1") == (
        "part-1",
    )
    assert store.get_part("part-1").state is PartState.INVALIDATED
    assert store.get_evidence_unit("unit-1") is None
    assert store.cached_evidence(claims[first_plan.part_id].idempotency_key) is None
    assert store.get_snapshot("snapshot-dependent") is None
    assert store.get_part("part-2").state is PartState.PROCESSED
    assert store.get_evidence_unit("unit-2") == second_unit
    assert (
        store.cached_evidence(claims[second_plan.part_id].idempotency_key)
        == second_unit
    )
    assert store.get_snapshot("snapshot-sibling") == sibling
    assert store.coverage("workspace-1", "revision-1") == sibling.coverage
    store.close()


def test_two_argument_invalidation_is_rejected_without_database_changes(
    tmp_path: Path,
):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    plan = part_plan()
    unit = evidence_unit()
    store.upsert_part_plans((plan,))
    [claimed] = store.claim_parts(
        plan.workspace_id, plan.revision_id, limit=1, now=NOW
    )
    publish_claim(store,
        plan.part_id,
        unit,
        cache_key=claimed.idempotency_key,
        completed_at=NOW,
    )
    saved = snapshot("preserved-snapshot", unit.evidence_unit_id)
    store.save_snapshot(saved)

    with pytest.raises(TypeError):
        store.invalidate_entry(plan.workspace_id, plan.entry_id)  # type: ignore[call-arg]

    assert store.get_part(plan.part_id).state is PartState.PROCESSED
    assert store.get_evidence_unit(unit.evidence_unit_id) == unit
    assert store.cached_evidence(claimed.idempotency_key) == unit
    assert store.get_snapshot(saved.snapshot_id) == saved
    store.close()


def test_workspace_cleanup_is_atomic_and_does_not_touch_other_workspaces(
    tmp_path: Path,
):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    first = part_plan()
    other = part_plan(
        part_id="other-part",
        workspace_id="workspace-2",
        revision_id="revision-2",
        entry_id="other-entry",
    )
    store.upsert_part_plans((first, other))
    [first_claim] = store.claim_parts(
        first.workspace_id, first.revision_id, limit=1, now=NOW
    )
    [other_claim] = store.claim_parts(
        other.workspace_id, other.revision_id, limit=1, now=NOW
    )
    publish_claim(store,
        first.part_id,
        evidence_unit(),
        cache_key=first_claim.idempotency_key,
        completed_at=NOW,
    )
    other_unit = evidence_unit(
        unit_id="other-unit",
        part_id="other-part",
        entry_id="other-entry",
    )
    publish_claim(store,
        other.part_id,
        other_unit,
        cache_key=other_claim.idempotency_key,
        completed_at=NOW,
    )

    with store._transaction() as connection:
        connection.execute(
            """CREATE TRIGGER reject_workspace_part_delete
               BEFORE DELETE ON evidence_parts
               WHEN OLD.workspace_id = 'workspace-1'
               BEGIN SELECT RAISE(ABORT, 'forced cleanup failure'); END"""
        )
    try:
        store.delete_workspace("workspace-1")
    except sqlite3.IntegrityError as error:
        assert "forced cleanup failure" in str(error)
    else:
        raise AssertionError("the trigger must reject workspace cleanup")
    assert store.get_evidence_unit("unit-1") is not None

    with store._transaction() as connection:
        connection.execute("DROP TRIGGER reject_workspace_part_delete")
    store.delete_workspace("workspace-1")
    try:
        store.get_part("part-1")
    except KeyError:
        pass
    else:
        raise AssertionError("workspace evidence parts must be deleted")
    assert store.cached_evidence(first_claim.idempotency_key) is None
    assert store.get_part("other-part") == other.model_copy(
        update={"state": PartState.PROCESSED}
    )
    assert store.cached_evidence(other_claim.idempotency_key) == other_unit
    store.close()
