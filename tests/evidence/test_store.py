from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
        locator="pages 1-3",
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
                relative_path=f"notes/{entry_id}.pdf",
                locator="pages 1-3",
            ),
        ),
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
    assert second.claim_parts(
        "workspace-1", "revision-current", limit=4, now=NOW
    ) == (
        current_low.model_copy(update={"state": PartState.RUNNING}),
    )
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


def test_publication_is_durable_and_false_only_for_an_identical_cache_entry(
    tmp_path: Path,
):
    path = tmp_path / "evidence.sqlite3"
    store = EvidenceStore(path)
    store.upsert_part_plans((part_plan(),))
    store.mark_running("part-1", attempt=1)
    unit = evidence_unit()

    assert store.publish_evidence(
        "part-1", unit, cache_key="cache-v1", completed_at=NOW
    ) is True
    assert store.publish_evidence(
        "part-1", unit, cache_key="cache-v1", completed_at=NOW
    ) is False
    assert store.get_part("part-1").state is PartState.PROCESSED
    assert store.cached_evidence("cache-v1") == unit
    store.close()

    restarted = EvidenceStore(path)
    assert restarted.cached_evidence("cache-v1") == unit
    restarted.close()


def test_identical_plan_upsert_preserves_processed_state_and_cache(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    plan = part_plan()
    unit = evidence_unit()
    store.upsert_part_plans((plan,))
    store.publish_evidence(
        "part-1", unit, cache_key="cache-v1", completed_at=NOW
    )

    store.upsert_part_plans((plan,))

    assert store.get_part("part-1") == plan.model_copy(
        update={"state": PartState.PROCESSED}
    )
    assert store.cached_evidence("cache-v1") == unit
    assert store.claim_parts(
        "workspace-1", "revision-1", limit=1, now=NOW
    ) == ()
    store.close()


def test_publication_rejects_changed_content_for_the_same_cache_identity(
    tmp_path: Path,
):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    first_plan = part_plan()
    second_plan = part_plan(part_id="part-2", entry_id="entry-2")
    store.upsert_part_plans((first_plan, second_plan))
    store.publish_evidence(
        "part-1", evidence_unit(), cache_key="cache-v1", completed_at=NOW
    )
    changed = evidence_unit(
        unit_id="unit-2",
        part_id="part-2",
        entry_id="entry-2",
        content="Changed evidence content.",
    )

    try:
        store.publish_evidence(
            "part-2", changed, cache_key="cache-v1", completed_at=NOW
        )
    except ValueError as error:
        assert "cache" in str(error).casefold()
    else:
        raise AssertionError("a cache identity cannot be rebound to different evidence")

    assert store.get_part("part-2") == second_plan
    assert store.get_evidence_unit("unit-2") is None
    store.close()


def test_publication_rolls_back_unit_and_part_when_cache_write_fails(tmp_path: Path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    store.upsert_part_plans((part_plan(),))
    running = store.mark_running("part-1", attempt=1)
    with store._transaction() as connection:
        connection.execute(
            """CREATE TRIGGER reject_evidence_cache
               BEFORE INSERT ON evidence_cache
               BEGIN SELECT RAISE(ABORT, 'forced cache failure'); END"""
        )

    try:
        store.publish_evidence(
            "part-1", evidence_unit(), cache_key="cache-v1", completed_at=NOW
        )
    except sqlite3.IntegrityError as error:
        assert "forced cache failure" in str(error)
    else:
        raise AssertionError("the trigger must reject the cache write")

    assert store.get_part("part-1") == running
    assert store.get_evidence_unit("unit-1") is None
    assert store.cached_evidence("cache-v1") is None
    store.close()


def test_snapshots_persist_exact_coverage_and_missing_dependencies_roll_back(
    tmp_path: Path,
):
    path = tmp_path / "evidence.sqlite3"
    store = EvidenceStore(path)
    store.upsert_part_plans((part_plan(),))
    store.publish_evidence(
        "part-1", evidence_unit(), cache_key="cache-v1", completed_at=NOW
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
    store.publish_evidence(
        "part-1", first_unit, cache_key="cache-1", completed_at=NOW
    )
    store.publish_evidence(
        "part-2", second_unit, cache_key="cache-2", completed_at=NOW
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
    assert store.cached_evidence("cache-1") is None
    assert store.get_snapshot("snapshot-dependent") is None
    assert store.get_part("part-2").state is PartState.PROCESSED
    assert store.get_evidence_unit("unit-2") == second_unit
    assert store.cached_evidence("cache-2") == second_unit
    assert store.get_snapshot("snapshot-sibling") == sibling
    assert store.coverage("workspace-1", "revision-1") == sibling.coverage
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
    store.publish_evidence(
        "part-1", evidence_unit(), cache_key="cache-1", completed_at=NOW
    )
    other_unit = evidence_unit(
        unit_id="other-unit",
        part_id="other-part",
        entry_id="other-entry",
    )
    store.publish_evidence(
        "other-part", other_unit, cache_key="cache-2", completed_at=NOW
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
    assert store.cached_evidence("cache-1") is None
    assert store.get_part("other-part") == other.model_copy(
        update={"state": PartState.PROCESSED}
    )
    assert store.cached_evidence("cache-2") == other_unit
    store.close()
