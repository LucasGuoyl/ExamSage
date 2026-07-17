from concurrent.futures import ThreadPoolExecutor
from datetime import timezone
from pathlib import Path
import sqlite3

from exam_predictor.runtime.models import EventType, RunStatus
from exam_predictor.runtime.store import RuntimeStore


def test_store_serializes_messages_and_events_without_secrets(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    run = store.create_run("course-1", "primary", "Explain limits.", RunStatus.RUNNING)
    event = store.append_event(
        run.run_id,
        EventType.STARTED,
        "planning",
        "Planning started.",
        {"safe": True},
    )
    assert event.sequence == 1
    assert store.get_run(run.run_id).message == "Explain limits."
    assert store.list_events(run.run_id, after=0) == [event]
    assert "api_key" not in (tmp_path / "runtime.sqlite3").read_bytes().decode(
        "utf-8", errors="ignore"
    )


def test_store_orders_the_global_serial_queue(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    first = store.create_run("course-1", "primary", "First", RunStatus.RUNNING)
    second = store.create_run("course-2", "primary", "Second", RunStatus.QUEUED)
    assert store.active_run().run_id == first.run_id
    assert store.next_queued_run().run_id == second.run_id


def test_paused_run_remains_globally_active(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    paused = store.create_run("course-1", "primary", "First", RunStatus.PAUSED)
    store.create_run("course-2", "primary", "Second", RunStatus.QUEUED)
    assert store.active_run().run_id == paused.run_id


def test_store_recovers_unclean_runs_as_paused(tmp_path: Path):
    path = tmp_path / "runtime.sqlite3"
    store = RuntimeStore(path)
    run = store.create_run("course-1", "primary", "Explain", RunStatus.RUNNING)
    recovered = RuntimeStore(path).recover_unfinished()
    assert recovered == [run.run_id]
    assert RuntimeStore(path).get_run(run.run_id).status is RunStatus.PAUSED


def test_snapshots_and_events_have_utc_aware_timestamps(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    run = store.create_run("course-1", "primary", "Explain", RunStatus.RUNNING)
    event = store.append_event(run.run_id, EventType.STARTED, "planning", "Started")

    assert run.created_at.utcoffset() == timezone.utc.utcoffset(run.created_at)
    assert run.updated_at.utcoffset() == timezone.utc.utcoffset(run.updated_at)
    assert event.created_at is not None
    assert event.created_at.utcoffset() == timezone.utc.utcoffset(event.created_at)


def test_store_enables_wal_and_foreign_keys_for_each_connection(tmp_path: Path):
    path = tmp_path / "runtime.sqlite3"
    store = RuntimeStore(path)
    run = store.create_run("course-1", "primary", "Explain", RunStatus.RUNNING)

    with store._connect() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            connection.execute(
                """INSERT INTO agent_events(
                       run_id, event_type, stage, message, payload_json, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                ("missing", "started", "stage", "message", "{}", run.created_at.isoformat()),
            )
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("agent_events must retain its foreign-key constraint")


def test_event_payload_round_trips_json_safely(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    run = store.create_run("course-1", "primary", "Explain", RunStatus.RUNNING)
    payload = {
        "unicode": "\u6570\u5b66",
        "nested": {"items": [True, None, 3.5]},
        "looks_like_sql": "'); DROP TABLE agent_runs; --",
    }

    event = store.append_event(run.run_id, EventType.PROGRESS, "research", "Working", payload)

    assert event.payload == payload
    assert store.get_run(run.run_id).status is RunStatus.RUNNING


def test_store_methods_are_safe_across_concurrent_threads(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")

    def create(index: int):
        return store.create_run("course", "primary", f"Message {index}", RunStatus.QUEUED)

    with ThreadPoolExecutor(max_workers=4) as executor:
        runs = list(executor.map(create, range(8)))

    assert {run.run_id for run in runs} == {
        run.run_id for run in store.list_by_status({RunStatus.QUEUED})
    }


def test_recovery_is_idempotent_and_leaves_settled_runs_unchanged(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    running = store.create_run("course-1", "primary", "Running", RunStatus.RUNNING)
    stopping = store.create_run("course-2", "primary", "Stopping", RunStatus.STOPPING)
    unchanged = {
        status: store.create_run("course", "primary", status.value, status)
        for status in (
            RunStatus.QUEUED,
            RunStatus.PAUSED,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
        )
    }

    assert store.recover_unfinished() == [running.run_id, stopping.run_id]
    assert store.recover_unfinished() == []
    for run in (running, stopping):
        events = store.list_events(run.run_id)
        assert len(events) == 1
        assert events[0].event_type is EventType.PAUSED
        assert events[0].message == (
            "ExamSage stopped before this run completed. Select Resume to continue."
        )
    for status, run in unchanged.items():
        assert store.get_run(run.run_id).status is status
        assert store.list_events(run.run_id) == []


def test_status_updates_and_missing_runs(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    run = store.create_run("course", "primary", "Explain", RunStatus.QUEUED)

    updated = store.set_status(run.run_id, RunStatus.FAILED, "provider failed")

    assert updated.status is RunStatus.FAILED
    assert updated.error == "provider failed"
    assert store.list_by_status({RunStatus.FAILED}) == [updated]
    assert store.list_by_status(set()) == []

    for operation in (
        lambda: store.get_run("missing"),
        lambda: store.set_status("missing", RunStatus.PAUSED),
        lambda: store.append_event("missing", EventType.PAUSED, "paused", "Paused"),
    ):
        try:
            operation()
        except (KeyError, sqlite3.IntegrityError):
            pass
        else:
            raise AssertionError("missing run operation should fail")
