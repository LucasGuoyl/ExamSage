from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading

import pytest

import exam_predictor.runtime.store as store_module
from exam_predictor.runtime.models import (
    EventType,
    ProviderProfile,
    RunStatus,
    SavedProviderProfile,
)
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


def test_every_store_operation_explicitly_closes_its_sqlite_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sqlite_connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        was_closed = False

        def close(self) -> None:
            self.was_closed = True
            super().close()

    connections: list[TrackingConnection] = []

    def tracked_connect(*args, **kwargs):
        kwargs["factory"] = TrackingConnection
        connection = sqlite_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(store_module.sqlite3, "connect", tracked_connect)
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    run = store.create_run("course-1", "primary", "Explain", RunStatus.RUNNING)
    store.get_run(run.run_id)
    store.active_run()
    store.next_queued_run()
    store.set_status(run.run_id, RunStatus.STOPPING)
    store.append_event(run.run_id, EventType.PROGRESS, "planning", "Working")
    store.list_events(run.run_id)
    store.list_by_status({RunStatus.STOPPING})
    store.recover_unfinished()
    store.set_status_and_append_event(
        run.run_id,
        RunStatus.COMPLETED,
        EventType.COMPLETED,
        "complete",
        "Done",
    )
    store.save_provider_profile(saved_provider_profile())
    store.list_saved_provider_profiles()
    store.mark_provider_reconnect_required("primary")

    assert connections
    assert all(connection.was_closed for connection in connections)


def saved_provider_profile(
    *,
    profile_id: str = "primary",
    credential_expected: bool = True,
    reconnect_required: bool = False,
) -> SavedProviderProfile:
    return SavedProviderProfile(
        profile=ProviderProfile(
            profile_id=profile_id,
            provider="custom",
            base_url="https://models.example/v1",
            balanced_model="course-model",
        ),
        capabilities={"chat": True, "web_search": False},
        credential_expected=credential_expected,
        reconnect_required=reconnect_required,
        updated_at=datetime(2026, 7, 24, 12, tzinfo=timezone.utc),
    )


def test_saved_provider_profile_round_trips_without_credential_material(tmp_path: Path):
    path = tmp_path / "runtime.sqlite3"
    sentinel = "vault-secret-not-for-sqlite"
    store = RuntimeStore(path)

    store.save_provider_profile(saved_provider_profile())

    assert store.list_saved_provider_profiles() == [saved_provider_profile()]
    serialized = path.read_bytes().decode("latin1")
    assert "course-model" in serialized
    assert "web_search" in serialized
    assert sentinel not in serialized
    assert "api_key" not in serialized


def test_saved_provider_profile_upsert_and_reconnect_marker_are_atomic(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.save_provider_profile(saved_provider_profile())
    replacement = saved_provider_profile(
        credential_expected=False,
        reconnect_required=True,
    ).model_copy(
        update={"capabilities": {"chat": False}, "updated_at": datetime(2026, 7, 24, 13, tzinfo=timezone.utc)}
    )

    store.save_provider_profile(replacement)
    assert store.list_saved_provider_profiles() == [replacement]

    store.save_provider_profile(saved_provider_profile())
    store.mark_provider_reconnect_required("primary")
    marked = store.list_saved_provider_profiles()[0]
    assert marked.credential_expected is False
    assert marked.reconnect_required is True
    assert marked.updated_at > saved_provider_profile().updated_at


def test_mark_provider_reconnect_required_rejects_missing_profile(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")

    with pytest.raises(KeyError, match="Saved provider profile 'missing' was not found"):
        store.mark_provider_reconnect_required("missing")


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("profile_json", '{"profile_id": "primary", "provider": "unknown"}'),
        ("capabilities_json", '{"chat": "not-a-boolean"}'),
    ],
)
def test_saved_provider_profile_rows_are_validated_on_read(
    tmp_path: Path,
    column: str,
    value: str,
):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.save_provider_profile(saved_provider_profile())
    with store._connection() as connection:
        connection.execute(
            f"UPDATE saved_provider_profiles SET {column} = ? WHERE profile_id = ?",
            (value, "primary"),
        )

    assert store.list_saved_provider_profiles() == []


def test_malformed_saved_profile_does_not_abort_other_profile_reads(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    valid = saved_provider_profile(profile_id="valid")
    store.save_provider_profile(valid)
    with store._connection() as connection:
        connection.execute(
            """INSERT INTO saved_provider_profiles(
                   profile_id, profile_json, capabilities_json,
                   credential_expected, reconnect_required, updated_at
               ) VALUES (?, ?, ?, 1, 0, ?)""",
            (
                "malformed",
                '{"profile_id":"malformed","provider":"unknown"}',
                '{"chat":true}',
                valid.updated_at.isoformat(),
            ),
        )

    assert store.list_saved_provider_profiles() == [valid]


def test_saved_profile_sql_and_json_ids_must_match(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    profile = saved_provider_profile(profile_id="json-id")
    with store._connection() as connection:
        connection.execute(
            """INSERT INTO saved_provider_profiles(
                   profile_id, profile_json, capabilities_json,
                   credential_expected, reconnect_required, updated_at
               ) VALUES (?, ?, ?, 1, 0, ?)""",
            (
                "sql-id",
                profile.profile.model_dump_json(),
                '{"chat":true}',
                profile.updated_at.isoformat(),
            ),
        )

    assert store.list_saved_provider_profiles() == []


def test_unsafe_saved_custom_url_is_filtered_without_echoing_query_secret(
    tmp_path: Path,
):
    sentinel = "saved-query-sentinel"
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    unsafe = saved_provider_profile().model_copy(
        update={
            "profile": ProviderProfile(
                profile_id="unsafe",
                provider="custom",
                base_url=f"https://models.example/v1?api_key={sentinel}",
            )
        }
    )
    with store._connection() as connection:
        connection.execute(
            """INSERT INTO saved_provider_profiles(
                   profile_id, profile_json, capabilities_json,
                   credential_expected, reconnect_required, updated_at
               ) VALUES (?, ?, ?, 1, 0, ?)""",
            (
                "unsafe",
                unsafe.profile.model_dump_json(),
                '{"chat":true}',
                unsafe.updated_at.isoformat(),
            ),
        )

    assert store.list_saved_provider_profiles() == []
    assert sentinel not in repr(store.list_saved_provider_profiles())


def test_connect_closes_new_connection_when_pragma_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sqlite_connect = sqlite3.connect

    class FailingPragmaConnection(sqlite3.Connection):
        was_closed = False

        def execute(self, sql, parameters=()):
            if sql == "PRAGMA foreign_keys=ON":
                raise sqlite3.OperationalError("forced pragma failure")
            return super().execute(sql, parameters)

        def close(self) -> None:
            self.was_closed = True
            super().close()

    connections: list[FailingPragmaConnection] = []

    def tracked_connect(*args, **kwargs):
        kwargs["factory"] = FailingPragmaConnection
        connection = sqlite_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(store_module.sqlite3, "connect", tracked_connect)
    store = RuntimeStore.__new__(RuntimeStore)
    store.db_path = tmp_path / "runtime.sqlite3"

    with pytest.raises(sqlite3.OperationalError, match="forced pragma failure"):
        store._connect()

    assert len(connections) == 1
    assert connections[0].was_closed


def test_store_orders_the_global_serial_queue(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    first = store.create_run("course-1", "primary", "First", RunStatus.RUNNING)
    second = store.create_run("course-2", "primary", "Second", RunStatus.QUEUED)
    assert store.active_run().run_id == first.run_id
    assert store.next_queued_run().run_id == second.run_id


def test_global_queue_preserves_insertion_order_for_identical_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    timestamp = "2026-07-17T12:00:00+00:00"
    monkeypatch.setattr(RuntimeStore, "_now", staticmethod(lambda: timestamp))
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    runs = [
        store.create_run(f"course-{index}", "primary", f"Message {index}", RunStatus.QUEUED)
        for index in range(3)
    ]

    assert store.next_queued_run().run_id == runs[0].run_id
    store.set_status(runs[0].run_id, RunStatus.COMPLETED)
    assert store.next_queued_run().run_id == runs[1].run_id
    store.set_status(runs[1].run_id, RunStatus.FAILED)
    assert store.next_queued_run().run_id == runs[2].run_id
    store.set_status(runs[2].run_id, RunStatus.COMPLETED)
    assert store.next_queued_run() is None


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

    with store._connection() as connection:
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


def test_store_indexes_match_global_queue_and_event_cursor_queries(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")

    with store._connection() as connection:
        run_indexes = {
            row["name"]: tuple(
                column["name"]
                for column in connection.execute(f"PRAGMA index_info('{row['name']}')")
            )
            for row in connection.execute("PRAGMA index_list('agent_runs')")
        }
        event_indexes = {
            row["name"]: tuple(
                column["name"]
                for column in connection.execute(f"PRAGMA index_info('{row['name']}')")
            )
            for row in connection.execute("PRAGMA index_list('agent_events')")
        }

    assert ("status", "created_at") in run_indexes.values()
    assert ("run_id", "sequence") in event_indexes.values()


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


def test_status_event_transition_rolls_back_when_event_insert_fails(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    run = store.create_run("course-1", "primary", "Explain", RunStatus.RUNNING)
    with store._connection() as connection:
        connection.execute(
            """CREATE TRIGGER reject_failed_event_insert
               BEFORE INSERT ON agent_events
               WHEN NEW.event_type = 'failed'
               BEGIN
                 SELECT RAISE(ABORT, 'forced event insert failure');
               END"""
        )

    with pytest.raises(sqlite3.IntegrityError, match="forced event insert failure"):
        store.set_status_and_append_event(
            run.run_id,
            RunStatus.FAILED,
            EventType.FAILED,
            "failed",
            "Failed",
            error="safe failure",
        )

    unchanged = store.get_run(run.run_id)
    assert unchanged.status is RunStatus.RUNNING
    assert unchanged.error is None
    assert store.list_events(run.run_id) == []


@pytest.mark.parametrize(
    "unsafe_value",
    [
        pytest.param(RuntimeError("provider failed"), id="exception-instance"),
        pytest.param(threading.Lock(), id="threading-lock"),
        pytest.param(object(), id="opaque-object"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_event_payload_rejects_unsafe_values_without_partial_writes(
    tmp_path: Path, unsafe_value: object
):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    run = store.create_run("course-1", "primary", "Explain", RunStatus.RUNNING)

    with pytest.raises((TypeError, ValueError)):
        store.append_event(
            run.run_id,
            EventType.PROGRESS,
            "research",
            "Working",
            {"unsafe": unsafe_value},
        )

    assert store.list_events(run.run_id) == []


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


def test_recovery_reuses_current_paused_boundary_without_duplicate_event(
    tmp_path: Path,
):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    run = store.create_run("course-1", "primary", "Explain", RunStatus.STOPPING)
    store.append_event(run.run_id, EventType.STOP_REQUESTED, "stopping", "Stop requested.")
    existing_pause = store.append_event(
        run.run_id,
        EventType.PAUSED,
        "paused",
        "The run is paused at a safe boundary.",
    )

    assert store.recover_unfinished() == [run.run_id]

    assert store.get_run(run.run_id).status is RunStatus.PAUSED
    paused_events = [
        event
        for event in store.list_events(run.run_id)
        if event.event_type is EventType.PAUSED
    ]
    assert paused_events == [existing_pause]


def test_simultaneous_recovery_transitions_and_emits_once(tmp_path: Path):
    path = tmp_path / "runtime.sqlite3"
    seed = RuntimeStore(path)
    runs = [
        seed.create_run("course-1", "primary", "Running", RunStatus.RUNNING),
        seed.create_run("course-2", "primary", "Stopping", RunStatus.STOPPING),
    ]
    stores = (RuntimeStore(path), RuntimeStore(path))
    barrier = threading.Barrier(len(stores))

    def recover(store: RuntimeStore) -> list[str]:
        barrier.wait()
        return store.recover_unfinished()

    with ThreadPoolExecutor(max_workers=len(stores)) as executor:
        recovery_results = list(executor.map(recover, stores))

    recovered_ids = [run_id for result in recovery_results for run_id in result]
    assert sorted(recovered_ids) == sorted(run.run_id for run in runs)
    for run in runs:
        assert seed.get_run(run.run_id).status is RunStatus.PAUSED
        events = seed.list_events(run.run_id)
        assert len(events) == 1
        assert events[0].event_type is EventType.PAUSED


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
        lambda: store.set_status_and_append_event(
            "missing",
            RunStatus.PAUSED,
            EventType.PAUSED,
            "paused",
            "Paused",
        ),
    ):
        try:
            operation()
        except (KeyError, sqlite3.IntegrityError):
            pass
        else:
            raise AssertionError("missing run operation should fail")
