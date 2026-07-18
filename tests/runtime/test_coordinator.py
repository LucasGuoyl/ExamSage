import time
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from langgraph.types import Command

from exam_predictor.runtime.coordinator import RuntimeCoordinator
from exam_predictor.runtime.models import (
    ConnectProviderRequest,
    EventType,
    ProviderProfile,
    RunStatus,
)
from exam_predictor.runtime.provider_sessions import ProviderSessionRegistry
from exam_predictor.runtime.store import RuntimeStore


def wait_for_status(store: RuntimeStore, run_id: str, status: RunStatus) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if store.get_run(run_id).status is status:
            return
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not reach {status}")


class GraphHarness:
    def __init__(
        self,
        *,
        pause_first: bool = False,
        emit_pause: bool = False,
        fail_first: Exception | None = None,
        honor_stop: bool = False,
    ):
        self.pause_first = pause_first
        self.emit_pause = emit_pause
        self.fail_first = fail_first
        self.honor_stop = honor_stop
        self.first_started = Event()
        self.release_first = Event()
        self.initial_calls = 0
        self.resume_calls = 0
        self.configs: list[dict] = []

    def factory(self, dependencies, saver):
        del saver
        harness = self

        class FakeGraph:
            def invoke(self, value, config):
                harness.configs.append(config)
                if isinstance(value, Command):
                    assert value.resume == {"action": "resume"}
                    harness.resume_calls += 1
                    return {"assistant_message": "resumed answer"}
                harness.initial_calls += 1
                if harness.initial_calls == 1:
                    harness.first_started.set()
                    assert harness.release_first.wait(timeout=2)
                    if harness.fail_first is not None:
                        raise harness.fail_first
                    should_pause = harness.pause_first or (
                        harness.honor_stop
                        and dependencies.controls.is_stop_requested(value["run_id"])
                    )
                    if should_pause:
                        if harness.emit_pause:
                            dependencies.emit(
                                value["run_id"],
                                "paused",
                                "paused",
                                "The run is paused at a safe boundary.",
                            )
                        return {"__interrupt__": [{"kind": "stopped"}]}
                return {"assistant_message": "completed answer"}

        return FakeGraph()


def registry() -> ProviderSessionRegistry:
    provider = SimpleNamespace(
        name="fake",
        capabilities=SimpleNamespace(chat=True),
    )
    sessions = ProviderSessionRegistry(factory=lambda config: provider)
    sessions.connect(
        ConnectProviderRequest(
            profile=ProviderProfile(profile_id="primary", provider="gemini"),
            api_key="test-only-key",
        )
    )
    return sessions


def runtime_for(
    tmp_path: Path,
    harness: GraphHarness,
    *,
    sessions: ProviderSessionRegistry | None = None,
) -> tuple[RuntimeStore, RuntimeCoordinator]:
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=sessions or registry(),
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        graph_factory=harness.factory,
    )
    return store, runtime


def test_second_message_queues_until_first_finishes(tmp_path: Path):
    harness = GraphHarness()
    store, runtime = runtime_for(tmp_path, harness)
    runtime.start()
    try:
        first = runtime.submit_message("course-1", "primary", "First")
        assert harness.first_started.wait(timeout=2)
        second = runtime.submit_message("course-2", "primary", "Second")
        assert second.status is RunStatus.QUEUED
        harness.release_first.set()
        wait_for_status(store, first.run_id, RunStatus.COMPLETED)
        wait_for_status(store, second.run_id, RunStatus.COMPLETED)
        assert harness.initial_calls == 2
    finally:
        runtime.shutdown()


def test_stop_pauses_and_resume_continues_same_run(tmp_path: Path):
    harness = GraphHarness(pause_first=True)
    store, runtime = runtime_for(tmp_path, harness)
    runtime.start()
    try:
        run = runtime.submit_message("course-1", "primary", "Explain")
        assert harness.first_started.wait(timeout=2)
        runtime.stop(run.run_id)
        harness.release_first.set()
        wait_for_status(store, run.run_id, RunStatus.PAUSED)
        runtime.resume(run.run_id)
        wait_for_status(store, run.run_id, RunStatus.COMPLETED)
        types = [event.event_type for event in store.list_events(run.run_id)]
        assert EventType.STOP_REQUESTED in types
        assert EventType.RESUMED in types
        assert types.count(EventType.PAUSED) == 1
        assert harness.resume_calls == 1
        assert harness.configs == [
            {"configurable": {"thread_id": "course-1"}},
            {"configurable": {"thread_id": "course-1"}},
        ]
    finally:
        runtime.shutdown()


def test_coordinator_does_not_duplicate_graph_emitted_paused_event(tmp_path: Path):
    harness = GraphHarness(pause_first=True, emit_pause=True)
    store, runtime = runtime_for(tmp_path, harness)
    runtime.start()
    try:
        run = runtime.submit_message("course-1", "primary", "Explain")
        assert harness.first_started.wait(timeout=2)
        runtime.stop(run.run_id)
        harness.release_first.set()
        wait_for_status(store, run.run_id, RunStatus.PAUSED)

        paused_events = [
            event
            for event in store.list_events(run.run_id)
            if event.event_type is EventType.PAUSED
        ]
        assert len(paused_events) == 1
        assert paused_events[0].message == "The run is paused at a safe boundary."
    finally:
        runtime.shutdown()


def test_resume_requires_reconnected_provider_without_changing_status(tmp_path: Path):
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    run = store.create_run("course-1", "primary", "Explain", RunStatus.PAUSED)
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=ProviderSessionRegistry(factory=lambda config: object()),
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        graph_factory=GraphHarness().factory,
    )
    try:
        with pytest.raises(KeyError, match="Connect provider profile 'primary'"):
            runtime.resume(run.run_id)
        assert store.get_run(run.run_id).status is RunStatus.PAUSED
    finally:
        runtime.shutdown()


def test_restart_recovers_unfinished_run_and_keeps_queued_work_blocked(tmp_path: Path):
    store_path = tmp_path / "runtime.sqlite3"
    seeded_store = RuntimeStore(store_path)
    unfinished = seeded_store.create_run(
        "course-1",
        "primary",
        "First",
        RunStatus.RUNNING,
    )
    queued = seeded_store.create_run(
        "course-2",
        "primary",
        "Second",
        RunStatus.QUEUED,
    )
    harness = GraphHarness()
    reconstructed_store = RuntimeStore(store_path)
    runtime = RuntimeCoordinator(
        store=reconstructed_store,
        provider_sessions=registry(),
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        graph_factory=harness.factory,
    )

    runtime.start()
    try:
        assert reconstructed_store.get_run(unfinished.run_id).status is RunStatus.PAUSED
        assert reconstructed_store.get_run(queued.run_id).status is RunStatus.QUEUED
        paused_events = [
            event
            for event in reconstructed_store.list_events(unfinished.run_id)
            if event.event_type is EventType.PAUSED
        ]
        assert len(paused_events) == 1
        assert harness.initial_calls == 0
    finally:
        runtime.shutdown()


def test_paused_run_blocks_queued_work_until_it_resumes(tmp_path: Path):
    harness = GraphHarness(pause_first=True)
    store, runtime = runtime_for(tmp_path, harness)
    runtime.start()
    try:
        first = runtime.submit_message("course-1", "primary", "First")
        assert harness.first_started.wait(timeout=2)
        second = runtime.submit_message("course-2", "primary", "Second")
        runtime.stop(first.run_id)
        harness.release_first.set()
        wait_for_status(store, first.run_id, RunStatus.PAUSED)
        time.sleep(0.05)
        assert store.get_run(second.run_id).status is RunStatus.QUEUED
        assert harness.initial_calls == 1

        runtime.resume(first.run_id)
        wait_for_status(store, first.run_id, RunStatus.COMPLETED)
        wait_for_status(store, second.run_id, RunStatus.COMPLETED)
        assert harness.initial_calls == 2
    finally:
        runtime.shutdown()


def test_failed_run_is_safe_and_starts_next_queued_run(tmp_path: Path):
    secret = "test-only-key"
    harness = GraphHarness(fail_first=RuntimeError(f"provider leaked {secret}"))
    store, runtime = runtime_for(tmp_path, harness)
    runtime.start()
    try:
        first = runtime.submit_message("course-1", "primary", "First")
        assert harness.first_started.wait(timeout=2)
        second = runtime.submit_message("course-2", "primary", "Second")
        harness.release_first.set()
        wait_for_status(store, first.run_id, RunStatus.FAILED)
        wait_for_status(store, second.run_id, RunStatus.COMPLETED)

        assert secret not in store.get_run(first.run_id).error
        assert secret.encode() not in (tmp_path / "runtime.sqlite3").read_bytes()
        assert harness.initial_calls == 2
    finally:
        runtime.shutdown()


def test_pause_all_requests_safe_pause_for_running_work(tmp_path: Path):
    harness = GraphHarness(honor_stop=True)
    store, runtime = runtime_for(tmp_path, harness)
    runtime.start()
    try:
        run = runtime.submit_message("course-1", "primary", "Explain")
        assert harness.first_started.wait(timeout=2)
        runtime.pause_all()
        harness.release_first.set()
        wait_for_status(store, run.run_id, RunStatus.PAUSED)
    finally:
        runtime.shutdown()


def test_shutdown_pauses_active_work_and_leaves_queue_order_intact(tmp_path: Path):
    harness = GraphHarness(honor_stop=True)
    store, runtime = runtime_for(tmp_path, harness)
    runtime.start()
    first = runtime.submit_message("course-1", "primary", "First")
    assert harness.first_started.wait(timeout=2)
    second = runtime.submit_message("course-2", "primary", "Second")

    shutdown = Thread(target=runtime.shutdown)
    shutdown.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if runtime.controls.is_stop_requested(first.run_id):
            break
        time.sleep(0.01)
    else:
        raise AssertionError("shutdown did not request a stop")
    harness.release_first.set()
    shutdown.join(timeout=2)

    assert store.get_run(first.run_id).status is RunStatus.PAUSED
    assert store.get_run(second.run_id).status is RunStatus.QUEUED
    assert not shutdown.is_alive()
    assert not runtime._thread.is_alive()


def test_shutdown_does_not_start_queued_work_when_active_work_finishes(tmp_path: Path):
    harness = GraphHarness()
    store, runtime = runtime_for(tmp_path, harness)
    runtime.start()
    first = runtime.submit_message("course-1", "primary", "First")
    assert harness.first_started.wait(timeout=2)
    second = runtime.submit_message("course-2", "primary", "Second")

    shutdown = Thread(target=runtime.shutdown)
    shutdown.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if runtime.controls.is_stop_requested(first.run_id):
            break
        time.sleep(0.01)
    else:
        raise AssertionError("shutdown did not request a stop")
    harness.release_first.set()
    shutdown.join(timeout=2)

    assert store.get_run(first.run_id).status is RunStatus.COMPLETED
    assert store.get_run(second.run_id).status is RunStatus.QUEUED
    assert harness.initial_calls == 1
    assert not shutdown.is_alive()
    assert not runtime._thread.is_alive()


def test_shutdown_timeout_is_explicit_and_repeated_shutdown_cleans_worker(tmp_path: Path):
    harness = GraphHarness(honor_stop=True)
    store, runtime = runtime_for(tmp_path, harness)
    runtime.start()
    run = runtime.submit_message("course-1", "primary", "Explain")
    assert harness.first_started.wait(timeout=2)

    try:
        with pytest.raises(
            TimeoutError,
            match="RuntimeCoordinator worker did not stop before the shutdown timeout",
        ):
            runtime.shutdown(timeout=0.01)
        assert runtime._thread.is_alive()
    finally:
        harness.release_first.set()
        runtime.shutdown(timeout=2)

    assert store.get_run(run.run_id).status is RunStatus.PAUSED
    assert not runtime._thread.is_alive()
