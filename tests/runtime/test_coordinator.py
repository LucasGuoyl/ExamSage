import json
import time
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from exam_predictor.graphs.kernel import KernelDependencies, build_kernel_graph
from exam_predictor.runtime.control import RunControlRegistry
from exam_predictor.runtime.coordinator import (
    ProviderProfileInUseError,
    RuntimeCoordinator,
)
from exam_predictor.runtime.credential_vault import VaultUnavailableError
from exam_predictor.runtime.models import (
    ConnectProviderRequest,
    EventType,
    ProviderProfile,
    RunStatus,
)
from exam_predictor.runtime.provider_sessions import ProviderSessionRegistry
from exam_predictor.runtime.store import RuntimeStore
from exam_predictor.tools.kernel import KernelPlanner, KernelToolRegistry


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
        self.run_id: str | None = None

    def factory(self, dependencies, saver):
        del saver
        harness = self

        class FakeGraph:
            def invoke(self, value, config):
                harness.configs.append(config)
                if isinstance(value, Command):
                    assert value.resume == {"action": "resume"}
                    harness.resume_calls += 1
                    dependencies.emit(
                        harness.run_id,
                        "resumed",
                        "planning",
                        "The run resumed from its checkpoint.",
                    )
                    return {"assistant_message": "resumed answer"}
                harness.initial_calls += 1
                harness.run_id = value["run_id"]
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


class FakeVault:
    def __init__(self) -> None:
        self.secrets: dict[str, str] = {}
        self.calls: list[tuple[str, str]] = []
        self.fail_save = False
        self.fail_load = False

    def save(self, profile_id: str, api_key: str) -> None:
        self.calls.append(("save", profile_id))
        if self.fail_save:
            raise VaultUnavailableError("Secure credential storage is unavailable")
        self.secrets[profile_id] = api_key

    def load(self, profile_id: str) -> str | None:
        self.calls.append(("load", profile_id))
        if self.fail_load:
            raise VaultUnavailableError("Secure credential storage is unavailable")
        return self.secrets.get(profile_id)

    def exists(self, profile_id: str) -> bool:
        return profile_id in self.secrets

    def delete(self, profile_id: str) -> None:
        self.calls.append(("delete", profile_id))
        self.secrets.pop(profile_id, None)


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


def provider_request(api_key: str) -> ConnectProviderRequest:
    return ConnectProviderRequest(
        profile=ProviderProfile(profile_id="primary", provider="gemini"),
        api_key=api_key,
    )


def credential_sessions() -> ProviderSessionRegistry:
    return ProviderSessionRegistry(
        factory=lambda config: SimpleNamespace(
            identity=config["api_key"],
            name="fake",
            capabilities=SimpleNamespace(chat=True, web_search=False),
        )
    )


def test_connect_validates_then_saves_credential_and_nonsecret_profile(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    sentinel = "vault-secret-for-connect"
    vault = FakeVault()
    sessions = credential_sessions()
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=sessions,
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        vault=vault,
    )

    descriptor = runtime.connect_provider(provider_request(sentinel))

    assert sessions.get_provider("primary").identity == sentinel
    assert vault.secrets == {"primary": sentinel}
    assert descriptor.credential_saved is True
    assert descriptor.credential_warning is None
    saved = store.list_saved_provider_profiles()
    assert len(saved) == 1
    assert saved[0].credential_expected is True
    assert saved[0].reconnect_required is False
    serialized = "\n".join(
        [
            (tmp_path / "runtime.sqlite3").read_bytes().decode("latin1"),
            descriptor.model_dump_json(),
            repr(store.list_saved_provider_profiles()),
            repr(vault.calls),
            caplog.text,
        ]
    )
    assert sentinel not in serialized


def test_vault_save_failure_keeps_validated_memory_session_and_safe_metadata(tmp_path: Path):
    sentinel = "vault-secret-save-outage"
    vault = FakeVault()
    vault.fail_save = True
    sessions = credential_sessions()
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=sessions,
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        vault=vault,
    )

    descriptor = runtime.connect_provider(provider_request(sentinel))

    assert sessions.get_provider("primary").identity == sentinel
    assert descriptor.credential_saved is False
    assert descriptor.credential_warning == RuntimeCoordinator.CREDENTIAL_WARNING
    saved = store.list_saved_provider_profiles()[0]
    assert saved.credential_expected is False
    assert saved.reconnect_required is True
    assert sentinel not in descriptor.model_dump_json()
    assert sentinel not in (tmp_path / "runtime.sqlite3").read_bytes().decode("latin1")


def test_start_restores_saved_provider_without_starting_paused_work(tmp_path: Path):
    vault = FakeVault()
    first_sessions = credential_sessions()
    path = tmp_path / "runtime.sqlite3"
    store = RuntimeStore(path)
    first = RuntimeCoordinator(
        store=store,
        provider_sessions=first_sessions,
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        vault=vault,
    )
    first.connect_provider(provider_request("restored-vault-secret"))
    paused = store.create_run("course-1", "primary", "Explain", RunStatus.PAUSED)

    restored_sessions = credential_sessions()
    restored = RuntimeCoordinator(
        store=RuntimeStore(path),
        provider_sessions=restored_sessions,
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        vault=vault,
    )
    restored.start()
    try:
        assert restored_sessions.get_provider("primary").identity == "restored-vault-secret"
        assert store.get_run(paused.run_id).status is RunStatus.PAUSED
    finally:
        restored.shutdown()


@pytest.mark.parametrize("vault_unavailable", [False, True])
def test_start_marks_reconnect_when_saved_credential_cannot_be_loaded(
    tmp_path: Path,
    vault_unavailable: bool,
):
    vault = FakeVault()
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    first = RuntimeCoordinator(
        store=store,
        provider_sessions=credential_sessions(),
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        vault=vault,
    )
    first.connect_provider(provider_request("revoked-secret"))
    vault.secrets.clear()
    vault.fail_load = vault_unavailable

    restored_sessions = credential_sessions()
    restored = RuntimeCoordinator(
        store=RuntimeStore(store.db_path),
        provider_sessions=restored_sessions,
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        vault=vault,
    )
    restored.start()
    try:
        saved = store.list_saved_provider_profiles()[0]
        assert saved.credential_expected is False
        assert saved.reconnect_required is True
        assert restored_sessions.has_provider("primary") is False
    finally:
        restored.shutdown()


def test_start_marks_reconnect_when_provider_rejects_restored_credential(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    sentinel = "vault-secret-revoked-by-provider"
    vault = FakeVault()
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    first = RuntimeCoordinator(
        store=store,
        provider_sessions=credential_sessions(),
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        vault=vault,
    )
    first.connect_provider(provider_request(sentinel))

    def rejecting_factory(config):
        raise RuntimeError(f"rejected {config['api_key']} at C:/private/course")

    restored_sessions = ProviderSessionRegistry(factory=rejecting_factory)
    restored = RuntimeCoordinator(
        store=RuntimeStore(store.db_path),
        provider_sessions=restored_sessions,
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        vault=vault,
    )
    restored.start()
    try:
        saved = store.list_saved_provider_profiles()[0]
        assert saved.credential_expected is False
        assert saved.reconnect_required is True
        assert restored_sessions.has_provider("primary") is False
        serialized = "\n".join(
            [
                store.db_path.read_bytes().decode("latin1"),
                repr(store.list_saved_provider_profiles()),
                caplog.text,
            ]
        )
        assert sentinel not in serialized
        assert "C:/private/course" not in serialized
    finally:
        restored.shutdown()


def test_forget_disconnects_and_deletes_only_the_credential(tmp_path: Path):
    vault = FakeVault()
    sessions = credential_sessions()
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=sessions,
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        vault=vault,
    )
    runtime.connect_provider(provider_request("forget-secret"))
    run = store.create_run("course-1", "primary", "Explain", RunStatus.PAUSED)
    event = store.append_event(run.run_id, EventType.PAUSED, "paused", "Paused")

    runtime.forget_provider_credential("primary")

    assert sessions.has_provider("primary") is False
    assert vault.secrets == {}
    assert store.get_run(run.run_id) == run
    assert store.list_events(run.run_id) == [event]
    saved = store.list_saved_provider_profiles()[0]
    assert saved.credential_expected is False
    assert saved.reconnect_required is True


@pytest.mark.parametrize("status", [RunStatus.RUNNING, RunStatus.STOPPING])
def test_forget_rejects_profiles_used_by_executing_runs_without_mutation(
    tmp_path: Path,
    status: RunStatus,
):
    vault = FakeVault()
    sessions = credential_sessions()
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=sessions,
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        vault=vault,
    )
    runtime.connect_provider(provider_request("active-secret"))
    store.create_run("course-1", "primary", "Explain", status)

    with pytest.raises(ProviderProfileInUseError, match="currently in use"):
        runtime.forget_provider_credential("primary")

    assert sessions.has_provider("primary") is True
    assert vault.secrets == {"primary": "active-secret"}
    assert ("delete", "primary") not in vault.calls


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
def test_replacing_provider_used_by_running_run_is_rejected_and_original_completes(
    tmp_path: Path,
):
    started = Event()
    release = Event()
    observed: list[str] = []

    def factory(config):
        return SimpleNamespace(
            identity=config["api_key"],
            name="fake",
            capabilities=SimpleNamespace(chat=True),
        )

    sessions = ProviderSessionRegistry(factory=factory)
    store = RuntimeStore(tmp_path / "runtime.sqlite3")

    def graph_factory(dependencies, _saver):
        class ProviderLookupGraph:
            def invoke(self, value, _config):
                observed.append(
                    dependencies.provider_sessions.get_provider(
                        value["provider_profile_id"]
                    ).identity
                )
                started.set()
                assert release.wait(timeout=2)
                observed.append(
                    dependencies.provider_sessions.get_provider(
                        value["provider_profile_id"]
                    ).identity
                )
                return {"assistant_message": "completed answer"}

        return ProviderLookupGraph()

    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=sessions,
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        graph_factory=graph_factory,
    )
    runtime.connect_provider(provider_request("original-key"))
    runtime.start()
    try:
        run = runtime.submit_message("course-1", "primary", "Explain")
        assert started.wait(timeout=2)
        with pytest.raises(RuntimeError, match="currently in use"):
            runtime.connect_provider(provider_request("replacement-key"))
        release.set()
        wait_for_status(store, run.run_id, RunStatus.COMPLETED)
        assert observed == ["original-key", "original-key"]
    finally:
        release.set()
        runtime.shutdown()


def test_replacing_provider_used_only_by_paused_run_is_allowed(tmp_path: Path):
    sessions = ProviderSessionRegistry(
        factory=lambda config: SimpleNamespace(
            identity=config["api_key"],
            name="fake",
            capabilities=SimpleNamespace(chat=True),
        )
    )
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.create_run("course-1", "primary", "Explain", RunStatus.PAUSED)
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=sessions,
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
    )
    runtime.connect_provider(provider_request("original-key"))

    descriptor = runtime.connect_provider(provider_request("replacement-key"))

    assert descriptor.profile.profile_id == "primary"
    assert sessions.get_provider("primary").identity == "replacement-key"


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


def test_resume_emits_one_resumed_event_at_graph_continuation(tmp_path: Path):
    planner_started = Event()
    release_planner = Event()
    provider = SimpleNamespace(
        name="fake",
        capabilities=SimpleNamespace(chat=True),
        models=SimpleNamespace(fast="fast", balanced="balanced"),
        calls=0,
    )

    def complete(**_kwargs):
        provider.calls += 1
        if provider.calls == 1:
            planner_started.set()
            assert release_planner.wait(timeout=2)
            content = json.dumps(
                {
                    "tool": "tutor_reply",
                    "arguments": {},
                    "reason": "Tutor request",
                }
            )
        else:
            content = "A limit is the value approached by a function."
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    provider.create_chat_completion = complete
    sessions = ProviderSessionRegistry(factory=lambda _config: provider)
    sessions.connect(
        ConnectProviderRequest(
            profile=ProviderProfile(profile_id="primary", provider="gemini"),
            api_key="test-only-key",
        )
    )
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=sessions,
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
    )

    runtime.start()
    try:
        run = runtime.submit_message("course-1", "primary", "Explain limits.")
        assert planner_started.wait(timeout=2)
        runtime.stop(run.run_id)
        release_planner.set()
        wait_for_status(store, run.run_id, RunStatus.PAUSED)
        runtime.resume(run.run_id)
        wait_for_status(store, run.run_id, RunStatus.COMPLETED)
        types = [event.event_type for event in store.list_events(run.run_id)]
        assert types.count(EventType.RESUMED) == 1
        assert types.count(EventType.PAUSED) == 1
    finally:
        release_planner.set()
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


def test_restart_resume_without_checkpoint_restarts_same_run_from_durable_payload(
    tmp_path: Path,
):
    store_path = tmp_path / "runtime.sqlite3"
    seeded_store = RuntimeStore(store_path)
    unfinished = seeded_store.create_run(
        "course-1",
        "primary",
        "Explain limits.",
        RunStatus.RUNNING,
    )
    provider = SimpleNamespace(
        name="fake",
        capabilities=SimpleNamespace(chat=True),
        models=SimpleNamespace(fast="fast", balanced="balanced"),
        calls=0,
    )

    def complete(**_kwargs):
        provider.calls += 1
        content = (
            json.dumps(
                {
                    "tool": "tutor_reply",
                    "arguments": {},
                    "reason": "Tutor request",
                }
            )
            if provider.calls == 1
            else "A limit is the value approached by a function."
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    provider.create_chat_completion = complete
    sessions = ProviderSessionRegistry(factory=lambda _config: provider)
    sessions.connect(
        ConnectProviderRequest(
            profile=ProviderProfile(profile_id="primary", provider="gemini"),
            api_key="test-only-key",
        )
    )
    runtime = RuntimeCoordinator(
        store=RuntimeStore(store_path),
        provider_sessions=sessions,
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
    )

    runtime.start()
    try:
        assert runtime.store.get_run(unfinished.run_id).status is RunStatus.PAUSED
        resumed = runtime.resume(unfinished.run_id)
        assert resumed.run_id == unfinished.run_id
        wait_for_status(runtime.store, unfinished.run_id, RunStatus.COMPLETED)
        events = runtime.store.list_events(unfinished.run_id)
        assert [event.event_type for event in events].count(EventType.RESUMED) == 1
        assert EventType.FAILED not in [event.event_type for event in events]
    finally:
        runtime.shutdown()


def test_restart_resume_continues_pending_pause_checkpoint_with_one_click(
    tmp_path: Path,
):
    store_path = tmp_path / "runtime.sqlite3"
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    store = RuntimeStore(store_path)
    interrupted = store.create_run(
        "course-1",
        "primary",
        "Explain limits.",
        RunStatus.STOPPING,
    )
    store.append_event(interrupted.run_id, EventType.STARTED, "queue", "Started")
    store.append_event(
        interrupted.run_id,
        EventType.STOP_REQUESTED,
        "stopping",
        "Stop requested.",
    )
    store.append_event(
        interrupted.run_id,
        EventType.PAUSED,
        "paused",
        "The run is paused at a safe boundary.",
    )
    queued = store.create_run(
        "course-2",
        "primary",
        "Explain derivatives.",
        RunStatus.QUEUED,
    )
    store.append_event(queued.run_id, EventType.QUEUED, "queue", "Queued")

    provider = SimpleNamespace(
        name="fake",
        capabilities=SimpleNamespace(chat=True),
        models=SimpleNamespace(fast="fast", balanced="balanced"),
        calls=0,
    )

    def complete(**_kwargs):
        provider.calls += 1
        content = (
            json.dumps(
                {
                    "tool": "tutor_reply",
                    "arguments": {},
                    "reason": "Tutor request",
                }
            )
            if provider.calls % 2 == 1
            else "A clear worked answer."
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    provider.create_chat_completion = complete
    sessions = ProviderSessionRegistry(factory=lambda _config: provider)
    sessions.connect(provider_request("test-only-key"))
    seed_controls = RunControlRegistry()
    seed_controls.request_stop(interrupted.run_id)
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = build_kernel_graph(
            KernelDependencies(
                provider_sessions=sessions,
                planner=KernelPlanner(),
                tools=KernelToolRegistry(),
                controls=seed_controls,
                emit=lambda *_args, **_kwargs: None,
            ),
            saver,
        )
        config = {"configurable": {"thread_id": interrupted.thread_id}}
        stream = graph.stream(
            RuntimeCoordinator._initial_state(interrupted),
            config,
            stream_mode="updates",
        )
        assert next(stream) == {"stop_before_plan": {"pause_pending": True}}
        stream.close()
        checkpoint = graph.get_state(config)
        assert checkpoint.next == ("pause_before_plan",)
        assert not any(task.interrupts for task in checkpoint.tasks)

    runtime = RuntimeCoordinator(
        store=RuntimeStore(store_path),
        provider_sessions=sessions,
        checkpoints_path=checkpoint_path,
    )
    runtime.start()
    try:
        assert runtime.store.get_run(interrupted.run_id).status is RunStatus.PAUSED
        assert runtime.store.get_run(queued.run_id).status is RunStatus.QUEUED
        assert [
            event.event_type for event in runtime.store.list_events(interrupted.run_id)
        ].count(EventType.PAUSED) == 1

        runtime.resume(interrupted.run_id)

        wait_for_status(runtime.store, interrupted.run_id, RunStatus.COMPLETED)
        wait_for_status(runtime.store, queued.run_id, RunStatus.COMPLETED)
        lifecycle = [
            event.event_type for event in runtime.store.list_events(interrupted.run_id)
        ]
        assert lifecycle.count(EventType.PAUSED) == 1
        assert lifecycle.count(EventType.RESUMED) == 1
        assert provider.calls == 4
    finally:
        runtime.shutdown()
    assert b"test-only-key" not in checkpoint_path.read_bytes()


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


def test_completed_status_and_terminal_event_become_visible_atomically(tmp_path: Path):
    class BlockingCompletedEventStore(RuntimeStore):
        def __init__(self, db_path):
            self.before_completed_insert = Event()
            self.release_completed_insert = Event()
            super().__init__(db_path)

        def _connect(self):
            connection = super()._connect()
            connection.create_function(
                "block_completed_insert",
                0,
                self._block_completed_insert,
            )
            return connection

        def _block_completed_insert(self):
            self.before_completed_insert.set()
            assert self.release_completed_insert.wait(timeout=2)
            return 0

    harness = GraphHarness()
    harness.release_first.set()
    path = tmp_path / "runtime.sqlite3"
    store = BlockingCompletedEventStore(path)
    observer = RuntimeStore(path)
    with store._connection() as connection:
        connection.execute(
            """CREATE TRIGGER block_completed_event_insert
               BEFORE INSERT ON agent_events
               WHEN NEW.event_type = 'completed'
               BEGIN
                 SELECT block_completed_insert();
               END"""
        )
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=registry(),
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        graph_factory=harness.factory,
    )
    runtime.start()
    try:
        run = runtime.submit_message("course-1", "primary", "Explain")
        assert store.before_completed_insert.wait(timeout=2)

        assert observer.get_run(run.run_id).status is RunStatus.RUNNING
        assert EventType.COMPLETED not in {
            event.event_type for event in observer.list_events(run.run_id)
        }

        store.release_completed_insert.set()
        wait_for_status(observer, run.run_id, RunStatus.COMPLETED)
        assert EventType.COMPLETED in {
            event.event_type for event in observer.list_events(run.run_id)
        }
    finally:
        store.release_completed_insert.set()
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
