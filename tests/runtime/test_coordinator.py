import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from exam_predictor.evidence.scheduler import SchedulerOutcome, SchedulerStatus
from exam_predictor.evidence.service import (
    EvidenceFrontierResult,
    EvidenceInspection,
    EvidenceRunResult,
)
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
from exam_predictor.workspace.browser_intake import BrowserIntakeWriter
from exam_predictor.workspace.models import (
    SourceMode,
    WorkspaceRecord,
    WorkspaceState,
)
from exam_predictor.workspace.scanner import WorkspaceScanner
from exam_predictor.workspace.service import WorkspaceService
from exam_predictor.workspace.store import WorkspaceStore
from exam_predictor.workspace.transmission import SourceAuthorizationError


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
        self.initial_values: list[dict] = []
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
                harness.initial_values.append(dict(value))
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
        self.fail_delete = False

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
        if self.fail_delete:
            raise VaultUnavailableError("Secure credential storage is unavailable")
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


def custom_provider_request(
    api_key: str,
    *,
    base_url: str,
) -> ConnectProviderRequest:
    return ConnectProviderRequest(
        profile=ProviderProfile(
            profile_id="primary",
            provider="custom",
            base_url=base_url,
        ),
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
    assert sessions.list_profiles() == [descriptor]
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
    assert sessions.list_profiles() == [descriptor]
    saved = store.list_saved_provider_profiles()[0]
    assert saved.credential_expected is False
    assert saved.reconnect_required is True
    assert sentinel not in descriptor.model_dump_json()
    assert sentinel not in (tmp_path / "runtime.sqlite3").read_bytes().decode("latin1")


def test_unsafe_custom_url_is_rejected_before_factory_vault_or_store(
    tmp_path: Path,
):
    sentinel = "runtime-query-sentinel"
    factory_calls: list[dict] = []
    vault = FakeVault()
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=ProviderSessionRegistry(
            factory=lambda config: factory_calls.append(config) or object()
        ),
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        vault=vault,
    )
    request = ConnectProviderRequest(
        profile=ProviderProfile(
            profile_id="custom",
            provider="custom",
            base_url=f"https://models.example/v1?api_key={sentinel}",
        ),
        api_key="provider-key",
    )

    with pytest.raises(RuntimeError) as captured:
        runtime.connect_provider(request)

    surfaces = json.dumps({"detail": str(captured.value)})
    assert sentinel not in surfaces
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert factory_calls == []
    assert vault.calls == []
    assert store.list_saved_provider_profiles() == []


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
        restored_descriptor = restored_sessions.list_profiles()[0]
        assert restored_descriptor.credential_saved is True
        assert restored_descriptor.credential_warning is None
        assert store.get_run(paused.run_id).status is RunStatus.PAUSED
    finally:
        restored.shutdown()


def test_start_skips_malformed_and_mismatched_profiles_without_vault_access(
    tmp_path: Path,
):
    vault = FakeVault()
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=credential_sessions(),
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        vault=vault,
    )
    runtime.connect_provider(provider_request("valid-secret"))
    mismatched = ProviderProfile(profile_id="json-id", provider="gemini")
    with store._connection() as connection:
        connection.execute(
            """INSERT INTO saved_provider_profiles(
                   profile_id, profile_json, capabilities_json,
                   credential_expected, reconnect_required, updated_at
               ) VALUES (?, ?, ?, 1, 0, ?)""",
            (
                "sql-id",
                mismatched.model_dump_json(),
                '{"chat":true}',
                "2026-07-24T12:00:00+00:00",
            ),
        )
        connection.execute(
            """INSERT INTO saved_provider_profiles(
                   profile_id, profile_json, capabilities_json,
                   credential_expected, reconnect_required, updated_at
               ) VALUES (?, ?, ?, 1, 0, ?)""",
            (
                "malformed",
                '{"profile_id":"malformed","provider":"unknown"}',
                '{"chat":true}',
                "2026-07-24T12:00:00+00:00",
            ),
        )
    vault.calls.clear()

    restored_sessions = credential_sessions()
    restored = RuntimeCoordinator(
        store=RuntimeStore(store.db_path),
        provider_sessions=restored_sessions,
        checkpoints_path=tmp_path / "restored-checkpoints.sqlite3",
        vault=vault,
    )
    restored.start()
    try:
        assert vault.calls == [("load", "primary")]
        assert restored_sessions.has_provider("primary") is True
        assert restored_sessions.has_provider("sql-id") is False
        assert restored_sessions.has_provider("json-id") is False
    finally:
        restored.shutdown()


def test_store_failure_after_vault_save_compensates_and_keeps_memory_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sentinel = "replacement-after-store-failure"
    vault = FakeVault()
    sessions = credential_sessions()
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=sessions,
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        vault=vault,
    )
    runtime.connect_provider(provider_request("original-secret"))
    original_save = store.save_provider_profile
    save_calls = 0

    def fail_final_save(profile):
        nonlocal save_calls
        save_calls += 1
        if save_calls == 2:
            raise sqlite3.OperationalError(f"database C:/private/{sentinel}")
        original_save(profile)

    monkeypatch.setattr(store, "save_provider_profile", fail_final_save)

    descriptor = runtime.connect_provider(provider_request(sentinel))

    assert descriptor.credential_saved is False
    assert descriptor.credential_warning == RuntimeCoordinator.CREDENTIAL_WARNING
    assert sessions.get_provider("primary").identity == sentinel
    assert sessions.list_profiles() == [descriptor]
    assert vault.secrets == {}
    guarded = store.list_saved_provider_profiles()[0]
    assert guarded.credential_expected is False
    assert guarded.reconnect_required is True


def test_guard_failure_never_overwrites_existing_vault_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sentinel = "guard-failure-new-secret"
    vault = FakeVault()
    sessions = credential_sessions()
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=sessions,
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        vault=vault,
    )
    runtime.connect_provider(
        custom_provider_request(
            "original-secret",
            base_url="https://old.example/v1",
        )
    )
    vault.calls.clear()

    def fail_guard(_profile):
        raise sqlite3.OperationalError(f"database C:/private/{sentinel}")

    monkeypatch.setattr(store, "save_provider_profile", fail_guard)

    with pytest.raises(
        RuntimeError,
        match="^Provider credential state could not be prepared safely\\.$",
    ) as captured:
        runtime.connect_provider(
            custom_provider_request(
                sentinel,
                base_url="https://new.example/v1",
            )
        )

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert sentinel not in str(captured.value)
    assert sessions.has_provider("primary") is False
    assert vault.calls == []
    assert vault.secrets == {"primary": "original-secret"}
    saved = store.list_saved_provider_profiles()[0]
    assert saved.profile.base_url == "https://old.example/v1"
    assert saved.credential_expected is True


def test_three_failures_never_restore_new_secret_with_old_custom_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sentinel = "three-failure-new-secret"
    vault = FakeVault()
    initial_endpoints: list[str] = []

    def initial_factory(config):
        initial_endpoints.append(config["base_url"])
        return SimpleNamespace(
            identity=config["api_key"],
            name="fake",
            capabilities=SimpleNamespace(chat=True),
        )

    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    sessions = ProviderSessionRegistry(factory=initial_factory)
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=sessions,
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        vault=vault,
    )
    runtime.connect_provider(
        custom_provider_request(
            "old-secret",
            base_url="https://old.example/v1",
        )
    )
    original_save = store.save_provider_profile
    save_calls = 0

    def fail_final_save(profile):
        nonlocal save_calls
        save_calls += 1
        if save_calls == 2:
            raise sqlite3.OperationalError(f"finalize C:/private/{sentinel}")
        original_save(profile)

    monkeypatch.setattr(store, "save_provider_profile", fail_final_save)
    monkeypatch.setattr(
        store,
        "mark_provider_reconnect_required",
        lambda _profile_id: (_ for _ in ()).throw(
            sqlite3.OperationalError(f"marker C:/private/{sentinel}")
        ),
    )
    vault.fail_delete = True

    with pytest.raises(
        RuntimeError,
        match="^Provider credential state could not be persisted safely\\.$",
    ) as captured:
        runtime.connect_provider(
            custom_provider_request(
                sentinel,
                base_url="https://new.example/v1",
            )
        )

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert sentinel not in str(captured.value)
    assert sessions.has_provider("primary") is False
    guarded = store.list_saved_provider_profiles()[0]
    assert guarded.profile.base_url == "https://new.example/v1"
    assert guarded.credential_expected is False
    assert guarded.reconnect_required is True

    restored_endpoints: list[str] = []
    restored = RuntimeCoordinator(
        store=RuntimeStore(store.db_path),
        provider_sessions=ProviderSessionRegistry(
            factory=lambda config: restored_endpoints.append(config["base_url"])
            or SimpleNamespace(
                name="fake",
                capabilities=SimpleNamespace(chat=True),
            )
        ),
        checkpoints_path=tmp_path / "restored-checkpoints.sqlite3",
        vault=vault,
    )
    vault.calls.clear()
    restored.start()
    try:
        assert restored_endpoints == []
        assert vault.calls == []
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


def test_forget_delete_failure_disconnects_without_marking_reconnect_and_can_retry(
    tmp_path: Path,
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
    runtime.connect_provider(provider_request("retry-secret"))
    vault.fail_delete = True

    with pytest.raises(VaultUnavailableError) as captured:
        runtime.forget_provider_credential("primary")

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert sessions.has_provider("primary") is False
    unchanged = store.list_saved_provider_profiles()[0]
    assert unchanged.credential_expected is True
    assert unchanged.reconnect_required is False
    assert vault.secrets == {"primary": "retry-secret"}

    vault.fail_delete = False
    runtime.forget_provider_credential("primary")
    marked = store.list_saved_provider_profiles()[0]
    assert vault.secrets == {}
    assert marked.credential_expected is False
    assert marked.reconnect_required is True


def test_start_self_heals_memory_session_after_forget_delete_failure(tmp_path: Path):
    vault = FakeVault()
    sessions = credential_sessions()
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=sessions,
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        vault=vault,
    )
    runtime.connect_provider(provider_request("self-heal-secret"))
    vault.fail_delete = True
    with pytest.raises(VaultUnavailableError):
        runtime.forget_provider_credential("primary")

    runtime.start()
    try:
        assert sessions.get_provider("primary").identity == "self-heal-secret"
        saved = store.list_saved_provider_profiles()[0]
        assert saved.credential_expected is True
        assert saved.reconnect_required is False
    finally:
        runtime.shutdown()


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


def test_workspace_submission_derives_owned_checkpoint_thread_and_json_safe_state(
    tmp_path: Path,
):
    workspace_id = "8d6f8d1f9ed34b3f9228dcd3cb6290c4"
    repository = SimpleNamespace(
        get_workspace=lambda candidate: (
            SimpleNamespace(state=WorkspaceState.APPROVED)
            if candidate == workspace_id
            else None
        )
    )
    harness = GraphHarness()
    harness.release_first.set()
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=registry(),
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        graph_factory=harness.factory,
        workspace_repository=repository,
    )

    runtime.start()
    try:
        run = runtime.submit_message(
            "client-supplied-thread",
            "primary",
            "Review my sources",
            workspace_id=workspace_id,
        )
        wait_for_status(store, run.run_id, RunStatus.COMPLETED)

        assert run.thread_id == f"workspace:{workspace_id}"
        assert run.workspace_id == workspace_id
        assert harness.configs == [
            {"configurable": {"thread_id": f"workspace:{workspace_id}"}}
        ]
        assert harness.initial_values[0]["workspace_id"] == workspace_id
        serialized_state = json.dumps(harness.initial_values[0])
        assert workspace_id in serialized_state
        assert str(tmp_path) not in serialized_state
    finally:
        runtime.shutdown()


def test_workspace_submission_requires_repository_to_find_workspace(tmp_path: Path):
    runtime = RuntimeCoordinator(
        store=RuntimeStore(tmp_path / "runtime.sqlite3"),
        provider_sessions=registry(),
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        workspace_repository=SimpleNamespace(get_workspace=lambda _workspace_id: None),
    )

    with pytest.raises(KeyError, match="Workspace"):
        runtime.submit_message(
            "client-thread",
            "primary",
            "Review my sources",
            workspace_id="8d6f8d1f9ed34b3f9228dcd3cb6290c4",
        )


@pytest.mark.parametrize(
    "workspace_state",
    [WorkspaceState.DELETING, WorkspaceState.CLEANUP_PENDING],
)
def test_workspace_submission_cannot_recreate_run_during_deletion(
    tmp_path: Path,
    workspace_state: WorkspaceState,
):
    workspace_id = "8d6f8d1f9ed34b3f9228dcd3cb6290c4"
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=registry(),
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        workspace_repository=SimpleNamespace(
            get_workspace=lambda _workspace_id: SimpleNamespace(
                state=workspace_state
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="^Workspace cannot accept new Agent runs\\.$",
    ) as captured:
        runtime.submit_message(
            "client-thread",
            "primary",
            "Review my sources",
            workspace_id=workspace_id,
        )

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert store.list_by_status(set(RunStatus)) == []


def test_coordinator_workspace_cleanup_deletes_owned_checkpoints_then_run_metadata(
    tmp_path: Path,
):
    workspace_id = "8d6f8d1f9ed34b3f9228dcd3cb6290c4"
    workspace_thread = f"workspace:{workspace_id}"
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    associated = store.create_run(
        workspace_thread,
        "primary",
        "Review",
        RunStatus.COMPLETED,
        workspace_id=workspace_id,
    )
    legacy = store.create_run(
        "legacy-thread",
        "primary",
        "Legacy",
        RunStatus.COMPLETED,
    )
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        saver.setup()
        for thread_id in (workspace_thread, legacy.thread_id):
            saver.conn.execute(
                """INSERT INTO checkpoints(
                       thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata
                   ) VALUES (?, '', 'checkpoint-1', X'01', X'01')""",
                (thread_id,),
            )
        saver.conn.commit()
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=registry(),
        checkpoints_path=checkpoint_path,
    )

    runtime.delete_settled_workspace_runs(workspace_id)

    with pytest.raises(KeyError):
        store.get_run(associated.run_id)
    assert store.get_run(legacy.run_id) == legacy
    with sqlite3.connect(checkpoint_path) as connection:
        remaining_threads = tuple(
            row[0]
            for row in connection.execute(
                "SELECT thread_id FROM checkpoints ORDER BY thread_id"
            )
        )
    assert remaining_threads == ("legacy-thread",)


def test_checkpoint_cleanup_failure_preserves_workspace_run_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace_id = "8d6f8d1f9ed34b3f9228dcd3cb6290c4"
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    run = store.create_run(
        f"workspace:{workspace_id}",
        "primary",
        "Review",
        RunStatus.COMPLETED,
        workspace_id=workspace_id,
    )
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=registry(),
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
    )

    def fail_checkpoint_open(_path):
        raise OSError("C:/private/checkpoint-secret")

    monkeypatch.setattr(SqliteSaver, "from_conn_string", fail_checkpoint_open)

    with pytest.raises(
        RuntimeError,
        match="^Agent checkpoint cleanup failed\\.$",
    ) as captured:
        runtime.delete_settled_workspace_runs(workspace_id)

    assert store.get_run(run.run_id) == run
    assert "private" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_source_authorization_failure_pauses_safely_without_starting_queued_work(
    tmp_path: Path,
):
    error = SourceAuthorizationError(
        "approved_source_changed",
        "8d6f8d1f9ed34b3f9228dcd3cb6290c4",
        "entry-7",
    )
    harness = GraphHarness(fail_first=error)
    store, runtime = runtime_for(tmp_path, harness)
    runtime.start()
    try:
        run = runtime.submit_message("course-1", "primary", "Review")
        assert harness.first_started.wait(timeout=2)
        queued = runtime.submit_message("course-2", "primary", "Wait")
        harness.release_first.set()
        wait_for_status(store, run.run_id, RunStatus.PAUSED)

        paused_event = store.list_events(run.run_id)[-1]
        assert paused_event.event_type is EventType.PAUSED
        assert paused_event.stage == "source_changed"
        assert paused_event.message == (
            "A course source changed. Rescan and approve it before resuming."
        )
        assert paused_event.payload == {
            "code": "approved_source_changed",
            "entry_id": "entry-7",
        }
        assert store.get_run(queued.run_id).status is RunStatus.QUEUED
        assert not runtime.controls.is_stop_requested(run.run_id)
    finally:
        harness.release_first.set()
        runtime.shutdown()


def test_workspace_delete_cutover_blocks_a_concurrent_run_creation(
    tmp_path: Path,
):
    workspace_id = "8d6f8d1f9ed34b3f9228dcd3cb6290c4"
    source_root = tmp_path / "course-delete-race"
    source_root.mkdir()
    source = source_root / "notes.txt"
    source.write_bytes(b"native course bytes")
    workspace_store = WorkspaceStore(tmp_path / "workspace.sqlite3")
    now = datetime.now(UTC)
    workspace_store.create_workspace(
        WorkspaceRecord(
            workspace_id=workspace_id,
            display_name="Course",
            source_mode=SourceMode.NATIVE_FOLDER,
            canonical_root=source_root.resolve(),
            root_device=str(source_root.stat().st_dev),
            root_file_id=str(source_root.stat().st_ino),
            state=WorkspaceState.READY,
            created_at=now,
            updated_at=now,
        )
    )
    runtime = RuntimeCoordinator(
        store=RuntimeStore(tmp_path / "runtime.sqlite3"),
        provider_sessions=registry(),
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        workspace_repository=workspace_store,
    )
    guard_entered = Event()
    release_guard = Event()

    class BlockingRunGuard:
        @staticmethod
        def has_unsettled_runs(candidate: str) -> bool:
            assert candidate == workspace_id
            guard_entered.set()
            assert release_guard.wait(timeout=2)
            return runtime.has_unsettled_runs(candidate)

        @staticmethod
        def delete_settled_workspace_runs(candidate: str) -> None:
            runtime.delete_settled_workspace_runs(candidate)

    service = WorkspaceService(
        store=workspace_store,
        scanner=WorkspaceScanner(),
        picker=SimpleNamespace(choose_folder=lambda: None),
        browser_intake=BrowserIntakeWriter(tmp_path / "workspaces"),
        run_guard=BlockingRunGuard(),
    )
    deletion_errors: list[BaseException] = []
    submission_errors: list[BaseException] = []
    submit_finished = Event()

    def delete_workspace() -> None:
        try:
            service.delete_workspace(workspace_id)
        except BaseException as error:
            deletion_errors.append(error)

    def submit_run() -> None:
        try:
            runtime.submit_message(
                "client-thread",
                "primary",
                "Build my study map.",
                workspace_id=workspace_id,
            )
        except BaseException as error:
            submission_errors.append(error)
        finally:
            submit_finished.set()

    deletion = Thread(target=delete_workspace)
    submission = Thread(target=submit_run)
    try:
        deletion.start()
        assert guard_entered.wait(timeout=2)
        submission.start()
        assert not submit_finished.wait(timeout=0.1)
        release_guard.set()
        deletion.join(timeout=2)
        submission.join(timeout=2)

        assert not deletion.is_alive()
        assert not submission.is_alive()
        assert deletion_errors == []
        assert len(submission_errors) == 1
        assert isinstance(submission_errors[0], (KeyError, ValueError))
        assert runtime.store.list_by_status(set(RunStatus)) == []
        assert workspace_store.get_workspace(workspace_id) is None
        assert source.read_bytes() == b"native course bytes"
    finally:
        release_guard.set()
        deletion.join(timeout=2)
        submission.join(timeout=2)
        runtime.shutdown()
        workspace_store.close()


def test_source_change_interrupt_uses_actionable_pause_and_can_resume(
    tmp_path: Path,
):
    class SourceChangedHarness:
        @staticmethod
        def factory(_dependencies, _saver):
            class FakeGraph:
                @staticmethod
                def invoke(value, _config):
                    if isinstance(value, Command):
                        assert value.resume == {"action": "resume"}
                        return {"assistant_message": "resumed"}
                    return {
                        "__interrupt__": (
                            SimpleNamespace(
                                value={
                                    "kind": "source_changed",
                                    "run_id": value["run_id"],
                                    "code": "source_approval_revoked",
                                    "entry_id": "entry-replaced",
                                }
                            ),
                        )
                    }

            return FakeGraph()

    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=registry(),
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        graph_factory=SourceChangedHarness.factory,
    )
    runtime.start()
    try:
        run = runtime.submit_message("course-1", "primary", "Review")
        wait_for_status(store, run.run_id, RunStatus.PAUSED)
        paused = store.list_events(run.run_id)[-1]
        assert paused.stage == "source_changed"
        assert paused.payload == {
            "code": "source_approval_revoked",
            "entry_id": "entry-replaced",
        }

        runtime.resume(run.run_id)
        wait_for_status(store, run.run_id, RunStatus.COMPLETED)
    finally:
        runtime.shutdown()


def test_coordinator_injects_evidence_service_into_the_real_kernel_graph(
    tmp_path: Path,
):
    workspace_id = "8d6f8d1f9ed34b3f9228dcd3cb6290c4"
    revision_id = "revision_coordinator_evidence_01"
    service_calls: list[str] = []

    class EvidenceProvider:
        name = "fake"
        capabilities = SimpleNamespace(chat=True)
        models = SimpleNamespace(fast="fast", balanced="balanced")

        def create_chat_completion(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "tool": "build_study_map",
                                    "arguments": {},
                                    "reason": "Build course evidence.",
                                }
                            )
                        )
                    )
                ]
            )

    class EvidenceService:
        def __init__(self):
            self.controls = None
            self.stop_once = True

        def start(self):
            service_calls.append("start")
            return ()

        def inspect(self, candidate):
            service_calls.append("inspect")
            return EvidenceInspection(
                workspace_id=candidate,
                revision_id=revision_id,
                approval_id="approval-coordinator",
                approval_required=False,
                approved_source_count=1,
                approved_bytes=10,
            )

        def prepare_analysis(self, candidate, run_id=None):
            service_calls.append("prepare")
            return self.inspect(candidate)

        def analyze_frontier(self, candidate, revision, run_id):
            service_calls.append("analyze")
            if self.stop_once:
                self.stop_once = False
                self.controls.request_stop(run_id)
            return EvidenceFrontierResult(
                workspace_id=candidate,
                revision_id=revision,
                outcome=SchedulerOutcome(
                    status=SchedulerStatus.COMPLETE,
                    processed_part_ids=("part-coordinator",),
                    pending_count=0,
                ),
            )

        def publish_frontier(
            self,
            candidate,
            revision,
            outcome,
            *,
            run_id,
            response_language=None,
        ):
            service_calls.append("publish")
            assert response_language == "en"
            return EvidenceRunResult(
                workspace_id=candidate,
                revision_id=revision,
                status="complete",
                outcome=outcome,
            )

    sessions = ProviderSessionRegistry(factory=lambda _config: EvidenceProvider())
    sessions.connect(provider_request("test-only-key"))
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    evidence_service = EvidenceService()
    runtime = RuntimeCoordinator(
        store=store,
        provider_sessions=sessions,
        checkpoints_path=tmp_path / "checkpoints.sqlite3",
        workspace_repository=SimpleNamespace(
            get_workspace=lambda _candidate: SimpleNamespace(
                state=WorkspaceState.APPROVED
            )
        ),
        evidence_service=evidence_service,
    )
    evidence_service.controls = runtime.controls
    runtime.start()
    try:
        run = runtime.submit_message(
            "ignored-client-thread",
            "primary",
            "Build my study map.",
            workspace_id=workspace_id,
        )
        wait_for_status(store, run.run_id, RunStatus.PAUSED)
        assert service_calls == [
            "start",
            "inspect",
            "prepare",
            "inspect",
            "analyze",
        ]
        runtime.resume(run.run_id)
        wait_for_status(store, run.run_id, RunStatus.COMPLETED)
        events = store.list_events(run.run_id)
    finally:
        runtime.shutdown()

    assert service_calls == [
        "start",
        "inspect",
        "prepare",
        "inspect",
        "analyze",
        "publish",
    ]
    assert [event.event_type for event in events] == [
        EventType.STARTED,
        EventType.PROGRESS,
        EventType.TOOL_STARTED,
        EventType.PROGRESS,
        EventType.PAUSED,
        EventType.RESUMED,
        EventType.TOOL_COMPLETED,
        EventType.MESSAGE,
        EventType.COMPLETED,
    ]
    assert run.thread_id == f"workspace:{workspace_id}"


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
