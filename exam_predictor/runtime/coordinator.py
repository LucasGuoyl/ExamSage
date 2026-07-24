from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from exam_predictor.graphs.kernel import KernelDependencies, build_kernel_graph
from exam_predictor.tools.kernel import KernelPlanner, KernelToolRegistry

from .control import RunControlRegistry
from .credential_vault import CredentialVault, VaultUnavailableError
from .models import (
    ConnectProviderRequest,
    EventType,
    ProviderDescriptor,
    RunSnapshot,
    RunStatus,
    SavedProviderProfile,
)
from .provider_sessions import ProviderSessionRegistry
from .store import RuntimeStore


class ProviderProfileInUseError(RuntimeError):
    pass


class RuntimeCoordinator:
    CREDENTIAL_WARNING = (
        "Provider connected, but the credential could not be saved securely. "
        "Reconnect after restart."
    )

    def __init__(
        self,
        *,
        store: RuntimeStore,
        provider_sessions: ProviderSessionRegistry,
        checkpoints_path: str | Path,
        graph_factory: Callable[..., Any] = build_kernel_graph,
        vault: CredentialVault | None = None,
    ):
        self.store = store
        self.provider_sessions = provider_sessions
        self.checkpoints_path = Path(checkpoints_path)
        self.graph_factory = graph_factory
        self.vault = vault
        self.controls = RunControlRegistry()
        self._commands: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._shutdown_requested = False

    def _emit(
        self,
        run_id: str,
        event_type: str,
        stage: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.store.append_event(run_id, EventType(event_type), stage, message, payload)

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            if self._shutdown_requested:
                raise RuntimeError("A shut down RuntimeCoordinator cannot be restarted.")
            self.store.recover_unfinished()
            self._restore_saved_profiles()
            self.checkpoints_path.parent.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(
                target=self._loop,
                name="examsage-agent",
                daemon=True,
            )
            self._thread.start()
            if self.store.active_run() is None:
                self._start_next()

    def submit_message(
        self,
        thread_id: str,
        provider_profile_id: str,
        message: str,
    ) -> RunSnapshot:
        with self._lock:
            self._ensure_accepting_commands()
            self.provider_sessions.get_provider(provider_profile_id)
            active = self.store.active_run()
            queued = self.store.next_queued_run()
            status = RunStatus.QUEUED if active or queued else RunStatus.RUNNING
            run = self.store.create_run(
                thread_id,
                provider_profile_id,
                message,
                status,
            )
            event_type = EventType.QUEUED if status is RunStatus.QUEUED else EventType.STARTED
            text = (
                "Message queued behind the active run."
                if status is RunStatus.QUEUED
                else "Agent run started."
            )
            self.store.append_event(run.run_id, event_type, "queue", text)
            if status is RunStatus.RUNNING:
                self._commands.put(("start", run.run_id))
            elif active is None:
                self._start_next()
            return run

    def connect_provider(
        self,
        request: ConnectProviderRequest,
    ) -> ProviderDescriptor:
        profile_id = request.profile.profile_id
        with self._lock:
            if self.provider_sessions.has_provider(profile_id):
                self._ensure_profile_replaceable(profile_id)
            descriptor = self.provider_sessions.connect(request)
            credential_saved = False
            if self.vault is not None:
                save_failed = False
                try:
                    self.vault.save(profile_id, request.api_key.get_secret_value())
                except Exception:
                    save_failed = True
                credential_saved = not save_failed
            warning = None if credential_saved else self.CREDENTIAL_WARNING
            descriptor = descriptor.model_copy(
                update={
                    "credential_saved": credential_saved,
                    "credential_warning": warning,
                }
            )
            saved_profile = SavedProviderProfile(
                profile=descriptor.profile,
                capabilities=descriptor.capabilities,
                credential_expected=credential_saved,
                reconnect_required=not credential_saved,
                updated_at=datetime.now(timezone.utc),
            )
            store_failed = False
            try:
                self.store.save_provider_profile(saved_profile)
            except Exception:
                store_failed = True
            if not store_failed:
                self.provider_sessions.update_descriptor(descriptor)
                return descriptor

            compensation_failed = False
            if credential_saved and self.vault is not None:
                try:
                    self.vault.delete(profile_id)
                except Exception:
                    compensation_failed = True
            if compensation_failed:
                try:
                    self.store.mark_provider_reconnect_required(profile_id)
                except Exception:
                    pass
                self.provider_sessions.disconnect(profile_id)
                raise RuntimeError(
                    "Provider credential state could not be persisted safely."
                ) from None

            descriptor = descriptor.model_copy(
                update={
                    "credential_saved": False,
                    "credential_warning": self.CREDENTIAL_WARNING,
                }
            )
            self.provider_sessions.update_descriptor(descriptor)
            return descriptor

    def forget_provider_credential(self, profile_id: str) -> None:
        with self._lock:
            self._ensure_profile_replaceable(profile_id)
            self.provider_sessions.disconnect(profile_id)
            if self.vault is not None:
                delete_failed = False
                try:
                    self.vault.delete(profile_id)
                except Exception:
                    delete_failed = True
                if delete_failed:
                    raise VaultUnavailableError(
                        "Secure credential storage is unavailable"
                    )
            self.store.mark_provider_reconnect_required(profile_id)

    def _ensure_profile_replaceable(self, profile_id: str) -> None:
        active = self.store.list_by_status(
            {RunStatus.RUNNING, RunStatus.STOPPING}
        )
        if any(run.provider_profile_id == profile_id for run in active):
            raise ProviderProfileInUseError(
                "Provider profile is currently in use by an active run."
            )

    def _restore_saved_profiles(self) -> None:
        for saved in self.store.list_saved_provider_profiles():
            profile_id = saved.profile.profile_id
            if (
                saved.reconnect_required
                or not saved.credential_expected
                or self.provider_sessions.has_provider(profile_id)
            ):
                continue
            api_key: str | None = None
            load_failed = self.vault is None
            if self.vault is not None:
                try:
                    api_key = self.vault.load(profile_id)
                except Exception:
                    load_failed = True
            if load_failed or api_key is None:
                self.store.mark_provider_reconnect_required(profile_id)
                continue
            restore_failed = False
            try:
                descriptor = self.provider_sessions.restore(saved.profile, api_key)
                self.provider_sessions.update_descriptor(
                    descriptor.model_copy(
                        update={
                            "credential_saved": True,
                            "credential_warning": None,
                        }
                    )
                )
            except Exception:
                restore_failed = True
            if restore_failed:
                self.store.mark_provider_reconnect_required(profile_id)

    def stop(self, run_id: str) -> RunSnapshot:
        with self._lock:
            run = self.store.get_run(run_id)
            if run.status is not RunStatus.RUNNING:
                raise ValueError("Only a running Agent run can be stopped.")
            self.controls.request_stop(run_id)
            self.store.append_event(
                run_id,
                EventType.STOP_REQUESTED,
                "stopping",
                "Stop requested.",
            )
            return self.store.set_status(run_id, RunStatus.STOPPING)

    def resume(self, run_id: str) -> RunSnapshot:
        with self._lock:
            run = self.store.get_run(run_id)
            if run.status is not RunStatus.PAUSED:
                raise ValueError("Only a paused Agent run can be resumed.")
            self.provider_sessions.get_provider(run.provider_profile_id)
            self._ensure_accepting_commands()
            self.store.set_status(run_id, RunStatus.RUNNING)
            self._commands.put(("resume", run_id))
            return self.store.get_run(run_id)

    def pause_all(self) -> None:
        with self._lock:
            active_statuses = {RunStatus.RUNNING, RunStatus.STOPPING}
            for run in self.store.list_by_status(active_statuses):
                self.controls.request_stop(run.run_id)

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._lock:
            self.pause_all()
            thread = self._thread
            if not self._shutdown_requested:
                self._shutdown_requested = True
                if thread is not None:
                    self._commands.put(None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise TimeoutError(
                    "RuntimeCoordinator worker did not stop before the shutdown timeout."
                )

    def _ensure_accepting_commands(self) -> None:
        if self._shutdown_requested:
            raise RuntimeError("The RuntimeCoordinator is shutting down.")

    def _loop(self) -> None:
        with SqliteSaver.from_conn_string(str(self.checkpoints_path)) as saver:
            dependencies = KernelDependencies(
                provider_sessions=self.provider_sessions,
                planner=KernelPlanner(),
                tools=KernelToolRegistry(),
                controls=self.controls,
                emit=self._emit,
            )
            graph = self.graph_factory(dependencies, saver)
            while True:
                command = self._commands.get()
                if command is None:
                    return
                action, run_id = command
                self._execute(graph, action, run_id)

    def _execute(self, graph: Any, action: str, run_id: str) -> None:
        run = self.store.get_run(run_id)
        existing_events = self.store.list_events(run_id)
        event_cursor = existing_events[-1].sequence if existing_events else 0
        config = {"configurable": {"thread_id": run.thread_id}}
        try:
            if action == "resume":
                result = self._invoke_resume(graph, config, run)
            else:
                result = graph.invoke(self._initial_state(run), config)
        except Exception as exc:
            message = f"{type(exc).__name__}: Agent run failed."
            with self._lock:
                self.store.set_status_and_append_event(
                    run_id,
                    RunStatus.FAILED,
                    EventType.FAILED,
                    "failed",
                    message,
                    error=message,
                )
                self.controls.discard(run_id)
                self._start_next()
            return

        with self._lock:
            if result.get("__interrupt__"):
                invocation_events = self.store.list_events(run_id, after=event_cursor)
                if any(
                    event.event_type is EventType.PAUSED for event in invocation_events
                ):
                    self.store.set_status(run_id, RunStatus.PAUSED)
                else:
                    self.store.set_status_and_append_event(
                        run_id,
                        RunStatus.PAUSED,
                        EventType.PAUSED,
                        "paused",
                        "The run is paused at a safe boundary.",
                    )
                return
            self.store.set_status_and_append_event(
                run_id,
                RunStatus.COMPLETED,
                EventType.COMPLETED,
                "complete",
                "Agent run completed.",
            )
            self.controls.discard(run_id)
            self._start_next()

    @staticmethod
    def _initial_state(run: RunSnapshot) -> dict[str, Any]:
        return {
            "run_id": run.run_id,
            "provider_profile_id": run.provider_profile_id,
            "user_message": run.message,
            "messages": [{"role": "user", "content": run.message}],
        }

    def _invoke_resume(
        self,
        graph: Any,
        config: dict[str, dict[str, str]],
        run: RunSnapshot,
    ) -> dict[str, Any]:
        get_state = getattr(graph, "get_state", None)
        if get_state is None:
            return graph.invoke(Command(resume={"action": "resume"}), config)

        checkpoint = get_state(config)
        checkpoint_values = getattr(checkpoint, "values", {})
        same_run = checkpoint_values.get("run_id") == run.run_id
        interrupted = same_run and any(
            getattr(task, "interrupts", ())
            for task in getattr(checkpoint, "tasks", ())
        )
        if interrupted:
            return graph.invoke(Command(resume={"action": "resume"}), config)

        pending_nodes = getattr(checkpoint, "next", ())
        pending_pause = (
            same_run
            and checkpoint_values.get("pause_pending") is True
            and len(pending_nodes) == 1
            and str(pending_nodes[0]).startswith("pause_before_")
        )
        if pending_pause:
            graph.update_state(config, {"pause_pending": False})
            self.controls.clear_stop(run.run_id)

        self.store.append_event(
            run.run_id,
            EventType.RESUMED,
            "planning",
            "The run resumed from durable state.",
        )
        if not same_run:
            return graph.invoke(self._initial_state(run), config)
        if pending_nodes:
            return graph.invoke(None, config)
        return dict(checkpoint_values)

    def _start_next(self) -> None:
        if self._shutdown_requested:
            return
        if self.store.active_run() is not None:
            return
        queued = self.store.next_queued_run()
        if queued is None:
            return
        self.store.set_status(queued.run_id, RunStatus.RUNNING)
        self.store.append_event(
            queued.run_id,
            EventType.STARTED,
            "queue",
            "Queued run started.",
        )
        self._commands.put(("start", queued.run_id))
