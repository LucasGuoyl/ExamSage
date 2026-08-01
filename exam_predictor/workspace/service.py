from __future__ import annotations

import os
import queue
import re
import threading
from dataclasses import dataclass
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Protocol
from uuid import uuid4

from exam_predictor.workspace.browser_intake import (
    BrowserIntakeError,
    BrowserIntakeWriter,
    BrowserUpload,
)
from exam_predictor.workspace.filesystem import SecureOpenError
from exam_predictor.workspace.models import (
    ApprovalRecord,
    ManifestEntry,
    ManifestRevision,
    SourceMode,
    SourceState,
    WorkspaceJob,
    WorkspaceJobStatus,
    WorkspaceRecord,
    WorkspaceState,
)
from exam_predictor.workspace.scanner import (
    RevalidationResult,
    ScanExecution,
    WorkspaceScanner,
)
from exam_predictor.workspace.store import (
    ActiveWorkspaceOperationError,
    InvalidApprovalError,
    ManifestNotFoundError,
    StaleManifestError,
    WorkspaceJobNotFoundError,
    WorkspaceNotFoundError,
    WorkspaceStore,
)

_BROWSER_TEMP_NAME = re.compile(r"\.browser-intake-[0-9a-f]{32}\.tmp\Z")


class WorkspaceRunGuard(Protocol):
    def has_unsettled_runs(self, workspace_id: str) -> bool:
        """Return true for queued, running, stopping, or paused linked work."""

    def delete_settled_workspace_runs(self, workspace_id: str) -> None:
        """Delete only settled run/checkpoint metadata owned by ExamSage."""


class FolderPicker(Protocol):
    def choose_folder(self) -> Path | None: ...


class WorkspaceOperationError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class OwnedTreeClaim:
    relative_path: str
    device_id: str
    file_id: str


class WorkspaceService:
    def __init__(
        self,
        *,
        store: WorkspaceStore,
        scanner: WorkspaceScanner,
        picker: FolderPicker,
        browser_intake: BrowserIntakeWriter,
        run_guard: WorkspaceRunGuard,
        remove_owned_tree: Callable[[OwnedTreeClaim], None] | None = None,
        evidence_cleanup: Callable[[str], object] | None = None,
        close_store_on_shutdown: bool = False,
    ) -> None:
        self._store = store
        self._scanner = scanner
        self._picker = picker
        self._browser_intake = browser_intake
        self._run_guard = run_guard
        self._remove_owned_tree = (
            remove_owned_tree or self._secure_cleanup_unavailable
        )
        self._evidence_cleanup = evidence_cleanup or (lambda _workspace_id: None)
        self._close_store_on_shutdown = close_store_on_shutdown
        self._jobs: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._active_workspaces: set[str] = set()
        self._active_jobs: dict[str, tuple[str, str]] = {}
        self._enqueued_job_ids: set[str] = set()
        self._creation_jobs: dict[str, str] = {}

    def start(self) -> None:
        """Recover jobs and cleanup records, then start one serialized job thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._jobs = queue.Queue()
            self._active_workspaces.clear()
            self._active_jobs.clear()
            self._enqueued_job_ids.clear()
            self._discover_browser_orphans()
            self._store.interrupt_claimed_creations()
            self._recover_cleanup()
            self._recover_deleting_workspaces()
            for job in self._store.recover_unfinished_jobs():
                self._queue_existing(job)
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name="workspace-service",
                daemon=True,
            )
            self._thread.start()

    def shutdown(self, timeout_seconds: float = 5.0) -> None:
        """Request shutdown, join the job thread, and close owned resources."""
        with self._lock:
            thread = self._thread
            if thread is not None and thread.is_alive():
                self._stop.set()
                self._jobs.put(None)
        if thread is not None:
            thread.join(timeout=max(timeout_seconds, 0.0))
        if thread is not None and thread.is_alive():
            return
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None
        if self._close_store_on_shutdown:
            self._store.close()

    def select_folder(self, idempotency_key: str) -> WorkspaceJob | None:
        """Open the injected picker; return None on cancel or enqueue a scan."""
        workspace_id = uuid4().hex
        try:
            request, claimed = self._store.claim_creation(
                idempotency_key,
                SourceMode.NATIVE_FOLDER,
                workspace_id,
            )
        except ActiveWorkspaceOperationError:
            raise WorkspaceOperationError("idempotency_conflict") from None
        if not claimed:
            return self._replay_creation(request.state, idempotency_key)
        try:
            selected = self._picker.choose_folder()
        except Exception as error:
            code = getattr(error, "code", "folder_picker_failed")
            self._store.fail_creation(idempotency_key, str(code))
            raise WorkspaceOperationError(str(code)) from None
        if selected is None:
            self._store.finish_cancelled_creation(idempotency_key)
            return None
        try:
            validation = self._validate_root(Path(selected))
        except WorkspaceOperationError as error:
            self._store.fail_creation(idempotency_key, error.code)
            raise
        now = datetime.now(UTC)
        workspace = WorkspaceRecord(
            workspace_id=workspace_id,
            display_name=validation.canonical_root.name or "Course workspace",
            source_mode=SourceMode.NATIVE_FOLDER,
            canonical_root=validation.canonical_root,
            root_device=validation.root_device,
            root_file_id=validation.root_file_id,
            state=WorkspaceState.READY,
            created_at=now,
            updated_at=now,
            last_access_verified_at=now,
        )
        job = self._new_job(workspace_id, idempotency_key, "initial_scan")
        try:
            _, stored, _ = self._store.create_workspace_with_initial_job(
                workspace,
                job,
                idempotency_key,
            )
        except Exception:
            self._store.fail_creation(idempotency_key, "creation_failed")
            raise WorkspaceOperationError("creation_failed") from None
        self._queue_existing(stored)
        return stored

    def create_browser_snapshot(
        self,
        display_name: str,
        files: Sequence[BrowserUpload],
        idempotency_key: str,
    ) -> WorkspaceJob:
        """Create an app-owned snapshot and enqueue its initial scan."""
        workspace_id = uuid4().hex
        try:
            request, claimed = self._store.claim_creation(
                idempotency_key,
                SourceMode.BROWSER_SNAPSHOT,
                workspace_id,
            )
        except ActiveWorkspaceOperationError:
            raise WorkspaceOperationError("idempotency_conflict") from None
        if not claimed:
            replay = self._replay_creation(request.state, idempotency_key)
            if replay is None:
                raise WorkspaceOperationError("creation_cancelled")
            return replay
        root: Path | None = None
        owned_validation: RevalidationResult | None = None
        try:
            root = self._browser_intake.create_snapshot(workspace_id, files)
            owned_validation = self._validate_root(root.parent)
            validation = self._validate_root(root)
        except (BrowserIntakeError, WorkspaceOperationError) as error:
            if root is not None:
                self._queue_orphan_cleanup(
                    workspace_id,
                    root.parent,
                    (
                        (
                            owned_validation.root_device,
                            owned_validation.root_file_id,
                        )
                        if owned_validation is not None
                        else None
                    ),
                )
            code = getattr(error, "code", "browser_intake_failed")
            self._store.fail_creation(idempotency_key, str(code))
            raise WorkspaceOperationError(str(code)) from None
        now = datetime.now(UTC)
        workspace = WorkspaceRecord(
            workspace_id=workspace_id,
            display_name=display_name.strip() or "Browser course",
            source_mode=SourceMode.BROWSER_SNAPSHOT,
            canonical_root=validation.canonical_root,
            root_device=validation.root_device,
            root_file_id=validation.root_file_id,
            state=WorkspaceState.READY,
            created_at=now,
            updated_at=now,
            last_access_verified_at=now,
        )
        job = self._new_job(workspace_id, idempotency_key, "initial_scan")
        try:
            _, stored, _ = self._store.create_workspace_with_initial_job(
                workspace,
                job,
                idempotency_key,
                owned_root_identity=(
                    owned_validation.root_device,
                    owned_validation.root_file_id,
                ),
            )
        except Exception:
            self._queue_orphan_cleanup(
                workspace_id,
                root.parent,
                (
                    owned_validation.root_device,
                    owned_validation.root_file_id,
                ),
            )
            self._store.fail_creation(idempotency_key, "creation_failed")
            raise WorkspaceOperationError("creation_failed") from None
        self._queue_existing(stored)
        return stored

    def rescan(self, workspace_id: str, idempotency_key: str) -> WorkspaceJob:
        """Return an identical active job or enqueue one new scan."""
        return self._enqueue(workspace_id, idempotency_key)

    def set_inclusion(
        self,
        workspace_id: str,
        revision_id: str,
        entry_id: str,
        included: bool,
        subtree: bool = False,
    ) -> ManifestRevision:
        """Create a draft after expanding an optional subtree."""
        with self._workspace_mutation(workspace_id):
            try:
                revision = self._store.get_manifest(workspace_id, revision_id)
            except WorkspaceNotFoundError:
                raise WorkspaceOperationError("workspace_not_found") from None
            except ManifestNotFoundError:
                raise WorkspaceOperationError("manifest_not_found") from None
            target = next(
                (entry for entry in revision.entries if entry.entry_id == entry_id),
                None,
            )
            if target is None:
                try:
                    return self._store.set_inclusion(
                        workspace_id, revision_id, (entry_id,), included
                    )
                except ManifestNotFoundError:
                    raise WorkspaceOperationError("manifest_not_found") from None
            entry_ids = (entry_id,)
            if subtree:
                source_target = target
                if target.archive_parent_entry_id is not None:
                    source_target = next(
                        (
                            entry
                            for entry in revision.entries
                            if entry.entry_id == target.archive_parent_entry_id
                        ),
                        target,
                    )
                relative_prefix = (
                    source_target.relative_path.rstrip("/")
                    if source_target.item_kind == "folder"
                    else source_target.relative_path.rpartition("/")[0]
                )

                def belongs_to_subtree(entry: ManifestEntry) -> bool:
                    if entry.archive_parent_entry_id is not None:
                        return False
                    if not relative_prefix:
                        return True
                    return (
                        entry.relative_path == relative_prefix
                        or entry.relative_path.startswith(f"{relative_prefix}/")
                    )

                def can_set_inclusion(entry: ManifestEntry) -> bool:
                    if entry.state in {
                        SourceState.PENDING_APPROVAL,
                        SourceState.APPROVED,
                        SourceState.CHANGED,
                    }:
                        return not included or entry.sha256 is not None
                    return (
                        entry.state is SourceState.EXCLUDED
                        and entry.inclusion_reason == "user_excluded"
                        and (not included or entry.sha256 is not None)
                    )

                entry_ids = tuple(
                    entry.entry_id
                    for entry in revision.entries
                    if belongs_to_subtree(entry) and can_set_inclusion(entry)
                )
            try:
                return self._store.set_inclusion(
                    workspace_id,
                    revision_id,
                    entry_ids,
                    included,
                )
            except ManifestNotFoundError:
                raise WorkspaceOperationError("manifest_not_found") from None

    def approve(self, workspace_id: str, revision_id: str) -> ApprovalRecord:
        """Revalidate included files and atomically approve the exact draft."""
        with self._workspace_mutation(workspace_id):
            try:
                revision = self._store.require_current_draft(
                    workspace_id, revision_id
                )
            except WorkspaceNotFoundError:
                raise WorkspaceOperationError("workspace_not_found") from None
            except ManifestNotFoundError:
                raise WorkspaceOperationError("manifest_not_found") from None
            workspace = self._require_workspace(workspace_id)
            selected = tuple(
                entry
                for entry in revision.entries
                if entry.included and entry.archive_parent_entry_id is None
            )
            try:
                validation = self._scanner.revalidate_entries(
                    workspace.canonical_root,
                    selected,
                )
                self._verify_stored_root(workspace, validation)
            except (SecureOpenError, WorkspaceOperationError) as error:
                self._store.mark_revision_attention_if_current(
                    workspace_id,
                    revision_id,
                    tuple(entry.entry_id for entry in selected),
                    getattr(error, "code", "source_root_invalid"),
                )
                raise StaleManifestError(
                    "The selected source changed and needs review."
                ) from None
            current = {entry.entry_id: entry for entry in validation.entries}
            changed = tuple(
                entry.entry_id
                for entry in selected
                if not self._entry_matches(entry, current.get(entry.entry_id))
            )
            if changed:
                self._store.mark_revision_attention_if_current(
                    workspace_id,
                    revision_id,
                    changed,
                    "source_changed_before_approval",
                )
                raise StaleManifestError(
                    "The selected source changed and needs review."
                )
            try:
                return self._store.approve(
                    workspace_id,
                    revision_id,
                    revision.policy_version,
                    verified_at=datetime.now(UTC),
                )
            except InvalidApprovalError:
                raise WorkspaceOperationError("invalid_approval") from None

    def delete_workspace(self, workspace_id: str) -> None:
        """Delete ExamSage-owned state only after the run guard settles."""
        with self._workspace_mutation(workspace_id):
            unsettled = False

            def has_unsettled_runs() -> bool:
                nonlocal unsettled
                unsettled = self._run_guard.has_unsettled_runs(workspace_id)
                return unsettled

            try:
                workspace = self._store.mark_deleting_if_settled(
                    workspace_id,
                    has_unsettled_runs,
                )
            except ActiveWorkspaceOperationError:
                if unsettled:
                    raise WorkspaceOperationError(
                        "workspace_has_unsettled_runs"
                    ) from None
                raise WorkspaceOperationError("workspace_operation_active") from None
            self._run_guard.delete_settled_workspace_runs(workspace_id)
            try:
                cleanup_result = self._evidence_cleanup(workspace_id)
            except Exception as error:
                code = getattr(error, "code", None)
                safe_code = (
                    code
                    if code in {"evidence_delete_pending", "evidence_runs_active"}
                    else "evidence_delete_pending"
                )
                raise WorkspaceOperationError(safe_code) from None
            cleanup_state = getattr(cleanup_result, "value", cleanup_result)
            if cleanup_state not in {None, "deleted"}:
                raise WorkspaceOperationError("evidence_delete_pending")
            if workspace.source_mode is SourceMode.NATIVE_FOLDER:
                self._store.delete_workspace_rows(workspace_id)
                return
            owned_path = workspace.canonical_root.parent
            owned_identity = self._store.get_owned_root_identity(workspace_id)
            self._assert_owned_snapshot_path(workspace, owned_path)
            record = self._store.queue_cleanup(
                workspace_id,
                owned_path,
                "cleanup_pending",
                deletion_root_identity=owned_identity,
            )
            if owned_identity is None:
                self._store.fail_cleanup(
                    record.cleanup_id,
                    "cleanup_identity_missing",
                )
                raise WorkspaceOperationError("cleanup_pending")
            claim = OwnedTreeClaim(
                relative_path=record.owned_relative_path,
                device_id=owned_identity[0],
                file_id=owned_identity[1],
            )
            try:
                self._remove_owned_tree(claim)
            except Exception:
                self._store.fail_cleanup(record.cleanup_id, "cleanup_failed")
                raise WorkspaceOperationError("cleanup_pending") from None
            self._store.complete_cleanup(record.cleanup_id)

    def delete_all_workspaces(self) -> tuple[str, ...]:
        """Safely delete settled workspaces and return sorted conflicts."""
        conflicts = sorted(
            workspace.workspace_id
            for workspace in self._store.list_workspaces()
            if self._run_guard.has_unsettled_runs(workspace.workspace_id)
        )
        conflict_set = set(conflicts)
        for workspace in sorted(
            self._store.list_workspaces(), key=lambda item: item.workspace_id
        ):
            if workspace.workspace_id not in conflict_set:
                self.delete_workspace(workspace.workspace_id)
        return tuple(conflicts)

    def _recover_deleting_workspaces(self) -> None:
        for workspace in self._store.list_workspaces():
            if workspace.state is not WorkspaceState.DELETING:
                continue
            try:
                self.delete_workspace(workspace.workspace_id)
            except WorkspaceOperationError:
                continue

    def _enqueue(self, workspace_id: str, idempotency_key: str) -> WorkspaceJob:
        with self._lock:
            active_job = self._active_jobs.get(workspace_id)
            if active_job is not None:
                active_key, active_job_id = active_job
                if active_key == idempotency_key:
                    return self._store.get_job(active_job_id)
                raise WorkspaceOperationError("workspace_operation_active")
            candidate = self._new_job(workspace_id, idempotency_key, "scan")
            try:
                stored = self._store.create_job(candidate, idempotency_key)
            except WorkspaceNotFoundError:
                raise WorkspaceOperationError("workspace_not_found") from None
            except ActiveWorkspaceOperationError:
                raise WorkspaceOperationError("workspace_operation_active") from None
            if stored.job_id != candidate.job_id:
                return stored
            self._active_workspaces.add(workspace_id)
            self._active_jobs[workspace_id] = (
                idempotency_key,
                stored.job_id,
            )
            self._queue_existing(stored)
            return stored

    @staticmethod
    def _new_job(
        workspace_id: str,
        idempotency_key: str,
        job_kind: str,
    ) -> WorkspaceJob:
        return WorkspaceJob(
            job_id=uuid4().hex,
            workspace_id=workspace_id,
            job_kind=job_kind,
            status=WorkspaceJobStatus.QUEUED,
            idempotency_key=idempotency_key,
            created_at=datetime.now(UTC),
        )

    def _queue_existing(self, job: WorkspaceJob) -> None:
        with self._lock:
            if job.job_id in self._enqueued_job_ids:
                return
            self._active_workspaces.add(job.workspace_id)
            self._active_jobs[job.workspace_id] = (
                job.idempotency_key,
                job.job_id,
            )
            self._enqueued_job_ids.add(job.job_id)
            self._jobs.put(job.job_id)

    def _loop(self) -> None:
        while True:
            job_id = self._jobs.get()
            if job_id is None:
                return
            if self._stop.is_set():
                return
            try:
                job = self._store.get_job(job_id)
                self._store.start_job(job_id)
                workspace = self._require_workspace(job.workspace_id)
                execution = self._scanner.scan_with_identity(
                    job.workspace_id,
                    workspace.canonical_root,
                    previous_entries=self._store.get_manifest_entries(
                        job.workspace_id
                    ),
                    emit=lambda progress: self._store.append_progress(
                        job_id, progress
                    ),
                )
                self._verify_stored_root(workspace, execution)
                result = execution.result
                self._store.record_access_verified(
                    job.workspace_id, result.completed_at
                )
                self._store.commit_scan(job.workspace_id, result, job_id)
            except WorkspaceOperationError as error:
                self._fail_if_active(job_id, error.code)
            except SecureOpenError as error:
                self._fail_if_active(job_id, error.code)
            except Exception:
                self._fail_if_active(job_id, "transient_local_io")
            finally:
                with self._lock:
                    self._enqueued_job_ids.discard(job_id)
                    try:
                        job = self._store.get_job(job_id)
                    except WorkspaceJobNotFoundError:
                        pass
                    else:
                        self._active_workspaces.discard(job.workspace_id)
                        if self._active_jobs.get(job.workspace_id) == (
                            job.idempotency_key,
                            job.job_id,
                        ):
                            self._active_jobs.pop(job.workspace_id, None)
            if self._stop.is_set():
                return

    def _fail_if_active(self, job_id: str, code: str) -> None:
        try:
            job = self._store.get_job(job_id)
            if job.status in {
                WorkspaceJobStatus.QUEUED,
                WorkspaceJobStatus.RUNNING,
            }:
                self._store.fail_job(job_id, code)
        except (ActiveWorkspaceOperationError, WorkspaceJobNotFoundError):
            return

    def _recover_cleanup(self) -> None:
        data_root = self._store.database_path.parent.absolute()
        for record in self._store.list_cleanup():
            workspace = self._store.get_workspace(record.workspace_id)
            if workspace is not None and self._run_guard.has_unsettled_runs(
                workspace.workspace_id
            ):
                self._store.fail_cleanup(
                    record.cleanup_id,
                    "workspace_has_unsettled_runs",
                )
                continue
            try:
                owned_path = self._cleanup_path(
                    data_root,
                    record.workspace_id,
                    record.owned_relative_path,
                )
                if workspace is not None:
                    if workspace.source_mode is not SourceMode.BROWSER_SNAPSHOT:
                        raise WorkspaceOperationError("cleanup_path_invalid")
                    self._assert_owned_snapshot_path(workspace, owned_path)
                if (
                    record.deletion_root_device is None
                    or record.deletion_root_file_id is None
                ):
                    self._store.fail_cleanup(
                        record.cleanup_id,
                        "cleanup_identity_missing",
                    )
                    continue
                try:
                    owned_path.lstat()
                except FileNotFoundError:
                    self._store.complete_cleanup(record.cleanup_id)
                    continue
                validation = self._scanner.revalidate_entries(owned_path)
                if (
                    validation.root_device != record.deletion_root_device
                    or validation.root_file_id != record.deletion_root_file_id
                ):
                    raise WorkspaceOperationError("cleanup_identity_changed")
                self._remove_owned_tree(
                    OwnedTreeClaim(
                        relative_path=record.owned_relative_path,
                        device_id=record.deletion_root_device,
                        file_id=record.deletion_root_file_id,
                    )
                )
                self._store.complete_cleanup(record.cleanup_id)
            except Exception:
                self._store.fail_cleanup(record.cleanup_id, "cleanup_retry_failed")

    def _queue_orphan_cleanup(
        self,
        workspace_id: str,
        owned_path: Path,
        deletion_root_identity: tuple[str, str] | None,
    ) -> None:
        data_root = self._store.database_path.parent.absolute()
        try:
            relative = owned_path.absolute().relative_to(data_root).as_posix()
            self._cleanup_path(data_root, workspace_id, relative)
            self._store.queue_cleanup(
                workspace_id,
                owned_path,
                "cleanup_pending",
                deletion_root_identity=deletion_root_identity,
            )
        except (ValueError, WorkspaceOperationError):
            return

    def _assert_owned_snapshot_path(
        self, workspace: WorkspaceRecord, owned_path: Path
    ) -> None:
        data_root = self._store.database_path.parent.absolute()
        expected = data_root / "workspaces" / workspace.workspace_id
        candidate = owned_path.absolute()
        if (
            candidate != expected
            or workspace.canonical_root.absolute()
            != expected / "browser-intake"
        ):
            raise WorkspaceOperationError("cleanup_path_invalid")

    @staticmethod
    def _cleanup_path(
        data_root: Path,
        workspace_id: str,
        owned_relative_path: str,
    ) -> Path:
        relative = PurePosixPath(owned_relative_path)
        parts = relative.parts
        exact_paths = {
            ("workspaces", workspace_id),
            ("workspaces", workspace_id, "browser-intake"),
        }
        is_task4_orphan = (
            len(parts) == 3
            and parts[:2] == ("workspaces", workspace_id)
            and parts[2].startswith(".examsage-browser-orphan-")
            and len(parts[2]) <= 96
        )
        is_browser_temp = (
            len(parts) == 3
            and parts[:2] == ("workspaces", workspace_id)
            and _BROWSER_TEMP_NAME.fullmatch(parts[2]) is not None
        )
        if parts not in exact_paths and not is_task4_orphan and not is_browser_temp:
            raise WorkspaceOperationError("cleanup_path_invalid")
        candidate = data_root.joinpath(*parts).absolute()
        if candidate != data_root.joinpath(*parts):
            raise WorkspaceOperationError("cleanup_path_invalid")
        return candidate

    def _discover_browser_orphans(self) -> None:
        workspaces_root = self._store.database_path.parent / "workspaces"
        claimed_workspaces = {
            request.workspace_id
            for request in self._store.list_recoverable_browser_creations()
        }
        try:
            workspaces = tuple(workspaces_root.iterdir())
        except OSError:
            return
        for workspace_root in workspaces:
            if not workspace_root.is_dir() or workspace_root.is_symlink():
                continue
            try:
                children = tuple(workspace_root.iterdir())
            except OSError:
                continue
            for child in children:
                is_task4_orphan = child.name.startswith(
                    ".examsage-browser-orphan-"
                )
                is_claimed_remnant = workspace_root.name in claimed_workspaces and (
                    child.name == "browser-intake"
                    or _BROWSER_TEMP_NAME.fullmatch(child.name) is not None
                )
                if not is_task4_orphan and not is_claimed_remnant:
                    continue
                identity = (
                    self._try_cleanup_identity(child)
                    if workspace_root.name in claimed_workspaces
                    else None
                )
                self._queue_orphan_cleanup(
                    workspace_root.name,
                    child,
                    identity,
                )

    def _try_cleanup_identity(self, owned_path: Path) -> tuple[str, str] | None:
        try:
            validation = self._scanner.revalidate_entries(owned_path)
        except (OSError, SecureOpenError):
            return None
        return validation.root_device, validation.root_file_id

    @staticmethod
    def _secure_cleanup_unavailable(claim: OwnedTreeClaim) -> None:
        del claim
        raise WorkspaceOperationError("secure_cleanup_unavailable")

    def _validate_root(self, root: Path) -> RevalidationResult:
        try:
            return self._scanner.revalidate_entries(root)
        except SecureOpenError as error:
            raise WorkspaceOperationError(error.code) from None

    @staticmethod
    def _verify_stored_root(
        workspace: WorkspaceRecord,
        validation: RevalidationResult | ScanExecution,
    ) -> None:
        same_path = os.path.normcase(str(workspace.canonical_root)) == os.path.normcase(
            str(validation.canonical_root)
        )
        if not (
            same_path
            and workspace.root_device == validation.root_device
            and workspace.root_file_id == validation.root_file_id
        ):
            raise WorkspaceOperationError("source_root_identity_changed")

    @staticmethod
    def _entry_matches(
        expected: ManifestEntry,
        actual: object,
    ) -> bool:
        if actual is None:
            return False
        return (
            getattr(actual, "failure_code", None) is None
            and expected.sha256 == getattr(actual, "sha256", None)
            and expected.size_bytes == getattr(actual, "size_bytes", None)
            and expected.modified_ns == getattr(actual, "modified_ns", None)
            and expected.device_id == getattr(actual, "device_id", None)
            and expected.file_id == getattr(actual, "file_id", None)
        )

    def _replay_creation(
        self,
        state: str,
        idempotency_key: str,
    ) -> WorkspaceJob | None:
        if state == "completed":
            job = self._store.get_creation_job(idempotency_key)
            if job is None:
                raise WorkspaceOperationError("creation_interrupted")
            return job
        if state == "cancelled":
            return None
        if state == "claimed":
            raise WorkspaceOperationError("creation_in_progress")
        if state == "interrupted":
            raise WorkspaceOperationError("creation_interrupted")
        raise WorkspaceOperationError("creation_failed")

    def _require_workspace(self, workspace_id: str) -> WorkspaceRecord:
        workspace = self._store.get_workspace(workspace_id)
        if workspace is None:
            raise WorkspaceOperationError("workspace_not_found")
        return workspace

    @contextmanager
    def _workspace_mutation(self, workspace_id: str) -> Iterator[None]:
        with self._lock:
            if workspace_id in self._active_workspaces:
                raise WorkspaceOperationError("workspace_operation_active")
            self._active_workspaces.add(workspace_id)
        try:
            yield
        finally:
            with self._lock:
                self._active_workspaces.discard(workspace_id)
