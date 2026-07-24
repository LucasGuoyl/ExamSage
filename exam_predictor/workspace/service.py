from __future__ import annotations

import os
import queue
import shutil
import stat
import threading
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
from exam_predictor.workspace.filesystem import SecureOpenError, is_reparse_point
from exam_predictor.workspace.models import (
    ApprovalRecord,
    ManifestEntry,
    ManifestRevision,
    SourceMode,
    WorkspaceJob,
    WorkspaceJobStatus,
    WorkspaceRecord,
    WorkspaceState,
)
from exam_predictor.workspace.scanner import RevalidationResult, WorkspaceScanner
from exam_predictor.workspace.store import (
    ActiveWorkspaceOperationError,
    StaleManifestError,
    WorkspaceJobNotFoundError,
    WorkspaceStore,
)


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


class WorkspaceService:
    def __init__(
        self,
        *,
        store: WorkspaceStore,
        scanner: WorkspaceScanner,
        picker: FolderPicker,
        browser_intake: BrowserIntakeWriter,
        run_guard: WorkspaceRunGuard,
        remove_owned_tree: Callable[[Path], None] | None = None,
        close_store_on_shutdown: bool = False,
    ) -> None:
        self._store = store
        self._scanner = scanner
        self._picker = picker
        self._browser_intake = browser_intake
        self._run_guard = run_guard
        self._remove_owned_tree = remove_owned_tree or shutil.rmtree
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
            self._recover_cleanup()
            for job in self._store.recover_running_jobs():
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
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None
        if self._close_store_on_shutdown:
            self._store.close()

    def select_folder(self, idempotency_key: str) -> WorkspaceJob | None:
        """Open the injected picker; return None on cancel or enqueue a scan."""
        existing = self._creation_job(idempotency_key)
        if existing is not None:
            return existing
        try:
            selected = self._picker.choose_folder()
        except Exception as error:
            code = getattr(error, "code", "folder_picker_failed")
            raise WorkspaceOperationError(str(code)) from None
        if selected is None:
            return None
        validation = self._validate_root(Path(selected))
        workspace_id = uuid4().hex
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
        self._store.create_workspace(workspace)
        job = self._enqueue(workspace_id, idempotency_key)
        with self._lock:
            self._creation_jobs[idempotency_key] = job.job_id
        return job

    def create_browser_snapshot(
        self,
        display_name: str,
        files: Sequence[BrowserUpload],
        idempotency_key: str,
    ) -> WorkspaceJob:
        """Create an app-owned snapshot and enqueue its initial scan."""
        existing = self._creation_job(idempotency_key)
        if existing is not None:
            return existing
        workspace_id = uuid4().hex
        root: Path | None = None
        try:
            root = self._browser_intake.create_snapshot(workspace_id, files)
            validation = self._validate_root(root)
        except (BrowserIntakeError, WorkspaceOperationError) as error:
            if root is not None:
                self._queue_orphan_cleanup(workspace_id, root.parent)
            code = getattr(error, "code", "browser_intake_failed")
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
        self._store.create_workspace(workspace)
        job = self._enqueue(workspace_id, idempotency_key)
        with self._lock:
            self._creation_jobs[idempotency_key] = job.job_id
        return job

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
            revision = self._store.get_manifest(workspace_id, revision_id)
            target = next(
                (entry for entry in revision.entries if entry.entry_id == entry_id),
                None,
            )
            if target is None:
                return self._store.set_inclusion(
                    workspace_id, revision_id, (entry_id,), included
                )
            entry_ids = (entry_id,)
            if subtree:
                prefix = f"{target.relative_path.rstrip('/')}/"
                entry_ids = tuple(
                    entry.entry_id
                    for entry in revision.entries
                    if entry.entry_id == entry_id
                    or entry.relative_path.startswith(prefix)
                )
            return self._store.set_inclusion(
                workspace_id,
                revision_id,
                entry_ids,
                included,
            )

    def approve(self, workspace_id: str, revision_id: str) -> ApprovalRecord:
        """Revalidate included files and atomically approve the exact draft."""
        with self._workspace_mutation(workspace_id):
            revision = self._store.get_manifest(workspace_id, revision_id)
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
                if selected:
                    self._store.mark_entry_changed(
                        workspace_id,
                        selected[0].entry_id,
                        getattr(error, "code", "source_root_invalid"),
                    )
                raise StaleManifestError(
                    "The selected source changed and needs review."
                ) from None
            current = {entry.entry_id: entry for entry in validation.entries}
            changed = next(
                (
                    entry
                    for entry in selected
                    if not self._entry_matches(entry, current.get(entry.entry_id))
                ),
                None,
            )
            if changed is not None:
                self._store.mark_entry_changed(
                    workspace_id,
                    changed.entry_id,
                    "source_changed_before_approval",
                )
                raise StaleManifestError(
                    "The selected source changed and needs review."
                )
            self._store.record_access_verified(workspace_id, datetime.now(UTC))
            return self._store.approve(
                workspace_id,
                revision_id,
                revision.policy_version,
            )

    def delete_workspace(self, workspace_id: str) -> None:
        """Delete ExamSage-owned state only after the run guard settles."""
        with self._workspace_mutation(workspace_id):
            if self._run_guard.has_unsettled_runs(workspace_id):
                raise WorkspaceOperationError("workspace_has_unsettled_runs")
            workspace = self._require_workspace(workspace_id)
            try:
                self._store.mark_deleting(workspace_id)
            except ActiveWorkspaceOperationError:
                raise WorkspaceOperationError("workspace_operation_active") from None
            self._run_guard.delete_settled_workspace_runs(workspace_id)
            if workspace.source_mode is SourceMode.NATIVE_FOLDER:
                self._store.delete_workspace_rows(workspace_id)
                return
            owned_path = workspace.canonical_root.parent
            try:
                self._remove_verified_owned_workspace(workspace, owned_path)
            except Exception:
                self._store.queue_cleanup(
                    workspace_id,
                    owned_path,
                    "cleanup_failed",
                )
                record = next(
                    record
                    for record in reversed(self._store.list_cleanup())
                    if record.workspace_id == workspace_id
                )
                self._store.fail_cleanup(record.cleanup_id, "cleanup_failed")
                raise WorkspaceOperationError("cleanup_pending") from None
            self._store.delete_workspace_rows(workspace_id)

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

    def _enqueue(self, workspace_id: str, idempotency_key: str) -> WorkspaceJob:
        with self._lock:
            active_job = self._active_jobs.get(workspace_id)
            if active_job is not None:
                active_key, active_job_id = active_job
                if active_key == idempotency_key:
                    return self._store.get_job(active_job_id)
                raise WorkspaceOperationError("workspace_operation_active")
            candidate = WorkspaceJob(
                job_id=uuid4().hex,
                workspace_id=workspace_id,
                job_kind="scan",
                status=WorkspaceJobStatus.QUEUED,
                idempotency_key=idempotency_key,
                created_at=datetime.now(UTC),
            )
            stored = self._store.create_job(candidate, idempotency_key)
            if stored.job_id != candidate.job_id:
                return stored
            self._active_workspaces.add(workspace_id)
            self._active_jobs[workspace_id] = (
                idempotency_key,
                stored.job_id,
            )
            self._queue_existing(stored)
            return stored

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
            try:
                job = self._store.get_job(job_id)
                self._store.start_job(job_id)
                workspace = self._require_workspace(job.workspace_id)
                validation = self._scanner.revalidate_entries(
                    workspace.canonical_root
                )
                self._verify_stored_root(workspace, validation)
                result = self._scanner.scan(
                    job.workspace_id,
                    workspace.canonical_root,
                    previous_entries=self._store.get_manifest_entries(
                        job.workspace_id
                    ),
                    emit=lambda progress: self._store.append_progress(
                        job_id, progress
                    ),
                )
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
                if owned_path.exists():
                    self._assert_safe_owned_tree(owned_path)
                    self._remove_owned_tree(owned_path)
                self._store.complete_cleanup(record.cleanup_id)
            except Exception:
                self._store.fail_cleanup(record.cleanup_id, "cleanup_retry_failed")

    def _queue_orphan_cleanup(
        self,
        workspace_id: str,
        owned_path: Path,
    ) -> None:
        data_root = self._store.database_path.parent.absolute()
        try:
            relative = owned_path.absolute().relative_to(data_root).as_posix()
            self._cleanup_path(data_root, workspace_id, relative)
            self._store.queue_cleanup(
                workspace_id,
                owned_path,
                "cleanup_pending",
            )
        except (ValueError, WorkspaceOperationError):
            return

    def _remove_verified_owned_workspace(
        self,
        workspace: WorkspaceRecord,
        owned_path: Path,
    ) -> None:
        self._assert_owned_snapshot_path(workspace, owned_path)
        validation = self._scanner.revalidate_entries(workspace.canonical_root)
        self._verify_stored_root(workspace, validation)
        self._assert_safe_owned_tree(owned_path)
        self._remove_owned_tree(owned_path)

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
        if parts not in {
            ("workspaces", workspace_id),
            ("workspaces", workspace_id, "browser-intake"),
        }:
            raise WorkspaceOperationError("cleanup_path_invalid")
        candidate = data_root.joinpath(*parts).absolute()
        if candidate != data_root.joinpath(*parts):
            raise WorkspaceOperationError("cleanup_path_invalid")
        return candidate

    @staticmethod
    def _assert_safe_owned_tree(root: Path) -> None:
        root_stat = root.stat(follow_symlinks=False)
        if (
            root.is_symlink()
            or is_reparse_point(root)
            or not stat.S_ISDIR(root_stat.st_mode)
        ):
            raise WorkspaceOperationError("cleanup_path_invalid")
        for directory, directories, files in os.walk(root, followlinks=False):
            current = Path(directory)
            for name in (*directories, *files):
                candidate = current / name
                metadata = candidate.stat(follow_symlinks=False)
                if candidate.is_symlink() or is_reparse_point(candidate):
                    raise WorkspaceOperationError("cleanup_path_invalid")
                if not (
                    stat.S_ISDIR(metadata.st_mode)
                    or stat.S_ISREG(metadata.st_mode)
                ):
                    raise WorkspaceOperationError("cleanup_path_invalid")

    def _validate_root(self, root: Path) -> RevalidationResult:
        try:
            return self._scanner.revalidate_entries(root)
        except SecureOpenError as error:
            raise WorkspaceOperationError(error.code) from None

    @staticmethod
    def _verify_stored_root(
        workspace: WorkspaceRecord,
        validation: RevalidationResult,
    ) -> None:
        same_path = os.path.normcase(
            os.path.abspath(workspace.canonical_root)
        ) == os.path.normcase(os.path.abspath(validation.canonical_root))
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

    def _creation_job(self, idempotency_key: str) -> WorkspaceJob | None:
        with self._lock:
            job_id = self._creation_jobs.get(idempotency_key)
        if job_id is None:
            return None
        try:
            return self._store.get_job(job_id)
        except WorkspaceJobNotFoundError:
            return None

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
