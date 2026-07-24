from __future__ import annotations

import io
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from exam_predictor.workspace.browser_intake import BrowserIntakeWriter, BrowserUpload
from exam_predictor.workspace.filesystem import SecureOpenError
from exam_predictor.workspace.models import (
    SourceState,
    WorkspaceJobStatus,
    WorkspaceState,
)
from exam_predictor.workspace.scanner import WorkspaceScanner
from exam_predictor.workspace.service import WorkspaceOperationError, WorkspaceService
from exam_predictor.workspace.store import StaleManifestError, WorkspaceStore


@dataclass
class FakePicker:
    selections: list[Path | None]

    def choose_folder(self) -> Path | None:
        return self.selections.pop(0)


@dataclass
class FakeRunGuard:
    unsettled: set[str] = field(default_factory=set)
    deleted: list[str] = field(default_factory=list)

    def has_unsettled_runs(self, workspace_id: str) -> bool:
        return workspace_id in self.unsettled

    def delete_settled_workspace_runs(self, workspace_id: str) -> None:
        self.deleted.append(workspace_id)


def _wait_for_job(
    store: WorkspaceStore,
    job_id: str,
    status: WorkspaceJobStatus = WorkspaceJobStatus.SUCCEEDED,
) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if store.get_job(job_id).status is status:
            return
        time.sleep(0.01)
    assert store.get_job(job_id).status is status


def _service(
    tmp_path: Path,
    store: WorkspaceStore,
    picker: FakePicker,
    guard: FakeRunGuard,
    *,
    scanner: WorkspaceScanner | None = None,
    remove_owned_tree=None,
) -> WorkspaceService:
    return WorkspaceService(
        store=store,
        scanner=scanner or WorkspaceScanner(),
        picker=picker,
        browser_intake=BrowserIntakeWriter(tmp_path / "workspaces"),
        run_guard=guard,
        remove_owned_tree=remove_owned_tree,
    )


@pytest.fixture
def store(tmp_path: Path):
    value = WorkspaceStore(tmp_path / "workspace.sqlite3")
    try:
        yield value
    finally:
        value.close()


def test_folder_cancel_does_not_create_workspace(
    tmp_path: Path, store: WorkspaceStore
):
    service = _service(tmp_path, store, FakePicker([None]), FakeRunGuard())

    assert service.select_folder("pick-1") is None
    assert store.list_workspaces() == ()


def test_initial_scan_emits_progress_and_preserves_one_unreadable_file(
    tmp_path: Path, store: WorkspaceStore
):
    native_root = tmp_path / "native"
    native_root.mkdir()
    (native_root / "notes.txt").write_bytes(b"notes")
    unreadable = native_root / "blocked.txt"
    unreadable.write_bytes(b"do not touch")
    scanner = WorkspaceScanner(
        is_reparse_point=lambda path: path.name == unreadable.name
    )
    service = _service(
        tmp_path,
        store,
        FakePicker([native_root]),
        FakeRunGuard(),
        scanner=scanner,
    )
    service.start()
    try:
        job = service.select_folder("pick-1")
        assert job is not None
        _wait_for_job(store, job.job_id)
        manifest = store.get_manifest(job.workspace_id)
        blocked = next(entry for entry in manifest.entries if entry.relative_path == "blocked.txt")
        assert blocked.state is SourceState.FAILED
        assert unreadable.read_bytes() == b"do not touch"
        assert [event.event_type for event in store.list_job_events(job.job_id)] == [
            "started",
            "scan_progress",
            "scan_progress",
            "approval_required",
        ]
        workspace = store.get_workspace(job.workspace_id)
        assert workspace is not None
        assert workspace.last_access_verified_at is not None
    finally:
        service.shutdown()


def test_rescan_is_idempotent_and_rejects_a_duplicate_active_mutation(
    tmp_path: Path, store: WorkspaceStore
):
    native_root = tmp_path / "native"
    native_root.mkdir()
    (native_root / "notes.txt").write_text("notes", encoding="utf-8")
    service = _service(
        tmp_path, store, FakePicker([native_root]), FakeRunGuard()
    )
    initial = service.select_folder("pick-1")
    assert initial is not None

    assert service.rescan(initial.workspace_id, "pick-1") == initial
    with pytest.raises(WorkspaceOperationError) as caught:
        service.rescan(initial.workspace_id, "pick-2")
    assert caught.value.code == "workspace_operation_active"


def test_stale_and_changed_approval_never_partially_approve(
    tmp_path: Path, store: WorkspaceStore
):
    native_root = tmp_path / "native"
    native_root.mkdir()
    source = native_root / "notes.txt"
    source.write_bytes(b"revision one")
    service = _service(
        tmp_path, store, FakePicker([native_root]), FakeRunGuard()
    )
    service.start()
    try:
        job = service.select_folder("pick-1")
        assert job is not None
        _wait_for_job(store, job.job_id)
        original = store.get_manifest(job.workspace_id)
        [entry] = original.entries
        newer = service.set_inclusion(
            job.workspace_id, original.revision_id, entry.entry_id, True
        )
        with pytest.raises(StaleManifestError):
            service.approve(job.workspace_id, original.revision_id)

        source.write_bytes(b"revision two")
        with pytest.raises(StaleManifestError):
            service.approve(job.workspace_id, newer.revision_id)

        changed = store.get_manifest(job.workspace_id)
        assert changed.revision_id != newer.revision_id
        assert changed.entries[0].state is SourceState.CHANGED
        assert store.get_approval(job.workspace_id) is None
    finally:
        service.shutdown()


def test_access_revoked_root_fails_rescan_without_adopting_replacement(
    tmp_path: Path, store: WorkspaceStore
):
    native_root = tmp_path / "native"
    native_root.mkdir()
    (native_root / "notes.txt").write_text("notes", encoding="utf-8")
    service = _service(
        tmp_path, store, FakePicker([native_root]), FakeRunGuard()
    )
    service.start()
    try:
        initial = service.select_folder("pick-1")
        assert initial is not None
        _wait_for_job(store, initial.job_id)
        moved = tmp_path / "moved"
        native_root.rename(moved)
        replacement = tmp_path / "native"
        replacement.mkdir()
        (replacement / "replacement.txt").write_text("replacement", encoding="utf-8")

        rescan = service.rescan(initial.workspace_id, "rescan-1")
        _wait_for_job(store, rescan.job_id, WorkspaceJobStatus.FAILED)

        failed = store.get_job(rescan.job_id)
        assert failed.safe_error_code == "source_root_identity_changed"
        workspace = store.get_workspace(initial.workspace_id)
        assert workspace is not None
        assert workspace.state is WorkspaceState.NEEDS_ATTENTION
        assert {entry.relative_path for entry in store.get_manifest(initial.workspace_id).entries} == {
            "notes.txt"
        }
    finally:
        service.shutdown()


def test_start_recovers_a_running_scan_once(
    tmp_path: Path, store: WorkspaceStore
):
    native_root = tmp_path / "native"
    native_root.mkdir()
    (native_root / "notes.txt").write_text("notes", encoding="utf-8")
    first = _service(tmp_path, store, FakePicker([native_root]), FakeRunGuard())
    job = first.select_folder("pick-1")
    assert job is not None
    store.start_job(job.job_id)

    restarted = _service(tmp_path, store, FakePicker([]), FakeRunGuard())
    restarted.start()
    try:
        _wait_for_job(store, job.job_id)
        events = store.list_job_events(job.job_id)
        assert [event.event_type for event in events].count("queued") == 1
        assert [event.event_type for event in events].count("approval_required") == 1
    finally:
        restarted.shutdown()


def test_partial_browser_cleanup_is_retried_on_start(
    tmp_path: Path, store: WorkspaceStore
):
    guard = FakeRunGuard()
    failed_once = False

    def fail_once(path: Path) -> None:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("locked")
        shutil.rmtree(path)

    first = _service(
        tmp_path,
        store,
        FakePicker([]),
        guard,
        remove_owned_tree=fail_once,
    )
    first.start()
    try:
        job = first.create_browser_snapshot(
            "Browser course",
            [BrowserUpload("notes.txt", 5, io.BytesIO(b"notes"))],
            "upload-1",
        )
        _wait_for_job(store, job.job_id)
        with pytest.raises(WorkspaceOperationError) as caught:
            first.delete_workspace(job.workspace_id)
        assert caught.value.code == "cleanup_pending"
        workspace = store.get_workspace(job.workspace_id)
        assert workspace is not None
        assert workspace.state is WorkspaceState.CLEANUP_PENDING
        assert store.list_cleanup()[0].attempt_count == 1
    finally:
        first.shutdown()

    restarted = _service(tmp_path, store, FakePicker([]), guard)
    restarted.start()
    try:
        assert store.get_workspace(job.workspace_id) is None
        assert store.list_cleanup() == ()
    finally:
        restarted.shutdown()


def test_browser_snapshot_orphan_is_durably_cleaned_after_validation_failure(
    tmp_path: Path, store: WorkspaceStore
):
    class RejectSnapshotRoot(WorkspaceScanner):
        def revalidate_entries(self, root, entries=()):
            raise SecureOpenError("source_root_invalid")

    failed = _service(
        tmp_path,
        store,
        FakePicker([]),
        FakeRunGuard(),
        scanner=RejectSnapshotRoot(),
    )
    with pytest.raises(WorkspaceOperationError) as caught:
        failed.create_browser_snapshot(
            "Browser",
            [BrowserUpload("notes.txt", 5, io.BytesIO(b"notes"))],
            "upload-1",
        )
    assert caught.value.code == "source_root_invalid"
    [cleanup] = store.list_cleanup()
    owned_path = store.database_path.parent / Path(cleanup.owned_relative_path)
    assert owned_path.exists()

    restarted = _service(tmp_path, store, FakePicker([]), FakeRunGuard())
    restarted.start()
    try:
        assert not owned_path.exists()
        assert store.list_cleanup() == ()
    finally:
        restarted.shutdown()


@pytest.mark.parametrize("run_state", ["queued", "running", "stopping", "paused"])
def test_deletion_blocks_every_nonterminal_run_state(
    tmp_path: Path, store: WorkspaceStore, run_state: str
):
    native_root = tmp_path / run_state
    native_root.mkdir()
    (native_root / "notes.txt").write_text(run_state, encoding="utf-8")
    guard = FakeRunGuard()
    service = _service(tmp_path, store, FakePicker([native_root]), guard)
    service.start()
    try:
        job = service.select_folder(f"pick-{run_state}")
        assert job is not None
        _wait_for_job(store, job.job_id)
        guard.unsettled.add(job.workspace_id)

        with pytest.raises(WorkspaceOperationError) as caught:
            service.delete_workspace(job.workspace_id)

        assert caught.value.code == "workspace_has_unsettled_runs"
        assert store.get_workspace(job.workspace_id) is not None
        assert native_root.exists()
    finally:
        service.shutdown()


def test_native_deletion_removes_only_owned_rows_and_never_source_bytes(
    tmp_path: Path, store: WorkspaceStore
):
    native_root = tmp_path / "native"
    native_root.mkdir()
    (native_root / "notes.txt").write_bytes(b"native")
    nested = native_root / "nested"
    nested.mkdir()
    (nested / "slides.pdf").write_bytes(b"slides")
    guard = FakeRunGuard()
    service = _service(tmp_path, store, FakePicker([native_root]), guard)
    service.start()
    try:
        job = service.select_folder("pick-1")
        assert job is not None
        _wait_for_job(store, job.job_id)
        before = {
            path.relative_to(native_root): path.read_bytes()
            for path in native_root.rglob("*")
            if path.is_file()
        }

        service.delete_workspace(job.workspace_id)

        after = {
            path.relative_to(native_root): path.read_bytes()
            for path in native_root.rglob("*")
            if path.is_file()
        }
        assert after == before
        assert native_root.exists()
        assert store.get_workspace(job.workspace_id) is None
        assert guard.deleted == [job.workspace_id]
    finally:
        service.shutdown()


def test_browser_approval_and_delete_all_remove_only_verified_owned_snapshots(
    tmp_path: Path, store: WorkspaceStore
):
    native_a = tmp_path / "native-a"
    native_b = tmp_path / "native-b"
    for root, content in ((native_a, "a"), (native_b, "b")):
        root.mkdir()
        (root / "notes.txt").write_text(content, encoding="utf-8")
    guard = FakeRunGuard()
    service = _service(
        tmp_path,
        store,
        FakePicker([native_a, native_b]),
        guard,
    )
    service.start()
    try:
        first = service.select_folder("pick-a")
        second = service.select_folder("pick-b")
        browser = service.create_browser_snapshot(
            "Browser",
            [BrowserUpload("upload.txt", 6, io.BytesIO(b"upload"))],
            "upload-1",
        )
        assert first is not None and second is not None
        for job in (first, second, browser):
            _wait_for_job(store, job.job_id)

        browser_manifest = store.get_manifest(browser.workspace_id)
        approval = service.approve(browser.workspace_id, browser_manifest.revision_id)
        assert approval.revision_id == browser_manifest.revision_id
        browser_workspace = store.get_workspace(browser.workspace_id)
        assert browser_workspace is not None
        owned_browser_root = browser_workspace.canonical_root.parent

        guard.unsettled.update({second.workspace_id, first.workspace_id})
        conflicts = service.delete_all_workspaces()

        assert conflicts == tuple(sorted((first.workspace_id, second.workspace_id)))
        assert store.get_workspace(browser.workspace_id) is None
        assert not owned_browser_root.exists()
        assert (native_a / "notes.txt").read_text(encoding="utf-8") == "a"
        assert (native_b / "notes.txt").read_text(encoding="utf-8") == "b"
    finally:
        service.shutdown()
