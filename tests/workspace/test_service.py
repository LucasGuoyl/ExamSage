from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from exam_predictor.workspace.browser_intake import BrowserIntakeWriter, BrowserUpload
from exam_predictor.ui.workspace_view import subtree_authority
from exam_predictor.workspace.filesystem import SecureOpenError
from exam_predictor.workspace.models import (
    SourceMode,
    SourceState,
    WorkspaceJobStatus,
    WorkspaceState,
)
from exam_predictor.workspace.picker import SubprocessFolderPicker
from exam_predictor.workspace.scanner import WorkspaceScanner
from exam_predictor.workspace.service import (
    OwnedTreeClaim,
    WorkspaceOperationError,
    WorkspaceService,
)
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
    evidence_cleanup=None,
) -> WorkspaceService:
    return WorkspaceService(
        store=store,
        scanner=scanner or WorkspaceScanner(),
        picker=picker,
        browser_intake=BrowserIntakeWriter(tmp_path / "workspaces"),
        run_guard=guard,
        remove_owned_tree=remove_owned_tree,
        evidence_cleanup=evidence_cleanup,
    )


def _secure_test_remover(data_root: Path, removed: list[Path] | None = None):
    def remove(claim: OwnedTreeClaim) -> None:
        path = data_root / Path(claim.relative_path)
        validation = WorkspaceScanner().revalidate_entries(path)
        if (
            validation.root_device != claim.device_id
            or validation.root_file_id != claim.file_id
        ):
            raise OSError("identity changed")
        shutil.rmtree(path)
        if removed is not None:
            removed.append(path)

    return remove


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")


def _directory_link_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError as symlink_error:
        if os.name != "nt":
            pytest.skip(f"directory links unavailable: {symlink_error}")

    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/j", str(link), str(target)],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip("directory links unavailable")


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


def test_picker_to_service_rejects_a_selected_link_root_before_granting_it(
    tmp_path: Path,
    store: WorkspaceStore,
    monkeypatch,
):
    target = tmp_path / "private-course"
    target.mkdir()
    (target / "notes.txt").write_text("private", encoding="utf-8")
    selected_link = tmp_path / "selected-course"
    _directory_link_or_skip(selected_link, target)

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(str(selected_link)).encode(),
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    service = WorkspaceService(
        store=store,
        scanner=WorkspaceScanner(),
        picker=SubprocessFolderPicker(Path(sys.executable)),
        browser_intake=BrowserIntakeWriter(tmp_path / "workspaces"),
        run_guard=FakeRunGuard(),
    )

    with pytest.raises(WorkspaceOperationError) as caught:
        service.select_folder("link-root")

    assert caught.value.code == "folder_picker_selection_invalid"
    assert store.list_workspaces() == ()
    assert selected_link.exists()
    assert (target / "notes.txt").read_text(encoding="utf-8") == "private"


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


def test_rescan_preserves_a_supported_source_excluded_by_the_user(
    tmp_path: Path, store: WorkspaceStore
):
    native_root = tmp_path / "native"
    native_root.mkdir()
    (native_root / "notes.txt").write_text("notes", encoding="utf-8")
    (native_root / "recording.mp4").write_bytes(b"unsupported recording")
    initial_source_snapshot = {
        path.relative_to(native_root).as_posix(): path.read_bytes()
        for path in sorted(native_root.rglob("*"))
        if path.is_file()
    }
    service = _service(
        tmp_path, store, FakePicker([native_root]), FakeRunGuard()
    )
    service.start()
    try:
        initial_job = service.select_folder("pick-1")
        assert initial_job is not None
        _wait_for_job(store, initial_job.job_id)
        initial = store.get_manifest(initial_job.workspace_id)
        notes = next(
            entry for entry in initial.entries if entry.relative_path == "notes.txt"
        )
        excluded = service.set_inclusion(
            initial_job.workspace_id,
            initial.revision_id,
            notes.entry_id,
            False,
        )
        excluded_notes = next(
            entry for entry in excluded.entries if entry.relative_path == "notes.txt"
        )
        assert excluded_notes.state is SourceState.EXCLUDED
        assert excluded_notes.inclusion_reason == "user_excluded"

        rescan_job = service.rescan(initial_job.workspace_id, "rescan-1")
        _wait_for_job(store, rescan_job.job_id)
        rescanned = store.get_manifest(initial_job.workspace_id)
        rescanned_notes = next(
            entry for entry in rescanned.entries if entry.relative_path == "notes.txt"
        )

        assert rescanned_notes.state is SourceState.EXCLUDED
        assert rescanned_notes.included is False
        assert rescanned_notes.inclusion_reason == "user_excluded"
        assert {
            path.relative_to(native_root).as_posix(): path.read_bytes()
            for path in sorted(native_root.rglob("*"))
            if path.is_file()
        } == initial_source_snapshot
    finally:
        service.shutdown()


def test_real_scan_ui_authority_and_service_expand_a_file_parent_subtree(
    tmp_path: Path,
    store: WorkspaceStore,
):
    native_root = tmp_path / "native"
    (native_root / "week-1").mkdir(parents=True)
    (native_root / "week-2").mkdir()
    (native_root / "week-1" / "notes.txt").write_bytes(b"notes")
    (native_root / "week-1" / "slides.txt").write_bytes(b"slides")
    (native_root / "week-2" / "other.txt").write_bytes(b"other")
    service = _service(
        tmp_path, store, FakePicker([native_root]), FakeRunGuard()
    )
    service.start()
    try:
        job = service.select_folder("pick-subtree")
        assert job is not None
        _wait_for_job(store, job.job_id)
        manifest = store.get_manifest(job.workspace_id)
        assert all(entry.item_kind != "folder" for entry in manifest.entries)
        notes = next(
            entry
            for entry in manifest.entries
            if entry.relative_path == "week-1/notes.txt"
        )

        authority = subtree_authority(manifest.entries, notes)

        assert authority is not None
        assert authority.entry_id == notes.entry_id
        assert authority.relative_prefix == "week-1"
        updated = service.set_inclusion(
            job.workspace_id,
            manifest.revision_id,
            authority.entry_id,
            False,
            subtree=True,
        )
        included_by_path = {
            entry.relative_path: entry.included
            for entry in updated.entries
            if entry.archive_parent_entry_id is None
        }
        assert included_by_path == {
            "week-1/notes.txt": False,
            "week-1/slides.txt": False,
            "week-2/other.txt": True,
        }
    finally:
        service.shutdown()


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


def test_start_recovers_every_durable_queued_scan(
    tmp_path: Path, store: WorkspaceStore
):
    native_root = tmp_path / "native"
    native_root.mkdir()
    (native_root / "notes.txt").write_text("notes", encoding="utf-8")
    first = _service(tmp_path, store, FakePicker([native_root]), FakeRunGuard())
    job = first.select_folder("pick-queued")
    assert job is not None
    assert store.get_job(job.job_id).status is WorkspaceJobStatus.QUEUED

    restarted = _service(tmp_path, store, FakePicker([]), FakeRunGuard())
    restarted.start()
    try:
        _wait_for_job(store, job.job_id)
        assert [
            event.event_type for event in store.list_job_events(job.job_id)
        ].count("approval_required") == 1
    finally:
        restarted.shutdown()


def test_initial_creation_idempotency_survives_service_restart(
    tmp_path: Path, store: WorkspaceStore
):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "notes.txt").write_text("first", encoding="utf-8")
    (second_root / "notes.txt").write_text("second", encoding="utf-8")
    first = _service(tmp_path, store, FakePicker([first_root]), FakeRunGuard())
    original = first.select_folder("durable-create")
    assert original is not None
    second_picker = FakePicker([second_root])
    restarted = _service(tmp_path, store, second_picker, FakeRunGuard())

    duplicate = restarted.select_folder("durable-create")

    assert duplicate == original
    assert second_picker.selections == [second_root]
    assert len(store.list_workspaces()) == 1


def test_scan_commit_uses_the_identity_returned_by_the_scan_open(
    tmp_path: Path, store: WorkspaceStore
):
    class MismatchingScanner(WorkspaceScanner):
        def scan_with_identity(self, *args, **kwargs):
            execution = super().scan_with_identity(*args, **kwargs)
            return replace(execution, root_file_id=f"{execution.root_file_id}-replacement")

    native_root = tmp_path / "native"
    native_root.mkdir()
    (native_root / "notes.txt").write_text("notes", encoding="utf-8")
    service = _service(
        tmp_path,
        store,
        FakePicker([native_root]),
        FakeRunGuard(),
        scanner=MismatchingScanner(),
    )
    service.start()
    try:
        job = service.select_folder("pick-identity")
        assert job is not None
        _wait_for_job(store, job.job_id, WorkspaceJobStatus.FAILED)
        assert store.get_job(job.job_id).safe_error_code == "source_root_identity_changed"
        assert store.get_workspace(job.workspace_id).state is WorkspaceState.NEEDS_ATTENTION
        assert store.get_manifest_entries(job.workspace_id) == ()
    finally:
        service.shutdown()


def test_partial_browser_cleanup_is_retried_on_start(
    tmp_path: Path, store: WorkspaceStore
):
    guard = FakeRunGuard()
    failed_once = False

    secure_remove = _secure_test_remover(store.database_path.parent)

    def fail_once(claim: OwnedTreeClaim) -> None:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("locked")
        secure_remove(claim)

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

    restarted = _service(
        tmp_path,
        store,
        FakePicker([]),
        guard,
        remove_owned_tree=fail_once,
    )
    restarted.start()
    try:
        assert store.get_workspace(job.workspace_id) is None
        assert store.list_cleanup() == ()
    finally:
        restarted.shutdown()


def test_unproven_browser_snapshot_orphan_remains_visible_and_is_never_deleted(
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
        assert owned_path.exists()
        pending = next(
            record
            for record in store.list_cleanup()
            if record.cleanup_id == cleanup.cleanup_id
        )
        assert pending.cleanup_id == cleanup.cleanup_id
        assert pending.deletion_root_device is None
        assert pending.deletion_root_file_id is None
        assert pending.attempt_count == 1
    finally:
        restarted.shutdown()


def test_cleanup_without_a_proven_identity_never_calls_the_tree_remover(
    tmp_path: Path, store: WorkspaceStore
):
    owned_path = tmp_path / "workspaces" / "orphan-workspace"
    owned_path.mkdir(parents=True)
    (owned_path / "marker.txt").write_text("keep", encoding="utf-8")
    store.queue_cleanup(
        "orphan-workspace",
        owned_path,
        "cleanup_pending",
    )
    removed: list[Path] = []
    service = _service(
        tmp_path,
        store,
        FakePicker([]),
        FakeRunGuard(),
        remove_owned_tree=lambda claim: removed.append(
            store.database_path.parent / Path(claim.relative_path)
        ),
    )

    service.start()
    try:
        assert removed == []
        assert owned_path.exists()
        [record] = store.list_cleanup()
        assert record.attempt_count == 1
    finally:
        service.shutdown()


def test_start_discovers_task4_browser_orphans_but_does_not_delete_them(
    tmp_path: Path, store: WorkspaceStore
):
    orphan = (
        tmp_path
        / "workspaces"
        / "browser-workspace"
        / ".examsage-browser-orphan-0123456789abcdef"
    )
    orphan.mkdir(parents=True)
    (orphan / "marker.txt").write_text("unproven", encoding="utf-8")
    service = _service(tmp_path, store, FakePicker([]), FakeRunGuard())

    service.start()
    try:
        assert orphan.exists()
        [record] = store.list_cleanup()
        assert record.workspace_id == "browser-workspace"
        assert record.owned_relative_path.endswith(orphan.name)
        assert record.deletion_root_device is None
        assert record.deletion_root_file_id is None
    finally:
        service.shutdown()


def test_start_records_a_claimed_browser_temporary_remnant(
    tmp_path: Path, store: WorkspaceStore
):
    workspace_id = "browser-temp-crash"
    store.claim_creation(
        "upload-temp-crash",
        SourceMode.BROWSER_SNAPSHOT,
        workspace_id,
    )
    remnant = (
        tmp_path
        / "workspaces"
        / workspace_id
        / ".browser-intake-0123456789abcdef0123456789abcdef.tmp"
    )
    remnant.mkdir(parents=True)
    service = _service(tmp_path, store, FakePicker([]), FakeRunGuard())

    service.start()
    try:
        [record] = store.list_cleanup()
        assert record.workspace_id == workspace_id
        assert record.owned_relative_path.endswith(remnant.name)
        assert record.deletion_root_device is not None
        assert record.deletion_root_file_id is not None
        assert remnant.exists()
    finally:
        service.shutdown()


def test_start_records_a_claimed_published_browser_intake_remnant(
    tmp_path: Path, store: WorkspaceStore
):
    workspace_id = "browser-published-crash"
    store.claim_creation(
        "upload-published-crash",
        SourceMode.BROWSER_SNAPSHOT,
        workspace_id,
    )
    writer = BrowserIntakeWriter(tmp_path / "workspaces")
    remnant = writer.create_snapshot(
        workspace_id,
        [BrowserUpload("notes.txt", 5, io.BytesIO(b"notes"))],
    )
    service = _service(tmp_path, store, FakePicker([]), FakeRunGuard())

    service.start()
    try:
        [record] = store.list_cleanup()
        assert record.workspace_id == workspace_id
        assert record.owned_relative_path.endswith("/browser-intake")
        assert record.deletion_root_device is not None
        assert record.deletion_root_file_id is not None
        assert remnant.exists()
    finally:
        service.shutdown()


def test_cleanup_recovery_keeps_a_dangling_link_pending(
    tmp_path: Path, store: WorkspaceStore
):
    owned_path = tmp_path / "workspaces" / "dangling-workspace" / "browser-intake"
    owned_path.parent.mkdir(parents=True)
    _symlink_or_skip(owned_path, tmp_path / "missing-cleanup-target")
    store.queue_cleanup(
        "dangling-workspace",
        owned_path,
        "cleanup_pending",
        deletion_root_identity=("7", "22"),
    )
    removed: list[OwnedTreeClaim] = []
    service = _service(
        tmp_path,
        store,
        FakePicker([]),
        FakeRunGuard(),
        remove_owned_tree=removed.append,
    )

    service.start()
    try:
        assert removed == []
        [record] = store.list_cleanup()
        assert record.attempt_count == 1
        assert os.path.lexists(owned_path)
    finally:
        service.shutdown()


def test_cleanup_recovery_keeps_an_inaccessible_path_pending(
    tmp_path: Path, store: WorkspaceStore, monkeypatch
):
    owned_path = tmp_path / "workspaces" / "blocked-workspace" / "browser-intake"
    owned_path.mkdir(parents=True)
    validation = WorkspaceScanner().revalidate_entries(owned_path)
    store.queue_cleanup(
        "blocked-workspace",
        owned_path,
        "cleanup_pending",
        deletion_root_identity=(
            validation.root_device,
            validation.root_file_id,
        ),
    )
    real_lstat = Path.lstat

    def inaccessible_lstat(path):
        if path == owned_path:
            raise PermissionError("blocked")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", inaccessible_lstat)
    removed: list[OwnedTreeClaim] = []
    service = _service(
        tmp_path,
        store,
        FakePicker([]),
        FakeRunGuard(),
        remove_owned_tree=removed.append,
    )

    service.start()
    try:
        assert removed == []
        [record] = store.list_cleanup()
        assert record.attempt_count == 1
    finally:
        service.shutdown()


def test_browser_deletion_refuses_a_replaced_parent_even_when_intake_is_original(
    tmp_path: Path, store: WorkspaceStore
):
    removed: list[Path] = []
    service = _service(
        tmp_path,
        store,
        FakePicker([]),
        FakeRunGuard(),
        remove_owned_tree=_secure_test_remover(
            store.database_path.parent,
            removed,
        ),
    )
    service.start()
    try:
        job = service.create_browser_snapshot(
            "Browser",
            [BrowserUpload("notes.txt", 5, io.BytesIO(b"notes"))],
            "upload-parent-identity",
        )
        _wait_for_job(store, job.job_id)
        workspace = store.get_workspace(job.workspace_id)
        assert workspace is not None
        original_parent = workspace.canonical_root.parent
        moved_parent = tmp_path / "moved-original-parent"
        original_parent.rename(moved_parent)
        original_parent.mkdir()
        (moved_parent / "browser-intake").rename(
            original_parent / "browser-intake"
        )
        (original_parent / "replacement-marker.txt").write_text(
            "replacement",
            encoding="utf-8",
        )

        with pytest.raises(WorkspaceOperationError) as caught:
            service.delete_workspace(job.workspace_id)

        assert caught.value.code == "cleanup_pending"
        assert removed == []
        assert (original_parent / "replacement-marker.txt").exists()
        assert store.get_workspace(job.workspace_id).state is WorkspaceState.CLEANUP_PENDING
    finally:
        service.shutdown()


def test_stale_approval_is_rejected_before_filesystem_revalidation(
    tmp_path: Path, store: WorkspaceStore
):
    class CountingScanner(WorkspaceScanner):
        def __init__(self):
            super().__init__()
            self.revalidations = 0

        def revalidate_entries(self, root, entries=()):
            self.revalidations += 1
            return super().revalidate_entries(root, entries)

    native_root = tmp_path / "native"
    native_root.mkdir()
    (native_root / "notes.txt").write_text("notes", encoding="utf-8")
    scanner = CountingScanner()
    service = _service(
        tmp_path,
        store,
        FakePicker([native_root]),
        FakeRunGuard(),
        scanner=scanner,
    )
    service.start()
    try:
        job = service.select_folder("pick-stale")
        assert job is not None
        _wait_for_job(store, job.job_id)
        original = store.get_manifest(job.workspace_id)
        [entry] = original.entries
        service.set_inclusion(
            job.workspace_id,
            original.revision_id,
            entry.entry_id,
            True,
        )
        before = scanner.revalidations

        with pytest.raises(StaleManifestError):
            service.approve(job.workspace_id, original.revision_id)

        assert scanner.revalidations == before
    finally:
        service.shutdown()


def test_approval_mismatch_never_marks_a_newer_revision(
    tmp_path: Path, store: WorkspaceStore
):
    native_root = tmp_path / "native"
    native_root.mkdir()
    source = native_root / "notes.txt"
    source.write_text("before", encoding="utf-8")

    class RacingScanner(WorkspaceScanner):
        raced_revision = None

        def revalidate_entries(self, root, entries=()):
            if entries and self.raced_revision is None:
                current = store.get_workspace(entries[0].workspace_id)
                assert current is not None
                self.raced_revision = store.set_inclusion(
                    entries[0].workspace_id,
                    current.current_draft_revision_id,
                    (entries[0].entry_id,),
                    True,
                )
            return super().revalidate_entries(root, entries)

    scanner = RacingScanner()
    service = _service(
        tmp_path,
        store,
        FakePicker([native_root]),
        FakeRunGuard(),
        scanner=scanner,
    )
    service.start()
    try:
        job = service.select_folder("pick-race")
        assert job is not None
        _wait_for_job(store, job.job_id)
        revision = store.get_manifest(job.workspace_id)
        source.write_text("after", encoding="utf-8")

        with pytest.raises(StaleManifestError):
            service.approve(job.workspace_id, revision.revision_id)

        assert scanner.raced_revision is not None
        current = store.get_manifest(job.workspace_id)
        assert current.revision_id == scanner.raced_revision.revision_id
        assert current.entries[0].state is SourceState.PENDING_APPROVAL
        assert store.get_workspace(job.workspace_id).state is WorkspaceState.APPROVAL_REQUIRED
    finally:
        service.shutdown()


def test_root_failure_marks_every_selected_entry_changed(
    tmp_path: Path, store: WorkspaceStore
):
    native_root = tmp_path / "native"
    native_root.mkdir()
    (native_root / "a.txt").write_text("a", encoding="utf-8")
    (native_root / "b.txt").write_text("b", encoding="utf-8")
    service = _service(tmp_path, store, FakePicker([native_root]), FakeRunGuard())
    service.start()
    try:
        job = service.select_folder("pick-root-failure")
        assert job is not None
        _wait_for_job(store, job.job_id)
        revision = store.get_manifest(job.workspace_id)
        native_root.rename(tmp_path / "revoked")

        with pytest.raises(StaleManifestError):
            service.approve(job.workspace_id, revision.revision_id)

        changed = store.get_manifest(job.workspace_id)
        assert {entry.state for entry in changed.entries} == {SourceState.CHANGED}
    finally:
        service.shutdown()


def test_empty_selection_root_failure_still_moves_workspace_to_attention(
    tmp_path: Path, store: WorkspaceStore
):
    native_root = tmp_path / "empty"
    native_root.mkdir()
    service = _service(tmp_path, store, FakePicker([native_root]), FakeRunGuard())
    service.start()
    try:
        job = service.select_folder("pick-empty")
        assert job is not None
        _wait_for_job(store, job.job_id)
        revision = store.get_manifest(job.workspace_id)
        native_root.rename(tmp_path / "revoked-empty")

        with pytest.raises(StaleManifestError):
            service.approve(job.workspace_id, revision.revision_id)

        workspace = store.get_workspace(job.workspace_id)
        assert workspace is not None
        assert workspace.state is WorkspaceState.NEEDS_ATTENTION
        assert workspace.current_draft_revision_id != revision.revision_id
    finally:
        service.shutdown()


def test_public_store_errors_are_mapped_to_safe_operation_codes(
    tmp_path: Path, store: WorkspaceStore
):
    service = _service(tmp_path, store, FakePicker([]), FakeRunGuard())

    with pytest.raises(WorkspaceOperationError) as caught:
        service.rescan("missing-workspace", "missing-rescan")

    assert caught.value.code == "workspace_not_found"


def test_shutdown_timeout_keeps_an_owned_store_open_until_worker_exits(tmp_path: Path):
    started = threading.Event()
    release = threading.Event()

    class BlockingScanner(WorkspaceScanner):
        def scan_with_identity(self, *args, **kwargs):
            started.set()
            assert release.wait(5)
            return super().scan_with_identity(*args, **kwargs)

    native_root = tmp_path / "native"
    native_root.mkdir()
    (native_root / "notes.txt").write_text("notes", encoding="utf-8")
    owned_store = WorkspaceStore(tmp_path / "owned.sqlite3")
    service = WorkspaceService(
        store=owned_store,
        scanner=BlockingScanner(),
        picker=FakePicker([native_root]),
        browser_intake=BrowserIntakeWriter(tmp_path / "workspaces"),
        run_guard=FakeRunGuard(),
        close_store_on_shutdown=True,
    )
    service.start()
    job = service.select_folder("pick-blocked")
    assert job is not None
    assert started.wait(5)

    service.shutdown(timeout_seconds=0.01)

    assert owned_store.get_workspace(job.workspace_id) is not None
    release.set()
    _wait_for_job(owned_store, job.job_id)
    service.shutdown()
    assert owned_store._closed is True


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


def test_workspace_deletion_cleans_derived_evidence_before_source_rows(
    tmp_path: Path,
    store: WorkspaceStore,
):
    native_root = tmp_path / "native-with-evidence"
    native_root.mkdir()
    (native_root / "notes.txt").write_bytes(b"native")
    guard = FakeRunGuard()
    cleanup_calls: list[str] = []

    def cleanup_evidence(workspace_id: str) -> None:
        workspace = store.get_workspace(workspace_id)
        assert workspace is not None
        assert workspace.state is WorkspaceState.DELETING
        assert guard.deleted == [workspace_id]
        cleanup_calls.append(workspace_id)

    service = _service(
        tmp_path,
        store,
        FakePicker([native_root]),
        guard,
        evidence_cleanup=cleanup_evidence,
    )
    service.start()
    try:
        job = service.select_folder("pick-evidence-cleanup")
        assert job is not None
        _wait_for_job(store, job.job_id)

        service.delete_workspace(job.workspace_id)

        assert cleanup_calls == [job.workspace_id]
        assert store.get_workspace(job.workspace_id) is None
        assert (native_root / "notes.txt").read_bytes() == b"native"
    finally:
        service.shutdown()


def test_pending_evidence_cleanup_blocks_rows_and_recovers_on_start(
    tmp_path: Path,
    store: WorkspaceStore,
):
    native_root = tmp_path / "native-pending-evidence"
    native_root.mkdir()
    (native_root / "notes.txt").write_bytes(b"native")
    guard = FakeRunGuard()

    class CleanupResult:
        def __init__(self, value: str) -> None:
            self.value = value

    first = _service(
        tmp_path,
        store,
        FakePicker([native_root]),
        guard,
        evidence_cleanup=lambda _workspace_id: CleanupResult("cleanup_pending"),
    )
    first.start()
    try:
        job = first.select_folder("pick-pending-evidence")
        assert job is not None
        _wait_for_job(store, job.job_id)

        with pytest.raises(WorkspaceOperationError) as caught:
            first.delete_workspace(job.workspace_id)

        assert caught.value.code == "evidence_delete_pending"
        assert store.get_workspace(job.workspace_id).state is WorkspaceState.DELETING
        assert (native_root / "notes.txt").read_bytes() == b"native"
    finally:
        first.shutdown()

    recovered: list[str] = []
    restarted = _service(
        tmp_path,
        store,
        FakePicker([]),
        guard,
        evidence_cleanup=lambda workspace_id: (
            recovered.append(workspace_id) or CleanupResult("deleted")
        ),
    )
    restarted.start()
    try:
        assert recovered == [job.workspace_id]
        assert store.get_workspace(job.workspace_id) is None
        assert (native_root / "notes.txt").read_bytes() == b"native"
    finally:
        restarted.shutdown()


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
        remove_owned_tree=_secure_test_remover(store.database_path.parent),
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
