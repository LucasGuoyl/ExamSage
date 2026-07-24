from __future__ import annotations

import io
import inspect
import os
import shutil

import pytest

from exam_predictor.workspace import browser_intake as browser_intake_module
from exam_predictor.workspace.browser_intake import (
    BrowserIntakeError,
    BrowserIntakeWriter,
    BrowserUpload,
    _PosixSnapshotSession,
    _WindowsSnapshotSession,
)
from exam_predictor.workspace.policy import DEFAULT_SCAN_POLICY


def _upload(relative_path: str, content: bytes, *, declared: int | None = None):
    return BrowserUpload(
        relative_path=relative_path,
        size_bytes=len(content) if declared is None else declared,
        stream=io.BytesIO(content),
    )


def test_browser_intake_publishes_nested_unicode_files_atomically(tmp_path):
    workspaces_root = tmp_path / "workspaces"
    writer = BrowserIntakeWriter(workspaces_root)

    root = writer.create_snapshot(
        "workspace-1",
        [
            _upload("Week 1/notes.txt", b"hello"),
            _upload("课程/讲义.md", "复习".encode()),
        ],
    )

    assert root == workspaces_root / "workspace-1" / "browser-intake"
    assert (root / "Week 1" / "notes.txt").read_bytes() == b"hello"
    assert (root / "课程" / "讲义.md").read_bytes() == "复习".encode()
    assert not list((workspaces_root / "workspace-1").glob(".browser-intake-*.tmp"))


def test_browser_intake_reads_streams_in_policy_bounded_chunks(tmp_path):
    class RecordingStream(io.BytesIO):
        def __init__(self, content):
            super().__init__(content)
            self.requested_sizes: list[int] = []

        def read(self, size=-1):
            self.requested_sizes.append(size)
            return super().read(size)

    stream = RecordingStream(b"abcdefgh")
    policy = DEFAULT_SCAN_POLICY.model_copy(update={"hash_chunk_bytes": 3})

    root = BrowserIntakeWriter(tmp_path / "workspaces", policy).create_snapshot(
        "workspace-1",
        [BrowserUpload("notes.txt", 8, stream)],
    )

    assert (root / "notes.txt").read_bytes() == b"abcdefgh"
    assert stream.requested_sizes == [3, 3, 3, 3]


@pytest.mark.parametrize(
    "paths",
    [
        ["Notes.txt", "notes.TXT"],
        ["course", "course/week.txt"],
        ["Course/week.txt", "course"],
    ],
)
def test_browser_intake_rejects_casefold_duplicates_and_file_parent_conflicts(
    tmp_path, paths
):
    workspaces_root = tmp_path / "workspaces"
    uploads = [_upload(path, b"x") for path in paths]

    with pytest.raises(BrowserIntakeError) as caught:
        BrowserIntakeWriter(workspaces_root).create_snapshot("workspace-1", uploads)

    assert caught.value.code == "browser_intake_path_conflict"
    assert not workspaces_root.exists()


@pytest.mark.parametrize(
    "relative_path",
    [
        "",
        ".",
        "../escape.txt",
        "course/../escape.txt",
        "/absolute.txt",
        "C:/drive.txt",
        "C:drive.txt",
        "course\\escape.txt",
        "course//notes.txt",
        "course/./notes.txt",
        "course/notes.txt\x00suffix",
    ],
)
def test_browser_intake_rejects_every_relative_path_escape_before_writing(
    tmp_path, relative_path
):
    workspaces_root = tmp_path / "workspaces"

    with pytest.raises(BrowserIntakeError) as caught:
        BrowserIntakeWriter(workspaces_root).create_snapshot(
            "workspace-1", [_upload(relative_path, b"private")]
        )

    assert caught.value.code == "browser_intake_path_invalid"
    if relative_path:
        assert relative_path not in str(caught.value)
    assert not workspaces_root.exists()


def test_browser_intake_validates_the_complete_plan_before_any_stream_read_or_write(
    tmp_path
):
    class ReadTrackingStream(io.BytesIO):
        read_called = False

        def read(self, size=-1):
            self.read_called = True
            return super().read(size)

    first_stream = ReadTrackingStream(b"safe")
    workspaces_root = tmp_path / "workspaces"

    with pytest.raises(BrowserIntakeError) as caught:
        BrowserIntakeWriter(workspaces_root).create_snapshot(
            "workspace-1",
            [
                BrowserUpload("safe.txt", 4, first_stream),
                _upload("../escape.txt", b"private"),
            ],
        )

    assert caught.value.code == "browser_intake_path_invalid"
    assert first_stream.read_called is False
    assert not workspaces_root.exists()


@pytest.mark.parametrize("declared", [-1, True, 1.5, "5"])
def test_browser_intake_rejects_invalid_declared_sizes_before_writing(tmp_path, declared):
    workspaces_root = tmp_path / "workspaces"

    with pytest.raises(BrowserIntakeError) as caught:
        BrowserIntakeWriter(workspaces_root).create_snapshot(
            "workspace-1",
            [BrowserUpload("notes.txt", declared, io.BytesIO(b"hello"))],
        )

    assert caught.value.code == "browser_intake_size_invalid"
    assert not workspaces_root.exists()


def test_browser_intake_rejects_declared_aggregate_limit_before_reading(tmp_path):
    policy = DEFAULT_SCAN_POLICY.model_copy(update={"max_workspace_bytes": 5})
    stream = io.BytesIO(b"1234")
    workspaces_root = tmp_path / "workspaces"

    with pytest.raises(BrowserIntakeError) as caught:
        BrowserIntakeWriter(workspaces_root, policy).create_snapshot(
            "workspace-1",
            [BrowserUpload("a.txt", 4, stream), _upload("b.txt", b"12")],
        )

    assert caught.value.code == "browser_intake_size_limit"
    assert stream.tell() == 0
    assert not workspaces_root.exists()


@pytest.mark.parametrize(
    ("content", "declared", "expected_code"),
    [
        (b"four", 5, "browser_intake_size_mismatch"),
        (b"six!!!", 5, "browser_intake_size_limit"),
    ],
)
def test_browser_intake_enforces_actual_size_and_removes_partial_snapshot(
    tmp_path, content, declared, expected_code
):
    workspaces_root = tmp_path / "workspaces"

    with pytest.raises(BrowserIntakeError) as caught:
        BrowserIntakeWriter(workspaces_root).create_snapshot(
            "workspace-1", [_upload("notes.txt", content, declared=declared)]
        )

    workspace = workspaces_root / "workspace-1"
    assert caught.value.code == expected_code
    assert not (workspace / "browser-intake").exists()
    assert not list(workspace.glob(".browser-intake-*.tmp"))


def test_browser_intake_maps_stream_failure_safely_and_cleans_only_its_temp_child(
    tmp_path
):
    class FailingStream:
        def __init__(self):
            self.calls = 0

        def read(self, size):
            del size
            self.calls += 1
            if self.calls == 1:
                return b"part"
            raise OSError("private client stream details")

    workspaces_root = tmp_path / "workspaces"
    workspace = workspaces_root / "workspace-1"
    unrelated = workspace / ".browser-intake-keep.tmp"
    unrelated.mkdir(parents=True)
    (unrelated / "keep.txt").write_bytes(b"keep")

    with pytest.raises(BrowserIntakeError) as caught:
        BrowserIntakeWriter(workspaces_root).create_snapshot(
            "workspace-1", [BrowserUpload("notes.txt", 8, FailingStream())]
        )

    assert caught.value.code == "browser_intake_write_failed"
    assert "private client stream details" not in str(caught.value)
    assert unrelated.is_dir()
    assert (unrelated / "keep.txt").read_bytes() == b"keep"
    assert list(workspace.glob(".browser-intake-*.tmp")) == [unrelated]


def test_browser_intake_rejects_existing_snapshot_without_reading_stream(tmp_path):
    class ForbiddenStream:
        def read(self, size):
            del size
            raise AssertionError("stream must not be read")

    workspaces_root = tmp_path / "workspaces"
    existing = workspaces_root / "workspace-1" / "browser-intake"
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_bytes(b"keep")

    with pytest.raises(BrowserIntakeError) as caught:
        BrowserIntakeWriter(workspaces_root).create_snapshot(
            "workspace-1", [BrowserUpload("notes.txt", 1, ForbiddenStream())]
        )

    assert caught.value.code == "browser_intake_exists"
    assert (existing / "keep.txt").read_bytes() == b"keep"


def test_browser_intake_maps_publish_race_to_existing_snapshot_conflict(
    tmp_path, monkeypatch
):
    workspaces_root = tmp_path / "workspaces"
    if os.name == "nt":
        original_rename = _WindowsSnapshotSession._rename_handle

        def race_with_empty_snapshot(session, handle, target):
            target.mkdir()
            return original_rename(session, handle, target)

        monkeypatch.setattr(
            _WindowsSnapshotSession, "_rename_handle", race_with_empty_snapshot
        )
    else:
        original_rename = _PosixSnapshotSession._rename_no_replace

        def race_with_empty_snapshot(session, source_name, destination_name):
            if destination_name == "browser-intake":
                os.mkdir(destination_name, dir_fd=session._workspace_fd)
            return original_rename(session, source_name, destination_name)

        monkeypatch.setattr(
            _PosixSnapshotSession, "_rename_no_replace", race_with_empty_snapshot
        )

    with pytest.raises(BrowserIntakeError) as caught:
        BrowserIntakeWriter(workspaces_root).create_snapshot(
            "workspace-1", [_upload("notes.txt", b"safe")]
        )

    assert caught.value.code == "browser_intake_exists"
    workspace = workspaces_root / "workspace-1"
    assert (workspace / "browser-intake").is_dir()
    assert not list(workspace.glob(".browser-intake-*.tmp"))


@pytest.mark.parametrize("workspace_id", ["", ".", "..", "../escape", "a/b", "a\\b", "C:"])
def test_browser_intake_rejects_workspace_id_escape_before_writing(tmp_path, workspace_id):
    workspaces_root = tmp_path / "workspaces"

    with pytest.raises(BrowserIntakeError) as caught:
        BrowserIntakeWriter(workspaces_root).create_snapshot(
            workspace_id, [_upload("notes.txt", b"safe")]
        )

    assert caught.value.code == "browser_intake_workspace_invalid"
    assert not workspaces_root.exists()


def test_browser_intake_flushes_and_fsyncs_each_file(tmp_path, monkeypatch):
    fsynced: list[int] = []
    monkeypatch.setattr(os, "fsync", fsynced.append)

    BrowserIntakeWriter(tmp_path / "workspaces").create_snapshot(
        "workspace-1",
        [_upload("a.txt", b"a"), _upload("nested/b.txt", b"b")],
    )

    assert len(fsynced) == 2
    assert all(isinstance(file_descriptor, int) for file_descriptor in fsynced)


def test_browser_intake_uses_platform_anchored_no_follow_write_primitives():
    posix_source = inspect.getsource(_PosixSnapshotSession)
    windows_source = inspect.getsource(_WindowsSnapshotSession)

    assert "dir_fd=" in posix_source
    assert "O_NOFOLLOW" in posix_source
    assert "O_EXCL" in posix_source
    assert "FILE_FLAG_OPEN_REPARSE_POINT" in windows_source
    assert "FILE_SHARE_DELETE" in windows_source
    assert "SetFileInformationByHandle" in windows_source


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-bound cleanup")
def test_windows_failure_cleanup_never_uses_path_based_rmtree(tmp_path, monkeypatch):
    class FailingStream:
        def read(self, size):
            del size
            raise OSError("stream failed")

    def forbidden_rmtree(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Windows cleanup must remain bound to created handles")

    monkeypatch.setattr(shutil, "rmtree", forbidden_rmtree)

    with pytest.raises(BrowserIntakeError) as caught:
        BrowserIntakeWriter(tmp_path / "workspaces").create_snapshot(
            "workspace-1", [BrowserUpload("notes.txt", 1, FailingStream())]
        )

    assert caught.value.code == "browser_intake_write_failed"


@pytest.mark.skipif(os.name != "nt", reason="Windows handle identity locking")
def test_windows_session_handle_blocks_temporary_root_replacement(tmp_path):
    workspace = tmp_path / "workspace-1"
    workspace.mkdir()
    session = _WindowsSnapshotSession(workspace)
    replacement_name = workspace / ".replacement.tmp"

    try:
        with pytest.raises(OSError):
            session._temporary_root.rename(replacement_name)
    finally:
        session.cleanup()

    assert not session._temporary_root.exists()
    assert not replacement_name.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-bound cleanup")
def test_windows_cleanup_refuses_a_replaced_file_identity(tmp_path):
    workspace = tmp_path / "workspace-1"
    workspace.mkdir()
    session = _WindowsSnapshotSession(workspace)
    target = session._temporary_root / "notes.txt"
    with session.open_destination(("notes.txt",)) as destination:
        destination.write(b"original")
        destination.flush()
        os.fsync(destination.fileno())
    session._close_children_for_publish()
    target.unlink()
    target.write_bytes(b"replacement")

    session._reopen_children_for_cleanup()
    session.cleanup()

    assert target.read_bytes() == b"replacement"


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory-descriptor race")
def test_posix_publish_rolls_back_a_replacement_injected_before_rename(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace-1"
    workspace.mkdir()
    session = _PosixSnapshotSession(workspace)
    original_root = workspace / ".original-root"
    replacement = workspace / ".replacement"
    replacement.mkdir()
    (replacement / "marker.txt").write_bytes(b"replacement")

    original_rename = session._rename_no_replace

    def substitute_then_rename(source_name, destination_name):
        if destination_name == "browser-intake":
            os.rename(
                session._temporary_name,
                original_root.name,
                src_dir_fd=session._workspace_fd,
                dst_dir_fd=session._workspace_fd,
            )
            os.rename(
                replacement.name,
                session._temporary_name,
                src_dir_fd=session._workspace_fd,
                dst_dir_fd=session._workspace_fd,
            )
        return original_rename(source_name, destination_name)

    monkeypatch.setattr(session, "_rename_no_replace", substitute_then_rename)

    with pytest.raises(BrowserIntakeError) as caught:
        session.publish("browser-intake")

    assert caught.value.code == "browser_intake_write_failed"
    assert not (workspace / "browser-intake").exists()
    [isolated] = workspace.glob(".examsage-unverified-*")
    assert (isolated / "marker.txt").read_bytes() == b"replacement"
    session.cleanup()


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor-bound cleanup race")
def test_posix_cleanup_refuses_a_replacement_injected_before_delete(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace-1"
    workspace.mkdir()
    session = _PosixSnapshotSession(workspace)
    original_root = workspace / ".original-root"
    replacement = workspace / ".replacement"
    replacement.mkdir()
    (replacement / "marker.txt").write_bytes(b"replacement")

    original_rename = session._rename_no_replace

    def substitute_then_rename(source_name, destination_name):
        if destination_name.startswith(".examsage-browser-orphan-"):
            os.rename(
                session._temporary_name,
                original_root.name,
                src_dir_fd=session._workspace_fd,
                dst_dir_fd=session._workspace_fd,
            )
            os.rename(
                replacement.name,
                session._temporary_name,
                src_dir_fd=session._workspace_fd,
                dst_dir_fd=session._workspace_fd,
            )
        return original_rename(source_name, destination_name)

    monkeypatch.setattr(session, "_rename_no_replace", substitute_then_rename)

    session.cleanup()

    [quarantine] = workspace.glob(".examsage-browser-orphan-*")
    assert (quarantine / "marker.txt").read_bytes() == b"replacement"


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor-bound cleanup race")
def test_posix_cleanup_after_verification_cleans_only_retained_identity(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace-1"
    workspace.mkdir()
    session = _PosixSnapshotSession(workspace)
    replacement = workspace / ".replacement"
    replacement.mkdir()
    (replacement / "marker.txt").write_bytes(b"replacement")

    original_identity_check = session._name_matches_temporary_identity

    def verify_then_substitute(name):
        matches = original_identity_check(name)
        if matches and name.startswith(".examsage-browser-orphan-"):
            quarantine = workspace / name
            quarantine.rename(workspace / ".original-root")
            replacement.rename(quarantine)
        return matches

    monkeypatch.setattr(
        session, "_name_matches_temporary_identity", verify_then_substitute
    )

    session.cleanup()

    [quarantine] = workspace.glob(".examsage-browser-orphan-*")
    assert (quarantine / "marker.txt").read_bytes() == b"replacement"


@pytest.mark.skipif(os.name != "nt", reason="Windows workspace handle identity")
def test_windows_session_rejects_workspace_retarget_between_prepare_and_open(
    tmp_path, monkeypatch
):
    workspaces_root = tmp_path / "workspaces"
    outside = tmp_path / "outside"
    outside.mkdir()
    writer = BrowserIntakeWriter(workspaces_root)

    original_start = browser_intake_module._start_snapshot_session

    def retarget_after_prepare(prepared):
        retargeted = prepared.__class__(
            canonical_root=prepared.canonical_root,
            workspace_root=outside,
            identity=prepared.identity,
        )
        return original_start(retargeted)

    monkeypatch.setattr(
        browser_intake_module, "_start_snapshot_session", retarget_after_prepare
    )

    with pytest.raises(BrowserIntakeError) as caught:
        writer.create_snapshot("workspace-1", [_upload("notes.txt", b"safe")])

    assert caught.value.code == "browser_intake_workspace_invalid"
    assert not list(outside.iterdir())


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-bound cleanup")
def test_windows_cleanup_does_not_adopt_a_replaced_directory_identity(tmp_path):
    workspace = tmp_path / "workspace-1"
    workspace.mkdir()
    session = _WindowsSnapshotSession(workspace)
    with session.open_destination(("course", "notes.txt")) as destination:
        destination.write(b"original")
        destination.flush()
        os.fsync(destination.fileno())
    session._close_children_for_publish()
    course = session._temporary_root / "course"
    original_course = session._temporary_root / "original-course"
    course.rename(original_course)
    course.mkdir()

    session._reopen_children_for_cleanup()
    session.cleanup()

    assert course.is_dir()


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor lifecycle")
def test_posix_session_closes_workspace_fd_if_temp_creation_fails(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace-1"
    workspace.mkdir()
    closed: list[int] = []
    monkeypatch.setattr(
        _PosixSnapshotSession,
        "_create_temporary_directory",
        lambda self: (_ for _ in ()).throw(OSError("create failed")),
    )
    original_close = os.close

    def recording_close(file_descriptor):
        closed.append(file_descriptor)
        original_close(file_descriptor)

    monkeypatch.setattr(os, "close", recording_close)

    with pytest.raises(OSError, match="create failed"):
        _PosixSnapshotSession(workspace)

    assert len(closed) == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows handle lifecycle")
def test_windows_session_closes_workspace_handle_if_temp_creation_fails(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace-1"
    workspace.mkdir()
    closed: list[int] = []
    original_close = _WindowsSnapshotSession._close_handle

    def recording_close(self, handle):
        closed.append(handle)
        original_close(self, handle)

    monkeypatch.setattr(_WindowsSnapshotSession, "_close_handle", recording_close)
    monkeypatch.setattr(
        _WindowsSnapshotSession,
        "_create_temporary_directory",
        lambda self: (_ for _ in ()).throw(OSError("create failed")),
    )

    with pytest.raises(OSError, match="create failed"):
        _WindowsSnapshotSession(workspace)

    assert len(closed) == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX creation identity")
def test_posix_temp_creation_refuses_a_replacement_before_open(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace-1"
    workspace.mkdir()
    replacement = workspace / ".replacement"
    replacement.mkdir()
    (replacement / "marker.txt").write_bytes(b"replacement")
    original_open = os.open
    injected = False
    session = None

    def replace_before_temp_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal injected
        if (
            not injected
            and isinstance(path, str)
            and path.startswith(".browser-intake-")
            and dir_fd is not None
        ):
            injected = True
            os.rename(
                path,
                ".original-created-root",
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            os.rename(
                replacement.name,
                path,
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_before_temp_open)

    try:
        with pytest.raises(BrowserIntakeError) as caught:
            session = _PosixSnapshotSession(workspace)
    finally:
        if session is not None:
            session.cleanup()

    assert caught.value.code == "browser_intake_write_failed"
    [candidate] = workspace.glob(".browser-intake-*.tmp")
    assert (candidate / "marker.txt").read_bytes() == b"replacement"


@pytest.mark.skipif(os.name != "nt", reason="Windows creation identity")
def test_windows_temp_creation_refuses_a_replacement_before_open(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace-1"
    workspace.mkdir()
    replacement = workspace / ".replacement"
    replacement.mkdir()
    (replacement / "marker.txt").write_bytes(b"replacement")
    original_open = _WindowsSnapshotSession._open_directory
    injected = False
    session = None

    def replace_before_temp_open(
        self, path, *, delete_access, containment_root
    ):
        nonlocal injected
        if not injected and path.name.startswith(".browser-intake-"):
            injected = True
            path.rename(workspace / ".original-created-root")
            replacement.rename(path)
        return original_open(
            self,
            path,
            delete_access=delete_access,
            containment_root=containment_root,
        )

    monkeypatch.setattr(
        _WindowsSnapshotSession, "_open_directory", replace_before_temp_open
    )

    try:
        with pytest.raises(BrowserIntakeError) as caught:
            session = _WindowsSnapshotSession(workspace)
    finally:
        if session is not None:
            session.cleanup()

    assert caught.value.code == "browser_intake_write_failed"
    [candidate] = workspace.glob(".browser-intake-*.tmp")
    assert (candidate / "marker.txt").read_bytes() == b"replacement"


@pytest.mark.skipif(os.name == "nt", reason="POSIX nested creation identity")
def test_posix_nested_parent_refuses_ordinary_directory_substitution(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace-1"
    workspace.mkdir()
    session = _PosixSnapshotSession(workspace)
    os.mkdir(".replacement", dir_fd=session._temporary_fd)
    replacement = session._temporary_name + "/.replacement"
    with open(workspace / replacement / "marker.txt", "wb") as marker:
        marker.write(b"replacement")
    original_open = os.open
    injected = False

    def replace_before_parent_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal injected
        if not injected and path == "course" and dir_fd is not None:
            injected = True
            os.rename(
                path,
                ".original-created-parent",
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            os.rename(
                ".replacement",
                path,
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_before_parent_open)

    try:
        with pytest.raises(BrowserIntakeError) as caught:
            with session.open_destination(("course", "notes.txt")):
                pytest.fail("substituted parent must never receive a destination")
        assert caught.value.code == "browser_intake_write_failed"
        assert not (workspace / "browser-intake").exists()
    finally:
        session.cleanup()

    [orphan] = workspace.glob(".examsage-browser-orphan-*")
    assert (orphan / "course" / "marker.txt").read_bytes() == b"replacement"
    assert not (orphan / "course" / "notes.txt").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows nested creation identity")
def test_windows_nested_parent_refuses_ordinary_directory_substitution(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace-1"
    workspace.mkdir()
    session = _WindowsSnapshotSession(workspace)
    replacement = session._temporary_root / ".replacement"
    replacement.mkdir()
    (replacement / "marker.txt").write_bytes(b"replacement")
    original_open = _WindowsSnapshotSession._open_directory
    injected = False

    def replace_before_parent_open(
        self, path, *, delete_access, containment_root
    ):
        nonlocal injected
        if not injected and path.name == "course":
            injected = True
            path.rename(session._temporary_root / ".original-created-parent")
            replacement.rename(path)
        return original_open(
            self,
            path,
            delete_access=delete_access,
            containment_root=containment_root,
        )

    monkeypatch.setattr(
        _WindowsSnapshotSession, "_open_directory", replace_before_parent_open
    )

    try:
        with pytest.raises(BrowserIntakeError) as caught:
            with session.open_destination(("course", "notes.txt")):
                pytest.fail("substituted parent must never receive a destination")
        assert caught.value.code == "browser_intake_write_failed"
        assert not (workspace / "browser-intake").exists()
    finally:
        session.cleanup()

    assert (session._temporary_root / "course" / "marker.txt").read_bytes() == (
        b"replacement"
    )
    assert not (session._temporary_root / "course" / "notes.txt").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle ownership")
def test_windows_destination_closes_handle_if_descriptor_transfer_fails(
    tmp_path, monkeypatch
):
    import msvcrt

    workspace = tmp_path / "workspace-1"
    workspace.mkdir()
    session = _WindowsSnapshotSession(workspace)
    original_close = session._close_handle
    closed: list[int] = []

    def recording_close(handle):
        closed.append(handle)
        original_close(handle)

    def fail_transfer(handle, flags):
        del handle, flags
        raise OSError("descriptor transfer failed")

    monkeypatch.setattr(session, "_close_handle", recording_close)
    monkeypatch.setattr(msvcrt, "open_osfhandle", fail_transfer)

    try:
        with pytest.raises(BrowserIntakeError) as caught:
            with session.open_destination(("notes.txt",)):
                pytest.fail("failed transfer must never yield a destination")
        assert caught.value.code == "browser_intake_write_failed"
        assert len(closed) == 1
    finally:
        session.cleanup()


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor lifecycle")
def test_posix_session_closes_workspace_fd_if_workspace_identity_query_fails(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace-1"
    workspace.mkdir()
    original_close = os.close
    closed: list[int] = []

    def fail_fstat(file_descriptor):
        del file_descriptor
        raise OSError("identity failed")

    def recording_close(file_descriptor):
        closed.append(file_descriptor)
        original_close(file_descriptor)

    monkeypatch.setattr(os, "fstat", fail_fstat)
    monkeypatch.setattr(os, "close", recording_close)

    with pytest.raises(OSError, match="identity failed"):
        _PosixSnapshotSession(workspace)

    assert len(closed) == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor lifecycle")
def test_posix_session_closes_all_descriptors_if_temp_identity_query_fails(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace-1"
    workspace.mkdir()
    original_fstat = os.fstat
    original_close = os.close
    fstat_calls = 0
    closed: list[int] = []

    def fail_temp_fstat(file_descriptor):
        nonlocal fstat_calls
        fstat_calls += 1
        if fstat_calls == 2:
            raise OSError("identity failed")
        return original_fstat(file_descriptor)

    def recording_close(file_descriptor):
        closed.append(file_descriptor)
        original_close(file_descriptor)

    monkeypatch.setattr(os, "fstat", fail_temp_fstat)
    monkeypatch.setattr(os, "close", recording_close)

    with pytest.raises(OSError, match="identity failed"):
        _PosixSnapshotSession(workspace)

    assert len(closed) >= 3


@pytest.mark.skipif(os.name != "nt", reason="Windows handle lifecycle")
def test_windows_session_closes_handles_if_temp_identity_query_fails(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace-1"
    workspace.mkdir()
    original_identity = _WindowsSnapshotSession._handle_identity
    original_close = _WindowsSnapshotSession._close_handle
    identity_calls = 0
    closed: list[int] = []

    def fail_temp_identity(self, handle):
        nonlocal identity_calls
        identity_calls += 1
        if identity_calls == 2:
            raise OSError("identity failed")
        return original_identity(self, handle)

    def recording_close(self, handle):
        closed.append(handle)
        original_close(self, handle)

    monkeypatch.setattr(_WindowsSnapshotSession, "_handle_identity", fail_temp_identity)
    monkeypatch.setattr(_WindowsSnapshotSession, "_close_handle", recording_close)

    with pytest.raises(OSError, match="identity failed"):
        _WindowsSnapshotSession(workspace)

    assert len(closed) == 2
