from __future__ import annotations

import io
import inspect
import os
import shutil

import pytest

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

        def race_with_empty_snapshot(session, destination_name):
            os.mkdir(destination_name, dir_fd=session._workspace_fd)
            return original_rename(session, destination_name)

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
