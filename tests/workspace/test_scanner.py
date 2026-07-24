from __future__ import annotations

import hashlib
import inspect
import io
import os
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import pytest

from exam_predictor.workspace.filesystem import RootAnchor, SecureFileOpener, SecureOpenError
from exam_predictor.workspace.models import SourceState
from exam_predictor.workspace.policy import DEFAULT_SCAN_POLICY
from exam_predictor.workspace.scanner import WorkspaceScanner


def _entry(result, relative_path, *, archive_member_path=None):
    return next(
        item
        for item in result.entries
        if item.relative_path == relative_path
        and item.archive_member_path == archive_member_path
    )


def _write_zip(path: Path, *members: tuple[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members:
            archive.writestr(name, content)


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")


def test_scanner_is_deterministic_for_nested_unicode_and_empty_folders(tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "Course B").mkdir()
    (tmp_path / "Course B" / "zeta.txt").write_text("z", encoding="utf-8")
    (tmp_path / "课程 A").mkdir()
    (tmp_path / "课程 A" / "讲义.md").write_text("内容", encoding="utf-8")
    (tmp_path / "alpha.txt").write_text("a", encoding="utf-8")

    result = WorkspaceScanner().scan("workspace-1", tmp_path / "empty" / "..")

    paths = [entry.relative_path for entry in result.entries]
    assert paths == sorted(paths, key=lambda value: (value.casefold(), value))
    assert paths == ["alpha.txt", "Course B/zeta.txt", "课程 A/讲义.md"]
    assert "empty" not in paths
    assert result.discovered_count == 3
    assert _entry(result, "Course B/zeta.txt").proposed_course_group == "Course B"
    assert _entry(result, "课程 A/讲义.md").proposed_course_group == "课程 A"
    assert _entry(result, "alpha.txt").proposed_course_group == "unclassified"


def test_scanner_does_not_adopt_a_resolved_root_with_a_different_identity(
    tmp_path, monkeypatch
):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "approved.txt").write_bytes(b"approved")
    (outside / "secret.txt").write_bytes(b"outside-secret")
    outside_resolved = outside.resolve()
    original_resolve = Path.resolve

    def substitute_during_resolve(self, *, strict=False):
        if self == root:
            return outside_resolved
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", substitute_during_resolve)

    with pytest.raises(SecureOpenError) as caught:
        WorkspaceScanner().scan("workspace-1", root)

    assert caught.value.code == "source_link_or_reparse"
    assert str(tmp_path) not in str(caught.value)


def test_scanner_maps_the_final_root_reparse_probe_to_a_safe_error(tmp_path):
    calls = 0

    def fail_second_probe(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermissionError(f"denied: {path}")
        return False

    with pytest.raises(SecureOpenError) as caught:
        WorkspaceScanner(is_reparse_point=fail_second_probe).scan(
            "workspace-1", tmp_path
        )

    assert caught.value.code == "source_root_invalid"
    assert str(tmp_path) not in str(caught.value)


def test_revalidate_entries_reuses_bounded_secure_hashing_and_root_identity(tmp_path):
    root = tmp_path / "course"
    root.mkdir()
    source = root / "notes.txt"
    source.write_bytes(b"revision one")
    scanner = WorkspaceScanner()
    scanned = scanner.scan("workspace-1", root)
    selected = tuple(entry for entry in scanned.entries if entry.included)

    validation = scanner.revalidate_entries(root, selected)

    with scanner._directory_opener.anchor_root(root) as root_anchor:
        assert root_anchor.identity is not None
        expected_identity = root_anchor.identity
    assert validation.canonical_root == root.resolve(strict=True)
    assert validation.root_device == str(expected_identity[0])
    assert validation.root_file_id == str(expected_identity[1])
    assert len(validation.entries) == 1
    assert validation.entries[0].entry_id == selected[0].entry_id
    assert validation.entries[0].sha256 == hashlib.sha256(b"revision one").hexdigest()
    assert validation.entries[0].failure_code is None


def test_scan_with_identity_returns_the_identity_from_the_scan_root_open(tmp_path):
    (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")

    execution = WorkspaceScanner().scan_with_identity("workspace-1", tmp_path)

    with WorkspaceScanner()._directory_opener.anchor_root(tmp_path) as root_anchor:
        assert root_anchor.identity is not None
        expected_identity = root_anchor.identity
    assert execution.result.workspace_id == "workspace-1"
    assert execution.canonical_root == tmp_path.resolve(strict=True)
    assert execution.root_device == str(expected_identity[0])
    assert execution.root_file_id == str(expected_identity[1])


def test_scanner_bounds_candidates_and_progress_events_by_policy(tmp_path):
    for index in range(20):
        (tmp_path / f"{index:02}.txt").write_text(str(index), encoding="utf-8")
    policy = DEFAULT_SCAN_POLICY.model_copy(update={"max_files": 3})
    progress = []

    result = WorkspaceScanner(policy).scan(
        "workspace-1",
        tmp_path,
        emit=progress.append,
    )

    assert len(result.entries) <= policy.max_files + 1
    assert len(progress) <= policy.max_files
    assert result.entries[-1].failure_code == "source_file_count_limit"


def test_scanner_hashes_supported_files_with_stable_sha256_and_progress(tmp_path):
    content = b"bounded hashing"
    (tmp_path / "notes.txt").write_bytes(content)
    progress = []

    result = WorkspaceScanner(
        DEFAULT_SCAN_POLICY.model_copy(update={"hash_chunk_bytes": 3})
    ).scan("workspace-1", tmp_path, emit=progress.append)

    entry = _entry(result, "notes.txt")
    assert entry.sha256 == hashlib.sha256(content).hexdigest()
    assert entry.state is SourceState.PENDING_APPROVAL
    assert entry.included is True
    assert result.bytes_hashed == len(content)
    assert progress[-1].discovered_count == 1
    assert progress[-1].bytes_hashed == len(content)


def test_scanner_reads_pre_and_post_hash_metadata_through_the_root_anchor(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"approved")

    class RecordingAnchoredOpener:
        def __init__(self) -> None:
            self.delegate = SecureFileOpener()
            self.events = []

        def stat_regular(
            self,
            canonical_root,
            relative_path,
            *,
            root_anchor=None,
        ):
            self.events.append(("stat", root_anchor))
            return self.delegate.stat_regular(
                canonical_root,
                relative_path,
                root_anchor=root_anchor,
            )

        def stat_open_file(self, source):
            self.events.append(("fstat", None))
            return self.delegate.stat_open_file(source)

        @contextmanager
        def open_regular(
            self,
            canonical_root,
            relative_path,
            *,
            root_anchor=None,
        ):
            self.events.append(("open", root_anchor))
            with self.delegate.open_regular(
                canonical_root,
                relative_path,
                root_anchor=root_anchor,
            ) as source:
                yield source

    opener = RecordingAnchoredOpener()
    scanner = WorkspaceScanner()
    scanner._secure_file_opener = opener

    scanner.scan("workspace-1", tmp_path)

    assert [event for event, _ in opener.events] == [
        "stat",
        "open",
        "fstat",
        "fstat",
        "stat",
    ]
    assert all(
        root_anchor is not None
        for event, root_anchor in opener.events
        if event != "fstat"
    )


def test_scanner_rejects_a_different_file_bound_to_the_actual_read_handle(tmp_path):
    source = tmp_path / "notes.txt"
    substituted = tmp_path / "substituted.bin"
    source.write_bytes(b"file-a")
    substituted.write_bytes(b"file-b")
    source_stat = source.stat()
    substituted_stat = substituted.stat()

    class ABAOpener:
        @staticmethod
        def stat_regular(
            canonical_root,
            relative_path,
            *,
            root_anchor=None,
        ):
            del canonical_root, relative_path, root_anchor
            return source_stat

        @contextmanager
        def open_regular(
            self,
            canonical_root,
            relative_path,
            *,
            root_anchor=None,
        ):
            del self, canonical_root, relative_path, root_anchor
            yield io.BytesIO(b"file-b")

        @staticmethod
        def stat_open_file(source_handle):
            del source_handle
            return substituted_stat

    scanner = WorkspaceScanner()
    scanner._secure_file_opener = ABAOpener()

    result = scanner.scan("workspace-1", tmp_path)

    entry = _entry(result, "notes.txt")
    assert entry.state is SourceState.FAILED
    assert entry.failure_code == "source_changed_during_scan"
    assert entry.sha256 is None


def test_scanner_keeps_unsupported_formats_visible_as_excluded(tmp_path):
    (tmp_path / "lecture.mp4").write_bytes(b"unsupported")

    result = WorkspaceScanner().scan("workspace-1", tmp_path)

    entry = _entry(result, "lecture.mp4")
    assert entry.state is SourceState.EXCLUDED
    assert entry.included is False
    assert entry.inclusion_reason == "unsupported_format"
    assert entry.sha256 is None
    assert result.bytes_hashed == 0


def test_scanner_rejects_symlinks_without_reading_the_target(tmp_path):
    outside = tmp_path.parent / "private.txt"
    outside.write_bytes(b"outside-secret")
    _symlink_or_skip(tmp_path / "linked.txt", outside)

    result = WorkspaceScanner().scan("workspace-1", tmp_path)

    entry = _entry(result, "linked.txt")
    assert entry.state is SourceState.FAILED
    assert entry.failure_code == "source_link_or_reparse"
    assert entry.sha256 is None


def test_scanner_rejects_injected_reparse_points_before_hashing(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_bytes(b"approved")

    result = WorkspaceScanner(is_reparse_point=lambda path: path == source).scan(
        "workspace-1", tmp_path
    )

    entry = _entry(result, "notes.txt")
    assert entry.state is SourceState.FAILED
    assert entry.failure_code == "source_link_or_reparse"
    assert result.bytes_hashed == 0


def test_scanner_anchors_each_directory_during_enumeration(tmp_path):
    (tmp_path / "course").mkdir()
    (tmp_path / "course" / "notes.txt").write_bytes(b"approved")

    class RecordingDirectoryOpener:
        def __init__(self) -> None:
            self.paths = []

        @contextmanager
        def anchor_root(self, root):
            platform = "windows" if os.name == "nt" else "posix"
            root_stat = root.resolve().stat(follow_symlinks=False)
            yield RootAnchor(
                canonical_root=root.resolve(),
                platform=platform,
                identity=(root_stat.st_dev, root_stat.st_ino),
            )

        @contextmanager
        def anchor_directory(self, canonical_root, relative_path=None):
            del canonical_root
            self.paths.append(relative_path.as_posix() if relative_path is not None else None)
            yield None

    opener = RecordingDirectoryOpener()
    scanner = WorkspaceScanner()
    scanner._directory_opener = opener

    scanner.scan("workspace-1", tmp_path)

    assert opener.paths == ["course"]


class _FailingOpener:
    @contextmanager
    def open_regular(
        self,
        canonical_root: Path,
        relative_path: PurePosixPath,
        *,
        root_anchor=None,
    ):
        del canonical_root, relative_path, root_anchor
        raise SecureOpenError("source_open_failed")
        yield io.BytesIO()  # pragma: no cover

    @staticmethod
    def stat_regular(
        canonical_root: Path,
        relative_path: PurePosixPath,
        *,
        root_anchor=None,
    ):
        del root_anchor
        return (canonical_root / relative_path.as_posix()).stat(
            follow_symlinks=False
        )


def test_scanner_isolates_an_open_failure_and_uses_only_safe_messages(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"approved")
    scanner = WorkspaceScanner()
    scanner._secure_file_opener = _FailingOpener()

    result = scanner.scan("workspace-1", tmp_path)

    entry = _entry(result, "notes.txt")
    assert entry.state is SourceState.FAILED
    assert entry.failure_code == "source_open_failed"
    assert entry.safe_message is not None
    assert str(tmp_path) not in entry.safe_message


@pytest.mark.parametrize(
    ("relative_path", "policy_update", "failure_code"),
    [
        ("nested/notes.txt", {"max_depth": 1}, "source_depth_limit"),
        ("notes.txt", {"max_path_chars": 5}, "source_path_limit"),
    ],
)
def test_scanner_applies_depth_and_path_limits(
    tmp_path, relative_path, policy_update, failure_code
):
    source = tmp_path / relative_path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"approved")

    result = WorkspaceScanner(DEFAULT_SCAN_POLICY.model_copy(update=policy_update)).scan(
        "workspace-1", tmp_path
    )

    entry = _entry(result, relative_path)
    assert entry.state is SourceState.FAILED
    assert entry.failure_code == failure_code
    assert entry.sha256 is None


def test_scanner_applies_file_count_limit_in_deterministic_order(tmp_path):
    (tmp_path / "b.txt").write_bytes(b"b")
    (tmp_path / "a.txt").write_bytes(b"a")
    policy = DEFAULT_SCAN_POLICY.model_copy(update={"max_files": 1})

    result = WorkspaceScanner(policy).scan("workspace-1", tmp_path)

    assert _entry(result, "a.txt").state is SourceState.PENDING_APPROVAL
    assert _entry(result, "b.txt").state is SourceState.FAILED
    assert _entry(result, "b.txt").failure_code == "source_file_count_limit"


def test_scanner_file_count_limit_bounds_scandir_consumption(tmp_path, monkeypatch):
    for index in range(100):
        (tmp_path / f"{index:03}.txt").write_bytes(b"x")
    policy = DEFAULT_SCAN_POLICY.model_copy(update={"max_files": 3})
    real_scandir = os.scandir
    consumed = 0

    class CountingIterator:
        def __init__(self, target):
            self._iterator = real_scandir(target)

        def __enter__(self):
            self._iterator.__enter__()
            return self

        def __exit__(self, *args):
            return self._iterator.__exit__(*args)

        def __iter__(self):
            return self

        def __next__(self):
            nonlocal consumed
            entry = next(self._iterator)
            consumed += 1
            return entry

    monkeypatch.setattr(os, "scandir", CountingIterator)

    result = WorkspaceScanner(policy).scan("workspace-1", tmp_path)

    assert consumed <= policy.max_files + 1
    assert len(result.entries) == policy.max_files + 1
    assert result.entries[-1].failure_code == "source_file_count_limit"


def test_scanner_applies_aggregate_selected_size_without_large_fixtures(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"1234")
    (tmp_path / "b.txt").write_bytes(b"5678")
    policy = DEFAULT_SCAN_POLICY.model_copy(update={"max_workspace_bytes": 5})

    result = WorkspaceScanner(policy).scan("workspace-1", tmp_path)

    assert _entry(result, "a.txt").state is SourceState.PENDING_APPROVAL
    assert _entry(result, "b.txt").state is SourceState.FAILED
    assert _entry(result, "b.txt").failure_code == "source_workspace_size_limit"
    assert result.bytes_hashed == 4


def test_scanner_applies_aggregate_limit_to_the_metadata_of_the_file_actually_hashed(
    tmp_path,
):
    source = tmp_path / "notes.txt"
    source.write_bytes(b"1")
    policy = DEFAULT_SCAN_POLICY.model_copy(update={"max_workspace_bytes": 5})

    class GrowBeforeReadScanner(WorkspaceScanner):
        def _read(self, path, root, root_anchor, relative_path, inspect_archive):
            path.write_bytes(b"123456")
            return super()._read(
                path,
                root,
                root_anchor,
                relative_path,
                inspect_archive,
            )

    result = GrowBeforeReadScanner(policy).scan("workspace-1", tmp_path)

    entry = _entry(result, "notes.txt")
    assert entry.state is SourceState.FAILED
    assert entry.failure_code == "source_workspace_size_limit"
    assert entry.sha256 is None


def test_scanner_marks_a_file_changed_during_hash_as_failed(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("before", encoding="utf-8")

    def mutate_after_first_chunk(path: Path, chunk_index: int) -> None:
        if path == source and chunk_index == 0:
            source.write_text("after", encoding="utf-8")

    class SnapshotOpener:
        def __init__(self) -> None:
            self.open_stat = None

        @staticmethod
        def stat_regular(
            canonical_root: Path,
            relative_path: PurePosixPath,
            *,
            root_anchor=None,
        ):
            del root_anchor
            return (canonical_root / relative_path.as_posix()).stat(
                follow_symlinks=False
            )

        @contextmanager
        def open_regular(
            self,
            canonical_root: Path,
            relative_path: PurePosixPath,
            *,
            root_anchor=None,
        ):
            del root_anchor
            self.open_stat = (
                canonical_root / relative_path.as_posix()
            ).stat(follow_symlinks=False)
            yield io.BytesIO(
                (canonical_root / relative_path.as_posix()).read_bytes()
            )

        def stat_open_file(self, source_handle):
            del source_handle
            assert self.open_stat is not None
            return self.open_stat

    scanner = WorkspaceScanner(after_hash_chunk=mutate_after_first_chunk)
    scanner._secure_file_opener = SnapshotOpener()
    result = scanner.scan("workspace-1", tmp_path)
    entry = _entry(result, "notes.txt")
    assert entry.state.value == "failed"
    assert entry.failure_code == "source_changed_during_scan"
    assert entry.sha256 is None


def test_scanner_preserves_approved_unchanged_changed_and_removed_states(tmp_path):
    unchanged = tmp_path / "unchanged.txt"
    changed = tmp_path / "changed.txt"
    removed = tmp_path / "removed.txt"
    unchanged.write_bytes(b"same")
    changed.write_bytes(b"before")
    removed.write_bytes(b"gone")
    first = WorkspaceScanner().scan("workspace-1", tmp_path)
    previous = tuple(
        entry.model_copy(update={"state": SourceState.APPROVED}) for entry in first.entries
    )
    changed.write_bytes(b"after-and-different")
    removed.unlink()

    second = WorkspaceScanner().scan("workspace-1", tmp_path, previous_entries=previous)

    assert _entry(second, "unchanged.txt").state is SourceState.APPROVED
    assert _entry(second, "changed.txt").state is SourceState.CHANGED
    removed_entry = _entry(second, "removed.txt")
    assert removed_entry.state is SourceState.REMOVED
    assert removed_entry.included is False


def test_scanner_reuses_an_interrupted_digest_only_for_exact_metadata(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"same")
    first = WorkspaceScanner().scan("workspace-1", tmp_path)
    scanner = WorkspaceScanner()
    scanner._secure_file_opener = _FailingOpener()

    second = scanner.scan("workspace-1", tmp_path, previous_entries=first.entries)

    assert _entry(second, "notes.txt").sha256 == _entry(first, "notes.txt").sha256
    assert _entry(second, "notes.txt").state is SourceState.PENDING_APPROVAL
    assert second.bytes_hashed == 0


def test_interrupted_reuse_path_contains_no_unreachable_archive_branch():
    source = inspect.getsource(WorkspaceScanner._scan_candidate)

    assert 'if format_category == "archive":' not in source


def test_scanner_rehashes_approved_files_even_when_metadata_is_restored(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_bytes(b"safe")
    first = WorkspaceScanner().scan("workspace-1", tmp_path)
    approved = tuple(
        entry.model_copy(update={"state": SourceState.APPROVED}) for entry in first.entries
    )
    original = source.stat()
    source.write_bytes(b"evil")
    source.touch()
    os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))

    second = WorkspaceScanner().scan("workspace-1", tmp_path, previous_entries=approved)

    entry = _entry(second, "notes.txt")
    assert entry.state is SourceState.CHANGED
    assert entry.sha256 == hashlib.sha256(b"evil").hexdigest()
    assert second.bytes_hashed == 4


def test_scanner_preserves_changed_state_until_the_new_digest_is_approved(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_bytes(b"before")
    first = WorkspaceScanner().scan("workspace-1", tmp_path)
    approved = tuple(
        entry.model_copy(update={"state": SourceState.APPROVED}) for entry in first.entries
    )
    source.write_bytes(b"changed-content")
    changed = WorkspaceScanner().scan("workspace-1", tmp_path, previous_entries=approved)

    repeated = WorkspaceScanner().scan(
        "workspace-1", tmp_path, previous_entries=changed.entries
    )

    assert _entry(changed, "notes.txt").state is SourceState.CHANGED
    assert _entry(repeated, "notes.txt").state is SourceState.CHANGED


def test_scanner_keeps_a_twice_changed_file_changed_until_approval(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_bytes(b"before")
    first = WorkspaceScanner().scan("workspace-1", tmp_path)
    approved = tuple(
        entry.model_copy(update={"state": SourceState.APPROVED}) for entry in first.entries
    )
    source.write_bytes(b"first-change")
    changed = WorkspaceScanner().scan("workspace-1", tmp_path, previous_entries=approved)
    source.write_bytes(b"second-change")

    repeated = WorkspaceScanner().scan(
        "workspace-1", tmp_path, previous_entries=changed.entries
    )

    assert _entry(changed, "notes.txt").state is SourceState.CHANGED
    assert _entry(repeated, "notes.txt").state is SourceState.CHANGED


def test_scanner_inspects_zip_metadata_through_the_secure_opener(tmp_path):
    archive_path = tmp_path / "bundle.zip"
    _write_zip(archive_path, ("notes/week1.txt", b"safe"), ("../escape.txt", b"unsafe"))

    class RecordingOpener:
        def __init__(self) -> None:
            self.paths = []
            self.delegate = SecureFileOpener()

        def stat_regular(
            self,
            canonical_root: Path,
            relative_path: PurePosixPath,
            *,
            root_anchor=None,
        ):
            return self.delegate.stat_regular(
                canonical_root,
                relative_path,
                root_anchor=root_anchor,
            )

        def stat_open_file(self, source):
            return self.delegate.stat_open_file(source)

        @contextmanager
        def open_regular(
            self,
            canonical_root: Path,
            relative_path: PurePosixPath,
            *,
            root_anchor=None,
        ):
            self.paths.append(relative_path.as_posix())
            with self.delegate.open_regular(
                canonical_root,
                relative_path,
                root_anchor=root_anchor,
            ) as source:
                yield source

    opener = RecordingOpener()
    scanner = WorkspaceScanner()
    scanner._secure_file_opener = opener

    result = scanner.scan("workspace-1", tmp_path)

    parent = _entry(result, "bundle.zip")
    safe_member = _entry(result, "bundle.zip", archive_member_path="notes/week1.txt")
    unsafe_member = _entry(result, "bundle.zip", archive_member_path="../escape.txt")
    assert parent.sha256 is not None
    assert safe_member.archive_parent_entry_id == parent.entry_id
    assert safe_member.included is False
    assert unsafe_member.state is SourceState.FAILED
    assert unsafe_member.failure_code == "archive_traversal"
    assert opener.paths == ["bundle.zip"]


def test_scanner_marks_invalid_zip_metadata_failed_without_aborting_other_files(tmp_path):
    (tmp_path / "bad.zip").write_bytes(b"not-a-zip")
    (tmp_path / "notes.txt").write_bytes(b"safe")

    result = WorkspaceScanner().scan("workspace-1", tmp_path)

    assert _entry(result, "bad.zip").state is SourceState.FAILED
    assert _entry(result, "bad.zip").failure_code == "archive_invalid"
    assert _entry(result, "notes.txt").state is SourceState.PENDING_APPROVAL


def test_entry_ids_are_stable_and_scoped_to_the_workspace(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"same")

    first = WorkspaceScanner().scan("workspace-1", tmp_path)
    repeated = WorkspaceScanner().scan("workspace-1", tmp_path)
    another_workspace = WorkspaceScanner().scan("workspace-2", tmp_path)

    assert first.entries[0].entry_id == repeated.entries[0].entry_id
    assert first.entries[0].entry_id != another_workspace.entries[0].entry_id
