from __future__ import annotations

import io
import os
from pathlib import Path, PurePosixPath

import pytest

from exam_predictor.workspace.filesystem import (
    FILE_ATTRIBUTE_DIRECTORY,
    FILE_ATTRIBUTE_NORMAL,
    FILE_ATTRIBUTE_REPARSE_POINT,
    FILE_FLAG_BACKUP_SEMANTICS,
    FILE_FLAG_OPEN_REPARSE_POINT,
    FILE_SHARE_READ,
    FILE_TYPE_DISK,
    SecureFileOpener,
    SecureOpenError,
    is_reparse_point,
)


def _symlink_or_skip(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")


def test_reparse_attribute_errors_are_not_treated_as_safe():
    class UnreadablePath:
        def stat(self, *, follow_symlinks):
            del follow_symlinks
            raise PermissionError("denied")

    with pytest.raises(PermissionError):
        is_reparse_point(UnreadablePath())  # type: ignore[arg-type]


def test_secure_opener_never_reads_a_link_substitution(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside.txt"
    root.mkdir()
    outside.write_bytes(b"outside-secret")
    source = root / "notes.txt"
    source.write_bytes(b"approved")
    source.unlink()
    _symlink_or_skip(source, outside)

    with pytest.raises(SecureOpenError) as caught:
        with SecureFileOpener().open_regular(root.resolve(), PurePosixPath("notes.txt")) as handle:
            handle.read()

    assert caught.value.code == "source_link_or_reparse"


@pytest.mark.skipif(os.name == "nt", reason="POSIX openat behavior")
def test_posix_secure_opener_reads_a_normal_nested_file(tmp_path):
    root = tmp_path / "root"
    nested = root / "week-1"
    nested.mkdir(parents=True)
    (nested / "notes.txt").write_bytes(b"approved")

    with SecureFileOpener(platform="posix").open_regular(
        root.resolve(), PurePosixPath("week-1/notes.txt")
    ) as handle:
        assert handle.read() == b"approved"


def test_posix_child_directory_anchor_opens_relative_to_the_held_parent(monkeypatch):
    opened = []
    closed = []

    def fake_open(path, flags, *, dir_fd=None):
        opened.append((path, flags, dir_fd))
        return 22

    monkeypatch.setattr(os, "O_DIRECTORY", 0x10000, raising=False)
    monkeypatch.setattr(os, "O_NOFOLLOW", 0x20000, raising=False)
    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "close", closed.append)

    with SecureFileOpener(platform="posix").anchor_child_directory(
        11, "week-1"
    ) as child_fd:
        assert child_fd == 22

    assert opened[0][0] == "week-1"
    assert opened[0][2] == 11
    assert opened[0][1] & os.O_DIRECTORY
    assert opened[0][1] & os.O_NOFOLLOW
    assert closed == [22]


@pytest.mark.skipif(os.name == "nt", reason="POSIX openat behavior")
@pytest.mark.parametrize("swap", ["root", "parent", "final"])
def test_posix_secure_opener_rejects_root_parent_and_final_symlink_swaps(tmp_path, swap):
    original_root = tmp_path / "root"
    outside_root = tmp_path / "outside"
    (original_root / "week-1").mkdir(parents=True)
    (outside_root / "week-1").mkdir(parents=True)
    (original_root / "week-1" / "notes.txt").write_bytes(b"approved")
    (outside_root / "week-1" / "notes.txt").write_bytes(b"outside-secret")
    canonical_root = original_root.resolve()

    if swap == "root":
        original_root.rename(tmp_path / "old-root")
        _symlink_or_skip(original_root, outside_root, target_is_directory=True)
    elif swap == "parent":
        (original_root / "week-1").rename(original_root / "old-week")
        _symlink_or_skip(
            original_root / "week-1", outside_root / "week-1", target_is_directory=True
        )
    else:
        source = original_root / "week-1" / "notes.txt"
        source.unlink()
        _symlink_or_skip(source, outside_root / "week-1" / "notes.txt")

    with pytest.raises(SecureOpenError) as caught:
        with SecureFileOpener(platform="posix").open_regular(
            canonical_root, PurePosixPath("week-1/notes.txt")
        ) as handle:
            handle.read()

    assert caught.value.code == "source_link_or_reparse"


class _ClosingBytesIO(io.BytesIO):
    def __init__(self, value: bytes, adapter: "FakeWindowsAdapter", handle: int) -> None:
        super().__init__(value)
        self._adapter = adapter
        self._handle = handle

    def close(self) -> None:
        self._adapter.closed.append(self._handle)
        super().close()


class FakeWindowsAdapter:
    def __init__(self) -> None:
        self.next_handle = 10
        self.opened: list[tuple[str, int, int]] = []
        self.closed: list[int] = []
        self.events: list[str] = []
        self.attributes: dict[int, int] = {}
        self.final_paths: dict[int, str] = {}
        self.file_types: dict[int, int] = {}
        self.contents = b"approved"

    def create_file(self, path: str, *, flags: int, share_mode: int) -> int:
        handle = self.next_handle
        self.next_handle += 1
        self.opened.append((path, flags, share_mode))
        self.events.append(f"open:{path}")
        is_file = path.casefold().endswith("notes.txt")
        self.attributes[handle] = FILE_ATTRIBUTE_NORMAL if is_file else FILE_ATTRIBUTE_DIRECTORY
        self.final_paths[handle] = path
        self.file_types[handle] = FILE_TYPE_DISK
        return handle

    def get_attributes(self, handle: int) -> int:
        self.events.append(f"attributes:{handle}")
        return self.attributes[handle]

    def get_file_type(self, handle: int) -> int:
        self.events.append(f"type:{handle}")
        return self.file_types[handle]

    def get_final_path(self, handle: int) -> str:
        self.events.append(f"final:{handle}")
        return self.final_paths[handle]

    def open_binary(self, handle: int):
        self.events.append(f"wrap:{handle}")
        return _ClosingBytesIO(self.contents, self, handle)

    def close_handle(self, handle: int) -> None:
        self.closed.append(handle)


def _windows_opener(adapter: FakeWindowsAdapter) -> SecureFileOpener:
    return SecureFileOpener(platform="windows", windows_adapter=adapter)


def test_windows_secure_opener_uses_reparse_point_flags_and_checks_before_wrapping():
    adapter = FakeWindowsAdapter()

    with _windows_opener(adapter).open_regular(
        Path("C:/course"), PurePosixPath("week-1/notes.txt")
    ) as handle:
        assert handle.read() == b"approved"

    assert len(adapter.opened) == 3
    assert all(flags & FILE_FLAG_OPEN_REPARSE_POINT for _, flags, _ in adapter.opened)
    assert all(flags & FILE_FLAG_BACKUP_SEMANTICS for _, flags, _ in adapter.opened[:-1])
    assert all(share_mode == FILE_SHARE_READ for _, _, share_mode in adapter.opened)
    final_handle = max(adapter.attributes)
    assert adapter.events.index(f"attributes:{final_handle}") < adapter.events.index(
        f"wrap:{final_handle}"
    )
    assert adapter.events.index(f"type:{final_handle}") < adapter.events.index(
        f"wrap:{final_handle}"
    )
    assert adapter.events.index(f"final:{final_handle}") < adapter.events.index(
        f"wrap:{final_handle}"
    )
    assert sorted(adapter.closed) == [10, 11, 12]


def test_windows_directory_anchor_holds_verified_handles_until_exit():
    adapter = FakeWindowsAdapter()
    opener = _windows_opener(adapter)

    with opener.anchor_directory(Path("C:/course"), PurePosixPath("week-1")) as target:
        assert target is None
        assert adapter.closed == []
        assert len(adapter.opened) == 2
        assert all(flags & FILE_FLAG_OPEN_REPARSE_POINT for _, flags, _ in adapter.opened)
        assert all(flags & FILE_FLAG_BACKUP_SEMANTICS for _, flags, _ in adapter.opened)
        assert all(share_mode == FILE_SHARE_READ for _, _, share_mode in adapter.opened)

    assert adapter.closed == [11, 10]


def test_windows_secure_opener_rejects_reparse_attributes_and_closes_all_handles():
    adapter = FakeWindowsAdapter()
    original = adapter.get_attributes

    def reparse_parent(handle: int) -> int:
        attributes = original(handle)
        if handle == 11:
            return attributes | FILE_ATTRIBUTE_REPARSE_POINT
        return attributes

    adapter.get_attributes = reparse_parent  # type: ignore[method-assign]

    with pytest.raises(SecureOpenError) as caught:
        with _windows_opener(adapter).open_regular(
            Path("C:/course"), PurePosixPath("week-1/notes.txt")
        ):
            pass

    assert caught.value.code == "source_link_or_reparse"
    assert sorted(adapter.closed) == [10, 11]
    assert not any(event.startswith("wrap:") for event in adapter.events)


def test_windows_secure_opener_rejects_final_path_outside_root_before_read():
    adapter = FakeWindowsAdapter()
    original = adapter.get_final_path

    def outside_final_path(handle: int) -> str:
        if handle == 12:
            return r"\\?\C:\private\notes.txt"
        return original(handle)

    adapter.get_final_path = outside_final_path  # type: ignore[method-assign]

    with pytest.raises(SecureOpenError) as caught:
        with _windows_opener(adapter).open_regular(
            Path("C:/course"), PurePosixPath("week-1/notes.txt")
        ):
            pass

    assert caught.value.code == "source_outside_root"
    assert sorted(adapter.closed) == [10, 11, 12]
    assert not any(event.startswith("wrap:") for event in adapter.events)


@pytest.mark.parametrize(
    ("attributes", "file_type", "expected_code"),
    [
        (FILE_ATTRIBUTE_DIRECTORY, FILE_TYPE_DISK, "source_not_regular"),
        (FILE_ATTRIBUTE_NORMAL, 2, "source_not_regular"),
    ],
)
def test_windows_secure_opener_requires_a_disk_regular_file(
    attributes, file_type, expected_code
):
    adapter = FakeWindowsAdapter()
    original_attributes = adapter.get_attributes
    original_type = adapter.get_file_type

    def final_attributes(handle: int) -> int:
        if handle == 12:
            return attributes
        return original_attributes(handle)

    def final_type(handle: int) -> int:
        if handle == 12:
            return file_type
        return original_type(handle)

    adapter.get_attributes = final_attributes  # type: ignore[method-assign]
    adapter.get_file_type = final_type  # type: ignore[method-assign]

    with pytest.raises(SecureOpenError) as caught:
        with _windows_opener(adapter).open_regular(
            Path("C:/course"), PurePosixPath("week-1/notes.txt")
        ):
            pass

    assert caught.value.code == expected_code
    assert sorted(adapter.closed) == [10, 11, 12]


@pytest.mark.parametrize(
    "relative_path",
    [
        PurePosixPath("."),
        PurePosixPath("../notes.txt"),
        PurePosixPath("/notes.txt"),
        PurePosixPath("nested/notes.txt:secret"),
    ],
)
def test_secure_opener_rejects_paths_that_do_not_name_a_child(relative_path, tmp_path):
    with pytest.raises(SecureOpenError) as caught:
        with SecureFileOpener().open_regular(tmp_path.resolve(), relative_path):
            pass

    assert caught.value.code == "source_path_invalid"
