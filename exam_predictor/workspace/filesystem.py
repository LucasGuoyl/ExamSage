from __future__ import annotations

import errno
import hashlib
import ntpath
import os
import secrets
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator, Literal, Protocol


FILE_ATTRIBUTE_DIRECTORY = 0x10
FILE_ATTRIBUTE_NORMAL = 0x80
FILE_ATTRIBUTE_REPARSE_POINT = 0x400
FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
FILE_SHARE_READ = 0x00000001
FILE_TYPE_DISK = 0x0001


class SecureOpenError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class RootAnchor:
    canonical_root: Path
    platform: str
    directory_fd: int | None = None
    identity: tuple[int, int] | None = None


class OwnedFilesystemError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class OwnedDirectoryAnchor:
    path: Path
    platform: str
    identity: tuple[int, int]
    descriptor: int


@dataclass(frozen=True)
class OwnedReadFile:
    parent: OwnedDirectoryAnchor
    name: str
    descriptor: int
    identity: tuple[int, int]


@dataclass(frozen=True)
class OwnedMutationFile:
    parent: OwnedDirectoryAnchor
    name: str
    descriptor: int
    identity: tuple[int, int]
    created: bool = False


@dataclass
class OwnedTemporaryFile:
    parent: OwnedDirectoryAnchor
    name: str
    descriptor: int
    identity: tuple[int, int]
    _released: bool = False

    def release(self) -> None:
        self._released = True


# Compatibility for callers migrated in the same Task 3 change set.
OwnedOpenFile = OwnedMutationFile


@dataclass(frozen=True)
class OwnedMutationResult:
    identity: tuple[int, int]
    sha256: str
    size: int
    rename_write_through: bool
    final_file_flushed: bool
    parent_directory_flushed: bool


class _WindowsAdapter(Protocol):
    def create_file(self, path: str, *, flags: int, share_mode: int) -> int: ...

    def get_attributes(self, handle: int) -> int: ...

    def get_file_type(self, handle: int) -> int: ...

    def get_file_identity(self, handle: int) -> tuple[int, int]: ...

    def get_final_path(self, handle: int) -> str: ...

    def open_binary(self, handle: int) -> BinaryIO: ...

    def close_handle(self, handle: int) -> None: ...


def is_reparse_point(path: Path) -> bool:
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & FILE_ATTRIBUTE_REPARSE_POINT)


def _validate_relative_path(relative_path: PurePosixPath) -> tuple[str, ...]:
    parts = relative_path.parts
    if (
        not parts
        or relative_path.is_absolute()
        or ".." in parts
        or any(part in {"", "."} or "\\" in part or "\x00" in part or ":" in part for part in parts)
    ):
        raise SecureOpenError("source_path_invalid")
    return parts


def _map_posix_open_error(error: OSError) -> SecureOpenError:
    if error.errno in {errno.ELOOP, errno.ENOTDIR}:
        return SecureOpenError("source_link_or_reparse")
    return SecureOpenError("source_open_failed")


def _normalize_windows_final_path(path: str) -> str:
    if path.startswith("\\\\?\\UNC\\"):
        path = "\\\\" + path[8:]
    elif path.startswith("\\\\?\\"):
        path = path[4:]
    return ntpath.normcase(ntpath.abspath(path))


def _is_windows_path_beneath(path: str, root: str) -> bool:
    normalized_path = _normalize_windows_final_path(path)
    normalized_root = _normalize_windows_final_path(root)
    try:
        return ntpath.commonpath((normalized_path, normalized_root)) == normalized_root
    except ValueError:
        return False


class _NativeWindowsAdapter:
    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("native Windows file APIs are unavailable")

        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._wintypes = wintypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class ByHandleFileInformation(ctypes.Structure):
            _fields_ = [
                ("dwFileAttributes", wintypes.DWORD),
                ("ftCreationTime", wintypes.FILETIME),
                ("ftLastAccessTime", wintypes.FILETIME),
                ("ftLastWriteTime", wintypes.FILETIME),
                ("dwVolumeSerialNumber", wintypes.DWORD),
                ("nFileSizeHigh", wintypes.DWORD),
                ("nFileSizeLow", wintypes.DWORD),
                ("nNumberOfLinks", wintypes.DWORD),
                ("nFileIndexHigh", wintypes.DWORD),
                ("nFileIndexLow", wintypes.DWORD),
            ]

        self._file_information_type = ByHandleFileInformation
        self._kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        self._kernel32.CreateFileW.restype = wintypes.HANDLE
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._kernel32.GetFileType.argtypes = [wintypes.HANDLE]
        self._kernel32.GetFileType.restype = wintypes.DWORD
        self._kernel32.GetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ByHandleFileInformation),
        ]
        self._kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
        self._kernel32.GetFinalPathNameByHandleW.argtypes = [
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        self._kernel32.GetFinalPathNameByHandleW.restype = wintypes.DWORD

    def create_file(self, path: str, *, flags: int, share_mode: int) -> int:
        generic_read = 0x80000000
        open_existing = 3
        handle = self._kernel32.CreateFileW(
            path,
            generic_read,
            share_mode,
            None,
            open_existing,
            flags,
            None,
        )
        invalid_handle_value = self._ctypes.c_void_p(-1).value
        if handle == invalid_handle_value:
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        return int(handle)

    def get_attributes(self, handle: int) -> int:
        information = self._file_information_type()
        if not self._kernel32.GetFileInformationByHandle(handle, self._ctypes.byref(information)):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        return int(information.dwFileAttributes)

    def get_file_type(self, handle: int) -> int:
        return int(self._kernel32.GetFileType(handle))

    def get_file_identity(self, handle: int) -> tuple[int, int]:
        information = self._file_information_type()
        if not self._kernel32.GetFileInformationByHandle(handle, self._ctypes.byref(information)):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        file_index = (int(information.nFileIndexHigh) << 32) | int(information.nFileIndexLow)
        return int(information.dwVolumeSerialNumber), file_index

    def get_final_path(self, handle: int) -> str:
        required = self._kernel32.GetFinalPathNameByHandleW(handle, None, 0, 0)
        if not required:
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        buffer = self._ctypes.create_unicode_buffer(required + 1)
        written = self._kernel32.GetFinalPathNameByHandleW(handle, buffer, len(buffer), 0)
        if not written or written >= len(buffer):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        return buffer.value

    def open_binary(self, handle: int) -> BinaryIO:
        import msvcrt

        file_descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        try:
            return os.fdopen(file_descriptor, "rb", buffering=0, closefd=True)
        except BaseException:
            os.close(file_descriptor)
            raise

    def close_handle(self, handle: int) -> None:
        self._kernel32.CloseHandle(handle)


class SecureFileOpener:
    def __init__(
        self,
        *,
        platform: str | None = None,
        windows_adapter: _WindowsAdapter | None = None,
    ) -> None:
        self._platform = platform or ("windows" if os.name == "nt" else "posix")
        self._windows_adapter = windows_adapter

    @contextmanager
    def open_regular(
        self,
        canonical_root: Path,
        relative_path: PurePosixPath,
        *,
        root_anchor: RootAnchor | None = None,
    ) -> Iterator[BinaryIO]:
        parts = _validate_relative_path(relative_path)
        if root_anchor is not None and root_anchor.platform != self._platform:
            raise SecureOpenError("source_root_invalid")
        if self._platform == "windows":
            with self._open_windows(canonical_root, parts) as source:
                yield source
            return
        if self._platform != "posix":
            raise ValueError(f"unsupported secure-open platform: {self._platform}")
        anchored_fd = root_anchor.directory_fd if root_anchor is not None else None
        with self._open_posix(canonical_root, parts, root_fd=anchored_fd) as source:
            yield source

    def stat_regular(
        self,
        canonical_root: Path,
        relative_path: PurePosixPath,
        *,
        root_anchor: RootAnchor | None = None,
    ) -> os.stat_result:
        with self.open_regular(
            canonical_root,
            relative_path,
            root_anchor=root_anchor,
        ) as source:
            return os.fstat(source.fileno())

    @staticmethod
    def stat_open_file(source: BinaryIO) -> os.stat_result:
        return os.fstat(source.fileno())

    @contextmanager
    def anchor_root(self, root: Path) -> Iterator[RootAnchor]:
        if self._platform == "windows":
            with self._anchor_windows_root(root) as anchor:
                yield anchor
            return
        if self._platform != "posix":
            raise ValueError(f"unsupported secure-open platform: {self._platform}")
        with self._anchor_posix_root(root) as anchor:
            yield anchor

    @contextmanager
    def _anchor_posix_root(self, root: Path) -> Iterator[RootAnchor]:
        root_fd: int | None = None
        try:
            flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            root_fd = os.open(root, flags)
            root_stat = os.fstat(root_fd)
            if not stat.S_ISDIR(root_stat.st_mode):
                raise SecureOpenError("source_root_invalid")
            canonical_root = root.resolve(strict=True)
            canonical_stat = canonical_root.stat(follow_symlinks=False)
            if (root_stat.st_dev, root_stat.st_ino) != (
                canonical_stat.st_dev,
                canonical_stat.st_ino,
            ):
                raise SecureOpenError("source_link_or_reparse")
            yield RootAnchor(
                canonical_root=canonical_root,
                platform="posix",
                directory_fd=root_fd,
                identity=(root_stat.st_dev, root_stat.st_ino),
            )
        except SecureOpenError:
            raise
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise SecureOpenError("source_link_or_reparse") from None
            raise SecureOpenError("source_root_invalid") from None
        finally:
            if root_fd is not None:
                os.close(root_fd)

    @contextmanager
    def anchor_directory(
        self,
        canonical_root: Path,
        relative_path: PurePosixPath | None = None,
    ) -> Iterator[int | None]:
        parts = () if relative_path is None else _validate_relative_path(relative_path)
        if self._platform == "windows":
            with self._anchor_windows_directory(canonical_root, parts):
                yield None
            return
        if self._platform != "posix":
            raise ValueError(f"unsupported secure-open platform: {self._platform}")
        with self._anchor_posix_directory(canonical_root, parts) as directory_fd:
            yield directory_fd

    @contextmanager
    def _anchor_posix_directory(self, canonical_root: Path, parts: tuple[str, ...]) -> Iterator[int]:
        directory_fds: list[int] = []
        try:
            directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
            directory_flags |= getattr(os, "O_NOFOLLOW", 0)
            directory_fds.append(os.open(canonical_root, directory_flags))
            for part in parts:
                directory_fds.append(os.open(part, directory_flags, dir_fd=directory_fds[-1]))
            yield directory_fds[-1]
        except SecureOpenError:
            raise
        except OSError as error:
            raise _map_posix_open_error(error) from None
        finally:
            for directory_fd in reversed(directory_fds):
                os.close(directory_fd)

    @contextmanager
    def anchor_child_directory(self, parent_fd: int, child_name: str) -> Iterator[int]:
        parts = _validate_relative_path(PurePosixPath(child_name))
        if self._platform != "posix" or len(parts) != 1:
            raise SecureOpenError("source_path_invalid")
        child_fd: int | None = None
        try:
            flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            child_fd = os.open(parts[0], flags, dir_fd=parent_fd)
            yield child_fd
        except SecureOpenError:
            raise
        except OSError as error:
            raise _map_posix_open_error(error) from None
        finally:
            if child_fd is not None:
                os.close(child_fd)

    @contextmanager
    def _open_posix(
        self,
        canonical_root: Path,
        parts: tuple[str, ...],
        *,
        root_fd: int | None = None,
    ) -> Iterator[BinaryIO]:
        directory_fds: list[int] = []
        file_fd: int | None = None
        try:
            root_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
            root_flags |= getattr(os, "O_NOFOLLOW", 0)
            directory_fds.append(os.open(canonical_root, root_flags) if root_fd is None else os.dup(root_fd))
            directory_flags = root_flags
            for part in parts[:-1]:
                directory_fds.append(os.open(part, directory_flags, dir_fd=directory_fds[-1]))

            file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            file_flags |= getattr(os, "O_NOFOLLOW", 0)
            file_fd = os.open(parts[-1], file_flags, dir_fd=directory_fds[-1])
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                raise SecureOpenError("source_not_regular")
            source = os.fdopen(file_fd, "rb", buffering=0, closefd=True)
            file_fd = None
            with source:
                yield source
        except SecureOpenError:
            raise
        except OSError as error:
            raise _map_posix_open_error(error) from None
        finally:
            if file_fd is not None:
                os.close(file_fd)
            for directory_fd in reversed(directory_fds):
                os.close(directory_fd)

    @contextmanager
    def _open_windows(self, canonical_root: Path, parts: tuple[str, ...]) -> Iterator[BinaryIO]:
        adapter = self._windows_adapter or _NativeWindowsAdapter()
        root_path = ntpath.abspath(str(canonical_root))
        directory_handles: list[int] = []
        file_handle: int | None = None
        try:
            directory_flags = FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS
            for count in range(len(parts)):
                directory_path = ntpath.join(root_path, *parts[:count])
                directory_handle = adapter.create_file(
                    directory_path,
                    flags=directory_flags,
                    share_mode=FILE_SHARE_READ,
                )
                directory_handles.append(directory_handle)
                attributes = adapter.get_attributes(directory_handle)
                if attributes & FILE_ATTRIBUTE_REPARSE_POINT:
                    raise SecureOpenError("source_link_or_reparse")
                if not attributes & FILE_ATTRIBUTE_DIRECTORY:
                    raise SecureOpenError("source_not_regular")
                if adapter.get_file_type(directory_handle) != FILE_TYPE_DISK:
                    raise SecureOpenError("source_not_regular")

            file_path = ntpath.join(root_path, *parts)
            file_handle = adapter.create_file(
                file_path,
                flags=FILE_FLAG_OPEN_REPARSE_POINT,
                share_mode=FILE_SHARE_READ,
            )
            attributes = adapter.get_attributes(file_handle)
            if attributes & FILE_ATTRIBUTE_REPARSE_POINT:
                raise SecureOpenError("source_link_or_reparse")
            if attributes & FILE_ATTRIBUTE_DIRECTORY:
                raise SecureOpenError("source_not_regular")
            if adapter.get_file_type(file_handle) != FILE_TYPE_DISK:
                raise SecureOpenError("source_not_regular")

            anchored_root = adapter.get_final_path(directory_handles[0])
            final_path = adapter.get_final_path(file_handle)
            if not _is_windows_path_beneath(final_path, anchored_root):
                raise SecureOpenError("source_outside_root")

            source = adapter.open_binary(file_handle)
            file_handle = None
            with source:
                yield source
        except SecureOpenError:
            raise
        except OSError:
            raise SecureOpenError("source_open_failed") from None
        finally:
            if file_handle is not None:
                adapter.close_handle(file_handle)
            for directory_handle in reversed(directory_handles):
                adapter.close_handle(directory_handle)

    @contextmanager
    def _anchor_windows_directory(self, canonical_root: Path, parts: tuple[str, ...]) -> Iterator[None]:
        adapter = self._windows_adapter or _NativeWindowsAdapter()
        root_path = ntpath.abspath(str(canonical_root))
        directory_handles: list[int] = []
        try:
            flags = FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS
            for count in range(len(parts) + 1):
                directory_path = ntpath.join(root_path, *parts[:count])
                directory_handle = adapter.create_file(
                    directory_path,
                    flags=flags,
                    share_mode=FILE_SHARE_READ,
                )
                directory_handles.append(directory_handle)
                attributes = adapter.get_attributes(directory_handle)
                if attributes & FILE_ATTRIBUTE_REPARSE_POINT:
                    raise SecureOpenError("source_link_or_reparse")
                if not attributes & FILE_ATTRIBUTE_DIRECTORY:
                    raise SecureOpenError("source_not_regular")
                if adapter.get_file_type(directory_handle) != FILE_TYPE_DISK:
                    raise SecureOpenError("source_not_regular")

            anchored_root = adapter.get_final_path(directory_handles[0])
            final_path = adapter.get_final_path(directory_handles[-1])
            if not _is_windows_path_beneath(final_path, anchored_root):
                raise SecureOpenError("source_outside_root")
            yield
        except SecureOpenError:
            raise
        except OSError:
            raise SecureOpenError("source_open_failed") from None
        finally:
            for directory_handle in reversed(directory_handles):
                adapter.close_handle(directory_handle)

    @contextmanager
    def _anchor_windows_root(self, root: Path) -> Iterator[RootAnchor]:
        adapter = self._windows_adapter or _NativeWindowsAdapter()
        root_handle: int | None = None
        try:
            flags = FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS
            root_handle = adapter.create_file(
                ntpath.abspath(str(root)),
                flags=flags,
                share_mode=FILE_SHARE_READ,
            )
            attributes = adapter.get_attributes(root_handle)
            if attributes & FILE_ATTRIBUTE_REPARSE_POINT:
                raise SecureOpenError("source_link_or_reparse")
            if not attributes & FILE_ATTRIBUTE_DIRECTORY:
                raise SecureOpenError("source_root_invalid")
            if adapter.get_file_type(root_handle) != FILE_TYPE_DISK:
                raise SecureOpenError("source_root_invalid")
            canonical_root = Path(_normalize_windows_final_path(adapter.get_final_path(root_handle)))
            yield RootAnchor(
                canonical_root=canonical_root,
                platform="windows",
                identity=adapter.get_file_identity(root_handle),
            )
        except SecureOpenError:
            raise
        except OSError:
            raise SecureOpenError("source_root_invalid") from None
        finally:
            if root_handle is not None:
                adapter.close_handle(root_handle)


class OwnedArtifactFilesystem:
    """Identity-bound filesystem mutations beneath held directory anchors."""

    MOVEFILE_REPLACE_EXISTING = 0x00000001
    MOVEFILE_WRITE_THROUGH = 0x00000008

    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _DELETE = 0x00010000
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _FILE_SHARE_DELETE = 0x00000004
    _CREATE_NEW = 1
    _OPEN_EXISTING = 3
    _FILE_ATTRIBUTE_DIRECTORY = 0x10
    _FILE_ATTRIBUTE_NORMAL = 0x80
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x400
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_TYPE_DISK = 0x0001
    _FILE_DISPOSITION_INFO = 4

    def __init__(self) -> None:
        self._platform = "windows" if os.name == "nt" else "posix"
        self._kernel32 = None
        self._file_information_type = None
        self._file_disposition_type = None
        if self._platform == "windows":
            self._configure_windows()

    @contextmanager
    def anchor_directory(self, path: Path) -> Iterator[OwnedDirectoryAnchor]:
        absolute = Path(path).absolute()
        anchor: OwnedDirectoryAnchor | None = None
        try:
            anchor = self._open_directory(absolute)
            self._assert_anchor(anchor, anchor.identity)
            yield anchor
        except OwnedFilesystemError:
            raise
        except FileNotFoundError:
            raise OwnedFilesystemError("owned_root_missing") from None
        except OSError:
            raise OwnedFilesystemError("owned_operation_failed") from None
        finally:
            if anchor is not None:
                self._close_directory(anchor)

    @contextmanager
    def create_child_directory(
        self,
        parent: OwnedDirectoryAnchor,
        name: str,
        *,
        expected_parent_identity: tuple[int, int],
    ) -> Iterator[OwnedDirectoryAnchor]:
        self._validate_name(name)
        self._assert_anchor(parent, expected_parent_identity)
        child_path = parent.path / name
        try:
            child_stat = os.lstat(child_path)
        except FileNotFoundError:
            self._mkdir_child(parent, name)
        except OSError:
            raise OwnedFilesystemError("owned_identity_changed") from None
        else:
            if (
                stat.S_ISLNK(child_stat.st_mode)
                or self._is_reparse_stat(child_stat)
                or not stat.S_ISDIR(child_stat.st_mode)
            ):
                raise OwnedFilesystemError("owned_identity_changed")
        child = self._open_child_directory(parent, name)
        try:
            self._assert_anchor(parent, expected_parent_identity)
            yield child
        finally:
            self._close_directory(child)

    @contextmanager
    def create_new_child_directory(
        self,
        parent: OwnedDirectoryAnchor,
        name: str,
        *,
        expected_parent_identity: tuple[int, int],
    ) -> Iterator[OwnedDirectoryAnchor]:
        self._validate_name(name)
        self._assert_anchor(parent, expected_parent_identity)
        try:
            self._mkdir_child(parent, name)
        except FileExistsError:
            raise OwnedFilesystemError("owned_destination_exists") from None
        except OwnedFilesystemError:
            raise
        except OSError:
            raise OwnedFilesystemError("owned_operation_failed") from None
        child: OwnedDirectoryAnchor | None = None
        try:
            child = self._open_child_directory(parent, name)
            self._assert_anchor(parent, expected_parent_identity)
            yield child
        finally:
            if child is not None:
                self._close_directory(child)

    @contextmanager
    def anchor_child_directory(
        self,
        parent: OwnedDirectoryAnchor,
        name: str,
        *,
        expected_parent_identity: tuple[int, int],
    ) -> Iterator[OwnedDirectoryAnchor]:
        self._validate_name(name)
        self._assert_anchor(parent, expected_parent_identity)
        child: OwnedDirectoryAnchor | None = None
        try:
            child = self._open_child_directory(parent, name)
            self._assert_anchor(parent, expected_parent_identity)
            yield child
        except FileNotFoundError:
            raise OwnedFilesystemError("owned_not_found") from None
        except OwnedFilesystemError:
            raise
        except OSError:
            raise OwnedFilesystemError("owned_operation_failed") from None
        finally:
            if child is not None:
                self._close_directory(child)

    @contextmanager
    def create_temporary_file(
        self,
        parent: OwnedDirectoryAnchor,
        name: str,
        *,
        expected_parent_identity: tuple[int, int],
    ) -> Iterator[OwnedTemporaryFile]:
        self._validate_name(name)
        self._assert_anchor(parent, expected_parent_identity)
        descriptor: int | None = None
        try:
            if self._platform == "posix":
                flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(name, flags, 0o600, dir_fd=parent.descriptor)
            else:
                descriptor = self._create_temporary_windows(parent.path / name)
            opened = os.fstat(descriptor)
            self._validate_regular(opened)
            temporary = OwnedTemporaryFile(
                parent=parent,
                name=name,
                descriptor=descriptor,
                identity=(opened.st_dev, opened.st_ino),
            )
            self._assert_named_file(parent, name, temporary.identity)
            self._assert_anchor(parent, expected_parent_identity)
            try:
                yield temporary
            finally:
                if not temporary._released:
                    self._discard_temporary_file(temporary, expected_parent_identity)
        except OwnedFilesystemError:
            raise
        except OSError:
            raise OwnedFilesystemError("owned_operation_failed") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @contextmanager
    def open_mutation_file(
        self,
        parent: OwnedDirectoryAnchor,
        name: str,
        *,
        expected_parent_identity: tuple[int, int],
    ) -> Iterator[OwnedMutationFile]:
        """Open an existing authority file without ever creating its name."""
        self._validate_name(name)
        self._assert_anchor(parent, expected_parent_identity)
        descriptor: int | None = None
        try:
            if self._platform == "posix":
                flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(name, flags, dir_fd=parent.descriptor)
            else:
                descriptor = self._open_file_windows(parent.path / name, mutable=True, pin=True)
            opened = os.fstat(descriptor)
            self._validate_regular(opened)
            source = OwnedMutationFile(
                parent,
                name,
                descriptor,
                (opened.st_dev, opened.st_ino),
                False,
            )
            self._assert_named_file(parent, name, source.identity)
            self._assert_anchor(parent, expected_parent_identity)
            yield source
        except FileNotFoundError:
            raise OwnedFilesystemError("owned_not_found") from None
        except OwnedFilesystemError:
            raise
        except OSError:
            raise OwnedFilesystemError("owned_operation_failed") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @contextmanager
    def open_or_create_mutation_file(
        self,
        parent: OwnedDirectoryAnchor,
        name: str,
        *,
        expected_parent_identity: tuple[int, int],
    ) -> Iterator[OwnedMutationFile]:
        self._validate_name(name)
        self._assert_anchor(parent, expected_parent_identity)
        descriptor: int | None = None
        created = False
        try:
            if self._platform == "posix":
                flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                try:
                    descriptor = os.open(name, flags, dir_fd=parent.descriptor)
                except FileNotFoundError:
                    descriptor = os.open(
                        name,
                        flags | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=parent.descriptor,
                    )
                    created = True
            else:
                try:
                    descriptor = self._open_file_windows(parent.path / name, mutable=True, pin=True)
                except FileNotFoundError:
                    descriptor = self._create_mutation_windows(parent.path / name, pin=True)
                    created = True
            opened = os.fstat(descriptor)
            self._validate_regular(opened)
            source = OwnedMutationFile(parent, name, descriptor, (opened.st_dev, opened.st_ino), created)
            self._assert_named_file(parent, name, source.identity)
            self._assert_anchor(parent, expected_parent_identity)
            if created:
                if self._platform == "posix":
                    os.fsync(parent.descriptor)
                else:
                    self._flush_directory_windows(parent)
            yield source
        except OwnedFilesystemError:
            raise
        except OSError:
            raise OwnedFilesystemError("owned_operation_failed") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def verify_directory_anchor(self, anchor: OwnedDirectoryAnchor) -> None:
        """Revalidate an owned directory handle against its current name."""
        self._assert_anchor(anchor, anchor.identity)

    def verify_mutation_file(self, source: OwnedMutationFile) -> None:
        """Revalidate an open regular file, link count, parent, name, and identity."""
        self._validate_regular(os.fstat(source.descriptor), source.identity)
        self._assert_named_file(source.parent, source.name, source.identity)
        self._assert_anchor(source.parent, source.parent.identity)

    @classmethod
    def directory_quarantine_name(
        cls,
        name: str,
        identity: tuple[int, int],
    ) -> str:
        """Return the deterministic owned-directory tombstone for recovery."""
        cls._validate_name(name)
        return cls._directory_quarantine_name(name, identity)

    @contextmanager
    def open_claimed_file(
        self,
        parent: OwnedDirectoryAnchor,
        name: str,
        *,
        expected_parent_identity: tuple[int, int],
        expected_source_identity: tuple[int, int],
        expected_sha256: str,
        expected_size: int,
    ) -> Iterator[OwnedReadFile]:
        self._validate_name(name)
        self._assert_anchor(parent, expected_parent_identity)
        descriptor: int | None = None
        try:
            if self._platform == "posix":
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(name, flags, dir_fd=parent.descriptor)
            else:
                descriptor = self._open_file_windows(parent.path / name, mutable=False)
            opened = os.fstat(descriptor)
            self._validate_regular(opened, expected_source_identity)
            source = OwnedReadFile(parent, name, descriptor, expected_source_identity)
            digest, size = self.hash_open_file(source)
            if digest != expected_sha256 or size != expected_size:
                raise OwnedFilesystemError("owned_content_changed")
            self._assert_named_file(parent, name, expected_source_identity)
            self._assert_anchor(parent, expected_parent_identity)
            yield source
        except OwnedFilesystemError:
            raise
        except OSError:
            raise OwnedFilesystemError("owned_operation_failed") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @contextmanager
    def open_claimed_file_for_mutation(
        self,
        parent: OwnedDirectoryAnchor,
        name: str,
        *,
        expected_parent_identity: tuple[int, int],
        expected_source_identity: tuple[int, int],
        expected_sha256: str,
        expected_size: int,
    ) -> Iterator[OwnedMutationFile]:
        self._validate_name(name)
        self._assert_anchor(parent, expected_parent_identity)
        descriptor: int | None = None
        try:
            if self._platform == "posix":
                flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(name, flags, dir_fd=parent.descriptor)
            else:
                descriptor = self._open_file_windows(parent.path / name, mutable=True)
            opened = os.fstat(descriptor)
            self._validate_regular(opened, expected_source_identity)
            source = OwnedMutationFile(parent, name, descriptor, expected_source_identity)
            digest, size = self.hash_open_file(source)
            if digest != expected_sha256 or size != expected_size:
                raise OwnedFilesystemError("owned_content_changed")
            self._assert_named_file(parent, name, expected_source_identity)
            self._assert_anchor(parent, expected_parent_identity)
            yield source
        except OwnedFilesystemError:
            raise
        except OSError:
            raise OwnedFilesystemError("owned_operation_failed") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def hash_open_file(
        self,
        source: OwnedReadFile | OwnedMutationFile | OwnedTemporaryFile,
    ) -> tuple[str, int]:
        try:
            opened = os.fstat(source.descriptor)
            self._validate_regular(opened, source.identity)
            os.lseek(source.descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            while chunk := os.read(source.descriptor, 1024 * 1024):
                digest.update(chunk)
            os.lseek(source.descriptor, 0, os.SEEK_SET)
            return digest.hexdigest(), opened.st_size
        except OwnedFilesystemError:
            raise
        except OSError:
            raise OwnedFilesystemError("owned_operation_failed") from None

    def read_named_file(
        self,
        parent: OwnedDirectoryAnchor,
        name: str,
        *,
        expected_parent_identity: tuple[int, int],
        maximum_bytes: int,
    ) -> tuple[tuple[int, int], bytes]:
        self._validate_name(name)
        self._assert_anchor(parent, expected_parent_identity)
        descriptor: int | None = None
        try:
            if self._platform == "posix":
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(name, flags, dir_fd=parent.descriptor)
            else:
                descriptor = self._open_file_windows(parent.path / name, mutable=False)
            opened = os.fstat(descriptor)
            self._validate_regular(opened)
            identity = (opened.st_dev, opened.st_ino)
            self._assert_named_file(parent, name, identity)
            content = os.read(descriptor, maximum_bytes + 1)
            self._assert_anchor(parent, expected_parent_identity)
            return identity, content
        except OwnedFilesystemError:
            raise
        except FileNotFoundError:
            raise OwnedFilesystemError("owned_not_found") from None
        except OSError:
            raise OwnedFilesystemError("owned_operation_failed") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def replace_open_file(
        self,
        parent: OwnedDirectoryAnchor,
        source: OwnedMutationFile | OwnedTemporaryFile,
        source_name: str,
        destination_name: str,
        *,
        expected_parent_identity: tuple[int, int],
        expected_source_identity: tuple[int, int],
        expected_sha256: str,
        expected_size: int,
        replace_existing: bool,
    ) -> OwnedMutationResult:
        self._validate_name(source_name)
        self._validate_name(destination_name)
        self._assert_anchor(parent, expected_parent_identity)
        opened = os.fstat(source.descriptor)
        self._validate_regular(opened, expected_source_identity)
        self._assert_named_file(parent, source_name, expected_source_identity)
        digest, size = self.hash_open_file(source)
        if digest != expected_sha256 or size != expected_size:
            raise OwnedFilesystemError("owned_content_changed")
        self.before_mutation("replace", parent, source_name)
        self._assert_named_file(parent, source_name, expected_source_identity)
        final_digest, final_size = self.hash_open_file(source)
        if final_digest != expected_sha256 or final_size != expected_size:
            raise OwnedFilesystemError("owned_content_changed")
        if self._platform == "posix":
            if replace_existing:
                os.replace(
                    source_name,
                    destination_name,
                    src_dir_fd=parent.descriptor,
                    dst_dir_fd=parent.descriptor,
                )
            else:
                self._rename_noreplace_posix(parent, source_name, destination_name)
            os.fsync(source.descriptor)
            os.fsync(parent.descriptor)
            write_through = False
            parent_flushed = True
        else:
            flags = self.MOVEFILE_WRITE_THROUGH
            if replace_existing:
                flags |= self.MOVEFILE_REPLACE_EXISTING
            self._move_file_ex_windows(
                parent.path / source_name,
                parent.path / destination_name,
                flags,
            )
            self._flush_file_windows(source.descriptor)
            parent_flushed = self._flush_directory_windows(parent)
            write_through = True
        if isinstance(source, OwnedTemporaryFile):
            source.release()
        try:
            self._assert_named_file(parent, destination_name, expected_source_identity)
        except OwnedFilesystemError:
            self._quarantine_untrusted_name(parent, destination_name)
            raise OwnedFilesystemError("owned_identity_changed") from None
        self._assert_anchor(parent, expected_parent_identity)
        return OwnedMutationResult(
            identity=expected_source_identity,
            sha256=digest,
            size=size,
            rename_write_through=write_through,
            final_file_flushed=True,
            parent_directory_flushed=parent_flushed,
        )

    def move_claimed_file(
        self,
        parent: OwnedDirectoryAnchor,
        source_name: str,
        destination_name: str,
        *,
        expected_parent_identity: tuple[int, int],
        expected_source_identity: tuple[int, int],
        expected_sha256: str,
        expected_size: int,
        replace_existing: bool,
    ) -> OwnedMutationResult:
        with self.open_claimed_file_for_mutation(
            parent,
            source_name,
            expected_parent_identity=expected_parent_identity,
            expected_source_identity=expected_source_identity,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        ) as source:
            return self.replace_open_file(
                parent,
                source,
                source_name,
                destination_name,
                expected_parent_identity=expected_parent_identity,
                expected_source_identity=expected_source_identity,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
                replace_existing=replace_existing,
            )

    def delete_claimed_file(
        self,
        parent: OwnedDirectoryAnchor,
        source_name: str,
        *,
        expected_parent_identity: tuple[int, int],
        expected_source_identity: tuple[int, int],
        expected_sha256: str,
        expected_size: int,
        quarantine_name: str | None = None,
    ) -> None:
        quarantine_name = quarantine_name or f".owned-quarantine-{secrets.token_hex(16)}"
        with self.open_claimed_file_for_mutation(
            parent,
            source_name,
            expected_parent_identity=expected_parent_identity,
            expected_source_identity=expected_source_identity,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        ) as source:
            self.delete_open_file(
                parent,
                source,
                source_name,
                quarantine_name=quarantine_name,
                expected_parent_identity=expected_parent_identity,
                expected_source_identity=expected_source_identity,
            )

    def delete_reserved_file(
        self,
        parent: OwnedDirectoryAnchor,
        source_name: str,
        *,
        expected_parent_identity: tuple[int, int],
        expected_source_identity: tuple[int, int],
        quarantine_name: str | None = None,
    ) -> None:
        """Delete the exact occupant of a durable exclusive-name reservation."""
        self._validate_name(source_name)
        self._assert_anchor(parent, expected_parent_identity)
        descriptor: int | None = None
        try:
            if self._platform == "posix":
                flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(source_name, flags, dir_fd=parent.descriptor)
            else:
                descriptor = self._open_file_windows(parent.path / source_name, mutable=True)
            opened = os.fstat(descriptor)
            self._validate_regular(opened, expected_source_identity)
            identity = (opened.st_dev, opened.st_ino)
            self._assert_named_file(parent, source_name, identity)
            self.before_mutation("delete_reserved", parent, source_name)
            self._assert_named_file(parent, source_name, identity)
            quarantine = quarantine_name or f".owned-reserved-{secrets.token_hex(16)}"
            self._validate_name(quarantine)
            if self._platform == "posix":
                if source_name != quarantine:
                    self._rename_noreplace_posix(parent, source_name, quarantine)
                self._assert_named_file(parent, quarantine, identity)
                os.unlink(quarantine, dir_fd=parent.descriptor)
                os.fsync(parent.descriptor)
            else:
                import msvcrt

                handle = msvcrt.get_osfhandle(descriptor)
                if source_name != quarantine:
                    self._rename_handle_windows(handle, parent.path / quarantine)
                self._assert_named_file(parent, quarantine, identity)
                self._mark_delete_handle_windows(handle)
            self._assert_anchor(parent, expected_parent_identity)
        except FileNotFoundError:
            raise OwnedFilesystemError("owned_not_found") from None
        except OwnedFilesystemError:
            raise
        except OSError:
            raise OwnedFilesystemError("owned_operation_failed") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def delete_open_file(
        self,
        parent: OwnedDirectoryAnchor,
        source: OwnedMutationFile | OwnedTemporaryFile,
        source_name: str,
        *,
        quarantine_name: str | None = None,
        expected_parent_identity: tuple[int, int],
        expected_source_identity: tuple[int, int],
    ) -> None:
        quarantine_name = quarantine_name or f".owned-quarantine-{secrets.token_hex(16)}"
        digest, size = self.hash_open_file(source)
        if self._platform == "windows":
            import msvcrt

            self._assert_anchor(parent, expected_parent_identity)
            self._assert_named_file(parent, source_name, expected_source_identity)
            self.before_mutation("delete", parent, source_name)
            self._assert_named_file(parent, source_name, expected_source_identity)
            handle = msvcrt.get_osfhandle(source.descriptor)
            if source_name != quarantine_name:
                destination = parent.path / quarantine_name
                self._rename_handle_windows(handle, destination)
            self._flush_file_windows(source.descriptor)
            self._flush_directory_windows(parent)
            self._assert_named_file(parent, quarantine_name, expected_source_identity)
            self._mark_delete_windows(source.descriptor)
            if isinstance(source, OwnedTemporaryFile):
                source.release()
            return
        if source_name == quarantine_name:
            self._assert_anchor(parent, expected_parent_identity)
            opened = os.fstat(source.descriptor)
            self._validate_regular(opened, expected_source_identity)
            self._assert_named_file(parent, source_name, expected_source_identity)
            self.before_mutation("delete", parent, source_name)
            self._assert_named_file(parent, source_name, expected_source_identity)
            os.unlink(source_name, dir_fd=parent.descriptor)
            os.fsync(parent.descriptor)
            if isinstance(source, OwnedTemporaryFile):
                source.release()
            return
        self.replace_open_file(
            parent,
            source,
            source_name,
            quarantine_name,
            expected_parent_identity=expected_parent_identity,
            expected_source_identity=expected_source_identity,
            expected_sha256=digest,
            expected_size=size,
            replace_existing=False,
        )
        self._assert_named_file(parent, quarantine_name, expected_source_identity)
        if self._platform == "posix":
            os.unlink(quarantine_name, dir_fd=parent.descriptor)
            os.fsync(parent.descriptor)
        else:
            self._mark_delete_windows(source.descriptor)
        if isinstance(source, OwnedTemporaryFile):
            source.release()

    def _discard_temporary_file(
        self,
        temporary: OwnedTemporaryFile,
        expected_parent_identity: tuple[int, int],
    ) -> None:
        self._assert_anchor(temporary.parent, expected_parent_identity)
        self._assert_named_file(temporary.parent, temporary.name, temporary.identity)
        if self._platform == "windows":
            self.before_mutation("discard_temporary", temporary.parent, temporary.name)
            self._assert_named_file(temporary.parent, temporary.name, temporary.identity)
            try:
                self._flush_directory_windows(temporary.parent)
            finally:
                self._mark_delete_windows(temporary.descriptor)
            return
        os.unlink(temporary.name, dir_fd=temporary.parent.descriptor)
        os.fsync(temporary.parent.descriptor)

    def list_names(self, anchor: OwnedDirectoryAnchor) -> tuple[str, ...]:
        self._assert_anchor(anchor, anchor.identity)
        try:
            with os.scandir(anchor.descriptor if self._platform == "posix" else anchor.path) as iterator:
                return tuple(
                    sorted(
                        (entry.name for entry in iterator),
                        key=lambda value: (value.casefold(), value),
                    )
                )
        except OSError:
            raise OwnedFilesystemError("owned_operation_failed") from None

    def directory_names_with_identity(
        self,
        parent: OwnedDirectoryAnchor,
        *,
        expected_parent_identity: tuple[int, int],
        expected_child_identity: tuple[int, int],
    ) -> tuple[str, ...]:
        """Return names in one pinned parent that still identify an owned directory."""
        self._assert_anchor(parent, expected_parent_identity)
        matches: list[str] = []
        for name in self.list_names(parent):
            child: OwnedDirectoryAnchor | None = None
            try:
                child = self._open_child_directory(parent, name)
            except (FileNotFoundError, NotADirectoryError, OwnedFilesystemError, OSError):
                continue
            try:
                if child.identity == expected_child_identity:
                    matches.append(name)
            finally:
                self._close_directory(child)
        return tuple(sorted(matches, key=lambda value: (value.casefold(), value)))

    def name_exists(
        self,
        parent: OwnedDirectoryAnchor,
        name: str,
        *,
        expected_parent_identity: tuple[int, int],
    ) -> bool:
        self._validate_name(name)
        self._assert_anchor(parent, expected_parent_identity)
        return self._name_exists(parent, name)

    def name_has_identity(
        self,
        parent: OwnedDirectoryAnchor,
        name: str,
        *,
        expected_parent_identity: tuple[int, int],
        expected_source_identity: tuple[int, int],
    ) -> bool:
        self._validate_name(name)
        self._assert_anchor(parent, expected_parent_identity)
        if not self._name_exists(parent, name):
            return False
        try:
            self._assert_named_file(parent, name, expected_source_identity)
        except OwnedFilesystemError:
            return False
        return True

    def classify_named_identity(
        self,
        parent: OwnedDirectoryAnchor,
        name: str,
        *,
        expected_parent_identity: tuple[int, int],
        expected_source_identity: tuple[int, int] | None,
    ) -> Literal["absent", "owned", "foreign"]:
        """Classify one no-follow name from a single directory-entry observation."""
        self._validate_name(name)
        self._assert_anchor(parent, expected_parent_identity)
        try:
            if self._platform == "posix":
                named = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
            else:
                named = os.lstat(parent.path / name)
        except FileNotFoundError:
            return "absent"
        except PermissionError:
            raise OwnedFilesystemError("owned_operation_failed") from None
        except OSError:
            raise OwnedFilesystemError("owned_identity_changed") from None
        if expected_source_identity is None:
            return "foreign"
        try:
            self._validate_regular(named, expected_source_identity)
        except OwnedFilesystemError:
            return "foreign"
        return "owned"

    def remove_empty_directory(
        self,
        parent: OwnedDirectoryAnchor,
        name: str,
        *,
        expected_parent_identity: tuple[int, int],
        expected_child_identity: tuple[int, int],
    ) -> bool:
        self._validate_name(name)
        self._assert_anchor(parent, expected_parent_identity)
        quarantine = self._directory_quarantine_name(name, expected_child_identity)
        original_exists = self._name_exists(parent, name)
        quarantine_exists = self._name_exists(parent, quarantine)
        if original_exists and quarantine_exists:
            raise OwnedFilesystemError("owned_identity_changed")
        if not original_exists and not quarantine_exists:
            return False
        active_name = name if original_exists else quarantine
        child = (
            self._open_directory_windows(parent.path / active_name, delete_access=True)
            if self._platform == "windows"
            else self._open_child_directory(parent, active_name)
        )
        try:
            if child.identity != expected_child_identity or self.list_names(child):
                raise OwnedFilesystemError("owned_identity_changed")
            self.before_mutation("remove_directory", parent, active_name)
            try:
                if child.identity != expected_child_identity or self.list_names(child):
                    raise OwnedFilesystemError("owned_identity_changed")
            except OwnedFilesystemError:
                if self._platform == "posix":
                    self._quarantine_changed_directory_name(
                        parent,
                        active_name,
                        quarantine,
                    )
                raise OwnedFilesystemError("owned_identity_changed") from None
            self._quarantine_open_directory(parent, child, active_name, quarantine)
        finally:
            self._close_directory(child)
        return True

    def quarantine_directory_tree(
        self,
        parent: OwnedDirectoryAnchor,
        name: str,
        *,
        expected_parent_identity: tuple[int, int],
        expected_child_identity: tuple[int, int],
    ) -> str | None:
        """Move an owned directory tree to a durable, identity-bound tombstone.

        Directory removal by name cannot be made object-bound on POSIX.  Keeping
        the quarantined empty tree makes retries deterministic and prevents a
        last-moment path substitution from causing deletion of a foreign object.
        """
        self._validate_name(name)
        self._assert_anchor(parent, expected_parent_identity)
        quarantine = self._directory_quarantine_name(name, expected_child_identity)
        original_exists = self._name_exists(parent, name)
        quarantine_exists = self._name_exists(parent, quarantine)
        if original_exists and quarantine_exists:
            raise OwnedFilesystemError("owned_identity_changed")
        if not original_exists and not quarantine_exists:
            return None
        active_name = name if original_exists else quarantine
        child = (
            self._open_directory_windows(parent.path / active_name, delete_access=True)
            if self._platform == "windows"
            else self._open_child_directory(parent, active_name)
        )
        try:
            if child.identity != expected_child_identity:
                raise OwnedFilesystemError("owned_identity_changed")
            self.before_mutation("quarantine_directory", parent, active_name)
            if child.identity != expected_child_identity:
                raise OwnedFilesystemError("owned_identity_changed")
            self._quarantine_open_directory(parent, child, active_name, quarantine)
        finally:
            self._close_directory(child)
        return quarantine

    def _quarantine_open_directory(
        self,
        parent: OwnedDirectoryAnchor,
        child: OwnedDirectoryAnchor,
        active_name: str,
        quarantine_name: str,
    ) -> None:
        if active_name != quarantine_name:
            if self._platform == "posix":
                self._rename_noreplace_posix(parent, active_name, quarantine_name)
                os.fsync(parent.descriptor)
            else:
                self._rename_handle_windows(child.descriptor, parent.path / quarantine_name)
                self._flush_directory_windows(parent)
        try:
            self._assert_named_directory(parent, quarantine_name, child.identity)
        except OwnedFilesystemError:
            raise OwnedFilesystemError("owned_identity_changed") from None

    def _quarantine_changed_directory_name(
        self,
        parent: OwnedDirectoryAnchor,
        active_name: str,
        quarantine_name: str,
    ) -> None:
        """Isolate a POSIX name that stopped identifying the held directory."""
        try:
            self._rename_noreplace_posix(parent, active_name, quarantine_name)
            os.fsync(parent.descriptor)
        except (OSError, OwnedFilesystemError):
            raise OwnedFilesystemError("owned_identity_changed") from None

    def _quarantine_untrusted_name(
        self,
        parent: OwnedDirectoryAnchor,
        name: str,
    ) -> None:
        quarantine = f".owned-rejected-{secrets.token_hex(16)}"
        try:
            if self._platform == "posix":
                self._rename_noreplace_posix(parent, name, quarantine)
                os.fsync(parent.descriptor)
            else:
                self._move_file_ex_windows(
                    parent.path / name,
                    parent.path / quarantine,
                    self.MOVEFILE_WRITE_THROUGH,
                )
                self._flush_directory_windows(parent)
        except (OSError, OwnedFilesystemError):
            raise OwnedFilesystemError("owned_identity_changed") from None

    @staticmethod
    def _directory_quarantine_name(
        name: str,
        identity: tuple[int, int],
    ) -> str:
        digest = hashlib.sha256(
            f"{name}\0{identity[0]}\0{identity[1]}".encode("utf-8")
        ).hexdigest()[:32]
        return f".owned-directory-{digest}"

    def anchor_identity(self, anchor: OwnedDirectoryAnchor) -> tuple[int, int]:
        self._assert_anchor(anchor, anchor.identity)
        return anchor.identity

    def before_mutation(self, operation: str, parent: OwnedDirectoryAnchor, source_name: str) -> None:
        del operation, parent, source_name

    @staticmethod
    def _validate_name(name: str) -> None:
        if (
            not isinstance(name, str)
            or name in {"", ".", ".."}
            or "/" in name
            or "\\" in name
            or ":" in name
            or "\x00" in name
        ):
            raise OwnedFilesystemError("owned_path_invalid")

    @staticmethod
    def _is_reparse_stat(value: os.stat_result) -> bool:
        return bool(getattr(value, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT)

    @staticmethod
    def _validate_regular(value: os.stat_result, expected: tuple[int, int] | None = None) -> None:
        identity = (value.st_dev, value.st_ino)
        if (
            not stat.S_ISREG(value.st_mode)
            or value.st_nlink != 1
            or OwnedArtifactFilesystem._is_reparse_stat(value)
            or (expected is not None and identity != expected)
        ):
            raise OwnedFilesystemError("owned_identity_changed")

    def _open_directory(self, path: Path) -> OwnedDirectoryAnchor:
        if self._platform == "posix":
            flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            if not stat.S_ISDIR(opened.st_mode):
                os.close(descriptor)
                raise OwnedFilesystemError("owned_identity_changed")
            return OwnedDirectoryAnchor(
                path=path,
                platform="posix",
                identity=(opened.st_dev, opened.st_ino),
                descriptor=descriptor,
            )
        return self._open_directory_windows(path)

    def _open_child_directory(self, parent: OwnedDirectoryAnchor, name: str) -> OwnedDirectoryAnchor:
        if self._platform == "windows":
            return self._open_directory_windows(parent.path / name)
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=parent.descriptor)
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode):
            os.close(descriptor)
            raise OwnedFilesystemError("owned_identity_changed")
        return OwnedDirectoryAnchor(
            path=parent.path / name,
            platform="posix",
            identity=(opened.st_dev, opened.st_ino),
            descriptor=descriptor,
        )

    def _mkdir_child(self, parent: OwnedDirectoryAnchor, name: str) -> None:
        self._assert_anchor(parent, parent.identity)
        if self._platform == "posix":
            os.mkdir(name, mode=0o700, dir_fd=parent.descriptor)
            os.fsync(parent.descriptor)
        else:
            os.mkdir(parent.path / name, mode=0o700)
        self._assert_anchor(parent, parent.identity)

    def _assert_anchor(self, anchor: OwnedDirectoryAnchor, expected: tuple[int, int]) -> None:
        if anchor.platform != self._platform or anchor.identity != expected:
            raise OwnedFilesystemError("owned_identity_changed")
        try:
            if self._platform == "posix":
                opened = os.fstat(anchor.descriptor)
                current = (opened.st_dev, opened.st_ino)
                if not stat.S_ISDIR(opened.st_mode) or current != expected:
                    raise OwnedFilesystemError("owned_identity_changed")
                named = os.lstat(anchor.path)
                if (
                    not stat.S_ISDIR(named.st_mode)
                    or stat.S_ISLNK(named.st_mode)
                    or self._is_reparse_stat(named)
                    or (named.st_dev, named.st_ino) != expected
                ):
                    raise OwnedFilesystemError("owned_identity_changed")
                return
            current = self._identity_windows(anchor.descriptor, anchor.path)
            final_path = _normalize_windows_final_path(self._final_path_windows(anchor.descriptor))
            if current != expected or final_path != ntpath.normcase(ntpath.abspath(str(anchor.path))):
                raise OwnedFilesystemError("owned_identity_changed")
        except OwnedFilesystemError:
            raise
        except PermissionError:
            raise OwnedFilesystemError("owned_operation_failed") from None
        except OSError:
            raise OwnedFilesystemError("owned_identity_changed") from None

    def _assert_named_file(
        self,
        parent: OwnedDirectoryAnchor,
        name: str,
        expected: tuple[int, int],
    ) -> None:
        try:
            if self._platform == "posix":
                named = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
            else:
                named = os.lstat(parent.path / name)
            self._validate_regular(named, expected)
        except OwnedFilesystemError:
            raise
        except OSError:
            raise OwnedFilesystemError("owned_identity_changed") from None

    def _assert_named_directory(
        self,
        parent: OwnedDirectoryAnchor,
        name: str,
        expected: tuple[int, int],
    ) -> None:
        try:
            if self._platform == "posix":
                named = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
            else:
                named = os.lstat(parent.path / name)
            if (
                not stat.S_ISDIR(named.st_mode)
                or stat.S_ISLNK(named.st_mode)
                or self._is_reparse_stat(named)
                or (named.st_dev, named.st_ino) != expected
            ):
                raise OwnedFilesystemError("owned_identity_changed")
        except OwnedFilesystemError:
            raise
        except OSError:
            raise OwnedFilesystemError("owned_identity_changed") from None

    def _name_exists(self, parent: OwnedDirectoryAnchor, name: str) -> bool:
        try:
            if self._platform == "posix":
                os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
            else:
                os.lstat(parent.path / name)
        except FileNotFoundError:
            return False
        except PermissionError:
            raise OwnedFilesystemError("owned_operation_failed") from None
        except OSError:
            raise OwnedFilesystemError("owned_identity_changed") from None
        return True

    def _close_directory(self, anchor: OwnedDirectoryAnchor) -> None:
        if self._platform == "posix":
            os.close(anchor.descriptor)
        else:
            self._close_handle_windows(anchor.descriptor)

    def _configure_windows(self) -> None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class ByHandleFileInformation(ctypes.Structure):
            _fields_ = [
                ("dwFileAttributes", wintypes.DWORD),
                ("ftCreationTime", wintypes.FILETIME),
                ("ftLastAccessTime", wintypes.FILETIME),
                ("ftLastWriteTime", wintypes.FILETIME),
                ("dwVolumeSerialNumber", wintypes.DWORD),
                ("nFileSizeHigh", wintypes.DWORD),
                ("nFileSizeLow", wintypes.DWORD),
                ("nNumberOfLinks", wintypes.DWORD),
                ("nFileIndexHigh", wintypes.DWORD),
                ("nFileIndexLow", wintypes.DWORD),
            ]

        class FileDispositionInformation(ctypes.Structure):
            _fields_ = [("DeleteFile", wintypes.BOOL)]

        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.GetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ByHandleFileInformation),
        ]
        kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
        kernel32.GetFileType.argtypes = [wintypes.HANDLE]
        kernel32.GetFileType.restype = wintypes.DWORD
        kernel32.GetFinalPathNameByHandleW.argtypes = [
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        kernel32.GetFinalPathNameByHandleW.restype = wintypes.DWORD
        kernel32.MoveFileExW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
        ]
        kernel32.MoveFileExW.restype = wintypes.BOOL
        kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
        kernel32.FlushFileBuffers.restype = wintypes.BOOL
        kernel32.SetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.SetFileInformationByHandle.restype = wintypes.BOOL
        self._kernel32 = kernel32
        self._file_information_type = ByHandleFileInformation
        self._file_disposition_type = FileDispositionInformation

    def _open_directory_windows(self, path: Path, *, delete_access: bool = False) -> OwnedDirectoryAnchor:
        handle = self._create_file_windows(
            path,
            access=self._DELETE if delete_access else 0,
            share_mode=self._FILE_SHARE_READ | self._FILE_SHARE_WRITE,
            disposition=self._OPEN_EXISTING,
            flags=self._FILE_FLAG_BACKUP_SEMANTICS | self._FILE_FLAG_OPEN_REPARSE_POINT,
        )
        try:
            information = self._information_windows(handle)
            if (
                not information.dwFileAttributes & self._FILE_ATTRIBUTE_DIRECTORY
                or information.dwFileAttributes & self._FILE_ATTRIBUTE_REPARSE_POINT
                or self._kernel32.GetFileType(handle) != self._FILE_TYPE_DISK
            ):
                raise OwnedFilesystemError("owned_identity_changed")
            identity = self._identity_windows(handle, path)
            final_path = _normalize_windows_final_path(self._final_path_windows(handle))
            if final_path != ntpath.normcase(ntpath.abspath(str(path))):
                raise OwnedFilesystemError("owned_identity_changed")
            return OwnedDirectoryAnchor(path, "windows", identity, handle)
        except BaseException:
            self._close_handle_windows(handle)
            raise

    def _create_temporary_windows(self, path: Path) -> int:
        import msvcrt

        handle = self._create_file_windows(
            path,
            access=self._GENERIC_READ | self._GENERIC_WRITE | self._DELETE,
            share_mode=(self._FILE_SHARE_READ | self._FILE_SHARE_WRITE | self._FILE_SHARE_DELETE),
            disposition=self._CREATE_NEW,
            flags=self._FILE_ATTRIBUTE_NORMAL | self._FILE_FLAG_OPEN_REPARSE_POINT,
        )
        try:
            return msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
        except BaseException:
            self._close_handle_windows(handle)
            raise

    def _create_mutation_windows(self, path: Path, *, pin: bool) -> int:
        import msvcrt

        share_mode = self._FILE_SHARE_READ | self._FILE_SHARE_WRITE
        access = self._GENERIC_READ | self._GENERIC_WRITE
        if not pin:
            share_mode |= self._FILE_SHARE_DELETE
            access |= self._DELETE
        handle = self._create_file_windows(
            path,
            access=access,
            share_mode=share_mode,
            disposition=self._CREATE_NEW,
            flags=self._FILE_ATTRIBUTE_NORMAL | self._FILE_FLAG_OPEN_REPARSE_POINT,
        )
        try:
            return msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
        except BaseException:
            self._close_handle_windows(handle)
            raise

    def _open_file_windows(self, path: Path, *, mutable: bool, pin: bool = False) -> int:
        import msvcrt

        access = self._GENERIC_READ
        descriptor_flags = os.O_RDONLY | os.O_BINARY
        share_mode = self._FILE_SHARE_READ | self._FILE_SHARE_WRITE
        if not pin:
            share_mode |= self._FILE_SHARE_DELETE
        if mutable:
            access |= self._GENERIC_WRITE
            descriptor_flags = os.O_RDWR | os.O_BINARY
            if not pin:
                access |= self._DELETE
        handle = self._create_file_windows(
            path,
            access=access,
            share_mode=share_mode,
            disposition=self._OPEN_EXISTING,
            flags=self._FILE_ATTRIBUTE_NORMAL | self._FILE_FLAG_OPEN_REPARSE_POINT,
        )
        try:
            return msvcrt.open_osfhandle(handle, descriptor_flags)
        except BaseException:
            self._close_handle_windows(handle)
            raise

    def _create_file_windows(
        self,
        path: Path,
        *,
        access: int,
        share_mode: int,
        disposition: int,
        flags: int,
    ) -> int:
        import ctypes

        handle = self._kernel32.CreateFileW(str(path), access, share_mode, None, disposition, flags, None)
        invalid = ctypes.c_void_p(-1).value
        value = handle if isinstance(handle, int) else handle.value
        if value == invalid:
            raise ctypes.WinError(ctypes.get_last_error())
        return int(value)

    def _information_windows(self, handle: int):
        import ctypes

        information = self._file_information_type()
        if not self._kernel32.GetFileInformationByHandle(handle, ctypes.byref(information)):
            raise ctypes.WinError(ctypes.get_last_error())
        return information

    def _identity_windows(self, handle: int, path: Path) -> tuple[int, int]:
        information = self._information_windows(handle)
        file_id = (int(information.nFileIndexHigh) << 32) | int(information.nFileIndexLow)
        named = os.lstat(path)
        if named.st_ino != file_id:
            raise OwnedFilesystemError("owned_identity_changed")
        return named.st_dev, file_id

    def _final_path_windows(self, handle: int) -> str:
        import ctypes

        size = self._kernel32.GetFinalPathNameByHandleW(handle, None, 0, 0)
        if not size:
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_unicode_buffer(size + 1)
        written = self._kernel32.GetFinalPathNameByHandleW(handle, buffer, len(buffer), 0)
        if not written or written >= len(buffer):
            raise ctypes.WinError(ctypes.get_last_error())
        return buffer.value

    def _move_file_ex_windows(self, source: Path, destination: Path, flags: int) -> None:
        import ctypes

        if not self._kernel32.MoveFileExW(str(source), str(destination), flags):
            error = ctypes.get_last_error()
            if not flags & self.MOVEFILE_REPLACE_EXISTING and error in {80, 183}:
                raise OwnedFilesystemError("owned_destination_exists")
            raise ctypes.WinError(error)

    @staticmethod
    def _rename_noreplace_posix(
        parent: OwnedDirectoryAnchor,
        source_name: str,
        destination_name: str,
    ) -> None:
        import ctypes
        import sys

        library = ctypes.CDLL(None, use_errno=True)
        source = os.fsencode(source_name)
        destination = os.fsencode(destination_name)
        if sys.platform.startswith("linux") and hasattr(library, "renameat2"):
            rename = library.renameat2
            rename.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            rename.restype = ctypes.c_int
            result = rename(parent.descriptor, source, parent.descriptor, destination, 1)
        elif sys.platform == "darwin" and hasattr(library, "renameatx_np"):
            rename = library.renameatx_np
            rename.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            rename.restype = ctypes.c_int
            result = rename(parent.descriptor, source, parent.descriptor, destination, 0x00000004)
        else:
            raise OwnedFilesystemError("owned_operation_failed")
        if result == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise OwnedFilesystemError("owned_destination_exists")
        raise OSError(error, os.strerror(error))

    def _flush_file_windows(self, descriptor: int) -> None:
        import ctypes
        import msvcrt

        handle = msvcrt.get_osfhandle(descriptor)
        if not self._kernel32.FlushFileBuffers(handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def _flush_directory_windows(self, anchor: OwnedDirectoryAnchor) -> bool:
        import ctypes

        handle = self._create_file_windows(
            anchor.path,
            access=self._GENERIC_WRITE,
            share_mode=self._FILE_SHARE_READ | self._FILE_SHARE_WRITE,
            disposition=self._OPEN_EXISTING,
            flags=self._FILE_FLAG_BACKUP_SEMANTICS | self._FILE_FLAG_OPEN_REPARSE_POINT,
        )
        try:
            if self._kernel32.FlushFileBuffers(handle):
                return True
            error = ctypes.get_last_error()
            if error in {1, 6, 50}:
                return False
            raise ctypes.WinError(error)
        finally:
            self._close_handle_windows(handle)

    def _mark_delete_windows(self, descriptor: int) -> None:
        import msvcrt

        handle = msvcrt.get_osfhandle(descriptor)
        self._mark_delete_handle_windows(handle)

    def _mark_delete_handle_windows(self, handle: int) -> None:
        import ctypes

        information = self._file_disposition_type(True)
        if not self._kernel32.SetFileInformationByHandle(
            handle,
            self._FILE_DISPOSITION_INFO,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    def _rename_handle_windows(self, handle: int, destination: Path) -> None:
        import ctypes
        from ctypes import wintypes

        destination_text = str(destination)

        class FileRenameInformation(ctypes.Structure):
            _fields_ = [
                ("ReplaceIfExists", wintypes.BOOLEAN),
                ("RootDirectory", wintypes.HANDLE),
                ("FileNameLength", wintypes.DWORD),
                ("FileName", wintypes.WCHAR * (len(destination_text) + 1)),
            ]

        information = FileRenameInformation()
        information.ReplaceIfExists = False
        information.RootDirectory = None
        information.FileNameLength = len(destination_text.encode("utf-16-le"))
        information.FileName = destination_text
        if not self._kernel32.SetFileInformationByHandle(
            handle,
            3,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    def _close_handle_windows(self, handle: int) -> None:
        self._kernel32.CloseHandle(handle)
