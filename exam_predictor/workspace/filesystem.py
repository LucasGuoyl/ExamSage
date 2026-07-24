from __future__ import annotations

import errno
import ntpath
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator, Protocol


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
        or any(
            part in {"", "."} or "\\" in part or "\x00" in part or ":" in part
            for part in parts
        )
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
        if not self._kernel32.GetFileInformationByHandle(
            handle, self._ctypes.byref(information)
        ):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        file_index = (int(information.nFileIndexHigh) << 32) | int(
            information.nFileIndexLow
        )
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
    def _anchor_posix_directory(
        self, canonical_root: Path, parts: tuple[str, ...]
    ) -> Iterator[int]:
        directory_fds: list[int] = []
        try:
            directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
            directory_flags |= getattr(os, "O_NOFOLLOW", 0)
            directory_fds.append(os.open(canonical_root, directory_flags))
            for part in parts:
                directory_fds.append(
                    os.open(part, directory_flags, dir_fd=directory_fds[-1])
                )
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
            directory_fds.append(
                os.open(canonical_root, root_flags)
                if root_fd is None
                else os.dup(root_fd)
            )
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
    def _anchor_windows_directory(
        self, canonical_root: Path, parts: tuple[str, ...]
    ) -> Iterator[None]:
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
            canonical_root = Path(
                _normalize_windows_final_path(adapter.get_final_path(root_handle))
            )
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
