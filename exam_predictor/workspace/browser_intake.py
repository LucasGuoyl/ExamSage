from __future__ import annotations

import contextlib
import ctypes
import errno
import ntpath
import os
import secrets
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Protocol, Sequence

from exam_predictor.workspace.filesystem import SecureFileOpener, is_reparse_point
from exam_predictor.workspace.models import ScanPolicy
from exam_predictor.workspace.policy import DEFAULT_SCAN_POLICY


@dataclass(frozen=True)
class BrowserUpload:
    relative_path: str
    size_bytes: int
    stream: BinaryIO


class BrowserIntakeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class OwnedTreeRemovalError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class OwnedTreeClaimLike(Protocol):
    relative_path: str
    device_id: str
    file_id: str


@dataclass(frozen=True)
class _PlannedUpload:
    upload: BrowserUpload
    parts: tuple[str, ...]


@dataclass(frozen=True)
class _PreparedWorkspace:
    canonical_root: Path
    workspace_root: Path
    identity: tuple[int, int]


class _SnapshotSession(Protocol):
    @contextlib.contextmanager
    def open_destination(self, parts: tuple[str, ...]) -> Iterator[BinaryIO]: ...

    def publish(self, destination_name: str) -> None: ...

    def cleanup(self) -> None: ...


class _PosixSnapshotSession:
    """Build a snapshot through directory descriptors without following links."""

    _AT_FDCWD = -100
    _RENAME_NOREPLACE = 1
    _RENAME_EXCL = 0x00000004

    def __init__(self, workspace: _PreparedWorkspace | Path) -> None:
        import fcntl

        if isinstance(workspace, Path):
            workspace_root = workspace.resolve(strict=True)
            workspace_stat = workspace_root.stat(follow_symlinks=False)
            workspace = _PreparedWorkspace(
                canonical_root=workspace_root.parent,
                workspace_root=workspace_root,
                identity=(workspace_stat.st_dev, workspace_stat.st_ino),
            )
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        self._directory_flags = directory_flags
        self._workspace_fd = os.open(workspace.workspace_root, directory_flags)
        self._lock_fd: int | None = None
        try:
            workspace_handle_stat = os.fstat(self._workspace_fd)
            if (
                workspace_handle_stat.st_dev,
                workspace_handle_stat.st_ino,
            ) != workspace.identity:
                raise BrowserIntakeError("browser_intake_workspace_invalid")
            lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            lock_flags |= getattr(os, "O_NOFOLLOW", 0)
            self._lock_fd = os.open(
                ".browser-intake.lock",
                lock_flags,
                0o600,
                dir_fd=self._workspace_fd,
            )
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            self._temporary_name, creation_identity = (
                self._create_temporary_directory()
            )
            temporary_fd = os.open(
                self._temporary_name,
                directory_flags,
                dir_fd=self._workspace_fd,
            )
            temporary_stat = os.fstat(temporary_fd)
            if (temporary_stat.st_dev, temporary_stat.st_ino) != creation_identity:
                raise BrowserIntakeError("browser_intake_write_failed")
            self._temporary_fd = temporary_fd
        except BaseException:
            if "temporary_fd" in locals():
                os.close(temporary_fd)
            if self._lock_fd is not None:
                os.close(self._lock_fd)
            os.close(self._workspace_fd)
            raise
        self._identity = temporary_stat
        self._directory_fds: dict[tuple[str, ...], int] = {}
        self._directory_identities: dict[tuple[str, ...], tuple[int, int]] = {}
        self._cleanup_identity_bound = True
        self._closed = False

    def _create_temporary_directory(self) -> tuple[str, tuple[int, int]]:
        for _ in range(100):
            name = f".browser-intake-{secrets.token_hex(16)}.tmp"
            try:
                os.mkdir(name, mode=0o700, dir_fd=self._workspace_fd)
            except FileExistsError:
                continue
            created = os.stat(
                name,
                dir_fd=self._workspace_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(created.st_mode):
                raise BrowserIntakeError("browser_intake_write_failed")
            return name, (created.st_dev, created.st_ino)
        raise BrowserIntakeError("browser_intake_write_failed")

    @contextlib.contextmanager
    def open_destination(self, parts: tuple[str, ...]) -> Iterator[BinaryIO]:
        parent_fd = os.dup(self._temporary_fd)
        file_descriptor: int | None = None
        try:
            current_key: tuple[str, ...] = ()
            for part in parts[:-1]:
                current_key += (part.casefold(),)
                retained_fd = self._directory_fds.get(current_key)
                if retained_fd is not None:
                    child_fd = os.dup(retained_fd)
                    os.close(parent_fd)
                    parent_fd = child_fd
                    continue

                candidate_fd: int | None = None
                try:
                    os.mkdir(part, mode=0o700, dir_fd=parent_fd)
                except FileExistsError:
                    self._cleanup_identity_bound = False
                    raise BrowserIntakeError("browser_intake_write_failed") from None
                try:
                    created = os.stat(
                        part,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    if not stat.S_ISDIR(created.st_mode):
                        raise BrowserIntakeError("browser_intake_write_failed")
                    creation_identity = (created.st_dev, created.st_ino)
                    candidate_fd = os.open(
                        part,
                        self._directory_flags,
                        dir_fd=parent_fd,
                    )
                    opened = os.fstat(candidate_fd)
                    if (opened.st_dev, opened.st_ino) != creation_identity:
                        raise BrowserIntakeError("browser_intake_write_failed")
                except BaseException:
                    self._cleanup_identity_bound = False
                    if candidate_fd is not None:
                        os.close(candidate_fd)
                    raise
                self._directory_fds[current_key] = candidate_fd
                self._directory_identities[current_key] = creation_identity
                child_fd = os.dup(candidate_fd)
                os.close(parent_fd)
                parent_fd = child_fd

            file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            file_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            file_descriptor = os.open(
                parts[-1],
                file_flags,
                0o600,
                dir_fd=parent_fd,
            )
            with os.fdopen(file_descriptor, "wb", closefd=True) as destination:
                file_descriptor = None
                yield destination
        except BrowserIntakeError:
            raise
        except OSError:
            raise BrowserIntakeError("browser_intake_write_failed") from None
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)
            os.close(parent_fd)

    def publish(self, destination_name: str) -> None:
        self._verify_temporary_identity()
        try:
            self._rename_no_replace(self._temporary_name, destination_name)
        except FileExistsError:
            raise BrowserIntakeError("browser_intake_exists") from None
        except OSError:
            raise BrowserIntakeError("browser_intake_write_failed") from None
        if not self._name_matches_temporary_identity(destination_name):
            isolation_name = f".examsage-unverified-{secrets.token_hex(16)}"
            with contextlib.suppress(OSError):
                self._rename_no_replace(destination_name, isolation_name)
            raise BrowserIntakeError("browser_intake_write_failed")
        self._closed = True
        for parent_fd in self._directory_fds.values():
            os.close(parent_fd)
        self._directory_fds.clear()
        os.close(self._temporary_fd)
        if self._lock_fd is not None:
            os.close(self._lock_fd)
        os.close(self._workspace_fd)

    def _rename_no_replace(self, source_name: str, destination_name: str) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        old_name = os.fsencode(source_name)
        new_name = os.fsencode(destination_name)
        if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
            rename = libc.renameat2
            rename.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            result = rename(
                self._workspace_fd,
                old_name,
                self._workspace_fd,
                new_name,
                self._RENAME_NOREPLACE,
            )
        elif sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
            rename = libc.renameatx_np
            rename.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            result = rename(
                self._workspace_fd,
                old_name,
                self._workspace_fd,
                new_name,
                self._RENAME_EXCL,
            )
        else:
            raise OSError(errno.ENOTSUP, "atomic no-replace rename unavailable")
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error_number, "snapshot already exists")
        raise OSError(error_number, "snapshot publish failed")

    def _verify_temporary_identity(self) -> None:
        try:
            current = os.stat(
                self._temporary_name,
                dir_fd=self._workspace_fd,
                follow_symlinks=False,
            )
        except OSError:
            raise BrowserIntakeError("browser_intake_write_failed") from None
        if not stat.S_ISDIR(current.st_mode) or not os.path.samestat(
            self._identity, current
        ):
            raise BrowserIntakeError("browser_intake_write_failed")

    def _name_matches_temporary_identity(self, name: str) -> bool:
        try:
            current = os.stat(
                name,
                dir_fd=self._workspace_fd,
                follow_symlinks=False,
            )
        except OSError:
            return False
        return stat.S_ISDIR(current.st_mode) and os.path.samestat(
            self._identity, current
        )

    def cleanup(self) -> None:
        if self._closed:
            return
        try:
            self._verify_temporary_identity()
            quarantine_name = f".examsage-browser-orphan-{secrets.token_hex(16)}"
            self._rename_no_replace(self._temporary_name, quarantine_name)
            if not self._name_matches_temporary_identity(quarantine_name):
                return
            if not self._cleanup_identity_bound:
                return
            if shutil.rmtree.avoids_symlink_attacks:
                with contextlib.suppress(OSError):
                    shutil.rmtree(".", dir_fd=self._temporary_fd)
                # POSIX has no portable directory-fd unlink. This exact-fd attempt
                # usually fails safely; the empty randomized orphan is then left
                # for Task 5 cleanup recovery rather than rmdir a mutable name.
                with contextlib.suppress(OSError):
                    os.rmdir(".", dir_fd=self._temporary_fd)
        except (BrowserIntakeError, OSError):
            pass
        finally:
            self._closed = True
            for parent_fd in self._directory_fds.values():
                os.close(parent_fd)
            self._directory_fds.clear()
            os.close(self._temporary_fd)
            if self._lock_fd is not None:
                os.close(self._lock_fd)
            os.close(self._workspace_fd)


class _WindowsSnapshotSession:
    """Hold no-delete-share handles so reparse swaps cannot redirect writes."""

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
    _FILE_RENAME_INFO = 3

    def __init__(self, workspace: _PreparedWorkspace | Path) -> None:
        if os.name != "nt":
            raise RuntimeError("Windows snapshot handles are unavailable")

        from ctypes import wintypes

        self._wintypes = wintypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_functions()
        if isinstance(workspace, Path):
            workspace_root = workspace.resolve(strict=True)
            workspace_stat = workspace_root.stat(follow_symlinks=False)
            workspace = _PreparedWorkspace(
                canonical_root=workspace_root.parent,
                workspace_root=workspace_root,
                identity=(workspace_stat.st_dev, workspace_stat.st_ino),
            )
        self._workspace_root = workspace.workspace_root
        self._workspace_handle = self._open_directory(
            self._workspace_root,
            delete_access=False,
            containment_root=workspace.canonical_root,
        )
        root_handle: int | None = None
        try:
            if self._handle_identity(self._workspace_handle)[1] != workspace.identity[1]:
                raise BrowserIntakeError("browser_intake_workspace_invalid")
            (
                self._temporary_name,
                self._temporary_root,
                creation_file_id,
            ) = (
                self._create_temporary_directory()
            )
            root_handle = self._open_directory(
                self._temporary_root,
                delete_access=True,
                containment_root=self._temporary_root,
            )
            root_identity = self._handle_identity(root_handle)
            if root_identity[1] != creation_file_id:
                raise BrowserIntakeError("browser_intake_write_failed")
        except BaseException:
            if root_handle is not None:
                self._close_handle(root_handle)
            self._close_handle(self._workspace_handle)
            raise
        self._directory_handles: dict[tuple[str, ...], int] = {(): root_handle}
        self._directory_paths: dict[tuple[str, ...], Path] = {(): self._temporary_root}
        self._directory_identities: dict[tuple[str, ...], tuple[int, int]] = {
            (): root_identity
        }
        self._file_descriptors: list[int] = []
        self._file_records: list[tuple[Path, tuple[int, int]]] = []
        self._cleanup_identity_bound = True
        self._closed = False

    def _configure_functions(self) -> None:
        wintypes = self._wintypes

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
        self._kernel32.GetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ByHandleFileInformation),
        ]
        self._kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
        self._kernel32.GetFileType.argtypes = [wintypes.HANDLE]
        self._kernel32.GetFileType.restype = wintypes.DWORD
        self._kernel32.GetFinalPathNameByHandleW.argtypes = [
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        self._kernel32.GetFinalPathNameByHandleW.restype = wintypes.DWORD
        self._kernel32.SetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        self._kernel32.SetFileInformationByHandle.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL

    def _create_temporary_directory(self) -> tuple[str, Path, int]:
        for _ in range(100):
            name = f".browser-intake-{secrets.token_hex(16)}.tmp"
            path = self._workspace_root / name
            try:
                path.mkdir()
            except FileExistsError:
                continue
            created = path.stat(follow_symlinks=False)
            return name, path, created.st_ino
        raise BrowserIntakeError("browser_intake_write_failed")

    def _open_directory(
        self,
        path: Path,
        *,
        delete_access: bool,
        containment_root: Path,
    ) -> int:
        access = self._GENERIC_READ | (self._DELETE if delete_access else 0)
        handle = self._create_file_handle(
            path,
            access=access,
            share_mode=self._FILE_SHARE_READ | self._FILE_SHARE_WRITE,
            disposition=self._OPEN_EXISTING,
            flags=self._FILE_FLAG_BACKUP_SEMANTICS
            | self._FILE_FLAG_OPEN_REPARSE_POINT,
        )
        try:
            attributes = self._attributes(handle)
            if (
                not attributes & self._FILE_ATTRIBUTE_DIRECTORY
                or attributes & self._FILE_ATTRIBUTE_REPARSE_POINT
                or self._kernel32.GetFileType(handle) != self._FILE_TYPE_DISK
                or not self._final_path_beneath(handle, containment_root)
            ):
                raise BrowserIntakeError("browser_intake_workspace_invalid")
            return handle
        except BaseException:
            self._close_handle(handle)
            raise

    def _create_file_handle(
        self,
        path: Path,
        *,
        access: int,
        share_mode: int,
        disposition: int,
        flags: int,
    ) -> int:
        handle = self._kernel32.CreateFileW(
            str(path),
            access,
            share_mode,
            None,
            disposition,
            flags,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        value = handle if isinstance(handle, int) else handle.value
        if value == invalid:
            raise ctypes.WinError(ctypes.get_last_error())
        return int(value)

    def _attributes(self, handle: int) -> int:
        return int(self._information(handle).dwFileAttributes)

    def _information(self, handle: int):
        information = self._file_information_type()
        if not self._kernel32.GetFileInformationByHandle(
            handle, ctypes.byref(information)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return information

    def _handle_identity(self, handle: int) -> tuple[int, int]:
        information = self._information(handle)
        file_index = (int(information.nFileIndexHigh) << 32) | int(
            information.nFileIndexLow
        )
        return int(information.dwVolumeSerialNumber), file_index

    def _final_path(self, handle: int) -> str:
        required = self._kernel32.GetFinalPathNameByHandleW(handle, None, 0, 0)
        if not required:
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_unicode_buffer(required + 1)
        written = self._kernel32.GetFinalPathNameByHandleW(
            handle, buffer, len(buffer), 0
        )
        if not written or written >= len(buffer):
            raise ctypes.WinError(ctypes.get_last_error())
        path = buffer.value
        if path.startswith("\\\\?\\UNC\\"):
            path = "\\\\" + path[8:]
        elif path.startswith("\\\\?\\"):
            path = path[4:]
        return ntpath.normcase(ntpath.abspath(path))

    def _final_path_beneath(self, handle: int, root: Path) -> bool:
        final_path = self._final_path(handle)
        normalized_root = ntpath.normcase(ntpath.abspath(str(root)))
        try:
            return ntpath.commonpath((final_path, normalized_root)) == normalized_root
        except ValueError:
            return False

    def _ensure_parent(self, parts: tuple[str, ...]) -> Path:
        current_key: tuple[str, ...] = ()
        current_path = self._temporary_root
        for part in parts:
            current_key += (part.casefold(),)
            current_path /= part
            if current_key in self._directory_handles:
                current_path = self._directory_paths[current_key]
                continue
            try:
                current_path.mkdir()
            except FileExistsError:
                self._cleanup_identity_bound = False
                raise BrowserIntakeError("browser_intake_write_failed") from None
            handle: int | None = None
            try:
                created = current_path.stat(follow_symlinks=False)
                creation_file_id = created.st_ino
                handle = self._open_directory(
                    current_path,
                    delete_access=True,
                    containment_root=self._temporary_root,
                )
                identity = self._handle_identity(handle)
                if identity[1] != creation_file_id:
                    raise BrowserIntakeError("browser_intake_write_failed")
            except BaseException:
                self._cleanup_identity_bound = False
                if handle is not None:
                    self._close_handle(handle)
                raise
            self._directory_handles[current_key] = handle
            self._directory_paths[current_key] = current_path
            self._directory_identities[current_key] = identity
        return current_path

    @contextlib.contextmanager
    def open_destination(self, parts: tuple[str, ...]) -> Iterator[BinaryIO]:
        import msvcrt

        parent = self._ensure_parent(parts[:-1])
        target = parent / parts[-1]
        handle: int | None = None
        try:
            handle = self._create_file_handle(
                target,
                access=self._GENERIC_READ | self._GENERIC_WRITE | self._DELETE,
                share_mode=0,
                disposition=self._CREATE_NEW,
                flags=self._FILE_ATTRIBUTE_NORMAL
                | self._FILE_FLAG_OPEN_REPARSE_POINT,
            )
            attributes = self._attributes(handle)
            if (
                attributes & (
                    self._FILE_ATTRIBUTE_DIRECTORY
                    | self._FILE_ATTRIBUTE_REPARSE_POINT
                )
                or self._kernel32.GetFileType(handle) != self._FILE_TYPE_DISK
                or not self._final_path_beneath(handle, self._temporary_root)
            ):
                self._close_handle(handle)
                handle = None
                raise BrowserIntakeError("browser_intake_path_invalid")
            identity = self._handle_identity(handle)
            file_descriptor = msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)
            handle = None
            self._file_descriptors.append(file_descriptor)
            self._file_records.append((target, identity))
            with os.fdopen(file_descriptor, "wb", closefd=False) as destination:
                yield destination
        except BrowserIntakeError:
            raise
        except OSError:
            raise BrowserIntakeError("browser_intake_write_failed") from None
        finally:
            if handle is not None:
                self._close_handle(handle)

    def publish(self, destination_name: str) -> None:
        self._close_children_for_publish()
        destination = self._workspace_root / destination_name
        try:
            self._rename_handle(self._directory_handles[()], destination)
        except OSError:
            self._reopen_children_for_cleanup()
            if os.path.lexists(destination):
                raise BrowserIntakeError("browser_intake_exists") from None
            raise BrowserIntakeError("browser_intake_write_failed") from None
        self._closed = True
        self._close_handle(self._directory_handles.pop(()))
        self._close_handle(self._workspace_handle)

    def _rename_handle(self, handle: int, destination: Path) -> None:
        wintypes = self._wintypes
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
            self._FILE_RENAME_INFO,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    def _close_children_for_publish(self) -> None:
        for file_descriptor in self._file_descriptors:
            os.close(file_descriptor)
        self._file_descriptors.clear()
        for key in sorted(
            (key for key in self._directory_handles if key),
            key=len,
            reverse=True,
        ):
            self._close_handle(self._directory_handles.pop(key))

    def _reopen_children_for_cleanup(self) -> None:
        import msvcrt

        try:
            for key in sorted(
                (key for key in self._directory_paths if key), key=len
            ):
                candidate_handle = self._open_directory(
                    self._directory_paths[key],
                    delete_access=True,
                    containment_root=self._temporary_root,
                )
                try:
                    candidate_identity = self._handle_identity(candidate_handle)
                except BaseException:
                    self._close_handle(candidate_handle)
                    raise
                if candidate_identity != self._directory_identities[key]:
                    self._close_handle(candidate_handle)
                    raise OSError(
                        errno.ESTALE, "snapshot directory identity changed"
                    )
                self._directory_handles[key] = candidate_handle
            for path, expected_identity in self._file_records:
                handle = self._create_file_handle(
                    path,
                    access=self._GENERIC_READ | self._DELETE,
                    share_mode=0,
                    disposition=self._OPEN_EXISTING,
                    flags=self._FILE_ATTRIBUTE_NORMAL
                    | self._FILE_FLAG_OPEN_REPARSE_POINT,
                )
                try:
                    attributes = self._attributes(handle)
                    candidate_identity = self._handle_identity(handle)
                    candidate_is_safe = not (
                        attributes
                        & (
                            self._FILE_ATTRIBUTE_DIRECTORY
                            | self._FILE_ATTRIBUTE_REPARSE_POINT
                        )
                    ) and self._final_path_beneath(
                        handle, self._temporary_root
                    )
                except BaseException:
                    self._close_handle(handle)
                    raise
                if not candidate_is_safe or candidate_identity != expected_identity:
                    self._close_handle(handle)
                    raise OSError(
                        errno.ELOOP, "snapshot child identity changed"
                    )
                try:
                    file_descriptor = msvcrt.open_osfhandle(
                        handle, os.O_RDONLY | os.O_BINARY
                    )
                except BaseException:
                    self._close_handle(handle)
                    raise
                self._file_descriptors.append(file_descriptor)
        except OSError:
            return

    def cleanup(self) -> None:
        if self._closed:
            return
        if not self._cleanup_identity_bound:
            for file_descriptor in self._file_descriptors:
                os.close(file_descriptor)
            self._file_descriptors.clear()
            for handle in self._directory_handles.values():
                self._close_handle(handle)
            self._directory_handles.clear()
            self._close_handle(self._workspace_handle)
            self._closed = True
            return
        for file_descriptor in self._file_descriptors:
            self._mark_delete(file_descriptor)
            os.close(file_descriptor)
        self._file_descriptors.clear()
        for key in sorted(self._directory_handles, key=len, reverse=True):
            handle = self._directory_handles[key]
            self._mark_delete(handle)
            self._close_handle(handle)
        self._directory_handles.clear()
        self._close_handle(self._workspace_handle)
        self._closed = True

    def _mark_delete(self, handle_or_descriptor: int) -> None:
        import msvcrt

        handle = handle_or_descriptor
        if handle_or_descriptor in self._file_descriptors:
            handle = msvcrt.get_osfhandle(handle_or_descriptor)

        class FileDispositionInformation(ctypes.Structure):
            _fields_ = [("DeleteFile", self._wintypes.BOOL)]

        information = FileDispositionInformation(True)
        self._kernel32.SetFileInformationByHandle(
            handle,
            self._FILE_DISPOSITION_INFO,
            ctypes.byref(information),
            ctypes.sizeof(information),
        )

    def _close_handle(self, handle: int) -> None:
        self._kernel32.CloseHandle(handle)


def _start_snapshot_session(workspace: _PreparedWorkspace) -> _SnapshotSession:
    if os.name == "nt":
        return _WindowsSnapshotSession(workspace)
    return _PosixSnapshotSession(workspace)


class BrowserIntakeWriter:
    def __init__(
        self,
        workspaces_root: Path,
        policy: ScanPolicy = DEFAULT_SCAN_POLICY,
    ) -> None:
        self._workspaces_root = workspaces_root
        self._policy = policy

    def create_snapshot(
        self, workspace_id: str, files: Sequence[BrowserUpload]
    ) -> Path:
        """Create and atomically publish one validated browser-intake snapshot."""
        self._validate_workspace_id(workspace_id)
        plan = self._validate_plan(files)
        prepared_workspace = self._prepare_workspace(workspace_id)
        workspace_root = prepared_workspace.workspace_root
        published_root = workspace_root / "browser-intake"
        if os.path.lexists(published_root):
            raise BrowserIntakeError("browser_intake_exists")

        try:
            session = _start_snapshot_session(prepared_workspace)
        except BrowserIntakeError:
            raise
        except OSError:
            raise BrowserIntakeError("browser_intake_write_failed") from None
        aggregate = 0
        try:
            for planned in plan:
                with session.open_destination(planned.parts) as destination:
                    aggregate = self._write_upload(
                        destination,
                        planned.upload,
                        aggregate,
                    )
            session.publish("browser-intake")
        except BrowserIntakeError:
            session.cleanup()
            raise
        except Exception:
            session.cleanup()
            raise BrowserIntakeError("browser_intake_write_failed") from None
        return published_root

    def _validate_plan(
        self, files: Sequence[BrowserUpload]
    ) -> tuple[_PlannedUpload, ...]:
        if self._policy.hash_chunk_bytes <= 0:
            raise BrowserIntakeError("browser_intake_policy_invalid")
        if len(files) > self._policy.max_files:
            raise BrowserIntakeError("browser_intake_file_count_limit")

        planned: list[_PlannedUpload] = []
        folded_files: set[str] = set()
        folded_parents: set[str] = set()
        aggregate = 0
        for upload in files:
            parts = self._validate_relative_path(upload.relative_path)
            if (
                not isinstance(upload.size_bytes, int)
                or isinstance(upload.size_bytes, bool)
                or upload.size_bytes < 0
            ):
                raise BrowserIntakeError("browser_intake_size_invalid")
            if not callable(getattr(upload.stream, "read", None)):
                raise BrowserIntakeError("browser_intake_stream_invalid")

            folded = "/".join(parts).casefold()
            folded_parent_paths = {
                "/".join(parts[:index]).casefold()
                for index in range(1, len(parts))
            }
            if (
                folded in folded_files
                or folded in folded_parents
                or folded_parent_paths.intersection(folded_files)
            ):
                raise BrowserIntakeError("browser_intake_path_conflict")
            folded_files.add(folded)
            folded_parents.update(folded_parent_paths)

            aggregate += upload.size_bytes
            if aggregate > self._policy.max_workspace_bytes:
                raise BrowserIntakeError("browser_intake_size_limit")
            planned.append(_PlannedUpload(upload=upload, parts=parts))
        return tuple(planned)

    def _validate_relative_path(self, relative_path: str) -> tuple[str, ...]:
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or len(relative_path) > self._policy.max_path_chars
            or "\\" in relative_path
            or "\x00" in relative_path
            or relative_path.startswith("/")
        ):
            raise BrowserIntakeError("browser_intake_path_invalid")
        parts = tuple(relative_path.split("/"))
        if (
            len(parts) > self._policy.max_depth
            or any(part in {"", ".", ".."} or ":" in part for part in parts)
        ):
            raise BrowserIntakeError("browser_intake_path_invalid")
        return parts

    @staticmethod
    def _validate_workspace_id(workspace_id: str) -> None:
        if (
            not isinstance(workspace_id, str)
            or not workspace_id
            or workspace_id in {".", ".."}
            or "/" in workspace_id
            or "\\" in workspace_id
            or "\x00" in workspace_id
            or ":" in workspace_id
        ):
            raise BrowserIntakeError("browser_intake_workspace_invalid")

    def _prepare_workspace(self, workspace_id: str) -> _PreparedWorkspace:
        try:
            self._workspaces_root.mkdir(parents=True, exist_ok=True)
            if self._is_link_or_reparse(self._workspaces_root):
                raise BrowserIntakeError("browser_intake_workspace_invalid")
            canonical_root = self._workspaces_root.resolve(strict=True)
            workspace_root = canonical_root / workspace_id
            if os.path.lexists(workspace_root):
                if self._is_link_or_reparse(workspace_root) or not workspace_root.is_dir():
                    raise BrowserIntakeError("browser_intake_workspace_invalid")
            else:
                workspace_root.mkdir()
            canonical_workspace = workspace_root.resolve(strict=True)
            if canonical_workspace.parent != canonical_root:
                raise BrowserIntakeError("browser_intake_workspace_invalid")
            workspace_stat = canonical_workspace.stat(follow_symlinks=False)
            return _PreparedWorkspace(
                canonical_root=canonical_root,
                workspace_root=canonical_workspace,
                identity=(workspace_stat.st_dev, workspace_stat.st_ino),
            )
        except BrowserIntakeError:
            raise
        except OSError:
            raise BrowserIntakeError("browser_intake_workspace_invalid") from None

    def _write_upload(
        self,
        destination: BinaryIO,
        upload: BrowserUpload,
        aggregate: int,
    ) -> int:
        written = 0
        try:
            while True:
                chunk = upload.stream.read(self._policy.hash_chunk_bytes)
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise BrowserIntakeError("browser_intake_stream_invalid")
                if not chunk:
                    break
                written += len(chunk)
                aggregate += len(chunk)
                if (
                    written > upload.size_bytes
                    or aggregate > self._policy.max_workspace_bytes
                ):
                    raise BrowserIntakeError("browser_intake_size_limit")
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        except BrowserIntakeError:
            raise
        except Exception:
            raise BrowserIntakeError("browser_intake_write_failed") from None
        if written != upload.size_bytes:
            raise BrowserIntakeError("browser_intake_size_mismatch")
        return aggregate

    @staticmethod
    def _is_link_or_reparse(path: Path) -> bool:
        return path.is_symlink() or is_reparse_point(path)


class OwnedTreeRemover:
    """Remove one identity-bound tree below the ExamSage-owned data root."""

    def __init__(self, data_root: Path) -> None:
        self._data_root = Path(data_root).absolute()

    def __call__(self, claim: OwnedTreeClaimLike) -> None:
        parts = self._validated_parts(claim.relative_path)
        try:
            if os.name == "nt":
                _WindowsOwnedTreeRemover(self._data_root).remove(parts, claim)
            else:
                self._remove_posix(parts, claim)
        except OwnedTreeRemovalError:
            raise
        except OSError:
            raise OwnedTreeRemovalError("cleanup_failed") from None

    @staticmethod
    def _validated_parts(relative_path: str) -> tuple[str, ...]:
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or "\\" in relative_path
            or "\x00" in relative_path
            or relative_path.startswith("/")
        ):
            raise OwnedTreeRemovalError("cleanup_path_invalid")
        parts = tuple(relative_path.split("/"))
        if (
            len(parts) < 2
            or parts[0] != "workspaces"
            or any(part in {"", ".", ".."} or ":" in part for part in parts)
        ):
            raise OwnedTreeRemovalError("cleanup_path_invalid")
        return parts

    def _remove_posix(
        self,
        parts: tuple[str, ...],
        claim: OwnedTreeClaimLike,
    ) -> None:
        opener = SecureFileOpener(platform="posix")
        with contextlib.ExitStack() as stack:
            root_anchor = stack.enter_context(opener.anchor_root(self._data_root))
            if root_anchor.directory_fd is None:
                raise OwnedTreeRemovalError("cleanup_identity_unavailable")
            directory_fds = [root_anchor.directory_fd]
            for part in parts:
                directory_fds.append(
                    stack.enter_context(
                        opener.anchor_child_directory(directory_fds[-1], part)
                    )
                )
            target_fd = directory_fds[-1]
            target_stat = os.fstat(target_fd)
            if (
                str(target_stat.st_dev) != claim.device_id
                or str(target_stat.st_ino) != claim.file_id
            ):
                raise OwnedTreeRemovalError("cleanup_identity_changed")
            self._remove_posix_contents(target_fd, opener)
            current = os.stat(
                parts[-1],
                dir_fd=directory_fds[-2],
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(current.st_mode) or not os.path.samestat(
                target_stat, current
            ):
                raise OwnedTreeRemovalError("cleanup_identity_changed")
            os.rmdir(parts[-1], dir_fd=directory_fds[-2])

    def _remove_posix_contents(
        self,
        directory_fd: int,
        opener: SecureFileOpener,
    ) -> None:
        with os.scandir(directory_fd) as iterator:
            names = sorted((entry.name for entry in iterator), key=str.casefold)
        for name in names:
            try:
                named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError:
                raise OwnedTreeRemovalError("cleanup_identity_changed") from None
            if stat.S_ISLNK(named.st_mode) or bool(
                getattr(named, "st_file_attributes", 0) & 0x400
            ):
                raise OwnedTreeRemovalError("cleanup_link_or_reparse")
            if stat.S_ISDIR(named.st_mode):
                with opener.anchor_child_directory(directory_fd, name) as child_fd:
                    opened = os.fstat(child_fd)
                    if not os.path.samestat(named, opened):
                        raise OwnedTreeRemovalError("cleanup_identity_changed")
                    self._remove_posix_contents(child_fd, opener)
                current = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if not stat.S_ISDIR(current.st_mode) or not os.path.samestat(
                    named, current
                ):
                    raise OwnedTreeRemovalError("cleanup_identity_changed")
                os.rmdir(name, dir_fd=directory_fd)
                continue
            if not stat.S_ISREG(named.st_mode):
                raise OwnedTreeRemovalError("cleanup_special_file")
            descriptor: int | None = None
            try:
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(name, flags, dir_fd=directory_fd)
                opened = os.fstat(descriptor)
                current = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or not os.path.samestat(named, opened)
                    or not os.path.samestat(opened, current)
                ):
                    raise OwnedTreeRemovalError("cleanup_identity_changed")
                os.unlink(name, dir_fd=directory_fd)
            finally:
                if descriptor is not None:
                    os.close(descriptor)


class _WindowsOwnedTreeRemover(_WindowsSnapshotSession):
    """Reuse intake's no-reparse handles for identity-bound Windows deletion."""

    def __init__(self, data_root: Path) -> None:
        if os.name != "nt":
            raise RuntimeError("Windows owned-tree handles are unavailable")
        from ctypes import wintypes

        self._wintypes = wintypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_functions()
        self._data_root = data_root
        self._claimed_root = data_root

    def remove(
        self,
        parts: tuple[str, ...],
        claim: OwnedTreeClaimLike,
    ) -> None:
        handles: list[int] = []
        try:
            data_handle = self._open_directory(
                self._data_root,
                delete_access=False,
                containment_root=self._data_root,
            )
            handles.append(data_handle)
            canonical_data_root = Path(self._final_path(data_handle))
            current = canonical_data_root
            for index, part in enumerate(parts):
                current /= part
                handle = self._open_directory(
                    current,
                    delete_access=index == len(parts) - 1,
                    containment_root=canonical_data_root,
                )
                handles.append(handle)
            target_handle = handles[-1]
            target_identity = self._handle_identity(target_handle)
            if (
                str(target_identity[0]) != claim.device_id
                or str(target_identity[1]) != claim.file_id
            ):
                raise OwnedTreeRemovalError("cleanup_identity_changed")
            expected = ntpath.normcase(ntpath.abspath(str(current)))
            if self._final_path(target_handle) != expected:
                raise OwnedTreeRemovalError("cleanup_path_invalid")
            self._claimed_root = current
            self._remove_open_directory(current, target_handle)
            self._mark_delete_checked(target_handle)
        except BrowserIntakeError:
            raise OwnedTreeRemovalError("cleanup_link_or_reparse") from None
        finally:
            for handle in reversed(handles):
                self._close_handle(handle)

    def _remove_open_directory(self, path: Path, directory_handle: int) -> None:
        if not self._final_path_beneath(directory_handle, self._claimed_root):
            raise OwnedTreeRemovalError("cleanup_path_invalid")
        try:
            with os.scandir(path) as iterator:
                entries = sorted(
                    tuple(iterator),
                    key=lambda entry: (entry.name.casefold(), entry.name),
                )
        except OSError:
            raise OwnedTreeRemovalError("cleanup_failed") from None
        for entry in entries:
            child = path / entry.name
            try:
                named = child.stat(follow_symlinks=False)
            except OSError:
                raise OwnedTreeRemovalError("cleanup_identity_changed") from None
            attributes = getattr(named, "st_file_attributes", 0)
            if child.is_symlink() or attributes & self._FILE_ATTRIBUTE_REPARSE_POINT:
                raise OwnedTreeRemovalError("cleanup_link_or_reparse")
            if stat.S_ISDIR(named.st_mode):
                child_handle = self._open_directory(
                    child,
                    delete_access=True,
                    containment_root=self._claimed_root,
                )
                try:
                    if self._handle_identity(child_handle)[1] != named.st_ino:
                        raise OwnedTreeRemovalError("cleanup_identity_changed")
                    self._remove_open_directory(child, child_handle)
                    self._mark_delete_checked(child_handle)
                finally:
                    self._close_handle(child_handle)
                continue
            if not stat.S_ISREG(named.st_mode):
                raise OwnedTreeRemovalError("cleanup_special_file")
            handle = self._create_file_handle(
                child,
                access=self._GENERIC_READ | self._DELETE,
                share_mode=self._FILE_SHARE_READ | self._FILE_SHARE_WRITE,
                disposition=self._OPEN_EXISTING,
                flags=self._FILE_ATTRIBUTE_NORMAL | self._FILE_FLAG_OPEN_REPARSE_POINT,
            )
            try:
                handle_attributes = self._attributes(handle)
                if (
                    handle_attributes
                    & (
                        self._FILE_ATTRIBUTE_DIRECTORY
                        | self._FILE_ATTRIBUTE_REPARSE_POINT
                    )
                    or self._kernel32.GetFileType(handle) != self._FILE_TYPE_DISK
                    or not self._final_path_beneath(handle, self._claimed_root)
                    or self._handle_identity(handle)[1] != named.st_ino
                ):
                    raise OwnedTreeRemovalError("cleanup_identity_changed")
                self._mark_delete_checked(handle)
            finally:
                self._close_handle(handle)

    def _mark_delete_checked(self, handle: int) -> None:
        class FileDispositionInformation(ctypes.Structure):
            _fields_ = [("DeleteFile", self._wintypes.BOOL)]

        information = FileDispositionInformation(True)
        if not self._kernel32.SetFileInformationByHandle(
            handle,
            self._FILE_DISPOSITION_INFO,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
