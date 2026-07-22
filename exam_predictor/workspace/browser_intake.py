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

from exam_predictor.workspace.filesystem import is_reparse_point
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


@dataclass(frozen=True)
class _PlannedUpload:
    upload: BrowserUpload
    parts: tuple[str, ...]


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

    def __init__(self, workspace_root: Path) -> None:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        self._directory_flags = directory_flags
        self._workspace_fd = os.open(workspace_root, directory_flags)
        self._temporary_name = self._create_temporary_directory()
        try:
            self._temporary_fd = os.open(
                self._temporary_name,
                directory_flags,
                dir_fd=self._workspace_fd,
            )
        except BaseException:
            os.rmdir(self._temporary_name, dir_fd=self._workspace_fd)
            os.close(self._workspace_fd)
            raise
        self._identity = os.fstat(self._temporary_fd)
        self._closed = False

    def _create_temporary_directory(self) -> str:
        for _ in range(100):
            name = f".browser-intake-{secrets.token_hex(16)}.tmp"
            try:
                os.mkdir(name, mode=0o700, dir_fd=self._workspace_fd)
            except FileExistsError:
                continue
            return name
        raise BrowserIntakeError("browser_intake_write_failed")

    @contextlib.contextmanager
    def open_destination(self, parts: tuple[str, ...]) -> Iterator[BinaryIO]:
        parent_fd = os.dup(self._temporary_fd)
        file_descriptor: int | None = None
        try:
            for part in parts[:-1]:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=parent_fd)
                except FileExistsError:
                    pass
                child_fd = os.open(part, self._directory_flags, dir_fd=parent_fd)
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
            self._rename_no_replace(destination_name)
        except FileExistsError:
            raise BrowserIntakeError("browser_intake_exists") from None
        except OSError:
            raise BrowserIntakeError("browser_intake_write_failed") from None
        self._closed = True
        os.close(self._temporary_fd)
        os.close(self._workspace_fd)

    def _rename_no_replace(self, destination_name: str) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        old_name = os.fsencode(self._temporary_name)
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

    def cleanup(self) -> None:
        if self._closed:
            return
        try:
            self._verify_temporary_identity()
            if shutil.rmtree.avoids_symlink_attacks:
                shutil.rmtree(self._temporary_name, dir_fd=self._workspace_fd)
        except (BrowserIntakeError, OSError):
            pass
        finally:
            self._closed = True
            os.close(self._temporary_fd)
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

    def __init__(self, workspace_root: Path) -> None:
        if os.name != "nt":
            raise RuntimeError("Windows snapshot handles are unavailable")

        from ctypes import wintypes

        self._wintypes = wintypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_functions()
        self._workspace_root = workspace_root.resolve(strict=True)
        self._workspace_handle = self._open_directory(
            self._workspace_root,
            delete_access=False,
            containment_root=self._workspace_root,
        )
        self._temporary_name, self._temporary_root = self._create_temporary_directory()
        try:
            root_handle = self._open_directory(
                self._temporary_root,
                delete_access=True,
                containment_root=self._temporary_root,
            )
        except BaseException:
            self._temporary_root.rmdir()
            self._close_handle(self._workspace_handle)
            raise
        self._directory_handles: dict[tuple[str, ...], int] = {(): root_handle}
        self._directory_paths: dict[tuple[str, ...], Path] = {(): self._temporary_root}
        self._directory_identities: dict[tuple[str, ...], tuple[int, int]] = {
            (): self._handle_identity(root_handle)
        }
        self._file_descriptors: list[int] = []
        self._file_records: list[tuple[Path, tuple[int, int]]] = []
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

    def _create_temporary_directory(self) -> tuple[str, Path]:
        for _ in range(100):
            name = f".browser-intake-{secrets.token_hex(16)}.tmp"
            path = self._workspace_root / name
            try:
                path.mkdir()
            except FileExistsError:
                continue
            return name, path
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
                pass
            handle = self._open_directory(
                current_path,
                delete_access=True,
                containment_root=self._temporary_root,
            )
            self._directory_handles[current_key] = handle
            self._directory_paths[current_key] = current_path
            self._directory_identities[current_key] = self._handle_identity(handle)
        return current_path

    @contextlib.contextmanager
    def open_destination(self, parts: tuple[str, ...]) -> Iterator[BinaryIO]:
        import msvcrt

        parent = self._ensure_parent(parts[:-1])
        target = parent / parts[-1]
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
                raise BrowserIntakeError("browser_intake_path_invalid")
            file_descriptor = msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)
            self._file_descriptors.append(file_descriptor)
            self._file_records.append((target, self._handle_identity(handle)))
            with os.fdopen(file_descriptor, "wb", closefd=False) as destination:
                yield destination
        except BrowserIntakeError:
            raise
        except OSError:
            raise BrowserIntakeError("browser_intake_write_failed") from None

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
                self._directory_handles[key] = self._open_directory(
                    self._directory_paths[key],
                    delete_access=True,
                    containment_root=self._temporary_root,
                )
                if self._handle_identity(self._directory_handles[key]) != (
                    self._directory_identities[key]
                ):
                    raise OSError(errno.ESTALE, "snapshot directory identity changed")
            for path, expected_identity in self._file_records:
                handle = self._create_file_handle(
                    path,
                    access=self._GENERIC_READ | self._DELETE,
                    share_mode=0,
                    disposition=self._OPEN_EXISTING,
                    flags=self._FILE_ATTRIBUTE_NORMAL
                    | self._FILE_FLAG_OPEN_REPARSE_POINT,
                )
                attributes = self._attributes(handle)
                if (
                    attributes
                    & (
                        self._FILE_ATTRIBUTE_DIRECTORY
                        | self._FILE_ATTRIBUTE_REPARSE_POINT
                    )
                    or not self._final_path_beneath(handle, self._temporary_root)
                    or self._handle_identity(handle) != expected_identity
                ):
                    self._close_handle(handle)
                    raise OSError(errno.ELOOP, "snapshot child identity changed")
                self._file_descriptors.append(
                    msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
                )
        except OSError:
            return

    def cleanup(self) -> None:
        if self._closed:
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


def _start_snapshot_session(workspace_root: Path) -> _SnapshotSession:
    if os.name == "nt":
        return _WindowsSnapshotSession(workspace_root)
    return _PosixSnapshotSession(workspace_root)


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
        workspace_root = self._prepare_workspace(workspace_id)
        published_root = workspace_root / "browser-intake"
        if os.path.lexists(published_root):
            raise BrowserIntakeError("browser_intake_exists")

        try:
            session = _start_snapshot_session(workspace_root)
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

    def _prepare_workspace(self, workspace_id: str) -> Path:
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
            return canonical_workspace
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
