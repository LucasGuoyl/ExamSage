from __future__ import annotations

import contextlib
import ctypes
import errno
import hashlib
import json
import math
import ntpath
import os
import re
import secrets
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator, Protocol

from exam_predictor.workspace.browser_intake import (
    BrowserIntakeError,
    OwnedTreeRemovalError,
    _WindowsOwnedTreeRemover,
)
from exam_predictor.workspace.filesystem import (
    SecureFileOpener,
    SecureOpenError,
    is_reparse_point,
)


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JSON_TYPES = frozenset({"units", "snapshots"})
_REPARSE_ATTRIBUTE = 0x400
_OWNERSHIP_NAME = ".artifact-ownership.json"
_OWNERSHIP_VERSION = 1
_MAX_MARKER_BYTES = 1024 * 1024
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 100_000
_MAX_JSON_STRING_BYTES = 1024 * 1024


def _windows_final_path(kernel32: Any, handle: int) -> str:
    size = kernel32.GetFinalPathNameByHandleW(handle, None, 0, 0)
    if size == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_unicode_buffer(size + 1)
    written = kernel32.GetFinalPathNameByHandleW(handle, buffer, len(buffer), 0)
    if written == 0 or written >= len(buffer):
        raise ctypes.WinError(ctypes.get_last_error())
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return ntpath.normcase(ntpath.abspath(value))


def _windows_path_beneath(candidate: str, root: str) -> bool:
    try:
        return ntpath.commonpath((candidate, root)) == root
    except ValueError:
        return False


def _hash_windows_handle(kernel32: Any, handle: int) -> str:
    from ctypes import wintypes

    kernel32.SetFilePointerEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_longlong),
        wintypes.DWORD,
    ]
    kernel32.SetFilePointerEx.restype = wintypes.BOOL
    kernel32.ReadFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    kernel32.ReadFile.restype = wintypes.BOOL
    if not kernel32.SetFilePointerEx(handle, 0, None, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    digest = hashlib.sha256()
    buffer = ctypes.create_string_buffer(1024 * 1024)
    while True:
        read = wintypes.DWORD()
        if not kernel32.ReadFile(handle, buffer, len(buffer), ctypes.byref(read), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if read.value == 0:
            break
        digest.update(buffer.raw[: read.value])
    if not kernel32.SetFilePointerEx(handle, 0, None, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    return digest.hexdigest()


class ArtifactBoundaryError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ArtifactCleanupState(StrEnum):
    DELETED = "deleted"
    CLEANUP_PENDING = "cleanup_pending"


@dataclass(frozen=True)
class _OwnedArtifactClaim:
    relative_path: str
    device_id: str
    file_id: str


class ArtifactFilesystemOps(Protocol):
    def atomic_replace(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        directory_fd: int | None,
        expected_identity: Identity,
        expected_parent_identity: Identity,
        expected_sha256: str,
    ) -> tuple[Identity, str] | None: ...

    def remove_owned_tree(
        self,
        data_root: Path,
        claim: _OwnedArtifactClaim,
    ) -> None: ...

    def remove_file(
        self,
        parent: Path,
        name: str,
        *,
        directory_fd: int | None,
        expected_identity: Identity,
    ) -> None: ...


class NativeArtifactFilesystemOps:
    def durability_event(self, event: str) -> None:
        del event

    def atomic_replace(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        directory_fd: int | None,
        expected_identity: Identity,
        expected_parent_identity: Identity,
        expected_sha256: str,
    ) -> tuple[Identity, str] | None:
        if os.name == "nt":
            return self._atomic_replace_windows(
                Path(source),
                Path(destination),
                expected_identity=expected_identity,
                expected_parent_identity=expected_parent_identity,
                expected_sha256=expected_sha256,
            )
        parent_identity = (
            self._path_directory_identity(Path(source).parent)
            if directory_fd is None
            else self._fd_identity(directory_fd)
        )
        if parent_identity != expected_parent_identity:
            raise ArtifactBoundaryError("artifact_identity_changed")
        descriptor = (
            os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            if directory_fd is None
            else os.open(
                source,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        )
        try:
            opened = os.fstat(descriptor)
            if (
                (opened.st_dev, opened.st_ino) != expected_identity
                or opened.st_nlink != 1
                or self._hash_descriptor(descriptor) != expected_sha256
            ):
                raise ArtifactBoundaryError("artifact_hash_mismatch")
        finally:
            os.close(descriptor)
        if directory_fd is None:
            os.replace(source, destination)
        else:
            os.replace(
                source,
                destination,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        published = (
            os.open(destination, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            if directory_fd is None
            else os.open(
                destination,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        )
        try:
            value = os.fstat(published)
            identity = (value.st_dev, value.st_ino)
            return identity, self._hash_descriptor(published)
        finally:
            os.close(published)

    def remove_owned_tree(
        self,
        data_root: Path,
        claim: _OwnedArtifactClaim,
    ) -> None:
        parts = tuple(claim.relative_path.split("/"))
        if len(parts) != 3 or parts[0] != "workspaces" or parts[2] != "evidence":
            raise OwnedTreeRemovalError("cleanup_path_invalid")
        if os.name == "nt":
            _WindowsEvidenceTreeRemover(data_root, self.before_remove).remove(parts, claim)
            return
        self._remove_owned_tree_posix(data_root, parts, claim)

    def remove_file(
        self,
        parent: Path,
        name: str,
        *,
        directory_fd: int | None,
        expected_identity: Identity,
    ) -> None:
        if os.name == "nt":
            remover = _WindowsOwnedTreeRemover(parent)
            parent_handle = remover._open_directory(parent, delete_access=False, containment_root=parent)
            handle: int | None = None
            try:
                path = parent / name
                handle = remover._create_file_handle(
                    path,
                    access=remover._GENERIC_READ | remover._DELETE,
                    share_mode=remover._FILE_SHARE_READ | remover._FILE_SHARE_WRITE,
                    disposition=remover._OPEN_EXISTING,
                    flags=remover._FILE_ATTRIBUTE_NORMAL | remover._FILE_FLAG_OPEN_REPARSE_POINT,
                )
                information = remover._information(handle)
                file_id = (int(information.nFileIndexHigh) << 32) | int(information.nFileIndexLow)
                named = path.stat(follow_symlinks=False)
                if (
                    file_id != expected_identity[1]
                    or named.st_ino != expected_identity[1]
                    or information.nNumberOfLinks != 1
                    or named.st_nlink != 1
                    or not remover._final_path_beneath(handle, parent)
                ):
                    raise ArtifactBoundaryError("artifact_identity_changed")
                remover._mark_delete_checked(handle)
            finally:
                if handle is not None:
                    remover._close_handle(handle)
                remover._close_handle(parent_handle)
            return
        descriptor = (
            os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
            if directory_fd is not None
            else os.open(parent / name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        )
        try:
            opened = os.fstat(descriptor)
            current = (
                os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if directory_fd is not None
                else (parent / name).stat(follow_symlinks=False)
            )
            if (
                (opened.st_dev, opened.st_ino) != expected_identity
                or opened.st_nlink != 1
                or current.st_nlink != 1
                or not os.path.samestat(opened, current)
            ):
                raise ArtifactBoundaryError("artifact_identity_changed")
            if directory_fd is not None:
                os.unlink(name, dir_fd=directory_fd)
            else:
                (parent / name).unlink()
        finally:
            os.close(descriptor)

    @staticmethod
    def _hash_descriptor(descriptor: int) -> str:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return digest.hexdigest()

    @staticmethod
    def _fd_identity(descriptor: int) -> Identity:
        value = os.fstat(descriptor)
        return value.st_dev, value.st_ino

    @staticmethod
    def _path_directory_identity(path: Path) -> Identity:
        value = path.stat(follow_symlinks=False)
        return value.st_dev, value.st_ino

    def before_remove(
        self,
        data_root: Path,
        claim: _OwnedArtifactClaim,
    ) -> None:
        del data_root, claim

    def _remove_owned_tree_posix(
        self,
        data_root: Path,
        parts: tuple[str, ...],
        claim: _OwnedArtifactClaim,
    ) -> None:
        opener = SecureFileOpener(platform="posix")
        with contextlib.ExitStack() as stack:
            root = stack.enter_context(opener.anchor_root(data_root))
            if root.directory_fd is None:
                raise OwnedTreeRemovalError("cleanup_identity_unavailable")
            workspaces_fd = stack.enter_context(
                opener.anchor_child_directory(root.directory_fd, "workspaces")
            )
            workspace_fd = stack.enter_context(opener.anchor_child_directory(workspaces_fd, parts[1]))
            evidence_fd = stack.enter_context(opener.anchor_child_directory(workspace_fd, "evidence"))
            evidence_stat = os.fstat(evidence_fd)
            if str(evidence_stat.st_dev) != claim.device_id or str(evidence_stat.st_ino) != claim.file_id:
                raise OwnedTreeRemovalError("cleanup_identity_changed")

            root_names = self._scandir_names(evidence_fd)
            allowed = {_OWNERSHIP_NAME, *_JSON_TYPES, "parts"}
            if not set(root_names).issubset(allowed):
                raise OwnedTreeRemovalError("cleanup_unknown_entry")

            directory_records: list[tuple[str, int, os.stat_result]] = []
            file_records: list[tuple[int, str, int, os.stat_result]] = []
            for directory_name in ("parts", "units", "snapshots"):
                if directory_name not in root_names:
                    continue
                named_directory = os.stat(
                    directory_name,
                    dir_fd=evidence_fd,
                    follow_symlinks=False,
                )
                if not stat.S_ISDIR(named_directory.st_mode):
                    raise OwnedTreeRemovalError("cleanup_identity_changed")
                directory_fd = stack.enter_context(opener.anchor_child_directory(evidence_fd, directory_name))
                opened_directory = os.fstat(directory_fd)
                if not os.path.samestat(named_directory, opened_directory):
                    raise OwnedTreeRemovalError("cleanup_identity_changed")
                directory_records.append((directory_name, directory_fd, opened_directory))
                for filename in self._scandir_names(directory_fd):
                    file_records.append(
                        (
                            directory_fd,
                            filename,
                            self._open_bound_regular_posix(directory_fd, filename),
                            os.stat(
                                filename,
                                dir_fd=directory_fd,
                                follow_symlinks=False,
                            ),
                        )
                    )
                    stack.callback(os.close, file_records[-1][2])

            marker_record: tuple[int, str, int, os.stat_result] | None = None
            if _OWNERSHIP_NAME in root_names:
                marker_fd = self._open_bound_regular_posix(evidence_fd, _OWNERSHIP_NAME)
                stack.callback(os.close, marker_fd)
                marker_record = (
                    evidence_fd,
                    _OWNERSHIP_NAME,
                    marker_fd,
                    os.stat(
                        _OWNERSHIP_NAME,
                        dir_fd=evidence_fd,
                        follow_symlinks=False,
                    ),
                )
            else:
                raise OwnedTreeRemovalError("cleanup_identity_changed")

            self.before_remove(data_root, claim)
            if self._scandir_names(evidence_fd) != root_names:
                raise OwnedTreeRemovalError("cleanup_identity_changed")
            for name, directory_fd, opened in directory_records:
                current = os.stat(name, dir_fd=evidence_fd, follow_symlinks=False)
                if not os.path.samestat(opened, current) or self._scandir_names(directory_fd) != tuple(
                    record[1] for record in file_records if record[0] == directory_fd
                ):
                    raise OwnedTreeRemovalError("cleanup_identity_changed")
            for parent_fd, name, descriptor, named in file_records + [marker_record]:
                opened = os.fstat(descriptor)
                current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if (
                    opened.st_nlink != 1
                    or current.st_nlink != 1
                    or not os.path.samestat(named, opened)
                    or not os.path.samestat(opened, current)
                ):
                    raise OwnedTreeRemovalError("cleanup_identity_changed")

            for parent_fd, name, _descriptor, _named in file_records:
                os.unlink(name, dir_fd=parent_fd)
            os.unlink(_OWNERSHIP_NAME, dir_fd=evidence_fd)
            for name, directory_fd, opened in directory_records:
                if self._scandir_names(directory_fd):
                    raise OwnedTreeRemovalError("cleanup_identity_changed")
                current = os.stat(name, dir_fd=evidence_fd, follow_symlinks=False)
                if not os.path.samestat(opened, current):
                    raise OwnedTreeRemovalError("cleanup_identity_changed")
                os.rmdir(name, dir_fd=evidence_fd)
            if self._scandir_names(evidence_fd):
                raise OwnedTreeRemovalError("cleanup_identity_changed")
            current_evidence = os.stat("evidence", dir_fd=workspace_fd, follow_symlinks=False)
            if not os.path.samestat(evidence_stat, current_evidence):
                raise OwnedTreeRemovalError("cleanup_identity_changed")
            os.rmdir("evidence", dir_fd=workspace_fd)

    @staticmethod
    def _scandir_names(directory_fd: int) -> tuple[str, ...]:
        with os.scandir(directory_fd) as iterator:
            return tuple(
                sorted(
                    (entry.name for entry in iterator),
                    key=lambda value: (value.casefold(), value),
                )
            )

    @staticmethod
    def _open_bound_regular_posix(parent_fd: int, name: str) -> int:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or named.st_nlink != 1
            or not os.path.samestat(opened, named)
        ):
            os.close(descriptor)
            raise OwnedTreeRemovalError("cleanup_identity_changed")
        return descriptor

    def _atomic_replace_windows(
        self,
        source: Path,
        destination: Path,
        *,
        expected_identity: Identity,
        expected_parent_identity: Identity,
        expected_sha256: str,
    ) -> tuple[Identity, str]:
        from ctypes import wintypes

        generic_read = 0x80000000
        generic_write = 0x40000000
        delete = 0x00010000
        open_existing = 3
        file_attribute_directory = 0x10
        file_attribute_normal = 0x80
        file_attribute_reparse_point = 0x400
        file_flag_open_reparse_point = 0x00200000
        file_flag_backup_semantics = 0x02000000
        file_flag_write_through = 0x80000000
        file_share_read = 0x00000001
        file_share_write = 0x00000002
        file_share_delete = 0x00000004
        file_type_disk = 0x0001
        file_rename_info = 3

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

        destination_text = str(destination)

        class FileRenameInformation(ctypes.Structure):
            _fields_ = [
                ("ReplaceIfExists", wintypes.BOOLEAN),
                ("RootDirectory", wintypes.HANDLE),
                ("FileNameLength", wintypes.DWORD),
                ("FileName", wintypes.WCHAR * (len(destination_text) + 1)),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
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
        kernel32.SetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.SetFileInformationByHandle.restype = wintypes.BOOL
        kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
        kernel32.FlushFileBuffers.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        invalid_handle = ctypes.c_void_p(-1).value
        named_parent = source.parent.stat(follow_symlinks=False)
        parent_handle = kernel32.CreateFileW(
            str(source.parent),
            generic_read,
            file_share_read | file_share_write | file_share_delete,
            None,
            open_existing,
            file_flag_backup_semantics | file_flag_open_reparse_point,
            None,
        )
        if parent_handle == invalid_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        handle: int | None = None
        try:
            parent_information = ByHandleFileInformation()
            if not kernel32.GetFileInformationByHandle(parent_handle, ctypes.byref(parent_information)):
                raise ctypes.WinError(ctypes.get_last_error())
            parent_file_id = (int(parent_information.nFileIndexHigh) << 32) | int(
                parent_information.nFileIndexLow
            )
            parent_identity = (named_parent.st_dev, named_parent.st_ino)
            final_parent = _windows_final_path(kernel32, parent_handle)
            if (
                not parent_information.dwFileAttributes & file_attribute_directory
                or parent_information.dwFileAttributes & file_attribute_reparse_point
                or kernel32.GetFileType(parent_handle) != file_type_disk
                or parent_file_id != named_parent.st_ino
                or parent_identity != expected_parent_identity
                or final_parent != ntpath.normcase(ntpath.abspath(str(source.parent)))
            ):
                raise ArtifactBoundaryError("artifact_identity_changed")

            named = source.stat(follow_symlinks=False)
            handle = kernel32.CreateFileW(
                str(source),
                generic_read | generic_write | delete,
                0,
                None,
                open_existing,
                file_attribute_normal | file_flag_open_reparse_point | file_flag_write_through,
                None,
            )
            if handle == invalid_handle:
                handle = None
                raise ctypes.WinError(ctypes.get_last_error())
            information = ByHandleFileInformation()
            if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(information)):
                raise ctypes.WinError(ctypes.get_last_error())
            handle_file_id = (int(information.nFileIndexHigh) << 32) | int(information.nFileIndexLow)
            named_identity = (named.st_dev, named.st_ino)
            if (
                information.dwFileAttributes & (file_attribute_directory | file_attribute_reparse_point)
                or information.nNumberOfLinks != 1
                or kernel32.GetFileType(handle) != file_type_disk
                or handle_file_id != named.st_ino
                or named_identity != expected_identity
            ):
                raise ArtifactBoundaryError("artifact_identity_changed")
            if _hash_windows_handle(kernel32, handle) != expected_sha256:
                raise ArtifactBoundaryError("artifact_hash_mismatch")
            final_path = _windows_final_path(kernel32, handle)
            if not _windows_path_beneath(final_path, final_parent):
                raise ArtifactBoundaryError("artifact_identity_changed")

            rename = FileRenameInformation()
            rename.ReplaceIfExists = True
            rename.RootDirectory = None
            rename.FileNameLength = len(destination_text.encode("utf-16-le"))
            rename.FileName = destination_text
            if not kernel32.SetFileInformationByHandle(
                handle,
                file_rename_info,
                ctypes.byref(rename),
                ctypes.sizeof(rename),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            self.durability_event("rename_write_through")
            if not kernel32.FlushFileBuffers(handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self.durability_event("final_file_flushed")
            if kernel32.FlushFileBuffers(parent_handle):
                self.durability_event("parent_directory_flushed")
            else:
                flush_error = ctypes.get_last_error()
                if flush_error not in {1, 5, 6}:
                    raise ctypes.WinError(flush_error)
                self.durability_event("parent_flush_unavailable")
            return named_identity, _hash_windows_handle(kernel32, handle)
        finally:
            if handle is not None:
                kernel32.CloseHandle(handle)
            kernel32.CloseHandle(parent_handle)


class _WindowsEvidenceTreeRemover(_WindowsOwnedTreeRemover):
    def __init__(self, data_root: Path, before_remove: Any) -> None:
        super().__init__(data_root)
        self._before_remove = before_remove

    def remove(
        self,
        parts: tuple[str, ...],
        claim: _OwnedArtifactClaim,
    ) -> None:
        directory_handles: list[int] = []
        file_handles: list[int] = []
        try:
            data_handle = self._open_directory(
                self._data_root,
                delete_access=False,
                containment_root=self._data_root,
            )
            directory_handles.append(data_handle)
            canonical_root = Path(self._final_path(data_handle))
            current = canonical_root
            for index, part in enumerate(parts):
                current /= part
                handle = self._open_directory(
                    current,
                    delete_access=index == len(parts) - 1,
                    containment_root=canonical_root,
                )
                directory_handles.append(handle)
            evidence_handle = directory_handles[-1]
            evidence_identity = self._handle_identity(evidence_handle)
            if str(evidence_identity[0]) != claim.device_id or str(evidence_identity[1]) != claim.file_id:
                raise OwnedTreeRemovalError("cleanup_identity_changed")
            evidence_path = current
            self._claimed_root = evidence_path
            root_names = self._path_names(evidence_path)
            allowed = {_OWNERSHIP_NAME, "parts", "units", "snapshots"}
            if not set(root_names).issubset(allowed):
                raise OwnedTreeRemovalError("cleanup_unknown_entry")

            collection_records: list[tuple[str, Path, int, int]] = []
            file_records: list[tuple[Path, str, int, int]] = []
            for collection_name in ("parts", "units", "snapshots"):
                if collection_name not in root_names:
                    continue
                collection_path = evidence_path / collection_name
                named = collection_path.stat(follow_symlinks=False)
                collection_handle = self._open_directory(
                    collection_path,
                    delete_access=True,
                    containment_root=evidence_path,
                )
                directory_handles.append(collection_handle)
                file_id = self._handle_identity(collection_handle)[1]
                if file_id != named.st_ino:
                    raise OwnedTreeRemovalError("cleanup_identity_changed")
                collection_records.append(
                    (
                        collection_name,
                        collection_path,
                        collection_handle,
                        file_id,
                    )
                )
                for filename in self._path_names(collection_path):
                    file_handle, child_file_id = self._open_bound_file(
                        collection_path / filename, evidence_path
                    )
                    file_handles.append(file_handle)
                    file_records.append(
                        (
                            collection_path,
                            filename,
                            file_handle,
                            child_file_id,
                        )
                    )

            if _OWNERSHIP_NAME not in root_names:
                raise OwnedTreeRemovalError("cleanup_identity_changed")
            marker_handle, marker_file_id = self._open_bound_file(
                evidence_path / _OWNERSHIP_NAME, evidence_path
            )
            file_handles.append(marker_handle)
            marker_record = (
                evidence_path,
                _OWNERSHIP_NAME,
                marker_handle,
                marker_file_id,
            )

            self._before_remove(self._data_root, claim)
            if self._path_names(evidence_path) != root_names:
                raise OwnedTreeRemovalError("cleanup_identity_changed")
            for name, path, handle, file_id in collection_records:
                if (
                    path.stat(follow_symlinks=False).st_ino != file_id
                    or self._path_names(path)
                    != tuple(record[1] for record in file_records if record[0] == path)
                    or self._handle_identity(handle)[1] != file_id
                ):
                    raise OwnedTreeRemovalError("cleanup_identity_changed")
            for parent, name, handle, file_id in file_records + [marker_record]:
                named = (parent / name).stat(follow_symlinks=False)
                information = self._information(handle)
                current_file_id = (int(information.nFileIndexHigh) << 32) | int(information.nFileIndexLow)
                if (
                    named.st_ino != file_id
                    or current_file_id != file_id
                    or named.st_nlink != 1
                    or information.nNumberOfLinks != 1
                ):
                    raise OwnedTreeRemovalError("cleanup_identity_changed")

            for _parent, _name, handle, _file_id in file_records + [marker_record]:
                self._mark_delete_checked(handle)
                self._close_handle(handle)
                file_handles.remove(handle)
            for _name, _path, handle, _file_id in reversed(collection_records):
                self._mark_delete_checked(handle)
                self._close_handle(handle)
                directory_handles.remove(handle)
            self._mark_delete_checked(evidence_handle)
        except BrowserIntakeError:
            raise OwnedTreeRemovalError("cleanup_identity_changed") from None
        finally:
            for handle in reversed(file_handles):
                self._close_handle(handle)
            for handle in reversed(directory_handles):
                self._close_handle(handle)

    @staticmethod
    def _path_names(path: Path) -> tuple[str, ...]:
        try:
            with os.scandir(path) as iterator:
                return tuple(
                    sorted(
                        (entry.name for entry in iterator),
                        key=lambda value: (value.casefold(), value),
                    )
                )
        except OSError:
            raise OwnedTreeRemovalError("cleanup_failed") from None

    def _open_bound_file(self, path: Path, root: Path) -> tuple[int, int]:
        named = path.stat(follow_symlinks=False)
        handle = self._create_file_handle(
            path,
            access=self._GENERIC_READ | self._DELETE,
            share_mode=self._FILE_SHARE_READ | self._FILE_SHARE_WRITE,
            disposition=self._OPEN_EXISTING,
            flags=self._FILE_ATTRIBUTE_NORMAL | self._FILE_FLAG_OPEN_REPARSE_POINT,
        )
        try:
            information = self._information(handle)
            file_id = (int(information.nFileIndexHigh) << 32) | int(information.nFileIndexLow)
            if (
                information.dwFileAttributes
                & (self._FILE_ATTRIBUTE_DIRECTORY | self._FILE_ATTRIBUTE_REPARSE_POINT)
                or information.nNumberOfLinks != 1
                or named.st_nlink != 1
                or file_id != named.st_ino
                or self._kernel32.GetFileType(handle) != self._FILE_TYPE_DISK
                or not self._final_path_beneath(handle, root)
            ):
                raise OwnedTreeRemovalError("cleanup_identity_changed")
            return handle, file_id
        except BaseException:
            self._close_handle(handle)
            raise


Identity = tuple[int, int]


class EvidenceArtifactStore:
    """Publish and remove only identity-bound ExamSage evidence artifacts."""

    def __init__(
        self,
        root: Path,
        *,
        filesystem_ops: ArtifactFilesystemOps | None = None,
    ) -> None:
        requested_root = Path(root).absolute()
        self._validate_ancestor_chain(requested_root)
        self._reject_link_or_reparse(requested_root, allow_missing=True)
        try:
            requested_root.mkdir(parents=True, exist_ok=True)
            self._reject_link_or_reparse(requested_root)
            self._root = requested_root.resolve(strict=True)
            self._root_identity = self._directory_identity(self._root)
            workspaces = self._root / "workspaces"
            self._create_directory(workspaces)
            self._workspaces_identity = self._directory_identity(workspaces)
        except ArtifactBoundaryError:
            raise
        except OSError:
            raise ArtifactBoundaryError("artifact_root_invalid") from None
        self._native_filesystem_ops = NativeArtifactFilesystemOps()
        self._filesystem_ops = filesystem_ops or self._native_filesystem_ops
        self._directory_identities: dict[tuple[str, str], Identity] = {}
        self._file_identities: dict[tuple[str, str, str], Identity] = {}
        self._ownership: dict[str, dict[str, Any]] = {}

    def publish_part(
        self,
        workspace_id: str,
        part_id: str,
        content: bytes | bytearray | memoryview,
        *,
        expected_sha256: str,
    ) -> str:
        self._validate_identifier(workspace_id)
        self._validate_identifier(part_id)
        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise ArtifactBoundaryError("artifact_content_invalid")
        return self._publish_bytes(
            workspace_id,
            "parts",
            part_id,
            bytes(content),
            expected_sha256,
            suffix="",
        )

    @contextmanager
    def open_part(self, workspace_id: str, part_id: str) -> Iterator[BinaryIO]:
        self._validate_identifier(workspace_id)
        self._validate_identifier(part_id)
        relative = self._artifact_relative_path(
            workspace_id,
            "parts",
            part_id,
            suffix="",
        )
        self._prepare_collection(workspace_id, "parts", create=False)
        expected, expected_digest = self._artifact_claim(workspace_id, "parts", part_id)
        opener = SecureFileOpener()
        try:
            with opener.open_regular(self._root, relative) as source:
                opened = os.fstat(source.fileno())
                self._validate_regular_identity(opened, expected=expected)
                named = self._artifact_path(relative).stat(follow_symlinks=False)
                self._validate_regular_identity(named, expected=(opened.st_dev, opened.st_ino))
                self._validate_hierarchy(workspace_id, "parts")
                digest = hashlib.sha256()
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                if digest.hexdigest() != expected_digest:
                    raise ArtifactBoundaryError("artifact_identity_changed")
                source.seek(0)
                self._file_identities[(workspace_id, "parts", part_id)] = (
                    opened.st_dev,
                    opened.st_ino,
                )
                yield source
                self._validate_hierarchy(workspace_id, "parts")
        except ArtifactBoundaryError:
            raise
        except SecureOpenError as error:
            if error.code in {
                "source_link_or_reparse",
                "source_outside_root",
                "source_not_regular",
            }:
                raise ArtifactBoundaryError("artifact_identity_changed") from None
            raise ArtifactBoundaryError("artifact_open_failed") from None
        except OSError:
            raise ArtifactBoundaryError("artifact_identity_changed") from None

    def publish_json(
        self,
        workspace_id: str,
        artifact_type: str,
        artifact_id: str,
        document: Any,
        *,
        expected_sha256: str,
    ) -> str:
        self._validate_identifier(workspace_id)
        self._validate_json_type(artifact_type)
        self._validate_identifier(artifact_id)
        self._validate_json_object(document)
        try:
            encoded = json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            raise ArtifactBoundaryError("artifact_json_invalid") from None
        if len(encoded) > _MAX_JSON_BYTES:
            raise ArtifactBoundaryError("artifact_json_too_large")
        return self._publish_bytes(
            workspace_id,
            artifact_type,
            artifact_id,
            encoded,
            expected_sha256,
            suffix=".json",
        )

    def read_json(
        self,
        workspace_id: str,
        artifact_type: str,
        artifact_id: str,
    ) -> Any:
        self._validate_identifier(workspace_id)
        self._validate_json_type(artifact_type)
        self._validate_identifier(artifact_id)
        relative = self._artifact_relative_path(
            workspace_id,
            artifact_type,
            artifact_id,
            suffix=".json",
        )
        self._prepare_collection(workspace_id, artifact_type, create=False)
        filename = f"{artifact_id}.json"
        expected, expected_digest = self._artifact_claim(workspace_id, artifact_type, filename)
        opener = SecureFileOpener()
        try:
            with opener.open_regular(self._root, relative) as source:
                opened = os.fstat(source.fileno())
                self._validate_regular_identity(opened, expected=expected)
                named = self._artifact_path(relative).stat(follow_symlinks=False)
                self._validate_regular_identity(named, expected=(opened.st_dev, opened.st_ino))
                self._validate_hierarchy(workspace_id, artifact_type)
                content = source.read(_MAX_JSON_BYTES + 1)
                if len(content) > _MAX_JSON_BYTES:
                    raise ArtifactBoundaryError("artifact_json_too_large")
                self._validate_hierarchy(workspace_id, artifact_type)
                if hashlib.sha256(content).hexdigest() != expected_digest:
                    raise ArtifactBoundaryError("artifact_identity_changed")
            result = json.loads(content.decode("utf-8"))
            self._validate_json_object(result)
        except ArtifactBoundaryError:
            raise
        except SecureOpenError as error:
            if error.code in {
                "source_link_or_reparse",
                "source_outside_root",
                "source_not_regular",
            }:
                raise ArtifactBoundaryError("artifact_identity_changed") from None
            raise ArtifactBoundaryError("artifact_open_failed") from None
        except (OSError, UnicodeError, ValueError, RecursionError):
            raise ArtifactBoundaryError("artifact_json_invalid") from None
        self._file_identities[(workspace_id, artifact_type, artifact_id)] = (
            opened.st_dev,
            opened.st_ino,
        )
        return result

    @staticmethod
    def _validate_json_object(document: Any) -> None:
        if not isinstance(document, dict):
            raise ArtifactBoundaryError("artifact_json_invalid")
        stack: list[tuple[Any, int]] = [(document, 0)]
        seen_containers: set[int] = set()
        nodes = 0
        while stack:
            value, depth = stack.pop()
            nodes += 1
            if nodes > _MAX_JSON_NODES:
                raise ArtifactBoundaryError("artifact_json_too_large")
            if depth > _MAX_JSON_DEPTH:
                raise ArtifactBoundaryError("artifact_json_too_deep")
            if isinstance(value, dict):
                identity = id(value)
                if identity in seen_containers:
                    raise ArtifactBoundaryError("artifact_json_invalid")
                seen_containers.add(identity)
                for key, child in value.items():
                    if not isinstance(key, str):
                        raise ArtifactBoundaryError("artifact_json_invalid")
                    if len(key.encode("utf-8")) > _MAX_JSON_STRING_BYTES:
                        raise ArtifactBoundaryError("artifact_json_too_large")
                    stack.append((child, depth + 1))
                continue
            if isinstance(value, list):
                identity = id(value)
                if identity in seen_containers:
                    raise ArtifactBoundaryError("artifact_json_invalid")
                seen_containers.add(identity)
                stack.extend((child, depth + 1) for child in value)
                continue
            if isinstance(value, str):
                if len(value.encode("utf-8")) > _MAX_JSON_STRING_BYTES:
                    raise ArtifactBoundaryError("artifact_json_too_large")
                continue
            if value is None or isinstance(value, (bool, int)):
                continue
            if isinstance(value, float) and math.isfinite(value):
                continue
            raise ArtifactBoundaryError("artifact_json_invalid")

    def delete_workspace(self, workspace_id: str) -> ArtifactCleanupState:
        self._validate_identifier(workspace_id)
        self._validate_base_identities()
        workspace = self._root / "workspaces" / workspace_id
        try:
            workspace_stat = self._lstat_or_none(workspace)
        except ArtifactBoundaryError:
            return ArtifactCleanupState.CLEANUP_PENDING
        if workspace_stat is None:
            return ArtifactCleanupState.DELETED
        self._prepare_workspace(workspace_id, create=False)
        evidence = workspace / "evidence"
        try:
            evidence_stat = self._lstat_or_none(evidence)
        except ArtifactBoundaryError:
            return ArtifactCleanupState.CLEANUP_PENDING
        if evidence_stat is None:
            return ArtifactCleanupState.DELETED
        evidence_identity = self._prepare_evidence(workspace_id, create=False)
        self._validate_cleanup_tree(evidence, evidence, evidence_identity)
        claim_identity = self._cleanup_claim_identity(evidence_identity)
        claim = _OwnedArtifactClaim(
            relative_path=f"workspaces/{workspace_id}/evidence",
            device_id=str(claim_identity[0]),
            file_id=str(claim_identity[1]),
        )
        try:
            remove_owned_tree = getattr(
                self._filesystem_ops,
                "remove_owned_tree",
                self._native_filesystem_ops.remove_owned_tree,
            )
            remove_owned_tree(self._root, claim)
        except OwnedTreeRemovalError as error:
            if error.code in {
                "cleanup_identity_changed",
                "cleanup_link_or_reparse",
                "cleanup_path_invalid",
                "cleanup_special_file",
                "cleanup_identity_unavailable",
            }:
                raise ArtifactBoundaryError("artifact_identity_changed") from None
            return ArtifactCleanupState.CLEANUP_PENDING
        except OSError as error:
            if self._is_retryable_cleanup_error(error):
                return ArtifactCleanupState.CLEANUP_PENDING
            return ArtifactCleanupState.CLEANUP_PENDING

        try:
            remaining = self._lstat_or_none(evidence)
        except ArtifactBoundaryError:
            return ArtifactCleanupState.CLEANUP_PENDING
        if remaining is not None:
            current = self._directory_identity(evidence)
            if current != evidence_identity:
                raise ArtifactBoundaryError("artifact_identity_changed")
            return ArtifactCleanupState.CLEANUP_PENDING
        self._forget_workspace(workspace_id)
        return ArtifactCleanupState.DELETED

    def _publish_bytes(
        self,
        workspace_id: str,
        artifact_type: str,
        artifact_id: str,
        content: bytes,
        expected_sha256: str,
        *,
        suffix: str,
    ) -> str:
        self._validate_expected_hash(expected_sha256)
        relative = self._artifact_relative_path(
            workspace_id,
            artifact_type,
            artifact_id,
            suffix=suffix,
        )
        parent = self._prepare_collection(workspace_id, artifact_type, create=True)
        target_name = f"{artifact_id}{suffix}"
        target_path = parent / target_name
        self._validate_replace_target(target_path)
        temporary_name = f".examsage-artifact-{secrets.token_hex(16)}.tmp"
        temporary_path = parent / temporary_name
        temporary_identity: Identity | None = None
        published_identity: Identity | None = None
        opener = SecureFileOpener()
        parent_relative = relative.parent
        try:
            with contextlib.ExitStack() as anchor_stack:
                directory_fd = anchor_stack.enter_context(
                    opener.anchor_directory(self._root, parent_relative)
                )
                descriptor = self._create_temporary(
                    temporary_path,
                    temporary_name,
                    directory_fd,
                )
                with os.fdopen(descriptor, "wb", closefd=True) as destination:
                    destination.write(content)
                    destination.flush()
                    os.fsync(destination.fileno())
                    temporary_stat = os.fstat(destination.fileno())
                    self._validate_regular_identity(temporary_stat)
                    temporary_identity = (
                        temporary_stat.st_dev,
                        temporary_stat.st_ino,
                    )
                if hashlib.sha256(content).hexdigest() != expected_sha256:
                    raise ArtifactBoundaryError("artifact_hash_mismatch")
                self._validate_hierarchy(workspace_id, artifact_type)
                named_temporary = self._stat_at(
                    temporary_path,
                    temporary_name,
                    directory_fd,
                )
                self._validate_regular_identity(
                    named_temporary,
                    expected=temporary_identity,
                )
                if os.name == "nt":
                    anchor_stack.close()
                    self._validate_hierarchy(workspace_id, artifact_type)
                    self._validate_regular_identity(
                        temporary_path.stat(follow_symlinks=False),
                        expected=temporary_identity,
                    )
                source: str | Path = temporary_name if directory_fd is not None else temporary_path
                destination: str | Path = target_name if directory_fd is not None else target_path
                atomic_result = self._filesystem_ops.atomic_replace(
                    source,
                    destination,
                    directory_fd=directory_fd,
                    expected_identity=temporary_identity,
                    expected_parent_identity=self._directory_identities[(workspace_id, artifact_type)],
                    expected_sha256=expected_sha256,
                )
                if atomic_result is None:
                    published = self._open_and_hash_at(
                        target_path,
                        target_name,
                        directory_fd,
                    )
                    self._validate_regular_identity(
                        published[0],
                        expected=temporary_identity,
                    )
                    published_identity = (
                        published[0].st_dev,
                        published[0].st_ino,
                    )
                    published_digest = published[1]
                else:
                    published_identity, published_digest = atomic_result
                    if published_identity != temporary_identity:
                        raise ArtifactBoundaryError("artifact_identity_changed")
                    self._validate_hierarchy(workspace_id, artifact_type)
                    self._validate_regular_identity(
                        target_path.stat(follow_symlinks=False),
                        expected=published_identity,
                    )
                if published_digest != expected_sha256:
                    if published_identity is not None:
                        self._filesystem_ops.remove_file(
                            parent,
                            target_name,
                            directory_fd=directory_fd,
                            expected_identity=published_identity,
                        )
                        published_identity = None
                    raise ArtifactBoundaryError("artifact_hash_mismatch")
                if directory_fd is not None:
                    os.fsync(directory_fd)
                self._validate_hierarchy(workspace_id, artifact_type)
        except ArtifactBoundaryError:
            self._remove_temporary_if_owned(
                temporary_path,
                temporary_name,
                temporary_identity,
            )
            raise
        except (OSError, SecureOpenError):
            identity_error: ArtifactBoundaryError | None = None
            try:
                self._validate_hierarchy(workspace_id, artifact_type)
            except ArtifactBoundaryError as error:
                identity_error = error
            self._remove_temporary_if_owned(
                temporary_path,
                temporary_name,
                temporary_identity,
            )
            if identity_error is not None:
                raise ArtifactBoundaryError("artifact_identity_changed") from None
            raise ArtifactBoundaryError("artifact_publish_failed") from None

        self._validate_hierarchy(workspace_id, artifact_type)
        if published_identity is None:
            raise ArtifactBoundaryError("artifact_publish_failed")
        self._file_identities[(workspace_id, artifact_type, artifact_id)] = published_identity
        self._record_artifact(
            workspace_id,
            artifact_type,
            f"{artifact_id}{suffix}",
            published_identity,
            expected_sha256,
        )
        return expected_sha256

    def _create_temporary(
        self,
        path: Path,
        name: str,
        directory_fd: int | None,
    ) -> int:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        if directory_fd is None:
            return os.open(path, flags, 0o600)
        return os.open(name, flags, 0o600, dir_fd=directory_fd)

    def _open_and_hash_at(
        self,
        path: Path,
        name: str,
        directory_fd: int | None,
    ) -> tuple[os.stat_result, str]:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = (
            os.open(path, flags) if directory_fd is None else os.open(name, flags, dir_fd=directory_fd)
        )
        try:
            opened = os.fstat(descriptor)
            self._validate_regular_identity(opened)
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            named = self._stat_at(path, name, directory_fd)
            self._validate_regular_identity(
                named,
                expected=(opened.st_dev, opened.st_ino),
            )
            return opened, digest.hexdigest()
        finally:
            os.close(descriptor)

    def _remove_temporary_if_owned(
        self,
        path: Path,
        name: str,
        identity: Identity | None,
    ) -> None:
        if identity is None:
            return
        with contextlib.suppress(OSError, ArtifactBoundaryError):
            current = path.stat(follow_symlinks=False)
            self._validate_regular_identity(current, expected=identity)
            path.unlink()

    def _prepare_collection(
        self,
        workspace_id: str,
        artifact_type: str,
        *,
        create: bool,
    ) -> Path:
        evidence_identity = self._prepare_evidence(workspace_id, create=create)
        del evidence_identity
        collection = self._root / "workspaces" / workspace_id / "evidence" / artifact_type
        key = (workspace_id, artifact_type)
        if self._lstat_or_none(collection) is None:
            if not create:
                raise ArtifactBoundaryError("artifact_not_found")
            self._create_directory(collection)
        current = self._directory_identity(collection)
        expected = self._directory_identities.get(key)
        if expected is not None and current != expected:
            raise ArtifactBoundaryError("artifact_identity_changed")
        self._directory_identities[key] = current
        return collection

    def _prepare_evidence(self, workspace_id: str, *, create: bool) -> Identity:
        workspace_identity = self._prepare_workspace(workspace_id, create=create)
        del workspace_identity
        evidence = self._root / "workspaces" / workspace_id / "evidence"
        key = (workspace_id, "evidence")
        evidence_created = False
        if self._lstat_or_none(evidence) is None:
            if not create:
                raise ArtifactBoundaryError("artifact_not_found")
            self._create_directory(evidence)
            evidence_created = True
        current = self._directory_identity(evidence)
        expected = self._directory_identities.get(key)
        if expected is not None and current != expected:
            raise ArtifactBoundaryError("artifact_identity_changed")
        self._directory_identities[key] = current
        self._load_or_create_ownership(
            workspace_id,
            evidence,
            current,
            created=evidence_created,
        )
        return current

    def _prepare_workspace(self, workspace_id: str, *, create: bool) -> Identity:
        self._validate_base_identities()
        workspace = self._root / "workspaces" / workspace_id
        key = (workspace_id, "workspace")
        if self._lstat_or_none(workspace) is None:
            if not create:
                raise ArtifactBoundaryError("artifact_not_found")
            self._create_directory(workspace)
        current = self._directory_identity(workspace)
        expected = self._directory_identities.get(key)
        if expected is not None and current != expected:
            raise ArtifactBoundaryError("artifact_identity_changed")
        self._directory_identities[key] = current
        return current

    def _load_or_create_ownership(
        self,
        workspace_id: str,
        evidence: Path,
        evidence_identity: Identity,
        *,
        created: bool,
    ) -> None:
        marker = evidence / _OWNERSHIP_NAME
        marker_stat = self._lstat_or_none(marker)
        if marker_stat is None:
            if not created:
                raise ArtifactBoundaryError("artifact_identity_changed")
            self._write_ownership_marker(workspace_id, evidence, evidence_identity, {})
            return
        self._validate_regular_identity(marker_stat)
        relative = PurePosixPath("workspaces", workspace_id, "evidence", _OWNERSHIP_NAME)
        opener = SecureFileOpener()
        try:
            with opener.open_regular(self._root, relative) as source:
                opened = os.fstat(source.fileno())
                self._validate_regular_identity(
                    opened,
                    expected=(marker_stat.st_dev, marker_stat.st_ino),
                )
                content = source.read(_MAX_MARKER_BYTES + 1)
        except (OSError, SecureOpenError):
            raise ArtifactBoundaryError("artifact_identity_changed") from None
        if len(content) > _MAX_MARKER_BYTES:
            raise ArtifactBoundaryError("artifact_identity_changed")
        try:
            payload = json.loads(content.decode("utf-8"))
        except (ValueError, RecursionError, UnicodeError):
            raise ArtifactBoundaryError("artifact_identity_changed") from None
        if not isinstance(payload, dict) or set(payload) != {
            "version",
            "root",
            "workspace",
            "evidence",
            "marker",
            "artifacts",
        }:
            raise ArtifactBoundaryError("artifact_identity_changed")
        workspace_identity = self._directory_identities.get((workspace_id, "workspace"))
        expected_values = {
            "version": _OWNERSHIP_VERSION,
            "root": list(self._root_identity),
            "workspace": list(workspace_identity or ()),
            "evidence": list(evidence_identity),
            "marker": [opened.st_dev, opened.st_ino],
        }
        if any(payload.get(key) != value for key, value in expected_values.items()):
            raise ArtifactBoundaryError("artifact_identity_changed")
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict) or len(artifacts) > 100_000:
            raise ArtifactBoundaryError("artifact_identity_changed")
        for name, claim in artifacts.items():
            if (
                not isinstance(name, str)
                or not isinstance(claim, dict)
                or set(claim) != {"identity", "sha256"}
                or not isinstance(claim["identity"], list)
                or len(claim["identity"]) != 2
                or not all(isinstance(value, int) for value in claim["identity"])
                or not isinstance(claim["sha256"], str)
                or _SHA256.fullmatch(claim["sha256"]) is None
            ):
                raise ArtifactBoundaryError("artifact_identity_changed")
        self._ownership[workspace_id] = payload

    def _write_ownership_marker(
        self,
        workspace_id: str,
        evidence: Path,
        evidence_identity: Identity,
        artifacts: dict[str, Any],
    ) -> None:
        temporary_name = f".examsage-ownership-{secrets.token_hex(16)}.tmp"
        temporary_path = evidence / temporary_name
        marker_path = evidence / _OWNERSHIP_NAME
        temporary_identity: Identity | None = None
        relative = PurePosixPath("workspaces", workspace_id, "evidence")
        opener = SecureFileOpener()
        try:
            with contextlib.ExitStack() as anchor_stack:
                directory_fd = anchor_stack.enter_context(opener.anchor_directory(self._root, relative))
                descriptor = self._create_temporary(temporary_path, temporary_name, directory_fd)
                with os.fdopen(descriptor, "wb", closefd=True) as destination:
                    temporary_stat = os.fstat(destination.fileno())
                    self._validate_regular_identity(temporary_stat)
                    temporary_identity = (
                        temporary_stat.st_dev,
                        temporary_stat.st_ino,
                    )
                    workspace_identity = self._directory_identities[(workspace_id, "workspace")]
                    payload = {
                        "version": _OWNERSHIP_VERSION,
                        "root": list(self._root_identity),
                        "workspace": list(workspace_identity),
                        "evidence": list(evidence_identity),
                        "marker": list(temporary_identity),
                        "artifacts": artifacts,
                    }
                    encoded = json.dumps(
                        payload,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    if len(encoded) > _MAX_MARKER_BYTES:
                        raise ArtifactBoundaryError("artifact_marker_full")
                    destination.write(encoded)
                    destination.flush()
                    os.fsync(destination.fileno())
                named = self._stat_at(temporary_path, temporary_name, directory_fd)
                self._validate_regular_identity(named, expected=temporary_identity)
                if os.name == "nt":
                    anchor_stack.close()
                    self._validate_hierarchy_for_marker(workspace_id, evidence_identity)
                source: str | Path = temporary_name if directory_fd is not None else temporary_path
                destination_name: str | Path = _OWNERSHIP_NAME if directory_fd is not None else marker_path
                result = self._filesystem_ops.atomic_replace(
                    source,
                    destination_name,
                    directory_fd=directory_fd,
                    expected_identity=temporary_identity,
                    expected_parent_identity=evidence_identity,
                    expected_sha256=hashlib.sha256(encoded).hexdigest(),
                )
                if result is not None and result[0] != temporary_identity:
                    raise ArtifactBoundaryError("artifact_identity_changed")
                if directory_fd is not None:
                    os.fsync(directory_fd)
        except ArtifactBoundaryError:
            self._remove_temporary_if_owned(temporary_path, temporary_name, temporary_identity)
            raise
        except (OSError, SecureOpenError):
            self._remove_temporary_if_owned(temporary_path, temporary_name, temporary_identity)
            raise ArtifactBoundaryError("artifact_publish_failed") from None
        marker_stat = marker_path.stat(follow_symlinks=False)
        self._validate_regular_identity(marker_stat, expected=temporary_identity)
        self._ownership[workspace_id] = payload

    def _validate_hierarchy_for_marker(self, workspace_id: str, evidence_identity: Identity) -> None:
        self._validate_base_identities()
        self._require_directory_identity(workspace_id, "workspace")
        evidence = self._root / "workspaces" / workspace_id / "evidence"
        if self._directory_identity(evidence) != evidence_identity:
            raise ArtifactBoundaryError("artifact_identity_changed")

    def _record_artifact(
        self,
        workspace_id: str,
        artifact_type: str,
        filename: str,
        identity: Identity,
        digest: str,
    ) -> None:
        ownership = self._ownership.get(workspace_id)
        if ownership is None:
            raise ArtifactBoundaryError("artifact_identity_changed")
        artifacts = dict(ownership["artifacts"])
        artifacts[f"{artifact_type}/{filename}"] = {
            "identity": list(identity),
            "sha256": digest,
        }
        evidence = self._root / "workspaces" / workspace_id / "evidence"
        evidence_identity = self._directory_identities[(workspace_id, "evidence")]
        self._write_ownership_marker(workspace_id, evidence, evidence_identity, artifacts)

    def _artifact_claim(self, workspace_id: str, artifact_type: str, filename: str) -> tuple[Identity, str]:
        ownership = self._ownership.get(workspace_id)
        claim = (
            ownership.get("artifacts", {}).get(f"{artifact_type}/{filename}")
            if ownership is not None
            else None
        )
        if not isinstance(claim, dict):
            raise ArtifactBoundaryError("artifact_identity_changed")
        return tuple(claim["identity"]), claim["sha256"]

    def _validate_hierarchy(self, workspace_id: str, artifact_type: str) -> None:
        self._validate_base_identities()
        self._require_directory_identity(workspace_id, "workspace")
        self._require_directory_identity(workspace_id, "evidence")
        self._require_directory_identity(workspace_id, artifact_type)

    def _require_directory_identity(self, workspace_id: str, name: str) -> None:
        expected = self._directory_identities.get((workspace_id, name))
        if expected is None:
            raise ArtifactBoundaryError("artifact_identity_changed")
        base = self._root / "workspaces" / workspace_id
        if name == "workspace":
            path = base
        elif name == "evidence":
            path = base / "evidence"
        else:
            path = base / "evidence" / name
        if self._directory_identity(path) != expected:
            raise ArtifactBoundaryError("artifact_identity_changed")

    def _validate_base_identities(self) -> None:
        if self._directory_identity(self._root) != self._root_identity:
            raise ArtifactBoundaryError("artifact_identity_changed")
        workspaces = self._root / "workspaces"
        if self._directory_identity(workspaces) != self._workspaces_identity:
            raise ArtifactBoundaryError("artifact_identity_changed")

    def _directory_identity(self, path: Path) -> Identity:
        try:
            self._reject_link_or_reparse(path)
            value = path.stat(follow_symlinks=False)
            if not stat.S_ISDIR(value.st_mode):
                raise ArtifactBoundaryError("artifact_identity_changed")
            if path.resolve(strict=True) != path.absolute():
                raise ArtifactBoundaryError("artifact_identity_changed")
            return value.st_dev, value.st_ino
        except ArtifactBoundaryError:
            raise
        except OSError:
            raise ArtifactBoundaryError("artifact_identity_changed") from None

    @staticmethod
    def _validate_regular_identity(
        value: os.stat_result,
        *,
        expected: Identity | None = None,
    ) -> None:
        attributes = getattr(value, "st_file_attributes", 0)
        identity = (value.st_dev, value.st_ino)
        if (
            not stat.S_ISREG(value.st_mode)
            or attributes & _REPARSE_ATTRIBUTE
            or value.st_nlink != 1
            or (expected is not None and identity != expected)
        ):
            raise ArtifactBoundaryError("artifact_identity_changed")

    def _validate_replace_target(self, target: Path) -> None:
        if self._lstat_or_none(target) is None:
            return
        try:
            self._reject_link_or_reparse(target)
            self._validate_regular_identity(target.stat(follow_symlinks=False))
        except OSError:
            raise ArtifactBoundaryError("artifact_identity_changed") from None

    @staticmethod
    def _stat_at(
        path: Path,
        name: str,
        directory_fd: int | None,
    ) -> os.stat_result:
        if directory_fd is None:
            return path.stat(follow_symlinks=False)
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)

    @staticmethod
    def _create_directory(path: Path) -> None:
        if EvidenceArtifactStore._lstat_or_none(path) is not None:
            EvidenceArtifactStore._reject_link_or_reparse(path)
            if not path.is_dir():
                raise ArtifactBoundaryError("artifact_identity_changed")
            return
        path.mkdir(mode=0o700)
        EvidenceArtifactStore._reject_link_or_reparse(path)

    @staticmethod
    def _lstat_or_none(path: Path) -> os.stat_result | None:
        try:
            return os.lstat(path)
        except FileNotFoundError:
            return None
        except OSError:
            raise ArtifactBoundaryError("artifact_identity_changed") from None

    @classmethod
    def _validate_ancestor_chain(cls, path: Path) -> None:
        chain: list[Path] = []
        current = path
        while True:
            chain.append(current)
            if current.parent == current:
                break
            current = current.parent
        for candidate in reversed(chain):
            value = cls._lstat_or_none(candidate)
            if value is None:
                continue
            attributes = getattr(value, "st_file_attributes", 0)
            if stat.S_ISLNK(value.st_mode) or attributes & _REPARSE_ATTRIBUTE:
                raise ArtifactBoundaryError("artifact_identity_changed")

    @staticmethod
    def _reject_link_or_reparse(path: Path, *, allow_missing: bool = False) -> None:
        if EvidenceArtifactStore._lstat_or_none(path) is None:
            if allow_missing:
                return
            raise ArtifactBoundaryError("artifact_identity_changed")
        try:
            if path.is_symlink() or is_reparse_point(path):
                raise ArtifactBoundaryError("artifact_identity_changed")
        except ArtifactBoundaryError:
            raise
        except OSError:
            raise ArtifactBoundaryError("artifact_identity_changed") from None

    @staticmethod
    def _artifact_relative_path(
        workspace_id: str,
        artifact_type: str,
        artifact_id: str,
        *,
        suffix: str,
    ) -> PurePosixPath:
        return PurePosixPath(
            "workspaces",
            workspace_id,
            "evidence",
            artifact_type,
            f"{artifact_id}{suffix}",
        )

    def _artifact_path(self, relative: PurePosixPath) -> Path:
        return self._root.joinpath(*relative.parts)

    @staticmethod
    def _validate_identifier(identifier: str) -> None:
        if not isinstance(identifier, str) or _IDENTIFIER.fullmatch(identifier) is None:
            raise ArtifactBoundaryError("artifact_identifier_invalid")

    @staticmethod
    def _validate_json_type(artifact_type: str) -> None:
        if artifact_type not in _JSON_TYPES:
            raise ArtifactBoundaryError("artifact_type_invalid")

    @staticmethod
    def _validate_expected_hash(expected_sha256: str) -> None:
        if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
            raise ArtifactBoundaryError("artifact_hash_invalid")

    @staticmethod
    def _is_retryable_cleanup_error(error: OSError) -> bool:
        return getattr(error, "winerror", None) in {32, 33} or error.errno in {
            errno.EACCES,
            errno.EBUSY,
            errno.EPERM,
        }

    def _forget_workspace(self, workspace_id: str) -> None:
        self._directory_identities = {
            key: value for key, value in self._directory_identities.items() if key[0] != workspace_id
        }
        self._file_identities = {
            key: value for key, value in self._file_identities.items() if key[0] != workspace_id
        }

    def _validate_cleanup_tree(
        self,
        evidence_root: Path,
        directory: Path,
        expected_identity: Identity,
    ) -> None:
        if self._directory_identity(directory) != expected_identity:
            raise ArtifactBoundaryError("artifact_identity_changed")
        try:
            with os.scandir(directory) as iterator:
                names = sorted(
                    (entry.name for entry in iterator),
                    key=lambda name: (name.casefold(), name),
                )
        except OSError:
            raise ArtifactBoundaryError("artifact_identity_changed") from None
        opener = SecureFileOpener()
        for name in names:
            child = directory / name
            try:
                named = child.stat(follow_symlinks=False)
            except OSError:
                raise ArtifactBoundaryError("artifact_identity_changed") from None
            attributes = getattr(named, "st_file_attributes", 0)
            if stat.S_ISLNK(named.st_mode) or attributes & _REPARSE_ATTRIBUTE:
                raise ArtifactBoundaryError("artifact_identity_changed")
            identity = (named.st_dev, named.st_ino)
            if stat.S_ISDIR(named.st_mode):
                self._validate_cleanup_tree(evidence_root, child, identity)
                if self._directory_identity(child) != identity:
                    raise ArtifactBoundaryError("artifact_identity_changed")
                continue
            self._validate_regular_identity(named)
            relative = PurePosixPath(*child.relative_to(evidence_root).parts)
            try:
                with opener.open_regular(evidence_root, relative) as source:
                    opened = os.fstat(source.fileno())
                    self._validate_regular_identity(opened, expected=identity)
            except (OSError, SecureOpenError):
                raise ArtifactBoundaryError("artifact_identity_changed") from None
        if self._directory_identity(directory) != expected_identity:
            raise ArtifactBoundaryError("artifact_identity_changed")

    def _cleanup_claim_identity(self, identity: Identity) -> Identity:
        if os.name != "nt":
            return identity
        from ctypes import wintypes

        volume_serial = wintypes.DWORD()
        maximum_component_length = wintypes.DWORD()
        filesystem_flags = wintypes.DWORD()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetVolumeInformationW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPWSTR,
            wintypes.DWORD,
        ]
        kernel32.GetVolumeInformationW.restype = wintypes.BOOL
        if not kernel32.GetVolumeInformationW(
            self._root.anchor,
            None,
            0,
            ctypes.byref(volume_serial),
            ctypes.byref(maximum_component_length),
            ctypes.byref(filesystem_flags),
            None,
            0,
        ):
            raise ArtifactBoundaryError("artifact_identity_changed")
        return int(volume_serial.value), identity[1]
