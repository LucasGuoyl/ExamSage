from __future__ import annotations

import contextlib
import ctypes
import errno
import hashlib
import json
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
    OwnedTreeRemovalError,
    OwnedTreeRemover,
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
        directory_fd: int | None = None,
        expected_identity: Identity | None = None,
        expected_parent_identity: Identity | None = None,
    ) -> tuple[Identity, str] | None: ...

    def remove_owned_tree(
        self,
        data_root: Path,
        claim: _OwnedArtifactClaim,
    ) -> None: ...


class _NativeArtifactFilesystemOps:
    def atomic_replace(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        directory_fd: int | None = None,
        expected_identity: Identity | None = None,
        expected_parent_identity: Identity | None = None,
    ) -> tuple[Identity, str] | None:
        if os.name == "nt":
            return self._atomic_replace_windows(
                Path(source),
                Path(destination),
                expected_identity=expected_identity,
                expected_parent_identity=expected_parent_identity,
            )
        if directory_fd is None:
            os.replace(source, destination)
            return None
        os.replace(
            source,
            destination,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        return None

    def remove_owned_tree(
        self,
        data_root: Path,
        claim: _OwnedArtifactClaim,
    ) -> None:
        OwnedTreeRemover(data_root)(claim)

    @staticmethod
    def _atomic_replace_windows(
        source: Path,
        destination: Path,
        *,
        expected_identity: Identity | None,
        expected_parent_identity: Identity | None,
    ) -> tuple[Identity, str]:
        import msvcrt
        from ctypes import wintypes

        generic_read = 0x80000000
        delete = 0x00010000
        open_existing = 3
        file_attribute_directory = 0x10
        file_attribute_normal = 0x80
        file_attribute_reparse_point = 0x400
        file_flag_open_reparse_point = 0x00200000
        file_flag_backup_semantics = 0x02000000
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
                or (expected_parent_identity is not None and parent_identity != expected_parent_identity)
                or final_parent != ntpath.normcase(ntpath.abspath(str(source.parent)))
            ):
                raise ArtifactBoundaryError("artifact_identity_changed")

            named = source.stat(follow_symlinks=False)
            handle = kernel32.CreateFileW(
                str(source),
                generic_read | delete,
                0,
                None,
                open_existing,
                file_attribute_normal | file_flag_open_reparse_point,
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
                or (expected_identity is not None and named_identity != expected_identity)
            ):
                raise ArtifactBoundaryError("artifact_identity_changed")
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
            descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
            handle = None
            digest = hashlib.sha256()
            try:
                while chunk := os.read(descriptor, 1024 * 1024):
                    digest.update(chunk)
            finally:
                os.close(descriptor)
            return named_identity, digest.hexdigest()
        finally:
            if handle is not None:
                kernel32.CloseHandle(handle)
            kernel32.CloseHandle(parent_handle)


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
        self._native_filesystem_ops = _NativeArtifactFilesystemOps()
        self._filesystem_ops = filesystem_ops or self._native_filesystem_ops
        self._directory_identities: dict[tuple[str, str], Identity] = {}
        self._file_identities: dict[tuple[str, str, str], Identity] = {}

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
        expected = self._file_identities.get((workspace_id, "parts", part_id))
        opener = SecureFileOpener()
        try:
            with opener.open_regular(self._root, relative) as source:
                opened = os.fstat(source.fileno())
                self._validate_regular_identity(opened, expected=expected)
                named = self._artifact_path(relative).stat(follow_symlinks=False)
                self._validate_regular_identity(named, expected=(opened.st_dev, opened.st_ino))
                self._validate_hierarchy(workspace_id, "parts")
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
        expected = self._file_identities.get((workspace_id, artifact_type, artifact_id))
        opener = SecureFileOpener()
        try:
            with opener.open_regular(self._root, relative) as source:
                opened = os.fstat(source.fileno())
                self._validate_regular_identity(opened, expected=expected)
                named = self._artifact_path(relative).stat(follow_symlinks=False)
                self._validate_regular_identity(named, expected=(opened.st_dev, opened.st_ino))
                self._validate_hierarchy(workspace_id, artifact_type)
                content = source.read()
                self._validate_hierarchy(workspace_id, artifact_type)
            result = json.loads(content.decode("utf-8"))
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
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ArtifactBoundaryError("artifact_json_invalid") from None
        self._file_identities[(workspace_id, artifact_type, artifact_id)] = (
            opened.st_dev,
            opened.st_ino,
        )
        return result

    def delete_workspace(self, workspace_id: str) -> ArtifactCleanupState:
        self._validate_identifier(workspace_id)
        self._validate_base_identities()
        workspace = self._root / "workspaces" / workspace_id
        if not os.path.lexists(workspace):
            return ArtifactCleanupState.DELETED
        self._prepare_workspace(workspace_id, create=False)
        evidence = workspace / "evidence"
        if not os.path.lexists(evidence):
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

        if os.path.lexists(evidence):
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
                atomic_replace = getattr(
                    self._filesystem_ops,
                    "atomic_replace",
                    self._native_filesystem_ops.atomic_replace,
                )
                if atomic_replace == self._native_filesystem_ops.atomic_replace:
                    atomic_result = atomic_replace(
                        source,
                        destination,
                        directory_fd=directory_fd,
                        expected_identity=temporary_identity,
                        expected_parent_identity=self._directory_identities[(workspace_id, artifact_type)],
                    )
                else:
                    atomic_result = atomic_replace(
                        source,
                        destination,
                        directory_fd=directory_fd,
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
        if not os.path.lexists(collection):
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
        if not os.path.lexists(evidence):
            if not create:
                raise ArtifactBoundaryError("artifact_not_found")
            self._create_directory(evidence)
        current = self._directory_identity(evidence)
        expected = self._directory_identities.get(key)
        if expected is not None and current != expected:
            raise ArtifactBoundaryError("artifact_identity_changed")
        self._directory_identities[key] = current
        return current

    def _prepare_workspace(self, workspace_id: str, *, create: bool) -> Identity:
        self._validate_base_identities()
        workspace = self._root / "workspaces" / workspace_id
        key = (workspace_id, "workspace")
        if not os.path.lexists(workspace):
            if not create:
                raise ArtifactBoundaryError("artifact_not_found")
            self._create_directory(workspace)
        current = self._directory_identity(workspace)
        expected = self._directory_identities.get(key)
        if expected is not None and current != expected:
            raise ArtifactBoundaryError("artifact_identity_changed")
        self._directory_identities[key] = current
        return current

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
        if not os.path.lexists(target):
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
        if os.path.lexists(path):
            EvidenceArtifactStore._reject_link_or_reparse(path)
            if not path.is_dir():
                raise ArtifactBoundaryError("artifact_identity_changed")
            return
        path.mkdir(mode=0o700)
        EvidenceArtifactStore._reject_link_or_reparse(path)

    @staticmethod
    def _reject_link_or_reparse(path: Path, *, allow_missing: bool = False) -> None:
        if not os.path.lexists(path):
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
