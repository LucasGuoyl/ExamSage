from __future__ import annotations

import hashlib
import os
import re
import stat
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import NAMESPACE_URL, UUID, uuid5

from exam_predictor.workspace.archive import ArchiveInspector
from exam_predictor.workspace.filesystem import (
    RootAnchor,
    SecureFileOpener,
    SecureOpenError,
    is_reparse_point as native_is_reparse_point,
)
from exam_predictor.workspace.models import (
    ArchiveMember,
    ManifestEntry,
    ScanPolicy,
    ScanProgress,
    ScanResult,
    SourceState,
)
from exam_predictor.workspace.policy import DEFAULT_SCAN_POLICY, classify_format


SOURCE_LINK_OR_REPARSE = "source_link_or_reparse"
SOURCE_NOT_REGULAR = "source_not_regular"
SOURCE_OPEN_FAILED = "source_open_failed"
SOURCE_CHANGED_DURING_SCAN = "source_changed_during_scan"
SOURCE_DEPTH_LIMIT = "source_depth_limit"
SOURCE_PATH_LIMIT = "source_path_limit"
SOURCE_FILE_COUNT_LIMIT = "source_file_count_limit"
SOURCE_WORKSPACE_SIZE_LIMIT = "source_workspace_size_limit"
ARCHIVE_INVALID = "archive_invalid"


@dataclass(frozen=True)
class _Candidate:
    path: Path
    relative_path: str
    item_kind: str
    stat_result: os.stat_result | None
    failure_code: str | None = None


@dataclass(frozen=True)
class _ReadOutcome:
    sha256: str | None
    bytes_hashed: int
    archive_members: tuple[ArchiveMember, ...] = ()
    failure_code: str | None = None
    stat_result: os.stat_result | None = None


@dataclass(frozen=True)
class RevalidatedEntry:
    entry_id: str
    sha256: str | None
    size_bytes: int
    modified_ns: int | None
    device_id: str | None
    file_id: str | None
    failure_code: str | None = None


@dataclass(frozen=True)
class RevalidationResult:
    canonical_root: Path
    root_device: str
    root_file_id: str
    entries: tuple[RevalidatedEntry, ...]


@dataclass(frozen=True)
class ScanExecution:
    result: ScanResult
    canonical_root: Path
    root_device: str
    root_file_id: str


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _workspace_namespace(workspace_id: str) -> UUID:
    try:
        return UUID(workspace_id)
    except ValueError:
        return uuid5(NAMESPACE_URL, workspace_id)


def _stable_entry_id(namespace: UUID, identity: str) -> str:
    return str(uuid5(namespace, identity))


def _safe_message(relative_path: str, failure_code: str) -> str:
    return f"Could not scan {relative_path}: {failure_code}."


def _sanitize_group(value: str) -> str:
    characters = [character if character.isalnum() else " " for character in value]
    return " ".join("".join(characters).split())[:120]


def _propose_course_group(relative_path: str) -> str:
    parts = PurePosixPath(relative_path).parts
    if len(parts) > 1:
        candidate = _sanitize_group(parts[0])
        if candidate and not re.fullmatch(r"(?i)(week|lecture|notes?|materials?)\s*\d*", candidate):
            return candidate
        return "unclassified"

    tokens = _sanitize_group(PurePosixPath(relative_path).stem).split()
    for token in tokens:
        if any(character.isalpha() for character in token) and any(
            character.isdigit() for character in token
        ):
            return token
    return "unclassified"


def _metadata_fields(value: os.stat_result | None) -> dict[str, int | str | None]:
    if value is None:
        return {
            "size_bytes": 0,
            "modified_ns": None,
            "device_id": None,
            "file_id": None,
        }
    return {
        "size_bytes": max(value.st_size, 0),
        "modified_ns": value.st_mtime_ns,
        "device_id": str(value.st_dev),
        "file_id": str(value.st_ino),
    }


class WorkspaceScanner:
    def __init__(
        self,
        policy: ScanPolicy = DEFAULT_SCAN_POLICY,
        *,
        archive_inspector: ArchiveInspector | None = None,
        after_hash_chunk: Callable[[Path, int], None] | None = None,
        is_reparse_point: Callable[[Path], bool] | None = None,
    ) -> None:
        self._policy = policy
        self._archive_inspector = archive_inspector or ArchiveInspector(policy)
        self._after_hash_chunk = after_hash_chunk
        self._is_reparse_point = is_reparse_point or native_is_reparse_point
        self._secure_file_opener = SecureFileOpener()
        self._directory_opener = SecureFileOpener()

    def scan(
        self,
        workspace_id: str,
        root: Path,
        *,
        previous_entries: Sequence[ManifestEntry] = (),
        emit: Callable[[ScanProgress], None] | None = None,
    ) -> ScanResult:
        return self.scan_with_identity(
            workspace_id,
            root,
            previous_entries=previous_entries,
            emit=emit,
        ).result

    def scan_with_identity(
        self,
        workspace_id: str,
        root: Path,
        *,
        previous_entries: Sequence[ManifestEntry] = (),
        emit: Callable[[ScanProgress], None] | None = None,
    ) -> ScanExecution:
        with self._directory_opener.anchor_root(root) as root_anchor:
            canonical_root = self._canonical_root(root, root_anchor)
            if root_anchor.identity is None:
                raise SecureOpenError("source_root_identity_unavailable")
            result = self._scan_anchored(
                workspace_id,
                canonical_root,
                root_anchor,
                previous_entries=previous_entries,
                emit=emit,
            )
            return ScanExecution(
                result=result,
                canonical_root=canonical_root,
                root_device=str(root_anchor.identity[0]),
                root_file_id=str(root_anchor.identity[1]),
            )

    def revalidate_entries(
        self,
        root: Path,
        entries: Sequence[ManifestEntry] = (),
    ) -> RevalidationResult:
        """Securely re-stat and hash an exact, policy-bounded set of native files."""
        if len(entries) > self._policy.max_files:
            raise ValueError("revalidation_file_count_limit")
        with self._directory_opener.anchor_root(root) as root_anchor:
            canonical_root = self._canonical_root(root, root_anchor)
            if root_anchor.identity is None:
                raise SecureOpenError("source_root_identity_unavailable")
            aggregate_size = 0
            validated: list[RevalidatedEntry] = []
            for entry in entries:
                failure_code: str | None = None
                outcome = _ReadOutcome(None, 0, failure_code=SOURCE_NOT_REGULAR)
                relative = PurePosixPath(entry.relative_path)
                if entry.archive_parent_entry_id is None and entry.item_kind == "file":
                    try:
                        metadata = self._secure_file_opener.stat_regular(
                            canonical_root,
                            relative,
                            root_anchor=root_anchor,
                        )
                    except SecureOpenError as error:
                        failure_code = error.code
                    else:
                        aggregate_size += max(metadata.st_size, 0)
                        if aggregate_size > self._policy.max_workspace_bytes:
                            failure_code = SOURCE_WORKSPACE_SIZE_LIMIT
                        else:
                            outcome = self._read(
                                canonical_root / Path(*relative.parts),
                                canonical_root,
                                root_anchor,
                                entry.relative_path,
                                False,
                            )
                            failure_code = outcome.failure_code
                metadata = outcome.stat_result
                validated.append(
                    RevalidatedEntry(
                        entry_id=entry.entry_id,
                        sha256=outcome.sha256,
                        size_bytes=max(metadata.st_size, 0) if metadata is not None else 0,
                        modified_ns=metadata.st_mtime_ns if metadata is not None else None,
                        device_id=str(metadata.st_dev) if metadata is not None else None,
                        file_id=str(metadata.st_ino) if metadata is not None else None,
                        failure_code=failure_code,
                    )
                )
            return RevalidationResult(
                canonical_root=canonical_root,
                root_device=str(root_anchor.identity[0]),
                root_file_id=str(root_anchor.identity[1]),
                entries=tuple(validated),
            )

    def _scan_anchored(
        self,
        workspace_id: str,
        canonical_root: Path,
        root_anchor: RootAnchor,
        *,
        previous_entries: Sequence[ManifestEntry],
        emit: Callable[[ScanProgress], None] | None,
    ) -> ScanResult:
        namespace = _workspace_namespace(workspace_id)
        candidates = self._enumerate(canonical_root, root_anchor)
        previous_native = {
            entry.relative_path: entry
            for entry in previous_entries
            if entry.archive_parent_entry_id is None
        }

        entries: list[ManifestEntry] = []
        discovered_count = 0
        bytes_hashed = 0
        failure_count = 0
        selected_bytes = 0
        file_count = 0
        emitted_progress = 0

        def bounded_emit(
            discovered: int,
            hashed: int,
            failures: int,
            relative_path: str,
        ) -> None:
            nonlocal emitted_progress
            if emit is None or emitted_progress >= self._policy.max_files:
                return
            self._emit(emit, discovered, hashed, failures, relative_path)
            emitted_progress += 1

        for candidate in candidates:
            if candidate.item_kind != "folder":
                file_count += 1
            entry, members, read_bytes, reserved_bytes = self._scan_candidate(
                workspace_id,
                namespace,
                canonical_root,
                root_anchor,
                candidate,
                previous_native.get(candidate.relative_path),
                file_count=file_count,
                selected_bytes=selected_bytes,
            )
            selected_bytes += reserved_bytes
            bytes_hashed += read_bytes
            entries.append(entry)
            discovered_count += 1
            if entry.state is SourceState.FAILED:
                failure_count += 1
            bounded_emit(
                discovered_count,
                bytes_hashed,
                failure_count,
                candidate.relative_path,
            )

            for member_index, member in enumerate(members):
                member_entry = self._archive_member_entry(
                    workspace_id,
                    namespace,
                    entry,
                    member,
                    member_index,
                )
                entries.append(member_entry)
                discovered_count += 1
                if member_entry.state is SourceState.FAILED:
                    failure_count += 1
                bounded_emit(
                    discovered_count,
                    bytes_hashed,
                    failure_count,
                    candidate.relative_path,
                )

        current_entry_ids = {entry.entry_id for entry in entries}
        for previous in previous_entries:
            if previous.entry_id in current_entry_ids:
                continue
            entries.append(
                previous.model_copy(
                    update={
                        "state": SourceState.REMOVED,
                        "included": False,
                        "inclusion_reason": "source_removed",
                        "failure_code": None,
                        "safe_message": None,
                    }
                )
            )

        entries.sort(
            key=lambda entry: (
                entry.relative_path.casefold(),
                entry.relative_path,
                (entry.archive_member_path or "").casefold(),
                entry.archive_member_path or "",
                entry.entry_id,
            )
        )
        return ScanResult(
            workspace_id=workspace_id,
            entries=tuple(entries),
            discovered_count=discovered_count,
            bytes_hashed=bytes_hashed,
            failure_count=failure_count,
            completed_at=datetime.now(UTC),
        )

    def _canonical_root(self, root: Path, root_anchor: RootAnchor) -> Path:
        try:
            if root.is_symlink() or self._is_reparse_point(root):
                raise SecureOpenError(SOURCE_LINK_OR_REPARSE)
            canonical_root = root.resolve(strict=True)
            root_stat = canonical_root.stat(follow_symlinks=False)
            if not stat.S_ISDIR(root_stat.st_mode) or self._is_reparse_point(
                canonical_root
            ):
                raise SecureOpenError("source_root_invalid")
        except SecureOpenError:
            raise
        except OSError:
            raise SecureOpenError("source_root_invalid") from None
        if root_anchor.identity is not None and root_anchor.platform == "posix":
            if root_anchor.identity != (root_stat.st_dev, root_stat.st_ino):
                raise SecureOpenError(SOURCE_LINK_OR_REPARSE)
        elif os.path.normcase(os.path.abspath(canonical_root)) != os.path.normcase(
            os.path.abspath(root_anchor.canonical_root)
        ):
            raise SecureOpenError(SOURCE_LINK_OR_REPARSE)
        return canonical_root

    def _enumerate(self, root: Path, root_anchor: RootAnchor) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        processed_candidates = 0
        limit_reached = False

        def walk(
            directory: Path,
            parent_parts: tuple[str, ...],
            directory_fd: int | None,
        ) -> None:
            nonlocal processed_candidates, limit_reached
            if limit_reached:
                return
            scan_target = directory if directory_fd is None else directory_fd
            with os.scandir(scan_target) as iterator:
                directory_entries = sorted(
                    iterator,
                    key=lambda item: (item.name.casefold(), item.name),
                )
            for directory_entry in directory_entries:
                if limit_reached:
                    return
                parts = (*parent_parts, directory_entry.name)
                relative_path = PurePosixPath(*parts).as_posix()
                path = directory / directory_entry.name
                if processed_candidates >= self._policy.max_files:
                    candidates.append(
                        _Candidate(
                            path,
                            relative_path,
                            "file",
                            None,
                            SOURCE_FILE_COUNT_LIMIT,
                        )
                    )
                    limit_reached = True
                    return
                processed_candidates += 1
                try:
                    metadata = directory_entry.stat(follow_symlinks=False)
                except OSError:
                    candidates.append(
                        _Candidate(path, relative_path, "file", None, SOURCE_OPEN_FAILED)
                    )
                    continue

                try:
                    is_link_like = (
                        directory_entry.is_symlink() or self._is_reparse_point(path)
                    )
                except OSError:
                    candidates.append(
                        _Candidate(path, relative_path, "file", metadata, SOURCE_OPEN_FAILED)
                    )
                    continue
                if is_link_like:
                    candidates.append(
                        _Candidate(
                            path,
                            relative_path,
                            "link",
                            metadata,
                            SOURCE_LINK_OR_REPARSE,
                        )
                    )
                    continue

                if stat.S_ISDIR(metadata.st_mode):
                    directory_failure = self._path_limit_failure(relative_path)
                    if directory_failure:
                        candidates.append(
                            _Candidate(
                                path,
                                relative_path,
                                "folder",
                                metadata,
                                directory_failure,
                            )
                        )
                        continue
                    try:
                        if directory_fd is None:
                            with self._directory_opener.anchor_directory(
                                root, PurePosixPath(*parts)
                            ) as child_fd:
                                walk(path, parts, child_fd)
                        else:
                            with self._directory_opener.anchor_child_directory(
                                directory_fd, directory_entry.name
                            ) as child_fd:
                                walk(path, parts, child_fd)
                    except SecureOpenError as error:
                        candidates.append(
                            _Candidate(
                                path,
                                relative_path,
                                "folder",
                                metadata,
                                error.code,
                            )
                        )
                    except OSError:
                        candidates.append(
                            _Candidate(
                                path,
                                relative_path,
                                "folder",
                                metadata,
                                SOURCE_OPEN_FAILED,
                            )
                        )
                    continue

                candidates.append(_Candidate(path, relative_path, "file", metadata))

        try:
            walk(root, (), root_anchor.directory_fd)
        except SecureOpenError:
            raise
        except OSError:
            raise SecureOpenError("source_root_unreadable") from None
        candidates.sort(key=lambda item: (item.relative_path.casefold(), item.relative_path))
        return candidates

    def _scan_candidate(
        self,
        workspace_id: str,
        namespace: UUID,
        root: Path,
        root_anchor: RootAnchor,
        candidate: _Candidate,
        previous: ManifestEntry | None,
        *,
        file_count: int,
        selected_bytes: int,
    ) -> tuple[ManifestEntry, tuple[ArchiveMember, ...], int, int]:
        entry_id = _stable_entry_id(namespace, candidate.relative_path)
        metadata_fields = _metadata_fields(candidate.stat_result)
        base = {
            "entry_id": entry_id,
            "workspace_id": workspace_id,
            "relative_path": candidate.relative_path,
            "item_kind": candidate.item_kind,
            "format_category": None,
            **metadata_fields,
            "proposed_course_group": _propose_course_group(candidate.relative_path),
        }

        failure_code = candidate.failure_code or self._path_limit_failure(candidate.relative_path)
        if failure_code is None and candidate.item_kind != "folder" and file_count > self._policy.max_files:
            failure_code = SOURCE_FILE_COUNT_LIMIT
        if failure_code:
            return self._failed_entry(base, failure_code), (), 0, 0

        if candidate.item_kind == "folder":
            return self._failed_entry(base, SOURCE_OPEN_FAILED), (), 0, 0

        metadata = candidate.stat_result
        if metadata is None or not stat.S_ISREG(metadata.st_mode):
            return self._excluded_entry(base, SOURCE_NOT_REGULAR), (), 0, 0

        format_category = classify_format(candidate.relative_path)
        base["format_category"] = format_category
        if format_category is None:
            return self._excluded_entry(base, "unsupported_format"), (), 0, 0

        if selected_bytes + metadata.st_size > self._policy.max_workspace_bytes:
            return self._failed_entry(base, SOURCE_WORKSPACE_SIZE_LIMIT), (), 0, 0

        if (
            previous is not None
            and previous.state is SourceState.PENDING_APPROVAL
            and format_category != "archive"
        ):
            reusable_metadata = self._reusable_metadata(
                root,
                root_anchor,
                candidate.relative_path,
                previous,
            )
            if reusable_metadata is not None:
                base.update(_metadata_fields(reusable_metadata))
                return (
                    ManifestEntry(
                        **base,
                        sha256=previous.sha256,
                        state=SourceState.PENDING_APPROVAL,
                        included=previous.included,
                        inclusion_reason=previous.inclusion_reason,
                    ),
                    (),
                    0,
                    reusable_metadata.st_size,
                )

        outcome = self._read(
            candidate.path,
            root,
            root_anchor,
            candidate.relative_path,
            format_category == "archive",
        )
        if outcome.failure_code:
            return self._failed_entry(base, outcome.failure_code), (), outcome.bytes_hashed, 0
        base.update(_metadata_fields(outcome.stat_result))
        actual_size = outcome.stat_result.st_size if outcome.stat_result is not None else 0
        if selected_bytes + actual_size > self._policy.max_workspace_bytes:
            return (
                self._failed_entry(base, SOURCE_WORKSPACE_SIZE_LIMIT),
                (),
                outcome.bytes_hashed,
                0,
            )

        state = SourceState.PENDING_APPROVAL
        if previous is not None:
            if previous.state is SourceState.APPROVED:
                state = (
                    SourceState.APPROVED
                    if previous.sha256 == outcome.sha256
                    else SourceState.CHANGED
                )
            elif previous.state is SourceState.CHANGED:
                state = SourceState.CHANGED
        return (
            ManifestEntry(
                **base,
                sha256=outcome.sha256,
                state=state,
                included=True,
            ),
            outcome.archive_members,
            outcome.bytes_hashed,
            actual_size,
        )

    def _read(
        self,
        path: Path,
        root: Path,
        root_anchor: RootAnchor,
        relative_path: str,
        inspect_archive: bool,
    ) -> _ReadOutcome:
        bytes_hashed = 0
        members: tuple[ArchiveMember, ...] = ()
        try:
            relative = PurePosixPath(relative_path)
            before = self._secure_file_opener.stat_regular(
                root,
                relative,
                root_anchor=root_anchor,
            )
            digest = hashlib.sha256()
            with self._secure_file_opener.open_regular(
                root,
                relative,
                root_anchor=root_anchor,
            ) as source:
                opened_before = self._secure_file_opener.stat_open_file(source)
                for chunk_index, chunk in enumerate(
                    iter(lambda: source.read(self._policy.hash_chunk_bytes), b"")
                ):
                    digest.update(chunk)
                    bytes_hashed += len(chunk)
                    if self._after_hash_chunk is not None:
                        self._after_hash_chunk(path, chunk_index)
                if inspect_archive:
                    source.seek(0)
                    members = tuple(
                        self._archive_inspector.inspect(
                            source,
                            parent_entry_id="pending",
                        )
                    )
                opened_after = self._secure_file_opener.stat_open_file(source)
            after = self._secure_file_opener.stat_regular(
                root,
                relative,
                root_anchor=root_anchor,
            )
        except SecureOpenError as error:
            return _ReadOutcome(None, bytes_hashed, failure_code=error.code)
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile):
            code = ARCHIVE_INVALID if inspect_archive else SOURCE_OPEN_FAILED
            return _ReadOutcome(None, bytes_hashed, failure_code=code)

        identities = {
            _stat_identity(before),
            _stat_identity(opened_before),
            _stat_identity(opened_after),
            _stat_identity(after),
        }
        if len(identities) != 1:
            return _ReadOutcome(None, bytes_hashed, failure_code=SOURCE_CHANGED_DURING_SCAN)
        return _ReadOutcome(
            digest.hexdigest(),
            bytes_hashed,
            members,
            stat_result=opened_before,
        )

    def _reusable_metadata(
        self,
        root: Path,
        root_anchor: RootAnchor,
        relative_path: str,
        previous: ManifestEntry,
    ) -> os.stat_result | None:
        if previous.sha256 is None:
            return None
        try:
            before = self._secure_file_opener.stat_regular(
                root,
                PurePosixPath(relative_path),
                root_anchor=root_anchor,
            )
            expected = (
                previous.device_id,
                previous.file_id,
                previous.size_bytes,
                previous.modified_ns,
            )
            current = (str(before.st_dev), str(before.st_ino), before.st_size, before.st_mtime_ns)
        except (OSError, SecureOpenError):
            return None
        if expected == current:
            return before
        return None

    def _archive_member_entry(
        self,
        workspace_id: str,
        namespace: UUID,
        parent: ManifestEntry,
        member: ArchiveMember,
        member_index: int,
    ) -> ManifestEntry:
        identity = f"{parent.relative_path}!/{member.display_path}#{member_index}"
        failure_code = member.failure_code
        return ManifestEntry(
            entry_id=_stable_entry_id(namespace, identity),
            workspace_id=workspace_id,
            relative_path=parent.relative_path,
            item_kind="archive_member",
            format_category=classify_format(member.display_path),
            size_bytes=member.size_bytes,
            state=member.state,
            included=False,
            inclusion_reason="archive_preview",
            proposed_course_group=parent.proposed_course_group,
            failure_code=failure_code,
            safe_message=(
                _safe_message(parent.relative_path, failure_code) if failure_code is not None else None
            ),
            archive_parent_entry_id=parent.entry_id,
            archive_member_path=member.display_path,
        )

    def _path_limit_failure(self, relative_path: str) -> str | None:
        if len(PurePosixPath(relative_path).parts) > self._policy.max_depth:
            return SOURCE_DEPTH_LIMIT
        if len(relative_path) > self._policy.max_path_chars:
            return SOURCE_PATH_LIMIT
        return None

    @staticmethod
    def _failed_entry(base: dict[str, object], failure_code: str) -> ManifestEntry:
        relative_path = str(base["relative_path"])
        return ManifestEntry(
            **base,
            state=SourceState.FAILED,
            included=False,
            inclusion_reason=failure_code,
            failure_code=failure_code,
            safe_message=_safe_message(relative_path, failure_code),
        )

    @staticmethod
    def _excluded_entry(base: dict[str, object], reason: str) -> ManifestEntry:
        return ManifestEntry(
            **base,
            state=SourceState.EXCLUDED,
            included=False,
            inclusion_reason=reason,
        )

    @staticmethod
    def _emit(
        emit: Callable[[ScanProgress], None] | None,
        discovered_count: int,
        bytes_hashed: int,
        failure_count: int,
        current_relative_path: str,
    ) -> None:
        if emit is not None:
            emit(
                ScanProgress(
                    discovered_count=discovered_count,
                    bytes_hashed=bytes_hashed,
                    failure_count=failure_count,
                    current_relative_path=current_relative_path,
                )
            )
