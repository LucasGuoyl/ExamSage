"""Read-only visibility and identity-bound cleanup for legacy upload copies."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from exam_predictor.workspace.browser_intake import (
    OwnedTreeRemovalError,
    OwnedTreeRemover,
)
from exam_predictor.workspace.filesystem import (
    OwnedArtifactFilesystem,
    OwnedDirectoryAnchor,
    OwnedFilesystemError,
    OwnedMutationFile,
    is_reparse_point,
)


_SESSION_ID = re.compile(r"[0-9a-f]{32}\Z")
_MAX_DIAGNOSTIC_ENTRIES = 100_000
_LEASE_PREFIX = ".examsage-active-"
_CLEANUP_CLAIM_PREFIX = ".examsage-cleanup-"


class LegacyIntakeError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class LegacyIntakeLease:
    """One process-held lease proving a legacy intake session is active."""

    def __init__(
        self,
        stack: ExitStack,
        filesystem: OwnedArtifactFilesystem,
        data_root: Path,
        intake: OwnedDirectoryAnchor,
        session: OwnedDirectoryAnchor,
        lock_file: OwnedMutationFile | None,
        locked: bool,
    ) -> None:
        self._stack = stack
        self._filesystem = filesystem
        self._data_root = data_root
        self._intake = intake
        self._session = session
        self._lock_file = lock_file
        self._descriptor = (
            intake.descriptor if lock_file is None else lock_file.descriptor
        )
        self._locked = locked
        self._closed = False

    @property
    def session_identity(self) -> tuple[int, int]:
        return self._session.identity

    @property
    def session_id(self) -> str:
        return self._session.path.name

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if not self._locked:
                return
            if os.name == "nt":
                import msvcrt

                os.lseek(self._descriptor, 0, os.SEEK_SET)
                msvcrt.locking(self._descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        finally:
            self._stack.close()

    def verify(self) -> None:
        self._filesystem.verify_directory_anchor(self._session)
        if self._lock_file is not None:
            self._filesystem.verify_mutation_file(self._lock_file)

    def cleanup_claim_exists(self) -> bool:
        self.verify()
        try:
            with self._filesystem.open_mutation_file(
                self._intake,
                _cleanup_claim_name(self.session_id),
                expected_parent_identity=self._intake.identity,
            ) as claim:
                if _read_claim_payload(claim) != _claim_payload(
                    self.session_id,
                    self.session_identity,
                    _removal_identity(self._data_root, self.session_id),
                ):
                    raise LegacyIntakeError("legacy_intake_unverified")
                return True
        except OwnedFilesystemError as error:
            if error.code == "owned_not_found":
                return False
            raise LegacyIntakeError("legacy_intake_unverified") from None

    def claim_cleanup(self) -> None:
        if self.cleanup_claim_exists():
            return
        try:
            with self._filesystem.create_temporary_file(
                self._intake,
                _cleanup_claim_name(self.session_id),
                expected_parent_identity=self._intake.identity,
            ) as claim:
                os.write(
                    claim.descriptor,
                    _claim_payload(
                        self.session_id,
                        self.session_identity,
                        _removal_identity(self._data_root, self.session_id),
                    ),
                )
                os.fsync(claim.descriptor)
                claim.release()
            self.verify()
        except OwnedFilesystemError:
            raise LegacyIntakeError("legacy_intake_unverified") from None

    def __enter__(self) -> LegacyIntakeLease:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class LegacyIntakeSummary:
    session_count: int
    file_count: int
    total_bytes: int
    unknown_entry_count: int = 0
    unsafe_session_count: int = 0
    session_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LegacyIntakeCleanupResult:
    deleted_session_ids: tuple[str, ...]
    deleted_bytes: int


@dataclass(frozen=True)
class _SessionClaim:
    relative_path: str
    device_id: str
    file_id: str


@dataclass(frozen=True)
class _SessionScan:
    session_id: str
    file_count: int
    total_bytes: int
    identity: tuple[int, int]


def diagnose_legacy_intake(data_root: Path) -> LegacyIntakeSummary:
    """Count only fixed-root, non-link legacy session copies without changing them."""
    intake = _verified_intake_root(data_root, missing_ok=True)
    if intake is None:
        return LegacyIntakeSummary(0, 0, 0)
    scans: list[_SessionScan] = []
    unknown = 0
    unsafe = 0
    try:
        entries = sorted(os.scandir(intake), key=lambda item: item.name.casefold())
    except OSError:
        raise LegacyIntakeError("legacy_intake_unavailable") from None
    for entry in entries:
        if entry.name.startswith((_LEASE_PREFIX, _CLEANUP_CLAIM_PREFIX)):
            continue
        if _SESSION_ID.fullmatch(entry.name) is None:
            unknown += 1
            continue
        try:
            scans.append(_scan_session(intake, entry.name))
        except LegacyIntakeError:
            unsafe += 1
    recovered, unsafe_recovery = _discover_recoverable_sessions(
        Path(data_root).absolute(),
        intake,
        {item.session_id for item in scans},
    )
    scans.extend(recovered)
    unsafe += unsafe_recovery
    return LegacyIntakeSummary(
        session_count=len(scans),
        file_count=sum(item.file_count for item in scans),
        total_bytes=sum(item.total_bytes for item in scans),
        unknown_entry_count=unknown,
        unsafe_session_count=unsafe,
        session_ids=tuple(item.session_id for item in scans),
    )


def acquire_legacy_intake_lease(
    data_root: Path,
    session_id: str,
) -> LegacyIntakeLease:
    """Hold a cross-process lease while the legacy UI uses one intake copy."""
    if _SESSION_ID.fullmatch(session_id) is None:
        raise LegacyIntakeError("legacy_intake_session_invalid")
    intake = _verified_intake_root(data_root, missing_ok=False)
    assert intake is not None
    _scan_session(intake, session_id)
    return _acquire_session_lease(Path(data_root).absolute(), session_id)


def cleanup_legacy_intake(
    data_root: Path,
    *,
    session_ids: Iterable[str] | None = None,
    active_session_ids: Iterable[str] = (),
    before_remove: Callable[[str, Path], None] | None = None,
    after_isolate: Callable[[str, Path], None] | None = None,
) -> LegacyIntakeCleanupResult:
    """Delete selected verified copies, never native sources or unknown intake paths."""
    root = Path(data_root).absolute()
    intake = _verified_intake_root(root, missing_ok=False)
    assert intake is not None
    summary = diagnose_legacy_intake(root)
    selected = tuple(summary.session_ids if session_ids is None else session_ids)
    if len(set(selected)) != len(selected) or any(
        not isinstance(item, str) or _SESSION_ID.fullmatch(item) is None
        for item in selected
    ):
        raise LegacyIntakeError("legacy_intake_session_invalid")
    active = frozenset(active_session_ids)
    if active.intersection(selected):
        raise LegacyIntakeError("legacy_intake_active")

    verified = set(summary.session_ids)
    if any(item not in verified for item in selected):
        raise LegacyIntakeError("legacy_intake_unverified")

    recovered = _recover_isolated_sessions(root, selected)
    remaining = tuple(item for item in selected if item not in recovered)
    if not remaining:
        return LegacyIntakeCleanupResult(
            tuple(item for item in selected if item in recovered),
            sum(recovered.values()),
        )
    selected = remaining

    leases: list[LegacyIntakeLease] = []
    try:
        for index, session_id in enumerate(selected):
            leases.append(
                _acquire_session_lease(
                    root,
                    session_id,
                    for_cleanup=True,
                    acquire_lock=os.name == "nt" or index == 0,
                )
            )
        for lease in leases:
            lease.claim_cleanup()
        scans = {item: _scan_session(intake, item) for item in selected}
        session_identities = {
            lease.session_id: lease.session_identity for lease in leases
        }
        for lease in reversed(leases):
            lease.close()
        leases.clear()
        remover = OwnedTreeRemover(root)
        deleted: list[str] = []
        deleted_bytes = 0
        for session_id in selected:
            scan = scans[session_id]
            session_path = intake / session_id
            if before_remove is not None:
                before_remove(session_id, session_path)
            try:
                quarantine = _isolate_session(
                    root,
                    session_id,
                    session_identities[session_id],
                )
                if quarantine is None:
                    raise LegacyIntakeError("legacy_intake_identity_changed")
                identity = remover.capture_identity(f"intake/{quarantine}")
                claim_data = _load_recovery_claim(root, session_id)
                if claim_data is None or tuple(
                    int(value) for value in identity
                ) != claim_data[1]:
                    raise LegacyIntakeError("legacy_intake_identity_changed")
            except OwnedFilesystemError:
                raise LegacyIntakeError("legacy_intake_identity_changed") from None
            if after_isolate is not None:
                after_isolate(session_id, intake / quarantine)
            claim = _SessionClaim(
                relative_path=f"intake/{quarantine}",
                device_id=identity[0],
                file_id=identity[1],
            )
            try:
                remover(claim)
            except OwnedTreeRemovalError as error:
                if error.code == "cleanup_identity_changed":
                    code = "legacy_intake_identity_changed"
                elif error.code in {
                    "cleanup_link_or_reparse",
                    "cleanup_path_invalid",
                    "cleanup_special_file",
                }:
                    code = "legacy_intake_unverified"
                else:
                    code = "legacy_intake_cleanup_failed"
                raise LegacyIntakeError(code) from None
            deleted.append(session_id)
            deleted_bytes += scan.total_bytes
            _remove_cleanup_claim(root, session_id)
            if os.name == "nt":
                _remove_reservation(root, _lease_name(session_id))
        all_deleted = tuple(
            item for item in summary.session_ids if item in recovered or item in deleted
        )
        return LegacyIntakeCleanupResult(
            all_deleted,
            deleted_bytes + sum(recovered.values()),
        )
    finally:
        for lease in reversed(leases):
            lease.close()


def _acquire_session_lease(
    data_root: Path,
    session_id: str,
    *,
    for_cleanup: bool = False,
    acquire_lock: bool = True,
) -> LegacyIntakeLease:
    stack = ExitStack()
    filesystem = OwnedArtifactFilesystem()
    try:
        root = stack.enter_context(filesystem.anchor_directory(data_root))
        intake = stack.enter_context(
            filesystem.anchor_child_directory(
                root,
                "intake",
                expected_parent_identity=root.identity,
            )
        )
        session = stack.enter_context(
            filesystem.anchor_child_directory(
                intake,
                session_id,
                expected_parent_identity=intake.identity,
            )
        )
        lock_file: OwnedMutationFile | None = None
        if os.name == "nt":
            import msvcrt

            lock_file = stack.enter_context(
                filesystem.open_or_create_mutation_file(
                    intake,
                    _lease_name(session_id),
                    expected_parent_identity=intake.identity,
                )
            )
            if os.fstat(lock_file.descriptor).st_size == 0:
                os.write(lock_file.descriptor, b"\0")
                os.fsync(lock_file.descriptor)
            os.lseek(lock_file.descriptor, 0, os.SEEK_SET)
            msvcrt.locking(lock_file.descriptor, msvcrt.LK_NBLCK, 1)
            descriptor = lock_file.descriptor
        elif acquire_lock:
            import fcntl

            descriptor = intake.descriptor
            operation = fcntl.LOCK_EX if for_cleanup else fcntl.LOCK_SH
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        stack.close()
        raise LegacyIntakeError("legacy_intake_active") from None
    except OwnedFilesystemError:
        stack.close()
        raise LegacyIntakeError("legacy_intake_unverified") from None
    lease = LegacyIntakeLease(
        stack,
        filesystem,
        data_root,
        intake,
        session,
        lock_file,
        acquire_lock,
    )
    try:
        lease.verify()
        if lease.cleanup_claim_exists() and not for_cleanup:
            raise LegacyIntakeError("legacy_intake_active")
    except Exception:
        lease.close()
        raise
    return lease


def _cleanup_claim_name(session_id: str) -> str:
    return f"{_CLEANUP_CLAIM_PREFIX}{session_id}"


def _lease_name(session_id: str) -> str:
    return f"{_LEASE_PREFIX}{session_id}.lock"


def _isolate_session(
    data_root: Path,
    session_id: str,
    expected_identity: tuple[int, int],
) -> str | None:
    filesystem = OwnedArtifactFilesystem()
    with filesystem.anchor_directory(data_root) as root:
        with filesystem.anchor_child_directory(
            root,
            "intake",
            expected_parent_identity=root.identity,
        ) as intake:
            return filesystem.quarantine_directory_tree(
                intake,
                session_id,
                expected_parent_identity=intake.identity,
                expected_child_identity=expected_identity,
            )


def _remove_cleanup_claim(data_root: Path, session_id: str) -> None:
    _remove_reservation(data_root, _cleanup_claim_name(session_id))


def _remove_reservation(data_root: Path, name: str) -> None:
    filesystem = OwnedArtifactFilesystem()
    with filesystem.anchor_directory(data_root) as root:
        with filesystem.anchor_child_directory(
            root,
            "intake",
            expected_parent_identity=root.identity,
        ) as intake:
            try:
                with filesystem.open_mutation_file(
                    intake,
                    name,
                    expected_parent_identity=intake.identity,
                ) as claim:
                    claim_identity = claim.identity
                filesystem.delete_reserved_file(
                    intake,
                    name,
                    expected_parent_identity=intake.identity,
                    expected_source_identity=claim_identity,
                )
            except OwnedFilesystemError as error:
                if error.code != "owned_not_found":
                    raise LegacyIntakeError("legacy_intake_cleanup_failed") from None


def _removal_identity(data_root: Path, session_id: str) -> tuple[int, int]:
    identity = OwnedTreeRemover(data_root).capture_identity(
        f"intake/{session_id}"
    )
    return int(identity[0]), int(identity[1])


def _claim_payload(
    session_id: str,
    identity: tuple[int, int],
    removal_identity: tuple[int, int],
) -> bytes:
    quarantine = OwnedArtifactFilesystem.directory_quarantine_name(
        session_id,
        identity,
    )
    return (
        f"v1\n{session_id}\n{identity[0]}\n{identity[1]}\n"
        f"{removal_identity[0]}\n{removal_identity[1]}\n{quarantine}\n"
    ).encode()


def _read_claim_payload(claim: OwnedMutationFile) -> bytes:
    os.lseek(claim.descriptor, 0, os.SEEK_SET)
    payload = os.read(claim.descriptor, 513)
    if len(payload) > 512:
        raise LegacyIntakeError("legacy_intake_unverified")
    return payload


def _parse_claim_payload(
    payload: bytes,
    session_id: str,
) -> tuple[tuple[int, int], tuple[int, int], str]:
    try:
        (
            version,
            claimed_session,
            device,
            file_id,
            removal_device,
            removal_file_id,
            quarantine,
            empty,
        ) = (
            payload.decode("ascii").split("\n")
        )
        identity = (int(device), int(file_id))
        removal_identity = (int(removal_device), int(removal_file_id))
    except (UnicodeDecodeError, ValueError):
        raise LegacyIntakeError("legacy_intake_unverified") from None
    expected = OwnedArtifactFilesystem.directory_quarantine_name(
        session_id,
        identity,
    )
    if (
        version != "v1"
        or claimed_session != session_id
        or empty != ""
        or quarantine != expected
        or any(value < 0 for value in (*identity, *removal_identity))
    ):
        raise LegacyIntakeError("legacy_intake_unverified")
    return identity, removal_identity, quarantine


def _load_recovery_claim(
    data_root: Path,
    session_id: str,
) -> tuple[tuple[int, int], tuple[int, int], str] | None:
    filesystem = OwnedArtifactFilesystem()
    try:
        with filesystem.anchor_directory(data_root) as root:
            with filesystem.anchor_child_directory(
                root,
                "intake",
                expected_parent_identity=root.identity,
            ) as intake:
                with filesystem.open_mutation_file(
                    intake,
                    _cleanup_claim_name(session_id),
                    expected_parent_identity=intake.identity,
                ) as claim:
                    return _parse_claim_payload(
                        _read_claim_payload(claim),
                        session_id,
                    )
    except OwnedFilesystemError as error:
        if error.code == "owned_not_found":
            return None
        raise LegacyIntakeError("legacy_intake_unverified") from None


def _verified_recovery_scan(
    data_root: Path,
    intake_path: Path,
    session_id: str,
) -> _SessionScan | None:
    claim = _load_recovery_claim(data_root, session_id)
    if claim is None or (intake_path / session_id).exists():
        return None
    identity, _removal_identity_value, quarantine = claim
    filesystem = OwnedArtifactFilesystem()
    try:
        with filesystem.anchor_directory(data_root) as root:
            with filesystem.anchor_child_directory(
                root,
                "intake",
                expected_parent_identity=root.identity,
            ) as intake:
                with filesystem.anchor_child_directory(
                    intake,
                    quarantine,
                    expected_parent_identity=intake.identity,
                ) as isolated:
                    if isolated.identity != identity:
                        raise LegacyIntakeError("legacy_intake_unverified")
        return _scan_directory(intake_path / quarantine, session_id)
    except OwnedFilesystemError:
        raise LegacyIntakeError("legacy_intake_unverified") from None


def _discover_recoverable_sessions(
    data_root: Path,
    intake: Path,
    known_session_ids: set[str],
) -> tuple[list[_SessionScan], int]:
    scans: list[_SessionScan] = []
    unsafe = 0
    try:
        names = tuple(entry.name for entry in os.scandir(intake))
    except OSError:
        raise LegacyIntakeError("legacy_intake_unavailable") from None
    for name in names:
        if not name.startswith(_CLEANUP_CLAIM_PREFIX):
            continue
        session_id = name.removeprefix(_CLEANUP_CLAIM_PREFIX)
        if _SESSION_ID.fullmatch(session_id) is None or session_id in known_session_ids:
            continue
        try:
            scan = _verified_recovery_scan(data_root, intake, session_id)
        except LegacyIntakeError:
            unsafe += 1
            continue
        if scan is not None:
            scans.append(scan)
    return scans, unsafe


def _recover_isolated_sessions(
    data_root: Path,
    session_ids: Iterable[str],
) -> dict[str, int]:
    intake = data_root / "intake"
    remover = OwnedTreeRemover(data_root)
    recovered: dict[str, int] = {}
    for session_id in session_ids:
        scan = _verified_recovery_scan(data_root, intake, session_id)
        if scan is None:
            continue
        persisted_identity, removal_identity, quarantine = _load_recovery_claim(
            data_root,
            session_id,
        ) or (None, None, None)
        if (
            quarantine is None
            or persisted_identity is None
            or removal_identity is None
        ):
            raise LegacyIntakeError("legacy_intake_unverified")
        identity = remover.capture_identity(f"intake/{quarantine}")
        if tuple(int(value) for value in identity) != removal_identity:
            raise LegacyIntakeError("legacy_intake_identity_changed")
        try:
            remover(
                _SessionClaim(
                    relative_path=f"intake/{quarantine}",
                    device_id=identity[0],
                    file_id=identity[1],
                )
            )
        except OwnedTreeRemovalError:
            raise LegacyIntakeError("legacy_intake_cleanup_failed") from None
        _remove_cleanup_claim(data_root, session_id)
        if os.name == "nt":
            _remove_reservation(data_root, _lease_name(session_id))
        recovered[session_id] = scan.total_bytes
    return recovered


def _verified_intake_root(data_root: Path, *, missing_ok: bool) -> Path | None:
    root = Path(data_root).absolute()
    if not root.exists():
        if missing_ok:
            return None
        raise LegacyIntakeError("legacy_intake_unavailable")
    if not root.is_dir() or root.is_symlink() or is_reparse_point(root):
        raise LegacyIntakeError("legacy_intake_unverified")
    intake = root / "intake"
    if not intake.exists():
        if missing_ok:
            return None
        raise LegacyIntakeError("legacy_intake_unavailable")
    if not intake.is_dir() or intake.is_symlink() or is_reparse_point(intake):
        raise LegacyIntakeError("legacy_intake_unverified")
    return intake


def _scan_session(intake: Path, session_id: str) -> _SessionScan:
    return _scan_directory(intake / session_id, session_id)


def _scan_directory(path: Path, session_id: str) -> _SessionScan:
    try:
        named = path.stat(follow_symlinks=False)
    except OSError:
        raise LegacyIntakeError("legacy_intake_unverified") from None
    if (
        not stat.S_ISDIR(named.st_mode)
        or path.is_symlink()
        or is_reparse_point(path)
    ):
        raise LegacyIntakeError("legacy_intake_unverified")
    file_count = 0
    total_bytes = 0
    visited = 0
    pending = [path]
    while pending:
        current = pending.pop()
        try:
            entries = tuple(os.scandir(current))
        except OSError:
            raise LegacyIntakeError("legacy_intake_unverified") from None
        for entry in entries:
            visited += 1
            if visited > _MAX_DIAGNOSTIC_ENTRIES:
                raise LegacyIntakeError("legacy_intake_unverified")
            child = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                raise LegacyIntakeError("legacy_intake_unverified") from None
            if entry.is_symlink() or is_reparse_point(child):
                raise LegacyIntakeError("legacy_intake_unverified")
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(child)
            elif stat.S_ISREG(metadata.st_mode):
                file_count += 1
                total_bytes += metadata.st_size
            else:
                raise LegacyIntakeError("legacy_intake_unverified")
    try:
        current = path.stat(follow_symlinks=False)
    except OSError:
        raise LegacyIntakeError("legacy_intake_unverified") from None
    if not os.path.samestat(named, current):
        raise LegacyIntakeError("legacy_intake_identity_changed")
    return _SessionScan(
        session_id=session_id,
        file_count=file_count,
        total_bytes=total_bytes,
        identity=(named.st_dev, named.st_ino),
    )
