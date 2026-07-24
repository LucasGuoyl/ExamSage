from __future__ import annotations

import os
import secrets
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import BinaryIO

from pydantic import SecretStr

from exam_predictor.workspace.filesystem import (
    RootAnchor,
    SecureFileOpener,
    SecureOpenError,
)
from exam_predictor.workspace.models import (
    ApprovedSource,
    ManifestEntry,
    ScanPolicy,
    SourceState,
    WorkspaceRecord,
    WorkspaceState,
)
from exam_predictor.workspace.policy import DEFAULT_SCAN_POLICY
from exam_predictor.workspace.scanner import RevalidatedEntry, WorkspaceScanner
from exam_predictor.workspace.store import StaleManifestError, WorkspaceStore


class SourceAuthorizationError(RuntimeError):
    def __init__(self, code: str, workspace_id: str, entry_id: str) -> None:
        self.code = code
        self.workspace_id = workspace_id
        self.entry_id = entry_id
        super().__init__(code)


@dataclass(frozen=True)
class _ReadGrant:
    workspace_id: str
    entry_id: str
    relative_path: str
    size_bytes: int
    modified_ns: int
    device_id: str
    file_id: str
    sha256: str
    root_device: str
    root_file_id: str
    expires_at: float


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
        os.path.abspath(right)
    )


def _root_matches(
    workspace: WorkspaceRecord,
    root_anchor: RootAnchor,
) -> bool:
    if (
        root_anchor.identity is None
        or workspace.root_device is None
        or workspace.root_file_id is None
    ):
        return False
    return (
        _same_path(workspace.canonical_root, root_anchor.canonical_root)
        and str(root_anchor.identity[0]) == workspace.root_device
        and str(root_anchor.identity[1]) == workspace.root_file_id
    )


def _entry_matches(
    entry: ManifestEntry,
    validation: RevalidatedEntry,
    approved_sha256: str,
) -> bool:
    return (
        validation.failure_code is None
        and validation.entry_id == entry.entry_id
        and validation.size_bytes == entry.size_bytes
        and validation.modified_ns == entry.modified_ns
        and validation.device_id == entry.device_id
        and validation.file_id == entry.file_id
        and validation.sha256 == entry.sha256
        and validation.sha256 == approved_sha256
    )


def _grant_entry(grant: _ReadGrant) -> ManifestEntry:
    return ManifestEntry(
        entry_id=grant.entry_id,
        workspace_id=grant.workspace_id,
        relative_path=grant.relative_path,
        item_kind="file",
        size_bytes=grant.size_bytes,
        modified_ns=grant.modified_ns,
        device_id=grant.device_id,
        file_id=grant.file_id,
        sha256=grant.sha256,
        state=SourceState.APPROVED,
        included=True,
    )


class WorkspaceTransmissionGate:
    def __init__(
        self,
        store: WorkspaceStore,
        policy: ScanPolicy = DEFAULT_SCAN_POLICY,
        token_ttl_seconds: float = 60.0,
    ) -> None:
        self._store = store
        self._policy = policy
        self._token_ttl_seconds = token_ttl_seconds
        self._scanner = WorkspaceScanner(policy)
        self._opener = SecureFileOpener()
        self._lock = threading.RLock()
        self._read_grants: dict[str, _ReadGrant] = {}

    def authorize(
        self,
        workspace_id: str,
        entry_ids: Sequence[str],
    ) -> Sequence[ApprovedSource]:
        requested = tuple(entry_ids)
        if not requested or len(set(requested)) != len(requested):
            raise SourceAuthorizationError(
                "source_request_invalid",
                workspace_id,
                requested[0] if requested else "",
            )

        workspace = self._store.get_workspace(workspace_id)
        if workspace is None:
            raise SourceAuthorizationError(
                "workspace_not_found",
                workspace_id,
                requested[0],
            )
        approval = self._store.get_approval(workspace_id)
        if approval is None or workspace.state is not WorkspaceState.APPROVED:
            raise SourceAuthorizationError(
                "source_approval_required",
                workspace_id,
                requested[0],
            )
        if approval.policy_version != self._policy.policy_version:
            raise SourceAuthorizationError(
                "source_approval_stale_policy",
                workspace_id,
                requested[0],
            )

        revision = self._store.get_manifest(workspace_id, approval.revision_id)
        manifest_by_id = {entry.entry_id: entry for entry in revision.entries}
        approved_by_id = {
            approved.entry_id: approved.sha256 for approved in approval.entries
        }
        selected: list[tuple[ManifestEntry, str]] = []
        for entry_id in requested:
            entry = manifest_by_id.get(entry_id)
            approved_sha256 = approved_by_id.get(entry_id)
            if entry is None or approved_sha256 is None:
                raise SourceAuthorizationError(
                    "source_not_approved",
                    workspace_id,
                    entry_id,
                )
            if entry.sha256 != approved_sha256:
                raise SourceAuthorizationError(
                    "source_approval_mismatch",
                    workspace_id,
                    entry_id,
                )
            if (
                entry.item_kind != "file"
                or entry.archive_parent_entry_id is not None
                or not entry.included
                or entry.state is not SourceState.APPROVED
                or entry.modified_ns is None
                or entry.device_id is None
                or entry.file_id is None
                or entry.sha256 is None
            ):
                raise SourceAuthorizationError(
                    "source_not_approved",
                    workspace_id,
                    entry_id,
                )
            selected.append((entry, approved_sha256))

        root = self._store.source_root(workspace_id)
        changed_entry_ids: tuple[str, ...] = ()
        failure_code: str | None = None
        failure_entry_id = requested[0]
        validated_by_id: dict[str, RevalidatedEntry] = {}
        try:
            with self._opener.anchor_root(root) as root_anchor:
                if not _root_matches(workspace, root_anchor):
                    failure_code = "source_root_identity_changed"
                    changed_entry_ids = requested
                else:
                    validation = self._scanner.revalidate_entries(
                        root,
                        [entry for entry, _ in selected],
                        root_anchor=root_anchor,
                    )
                    if (
                        not _same_path(validation.canonical_root, root_anchor.canonical_root)
                        or validation.root_device != workspace.root_device
                        or validation.root_file_id != workspace.root_file_id
                    ):
                        failure_code = "source_root_identity_changed"
                        changed_entry_ids = requested
                    else:
                        validated_by_id = {
                            item.entry_id: item for item in validation.entries
                        }
                        mismatched = tuple(
                            entry.entry_id
                            for entry, approved_sha256 in selected
                            if (
                                (current := validated_by_id.get(entry.entry_id)) is None
                                or not _entry_matches(
                                    entry,
                                    current,
                                    approved_sha256,
                                )
                            )
                        )
                        if mismatched:
                            failure_code = "approved_source_changed"
                            failure_entry_id = mismatched[0]
                            changed_entry_ids = mismatched
        except SecureOpenError as error:
            failure_code = (
                "source_root_invalid"
                if error.code
                in {
                    "source_link_or_reparse",
                    "source_open_failed",
                    "source_root_invalid",
                    "source_root_identity_unavailable",
                }
                else "approved_source_changed"
            )
            changed_entry_ids = requested

        if failure_code is not None:
            self._mark_changed(workspace, changed_entry_ids)
            raise SourceAuthorizationError(
                failure_code,
                workspace_id,
                failure_entry_id,
            ) from None

        self._store.record_access_verified(workspace_id, datetime.now(UTC))
        expires_at = monotonic() + self._token_ttl_seconds
        descriptors: list[ApprovedSource] = []
        grants: list[tuple[str, _ReadGrant]] = []
        for entry, approved_sha256 in selected:
            current = validated_by_id[entry.entry_id]
            token = secrets.token_urlsafe(32)
            grants.append(
                (
                    token,
                    _ReadGrant(
                        workspace_id=workspace_id,
                        entry_id=entry.entry_id,
                        relative_path=entry.relative_path,
                        size_bytes=current.size_bytes,
                        modified_ns=current.modified_ns,  # type: ignore[arg-type]
                        device_id=current.device_id,  # type: ignore[arg-type]
                        file_id=current.file_id,  # type: ignore[arg-type]
                        sha256=approved_sha256,
                        root_device=workspace.root_device,  # type: ignore[arg-type]
                        root_file_id=workspace.root_file_id,  # type: ignore[arg-type]
                        expires_at=expires_at,
                    ),
                )
            )
            descriptors.append(
                ApprovedSource(
                    workspace_id=workspace_id,
                    entry_id=entry.entry_id,
                    relative_path=entry.relative_path,
                    size_bytes=current.size_bytes,
                    sha256=approved_sha256,
                    read_token=SecretStr(token),
                )
            )
        with self._lock:
            self._read_grants.update(grants)
        return tuple(descriptors)

    @contextmanager
    def open_approved(
        self,
        read_token: SecretStr | str,
    ) -> Iterator[BinaryIO]:
        token = (
            read_token.get_secret_value()
            if isinstance(read_token, SecretStr)
            else read_token
        )
        with self._lock:
            grant = self._read_grants.pop(token, None)
        if grant is None:
            raise SourceAuthorizationError("read_token_invalid", "", "")
        if monotonic() >= grant.expires_at:
            raise SourceAuthorizationError(
                "read_token_expired",
                grant.workspace_id,
                grant.entry_id,
            )

        workspace = self._store.get_workspace(grant.workspace_id)
        failure_code: str | None = None
        if workspace is None:
            failure_code = "source_root_invalid"
        elif (
            workspace.root_device != grant.root_device
            or workspace.root_file_id != grant.root_file_id
        ):
            failure_code = "source_root_identity_changed"

        if failure_code is None and workspace is not None:
            root = self._store.source_root(grant.workspace_id)
            try:
                with self._opener.anchor_root(root) as root_anchor:
                    if not _root_matches(workspace, root_anchor):
                        failure_code = "source_root_identity_changed"
                    else:
                        with self._scanner.open_revalidated_entry(
                            root,
                            _grant_entry(grant),
                            root_anchor=root_anchor,
                        ) as (current, source):
                            if not _entry_matches(
                                _grant_entry(grant),
                                current,
                                grant.sha256,
                            ):
                                failure_code = "approved_source_changed"
                            else:
                                yield source
            except SecureOpenError:
                failure_code = "approved_source_changed"

        if failure_code is not None:
            if workspace is not None:
                self._mark_changed(workspace, (grant.entry_id,))
            raise SourceAuthorizationError(
                failure_code,
                grant.workspace_id,
                grant.entry_id,
            ) from None

    def _mark_changed(
        self,
        workspace: WorkspaceRecord,
        entry_ids: Sequence[str],
    ) -> None:
        if not entry_ids or workspace.current_draft_revision_id is None:
            return
        update_failed = False
        try:
            self._store.mark_revision_attention_if_current(
                workspace.workspace_id,
                workspace.current_draft_revision_id,
                entry_ids,
                "approved_source_changed",
            )
        except StaleManifestError:
            current = self._store.get_workspace(workspace.workspace_id)
            if current is None or current.state is not WorkspaceState.NEEDS_ATTENTION:
                update_failed = True
        except Exception:
            update_failed = True
        if update_failed:
            raise SourceAuthorizationError(
                "source_state_update_failed",
                workspace.workspace_id,
                entry_ids[0],
            ) from None
