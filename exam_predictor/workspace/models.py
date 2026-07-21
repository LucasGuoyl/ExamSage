from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal
from unicodedata import category

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class SourceMode(StrEnum):
    NATIVE_FOLDER = "native_folder"
    BROWSER_SNAPSHOT = "browser_snapshot"


class WorkspaceState(StrEnum):
    READY = "ready"
    SCANNING = "scanning"
    APPROVAL_REQUIRED = "approval_required"
    APPROVED = "approved"
    NEEDS_ATTENTION = "needs_attention"
    DELETING = "deleting"
    CLEANUP_PENDING = "cleanup_pending"


class SourceState(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    EXCLUDED = "excluded"
    FAILED = "failed"
    CHANGED = "changed"
    REMOVED = "removed"


class WorkspaceJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


def normalize_relative_path(value: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise ValueError("relative_path must be a POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
        raise ValueError("relative_path must remain within the workspace")
    normalized = path.as_posix()
    if normalized in {".", ""}:
        raise ValueError("relative_path must identify an item")
    return normalized


class ScanPolicy(FrozenModel):
    policy_version: str = "workspace-v1"
    max_workspace_bytes: int = 1_073_741_824
    max_files: int = 20_000
    max_depth: int = 32
    max_path_chars: int = 1_024
    max_archive_members: int = 10_000
    max_archive_expanded_bytes: int = 2_147_483_648
    max_archive_ratio: float = 200.0
    hash_chunk_bytes: int = 1_048_576


class ManifestEntry(FrozenModel):
    entry_id: str
    workspace_id: str
    relative_path: str
    item_kind: str
    format_category: str | None = None
    size_bytes: int = Field(ge=0)
    modified_ns: int | None = None
    device_id: str | None = None
    file_id: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    state: SourceState
    included: bool
    inclusion_reason: str | None = None
    proposed_course_group: str = "unclassified"
    failure_code: str | None = None
    safe_message: str | None = None
    archive_parent_entry_id: str | None = None
    archive_member_path: str | None = None

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return normalize_relative_path(value)


class ManifestRevision(FrozenModel):
    revision_id: str
    workspace_id: str
    parent_revision_id: str | None = None
    scan_job_id: str | None = None
    policy_version: str
    entries: tuple[ManifestEntry, ...]
    created_at: datetime


class ApprovedEntryHash(FrozenModel):
    entry_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ApprovalRecord(FrozenModel):
    approval_id: str
    workspace_id: str
    revision_id: str
    entries: tuple[ApprovedEntryHash, ...]
    policy_version: str
    approved_at: datetime


class WorkspaceRecord(FrozenModel):
    workspace_id: str
    display_name: str
    source_mode: SourceMode
    canonical_root: Path
    root_device: str | None = None
    root_file_id: str | None = None
    state: WorkspaceState
    current_draft_revision_id: str | None = None
    current_approved_revision_id: str | None = None
    created_at: datetime
    updated_at: datetime
    last_scanned_at: datetime | None = None
    last_access_verified_at: datetime | None = None


class WorkspaceSummary(FrozenModel):
    workspace_id: str
    display_name: str
    source_mode: SourceMode
    state: WorkspaceState
    counts: dict[SourceState, int]
    updated_at: datetime


class WorkspaceDetail(WorkspaceSummary):
    current_draft_revision_id: str | None = None
    current_approved_revision_id: str | None = None
    created_at: datetime
    last_scanned_at: datetime | None = None
    last_access_verified_at: datetime | None = None


class ScanProgress(FrozenModel):
    discovered_count: int = Field(ge=0)
    bytes_hashed: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    current_relative_path: str | None = None


class ScanResult(FrozenModel):
    workspace_id: str
    entries: tuple[ManifestEntry, ...]
    discovered_count: int = Field(ge=0)
    bytes_hashed: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    completed_at: datetime


class WorkspaceJob(FrozenModel):
    job_id: str
    workspace_id: str
    job_kind: str
    status: WorkspaceJobStatus
    idempotency_key: str
    safe_error_code: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class WorkspaceEvent(FrozenModel):
    sequence: int = Field(ge=1)
    job_id: str
    event_type: str
    message: str
    payload: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    created_at: datetime


class ArchiveMember(FrozenModel):
    parent_entry_id: str
    display_path: str
    item_kind: str
    size_bytes: int = Field(ge=0)
    compressed_bytes: int = Field(ge=0)
    state: SourceState
    failure_code: str | None = None

    @field_validator("display_path")
    @classmethod
    def normalize_display_path(cls, value: str) -> str:
        return "".join(character for character in value if category(character) != "Cc")[:1_024]


class CleanupRecord(FrozenModel):
    cleanup_id: str
    workspace_id: str
    owned_relative_path: str
    safe_error_code: str
    attempt_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


class ApprovedSource(FrozenModel):
    workspace_id: str
    entry_id: str
    relative_path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    read_token: SecretStr


class ManifestPage(FrozenModel):
    items: tuple[ManifestEntry, ...]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=500)
    counts: dict[SourceState, int]


class EntryInclusionRequest(FrozenModel):
    revision_id: str
    included: bool
    subtree: bool = False


class ApprovalRequest(FrozenModel):
    revision_id: str


class DeleteAllWorkspacesRequest(FrozenModel):
    confirmation: Literal["DELETE ALL"]
