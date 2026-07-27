from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import ConfigDict, Field, field_validator, model_validator

from exam_predictor.workspace.models import FrozenModel, normalize_relative_path


class PartState(StrEnum):
    PLANNED = "planned"
    AUTHORIZED = "authorized"
    PREPARED = "prepared"
    RUNNING = "running"
    PROCESSED = "processed"
    RETRY_WAIT = "retry_wait"
    FAILED = "failed"
    INVALIDATED = "invalidated"


class SnapshotStatus(StrEnum):
    INITIAL = "initial"
    COMPLETE = "complete"


class EvidenceFrozenModel(FrozenModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourcePartPlan(EvidenceFrozenModel):
    part_id: str
    workspace_id: str
    revision_id: str
    entry_id: str
    relative_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    part_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordinal: int = Field(ge=0)
    locator: str
    media_type: str
    size_bytes: int = Field(ge=0)
    scheduling_class: str
    priority: int = Field(ge=0)
    state: PartState
    idempotency_key: str

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return normalize_relative_path(value)


class EvidenceCitation(EvidenceFrozenModel):
    citation_id: str
    evidence_unit_id: str
    source_part_id: str
    relative_path: str
    locator: str

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return normalize_relative_path(value)


class EvidenceUnit(EvidenceFrozenModel):
    evidence_unit_id: str
    source_part_id: str
    content: str
    citations: tuple[EvidenceCitation, ...]


class CoverageItem(EvidenceFrozenModel):
    topic: str
    covered: bool


class CoverageSummary(EvidenceFrozenModel):
    items: tuple[CoverageItem, ...]


class KnowledgeNode(EvidenceFrozenModel):
    node_id: str
    title: str
    focus_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_unit_ids: tuple[str, ...]

    @field_validator("evidence_unit_ids")
    @classmethod
    def validate_evidence_dependencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("evidence dependencies must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("evidence dependencies must be unique")
        return value


class StudyMapSnapshot(EvidenceFrozenModel):
    snapshot_id: str
    workspace_id: str
    revision_id: str
    status: SnapshotStatus
    nodes: tuple[KnowledgeNode, ...]
    coverage: CoverageSummary | None
    evidence_unit_ids: tuple[str, ...]
    created_at: datetime

    @model_validator(mode="after")
    def validate_initial_dependencies(self) -> StudyMapSnapshot:
        if self.status is SnapshotStatus.INITIAL and (
            self.coverage is None or not self.coverage.items or not self.evidence_unit_ids
        ):
            raise ValueError("initial snapshots require coverage and evidence dependencies")
        if len(set(self.evidence_unit_ids)) != len(self.evidence_unit_ids):
            raise ValueError("evidence dependencies must be unique")
        return self
