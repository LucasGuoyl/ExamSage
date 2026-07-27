from __future__ import annotations

from datetime import datetime
from enum import StrEnum
import re

from pydantic import ConfigDict, Field, field_validator, model_validator

from exam_predictor.workspace.models import FrozenModel, normalize_relative_path


_SENSITIVE_EVIDENCE_PATTERNS = (
    re.compile(r"\bauthorization\s*:\s*(?:bearer|basic)\s+\S+", re.IGNORECASE),
    re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{30,}"),
    re.compile(r"https?://[^\s/@:]+:[^\s/@]+@", re.IGNORECASE),
    re.compile(
        r"https?://\S+[?&](?:x-amz-(?:signature|credential|security-token)|"
        r"x-goog-(?:signature|credential)|sig|signature|token|access_token)=",
        re.IGNORECASE,
    ),
    re.compile(r"(?:[A-Za-z]:[\\/]|\\\\[^\\/]+[\\/])", re.IGNORECASE),
    re.compile(
        r"(?<!\S)/(?:etc|usr|bin|sbin|opt|root|proc|sys|dev|run|mnt|media|home|var|tmp)(?:/|$)",
        re.IGNORECASE,
    ),
    re.compile(r"^/$"),
    re.compile(r"traceback \(most recent call last\)|<(?:openai|google\.genai|httpx)\.", re.IGNORECASE),
    re.compile(
        r"^(?:[A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning)|KeyboardInterrupt|SystemExit):\s+\S+",
        re.MULTILINE,
    ),
)


def validate_safe_evidence_text(value: str) -> str:
    if any(pattern.search(value) for pattern in _SENSITIVE_EVIDENCE_PATTERNS):
        raise ValueError("evidence text must not contain credentials, handles, or absolute paths")
    return value


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

    @field_validator("locator")
    @classmethod
    def validate_locator(cls, value: str) -> str:
        return validate_safe_evidence_text(value)


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

    @field_validator("locator")
    @classmethod
    def validate_locator(cls, value: str) -> str:
        return validate_safe_evidence_text(value)


class EvidenceUnit(EvidenceFrozenModel):
    evidence_unit_id: str
    source_part_id: str
    content: str
    citations: tuple[EvidenceCitation, ...]

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        return validate_safe_evidence_text(value)

    @field_validator("citations")
    @classmethod
    def validate_citations(cls, value: tuple[EvidenceCitation, ...]) -> tuple[EvidenceCitation, ...]:
        if not value:
            raise ValueError("evidence units require at least one citation")
        return value


class CoverageItem(EvidenceFrozenModel):
    topic: str
    covered: bool

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("coverage topics must not be empty")
        return normalized


class CoverageSummary(EvidenceFrozenModel):
    items: tuple[CoverageItem, ...]
    covered_count: int = Field(ge=0)
    total_count: int = Field(ge=1)

    @property
    def coverage_fraction(self) -> float:
        return self.covered_count / self.total_count

    @property
    def is_partial(self) -> bool:
        return 0 < self.covered_count < self.total_count

    @model_validator(mode="after")
    def validate_exact_coverage(self) -> CoverageSummary:
        if self.total_count != len(self.items):
            raise ValueError("total_count must equal the number of coverage items")
        if self.covered_count != sum(item.covered for item in self.items):
            raise ValueError("covered_count must equal the covered coverage items")
        if len({item.topic.casefold() for item in self.items}) != len(self.items):
            raise ValueError("coverage topics must be unique")
        return self


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
            self.coverage is None or not self.coverage.is_partial or not self.evidence_unit_ids
        ):
            raise ValueError("initial snapshots require coverage and evidence dependencies")
        if len(set(self.evidence_unit_ids)) != len(self.evidence_unit_ids):
            raise ValueError("evidence dependencies must be unique")
        return self
