from __future__ import annotations

from datetime import datetime
from enum import StrEnum
import re
from typing import Literal

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
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError("evidence text must contain valid Unicode scalar values")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("evidence text must contain valid Unicode scalar values") from None
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
    entry_id: str | None = None
    relative_path: str | None = None
    approved_bytes: int = Field(default=0, ge=0)
    planned_part_count: int = Field(default=0, ge=0)
    processed_part_count: int = Field(default=0, ge=0)
    pending_part_count: int = Field(default=0, ge=0)
    retrying_part_count: int = Field(default=0, ge=0)
    failed_part_count: int = Field(default=0, ge=0)
    invalidated_part_count: int = Field(default=0, ge=0)
    processed_locators: tuple[str, ...] = ()
    last_successful_evidence_at: datetime | None = None
    influenced_current_snapshot: bool = False
    next_action: Literal[
        "none",
        "prepare",
        "analyze",
        "resume",
        "retry",
        "reapprove",
    ] = "none"

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("coverage topics must not be empty")
        return validate_safe_evidence_text(normalized)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str | None) -> str | None:
        return None if value is None else normalize_relative_path(value)

    @field_validator("processed_locators")
    @classmethod
    def validate_processed_locators(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("processed locators must be unique")
        return tuple(validate_safe_evidence_text(item) for item in value)

    @model_validator(mode="after")
    def validate_entry_ledger(self) -> CoverageItem:
        if self.entry_id is None:
            return self
        counts = (
            self.processed_part_count
            + self.pending_part_count
            + self.retrying_part_count
            + self.failed_part_count
            + self.invalidated_part_count
        )
        if self.relative_path is None or counts != self.planned_part_count:
            raise ValueError("coverage entry part counts must be exact")
        expected_covered = (
            self.planned_part_count > 0
            and self.processed_part_count == self.planned_part_count
        )
        if self.covered != expected_covered:
            raise ValueError("coverage entry covered state must match processed parts")
        if len(self.processed_locators) != self.processed_part_count:
            raise ValueError("processed locators must match processed part count")
        return self


class CoverageSummary(EvidenceFrozenModel):
    items: tuple[CoverageItem, ...]
    covered_count: int = Field(ge=0)
    total_count: int = Field(ge=1)
    approved_bytes: int = Field(default=0, ge=0)
    part_total_count: int = Field(default=0, ge=0)
    part_processed_count: int = Field(default=0, ge=0)
    part_pending_count: int = Field(default=0, ge=0)
    part_retrying_count: int = Field(default=0, ge=0)
    part_failed_count: int = Field(default=0, ge=0)
    part_invalidated_count: int = Field(default=0, ge=0)

    @property
    def coverage_fraction(self) -> float:
        return self.covered_count / self.total_count

    @property
    def part_coverage_fraction(self) -> float:
        if self.part_total_count == 0:
            return 0.0
        return self.part_processed_count / self.part_total_count

    @property
    def is_partial(self) -> bool:
        return (
            0 < self.covered_count < self.total_count
            or 0 < self.part_processed_count < self.part_total_count
        )

    @model_validator(mode="after")
    def validate_exact_coverage(self) -> CoverageSummary:
        if self.total_count != len(self.items):
            raise ValueError("total_count must equal the number of coverage items")
        if self.covered_count != sum(item.covered for item in self.items):
            raise ValueError("covered_count must equal the covered coverage items")
        if len({item.topic.casefold() for item in self.items}) != len(self.items):
            raise ValueError("coverage topics must be unique")
        if all(item.entry_id is not None for item in self.items):
            exact = {
                "approved_bytes": sum(item.approved_bytes for item in self.items),
                "part_total_count": sum(item.planned_part_count for item in self.items),
                "part_processed_count": sum(
                    item.processed_part_count for item in self.items
                ),
                "part_pending_count": sum(item.pending_part_count for item in self.items),
                "part_retrying_count": sum(
                    item.retrying_part_count for item in self.items
                ),
                "part_failed_count": sum(item.failed_part_count for item in self.items),
                "part_invalidated_count": sum(
                    item.invalidated_part_count for item in self.items
                ),
            }
            if any(getattr(self, name) != count for name, count in exact.items()):
                raise ValueError("coverage summary part and byte totals must be exact")
        return self


class CourseGroup(EvidenceFrozenModel):
    group_id: str
    title: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_unit_ids: tuple[str, ...]

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("course-group titles must not be empty")
        return validate_safe_evidence_text(normalized)

    @field_validator("evidence_unit_ids")
    @classmethod
    def validate_evidence_dependencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("course groups require evidence dependencies")
        if len(set(value)) != len(value):
            raise ValueError("course-group evidence dependencies must be unique")
        return value

    @property
    def confidence_band(self) -> Literal["Strong", "Moderate", "Limited"]:
        return _confidence_band(self.confidence)


class KnowledgeNode(EvidenceFrozenModel):
    node_id: str
    title: str
    focus_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_unit_ids: tuple[str, ...]
    parent_node_id: str | None = None
    prerequisite_node_ids: tuple[str, ...] = ()
    course_group_id: str | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("knowledge-node titles must not be empty")
        return validate_safe_evidence_text(normalized)

    @field_validator("evidence_unit_ids")
    @classmethod
    def validate_evidence_dependencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("evidence dependencies must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("evidence dependencies must be unique")
        return value

    @field_validator("prerequisite_node_ids")
    @classmethod
    def validate_prerequisites(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("prerequisite dependencies must be unique")
        return value

    @model_validator(mode="after")
    def validate_no_self_reference(self) -> KnowledgeNode:
        if self.parent_node_id == self.node_id or self.node_id in self.prerequisite_node_ids:
            raise ValueError("knowledge nodes cannot depend on themselves")
        return self

    @property
    def evidence_count(self) -> int:
        return len(self.evidence_unit_ids)

    @property
    def focus_band(self) -> Literal["High", "Medium", "Low"]:
        if self.focus_score >= 2 / 3:
            return "High"
        if self.focus_score >= 1 / 3:
            return "Medium"
        return "Low"

    @property
    def confidence_band(self) -> Literal["Strong", "Moderate", "Limited"]:
        return _confidence_band(self.confidence)


class StudyMapSnapshot(EvidenceFrozenModel):
    snapshot_id: str
    workspace_id: str
    revision_id: str
    status: SnapshotStatus
    nodes: tuple[KnowledgeNode, ...]
    coverage: CoverageSummary | None
    evidence_unit_ids: tuple[str, ...]
    created_at: datetime
    course_groups: tuple[CourseGroup, ...] = ()
    limitations: tuple[str, ...] = ()
    superseded_snapshot_id: str | None = None

    @field_validator("limitations")
    @classmethod
    def validate_limitations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(validate_safe_evidence_text(item.strip()) for item in value if item.strip())

    @model_validator(mode="after")
    def validate_initial_dependencies(self) -> StudyMapSnapshot:
        if self.status is SnapshotStatus.INITIAL and (
            self.coverage is None or not self.coverage.is_partial or not self.evidence_unit_ids
        ):
            raise ValueError("initial snapshots require coverage and evidence dependencies")
        if len(set(self.evidence_unit_ids)) != len(self.evidence_unit_ids):
            raise ValueError("evidence dependencies must be unique")
        node_ids = tuple(node.node_id for node in self.nodes)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("knowledge-node IDs must be unique")
        group_ids = tuple(group.group_id for group in self.course_groups)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("course-group IDs must be unique")
        known_nodes = set(node_ids)
        known_groups = set(group_ids)
        if any(
            (node.parent_node_id is not None and node.parent_node_id not in known_nodes)
            or any(item not in known_nodes for item in node.prerequisite_node_ids)
            or (node.course_group_id is not None and node.course_group_id not in known_groups)
            for node in self.nodes
        ):
            raise ValueError("study-map relationships must reference known nodes and groups")
        return self


def _confidence_band(value: float) -> Literal["Strong", "Moderate", "Limited"]:
    if value >= 2 / 3:
        return "Strong"
    if value >= 1 / 3:
        return "Moderate"
    return "Limited"
