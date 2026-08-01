"""Strict evidence validation and dependency-bound study-map publication."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import UTC, datetime
from hashlib import sha256
import json
import re
from threading import RLock
import time
from typing import Callable, ContextManager, Literal, Protocol

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from exam_predictor.evidence.artifacts import EvidenceArtifactStore
from exam_predictor.evidence.models import (
    CourseGroup,
    CoverageItem,
    CoverageSummary,
    EvidenceCitation,
    EvidenceFrozenModel,
    EvidenceUnit,
    KnowledgeNode,
    PartState,
    SnapshotStatus,
    SourcePartPlan,
    StudyMapSnapshot,
    validate_safe_evidence_text,
)
from exam_predictor.evidence.policy import EvidencePolicy
from exam_predictor.evidence.providers import EvidencePartResult
from exam_predictor.evidence.store import EvidenceStore


_MAX_PROVIDER_JSON_BYTES = 2 * 1024 * 1024
_MAX_EVIDENCE_SUMMARY_CHARS = 12_000
_DEFAULT_SYNTHESIS_BATCH_SIZE = 16
_PROBABILITY_LABEL = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*%|\bexam\s+probabilit(?:y|ies)\b|"
    r"\b(?:chance|likelihood)\s+of\s+(?:appearing|being tested)\b)",
    re.IGNORECASE,
)
_SAFE_VALIDATION_MESSAGES = {
    "evidence_invalid": "The provider evidence response is invalid.",
    "evidence_repair_failed": "The provider evidence repair failed.",
    "study_map_invalid": "The provider study map is invalid.",
    "study_map_provider_failed": "The provider could not synthesize the study map.",
}


class EvidenceValidationError(ValueError):
    """A stable validation failure that never retains provider exception state."""

    def __init__(self, code: str, *, issues: tuple[str, ...] = ()) -> None:
        if code not in _SAFE_VALIDATION_MESSAGES:
            raise ValueError("unsupported evidence validation error code")
        self.code = code
        self.issues = issues
        super().__init__(_SAFE_VALIDATION_MESSAGES[code])


class EvidenceSynthesisDeadlineExceeded(RuntimeError):
    """No new synthesis request may start inside the final ten seconds."""


class EvidenceRepairRequest(EvidenceFrozenModel):
    source_part_id: str
    locator: str = Field(min_length=1, max_length=512)
    raw_output: str = Field(repr=False, exclude=True)
    errors: tuple[str, ...]
    deadline_seconds: float = Field(ge=10.0, le=300.0)


class EvidenceRepairer(Protocol):
    def repair_evidence(self, request: EvidenceRepairRequest) -> str: ...


class _ProviderPayloadModel(EvidenceFrozenModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class _Definition(_ProviderPayloadModel):
    term: str = Field(min_length=1, max_length=512)
    explanation: str = Field(min_length=1, max_length=4_096)

    @field_validator("term", "explanation")
    @classmethod
    def validate_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("definition text must not be empty")
        return validate_safe_evidence_text(normalized)


class _RawEvidence(_ProviderPayloadModel):
    locator: str = Field(min_length=1, max_length=512)
    detected_language: str = Field(min_length=1, max_length=64)
    material_role: str = Field(min_length=1, max_length=128)
    headings: tuple[str, ...] = Field(max_length=128)
    concepts: tuple[str, ...] = Field(max_length=256)
    definitions: tuple[_Definition, ...] = Field(max_length=128)
    formulas: tuple[str, ...] = Field(max_length=128)
    procedures: tuple[str, ...] = Field(max_length=128)
    examples: tuple[str, ...] = Field(max_length=128)
    assessment_items: tuple[str, ...] = Field(max_length=128)
    visual_descriptions: tuple[str, ...] = Field(max_length=128)
    ocr_text: tuple[str, ...] = Field(max_length=128)
    limitations: tuple[str, ...] = Field(max_length=128)
    warnings: tuple[str, ...] = Field(max_length=128)
    prompt_injection_indicators: tuple[str, ...] = Field(max_length=128)

    @field_validator("locator", "detected_language", "material_role")
    @classmethod
    def validate_scalar_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("evidence text must not be empty")
        return validate_safe_evidence_text(normalized)

    @field_validator(
        "headings",
        "concepts",
        "formulas",
        "procedures",
        "examples",
        "assessment_items",
        "visual_descriptions",
        "ocr_text",
        "limitations",
        "warnings",
        "prompt_injection_indicators",
    )
    @classmethod
    def validate_text_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for item in value:
            text = item.strip()
            if not text or len(text) > 4_096:
                raise ValueError("evidence list text is empty or too long")
            normalized.append(validate_safe_evidence_text(text))
        return tuple(normalized)


class EvidenceValidator:
    """Convert one strict provider result into one immutable cited evidence unit."""

    def __init__(
        self,
        *,
        repairer: EvidenceRepairer | None = None,
        policy: EvidencePolicy = EvidencePolicy(),
    ) -> None:
        self._repairer = repairer
        self._policy = policy

    def __call__(self, result: EvidencePartResult, plan: SourcePartPlan) -> EvidenceUnit:
        return self.validate(
            result,
            plan,
            deadline_seconds=self._policy.provider_timeout_seconds,
        )

    def validate(
        self,
        result: EvidencePartResult,
        plan: SourcePartPlan,
        *,
        deadline_seconds: float,
    ) -> EvidenceUnit:
        raw_output = result.raw_output
        issues: tuple[str, ...]
        try:
            evidence = self._validate_once(result, plan, raw_output)
        except (ValidationError, ValueError, json.JSONDecodeError) as error:
            issues = _safe_validation_issues(error)
        else:
            return self._unit(result, plan, evidence)

        if self._repairer is None or self._policy.max_repair_attempts == 0:
            raise EvidenceValidationError("evidence_invalid", issues=issues) from None
        if deadline_seconds < 10.0:
            raise EvidenceValidationError("evidence_repair_failed") from None
        try:
            repaired = self._repairer.repair_evidence(
                EvidenceRepairRequest(
                    source_part_id=plan.part_id,
                    locator=plan.locator,
                    raw_output=raw_output,
                    errors=issues,
                    deadline_seconds=min(
                        self._policy.provider_timeout_seconds,
                        deadline_seconds,
                    ),
                )
            )
        except Exception:
            raise EvidenceValidationError("evidence_repair_failed") from None
        if not isinstance(repaired, str):
            raise EvidenceValidationError("evidence_repair_failed") from None
        repaired_result = result.model_copy(update={"raw_output": repaired})
        try:
            evidence = self._validate_once(repaired_result, plan, repaired)
        except (ValidationError, ValueError, json.JSONDecodeError) as error:
            raise EvidenceValidationError(
                "evidence_invalid",
                issues=_safe_validation_issues(error),
            ) from None
        return self._unit(repaired_result, plan, evidence)

    @staticmethod
    def _validate_once(
        result: EvidencePartResult,
        plan: SourcePartPlan,
        raw_output: str,
    ) -> _RawEvidence:
        if (
            result.source_part_id != plan.part_id
            or result.locator != plan.locator
            or len(raw_output.encode("utf-8")) > _MAX_PROVIDER_JSON_BYTES
        ):
            raise ValueError("provider identity or response bound is invalid")
        document = json.loads(raw_output)
        evidence = _RawEvidence.model_validate(document)
        if evidence.locator != plan.locator:
            raise ValueError("provider locator does not match the prepared part")
        return evidence

    @staticmethod
    def _unit(
        result: EvidencePartResult,
        plan: SourcePartPlan,
        evidence: _RawEvidence,
    ) -> EvidenceUnit:
        canonical = _canonical_json(evidence.model_dump(mode="json"))
        identity = "\0".join(
            (
                plan.part_id,
                result.provider,
                result.model_id,
                result.prompt_version,
                canonical,
            )
        )
        unit_id = f"unit_{sha256(identity.encode()).hexdigest()}"
        citation_id = f"citation_{sha256((unit_id + chr(0) + plan.locator).encode()).hexdigest()}"
        return EvidenceUnit(
            evidence_unit_id=unit_id,
            source_part_id=plan.part_id,
            content=canonical,
            citations=(
                EvidenceCitation(
                    citation_id=citation_id,
                    evidence_unit_id=unit_id,
                    source_part_id=plan.part_id,
                    relative_path=plan.relative_path,
                    locator=plan.locator,
                ),
            ),
        )


class EvidenceSynthesisItem(_ProviderPayloadModel):
    evidence_unit_id: str
    source_part_id: str
    content: str = Field(repr=False, max_length=_MAX_EVIDENCE_SUMMARY_CHARS)
    relative_path: str
    locator: str


class StudyMapSynthesisRequest(_ProviderPayloadModel):
    phase: Literal["batch", "final"]
    workspace_id: str
    revision_id: str
    status: SnapshotStatus
    evidence: tuple[EvidenceSynthesisItem, ...] = Field(max_length=32)
    drafts: tuple[str, ...] = Field(default=(), max_length=64, repr=False)
    response_language: str | None = Field(default=None, max_length=64)
    deadline_seconds: float = Field(ge=10.0, le=300.0)


class StudyMapSynthesizer(Protocol):
    def synthesize_study_map(self, request: StudyMapSynthesisRequest) -> str: ...


class _SynthesisPayload(_ProviderPayloadModel):
    course_groups: tuple[CourseGroup, ...]
    nodes: tuple[KnowledgeNode, ...]
    limitations: tuple[str, ...]
    evidence_unit_ids: tuple[str, ...]


class EvidenceAnswerContext(EvidenceFrozenModel):
    snapshot: StudyMapSnapshot
    evidence_units: tuple[EvidenceUnit, ...]
    limitations: tuple[str, ...]


class ApprovedCoverageEntry(EvidenceFrozenModel):
    entry_id: str
    relative_path: str
    approved_bytes: int = Field(ge=0)
    included: bool = True

    @model_validator(mode="after")
    def validate_inclusion(self) -> ApprovedCoverageEntry:
        if not self.included and self.approved_bytes != 0:
            raise ValueError("excluded manifest entries cannot have approved bytes")
        return self


CoverageSource = Callable[[str, str], tuple[ApprovedCoverageEntry, ...]]
PublicationGuard = Callable[[], ContextManager[object]]


class StudyMapBuilder:
    """Synthesize and publish immutable snapshots from processed evidence only."""

    def __init__(
        self,
        store: EvidenceStore,
        artifact_store: EvidenceArtifactStore,
        synthesizer: StudyMapSynthesizer,
        *,
        coverage_source: CoverageSource,
        policy: EvidencePolicy = EvidencePolicy(),
        now: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        batch_size: int = _DEFAULT_SYNTHESIS_BATCH_SIZE,
    ) -> None:
        if not 1 <= batch_size <= 32:
            raise ValueError("study-map batch size must be between 1 and 32")
        self._store = store
        self._artifact_store = artifact_store
        self._synthesizer = synthesizer
        self._policy = policy
        self._now = now or (lambda: datetime.now(UTC))
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._batch_size = batch_size
        self._coverage_source = coverage_source
        self._artifact_lock = RLock()

    def publish_initial(
        self,
        workspace_id: str,
        revision_id: str,
        *,
        publication_guard: PublicationGuard | None = None,
        synthesis_guard: PublicationGuard | None = None,
        response_language: str | None = None,
        deadline: float | None = None,
    ) -> StudyMapSnapshot | None:
        if not self._store.is_current_revision(workspace_id, revision_id):
            return None
        self.recover_snapshot_revocations(workspace_id)
        current = self._store.current_snapshot(workspace_id, revision_id)
        if current is not None:
            with self._publication_context(publication_guard):
                self._ensure_snapshot_artifact(current)
            return current
        parts = self._store.list_parts(workspace_id, revision_id)
        if not parts:
            return None
        coverage = self._coverage(workspace_id, revision_id, parts)
        if not coverage.is_partial or not self._high_priority_ready(parts):
            return None
        units = self._store.list_evidence_units(workspace_id, revision_id)
        if not units:
            return None
        self._require_one_unit_per_processed_part(parts, units)
        return self._publish(
            workspace_id,
            revision_id,
            SnapshotStatus.INITIAL,
            coverage,
            units,
            parts,
            superseded_snapshot_id=None,
            publication_guard=publication_guard,
            synthesis_guard=synthesis_guard,
            response_language=response_language,
            deadline=deadline,
        )

    def publish_complete(
        self,
        workspace_id: str,
        revision_id: str,
        *,
        publication_guard: PublicationGuard | None = None,
        synthesis_guard: PublicationGuard | None = None,
        response_language: str | None = None,
        deadline: float | None = None,
    ) -> StudyMapSnapshot | None:
        if not self._store.is_current_revision(workspace_id, revision_id):
            return None
        self.recover_snapshot_revocations(workspace_id)
        current = self._store.current_snapshot(workspace_id, revision_id)
        parts = self._store.list_parts(workspace_id, revision_id)
        if not parts or any(
            part.state not in {PartState.PROCESSED, PartState.FAILED} for part in parts
        ):
            return None
        units = self._store.list_evidence_units(workspace_id, revision_id)
        self._require_one_unit_per_processed_part(parts, units)
        coverage = self._coverage(workspace_id, revision_id, parts)
        if any(
            not item.excluded
            and (
                item.planned_part_count == 0
                or item.pending_part_count > 0
                or item.retrying_part_count > 0
                or item.invalidated_part_count > 0
                or item.processed_part_count + item.failed_part_count
                != item.planned_part_count
            )
            for item in coverage.items
        ):
            return None
        dependency_ids = tuple(sorted(unit.evidence_unit_id for unit in units))
        if (
            current is not None
            and current.status is SnapshotStatus.COMPLETE
            and current.coverage is not None
            and _coverage_without_influence(current.coverage)
            == _coverage_without_influence(coverage)
            and current.evidence_unit_ids == dependency_ids
            and (
                response_language is None
                or current.response_language == response_language
            )
        ):
            with self._publication_context(publication_guard):
                self._ensure_snapshot_artifact(current)
            return current
        return self._publish(
            workspace_id,
            revision_id,
            SnapshotStatus.COMPLETE,
            coverage,
            units,
            parts,
            superseded_snapshot_id=(None if current is None else current.snapshot_id),
            publication_guard=publication_guard,
            synthesis_guard=synthesis_guard,
            response_language=response_language,
            deadline=deadline,
        )

    def answer_context(
        self,
        workspace_id: str,
        revision_id: str,
        *,
        max_units: int = 32,
    ) -> EvidenceAnswerContext | None:
        if not 1 <= max_units <= 64:
            raise ValueError("answer context must contain between 1 and 64 evidence units")
        if not self._store.is_current_revision(workspace_id, revision_id):
            return None
        context = self._store.current_snapshot_context(workspace_id, revision_id)
        if context is None:
            return None
        snapshot, all_units = context
        units = all_units[:max_units]
        limitations = list(snapshot.limitations)
        if (
            snapshot.coverage is not None
            and snapshot.coverage.covered_count < snapshot.coverage.included_count
        ):
            limitations.append(
                "The current map does not contain validated evidence for every planned source part."
            )
        if len(snapshot.evidence_unit_ids) > max_units:
            limitations.append("Additional cited evidence was omitted from this bounded answer context.")
        return EvidenceAnswerContext(
            snapshot=snapshot,
            evidence_units=units,
            limitations=tuple(dict.fromkeys(limitations)),
        )

    def coverage(
        self,
        workspace_id: str,
        revision_id: str,
    ) -> CoverageSummary:
        parts = self._store.list_parts(workspace_id, revision_id)
        return self._coverage(workspace_id, revision_id, parts)

    def invalidate_entry(
        self,
        workspace_id: str,
        revision_id: str,
        entry_id: str,
    ) -> tuple[str, ...]:
        part_ids, _snapshot_ids = self._store.invalidate_entry_with_revocations(
            workspace_id,
            revision_id,
            entry_id,
        )
        self.recover_snapshot_revocations(workspace_id)
        return part_ids

    def recover_snapshot_revocations(self, workspace_id: str) -> tuple[str, ...]:
        pending = frozenset(self._store.pending_snapshot_revocations(workspace_id))
        completed: list[str] = []
        for snapshot_id in self._store.snapshot_revocations(workspace_id):
            self._revoke_snapshot_artifact(workspace_id, snapshot_id)
            self._store.complete_snapshot_revocation(workspace_id, snapshot_id)
            if snapshot_id in pending:
                completed.append(snapshot_id)
        return tuple(completed)

    @staticmethod
    def _high_priority_ready(parts: tuple[SourcePartPlan, ...]) -> bool:
        high_priority = tuple(part for part in parts if part.priority <= 1)
        if not high_priority:
            return any(part.state is PartState.PROCESSED for part in parts)
        return any(part.state is PartState.PROCESSED for part in high_priority)

    def _coverage(
        self,
        workspace_id: str,
        revision_id: str,
        parts: tuple[SourcePartPlan, ...],
    ) -> CoverageSummary:
        entries = self._coverage_source(workspace_id, revision_id)
        return _part_coverage(
            parts,
            entries,
            self._store.latest_evidence_times(workspace_id, revision_id),
            self._store.latest_part_error_codes(workspace_id, revision_id),
        )

    @staticmethod
    def _require_one_unit_per_processed_part(
        parts: tuple[SourcePartPlan, ...],
        units: tuple[EvidenceUnit, ...],
    ) -> None:
        processed_ids = {
            part.part_id for part in parts if part.state is PartState.PROCESSED
        }
        unit_part_ids = {unit.source_part_id for unit in units}
        if len(units) != len(unit_part_ids) or unit_part_ids != processed_ids:
            raise EvidenceValidationError("study_map_invalid")

    def _publish(
        self,
        workspace_id: str,
        revision_id: str,
        status: SnapshotStatus,
        coverage: CoverageSummary,
        units: tuple[EvidenceUnit, ...],
        parts: tuple[SourcePartPlan, ...],
        *,
        superseded_snapshot_id: str | None,
        publication_guard: PublicationGuard | None,
        synthesis_guard: PublicationGuard | None,
        response_language: str | None,
        deadline: float | None,
    ) -> StudyMapSnapshot:
        with self._publication_context(synthesis_guard):
            payload = self._synthesize(
                workspace_id,
                revision_id,
                status,
                units,
                response_language=response_language,
                deadline=deadline,
            )
        coverage = _mark_snapshot_influence(coverage, parts, units, payload)
        created_at = self._now()
        dependency_ids = tuple(sorted(payload.evidence_unit_ids))
        units_by_id = {unit.evidence_unit_id: unit for unit in units}
        citations = tuple(
            citation
            for evidence_unit_id in dependency_ids
            for citation in units_by_id[evidence_unit_id].citations
        )
        identity_payload = {
            "workspace_id": workspace_id,
            "revision_id": revision_id,
            "status": status.value,
            "course_groups": [group.model_dump(mode="json") for group in payload.course_groups],
            "nodes": [node.model_dump(mode="json") for node in payload.nodes],
            "coverage": coverage.model_dump(mode="json"),
            "evidence_unit_ids": dependency_ids,
            "citations": [item.model_dump(mode="json") for item in citations],
            "limitations": payload.limitations,
            "response_language": response_language,
            "superseded_snapshot_id": superseded_snapshot_id,
        }
        snapshot_id = f"snapshot_{sha256(_canonical_json(identity_payload).encode()).hexdigest()}"
        snapshot = StudyMapSnapshot(
            snapshot_id=snapshot_id,
            workspace_id=workspace_id,
            revision_id=revision_id,
            status=status,
            nodes=payload.nodes,
            coverage=coverage,
            evidence_unit_ids=dependency_ids,
            created_at=created_at,
            course_groups=payload.course_groups,
            limitations=payload.limitations,
            citations=citations,
            response_language=response_language,
            superseded_snapshot_id=superseded_snapshot_id,
        )
        snapshot_document = snapshot.model_dump(mode="json")
        expected_sha256 = sha256(_canonical_json(snapshot_document).encode()).hexdigest()
        with self._publication_context(publication_guard):
            inserted = self._store.save_snapshot(snapshot)
            persisted = (
                snapshot if inserted else self._store.get_snapshot(snapshot.snapshot_id)
            )
            if persisted is None:
                raise EvidenceValidationError("study_map_invalid")
            try:
                self._ensure_snapshot_artifact(
                    persisted,
                    expected_sha256=(expected_sha256 if inserted else None),
                )
            except Exception:
                if inserted:
                    try:
                        self._revoke_snapshot_artifact(
                            workspace_id,
                            snapshot.snapshot_id,
                        )
                    except Exception:
                        pass
                    self._store.delete_snapshot(snapshot.snapshot_id)
                raise
            if (
                not self._store.is_current_revision(workspace_id, revision_id)
                or self._store.get_snapshot(snapshot.snapshot_id) is None
            ):
                self._revoke_snapshot_artifact(
                    workspace_id,
                    snapshot.snapshot_id,
                )
                self._store.delete_snapshot(snapshot.snapshot_id)
                raise ValueError("study-map snapshot revision is no longer current")
        return persisted

    @staticmethod
    def _publication_context(
        publication_guard: PublicationGuard | None,
    ) -> ContextManager[object]:
        return nullcontext() if publication_guard is None else publication_guard()

    def _ensure_snapshot_artifact(
        self,
        snapshot: StudyMapSnapshot,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        document = snapshot.model_dump(mode="json")
        digest = expected_sha256 or sha256(_canonical_json(document).encode()).hexdigest()
        with self._artifact_lock:
            if self._store.snapshot_is_revoked(
                snapshot.workspace_id,
                snapshot.snapshot_id,
            ):
                raise ValueError("study-map snapshot artifact is permanently revoked")
            self._artifact_store.publish_json(
                snapshot.workspace_id,
                "snapshots",
                snapshot.snapshot_id,
                snapshot,
                expected_sha256=digest,
            )

    def _revoke_snapshot_artifact(
        self,
        workspace_id: str,
        snapshot_id: str,
    ) -> None:
        with self._artifact_lock:
            self._artifact_store.revoke_json(
                workspace_id,
                "snapshots",
                snapshot_id,
            )

    def _synthesize(
        self,
        workspace_id: str,
        revision_id: str,
        status: SnapshotStatus,
        units: tuple[EvidenceUnit, ...],
        *,
        response_language: str | None,
        deadline: float | None,
    ) -> _SynthesisPayload:
        if not units:
            return _SynthesisPayload(
                course_groups=(),
                nodes=(),
                limitations=("No source part produced validated evidence.",),
                evidence_unit_ids=(),
            )
        items = tuple(_synthesis_item(unit) for unit in units)
        allowed = frozenset(item.evidence_unit_id for item in items)
        batches = tuple(
            items[index : index + self._batch_size]
            for index in range(0, len(items), self._batch_size)
        )
        if len(batches) == 1:
            output = self._call_synthesizer(
                StudyMapSynthesisRequest(
                    phase="final",
                    workspace_id=workspace_id,
                    revision_id=revision_id,
                    status=status,
                    evidence=batches[0],
                    response_language=response_language,
                    deadline_seconds=self._synthesis_deadline_seconds(deadline),
                )
            )
            return _parse_synthesis(output, allowed)

        def synthesize_batch(batch: tuple[EvidenceSynthesisItem, ...]) -> str:
            output = self._call_synthesizer(
                StudyMapSynthesisRequest(
                    phase="batch",
                    workspace_id=workspace_id,
                    revision_id=revision_id,
                    status=status,
                    evidence=batch,
                    response_language=response_language,
                    deadline_seconds=self._synthesis_deadline_seconds(deadline),
                )
            )
            payload = _parse_synthesis(
                output,
                frozenset(item.evidence_unit_id for item in batch),
            )
            return _canonical_json(payload.model_dump(mode="json"))

        with ThreadPoolExecutor(
            max_workers=min(self._policy.synthesis_concurrency, len(batches)),
            thread_name_prefix="examsage-study-map",
        ) as executor:
            drafts = tuple(executor.map(synthesize_batch, batches))
        final = self._call_synthesizer(
            StudyMapSynthesisRequest(
                phase="final",
                workspace_id=workspace_id,
                revision_id=revision_id,
                status=status,
                evidence=(),
                drafts=drafts,
                response_language=response_language,
                deadline_seconds=self._synthesis_deadline_seconds(deadline),
            )
        )
        return _parse_synthesis(final, allowed)

    def _synthesis_deadline_seconds(self, deadline: float | None) -> float:
        if deadline is None:
            return self._policy.provider_timeout_seconds
        remaining = deadline - self._monotonic_clock()
        if remaining < 10.0:
            raise EvidenceSynthesisDeadlineExceeded
        return min(self._policy.provider_timeout_seconds, remaining)

    def _call_synthesizer(self, request: StudyMapSynthesisRequest) -> str:
        try:
            output = self._synthesizer.synthesize_study_map(request)
        except EvidenceValidationError:
            raise
        except Exception:
            raise EvidenceValidationError("study_map_provider_failed") from None
        if not isinstance(output, str) or len(output.encode("utf-8")) > _MAX_PROVIDER_JSON_BYTES:
            raise EvidenceValidationError("study_map_invalid") from None
        return output


def _synthesis_item(unit: EvidenceUnit) -> EvidenceSynthesisItem:
    citation = unit.citations[0]
    return EvidenceSynthesisItem(
        evidence_unit_id=unit.evidence_unit_id,
        source_part_id=unit.source_part_id,
        content=unit.content[:_MAX_EVIDENCE_SUMMARY_CHARS],
        relative_path=citation.relative_path,
        locator=citation.locator,
    )


def _parse_synthesis(raw_output: str, allowed_evidence: frozenset[str]) -> _SynthesisPayload:
    try:
        payload = _SynthesisPayload.model_validate(json.loads(raw_output))
        _validate_provider_strings(payload.model_dump(mode="python"))
        _validate_synthesis_payload(payload, allowed_evidence)
        return payload
    except EvidenceValidationError:
        raise
    except (ValidationError, ValueError, json.JSONDecodeError) as error:
        raise EvidenceValidationError(
            "study_map_invalid",
            issues=_safe_validation_issues(error),
        ) from None


def _validate_synthesis_payload(
    payload: _SynthesisPayload,
    allowed_evidence: frozenset[str],
) -> None:
    node_ids = tuple(node.node_id for node in payload.nodes)
    group_ids = tuple(group.group_id for group in payload.course_groups)
    if len(set(node_ids)) != len(node_ids) or len(set(group_ids)) != len(group_ids):
        raise ValueError("study map contains duplicate identities")
    if allowed_evidence and not payload.nodes:
        raise ValueError("study map requires at least one cited node")
    known_nodes = set(node_ids)
    known_groups = set(group_ids)
    for node in payload.nodes:
        if _PROBABILITY_LABEL.search(node.title):
            raise ValueError("study map contains a literal probability label")
        if node.parent_node_id is not None and node.parent_node_id not in known_nodes:
            raise ValueError("study map contains an unknown parent")
        if any(item not in known_nodes for item in node.prerequisite_node_ids):
            raise ValueError("study map contains an unknown prerequisite")
        if node.course_group_id is not None and node.course_group_id not in known_groups:
            raise ValueError("study map contains an unknown course group")
    if any(_PROBABILITY_LABEL.search(group.title) for group in payload.course_groups):
        raise ValueError("course group contains a literal probability label")
    if any(_PROBABILITY_LABEL.search(item) for item in payload.limitations):
        raise ValueError("study map limitation contains a literal probability label")
    for item in payload.limitations:
        validate_safe_evidence_text(item)
    dependency_ids = {
        evidence_id
        for node in payload.nodes
        for evidence_id in node.evidence_unit_ids
    } | {
        evidence_id
        for group in payload.course_groups
        for evidence_id in group.evidence_unit_ids
    }
    if set(payload.evidence_unit_ids) != dependency_ids:
        raise ValueError("study map dependency closure is inconsistent")
    if len(set(payload.evidence_unit_ids)) != len(payload.evidence_unit_ids):
        raise ValueError("study map dependencies must be unique")
    if not dependency_ids <= allowed_evidence:
        raise ValueError("study map cites evidence outside processed source parts")
    _reject_graph_cycle(
        {
            node.node_id: tuple(
                item
                for item in (node.parent_node_id, *node.prerequisite_node_ids)
                if item is not None
            )
            for node in payload.nodes
        }
    )


def _reject_graph_cycle(graph: dict[str, tuple[str, ...]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("study map relationships contain a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in graph[node_id]:
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in graph:
        visit(node_id)


def _part_coverage(
    parts: tuple[SourcePartPlan, ...],
    entries: tuple[ApprovedCoverageEntry, ...],
    latest_evidence_times: dict[str, datetime],
    latest_error_codes: dict[str, str],
) -> CoverageSummary:
    if len({entry.entry_id for entry in entries}) != len(entries):
        raise EvidenceValidationError("study_map_invalid")
    known_entries = {entry.entry_id for entry in entries}
    if any(part.entry_id not in known_entries for part in parts):
        raise EvidenceValidationError("study_map_invalid")
    excluded_entries = {entry.entry_id for entry in entries if not entry.included}
    if any(part.entry_id in excluded_entries for part in parts):
        raise EvidenceValidationError("study_map_invalid")
    parts_by_entry = {
        entry.entry_id: tuple(part for part in parts if part.entry_id == entry.entry_id)
        for entry in entries
    }
    items: list[CoverageItem] = []
    for entry in entries:
        if not entry.included:
            items.append(
                CoverageItem(
                    topic=entry.relative_path,
                    covered=False,
                    excluded=True,
                    entry_id=entry.entry_id,
                    relative_path=entry.relative_path,
                )
            )
            continue
        entry_parts = parts_by_entry[entry.entry_id]
        processed = tuple(
            part for part in entry_parts if part.state is PartState.PROCESSED
        )
        pending = tuple(
            part
            for part in entry_parts
            if part.state
            in {
                PartState.PLANNED,
                PartState.AUTHORIZED,
                PartState.PREPARED,
            }
        )
        running = tuple(
            part for part in entry_parts if part.state is PartState.RUNNING
        )
        retrying = tuple(
            part for part in entry_parts if part.state is PartState.RETRY_WAIT
        )
        failed = tuple(part for part in entry_parts if part.state is PartState.FAILED)
        invalidated = tuple(
            part for part in entry_parts if part.state is PartState.INVALIDATED
        )
        covered = bool(entry_parts) and len(processed) == len(entry_parts)
        if not entry_parts:
            next_action = "prepare"
        elif invalidated:
            next_action = "reapprove"
        elif retrying:
            next_action = "resume"
        elif running or pending:
            next_action = "analyze"
        elif failed:
            next_action = "retry"
        else:
            next_action = "none"
        success_times = tuple(
            latest_evidence_times[part.part_id]
            for part in processed
            if part.part_id in latest_evidence_times
        )
        items.append(
            CoverageItem(
                topic=entry.relative_path,
                covered=covered,
                entry_id=entry.entry_id,
                relative_path=entry.relative_path,
                approved_bytes=entry.approved_bytes,
                planned_part_count=len(entry_parts),
                processed_part_count=len(processed),
                running_part_count=len(running),
                pending_part_count=len(pending),
                retrying_part_count=len(retrying),
                failed_part_count=len(failed),
                invalidated_part_count=len(invalidated),
                processed_locators=tuple(part.locator for part in processed),
                current_locator=(
                    next(
                        (
                            part.locator
                            for part in (
                                *running,
                                *invalidated,
                                *retrying,
                                *pending,
                                *failed,
                                *processed[-1:],
                            )
                        ),
                        None,
                    )
                ),
                failure_codes=tuple(
                    dict.fromkeys(
                        latest_error_codes.get(part.part_id) or part.safe_error_code
                        for part in (*retrying, *failed)
                        if latest_error_codes.get(part.part_id) or part.safe_error_code
                    )
                ),
                last_successful_evidence_at=(max(success_times) if success_times else None),
                next_action=next_action,
            )
        )
    item_tuple = tuple(items)
    return CoverageSummary(
        items=item_tuple,
        covered_count=sum(item.covered for item in item_tuple),
        total_count=len(item_tuple),
        excluded_count=sum(item.excluded for item in item_tuple),
        approved_bytes=sum(item.approved_bytes for item in item_tuple),
        part_total_count=sum(item.planned_part_count for item in item_tuple),
        part_processed_count=sum(item.processed_part_count for item in item_tuple),
        part_running_count=sum(item.running_part_count for item in item_tuple),
        part_pending_count=sum(item.pending_part_count for item in item_tuple),
        part_retrying_count=sum(item.retrying_part_count for item in item_tuple),
        part_failed_count=sum(item.failed_part_count for item in item_tuple),
        part_invalidated_count=sum(item.invalidated_part_count for item in item_tuple),
    )


def _mark_snapshot_influence(
    coverage: CoverageSummary,
    parts: tuple[SourcePartPlan, ...],
    units: tuple[EvidenceUnit, ...],
    payload: _SynthesisPayload,
) -> CoverageSummary:
    unit_parts = {unit.evidence_unit_id: unit.source_part_id for unit in units}
    part_entries = {part.part_id: part.entry_id for part in parts}
    influenced_entries = {
        part_entries[unit_parts[evidence_id]]
        for evidence_id in payload.evidence_unit_ids
        if evidence_id in unit_parts and unit_parts[evidence_id] in part_entries
    }
    return coverage.model_copy(
        update={
            "items": tuple(
                item.model_copy(
                    update={
                        "influenced_current_snapshot": item.entry_id in influenced_entries
                    }
                )
                for item in coverage.items
            )
        }
    )


def _coverage_without_influence(coverage: CoverageSummary) -> CoverageSummary:
    return coverage.model_copy(
        update={
            "items": tuple(
                item.model_copy(update={"influenced_current_snapshot": False})
                for item in coverage.items
            )
        }
    )


def _safe_validation_issues(error: BaseException) -> tuple[str, ...]:
    if isinstance(error, ValidationError):
        issues = []
        for item in error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )[:32]:
            location = ".".join(str(part) for part in item.get("loc", ())) or "root"
            issue_type = str(item.get("type", "invalid"))
            issues.append(f"{location}:{issue_type}")
        return tuple(issues) or ("root:invalid",)
    if isinstance(error, json.JSONDecodeError):
        return ("root:json_invalid",)
    return ("root:value_invalid",)


def _validate_provider_strings(value: object) -> None:
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            validate_safe_evidence_text(current)
        elif isinstance(current, dict):
            stack.extend(current.keys())
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
