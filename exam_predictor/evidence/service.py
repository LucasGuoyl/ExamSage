from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from hashlib import sha256
from typing import Literal, Protocol

from pydantic import ConfigDict, Field, field_validator

from exam_predictor.evidence.artifacts import (
    ArtifactCleanupState,
    EvidenceArtifactStore,
)
from exam_predictor.evidence.models import (
    CoverageSummary,
    EvidenceCitation,
    EvidenceFrozenModel,
    EvidenceUnit,
    PartState,
    SourcePartPlan,
    StudyMapSnapshot,
    validate_safe_evidence_text,
)
from exam_predictor.evidence.policy import source_priority
from exam_predictor.evidence.preparation import (
    ArchivePreviewAuthority,
    PreparedPartRequest,
    SourcePartPreparer,
    SourcePreparationError,
)
from exam_predictor.evidence.scheduler import (
    EvidenceAuthorizationError,
    EvidenceScheduler,
    SchedulerOutcome,
    SchedulerStatus,
)
from exam_predictor.evidence.store import EvidenceStore
from exam_predictor.evidence.study_map import (
    EvidenceAnswerContext,
    StudyMapBuilder,
)
from exam_predictor.workspace.models import (
    ApprovalRecord,
    ManifestEntry,
    ManifestRevision,
    SourceState,
    WorkspaceState,
)
from exam_predictor.workspace.store import (
    TransmissionAuthorityRevokedError,
    TransmissionAuthoritySnapshot,
    WorkspaceStore,
)
from exam_predictor.workspace.transmission import (
    SourceAuthorizationError,
    WorkspaceTransmissionGate,
)


EventEmitter = Callable[[str, dict[str, object]], None]


class EvidenceRunGuard(Protocol):
    def has_unsettled_runs(self, workspace_id: str) -> bool: ...


class EvidenceAnswerComposer(Protocol):
    def answer_from_evidence(self, request: EvidenceAnswerRequest) -> str: ...


class EvidenceServiceError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class EvidenceInspection(EvidenceFrozenModel):
    workspace_id: str
    revision_id: str | None = None
    approval_id: str | None = None
    approval_required: bool
    approved_source_count: int = Field(default=0, ge=0)
    approved_bytes: int = Field(default=0, ge=0)
    coverage: CoverageSummary | None = None
    snapshot: StudyMapSnapshot | None = None


class EvidenceRunResult(EvidenceFrozenModel):
    workspace_id: str
    revision_id: str
    status: Literal["complete", "paused"]
    snapshot: StudyMapSnapshot | None = None
    outcome: SchedulerOutcome
    safe_error_code: str | None = None


class EvidenceAnswerRequest(EvidenceFrozenModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    question: str = Field(min_length=1, max_length=4_000)
    context: EvidenceAnswerContext
    response_language: str | None = Field(default=None, max_length=64)

    @field_validator("question", "response_language")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("answer text must not be empty")
        return validate_safe_evidence_text(normalized)


class EvidenceAnswerResult(EvidenceFrozenModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    workspace_id: str
    revision_id: str
    snapshot_id: str
    answer: str
    citations: tuple[EvidenceCitation, ...]
    limitations: tuple[str, ...]

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("evidence answers must not be empty")
        return validate_safe_evidence_text(normalized)


class EvidenceService:
    """Workspace-facing evidence orchestration with approval-bound source reads."""

    def __init__(
        self,
        *,
        workspace_store: WorkspaceStore,
        transmission_gate: WorkspaceTransmissionGate,
        evidence_store: EvidenceStore,
        artifact_store: EvidenceArtifactStore,
        preparer: SourcePartPreparer,
        scheduler: EvidenceScheduler,
        study_map_builder: StudyMapBuilder,
        answer_composer: EvidenceAnswerComposer | None = None,
        run_guard: EvidenceRunGuard | None = None,
        emit: EventEmitter | None = None,
    ) -> None:
        self._workspace_store = workspace_store
        self._transmission_gate = transmission_gate
        self._evidence_store = evidence_store
        self._artifact_store = artifact_store
        self._preparer = preparer
        self._scheduler = scheduler
        self._builder = study_map_builder
        self._answer_composer = answer_composer
        self._run_guard = run_guard
        self._emit = emit or (lambda _event_type, _payload: None)

    def start(self) -> tuple[str, ...]:
        """Recover interrupted local state without starting provider work."""

        recovered = self._evidence_store.recover_unfinished()
        self._recover_workspace_deletions()
        return recovered

    def inspect(self, workspace_id: str) -> EvidenceInspection:
        self._ensure_workspace_active(workspace_id)
        authority = self._workspace_store.transmission_authority_snapshot(workspace_id)
        if authority is None:
            raise EvidenceServiceError("workspace_not_found")
        if not self._authority_is_approved(authority):
            workspace = authority.workspace
            return EvidenceInspection(
                workspace_id=workspace_id,
                revision_id=(
                    workspace.current_approved_revision_id
                    or workspace.current_draft_revision_id
                ),
                approval_required=True,
            )
        approval = authority.approval
        revision = authority.revision
        assert approval is not None and revision is not None
        entries = self._approved_entries(approval, revision)
        coverage = self._builder.coverage(workspace_id, revision.revision_id)
        return EvidenceInspection(
            workspace_id=workspace_id,
            revision_id=revision.revision_id,
            approval_id=approval.approval_id,
            approval_required=False,
            approved_source_count=len(entries),
            approved_bytes=sum(entry.size_bytes for entry in entries),
            coverage=coverage,
            snapshot=self._evidence_store.current_snapshot(
                workspace_id,
                revision.revision_id,
            ),
        )

    def build_study_map(
        self,
        workspace_id: str,
        run_id: str,
    ) -> EvidenceRunResult:
        authority = self._require_authority(workspace_id)
        self._prepare_current(authority)
        return self._analyze_current(authority, run_id)

    def continue_analysis(
        self,
        workspace_id: str,
        run_id: str,
    ) -> EvidenceRunResult:
        authority = self._require_authority(workspace_id)
        self._prepare_current(authority)
        return self._analyze_current(authority, run_id)

    def answer_from_evidence(
        self,
        workspace_id: str,
        question: str,
        *,
        response_language: str | None = None,
    ) -> EvidenceAnswerResult:
        authority = self._require_authority(workspace_id)
        revision = authority.revision
        assert revision is not None
        context = self._builder.answer_context(workspace_id, revision.revision_id)
        if context is None:
            raise EvidenceServiceError("evidence_not_ready")
        if not self._authority_matches(authority):
            raise EvidenceServiceError("source_approval_revoked")
        try:
            request = EvidenceAnswerRequest(
                question=question,
                context=context,
                response_language=response_language,
            )
        except Exception:
            raise EvidenceServiceError("evidence_question_invalid") from None
        approval = authority.approval
        assert approval is not None
        try:
            with self._workspace_store.hold_transmission_authority(
                workspace_id,
                approval_id=approval.approval_id,
                revision_id=revision.revision_id,
            ):
                self._ensure_workspace_active(workspace_id)
                if self._answer_composer is None:
                    answer = "\n\n".join(
                        unit.content for unit in context.evidence_units
                    )
                else:
                    answer = self._answer_composer.answer_from_evidence(request)
                return EvidenceAnswerResult(
                    workspace_id=workspace_id,
                    revision_id=revision.revision_id,
                    snapshot_id=context.snapshot.snapshot_id,
                    answer=answer,
                    citations=_citations(context.evidence_units),
                    limitations=context.limitations,
                )
        except TransmissionAuthorityRevokedError:
            raise EvidenceServiceError("source_approval_revoked") from None
        except EvidenceServiceError:
            raise
        except Exception:
            raise EvidenceServiceError("evidence_answer_failed") from None

    def delete_workspace_evidence(
        self,
        workspace_id: str,
    ) -> ArtifactCleanupState:
        try:
            with self._workspace_store.hold_workspace_authority(workspace_id):
                if self._run_guard is not None and self._run_guard.has_unsettled_runs(
                    workspace_id
                ):
                    raise EvidenceServiceError("evidence_runs_active")
                self._evidence_store.begin_workspace_deletion(workspace_id)
        except EvidenceServiceError:
            raise
        except Exception:
            raise EvidenceServiceError("evidence_delete_pending") from None
        try:
            state = self._artifact_store.delete_workspace(workspace_id)
        except Exception:
            raise EvidenceServiceError("evidence_delete_pending") from None
        if state is ArtifactCleanupState.DELETED:
            try:
                self._evidence_store.complete_workspace_deletion(workspace_id)
            except Exception:
                raise EvidenceServiceError("evidence_delete_pending") from None
        return state

    def _recover_workspace_deletions(self) -> None:
        for workspace_id in self._evidence_store.pending_workspace_deletions():
            try:
                with self._workspace_store.hold_workspace_authority(workspace_id):
                    if self._run_guard is not None and self._run_guard.has_unsettled_runs(
                        workspace_id
                    ):
                        continue
                    state = self._artifact_store.delete_workspace(workspace_id)
                    if state is ArtifactCleanupState.DELETED:
                        self._evidence_store.complete_workspace_deletion(workspace_id)
            except Exception:
                continue

    def _prepare_current(self, authority: TransmissionAuthoritySnapshot) -> None:
        approval = authority.approval
        revision = authority.revision
        assert approval is not None and revision is not None
        workspace_id = authority.workspace.workspace_id
        self._ensure_workspace_active(workspace_id)
        previous_revision_id = self._evidence_store.current_revision_id(workspace_id)
        if previous_revision_id not in {None, revision.revision_id}:
            self._invalidate_changed_entries(
                workspace_id,
                previous_revision_id,
                self._approved_entries(approval, revision),
            )
        existing = self._evidence_store.list_parts(workspace_id, revision.revision_id)
        prepared_entries = {
            part.entry_id
            for part in existing
            if part.state is not PartState.INVALIDATED
        }
        for entry in self._approved_entries(approval, revision):
            if entry.entry_id in prepared_entries:
                continue
            with self._workspace_store.hold_workspace_authority(workspace_id):
                self._ensure_workspace_active(workspace_id)
                if not self._authority_matches(authority):
                    raise SourceAuthorizationError(
                        "source_approval_revoked",
                        workspace_id,
                        entry.entry_id,
                    )
                descriptors = self._transmission_gate.authorize(
                    workspace_id,
                    (entry.entry_id,),
                )
                if len(descriptors) != 1:
                    raise SourceAuthorizationError(
                        "source_store_unavailable",
                        workspace_id,
                        entry.entry_id,
                    )
                descriptor = descriptors[0]
                request = PreparedPartRequest(
                    workspace_id=workspace_id,
                    revision_id=revision.revision_id,
                    entry_id=entry.entry_id,
                    relative_path=entry.relative_path,
                    format_category=entry.format_category or "unknown",
                    source_size_bytes=descriptor.size_bytes,
                    source_sha256=descriptor.sha256,
                    archive_previews=_archive_previews(revision, entry),
                )
                try:
                    with self._transmission_gate.open_approved(
                        descriptor.read_token
                    ) as stream:
                        plans = self._preparer.prepare(request, stream)
                except SourcePreparationError as error:
                    plans = (_failed_plan(request, entry, error),)
                    self._emit(
                        "part_failed",
                        {
                            "workspace_id": workspace_id,
                            "revision_id": revision.revision_id,
                            "entry_id": entry.entry_id,
                            "relative_path": entry.relative_path,
                            "locator": error.locator,
                            "code": error.code,
                        },
                    )
                self._evidence_store.upsert_part_plans(plans)
                prepared_entries.add(entry.entry_id)
                self._emit(
                    "source_prepared",
                    {
                        "workspace_id": workspace_id,
                        "revision_id": revision.revision_id,
                        "entry_id": entry.entry_id,
                        "relative_path": entry.relative_path,
                        "part_count": len(plans),
                    },
                )

    def _analyze_current(
        self,
        authority: TransmissionAuthoritySnapshot,
        run_id: str,
    ) -> EvidenceRunResult:
        revision = authority.revision
        assert revision is not None
        workspace_id = authority.workspace.workspace_id
        last_outcome = SchedulerOutcome(
            status=SchedulerStatus.PAUSED,
            pending_count=len(
                self._evidence_store.list_parts(workspace_id, revision.revision_id)
            ),
        )
        while True:
            self._ensure_workspace_active(workspace_id)
            if not self._authority_matches(authority):
                return self._paused_result(
                    workspace_id,
                    revision.revision_id,
                    last_outcome,
                    "source_approval_revoked",
                )
            approval = authority.approval
            assert approval is not None

            @contextmanager
            def part_authority_guard(_part: SourcePartPlan):
                try:
                    with self._workspace_store.hold_transmission_authority(
                        workspace_id,
                        approval_id=approval.approval_id,
                        revision_id=revision.revision_id,
                    ) as held:
                        self._ensure_workspace_active(workspace_id)
                        yield held
                except TransmissionAuthorityRevokedError:
                    raise EvidenceAuthorizationError from None

            last_outcome = self._scheduler.run_frontier(
                run_id,
                workspace_id,
                revision.revision_id,
                authorize_part=part_authority_guard,
            )
            self._ensure_workspace_active(workspace_id)
            if not self._authority_matches(authority):
                return self._paused_result(
                    workspace_id,
                    revision.revision_id,
                    last_outcome,
                    "source_approval_revoked",
                )
            @contextmanager
            def publication_guard():
                with self._workspace_store.hold_transmission_authority(
                    workspace_id,
                    approval_id=approval.approval_id,
                    revision_id=revision.revision_id,
                ) as held:
                    self._ensure_workspace_active(workspace_id)
                    yield held

            try:
                snapshot = self._builder.publish_initial(
                    workspace_id,
                    revision.revision_id,
                    publication_guard=publication_guard,
                    synthesis_guard=publication_guard,
                )
                complete = (
                    self._builder.publish_complete(
                        workspace_id,
                        revision.revision_id,
                        publication_guard=publication_guard,
                        synthesis_guard=publication_guard,
                    )
                    if last_outcome.pending_count == 0
                    else None
                )
            except TransmissionAuthorityRevokedError:
                return self._paused_result(
                    workspace_id,
                    revision.revision_id,
                    last_outcome,
                    "source_approval_revoked",
                )
            if last_outcome.pending_count == 0:
                if complete is not None:
                    return EvidenceRunResult(
                        workspace_id=workspace_id,
                        revision_id=revision.revision_id,
                        status="complete",
                        snapshot=complete,
                        outcome=last_outcome,
                    )
            if last_outcome.status is SchedulerStatus.PAUSED:
                return EvidenceRunResult(
                    workspace_id=workspace_id,
                    revision_id=revision.revision_id,
                    status="paused",
                    snapshot=snapshot,
                    outcome=last_outcome,
                )

    def _invalidate_changed_entries(
        self,
        workspace_id: str,
        previous_revision_id: str,
        current_entries: tuple[ManifestEntry, ...],
    ) -> None:
        current_hashes = {
            entry.entry_id: entry.sha256 for entry in current_entries
        }
        previous_parts = self._evidence_store.list_parts(
            workspace_id,
            previous_revision_id,
        )
        previous_hashes: dict[str, str] = {}
        for part in previous_parts:
            previous_hashes.setdefault(part.entry_id, part.source_sha256)
        for entry_id, old_hash in previous_hashes.items():
            if current_hashes.get(entry_id) != old_hash:
                self._builder.invalidate_entry(
                    workspace_id,
                    previous_revision_id,
                    entry_id,
                )

    def _require_authority(
        self,
        workspace_id: str,
    ) -> TransmissionAuthoritySnapshot:
        self._ensure_workspace_active(workspace_id)
        authority = self._workspace_store.transmission_authority_snapshot(workspace_id)
        if authority is None:
            raise SourceAuthorizationError("workspace_not_found", workspace_id, "")
        if not self._authority_is_approved(authority):
            raise SourceAuthorizationError(
                "source_approval_required",
                workspace_id,
                "",
            )
        return authority

    def _ensure_workspace_active(self, workspace_id: str) -> None:
        if self._evidence_store.workspace_deletion_pending(workspace_id):
            raise EvidenceServiceError("evidence_delete_pending")

    def _authority_matches(self, expected: TransmissionAuthoritySnapshot) -> bool:
        current = self._workspace_store.transmission_authority_snapshot(
            expected.workspace.workspace_id
        )
        if not self._authority_is_approved(current):
            return False
        assert current is not None
        return (
            current.workspace.current_draft_revision_id
            == expected.workspace.current_draft_revision_id
            and current.workspace.current_approved_revision_id
            == expected.workspace.current_approved_revision_id
            and current.approval == expected.approval
            and current.revision == expected.revision
        )

    @staticmethod
    def _authority_is_approved(
        authority: TransmissionAuthoritySnapshot | None,
    ) -> bool:
        return bool(
            authority is not None
            and authority.workspace.state is WorkspaceState.APPROVED
            and authority.approval is not None
            and authority.revision is not None
            and authority.workspace.current_draft_revision_id
            == authority.revision.revision_id
            and authority.workspace.current_approved_revision_id
            == authority.revision.revision_id
            and authority.approval.revision_id == authority.revision.revision_id
        )

    @staticmethod
    def _approved_entries(
        approval: ApprovalRecord,
        revision: ManifestRevision,
    ) -> tuple[ManifestEntry, ...]:
        approved_hashes = {
            entry.entry_id: entry.sha256 for entry in approval.entries
        }
        entries = tuple(
            entry
            for entry in revision.entries
            if entry.entry_id in approved_hashes
            and entry.sha256 == approved_hashes[entry.entry_id]
            and entry.item_kind == "file"
            and entry.archive_parent_entry_id is None
            and entry.included
            and entry.state is SourceState.APPROVED
        )
        if len(entries) != len(approval.entries):
            raise SourceAuthorizationError(
                "source_approval_mismatch",
                revision.workspace_id,
                "",
            )
        return tuple(sorted(entries, key=lambda item: (item.relative_path, item.entry_id)))

    @staticmethod
    def _paused_result(
        workspace_id: str,
        revision_id: str,
        outcome: SchedulerOutcome,
        code: str,
    ) -> EvidenceRunResult:
        return EvidenceRunResult(
            workspace_id=workspace_id,
            revision_id=revision_id,
            status="paused",
            outcome=outcome,
            safe_error_code=code,
        )


def _archive_previews(
    revision: ManifestRevision,
    parent: ManifestEntry,
) -> tuple[ArchivePreviewAuthority, ...]:
    assert parent.sha256 is not None
    return tuple(
        ArchivePreviewAuthority(
            workspace_id=revision.workspace_id,
            revision_id=revision.revision_id,
            parent_entry_id=parent.entry_id,
            parent_source_sha256=parent.sha256,
            entry=entry,
            approved=True,
        )
        for entry in revision.entries
        if entry.archive_parent_entry_id == parent.entry_id
        and entry.inclusion_reason == "archive_preview"
        and entry.failure_code is None
    )


def _failed_plan(
    request: PreparedPartRequest,
    entry: ManifestEntry,
    error: SourcePreparationError,
) -> SourcePartPlan:
    priority, scheduling_class = source_priority(
        request.relative_path,
        request.format_category,
    )
    identity = sha256(
        (
            f"{request.workspace_id}\0{request.revision_id}\0{request.entry_id}\0"
            f"{request.source_sha256}\0{error.locator}\0{error.code}"
        ).encode("utf-8")
    ).hexdigest()
    return SourcePartPlan(
        part_id=f"part_{identity}",
        workspace_id=request.workspace_id,
        revision_id=request.revision_id,
        entry_id=request.entry_id,
        relative_path=request.relative_path,
        source_sha256=request.source_sha256,
        part_sha256=sha256(error.code.encode("utf-8")).hexdigest(),
        ordinal=0,
        locator=error.locator,
        media_type="application/octet-stream",
        size_bytes=entry.size_bytes,
        scheduling_class=scheduling_class,
        priority=priority,
        state=PartState.FAILED,
        idempotency_key=f"prepare_failure_{identity}",
    )


def _citations(units: tuple[EvidenceUnit, ...]) -> tuple[EvidenceCitation, ...]:
    citations: dict[str, EvidenceCitation] = {}
    for unit in units:
        for citation in unit.citations:
            citations.setdefault(citation.citation_id, citation)
    return tuple(citations[citation_id] for citation_id in sorted(citations))
