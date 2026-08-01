from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from threading import Event

import pytest

from exam_predictor.evidence.artifacts import ArtifactCleanupState, EvidenceArtifactStore
from exam_predictor.evidence.policy import EvidencePolicy
from exam_predictor.evidence.preparation import (
    SourcePartPreparer,
    SourcePreparationError,
)
from exam_predictor.evidence.providers import (
    AnalyzeSourcePartRequest,
    EvidencePartResult,
    EvidenceProviderError,
    EvidenceRouteIdentity,
)
from exam_predictor.evidence.scheduler import EvidenceScheduler
from exam_predictor.evidence.service import EvidenceService
from exam_predictor.evidence.service import EvidenceServiceError
from exam_predictor.evidence.store import EvidenceStore
from exam_predictor.evidence.study_map import (
    ApprovedCoverageEntry,
    EvidenceValidator,
    StudyMapBuilder,
    StudyMapSynthesisRequest,
)
from exam_predictor.runtime.control import RunControlRegistry
from exam_predictor.workspace.models import (
    SourceMode,
    WorkspaceJob,
    WorkspaceJobStatus,
    WorkspaceRecord,
    WorkspaceState,
)
from exam_predictor.workspace.scanner import WorkspaceScanner
from exam_predictor.workspace.store import WorkspaceStore
from exam_predictor.workspace.transmission import (
    SourceAuthorizationError,
    WorkspaceTransmissionGate,
)


NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)
WORKSPACE_ID = "workspace_service_0001"


class RecordingGate(WorkspaceTransmissionGate):
    def __init__(self, store: WorkspaceStore) -> None:
        super().__init__(store)
        self.descriptors = []

    def authorize(self, workspace_id: str, entry_ids):
        descriptors = super().authorize(workspace_id, entry_ids)
        self.descriptors.extend(descriptors)
        return descriptors


class RecordingProvider:
    def __init__(self, on_call=None) -> None:
        self.requests: list[AnalyzeSourcePartRequest] = []
        self._on_call = on_call

    def route_identity(self, model_route: str) -> EvidenceRouteIdentity:
        return EvidenceRouteIdentity(
            provider="openai",
            model_route=model_route,
            model_id="evidence-test-model",
            capability_fingerprint="a" * 64,
        )

    def analyze_source_part(
        self,
        request: AnalyzeSourcePartRequest,
    ) -> EvidencePartResult:
        self.requests.append(request)
        if self._on_call is not None:
            self._on_call(len(self.requests), request)
        return EvidencePartResult(
            source_part_id=request.source_part_id,
            locator=request.locator,
            provider="openai",
            model_id="evidence-test-model",
            prompt_version="source-analysis-v1",
            raw_output=json.dumps(
                {
                    "locator": request.locator,
                    "detected_language": "en",
                    "material_role": "course material",
                    "headings": ["Core topic"],
                    "concepts": ["bounded evidence"],
                    "definitions": [],
                    "formulas": [],
                    "procedures": [],
                    "examples": [],
                    "assessment_items": [],
                    "visual_descriptions": [],
                    "ocr_text": [],
                    "limitations": [],
                    "warnings": [],
                    "prompt_injection_indicators": [],
                }
            ),
        )


class Synthesizer:
    def synthesize_study_map(self, request: StudyMapSynthesisRequest) -> str:
        evidence_ids = tuple(item.evidence_unit_id for item in request.evidence)
        if not evidence_ids:
            evidence_ids = tuple(
                evidence_id
                for draft in request.drafts
                for evidence_id in json.loads(draft)["evidence_unit_ids"]
            )
        return json.dumps(
            {
                "course_groups": [
                    {
                        "group_id": "group-course",
                        "title": "Course",
                        "confidence": 0.8,
                        "evidence_unit_ids": list(evidence_ids),
                    }
                ],
                "nodes": [
                    {
                        "node_id": "node-core",
                        "title": "Core topic",
                        "parent_node_id": None,
                        "prerequisite_node_ids": [],
                        "course_group_id": "group-course",
                        "focus_score": 0.75,
                        "confidence": 0.8,
                        "evidence_unit_ids": list(evidence_ids),
                    }
                ],
                "limitations": [],
                "evidence_unit_ids": list(evidence_ids),
            }
        )


def _scan(
    store: WorkspaceStore,
    root: Path,
    *,
    job_suffix: str,
    create_workspace: bool,
    approve: bool,
):
    execution = WorkspaceScanner().scan_with_identity(WORKSPACE_ID, root)
    if create_workspace:
        store.create_workspace(
            WorkspaceRecord(
                workspace_id=WORKSPACE_ID,
                display_name="Course",
                source_mode=SourceMode.NATIVE_FOLDER,
                canonical_root=execution.canonical_root,
                root_device=execution.root_device,
                root_file_id=execution.root_file_id,
                state=WorkspaceState.SCANNING,
                created_at=NOW,
                updated_at=NOW,
            )
        )
    job = WorkspaceJob(
        job_id=f"scan_service_{job_suffix}",
        workspace_id=WORKSPACE_ID,
        job_kind="scan",
        status=WorkspaceJobStatus.QUEUED,
        idempotency_key=f"request_service_{job_suffix}",
        created_at=NOW,
    )
    store.create_job(job, job.idempotency_key)
    store.start_job(job.job_id)
    revision = store.commit_scan(WORKSPACE_ID, execution.result, job.job_id)
    if approve:
        store.approve(WORKSPACE_ID, revision.revision_id, revision.policy_version)
    return revision


def _workspace(
    store: WorkspaceStore,
    tmp_path: Path,
    sources: dict[str, bytes],
    *,
    approve: bool = True,
):
    root = tmp_path / "course"
    root.mkdir()
    for relative_path, content in sources.items():
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    revision = _scan(
        store,
        root,
        job_suffix="initial",
        create_workspace=True,
        approve=approve,
    )
    return root, revision


def _coverage_source(workspace_store: WorkspaceStore):
    def load(workspace_id: str, revision_id: str):
        authority = workspace_store.transmission_authority_snapshot(workspace_id)
        assert authority is not None
        assert authority.approval is not None
        assert authority.revision is not None
        assert authority.revision.revision_id == revision_id
        approved_ids = {entry.entry_id for entry in authority.approval.entries}
        return tuple(
            ApprovedCoverageEntry(
                entry_id=entry.entry_id,
                relative_path=entry.relative_path,
                approved_bytes=entry.size_bytes,
            )
            for entry in authority.revision.entries
            if entry.entry_id in approved_ids
        )

    return load


def _service(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
    provider: RecordingProvider,
    *,
    answer_composer=None,
    run_guard=None,
    scheduler_emit=None,
    synthesizer=None,
    policy=None,
    wall_clock=None,
    monotonic_clock=None,
):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(exist_ok=True)
    artifacts = EvidenceArtifactStore(artifact_root)
    evidence_store = EvidenceStore(tmp_path / "evidence.sqlite3")
    policy = policy or EvidencePolicy(multimodal_concurrency=1)
    gate = RecordingGate(workspace_store)
    builder = StudyMapBuilder(
        evidence_store,
        artifacts,
        synthesizer or Synthesizer(),
        coverage_source=_coverage_source(workspace_store),
        policy=policy,
        now=lambda: NOW,
        monotonic_clock=monotonic_clock,
    )
    scheduler = EvidenceScheduler(
        evidence_store,
        artifacts,
        provider,
        EvidenceValidator(),
        RunControlRegistry(),
        emit=scheduler_emit,
        policy=policy,
        wall_clock=wall_clock or (lambda: NOW),
        monotonic_clock=monotonic_clock,
    )
    service = EvidenceService(
        workspace_store=workspace_store,
        transmission_gate=gate,
        evidence_store=evidence_store,
        artifact_store=artifacts,
        preparer=SourcePartPreparer(artifacts, policy=policy),
        scheduler=scheduler,
        study_map_builder=builder,
        answer_composer=answer_composer,
        run_guard=run_guard,
        policy=policy,
        wall_clock=wall_clock or (lambda: NOW),
        monotonic_clock=monotonic_clock,
    )
    return service, gate, evidence_store, artifacts


class _ServiceClock:
    def __init__(self) -> None:
        self.current = NOW
        self.monotonic_value = 0.0

    def wall(self) -> datetime:
        return self.current

    def monotonic(self) -> float:
        return self.monotonic_value

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)
        self.monotonic_value += seconds


@pytest.fixture
def workspace_store(tmp_path: Path):
    store = WorkspaceStore(tmp_path / "workspace.sqlite3")
    try:
        yield store
    finally:
        store.close()


def test_unapproved_workspace_requires_approval_before_any_provider_call(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _workspace(workspace_store, tmp_path, {"notes.txt": b"private notes"}, approve=False)
    provider = RecordingProvider()
    service, _gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
    )
    try:
        with pytest.raises(SourceAuthorizationError) as caught:
            service.build_study_map(WORKSPACE_ID, "run_unapproved_0001")
        assert caught.value.code == "source_approval_required"
        assert provider.requests == []
    finally:
        artifacts.close()
        evidence_store.close()


def test_approved_workspace_with_no_selected_sources_fails_without_loop_or_provider(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _root, revision = _workspace(
        workspace_store,
        tmp_path,
        {"notes.txt": b"not selected"},
        approve=False,
    )
    entry = next(item for item in revision.entries if item.included)
    empty_revision = workspace_store.set_inclusion(
        WORKSPACE_ID,
        revision.revision_id,
        (entry.entry_id,),
        False,
    )
    workspace_store.approve(
        WORKSPACE_ID,
        empty_revision.revision_id,
        empty_revision.policy_version,
    )
    provider = RecordingProvider()
    service, _gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
    )
    try:
        with pytest.raises(EvidenceServiceError) as empty:
            service.build_study_map(WORKSPACE_ID, "run_empty_sources_01")
        assert empty.value.code == "evidence_sources_empty"
        assert provider.requests == []
        assert evidence_store.list_parts(
            WORKSPACE_ID,
            empty_revision.revision_id,
        ) == ()
    finally:
        artifacts.close()
        evidence_store.close()


def test_prepare_analysis_never_returns_a_new_revision_it_did_not_prepare(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _root, revision = _workspace(
        workspace_store,
        tmp_path,
        {"a.txt": b"first", "b.txt": b"second"},
    )
    excluded = revision.entries[0]
    provider = RecordingProvider()
    service, _gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
    )
    original_inspection = service._inspection_from_authority
    replacement_revision_id = None

    def approve_replacement(authority):
        nonlocal replacement_revision_id
        inspection = original_inspection(authority)
        replacement = workspace_store.set_inclusion(
            WORKSPACE_ID,
            revision.revision_id,
            (excluded.entry_id,),
            False,
        )
        workspace_store.approve(
            WORKSPACE_ID,
            replacement.revision_id,
            replacement.policy_version,
        )
        replacement_revision_id = replacement.revision_id
        return inspection

    service._inspection_from_authority = approve_replacement
    try:
        with pytest.raises(SourceAuthorizationError) as changed:
            service.prepare_analysis(WORKSPACE_ID)
        assert changed.value.code == "source_approval_revoked"
        assert len(evidence_store.list_parts(WORKSPACE_ID, revision.revision_id)) == 2
        assert replacement_revision_id is not None
        assert evidence_store.list_parts(
            WORKSPACE_ID,
            replacement_revision_id,
        ) == ()
        assert provider.requests == []
    finally:
        artifacts.close()
        evidence_store.close()


def test_service_consumes_one_gate_spool_and_provider_receives_only_prepared_bytes(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    source = b"approved course evidence"
    _workspace(workspace_store, tmp_path, {"notes.txt": source})
    provider = RecordingProvider()
    service, gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
    )
    try:
        result = service.build_study_map(WORKSPACE_ID, "run_build_00000001")
        assert result.status == "complete"
        assert result.snapshot is not None
        assert [request.content_bytes for request in provider.requests] == [source]
        assert len(gate.descriptors) == 1
        with pytest.raises(SourceAuthorizationError) as consumed:
            with gate.open_approved(gate.descriptors[0].read_token):
                pass
        assert consumed.value.code == "read_token_invalid"
    finally:
        artifacts.close()
        evidence_store.close()


def test_one_run_keeps_one_deadline_across_frontiers(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _workspace(
        workspace_store,
        tmp_path,
        {"a.txt": b"first", "b.txt": b"second"},
    )
    clock = _ServiceClock()
    provider = RecordingProvider()
    policy = EvidencePolicy(
        multimodal_concurrency=1,
        tool_deadline_seconds=60,
    )
    service, _gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
        policy=policy,
        wall_clock=clock.wall,
        monotonic_clock=clock.monotonic,
    )
    try:
        inspection = service.prepare_analysis(WORKSPACE_ID)
        assert inspection.revision_id is not None
        first = service.analyze_frontier(
            WORKSPACE_ID,
            inspection.revision_id,
            "run_deadline_0001",
        )
        clock.advance(61)
        second = service.analyze_frontier(
            WORKSPACE_ID,
            inspection.revision_id,
            "run_deadline_0001",
        )
    finally:
        artifacts.close()
        evidence_store.close()

    assert first.outcome.processed_part_ids
    assert second.outcome.status.value == "paused"
    assert len(provider.requests) == 1


def test_expired_run_deadline_prevents_new_synthesis_after_frontier(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _workspace(workspace_store, tmp_path, {"notes.txt": b"only source"})
    clock = _ServiceClock()

    class RecordingSynthesizer(Synthesizer):
        def __init__(self) -> None:
            self.requests: list[StudyMapSynthesisRequest] = []

        def synthesize_study_map(self, request: StudyMapSynthesisRequest) -> str:
            self.requests.append(request)
            return super().synthesize_study_map(request)

    synthesizer = RecordingSynthesizer()
    provider = RecordingProvider(on_call=lambda *_args: clock.advance(61))
    service, _gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
        synthesizer=synthesizer,
        policy=EvidencePolicy(
            multimodal_concurrency=1,
            tool_deadline_seconds=60,
        ),
        wall_clock=clock.wall,
        monotonic_clock=clock.monotonic,
    )
    run_id = "run_publish_deadline_0001"
    try:
        inspection = service.prepare_analysis(WORKSPACE_ID, run_id)
        assert inspection.revision_id is not None
        frontier = service.analyze_frontier(
            WORKSPACE_ID,
            inspection.revision_id,
            run_id,
        )
        result = service.publish_frontier(
            WORKSPACE_ID,
            inspection.revision_id,
            frontier.outcome,
            run_id=run_id,
        )
    finally:
        artifacts.close()
        evidence_store.close()

    assert frontier.outcome.processed_part_ids
    assert result.status == "paused"
    assert synthesizer.requests == []


def test_preparation_policy_upgrade_reprepares_current_revision(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _root, revision = _workspace(
        workspace_store,
        tmp_path,
        {"notes.txt": b"approved evidence"},
    )
    first_service, _gate, first_store, first_artifacts = _service(
        tmp_path,
        workspace_store,
        RecordingProvider(),
        policy=EvidencePolicy(policy_version="evidence-v1"),
    )
    first_service.prepare_analysis(WORKSPACE_ID)
    first_parts = first_store.list_parts(WORKSPACE_ID, revision.revision_id)
    first_artifacts.close()
    first_store.close()

    second_service, _gate, second_store, second_artifacts = _service(
        tmp_path,
        workspace_store,
        RecordingProvider(),
        policy=EvidencePolicy(policy_version="evidence-v2"),
    )
    try:
        second_service.prepare_analysis(WORKSPACE_ID)
        second_parts = second_store.list_parts(WORKSPACE_ID, revision.revision_id)
    finally:
        second_artifacts.close()
        second_store.close()

    assert len(first_parts) == len(second_parts) == 1
    assert second_parts[0].part_id != first_parts[0].part_id
    assert second_parts[0].preparation_policy_version == "evidence-v2"


def test_exclusion_between_frontiers_pauses_before_the_second_provider_call(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _root, revision = _workspace(
        workspace_store,
        tmp_path,
        {"a.txt": b"first", "b.txt": b"second"},
    )
    included = tuple(entry for entry in revision.entries if entry.included)

    def exclude_after_first(event_type: str, _payload: dict[str, object]):
        if event_type == "part_processed":
            workspace_store.set_inclusion(
                WORKSPACE_ID,
                revision.revision_id,
                (included[1].entry_id,),
                False,
            )

    provider = RecordingProvider()
    service, _gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
        scheduler_emit=exclude_after_first,
    )
    try:
        result = service.build_study_map(WORKSPACE_ID, "run_exclusion_0001")
        assert result.status == "paused"
        assert result.safe_error_code == "source_approval_revoked"
        assert len(provider.requests) == 1
    finally:
        artifacts.close()
        evidence_store.close()


def test_approval_is_rechecked_before_a_retry_in_the_same_frontier(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _root, revision = _workspace(
        workspace_store,
        tmp_path,
        {"retry.txt": b"retry evidence"},
    )
    entry = next(item for item in revision.entries if item.included)

    class RetryProvider(RecordingProvider):
        def analyze_source_part(self, request: AnalyzeSourcePartRequest):
            self.requests.append(request)
            if len(self.requests) == 1:
                raise EvidenceProviderError("provider_rate_limited", retryable=True)
            return super().analyze_source_part(request)

    def revoke_before_retry(event_type: str, _payload: dict[str, object]):
        if event_type == "part_retrying":
            workspace_store.set_inclusion(
                WORKSPACE_ID,
                revision.revision_id,
                (entry.entry_id,),
                False,
            )

    provider = RetryProvider()
    service, _gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
        scheduler_emit=revoke_before_retry,
    )
    try:
        result = service.build_study_map(WORKSPACE_ID, "run_retry_revoke_01")
        assert result.status == "paused"
        assert result.safe_error_code == "source_approval_revoked"
        assert len(provider.requests) == 1
    finally:
        artifacts.close()
        evidence_store.close()


def test_one_changed_file_invalidates_only_its_units_and_reuses_the_sibling_cache(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    root, first_revision = _workspace(
        workspace_store,
        tmp_path,
        {"changed.txt": b"old content", "stable.txt": b"stable content"},
    )
    provider = RecordingProvider()
    service, _gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
    )
    try:
        first = service.build_study_map(WORKSPACE_ID, "run_first_00000001")
        assert first.status == "complete"
        old_parts = evidence_store.list_parts(WORKSPACE_ID, first_revision.revision_id)
        old_units = {
            part.entry_id: next(
                unit
                for unit in evidence_store.list_evidence_units(
                    WORKSPACE_ID,
                    first_revision.revision_id,
                )
                if unit.source_part_id == part.part_id
            )
            for part in old_parts
        }
        changed_entry = next(
            entry for entry in first_revision.entries if entry.relative_path == "changed.txt"
        )
        stable_entry = next(
            entry for entry in first_revision.entries if entry.relative_path == "stable.txt"
        )

        (root / "changed.txt").write_bytes(b"new content")
        second_revision = _scan(
            workspace_store,
            root,
            job_suffix="changed",
            create_workspace=False,
            approve=True,
        )
        second = service.build_study_map(WORKSPACE_ID, "run_second_0000001")

        assert second.status == "complete"
        assert len(provider.requests) == 3
        assert provider.requests[-1].content_bytes == b"new content"
        remaining_old = evidence_store.list_evidence_units(
            WORKSPACE_ID,
            first_revision.revision_id,
        )
        assert old_units[stable_entry.entry_id] in remaining_old
        assert old_units[changed_entry.entry_id] not in remaining_old
        assert first.snapshot is not None
        assert evidence_store.get_snapshot(first.snapshot.snapshot_id) is None
        assert evidence_store.current_snapshot(
            WORKSPACE_ID,
            second_revision.revision_id,
        ) is not None
    finally:
        artifacts.close()
        evidence_store.close()


def test_inspection_does_not_read_or_transmit_unapproved_sources(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _workspace(workspace_store, tmp_path, {"notes.txt": b"private"}, approve=False)
    provider = RecordingProvider()
    service, gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
    )
    try:
        inspection = service.inspect(WORKSPACE_ID)
        assert inspection.approval_required is True
        assert inspection.approved_source_count == 0
        assert inspection.coverage is None
        assert gate.descriptors == []
        assert provider.requests == []
    finally:
        artifacts.close()
        evidence_store.close()


def test_explicit_second_build_retries_a_terminal_provider_failure(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _workspace(workspace_store, tmp_path, {"notes.txt": b"approved evidence"})
    failing = True

    def fail_while_enabled(_call_number, _request):
        if failing:
            raise RuntimeError("safe validation boundary")

    provider = RecordingProvider(on_call=fail_while_enabled)
    service, _gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
    )
    try:
        first = service.build_study_map(WORKSPACE_ID, "run_failed_first")
        assert first.status == "complete"
        first_inspection = service.inspect(WORKSPACE_ID)
        assert first_inspection.coverage is not None
        assert first_inspection.coverage.part_failed_count == 1
        assert first_inspection.coverage.items[0].failure_codes == (
            "evidence_validation_failed",
        )
        assert first.snapshot is not None
        first_snapshot_id = first.snapshot.snapshot_id
        calls_after_failure = len(provider.requests)

        failing = False
        second = service.build_study_map(WORKSPACE_ID, "run_failed_retry")
        assert second.status == "complete"
        second_inspection = service.inspect(WORKSPACE_ID)
        assert second_inspection.coverage is not None
        assert second_inspection.coverage.part_failed_count == 0
        assert second_inspection.coverage.part_processed_count == 1
        assert len(provider.requests) == calls_after_failure + 1
        assert second.snapshot is not None
        assert second.snapshot.snapshot_id != first_snapshot_id
        assert len(second.snapshot.evidence_unit_ids) == 1
        assert len(second.snapshot.citations) == 1
        assert second.snapshot.coverage is not None
        assert second.snapshot.coverage.part_processed_count == 1
        assert second.snapshot.coverage.part_failed_count == 0
    finally:
        artifacts.close()
        evidence_store.close()


def test_explicit_second_build_reprepares_a_terminal_converter_failure(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _workspace(workspace_store, tmp_path, {"notes.txt": b"approved evidence"})
    provider = RecordingProvider()
    service, _gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
    )
    original = service._preparer

    class FailOncePreparer:
        calls = 0

        def prepare(self, request, stream):
            self.calls += 1
            if self.calls == 1:
                raise SourcePreparationError("converter_failed", "source")
            return original.prepare(request, stream)

    service._preparer = FailOncePreparer()
    try:
        first = service.build_study_map(WORKSPACE_ID, "run_prepare_failed")
        assert first.status == "complete"
        first_inspection = service.inspect(WORKSPACE_ID)
        assert first_inspection.coverage is not None
        assert first_inspection.coverage.items[0].failure_codes == (
            "converter_failed",
        )
        assert provider.requests == []

        second = service.build_study_map(WORKSPACE_ID, "run_prepare_retry")
        assert second.status == "complete"
        second_inspection = service.inspect(WORKSPACE_ID)
        assert second_inspection.coverage is not None
        assert second_inspection.coverage.part_failed_count == 0
        assert second_inspection.coverage.part_processed_count == 1
        assert len(provider.requests) == 1
    finally:
        artifacts.close()
        evidence_store.close()


def test_answer_uses_only_stored_evidence_and_returns_exact_citations(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    root, _revision = _workspace(
        workspace_store,
        tmp_path,
        {"notes.txt": b"original approved evidence"},
    )

    class Composer:
        def __init__(self) -> None:
            self.requests = []

        def answer_from_evidence(self, request):
            self.requests.append(request)
            return "Answer grounded in the stored course evidence."

    composer = Composer()
    provider = RecordingProvider()
    service, gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
        answer_composer=composer,
    )
    try:
        built = service.build_study_map(WORKSPACE_ID, "run_answer_build_01")
        assert built.status == "complete"
        consumed_count = len(gate.descriptors)
        (root / "notes.txt").write_bytes(b"changed outside the stored evidence")

        answer = service.answer_from_evidence(
            WORKSPACE_ID,
            "What is covered?",
            response_language="English",
        )

        assert answer.answer == "Answer grounded in the stored course evidence."
        assert answer.citations
        assert len(composer.requests) == 1
        assert composer.requests[0].context.evidence_units[0].content
        assert len(gate.descriptors) == consumed_count
        assert len(provider.requests) == 1
    finally:
        artifacts.close()
        evidence_store.close()


def test_answer_rechecks_approval_before_calling_the_composer(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _root, revision = _workspace(
        workspace_store,
        tmp_path,
        {"notes.txt": b"approved evidence"},
    )
    entry = next(item for item in revision.entries if item.included)

    class Composer:
        calls = 0

        def answer_from_evidence(self, _request):
            self.calls += 1
            return "must not be called"

    composer = Composer()
    provider = RecordingProvider()
    service, _gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
        answer_composer=composer,
    )
    try:
        assert service.build_study_map(
            WORKSPACE_ID,
            "run_answer_revoke_build",
        ).status == "complete"
        original_context = service._builder.answer_context

        def revoke_then_return(*args, **kwargs):
            context = original_context(*args, **kwargs)
            workspace_store.set_inclusion(
                WORKSPACE_ID,
                revision.revision_id,
                (entry.entry_id,),
                False,
            )
            return context

        service._builder.answer_context = revoke_then_return
        with pytest.raises(EvidenceServiceError) as revoked:
            service.answer_from_evidence(WORKSPACE_ID, "What is covered?")
        assert revoked.value.code == "source_approval_revoked"
        assert composer.calls == 0
    finally:
        artifacts.close()
        evidence_store.close()


def test_answer_composer_failures_and_unsafe_output_are_always_redacted(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _workspace(workspace_store, tmp_path, {"notes.txt": b"approved evidence"})
    provider = RecordingProvider()
    service, _gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
    )
    secret = "sk-" + "proj-" + "abcdefghijklmnopqrstuvwxyz123456"
    try:
        assert service.build_study_map(
            WORKSPACE_ID,
            "run_answer_secret_build",
        ).status == "complete"

        class RaisingComposer:
            def answer_from_evidence(self, _request):
                raise RuntimeError(secret)

        service._answer_composer = RaisingComposer()
        with pytest.raises(EvidenceServiceError) as raised:
            service.answer_from_evidence(WORKSPACE_ID, "What is covered?")
        assert raised.value.code == "evidence_answer_failed"
        assert secret not in str(raised.value)

        class UnsafeComposer:
            def answer_from_evidence(self, _request):
                return f"Leaked credential: {secret}"

        service._answer_composer = UnsafeComposer()
        with pytest.raises(EvidenceServiceError) as unsafe:
            service.answer_from_evidence(WORKSPACE_ID, "What is covered?")
        assert unsafe.value.code == "evidence_answer_failed"
        assert secret not in str(unsafe.value)
    finally:
        artifacts.close()
        evidence_store.close()


def test_start_recovers_running_parts_without_starting_provider_work(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _workspace(workspace_store, tmp_path, {"notes.txt": b"approved evidence"})
    provider = RecordingProvider()
    service, _gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
    )
    try:
        authority = workspace_store.transmission_authority_snapshot(WORKSPACE_ID)
        assert authority is not None and authority.revision is not None
        service._prepare_current(authority)
        claimed = evidence_store.claim_parts_with_tokens(
            WORKSPACE_ID,
            authority.revision.revision_id,
            limit=1,
            now=NOW,
        )
        assert len(claimed) == 1

        assert service.start() == (claimed[0].plan.part_id,)
        assert evidence_store.get_part(claimed[0].plan.part_id).state.value == "retry_wait"
        assert provider.requests == []
    finally:
        artifacts.close()
        evidence_store.close()


def test_workspace_evidence_deletion_waits_for_unsettled_runs(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _workspace(workspace_store, tmp_path, {"notes.txt": b"approved evidence"})

    class Guard:
        unsettled = True

        def has_unsettled_runs(self, _workspace_id: str) -> bool:
            return self.unsettled

    guard = Guard()
    provider = RecordingProvider()
    service, _gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
        run_guard=guard,
    )
    try:
        built = service.build_study_map(WORKSPACE_ID, "run_delete_build_01")
        assert built.status == "complete"
        with pytest.raises(EvidenceServiceError) as active:
            service.delete_workspace_evidence(WORKSPACE_ID)
        assert active.value.code == "evidence_runs_active"
        assert evidence_store.current_revision_id(WORKSPACE_ID) == built.revision_id

        guard.unsettled = False
        assert service.delete_workspace_evidence(WORKSPACE_ID).value == "deleted"
        assert evidence_store.current_revision_id(WORKSPACE_ID) is None
    finally:
        artifacts.close()
        evidence_store.close()


def test_start_finishes_a_durable_deletion_after_database_cleanup_failed(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _workspace(workspace_store, tmp_path, {"notes.txt": b"approved evidence"})
    provider = RecordingProvider()
    service, _gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
    )
    try:
        built = service.build_study_map(WORKSPACE_ID, "run_delete_recovery_01")
        assert built.status == "complete"
        original_complete = evidence_store.complete_workspace_deletion

        def fail_database_cleanup(_workspace_id: str):
            raise RuntimeError("private database failure")

        evidence_store.complete_workspace_deletion = fail_database_cleanup
        with pytest.raises(EvidenceServiceError) as pending:
            service.delete_workspace_evidence(WORKSPACE_ID)
        assert pending.value.code == "evidence_delete_pending"
        assert evidence_store.pending_workspace_deletions() == (WORKSPACE_ID,)
        assert evidence_store.current_revision_id(WORKSPACE_ID) == built.revision_id
        provider_call_count = len(provider.requests)

        with pytest.raises(EvidenceServiceError) as answer_blocked:
            service.answer_from_evidence(WORKSPACE_ID, "What is covered?")
        assert answer_blocked.value.code == "evidence_delete_pending"
        with pytest.raises(EvidenceServiceError) as build_blocked:
            service.build_study_map(WORKSPACE_ID, "run_delete_blocked_01")
        assert build_blocked.value.code == "evidence_delete_pending"
        with pytest.raises(EvidenceServiceError) as inspect_blocked:
            service.inspect(WORKSPACE_ID)
        assert inspect_blocked.value.code == "evidence_delete_pending"
        assert len(provider.requests) == provider_call_count

        evidence_store.complete_workspace_deletion = original_complete
        assert service.start() == ()
        assert evidence_store.pending_workspace_deletions() == ()
        assert evidence_store.current_revision_id(WORKSPACE_ID) is None
        assert provider.requests == provider.requests[:1]
    finally:
        artifacts.close()
        evidence_store.close()


def test_start_does_not_replay_a_pending_deletion_while_runs_are_unsettled(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _workspace(workspace_store, tmp_path, {"notes.txt": b"approved evidence"})

    class Guard:
        unsettled = False

        def has_unsettled_runs(self, _workspace_id: str) -> bool:
            return self.unsettled

    guard = Guard()
    provider = RecordingProvider()
    service, _gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
        run_guard=guard,
    )
    try:
        built = service.build_study_map(WORKSPACE_ID, "run_delete_guard_01")
        assert built.status == "complete"
        evidence_store.begin_workspace_deletion(WORKSPACE_ID)
        guard.unsettled = True

        assert service.start() == ()
        assert evidence_store.pending_workspace_deletions() == (WORKSPACE_ID,)
        assert evidence_store.current_revision_id(WORKSPACE_ID) == built.revision_id

        guard.unsettled = False
        assert service.start() == ()
        assert evidence_store.pending_workspace_deletions() == ()
        assert evidence_store.current_revision_id(WORKSPACE_ID) is None
    finally:
        artifacts.close()
        evidence_store.close()


def test_synthesis_holds_only_same_workspace_authority_and_never_runs_after_revoke(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _root, revision = _workspace(
        workspace_store,
        tmp_path,
        {"notes.txt": b"approved evidence"},
    )
    entry = next(item for item in revision.entries if item.included)
    synthesis_entered = Event()
    synthesis_release = Event()
    mutation_started = Event()
    mutation_completed = Event()

    class BlockingSynthesizer(Synthesizer):
        def __init__(self) -> None:
            self.started_after_revoke: list[bool] = []

        def synthesize_study_map(self, request: StudyMapSynthesisRequest) -> str:
            self.started_after_revoke.append(mutation_completed.is_set())
            synthesis_entered.set()
            assert synthesis_release.wait(timeout=5)
            return super().synthesize_study_map(request)

    synthesizer = BlockingSynthesizer()
    provider = RecordingProvider()
    service, _gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
        synthesizer=synthesizer,
    )

    def revoke_source():
        mutation_started.set()
        changed = workspace_store.set_inclusion(
            WORKSPACE_ID,
            revision.revision_id,
            (entry.entry_id,),
            False,
        )
        mutation_completed.set()
        return changed

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            build = pool.submit(
                service.build_study_map,
                WORKSPACE_ID,
                "run_synthesis_guard_01",
            )
            assert synthesis_entered.wait(timeout=5)
            revoke = pool.submit(revoke_source)
            assert mutation_started.wait(timeout=5)
            assert not revoke.done()

            unrelated_read = pool.submit(workspace_store.list_workspaces)
            assert unrelated_read.result(timeout=1)

            synthesis_release.set()
            result = build.result(timeout=5)
            revoke.result(timeout=5)

        assert result.status in {"complete", "paused"}
        assert synthesizer.started_after_revoke
        assert not any(synthesizer.started_after_revoke)
    finally:
        synthesis_release.set()
        artifacts.close()
        evidence_store.close()


def test_deletion_cannot_commit_between_source_check_copy_and_part_persistence(
    tmp_path: Path,
    workspace_store: WorkspaceStore,
):
    _workspace(workspace_store, tmp_path, {"notes.txt": b"approved evidence"})
    authorize_entered = Event()
    authorize_release = Event()
    deletion_started = Event()
    provider_pending_states: list[bool] = []
    provider = RecordingProvider()
    service, _original_gate, evidence_store, artifacts = _service(
        tmp_path,
        workspace_store,
        provider,
    )

    class BlockingGate(RecordingGate):
        def authorize(self, workspace_id: str, entry_ids):
            authorize_entered.set()
            assert authorize_release.wait(timeout=5)
            assert not evidence_store.workspace_deletion_pending(workspace_id)
            return super().authorize(workspace_id, entry_ids)

    service._transmission_gate = BlockingGate(workspace_store)

    def delete_evidence():
        deletion_started.set()
        return service.delete_workspace_evidence(WORKSPACE_ID)

    provider._on_call = lambda _count, _request: provider_pending_states.append(
        evidence_store.workspace_deletion_pending(WORKSPACE_ID)
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            build = pool.submit(
                service.build_study_map,
                WORKSPACE_ID,
                "run_delete_prepare_race",
            )
            assert authorize_entered.wait(timeout=5)
            deletion = pool.submit(delete_evidence)
            assert deletion_started.wait(timeout=5)
            assert not deletion.done()
            authorize_release.set()
            try:
                result = build.result(timeout=10)
                assert result.status in {"complete", "paused"}
            except EvidenceServiceError as error:
                assert error.code == "evidence_delete_pending"
            assert deletion.result(timeout=10) is ArtifactCleanupState.DELETED

        assert not any(provider_pending_states)
        assert evidence_store.current_revision_id(WORKSPACE_ID) is None
    finally:
        authorize_release.set()
        artifacts.close()
        evidence_store.close()
