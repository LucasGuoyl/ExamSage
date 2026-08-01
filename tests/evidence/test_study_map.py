from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from threading import Barrier, Event, Lock

import pytest

from exam_predictor.evidence.artifacts import ArtifactBoundaryError, EvidenceArtifactStore
from exam_predictor.evidence.models import PartState, SnapshotStatus, SourcePartPlan
from exam_predictor.evidence.providers import EvidencePartResult
from exam_predictor.evidence.store import EvidenceStore
from exam_predictor.evidence.study_map import (
    EvidenceRepairRequest,
    ApprovedCoverageEntry,
    EvidenceValidationError,
    EvidenceValidator,
    StudyMapBuilder,
    StudyMapSynthesisRequest,
)


NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)
WORKSPACE_ID = "workspace_study_map_01"
REVISION_ID = "revision_study_map_01"


def _plan(
    name: str,
    *,
    priority: int,
    entry_id: str | None = None,
) -> SourcePartPlan:
    part_id = f"part_{name}_000000000000"
    return SourcePartPlan(
        part_id=part_id,
        workspace_id=WORKSPACE_ID,
        revision_id=REVISION_ID,
        entry_id=entry_id or f"entry_{name}",
        relative_path=f"course/{name}.pdf",
        source_sha256=sha256(f"source:{name}".encode()).hexdigest(),
        part_sha256=sha256(f"part:{name}".encode()).hexdigest(),
        ordinal=0,
        locator="pages 1-2",
        media_type="application/pdf",
        size_bytes=64,
        scheduling_class="assessment" if priority == 0 else "reference",
        priority=priority,
        state=PartState.PREPARED,
        idempotency_key=f"prepare:{part_id}",
    )


def _raw_evidence(locator: str = "pages 1-2") -> str:
    return json.dumps(
        {
            "locator": locator,
            "detected_language": "en",
            "material_role": "past exam",
            "headings": ["Limits and continuity"],
            "concepts": ["limits", "continuity"],
            "definitions": [{"term": "limit", "explanation": "Approached value"}],
            "formulas": ["lim f(x)"],
            "procedures": ["Check one-sided limits"],
            "examples": ["A removable discontinuity"],
            "assessment_items": ["Evaluate a limit"],
            "visual_descriptions": [],
            "ocr_text": [],
            "limitations": [],
            "warnings": [],
            "prompt_injection_indicators": [
                "The source asks the reader to ignore earlier instructions."
            ],
        }
    )


def _result(plan: SourcePartPlan, raw_output: str | None = None) -> EvidencePartResult:
    return EvidencePartResult(
        source_part_id=plan.part_id,
        locator=plan.locator,
        provider="openai",
        model_id="test-model",
        prompt_version="source-analysis-v1",
        raw_output=raw_output if raw_output is not None else _raw_evidence(plan.locator),
    )


class _Repairer:
    def __init__(self, repaired: str) -> None:
        self.repaired = repaired
        self.calls: list[EvidenceRepairRequest] = []

    def repair_evidence(self, request: EvidenceRepairRequest) -> str:
        self.calls.append(request)
        return self.repaired


class _Synthesizer:
    def __init__(self, output_factory=None) -> None:
        self.calls: list[StudyMapSynthesisRequest] = []
        self.output_factory = output_factory or self._valid_output

    @staticmethod
    def _valid_output(request: StudyMapSynthesisRequest) -> str:
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
                        "group_id": "group-calculus",
                        "title": "Calculus",
                        "confidence": 0.76,
                        "evidence_unit_ids": list(evidence_ids),
                    }
                ],
                "nodes": [
                    {
                        "node_id": "node-limits",
                        "title": "Limits",
                        "parent_node_id": None,
                        "prerequisite_node_ids": [],
                        "course_group_id": "group-calculus",
                        "focus_score": 0.82,
                        "confidence": 0.71,
                        "evidence_unit_ids": list(evidence_ids),
                    }
                ],
                "limitations": [],
                "evidence_unit_ids": list(evidence_ids),
            }
        )

    def synthesize_study_map(self, request: StudyMapSynthesisRequest) -> str:
        self.calls.append(request)
        return self.output_factory(request)


def _seed(tmp_path: Path, plans: tuple[SourcePartPlan, ...]):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifacts = EvidenceArtifactStore(artifact_root)
    store.upsert_part_plans(plans)
    return store, artifacts


def _coverage_source(plans: tuple[SourcePartPlan, ...]):
    entries = tuple(
        ApprovedCoverageEntry(
            entry_id=plan.entry_id,
            relative_path=plan.relative_path,
            approved_bytes=plan.size_bytes,
        )
        for plan in plans
    )
    return lambda _workspace, _revision: entries


def _publish_unit(store: EvidenceStore, plan: SourcePartPlan):
    validator = EvidenceValidator()
    unit = validator(_result(plan), plan)
    store.mark_running(
        plan.part_id,
        attempt=1,
        expected_state=PartState.PREPARED,
    )
    claim_token = store.publication_token(plan.part_id)
    assert store.publish_evidence(
        plan.part_id,
        unit,
        cache_key=f"cache:{plan.part_id}",
        claim_token=claim_token,
        completed_at=NOW,
    )
    return unit


def test_validator_rejects_malformed_json_with_a_stable_error():
    plan = _plan("malformed", priority=0)

    with pytest.raises(EvidenceValidationError) as caught:
        EvidenceValidator()(_result(plan, "not-json"), plan)

    assert caught.value.code == "evidence_invalid"
    assert caught.value.__cause__ is None
    assert "not-json" not in str(caught.value)


def test_validator_repairs_once_with_only_safe_schema_issues():
    plan = _plan("repair", priority=0)
    repairer = _Repairer(_raw_evidence(plan.locator))

    unit = EvidenceValidator(repairer=repairer)(_result(plan, '{"locator": 3}'), plan)

    assert unit.source_part_id == plan.part_id
    assert unit.citations[0].locator == plan.locator
    assert len(repairer.calls) == 1
    repair = repairer.calls[0]
    assert repair.raw_output == '{"locator": 3}'
    assert repair.errors
    assert repair.deadline_seconds == 90.0
    assert all("input" not in issue.casefold() for issue in repair.errors)
    assert "raw_output" not in repr(repair)


def test_validator_rejects_unknown_locator_after_the_single_repair():
    plan = _plan("locator", priority=0)
    repairer = _Repairer(_raw_evidence("pages 99-100"))

    with pytest.raises(EvidenceValidationError):
        EvidenceValidator(repairer=repairer)(
            _result(plan, _raw_evidence("page outside approved part")),
            plan,
        )

    assert len(repairer.calls) == 1


def test_prompt_injection_is_preserved_only_as_evidence_data():
    plan = _plan("injection", priority=0)

    unit = EvidenceValidator()(_result(plan), plan)
    content = json.loads(unit.content)

    assert content["prompt_injection_indicators"] == [
        "The source asks the reader to ignore earlier instructions."
    ]
    assert unit.citations[0].relative_path == plan.relative_path


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["nodes"].append(dict(payload["nodes"][0])),
        lambda payload: payload["nodes"][0].update(
            {"prerequisite_node_ids": ["node-missing"]}
        ),
        lambda payload: payload["nodes"][0].update({"title": "90% exam probability"}),
        lambda payload: payload["course_groups"][0].update({"confidence": 1.4}),
        lambda payload: payload["limitations"].append("Estimated 90% exam probability"),
        lambda payload: payload["nodes"][0].update({"node_id": chr(0xD800)}),
    ],
    ids=(
        "duplicate-node-id",
        "unknown-prerequisite",
        "literal-probability",
        "group-confidence",
        "probability-in-limitations",
        "invalid-unicode-identifier",
    ),
)
def test_synthesis_rejects_invalid_relationships_and_probability_claims(
    tmp_path: Path,
    mutate,
):
    plan = _plan("invalid-map", priority=0)
    store, artifacts = _seed(tmp_path, (plan,))
    _publish_unit(store, plan)

    def invalid_output(request: StudyMapSynthesisRequest) -> str:
        payload = json.loads(_Synthesizer._valid_output(request))
        mutate(payload)
        return json.dumps(payload)

    builder = StudyMapBuilder(
        store,
        artifacts,
        _Synthesizer(invalid_output),
        coverage_source=_coverage_source((plan,)),
        now=lambda: NOW,
    )
    try:
        with pytest.raises(EvidenceValidationError) as caught:
            builder.publish_complete(WORKSPACE_ID, REVISION_ID)
    finally:
        artifacts.close()
        store.close()

    assert caught.value.code == "study_map_invalid"


def test_synthesis_cannot_cite_pending_or_unknown_evidence(tmp_path: Path):
    plan = _plan("pending-citation", priority=0)
    store, artifacts = _seed(tmp_path, (plan,))
    unit = _publish_unit(store, plan)

    def unknown_citation(request: StudyMapSynthesisRequest) -> str:
        payload = json.loads(_Synthesizer._valid_output(request))
        payload["nodes"][0]["evidence_unit_ids"] = ["unit_pending_or_unknown"]
        payload["evidence_unit_ids"] = ["unit_pending_or_unknown"]
        return json.dumps(payload)

    builder = StudyMapBuilder(
        store,
        artifacts,
        _Synthesizer(unknown_citation),
        coverage_source=_coverage_source((plan,)),
        now=lambda: NOW,
    )
    try:
        with pytest.raises(EvidenceValidationError, match="study map"):
            builder.publish_complete(WORKSPACE_ID, REVISION_ID)
    finally:
        artifacts.close()
        store.close()

    assert unit.evidence_unit_id != "unit_pending_or_unknown"


def test_initial_snapshot_requires_processed_high_priority_evidence(tmp_path: Path):
    high = _plan("high-pending", priority=0)
    low = _plan("low-processed", priority=3)
    store, artifacts = _seed(tmp_path, (high, low))
    _publish_unit(store, low)
    synthesizer = _Synthesizer()
    builder = StudyMapBuilder(
        store,
        artifacts,
        synthesizer,
        coverage_source=_coverage_source((high, low)),
        now=lambda: NOW,
    )
    try:
        snapshot = builder.publish_initial(WORKSPACE_ID, REVISION_ID)
    finally:
        artifacts.close()
        store.close()

    assert snapshot is None
    assert synthesizer.calls == []


def test_failed_high_priority_source_does_not_unlock_a_lower_priority_initial_map(
    tmp_path: Path,
):
    high = _plan("high-failed", priority=0)
    low = _plan("low-processed-after-failure", priority=3)
    store, artifacts = _seed(tmp_path, (high, low))
    _publish_unit(store, low)
    store.mark_running(high.part_id, attempt=1, expected_state=PartState.PREPARED)
    store.record_attempt(
        high.part_id,
        attempt=1,
        route="test-model",
        outcome=PartState.FAILED,
        started_at=NOW,
        finished_at=NOW,
        safe_error_code="provider_media_unsupported",
        claim_token=store.publication_token(high.part_id),
    )
    synthesizer = _Synthesizer()
    builder = StudyMapBuilder(
        store,
        artifacts,
        synthesizer,
        coverage_source=_coverage_source((high, low)),
        now=lambda: NOW,
    )
    try:
        snapshot = builder.publish_initial(WORKSPACE_ID, REVISION_ID)
    finally:
        artifacts.close()
        store.close()

    assert snapshot is None
    assert synthesizer.calls == []


def test_initial_and_complete_snapshots_are_cited_atomic_and_resumable(tmp_path: Path):
    high = _plan("exam", priority=0)
    pending = _plan("reference", priority=3)
    store, artifacts = _seed(tmp_path, (high, pending))
    unit = _publish_unit(store, high)
    synthesizer = _Synthesizer()
    builder = StudyMapBuilder(
        store,
        artifacts,
        synthesizer,
        coverage_source=_coverage_source((high, pending)),
        now=lambda: NOW,
    )
    try:
        initial = builder.publish_initial(WORKSPACE_ID, REVISION_ID)
        assert initial is not None
        assert initial.status is SnapshotStatus.INITIAL
        assert initial.coverage is not None
        assert (initial.coverage.covered_count, initial.coverage.total_count) == (1, 2)
        assert initial.nodes[0].focus_band == "High"
        assert initial.nodes[0].confidence_band == "Strong"
        assert initial.nodes[0].evidence_count == 1
        assert artifacts.read_json(WORKSPACE_ID, "snapshots", initial.snapshot_id) == initial
        assert store.current_snapshot(WORKSPACE_ID, REVISION_ID) == initial

        context = builder.answer_context(WORKSPACE_ID, REVISION_ID)
        assert context.snapshot == initial
        assert context.evidence_units == (unit,)

        [claim] = store.claim_parts_with_tokens(
            WORKSPACE_ID,
            REVISION_ID,
            limit=1,
            now=NOW,
        )
        assert claim.plan.part_id == pending.part_id
        store.record_attempt(
            pending.part_id,
            attempt=1,
            route="test-model",
            outcome=PartState.FAILED,
            started_at=NOW,
            finished_at=NOW,
            safe_error_code="provider_media_unsupported",
            claim_token=claim.claim_token,
        )

        complete = builder.publish_complete(WORKSPACE_ID, REVISION_ID)
        assert complete is not None
        assert complete.status is SnapshotStatus.COMPLETE
        assert complete.superseded_snapshot_id == initial.snapshot_id
        assert complete.coverage is not None
        assert (complete.coverage.covered_count, complete.coverage.total_count) == (1, 2)
        assert store.current_snapshot(WORKSPACE_ID, REVISION_ID) == complete

        assert builder.invalidate_entry(WORKSPACE_ID, REVISION_ID, high.entry_id) == (
            high.part_id,
        )
        assert store.current_snapshot(WORKSPACE_ID, REVISION_ID) is None
        with pytest.raises(ArtifactBoundaryError):
            artifacts.read_json(WORKSPACE_ID, "snapshots", complete.snapshot_id)
    finally:
        artifacts.close()
        store.close()


def test_complete_snapshot_waits_for_every_nonterminal_part(tmp_path: Path):
    processed = _plan("processed", priority=0)
    pending = _plan("pending", priority=1)
    store, artifacts = _seed(tmp_path, (processed, pending))
    _publish_unit(store, processed)
    synthesizer = _Synthesizer()
    builder = StudyMapBuilder(
        store,
        artifacts,
        synthesizer,
        coverage_source=_coverage_source((processed, pending)),
        now=lambda: NOW,
    )
    try:
        snapshot = builder.publish_complete(WORKSPACE_ID, REVISION_ID)
    finally:
        artifacts.close()
        store.close()

    assert snapshot is None
    assert synthesizer.calls == []


def test_stale_revision_cannot_publish_a_late_snapshot(tmp_path: Path):
    old = _plan("old-revision", priority=0)
    store, artifacts = _seed(tmp_path, (old,))
    _publish_unit(store, old)
    current = _plan("current-revision", priority=0).model_copy(
        update={"revision_id": "revision_study_map_02"}
    )
    store.upsert_part_plans((current,))
    synthesizer = _Synthesizer()
    builder = StudyMapBuilder(
        store,
        artifacts,
        synthesizer,
        coverage_source=_coverage_source((old,)),
        now=lambda: NOW,
    )
    try:
        snapshot = builder.publish_complete(WORKSPACE_ID, REVISION_ID)
    finally:
        artifacts.close()
        store.close()

    assert snapshot is None
    assert synthesizer.calls == []


def test_large_evidence_sets_use_bounded_batches_before_final_synthesis(tmp_path: Path):
    plans = tuple(_plan(f"batch-{index}", priority=index) for index in range(5))
    store, artifacts = _seed(tmp_path, plans)
    units = tuple(_publish_unit(store, plan) for plan in plans)
    synthesizer = _Synthesizer()
    builder = StudyMapBuilder(
        store,
        artifacts,
        synthesizer,
        coverage_source=_coverage_source(plans),
        now=lambda: NOW,
        batch_size=2,
    )
    try:
        snapshot = builder.publish_complete(WORKSPACE_ID, REVISION_ID)
    finally:
        artifacts.close()
        store.close()

    assert snapshot is not None
    batch_calls = tuple(call for call in synthesizer.calls if call.phase == "batch")
    final_calls = tuple(call for call in synthesizer.calls if call.phase == "final")
    assert len(batch_calls) == 3
    assert all(1 <= len(call.evidence) <= 2 for call in batch_calls)
    assert len(final_calls) == 1
    assert final_calls[0].evidence == ()
    assert len(final_calls[0].drafts) == 3
    assert set(snapshot.evidence_unit_ids) == {
        unit.evidence_unit_id for unit in units
    }


def test_builder_uses_only_the_latest_evidence_version_for_each_part(tmp_path: Path):
    plan = _plan("reprocessed", priority=0)
    store, artifacts = _seed(tmp_path, (plan,))
    first = _publish_unit(store, plan)
    revised_payload = json.loads(_raw_evidence(plan.locator))
    revised_payload["headings"] = ["Revalidated limits evidence"]
    second = EvidenceValidator()(
        _result(plan, json.dumps(revised_payload)),
        plan,
    )
    store.mark_running(
        plan.part_id,
        attempt=2,
        expected_state=PartState.PROCESSED,
    )
    assert store.publish_evidence(
        plan.part_id,
        second,
        cache_key="cache:reprocessed:v2",
        claim_token=store.publication_token(plan.part_id),
        completed_at=NOW,
    )
    synthesizer = _Synthesizer()
    builder = StudyMapBuilder(
        store,
        artifacts,
        synthesizer,
        coverage_source=_coverage_source((plan,)),
        now=lambda: NOW,
    )
    try:
        snapshot = builder.publish_complete(WORKSPACE_ID, REVISION_ID)
        persisted_first = store.get_evidence_unit(first.evidence_unit_id)
    finally:
        artifacts.close()
        store.close()

    assert snapshot is not None
    assert persisted_first == first
    assert snapshot.evidence_unit_ids == (second.evidence_unit_id,)
    assert tuple(
        item.evidence_unit_id for item in synthesizer.calls[0].evidence
    ) == (second.evidence_unit_id,)


def test_invalid_unicode_uses_the_single_repair_boundary():
    plan = _plan("invalid-unicode", priority=0)
    document = json.loads(_raw_evidence(plan.locator))
    document["headings"] = [chr(0xD800)]
    repairer = _Repairer(_raw_evidence(plan.locator))

    unit = EvidenceValidator(repairer=repairer)(
        _result(plan, json.dumps(document)),
        plan,
    )

    assert unit.source_part_id == plan.part_id
    assert len(repairer.calls) == 1


def test_revision_change_during_synthesis_persists_neither_row_nor_artifact(
    tmp_path: Path,
):
    old = _plan("revision-race-old", priority=0)
    store, artifacts = _seed(tmp_path, (old,))
    _publish_unit(store, old)
    current = _plan("revision-race-current", priority=0).model_copy(
        update={"revision_id": "revision_study_map_race_02"}
    )

    def flip_revision(request: StudyMapSynthesisRequest) -> str:
        store.upsert_part_plans((current,))
        return _Synthesizer._valid_output(request)

    builder = StudyMapBuilder(
        store,
        artifacts,
        _Synthesizer(flip_revision),
        coverage_source=_coverage_source((old,)),
        now=lambda: NOW,
    )
    try:
        with pytest.raises(ValueError, match="revision"):
            builder.publish_complete(WORKSPACE_ID, REVISION_ID)
        assert store.current_snapshot(WORKSPACE_ID, REVISION_ID) is None
        snapshot_files = tuple((tmp_path / "artifacts").rglob("snapshot_*.json"))
    finally:
        artifacts.close()
        store.close()

    assert snapshot_files == ()


def test_reprocessing_during_synthesis_cannot_commit_the_superseded_unit(
    tmp_path: Path,
):
    plan = _plan("unit-race", priority=0)
    store, artifacts = _seed(tmp_path, (plan,))
    first = _publish_unit(store, plan)
    revised_payload = json.loads(_raw_evidence(plan.locator))
    revised_payload["concepts"] = ["new evidence version"]
    second = EvidenceValidator()(
        _result(plan, json.dumps(revised_payload)),
        plan,
    )

    def replace_unit(request: StudyMapSynthesisRequest) -> str:
        store.mark_running(
            plan.part_id,
            attempt=2,
            expected_state=PartState.PROCESSED,
        )
        assert store.publish_evidence(
            plan.part_id,
            second,
            cache_key="cache:unit-race:v2",
            claim_token=store.publication_token(plan.part_id),
            completed_at=NOW,
        )
        return _Synthesizer._valid_output(request)

    builder = StudyMapBuilder(
        store,
        artifacts,
        _Synthesizer(replace_unit),
        coverage_source=_coverage_source((plan,)),
        now=lambda: NOW,
    )
    try:
        with pytest.raises(ValueError, match="latest processed"):
            builder.publish_complete(WORKSPACE_ID, REVISION_ID)
        assert store.current_snapshot(WORKSPACE_ID, REVISION_ID) is None
        assert store.get_evidence_unit(first.evidence_unit_id) == first
        snapshot_files = tuple((tmp_path / "artifacts").rglob("snapshot_*.json"))
    finally:
        artifacts.close()
        store.close()

    assert snapshot_files == ()


def test_answer_context_rejects_a_snapshot_while_its_dependency_is_reprocessing(
    tmp_path: Path,
):
    plan = _plan("context-race", priority=0)
    store, artifacts = _seed(tmp_path, (plan,))
    _publish_unit(store, plan)
    builder = StudyMapBuilder(
        store,
        artifacts,
        _Synthesizer(),
        coverage_source=_coverage_source((plan,)),
        now=lambda: NOW,
    )
    try:
        snapshot = builder.publish_complete(WORKSPACE_ID, REVISION_ID)
        assert snapshot is not None
        store.mark_running(
            plan.part_id,
            attempt=2,
            expected_state=PartState.PROCESSED,
        )
        context = builder.answer_context(WORKSPACE_ID, REVISION_ID)
    finally:
        artifacts.close()
        store.close()

    assert context is None


def test_coverage_keeps_approved_entries_that_have_no_prepared_part(tmp_path: Path):
    processed = _plan("coverage-processed", priority=0)
    pending = _plan("coverage-pending", priority=2)
    store, artifacts = _seed(tmp_path, (processed, pending))
    _publish_unit(store, processed)
    entries = (
        ApprovedCoverageEntry(
            entry_id=processed.entry_id,
            relative_path=processed.relative_path,
            approved_bytes=100,
        ),
        ApprovedCoverageEntry(
            entry_id=pending.entry_id,
            relative_path=pending.relative_path,
            approved_bytes=200,
        ),
        ApprovedCoverageEntry(
            entry_id="entry_not_prepared",
            relative_path="course/not-prepared.pdf",
            approved_bytes=300,
        ),
    )
    builder = StudyMapBuilder(
        store,
        artifacts,
        _Synthesizer(),
        now=lambda: NOW,
        coverage_source=lambda _workspace, _revision: entries,
    )
    try:
        snapshot = builder.publish_initial(WORKSPACE_ID, REVISION_ID)
    finally:
        artifacts.close()
        store.close()

    assert snapshot is not None
    assert snapshot.coverage is not None
    assert snapshot.coverage.total_count == 3
    assert snapshot.coverage.part_total_count == 2
    omitted = next(
        item for item in snapshot.coverage.items if item.entry_id == "entry_not_prepared"
    )
    assert omitted.planned_part_count == 0
    assert omitted.next_action == "prepare"


def test_coverage_exposes_only_the_latest_stable_failure_reason_and_locator(
    tmp_path: Path,
):
    plan = _plan("coverage-failure", priority=0)
    store, artifacts = _seed(tmp_path, (plan,))
    store.mark_running(
        plan.part_id,
        attempt=1,
        expected_state=PartState.PREPARED,
    )
    store.record_attempt(
        plan.part_id,
        attempt=1,
        route="primary",
        outcome=PartState.FAILED,
        started_at=NOW,
        finished_at=NOW,
        safe_error_code="office_conversion_failed",
        claim_token=store.publication_token(plan.part_id),
    )
    builder = StudyMapBuilder(
        store,
        artifacts,
        _Synthesizer(),
        coverage_source=_coverage_source((plan,)),
        now=lambda: NOW,
    )
    try:
        summary = builder.coverage(WORKSPACE_ID, REVISION_ID)
    finally:
        artifacts.close()
        store.close()

    item = summary.items[0]
    assert item.failure_codes == ("office_conversion_failed",)
    assert item.current_locator == "pages 1-2"


def test_complete_snapshot_refuses_an_approved_entry_without_a_terminal_part(
    tmp_path: Path,
):
    processed = _plan("complete-with-gap", priority=0)
    store, artifacts = _seed(tmp_path, (processed,))
    _publish_unit(store, processed)
    entries = (
        ApprovedCoverageEntry(
            entry_id=processed.entry_id,
            relative_path=processed.relative_path,
            approved_bytes=100,
        ),
        ApprovedCoverageEntry(
            entry_id="entry_missing_part",
            relative_path="course/missing-part.pdf",
            approved_bytes=200,
        ),
    )
    synthesizer = _Synthesizer()
    builder = StudyMapBuilder(
        store,
        artifacts,
        synthesizer,
        coverage_source=lambda _workspace, _revision: entries,
        now=lambda: NOW,
    )
    try:
        snapshot = builder.publish_complete(WORKSPACE_ID, REVISION_ID)
    finally:
        artifacts.close()
        store.close()

    assert snapshot is None
    assert synthesizer.calls == []


def test_failed_artifact_revocation_remains_durable_and_resumable(
    tmp_path: Path,
):
    high = _plan("revocation-high", priority=0)
    low = _plan("revocation-low", priority=3)
    store, artifacts = _seed(tmp_path, (high, low))
    _publish_unit(store, high)
    builder = StudyMapBuilder(
        store,
        artifacts,
        _Synthesizer(),
        coverage_source=_coverage_source((high, low)),
        now=lambda: NOW,
    )
    try:
        initial = builder.publish_initial(WORKSPACE_ID, REVISION_ID)
        assert initial is not None
        _publish_unit(store, low)
        complete = builder.publish_complete(WORKSPACE_ID, REVISION_ID)
        assert complete is not None
        original_revoke = artifacts.revoke_json
        calls: list[str] = []

        def fail_second(workspace_id: str, artifact_type: str, artifact_id: str):
            calls.append(artifact_id)
            if len(calls) == 2:
                raise ArtifactBoundaryError("artifact_publish_failed")
            return original_revoke(workspace_id, artifact_type, artifact_id)

        artifacts.revoke_json = fail_second
        with pytest.raises(ArtifactBoundaryError):
            builder.invalidate_entry(WORKSPACE_ID, REVISION_ID, high.entry_id)

        assert store.current_snapshot(WORKSPACE_ID, REVISION_ID) is None
        pending = store.pending_snapshot_revocations(WORKSPACE_ID)
        assert len(pending) == 1
        artifacts.revoke_json = original_revoke
        assert builder.recover_snapshot_revocations(WORKSPACE_ID) == pending
        assert store.pending_snapshot_revocations(WORKSPACE_ID) == ()
        for snapshot in (initial, complete):
            with pytest.raises(ArtifactBoundaryError):
                artifacts.read_json(WORKSPACE_ID, "snapshots", snapshot.snapshot_id)
    finally:
        artifacts.close()
        store.close()


def test_concurrent_identical_publications_return_one_database_artifact_identity(
    tmp_path: Path,
):
    plan = _plan("concurrent-snapshot", priority=0)
    store, artifacts = _seed(tmp_path, (plan,))
    _publish_unit(store, plan)
    barrier = Barrier(2)

    class ConcurrentSynthesizer(_Synthesizer):
        def synthesize_study_map(self, request: StudyMapSynthesisRequest) -> str:
            barrier.wait(timeout=2)
            return super().synthesize_study_map(request)

    time_lock = Lock()
    time_counter = 0

    def advancing_now() -> datetime:
        nonlocal time_counter
        with time_lock:
            value = NOW + timedelta(seconds=time_counter)
            time_counter += 1
            return value

    builder = StudyMapBuilder(
        store,
        artifacts,
        ConcurrentSynthesizer(),
        coverage_source=_coverage_source((plan,)),
        now=advancing_now,
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            snapshots = tuple(
                executor.map(
                    lambda _index: builder.publish_complete(
                        WORKSPACE_ID,
                        REVISION_ID,
                    ),
                    range(2),
                )
            )
        persisted = store.current_snapshot(WORKSPACE_ID, REVISION_ID)
        assert persisted is not None
        artifact = artifacts.read_json(
            WORKSPACE_ID,
            "snapshots",
            persisted.snapshot_id,
        )
    finally:
        artifacts.close()
        store.close()

    assert snapshots == (persisted, persisted)
    assert artifact == persisted


def test_in_flight_publisher_cannot_revive_a_revoked_snapshot(tmp_path: Path):
    plan = _plan("revoked-in-flight", priority=0)
    store, artifacts = _seed(tmp_path, (plan,))
    invalidator_artifacts = EvidenceArtifactStore(tmp_path / "artifacts")
    _publish_unit(store, plan)
    publisher = StudyMapBuilder(
        store,
        artifacts,
        _Synthesizer(),
        coverage_source=_coverage_source((plan,)),
        now=lambda: NOW,
    )
    invalidator = StudyMapBuilder(
        store,
        invalidator_artifacts,
        _Synthesizer(),
        coverage_source=_coverage_source((plan,)),
        now=lambda: NOW,
    )
    entered = Event()
    release = Event()
    original_publish = artifacts.publish_json

    def delayed_publish(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return original_publish(*args, **kwargs)

    artifacts.publish_json = delayed_publish
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                publisher.publish_complete,
                WORKSPACE_ID,
                REVISION_ID,
            )
            assert entered.wait(timeout=2)
            invalidator.invalidate_entry(WORKSPACE_ID, REVISION_ID, plan.entry_id)
            revoked = store.snapshot_revocations(WORKSPACE_ID)
            assert len(revoked) == 1
            release.set()
            with pytest.raises(ArtifactBoundaryError) as caught:
                future.result(timeout=2)
            assert caught.value.code == "artifact_revoked"

        assert store.current_snapshot(WORKSPACE_ID, REVISION_ID) is None
        assert store.pending_snapshot_revocations(WORKSPACE_ID) == ()
        assert store.snapshot_is_revoked(WORKSPACE_ID, revoked[0])
        with pytest.raises(ArtifactBoundaryError):
            artifacts.read_json(WORKSPACE_ID, "snapshots", revoked[0])
    finally:
        release.set()
        invalidator_artifacts.close()
        artifacts.close()
        store.close()


def test_unicode_study_map_uses_the_artifact_store_canonical_encoding(tmp_path: Path):
    plan = _plan("unicode-snapshot", priority=0)
    store, artifacts = _seed(tmp_path, (plan,))
    unit = _publish_unit(store, plan)

    def chinese_output(_request: StudyMapSynthesisRequest) -> str:
        return json.dumps(
            {
                "course_groups": [
                    {
                        "group_id": "group-calculus",
                        "title": "微积分",
                        "confidence": 0.76,
                        "evidence_unit_ids": [unit.evidence_unit_id],
                    }
                ],
                "nodes": [
                    {
                        "node_id": "node-limits",
                        "title": "极限",
                        "parent_node_id": None,
                        "prerequisite_node_ids": [],
                        "course_group_id": "group-calculus",
                        "focus_score": 0.82,
                        "confidence": 0.71,
                        "evidence_unit_ids": [unit.evidence_unit_id],
                    }
                ],
                "limitations": ["仅依据已批准的课程资料。"],
                "evidence_unit_ids": [unit.evidence_unit_id],
            },
            ensure_ascii=False,
        )

    builder = StudyMapBuilder(
        store,
        artifacts,
        _Synthesizer(chinese_output),
        coverage_source=_coverage_source((plan,)),
        now=lambda: NOW,
    )
    try:
        snapshot = builder.publish_complete(WORKSPACE_ID, REVISION_ID)
        assert snapshot is not None
        assert snapshot.nodes[0].title == "极限"
        assert artifacts.read_json(
            WORKSPACE_ID,
            "snapshots",
            snapshot.snapshot_id,
        ) == snapshot
    finally:
        artifacts.close()
        store.close()
