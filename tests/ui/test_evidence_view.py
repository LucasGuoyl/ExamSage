from __future__ import annotations

from datetime import UTC, datetime

import pytest
from streamlit.testing.v1 import AppTest

from exam_predictor.evidence.models import (
    CoverageItem,
    CoverageSummary,
    EvidenceCitation,
    EvidenceStatus,
    KnowledgeNode,
    SnapshotStatus,
    StudyMapSnapshot,
)
from exam_predictor.runtime.models import RunStatus
from exam_predictor.ui.evidence_view import (
    EvidencePhase,
    build_evidence_view_model,
    failure_reason,
)
from exam_predictor.ui import evidence_view


NOW = datetime(2026, 8, 1, tzinfo=UTC)


def coverage_item(
    path: str,
    *,
    processed: int = 0,
    running: int = 0,
    pending: int = 0,
    retrying: int = 0,
    failed: int = 0,
    invalidated: int = 0,
    next_action: str = "none",
    failure_codes: tuple[str, ...] = (),
    excluded: bool = False,
) -> CoverageItem:
    total = processed + running + pending + retrying + failed + invalidated
    return CoverageItem(
        topic=path,
        covered=bool(total and processed == total),
        entry_id=f"entry-{path}",
        relative_path=path,
        approved_bytes=0 if excluded else 100,
        excluded=excluded,
        planned_part_count=total,
        processed_part_count=processed,
        running_part_count=running,
        pending_part_count=pending,
        retrying_part_count=retrying,
        failed_part_count=failed,
        invalidated_part_count=invalidated,
        processed_locators=tuple(f"pages {index + 1}-{index + 2}" for index in range(processed)),
        current_locator="pages 1-2" if total else None,
        next_action=next_action,
        failure_codes=failure_codes,
    )


def coverage(*items: CoverageItem) -> CoverageSummary:
    return CoverageSummary(
        items=items,
        covered_count=sum(item.covered for item in items),
        total_count=len(items),
        excluded_count=sum(item.excluded for item in items),
        approved_bytes=sum(item.approved_bytes for item in items),
        part_total_count=sum(item.planned_part_count for item in items),
        part_processed_count=sum(item.processed_part_count for item in items),
        part_running_count=sum(item.running_part_count for item in items),
        part_pending_count=sum(item.pending_part_count for item in items),
        part_retrying_count=sum(item.retrying_part_count for item in items),
        part_failed_count=sum(item.failed_part_count for item in items),
        part_invalidated_count=sum(item.invalidated_part_count for item in items),
    )


def snapshot(summary: CoverageSummary, status: SnapshotStatus) -> StudyMapSnapshot:
    return StudyMapSnapshot(
        snapshot_id="snapshot-1",
        workspace_id="workspace-1",
        revision_id="revision-1",
        status=status,
        nodes=(
            KnowledgeNode(
                node_id="limits",
                title="Limits",
                focus_score=0.8,
                confidence=0.7,
                evidence_unit_ids=("unit-1",),
            ),
        ),
        coverage=summary,
        evidence_unit_ids=("unit-1",),
        citations=(
            EvidenceCitation(
                citation_id="citation-1",
                evidence_unit_id="unit-1",
                source_part_id="part-1",
                relative_path="week-1/exam.pdf",
                locator="pages 1-2",
            ),
        ),
        response_language="en",
        limitations=("The final tutorial has not been analyzed.",),
        created_at=NOW,
    )


@pytest.mark.parametrize(
    ("summary", "study_map", "run_status", "expected"),
    [
        (None, None, None, EvidencePhase.EMPTY),
        (coverage(coverage_item("notes.pdf", next_action="prepare")), None, None, EvidencePhase.PREPARING),
        (coverage(coverage_item("notes.pdf", pending=1, next_action="analyze")), None, RunStatus.RUNNING, EvidencePhase.ANALYZING),
        (coverage(coverage_item("notes.pdf", retrying=1, next_action="resume")), None, RunStatus.RUNNING, EvidencePhase.RETRYING),
        (coverage(coverage_item("notes.pdf", pending=1, next_action="analyze")), None, RunStatus.PAUSED, EvidencePhase.PAUSED),
        (
            coverage(
                coverage_item("exam.pdf", processed=1),
                coverage_item("notes.pdf", pending=1, next_action="analyze"),
            ),
            "initial",
            RunStatus.RUNNING,
            EvidencePhase.PARTIAL,
        ),
        (coverage(coverage_item("exam.pdf", processed=1)), "complete", None, EvidencePhase.COMPLETE),
        (coverage(coverage_item("exam.pdf", invalidated=1, next_action="reapprove")), None, None, EvidencePhase.CHANGED_APPROVAL),
        (
            coverage(coverage_item("slides.pptx", failed=1, next_action="retry", failure_codes=("converter_failed",))),
            None,
            None,
            EvidencePhase.CONVERTER_FAILURE,
        ),
        (
            coverage(coverage_item("lecture.mp4", retrying=1, next_action="resume", failure_codes=("provider_rate_limited",))),
            None,
            RunStatus.RUNNING,
            EvidencePhase.PROVIDER_CAPACITY,
        ),
    ],
)
def test_evidence_phase_covers_every_progress_and_recovery_state(
    summary,
    study_map,
    run_status,
    expected,
):
    if study_map == "initial":
        study_map = snapshot(summary, SnapshotStatus.INITIAL)
    elif study_map == "complete":
        study_map = snapshot(summary, SnapshotStatus.COMPLETE)

    model = build_evidence_view_model(summary, study_map, run_status=run_status)

    assert model.phase is expected


def test_view_model_uses_complete_server_coverage_and_exposes_source_locators():
    summary = coverage(
        coverage_item("week-1/exam.pdf", processed=2),
        coverage_item("week-2/slides.pptx", running=3, next_action="analyze"),
        coverage_item("week-3/tutorial.docx", failed=1, next_action="retry", failure_codes=("converter_failed",)),
    )

    model = build_evidence_view_model(summary, None, run_status=RunStatus.RUNNING)

    assert model.source_count == 3
    assert model.covered_source_count == 1
    assert model.part_count == 6
    assert model.processed_part_count == 2
    assert model.current_source == "week-2/slides.pptx"
    assert model.citations == ()
    assert model.failed_sources == (
        ("week-3/tutorial.docx", ("converter_failed",)),
    )


def test_view_model_keeps_excluded_manifest_sources_visible():
    summary = coverage(
        coverage_item("exam.pdf", processed=1),
        coverage_item("optional.zip", excluded=True),
    )

    model = build_evidence_view_model(summary, None)

    assert model.source_count == 1
    assert model.excluded_source_count == 1
    assert model.items[1].excluded is True


def test_initial_map_keeps_tree_evidence_dependencies_and_limitations_visible():
    summary = coverage(
        coverage_item("exam.pdf", processed=1),
        coverage_item("tutorial.pdf", pending=1, next_action="analyze"),
    )
    model = build_evidence_view_model(
        summary,
        snapshot(summary, SnapshotStatus.INITIAL),
        run_status=RunStatus.RUNNING,
    )

    assert model.phase is EvidencePhase.PARTIAL
    assert model.nodes[0].title == "Limits"
    assert model.nodes[0].evidence_unit_ids == ("unit-1",)
    assert model.limitations == ("The final tutorial has not been analyzed.",)
    assert model.citations == ("week-1/exam.pdf — pages 1-2",)


def test_complete_snapshot_with_terminal_failures_stays_complete_and_retryable():
    summary = coverage(
        coverage_item("exam.pdf", processed=1),
        coverage_item(
            "slides.pptx",
            failed=1,
            next_action="retry",
            failure_codes=("converter_failed",),
        ),
    )

    model = build_evidence_view_model(
        summary,
        snapshot(summary, SnapshotStatus.COMPLETE),
    )

    assert model.phase is EvidencePhase.COMPLETE
    assert model.can_retry
    assert model.failed_sources == (
        ("slides.pptx", ("converter_failed",)),
    )
    assert model.current_source is None
    assert model.current_locator is None


def test_old_complete_snapshot_does_not_hide_live_retry_progress():
    old_summary = coverage(
        coverage_item("exam.pdf", processed=1),
        coverage_item(
            "slides.pptx",
            failed=1,
            next_action="retry",
            failure_codes=("converter_failed",),
        ),
    )
    live_summary = coverage(
        coverage_item("exam.pdf", processed=1),
        coverage_item("slides.pptx", running=1, next_action="analyze"),
    )

    model = build_evidence_view_model(
        live_summary,
        snapshot(old_summary, SnapshotStatus.COMPLETE),
        run_status=RunStatus.RUNNING,
    )

    assert model.phase is EvidencePhase.PARTIAL
    assert model.current_source == "slides.pptx"
    assert not model.can_retry


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_safe_failure_reasons_are_actionable_in_both_languages(language: str):
    converter = failure_reason("converter_failed", language)
    capacity = failure_reason("provider_rate_limited", language)
    unknown = failure_reason("future_safe_code", language)

    assert converter and capacity and unknown
    assert "converter_failed" not in converter
    assert "provider_rate_limited" not in capacity
    assert "future_safe_code" not in unknown


class EvidencePanelClient:
    def __init__(
        self,
        summary: CoverageSummary,
        study_map: StudyMapSnapshot,
    ) -> None:
        self.summary = summary
        self.study_map = study_map
        self.coverage_calls = 0
        self.snapshot_calls = 0

    def get_evidence_coverage(self, _workspace_id: str) -> CoverageSummary:
        self.coverage_calls += 1
        return self.summary

    def get_evidence_status(self, _workspace_id: str) -> EvidenceStatus:
        return EvidenceStatus(
            workspace_id="workspace-1",
            revision_id="revision-1",
            approval_required=False,
            prior_approval_exists=True,
            approved_source_count=self.summary.total_count,
            approved_bytes=self.summary.approved_bytes,
        )

    def get_current_evidence_snapshot(
        self,
        _workspace_id: str,
    ) -> StudyMapSnapshot:
        self.snapshot_calls += 1
        return self.study_map


def test_evidence_panel_renders_complete_server_counts_tree_citations_and_limitations(
    monkeypatch,
):
    summary = coverage(
        coverage_item("week-1/exam.pdf", processed=1),
        coverage_item("week-2/tutorial.pdf", pending=1, next_action="analyze"),
    )
    client = EvidencePanelClient(
        summary,
        snapshot(summary, SnapshotStatus.INITIAL),
    )
    monkeypatch.setattr(evidence_view, "_PANEL_TEST_CLIENT", client, raising=False)
    app = AppTest.from_string(
        "from exam_predictor.ui import evidence_view\n"
        "evidence_view.render_evidence_panel("
        "evidence_view._PANEL_TEST_CLIENT, 'workspace-1')"
    ).run()

    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Sources covered"] == "1/2"
    assert metrics["Parts processed"] == "1/2"
    plain_text = " ".join(item.value for item in app.text)
    captions = " ".join(item.value for item in app.caption)
    assert "Limits" in plain_text
    assert "Current source" not in captions
    assert any(item.label == "Source citations" for item in app.expander)
    assert any(item.label == "Limitations" for item in app.expander)
    assert client.coverage_calls == 1
    assert client.snapshot_calls == 1


def test_changed_approval_status_is_reachable_without_a_coverage_response():
    model = build_evidence_view_model(
        None,
        None,
        source_changed=True,
    )

    assert model.phase is EvidencePhase.CHANGED_APPROVAL
    assert model.needs_reapproval


def test_untrusted_study_map_markdown_is_rendered_as_plain_text(monkeypatch):
    summary = coverage(
        coverage_item("week-1/exam.pdf", processed=1),
        coverage_item("week-2/tutorial.pdf", pending=1, next_action="analyze"),
    )
    unsafe_snapshot = snapshot(summary, SnapshotStatus.INITIAL)
    unsafe_snapshot = unsafe_snapshot.model_copy(
        update={
            "nodes": (
                unsafe_snapshot.nodes[0].model_copy(
                    update={"title": "[Course link](https://example.invalid)"}
                ),
            ),
            "limitations": ("[More](https://example.invalid/more)",),
        }
    )
    client = EvidencePanelClient(summary, unsafe_snapshot)
    monkeypatch.setattr(evidence_view, "_PANEL_TEST_CLIENT", client, raising=False)
    app = AppTest.from_string(
        "from exam_predictor.ui import evidence_view\n"
        "evidence_view.render_evidence_panel("
        "evidence_view._PANEL_TEST_CLIENT, 'workspace-1')"
    ).run()

    plain_text = " ".join(item.value for item in app.text)
    markdown = " ".join(item.value for item in app.markdown)
    assert "[Course link](https://example.invalid)" in plain_text
    assert "[More](https://example.invalid/more)" in plain_text
    assert "example.invalid" not in markdown
