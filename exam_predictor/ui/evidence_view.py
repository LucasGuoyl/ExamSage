from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import streamlit as st

from exam_predictor.evidence.models import (
    CoverageItem,
    CoverageSummary,
    EvidenceStatus,
    KnowledgeNode,
    SnapshotStatus,
    StudyMapSnapshot,
)
from exam_predictor.runtime.client import WorkerClient, WorkerClientError
from exam_predictor.runtime.models import RunStatus
from exam_predictor.ui.i18n import get_ui_language, text


class EvidencePhase(StrEnum):
    EMPTY = "empty"
    PREPARING = "preparing"
    ANALYZING = "analyzing"
    RETRYING = "retrying"
    PAUSED = "paused"
    PARTIAL = "partial"
    COMPLETE = "complete"
    CHANGED_APPROVAL = "changed_approval"
    CONVERTER_FAILURE = "converter_failure"
    PROVIDER_CAPACITY = "provider_capacity"
    FAILED = "failed"


_CONVERTER_CODES = frozenset(
    {
        "office_conversion_failed",
        "converter_failed",
        "converter_unavailable",
        "office_converter_failed",
        "office_converter_timeout",
        "office_converter_unavailable",
        "pdf_conversion_failed",
        "presentation_conversion_failed",
        "document_conversion_failed",
    }
)
_STRUCTURED_DATA_CODES = frozenset(
    {
        "structured_data_limit_exceeded",
        "structured_data_malformed",
        "structured_data_too_large",
    }
)
_PROVIDER_CAPACITY_CODES = frozenset(
    {
        "provider_connection_failed",
        "provider_rate_limited",
        "provider_timeout",
        "provider_unavailable",
    }
)
_PROVIDER_MEDIA_CODES = frozenset(
    {"provider_media_unsupported", "provider_model_unsupported"}
)
_SOURCE_CHANGED_CODES = frozenset(
    {"source_approval_revoked", "source_changed", "source_revision_changed"}
)


@dataclass(frozen=True)
class EvidenceViewModel:
    phase: EvidencePhase
    source_count: int = 0
    excluded_source_count: int = 0
    covered_source_count: int = 0
    part_count: int = 0
    processed_part_count: int = 0
    approved_bytes: int = 0
    current_source: str | None = None
    current_locator: str | None = None
    items: tuple[CoverageItem, ...] = ()
    nodes: tuple[KnowledgeNode, ...] = ()
    citations: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    failed_sources: tuple[tuple[str, tuple[str, ...]], ...] = ()
    snapshot_status: SnapshotStatus | None = None
    response_language: str | None = None
    can_retry: bool = False
    needs_reapproval: bool = False


def _failure_codes(coverage: CoverageSummary | None) -> frozenset[str]:
    if coverage is None:
        return frozenset()
    return frozenset(code for item in coverage.items for code in item.failure_codes)


def _same_coverage_state(
    left: CoverageSummary,
    right: CoverageSummary,
) -> bool:
    def without_influence(value: CoverageSummary) -> CoverageSummary:
        return value.model_copy(
            update={
                "items": tuple(
                    item.model_copy(update={"influenced_current_snapshot": False})
                    for item in value.items
                )
            }
        )

    return without_influence(left) == without_influence(right)


def _phase(
    coverage: CoverageSummary | None,
    snapshot: StudyMapSnapshot | None,
    run_status: RunStatus | None,
    source_changed: bool,
) -> EvidencePhase:
    if source_changed:
        return EvidencePhase.CHANGED_APPROVAL
    if coverage is None:
        return EvidencePhase.EMPTY
    codes = _failure_codes(coverage)
    if coverage.part_invalidated_count or any(
        item.next_action == "reapprove" for item in coverage.items
    ):
        return EvidencePhase.CHANGED_APPROVAL
    if run_status is RunStatus.PAUSED:
        return EvidencePhase.PAUSED
    if (
        snapshot is not None
        and snapshot.status is SnapshotStatus.COMPLETE
        and snapshot.coverage is not None
        and _same_coverage_state(snapshot.coverage, coverage)
    ):
        return EvidencePhase.COMPLETE
    if codes & _PROVIDER_CAPACITY_CODES:
        return EvidencePhase.PROVIDER_CAPACITY
    if codes & (_CONVERTER_CODES | _STRUCTURED_DATA_CODES):
        return EvidencePhase.CONVERTER_FAILURE
    if snapshot is not None and snapshot.status is SnapshotStatus.INITIAL:
        return EvidencePhase.PARTIAL
    if coverage.part_retrying_count:
        return EvidencePhase.RETRYING
    if coverage.part_failed_count:
        return EvidencePhase.FAILED
    if coverage.part_processed_count:
        return EvidencePhase.PARTIAL
    if coverage.part_total_count == 0 or any(
        item.next_action == "prepare" for item in coverage.items
    ):
        return EvidencePhase.PREPARING
    return EvidencePhase.ANALYZING


def build_evidence_view_model(
    coverage: CoverageSummary | None,
    snapshot: StudyMapSnapshot | None,
    *,
    run_status: RunStatus | None = None,
    source_changed: bool = False,
) -> EvidenceViewModel:
    phase = _phase(coverage, snapshot, run_status, source_changed)
    if coverage is None:
        return EvidenceViewModel(
            phase=phase,
            needs_reapproval=phase is EvidencePhase.CHANGED_APPROVAL,
        )
    actionable = next(
        (
            item
            for item in coverage.items
            if item.running_part_count or item.retrying_part_count
        ),
        None,
    )
    citations = tuple(
        f"{item.relative_path} — {item.locator}"
        for item in (() if snapshot is None else snapshot.citations)
    )
    failed_sources = tuple(
        (item.relative_path or item.topic, item.failure_codes)
        for item in coverage.items
        if item.failed_part_count or item.retrying_part_count
    )
    return EvidenceViewModel(
        phase=phase,
        source_count=coverage.included_count,
        excluded_source_count=coverage.excluded_count,
        covered_source_count=coverage.covered_count,
        part_count=coverage.part_total_count,
        processed_part_count=coverage.part_processed_count,
        approved_bytes=coverage.approved_bytes,
        current_source=(
            None if actionable is None else actionable.relative_path or actionable.topic
        ),
        current_locator=(
            None if actionable is None else actionable.current_locator
        ),
        items=coverage.items,
        nodes=() if snapshot is None else snapshot.nodes,
        citations=citations,
        limitations=() if snapshot is None else snapshot.limitations,
        failed_sources=failed_sources,
        snapshot_status=None if snapshot is None else snapshot.status,
        response_language=None if snapshot is None else snapshot.response_language,
        can_retry=bool(coverage.part_retrying_count or coverage.part_failed_count),
        needs_reapproval=phase is EvidencePhase.CHANGED_APPROVAL,
    )


def failure_reason(code: str, language: str) -> str:
    if code in _CONVERTER_CODES:
        key = "failure_office_conversion"
    elif code in _STRUCTURED_DATA_CODES:
        key = "failure_structured_data"
    elif code in _PROVIDER_CAPACITY_CODES:
        key = "failure_provider_capacity"
    elif code in _PROVIDER_MEDIA_CODES:
        key = "failure_provider_media"
    elif code in _SOURCE_CHANGED_CODES:
        key = "failure_source_changed"
    else:
        key = "failure_generic"
    return text(key, language)


def _phase_message(phase: EvidencePhase, language: str) -> str:
    return text(
        {
            EvidencePhase.EMPTY: "evidence_empty",
            EvidencePhase.PREPARING: "evidence_preparing",
            EvidencePhase.ANALYZING: "evidence_analyzing",
            EvidencePhase.RETRYING: "evidence_retrying",
            EvidencePhase.PAUSED: "evidence_paused",
            EvidencePhase.PARTIAL: "evidence_partial",
            EvidencePhase.COMPLETE: "evidence_complete",
            EvidencePhase.CHANGED_APPROVAL: "evidence_changed",
            EvidencePhase.CONVERTER_FAILURE: "evidence_converter_failure",
            EvidencePhase.PROVIDER_CAPACITY: "evidence_provider_capacity",
            EvidencePhase.FAILED: "evidence_failed",
        }[phase],
        language,
    )


def _render_phase(model: EvidenceViewModel, language: str) -> None:
    message = _phase_message(model.phase, language)
    if model.phase is EvidencePhase.COMPLETE and not model.failed_sources:
        st.success(message)
    elif model.phase in {
        EvidencePhase.CHANGED_APPROVAL,
        EvidencePhase.CONVERTER_FAILURE,
        EvidencePhase.PROVIDER_CAPACITY,
        EvidencePhase.FAILED,
    }:
        st.warning(message)
    else:
        st.info(message)


def _render_source_progress(model: EvidenceViewModel, language: str) -> None:
    if not model.items:
        return
    for item in model.items:
        label = item.relative_path or item.topic
        if item.excluded:
            st.caption(f"{label} · {text('source_excluded', language)}")
            continue
        total = item.planned_part_count
        progress = 0.0 if total == 0 else item.processed_part_count / total
        st.progress(progress, text=label)
        details = [
            text(
                "source_progress",
                language,
                processed=item.processed_part_count,
                total=total,
            )
        ]
        for key, count in (
            ("source_running", item.running_part_count),
            ("source_retrying", item.retrying_part_count),
            ("source_failed", item.failed_part_count),
            ("source_changed", item.invalidated_part_count),
        ):
            if count:
                details.append(f"{text(key, language)}: {count}")
        st.caption(" · ".join(details))


def _render_study_map(model: EvidenceViewModel, language: str) -> None:
    if not model.nodes:
        return
    st.subheader(text("study_map", language))
    known = {node.node_id: node for node in model.nodes}

    def render_node(node: KnowledgeNode, depth: int = 0) -> None:
        st.text(f"{'  ' * depth}• {node.title}")
        st.caption(
            f"{text('focus', language)} {node.focus_score:.0%} · "
            f"{text('confidence', language)} {node.confidence:.0%}"
        )
        st.caption(
            text(
                "evidence_dependencies",
                language,
                ids=", ".join(node.evidence_unit_ids),
            )
        )
        for child in model.nodes:
            if child.parent_node_id == node.node_id and child.node_id in known:
                render_node(child, depth + 1)

    for node in model.nodes:
        if node.parent_node_id is None:
            render_node(node)


def render_evidence_panel(
    client: WorkerClient,
    workspace_id: str,
    *,
    run_status: RunStatus | None = None,
    on_retry: Callable[[str | None], None] | None = None,
) -> EvidenceViewModel:
    """Render the read-only evidence projection and return its deterministic state."""
    language = get_ui_language(st.session_state)
    st.header(text("evidence_title", language))
    if st.button(text("refresh_evidence", language), key="refresh-evidence"):
        st.rerun()

    coverage: CoverageSummary | None = None
    snapshot: StudyMapSnapshot | None = None
    status: EvidenceStatus | None = None
    try:
        status = client.get_evidence_status(workspace_id)
    except WorkerClientError as error:
        if "evidence_not_ready" not in str(error):
            st.error(text("workspace_request_failed", language))
    if status is not None and not status.approval_required:
        try:
            coverage = client.get_evidence_coverage(workspace_id)
        except WorkerClientError as error:
            if "evidence_not_ready" not in str(error):
                st.error(text("workspace_request_failed", language))
        if coverage is not None:
            try:
                snapshot = client.get_current_evidence_snapshot(workspace_id)
            except WorkerClientError as error:
                if "evidence_not_ready" not in str(error):
                    st.error(text("workspace_request_failed", language))

    model = build_evidence_view_model(
        coverage,
        snapshot,
        run_status=run_status,
        source_changed=(
            status is not None
            and status.approval_required
            and status.prior_approval_exists
        ),
    )
    _render_phase(model, language)
    if coverage is None:
        return model

    banner_key = (
        "coverage_complete_banner"
        if model.phase is EvidencePhase.COMPLETE
        else "coverage_banner"
    )
    st.info(
        text(
            banner_key,
            language,
            covered=model.covered_source_count,
            total=model.source_count,
        )
    )
    source_metric, part_metric, byte_metric = st.columns(3)
    source_metric.metric(
        text("sources_covered", language),
        f"{model.covered_source_count}/{model.source_count}",
    )
    part_metric.metric(
        text("parts_processed", language),
        f"{model.processed_part_count}/{model.part_count}",
    )
    byte_metric.metric(text("approved_bytes", language), model.approved_bytes)
    if model.excluded_source_count:
        st.caption(
            text(
                "sources_excluded",
                language,
                count=model.excluded_source_count,
            )
        )

    if model.current_source is not None:
        st.caption(f"{text('current_source', language)}: {model.current_source}")
    if model.current_locator is not None:
        st.caption(f"{text('current_locator', language)}: {model.current_locator}")
    _render_source_progress(model, language)
    _render_study_map(model, language)

    if model.citations:
        with st.expander(text("citations", language)):
            for citation in model.citations:
                st.text(citation)
    if model.limitations:
        with st.expander(text("limitations", language), expanded=True):
            for limitation in model.limitations:
                st.text(f"• {limitation}")
    if model.failed_sources:
        with st.expander(text("failed_sources", language), expanded=True):
            for path, codes in model.failed_sources:
                st.text(path)
                if codes:
                    for code in codes:
                        st.text(f"• {failure_reason(code, language)}")
                else:
                    st.text(f"• {text('failure_generic', language)}")
    if model.needs_reapproval:
        st.warning(text("reapprove_hint", language))
    if model.can_retry and st.button(
        text("retry_evidence", language),
        key="retry-evidence",
        disabled=on_retry is None,
    ) and on_retry is not None:
        on_retry(model.response_language)
    return model
