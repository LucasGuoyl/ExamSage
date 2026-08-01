from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from exam_predictor.evidence.models import (
    CoverageItem,
    CoverageSummary,
    EvidenceCitation,
    EvidenceUnit,
    KnowledgeNode,
    PartState,
    SnapshotStatus,
    SourcePartPlan,
    StudyMapSnapshot,
)


NOW = datetime(2026, 7, 27, tzinfo=UTC)
SOURCE_PART = dict(
    part_id="part-1",
    workspace_id="w" * 32,
    revision_id="r" * 32,
    entry_id="e" * 32,
    relative_path="course/syllabus.pdf",
    source_sha256="a" * 64,
    part_sha256="b" * 64,
    ordinal=0,
    locator="pages 1-20",
    media_type="application/pdf",
    size_bytes=100,
    scheduling_class="syllabus",
    priority=0,
    state=PartState.PLANNED,
    idempotency_key="i" * 32,
)


def test_source_part_rejects_absolute_paths_and_secret_fields():
    with pytest.raises(ValidationError):
        SourcePartPlan(**{**SOURCE_PART, "relative_path": "C:/secret.pdf"})

    with pytest.raises(ValidationError):
        SourcePartPlan(**SOURCE_PART, api_key="not-allowed")


@pytest.mark.parametrize(
    ("model", "factory", "field", "unsafe_value"),
    [
        (
            SourcePartPlan,
            lambda: SOURCE_PART,
            "locator",
            "https://storage.example/file?X-Amz-Signature=abc",
        ),
        (SourcePartPlan, lambda: SOURCE_PART, "locator", "C:/private/provider.log"),
        (SourcePartPlan, lambda: SOURCE_PART, "locator", "/etc/passwd"),
        (SourcePartPlan, lambda: SOURCE_PART, "locator", "/"),
        (
            EvidenceCitation,
            lambda: {
                "citation_id": "citation-1",
                "evidence_unit_id": "unit-1",
                "source_part_id": "part-1",
                "relative_path": "course/syllabus.pdf",
                "locator": "page 1",
            },
            "locator",
            "Authorization: Bearer provider-secret",
        ),
        (
            EvidenceUnit,
            lambda: {
                "evidence_unit_id": "unit-1",
                "source_part_id": "part-1",
                "content": "Ordinary course text.",
                "citations": (
                    EvidenceCitation(
                        citation_id="citation-1",
                        evidence_unit_id="unit-1",
                        source_part_id="part-1",
                        relative_path="course/syllabus.pdf",
                        locator="page 1",
                    ),
                ),
            },
            "content",
            "Provider returned sk-" + "proj-abcdefghijklmnopqrstuvwxyz1234567890",
        ),
        (
            EvidenceUnit,
            lambda: {
                "evidence_unit_id": "unit-1",
                "source_part_id": "part-1",
                "content": "Ordinary course text.",
                "citations": (
                    EvidenceCitation(
                        citation_id="citation-1",
                        evidence_unit_id="unit-1",
                        source_part_id="part-1",
                        relative_path="course/syllabus.pdf",
                        locator="page 1",
                    ),
                ),
            },
            "content",
            "Traceback (most recent call last): provider failure",
        ),
        (
            EvidenceUnit,
            lambda: {
                "evidence_unit_id": "unit-1",
                "source_part_id": "part-1",
                "content": "Ordinary course text.",
                "citations": (
                    EvidenceCitation(
                        citation_id="citation-1",
                        evidence_unit_id="unit-1",
                        source_part_id="part-1",
                        relative_path="course/syllabus.pdf",
                        locator="page 1",
                    ),
                ),
            },
            "content",
            "RuntimeError: provider failure",
        ),
        (
            EvidenceUnit,
            lambda: {
                "evidence_unit_id": "unit-1",
                "source_part_id": "part-1",
                "content": "Ordinary course text.",
                "citations": (
                    EvidenceCitation(
                        citation_id="citation-1",
                        evidence_unit_id="unit-1",
                        source_part_id="part-1",
                        relative_path="course/syllabus.pdf",
                        locator="page 1",
                    ),
                ),
            },
            "content",
            "<openai.OpenAI object at 0x1234>",
        ),
    ],
)
def test_evidence_text_fields_reject_sensitive_values(model, factory, field, unsafe_value):
    with pytest.raises(ValidationError):
        model(**{**factory(), field: unsafe_value})


def test_evidence_text_fields_allow_ordinary_course_text():
    unit = EvidenceUnit(
        evidence_unit_id="unit-1",
        source_part_id="part-1",
        content="The lecture discusses limits, continuity, and exam preparation.",
        citations=(
            EvidenceCitation(
                citation_id="citation-1",
                evidence_unit_id="unit-1",
                source_part_id="part-1",
                relative_path="course/syllabus.pdf",
                locator="page 1",
            ),
        ),
    )

    assert unit.content.startswith("The lecture")


def test_source_part_normalizes_relative_paths_and_requires_lowercase_hashes():
    part = SourcePartPlan(**{**SOURCE_PART, "relative_path": "course/./syllabus.pdf"})

    assert part.relative_path == "course/syllabus.pdf"
    with pytest.raises(ValidationError):
        SourcePartPlan(**{**SOURCE_PART, "source_sha256": "A" * 64})


def test_evidence_dependencies_are_nonempty_and_unique():
    with pytest.raises(ValidationError):
        KnowledgeNode(
            node_id="node-1",
            title="Limits",
            focus_score=0.9,
            confidence=0.5,
            evidence_unit_ids=("unit-1", "unit-1"),
        )


def test_knowledge_node_keeps_focus_score_separate_from_confidence():
    node = KnowledgeNode(
        node_id="node-1",
        title="Limits",
        focus_score=0.9,
        confidence=0.5,
        evidence_unit_ids=("unit-1",),
    )

    assert (node.focus_score, node.confidence) == (0.9, 0.5)


def test_initial_snapshot_requires_nonempty_coverage_and_evidence_dependencies():
    with pytest.raises(ValidationError):
        StudyMapSnapshot(
            snapshot_id="s" * 32,
            workspace_id="w" * 32,
            revision_id="r" * 32,
            status=SnapshotStatus.INITIAL,
            nodes=(),
            coverage=None,
            evidence_unit_ids=(),
            created_at=NOW,
        )


def test_initial_snapshot_accepts_coverage_and_unique_evidence_dependencies():
    snapshot = StudyMapSnapshot(
        snapshot_id="s" * 32,
        workspace_id="w" * 32,
        revision_id="r" * 32,
        status=SnapshotStatus.INITIAL,
        nodes=(
            KnowledgeNode(
                node_id="node-1",
                title="Limits",
                focus_score=0.9,
                confidence=0.5,
                evidence_unit_ids=("unit-1",),
            ),
        ),
        coverage=CoverageSummary(
            items=(
                CoverageItem(topic="Limits", covered=True),
                CoverageItem(topic="Continuity", covered=False),
            ),
            covered_count=1,
            total_count=2,
        ),
        evidence_unit_ids=("unit-1",),
        created_at=NOW,
    )

    assert snapshot.status is SnapshotStatus.INITIAL


def test_evidence_models_are_frozen_and_forbid_extra_fields():
    citation = EvidenceCitation(
        citation_id="citation-1",
        evidence_unit_id="unit-1",
        source_part_id="part-1",
        relative_path="course/syllabus.pdf",
        locator="page 1",
    )
    unit = EvidenceUnit(
        evidence_unit_id="unit-1",
        source_part_id="part-1",
        content="A limit is ...",
        citations=(citation,),
    )

    with pytest.raises(ValidationError):
        unit.content = "changed"
    with pytest.raises(ValidationError):
        EvidenceUnit(**unit.model_dump(), secret="not-allowed")


def test_evidence_unit_requires_at_least_one_citation():
    with pytest.raises(ValidationError):
        EvidenceUnit(
            evidence_unit_id="unit-1",
            source_part_id="part-1",
            content="A limit is ...",
            citations=(),
        )


def test_coverage_summary_has_exact_noncontradictory_partial_counts():
    summary = CoverageSummary(
        items=(
            CoverageItem(topic="Limits", covered=True),
            CoverageItem(topic="Continuity", covered=False),
            CoverageItem(topic="Derivatives", covered=False),
        ),
        covered_count=1,
        total_count=3,
    )

    assert summary.coverage_fraction == pytest.approx(1 / 3)
    assert summary.is_partial is True
    with pytest.raises(ValidationError):
        CoverageSummary(
            items=(CoverageItem(topic="Limits", covered=True),),
            covered_count=0,
            total_count=1,
        )
    with pytest.raises(ValidationError):
        CoverageSummary(
            items=(CoverageItem(topic=" ", covered=True),),
            covered_count=1,
            total_count=1,
        )
    with pytest.raises(ValidationError):
        CoverageSummary(
            items=(
                CoverageItem(topic="Limits", covered=True),
                CoverageItem(topic="Limits", covered=False),
            ),
            covered_count=1,
            total_count=2,
        )


def test_coverage_summary_and_snapshot_round_trip_without_derived_field_extras():
    coverage = CoverageSummary(
        items=(
            CoverageItem(topic="Limits", covered=True),
            CoverageItem(topic="Continuity", covered=False),
        ),
        covered_count=1,
        total_count=2,
    )
    snapshot = StudyMapSnapshot(
        snapshot_id="s" * 32,
        workspace_id="w" * 32,
        revision_id="r" * 32,
        status=SnapshotStatus.INITIAL,
        nodes=(
            KnowledgeNode(
                node_id="node-1",
                title="Limits",
                focus_score=0.9,
                confidence=0.5,
                evidence_unit_ids=("unit-1",),
            ),
        ),
        coverage=coverage,
        evidence_unit_ids=("unit-1",),
        created_at=NOW,
    )

    assert CoverageSummary.model_validate(coverage.model_dump()) == coverage
    assert StudyMapSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot
