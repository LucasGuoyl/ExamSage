from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from exam_predictor.budget import estimate_run_cost
from exam_predictor.cloud_analyzer import normalize_course_uploads
from exam_predictor.exporter import report_to_pdf_bytes
from exam_predictor.generator import generate_for_knowledge_point
from exam_predictor.pipeline import ExamPredictor
from exam_predictor.providers import BaseProvider, BudgetExceeded, ModelRouting
from exam_predictor.schema import (
    Chunk,
    ChunkFeatures,
    GeneratedQuestion,
    KnowledgePointScore,
    MarkingCriterion,
    PredictionReport,
)
from exam_predictor.security import UploadSecurityError, safe_extract_zip, validate_public_https_url
from exam_predictor.state import CourseStore


class FakeProvider(BaseProvider):
    name = "fake"

    def __init__(self, approved_max_usd: float | None = None):
        super().__init__(
            "fake-key",
            ModelRouting("fast", "balanced", "reasoning", "embedding"),
            approved_max_usd=approved_max_usd,
        )

    def create_chat_completion(self, **kwargs):  # pragma: no cover - not needed here
        raise NotImplementedError

    def embed(self, texts):
        return [[1.0] + [0.0] * 1535 for _ in texts]

    def analyze_file(self, path, prompt):
        return json.dumps({
            "document_title": "Calculus Notes",
            "detected_language": "en",
            "course_name": "Calculus I",
            "material_kind": "lecture",
            "sections": [{
                "locator": "section 1",
                "title": "Limits",
                "text": "A limit describes the value approached by a function.",
                "visual_description": "",
            }],
            "exam_questions": [],
            "syllabus_points": ["Compute one-sided limits"],
            "warnings": [],
        })

    def web_search(self, query, prompt=None):  # pragma: no cover - not needed here
        raise NotImplementedError


def test_budget_estimate_has_breakdown(tmp_path: Path):
    document = tmp_path / "notes.pdf"
    document.write_bytes(b"x" * 10_000)
    estimate = estimate_run_cost("gemini", [document], web_queries=3)
    assert estimate.estimated_max >= estimate.estimated_min >= 0
    assert len(estimate.breakdown) == 4
    assert estimate.pricing_updated_at


def test_budget_guard_stops_before_request():
    provider = FakeProvider(approved_max_usd=0.01)
    with pytest.raises(BudgetExceeded):
        provider._guard("expensive call", reserved_usd=0.02)
    assert provider.ledger.events == []


def test_zip_traversal_is_blocked(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "no")
    with pytest.raises(UploadSecurityError):
        safe_extract_zip(archive, tmp_path / "out")


def test_private_urls_are_blocked():
    assert validate_public_https_url("https://example.edu/course") == "https://example.edu/course"
    with pytest.raises(UploadSecurityError):
        validate_public_https_url("http://example.edu")
    with pytest.raises(UploadSecurityError):
        validate_public_https_url("https://127.0.0.1/private")


def test_cloud_normalizer_uses_provider_and_writes_standard_course(tmp_path: Path):
    source = tmp_path / "notes.md"
    source.write_text("# Limits\nA function approaches a value.", encoding="utf-8")
    output = tmp_path / "normalized"
    result = normalize_course_uploads(
        FakeProvider(),
        [source],
        output,
        user_request="Prepare for my final.",
    )
    assert result.course_name == "Calculus I"
    assert list((output / "slides").glob("*.md"))
    assert (output / "syllabus.md").exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["user_request"] == "Prepare for my final."


def test_practice_allocation_is_between_six_and_twenty_four():
    features = ChunkFeatures(
        chunk_id="kp",
        evidence={"total_questions_matched": 5},
    )
    point = KnowledgePointScore(
        knowledge_point_id="kp",
        title="Limits",
        chunk_ids=["c1"],
        score=0.9,
        confidence=0.5,
        features=features,
        representative_text="Limits",
    )
    assert ExamPredictor._practice_target(point) == 24
    point.score = 0.1
    point.features.evidence["total_questions_matched"] = 0
    assert ExamPredictor._practice_target(point) == 6


def test_course_store_never_requires_or_returns_api_key(tmp_path: Path):
    store = CourseStore(tmp_path)
    course_id = store.save_course(
        "Calculus I", "openai", tmp_path / "course", {"course_name": "Calculus I"}
    )
    store.add_message(course_id, "user", "Explain limits")
    saved = store.get_course(course_id)
    assert saved and "api_key" not in json.dumps(saved)
    assert store.messages(course_id)[0]["content"] == "Explain limits"


def test_generated_total_marks_always_matches_rubric():
    payload = {
        "questions": [{
            "text": "Compute the limit and justify each step.",
            "answer_sketch": "Step 1: factor. Step 2: cancel. Step 3: substitute.",
            "question_type": "computation",
            "estimated_difficulty": 0.5,
            "suggested_marks": 99,
            "marking_scheme": [
                {"criterion": "Factor correctly", "marks": 2, "explanation": "Valid factorization"},
                {"criterion": "Finish the limit", "marks": 3, "explanation": "Correct substitution"},
            ],
        }]
    }

    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload))
            )])

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    questions = generate_for_knowledge_point(
        client,
        "fake-model",
        "Limits",
        [Chunk(id="c1", source="notes", text="Limit laws", metadata={"kp_id": "kp1"})],
        [],
        n_candidates=1,
    )
    assert len(questions) == 1
    assert questions[0].suggested_marks == 5
    assert sum(item.marks for item in questions[0].marking_scheme) == 5


def test_markdown_and_pdf_exports_include_scored_answer():
    point = KnowledgePointScore(
        knowledge_point_id="kp1",
        title="Limits",
        chunk_ids=["c1"],
        score=0.8,
        confidence=0.6,
        features=ChunkFeatures(chunk_id="kp1"),
        representative_text="Limits",
    )
    question = GeneratedQuestion(
        knowledge_point_id="kp1",
        text="Compute the limit.",
        answer_sketch="Factor, cancel, then substitute.",
        suggested_marks=5,
        marking_scheme=[MarkingCriterion(criterion="Correct method", marks=5)],
    )
    report = PredictionReport(
        course_name="Calculus I",
        n_chunks=1,
        n_past_questions=1,
        predictions=[point],
        generated_questions=[question],
        overall_confidence=0.5,
    )
    markdown = ExamPredictor._format_markdown_report(report)
    assert "Suggested marks:** 5" in markdown
    assert "Correct method" in markdown
    assert report_to_pdf_bytes(report).startswith(b"%PDF")
