"""Pydantic data models for the entire pipeline.

Three core entities:
  - Chunk: a unit of source material (slide section, textbook paragraph)
  - ExamQuestion: a past exam question (with optional metadata)
  - KnowledgePointScore: a scored prediction for one knowledge point

Plus auxiliary models for features, generated questions, and final reports.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class Chunk(BaseModel):
    """A unit of source material — a slide section, textbook paragraph, etc."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    source: str                              # e.g. "lecture_3.pdf"
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # populated downstream:
    embedding: Optional[list[float]] = None


class ExamQuestion(BaseModel):
    """A past exam question (or a question from a mock paper)."""

    id: str
    text: str
    year: Optional[int] = None
    paper: Optional[str] = None              # e.g. "2023_final" or "2022_mock_jiangsu"
    answer: Optional[str] = None
    # auto-extracted attributes:
    topic_labels: list[str] = Field(default_factory=list)
    question_type: Optional[str] = None      # MCQ / SAQ / proof / computation / essay
    difficulty: Optional[float] = None       # 0-1 scale
    bloom_level: Optional[str] = None        # remember / understand / apply / analyze / ...
    # populated downstream:
    embedding: Optional[list[float]] = None
    aligned_chunk_ids: list[str] = Field(default_factory=list)


class ChunkFeatures(BaseModel):
    """Multi-signal features for a chunk, used by the fusion layer (D)."""

    chunk_id: str
    exam_frequency: float = 0.0              # from aligner (C-stage heat map)
    explicit_emphasis: float = 0.0           # keyword density of "important" etc.
    chunk_size: float = 0.0                  # normalized text length
    tutorial_overlap: float = 0.0            # similarity to tutorial/problem-set items
    syllabus_emphasis: float = 0.0           # weight from syllabus verbs
    # ── new cross-course signals (work with zero past papers) ──
    llm_pedagogy_score: float = 0.0          # LLM-judged exam-worthiness from first principles
    structural_signals: float = 0.0          # rule-based: definitions, theorems, action verbs
    # raw evidence (for explanation in output):
    evidence: dict[str, Any] = Field(default_factory=dict)


class KnowledgePointScore(BaseModel):
    """The scored output for one knowledge point (one or more chunks)."""

    knowledge_point_id: str
    title: str
    chunk_ids: list[str]
    score: float                             # final fused probability (0-1)
    confidence: float                        # how confident is the score (data-dep.)
    features: ChunkFeatures
    representative_text: str                 # short snippet for display
    # populated by summarise_knowledge_points() before generation:
    description: Optional[str] = None        # LLM-generated 2-3 sentence description
    exam_directions: list[str] = Field(default_factory=list)  # likely exam angles


class GeneratedQuestion(BaseModel):
    """A generated candidate question, with rerank metadata."""

    knowledge_point_id: str
    text: str
    answer_sketch: Optional[str] = None
    question_type: Optional[str] = None
    estimated_difficulty: Optional[float] = None
    suggested_marks: Optional[int] = None
    marking_scheme: list["MarkingCriterion"] = Field(default_factory=list)
    source_kind: str = "generated"          # uploaded | external | generated_variant
    source_reference: Optional[str] = None
    # rerank scores:
    style_match_score: Optional[float] = None
    novelty_score: Optional[float] = None
    overall_score: Optional[float] = None


class MarkingCriterion(BaseModel):
    """One transparent step in a suggested marking rubric."""

    criterion: str
    marks: int = Field(ge=0)
    explanation: Optional[str] = None


class SourceCitation(BaseModel):
    """A public source consulted by the research tool."""

    title: str
    url: str
    domain: Optional[str] = None
    snippet: Optional[str] = None
    source_type: str = "web"
    trust_level: str = "supporting"
    accessed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class WebEvidence(BaseModel):
    """External evidence that clarifies a topic without masquerading as course evidence."""

    knowledge_point_id: Optional[str] = None
    query: str
    summary: str
    citations: list[SourceCitation] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class KnowledgeTreeNode(BaseModel):
    """A recursive, chapter-first view of a course."""

    id: str
    title: str
    summary: Optional[str] = None
    knowledge_point_ids: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    children: list["KnowledgeTreeNode"] = Field(default_factory=list)


class CostBreakdown(BaseModel):
    label: str
    estimated_min: float = Field(ge=0)
    estimated_max: float = Field(ge=0)
    unit: str = "USD"
    assumptions: list[str] = Field(default_factory=list)


class CostEstimate(BaseModel):
    """A conservative pre-run estimate; never presented as an exact invoice."""

    provider: str
    estimated_min: float = Field(ge=0)
    estimated_max: float = Field(ge=0)
    currency: str = "USD"
    breakdown: list[CostBreakdown] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    pricing_updated_at: Optional[str] = None


class PredictionReport(BaseModel):
    """The final report returned by the pipeline."""

    course_name: str
    n_chunks: int
    n_past_questions: int
    predictions: list[KnowledgePointScore]
    generated_questions: list[GeneratedQuestion]
    overall_confidence: float                # narrows with more data
    warnings: list[str] = Field(default_factory=list)
    knowledge_tree: list[KnowledgeTreeNode] = Field(default_factory=list)
    web_evidence: list[WebEvidence] = Field(default_factory=list)
    study_guide: Optional[str] = None
    provider: Optional[str] = None
    source_language: Optional[str] = None
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
