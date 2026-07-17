"""Smoke tests — verify imports and core data structures work without an API key.

Does NOT test LLM/embedding calls (those require credentials).
Run: python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_imports():
    """All modules should import cleanly."""
    import importlib

    modules = [
        "schema", "ingest", "chunker", "embedder", "vector_store", "aligner",
        "fusion", "generator", "reranker", "evaluator", "pipeline", "providers",
        "security", "cloud_analyzer", "agent", "state", "budget",
        "runtime.models",
    ]
    for module in modules:
        importlib.import_module(f"exam_predictor.{module}")


def test_schema_roundtrip():
    """Pydantic models should serialize and deserialize."""
    from exam_predictor.schema import Chunk, ExamQuestion, KnowledgePointScore, ChunkFeatures

    c = Chunk(id="c1", source="x.pdf", text="hello")
    d = c.model_dump()
    assert d["id"] == "c1"

    q = ExamQuestion(id="q1", text="What is X?", year=2023)
    assert q.year == 2023

    f = ChunkFeatures(chunk_id="c1", exam_frequency=0.7)
    kp = KnowledgePointScore(
        knowledge_point_id="kp1",
        title="Topic 1",
        chunk_ids=["c1"],
        score=0.8,
        confidence=0.5,
        features=f,
        representative_text="hello",
    )
    assert kp.score == 0.8


def test_chunker_short():
    """Short text should return one chunk unchanged."""
    from exam_predictor.chunker import chunk_units
    chunks = chunk_units([("hello world", {"source": "test.md"})], chunk_size=800)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_chunker_long():
    """Long text should be split into multiple overlapping chunks."""
    from exam_predictor.chunker import chunk_units
    text = ". ".join([f"sentence {i}" for i in range(200)]) + "."
    chunks = chunk_units([(text, {"source": "long.md"})], chunk_size=200, overlap=50)
    assert len(chunks) > 1
    # ids should be unique
    assert len({c.id for c in chunks}) == len(chunks)


def test_emphasis_score():
    """Emphasis keyword counting should work for both Chinese and English."""
    from exam_predictor.fusion import extract_emphasis_score
    score_zh = extract_emphasis_score(
        "这是一个重点内容,也是考点,需要记住",
        keywords_zh=["重点", "考点", "记住"],
        keywords_en=[],
    )
    score_en = extract_emphasis_score(
        "This is important and you should remember it for the exam",
        keywords_zh=[],
        keywords_en=["important", "remember", "exam"],
    )
    score_neutral = extract_emphasis_score(
        "This paragraph mentions nothing special at all",
        keywords_zh=["重点"],
        keywords_en=["important"],
    )
    assert score_zh > 0
    assert score_en > 0
    assert score_neutral == 0


def test_vector_store():
    """FAISS store add/search should work end-to-end."""
    import numpy as np
    from exam_predictor.vector_store import FAISSStore

    store = FAISSStore(dim=4)
    embs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.7, 0.7, 0.0, 0.0],
    ], dtype=np.float32)
    # normalize
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    items = ["A", "B", "C"]
    store.add(embs, items)
    q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    results = store.search(q, top_k=2)
    assert len(results[0]) == 2
    assert results[0][0][0] == "A"  # exact match should win


if __name__ == "__main__":
    test_imports()
    test_schema_roundtrip()
    test_chunker_short()
    test_chunker_long()
    test_emphasis_score()
    test_vector_store()
    print("All smoke tests passed.")
