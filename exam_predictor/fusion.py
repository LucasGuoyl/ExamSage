"""Stage D: Multi-signal fusion.

For each chunk (or knowledge point), extract several features:
  - exam_frequency: from aligner (Stage C)
  - explicit_emphasis: keyword density of "important", "考点" etc.
  - chunk_size: log-normalized text length (proxy for teaching time)
  - tutorial_overlap: similarity to tutorial / problem-set items
  - syllabus_emphasis: from action verbs in learning outcomes

Combine into a per-chunk score via configurable weighted sum. The
output is the input to the generation stage (E).
"""

from __future__ import annotations

import math
import re

import numpy as np

from .aligner import AlignmentResult
from .embedder import Embedder
from .schema import Chunk, ChunkFeatures


# ---------------------------------------------------------------------------
# Individual feature extractors
# ---------------------------------------------------------------------------
def extract_emphasis_score(text: str, keywords_zh: list[str], keywords_en: list[str]) -> float:
    """Density of emphasis keywords. Normalized to roughly [0, 1]."""
    text_lower = text.lower()
    count = 0
    for kw in keywords_en:
        count += len(re.findall(rf"\b{re.escape(kw.lower())}\b", text_lower))
    for kw in keywords_zh:
        count += text.count(kw)
    # Normalize: 1 hit per 200 chars = score of 0.5, saturating
    density = count / max(len(text) / 200, 1.0)
    return float(min(density / 2.0, 1.0))


def extract_chunk_size_score(chunk: Chunk, all_chunks: list[Chunk]) -> float:
    """Normalized log-length. Longer chunks → more content was spent on this topic."""
    lengths = np.log1p([len(c.text) for c in all_chunks])
    lo, hi = lengths.min(), lengths.max()
    if hi - lo < 1e-6:
        return 0.5
    return float((math.log1p(len(chunk.text)) - lo) / (hi - lo))


def compute_tutorial_overlap(
    chunks: list[Chunk],
    tutorial_items: list[str],
    embedder: Embedder,
    top_k: int = 3,
    sim_threshold: float = 0.55,
) -> dict[str, float]:
    """For each chunk, measure max similarity to any tutorial item.

    Tutorial questions resembling past exam content historically correlate
    strongly with what's tested. This is a "cheap but high-signal" feature.
    """
    if not tutorial_items or not chunks:
        return {c.id: 0.0 for c in chunks}

    # ensure chunk embeddings present
    needs = [c for c in chunks if c.embedding is None]
    if needs:
        vecs = embedder.embed([c.text for c in needs])
        for c, v in zip(needs, vecs):
            c.embedding = v.tolist()

    tut_embs = embedder.embed(tutorial_items)
    chunk_embs = np.array([c.embedding for c in chunks], dtype=np.float32)

    # Cosine sim matrix (since both normalized)
    sim = chunk_embs @ tut_embs.T              # shape (n_chunks, n_tutorials)
    max_sims = sim.max(axis=1)                 # per-chunk best tutorial match
    # Apply threshold: below threshold = 0
    max_sims = np.where(max_sims >= sim_threshold, max_sims, 0.0)
    return {c.id: float(s) for c, s in zip(chunks, max_sims)}


_VERB_RE = re.compile(r"\b(\w+)\b")


def extract_syllabus_emphasis(
    chunks: list[Chunk],
    syllabus_text: str | None,
    verb_weights: dict[str, float],
    embedder: Embedder,
) -> dict[str, float]:
    """If a syllabus is provided, weight chunks by how strongly their topic
    is described with action verbs.

    Mechanism: split syllabus into bullets, score each bullet by verb weight,
    then for each chunk find the closest bullet (by embedding) and inherit its score.
    """
    if not syllabus_text:
        return {c.id: 0.0 for c in chunks}

    # split into bullets/lines
    bullets = [b.strip() for b in re.split(r"[\n;。.;]", syllabus_text) if b.strip()]
    if not bullets:
        return {c.id: 0.0 for c in chunks}

    # score each bullet
    bullet_scores: list[float] = []
    for b in bullets:
        words = _VERB_RE.findall(b.lower())
        score = 0.0
        for w in words:
            if w in verb_weights:
                score = max(score, verb_weights[w])
        bullet_scores.append(score)

    # embed bullets
    bullet_embs = embedder.embed(bullets)
    needs = [c for c in chunks if c.embedding is None]
    if needs:
        vecs = embedder.embed([c.text for c in needs])
        for c, v in zip(needs, vecs):
            c.embedding = v.tolist()
    chunk_embs = np.array([c.embedding for c in chunks], dtype=np.float32)

    # Each chunk gets the score of its closest bullet
    sim = chunk_embs @ bullet_embs.T
    best_bullets = sim.argmax(axis=1)
    return {c.id: float(bullet_scores[idx]) for c, idx in zip(chunks, best_bullets)}


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
# Default weights for both regimes; overridable via config.yaml ─────────────
_NORMAL_WEIGHTS: dict[str, float] = {
    "exam_frequency":    0.35,
    "explicit_emphasis": 0.10,
    "chunk_size":        0.05,
    "tutorial_overlap":  0.15,
    "syllabus_emphasis": 0.10,
    "llm_pedagogy_score": 0.15,
    "structural_signals": 0.10,
}

# When past-paper count < low_data_threshold, rely more on cross-course priors
_SPARSE_WEIGHTS: dict[str, float] = {
    "exam_frequency":    0.10,
    "explicit_emphasis": 0.08,
    "chunk_size":        0.03,
    "tutorial_overlap":  0.10,
    "syllabus_emphasis": 0.14,
    "llm_pedagogy_score": 0.45,   # LLM cross-course prior dominates
    "structural_signals": 0.10,
}


def adaptive_weights(n_past_papers: int, config: dict) -> dict[str, float]:
    """Return fusion weights adapted to the available data volume.

    With abundant past papers (≥ threshold) we trust exam_frequency most.
    With sparse data we shift weight to llm_pedagogy_score and
    structural_signals which work with zero past papers.
    A smooth linear blend avoids hard cutoffs.
    """
    fusion_cfg  = config.get("fusion", {})
    normal_w    = fusion_cfg.get("weights",        _NORMAL_WEIGHTS)
    sparse_w    = fusion_cfg.get("sparse_weights", _SPARSE_WEIGHTS)
    threshold   = fusion_cfg.get("low_data_threshold", 10)

    if n_past_papers >= threshold:
        return dict(normal_w)

    alpha = n_past_papers / threshold   # 0.0 (no data) → 1.0 (full data)
    all_keys = set(normal_w) | set(sparse_w)
    return {
        k: sparse_w.get(k, 0.0) + alpha * (normal_w.get(k, 0.0) - sparse_w.get(k, 0.0))
        for k in all_keys
    }


def fuse(
    chunks: list[Chunk],
    alignment: AlignmentResult,
    config: dict,
    n_past_papers: int = 0,
    tutorial_items: list[str] | None = None,
    syllabus_text: str | None = None,
    embedder: Embedder | None = None,
    llm_scores: dict[str, float] | None = None,
    structural_scores: dict[str, float] | None = None,
) -> dict[str, ChunkFeatures]:
    """Compute ChunkFeatures for every chunk and the final fused score.

    New parameters:
        n_past_papers:      Controls adaptive weight mixing.
        llm_scores:         chunk_id → llm_pedagogy_score from scorer.py.
        structural_scores:  chunk_id → structural_signals  from scorer.py.

    Returns: chunk_id -> ChunkFeatures (with `evidence["final_score"]`).
    """
    fusion_cfg  = config.get("fusion", {})
    weights     = adaptive_weights(n_past_papers, config)
    keywords_zh = fusion_cfg.get("emphasis_keywords", {}).get("zh", [])
    keywords_en = fusion_cfg.get("emphasis_keywords", {}).get("en", [])
    verb_weights = fusion_cfg.get("syllabus_verb_weights", {})

    llm_s  = llm_scores        or {c.id: 0.0 for c in chunks}
    str_s  = structural_scores or {c.id: 0.0 for c in chunks}

    tutorial_overlap = (
        compute_tutorial_overlap(chunks, tutorial_items, embedder)
        if tutorial_items and embedder
        else {c.id: 0.0 for c in chunks}
    )
    syllabus_emph = (
        extract_syllabus_emphasis(chunks, syllabus_text, verb_weights, embedder)
        if syllabus_text and embedder
        else {c.id: 0.0 for c in chunks}
    )

    out: dict[str, ChunkFeatures] = {}
    for chunk in chunks:
        feats = ChunkFeatures(
            chunk_id=chunk.id,
            exam_frequency=alignment.chunk_heat.get(chunk.id, 0.0),
            explicit_emphasis=extract_emphasis_score(chunk.text, keywords_zh, keywords_en),
            chunk_size=extract_chunk_size_score(chunk, chunks),
            tutorial_overlap=tutorial_overlap.get(chunk.id, 0.0),
            syllabus_emphasis=syllabus_emph.get(chunk.id, 0.0),
            llm_pedagogy_score=llm_s.get(chunk.id, 0.0),
            structural_signals=str_s.get(chunk.id, 0.0),
        )
        final = (
            weights.get("exam_frequency",    0.0) * feats.exam_frequency
            + weights.get("explicit_emphasis", 0.0) * feats.explicit_emphasis
            + weights.get("chunk_size",        0.0) * feats.chunk_size
            + weights.get("tutorial_overlap",  0.0) * feats.tutorial_overlap
            + weights.get("syllabus_emphasis", 0.0) * feats.syllabus_emphasis
            + weights.get("llm_pedagogy_score",0.0) * feats.llm_pedagogy_score
            + weights.get("structural_signals",0.0) * feats.structural_signals
        )
        feats.evidence = {
            "final_score":        float(final),
            "raw_question_count": alignment.raw_counts.get(chunk.id, 0),
            "weight_regime":      "sparse" if n_past_papers < fusion_cfg.get("low_data_threshold", 10) else "normal",
        }
        out[chunk.id] = feats
    return out


def confidence_from_data(n_past_papers: int, n_chunks: int) -> float:
    """Heuristic confidence score that widens (drops) with less data.

    With ≥30 past questions: confidence ~ 0.85
    With 10:                  confidence ~ 0.55
    With 3:                   confidence ~ 0.30
    """
    n = max(n_past_papers, 1)
    raw = 1.0 - math.exp(-n / 15.0)
    # also penalize tiny chunk pools (no signal to align against)
    chunk_penalty = min(n_chunks / 50.0, 1.0)
    return float(raw * chunk_penalty * 0.95)
