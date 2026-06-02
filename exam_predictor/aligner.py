"""Stage C: Alignment.

The core operation: for each past exam question, find the top-K most
similar chunks in the course materials. Aggregate across all questions
to get a per-chunk "exam frequency" score — the heat map.

Outputs:
  - For each chunk: how many past questions tested content close to it
  - For each question: which chunks it most likely came from

This score becomes the strongest single feature in the D-stage fusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embedder import Embedder
from .schema import Chunk, ExamQuestion
from .vector_store import FAISSStore


@dataclass
class AlignmentResult:
    """Output of alignment stage."""

    chunk_heat: dict[str, float]                 # chunk_id -> exam_frequency_score in [0, 1]
    question_alignments: dict[str, list[tuple[str, float]]]   # qid -> [(chunk_id, sim)]
    raw_counts: dict[str, int]                   # chunk_id -> n questions matched


def _ensure_embeddings(
    items: list,
    embedder: Embedder,
    text_attr: str = "text",
) -> None:
    """Embed any items that don't already have an embedding."""
    missing = [it for it in items if it.embedding is None]
    if not missing:
        return
    texts = [getattr(it, text_attr) for it in missing]
    vecs = embedder.embed(texts)
    for it, v in zip(missing, vecs):
        it.embedding = v.tolist()


def align_questions_to_chunks(
    questions: list[ExamQuestion],
    chunks: list[Chunk],
    embedder: Embedder,
    top_k: int = 5,
    similarity_threshold: float = 0.55,
) -> AlignmentResult:
    """Build the chunk heat map.

    Algorithm:
      1. Embed all chunks and questions (if not already done).
      2. Index chunks in a FAISS store.
      3. For each question, retrieve top-K chunks above threshold.
      4. Each matched chunk receives a contribution = similarity * recency_weight.
         Recency weight gives more weight to recent years (configurable).
      5. Aggregate per chunk; min-max normalize to [0, 1].
    """
    if not chunks:
        raise ValueError("No chunks to align against.")
    if not questions:
        return AlignmentResult(chunk_heat={}, question_alignments={}, raw_counts={})

    _ensure_embeddings(chunks, embedder)
    _ensure_embeddings(questions, embedder)

    # Build chunk index
    chunk_embs = np.array([c.embedding for c in chunks], dtype=np.float32)
    store: FAISSStore[Chunk] = FAISSStore(dim=chunk_embs.shape[1])
    store.add(chunk_embs, chunks)

    # Compute year weights (more recent = more predictive). Defaults to 1 if no year info.
    years = [q.year for q in questions if q.year is not None]
    if years:
        max_year, min_year = max(years), min(years)
        year_range = max(max_year - min_year, 1)

        def year_weight(y: int | None) -> float:
            if y is None:
                return 0.7  # unknown year gets neutral-low weight
            # linear decay: most recent = 1.0, oldest = 0.5
            return 0.5 + 0.5 * (y - min_year) / year_range
    else:
        def year_weight(y: int | None) -> float:
            return 1.0

    # Search
    q_embs = np.array([q.embedding for q in questions], dtype=np.float32)
    results = store.search(q_embs, top_k=top_k)

    # Aggregate
    chunk_heat_raw: dict[str, float] = {c.id: 0.0 for c in chunks}
    chunk_counts: dict[str, int] = {c.id: 0 for c in chunks}
    question_alignments: dict[str, list[tuple[str, float]]] = {}

    for question, row in zip(questions, results):
        aligned_ids: list[str] = []
        kept: list[tuple[str, float]] = []
        for chunk, sim in row:
            if sim < similarity_threshold:
                continue
            w = year_weight(question.year)
            chunk_heat_raw[chunk.id] += sim * w
            chunk_counts[chunk.id] += 1
            aligned_ids.append(chunk.id)
            kept.append((chunk.id, sim))
        question.aligned_chunk_ids = aligned_ids
        question_alignments[question.id] = kept

    # Normalize heat to [0, 1]
    if chunk_heat_raw:
        max_v = max(chunk_heat_raw.values()) or 1.0
        chunk_heat = {cid: v / max_v for cid, v in chunk_heat_raw.items()}
    else:
        chunk_heat = chunk_heat_raw

    return AlignmentResult(
        chunk_heat=chunk_heat,
        question_alignments=question_alignments,
        raw_counts=chunk_counts,
    )


def aggregate_chunks_to_knowledge_points(
    chunks: list[Chunk],
    chunk_heat: dict[str, float],
    embedder: Embedder,
    similarity_threshold: float = 0.75,
) -> list[dict]:
    """Cluster chunks into knowledge points using greedy embedding-similarity clustering.

    Returns a list of dicts:
        {"id": str, "chunk_ids": [...], "title": str, "score": float (mean heat)}

    The simplest possible clustering — works fine for course materials where
    related chunks have very high pairwise similarity. Replace with HDBSCAN
    or BERTopic for production.
    """
    _ensure_embeddings(chunks, embedder)
    if not chunks:
        return []
    embs = np.array([c.embedding for c in chunks], dtype=np.float32)
    n = len(chunks)
    assigned = [-1] * n
    cluster_centers: list[np.ndarray] = []

    # Process chunks in order of decreasing heat (anchor on important content first)
    order = sorted(range(n), key=lambda i: -chunk_heat.get(chunks[i].id, 0.0))
    for i in order:
        if assigned[i] != -1:
            continue
        # find closest existing cluster
        best_c, best_sim = -1, -1.0
        for c_idx, center in enumerate(cluster_centers):
            sim = float(np.dot(embs[i], center))
            if sim > best_sim:
                best_c, best_sim = c_idx, sim
        if best_sim >= similarity_threshold and best_c != -1:
            assigned[i] = best_c
            # update center (running average — simple)
            members = [j for j in range(n) if assigned[j] == best_c]
            cluster_centers[best_c] = embs[members].mean(axis=0)
            cluster_centers[best_c] /= np.linalg.norm(cluster_centers[best_c]) or 1.0
        else:
            cluster_centers.append(embs[i].copy())
            assigned[i] = len(cluster_centers) - 1

    # build output
    kps: list[dict] = []
    for c_idx in range(len(cluster_centers)):
        member_idxs = [i for i in range(n) if assigned[i] == c_idx]
        if not member_idxs:
            continue
        member_chunks = [chunks[i] for i in member_idxs]
        scores = [chunk_heat.get(c.id, 0.0) for c in member_chunks]
        # Title: take the first ~80 chars of the highest-heat chunk
        top = max(member_chunks, key=lambda c: chunk_heat.get(c.id, 0.0))
        title = top.text[:80].replace("\n", " ").strip()
        kps.append({
            "id": f"kp_{c_idx}",
            "chunk_ids": [c.id for c in member_chunks],
            "title": title,
            "mean_heat": float(np.mean(scores)),
            "max_heat": float(np.max(scores)),
            "size": len(member_chunks),
        })
    kps.sort(key=lambda k: -k["mean_heat"])
    return kps
