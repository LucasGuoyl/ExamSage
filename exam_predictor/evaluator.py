"""Hold-out evaluator for Gaokao-style benchmarks.

Workflow:
  - Train years 2010-2023 as "past data"
  - Test year 2024 as the held-out ground truth
  - Run the full pipeline on train data
  - Compute:
      * Top-K coverage: what fraction of held-out questions align (sim > τ)
        to one of the top-K predicted knowledge points?
      * MRR (Mean Reciprocal Rank) of held-out question → best predicted KP

This gives you the ablation rig needed to tune fusion weights and validate
that the pipeline really learns something beyond random.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .embedder import Embedder
from .schema import Chunk, ExamQuestion, KnowledgePointScore


@dataclass
class EvalResult:
    top_k: int
    coverage: float                         # % of test questions covered by top-K KPs
    mrr: float                              # mean reciprocal rank
    per_question: list[dict]                # detailed per-q breakdown


def evaluate_topk_coverage(
    test_questions: list[ExamQuestion],
    ranked_kps: list[KnowledgePointScore],
    chunks: list[Chunk],
    embedder: Embedder,
    top_k: int = 10,
    coverage_threshold: float = 0.55,
) -> EvalResult:
    """For each held-out question, find which (if any) of the top-K predicted
    knowledge points it semantically aligns to (max chunk similarity ≥ τ).

    Returns:
      - coverage: fraction of test questions covered
      - mrr: mean reciprocal of rank where the matching KP appeared
    """
    if not test_questions or not ranked_kps:
        return EvalResult(top_k=top_k, coverage=0.0, mrr=0.0, per_question=[])

    # ensure embeddings
    need_q = [q for q in test_questions if q.embedding is None]
    if need_q:
        v = embedder.embed([q.text for q in need_q])
        for q, e in zip(need_q, v):
            q.embedding = e.tolist()

    chunk_map = {c.id: c for c in chunks}
    need_c = [c for c in chunks if c.embedding is None]
    if need_c:
        v = embedder.embed([c.text for c in need_c])
        for c, e in zip(need_c, v):
            c.embedding = e.tolist()

    # For efficiency: build per-KP embedding matrix of its member chunks
    kp_to_chunk_embs: dict[str, np.ndarray] = {}
    for kp in ranked_kps[:top_k]:
        member = [chunk_map[cid] for cid in kp.chunk_ids if cid in chunk_map]
        if not member:
            continue
        kp_to_chunk_embs[kp.knowledge_point_id] = np.array(
            [c.embedding for c in member], dtype=np.float32,
        )

    per_q: list[dict] = []
    covered_count = 0
    reciprocal_ranks: list[float] = []

    for q in test_questions:
        q_emb = np.array(q.embedding, dtype=np.float32)
        rank_found: int | None = None
        best_sim_per_kp: list[tuple[str, float]] = []
        for rank, kp in enumerate(ranked_kps[:top_k], 1):
            embs = kp_to_chunk_embs.get(kp.knowledge_point_id)
            if embs is None:
                continue
            sims = embs @ q_emb
            max_sim = float(sims.max())
            best_sim_per_kp.append((kp.knowledge_point_id, max_sim))
            if max_sim >= coverage_threshold and rank_found is None:
                rank_found = rank

        if rank_found is not None:
            covered_count += 1
            reciprocal_ranks.append(1.0 / rank_found)
        else:
            reciprocal_ranks.append(0.0)
        per_q.append({
            "question_id": q.id,
            "year": q.year,
            "matched_at_rank": rank_found,
            "top_kp_similarities": sorted(best_sim_per_kp, key=lambda x: -x[1])[:3],
        })

    return EvalResult(
        top_k=top_k,
        coverage=covered_count / len(test_questions),
        mrr=float(np.mean(reciprocal_ranks)),
        per_question=per_q,
    )


def random_baseline(
    test_questions: list[ExamQuestion],
    ranked_kps: list[KnowledgePointScore],
    chunks: list[Chunk],
    embedder: Embedder,
    top_k: int = 10,
    coverage_threshold: float = 0.55,
    n_runs: int = 5,
    seed: int = 0,
) -> dict:
    """Random baseline: shuffle KPs, see how often a random top-K covers.

    A pipeline that beats random by < 2x is not learning anything useful.
    """
    import random

    rng = random.Random(seed)
    coverages: list[float] = []
    for _ in range(n_runs):
        shuffled = list(ranked_kps)
        rng.shuffle(shuffled)
        r = evaluate_topk_coverage(
            test_questions, shuffled, chunks, embedder,
            top_k=top_k, coverage_threshold=coverage_threshold,
        )
        coverages.append(r.coverage)
    return {
        "mean_coverage": float(np.mean(coverages)),
        "std_coverage": float(np.std(coverages)),
        "n_runs": n_runs,
    }
