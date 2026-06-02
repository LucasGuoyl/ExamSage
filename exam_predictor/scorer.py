"""Pedagogical importance scorer — works across courses with zero past papers.

Two complementary signals that do NOT require historical exam data, making
the pipeline useful for any course even on first run:

1. structural_signals (fast, zero-cost):
   Rule-based detection of the formatting and language patterns instructors
   use to mark important content — definitions, theorems, action verbs
   (derive/prove/compute), key-points sections, speaker-note exam cues.

2. llm_pedagogy_score (LLM-based, stronger):
   Asks the LLM to rate each chunk's exam probability from first principles,
   leveraging the model's cross-course training on millions of textbooks and
   exam papers. Batches multiple chunks per call to keep API costs low.

Adaptive weighting: when n_past_papers < threshold these signals dominate;
as historical data grows, exam_frequency gradually takes over (see fusion.py).
"""

from __future__ import annotations

import json
import logging
import re
import time

from tenacity import retry, stop_after_attempt, wait_exponential

from .schema import Chunk

log = logging.getLogger(__name__)


# ── 1. Structural / formatting signals (no API call) ─────────────────────────

_SECTION_KW_RE = re.compile(
    r"\b(key\s+point|summary|summari[sz]|important|must\s+know|"
    r"definition|theorem|lemma|proof|corollar|formula|"
    r"key\s+concept|learning\s+outcome|exam\s+tip|"
    r"要点|重点|总结|定义|定理|公式|考点|必考|掌握|记住)\b",
    re.I,
)
_DEFINITION_RE = re.compile(
    r"(is\s+defined\s+as|we\s+define|definition\s*[:\s]|"
    r"称为|定义为|设\s*\w+\s*为)",
    re.I,
)
_ACTION_VERB_RE = re.compile(
    r"\b(derive|prove|show\s+that|compute|calculate|solve|apply|"
    r"formulate|analyze|analyse|compare|contrast|evaluate|"
    r"推导|证明|计算|求解|分析|比较|评估|建立模型)\b",
    re.I,
)


def extract_structural_signals(chunk: Chunk) -> float:
    """Fast rule-based importance score in [0, 1].

    Detects instructor emphasis cues that are universal across courses:
    - 'Definition / Theorem / Proof' style headings or bodies
    - Action verbs that signal testable, higher-order content
    - Explicit emphasis keywords ('key point', '重点', etc.)
    - Speaker-note references to exams
    """
    text = chunk.text
    norm = max(len(text) / 300.0, 1.0)
    score = 0.0

    # Section heading signals (strong indicator if the heading itself says 'summary')
    header = str(
        chunk.metadata.get("title") or
        chunk.metadata.get("header") or ""
    )
    if _SECTION_KW_RE.search(header):
        score += 0.35

    # Body keyword density (normalised to avoid penalising long chunks)
    score += min(len(_SECTION_KW_RE.findall(text))  / norm * 0.25, 0.25)
    score += min(len(_DEFINITION_RE.findall(text))   / norm * 0.20, 0.20)
    score += min(len(_ACTION_VERB_RE.findall(text))  / norm * 0.25, 0.25)

    # Speaker notes that explicitly mention exam/test
    if chunk.metadata.get("has_notes") and re.search(
        r"exam|quiz|test|考试|考点", text, re.I
    ):
        score += 0.20

    return float(min(score, 1.0))


# ── 1b. Low-value content detector (filters non-knowledge chunks) ─────────────

# Pure title / cover-slide patterns: "Lecture 3", "Chapter 2", "第三讲", etc.
_TITLE_ONLY_RE = re.compile(
    r"^\s*(lecture|chapter|week|section|part|unit|module|topic|slide|"
    r"第\s*[一二三四五六七八九十\d]+\s*[章讲节课部分篇])"
    r"\s*[\d\.:：、\-—]*\s*[\w一-鿿 ,，.]{0,40}$",
    re.I,
)
# Agenda / outline / review / summary / references — structural filler
_AGENDA_RE = re.compile(
    r"(outline|agenda|overview|table\s+of\s+contents|"
    r"learning\s+(outcomes?|objectives?)|course\s+objectives?|roadmap|"
    r"what\s+we['’ ]*(ll|will)\s+cover|in\s+this\s+(lecture|chapter)|"
    r"recap|review\s+of|last\s+(week|lecture|time)|previously\s+on|"
    r"references?|bibliography|further\s+reading|"
    r"acknowledge?ments?|thank\s+you|questions\s*\??|q\s*&\s*a|"
    r"目录|提纲|大纲|本(章|节|讲|课)(小结|回顾|提要)|课程回顾|"
    r"学习目标|教学目标|致谢|参考文献|思考题|课后(回顾|复习))",
    re.I,
)
# Administrative / logistics content
_ADMIN_RE = re.compile(
    r"(assignment\s+due|homework\s+due|deadline|due\s+date|office\s+hours?|"
    r"grading\s+(policy|scheme)|exam\s+(date|schedule)|"
    r"contact\s+(me|us|information)|teaching\s+assistant|instructor['’ ]*s?\s+email|"
    r"作业(要求|截止|提交)|截止日期|答疑(时间|安排)|评分(标准|政策)|"
    r"课程安排|联系方式|助教)",
    re.I,
)


def is_low_value_chunk(chunk: Chunk, min_chars: int = 60) -> tuple[bool, str]:
    """Detect chunks that are NOT real knowledge points and should be excluded.

    Catches the common false positives the user reported: title/cover slides,
    agenda/outline pages, "review of last week", references, acknowledgements,
    and administrative/logistics text.

    Returns (is_low_value, reason). Designed to be conservative — only fires on
    clear non-content, with length guards so genuine material is never dropped.
    """
    text = chunk.text.strip()
    header = str(
        chunk.metadata.get("title") or chunk.metadata.get("header") or ""
    ).strip()

    # 1. Too short to carry a testable concept (likely a title/cover slide)
    if len(text) < min_chars:
        return True, "too_short"

    # 2. Body is essentially just a section/lecture title
    first_line = text.splitlines()[0].strip()
    if _TITLE_ONLY_RE.match(text) or (
        len(text) < 120 and _TITLE_ONLY_RE.match(first_line)
    ):
        return True, "title_slide"

    # 3. Agenda / outline / review / references — in a header, or a short body
    if _AGENDA_RE.search(header):
        return True, "agenda_or_review"
    if _AGENDA_RE.search(text) and len(text) < 220:
        return True, "agenda_or_review"

    # 4. Administrative / logistics (short blocks only, to avoid false hits)
    if _ADMIN_RE.search(text) and len(text) < 280:
        return True, "administrative"

    return False, ""


# ── 2. LLM pedagogy scorer (batched) ─────────────────────────────────────────

_BATCH_SYSTEM = """\
You are an experienced university lecturer and exam setter who has taught
hundreds of different courses. Your job is to assess how likely each piece
of lecture content is to appear in a final exam.

Universal exam-worthiness criteria (apply to ANY subject):
  • Foundational concepts the whole subject builds on          → 0.8–1.0
  • Derivations, proofs, multi-step calculations               → 0.8–1.0
  • Core definitions and theorems students must master         → 0.7–0.9
  • Application techniques and problem-solving methods         → 0.6–0.8
  • Comparisons, trade-off analysis, when-to-use judgements    → 0.5–0.7
  • Worked examples that illustrate a technique                → 0.3–0.5
  • Motivating narrative, history, or background context       → 0.1–0.3
  • Administrative text, acknowledgements, course logistics    → 0.0–0.1

Output ONLY a valid JSON array — no prose, no code fences:
[{"id": <int>, "score": <float 0–1>, "type": "<definition|theorem|derivation|calculation|application|comparison|example|background>"}]

The id must match the chunk number given in the input (0-indexed within this batch)."""

_BATCH_USER = """\
Course context: {course_context}

Rate the exam probability of each chunk:

{chunks_block}"""


def _format_chunks_block(batch: list[tuple[int, Chunk]]) -> str:
    parts = []
    for local_i, chunk in batch:
        preview = chunk.text[:500].replace("\n", " ").strip()
        parts.append(f"Chunk {local_i}: {preview}")
    return "\n\n".join(parts)


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=10))
def _call_batch(
    client,
    model: str,
    course_context: str,
    batch: list[tuple[int, Chunk]],
) -> list[dict]:
    user_msg = _BATCH_USER.format(
        course_context=course_context or "university-level course",
        chunks_block=_format_chunks_block(batch),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _BATCH_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=300,
    )
    raw = (resp.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found: {raw[:150]}")
    return json.loads(raw[start: end + 1])


def score_chunks_pedagogically(
    chunks: list[Chunk],
    client,
    model: str,
    course_context: str = "",
    batch_size: int = 5,
    max_chunks: int = 80,
    inter_batch_delay: float = 0.3,
) -> dict[str, float]:
    """Return chunk_id → llm_pedagogy_score ∈ [0, 1] for every chunk.

    Sends `batch_size` chunks per LLM call to balance quality and cost.
    Chunks beyond `max_chunks` receive a neutral score (0.5) to cap spend.

    Args:
        chunks:              All chunks from Stage 1.
        client:              OpenAI-compatible client.
        model:               LLM model name.
        course_context:      Short description, e.g. "undergraduate Power Systems".
                             Helps the LLM apply appropriate subject-specific priors.
        batch_size:          Chunks per LLM call (5 is a good default).
        max_chunks:          Hard cap on scored chunks (cost control).
        inter_batch_delay:   Seconds between batches (rate-limit safety).
    """
    results: dict[str, float] = {c.id: 0.5 for c in chunks}  # neutral default

    to_score = chunks[:max_chunks]
    for batch_start in range(0, len(to_score), batch_size):
        batch_chunks = to_score[batch_start: batch_start + batch_size]
        # Use 0-based indices local to this batch
        batch = [(local_i, chunk) for local_i, chunk in enumerate(batch_chunks)]
        try:
            raw_results = _call_batch(client, model, course_context, batch)
            id_to_score = {
                r["id"]: float(r.get("score", 0.5))
                for r in raw_results
                if isinstance(r, dict) and "id" in r
            }
            for local_i, chunk in batch:
                results[chunk.id] = id_to_score.get(local_i, 0.5)
        except Exception as e:
            log.warning("LLM pedagogy batch %d failed: %s", batch_start, e)

        if inter_batch_delay and batch_start + batch_size < len(to_score):
            time.sleep(inter_batch_delay)

    return results
