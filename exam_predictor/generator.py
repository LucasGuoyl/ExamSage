"""Stage E: Question generation.

For the top-K knowledge points (by D-stage fused score), generate
candidate exam questions via few-shot prompting:

  System: explain the task
  User:   here are <few_shot_examples> past papers from this course/exam
          here is one specific knowledge point: <chunks of relevant material>
          please generate <N> candidate questions in the same style.

Output is parsed back into GeneratedQuestion objects.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Iterable

from tenacity import retry, stop_after_attempt, wait_exponential

from .schema import Chunk, ExamQuestion, GeneratedQuestion


log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert exam-question designer. Your task is to
generate plausible practice questions for a university (or high-school) exam,
mimicking the style, format, and difficulty of provided past papers.

You will be given:
  1. A small set of REAL past exam questions from this course (your style anchor).
  2. A specific KNOWLEDGE POINT — the topic to test.
  3. Supporting MATERIAL from the lecture notes covering this knowledge point.

You must:
  - Match the past-paper STYLE: question type, length, formality, language.
  - Test the knowledge point at an APPROPRIATE difficulty (consistent with examples).
  - Produce DIVERSE candidates — not minor rewordings of each other.
  - Output ONLY valid JSON in the specified format. No commentary, no markdown fences.

Output schema:
{
  "questions": [
    {
      "text": "<the question>",
      "answer_sketch": "<brief outline of expected answer, 1-3 sentences>",
      "question_type": "MCQ | SAQ | computation | proof | essay | other",
      "estimated_difficulty": <float 0-1, where 0.5 ≈ typical, 0.8 ≈ challenging>
    },
    ...
  ]
}"""


USER_TEMPLATE = """## PAST EXAM QUESTIONS (style anchor)

{few_shot_block}

## KNOWLEDGE POINT TO TEST

Title: {kp_title}

## SUPPORTING MATERIAL

{material_block}

## TASK

Generate {n} diverse, high-quality candidate exam questions that test this
knowledge point in the style of the past papers above. Output JSON only."""


def _format_few_shot(questions: list[ExamQuestion]) -> str:
    parts = []
    for i, q in enumerate(questions, 1):
        head = f"### Example {i}"
        if q.year:
            head += f" ({q.year})"
        parts.append(f"{head}\n{q.text.strip()}")
    return "\n\n".join(parts)


def _format_material(chunks: list[Chunk], max_chars: int = 3000) -> str:
    blocks = []
    used = 0
    for c in chunks:
        block = c.text.strip()
        if used + len(block) > max_chars:
            block = block[: max(max_chars - used, 0)]
        if not block:
            break
        blocks.append(f"[from {c.source}]\n{block}")
        used += len(block)
        if used >= max_chars:
            break
    return "\n\n---\n\n".join(blocks)


def _extract_json(s: str) -> dict:
    """Robustly pull a JSON object out of an LLM response.

    Handles cases where the model wraps in ```json ... ``` or adds preamble.
    """
    s = s.strip()
    # strip fences
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # find first {...} balanced block
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {s[:200]}")
    return json.loads(s[start : end + 1])


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=10))
def _call_llm(client, model: str, messages: list[dict], **kwargs) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    return resp.choices[0].message.content or ""


def generate_for_knowledge_point(
    client,
    model: str,
    kp_title: str,
    kp_chunks: list[Chunk],
    style_examples: list[ExamQuestion],
    n_candidates: int = 5,
    temperature: float = 0.5,
    max_tokens: int = 2000,
) -> list[GeneratedQuestion]:
    """Generate N candidate questions for one knowledge point."""
    kp_id = kp_chunks[0].metadata.get("kp_id", "kp") if kp_chunks else "kp"
    few_shot_block = _format_few_shot(style_examples)
    material_block = _format_material(kp_chunks)

    user = USER_TEMPLATE.format(
        few_shot_block=few_shot_block or "(no examples available — generate generic-format questions)",
        kp_title=kp_title,
        material_block=material_block or "(no material — generate based on knowledge-point title only)",
        n=n_candidates,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    try:
        raw = _call_llm(
            client, model, messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        parsed = _extract_json(raw)
        items = parsed.get("questions", [])
    except Exception as e:
        log.warning("Generation failed for %s: %s", kp_title, e)
        return []

    out: list[GeneratedQuestion] = []
    for item in items:
        if not isinstance(item, dict) or "text" not in item:
            continue
        out.append(GeneratedQuestion(
            knowledge_point_id=kp_id,
            text=str(item["text"]).strip(),
            answer_sketch=item.get("answer_sketch"),
            question_type=item.get("question_type"),
            estimated_difficulty=item.get("estimated_difficulty"),
        ))
    return out


def sample_style_examples(
    past_questions: list[ExamQuestion],
    n: int = 3,
    seed: int | None = None,
) -> list[ExamQuestion]:
    """Pick N representative past questions as few-shot anchors.

    Strategy: prefer diverse years and types if available; otherwise random sample.
    """
    if not past_questions:
        return []
    if len(past_questions) <= n:
        return past_questions

    rng = random.Random(seed)
    # bucket by year
    by_year: dict[int | None, list[ExamQuestion]] = {}
    for q in past_questions:
        by_year.setdefault(q.year, []).append(q)
    out: list[ExamQuestion] = []
    years = list(by_year.keys())
    rng.shuffle(years)
    for y in years:
        out.append(rng.choice(by_year[y]))
        if len(out) >= n:
            break
    # fallback fill
    if len(out) < n:
        remaining = [q for q in past_questions if q not in out]
        rng.shuffle(remaining)
        out.extend(remaining[: n - len(out)])
    return out[:n]


# ── Knowledge-point summarisation ─────────────────────────────────────────────

_SUMMARISE_SYSTEM = """\
You are an expert academic advisor helping students prepare for university exams.
For each knowledge point you receive, produce in the SAME LANGUAGE as the content:
  1. clean_title — a precise, self-contained title (≤ 10 words)
  2. description — 2–3 sentences: what this concept is and why it matters for exams
  3. exam_directions — exactly 3 specific angles a professor is likely to test

Output ONLY a valid JSON array — no prose, no code fences:
[
  {
    "id": <int 0-indexed within this batch>,
    "clean_title": "...",
    "description": "...",
    "exam_directions": ["...", "...", "..."]
  }
]"""

_SUMMARISE_USER = """\
Course context: {course_context}

{kp_blocks}

Summarise each knowledge point above. Match the language of the content."""


def _kp_block(idx: int, raw_title: str, chunks: list[Chunk]) -> str:
    preview = "\n".join(c.text[:350].replace("\n", " ") for c in chunks[:2])
    return f"[{idx}] Raw title: {raw_title[:80]}\nContent preview:\n{preview}"


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=10))
def _call_summarise(client, model: str, course_context: str, kp_blocks: str) -> list[dict]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SUMMARISE_SYSTEM},
            {"role": "user",   "content": _SUMMARISE_USER.format(
                course_context=course_context or "university-level course",
                kp_blocks=kp_blocks,
            )},
        ],
        temperature=0.2,
        max_tokens=1500,
    )
    raw = (resp.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array: {raw[:150]}")
    return json.loads(raw[start: end + 1])


def summarise_knowledge_points(
    client,
    model: str,
    kps: list,
    chunk_by_id: dict,
    course_context: str = "",
    batch_size: int = 4,
) -> None:
    """Enrich each KnowledgePointScore in-place with description and exam_directions.

    Also replaces the raw-text title with a clean LLM-generated one.
    Runs ceil(n / batch_size) LLM calls; failures are logged, not raised.
    """
    for batch_start in range(0, len(kps), batch_size):
        batch = kps[batch_start: batch_start + batch_size]
        blocks_text = "\n\n".join(
            _kp_block(
                local_i,
                kp.title,
                [chunk_by_id[cid] for cid in kp.chunk_ids if cid in chunk_by_id],
            )
            for local_i, kp in enumerate(batch)
        )
        try:
            results = _call_summarise(client, model, course_context, blocks_text)
            id_map = {
                r["id"]: r for r in results
                if isinstance(r, dict) and "id" in r
            }
            for local_i, kp in enumerate(batch):
                r = id_map.get(local_i, {})
                if r.get("clean_title"):
                    kp.title = str(r["clean_title"]).strip()
                if r.get("description"):
                    kp.description = str(r["description"]).strip()
                if r.get("exam_directions"):
                    kp.exam_directions = [str(d) for d in r["exam_directions"][:3]]
        except Exception as exc:
            log.warning("KP summarisation batch %d failed: %s", batch_start, exc)
