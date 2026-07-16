"""Pre-run cloud cost estimates for informed user confirmation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from .schema import CostBreakdown, CostEstimate


PRICING_UPDATED_AT = "2026-07-15"


def estimate_run_cost(
    provider: str,
    files: Iterable[str | Path],
    *,
    knowledge_points: int = 10,
    practice_questions_per_point: int = 12,
    web_queries: int = 6,
) -> CostEstimate:
    """Return a deliberately broad estimate, not a provider invoice quote.

    File bytes are only a proxy for parsed tokens/images. The estimate widens
    for PDFs and images because page count and visual detail strongly affect use.
    """

    paths = [Path(path) for path in files]
    total_mb = sum(path.stat().st_size for path in paths if path.exists()) / (1024 * 1024)
    visual_count = sum(
        1 for path in paths
        if path.suffix.lower() in {".pdf", ".ppt", ".pptx", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
    )
    provider_key = provider.lower()
    if provider_key == "openai":
        input_rate, generation_output_rate, search_rate = 2.50, 30.00, 0.014
    elif provider_key in {"gemini", "google"}:
        input_rate, generation_output_rate, search_rate = 1.50, 10.00, 0.014
    else:
        input_rate, generation_output_rate, search_rate = 5.00, 30.00, 0.02

    # Course parsing: 1 MB of mixed educational documents is often far less
    # than one million text tokens, but scanned pages create extra image input.
    input_m_tokens_low = max(0.01, total_mb * 0.025)
    input_m_tokens_high = max(0.03, total_mb * (0.10 + 0.02 * min(visual_count, 5)))
    parse_low = input_m_tokens_low * input_rate
    parse_high = input_m_tokens_high * input_rate + visual_count * 0.03

    # Scoring, summaries, chapter tree and study guide.
    analysis_calls = max(5, math.ceil(knowledge_points / 3) + 5)
    analysis_low = analysis_calls * 0.015
    analysis_high = analysis_calls * 0.12

    # Detailed answers and rubrics dominate generation output.
    question_total = max(1, knowledge_points * practice_questions_per_point)
    generation_low = question_total * (450 / 1_000_000 * generation_output_rate)
    generation_high = question_total * (1200 / 1_000_000 * generation_output_rate + 0.006)

    search_low = max(0, web_queries) * search_rate
    search_high = max(0, web_queries) * (search_rate + 0.06)

    breakdown = [
        CostBreakdown(
            label="Cloud file and image understanding",
            estimated_min=parse_low,
            estimated_max=parse_high,
            assumptions=[f"{total_mb:.1f} MB across {len(paths)} files", f"{visual_count} visual-heavy files"],
        ),
        CostBreakdown(
            label="Knowledge analysis and report",
            estimated_min=analysis_low,
            estimated_max=analysis_high,
            assumptions=[f"About {analysis_calls} model requests", f"Up to {knowledge_points} priority topics"],
        ),
        CostBreakdown(
            label="Practice questions, worked answers and rubrics",
            estimated_min=generation_low,
            estimated_max=generation_high,
            assumptions=[f"About {question_total} questions"],
        ),
        CostBreakdown(
            label="Grounded web research",
            estimated_min=search_low,
            estimated_max=search_high,
            assumptions=[f"Up to {web_queries} search requests"],
        ),
    ]
    low = sum(item.estimated_min for item in breakdown)
    high = sum(item.estimated_max for item in breakdown)
    return CostEstimate(
        provider=provider_key,
        estimated_min=round(low, 2),
        estimated_max=round(max(low, high), 2),
        breakdown=breakdown,
        assumptions=[
            "This is a conservative range, not a bill or guarantee.",
            "Scanned pages, dense diagrams, long answers and retries can increase usage.",
            "The run stops before a request that would cross the user-approved ceiling.",
            "Provider pricing and free-tier treatment can change; verify in the provider console.",
        ],
        pricing_updated_at=PRICING_UPDATED_AT,
    )
