from pydantic import ConfigDict, Field

from exam_predictor.workspace.models import FrozenModel


class EvidencePolicy(FrozenModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    policy_version: str = "evidence-v1"
    schema_version: str = "evidence-schema-v1"
    prompt_version: str = "source-analysis-v1"
    multimodal_concurrency: int = Field(default=2, ge=1, le=4)
    synthesis_concurrency: int = Field(default=4, ge=1, le=4)
    provider_timeout_seconds: float = Field(default=90.0, ge=10.0, le=300.0)
    tool_deadline_seconds: float = Field(default=3600.0, ge=60.0, le=14400.0)
    max_attempts_per_route: int = Field(default=3, ge=1, le=3)
    max_repair_attempts: int = Field(default=1, ge=0, le=1)
    pdf_pages_per_part: int = Field(default=24, ge=1, le=80)
    max_part_bytes: int = Field(default=10 * 1024 * 1024, ge=1024, le=48 * 1024 * 1024)
    first_map_deadline_seconds: float = Field(default=180.0, ge=30.0, le=600.0)


def source_priority(relative_path: str, format_category: str | None) -> tuple[int, str]:
    normalized = relative_path.casefold()
    classes = (
        (0, "syllabus", ("syllabus", "specification", "learning-outcome", "revision-guide")),
        (1, "assessment", ("exam", "past-paper", "mark-scheme", "problem", "tutorial", "assignment")),
        (2, "teaching", ("lecture", "slide", "course-note")),
        (3, "reference", ("textbook", "reference", "handbook")),
    )
    for priority, reason, tokens in classes:
        if any(token in normalized for token in tokens):
            return priority, reason
    return 4, "supplemental"


def representative_ordinals(total_parts: int) -> tuple[int, ...]:
    if total_parts <= 0:
        return ()
    anchors = dict.fromkeys((0, total_parts // 2, total_parts - 1))
    return tuple((*anchors, *(index for index in range(total_parts) if index not in anchors)))
