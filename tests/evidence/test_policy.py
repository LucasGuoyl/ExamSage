import pytest
from pydantic import ValidationError

from exam_predictor.evidence.policy import EvidencePolicy, representative_ordinals, source_priority


def test_policy_defaults_and_bounds_are_deterministic():
    policy = EvidencePolicy()

    assert policy.policy_version == "evidence-v1"
    assert policy.multimodal_concurrency == 2
    assert policy.retry_backoff_seconds == 1.0
    with pytest.raises(ValidationError):
        EvidencePolicy(multimodal_concurrency=5)
    with pytest.raises(ValidationError):
        EvidencePolicy(api_key="not-allowed")
    with pytest.raises(ValidationError):
        EvidencePolicy(retry_backoff_seconds=0.0)


@pytest.mark.parametrize(
    ("relative_path", "expected"),
    [
        ("Course/Syllabus.pdf", (0, "syllabus")),
        ("past-exam.pdf", (1, "assessment")),
        ("lecture-slides.pdf", (2, "teaching")),
        ("textbook.pdf", (3, "reference")),
        ("miscellaneous.pdf", (4, "supplemental")),
    ],
)
def test_source_priority_uses_stable_path_classes(relative_path, expected):
    assert source_priority(relative_path, None) == expected


def test_representative_ordinals_puts_anchors_first_without_duplicates():
    assert representative_ordinals(0) == ()
    assert representative_ordinals(1) == (0,)
    assert representative_ordinals(2) == (0, 1)
    assert representative_ordinals(5) == (0, 2, 4, 1, 3)
