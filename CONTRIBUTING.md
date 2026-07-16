# Contributing to ExamSage

Thank you for helping make evidence-aware revision tools more accessible.

## Principles

- Never describe a focus score as a guaranteed exam probability.
- Preserve the one-provider/one-key beginner experience.
- Keep AI inference in the selected cloud provider; deterministic local processing is welcome.
- Do not add telemetry, a maintainer-operated data relay, or secret persistence without an explicit public design review.
- Label uploaded evidence, external evidence, and generated variants separately.
- Use licensed, consented, or synthetic evaluation material. Do not commit private courses or complete copyrighted exams.
- Maintain English UI copy and multilingual content behavior.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements-dev.txt
pytest
ruff check exam_predictor tests app.py
```

Tests must run without an API key. Provider integrations should use fakes for unit tests and opt-in environment variables for live tests.

## Pull requests

Keep changes focused. Explain the student problem, security/privacy impact, cost impact, test evidence, and screenshots for UI changes. Update README and configuration examples when behavior changes.

Good first issues:

- contract tests for OpenAI and Gemini SDK response parsing;
- accessible chapter-tree navigation;
- evaluation metrics for ranking calibration and question/rubric quality;
- additional safe document converters;
- signed installers and automatic update design;
- source-quality ranking for global university material.
