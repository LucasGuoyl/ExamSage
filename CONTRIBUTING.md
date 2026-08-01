# Contributing to ExamSage

Thank you for helping build evidence-aware revision tools.

## Principles

- Never describe a focus score as a guaranteed exam probability.
- Preserve the one-provider, one-key beginner model.
- Keep the interface complete and equivalent in English and Simplified Chinese.
- Generated academic content follows the current user-message language unless the user explicitly requests another output language; changing interface language must not silently regenerate artifacts.
- Keep AI inference in the selected provider and deterministic safety/preparation locally.
- Do not add telemetry, a maintainer-operated relay, or plaintext secret persistence.
- Keep approval, source identity, provider transmission, evidence, citations, and deletion boundaries explicit.
- Label partial coverage, failed sources, external evidence, and generated material honestly.
- Use licensed, consented, or synthetic evaluation material. Never commit private courses, credentials, live benchmark output with sensitive fields, or complete copyrighted exams.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

On macOS/Linux, use `.venv/bin/python` instead. All deterministic tests must run without an API key. Provider boundaries use fakes in regular tests.

## Required release gate

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall -q app.py exam_predictor scripts tests
.\.venv\Scripts\python.exe scripts\check_secret_patterns.py --root .
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

Feature and bug-fix work should start with a failing focused test. Add security-boundary tests for identity replacement, link substitution, malformed provider output, timeout/retry exhaustion, Stop/Resume idempotency, cache invalidation, and cleanup races where relevant.

## Live tests and benchmarks

Live provider or platform checks must be explicitly opt-in and must never run in CI. Use only the declared CC0 synthetic reference course for `scripts/benchmark_initial_map.py`. Supply `--live`, an explicit connected provider profile, fixture path, and a new output path. Never use a private course, print credentials or source text, overwrite an existing result, or describe fake-provider timing as live evidence.

Record the exact provider, explicit model routes, OS, result, date, safe limitations, and whether the check was automated or live in `docs/manual-tests/`. Mark unavailable checks `outstanding`.

## Pull requests

Keep changes focused. Explain the student problem, security/privacy impact, provider/cost impact, compatibility impact, test evidence, and screenshots for UI changes. Update README, privacy, security, and manual checkpoints whenever behavior changes. Major subprojects require an independent review with every Critical or Important finding fixed before completion.

Useful follow-on areas include request-specific grounded research, adaptive practice, worked solutions and rubrics, the final accessible three-pane experience, signed installers, academic-quality evaluation datasets, and safe converter sandbox implementations.
