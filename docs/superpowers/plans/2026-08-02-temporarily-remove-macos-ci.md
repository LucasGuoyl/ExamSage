# Temporarily Remove macOS CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the current ExamSage application on `main` while temporarily removing macOS jobs from the GitHub Actions test matrix.

**Architecture:** Change only the operating-system values in the existing workflow matrix. Preserve the existing push and pull-request triggers, Python 3.11 and 3.12 coverage, installation command, pytest command, and Ruff command.

**Tech Stack:** GitHub Actions YAML, Python with PyYAML for structural verification, pytest, Ruff, Git.

## Global Constraints

- Keep the current application code and new interface unchanged.
- Keep `ubuntu-latest` and `windows-latest` in the CI matrix.
- Keep Python versions `3.11` and `3.12` in the CI matrix.
- Do not change README platform support or usage instructions.
- Push the verified result directly to `main`.

---

### Task 1: Restrict the GitHub Actions matrix to Ubuntu and Windows

**Files:**
- Modify: `.github/workflows/tests.yml:13`
- Create: `docs/superpowers/plans/2026-08-02-temporarily-remove-macos-ci.md`

**Interfaces:**
- Consumes: GitHub Actions matrix expansion through `matrix.os` and `matrix.python-version`.
- Produces: Four CI jobs covering Ubuntu and Windows on Python 3.11 and 3.12.

- [x] **Step 1: Verify the current matrix still contains macOS**

Run:

```powershell
@'
from pathlib import Path
import yaml

workflow = yaml.safe_load(Path('.github/workflows/tests.yml').read_text(encoding='utf-8'))
operating_systems = workflow['jobs']['test']['strategy']['matrix']['os']
assert 'macos-latest' in operating_systems, operating_systems
'@ | python -
```

Expected: exit code `0`, proving the pre-change workflow includes `macos-latest`.

- [x] **Step 2: Apply the minimal workflow change**

Replace:

```yaml
os: [ubuntu-latest, windows-latest, macos-latest]
```

with:

```yaml
os: [ubuntu-latest, windows-latest]
```

- [x] **Step 3: Verify the resulting workflow structure**

Run:

```powershell
@'
from pathlib import Path
import yaml

workflow = yaml.safe_load(Path('.github/workflows/tests.yml').read_text(encoding='utf-8'))
matrix = workflow['jobs']['test']['strategy']['matrix']
assert matrix['os'] == ['ubuntu-latest', 'windows-latest'], matrix['os']
assert matrix['python-version'] == ['3.11', '3.12'], matrix['python-version']
assert workflow['jobs']['test']['runs-on'] == '${{ matrix.os }}'
'@ | python -
```

Expected: exit code `0`.

- [x] **Step 4: Run repository verification**

Run:

```powershell
python -m pytest
ruff check exam_predictor tests app.py
```

Expected: both commands exit with code `0`.

- [ ] **Step 5: Review and commit the scoped change**

Run:

```powershell
git diff --check
git diff -- .github/workflows/tests.yml docs/superpowers/plans/2026-08-02-temporarily-remove-macos-ci.md
git add .github/workflows/tests.yml docs/superpowers/plans/2026-08-02-temporarily-remove-macos-ci.md
git commit -m "ci: temporarily remove macos test jobs"
```

Expected: one commit containing only the workflow change and its implementation plan.

- [ ] **Step 6: Push and inspect GitHub Actions**

Run:

```powershell
git push origin HEAD:main
```

Expected: the remote `main` branch advances to the new commit and GitHub Actions creates four jobs without a macOS job.
