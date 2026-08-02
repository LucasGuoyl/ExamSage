# Temporarily Remove macOS CI Design

## Goal

Keep the current ExamSage application and its new interface on `main`, while temporarily stopping GitHub Actions from running the known-failing macOS test combinations.

## Scope

- Remove `macos-latest` from the operating-system matrix in `.github/workflows/tests.yml`.
- Keep Python 3.11 and 3.12 checks on `ubuntu-latest` and `windows-latest`.
- Do not change application code, tests, launchers, platform support statements, or README usage instructions.
- Do not revert commit `a97c836` or any earlier feature commit.

## Result

Each push to `main` and each pull request will run four jobs: Ubuntu and Windows on Python 3.11 and 3.12. The macOS-specific test failures remain documented by the existing GitHub Actions run and can be fixed before macOS CI is restored.

## Verification

- Parse the workflow as YAML and confirm the matrix contains exactly `ubuntu-latest` and `windows-latest`.
- Confirm both supported Python versions remain present.
- Run the repository's full local test and Ruff checks before pushing.
- Push the resulting commit to `main` and inspect the new GitHub Actions run.
