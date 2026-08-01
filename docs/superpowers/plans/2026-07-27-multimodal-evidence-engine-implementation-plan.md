# ExamSage Multimodal Evidence Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect approved course workspaces to resumable OpenAI/Gemini multimodal tools that publish an honestly partial initial study map quickly and then complete every approved source without repeating validated work.

**Architecture:** Add a focused `exam_predictor.evidence` package containing immutable contracts, an additive SQLite repository, safe part preparation, provider adapters, a bounded scheduler, evidence validation, coverage, and incremental study-map construction. The existing Worker composes these services with `WorkspaceTransmissionGate`; LangGraph invokes typed evidence tools and streams durable progress while preserving Stop/Resume and the legacy feature flag.

**Tech Stack:** Python 3.11/3.12, Pydantic 2, SQLite, LangGraph 1.2, FastAPI, Streamlit, `google-genai` 1.x, `openai` 2.x, `pypdf` 6.x, `python-pptx` 1.x, standard-library ZIP/XML/concurrency primitives, pytest, Ruff.

## Global Constraints

- Work only in the existing isolated branch/worktree `codex/agent-kernel` at `.worktrees/agent-kernel`.
- Preserve all Subproject 1–2 security invariants and additive migrations.
- The transmission gate is the only route from an approved workspace to provider-bound bytes.
- Never persist an API key, authorization header, signed URL, unrestricted path, SDK client, open handle, or raw exception.
- Original native-folder sources remain read-only and are never cleanup targets.
- One provider profile and one credential must cover chat, multimodal analysis, embeddings, and later search.
- No local AI or OCR model may be introduced.
- Every approved supported file must end as processed, pending/retryable, or visibly failed; nothing is silently omitted.
- Publish only evidence-backed nodes and label initial snapshots with exact partial coverage.
- Default multimodal concurrency is 2; default text synthesis concurrency is at most 4.
- Provider and tool deadlines, attempts, and backoff are deterministic policy values; the model cannot raise them.
- Stop/restart/Resume must not repeat a completed provider operation.
- Agent study-map work must not call `ExamSageAgent.build_course()`.
- Use TDD for every task, run the stated focused gate, request an independent review after each task, and make one focused commit per accepted task.
- Do not implement Subproject 4 web/practice/rubric/export work or Subproject 5 final layout in this plan.
- Every new evidence-engine UI string must use shared English/Simplified-Chinese copy keys. Persist the UI
  language locally, and keep generated-content language tied to the current user message unless explicitly
  overridden.

---

## File map

| Path | Responsibility |
|---|---|
| `exam_predictor/evidence/models.py` | Immutable source-part, evidence, coverage, citation, course-group, and study-map contracts. |
| `exam_predictor/evidence/policy.py` | Deadlines, attempt limits, concurrency, preparation bounds, schema/prompt versions, and deterministic source priority. |
| `exam_predictor/evidence/store.py` | Additive evidence SQLite schema, repositories, claims, cache keys, invalidation, coverage, and snapshots. |
| `exam_predictor/evidence/artifacts.py` | Identity-bound ExamSage-owned part/evidence artifact publication, reads, reference cleanup, and atomic writes. |
| `exam_predictor/evidence/preparation.py` | Provider-neutral bounded part preparation for PDFs, text/data, images, OOXML, ZIP members, and converter outputs. |
| `exam_predictor/evidence/converter.py` | Resource-limited legacy DOC/PPT/XLS converter interface and subprocess implementation. |
| `exam_predictor/evidence/prompts.py` | Versioned untrusted-source analysis and study-map prompts. |
| `exam_predictor/evidence/providers.py` | Provider-neutral request/result protocol plus OpenAI/Gemini adapters and safe error normalization. |
| `exam_predictor/evidence/scheduler.py` | Bounded concurrent execution, retry/deadline/Stop checks, idempotent publication, and progressive events. |
| `exam_predictor/evidence/study_map.py` | Hierarchical synthesis, validation, evidence dependencies, focus/confidence separation, and snapshots. |
| `exam_predictor/evidence/service.py` | Workspace-facing orchestration across gate, preparation, cache, scheduler, coverage, and snapshots. |
| `exam_predictor/tools/evidence.py` | Typed LangGraph tool inputs/results and evidence-tool registry integration. |
| `exam_predictor/graphs/evidence.py` | Resumable evidence subgraph and interrupt boundaries. |
| `exam_predictor/worker/evidence_routes.py` | Authenticated coverage and snapshot read routes. |
| `exam_predictor/ui/evidence_view.py` | Functional progress, coverage, initial/final map, failure, Stop, and Resume rendering. |

## Task 1: Evidence contracts and deterministic policy

**Files:**
- Create: `exam_predictor/evidence/__init__.py`
- Create: `exam_predictor/evidence/models.py`
- Create: `exam_predictor/evidence/policy.py`
- Create: `tests/evidence/__init__.py`
- Create: `tests/evidence/test_models.py`
- Create: `tests/evidence/test_policy.py`

**Interfaces:**
- Consumes: `workspace.models.ManifestEntry`, `SourceState`, and `normalize_relative_path`.
- Produces: `EvidencePolicy`, `PartState`, `SnapshotStatus`, `SourcePartPlan`, `EvidenceCitation`, `EvidenceUnit`, `CoverageItem`, `CoverageSummary`, `KnowledgeNode`, `StudyMapSnapshot`, `source_priority()`.

- [ ] **Step 1: Write failing immutable-contract tests**

```python
def test_source_part_rejects_absolute_paths_and_secret_fields():
    with pytest.raises(ValidationError):
        SourcePartPlan(
            part_id="part-1", workspace_id="w" * 32, revision_id="r" * 32,
            entry_id="e" * 32, relative_path="C:/secret.pdf", source_sha256="a" * 64,
            part_sha256="b" * 64, ordinal=0, locator="pages 1-20",
            media_type="application/pdf", size_bytes=100, scheduling_class="syllabus",
            priority=0, state=PartState.PLANNED, idempotency_key="i" * 32,
        )


def test_initial_snapshot_requires_nonempty_coverage_and_evidence_dependencies():
    with pytest.raises(ValidationError):
        StudyMapSnapshot(
            snapshot_id="s" * 32, workspace_id="w" * 32, revision_id="r" * 32,
            status=SnapshotStatus.INITIAL, nodes=(), coverage=None,
            evidence_unit_ids=(), created_at=NOW,
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/evidence/test_models.py tests/evidence/test_policy.py -q`

Expected: collection fails because `exam_predictor.evidence` does not exist.

- [ ] **Step 3: Implement exact enums and frozen Pydantic models**

```python
class PartState(StrEnum):
    PLANNED = "planned"
    AUTHORIZED = "authorized"
    PREPARED = "prepared"
    RUNNING = "running"
    PROCESSED = "processed"
    RETRY_WAIT = "retry_wait"
    FAILED = "failed"
    INVALIDATED = "invalidated"


class SnapshotStatus(StrEnum):
    INITIAL = "initial"
    COMPLETE = "complete"


class EvidencePolicy(FrozenModel):
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
```

Implement validators that normalize relative paths, require 64-character lowercase hashes, require unique
evidence dependencies, separate `focus_score` from `confidence`, and forbid extra fields.

- [ ] **Step 4: Implement deterministic source priority and representative ordinals**

```python
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
```

Priority order must be syllabus/guidance, exams/problems, slides/notes, textbooks/references, supplemental.
For `total_parts >= 3`, representative ordinals start with `0`, midpoint, and last, without duplicates.

- [ ] **Step 5: Run focused tests and static checks**

Run: `.\.venv\Scripts\python.exe -m pytest tests/evidence/test_models.py tests/evidence/test_policy.py -q`

Run: `.\.venv\Scripts\python.exe -m ruff check exam_predictor/evidence tests/evidence`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add exam_predictor/evidence tests/evidence
git commit -m "feat: define multimodal evidence contracts"
```

## Task 2: Durable evidence store and additive recovery

**Files:**
- Create: `exam_predictor/evidence/store.py`
- Create: `tests/evidence/test_store.py`

**Interfaces:**
- Consumes: Task 1 models and `RuntimeStore`/`WorkspaceStore` connection conventions.
- Produces: `EvidenceStore(path)`, `migrate()`, `upsert_part_plans()`, `claim_parts()`, `record_attempt()`, `publish_evidence()`, `save_snapshot()`, `coverage()`, `invalidate_entry()`, `recover_unfinished()`.

- [ ] **Step 1: Write failing schema and migration tests**

```python
def test_migration_is_additive_idempotent_and_never_stores_secret_columns(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    store.migrate()
    store.migrate()
    names = sqlite_names_and_columns(tmp_path / "evidence.sqlite3")
    assert "evidence_parts" in names
    assert "evidence_units" in names
    assert not any("key" in column or "authorization" in column for column in all_columns(names))


def test_recovery_pauses_running_parts_without_consuming_attempt(tmp_path):
    store = seeded_store(tmp_path)
    store.mark_running("part-1", attempt=1)
    store.recover_unfinished()
    part = store.get_part("part-1")
    assert part.state is PartState.RETRY_WAIT
    assert part.attempt_count == 1
```

- [ ] **Step 2: Verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/evidence/test_store.py -q`

Expected: import failure for `EvidenceStore`.

- [ ] **Step 3: Implement schema version 1**

Create tables `evidence_meta`, `evidence_parts`, `evidence_attempts`, `evidence_units`,
`evidence_cache`, `study_map_snapshots`, `study_map_dependencies`, and indexes on workspace/revision/state,
cache key, source hash, and next-attempt time. Use WAL, foreign keys, explicit connection closure, UTC ISO
timestamps, canonical JSON, and transactions matching existing stores.

- [ ] **Step 4: Implement atomic claims and idempotent publication**

Implement `claim_parts(workspace_id, revision_id, *, limit, now)` returning a tuple of exact
`SourcePartPlan` rows and `publish_evidence(part_id, unit, *, cache_key, completed_at)` returning `False`
only when the identical idempotency key and evidence-unit hash were already published.

Claims use `BEGIN IMMEDIATE`; publication writes unit, cache row, part state, and dependency atomically.

- [ ] **Step 5: Implement coverage, invalidation, cleanup, and recovery tests**

Cover restart recovery, stale revision claims, identical cache reuse, changed-entry invalidation, unaffected
entry preservation, workspace deletion, and failure rollback.

- [ ] **Step 6: Run focused and regression tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/evidence/test_store.py tests/runtime/test_store.py tests/workspace/test_store.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add exam_predictor/evidence/store.py tests/evidence/test_store.py
git commit -m "feat: persist resumable evidence state"
```

## Task 3: Identity-bound evidence artifact store

**Files:**
- Create: `exam_predictor/evidence/artifacts.py`
- Create: `tests/evidence/test_artifacts.py`

**Interfaces:**
- Consumes: Task 1 identifiers and the ownership patterns in `workspace/browser_intake.py`.
- Produces: `EvidenceArtifactStore(root)`, `publish_part()`, `open_part()`, `publish_json()`, `read_json()`, `delete_workspace()`.

- [ ] **Step 1: Write failing containment and atomic-publication tests**

Create three concrete tests: replace a workspace artifact directory with a link and assert
`ArtifactBoundaryError("artifact_identity_changed")`; publish known bytes and assert the reopened SHA-256;
pass a native source path as a cleanup target and assert deletion is refused while source bytes remain exact.

- [ ] **Step 2: Verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/evidence/test_artifacts.py -q`

- [ ] **Step 3: Implement owned relative layout and atomic writes**

Use only:

```text
<data-dir>/workspaces/<workspace-id>/evidence/parts/<part-id>
<data-dir>/workspaces/<workspace-id>/evidence/units/<unit-id>.json
<data-dir>/workspaces/<workspace-id>/evidence/snapshots/<snapshot-id>.json
```

Validate every identifier against `^[A-Za-z0-9_-]{16,128}$`, open without following links/reparse points,
write to a sibling temporary file, fsync, replace atomically, and verify the expected SHA-256 before publish.

- [ ] **Step 4: Add restart, cleanup-pending, and Windows handle tests**

Inject filesystem operations so Windows sharing violations and POSIX identity substitution settle as a safe
retry state without widening the target.

- [ ] **Step 5: Run tests and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/evidence/test_artifacts.py tests/workspace/test_browser_intake.py -q`

```powershell
git add exam_predictor/evidence/artifacts.py tests/evidence/test_artifacts.py
git commit -m "feat: store evidence artifacts safely"
```

## Task 4: PDF, text, structured-data, and image part preparation

**Files:**
- Create: `exam_predictor/evidence/preparation.py`
- Create: `tests/evidence/test_preparation.py`
- Modify: `requirements.txt`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: a binary stream opened by `WorkspaceTransmissionGate.open_approved`, Task 1 policy/models, and Task 3 artifact store.
- Produces: `SourcePartPreparer.prepare(request, stream) -> tuple[SourcePartPlan, ...]` and `PreparedPartRequest`.

- [ ] **Step 1: Write failing representative PDF tests**

```python
def test_long_pdf_plans_begin_middle_end_before_remaining_pages(tmp_path):
    parts = prepare_pdf(make_pdf(tmp_path, pages=357), pages_per_part=24)
    assert [part.ordinal for part in parts[:3]] == [0, 7, 14]
    assert sorted(part.ordinal for part in parts) == list(range(15))
    assert all(part.size_bytes <= 10 * 1024 * 1024 for part in parts)


def test_each_part_has_stable_hash_locator_and_idempotency_key(tmp_path):
    first = prepare_pdf(make_pdf(tmp_path, pages=50), pages_per_part=24)
    second = prepare_pdf(make_pdf(tmp_path, pages=50), pages_per_part=24)
    assert [(item.locator, item.part_sha256, item.idempotency_key) for item in first] == [
        (item.locator, item.part_sha256, item.idempotency_key) for item in second
    ]
```

- [ ] **Step 2: Verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/evidence/test_preparation.py -q`

- [ ] **Step 3: Implement bounded PDF preparation**

Read from the authorized spool, use `pypdf` to write bounded page groups, shrink groups that exceed
`max_part_bytes`, reject an indivisible oversized page visibly, and publish each prepared part through the
artifact store. Locators use `pages X-Y`; ordinals are stable for identical bytes and policy.

- [ ] **Step 4: Implement bounded text/data/image preparation**

- Decode TXT/MD/HTML/CSV/TSV/JSON/YAML with UTF-8 then bounded replacement fallback; never fetch HTML URLs.
- Split text at heading/line boundaries with overlap metadata, not arbitrary byte truncation.
- Preserve spreadsheet-like rows in bounded ranges for CSV/TSV.
- Validate image type and dimensions from headers; reject decompression-risk dimensions before provider use.
- Extract a bounded representative frame set from animated GIF only when Pillow is installed; add
  `Pillow>=11,<13` to both dependency files and record omitted-frame counts.

- [ ] **Step 5: Add corruption, encrypted PDF, oversized page, encoding, HTML injection, and image-bomb tests**

Every failure returns a stable code and relative locator; no raw content appears in the error.

- [ ] **Step 6: Run tests and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/evidence/test_preparation.py tests/test_agent_core.py -q`

```powershell
git add exam_predictor/evidence/preparation.py tests/evidence/test_preparation.py requirements.txt pyproject.toml
git commit -m "feat: prepare bounded multimodal source parts"
```

## Task 5: OOXML, ZIP-member, and legacy-converter preparation

**Files:**
- Modify: `exam_predictor/evidence/preparation.py`
- Create: `exam_predictor/evidence/converter.py`
- Create: `tests/evidence/test_ooxml_preparation.py`
- Create: `tests/evidence/test_converter.py`

**Interfaces:**
- Consumes: Task 4 preparer/artifact store and safe archive policy from `workspace/archive.py`.
- Produces: slide-, section-, sheet-, media-, and archive-member-located parts; `LegacyOfficeConverter` protocol.

- [ ] **Step 1: Write failing PPTX/DOCX/XLSX relationship tests**

Construct synthetic OOXML packages and assert slide order, heading order, sheet/range order, formulas,
displayed values, and embedded image relationships remain attached to their source locator. Assert a single
embedded image is not submitted twice for one provider contract.

- [ ] **Step 2: Write failing ZIP authority tests**

Assert only members already represented as safe preview entries may be prepared; traversal, links, encrypted
members, bombs, and unapproved nested files fail before artifact publication.

- [ ] **Step 3: Implement deterministic OOXML and ZIP preparation**

Use `zipfile`, `defusedxml`, and existing safe archive rules. Add `defusedxml>=0.7,<1` to dependencies. Local
XML extraction is deterministic and does not infer meaning. Group parts under provider byte/token bounds.

- [ ] **Step 4: Implement converter protocol and resource-limited subprocess adapter**

```python
class LegacyOfficeConverter(Protocol):
    def available(self) -> bool:
        """Return whether the fixed converter executable is available."""

    def convert(self, source: BinaryIO, *, suffix: str, deadline: float) -> ConvertedDocument:
        """Convert one authorized stream into one bounded owned document."""
```

The default adapter discovers LibreOffice without shell interpolation, writes only beneath an owned temporary
directory, invokes a fixed argument vector, enforces wall-clock timeout and output bounds, rejects links and
unexpected outputs, and reports `converter_unavailable` or `converter_failed` visibly.

- [ ] **Step 5: Run focused and security regressions**

Run: `.\.venv\Scripts\python.exe -m pytest tests/evidence/test_ooxml_preparation.py tests/evidence/test_converter.py tests/workspace/test_archive.py -q`

- [ ] **Step 6: Commit**

```powershell
git add exam_predictor/evidence/preparation.py exam_predictor/evidence/converter.py tests/evidence requirements.txt pyproject.toml
git commit -m "feat: prepare office and archive evidence"
```

## Task 6: Provider evidence contracts, explicit timeouts, and safe errors

**Files:**
- Create: `exam_predictor/evidence/prompts.py`
- Create: `exam_predictor/evidence/providers.py`
- Modify: `exam_predictor/providers.py`
- Create: `tests/evidence/test_providers.py`
- Modify: `tests/test_gemini_transport.py`

**Interfaces:**
- Consumes: Task 1 evidence contracts and prepared bounded bytes from Task 4/5.
- Produces: `EvidenceProvider` protocol, `ProviderEvidenceAdapter`, `AnalyzeSourcePartRequest`, `EvidenceProviderError`, OpenAI/Gemini implementations.

- [ ] **Step 1: Write failing timeout and retry-ownership tests**

Create concrete fake-constructor tests that capture keyword arguments and assert OpenAI receives
`timeout=90.0, max_retries=0`; Gemini receives `HttpOptions.timeout == 90000` and one SDK attempt; and a
fake 503 containing a sentinel key/body normalizes to code `provider_unavailable`, `retryable=True`, bounded
`retry_after_seconds`, with the sentinel absent from `str(error)`, `repr(error)`, and serialized events.

- [ ] **Step 2: Verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/evidence/test_providers.py tests/test_gemini_transport.py -q`

- [ ] **Step 3: Configure SDK boundaries from official contracts**

- OpenAI: construct `OpenAI(api_key=api_key, timeout=policy.provider_timeout_seconds, max_retries=0)` so ExamSage owns the bounded retry budget.
- Gemini: construct `genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(policy.provider_timeout_seconds * 1000), retry_options=types.HttpRetryOptions(attempts=1)))` and verify the installed SDK accepts the exact fields.
- Preserve existing one-key model routing and add no second client credential.

Official references:

- `https://github.com/openai/openai-python#timeouts`
- `https://github.com/openai/openai-python#retries`
- `https://googleapis.github.io/python-genai/genai.html`
- `https://googleapis.github.io/python-genai/#genai.types.HttpOptions`

- [ ] **Step 4: Implement strict structured multimodal analysis**

Use versioned prompts that delimit untrusted content, request JSON MIME/schema where the provider supports
it, set bounded output tokens, preserve locators, and return only typed raw results for validation. Use inline
bytes when within provider support; otherwise use provider file upload and best-effort deletion.

- [ ] **Step 5: Normalize retryable errors**

Map 429, 5xx, timeouts, and protocol disconnects to safe retryable codes; capture numeric `Retry-After` only
within policy bounds; map credentials, unsupported model/media, and invalid request to non-retryable action
codes. Never include raw response bodies or request URLs in user-visible text.

- [ ] **Step 6: Run provider and legacy regressions**

Run: `.\.venv\Scripts\python.exe -m pytest tests/evidence/test_providers.py tests/test_gemini_transport.py tests/test_agent_core.py tests/tools/test_kernel_tools.py -q`

- [ ] **Step 7: Commit**

```powershell
git add exam_predictor/evidence/prompts.py exam_predictor/evidence/providers.py exam_predictor/providers.py tests/evidence/test_providers.py tests/test_gemini_transport.py
git commit -m "feat: add bounded multimodal provider contracts"
```

## Task 7: Concurrent scheduler, cache, checkpoints, and Stop/Resume

**Files:**
- Create: `exam_predictor/evidence/scheduler.py`
- Create: `tests/evidence/test_scheduler.py`

**Interfaces:**
- Consumes: Tasks 1–6, `RunControlRegistry`, runtime event emitter, provider session, and injected monotonic/wall clocks.
- Produces: `EvidenceScheduler.run_frontier()`, `run_to_completion()`, `SchedulerOutcome`.

- [ ] **Step 1: Write deterministic failing concurrency tests**

Create deterministic tests with a barrier-counting fake provider: six ready parts must observe a maximum
of two concurrent calls; the first claims must be syllabus then assessment then beginning/middle/end of the
large reference; a fake 503 with `Retry-After: 2` must advance the fake clock exactly two seconds and pause
after the third failed route attempt; a Stop set after the first publication must prevent the second call,
and Resume must begin at the second part.

Use barriers, fake clocks, and fake providers; do not use wall-clock sleeps.

- [ ] **Step 2: Verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/evidence/test_scheduler.py -q`

- [ ] **Step 3: Implement bounded executor and deterministic claims**

Use `ThreadPoolExecutor(max_workers=policy.multimodal_concurrency)` around synchronous provider adapters.
Claim only the next bounded frontier, check Stop before authorization/provider/publication, and close every
future/executor deterministically on pause or error.

- [ ] **Step 4: Implement cache and crash-window idempotency**

Before provider use, check the full cache key. After provider success, validate then publish evidence and part
completion in one store operation. A crash after provider return but before publication may repeat at most that
unpublished attempt; a published result never repeats.

- [ ] **Step 5: Implement deadline and progress events**

Emit `part_planned`, `part_started`, `part_retrying`, `part_processed`, `part_failed`, `coverage_updated`, and
`initial_map_due` events containing only safe identifiers, relative paths, locators, counts, and next action.

- [ ] **Step 6: Run scheduler, runtime, and secret tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/evidence/test_scheduler.py tests/runtime/test_coordinator.py tests/test_secret_audit.py -q`

- [ ] **Step 7: Commit**

```powershell
git add exam_predictor/evidence/scheduler.py tests/evidence/test_scheduler.py
git commit -m "feat: schedule resumable evidence analysis"
```

## Task 8: Evidence validation and incremental study-map snapshots

**Files:**
- Create: `exam_predictor/evidence/study_map.py`
- Create: `tests/evidence/test_study_map.py`

**Interfaces:**
- Consumes: validated evidence units, coverage, provider study-map synthesis, and Task 2 store.
- Produces: `EvidenceValidator`, `StudyMapBuilder.publish_initial()`, `publish_complete()`, `answer_context()`.

- [ ] **Step 1: Write failing validation tests**

Cover malformed JSON, unknown locators, duplicate evidence IDs, invalid prerequisite references, literal
probability labels, citations to pending parts, course-group confidence, prompt injection, and one repair.

- [ ] **Step 2: Verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/evidence/test_study_map.py -q`

- [ ] **Step 3: Implement evidence validator and bounded repair**

Validate provider output into Task 1 models. Repair receives only schema error paths/codes plus the original
provider response, never secrets or unrestricted source handles. A second invalid result marks the part failed.

- [ ] **Step 4: Implement hierarchical synthesis**

Build compact evidence summaries in bounded batches, synthesize course groups and a chapter tree, validate
that each knowledge node cites processed evidence, compute deterministic evidence counts, and retain provider
focus/confidence signals only within `[0, 1]`. UI bands are High/Medium/Low and Strong/Moderate/Limited.

- [ ] **Step 5: Implement initial/complete publication rules and dependency invalidation**

Initial publication requires at least one node and nonempty complete coverage. Complete publication requires
all approved parts processed or terminal failed. Save snapshot JSON atomically and rows transactionally.

- [ ] **Step 6: Run tests and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/evidence/test_study_map.py tests/evidence/test_store.py -q`

```powershell
git add exam_predictor/evidence/study_map.py tests/evidence/test_study_map.py
git commit -m "feat: build incremental cited study maps"
```

## Task 9: Workspace evidence service and transmission-gate vertical slice

**Files:**
- Create: `exam_predictor/evidence/service.py`
- Create: `tests/evidence/test_service.py`

**Interfaces:**
- Consumes: `WorkspaceStore`, `WorkspaceTransmissionGate`, Tasks 1–8, provider sessions, controls, and event emitter.
- Produces: `EvidenceService.inspect()`, `build_study_map()`, `continue_analysis()`, `answer_from_evidence()`, `delete_workspace_evidence()`.

- [ ] **Step 1: Write failing authorization-boundary tests**

Create four vertical tests: an unapproved workspace produces `source_approval_required` and zero fake calls;
the fake provider receives bytes equal to the single consumed gate spool and a second open fails; exclusion
between two parts pauses before call two; changing one of two approved files invalidates only that file's
units and snapshot dependencies while the sibling cache remains reusable.

- [ ] **Step 2: Verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/evidence/test_service.py -q`

- [ ] **Step 3: Implement workspace inspection and preparation transaction**

Load the current approved revision, compute source priority, authorize one source at a time, consume the token
into a Worker-owned spool, prepare parts, publish plans, and discard the spool before the provider scheduler.

- [ ] **Step 4: Implement build/resume and focused evidence answers**

`build_study_map()` runs the initial frontier, publishes the initial snapshot as soon as valid, then continues
remaining parts until complete or paused. `answer_from_evidence()` uses stored evidence only and returns
citations plus current coverage limitations.

- [ ] **Step 5: Implement deletion and restart recovery**

Workspace deletion calls evidence cleanup only after unsettled-run guards. Startup recovers unfinished evidence
parts to paused/retry state and never starts provider work implicitly.

- [ ] **Step 6: Run evidence/workspace regressions and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/evidence/test_service.py tests/workspace/test_transmission.py tests/workspace/test_service.py -q`

```powershell
git add exam_predictor/evidence/service.py tests/evidence/test_service.py
git commit -m "feat: connect approved workspaces to evidence tools"
```

## Task 10: LangGraph evidence tools and resumable subgraph

**Files:**
- Create: `exam_predictor/tools/evidence.py`
- Create: `exam_predictor/graphs/evidence.py`
- Modify: `exam_predictor/tools/kernel.py`
- Modify: `exam_predictor/graphs/kernel.py`
- Modify: `exam_predictor/runtime/coordinator.py`
- Create: `tests/tools/test_evidence_tools.py`
- Create: `tests/graphs/test_evidence_graph.py`
- Modify: `tests/tools/test_kernel_tools.py`
- Modify: `tests/graphs/test_kernel_graph.py`
- Modify: `tests/runtime/test_coordinator.py`

**Interfaces:**
- Consumes: Task 9 service and existing run/workspace/provider state.
- Produces: registered tools `inspect_course_sources`, `build_study_map`, `continue_source_analysis`, `answer_from_course_evidence` and `build_evidence_graph()`.

- [ ] **Step 1: Write failing planner and tool-registration tests**

Assert study-map requests select `build_study_map`, source-status questions select `inspect_course_sources`,
focused course questions with evidence select `answer_from_course_evidence`, and generic chat remains
`tutor_reply`. The planner cannot invent tools or entry IDs.

- [ ] **Step 2: Write failing graph Stop/checkpoint/resume tests**

Assert interrupts occur before source authorization, before each provider frontier, after validated evidence,
and before snapshot publication; Resume restores the exact workspace/revision/run state.

- [ ] **Step 3: Verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/tools/test_evidence_tools.py tests/graphs/test_evidence_graph.py -q`

- [ ] **Step 4: Implement typed evidence tools and planner context**

Extend `ToolPlan.tool` with the four registered names. Pass `workspace_id` plus compact coverage metadata to
planning. Evidence tool arguments contain only workspace ID and user intent; source selection remains
server-side.

- [ ] **Step 5: Implement and compose the evidence subgraph**

Nodes: hydrate authority, inspect coverage, plan frontier, interrupt gate, analyze frontier, validate/persist,
maybe publish initial, continue or pause, publish complete, compose cited response. Reuse existing control and
event contracts.

- [ ] **Step 6: Prove Agent mode never calls legacy build**

Patch `ExamSageAgent.build_course` to raise in a vertical test and complete a fake-provider study map through
the real graph.

- [ ] **Step 7: Run graph/runtime regressions and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/tools tests/graphs tests/runtime/test_coordinator.py tests/test_agent_kernel_acceptance.py -q`

```powershell
git add exam_predictor/tools exam_predictor/graphs exam_predictor/runtime/coordinator.py tests/tools tests/graphs tests/runtime/test_coordinator.py
git commit -m "feat: orchestrate multimodal evidence in LangGraph"
```

## Task 11: Worker composition, coverage API, and typed client

**Files:**
- Create: `exam_predictor/worker/evidence_routes.py`
- Modify: `exam_predictor/worker/api.py`
- Modify: `exam_predictor/runtime/client.py`
- Create: `tests/worker/test_evidence_api.py`
- Modify: `tests/worker/test_api.py`
- Modify: `tests/runtime/test_client.py`

**Interfaces:**
- Consumes: EvidenceStore, artifact store, provider adapters, transmission gate, service, runtime coordinator.
- Produces authenticated endpoints for coverage/current snapshot and typed client methods.

- [ ] **Step 1: Write failing production-composition test**

Start uninjected `create_worker_app`, connect a fake provider boundary, approve a real temporary workspace,
submit a study-map message, and assert the production composition reaches the evidence service without a
missing dependency or direct path read.

- [ ] **Step 2: Write auth-before-body and safe-error API tests**

Add:

```text
GET /v1/workspaces/{workspace_id}/evidence/coverage
GET /v1/workspaces/{workspace_id}/evidence/snapshots/current
```

Assert unauthenticated requests never parse IDs/bodies and errors expose no absolute paths/source text.

- [ ] **Step 3: Verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/worker/test_evidence_api.py tests/runtime/test_client.py -q`

- [ ] **Step 4: Compose dependencies and lifecycle**

Create one EvidenceStore, artifact store, transmission gate, provider adapter factory, EvidenceService, and
scheduler per Worker. Startup migrates/recover-pauses; shutdown requests safe pause, closes schedulers, then
workspace/runtime services in deterministic order.

- [ ] **Step 5: Implement routes and client methods**

Return only public Pydantic coverage/snapshot models. Client methods preserve the existing loopback/token,
timeout, redaction, and caller-owned stream contracts.

- [ ] **Step 6: Run worker/client regressions and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/worker tests/runtime/test_client.py tests/test_launcher.py -q`

```powershell
git add exam_predictor/worker exam_predictor/runtime/client.py tests/worker tests/runtime/test_client.py
git commit -m "feat: expose evidence engine through local worker"
```

## Task 12: Functional progressive evidence UI

**Files:**
- Create: `exam_predictor/ui/i18n.py`
- Create: `exam_predictor/ui/evidence_view.py`
- Modify: `exam_predictor/ui/agent_view.py`
- Modify: `exam_predictor/ui/workspace_view.py`
- Create: `tests/ui/test_i18n.py`
- Create: `tests/ui/test_evidence_view.py`
- Modify: `tests/ui/test_agent_view.py`
- Modify: `tests/ui/test_workspace_view.py`

**Interfaces:**
- Consumes: Task 11 client coverage/snapshot methods, existing durable run events, and the persisted UI
  language preference.
- Produces: switchable English/Simplified-Chinese copy, visible file/part coverage, activity, initial/final
  snapshot, safe actions, and no indefinite spinner. Generated academic prose continues to follow the
  current user-message language unless explicitly overridden.

- [ ] **Step 1: Write failing state-projection tests**

Test zero-data, preparing, analyzing, retrying, paused, partial initial, complete, changed approval, converter
failure, and provider-capacity states in both supported interface languages. Verify every copy key is complete,
the selected UI language persists, changing it does not mutate stored academic artifacts, and displayed counts
match the complete server coverage rather than the current manifest page.

- [ ] **Step 2: Verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ui/test_i18n.py tests/ui/test_evidence_view.py -q`

- [ ] **Step 3: Implement functional renderer**

Implement the shared copy catalog and persisted language preference, then show file and part progress, current
relative source/locator, initial-map coverage banner, chapter tree, evidence citations, limitations,
failed-source reasons, Stop, Resume, Rescan/Reapprove, and Retry actions. Polling remains bounded and offers
manual Refresh after the cap. UI-language changes never trigger provider calls or artifact regeneration.

- [ ] **Step 4: Remove misleading Agent-alpha capability copy**

Agent mode must no longer say course tools arrive later. It must state exactly which evidence tools are
available and must never display the legacy estimate/build controls.

- [ ] **Step 5: Run UI and smoke regressions and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ui tests/test_app_smoke.py tests/test_agent_kernel_acceptance.py -q`

```powershell
git add exam_predictor/ui tests/ui
git commit -m "feat: show progressive course evidence"
```

## Task 13: End-to-end acceptance, performance harness, and failure cleanup

**Files:**
- Create: `tests/test_multimodal_evidence_acceptance.py`
- Create: `tests/fixtures/reference_course/README.md`
- Create: `scripts/benchmark_initial_map.py`
- Create: `tests/test_evidence_benchmark.py`
- Modify: `scripts/check_secret_patterns.py`
- Modify: `tests/test_secret_audit.py`

**Interfaces:**
- Consumes: real Worker/SQLite/transmission/preparation/graph/UI client with fake provider and injected clocks.
- Produces: one reproducible acceptance gate and one opt-in live benchmark command.

- [ ] **Step 1: Build a licensed synthetic mixed-format reference pack fixture**

Generate fixtures during tests rather than committing copyrighted materials: syllabus Markdown, past-paper
JSON, PPTX with image/formula labels, 120-page PDF, XLSX-like OOXML fixture, scan image, and safe ZIP. Document
the synthetic license and expected knowledge relationships.

- [ ] **Step 2: Write the complete vertical RED test**

Exercise select/scan/exclude/approve/connect/chat/initial map/Stop/close/restart/Resume/final map/change/
reapprove/invalidate/delete. Assert first activity <=15 fake seconds and initial snapshot <=180 fake seconds,
exact coverage, zero duplicate completed calls, no legacy build, and byte-for-byte original sources.

- [ ] **Step 3: Implement only missing integration behavior until GREEN**

Do not weaken assertions or fake production components other than provider, vault, picker, clocks, and external
converter boundary.

- [ ] **Step 4: Add opt-in live benchmark**

`scripts/benchmark_initial_map.py` requires an explicit provider profile and approved synthetic fixture. It
records machine, provider, models, bytes/pages, calls, retries, time to activity, time to initial/final map,
coverage, estimated usage, and safe error codes. It never prints keys or source contents and never runs in CI.

- [ ] **Step 5: Harden abandoned legacy-intake visibility**

Add a read-only diagnostic count/size function and a safe user-triggered cleanup action that deletes only
verified `~/.examsage/intake/<session-id>` copies. Never auto-delete unknown paths and never touch native course
folders. Add tests for identity substitution and active-session refusal.

- [ ] **Step 6: Run acceptance and full regression**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_multimodal_evidence_acceptance.py tests/test_evidence_benchmark.py tests/test_secret_audit.py -q`

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all deterministic tests pass; platform/live skips remain explicitly reported.

- [ ] **Step 7: Commit**

```powershell
git add tests/test_multimodal_evidence_acceptance.py tests/fixtures/reference_course scripts/benchmark_initial_map.py tests/test_evidence_benchmark.py scripts/check_secret_patterns.py tests/test_secret_audit.py
git commit -m "test: verify progressive multimodal evidence"
```

## Task 14: Documentation, version 0.6.0, manual evidence, and subproject review

**Files:**
- Modify: `README.md`
- Modify: `PRIVACY.md`
- Modify: `SECURITY.md`
- Modify: `CONTRIBUTING.md`
- Modify: `exam_predictor/__init__.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_basic.py`
- Create: `docs/manual-tests/2026-07-27-multimodal-evidence-checkpoints.md`

**Interfaces:**
- Consumes: all Task 1–13 acceptance evidence.
- Produces: honest 0.6.0 documentation and a reviewed Subproject 3 handoff to Subproject 4.

- [ ] **Step 1: Write failing documentation/version assertions**

Assert all three public version surfaces are `0.6.0`; README distinguishes initial versus complete coverage,
documents one-key Agent launch and benchmark conditions, and no longer presents the legacy route as the
recommended product.

- [ ] **Step 2: Update documentation without overclaiming**

Document provider retention, approved-byte flow, evidence caching, source invalidation, timeouts/retries,
partial coverage, deletion, converter behavior, known limitations, and exact outstanding live checks.

- [ ] **Step 3: Record manual checkpoints honestly**

Record Windows native picker/Credential Manager, Gemini/OpenAI reference benchmark, browser fallback, macOS
Finder/Keychain, and clean-launch results. Mark unavailable checks `outstanding`; do not convert automated fakes
into live evidence.

- [ ] **Step 4: Run the final automated gate**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall -q app.py exam_predictor scripts tests
.\.venv\Scripts\python.exe scripts\check_secret_patterns.py --root .
.\.venv\Scripts\python.exe -m pip check
git diff --check 8335f29..HEAD
git status --short
```

- [ ] **Step 5: Request independent whole-subproject review**

Review `8335f29..HEAD` against the design and this plan. Fix every Critical/Important finding with focused TDD,
run one scoped re-review, and rerun the full gate.

- [ ] **Step 6: Commit**

```powershell
git add README.md PRIVACY.md SECURITY.md CONTRIBUTING.md exam_predictor/__init__.py pyproject.toml tests/test_basic.py docs/manual-tests/2026-07-27-multimodal-evidence-checkpoints.md
git commit -m "feat: complete multimodal evidence engine"
```

## Final Subproject 3 acceptance checklist

- [ ] Agent study-map messages consume exact approved hashes through `WorkspaceTransmissionGate`.
- [ ] No Agent evidence path invokes `ExamSageAgent.build_course()`.
- [ ] Every approved source has exact file and part coverage.
- [ ] Initial maps cite only processed evidence and expose pending/failed content.
- [ ] All approved sources eventually process or visibly fail.
- [ ] Source priority favors exam evidence and representative long-document parts.
- [ ] Multimodal concurrency, deadlines, retries, `Retry-After`, and repair are bounded.
- [ ] Stop/restart/Resume repeats no completed provider operation.
- [ ] Unchanged hashes reuse evidence; changed hashes invalidate exact dependencies.
- [ ] OpenAI and Gemini use one key, explicit timeouts, and ExamSage-owned retries.
- [ ] Secrets, absolute source paths, signed URLs, and source text stay out of prohibited surfaces.
- [ ] Functional UI shows progress, coverage, initial/final status, and actionable failure states.
- [ ] Synthetic performance acceptance passes and live checks are reported honestly.
- [ ] Full regression/static/security gates pass.
- [ ] Independent review has no Critical or Important finding.
