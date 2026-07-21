# ExamSage Secure Course Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Use
> `superpowers:test-driven-development` for every behavior change,
> `superpowers:systematic-debugging` for every unexpected failure, and
> `superpowers:verification-before-completion` before any completion claim. Steps use checkbox
> (`- [ ]`) syntax for execution tracking.

**Goal:** Add a secure, durable course workspace that lets a student choose a native folder or
upload a browser directory, see every discovered source, approve exact SHA-256 hashes, and make
only those approved hashes eligible for future provider tools. Persist provider credentials only
in Windows Credential Manager or macOS Keychain and restore sessions safely after restart.

**Architecture:** The authenticated loopback Worker remains the only filesystem, approval, and
credential authority. A new `exam_predictor.workspace` package owns deterministic scanning,
immutable manifest revisions, serialized workspace jobs, and the transmission gate. Streamlit
uses typed `WorkerClient` calls and displays relative paths only. The existing LangGraph kernel is
extended with an optional workspace association but does not parse or transmit source content in
this subproject.

**Tech Stack:** Python 3.11/3.12, Pydantic 2, SQLite, FastAPI 0.139.x, HTTPX 0.28.x,
Streamlit 1.49+, `keyring` 25.7.x, `python-multipart` 0.0.32.x, pytest, Ruff.

**Approved specification:**
`docs/superpowers/specs/2026-07-21-secure-course-workspace-design.md`

**Parent specification:** `docs/superpowers/specs/2026-07-17-langgraph-agent-design.md`,
especially Sections 5, 9, 15, 18, and Subproject 2 in Section 19.

## Global Constraints

- Python remains `>=3.11,<3.13`; keep `requirements.txt` and `pyproject.toml` consistent.
- Pin new ranges to `keyring>=25.7,<26`, `python-multipart>=0.0.32,<0.1`, and
  `streamlit>=1.49,<2`. Streamlit 1.49 is the first documented release with directory upload.
- Keep the Worker bound to `127.0.0.1`; every `/v1/*` route remains protected by the random
  per-launch `X-ExamSage-Token`, with authentication before body parsing.
- Native source roots are read-only. No scan, approval, rescan, transmission check, or workspace
  deletion may write, move, rename, or delete anything below a native root.
- Browser fallback writes only below `~/.examsage/workspaces/<workspace-id>/browser-intake/` by
  temporary-file plus atomic rename; an invalid relative path fails closed.
- No source content is provider-eligible before approval. Approval binds one immutable revision to
  exact entry IDs and SHA-256 values. The transmission gate is the only filesystem resolution API
  available to later provider tools.
- Never follow symlinks, junctions, Windows reparse points, archive links, special files, or paths
  that escape the canonical workspace root.
- The 1 GiB limit is aggregate selected source size. Tests use injected limits or sparse metadata;
  they do not allocate a 1 GiB fixture.
- Local course grouping uses path and filename tokens only. No semantic parsing, OCR, provider
  classification, chapter tree, web research, practice generation, or export work enters this plan.
- API keys may cross the authenticated loopback request once. They may exist only in a Worker
  memory session and the OS credential vault; never in SQLite, checkpoints, state, HTTP responses,
  events, logs, exception causes/contexts, artifacts, or Git changes.
- If the vault is unavailable, keep the successfully validated in-memory session and return a safe
  `credential_saved=false` warning. Never fall back to plaintext or environment files.
- Forget API key disconnects the profile and deletes its vault secret, but never deletes a
  workspace. Workspace deletion never deletes a credential or an original native source.
- A workspace linked to `queued`, `running`, `stopping`, or `paused` work is not deletable. This
  treats every resumable/nonterminal run as active authority and prevents orphaned checkpoints.
- All checkpointed values remain JSON-safe. Absolute roots, `Path` objects, API keys, file handles,
  keyring objects, SDK clients, locks, and exceptions stay outside graph state.
- Preserve the legacy build route, explicit `EXAMSAGE_AGENT_V2` feature flag, provider replacement
  lock, Stop/Resume behavior, pause recovery, and all existing tests.
- UI copy added by this plan is English; HTTP/events expose relative paths and stable error codes,
  never raw local paths or provider exceptions.
- Automated tests use fake pickers, fake vaults, fake providers, and temporary files only. Record
  Windows picker/Credential Manager, macOS Finder/Keychain, and browser fallback as separate manual
  checkpoints; unavailable platforms remain explicitly outstanding.
- Every task follows RED -> GREEN -> focused regression -> review -> small commit. Before final
  completion run full pytest, Ruff, compileall, `git diff --check`, and an independent code review.

## File Map

| Path | Responsibility |
|---|---|
| `exam_predictor/workspace/models.py` | Workspace, manifest, approval, scan, job, and API contracts. |
| `exam_predictor/workspace/policy.py` | Supported formats and deterministic security/size limits. |
| `exam_predictor/workspace/filesystem.py` | Root-anchored, link-safe, read-only file handles. |
| `exam_predictor/workspace/archive.py` | Safe ZIP metadata inspection without extraction. |
| `exam_predictor/workspace/scanner.py` | Read-only enumeration, hashing, grouping, and rescan comparison. |
| `exam_predictor/workspace/store.py` | SQLite migrations and immutable workspace repositories. |
| `exam_predictor/workspace/picker.py` | Injectable client for the short-lived native picker helper. |
| `exam_predictor/workspace/picker_helper.py` | Main-thread Windows/macOS folder dialog subprocess. |
| `exam_predictor/workspace/browser_intake.py` | Validated browser-directory snapshot writer. |
| `exam_predictor/workspace/service.py` | Serialized scan jobs, lifecycle rules, recovery, and deletion. |
| `exam_predictor/workspace/transmission.py` | Hash-bound authorization gate for future provider tools. |
| `exam_predictor/runtime/credential_vault.py` | Provider-neutral vault interface and keyring backend. |
| `exam_predictor/runtime/provider_sessions.py` | Connect, restore, enumerate, disconnect, and replacement lock. |
| `exam_predictor/runtime/store.py` | Additive workspace/run association and deletion-conflict queries. |
| `exam_predictor/runtime/coordinator.py` | Workspace-aware run submission and provider-vault lifecycle. |
| `exam_predictor/worker/workspace_routes.py` | Authenticated workspace, manifest, job, and credential routes. |
| `exam_predictor/worker/api.py` | Compose workspace router with existing Worker lifecycle/auth boundary. |
| `exam_predictor/runtime/client.py` | Typed workspace and saved-provider loopback client methods. |
| `exam_predictor/ui/workspace_view.py` | Functional choose/rescan/review/approve/delete workspace panel. |
| `exam_predictor/ui/agent_view.py` | Provider/chat integration with selected workspace. |
| `tests/workspace/*` | Policy, scanner, archive, store, service, picker, intake, and gate tests. |
| `tests/runtime/*` | Vault, session restore, run migration, and coordinator regression tests. |
| `tests/worker/test_workspace_api.py` | Auth, status, filtering, idempotency, and redaction API tests. |
| `tests/ui/test_workspace_view.py` | Manifest counts and action-state UI tests. |
| `tests/test_secure_workspace_acceptance.py` | Complete fake-boundary vertical slice. |

---

### Task 1: Workspace contracts, format policy, and dependency floors

**Files:**
- Modify: `requirements.txt`
- Modify: `pyproject.toml`
- Create: `exam_predictor/workspace/__init__.py`
- Create: `exam_predictor/workspace/models.py`
- Create: `exam_predictor/workspace/policy.py`
- Create: `tests/workspace/__init__.py`
- Create: `tests/workspace/test_models.py`
- Create: `tests/workspace/test_policy.py`

**Interfaces produced:**

```python
class SourceMode(StrEnum):
    NATIVE_FOLDER = "native_folder"
    BROWSER_SNAPSHOT = "browser_snapshot"


class WorkspaceState(StrEnum):
    READY = "ready"
    SCANNING = "scanning"
    APPROVAL_REQUIRED = "approval_required"
    APPROVED = "approved"
    NEEDS_ATTENTION = "needs_attention"
    DELETING = "deleting"
    CLEANUP_PENDING = "cleanup_pending"


class SourceState(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    EXCLUDED = "excluded"
    FAILED = "failed"
    CHANGED = "changed"
    REMOVED = "removed"


class WorkspaceJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScanPolicy(BaseModel, frozen=True):
    policy_version: str = "workspace-v1"
    max_workspace_bytes: int = 1_073_741_824
    max_files: int = 20_000
    max_depth: int = 32
    max_path_chars: int = 1_024
    max_archive_members: int = 10_000
    max_archive_expanded_bytes: int = 2_147_483_648
    max_archive_ratio: float = 200.0
    hash_chunk_bytes: int = 1_048_576


class ManifestEntry(BaseModel, frozen=True):
    entry_id: str
    workspace_id: str
    relative_path: str
    item_kind: str
    format_category: str | None = None
    size_bytes: int = Field(ge=0)
    modified_ns: int | None = None
    device_id: str | None = None
    file_id: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    state: SourceState
    included: bool
    inclusion_reason: str | None = None
    proposed_course_group: str = "unclassified"
    failure_code: str | None = None
    safe_message: str | None = None
    archive_parent_entry_id: str | None = None
    archive_member_path: str | None = None


class ManifestRevision(BaseModel, frozen=True):
    revision_id: str
    workspace_id: str
    parent_revision_id: str | None = None
    scan_job_id: str | None = None
    policy_version: str
    entries: tuple[ManifestEntry, ...]
    created_at: datetime


class ApprovedEntryHash(BaseModel, frozen=True):
    entry_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ApprovalRecord(BaseModel, frozen=True):
    approval_id: str
    workspace_id: str
    revision_id: str
    entries: tuple[ApprovedEntryHash, ...]
    policy_version: str
    approved_at: datetime


class WorkspaceRecord(BaseModel, frozen=True):
    workspace_id: str
    display_name: str
    source_mode: SourceMode
    canonical_root: Path
    root_device: str | None = None
    root_file_id: str | None = None
    state: WorkspaceState
    current_draft_revision_id: str | None = None
    current_approved_revision_id: str | None = None
    created_at: datetime
    updated_at: datetime
    last_scanned_at: datetime | None = None
    last_access_verified_at: datetime | None = None


class WorkspaceSummary(BaseModel, frozen=True):
    workspace_id: str
    display_name: str
    source_mode: SourceMode
    state: WorkspaceState
    counts: dict[SourceState, int]
    updated_at: datetime


class WorkspaceDetail(WorkspaceSummary):
    current_draft_revision_id: str | None = None
    current_approved_revision_id: str | None = None
    created_at: datetime
    last_scanned_at: datetime | None = None
    last_access_verified_at: datetime | None = None


class ScanProgress(BaseModel, frozen=True):
    discovered_count: int = Field(ge=0)
    bytes_hashed: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    current_relative_path: str | None = None


class ScanResult(BaseModel, frozen=True):
    workspace_id: str
    entries: tuple[ManifestEntry, ...]
    discovered_count: int = Field(ge=0)
    bytes_hashed: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    completed_at: datetime


class WorkspaceJob(BaseModel, frozen=True):
    job_id: str
    workspace_id: str
    job_kind: str
    status: WorkspaceJobStatus
    idempotency_key: str
    safe_error_code: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class WorkspaceEvent(BaseModel, frozen=True):
    sequence: int = Field(ge=1)
    job_id: str
    event_type: str
    message: str
    payload: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    created_at: datetime


class ArchiveMember(BaseModel, frozen=True):
    parent_entry_id: str
    display_path: str
    item_kind: str
    size_bytes: int = Field(ge=0)
    compressed_bytes: int = Field(ge=0)
    state: SourceState
    failure_code: str | None = None


class CleanupRecord(BaseModel, frozen=True):
    cleanup_id: str
    workspace_id: str
    owned_relative_path: str
    safe_error_code: str
    attempt_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


class ApprovedSource(BaseModel, frozen=True):
    workspace_id: str
    entry_id: str
    relative_path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    read_token: SecretStr


class ManifestPage(BaseModel, frozen=True):
    items: tuple[ManifestEntry, ...]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=500)
    counts: dict[SourceState, int]


class EntryInclusionRequest(BaseModel, frozen=True):
    revision_id: str
    included: bool
    subtree: bool = False


class ApprovalRequest(BaseModel, frozen=True):
    revision_id: str


class DeleteAllWorkspacesRequest(BaseModel, frozen=True):
    confirmation: Literal["DELETE ALL"]
```

`ManifestEntry`, `ManifestRevision`, `WorkspaceRecord`, `WorkspaceSummary`, `ApprovalRecord`,
`ScanProgress`, `ScanResult`, `WorkspaceJob`, `WorkspaceEvent`, `ManifestPage`, API request models,
`ArchiveMember`, `CleanupRecord`, and `ApprovedSource` are frozen Pydantic models.
`ManifestEntry.relative_path` is normalized to a nonempty POSIX relative path; absolute paths, `..`,
backslashes, NUL, and drive prefixes raise validation errors. `ArchiveMember.display_path` is a
control-character-stripped, length-bounded label and is never used for filesystem resolution.
`WorkspaceRecord` keeps the canonical root and optional stable `(device, inode)` grant identity only
inside the Worker repository, plus created/updated/last-scanned/last-access-verified timestamps.
`WorkspaceDetail` is the separate public DTO and deliberately has no root or local identity field.

- [ ] **Step 1: Write failing model and policy tests**

Create `tests/workspace/test_models.py` with exact boundary assertions:

```python
import pytest
from pydantic import ValidationError

from exam_predictor.workspace.models import ManifestEntry, SourceState


def test_manifest_entry_rejects_paths_that_can_escape_the_grant():
    for unsafe in ("../secret.pdf", "/etc/passwd", "C:/secret.pdf", "a\\b.pdf", "a\x00b"):
        with pytest.raises(ValidationError):
            ManifestEntry(
                entry_id="entry-1",
                workspace_id="workspace-1",
                relative_path=unsafe,
                item_kind="file",
                format_category="pdf",
                size_bytes=4,
                modified_ns=1,
                sha256="0" * 64,
                state=SourceState.PENDING_APPROVAL,
                included=True,
            )


def test_source_state_and_course_group_are_independent():
    entry = ManifestEntry(
        entry_id="entry-1",
        workspace_id="workspace-1",
        relative_path="Week 1/notes.pdf",
        item_kind="file",
        format_category="pdf",
        size_bytes=4,
        modified_ns=1,
        sha256="0" * 64,
        state=SourceState.PENDING_APPROVAL,
        included=True,
        proposed_course_group="unclassified",
    )
    assert entry.state is SourceState.PENDING_APPROVAL
    assert entry.proposed_course_group == "unclassified"
```

Create `tests/workspace/test_policy.py`:

```python
from exam_predictor.workspace.policy import DEFAULT_SCAN_POLICY, classify_format


def test_supported_formats_are_classified_without_reading_content():
    assert classify_format("lecture.PDF") == "pdf"
    assert classify_format("questions.xlsx") == "spreadsheet"
    assert classify_format("scan.tiff") == "image"
    assert classify_format("bundle.zip") == "archive"


def test_audio_video_executables_and_unknown_files_are_visible_exclusions():
    for name in ("lecture.mp4", "recording.mp3", "setup.exe", "blob.unknown"):
        assert classify_format(name) is None


def test_default_policy_is_bounded_and_versioned():
    assert DEFAULT_SCAN_POLICY.max_workspace_bytes == 1_073_741_824
    assert DEFAULT_SCAN_POLICY.hash_chunk_bytes == 1_048_576
    assert DEFAULT_SCAN_POLICY.policy_version == "workspace-v1"
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/workspace/test_models.py tests/workspace/test_policy.py -q
```

Expected: FAIL during collection because `exam_predictor.workspace` does not exist.

- [ ] **Step 2: Add the typed models and deterministic extension map**

Implement the enums and models above. Use these validators and helpers verbatim in behavior:

```python
def normalize_relative_path(value: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise ValueError("relative_path must be a POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
        raise ValueError("relative_path must remain within the workspace")
    normalized = path.as_posix()
    if normalized in {".", ""}:
        raise ValueError("relative_path must identify an item")
    return normalized


def classify_format(filename: str) -> str | None:
    return SUPPORTED_EXTENSIONS.get(Path(filename).suffix.casefold())
```

`SUPPORTED_EXTENSIONS` must cover every category in design Section 8.2 and no audio/video or
executable extension. Update both dependency files with the three exact ranges in Global
Constraints and change the expected project version in `tests/test_basic.py` only when Task 11
performs the 0.5.0 release bump.

Use this exact extension map:

```python
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".doc": "document",
    ".docx": "document",
    ".ppt": "presentation",
    ".pptx": "presentation",
    ".xls": "spreadsheet",
    ".xlsx": "spreadsheet",
    ".csv": "tabular",
    ".tsv": "tabular",
    ".json": "structured_data",
    ".yaml": "structured_data",
    ".yml": "structured_data",
    ".md": "text",
    ".txt": "text",
    ".html": "text",
    ".htm": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
    ".bmp": "image",
    ".tif": "image",
    ".tiff": "image",
    ".zip": "archive",
}
```

Run the focused tests. Expected: PASS.

- [ ] **Step 3: Verify dependency metadata and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/workspace/test_models.py tests/workspace/test_policy.py tests/test_basic.py -q
.\.venv\Scripts\python.exe -m ruff check exam_predictor/workspace tests/workspace
.\.venv\Scripts\python.exe -m compileall -q exam_predictor/workspace
git diff --check
git add requirements.txt pyproject.toml exam_predictor/workspace tests/workspace
git commit -m "feat: define secure workspace contracts"
```

Expected: focused tests pass, Ruff/compileall/diff check are clean, and the commit contains no
source scanner or Worker API behavior.

---

### Task 2: Safe ZIP preview and deterministic read-only scanner

**Files:**
- Create: `exam_predictor/workspace/filesystem.py`
- Create: `exam_predictor/workspace/archive.py`
- Create: `exam_predictor/workspace/scanner.py`
- Create: `tests/workspace/test_filesystem.py`
- Create: `tests/workspace/test_archive.py`
- Create: `tests/workspace/test_scanner.py`

**Interfaces produced:**

```python
class ArchiveInspector:
    def __init__(self, policy: ScanPolicy = DEFAULT_SCAN_POLICY) -> None:
        self._policy = policy

    def inspect(self, archive_stream: BinaryIO, *, parent_entry_id: str) -> Sequence[ArchiveMember]:
        """Return validated member metadata without extracting member content."""


class WorkspaceScanner:
    def __init__(self, policy: ScanPolicy = DEFAULT_SCAN_POLICY) -> None:
        self._policy = policy
    def scan(
        self,
        workspace_id: str,
        root: Path,
        *,
        previous_entries: Sequence[ManifestEntry] = (),
        emit: Callable[[ScanProgress], None] | None = None,
    ) -> ScanResult:
        """Return a deterministic immutable scan result for one canonical root."""


class SecureFileOpener:
    @contextmanager
    def open_regular(
        self, canonical_root: Path, relative_path: PurePosixPath
    ) -> Iterator[BinaryIO]:
        """Open one regular file beneath an anchored root without following links."""


class SecureOpenError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def is_reparse_point(path: Path) -> bool:
    """Return true when Windows file attributes contain FILE_ATTRIBUTE_REPARSE_POINT."""
```

`scan()` canonicalizes `root`, enumerates without following links, emits safe relative paths,
hashes supported regular files in bounded chunks, inspects ZIP metadata, classifies unsupported
items as excluded, and compares previous approved hashes into unchanged/changed/removed states.
Every scanner and transmission read goes through `SecureFileOpener`; ordinary `Path.open()` is not
permitted for native sources.

- [ ] **Step 1: Write failing secure-open tests**

Create `tests/workspace/test_filesystem.py`. On POSIX, test a normal nested file plus root, parent,
and final-component symlink swaps; assert no byte is returned after a swap. Mock the Windows adapter
to test `FILE_FLAG_OPEN_REPARSE_POINT`, reparse-attribute rejection, final-handle canonical path
containment, regular-file mode, and handle closure. The platform-neutral assertion is:

```python
def test_secure_opener_never_reads_a_link_substitution(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside.txt"
    root.mkdir()
    outside.write_bytes(b"outside-secret")
    source = root / "notes.txt"
    source.write_bytes(b"approved")
    source.unlink()
    source.symlink_to(outside)

    with pytest.raises(SecureOpenError) as caught:
        with SecureFileOpener().open_regular(root.resolve(), PurePosixPath("notes.txt")) as handle:
            handle.read()
    assert caught.value.code == "source_link_or_reparse"
```

The test skips real symlink creation only when the platform denies it; the injected Windows adapter
tests still run on every platform.

- [ ] **Step 2: Implement the root-anchored secure opener**

On POSIX, open the canonical root directory once, traverse every directory with
`os.open(part, O_DIRECTORY | O_NOFOLLOW, dir_fd=parent_fd)`, open the final component with
`O_RDONLY | O_NOFOLLOW`, require `stat.S_ISREG(os.fstat(fd).st_mode)`, and close every descriptor in
reverse order. On Windows, call `CreateFileW` with read-only sharing and
`FILE_FLAG_OPEN_REPARSE_POINT`, reject `FILE_ATTRIBUTE_REPARSE_POINT`, require a disk regular file,
obtain the final path from the unopened-for-reading handle, prove canonical containment, then wrap
the handle with `msvcrt.open_osfhandle`. No source bytes are read before those checks pass.

```python
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
root_fd = os.open(canonical_root, os.O_RDONLY | os.O_DIRECTORY)
directory_fds = [root_fd]
file_fd: int | None = None
try:
    for part in relative_path.parts[:-1]:
        directory_fds.append(
            os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fds[-1])
        )
    file_fd = os.open(relative_path.name, flags, dir_fd=directory_fds[-1])
    if not stat.S_ISREG(os.fstat(file_fd).st_mode):
        raise SecureOpenError("source_not_regular")
    source = os.fdopen(file_fd, "rb", buffering=0, closefd=True)
    file_fd = None
    with source:
        yield source
finally:
    if file_fd is not None:
        os.close(file_fd)
    for directory_fd in reversed(directory_fds):
        os.close(directory_fd)
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/workspace/test_filesystem.py -q
```

Expected: PASS, with unsafe substitutions rejected before a read.

- [ ] **Step 3: Write failing archive security tests**

Create in-memory ZIP fixtures with `zipfile.ZipFile`, including `notes/week1.txt`, `../escape.txt`,
`/absolute.txt`, an encrypted-flag metadata entry, a Unix symlink mode, excessive members, excessive
expanded bytes, and excessive ratio. Assert safe members return metadata only and every unsafe
member returns `state=failed` plus one of these stable codes:

```python
ARCHIVE_TRAVERSAL = "archive_traversal"
ARCHIVE_ABSOLUTE_PATH = "archive_absolute_path"
ARCHIVE_LINK = "archive_link"
ARCHIVE_ENCRYPTED = "archive_encrypted"
ARCHIVE_MEMBER_LIMIT = "archive_member_limit"
ARCHIVE_SIZE_LIMIT = "archive_size_limit"
ARCHIVE_RATIO_LIMIT = "archive_ratio_limit"
```

The core assertion must prove inspection does not extract:

```python
with archive_path.open("rb") as archive_stream:
    members = ArchiveInspector(policy).inspect(archive_stream, parent_entry_id="archive-1")
assert {member.relative_path for member in members} >= {"notes/week1.txt", "../escape.txt"}
assert not (tmp_path / "notes").exists()
assert next(member for member in members if member.relative_path == "../escape.txt").failure_code \
    == "archive_traversal"
```

- [ ] **Step 4: Write failing scanner and change-detection tests**

Tests must cover nested/empty folders, Unicode, unsupported formats, symlinks where supported,
injected reparse detection, unreadable/open failure, path/count/depth/aggregate limits, stable
SHA-256, and a file changed between pre/post stat. Use injected hooks instead of timing races:

```python
def test_scanner_marks_a_file_changed_during_hash_as_failed(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("before", encoding="utf-8")

    def mutate_after_first_chunk(path: Path, chunk_index: int) -> None:
        if path == source and chunk_index == 0:
            source.write_text("after", encoding="utf-8")

    result = WorkspaceScanner(after_hash_chunk=mutate_after_first_chunk).scan(
        "workspace-1", tmp_path
    )
    entry = next(item for item in result.entries if item.relative_path == "notes.txt")
    assert entry.state.value == "failed"
    assert entry.failure_code == "source_changed_during_scan"
    assert entry.sha256 is None
```

The production constructor therefore has one test-only-neutral hook:

```python
def __init__(
    self,
    policy: ScanPolicy = DEFAULT_SCAN_POLICY,
    *,
    archive_inspector: ArchiveInspector | None = None,
    after_hash_chunk: Callable[[Path, int], None] | None = None,
    is_reparse_point: Callable[[Path], bool] | None = None,
    ) -> None:
        """Initialize bounded scanner dependencies without touching the source root."""
```

- [ ] **Step 5: Implement ZIP metadata validation**

Normalize archive member paths with `PurePosixPath`; reject absolute/drive/`..` paths before any
other handling. Detect Unix links from `(external_attr >> 16) & 0o170000`, encrypted entries from
`flag_bits & 0x1`, and enforce count/expanded-size/ratio cumulatively. Return immutable
`ArchiveMember` metadata. Never call `ZipFile.extract`, `extractall`, or open member content.
Production scanner code passes a handle obtained from `SecureFileOpener`; `ArchiveInspector` never
resolves or opens a filesystem path itself.

```python
def _member_failure(info: ZipInfo, policy: ScanPolicy) -> str | None:
    raw = info.filename.replace("\\", "/")
    member_path = PurePosixPath(raw)
    if member_path.is_absolute() or re.match(r"^[A-Za-z]:", raw):
        return "archive_absolute_path"
    if ".." in member_path.parts:
        return "archive_traversal"
    unix_kind = (info.external_attr >> 16) & 0o170000
    if unix_kind == stat.S_IFLNK:
        return "archive_link"
    if info.flag_bits & 0x1:
        return "archive_encrypted"
    if len(member_path.parts) > policy.max_depth:
        return "archive_depth_limit"
    if info.file_size / max(info.compress_size, 1) > policy.max_archive_ratio:
        return "archive_ratio_limit"
    return None
```

Run `tests/workspace/test_archive.py`. Expected: PASS.

- [ ] **Step 6: Implement the read-only scanner**

Use `os.scandir()` and `DirEntry.stat(follow_symlinks=False)`. Before hashing, reject
`entry.is_symlink()`, platform reparse points, non-regular modes, depth/path/count violations, and
unsupported formats. Replace the illustrative `path.open()` below with
`SecureFileOpener.open_regular(root, relative_path)` in production; the digest loop is:

```python
before = path.stat(follow_symlinks=False)
digest = hashlib.sha256()
with secure_file_opener.open_regular(root, PurePosixPath(relative_path)) as source:
    for chunk_index, chunk in enumerate(iter(lambda: source.read(policy.hash_chunk_bytes), b"")):
        digest.update(chunk)
        if after_hash_chunk is not None:
            after_hash_chunk(path, chunk_index)
after = path.stat(follow_symlinks=False)
if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
):
    return failed_entry("source_changed_during_scan")
```

Generate stable entry IDs from `uuid5(UUID(workspace_id), relative_path)` using a workspace UUID;
propose a course group only from the first directory component or sanitized filename stem, and use
`unclassified` when the proposal is ambiguous. Sort entries by `relative_path.casefold()` so scans
are deterministic. Preserve previous missing entries as `removed`. Reuse an interrupted scan's
stored digest only when device, inode/file identity, size, and modification nanoseconds all match;
approval and transmission still perform a fresh stronger validation.

Run both Task 2 test files. Expected: PASS.

- [ ] **Step 7: Verify scanner regressions and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/workspace/test_filesystem.py tests/workspace/test_archive.py tests/workspace/test_scanner.py -q
.\.venv\Scripts\python.exe -m ruff check exam_predictor/workspace tests/workspace
.\.venv\Scripts\python.exe -m compileall -q exam_predictor/workspace
git diff --check
git add exam_predictor/workspace/filesystem.py exam_predictor/workspace/archive.py exam_predictor/workspace/scanner.py tests/workspace
git commit -m "feat: scan course folders without following links"
```

Expected: all focused tests pass and no provider, semantic parser, or source mutation appears.

---

### Task 3: Immutable workspace SQLite repository

**Files:**
- Create: `exam_predictor/workspace/store.py`
- Create: `tests/workspace/test_store.py`

**Interfaces produced:**

```python
class WorkspaceStore:
    def __init__(self, database_path: Path) -> None:
        """Open and migrate the workspace database."""

    def close(self) -> None:
        """Close the persistent connection exactly once."""

    def create_workspace(self, workspace: WorkspaceRecord) -> WorkspaceRecord:
        """Insert one workspace and return its persisted representation."""

    def list_workspaces(self) -> Sequence[WorkspaceSummary]:
        """Return summaries ordered by most recent update."""

    def get_workspace(self, workspace_id: str) -> WorkspaceRecord | None:
        """Return the internal record, including its grant root, or None."""

    def source_root(self, workspace_id: str) -> Path:
        """Return the internal canonical root for Worker-only services."""

    def get_manifest_entries(self, workspace_id: str) -> Sequence[ManifestEntry]:
        """Return the current draft entries or an empty sequence."""

    def commit_scan(self, workspace_id: str, result: ScanResult, job_id: str) -> ManifestRevision:
        """Atomically publish an immutable draft and its completion event."""

    def get_manifest(self, workspace_id: str, revision_id: str | None = None) -> ManifestRevision:
        """Return the requested revision or the current draft."""

    def set_inclusion(
        self, workspace_id: str, revision_id: str, entry_ids: Sequence[str], included: bool
    ) -> ManifestRevision:
        """Clone the current draft with exact inclusion changes."""

    def approve(self, workspace_id: str, revision_id: str, policy_version: str) -> ApprovalRecord:
        """Atomically bind selected entries and hashes to the current policy."""

    def get_approval(self, workspace_id: str) -> ApprovalRecord | None:
        """Return the current approval or None."""

    def create_job(self, job: WorkspaceJob, idempotency_key: str) -> WorkspaceJob:
        """Insert a job or return the existing idempotent job."""

    def get_job(self, job_id: str) -> WorkspaceJob:
        """Return one durable job or raise WorkspaceJobNotFoundError."""

    def start_job(self, job_id: str) -> WorkspaceJob:
        """Atomically mark a queued job running and append its started event."""

    def append_progress(self, job_id: str, progress: ScanProgress) -> WorkspaceEvent:
        """Append a bounded relative-path scan progress event."""

    def fail_job(self, job_id: str, safe_error_code: str) -> WorkspaceJob:
        """Atomically fail a job, update workspace state, and append its event."""

    def update_job(self, job: WorkspaceJob, event: WorkspaceEvent) -> None:
        """Persist the job transition and event in one transaction."""

    def list_job_events(self, job_id: str, after_sequence: int = 0) -> Sequence[WorkspaceEvent]:
        """Return ordered events after the exclusive cursor."""

    def mark_entry_changed(self, workspace_id: str, entry_id: str, code: str) -> None:
        """Create a changed draft and set needs-attention state."""

    def record_access_verified(self, workspace_id: str, verified_at: datetime) -> None:
        """Persist the last successful grant-root identity and access verification."""

    def mark_deleting(self, workspace_id: str) -> WorkspaceRecord:
        """Transition a settled workspace to deleting."""

    def delete_workspace_rows(self, workspace_id: str) -> None:
        """Delete only application database rows."""

    def queue_cleanup(self, workspace_id: str, owned_path: Path, code: str) -> None:
        """Persist a retry for one validated app-owned path."""

    def list_cleanup(self) -> Sequence[CleanupRecord]:
        """Return pending cleanup records in creation order."""
```

The schema uses `PRAGMA user_version=1`, WAL, foreign keys, explicit transactions, and tables
`workspaces`, `manifest_revisions`, `manifest_entries`, `approvals`, `workspace_jobs`,
`workspace_events`, and `cleanup_queue`. Manifest entries are keyed by `(revision_id, entry_id)`;
approvals store a canonical JSON array of `{entry_id, sha256}` pairs and never store file content.

Use this exact schema shape; timestamps are UTC ISO-8601 text and booleans are constrained integers:

```sql
CREATE TABLE workspaces (
  workspace_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  source_mode TEXT NOT NULL,
  canonical_root TEXT NOT NULL,
  root_device TEXT,
  root_file_id TEXT,
  state TEXT NOT NULL,
  current_draft_revision_id TEXT,
  current_approved_revision_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_scanned_at TEXT,
  last_access_verified_at TEXT
);
CREATE TABLE manifest_revisions (
  revision_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
  parent_revision_id TEXT,
  scan_job_id TEXT,
  policy_version TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE manifest_entries (
  revision_id TEXT NOT NULL REFERENCES manifest_revisions(revision_id) ON DELETE CASCADE,
  entry_id TEXT NOT NULL,
  relative_path TEXT NOT NULL,
  item_kind TEXT NOT NULL,
  format_category TEXT,
  size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
  modified_ns INTEGER,
  device_id TEXT,
  file_id TEXT,
  sha256 TEXT,
  state TEXT NOT NULL,
  included INTEGER NOT NULL CHECK(included IN (0, 1)),
  inclusion_reason TEXT,
  proposed_course_group TEXT NOT NULL,
  failure_code TEXT,
  safe_message TEXT,
  archive_parent_entry_id TEXT,
  archive_member_path TEXT,
  PRIMARY KEY(revision_id, entry_id)
);
CREATE TABLE approvals (
  approval_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
  revision_id TEXT NOT NULL REFERENCES manifest_revisions(revision_id) ON DELETE CASCADE,
  approved_entries_json TEXT NOT NULL,
  policy_version TEXT NOT NULL,
  approved_at TEXT NOT NULL
);
CREATE TABLE workspace_jobs (
  job_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
  job_kind TEXT NOT NULL,
  status TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  safe_error_code TEXT,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  UNIQUE(workspace_id, job_kind, idempotency_key)
);
CREATE TABLE workspace_events (
  sequence INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL REFERENCES workspace_jobs(job_id) ON DELETE CASCADE,
  event_type TEXT NOT NULL,
  message TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE cleanup_queue (
  cleanup_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  owned_relative_path TEXT NOT NULL,
  safe_error_code TEXT NOT NULL,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX idx_manifest_revisions_workspace_created
  ON manifest_revisions(workspace_id, created_at);
CREATE INDEX idx_manifest_entries_revision_state
  ON manifest_entries(revision_id, state, relative_path);
CREATE INDEX idx_workspace_events_job_sequence
  ON workspace_events(job_id, sequence);
```

`cleanup_queue.owned_relative_path` is always relative to the configured ExamSage data directory;
the canonical native source root is never accepted in that column.

- [ ] **Step 1: Write failing repository tests**

Tests must prove:

```python
def test_approval_is_atomic_and_rejects_a_stale_revision(store, scanned_workspace):
    first = store.get_manifest(scanned_workspace.workspace_id)
    second = store.set_inclusion(
        scanned_workspace.workspace_id, first.revision_id, [first.entries[0].entry_id], False
    )
    with pytest.raises(StaleManifestError):
        store.approve(scanned_workspace.workspace_id, first.revision_id, "workspace-v1")
    approval = store.approve(scanned_workspace.workspace_id, second.revision_id, "workspace-v1")
    assert approval.revision_id == second.revision_id
    assert all(item.entry_id != first.entries[0].entry_id for item in approval.entries)
```

Also cover immutable revisions, subtree-expanded entry sets, added/changed/removed rows, duplicate
idempotency keys returning the existing job, event sequence monotonicity, explicit `close()`,
transaction rollback, cleanup persistence, restart recovery, and safe missing-workspace errors.

Run `tests/workspace/test_store.py`. Expected: FAIL because the store does not exist.

- [ ] **Step 2: Create the schema and explicit transaction boundary**

Use one connection guarded by `threading.RLock`, `check_same_thread=False`, `row_factory=sqlite3.Row`,
and this transaction pattern:

```python
@contextmanager
def _transaction(self) -> Iterator[sqlite3.Connection]:
    with self._lock:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            yield self._connection
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()
```

Use foreign-key cascades only for ExamSage database rows. Never put filesystem deletion in a
database trigger or transaction.

- [ ] **Step 3: Implement immutable revisions, approval, jobs, and cleanup**

`commit_scan()` inserts a new revision and entries, updates `current_draft_revision_id`, and sets
workspace state to `approval_required` in one transaction. `set_inclusion()` clones the current
revision with only the requested inclusion changes. `approve()` verifies the requested revision is
still current, every included entry is hashable and in `pending_approval`/`approved`, inserts the
canonical approved set, and changes workspace plus selected entries to approved atomically.
`update_job()` writes the lifecycle state and its corresponding durable event in the same immediate
transaction so the UI cannot observe a state/event split.

Use stable domain errors:

```python
class WorkspaceNotFoundError(LookupError):
    pass


class ManifestNotFoundError(LookupError):
    pass


class WorkspaceJobNotFoundError(LookupError):
    pass


class StaleManifestError(RuntimeError):
    pass


class ActiveWorkspaceOperationError(RuntimeError):
    pass


class InvalidApprovalError(ValueError):
    pass
```

Run `tests/workspace/test_store.py`. Expected: PASS.

- [ ] **Step 4: Verify repository regressions and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/workspace/test_store.py tests/runtime/test_store.py -q
.\.venv\Scripts\python.exe -m ruff check exam_predictor/workspace/store.py tests/workspace/test_store.py
.\.venv\Scripts\python.exe -m compileall -q exam_predictor/workspace/store.py
git diff --check
git add exam_predictor/workspace/store.py tests/workspace/test_store.py
git commit -m "feat: persist immutable workspace manifests"
```

Expected: workspace and Agent runtime repositories both pass; stored JSON contains relative paths
and hashes but no file bytes or credentials.

---

### Task 4: Native picker boundary and browser intake snapshot

**Files:**
- Create: `exam_predictor/workspace/picker.py`
- Create: `exam_predictor/workspace/picker_helper.py`
- Create: `exam_predictor/workspace/browser_intake.py`
- Create: `tests/workspace/test_picker.py`
- Create: `tests/workspace/test_browser_intake.py`

**Interfaces produced:**

```python
class FolderPicker(Protocol):
    def choose_folder(self) -> Path | None:
        """Return the selected folder or None when the user cancels."""


class SubprocessFolderPicker:
    def __init__(self, python_executable: Path, timeout_seconds: float = 300.0) -> None:
        self._python_executable = python_executable
        self._timeout_seconds = timeout_seconds

    def choose_folder(self) -> Path | None:
        """Run the picker helper with captured stdio and validate its JSON response."""


class FolderPickerError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class BrowserIntakeWriter:
    def __init__(self, workspaces_root: Path, policy: ScanPolicy = DEFAULT_SCAN_POLICY) -> None:
        self._workspaces_root = workspaces_root
        self._policy = policy

    def create_snapshot(
        self, workspace_id: str, files: Sequence[BrowserUpload]
    ) -> Path:
        """Create and atomically publish one validated browser-intake snapshot."""


class BrowserIntakeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)
```

`BrowserUpload` contains a validated relative POSIX path, declared byte count, and a binary stream.
The path is never inferred from a client absolute path.

```python
@dataclass(frozen=True)
class BrowserUpload:
    relative_path: str
    size_bytes: int
    stream: BinaryIO
```

- [ ] **Step 1: Write failing picker boundary tests**

Inject `subprocess.run` and assert the command contains only the Python executable plus
`-m exam_predictor.workspace.picker_helper`; the selected path appears only in captured stdout,
not argv, error text, or logs:

```python
def test_picker_keeps_selected_path_out_of_argv(monkeypatch, tmp_path):
    selected = tmp_path / "Private Course"
    selected.mkdir()
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        return CompletedProcess(command, 0, stdout=json.dumps(str(selected)), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = SubprocessFolderPicker(Path(sys.executable)).choose_folder()
    assert result == selected.resolve()
    assert str(selected) not in " ".join(observed["command"])
```

Also test cancel (`null`), nonzero helper exit, malformed/non-string JSON, missing/non-directory paths,
timeout, and a safe `FolderPickerError` whose text omits stdout/stderr/path.

- [ ] **Step 2: Write failing browser snapshot tests**

Test nested Unicode files, streamed writes, duplicate paths, `..`/absolute/drive/backslash/NUL paths,
declared/actual size mismatch, aggregate limit, partial write cleanup, and existing snapshot conflict.
The essential assertions are:

```python
root = writer.create_snapshot(
    "workspace-1",
    [BrowserUpload(relative_path="Week 1/notes.txt", size_bytes=5, stream=BytesIO(b"hello"))],
)
assert root == workspaces_root / "workspace-1" / "browser-intake"
assert (root / "Week 1" / "notes.txt").read_bytes() == b"hello"
assert not list((workspaces_root / "workspace-1").glob(".browser-intake-*.tmp"))
```

- [ ] **Step 3: Implement the short-lived helper and picker client**

`picker_helper.main()` creates a hidden Tk root, calls `filedialog.askdirectory(mustexist=True)`,
writes exactly one JSON string (or JSON `null` on cancel) to stdout, and always destroys the root.
JSON keeps newline or Unicode path characters unambiguous while stdout remains a private captured
pipe. The helper runs only under `if __name__ == "__main__":`. `SubprocessFolderPicker` uses
`shell=False`, no inherited stdin, captured text stdout/stderr, `check=False`, and a timeout. It
canonicalizes the returned JSON string and requires `is_dir()`; errors use stable codes and never
include helper output.

```python
def main() -> None:
    root = tkinter.Tk()
    root.withdraw()
    try:
        selected = filedialog.askdirectory(mustexist=True) or None
        sys.stdout.write(json.dumps(selected, ensure_ascii=False))
        sys.stdout.flush()
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
```

Run `tests/workspace/test_picker.py`. Expected: PASS without opening a GUI.

- [ ] **Step 4: Implement atomic browser intake**

Validate all paths and declared sizes before writing. Resolve every parent under a unique temporary
snapshot root, reject duplicate case-folded paths, stream with `hash_chunk_bytes`, enforce actual
and aggregate bytes, `flush()` and `os.fsync()` each file, then rename the complete temporary root
to `browser-intake`. On any error, delete only the validated temporary root below the workspace's
ExamSage-owned directory.

```python
written = 0
with target.open("xb") as destination:
    while chunk := upload.stream.read(self._policy.hash_chunk_bytes):
        written += len(chunk)
        aggregate += len(chunk)
        if written > upload.size_bytes or aggregate > self._policy.max_workspace_bytes:
            raise BrowserIntakeError("browser_intake_size_limit")
        destination.write(chunk)
    destination.flush()
    os.fsync(destination.fileno())
if written != upload.size_bytes:
    raise BrowserIntakeError("browser_intake_size_mismatch")
```

Run `tests/workspace/test_browser_intake.py`. Expected: PASS.

- [ ] **Step 5: Verify intake boundaries and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/workspace/test_picker.py tests/workspace/test_browser_intake.py -q
.\.venv\Scripts\python.exe -m ruff check exam_predictor/workspace tests/workspace
.\.venv\Scripts\python.exe -m compileall -q exam_predictor/workspace
git diff --check
git add exam_predictor/workspace/picker.py exam_predictor/workspace/picker_helper.py exam_predictor/workspace/browser_intake.py tests/workspace
git commit -m "feat: add secure course folder intake"
```

Expected: tests pass with no GUI, native source write, browser path escape, or logged absolute path.

---

### Task 5: Serialized workspace service, recovery, and deletion

**Files:**
- Create: `exam_predictor/workspace/service.py`
- Create: `tests/workspace/test_service.py`

**Interfaces produced:**

```python
class WorkspaceRunGuard(Protocol):
    def has_unsettled_runs(self, workspace_id: str) -> bool:
        """Return true for queued, running, stopping, or paused linked work."""

    def delete_settled_workspace_runs(self, workspace_id: str) -> None:
        """Delete only settled run/checkpoint metadata owned by ExamSage."""


class WorkspaceService:
    def start(self) -> None:
        """Recover jobs and cleanup records, then start one serialized job thread."""

    def shutdown(self, timeout_seconds: float = 5.0) -> None:
        """Request shutdown, join the job thread, and close owned resources."""

    def select_folder(self, idempotency_key: str) -> WorkspaceJob | None:
        """Open the injected picker; return None on cancel or enqueue a scan."""

    def create_browser_snapshot(
        self, display_name: str, files: Sequence[BrowserUpload], idempotency_key: str
    ) -> WorkspaceJob:
        """Create an app-owned snapshot and enqueue its initial scan."""

    def rescan(self, workspace_id: str, idempotency_key: str) -> WorkspaceJob:
        """Return an identical active job or enqueue one new scan."""

    def set_inclusion(
        self,
        workspace_id: str,
        revision_id: str,
        entry_id: str,
        included: bool,
        subtree: bool = False,
    ) -> ManifestRevision:
        """Create a draft after expanding an optional subtree on the requested revision."""

    def approve(self, workspace_id: str, revision_id: str) -> ApprovalRecord:
        """Revalidate included files and atomically approve the exact draft."""

    def delete_workspace(self, workspace_id: str) -> None:
        """Delete ExamSage-owned state only after the run guard settles."""

    def delete_all_workspaces(self) -> None:
        """Apply the same safe deletion rule to every workspace."""


class WorkspaceOperationError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)
```

- [ ] **Step 1: Write failing service lifecycle tests**

Use an injected fake picker, real temporary roots/store, deterministic scanner, and fake run guard.
Cover cancel, initial scan progress, one unreadable file, idempotent rescan, duplicate active mutation,
stale approval, selected-file mutation immediately before approval, access-revoked root, restart of a
running scan, partial cleanup retry, and deletion blocking for every nonterminal run state.

The deletion proof must snapshot native bytes before and after:

```python
before = {path.relative_to(native_root): path.read_bytes() for path in native_root.rglob("*") if path.is_file()}
service.delete_workspace(workspace_id)
after = {path.relative_to(native_root): path.read_bytes() for path in native_root.rglob("*") if path.is_file()}
assert after == before
assert native_root.exists()
assert store.get_workspace(workspace_id) is None
```

- [ ] **Step 2: Implement one serialized workspace job loop**

Use `queue.Queue[str]`, one daemon thread, a stop event, and `WorkspaceStore` durable job rows. A
successful enqueue atomically records `queued`; the worker transitions `running`, emits bounded
`scan_progress`, commits the draft plus `succeeded`, and emits `approval_required`. Safe failures
transition the job to `failed` and workspace to `needs_attention`. Never keep a SQLite transaction
open while traversing or hashing files.

On `start()`, convert persisted `running` jobs to `queued` and enqueue them once. Resume pending
cleanup before new scans. One workspace may have only one queued/running scan or approval mutation.
Folder selection records the canonical root identity before creating the workspace; every successful
scan/rescan refreshes `last_access_verified_at`, while a missing/mismatched root moves the workspace
to `needs_attention` without silently adopting the replacement.

```python
def _loop(self) -> None:
    while True:
        job_id = self._jobs.get()
        if job_id is None:
            return
        job = self._store.get_job(job_id)
        try:
            self._store.start_job(job_id)
            result = self._scanner.scan(
                job.workspace_id,
                self._store.source_root(job.workspace_id),
                previous_entries=self._store.get_manifest_entries(job.workspace_id),
                emit=lambda progress: self._store.append_progress(job_id, progress),
            )
            self._store.commit_scan(job.workspace_id, result, job_id)
        except WorkspaceOperationError as exc:
            self._store.fail_job(job_id, exc.code)
        except Exception:
            self._store.fail_job(job_id, "transient_local_io")
```

- [ ] **Step 3: Implement approval revalidation and safe deletion**

Before `store.approve()`, re-stat and re-hash every included draft file through the scanner's bounded
validation helper. Any mismatch creates a new changed draft and raises `StaleManifestError` without
partial approval. Browser snapshots use the same validation.

Deletion order is: check run guard -> mark deleting -> delete settled linked run/checkpoint rows ->
delete only app-owned workspace directory -> delete workspace rows. If app-owned deletion fails,
persist `cleanup_pending` and retry on next start. Native roots never enter the filesystem-deletion
function. `delete_all_workspaces()` stops at conflicts and returns the conflicting safe workspace IDs.

Run `tests/workspace/test_service.py`. Expected: PASS.

- [ ] **Step 4: Verify service and store regressions and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/workspace/test_service.py tests/workspace/test_store.py tests/runtime/test_coordinator.py -q
.\.venv\Scripts\python.exe -m ruff check exam_predictor/workspace/service.py tests/workspace/test_service.py
.\.venv\Scripts\python.exe -m compileall -q exam_predictor/workspace/service.py
git diff --check
git add exam_predictor/workspace/service.py tests/workspace/test_service.py
git commit -m "feat: manage durable workspace lifecycle"
```

Expected: restart/idempotency/deletion tests pass and no original native file is changed.

---

### Task 6: Hash-bound transmission gate

**Files:**
- Create: `exam_predictor/workspace/transmission.py`
- Create: `tests/workspace/test_transmission.py`

**Interface produced:**

```python
class WorkspaceTransmissionGate:
    def __init__(
        self,
        store: WorkspaceStore,
        policy: ScanPolicy = DEFAULT_SCAN_POLICY,
        token_ttl_seconds: float = 60.0,
    ) -> None:
        self._store = store
        self._policy = policy
        self._token_ttl_seconds = token_ttl_seconds

    def authorize(
        self, workspace_id: str, entry_ids: Sequence[str]
    ) -> Sequence[ApprovedSource]:
        """Resolve only exact approved hashes after current path and file revalidation."""


class SourceAuthorizationError(RuntimeError):
    def __init__(self, code: str, workspace_id: str, entry_id: str) -> None:
        self.code = code
        self.workspace_id = workspace_id
        self.entry_id = entry_id
        super().__init__(code)
```

`ApprovedSource` contains only `workspace_id`, `entry_id`, safe `relative_path`, `size_bytes`,
`sha256`, and an opaque gate-owned `read_token`. It does not serialize or expose an absolute path.
A separate bounded `open_approved(read_token)` context manager resolves the token in Worker memory,
opens the file with no-follow semantics where supported, checks identity once more, and expires the
token after one use or 60 seconds.

- [ ] **Step 1: Write failing authorization tests**

Test pre-approval rejection, excluded/failed/unrequested entries, stale policy version, missing root,
root substitution, symlink/reparse substitution, containment escape, size/mtime/identity/hash change,
all-or-nothing multi-entry authorization, single-use/expiry tokens, and zero provider invocations.

```python
provider_calls = 0
source.write_text("changed", encoding="utf-8")
with pytest.raises(SourceAuthorizationError) as caught:
    gate.authorize(workspace_id, [entry_id])
assert caught.value.code == "approved_source_changed"
assert provider_calls == 0
assert store.get_workspace(workspace_id).state.value == "needs_attention"
```

- [ ] **Step 2: Implement canonical containment and current-hash checks**

Resolve the stored grant internally, require `candidate.relative_to(canonical_root)` to succeed,
walk each path component with `lstat`, reject link/reparse/special modes, and hash in bounded chunks.
Compare current identity, size, and digest with the approved record. If any requested entry fails,
persist a changed/needs-attention event for that entry and return no descriptors for the batch.

```python
def _contained_candidate(
    root: Path, relative_path: str, workspace_id: str, entry_id: str
) -> Path:
    canonical_root = root.resolve(strict=True)
    candidate = (canonical_root / PurePosixPath(relative_path)).resolve(strict=True)
    try:
        candidate.relative_to(canonical_root)
    except ValueError:
        raise SourceAuthorizationError("source_outside_workspace", workspace_id, entry_id) from None
    current = canonical_root
    for part in PurePosixPath(relative_path).parts:
        current = current / part
        mode = current.lstat().st_mode
        if is_reparse_point(current) or stat.S_ISLNK(mode) or not (
            stat.S_ISDIR(mode) or stat.S_ISREG(mode)
        ):
            raise SourceAuthorizationError(
                "source_link_or_special_file", workspace_id, entry_id
            ) from None
    return candidate
```

Keep read tokens in a lock-protected in-memory dictionary. Their representation contains only a
random token, entry ID, expiry, and expected metadata; it never enters a checkpoint or SQLite.
Both authorization hashing and `open_approved()` read through the Task 2 `SecureFileOpener`; the
gate never calls ordinary `Path.open()` on a native source.
Validate the entire requested batch before inserting any read token. If one entry fails, discard all
temporary validation results so a caller cannot retain a token for a partially authorized batch.
After a valid batch, call `record_access_verified()`; a root identity mismatch is an authorization
failure, not an opportunity to overwrite the stored folder grant.
Raise `SourceAuthorizationError` with only a stable code, workspace ID, and entry ID. Task 8 adds a
dedicated coordinator catch that pauses a dependent run and emits a user-action event before the
generic failure handler, guaranteeing a source change cannot be reported as a completed provider
operation.

- [ ] **Step 3: Verify security regressions and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/workspace/test_transmission.py tests/workspace/test_scanner.py tests/workspace/test_store.py -q
.\.venv\Scripts\python.exe -m ruff check exam_predictor/workspace/transmission.py tests/workspace/test_transmission.py
.\.venv\Scripts\python.exe -m compileall -q exam_predictor/workspace/transmission.py
git diff --check
git add exam_predictor/workspace/transmission.py tests/workspace/test_transmission.py
git commit -m "feat: enforce approved source transmission gate"
```

Expected: every bypass attempt fails before a provider boundary and valid tokens are bounded,
single-use, and secret/path safe.

---

### Task 7: OS credential vault and provider-session restoration

**Files:**
- Create: `exam_predictor/runtime/credential_vault.py`
- Modify: `exam_predictor/runtime/models.py`
- Modify: `exam_predictor/runtime/store.py`
- Modify: `exam_predictor/runtime/provider_sessions.py`
- Modify: `exam_predictor/runtime/coordinator.py`
- Create: `tests/runtime/test_credential_vault.py`
- Modify: `tests/runtime/test_store.py`
- Modify: `tests/runtime/test_provider_sessions.py`
- Modify: `tests/runtime/test_coordinator.py`

**Interfaces produced:**

```python
class CredentialVault(Protocol):
    def save(self, profile_id: str, api_key: str) -> None:
        """Persist one provider secret outside application files."""

    def load(self, profile_id: str) -> str | None:
        """Return the secret or None without logging it."""

    def exists(self, profile_id: str) -> bool:
        """Report whether a credential exists."""

    def delete(self, profile_id: str) -> None:
        """Remove the vault secret idempotently."""


class KeyringBackend(Protocol):
    def set_password(self, service: str, account: str, password: str) -> None:
        """Store one password."""

    def get_password(self, service: str, account: str) -> str | None:
        """Load one password."""

    def delete_password(self, service: str, account: str) -> None:
        """Delete one password."""


class KeyringCredentialVault:
    SERVICE_NAME = "ExamSage"

    def __init__(self, backend: KeyringBackend | None = None) -> None:
        self._backend = backend or cast(KeyringBackend, keyring)
```

`ProviderDescriptor` gains `credential_saved: bool = False` and `credential_warning: str | None`.
`RuntimeStore` adds a `saved_provider_profiles` table and exact methods
`save_provider_profile(profile: SavedProviderProfile)`, `list_saved_provider_profiles()`, and
`mark_provider_reconnect_required(profile_id)`. The model contains non-secret
profile/model/capability metadata and `credential_expected`, never a key. `ProviderSessionRegistry`
gains `disconnect()`, `list_profiles()`, and `restore(profile, api_key)` while retaining the existing
running-profile replacement guard.

The additive table is:

```sql
CREATE TABLE IF NOT EXISTS saved_provider_profiles (
  profile_id TEXT PRIMARY KEY,
  profile_json TEXT NOT NULL,
  capabilities_json TEXT NOT NULL,
  credential_expected INTEGER NOT NULL CHECK(credential_expected IN (0, 1)),
  reconnect_required INTEGER NOT NULL CHECK(reconnect_required IN (0, 1)),
  updated_at TEXT NOT NULL
);
```

`profile_json` is the canonical `ProviderProfile.model_dump_json()` and `capabilities_json` is a
canonical `dict[str, bool]`; both are validated back through Pydantic on read. A custom `base_url`
remains non-secret configuration and must pass the existing safe provider URL validation. Request
headers, access tokens, and query credentials are never persisted.

```python
class SavedProviderProfile(BaseModel):
    profile: ProviderProfile
    capabilities: dict[str, bool]
    credential_expected: bool
    reconnect_required: bool
    updated_at: datetime
```

- [ ] **Step 1: Write failing vault and secret-absence tests**

Fake the keyring backend and cover save/load/exists/delete, missing secret, backend unavailable,
backend error redaction, replace, restore, and forget. Use a sentinel and inspect every serializable
surface:

```python
sentinel = "vault-secret-" + secrets.token_hex(16)
vault.save("primary", sentinel)
coordinator.connect_provider(request_with(sentinel))
serialized = "\n".join(
    [database_path.read_bytes().decode("latin1"), repr(events), response.model_dump_json(), caplog.text]
)
assert sentinel not in serialized
```

Also walk `exception.__cause__` and `exception.__context__` and assert the sentinel is absent.

- [ ] **Step 2: Implement keyring adapter with no plaintext fallback**

Call `keyring.set_password/get_password/delete_password` under service `ExamSage` and account
`provider:<profile_id>`. Convert `NoKeyringError`, `InitError`, and backend exceptions into a
secret-free `VaultUnavailableError("Secure credential storage is unavailable")` raised after the
backend `except` block so neither `__cause__` nor `__context__` retains a provider exception.
Validate profile IDs against `^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$` before keyring access.

```python
def save(self, profile_id: str, api_key: str) -> None:
    account = self._account(profile_id)
    unavailable = False
    try:
        self._backend.set_password(self.SERVICE_NAME, account, api_key)
    except Exception:
        unavailable = True
    if unavailable:
        raise VaultUnavailableError("Secure credential storage is unavailable")


def load(self, profile_id: str) -> str | None:
    unavailable = False
    value: str | None = None
    try:
        value = self._backend.get_password(self.SERVICE_NAME, self._account(profile_id))
    except Exception:
        unavailable = True
    if unavailable:
        raise VaultUnavailableError("Secure credential storage is unavailable")
    return value
```

- [ ] **Step 3: Integrate connect, startup restore, and forget**

Connection order is validate provider -> create SDK client -> register in-memory session -> save
vault secret -> save non-secret profile. If vault save fails, return the connected descriptor with
`credential_saved=false` and the stable warning; do not disconnect the usable session. On startup,
load each configured profile and restore it only when a key exists; otherwise emit reconnect state.
Persist `credential_expected=false` and `reconnect_required=true` after a vault-save failure so a
restart never claims that a secret should exist.

`forget_provider_credential(profile_id)` first checks the existing running/stopping replacement lock,
then disconnects, deletes the vault entry idempotently, and marks the profile reconnect-required.
It does not touch workspaces or settled run history.

- [ ] **Step 4: Verify vault/provider regressions and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/runtime/test_credential_vault.py tests/runtime/test_store.py tests/runtime/test_provider_sessions.py tests/runtime/test_coordinator.py -q
.\.venv\Scripts\python.exe -m ruff check exam_predictor/runtime tests/runtime
.\.venv\Scripts\python.exe -m compileall -q exam_predictor/runtime
git diff --check
git add exam_predictor/runtime/credential_vault.py exam_predictor/runtime/models.py exam_predictor/runtime/store.py exam_predictor/runtime/provider_sessions.py exam_predictor/runtime/coordinator.py tests/runtime
git commit -m "feat: persist provider credentials in os vault"
```

Expected: fake-vault restore and forget pass, vault failure preserves only the memory session, and
the secret sentinel is absent from every inspected application surface.

---

### Task 8: Workspace-aware Agent runs and additive runtime migration

**Files:**
- Modify: `exam_predictor/runtime/models.py`
- Modify: `exam_predictor/runtime/store.py`
- Modify: `exam_predictor/runtime/coordinator.py`
- Modify: `tests/runtime/test_models.py`
- Modify: `tests/runtime/test_store.py`
- Modify: `tests/runtime/test_coordinator.py`

**Contract changes:**

```python
class SubmitMessageRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    provider_profile_id: str = Field(min_length=1, max_length=64)
    workspace_id: str | None = Field(default=None, min_length=32, max_length=36)
    message: str = Field(min_length=1, max_length=20_000)


class RunSnapshot(BaseModel):
    run_id: str
    thread_id: str
    provider_profile_id: str
    workspace_id: str | None = None
    message: str
    status: RunStatus
    error: str | None = None
    created_at: datetime
    updated_at: datetime
```

An associated run uses the server-derived checkpoint thread ID `workspace:<workspace_id>`. This
guarantees one workspace owns its checkpoint thread and makes deletion exact. Unassociated legacy
Agent chat preserves its supplied `thread_id` and behavior.

- [ ] **Step 1: Write failing additive-migration and run-guard tests**

Create a pre-migration `agent_runs` database using the current schema, open it with the new store,
and assert existing rows load with `workspace_id is None`. Create associated runs and assert:

```python
run = store.create_run(
    thread_id="workspace:8d6f8d1f9ed34b3f9228dcd3cb6290c4",
    provider_profile_id="primary",
    workspace_id="8d6f8d1f9ed34b3f9228dcd3cb6290c4",
    message="Review my sources",
    status=RunStatus.PAUSED,
)
assert store.has_unsettled_runs(run.workspace_id) is True
assert store.thread_ids_for_workspace(run.workspace_id) == (run.thread_id,)
```

Test that queued/running/stopping/paused are unsettled, completed/failed are settled, a workspace
thread cannot be associated with another workspace, and deleting settled workspace runs cascades
their events but leaves unassociated runs intact.

- [ ] **Step 2: Add the nullable column and indexes without destructive migration**

During `_initialize()`, inspect `PRAGMA table_info(agent_runs)`. If `workspace_id` is absent, execute
`ALTER TABLE agent_runs ADD COLUMN workspace_id TEXT`. Then create indexes on
`(workspace_id, status, created_at)` and `(workspace_id, thread_id)`. Do not rebuild, drop, or copy
the existing table. Update `_run()` and `create_run()` to read/write the optional field.

```python
columns = {str(row["name"]) for row in db.execute("PRAGMA table_info(agent_runs)")}
if "workspace_id" not in columns:
    db.execute("ALTER TABLE agent_runs ADD COLUMN workspace_id TEXT")
db.execute(
    "CREATE INDEX IF NOT EXISTS idx_agent_runs_workspace_status "
    "ON agent_runs(workspace_id, status, created_at)"
)
db.execute(
    "CREATE INDEX IF NOT EXISTS idx_agent_runs_workspace_thread "
    "ON agent_runs(workspace_id, thread_id)"
)
```

Add exact store methods:

```python
def has_unsettled_runs(self, workspace_id: str) -> bool:
    statuses = {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.STOPPING, RunStatus.PAUSED}
    return bool(self.list_for_workspace(workspace_id, statuses=statuses))


def thread_ids_for_workspace(self, workspace_id: str) -> Sequence[str]:
    """Return distinct checkpoint thread IDs in stable order."""


def delete_settled_workspace_runs(self, workspace_id: str) -> None:
    """Raise on an unsettled run, otherwise delete linked run/event rows atomically."""
```

- [ ] **Step 3: Associate coordinator submission and checkpoint cleanup**

Extend
`submit_message(thread_id: str, provider_profile_id: str, message: str, workspace_id: str | None = None)`.
When present, require the workspace
repository to find the workspace, derive `thread_id = f"workspace:{workspace_id}"`, and store the
association. Add coordinator methods implementing `WorkspaceRunGuard`. For settled deletion, open
`SqliteSaver.from_conn_string(checkpoints_path)`, call `delete_thread()` for each workspace-owned
thread, then delete linked run rows. If checkpoint deletion fails, keep run metadata and raise a
safe cleanup error so workspace deletion becomes `cleanup_pending`.

Include `workspace_id` in `_initial_state()` as a JSON-safe string but do not include root paths or
manifest content. In `_execute()`, catch `SourceAuthorizationError` before the generic exception
handler, set the run to `paused`, append a safe `source_changed` event, discard the active control,
and start no provider call. Resume re-enters the durable checkpoint and rechecks the gate after the
student rescans and reapproves.

```python
except SourceAuthorizationError as exc:
    with self._lock:
        self.store.set_status_and_append_event(
            run_id,
            RunStatus.PAUSED,
            EventType.PAUSED,
            "source_changed",
            "A course source changed. Rescan and approve it before resuming.",
            payload={"code": exc.code, "entry_id": exc.entry_id},
        )
        self.controls.discard(run_id)
    return
```

- [ ] **Step 4: Verify migration/kernel regressions and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/runtime/test_models.py tests/runtime/test_store.py tests/runtime/test_coordinator.py tests/graphs/test_kernel_graph.py -q
.\.venv\Scripts\python.exe -m ruff check exam_predictor/runtime tests/runtime
.\.venv\Scripts\python.exe -m compileall -q exam_predictor/runtime
git diff --check
git add exam_predictor/runtime/models.py exam_predictor/runtime/store.py exam_predictor/runtime/coordinator.py tests/runtime
git commit -m "feat: associate agent runs with workspaces"
```

Expected: old databases migrate in place, old unassociated runs still work, and workspace deletion
can identify every linked run/checkpoint without scanning arbitrary threads.

---

### Task 9: Authenticated Worker workspace API and typed client

**Files:**
- Create: `exam_predictor/worker/workspace_routes.py`
- Modify: `exam_predictor/worker/api.py`
- Modify: `exam_predictor/worker/main.py`
- Modify: `exam_predictor/runtime/client.py`
- Create: `tests/worker/test_workspace_api.py`
- Modify: `tests/worker/test_api.py`
- Modify: `tests/runtime/test_client.py`

**Routes:**

```text
POST   /v1/workspaces/select-folder
POST   /v1/workspaces/browser-snapshot
GET    /v1/workspaces
GET    /v1/workspaces/{workspace_id}
GET    /v1/workspaces/{workspace_id}/manifest?state=&course=&offset=&limit=
POST   /v1/workspaces/{workspace_id}/rescan
POST   /v1/workspaces/{workspace_id}/approval
PATCH  /v1/workspaces/{workspace_id}/entries/{entry_id}
DELETE /v1/workspaces/{workspace_id}
DELETE /v1/workspaces
GET    /v1/workspace-jobs/{job_id}
GET    /v1/workspace-jobs/{job_id}/events?after=
GET    /v1/providers/saved
DELETE /v1/providers/{profile_id}/credential
```

All success bodies use Pydantic response models from `workspace.models`. Public workspace models
contain `workspace_id`, display name, source mode, state, counts/revisions/timestamps, but not
canonical root, snapshot path, vault account, or raw failure exception.

- [ ] **Step 1: Write failing auth-before-body and safe-error API tests**

Compose the app with fake workspace service/vault/provider sessions. For every new POST/PATCH/DELETE
route, send a large malformed body without the token and assert 401 before multipart/JSON parsing.
Then cover picker cancel 204, select/rescan 202, idempotency header reuse, list/detail/manifest filters
and pagination, approval 200, stale 409, active-operation 409, invalid path/body 422, missing 404,
delete-all confirmation 422, saved providers, forget conflict, and absolute-path/secret redaction.

```python
response = client.post(
    "/v1/workspaces/browser-snapshot",
    content=b"not-a-valid-multipart-body" * 100_000,
    headers={"content-type": "multipart/form-data; boundary=broken"},
)
assert response.status_code == 401
assert workspace_service.calls == []
```

- [ ] **Step 2: Build a focused router and preserve the outer auth middleware**

`build_workspace_router(dependencies)` returns `APIRouter(prefix="/v1")` with no separate secret
handling.
`create_worker_app()` constructs `WorkspaceStore` at `data_dir/workspace.sqlite3`, a
`WorkspaceService`, `KeyringCredentialVault`, and the existing runtime coordinator, then includes
the router. Its lifespan starts both coordinators, restores saved providers, and shuts down service,
runtime, and stores in reverse order. Injected dependencies bypass real keyring/GUI in tests.

Keep `_V1TokenAuthBoundary` wrapping the fully composed app so authentication occurs before FastAPI
body parsing. Map domain errors centrally to stable codes such as `workspace_not_found`,
`stale_manifest`, `active_workspace_operation`, `vault_unavailable`, and `invalid_workspace_input`.

```python
@router.post("/workspaces/select-folder", status_code=status.HTTP_202_ACCEPTED)
def select_folder(
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> WorkspaceJob | Response:
    job = dependencies.workspace_service.select_folder(idempotency_key)
    return job if job is not None else Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/workspaces/{workspace_id}/approval")
def approve_workspace(workspace_id: str, request: ApprovalRequest) -> ApprovalRecord:
    return dependencies.workspace_service.approve(workspace_id, request.revision_id)
```

- [ ] **Step 3: Implement multipart browser fallback and job reads**

Accept `display_name` and `idempotency_key` form fields plus repeated `files`. Use each UploadFile's
directory-relative `filename`, declared size when available, and underlying stream; never call
`await file.read()` without a bound. `BrowserIntakeWriter` performs the authoritative streamed size
and path validation. Always close UploadFiles in `finally`.

Return 202 with `WorkspaceJob` for scans, 204 for picker cancel, and 200 for settled mutations.
Limit manifest `limit` to `1..500`; return `ManifestPage(items,total,offset,limit,counts)`.
For an entry PATCH with `subtree=true`, the Worker expands the selected entry's relative-path prefix
against the complete requested revision inside `WorkspaceService`; it never relies on the UI's
currently paginated rows.

- [ ] **Step 4: Add exact typed `WorkerClient` methods**

Add `list_workspaces`, `get_workspace`, `get_manifest`, `select_folder`, `upload_directory`, `rescan`,
`set_entry_inclusion`, `approve_workspace`, `delete_workspace`, `delete_all_workspaces`, `get_job`,
`workspace_events_after`, `list_saved_providers`, and `forget_provider_credential`. Reuse `_request()`
so localhost validation, auth token, secret redaction, timeouts, and client closure stay centralized.
Multipart file handles are opened with `ExitStack` and closed on both success and failure.

- [ ] **Step 5: Verify Worker/client regressions and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/worker/test_workspace_api.py tests/worker/test_api.py tests/runtime/test_client.py -q
.\.venv\Scripts\python.exe -m ruff check exam_predictor/worker exam_predictor/runtime/client.py tests/worker tests/runtime/test_client.py
.\.venv\Scripts\python.exe -m compileall -q exam_predictor/worker exam_predictor/runtime/client.py
git diff --check
git add exam_predictor/worker exam_predictor/runtime/client.py tests/worker tests/runtime/test_client.py
git commit -m "feat: expose secure workspace worker api"
```

Expected: new and legacy routes pass, unauthenticated multipart never reaches parsing, and no API
response contains an absolute root or API key.

---

### Task 10: Functional Streamlit workspace and credential controls

**Files:**
- Create: `exam_predictor/ui/workspace_view.py`
- Modify: `exam_predictor/ui/agent_view.py`
- Create: `tests/ui/test_workspace_view.py`
- Modify: `tests/ui/test_agent_view.py`
- Modify: `tests/test_app_smoke.py`

**Interfaces produced:**

```python
@dataclass(frozen=True)
class WorkspaceActionState:
    can_rescan: bool
    can_edit_inclusion: bool
    can_approve: bool
    can_delete: bool
    reason: str | None = None


def manifest_counts(entries: Sequence[ManifestEntry]) -> dict[SourceState, int]:
    """Return every SourceState with a visible zero-or-greater count."""


def action_state(workspace: WorkspaceDetail, manifest: ManifestPage) -> WorkspaceActionState:
    """Derive button enablement from durable Worker state only."""


def render_workspace_panel(client: WorkerClient) -> str | None:
    """Render workspace controls and return the selected workspace ID."""
```

- [ ] **Step 1: Write failing pure UI-state tests**

Test counts include all states, changed/failed/removed reasons stay visible, short hashes never become
approval identifiers, scanning disables approval/edit/delete, stale drafts disable approval, approved
state remains rescan-able, and delete-all requires a second explicit confirmation.

```python
state = action_state(scanning_workspace, manifest_page)
assert state.can_rescan is False
assert state.can_edit_inclusion is False
assert state.can_approve is False
assert state.can_delete is False
assert "scan" in state.reason.lower()
```

- [ ] **Step 2: Implement the native-first workspace panel**

At the top of the Agent view, render `Choose course folder`. On click, call `select_folder()` and
poll the returned job with bounded reruns; never use an unbounded spinner. Add a development fallback
expander using:

```python
uploaded_files = st.file_uploader(
    "Upload a course directory",
    accept_multiple_files="directory",
    max_upload_size=1024,
    key="workspace_directory_upload",
)
```

Send the relative `UploadedFile.name` values through `upload_directory()`. Show workspace selector,
state, rescan, counts, state/course filters, and a manifest dataframe containing relative path,
category, bytes, modified time, state, first 12 hash characters, group, and reason. Never render the
canonical root.

- [ ] **Step 3: Implement review, approval, deletion, and credential actions**

Provide per-row inclusion toggles plus a subtree action based on the visible relative-path prefix.
Send the current revision ID on every edit/approval. Disable approval when no included hashable file
exists or a scan/stale/attention state is active. Poll jobs using event sequence cursors and show
discovered count, bytes hashed, failures, and next action.

Require the user to type the workspace display name before delete, and type `DELETE ALL` before
delete-all. Render saved provider status and `Forget API key`; warn that forgetting disconnects the
provider but leaves course files/workspaces intact. Preserve the existing provider-connect and chat
flow. Submit chat with the selected `workspace_id`; the server derives its workspace thread.

- [ ] **Step 4: Verify UI/legacy smoke regressions and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ui/test_workspace_view.py tests/ui/test_agent_view.py tests/test_app_smoke.py -q
.\.venv\Scripts\python.exe -m ruff check exam_predictor/ui tests/ui
.\.venv\Scripts\python.exe -m compileall -q exam_predictor/ui
git diff --check
git add exam_predictor/ui/workspace_view.py exam_predictor/ui/agent_view.py tests/ui tests/test_app_smoke.py
git commit -m "feat: add course workspace controls to agent ui"
```

Expected: the English functional UI exposes every manifest state, all action enablement follows
Worker state, and the legacy feature-flag-disabled app smoke test remains green.

---

### Task 11: Secure-workspace acceptance, documentation, and 0.5.0 gate

**Files:**
- Create: `tests/test_secure_workspace_acceptance.py`
- Modify: `README.md`
- Modify: `PRIVACY.md`
- Modify: `SECURITY.md`
- Modify: `exam_predictor/__init__.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_basic.py`
- Create: `docs/manual-tests/2026-07-21-secure-workspace-checkpoints.md`

- [ ] **Step 1: Write the complete failing vertical-slice acceptance test**

Use an injected picker, fake vault, fake provider factory/call counter, authenticated FastAPI client,
real temporary SQLite files, and a real mixed temporary source folder. Execute exactly:

1. choose the mixed folder;
2. wait for scan completion and assert every discovered item has a visible state/reason;
3. exclude one supported source;
4. approve remaining exact hashes;
5. assert only approved entries pass `WorkspaceTransmissionGate`;
6. mutate one approved source and assert the gate blocks before the fake provider counter changes;
7. stop/recreate the Worker dependencies and restore workspace, approval, and fake-vault session;
8. delete the workspace and assert every original source byte matches its initial snapshot.

The test must also scan database bytes, checkpoint bytes, events, HTTP JSON, logs, exceptions,
artifacts, and `git diff` for a unique fake secret sentinel and assert absence.

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_secure_workspace_acceptance.py -q
```

Expected before final integration: at least one precise failure identifies a missing composition or
recovery edge; fix only that behavior with a focused test, then rerun until PASS.

- [ ] **Step 2: Update user, privacy, and security documentation**

Document the one-key startup, native folder picker, browser fallback, visible manifest states,
approval boundary, rescan/change behavior, 1 GiB aggregate limit, OS vault, Forget API key,
workspace deletion semantics, supported/deferred formats, and troubleshooting for moved folders,
vault unavailable, cleanup pending, and stale approval. State plainly that approved files will be
sent to the configured provider only when a later user task invokes a provider tool; Subproject 2
itself performs no cloud source analysis.

Update the threat model with canonical containment, symlink/reparse rejection, hash binding,
single-use gate tokens, auth-before-body, no plaintext fallback, and the remaining provider-side
retention responsibility.

- [ ] **Step 3: Record manual platform checkpoints honestly**

Create a table with rows `Windows native picker`, `Windows Credential Manager`, `macOS Finder
picker`, `macOS Keychain`, and `Browser directory fallback`; columns are date, platform/build,
result (`passed`, `failed`, or `outstanding`), and evidence. Mark only checks actually performed.
Do not convert an unavailable Mac or browser into a pass based on mocks.

- [ ] **Step 4: Bump exactly the public version surfaces to 0.5.0**

Change `exam_predictor.__version__`, `pyproject.toml [project].version`, and the matching
`tests/test_basic.py` assertion from `0.4.0` to `0.5.0`. Do not mass-replace historical docs or
fixtures.

- [ ] **Step 5: Run the complete automated acceptance gate**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall -q app.py exam_predictor scripts tests
git diff --check
git status --short
```

Expected: all tests pass with no real provider/network/keyring/GUI call; Ruff, compileall, and
whitespace checks are clean; status lists only intended Subproject 2 files.

- [ ] **Step 6: Request independent review and address findings**

Use `superpowers:requesting-code-review` against the approved spec and this plan. The reviewer must
check all security invariants, secret absence, source immutability, transaction/event atomicity,
restart/deletion behavior, auth-before-body, legacy regressions, and scope exclusions. Apply Critical
or Important fixes with `superpowers:receiving-code-review`, add a reproducing test first, and rerun
the complete gate.

- [ ] **Step 7: Commit the accepted subproject**

```powershell
git add README.md PRIVACY.md SECURITY.md exam_predictor/__init__.py pyproject.toml tests/test_basic.py tests/test_secure_workspace_acceptance.py docs/manual-tests/2026-07-21-secure-workspace-checkpoints.md
git diff --cached --check
git diff --cached --name-status
git commit -m "feat: complete secure course workspace"
```

Expected: a reviewable 0.5.0 commit, fresh full-suite evidence, no Critical/Important review finding,
and manual platform results that distinguish passed checks from outstanding checks.

## Final Acceptance Checklist

- [ ] Every discovered native or browser item has a visible reasoned manifest state.
- [ ] Zero source content is provider-eligible before exact revision/hash approval.
- [ ] New, changed, moved, removed, substituted, or link-like sources fail closed before provider use.
- [ ] Native source bytes are unchanged after scan, approval, rescan, transmission check, and delete.
- [ ] Browser snapshots and other deletion targets are proven below ExamSage-owned workspace roots.
- [ ] Credentials are absent from SQLite, checkpoints, events, HTTP, logs, exceptions, artifacts,
      diagnostics, and Git changes; vault unavailability has no plaintext fallback.
- [ ] Fake-vault restart restores a provider session; Forget key leaves workspaces intact.
- [ ] Cancellation, access loss, stale approval, restart, conflict, and partial cleanup settle into
      actionable durable states rather than unbounded UI spinners.
- [ ] Existing Agent-kernel Stop/Resume, provider replacement, auth, launcher, UI, and legacy tests pass.
- [ ] Full pytest, Ruff, compileall, `git diff --check`, and independent review are clean.
- [ ] Windows/macOS/browser manual checks are recorded as passed, failed, or outstanding with evidence.

## Specification Coverage Map

| Approved design section | Implemented and verified by |
|---|---|
| 1-3 Purpose, scope, product decisions | Global Constraints; Tasks 1-11; deferred-work exclusions |
| 4 Architecture and responsibility boundaries | File Map; Tasks 3-10 |
| 5 Local data layout | Tasks 3-5, 7-9, 11 |
| 6 Domain model and immutable revisions | Tasks 1 and 3 |
| 7 Native folder selection | Task 4; Task 11 manual checkpoints |
| 8 Scanning, hashing, limits, ZIP preview | Tasks 1-2; Task 11 acceptance |
| 9 Approval and transmission gate | Tasks 3, 5-6, and Task 8 pause integration |
| 10 Multi-course folders | Task 2 local path-only grouping; Task 10 filters |
| 11 Credential vault | Task 7; Tasks 9-11 integration and verification |
| 12 Worker API | Task 9 |
| 13 Functional user interface | Task 10 |
| 14 Lifecycle, concurrency, and recovery | Tasks 3, 5, 7-9 |
| 15 Deletion and retention | Tasks 5, 8-11 |
| 16 Typed/redacted errors | Tasks 2-10; Task 11 sentinel audit |
| 17 Testing strategy | Focused tests in every task; Task 11 vertical slice |
| 18 Acceptance gate | Task 11 complete gate and independent review |
| 19 Security invariants | Global Constraints; Tasks 2, 4, 6-9, 11 |
| 20 Migration and compatibility | Tasks 7-11; additive runtime migration and full legacy regression |

## Official References Checked for This Plan

- Streamlit 1.49.0 release notes document directory upload support for `st.file_uploader` and
  `st.chat_input`: https://docs.streamlit.io/develop/quick-reference/release-notes/2025
- Current `st.file_uploader` documentation defines `accept_multiple_files="directory"` and
  `max_upload_size`: https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
- `keyring` 25.7.0 provides the native Windows/macOS credential backend abstraction used here:
  https://pypi.org/project/keyring/25.7.0/
- `python-multipart` 0.0.32 provides FastAPI multipart parsing for the browser fallback:
  https://pypi.org/project/python-multipart/0.0.32/
