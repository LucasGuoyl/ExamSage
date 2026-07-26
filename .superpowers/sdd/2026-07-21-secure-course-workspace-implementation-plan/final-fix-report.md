# Final fix report: secure workspace integration gaps

## Status

All ten final-review findings are fixed on base `2a74490`. Each fix was driven by a focused failing
test before the production change. No provider source processing, OCR, semantic analysis, embedding,
grounded research, or report-generation work was added; those capabilities remain deferred to the
later provider-tool subproject.

## RED / GREEN evidence

### 1. Persist Worker-owned workspace authority on message submission

RED:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/worker/test_api.py::test_authenticated_message_submission_persists_server_workspace_authority -q
```

The stored run had `workspace_id=None`. GREEN adds `workspace_id` to the authenticated body, validates
it through `SubmitMessageRequest`, and passes it to `RuntimeCoordinator.submit_message`; the focused
Worker/client contract tests passed.

### 2. Compose the production identity-bound owned-tree remover

RED:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/worker/test_workspace_api.py::test_default_worker_composition_deletes_only_its_browser_snapshot -q
```

Default composition returned `cleanup_pending` because no remover was installed. GREEN wires
`OwnedTreeRemover` into `create_worker_app`. It validates `workspaces/...` claims, anchors every path
component without following links/reparse points, compares the recorded directory identity, rejects
special/link-like children, and removes only the held tree. Default-composition deletion and
restart/retry recovery both passed.

### 3. Make approval authority atomic with token publication and reads

RED tests deterministically excluded a source after filesystem validation but before publication and
also tried a token after its revision was excluded; both initially failed with `DID NOT RAISE`.

GREEN adds one atomic authority snapshot plus `hold_transmission_authority`, binding every grant to its
approval and revision. Final publication and each open recheck exact current authority. The bounded read
holds the store transaction/lock through the immutable spool context, so a concurrent consent mutation
waits until the read closes. The complete transmission module passed all 38 tests.

### 4. Reject lexical picker links before canonicalization

RED used a Windows directory junction from the real subprocess picker into `WorkspaceService`; the
selection was accepted and a workspace was created. GREEN retains the lexical absolute path, checks
every component for symlink/reparse metadata before canonicalization, and anchors the final directory
identity. The picker suite plus picker-to-service boundary test passed all 16 tests.

### 5. Do not charge preserved exclusions to the selected-byte budget

RED proved a preserved `user_excluded` file consumed the aggregate selected-byte budget and an excluded
ZIP failed before inspection. GREEN determines preserved exclusion before either aggregate cap check,
still hashes/inspects the source for safety and visibility, and reserves zero selected bytes. Both
ordinary-file and archive regressions passed.

### 6. Derive approval enablement from included rows only

RED mixed one safe included source with excluded `changed`, `failed`, and `removed` rows; the UI disabled
approval. GREEN blocks only included rows that are unhashable or not in an approval-eligible state,
while keeping every attention row visible. All five action-state tests passed.

### 7. Give normal file rows authoritative parent-subtree semantics

RED used a real scan with sibling files and no structural folder rows; `subtree_authority` returned
`None`. GREEN lets a file carry authority for its parent prefix (or workspace root), while retaining a
real folder authority when present. The service expands the prefix against the complete stored revision
and ignores archive previews/ineligible rows. The real scanner-to-UI-helper-to-service test proved both
siblings changed and an unrelated directory did not; existing folder/panel cases also passed.

### 8. Remove the nonexistent `RuntimeStore.close` lifecycle contract

RED exercised uninjected production lifespan startup/shutdown and failed with
`AttributeError: 'RuntimeStore' object has no attribute 'close'`. GREEN removes that invented contract;
the per-operation runtime store needs no close call. Production lifespan and cleanup-precedence tests
passed all four focused cases.

### 9. Preserve caller-owned upload stream state

RED left caller-owned cursors at EOF after success, HTTP failure, and transport failure, and raised a raw
`AttributeError` for a non-seekable object. GREEN publishes an explicit read/seek/tell stream contract,
rewinds only for transport, restores the original cursor through `ExitStack` on every outcome, never
closes caller streams, and rejects non-seekable input before a request. Eight focused upload ownership
cases passed; the Streamlit upload protocol now includes `tell`.

### 10. Remove provider-shaped test credentials and add a redacting static audit

The first RED showed the audit entry point was absent. After adding it, a second RED reported only:

```text
openai_api_key: tests/graphs/test_kernel_graph.py:18
```

It did not echo the matched value. GREEN replaces the provider-shaped fake with a neutral credential
sentinel. `scripts/check_secret_patterns.py` scans tracked/untracked nonignored repository files and
staged/unstaged added diff lines for common provider, SCM, cloud, Slack, and private-key shapes. Reports
contain only rule ID, redacted path, line, and source. Audit/redaction plus kernel graph tests passed.

## Deferred-work ledger

The handoff remains explicit and unchanged:

- `README.md` says the secure-workspace release catalogs, hashes, approves, and gates sources but does
  not connect them to cloud OCR, parsing, embeddings, grounded research, or report generation.
- `PRIVACY.md` says provider transmission occurs only when a later provider tool needs an approved file.
- The approved design lists semantic/multimodal understanding and provider-powered classification under
  **Deferred**, with Subproject 3 consuming the transmission gate later.

This wave added no provider call, OCR/parser integration, semantic extraction, or durable analysis state.

## Regression and static review

The combined focused sweep covered runtime client, workspace UI, both Worker API suites, picker,
scanner, service, store, transmission, kernel graph, and the secret audit. It reached 100% with two
platform/link skips and no failures. `ruff check .` and `git diff --check` also exited 0 before the final
gate.

Whole-diff review confirmed:

- native source deletion remains metadata-only;
- browser deletion stays inside the identity-bound application-owned tree and preserves retry state on
  uncertainty;
- grants remain process-local, single-use, short-lived, and now approval/revision-bound;
- source content and credentials are not introduced into SQLite, checkpoints, events, or diagnostics;
- subtree expansion is server-side over the complete revision, not a visible UI page;
- caller-owned streams are neither closed nor left repositioned.

## Final automated verification

The full suite was run exactly once on the final code tree:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

It exited 0 at 100%: **581 passed, 15 skipped (596 collected) in 27.4 seconds**. The skips are
platform/link-dependent coverage and were not converted into claimed live passes.

The remaining final gates all exited 0:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall -q app.py exam_predictor scripts tests
.\.venv\Scripts\python.exe scripts\check_secret_patterns.py --root .
git diff --check
```

Ruff reported `All checks passed!`; compileall and the whitespace check were clean; the audit reported
that repository files and added diff lines passed.
