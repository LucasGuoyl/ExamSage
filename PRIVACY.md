# ExamSage Privacy Model

Last updated: 1 August 2026

ExamSage is a local application, not a hosted service. The maintainers do not operate an ExamSage API relay, account database, analytics endpoint, or file server. ExamSage adds no telemetry, and Streamlit usage reporting is disabled in `.streamlit/config.toml`.

## Agent data flow

1. A native picker records a canonical reference to a selected folder, or the browser fallback streams an ExamSage-owned snapshot below the application data directory.
2. Local deterministic code inventories supported paths, rejects link-like or unsafe entries, previews archive metadata, hashes regular files, and displays the manifest. It does not infer course meaning at this stage.
3. The user excludes unwanted sources and approves the exact included entry IDs, manifest revision, policy version, and hashes.
4. Before every source use, the transmission gate rechecks the canonical root, file identity, metadata, and hash, then grants one short-lived, single-use read.
5. ExamSage locally prepares bounded source parts. Only the exact approved part required by an active evidence operation is sent directly to the selected provider.
6. Validated evidence units, coverage, citations, and initial/complete study-map snapshots are cached locally so unchanged work can be reused and resumed.

No Agent source is eligible for provider transmission before approval. Excluded sources remain local. Approval can limit what ExamSage sends, but it cannot control what a provider retains after receipt.

## Provider processing and retention

The selected provider receives approved prepared parts and prompts needed for the active operation. Its privacy, retention, safety-monitoring, regional-processing, account, and billing policies apply.

- Agent-mode OpenAI requests explicitly set `store: false`; legacy mode and custom endpoints may behave differently.
- Gemini temporary provider files are deleted best-effort after analysis.
- Provider-side operational, security, or abuse-prevention retention may still apply.
- OpenAI-compatible endpoints are experimental; their storage and deletion behavior depends on their operator.

Do not process material unless you are authorized to share it with that provider. Review current provider terms before using personal data, unreleased exams, embargoed research, NDA material, medical or legal records, export-controlled information, or institution-restricted content.

## API keys

The Agent route sends the key once to the authenticated loopback Worker and stores it through Windows Credential Manager or macOS Keychain. There is no plaintext fallback when the OS vault is unavailable. On restart, the Worker may restore an eligible provider session from the vault; it never puts the key in durable application state.

Credentials are excluded from SQLite, LangGraph checkpoints, manifests, evidence, prepared parts, maps, reports, events, logs, exceptions, HTTP responses, exports, benchmark reports, diagnostics, and backups. `Forget API key` disconnects that provider profile and removes its vault credential; it does not delete course workspaces.

The temporary legacy route keeps its key in Streamlit session memory. It is a compatibility/developer fallback, not the recommended evidence-engine path.

## Local evidence, caching, and invalidation

Application data defaults to `~/.examsage`; set `EXAMSAGE_DATA_DIR` to use another local location. ExamSage stores manifests, approvals, run/checkpoint metadata, prepared parts, validated evidence, citations, coverage, and study-map snapshots under its local data boundary. Evidence and map content are derived from course sources and can still be sensitive.

The cache identity includes source and part hashes plus provider route, schema, prompt, and policy versions. Unchanged approved content can be reused. If a source hash changes, ExamSage requires a new scan and approval and invalidates evidence and study-map dependencies derived from that source. It does not silently attach old evidence to changed bytes.

Backups exclude transient intake files and known secret-file extensions, but may include derived study content. The user remains responsible for OS account security, file permissions, disk encryption, malware protection, and backup access.

## Deletion

- Deleting a native workspace removes ExamSage-owned metadata, evidence, prepared artifacts, runs, events, and checkpoints. It never edits or deletes the selected native source folder.
- Deleting a browser fallback workspace may remove only the recorded identity-verified snapshot below the ExamSage data root.
- Unverifiable or partial deletion remains `cleanup pending` and can be retried; ExamSage does not expand the target.
- Old compatibility-route upload copies are removed only through an explicit cleanup control that refuses active, replaced, linked, unknown, or unverified entries.

Deleting local data does not retroactively delete data already processed or retained by a provider. Use the provider's own controls where applicable.

## Web research

The 0.6.0 evidence-engine workspace does not fetch arbitrary webpages. Request-specific grounded web research is a later product stage. The temporary legacy route may use the selected provider's native web-search feature and can store returned source URLs in local reports.

## Automated and live evidence

Automated acceptance uses synthetic material, fake providers, injected clocks, and fake vaults. It does not use a real keyring or contact a real provider. Opt-in live benchmarks use only the declared CC0 synthetic reference course and record sanitized measurements; outstanding platform and provider checks are listed in `docs/manual-tests/`.
