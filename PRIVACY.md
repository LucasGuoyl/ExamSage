# ExamSage Privacy Model

Last updated: 26 July 2026

ExamSage is a local application, not a hosted service. The project maintainers do not operate an ExamSage API relay, account database, analytics endpoint, or file server.

## Data flow

The default legacy build keeps its existing direct-upload workflow: local deterministic checks run
first, selected files and embedded Office images are sent directly to the chosen provider, and the
returned normalized course data is stored locally.

The optional Agent workspace has a separate approval boundary:

1. The native picker records a canonical reference to a user-selected folder, or the browser fallback
   creates a streamed snapshot below ExamSage's data directory.
2. Local deterministic code inventories supported paths, hashes regular files, previews ZIP metadata,
   and records a visible manifest state/reason. It does not infer or analyze course content.
3. The user excludes unwanted sources and approves the exact remaining manifest revision and hashes.
4. The transmission gate rechecks the canonical root, file identity, metadata, and hash before granting
   one short-lived read.
5. An approved file is sent to the configured provider only when a later user task invokes a provider
   tool that needs it. The secure-workspace subproject itself performs no cloud source analysis.

## API keys

The legacy build keeps its API key in Streamlit session memory only. The Agent route sends the key once
to the authenticated loopback Worker, which stores it through Windows Credential Manager or macOS
Keychain via the operating-system vault abstraction. There is no plaintext fallback when the vault is
unavailable.

Credentials are excluded from SQLite, LangGraph checkpoints, manifests, normalized files, reports,
events, logs, exceptions, HTTP responses, exports, diagnostics, and backups. `Forget API key`
disconnects the provider and deletes its vault credential; it does not delete course workspaces.

## Agent kernel alpha credentials and local processes

In the hidden Agent route, Streamlit sends the provider API key once to the local Worker over an
authenticated loopback request. The Worker accepts Agent API requests only on `127.0.0.1`, using a
random per-launch token shared with the Streamlit child process through their environment. Authentication
is checked before JSON or multipart bodies are parsed.

Closing ExamSage clears in-process provider clients. On restart, the Worker restores eligible provider
sessions from the OS vault without placing the key in durable application state. The automated
acceptance test injects an in-memory fake vault and fake provider; it never reads a real keyring or sends
a real provider request.

## Provider processing

The selected provider receives only content that the active workflow sends: direct legacy uploads, or
exactly approved Agent workspace files when a later provider tool is invoked. Its privacy, retention,
safety-monitoring, regional-processing, and billing policies apply. OpenAI requests set `store: false`
where supported. Gemini temporary files are deleted after analysis on a best-effort basis;
provider-side operational retention may still apply.

Do not upload data unless you are authorized to share it with the chosen provider. Review the provider's current terms before processing personal data, unreleased exams, research under NDA, medical/legal records, export-controlled material, or institution-restricted content.

## Local storage and backup

By default, application data is stored in `~/.examsage`. Set `EXAMSAGE_DATA_DIR` to choose another
local location. A native workspace stores metadata and approvals but leaves every source byte in the
selected folder unchanged. Deleting it removes only ExamSage-owned metadata, linked settled runs/events,
and checkpoints. A browser fallback stores an ExamSage-owned snapshot and may delete only that
identity-verified tree below the application data root. A deletion that cannot prove those boundaries
remains `cleanup pending`.

Backups exclude transient intake files and known secret-file extensions. The user is responsible for
operating-system accounts, disk encryption, backups, malware protection, and access permissions.

## Telemetry

ExamSage adds no telemetry. Streamlit's usage-stat collection is disabled in `.streamlit/config.toml`.

## Web research

When course evidence is sparse—or when a student asks for fresh sources—ExamSage sends a generated query to the selected provider's native web-search/grounding tool. Returned source URLs become part of the local report or chat.
