# ExamSage Privacy Model

Last updated: 15 July 2026

ExamSage is a local application, not a hosted service. The project maintainers do not operate an ExamSage API relay, account database, analytics endpoint, or file server.

## Data flow

1. Uploaded files are written to a temporary local intake folder.
2. Local deterministic code validates file type/size and safely extracts supported ZIP members.
3. Files and embedded Office images are sent directly to the AI provider selected by the user.
4. The provider returns extracted/structured content. ExamSage stores a normalized local course workspace, report, and chat history.
5. Transient intake files are deleted after a successful build. Users can delete a course in the UI.

## API keys

The API key is stored in Streamlit session memory only. It is not placed in SQLite, normalized course files, reports, logs, exports, or backups. Closing the app/session requires entering it again.

## Agent kernel alpha credentials and local processes

In the hidden Agent route, Streamlit sends the provider API key once to the local Worker over an
authenticated loopback request. The Worker keeps the key only in its process-memory provider session;
it is excluded from LangGraph state, checkpoints, run and event databases, logs, exceptions, HTTP
responses, and Git artifacts. The Worker accepts Agent API requests only on `127.0.0.1`, using a random
per-launch token shared with the Streamlit child process through their environment.

Closing ExamSage terminates these local child processes and clears the in-memory provider session. Until
an operating-system credential vault is added in a future subproject, restart the launcher, reconnect the
provider, and then select Resume for paused work. The automated acceptance test uses a fake provider and
a test-only credential; it never reads or sends a real provider key.

## Provider processing

The selected provider receives the uploaded data and user prompts. Its privacy, retention, safety-monitoring, regional-processing, and billing policies apply. OpenAI requests set `store: false` where supported. Gemini temporary files are deleted immediately after analysis on a best-effort basis; provider-side operational retention may still apply.

Do not upload data unless you are authorized to share it with the chosen provider. Review the provider's current terms before processing personal data, unreleased exams, research under NDA, medical/legal records, export-controlled material, or institution-restricted content.

## Local storage and backup

By default, data is stored in `~/.examsage`. Set `EXAMSAGE_DATA_DIR` to choose another local location. Backups exclude the transient `intake` directory and known secret-file extensions. The user is responsible for operating-system accounts, disk encryption, backups, malware protection, and access permissions.

## Telemetry

ExamSage adds no telemetry. Streamlit's usage-stat collection is disabled in `.streamlit/config.toml`.

## Web research

When course evidence is sparse—or when a student asks for fresh sources—ExamSage sends a generated query to the selected provider's native web-search/grounding tool. Returned source URLs become part of the local report or chat.
