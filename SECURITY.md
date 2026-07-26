# Security Policy

## Supported versions

Security fixes target the latest release on the default branch while ExamSage is in alpha.

## Reporting a vulnerability

Do not post an exploitable vulnerability, leaked key, private course file, or personal data in a public issue. Use GitHub's private vulnerability reporting for the repository. Include the affected version, reproduction steps, impact, and a proposed mitigation if available.

## Threat model

ExamSage treats uploads, archive members, document text, webpages, model output, and citations as untrusted.

Controls include:

- extension allow-list and a 1 GiB aggregate workspace cap;
- canonical root containment, stable root/file identity checks, and rejection of symlinks, junctions,
  reparse points, and other link-like substitutions;
- ZIP path-containment checks, link rejection, file-count limit, expanded-size limit, and
  compression-ratio limit;
- immutable manifest revisions and approval bound to the exact policy version, entry IDs, and SHA-256
  hashes;
- transmission-time path, identity, metadata, and hash revalidation followed by short-lived,
  process-local, single-use read tokens;
- rejection of URL credentials, non-HTTPS schemes, localhost names, and literal private/reserved IP addresses;
- prompts that delimit uploaded text as data and explicitly reject embedded instructions;
- prompt-injection detection and visible warnings;
- no shell/code execution from course content;
- no developer-operated proxy or telemetry;
- session-only legacy API keys; Agent credentials use the operating-system vault with no plaintext
  fallback, plus secret redaction in diagnostics;
- direct official SDK connections and `store: false`/best-effort deletion controls;
- user-confirmed cost ceilings before work begins.

## Local Agent Worker threat model

The hidden Agent Worker's HTTP server binds only to `127.0.0.1`. Every `/v1/*` route requires a
cryptographically random, per-launch `X-ExamSage-Token`; `/health` is the only unauthenticated route and
exposes readiness only. An outer ASGI boundary authenticates before FastAPI parses JSON or multipart
bodies. The launcher passes the token through the child-process environment, never as a command-line
argument.

On launcher exit, ExamSage requests pause before terminating the Streamlit and Worker child processes.
That supervisor sequence does not wait for a newly durable checkpoint before termination. Stop requests
are cooperative and are observed at safe graph boundaries; on the next Worker start, unfinished
`running` or `stopping` metadata is recovered to `paused`, and Resume continues from the latest durable
checkpoint only after the provider session is restored from the vault or reconnected.

SQLite checkpoints contain JSON-safe conversation and tool state, but never SDK clients, credentials,
locks, file handles, transmission tokens, canonical source roots, manifests, or exception objects. Run,
event, workspace, and approval databases likewise exclude credentials and raw source content. Provider
profiles contain only non-secret configuration; the Worker restores credentials from Windows Credential
Manager or macOS Keychain and marks the profile reconnect-required if secure loading fails.

Native workspace deletion removes ExamSage metadata only and never traverses or deletes the selected
source folder. Browser snapshot deletion is restricted to an exact workspace-owned path below the
application data root and requires the recorded directory identity to match. Partial or unverifiable
cleanup fails closed as `cleanup pending`.

Anyone with access to the local operating-system account may still read study content in native folders
or ExamSage-owned browser snapshots, so account access controls, permissions, backups, and disk
encryption remain part of the security boundary.

## Important residual risks

- A public hostname can resolve to a private IP after validation. ExamSage currently uses provider-native search rather than locally fetching arbitrary URLs, reducing but not eliminating provider-side URL risks.
- Model providers can retain data for safety, abuse prevention, or account-policy reasons even when application storage is disabled.
- Approval limits what ExamSage may send but cannot control provider-side retention after a later user
  task invokes a provider tool. Users remain responsible for the configured provider's terms and data
  controls.
- Prompt injection is not a solved problem. ExamSage never grants uploaded text tools or credentials, but model output must still be treated as untrusted advice.
- Generated answers can be wrong. Exam ranking and worked solutions require human verification.
- Local malware, browser extensions, compromised Python packages, or an unlocked operating-system account are outside the application boundary.

## Dependency safety

Use supported Python versions, keep dependencies current, review automated dependency updates, and run the CI suite before release. Signed native installers are a roadmap item; current scripts may trigger OS first-run warnings.
