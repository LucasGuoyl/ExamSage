# Security Policy

## Supported versions

Security fixes target the latest release on the default branch while ExamSage is in alpha.

## Reporting a vulnerability

Do not post an exploitable vulnerability, leaked key, private course file, or personal data in a public issue. Use GitHub private vulnerability reporting. Include the affected version, reproduction steps, impact, and a proposed mitigation if available.

## Threat model

ExamSage treats folder entries, archive members, document text, images, model output, citations, and local state outside its verified ownership boundary as untrusted.

Core controls include:

- supported-extension allow-list and a 1 GiB aggregate workspace cap;
- canonical root containment, stable root/file identity, metadata and hash checks, and rejection of symlinks, junctions, reparse points, and other link-like substitutions;
- ZIP containment, link rejection, depth, member-count, expanded-size, and compression-ratio limits;
- immutable manifest revisions and approval bound to the exact policy version, entry IDs, and SHA-256 hashes;
- transmission-time revalidation followed by short-lived, process-local, single-use reads;
- prompts that delimit course content as data, visible prompt-injection warnings, and no shell/code execution from source content;
- direct official provider SDK connections, explicit request timeouts, bounded application retries, Agent-mode OpenAI `store: false`, and best-effort Gemini remote-file deletion;
- OS-vault Agent credentials with no plaintext fallback and secret redaction in durable or diagnostic surfaces;
- user-approved cost ceilings for routes that expose them, with the provider console remaining authoritative.

## Local Worker and durable state

The Agent Worker binds only to `127.0.0.1`. Every `/v1/*` route requires a cryptographically random per-launch `X-ExamSage-Token`; `/health` exposes readiness only. An outer ASGI boundary authenticates before FastAPI parses JSON or multipart bodies. The launcher passes its token in the child-process environment, never in command arguments.

SQLite checkpoints contain JSON-safe conversation and tool state but never SDK clients, credentials, locks, file handles, transmission tokens, canonical source roots, manifests, or exception objects. Workspace, approval, run, and event stores likewise exclude credentials and raw source content. The separate evidence store intentionally contains validated source-derived evidence and maps; protected prepared parts and JSON artifacts are held under identity-checked ExamSage-owned directories.

Stop is cooperative at durable graph boundaries. Launcher exit requests pause, but does not guarantee a new checkpoint at the exact shutdown instant. Startup recovers unfinished `running` or `stopping` metadata as `paused`; Resume is explicit and continues from the latest durable boundary after the provider session is restored or reconnected.

## Evidence preparation and provider boundaries

Modern OOXML is parsed locally with bounded XML/archive handling. PDFs are divided into bounded page groups. Structured data, images, and safe ZIP members keep stable part identities and coverage. Prepared parts are content-addressed and published only below the evidence-artifact root through an ownership registry; identity or registry disagreement fails closed.

Legacy DOC/PPT/XLS conversion requires LibreOffice plus an injected secure sandbox runner that isolates filesystem and network access, uses a private profile, enforces process/output/deadline bounds, discards subprocess output, and kills the process tree on timeout. Installing LibreOffice alone does not enable conversion. If this boundary is unavailable or conversion output is unexpected, the source fails visibly and is not passed through an unrestricted subprocess.

Provider requests default to a 90-second timeout and the evidence tool to one persisted 60-minute run deadline. Multimodal concurrency defaults to 2. Each provider-profile route permits no more than 3 attempts and structured-output repair no more than once. Retryable errors and `Retry-After` are honored only inside the remaining deadline. Durably published validated source-part results are reused after Stop/restart/Resume. A crash after a provider response but before atomic publication can repeat that unpublished attempt, and unpublished planner or synthesis calls can also repeat; this is not an external exactly-once guarantee.

Provider/model output is schema-validated, size-bounded, citation-linked, and rejected when malformed or unsafe. These controls reduce risk but do not make model output trustworthy.

## Cache invalidation and deletion

Evidence cache keys bind source/part hashes and relevant route/schema/prompt/policy versions. A changed source requires rescan and approval; derived evidence and study-map dependencies for that source are invalidated. Initial maps cite processed evidence only and expose partial coverage. Complete maps are not published while parts remain nonterminal.

Native workspace deletion never traverses the selected source folder. Browser snapshot, prepared-artifact, and abandoned legacy-intake cleanup is limited to registered or fixed-root owned trees and requires recorded identity to match. Active leases, identity changes, links, unknown paths, partial failure, or crash recovery uncertainty fail closed as `cleanup pending`.

## Prohibited surfaces

Credentials, absolute source paths, signed URLs, and raw source text must not appear in run/event metadata, safe errors, logs, exceptions, HTTP diagnostics, benchmark reports, or backups. Durable logical provider-operation receipts contain only safe identifiers and routing metadata. Secret-pattern checks are part of the release gate.

## Important residual risks

- A selected provider can retain data under its policies even when application storage is disabled or temporary files are deleted.
- Prompt injection is not solved. Source text receives no credentials or tools, but model output and citations still require human review.
- Generated evidence, topic structure, ranking, OCR, and answers can be wrong.
- Anyone with access to the local OS account may read native sources, browser snapshots, prepared parts, evidence, and maps.
- Local malware, browser extensions, compromised dependencies, an unlocked account, or a compromised provider account are outside the application boundary.
- Native pickers, keyrings, provider behavior, clean launch, and performance require separate live validation; automated fakes are not evidence for those boundaries.
- Current launchers are not signed installers and can trigger first-run warnings.

## Dependency safety

Use Python 3.11 or 3.12, keep dependencies current, review dependency updates, and run the complete test, Ruff, byte-compilation, secret-pattern, dependency, and Git-whitespace gates before release.
