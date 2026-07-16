# Security Policy

## Supported versions

Security fixes target the latest release on the default branch while ExamSage is in alpha.

## Reporting a vulnerability

Do not post an exploitable vulnerability, leaked key, private course file, or personal data in a public issue. Use GitHub's private vulnerability reporting for the repository. Include the affected version, reproduction steps, impact, and a proposed mitigation if available.

## Threat model

ExamSage treats uploads, archive members, document text, webpages, model output, and citations as untrusted.

Controls include:

- extension allow-list and 1 GB course cap;
- ZIP path-containment checks, symlink rejection, file-count limit, expanded-size limit, and compression-ratio limit;
- rejection of URL credentials, non-HTTPS schemes, localhost names, and literal private/reserved IP addresses;
- prompts that delimit uploaded text as data and explicitly reject embedded instructions;
- prompt-injection detection and visible warnings;
- no shell/code execution from course content;
- no developer-operated proxy or telemetry;
- session-only API keys and secret redaction in diagnostics;
- direct official SDK connections and `store: false`/best-effort deletion controls;
- user-confirmed cost ceilings before work begins.

## Important residual risks

- A public hostname can resolve to a private IP after validation. ExamSage currently uses provider-native search rather than locally fetching arbitrary URLs, reducing but not eliminating provider-side URL risks.
- Model providers can retain data for safety, abuse prevention, or account-policy reasons even when application storage is disabled.
- Prompt injection is not a solved problem. ExamSage never grants uploaded text tools or credentials, but model output must still be treated as untrusted advice.
- Generated answers can be wrong. Exam ranking and worked solutions require human verification.
- Local malware, browser extensions, compromised Python packages, or an unlocked operating-system account are outside the application boundary.

## Dependency safety

Use supported Python versions, keep dependencies current, review automated dependency updates, and run the CI suite before release. Signed native installers are a roadmap item; current scripts may trigger OS first-run warnings.
