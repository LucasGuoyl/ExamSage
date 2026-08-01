# ExamSage 0.6.0 multimodal evidence checkpoints

Last updated: 1 August 2026

This record separates deterministic automated evidence from live provider and native-platform evidence. Automated fakes are never counted as proof of a real keyring, native picker, browser, network, provider, model, or clean launch.

| Checkpoint | Environment | Status | Evidence / exact outstanding work |
|---|---|---|---|
| Mixed-format select → scan → exclude → approve → analyze → Stop → restart → Resume → change → reapprove → invalidate → delete | Automated; real Worker/SQLite/transmission/preparation/graph/client, fake provider/vault/picker/clocks/external converter | passed (automated, not live) | Synthetic acceptance covers exact approved hashes, partial/final coverage, reuse of already durably published source-part results on the tested Stop/restart path, original-source preservation, and safe deletion. It does not prove provider-side exactly-once behavior, the crash-before-publication window, planner/synthesis replay behavior, a live provider, or OS integration. |
| Windows native folder picker | Windows desktop | outstanding | Select a real synthetic fixture through the native dialog; verify canonical manifest, cancel behavior, rescan, and original-folder preservation. |
| Windows Credential Manager | Windows desktop | outstanding | Connect a disposable provider key, close/relaunch, verify eligible restore, use Forget API key, and verify no plaintext fallback or leaked diagnostic. |
| OpenAI reference benchmark | Live OpenAI; explicit saved model routes; CC0 synthetic fixture | outstanding | Run the opt-in benchmark outside CI and record first activity, initial/final map timing, exact source/part coverage, logical calls, ExamSage-visible retries, safe errors, and model IDs. No live result is claimed. |
| Google Gemini reference benchmark | Live Gemini; explicit saved model routes; CC0 synthetic fixture | outstanding | Run the same benchmark and record the same fields plus best-effort temporary-file cleanup behavior. No live result is claimed. |
| Browser directory fallback | Supported desktop browser + local Worker | outstanding | Force native-picker unavailability, upload the synthetic directory, verify streamed snapshot identity, analysis, deletion, and that the original directory is untouched. |
| macOS Finder folder picker | macOS desktop | outstanding | Select/cancel/rescan the synthetic fixture and verify canonical manifest and original-folder preservation. |
| macOS Keychain | macOS desktop | outstanding | Connect/relaunch/restore/forget a disposable key and verify no plaintext fallback or diagnostic leak. |
| Windows clean launch | Supported clean Windows installation, Python 3.11 or 3.12 | outstanding | Start the Agent route from the documented launcher command with no existing `.venv` or ExamSage state; verify dependency installation, Worker readiness, UI open, Stop, and clean exit. |
| macOS clean launch | Supported clean macOS installation, Python 3.11 or 3.12 | outstanding | Repeat the clean-launch sequence and record Gatekeeper/first-run behavior. |

## Live benchmark conditions

The benchmark is valid only when all of the following hold:

1. `--live`, `--provider-profile`, and `--fixture` are explicit, the Worker environment is authenticated, and CI is not set.
2. The provider profile is connected and its fast, balanced, or reasoning model IDs are explicit saved routes.
3. The fixture declares `ExamSage synthetic reference course` and `CC0-1.0`; no real course or private material is used.
4. The report is written to a new path and reviewed for only safe metadata. Provider SDK-internal retries and billing cost are not inferred.
5. A result is recorded here as live evidence only after a person verifies the environment and report.

## Release interpretation

The automated row is sufficient to exercise the local deterministic architecture and fake external contracts. Every `outstanding` row remains a release limitation for real-platform/provider confidence and must not be reworded as passed until the stated live procedure is completed.
