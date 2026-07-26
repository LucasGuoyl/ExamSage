# Secure workspace manual platform checkpoints

Automated tests use injected picker, vault, browser-upload, and provider boundaries. Those tests do
not count as live platform evidence. No live GUI, operating-system keyring, or browser directory check
was performed for this release gate.

| Checkpoint | Date | Platform/build | Result | Evidence |
|---|---|---|---|---|
| Windows native picker | 2026-07-26 | Windows host; live app build not launched | outstanding | Automated acceptance used an injected fake picker; no native dialog was opened. |
| Windows Credential Manager | 2026-07-26 | Windows host; live keyring not accessed | outstanding | Automated acceptance used an in-memory fake vault; no real credential was stored or restored. |
| macOS Finder picker | 2026-07-26 | macOS unavailable | outstanding | No macOS host or Finder session was available. |
| macOS Keychain | 2026-07-26 | macOS unavailable | outstanding | No macOS host or Keychain session was available. |
| Browser directory fallback | 2026-07-26 | Interactive browser unavailable | outstanding | Automated upload tests do not prove a live browser directory-selection flow. |
