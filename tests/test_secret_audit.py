from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.check_secret_patterns import format_findings, scan_added_diff_text


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = REPOSITORY_ROOT / "scripts" / "check_secret_patterns.py"


def test_repository_and_added_diff_pass_the_redacting_secret_audit():
    assert AUDIT_SCRIPT.is_file(), "The static secret audit entry point is missing."

    completed = subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), "--root", str(REPOSITORY_ROOT)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_added_diff_diagnostics_never_echo_a_matched_value():
    simulated_secret = "sk-" + "A" * 24
    diff = (
        "+++ b/config.py\n"
        "@@ -0,0 +1 @@\n"
        f"+credential = {simulated_secret}\n"
    )

    findings = scan_added_diff_text(diff, source="test_diff")
    report = format_findings(findings)

    assert len(findings) == 1
    assert findings[0].rule_id == "openai_api_key"
    assert "config.py:1" in report
    assert simulated_secret not in report
