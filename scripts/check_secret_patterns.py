from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("google_api_key", re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{16,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
)


@dataclass(frozen=True, order=True)
class SecretFinding:
    rule_id: str
    path: str
    line_number: int
    source: str


def scan_text(
    text: str,
    *,
    path: str,
    source: str,
    starting_line: int = 1,
) -> tuple[SecretFinding, ...]:
    findings: list[SecretFinding] = []
    for line_number, line in enumerate(text.splitlines(), start=starting_line):
        for rule_id, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(
                    SecretFinding(rule_id, path, line_number, source)
                )
    return tuple(findings)


def scan_added_diff_text(
    diff: str,
    *,
    source: str,
) -> tuple[SecretFinding, ...]:
    findings: list[SecretFinding] = []
    path = "unknown"
    new_line_number = 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:]
            if path.startswith("b/"):
                path = path[2:]
            continue
        if line.startswith("@@ "):
            match = re.search(r"\+(\d+)", line)
            new_line_number = int(match.group(1)) if match else 0
            continue
        if line.startswith("+") and not line.startswith("+++"):
            findings.extend(
                scan_text(
                    line[1:],
                    path=path,
                    source=source,
                    starting_line=new_line_number,
                )
            )
            new_line_number += 1
        elif not line.startswith("-"):
            new_line_number += 1
    return tuple(findings)


def _git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("Secret audit could not inspect repository state.")
    return completed.stdout


def _repository_paths(root: Path) -> Iterable[Path]:
    output = _git(
        root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    )
    for encoded in output.split(b"\0"):
        if encoded:
            yield root / encoded.decode("utf-8", errors="surrogateescape")


def audit_repository(root: Path) -> tuple[SecretFinding, ...]:
    root = root.resolve(strict=True)
    findings: list[SecretFinding] = []
    for candidate in _repository_paths(root):
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
            if candidate.is_symlink() or not resolved.is_file():
                continue
            content = resolved.read_bytes().decode("utf-8", errors="ignore")
        except (OSError, RuntimeError, ValueError):
            continue
        findings.extend(
            scan_text(
                content,
                path=candidate.relative_to(root).as_posix(),
                source="repository",
            )
        )

    for arguments, source in (
        (("diff", "--no-ext-diff", "--no-color", "--unified=0"), "worktree_diff"),
        (
            ("diff", "--cached", "--no-ext-diff", "--no-color", "--unified=0"),
            "staged_diff",
        ),
    ):
        diff = _git(root, *arguments).decode("utf-8", errors="ignore")
        findings.extend(scan_added_diff_text(diff, source=source))
    return tuple(sorted(set(findings)))


def format_findings(findings: Sequence[SecretFinding]) -> str:
    lines = ["Potential secret patterns detected (matched values are redacted):"]
    for finding in findings:
        safe_path = finding.path
        for _rule_id, pattern in SECRET_PATTERNS:
            safe_path = pattern.sub("[REDACTED]", safe_path)
        lines.append(
            f"- {finding.rule_id}: {safe_path}:{finding.line_number} "
            f"({finding.source})"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit repository secret patterns.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    findings = audit_repository(arguments.root)
    if findings:
        print(format_findings(findings))
        return 1
    print("Secret pattern audit passed for repository files and added diff lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
