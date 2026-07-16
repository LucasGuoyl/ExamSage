"""Security boundaries for untrusted course uploads and public URLs."""

from __future__ import annotations

import ipaddress
import re
import shutil
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


MAX_COURSE_BYTES = 1 * 1024 * 1024 * 1024
MAX_ARCHIVE_FILES = 10_000
MAX_COMPRESSION_RATIO = 200

ALLOWED_EXTENSIONS = {
    ".pdf", ".ppt", ".pptx", ".doc", ".docx",
    ".xls", ".xlsx", ".csv", ".tsv",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff",
    ".md", ".markdown", ".txt", ".html", ".htm", ".json", ".yaml", ".yml",
    ".zip",
}

PROMPT_INJECTION_PATTERNS = (
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"reveal\s+(the\s+)?(system|developer)\s+prompt",
    r"print\s+(your\s+)?(api|secret)\s*key",
    r"exfiltrat(e|ion)",
    r"send\s+.*\s+to\s+https?://",
    r"do\s+not\s+tell\s+the\s+user",
)


class UploadSecurityError(ValueError):
    """An upload violates a declared safety boundary."""


@dataclass
class SecurityFinding:
    path: str
    severity: str
    message: str


@dataclass
class ValidationResult:
    total_bytes: int
    files: list[Path]
    findings: list[SecurityFinding] = field(default_factory=list)


def safe_filename(name: str) -> str:
    """Return a filesystem-neutral filename without path components."""

    clean = Path(name.replace("\\", "/")).name
    clean = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", clean).strip(" .")
    return clean[:180] or "upload"


def validate_uploads(paths: list[str | Path], max_bytes: int = MAX_COURSE_BYTES) -> ValidationResult:
    files: list[Path] = []
    findings: list[SecurityFinding] = []
    total = 0
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise UploadSecurityError(f"Upload is missing or not a file: {path.name}")
        suffix = path.suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise UploadSecurityError(
                f"Unsupported file type '{suffix or '(none)'}' for {path.name}."
            )
        size = path.stat().st_size
        total += size
        if total > max_bytes:
            raise UploadSecurityError("The course exceeds the 1 GB local workspace limit.")
        if size == 0:
            findings.append(SecurityFinding(str(path), "warning", "The file is empty."))
        files.append(path)
    return ValidationResult(total_bytes=total, files=files, findings=findings)


def scan_prompt_injection(text: str, path: str = "uploaded content") -> list[SecurityFinding]:
    """Flag suspicious instructions; the material is still treated as data only."""

    findings: list[SecurityFinding] = []
    lowered = text.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            findings.append(SecurityFinding(
                path=path,
                severity="warning",
                message=(
                    "Possible prompt-injection text was found. ExamSage will quote it as "
                    "untrusted course data and will not follow its instructions."
                ),
            ))
            break
    return findings


def safe_extract_zip(
    archive: str | Path,
    destination: str | Path,
    *,
    max_total_bytes: int = MAX_COURSE_BYTES,
) -> list[Path]:
    """Extract a ZIP while blocking traversal, symlinks and compression bombs."""

    archive = Path(archive)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    total_uncompressed = 0

    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_ARCHIVE_FILES:
            raise UploadSecurityError("ZIP contains too many files.")
        for info in infos:
            raw_name = info.filename.replace("\\", "/")
            if not raw_name or raw_name.endswith("/"):
                continue
            target = (destination / raw_name).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise UploadSecurityError("ZIP path traversal was blocked.") from exc
            unix_mode = info.external_attr >> 16
            if unix_mode and stat.S_ISLNK(unix_mode):
                raise UploadSecurityError("ZIP symbolic links are not allowed.")
            suffix = target.suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS - {".zip"}:
                # Unknown executables and scripts never leave the archive.
                continue
            total_uncompressed += info.file_size
            if total_uncompressed > max_total_bytes:
                raise UploadSecurityError("ZIP expands beyond the 1 GB course limit.")
            if info.compress_size == 0 and info.file_size > 0:
                raise UploadSecurityError("Suspicious ZIP compression ratio was blocked.")
            if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
                raise UploadSecurityError("Possible ZIP bomb was blocked.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(target)
    return extracted


def validate_public_https_url(url: str) -> str:
    """Reject local/private targets to prevent server-side request forgery."""

    parsed = urlparse(url.strip())
    if parsed.scheme != "https" or not parsed.hostname:
        raise UploadSecurityError("Only public HTTPS URLs are allowed.")
    host = parsed.hostname.strip("[]").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise UploadSecurityError("Local network URLs are not allowed.")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved):
        raise UploadSecurityError("Private or reserved network addresses are not allowed.")
    if parsed.username or parsed.password:
        raise UploadSecurityError("URLs containing credentials are not allowed.")
    return url.strip()


def redact_secrets(value: str) -> str:
    """Best-effort API-key redaction for user-visible diagnostics."""

    value = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED_API_KEY]", value)
    value = re.sub(r"\bAIza[A-Za-z0-9_-]{20,}\b", "[REDACTED_API_KEY]", value)
    return value
