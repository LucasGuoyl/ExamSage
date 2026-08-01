"""Bounded legacy Office conversion without shell interpolation."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import BinaryIO, Protocol

from pydantic import Field, model_validator

from exam_predictor.evidence.models import EvidenceFrozenModel


_FORMAT_MAP = {
    ".doc": ("docx", ".docx"),
    ".ppt": ("pptx", ".pptx"),
    ".xls": ("xlsx", ".xlsx"),
}
_DEFAULT_MAX_INPUT_BYTES = 64 * 1024 * 1024
_DEFAULT_MAX_OUTPUT_BYTES = 48 * 1024 * 1024
_MAX_DEADLINE_SECONDS = 300.0


class LegacyConversionError(RuntimeError):
    """A safe conversion failure without paths, source bytes, or subprocess output."""

    def __init__(self, code: str) -> None:
        if code not in {"converter_unavailable", "converter_failed"}:
            raise ValueError("unsupported converter error code")
        self.code = code
        super().__init__(code)


class ConvertedDocument(EvidenceFrozenModel):
    """One bounded converted document owned by the caller after return."""

    suffix: str
    content_bytes: bytes = Field(repr=False, exclude=True)
    content_size_bytes: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_content_identity(self) -> ConvertedDocument:
        if self.suffix not in {".docx", ".pptx", ".xlsx"}:
            raise ValueError("converted suffix is unsupported")
        if len(self.content_bytes) != self.content_size_bytes:
            raise ValueError("converted byte length does not match its declaration")
        if sha256(self.content_bytes).hexdigest() != self.content_sha256:
            raise ValueError("converted byte hash does not match its declaration")
        return self


class LegacyOfficeConverter(Protocol):
    def available(self) -> bool:
        """Return whether the fixed converter executable is available."""

    def convert(self, source: BinaryIO, *, suffix: str, deadline: float) -> ConvertedDocument:
        """Convert one authorized stream into one bounded owned document."""


class SandboxedConverterRunner(Protocol):
    """Security boundary for a native converter process tree.

    Implementations must isolate filesystem and network access, use a private
    profile, enforce CPU/memory/output quotas, discard process output, and kill
    the complete process tree when the deadline expires.
    """

    def run(
        self,
        arguments: Sequence[str],
        *,
        cwd: Path,
        deadline: float,
        max_output_bytes: int,
    ) -> int:
        """Run one fixed conversion and return only its exit code."""


class LibreOfficeConverter:
    """Prepare a fixed LibreOffice invocation for an injected secure sandbox."""

    def __init__(
        self,
        executable: Path | None = None,
        *,
        sandbox: SandboxedConverterRunner | None = None,
        max_input_bytes: int = _DEFAULT_MAX_INPUT_BYTES,
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        if max_input_bytes <= 0 or max_output_bytes <= 0:
            raise ValueError("converter bounds must be positive")
        self._configured_executable = executable
        self._sandbox = sandbox
        self._max_input_bytes = max_input_bytes
        self._max_output_bytes = max_output_bytes

    def available(self) -> bool:
        return self._sandbox is not None and self._fixed_executable() is not None

    def convert(self, source: BinaryIO, *, suffix: str, deadline: float) -> ConvertedDocument:
        executable = self._fixed_executable()
        sandbox = self._sandbox
        if executable is None or sandbox is None:
            raise LegacyConversionError("converter_unavailable")
        route = _FORMAT_MAP.get(suffix.casefold())
        if route is None or not isinstance(deadline, (int, float)) or isinstance(deadline, bool):
            raise LegacyConversionError("converter_failed")
        timeout = float(deadline)
        if timeout <= 0 or timeout > _MAX_DEADLINE_SECONDS:
            raise LegacyConversionError("converter_failed")

        conversion_format, output_suffix = route
        try:
            with TemporaryDirectory(prefix="examsage-convert-") as temporary:
                temporary_root = Path(temporary)
                input_path = temporary_root / f"source{suffix.casefold()}"
                output_root = temporary_root / "output"
                profile_root = temporary_root / "profile"
                output_root.mkdir()
                profile_root.mkdir()
                _copy_stream_bounded(source, input_path, self._max_input_bytes)
                arguments: Sequence[str] = (
                    str(executable),
                    "--headless",
                    "--nologo",
                    "--nodefault",
                    "--nolockcheck",
                    "--norestore",
                    "--safe-mode",
                    f"-env:UserInstallation={profile_root.resolve().as_uri()}",
                    "--convert-to",
                    conversion_format,
                    "--outdir",
                    str(output_root),
                    str(input_path),
                )
                return_code = sandbox.run(
                    arguments,
                    cwd=temporary_root,
                    deadline=timeout,
                    max_output_bytes=self._max_output_bytes,
                )
                if type(return_code) is not int or return_code != 0:
                    raise LegacyConversionError("converter_failed")
                expected = output_root / f"source{output_suffix}"
                entries = tuple(output_root.iterdir())
                if (
                    entries != (expected,)
                    or expected.is_symlink()
                    or not expected.is_file()
                ):
                    raise LegacyConversionError("converter_failed")
                content = _read_path_bounded(expected, self._max_output_bytes)
        except LegacyConversionError:
            raise
        except BaseException:
            raise LegacyConversionError("converter_failed") from None

        return ConvertedDocument(
            suffix=output_suffix,
            content_bytes=content,
            content_size_bytes=len(content),
            content_sha256=sha256(content).hexdigest(),
        )

    def _fixed_executable(self) -> Path | None:
        candidate = self._configured_executable
        if candidate is None:
            discovered = shutil.which("soffice") or shutil.which("libreoffice")
            if discovered is None:
                return None
            candidate = Path(discovered)
        try:
            if candidate.is_symlink():
                return None
            resolved = candidate.resolve(strict=True)
            if not resolved.is_file():
                return None
            return resolved
        except (OSError, RuntimeError):
            return None


def _copy_stream_bounded(source: BinaryIO, destination: Path, maximum: int) -> None:
    if not callable(getattr(source, "read", None)) or not callable(getattr(source, "seek", None)):
        raise LegacyConversionError("converter_failed")
    try:
        source.seek(0)
        total = 0
        with destination.open("xb") as output:
            while chunk := source.read(min(1024 * 1024, maximum - total + 1)):
                if not isinstance(chunk, bytes):
                    raise LegacyConversionError("converter_failed")
                total += len(chunk)
                if total > maximum:
                    raise LegacyConversionError("converter_failed")
                output.write(chunk)
        if total == 0:
            raise LegacyConversionError("converter_failed")
    except LegacyConversionError:
        raise
    except Exception:
        raise LegacyConversionError("converter_failed") from None


def _read_path_bounded(path: Path, maximum: int) -> bytes:
    try:
        with path.open("rb") as stream:
            content = stream.read(maximum + 1)
            if stream.read(1):
                raise LegacyConversionError("converter_failed")
    except LegacyConversionError:
        raise
    except OSError:
        raise LegacyConversionError("converter_failed") from None
    if not content or len(content) > maximum:
        raise LegacyConversionError("converter_failed")
    return content
