from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from io import BytesIO, TextIOWrapper
import json
from pathlib import PurePosixPath
from tempfile import SpooledTemporaryFile
from typing import Any, BinaryIO

from pydantic import ConfigDict, Field, field_validator, model_validator
from pypdf import PdfReader, PdfWriter
from pypdf.errors import FileNotDecryptedError, PdfReadError, PdfStreamError
import yaml

try:
    from PIL import Image
except ImportError:  # pragma: no cover - the dependency is mandatory in distributions
    Image = None  # type: ignore[assignment]

from exam_predictor.evidence.artifacts import ArtifactBoundaryError, EvidenceArtifactStore
from exam_predictor.evidence.models import EvidenceFrozenModel, PartState, SourcePartPlan
from exam_predictor.evidence.policy import EvidencePolicy, representative_ordinals, source_priority
from exam_predictor.workspace.models import ManifestEntry, ScanPolicy, normalize_relative_path
from exam_predictor.workspace.policy import DEFAULT_SCAN_POLICY


_TEXT_OVERLAP_LINES = 2
_MAX_STRUCTURE_DEPTH = 64
_MAX_STRUCTURE_NODES = 100_000
_MAX_IMAGE_DIMENSION = 32_768
_MAX_IMAGE_PIXELS = 25_000_000
_MAX_GIF_FRAMES = 256
_MAX_SELECTED_GIF_FRAMES = 3
_MAX_READ_CHUNK = 1024 * 1024
_SPOOL_MEMORY_BYTES = 1024 * 1024

_TEXT_MEDIA_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
}
_STRUCTURED_MEDIA_TYPES = {
    ".json": "application/json",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
}
_IMAGE_MEDIA_TYPES = {
    ".png": ("png", "image/png"),
    ".jpg": ("jpeg", "image/jpeg"),
    ".jpeg": ("jpeg", "image/jpeg"),
    ".webp": ("webp", "image/webp"),
    ".gif": ("gif", "image/gif"),
    ".bmp": ("bmp", "image/bmp"),
    ".tif": ("tiff", "image/tiff"),
    ".tiff": ("tiff", "image/tiff"),
}


class SourcePreparationError(RuntimeError):
    """A safe, stable preparation failure without parser or source text."""

    def __init__(self, code: str, locator: str) -> None:
        self.code = code
        self.locator = locator
        super().__init__(code)


class PreparedPartRequest(EvidenceFrozenModel):
    """Hash-bound authority and metadata for one caller-owned source stream."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    revision_id: str
    entry_id: str
    relative_path: str
    format_category: str
    source_size_bytes: int = Field(ge=0)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_preview_entries: tuple[ManifestEntry, ...] = ()

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return normalize_relative_path(value)

    @model_validator(mode="after")
    def validate_archive_preview_entries(self) -> PreparedPartRequest:
        entry_ids: set[str] = set()
        for entry in self.archive_preview_entries:
            if (
                entry.workspace_id != self.workspace_id
                or entry.archive_parent_entry_id != self.entry_id
                or entry.entry_id in entry_ids
            ):
                raise ValueError("archive preview authority does not match the source request")
            entry_ids.add(entry.entry_id)
        return self


@dataclass
class _StagedPart:
    ordinal: int
    locator: str
    media_type: str
    stream: BinaryIO
    size_bytes: int
    sha256: str

    def close(self) -> None:
        self.stream.close()


@dataclass(frozen=True)
class _Line:
    number: int
    content: bytes


class _LimitedLineIterator:
    def __init__(self, wrapper: TextIOWrapper, max_line_bytes: int) -> None:
        self._wrapper = wrapper
        self._max_line_bytes = max_line_bytes
        self.consumed: list[str] = []

    def __iter__(self) -> _LimitedLineIterator:
        return self

    def __next__(self) -> str:
        line = self._wrapper.readline(self._max_line_bytes + 1)
        if line == "":
            raise StopIteration
        if len(line.encode("utf-8")) > self._max_line_bytes:
            raise SourcePreparationError("tabular_row_too_large", "source")
        self.consumed.append(line)
        return line


class SourcePartPreparer:
    def __init__(
        self,
        artifact_store: EvidenceArtifactStore,
        *,
        policy: EvidencePolicy = EvidencePolicy(),
        scan_policy: ScanPolicy = DEFAULT_SCAN_POLICY,
        legacy_converter: object | None = None,
    ) -> None:
        self._artifact_store = artifact_store
        self._policy = policy
        self._scan_policy = scan_policy
        self._legacy_converter = legacy_converter

    def prepare(
        self,
        request: PreparedPartRequest,
        stream: BinaryIO,
    ) -> tuple[SourcePartPlan, ...]:
        self._verify_source(request, stream)
        suffix = PurePosixPath(request.relative_path).suffix.casefold()
        if request.format_category == "pdf" and suffix == ".pdf":
            staged = self._prepare_pdf(stream)
        elif request.format_category == "text" and suffix in _TEXT_MEDIA_TYPES:
            staged = self._prepare_text(stream, _TEXT_MEDIA_TYPES[suffix])
        elif request.format_category == "tabular" and suffix in {".csv", ".tsv"}:
            staged = self._prepare_tabular(stream, suffix)
        elif request.format_category == "structured_data" and suffix in _STRUCTURED_MEDIA_TYPES:
            staged = self._prepare_structured_data(request, stream, suffix)
        elif request.format_category == "image" and suffix in _IMAGE_MEDIA_TYPES:
            staged = self._prepare_image(request, stream, suffix)
        else:
            raise SourcePreparationError("source_format_unsupported", "source")
        return self._publish(request, staged)

    def _verify_source(self, request: PreparedPartRequest, stream: BinaryIO) -> None:
        if (
            isinstance(stream, (str, bytes, bytearray))
            or not callable(getattr(stream, "read", None))
            or not callable(getattr(stream, "seek", None))
        ):
            raise SourcePreparationError("source_stream_invalid", "source")
        if request.source_size_bytes > self._scan_policy.max_workspace_bytes:
            raise SourcePreparationError("source_size_exceeded", "source")
        chunk_bytes = max(1, min(self._scan_policy.hash_chunk_bytes, _MAX_READ_CHUNK))
        total = 0
        digest = hashlib.sha256()
        try:
            stream.seek(0)
            while True:
                chunk = stream.read(chunk_bytes)
                if not isinstance(chunk, bytes) or len(chunk) > chunk_bytes:
                    raise SourcePreparationError("source_stream_invalid", "source")
                if not chunk:
                    break
                total += len(chunk)
                if total > self._scan_policy.max_workspace_bytes:
                    raise SourcePreparationError("source_size_exceeded", "source")
                digest.update(chunk)
            stream.seek(0)
        except SourcePreparationError:
            raise
        except Exception:
            raise SourcePreparationError("source_stream_invalid", "source") from None
        if total != request.source_size_bytes:
            raise SourcePreparationError("source_size_mismatch", "source")
        if digest.hexdigest() != request.source_sha256:
            raise SourcePreparationError("source_hash_mismatch", "source")

    def _prepare_pdf(self, stream: BinaryIO) -> list[_StagedPart]:
        staged: list[_StagedPart] = []
        try:
            stream.seek(0)
            reader = PdfReader(stream, strict=True)
            if reader.is_encrypted:
                raise SourcePreparationError("pdf_encrypted", "source")
            page_count = len(reader.pages)
            if page_count == 0:
                raise SourcePreparationError("pdf_empty", "source")
            start = 0
            while start < page_count:
                end = min(start + self._policy.pdf_pages_per_part, page_count)
                while True:
                    candidate = self._write_pdf_range(reader, start, end)
                    if candidate.size_bytes <= self._policy.max_part_bytes:
                        candidate.ordinal = len(staged)
                        staged.append(candidate)
                        start = end
                        break
                    candidate.close()
                    if end - start == 1:
                        raise SourcePreparationError("pdf_page_too_large", f"page {start + 1}")
                    end = start + max(1, (end - start) // 2)
            return staged
        except SourcePreparationError:
            _close_staged(staged)
            raise
        except (FileNotDecryptedError, PdfReadError, PdfStreamError, ValueError, TypeError, KeyError):
            _close_staged(staged)
            raise SourcePreparationError("pdf_corrupt", "source") from None
        except Exception:
            _close_staged(staged)
            raise SourcePreparationError("pdf_corrupt", "source") from None

    def _write_pdf_range(self, reader: PdfReader, start: int, end: int) -> _StagedPart:
        output = SpooledTemporaryFile(max_size=_SPOOL_MEMORY_BYTES, mode="w+b")
        try:
            writer = PdfWriter()
            for page_index in range(start, end):
                writer.add_page(reader.pages[page_index])
            writer.write(output)
            return _finish_staged(
                output,
                ordinal=0,
                locator=f"pages {start + 1}-{end}",
                media_type="application/pdf",
            )
        except Exception:
            output.close()
            raise

    def _prepare_text(self, stream: BinaryIO, media_type: str) -> list[_StagedPart]:
        try:
            return self._prepare_text_once(stream, media_type, errors="strict")
        except UnicodeDecodeError:
            return self._prepare_text_once(stream, media_type, errors="replace")

    def _prepare_text_once(
        self,
        stream: BinaryIO,
        media_type: str,
        *,
        errors: str,
    ) -> list[_StagedPart]:
        stream.seek(0)
        wrapper = TextIOWrapper(stream, encoding="utf-8", errors=errors, newline="")
        staged: list[_StagedPart] = []
        current: list[_Line] = []
        current_size = 0
        overlap_count = 0
        line_number = 0
        try:
            while True:
                line = wrapper.readline(self._policy.max_part_bytes + 1)
                if line == "":
                    break
                line_number += 1
                encoded = line.encode("utf-8")
                if len(encoded) > self._policy.max_part_bytes:
                    raise SourcePreparationError("text_line_too_large", f"line {line_number}")
                if current and current_size + len(encoded) > self._policy.max_part_bytes:
                    staged.append(
                        self._stage_lines(
                            current,
                            media_type,
                            overlap_count=overlap_count,
                            ordinal=len(staged),
                        )
                    )
                    overlap = current[-_TEXT_OVERLAP_LINES:]
                    while (
                        overlap
                        and sum(len(item.content) for item in overlap) + len(encoded)
                        > self._policy.max_part_bytes
                    ):
                        overlap.pop(0)
                    current = list(overlap)
                    current_size = sum(len(item.content) for item in current)
                    overlap_count = len(current)
                current.append(_Line(line_number, encoded))
                current_size += len(encoded)
            if not current:
                raise SourcePreparationError("text_empty", "source")
            staged.append(
                self._stage_lines(
                    current,
                    media_type,
                    overlap_count=overlap_count,
                    ordinal=len(staged),
                )
            )
            return staged
        except Exception:
            _close_staged(staged)
            raise
        finally:
            try:
                wrapper.detach()
            except Exception:
                pass

    def _stage_lines(
        self,
        lines: list[_Line],
        media_type: str,
        *,
        overlap_count: int,
        ordinal: int,
    ) -> _StagedPart:
        locator = f"lines {lines[0].number}-{lines[-1].number}"
        if overlap_count:
            locator += f"; overlap {overlap_count} lines"
        return self._stage_bytes(
            b"".join(item.content for item in lines),
            ordinal=ordinal,
            locator=locator,
            media_type=media_type,
            oversized_code="text_line_too_large",
        )

    def _prepare_tabular(self, stream: BinaryIO, suffix: str) -> list[_StagedPart]:
        try:
            return self._prepare_tabular_once(stream, suffix, errors="strict")
        except UnicodeDecodeError:
            return self._prepare_tabular_once(stream, suffix, errors="replace")

    def _prepare_tabular_once(
        self,
        stream: BinaryIO,
        suffix: str,
        *,
        errors: str,
    ) -> list[_StagedPart]:
        stream.seek(0)
        wrapper = TextIOWrapper(stream, encoding="utf-8", errors=errors, newline="")
        lines = _LimitedLineIterator(wrapper, self._policy.max_part_bytes)
        reader = csv.reader(lines, delimiter="," if suffix == ".csv" else "\t", strict=True)
        media_type = "text/csv" if suffix == ".csv" else "text/tab-separated-values"
        staged: list[_StagedPart] = []
        current: list[bytes] = []
        current_size = 0
        first_row = 1
        row_number = 0
        try:
            while True:
                lines.consumed.clear()
                try:
                    next(reader)
                except StopIteration:
                    break
                row_number += 1
                encoded = "".join(lines.consumed).encode("utf-8")
                if len(encoded) > self._policy.max_part_bytes:
                    raise SourcePreparationError("tabular_row_too_large", f"row {row_number}")
                if current and current_size + len(encoded) > self._policy.max_part_bytes:
                    staged.append(
                        self._stage_bytes(
                            b"".join(current),
                            ordinal=len(staged),
                            locator=f"rows {first_row}-{row_number - 1}",
                            media_type=media_type,
                            oversized_code="tabular_row_too_large",
                        )
                    )
                    current = []
                    current_size = 0
                    first_row = row_number
                current.append(encoded)
                current_size += len(encoded)
            if row_number == 0:
                raise SourcePreparationError("tabular_empty", "source")
            staged.append(
                self._stage_bytes(
                    b"".join(current),
                    ordinal=len(staged),
                    locator=f"rows {first_row}-{row_number}",
                    media_type=media_type,
                    oversized_code="tabular_row_too_large",
                )
            )
            return staged
        except csv.Error:
            _close_staged(staged)
            raise SourcePreparationError("tabular_malformed", "source") from None
        except Exception:
            _close_staged(staged)
            raise
        finally:
            try:
                wrapper.detach()
            except Exception:
                pass

    def _prepare_structured_data(
        self,
        request: PreparedPartRequest,
        stream: BinaryIO,
        suffix: str,
    ) -> list[_StagedPart]:
        source_limit = min(
            self._scan_policy.max_workspace_bytes,
            self._policy.max_part_bytes * 4,
        )
        if request.source_size_bytes > source_limit:
            raise SourcePreparationError("structured_data_too_large", "source")
        content = _read_bounded(stream, source_limit, "structured_data_too_large")
        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError:
            decoded = content.decode("utf-8", errors="replace")
        try:
            document = json.loads(decoded) if suffix == ".json" else yaml.safe_load(decoded)
            _validate_structure(document)
        except Exception:
            raise SourcePreparationError("structured_data_malformed", "source") from None
        normalized = decoded.encode("utf-8")
        return self._prepare_text(BytesIO(normalized), _STRUCTURED_MEDIA_TYPES[suffix])

    def _prepare_image(
        self,
        request: PreparedPartRequest,
        stream: BinaryIO,
        suffix: str,
    ) -> list[_StagedPart]:
        if request.source_size_bytes > self._policy.max_part_bytes:
            raise SourcePreparationError("image_too_large", "image")
        content = _read_bounded(stream, self._policy.max_part_bytes, "image_too_large")
        expected_format, media_type = _IMAGE_MEDIA_TYPES[suffix]
        actual_format = _detect_image_format(content)
        if actual_format != expected_format:
            raise SourcePreparationError("image_magic_mismatch", "image")
        width, height = _image_dimensions(content, actual_format)
        _validate_image_dimensions(width, height)
        if Image is None:
            code = "image_frames_unavailable" if actual_format == "gif" else "image_parser_unavailable"
            raise SourcePreparationError(code, "image")
        try:
            with Image.open(BytesIO(content)) as image:
                if image.size != (width, height):
                    raise SourcePreparationError("image_header_mismatch", "image")
                frame_count = int(getattr(image, "n_frames", 1))
                if frame_count > _MAX_GIF_FRAMES:
                    raise SourcePreparationError("image_frame_count_exceeded", "image")
                if actual_format != "gif" or frame_count == 1:
                    image.verify()
                    return [
                        self._stage_bytes(
                            content,
                            ordinal=0,
                            locator="image 1",
                            media_type=media_type,
                            oversized_code="image_too_large",
                        )
                    ]
                selected = representative_ordinals(frame_count)[:_MAX_SELECTED_GIF_FRAMES]
                omitted = frame_count - len(selected)
                staged: list[_StagedPart] = []
                try:
                    for frame_index in selected:
                        image.seek(frame_index)
                        frame = image.convert("RGBA")
                        output = BytesIO()
                        frame.save(output, format="PNG", optimize=False, compress_level=9)
                        staged.append(
                            self._stage_bytes(
                                output.getvalue(),
                                ordinal=len(staged),
                                locator=(f"frame {frame_index + 1} of {frame_count}; omitted {omitted}"),
                                media_type="image/png",
                                oversized_code="image_frame_too_large",
                            )
                        )
                    return staged
                except Exception:
                    _close_staged(staged)
                    raise
        except SourcePreparationError:
            raise
        except Exception:
            raise SourcePreparationError("image_corrupt", "image") from None

    def _stage_bytes(
        self,
        content: bytes,
        *,
        ordinal: int,
        locator: str,
        media_type: str,
        oversized_code: str,
    ) -> _StagedPart:
        if len(content) > self._policy.max_part_bytes:
            raise SourcePreparationError(oversized_code, locator)
        output = SpooledTemporaryFile(max_size=_SPOOL_MEMORY_BYTES, mode="w+b")
        try:
            output.write(content)
            return _finish_staged(
                output,
                ordinal=ordinal,
                locator=locator,
                media_type=media_type,
            )
        except Exception:
            output.close()
            raise

    def _publish(
        self,
        request: PreparedPartRequest,
        staged: list[_StagedPart],
    ) -> tuple[SourcePartPlan, ...]:
        priority, scheduling_class = source_priority(
            request.relative_path,
            request.format_category,
        )
        plans: list[SourcePartPlan] = []
        try:
            for part in staged:
                part.stream.seek(0)
                content = part.stream.read(self._policy.max_part_bytes + 1)
                if len(content) != part.size_bytes or hashlib.sha256(content).hexdigest() != part.sha256:
                    raise SourcePreparationError("prepared_part_changed", part.locator)
                identity = _part_identity(
                    policy_version=self._policy.policy_version,
                    source_sha256=request.source_sha256,
                    part_sha256=part.sha256,
                    locator=part.locator,
                    media_type=part.media_type,
                )
                part_id = f"part_{identity}"
                try:
                    published_sha256 = self._artifact_store.publish_part(
                        request.workspace_id,
                        part_id,
                        content,
                        expected_sha256=part.sha256,
                    )
                except ArtifactBoundaryError as error:
                    raise SourcePreparationError(error.code, part.locator) from None
                except Exception:
                    raise SourcePreparationError("artifact_publish_failed", part.locator) from None
                if published_sha256 != part.sha256:
                    raise SourcePreparationError("artifact_hash_mismatch", part.locator)
                plans.append(
                    SourcePartPlan(
                        part_id=part_id,
                        workspace_id=request.workspace_id,
                        revision_id=request.revision_id,
                        entry_id=request.entry_id,
                        relative_path=request.relative_path,
                        source_sha256=request.source_sha256,
                        part_sha256=part.sha256,
                        ordinal=part.ordinal,
                        locator=part.locator,
                        media_type=part.media_type,
                        size_bytes=part.size_bytes,
                        scheduling_class=scheduling_class,
                        priority=priority,
                        state=PartState.PREPARED,
                        idempotency_key=f"prepare_{identity}",
                    )
                )
            return tuple(plans[index] for index in representative_ordinals(len(plans)))
        finally:
            _close_staged(staged)


def _finish_staged(
    stream: BinaryIO,
    *,
    ordinal: int,
    locator: str,
    media_type: str,
) -> _StagedPart:
    stream.seek(0, 2)
    size_bytes = stream.tell()
    stream.seek(0)
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(_MAX_READ_CHUNK)
        if not chunk:
            break
        digest.update(chunk)
    stream.seek(0)
    return _StagedPart(
        ordinal=ordinal,
        locator=locator,
        media_type=media_type,
        stream=stream,
        size_bytes=size_bytes,
        sha256=digest.hexdigest(),
    )


def _close_staged(staged: list[_StagedPart]) -> None:
    for part in staged:
        part.close()


def _read_bounded(stream: BinaryIO, limit: int, code: str) -> bytes:
    try:
        stream.seek(0)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = stream.read(min(_MAX_READ_CHUNK, limit - total + 1))
            if not isinstance(chunk, bytes):
                raise SourcePreparationError("source_stream_invalid", "source")
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise SourcePreparationError(code, "source")
        return b"".join(chunks)
    except SourcePreparationError:
        raise
    except Exception:
        raise SourcePreparationError("source_stream_invalid", "source") from None


def _validate_structure(document: Any) -> None:
    stack: list[tuple[Any, int]] = [(document, 0)]
    seen: set[int] = set()
    nodes = 0
    while stack:
        value, depth = stack.pop()
        if depth > _MAX_STRUCTURE_DEPTH:
            raise ValueError("structure depth exceeded")
        nodes += 1
        if nodes > _MAX_STRUCTURE_NODES:
            raise ValueError("structure node count exceeded")
        if isinstance(value, (dict, list, tuple, set)):
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
        if isinstance(value, dict):
            for key, item in value.items():
                stack.append((key, depth + 1))
                stack.append((item, depth + 1))
        elif isinstance(value, (list, tuple, set)):
            stack.extend((item, depth + 1) for item in value)
        elif isinstance(value, str) and len(value.encode("utf-8")) > _MAX_READ_CHUNK:
            raise ValueError("structure string exceeded")


def _detect_image_format(content: bytes) -> str | None:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if content.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if content.startswith(b"BM"):
        return "bmp"
    if content.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "webp"
    return None


def _image_dimensions(content: bytes, image_format: str) -> tuple[int, int]:
    try:
        if image_format == "png":
            if len(content) < 24 or content[12:16] != b"IHDR":
                raise ValueError
            return int.from_bytes(content[16:20], "big"), int.from_bytes(content[20:24], "big")
        if image_format == "gif":
            if len(content) < 10:
                raise ValueError
            return int.from_bytes(content[6:8], "little"), int.from_bytes(content[8:10], "little")
        if image_format == "bmp":
            if len(content) < 26:
                raise ValueError
            dib_size = int.from_bytes(content[14:18], "little")
            if dib_size == 12:
                return int.from_bytes(content[18:20], "little"), int.from_bytes(content[20:22], "little")
            return abs(int.from_bytes(content[18:22], "little", signed=True)), abs(
                int.from_bytes(content[22:26], "little", signed=True)
            )
        if image_format == "jpeg":
            return _jpeg_dimensions(content)
        if Image is None:
            raise SourcePreparationError("image_parser_unavailable", "image")
        with Image.open(BytesIO(content)) as image:
            return image.size
    except SourcePreparationError:
        raise
    except Exception:
        raise SourcePreparationError("image_corrupt", "image") from None


def _jpeg_dimensions(content: bytes) -> tuple[int, int]:
    position = 2
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while position + 4 <= len(content):
        if content[position] != 0xFF:
            position += 1
            continue
        while position < len(content) and content[position] == 0xFF:
            position += 1
        if position >= len(content):
            break
        marker = content[position]
        position += 1
        if marker in {0xD8, 0xD9}:
            continue
        if position + 2 > len(content):
            break
        segment_length = int.from_bytes(content[position : position + 2], "big")
        if segment_length < 2 or position + segment_length > len(content):
            break
        if marker in sof_markers:
            if segment_length < 7:
                break
            height = int.from_bytes(content[position + 3 : position + 5], "big")
            width = int.from_bytes(content[position + 5 : position + 7], "big")
            return width, height
        position += segment_length
    raise SourcePreparationError("image_corrupt", "image")


def _validate_image_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise SourcePreparationError("image_dimensions_invalid", "image")
    if width > _MAX_IMAGE_DIMENSION or height > _MAX_IMAGE_DIMENSION or width * height > _MAX_IMAGE_PIXELS:
        raise SourcePreparationError("image_dimensions_exceeded", "image")


def _part_identity(
    *,
    policy_version: str,
    source_sha256: str,
    part_sha256: str,
    locator: str,
    media_type: str,
) -> str:
    encoded = json.dumps(
        {
            "locator": locator,
            "media_type": media_type,
            "part_sha256": part_sha256,
            "policy_version": policy_version,
            "source_sha256": source_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
