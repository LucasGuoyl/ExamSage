from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
from io import BytesIO, TextIOWrapper
import json
import math
from pathlib import PurePosixPath
from tempfile import SpooledTemporaryFile
from typing import Any, BinaryIO, Literal
import unicodedata
import zipfile

from pydantic import ConfigDict, Field, field_validator, model_validator
from pypdf import PdfReader, PdfWriter
from pypdf.errors import FileNotDecryptedError, PdfReadError, PdfStreamError
import yaml
from yaml.events import (
    AliasEvent,
    MappingEndEvent,
    MappingStartEvent,
    ScalarEvent,
    SequenceEndEvent,
    SequenceStartEvent,
)

try:
    from PIL import Image
except ImportError:  # pragma: no cover - the dependency is mandatory in distributions
    Image = None  # type: ignore[assignment]

from exam_predictor.evidence.artifacts import ArtifactBoundaryError, EvidenceArtifactStore
from exam_predictor.evidence.converter import LegacyConversionError, LegacyOfficeConverter
from exam_predictor.evidence.models import EvidenceFrozenModel, PartState, SourcePartPlan
from exam_predictor.evidence.ooxml import OoxmlPreparationError, prepare_ooxml
from exam_predictor.evidence.policy import EvidencePolicy, representative_ordinals, source_priority
from exam_predictor.workspace.archive import ArchiveInspector
from exam_predictor.workspace.models import (
    ManifestEntry,
    ScanPolicy,
    SourceState,
    normalize_relative_path,
)
from exam_predictor.workspace.policy import DEFAULT_SCAN_POLICY


_TEXT_OVERLAP_LINES = 2
_MAX_STRUCTURE_DEPTH = 64
_MAX_STRUCTURE_NODES = 100_000
_MAX_STRUCTURE_TOKENS = 250_000
_MAX_STRUCTURE_STRING_BYTES = 1024 * 1024
_MAX_IMAGE_DIMENSION = 32_768
_MAX_IMAGE_PIXELS = 25_000_000
_MAX_GIF_FRAMES = 256
_MAX_SELECTED_GIF_FRAMES = 3
_MAX_READ_CHUNK = 1024 * 1024
_SPOOL_MEMORY_BYTES = 1024 * 1024
_MAX_ARCHIVE_MEMBER_INPUT_BYTES = 64 * 1024 * 1024
_MAX_OOXML_OUTPUT_BYTES = 64 * 1024 * 1024
_MAX_OOXML_OUTPUT_PARTS = 256

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
_OOXML_CATEGORIES = {
    ".docx": "document",
    ".pptx": "presentation",
    ".xlsx": "spreadsheet",
}
_LEGACY_OFFICE_CATEGORIES = {
    ".doc": "document",
    ".ppt": "presentation",
    ".xls": "spreadsheet",
}


class SourcePreparationError(RuntimeError):
    """A safe, stable preparation failure without parser or source text."""

    def __init__(self, code: str, locator: str) -> None:
        self.code = code
        self.locator = locator
        super().__init__(code)


class _StructuredDataLimitError(ValueError):
    pass


class _StructuredDataMalformedError(ValueError):
    pass


class _TabularMalformedError(ValueError):
    pass


class ArchivePreviewAuthority(EvidenceFrozenModel):
    """Explicit, revision-bound approval to prepare one safe archive preview."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    revision_id: str
    parent_entry_id: str
    parent_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entry: ManifestEntry
    approved: Literal[True]

    @model_validator(mode="after")
    def validate_preview_entry(self) -> ArchivePreviewAuthority:
        entry = self.entry
        member_path = entry.archive_member_path
        path_is_valid = False
        if member_path:
            try:
                path_is_valid = (
                    normalize_relative_path(member_path) == member_path
                    and len(member_path) <= DEFAULT_SCAN_POLICY.max_path_chars
                    and not any(
                        unicodedata.category(character) == "Cc"
                        for character in member_path
                    )
                )
            except ValueError:
                path_is_valid = False
        if (
            entry.workspace_id != self.workspace_id
            or entry.archive_parent_entry_id != self.parent_entry_id
            or entry.item_kind != "archive_member"
            or entry.inclusion_reason != "archive_preview"
            or entry.included
            or entry.failure_code is not None
            or entry.state not in {SourceState.PENDING_APPROVAL, SourceState.APPROVED}
            or entry.archive_member_index is None
            or entry.archive_member_crc32 is None
            or entry.archive_member_compressed_bytes is None
            or not path_is_valid
        ):
            raise ValueError("archive preview entry is not a safe scanner preview")
        return self


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
    archive_previews: tuple[ArchivePreviewAuthority, ...] = ()

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return normalize_relative_path(value)

    @model_validator(mode="after")
    def validate_archive_previews(self) -> PreparedPartRequest:
        entry_ids: set[str] = set()
        member_paths: set[str] = set()
        for authority in self.archive_previews:
            entry = authority.entry
            member_path = entry.archive_member_path
            if (
                authority.workspace_id != self.workspace_id
                or authority.revision_id != self.revision_id
                or authority.parent_entry_id != self.entry_id
                or authority.parent_source_sha256 != self.source_sha256
                or entry.relative_path != self.relative_path
                or member_path is None
                or entry.entry_id in entry_ids
                or member_path in member_paths
            ):
                raise ValueError("archive preview authority does not match the source request")
            entry_ids.add(entry.entry_id)
            member_paths.add(member_path)
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


class SourcePartPreparer:
    def __init__(
        self,
        artifact_store: EvidenceArtifactStore,
        *,
        policy: EvidencePolicy = EvidencePolicy(),
        scan_policy: ScanPolicy = DEFAULT_SCAN_POLICY,
        legacy_converter: LegacyOfficeConverter | None = None,
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
        snapshot = self._snapshot_source(request, stream)
        staged: list[_StagedPart] | None = None
        failure: SourcePreparationError | None = None
        try:
            suffix = PurePosixPath(request.relative_path).suffix.casefold()
            if request.format_category == "pdf" and suffix == ".pdf":
                staged = self._prepare_pdf(snapshot)
            elif request.format_category == "text" and suffix in _TEXT_MEDIA_TYPES:
                staged = self._prepare_text(snapshot, _TEXT_MEDIA_TYPES[suffix])
            elif request.format_category == "tabular" and suffix in {".csv", ".tsv"}:
                staged = self._prepare_tabular(snapshot, suffix)
            elif request.format_category == "structured_data" and suffix in _STRUCTURED_MEDIA_TYPES:
                staged = self._prepare_structured_data(request, snapshot, suffix)
            elif request.format_category == "image" and suffix in _IMAGE_MEDIA_TYPES:
                staged = self._prepare_image(request, snapshot, suffix)
            elif _OOXML_CATEGORIES.get(suffix) == request.format_category:
                staged = self._prepare_ooxml(snapshot, suffix)
            elif request.format_category == "archive" and suffix == ".zip":
                staged = self._prepare_archive(request, snapshot)
            elif _LEGACY_OFFICE_CATEGORIES.get(suffix) == request.format_category:
                staged = self._prepare_legacy_office(snapshot, suffix)
            else:
                raise SourcePreparationError("source_format_unsupported", "source")
        except SourcePreparationError as error:
            failure = SourcePreparationError(error.code, error.locator)
        except Exception:
            failure = SourcePreparationError("source_preparation_failed", "source")
        finally:
            _close_stream(snapshot)
        if failure is not None:
            raise failure
        if staged is None:
            raise SourcePreparationError("source_preparation_failed", "source")
        return self._publish(request, staged)

    def _snapshot_source(self, request: PreparedPartRequest, stream: BinaryIO) -> BinaryIO:
        if (
            isinstance(stream, (str, bytes, bytearray))
            or not callable(getattr(stream, "read", None))
            or not callable(getattr(stream, "seek", None))
        ):
            raise SourcePreparationError("source_stream_invalid", "source")
        if request.source_size_bytes > self._scan_policy.max_workspace_bytes:
            raise SourcePreparationError("source_size_exceeded", "source")
        chunk_bytes = max(1, min(self._scan_policy.hash_chunk_bytes, _MAX_READ_CHUNK))
        snapshot = _new_spool("source_spool_unavailable", "source")
        total = 0
        digest = hashlib.sha256()
        failure: SourcePreparationError | None = None
        try:
            stream.seek(0)
        except Exception:
            failure = SourcePreparationError("source_stream_invalid", "source")
        while failure is None:
            read_size = min(chunk_bytes, request.source_size_bytes - total + 1)
            try:
                chunk = stream.read(read_size)
            except Exception:
                failure = SourcePreparationError("source_stream_invalid", "source")
                break
            if not isinstance(chunk, bytes) or len(chunk) > read_size:
                failure = SourcePreparationError("source_stream_invalid", "source")
                break
            if not chunk:
                break
            total += len(chunk)
            if total > request.source_size_bytes:
                failure = SourcePreparationError("source_size_mismatch", "source")
                break
            digest.update(chunk)
            try:
                written = snapshot.write(chunk)
            except Exception:
                failure = SourcePreparationError("source_spool_unavailable", "source")
                break
            if written != len(chunk):
                failure = SourcePreparationError("source_spool_unavailable", "source")
                break
        if failure is not None:
            _close_stream(snapshot)
            raise failure
        if total != request.source_size_bytes:
            _close_stream(snapshot)
            raise SourcePreparationError("source_size_mismatch", "source")
        if digest.hexdigest() != request.source_sha256:
            _close_stream(snapshot)
            raise SourcePreparationError("source_hash_mismatch", "source")
        try:
            snapshot.seek(0)
        except Exception:
            failure = SourcePreparationError("source_spool_unavailable", "source")
        if failure is not None:
            _close_stream(snapshot)
            raise failure
        return snapshot

    def _prepare_pdf(self, stream: BinaryIO) -> list[_StagedPart]:
        staged: list[_StagedPart] = []
        failure: SourcePreparationError | None = None
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
        except SourcePreparationError as error:
            failure = error
        except (
            FileNotDecryptedError,
            PdfReadError,
            PdfStreamError,
            ValueError,
            TypeError,
            KeyError,
        ):
            failure = SourcePreparationError("pdf_corrupt", "source")
        except Exception:
            failure = SourcePreparationError("pdf_corrupt", "source")
        if failure is not None:
            _close_staged(staged)
            raise failure
        return staged

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
        delimiter = "," if suffix == ".csv" else "\t"
        media_type = "text/csv" if suffix == ".csv" else "text/tab-separated-values"
        staged: list[_StagedPart] = []
        current = bytearray()
        first_row = 1
        row_number = 0
        failure: SourcePreparationError | None = None
        try:
            for row_number, encoded in _iter_tabular_rows(
                wrapper,
                delimiter=delimiter,
                max_row_bytes=self._policy.max_part_bytes,
            ):
                if current and len(current) + len(encoded) > self._policy.max_part_bytes:
                    staged.append(
                        self._stage_bytes(
                            bytes(current),
                            ordinal=len(staged),
                            locator=f"rows {first_row}-{row_number - 1}",
                            media_type=media_type,
                            oversized_code="tabular_row_too_large",
                        )
                    )
                    current = bytearray()
                    first_row = row_number
                current.extend(encoded)
            if row_number == 0:
                raise SourcePreparationError("tabular_empty", "source")
            staged.append(
                self._stage_bytes(
                    bytes(current),
                    ordinal=len(staged),
                    locator=f"rows {first_row}-{row_number}",
                    media_type=media_type,
                    oversized_code="tabular_row_too_large",
                )
            )
            return staged
        except _TabularMalformedError:
            failure = SourcePreparationError("tabular_malformed", "source")
        except Exception:
            _close_staged(staged)
            raise
        finally:
            try:
                wrapper.detach()
            except Exception:
                pass
        if failure is not None:
            _close_staged(staged)
            raise failure
        raise AssertionError("tabular preparation reached an invalid state")

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
        failure_code: str | None = None
        document: Any = None
        try:
            if suffix == ".json":
                _preflight_json(decoded)
                document = json.loads(decoded, parse_constant=_reject_json_constant)
            else:
                _preflight_yaml(decoded)
                document = yaml.safe_load(decoded)
            _validate_structure(document)
        except _StructuredDataLimitError:
            failure_code = "structured_data_limit_exceeded"
        except Exception:
            failure_code = "structured_data_malformed"
        if failure_code is not None:
            raise SourcePreparationError(failure_code, "source")
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
        return self._prepare_image_content(content, suffix)

    def _prepare_image_content(self, content: bytes, suffix: str) -> list[_StagedPart]:
        expected_format, media_type = _IMAGE_MEDIA_TYPES[suffix]
        actual_format = _detect_image_format(content)
        if actual_format != expected_format:
            raise SourcePreparationError("image_magic_mismatch", "image")
        width, height = _image_dimensions(content, actual_format)
        _validate_image_dimensions(width, height)
        if Image is None:
            code = "image_frames_unavailable" if actual_format == "gif" else "image_parser_unavailable"
            raise SourcePreparationError(code, "image")
        prepared: list[_StagedPart] | None = None
        failure: SourcePreparationError | None = None
        try:
            with Image.open(BytesIO(content)) as image:
                if image.size != (width, height):
                    raise SourcePreparationError("image_header_mismatch", "image")
                frame_count = int(getattr(image, "n_frames", 1))
                if frame_count > _MAX_GIF_FRAMES:
                    raise SourcePreparationError("image_frame_count_exceeded", "image")
                if actual_format != "gif" or frame_count == 1:
                    image.verify()
                    prepared = [
                        self._stage_bytes(
                            content,
                            ordinal=0,
                            locator="image 1",
                            media_type=media_type,
                            oversized_code="image_too_large",
                        )
                    ]
                else:
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
                        prepared = staged
                    except Exception:
                        _close_staged(staged)
                        raise
        except SourcePreparationError as error:
            failure = error
        except Exception:
            failure = SourcePreparationError("image_corrupt", "image")
        if failure is not None:
            raise failure
        if prepared is None:
            raise SourcePreparationError("image_corrupt", "image")
        return prepared

    def _prepare_ooxml(self, stream: BinaryIO, suffix: str) -> list[_StagedPart]:
        try:
            prepared = prepare_ooxml(
                stream,
                suffix=suffix,
                scan_policy=self._scan_policy,
                max_part_bytes=self._policy.max_part_bytes,
            )
        except OoxmlPreparationError as error:
            raise SourcePreparationError(error.code, error.locator) from None
        staged: list[_StagedPart] = []
        staged_bytes = 0

        def append_checked(candidate: _StagedPart) -> None:
            nonlocal staged_bytes
            if (
                len(staged) >= _MAX_OOXML_OUTPUT_PARTS
                or staged_bytes + candidate.size_bytes > _MAX_OOXML_OUTPUT_BYTES
            ):
                candidate.close()
                raise SourcePreparationError("ooxml_output_limit", "source")
            staged_bytes += candidate.size_bytes
            staged.append(candidate)

        try:
            for ordinal, part in enumerate(prepared):
                if part.media_type.startswith("image/"):
                    suffix_by_media_type = {
                        "image/png": ".png",
                        "image/jpeg": ".jpg",
                        "image/gif": ".gif",
                        "image/bmp": ".bmp",
                        "image/tiff": ".tiff",
                        "image/webp": ".webp",
                    }
                    image_suffix = suffix_by_media_type.get(part.media_type)
                    if image_suffix is None:
                        raise SourcePreparationError("image_format_unsupported", part.locator)
                    image_parts = self._prepare_image_content(part.content, image_suffix)
                    for image_part in image_parts:
                        image_part.ordinal = len(staged)
                        image_part.locator = (
                            part.locator
                            if len(image_parts) == 1
                            else f"{part.locator} {image_part.locator}"
                        )
                        append_checked(image_part)
                else:
                    append_checked(
                        self._stage_bytes(
                            part.content,
                            ordinal=len(staged),
                            locator=part.locator,
                            media_type=part.media_type,
                            oversized_code="ooxml_part_too_large",
                        )
                    )
            return staged
        except Exception:
            _close_staged(staged)
            raise

    def _prepare_archive(
        self,
        request: PreparedPartRequest,
        stream: BinaryIO,
    ) -> list[_StagedPart]:
        if not request.archive_previews:
            raise SourcePreparationError("archive_no_authorized_members", "source")
        try:
            stream.seek(0)
            inspected = ArchiveInspector(self._scan_policy).inspect(
                stream,
                parent_entry_id=request.entry_id,
            )
            if not inspected or any(member.state is SourceState.FAILED for member in inspected):
                raise SourcePreparationError("archive_unsafe", "source")
            stream.seek(0)
            archive = zipfile.ZipFile(stream, "r")
        except SourcePreparationError:
            raise
        except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile):
            raise SourcePreparationError("archive_invalid", "source") from None

        staged: list[_StagedPart] = []
        staged_bytes = 0
        try:
            infos = archive.infolist()
            for authority in request.archive_previews:
                entry = authority.entry
                member_path = entry.archive_member_path
                if member_path is None:
                    raise SourcePreparationError("archive_member_not_approved", "source")
                locator = f"archive member {member_path}"
                member_index = entry.archive_member_index
                if member_index is None or member_index > len(infos):
                    raise SourcePreparationError("archive_member_changed", locator)
                info = infos[member_index - 1]
                if (
                    info.is_dir()
                    or info.file_size != entry.size_bytes
                    or info.compress_size != entry.archive_member_compressed_bytes
                    or info.CRC != entry.archive_member_crc32
                    or info.filename.replace("\\", "/") != member_path
                    or info.flag_bits & 0x1
                ):
                    raise SourcePreparationError("archive_member_changed", locator)
                member_limit = min(
                    self._scan_policy.max_archive_expanded_bytes,
                    self._scan_policy.max_workspace_bytes,
                    _MAX_ARCHIVE_MEMBER_INPUT_BYTES,
                )
                if info.file_size > member_limit:
                    raise SourcePreparationError("archive_member_too_large", locator)
                with archive.open(info, "r") as member, SpooledTemporaryFile(
                    max_size=_SPOOL_MEMORY_BYTES,
                    mode="w+b",
                ) as member_stream:
                    member_hash = hashlib.sha256()
                    member_size = 0
                    while chunk := member.read(_MAX_READ_CHUNK):
                        member_size += len(chunk)
                        if member_size > member_limit:
                            raise SourcePreparationError("archive_member_too_large", locator)
                        member_hash.update(chunk)
                        member_stream.write(chunk)
                    if member_size != info.file_size:
                        raise SourcePreparationError("archive_member_changed", locator)
                    member_stream.seek(0)
                    member_parts = self._prepare_archive_member(
                        request,
                        entry,
                        member_stream,
                        member_path,
                        member_size=member_size,
                        member_sha256=member_hash.hexdigest(),
                    )
                member_output_bytes = sum(part.size_bytes for part in member_parts)
                if (
                    len(staged) + len(member_parts) > _MAX_OOXML_OUTPUT_PARTS
                    or staged_bytes + member_output_bytes > _MAX_OOXML_OUTPUT_BYTES
                ):
                    _close_staged(member_parts)
                    raise SourcePreparationError("archive_output_limit", "source")
                for part in member_parts:
                    part.ordinal = len(staged)
                    part.locator = f"{locator} {part.locator}"
                    staged.append(part)
                staged_bytes += member_output_bytes
            return staged
        except Exception:
            _close_staged(staged)
            raise
        finally:
            archive.close()

    def _prepare_archive_member(
        self,
        parent_request: PreparedPartRequest,
        entry: ManifestEntry,
        stream: BinaryIO,
        member_path: str,
        *,
        member_size: int,
        member_sha256: str,
    ) -> list[_StagedPart]:
        suffix = PurePosixPath(member_path).suffix.casefold()
        if entry.format_category == "text" and suffix in _TEXT_MEDIA_TYPES:
            return self._prepare_text(stream, _TEXT_MEDIA_TYPES[suffix])
        if entry.format_category == "tabular" and suffix in {".csv", ".tsv"}:
            return self._prepare_tabular(stream, suffix)
        if entry.format_category == "structured_data" and suffix in _STRUCTURED_MEDIA_TYPES:
            member_request = PreparedPartRequest(
                workspace_id=parent_request.workspace_id,
                revision_id=parent_request.revision_id,
                entry_id=entry.entry_id,
                relative_path=member_path,
                format_category=entry.format_category,
                source_size_bytes=member_size,
                source_sha256=member_sha256,
            )
            return self._prepare_structured_data(member_request, stream, suffix)
        if entry.format_category == "image" and suffix in _IMAGE_MEDIA_TYPES:
            member_request = PreparedPartRequest(
                workspace_id=parent_request.workspace_id,
                revision_id=parent_request.revision_id,
                entry_id=entry.entry_id,
                relative_path=member_path,
                format_category=entry.format_category,
                source_size_bytes=member_size,
                source_sha256=member_sha256,
            )
            return self._prepare_image(member_request, stream, suffix)
        if entry.format_category == "pdf" and suffix == ".pdf":
            return self._prepare_pdf(stream)
        if _OOXML_CATEGORIES.get(suffix) == entry.format_category:
            return self._prepare_ooxml(stream, suffix)
        raise SourcePreparationError("archive_member_unsupported", f"archive member {member_path}")

    def _prepare_legacy_office(self, stream: BinaryIO, suffix: str) -> list[_StagedPart]:
        converter = self._legacy_converter
        try:
            available = converter is not None and converter.available()
        except Exception:
            available = False
        if not available or converter is None:
            raise SourcePreparationError("converter_unavailable", "source")
        try:
            converted = converter.convert(
                stream,
                suffix=suffix,
                deadline=self._policy.provider_timeout_seconds,
            )
        except LegacyConversionError as error:
            raise SourcePreparationError(error.code, "source") from None
        return self._prepare_ooxml(BytesIO(converted.content_bytes), converted.suffix)

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
        validated: list[tuple[_StagedPart, SourcePartPlan]] = []
        result: tuple[SourcePartPlan, ...] | None = None
        failure: SourcePreparationError | None = None
        current_locator = "source"
        try:
            priority, scheduling_class = source_priority(
                request.relative_path,
                request.format_category,
            )
            for part in staged:
                current_locator = part.locator
                content: bytes | None = None
                stream_failed = False
                try:
                    part.stream.seek(0)
                    candidate = part.stream.read(self._policy.max_part_bytes + 1)
                    if isinstance(candidate, bytes):
                        content = candidate
                    else:
                        stream_failed = True
                except Exception:
                    stream_failed = True
                if stream_failed or content is None:
                    failure = SourcePreparationError("prepared_part_unavailable", part.locator)
                    break
                if len(content) != part.size_bytes or hashlib.sha256(content).hexdigest() != part.sha256:
                    failure = SourcePreparationError("prepared_part_changed", part.locator)
                    break
                identity = _part_identity(
                    workspace_id=request.workspace_id,
                    revision_id=request.revision_id,
                    entry_id=request.entry_id,
                    source_identity_sha256=hashlib.sha256(request.relative_path.encode("utf-8")).hexdigest(),
                    policy_version=self._policy.policy_version,
                    schema_version=self._policy.schema_version,
                    source_sha256=request.source_sha256,
                    part_sha256=part.sha256,
                    locator=part.locator,
                    media_type=part.media_type,
                )
                part_id = f"part_{identity}"
                candidate_plan = SourcePartPlan(
                    part_id=part_id,
                    workspace_id=request.workspace_id,
                    revision_id=request.revision_id,
                    entry_id=request.entry_id,
                    relative_path=request.relative_path,
                    source_sha256=request.source_sha256,
                    part_sha256=part.sha256,
                    ordinal=part.ordinal,
                    scheduling_rank=part.ordinal,
                    locator=part.locator,
                    media_type=part.media_type,
                    size_bytes=part.size_bytes,
                    scheduling_class=scheduling_class,
                    priority=priority,
                    preparation_policy_version=self._policy.policy_version,
                    preparation_schema_version=self._policy.schema_version,
                    state=PartState.PREPARED,
                    idempotency_key=f"prepare_{identity}",
                )
                validated.append((part, candidate_plan))

            plans: list[SourcePartPlan] = []
            if failure is None:
                for part, candidate_plan in validated:
                    current_locator = part.locator
                    artifact_failure_code: str | None = None
                    published_sha256: str | None = None
                    try:
                        part.stream.seek(0)
                        content = part.stream.read(self._policy.max_part_bytes + 1)
                    except Exception:
                        content = None
                    if (
                        not isinstance(content, bytes)
                        or len(content) != part.size_bytes
                        or hashlib.sha256(content).hexdigest() != part.sha256
                    ):
                        failure = SourcePreparationError("prepared_part_changed", part.locator)
                        break
                    try:
                        published_sha256 = self._artifact_store.publish_part(
                            request.workspace_id,
                            candidate_plan.part_id,
                            content,
                            expected_sha256=part.sha256,
                        )
                    except ArtifactBoundaryError as error:
                        artifact_failure_code = error.code
                    except Exception:
                        artifact_failure_code = "artifact_publish_failed"
                    if artifact_failure_code is not None:
                        failure = SourcePreparationError(artifact_failure_code, part.locator)
                        break
                    if published_sha256 != part.sha256:
                        failure = SourcePreparationError("artifact_hash_mismatch", part.locator)
                        break
                    plans.append(candidate_plan)
            if failure is None:
                result = tuple(
                    plans[index].model_copy(update={"scheduling_rank": rank})
                    for rank, index in enumerate(representative_ordinals(len(plans)))
                )
        except SourcePreparationError as error:
            failure = SourcePreparationError(error.code, error.locator)
        except Exception:
            failure = SourcePreparationError("prepared_part_publication_failed", current_locator)
        cleanup_failed = _close_staged(staged)
        if failure is not None:
            raise failure
        if cleanup_failed:
            raise SourcePreparationError("prepared_part_cleanup_failed", current_locator)
        if result is None:
            raise SourcePreparationError("prepared_part_publication_failed", current_locator)
        return result


def _iter_tabular_rows(
    wrapper: TextIOWrapper,
    *,
    delimiter: str,
    max_row_bytes: int,
) -> Iterator[tuple[int, bytes]]:
    """Yield exact encoded records using only request-local, bounded state."""

    row_number = 1
    row_buffer = bytearray()
    in_quotes = False
    at_field_start = True
    after_quote = False
    while True:
        line = wrapper.readline(max_row_bytes + 1)
        if line == "":
            break
        encoded = line.encode("utf-8")
        if len(row_buffer) + len(encoded) > max_row_bytes:
            raise SourcePreparationError("tabular_row_too_large", f"row {row_number}")
        row_buffer.extend(encoded)

        if line.endswith("\r\n"):
            body = line[:-2]
            terminated = True
        elif line.endswith(("\n", "\r")):
            body = line[:-1]
            terminated = True
        else:
            body = line
            terminated = False

        index = 0
        while index < len(body):
            character = body[index]
            if in_quotes:
                if character == '"':
                    if index + 1 < len(body) and body[index + 1] == '"':
                        index += 2
                        continue
                    in_quotes = False
                    after_quote = True
                index += 1
                continue
            if after_quote:
                if character != delimiter:
                    raise _TabularMalformedError
                after_quote = False
                at_field_start = True
                index += 1
                continue
            if character == delimiter:
                at_field_start = True
            elif character == '"':
                if not at_field_start:
                    raise _TabularMalformedError
                in_quotes = True
                at_field_start = False
            else:
                at_field_start = False
            index += 1

        if terminated and not in_quotes:
            yield row_number, bytes(row_buffer)
            row_number += 1
            row_buffer = bytearray()
            at_field_start = True
            after_quote = False

    if in_quotes:
        raise _TabularMalformedError
    if row_buffer:
        yield row_number, bytes(row_buffer)


def _new_spool(code: str, locator: str) -> BinaryIO:
    spool: BinaryIO | None = None
    failed = False
    try:
        spool = SpooledTemporaryFile(max_size=_SPOOL_MEMORY_BYTES, mode="w+b")
    except Exception:
        failed = True
    if failed or spool is None:
        raise SourcePreparationError(code, locator)
    return spool


def _close_stream(stream: BinaryIO) -> None:
    try:
        stream.close()
    except Exception:
        pass


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


def _close_staged(staged: list[_StagedPart]) -> bool:
    failed = False
    for part in staged:
        try:
            part.close()
        except Exception:
            failed = True
    return failed


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


def _preflight_json(document: str) -> None:
    depth = 0
    nodes = 0
    tokens = 0
    index = 0

    def consume(*, node: bool = False) -> None:
        nonlocal nodes, tokens
        tokens += 1
        if tokens > _MAX_STRUCTURE_TOKENS:
            raise _StructuredDataLimitError
        if node:
            nodes += 1
            if nodes > _MAX_STRUCTURE_NODES:
                raise _StructuredDataLimitError

    while index < len(document):
        character = document[index]
        if character.isspace():
            index += 1
            continue
        if character in "[{":
            consume(node=True)
            depth += 1
            if depth > _MAX_STRUCTURE_DEPTH:
                raise _StructuredDataLimitError
            index += 1
            continue
        if character in "]}":
            consume()
            depth -= 1
            if depth < 0:
                raise _StructuredDataMalformedError
            index += 1
            continue
        if character in ",:":
            consume()
            index += 1
            continue
        if character == '"':
            consume(node=True)
            index = _scan_json_string(document, index + 1)
            continue
        end = index
        while end < len(document) and not (document[end].isspace() or document[end] in "[]{},:"):
            end += 1
        token = document[index:end]
        if not token:
            raise _StructuredDataMalformedError
        consume(node=True)
        if token in {"NaN", "Infinity", "-Infinity"}:
            raise _StructuredDataMalformedError
        index = end
    if depth != 0:
        raise _StructuredDataMalformedError


def _scan_json_string(document: str, index: int) -> int:
    encoded_bytes = 0
    while index < len(document):
        character = document[index]
        if character == '"':
            return index + 1
        if ord(character) < 0x20:
            raise _StructuredDataMalformedError
        if character != "\\":
            encoded_bytes += len(character.encode("utf-8"))
            index += 1
        else:
            index += 1
            if index >= len(document):
                raise _StructuredDataMalformedError
            escape = document[index]
            if escape == "u":
                digits = document[index + 1 : index + 5]
                if len(digits) != 4 or any(item not in "0123456789abcdefABCDEF" for item in digits):
                    raise _StructuredDataMalformedError
                encoded_bytes += 4
                index += 5
            elif escape in '"\\/bfnrt':
                encoded_bytes += 1
                index += 1
            else:
                raise _StructuredDataMalformedError
        if encoded_bytes > _MAX_STRUCTURE_STRING_BYTES:
            raise _StructuredDataLimitError
    raise _StructuredDataMalformedError


def _preflight_yaml(document: str) -> None:
    depth = 0
    nodes = 0
    tokens = 0
    for event in yaml.parse(document, Loader=yaml.SafeLoader):
        tokens += 1
        if tokens > _MAX_STRUCTURE_TOKENS:
            raise _StructuredDataLimitError
        if isinstance(event, (MappingStartEvent, SequenceStartEvent)):
            nodes += 1
            depth += 1
            if nodes > _MAX_STRUCTURE_NODES or depth > _MAX_STRUCTURE_DEPTH:
                raise _StructuredDataLimitError
        elif isinstance(event, (MappingEndEvent, SequenceEndEvent)):
            depth -= 1
            if depth < 0:
                raise _StructuredDataMalformedError
        elif isinstance(event, (ScalarEvent, AliasEvent)):
            nodes += 1
            if nodes > _MAX_STRUCTURE_NODES:
                raise _StructuredDataLimitError
            value = getattr(event, "value", "")
            if len(value.encode("utf-8")) > _MAX_STRUCTURE_STRING_BYTES:
                raise _StructuredDataLimitError
    if depth != 0:
        raise _StructuredDataMalformedError


def _reject_json_constant(_value: str) -> None:
    raise _StructuredDataMalformedError


def _validate_structure(document: Any) -> None:
    seen: set[int] = set()
    nodes = 0

    def visit(value: Any, depth: int) -> None:
        nonlocal nodes
        if depth > _MAX_STRUCTURE_DEPTH:
            raise _StructuredDataLimitError
        nodes += 1
        if nodes > _MAX_STRUCTURE_NODES:
            raise _StructuredDataLimitError
        if isinstance(value, (dict, list, tuple, set)):
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
        if isinstance(value, dict):
            for key, item in value.items():
                visit(key, depth + 1)
                visit(item, depth + 1)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item, depth + 1)
        elif isinstance(value, float) and not math.isfinite(value):
            raise _StructuredDataMalformedError
        elif isinstance(value, str) and len(value.encode("utf-8")) > _MAX_STRUCTURE_STRING_BYTES:
            raise _StructuredDataLimitError

    visit(document, 0)


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
    workspace_id: str,
    revision_id: str,
    entry_id: str,
    source_identity_sha256: str,
    policy_version: str,
    schema_version: str,
    source_sha256: str,
    part_sha256: str,
    locator: str,
    media_type: str,
) -> str:
    encoded = json.dumps(
        {
            "entry_id": entry_id,
            "locator": locator,
            "media_type": media_type,
            "part_sha256": part_sha256,
            "policy_version": policy_version,
            "schema_version": schema_version,
            "revision_id": revision_id,
            "source_identity_sha256": source_identity_sha256,
            "source_sha256": source_sha256,
            "workspace_id": workspace_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
