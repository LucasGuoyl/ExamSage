"""Deterministic, non-semantic OOXML relationship preparation."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import posixpath
from pathlib import PurePosixPath
from typing import BinaryIO
import zipfile

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

from exam_predictor.workspace.archive import ArchiveInspector
from exam_predictor.workspace.models import ScanPolicy, SourceState


_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_S_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_MAX_XML_BYTES = 8 * 1024 * 1024
_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_MAX_OUTPUT_PARTS = 256
_MAX_XML_NODES = 50_000
_MAX_XML_DEPTH = 64
_MAX_XML_TEXT_CHARS = 4 * 1024 * 1024
_MAX_TOTAL_XML_BYTES = 32 * 1024 * 1024
_MAX_TOTAL_XML_NODES = 200_000
_MAX_TOTAL_XML_TEXT_CHARS = 8 * 1024 * 1024
_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}


class OoxmlPreparationError(RuntimeError):
    def __init__(self, code: str, locator: str = "source") -> None:
        self.code = code
        self.locator = locator
        super().__init__(code)


@dataclass(frozen=True)
class OoxmlPart:
    locator: str
    media_type: str
    content: bytes


@dataclass(frozen=True)
class _Relationship:
    target: str
    relationship_type: str


class _Package:
    def __init__(self, stream: BinaryIO, *, scan_policy: ScanPolicy, max_part_bytes: int) -> None:
        self._stream = stream
        self._scan_policy = scan_policy
        self._max_part_bytes = max_part_bytes
        self._archive: zipfile.ZipFile | None = None
        self._names: set[str] = set()
        self._output_bytes = 0
        self._output_parts = 0
        self._emitted_media: set[str] = set()
        self._parsed_xml_bytes = 0
        self._parsed_xml_nodes = 0
        self._parsed_xml_text_chars = 0

    def __enter__(self) -> _Package:
        try:
            self._stream.seek(0)
            members = ArchiveInspector(self._scan_policy).inspect(
                self._stream,
                parent_entry_id="ooxml_container",
            )
            if not members or any(member.state is SourceState.FAILED for member in members):
                raise OoxmlPreparationError("ooxml_archive_unsafe")
            self._stream.seek(0)
            archive = zipfile.ZipFile(self._stream, "r")
            names: set[str] = set()
            for info in archive.infolist():
                normalized = _normalize_member_name(info.filename)
                if normalized in names:
                    archive.close()
                    raise OoxmlPreparationError("ooxml_archive_unsafe")
                names.add(normalized)
            self._archive = archive
            self._names = names
            return self
        except OoxmlPreparationError:
            raise
        except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile):
            raise OoxmlPreparationError("ooxml_corrupt") from None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._archive is not None:
            self._archive.close()

    def contains(self, name: str) -> bool:
        return _normalize_member_name(name) in self._names

    def read(self, name: str, *, maximum: int | None = None) -> bytes:
        archive = self._require_archive()
        normalized = _normalize_member_name(name)
        limit = self._max_part_bytes if maximum is None else min(maximum, self._max_part_bytes)
        try:
            info = archive.getinfo(normalized)
            if info.is_dir() or info.file_size > limit:
                raise OoxmlPreparationError("ooxml_part_too_large", "source")
            with archive.open(info, "r") as member:
                content = member.read(limit + 1)
                if len(content) > limit or member.read(1):
                    raise OoxmlPreparationError("ooxml_part_too_large", "source")
            if len(content) != info.file_size:
                raise OoxmlPreparationError("ooxml_member_changed", "source")
            return content
        except OoxmlPreparationError:
            raise
        except (KeyError, OSError, RuntimeError, zipfile.BadZipFile):
            raise OoxmlPreparationError("ooxml_corrupt", "source") from None

    def xml(self, name: str):
        content = self.read(name, maximum=_MAX_XML_BYTES)
        self._parsed_xml_bytes += len(content)
        if self._parsed_xml_bytes > _MAX_TOTAL_XML_BYTES:
            raise OoxmlPreparationError("ooxml_xml_limit", "source")
        try:
            depth = 0
            nodes = 0
            text_chars = 0
            for event, element in ElementTree.iterparse(
                BytesIO(content),
                events=("start", "end"),
            ):
                if event == "start":
                    depth += 1
                    nodes += 1
                    self._parsed_xml_nodes += 1
                    if (
                        nodes > _MAX_XML_NODES
                        or self._parsed_xml_nodes > _MAX_TOTAL_XML_NODES
                        or depth > _MAX_XML_DEPTH
                    ):
                        raise OoxmlPreparationError("ooxml_xml_limit", "source")
                else:
                    text_chars += len(element.text or "") + len(element.tail or "")
                    self._parsed_xml_text_chars += len(element.text or "") + len(
                        element.tail or ""
                    )
                    if (
                        text_chars > _MAX_XML_TEXT_CHARS
                        or self._parsed_xml_text_chars > _MAX_TOTAL_XML_TEXT_CHARS
                    ):
                        raise OoxmlPreparationError("ooxml_xml_limit", "source")
                    depth -= 1
                    element.clear()
            return ElementTree.fromstring(content)
        except OoxmlPreparationError:
            raise
        except (DefusedXmlException, ElementTree.ParseError, ValueError, TypeError):
            raise OoxmlPreparationError("ooxml_xml_invalid", "source") from None

    def part(self, locator: str, media_type: str, content: bytes) -> OoxmlPart:
        output_limit = min(
            _MAX_OUTPUT_BYTES,
            self._scan_policy.max_archive_expanded_bytes,
            self._max_part_bytes * 32,
        )
        if len(content) > self._max_part_bytes:
            raise OoxmlPreparationError("ooxml_part_too_large", locator)
        if (
            self._output_parts >= _MAX_OUTPUT_PARTS
            or self._output_bytes + len(content) > output_limit
        ):
            raise OoxmlPreparationError("ooxml_output_limit", "source")
        self._output_parts += 1
        self._output_bytes += len(content)
        return OoxmlPart(locator, media_type, content)

    def claim_media(self, target: str) -> bool:
        if target in self._emitted_media:
            return False
        self._emitted_media.add(target)
        return True

    def _require_archive(self) -> zipfile.ZipFile:
        if self._archive is None:
            raise OoxmlPreparationError("ooxml_corrupt")
        return self._archive


def prepare_ooxml(
    stream: BinaryIO,
    *,
    suffix: str,
    scan_policy: ScanPolicy,
    max_part_bytes: int,
) -> tuple[OoxmlPart, ...]:
    try:
        with _Package(stream, scan_policy=scan_policy, max_part_bytes=max_part_bytes) as package:
            if suffix == ".pptx":
                result = _prepare_pptx(package)
            elif suffix == ".docx":
                result = _prepare_docx(package)
            elif suffix == ".xlsx":
                result = _prepare_xlsx(package, max_part_bytes=max_part_bytes)
            else:
                raise OoxmlPreparationError("ooxml_format_unsupported")
    except OoxmlPreparationError:
        raise
    except BaseException:
        raise OoxmlPreparationError("ooxml_corrupt") from None
    if not result:
        raise OoxmlPreparationError("ooxml_empty")
    return tuple(result)


def _prepare_pptx(package: _Package) -> list[OoxmlPart]:
    presentation_name = "ppt/presentation.xml"
    presentation = package.xml(presentation_name)
    relationships = _relationships(package, presentation_name)
    parts: list[OoxmlPart] = []
    for slide_number, slide_id in enumerate(presentation.iter(f"{{{_P_NS}}}sldId"), start=1):
        relation_id = slide_id.get(f"{{{_DOC_REL_NS}}}id")
        slide_relationship = relationships.get(relation_id or "")
        if slide_relationship is None:
            raise OoxmlPreparationError("ooxml_relationship_invalid", f"slide {slide_number}")
        slide_name = slide_relationship.target
        slide = package.xml(slide_name)
        text = _joined_text(slide, f"{{{_A_NS}}}t")
        if text:
            parts.append(
                package.part(f"slide {slide_number}", "text/plain", text.encode("utf-8"))
            )
        parts.extend(_image_parts(package, slide_name, slide, f"slide {slide_number}"))
    return parts


def _prepare_docx(package: _Package) -> list[OoxmlPart]:
    document_name = "word/document.xml"
    document = package.xml(document_name)
    relationships = _relationships(package, document_name)
    sections: list[tuple[str, list[str], list[str]]] = []
    title = "Untitled"
    has_heading = False
    lines: list[str] = []
    image_targets: list[str] = []

    def finish() -> None:
        nonlocal lines, image_targets
        if has_heading or lines or image_targets:
            sections.append((title, lines, image_targets))
        lines = []
        image_targets = []

    for paragraph in document.iter(f"{{{_W_NS}}}p"):
        text = _joined_text(paragraph, f"{{{_W_NS}}}t")
        style = paragraph.find(f".//{{{_W_NS}}}pStyle")
        style_name = "" if style is None else style.get(f"{{{_W_NS}}}val", "")
        if text and style_name.casefold().startswith("heading"):
            finish()
            title = text
            has_heading = True
        elif text:
            lines.append(text)
        for blip in paragraph.iter(f"{{{_A_NS}}}blip"):
            relation_id = blip.get(f"{{{_DOC_REL_NS}}}embed")
            relationship = relationships.get(relation_id or "")
            if (
                relationship is not None
                and relationship.relationship_type.endswith("/image")
                and relationship.target not in image_targets
            ):
                image_targets.append(relationship.target)
    finish()

    parts: list[OoxmlPart] = []
    for section_number, (section_title, section_lines, targets) in enumerate(sections, start=1):
        text_lines = [section_title, *section_lines]
        parts.append(
            package.part(
                f"section {section_number}",
                "text/plain",
                "\n".join(text_lines).encode("utf-8"),
            )
        )
        parts.extend(_targets_to_image_parts(package, targets, f"section {section_number}"))
    footnote_relationships = [
        relationship
        for relationship in relationships.values()
        if relationship.relationship_type.endswith("/footnotes")
    ]
    if len(footnote_relationships) > 1:
        raise OoxmlPreparationError("ooxml_relationship_invalid", "source")
    if footnote_relationships:
        footnotes_name = footnote_relationships[0].target
        footnotes = package.xml(footnotes_name)
        footnote_text = _joined_text(footnotes, f"{{{_W_NS}}}t")
        if footnote_text:
            parts.append(package.part("footnotes 1", "text/plain", footnote_text.encode("utf-8")))
        parts.extend(_image_parts(package, footnotes_name, footnotes, "footnotes 1"))
    return parts


def _prepare_xlsx(package: _Package, *, max_part_bytes: int) -> list[OoxmlPart]:
    workbook_name = "xl/workbook.xml"
    workbook = package.xml(workbook_name)
    relationships = _relationships(package, workbook_name)
    shared_strings = _shared_strings(package)
    parts: list[OoxmlPart] = []
    for sheet_number, sheet in enumerate(workbook.iter(f"{{{_S_NS}}}sheet"), start=1):
        relation_id = sheet.get(f"{{{_DOC_REL_NS}}}id")
        worksheet_relationship = relationships.get(relation_id or "")
        if worksheet_relationship is None:
            raise OoxmlPreparationError("ooxml_relationship_invalid", f"sheet {sheet_number}")
        worksheet_name = worksheet_relationship.target
        worksheet = package.xml(worksheet_name)
        rows: list[tuple[int, bytes]] = []
        for row in worksheet.iter(f"{{{_S_NS}}}row"):
            row_number = _positive_int(row.get("r"), default=len(rows) + 1)
            lines: list[str] = []
            for cell in row.iter(f"{{{_S_NS}}}c"):
                reference = (cell.get("r") or "cell").strip() or "cell"
                formula = _child_text(cell, f"{{{_S_NS}}}f")
                value = _cell_value(cell, shared_strings)
                if formula is not None:
                    lines.append(f"{reference}\tformula={formula}\tvalue={value or ''}")
                else:
                    lines.append(f"{reference}\t{value or ''}")
            if lines:
                rows.append((row_number, ("\n".join(lines) + "\n").encode("utf-8")))
        parts.extend(
            _chunk_sheet(
                package,
                sheet_number,
                rows,
                max_part_bytes=max_part_bytes,
            )
        )
        image_prefix = f"sheet {sheet_number}"
        parts.extend(_image_parts(package, worksheet_name, worksheet, image_prefix))
        worksheet_relationships = _relationships(package, worksheet_name)
        drawing_ids = {
            element.get(f"{{{_DOC_REL_NS}}}id")
            for element in worksheet.iter(f"{{{_S_NS}}}drawing")
        }
        for drawing_id in sorted(candidate for candidate in drawing_ids if candidate):
            relationship = worksheet_relationships.get(drawing_id)
            if relationship is None or not relationship.relationship_type.endswith("/drawing"):
                raise OoxmlPreparationError("ooxml_relationship_invalid", image_prefix)
            drawing = package.xml(relationship.target)
            parts.extend(_image_parts(package, relationship.target, drawing, image_prefix))
    return parts


def _shared_strings(package: _Package) -> tuple[str, ...]:
    name = "xl/sharedStrings.xml"
    if not package.contains(name):
        return ()
    root = package.xml(name)
    return tuple(_joined_text(item, f"{{{_S_NS}}}t") for item in root.iter(f"{{{_S_NS}}}si"))


def _cell_value(cell, shared_strings: tuple[str, ...]) -> str | None:
    cell_type = cell.get("t", "")
    if cell_type == "inlineStr":
        return _joined_text(cell, f"{{{_S_NS}}}t")
    value = _child_text(cell, f"{{{_S_NS}}}v")
    if cell_type == "s" and value is not None:
        try:
            index = int(value)
            if index < 0:
                raise IndexError
            return shared_strings[index]
        except (ValueError, IndexError):
            raise OoxmlPreparationError("ooxml_cell_invalid") from None
    return value


def _chunk_sheet(
    package: _Package,
    sheet_number: int,
    rows: list[tuple[int, bytes]],
    *,
    max_part_bytes: int,
) -> list[OoxmlPart]:
    parts: list[OoxmlPart] = []
    current = bytearray()
    first_row = 0
    last_row = 0
    for row_number, content in rows:
        if len(content) > max_part_bytes:
            raise OoxmlPreparationError(
                "ooxml_part_too_large",
                f"sheet {sheet_number} row {row_number}",
            )
        if current and len(current) + len(content) > max_part_bytes:
            parts.append(
                package.part(
                    f"sheet {sheet_number} rows {first_row}-{last_row}",
                    "text/tab-separated-values",
                    bytes(current),
                )
            )
            current = bytearray()
        if not current:
            first_row = row_number
        current.extend(content)
        last_row = row_number
    if current:
        parts.append(
            package.part(
                f"sheet {sheet_number} rows {first_row}-{last_row}",
                "text/tab-separated-values",
                bytes(current),
            )
        )
    return parts


def _image_parts(package: _Package, owner_name: str, root, locator_prefix: str) -> list[OoxmlPart]:
    relationships = _relationships(package, owner_name)
    targets: list[str] = []
    for blip in root.iter(f"{{{_A_NS}}}blip"):
        relation_id = blip.get(f"{{{_DOC_REL_NS}}}embed")
        relationship = relationships.get(relation_id or "")
        if (
            relationship is not None
            and relationship.relationship_type.endswith("/image")
            and relationship.target not in targets
        ):
            targets.append(relationship.target)
    return _targets_to_image_parts(package, targets, locator_prefix)


def _targets_to_image_parts(package: _Package, targets: list[str], locator_prefix: str) -> list[OoxmlPart]:
    parts: list[OoxmlPart] = []
    for image_number, target in enumerate(targets, start=1):
        if not package.claim_media(target):
            continue
        media_type = _MEDIA_TYPES.get(PurePosixPath(target).suffix.casefold())
        if media_type is None:
            raise OoxmlPreparationError("ooxml_media_unsupported", locator_prefix)
        parts.append(
            package.part(
                f"{locator_prefix} image {image_number}",
                media_type,
                package.read(target),
            )
        )
    return parts


def _relationships(package: _Package, owner_name: str) -> dict[str, _Relationship]:
    owner = PurePosixPath(owner_name)
    relationship_name = (owner.parent / "_rels" / f"{owner.name}.rels").as_posix()
    if not package.contains(relationship_name):
        return {}
    root = package.xml(relationship_name)
    relationships: dict[str, _Relationship] = {}
    for relationship in root.iter(f"{{{_REL_NS}}}Relationship"):
        relation_id = relationship.get("Id")
        target = relationship.get("Target")
        if not relation_id or not target or relationship.get("TargetMode", "").casefold() == "external":
            continue
        resolved = _resolve_target(owner_name, target)
        if not package.contains(resolved):
            raise OoxmlPreparationError("ooxml_relationship_invalid", "source")
        if relation_id in relationships:
            raise OoxmlPreparationError("ooxml_relationship_invalid", "source")
        relationships[relation_id] = _Relationship(
            resolved,
            relationship.get("Type", ""),
        )
    return relationships


def _resolve_target(owner_name: str, target: str) -> str:
    if "\\" in target or target.startswith(("/", "\\")):
        raise OoxmlPreparationError("ooxml_relationship_invalid", "source")
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(owner_name), target))
    if resolved in {"", ".", ".."} or resolved.startswith("../"):
        raise OoxmlPreparationError("ooxml_relationship_invalid", "source")
    return _normalize_member_name(resolved)


def _normalize_member_name(name: str) -> str:
    if "\\" in name:
        raise OoxmlPreparationError("ooxml_archive_unsafe")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        raise OoxmlPreparationError("ooxml_archive_unsafe")
    normalized = path.as_posix()
    if normalized != name.rstrip("/"):
        raise OoxmlPreparationError("ooxml_archive_unsafe")
    return normalized


def _joined_text(root, tag: str) -> str:
    values = [element.text.strip() for element in root.iter(tag) if element.text and element.text.strip()]
    return "\n".join(values)


def _child_text(root, tag: str) -> str | None:
    element = root.find(tag)
    if element is None or element.text is None:
        return None
    value = element.text.strip()
    return value or None


def _positive_int(value: str | None, *, default: int) -> int:
    try:
        parsed = int(value or "")
    except ValueError:
        return default
    return parsed if parsed > 0 else default
