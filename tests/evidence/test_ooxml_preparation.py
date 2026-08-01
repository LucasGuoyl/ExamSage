from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
import zipfile

from PIL import Image
import pytest

from exam_predictor.evidence.artifacts import EvidenceArtifactStore
from exam_predictor.evidence.preparation import (
    ArchivePreviewAuthority,
    PreparedPartRequest,
    SourcePartPreparer,
    SourcePreparationError,
)
from exam_predictor.workspace.models import ManifestEntry, SourceState


WORKSPACE_ID = "workspace_ooxml_00000000000000001"
REVISION_ID = "revision_ooxml_00000000000000001"
ENTRY_ID = "entry_ooxml_000000000000000001"


@pytest.fixture
def artifact_store(tmp_path: Path):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = EvidenceArtifactStore(root)
    try:
        yield store
    finally:
        store.close()


def _archive(files: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name, content in files.items():
            package.writestr(name, content)
    return output.getvalue()


def _archive_entries(files: list[tuple[str, bytes]]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name, content in files:
            package.writestr(name, content)
    return output.getvalue()


def _png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


def _request(
    content: bytes,
    *,
    relative_path: str,
    format_category: str,
    archive_previews: tuple[ArchivePreviewAuthority, ...] = (),
) -> PreparedPartRequest:
    return PreparedPartRequest(
        workspace_id=WORKSPACE_ID,
        revision_id=REVISION_ID,
        entry_id=ENTRY_ID,
        relative_path=relative_path,
        format_category=format_category,
        source_size_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
        archive_previews=archive_previews,
    )


def _published(store: EvidenceArtifactStore, plans) -> list[tuple[str, str, bytes]]:
    result: list[tuple[str, str, bytes]] = []
    for plan in plans:
        with store.open_part(WORKSPACE_ID, plan.part_id) as stream:
            result.append((plan.locator, plan.media_type, stream.read()))
    return result


def test_pptx_preserves_slide_order_and_embedded_image_relationship_once(artifact_store):
    content = _archive(
        {
            "ppt/presentation.xml": b"""
                <p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                  <p:sldIdLst><p:sldId id="1" r:id="rId2"/><p:sldId id="2" r:id="rId1"/></p:sldIdLst>
                </p:presentation>
            """,
            "ppt/_rels/presentation.xml.rels": b"""
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                  <Relationship Id="rId1" Target="slides/slide2.xml"/>
                  <Relationship Id="rId2" Target="slides/slide1.xml"/>
                </Relationships>
            """,
            "ppt/slides/slide1.xml": b"""
                <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                    xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                  <a:t>First linked slide</a:t><a:blip r:embed="rImg1"/><a:blip r:embed="rImg1"/>
                </p:sld>
            """,
            "ppt/slides/_rels/slide1.xml.rels": b"""
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                  <Relationship Id="rImg1" Target="../media/image1.png"
                    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"/>
                </Relationships>
            """,
            "ppt/slides/slide2.xml": b"""
                <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                    xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
                  <a:t>Second linked slide</a:t>
                </p:sld>
            """,
            "ppt/media/image1.png": _png(),
        }
    )

    plans = SourcePartPreparer(artifact_store).prepare(
        _request(content, relative_path="lectures/week1.pptx", format_category="presentation"),
        BytesIO(content),
    )

    published = _published(artifact_store, plans)
    assert [(locator, media_type) for locator, media_type, _ in published] == [
        ("slide 1", "text/plain"),
        ("slide 1 image 1", "image/png"),
        ("slide 2", "text/plain"),
    ]
    assert b"First linked slide" in published[0][2]
    assert published[1][2] == _png()
    assert b"Second linked slide" in published[2][2]


def test_docx_preserves_heading_order_and_embedded_image_context(artifact_store):
    content = _archive(
        {
            "word/document.xml": b"""
                <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
                    xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                  <w:body>
                    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Unit One</w:t></w:r></w:p>
                    <w:p><w:r><w:t>Definition body</w:t></w:r><a:blip r:embed="rImg5"/></w:p>
                    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Unit Two</w:t></w:r></w:p>
                  </w:body>
                </w:document>
            """,
            "word/_rels/document.xml.rels": b"""
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                  <Relationship Id="rImg5" Target="media/diagram.png"
                    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"/>
                </Relationships>
            """,
            "word/media/diagram.png": _png(),
        }
    )

    plans = SourcePartPreparer(artifact_store).prepare(
        _request(content, relative_path="notes/course.docx", format_category="document"),
        BytesIO(content),
    )

    published = _published(artifact_store, plans)
    assert [locator for locator, _, _ in published] == [
        "section 1",
        "section 1 image 1",
        "section 2",
    ]
    assert b"Unit One\nDefinition body" in published[0][2]
    assert published[1][2] == _png()
    assert b"Unit Two" in published[2][2]


def test_xlsx_preserves_sheet_order_formula_and_displayed_value(artifact_store):
    content = _archive(
        {
            "xl/workbook.xml": b"""
                <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                  <sheets><sheet name="Marks" sheetId="1" r:id="rId1"/></sheets>
                </workbook>
            """,
            "xl/_rels/workbook.xml.rels": b"""
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                  <Relationship Id="rId1" Target="worksheets/sheet1.xml"/>
                </Relationships>
            """,
            "xl/worksheets/sheet1.xml": b"""
                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
                  <sheetData><row r="1">
                    <c r="A1" t="inlineStr"><is><t>Topic</t></is></c>
                    <c r="B1"><f>SUM(1,2)</f><v>3</v></c>
                  </row></sheetData>
                </worksheet>
            """,
        }
    )

    plans = SourcePartPreparer(artifact_store).prepare(
        _request(content, relative_path="data/marks.xlsx", format_category="spreadsheet"),
        BytesIO(content),
    )

    published = _published(artifact_store, plans)
    assert [(locator, media_type) for locator, media_type, _ in published] == [
        ("sheet 1 rows 1-1", "text/tab-separated-values")
    ]
    assert published[0][2] == b"A1\tTopic\nB1\tformula=SUM(1,2)\tvalue=3\n"


def _preview(source: bytes, *, size_bytes: int = 4) -> ArchivePreviewAuthority:
    with zipfile.ZipFile(BytesIO(source), "r") as archive:
        info = archive.infolist()[0]
    entry = ManifestEntry(
        entry_id="member_preview_00000000000000001",
        workspace_id=WORKSPACE_ID,
        relative_path="bundle.zip",
        item_kind="archive_member",
        format_category="text",
        size_bytes=size_bytes,
        sha256=None,
        state=SourceState.PENDING_APPROVAL,
        included=False,
        inclusion_reason="archive_preview",
        archive_parent_entry_id=ENTRY_ID,
        archive_member_path="member.txt",
        archive_member_index=1,
        archive_member_crc32=info.CRC,
        archive_member_compressed_bytes=info.compress_size,
    )
    return ArchivePreviewAuthority(
        workspace_id=WORKSPACE_ID,
        revision_id=REVISION_ID,
        parent_entry_id=ENTRY_ID,
        parent_source_sha256=hashlib.sha256(source).hexdigest(),
        entry=entry,
        approved=True,
    )


def test_zip_prepares_only_exact_authorized_preview_members(artifact_store):
    content = _archive({"member.txt": b"safe", "unapproved.txt": b"hidden"})

    plans = SourcePartPreparer(artifact_store).prepare(
        _request(
            content,
            relative_path="bundle.zip",
            format_category="archive",
            archive_previews=(_preview(content),),
        ),
        BytesIO(content),
    )

    published = _published(artifact_store, plans)
    assert published == [("archive member member.txt lines 1-1", "text/plain", b"safe")]


def test_zip_member_metadata_change_fails_before_publication(artifact_store):
    content = _archive({"member.txt": b"safe"})

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(artifact_store).prepare(
            _request(
                content,
                relative_path="bundle.zip",
                format_category="archive",
                archive_previews=(
                    _preview(content, size_bytes=999),
                ),
            ),
            BytesIO(content),
        )

    assert caught.value.code == "archive_member_changed"
    assert caught.value.locator == "archive member member.txt"


def test_zip_rejects_duplicate_normalized_member_names_before_publication(artifact_store):
    with pytest.warns(UserWarning, match="Duplicate name"):
        content = _archive_entries(
            [("member.txt", b"safe"), ("member.txt", b"evil")]
        )

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(artifact_store).prepare(
            _request(
                content,
                relative_path="bundle.zip",
                format_category="archive",
                archive_previews=(_preview(content),),
            ),
            BytesIO(content),
        )

    assert caught.value.code == "archive_unsafe"


def test_zip_applies_one_final_output_budget_across_all_authorized_members(artifact_store):
    files = {f"member-{index:03d}.txt": b"x" for index in range(257)}
    content = _archive(files)
    previews = []
    with zipfile.ZipFile(BytesIO(content), "r") as archive:
        for index, info in enumerate(archive.infolist(), start=1):
            entry = ManifestEntry(
                entry_id=f"member_preview_{index:020d}",
                workspace_id=WORKSPACE_ID,
                relative_path="bundle.zip",
                item_kind="archive_member",
                format_category="text",
                size_bytes=info.file_size,
                sha256=None,
                state=SourceState.PENDING_APPROVAL,
                included=False,
                inclusion_reason="archive_preview",
                archive_parent_entry_id=ENTRY_ID,
                archive_member_path=info.filename,
                archive_member_index=index,
                archive_member_crc32=info.CRC,
                archive_member_compressed_bytes=info.compress_size,
            )
            previews.append(
                ArchivePreviewAuthority(
                    workspace_id=WORKSPACE_ID,
                    revision_id=REVISION_ID,
                    parent_entry_id=ENTRY_ID,
                    parent_source_sha256=hashlib.sha256(content).hexdigest(),
                    entry=entry,
                    approved=True,
                )
            )

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(artifact_store).prepare(
            _request(
                content,
                relative_path="bundle.zip",
                format_category="archive",
                archive_previews=tuple(previews),
            ),
            BytesIO(content),
        )

    assert (caught.value.code, caught.value.locator) == ("archive_output_limit", "source")
    assert artifact_store._registry is not None
    assert artifact_store._registry.get_workspace(WORKSPACE_ID) is None


def test_ooxml_rejects_invalid_embedded_image_bytes(artifact_store):
    content = _archive(
        {
            "ppt/presentation.xml": b"""
                <p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                  <p:sldIdLst><p:sldId id="1" r:id="rId1"/></p:sldIdLst>
                </p:presentation>
            """,
            "ppt/_rels/presentation.xml.rels": b"""
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                  <Relationship Id="rId1" Target="slides/slide1.xml"/>
                </Relationships>
            """,
            "ppt/slides/slide1.xml": b"""
                <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                    xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                  <a:blip r:embed="rImg1"/>
                </p:sld>
            """,
            "ppt/slides/_rels/slide1.xml.rels": b"""
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                  <Relationship Id="rImg1" Target="../media/image1.png"
                    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"/>
                </Relationships>
            """,
            "ppt/media/image1.png": b"not a png",
        }
    )

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(artifact_store).prepare(
            _request(content, relative_path="slides.pptx", format_category="presentation"),
            BytesIO(content),
        )

    assert caught.value.code == "image_magic_mismatch"


def test_xlsx_rejects_negative_shared_string_index(artifact_store):
    content = _archive(
        {
            "xl/workbook.xml": b"""
                <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                  <sheets><sheet name="Private" sheetId="1" r:id="rId1"/></sheets>
                </workbook>
            """,
            "xl/_rels/workbook.xml.rels": b"""
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                  <Relationship Id="rId1" Target="worksheets/sheet1.xml"/>
                </Relationships>
            """,
            "xl/sharedStrings.xml": b"""
                <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
                  <si><t>must not be selected</t></si>
                </sst>
            """,
            "xl/worksheets/sheet1.xml": b"""
                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
                  <sheetData><row r="1"><c r="A1" t="s"><v>-1</v></c></row></sheetData>
                </worksheet>
            """,
        }
    )

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(artifact_store).prepare(
            _request(content, relative_path="data.xlsx", format_category="spreadsheet"),
            BytesIO(content),
        )

    assert caught.value.code == "ooxml_cell_invalid"


def test_docx_includes_footnotes_with_stable_locator(artifact_store):
    content = _archive(
        {
            "word/document.xml": b"""
                <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                  <w:body><w:p><w:r><w:t>Main body</w:t></w:r></w:p></w:body>
                </w:document>
            """,
            "word/_rels/document.xml.rels": b"""
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                  <Relationship Id="rFootnotes" Target="footnotes.xml"
                    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes"/>
                </Relationships>
            """,
            "word/footnotes.xml": b"""
                <w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                  <w:footnote w:id="1"><w:p><w:r><w:t>Footnote evidence</w:t></w:r></w:p></w:footnote>
                </w:footnotes>
            """,
        }
    )

    plans = SourcePartPreparer(artifact_store).prepare(
        _request(content, relative_path="notes.docx", format_category="document"),
        BytesIO(content),
    )

    published = _published(artifact_store, plans)
    assert [locator for locator, _, _ in published] == ["section 1", "footnotes 1"]
    assert published[1][2] == b"Footnote evidence"


def test_xlsx_follows_standard_worksheet_drawing_image_chain(artifact_store):
    content = _archive(
        {
            "xl/workbook.xml": b"""
                <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                  <sheets><sheet name="Visual" sheetId="1" r:id="rId1"/></sheets>
                </workbook>
            """,
            "xl/_rels/workbook.xml.rels": b"""
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                  <Relationship Id="rId1" Target="worksheets/sheet1.xml"/>
                </Relationships>
            """,
            "xl/worksheets/sheet1.xml": b"""
                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                  <drawing r:id="rDraw1"/>
                </worksheet>
            """,
            "xl/worksheets/_rels/sheet1.xml.rels": b"""
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                  <Relationship Id="rDraw1" Target="../drawings/drawing1.xml"
                    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing"/>
                </Relationships>
            """,
            "xl/drawings/drawing1.xml": b"""
                <xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
                    xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                  <a:blip r:embed="rImg1"/>
                </xdr:wsDr>
            """,
            "xl/drawings/_rels/drawing1.xml.rels": b"""
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                  <Relationship Id="rImg1" Target="../media/image1.png"
                    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"/>
                </Relationships>
            """,
            "xl/media/image1.png": _png(),
        }
    )

    plans = SourcePartPreparer(artifact_store).prepare(
        _request(content, relative_path="visual.xlsx", format_category="spreadsheet"),
        BytesIO(content),
    )

    assert _published(artifact_store, plans) == [
        ("sheet 1 image 1", "image/png", _png())
    ]


def test_ooxml_part_count_budget_fails_before_unbounded_output(artifact_store):
    slide_count = 257
    files: dict[str, bytes] = {
        "ppt/presentation.xml": (
            '<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<p:sldIdLst>"
            + "".join(
                f'<p:sldId id="{index}" r:id="rId{index}"/>'
                for index in range(1, slide_count + 1)
            )
            + "</p:sldIdLst></p:presentation>"
        ).encode(),
        "ppt/_rels/presentation.xml.rels": (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{index}" Target="slides/slide{index}.xml"/>'
                for index in range(1, slide_count + 1)
            )
            + "</Relationships>"
        ).encode(),
    }
    for index in range(1, slide_count + 1):
        files[f"ppt/slides/slide{index}.xml"] = (
            '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
            'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            f"<a:t>Slide {index}</a:t></p:sld>"
        ).encode()
    content = _archive(files)

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(artifact_store).prepare(
            _request(content, relative_path="many.pptx", format_category="presentation"),
            BytesIO(content),
        )

    assert caught.value.code == "ooxml_output_limit"
