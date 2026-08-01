from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
import zipfile

import pytest

from exam_predictor.evidence.converter import (
    ConvertedDocument,
    LegacyConversionError,
    LibreOfficeConverter,
)
from exam_predictor.evidence.artifacts import EvidenceArtifactStore
from exam_predictor.evidence.preparation import PreparedPartRequest, SourcePartPreparer, SourcePreparationError


def test_unavailable_converter_fails_without_reading_source(tmp_path: Path):
    source = BytesIO(b"legacy source")
    converter = LibreOfficeConverter(executable=tmp_path / "missing-soffice")

    assert converter.available() is False
    with pytest.raises(LegacyConversionError) as caught:
        converter.convert(source, suffix=".doc", deadline=5.0)

    assert caught.value.code == "converter_unavailable"
    assert source.tell() == 0


def test_converter_uses_fixed_argument_vector_and_returns_bounded_owned_bytes(tmp_path: Path):
    executable = tmp_path / "soffice.exe"
    executable.write_bytes(b"fixed executable identity")
    calls: list[dict[str, object]] = []

    class FakeSandbox:
        def run(self, arguments, **kwargs):
            calls.append({"arguments": tuple(arguments), **kwargs})
            output_directory = Path(arguments[arguments.index("--outdir") + 1])
            source_path = Path(arguments[-1])
            (output_directory / f"{source_path.stem}.docx").write_bytes(b"converted docx bytes")
            return 0

    converter = LibreOfficeConverter(
        executable=executable,
        sandbox=FakeSandbox(),
        max_output_bytes=1024,
    )

    converted = converter.convert(BytesIO(b"legacy source"), suffix=".doc", deadline=5.0)

    assert converted.suffix == ".docx"
    assert converted.content_bytes == b"converted docx bytes"
    assert "content_bytes" not in repr(converted)
    assert len(calls) == 1
    call = calls[0]
    arguments = call["arguments"]
    assert arguments[0] == str(executable.resolve())
    assert arguments[1:6] == ("--headless", "--nologo", "--nodefault", "--nolockcheck", "--norestore")
    assert "--convert-to" in arguments
    assert arguments[arguments.index("--convert-to") + 1] == "docx"
    assert call["deadline"] == 5.0
    assert call["max_output_bytes"] == 1024


def test_converter_rejects_unexpected_or_oversized_output(tmp_path: Path):
    executable = tmp_path / "soffice"
    executable.write_bytes(b"fixed executable identity")

    class FakeSandbox:
        def run(self, arguments, **_kwargs):
            output_directory = Path(arguments[arguments.index("--outdir") + 1])
            source_path = Path(arguments[-1])
            (output_directory / f"{source_path.stem}.docx").write_bytes(b"x" * 33)
            (output_directory / "unexpected.bin").write_bytes(b"unexpected")
            return 0

    converter = LibreOfficeConverter(
        executable=executable,
        sandbox=FakeSandbox(),
        max_output_bytes=32,
    )

    with pytest.raises(LegacyConversionError) as caught:
        converter.convert(BytesIO(b"legacy source"), suffix=".doc", deadline=5.0)

    assert caught.value.code == "converter_failed"


def test_default_converter_refuses_to_run_without_a_secure_sandbox(tmp_path: Path):
    executable = tmp_path / "soffice"
    executable.write_bytes(b"fixed executable identity")
    converter = LibreOfficeConverter(executable=executable)

    assert converter.available() is False
    with pytest.raises(LegacyConversionError) as caught:
        converter.convert(BytesIO(b"legacy source"), suffix=".doc", deadline=5.0)

    assert caught.value.code == "converter_unavailable"


def _docx_bytes() -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as package:
        package.writestr(
            "word/document.xml",
            b"""
                <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                  <w:body><w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
                    <w:r><w:t>Converted Unit</w:t></w:r></w:p></w:body>
                </w:document>
            """,
        )
    return output.getvalue()


def _legacy_request(content: bytes) -> PreparedPartRequest:
    return PreparedPartRequest(
        workspace_id="workspace_converter_00000000000001",
        revision_id="revision_converter_00000000000001",
        entry_id="entry_converter_00000000000000001",
        relative_path="legacy/course.doc",
        format_category="document",
        source_size_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )


class _FakeConverter:
    def __init__(self, *, available: bool = True) -> None:
        self.is_available = available
        self.calls: list[tuple[bytes, str, float]] = []

    def available(self) -> bool:
        return self.is_available

    def convert(self, source, *, suffix: str, deadline: float) -> ConvertedDocument:
        source.seek(0)
        self.calls.append((source.read(), suffix, deadline))
        content = _docx_bytes()
        return ConvertedDocument(
            suffix=".docx",
            content_bytes=content,
            content_size_bytes=len(content),
            content_sha256=hashlib.sha256(content).hexdigest(),
        )


def test_preparer_routes_legacy_document_through_converter_output(tmp_path: Path):
    root = tmp_path / "artifacts"
    root.mkdir()
    artifact_store = EvidenceArtifactStore(root)
    converter = _FakeConverter()
    content = b"legacy source"
    try:
        plans = SourcePartPreparer(artifact_store, legacy_converter=converter).prepare(
            _legacy_request(content),
            BytesIO(content),
        )
    finally:
        artifact_store.close()

    assert [(plan.locator, plan.media_type) for plan in plans] == [
        ("section 1", "text/plain")
    ]
    assert converter.calls == [(content, ".doc", 90.0)]


def test_preparer_reports_converter_unavailable_before_conversion(tmp_path: Path):
    root = tmp_path / "artifacts"
    root.mkdir()
    artifact_store = EvidenceArtifactStore(root)
    converter = _FakeConverter(available=False)
    content = b"legacy source"
    try:
        with pytest.raises(SourcePreparationError) as caught:
            SourcePartPreparer(artifact_store, legacy_converter=converter).prepare(
                _legacy_request(content),
                BytesIO(content),
            )
    finally:
        artifact_store.close()

    assert caught.value.code == "converter_unavailable"
    assert converter.calls == []
