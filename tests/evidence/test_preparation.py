from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
import urllib.request

import pytest
from PIL import Image
from pydantic import ValidationError
from pypdf.generic import DecodedStreamObject, NameObject
from pypdf import PdfWriter

from exam_predictor.evidence import preparation as preparation_module
from exam_predictor.evidence.artifacts import EvidenceArtifactStore
from exam_predictor.evidence.policy import EvidencePolicy
from exam_predictor.evidence.preparation import (
    PreparedPartRequest,
    SourcePartPreparer,
    SourcePreparationError,
)
from exam_predictor.workspace.models import ManifestEntry, ScanPolicy, SourceState


WORKSPACE_ID = "workspace_preparation_000000000001"
REVISION_ID = "revision_preparation_000000000001"
ENTRY_ID = "entry_preparation_000000000000001"


@pytest.fixture
def artifact_store(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = EvidenceArtifactStore(root)
    try:
        yield store
    finally:
        store.close()


def _pdf_bytes(*, pages: int) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _pdf_with_payloads(payload_sizes: tuple[int, ...]) -> bytes:
    writer = PdfWriter()
    for payload_size in payload_sizes:
        page = writer.add_blank_page(width=72, height=72)
        content = DecodedStreamObject()
        content.set_data((b"0 0 m 1 1 l S\n" * ((payload_size // 16) + 1))[:payload_size])
        page[NameObject("/Contents")] = writer._add_object(content)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _encrypted_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt("secret")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _image_bytes(image_format: str, *, frames: int = 1) -> bytes:
    images = [Image.new("RGB", (3, 2), (index % 255, 20, 30)) for index in range(frames)]
    output = BytesIO()
    if frames == 1:
        images[0].save(output, format=image_format)
    else:
        images[0].save(
            output,
            format=image_format,
            save_all=True,
            append_images=images[1:],
            duration=20,
            loop=0,
            optimize=False,
        )
    return output.getvalue()


def _request(
    content: bytes,
    *,
    relative_path: str,
    format_category: str,
    **overrides,
) -> PreparedPartRequest:
    values = {
        "workspace_id": WORKSPACE_ID,
        "revision_id": REVISION_ID,
        "entry_id": ENTRY_ID,
        "relative_path": relative_path,
        "format_category": format_category,
        "source_size_bytes": len(content),
        "source_sha256": hashlib.sha256(content).hexdigest(),
    }
    values.update(overrides)
    return PreparedPartRequest(**values)


def _prepare_pdf(
    artifact_store: EvidenceArtifactStore,
    content: bytes,
    *,
    pages_per_part: int = 24,
):
    request = _request(
        content,
        relative_path="references/long-book.pdf",
        format_category="pdf",
    )
    preparer = SourcePartPreparer(
        artifact_store,
        policy=EvidencePolicy(pdf_pages_per_part=pages_per_part),
    )
    return preparer.prepare(request, BytesIO(content))


class _RecordingArtifactStore:
    def __init__(self) -> None:
        self.published: list[tuple[str, str, bytes, str]] = []

    def publish_part(
        self,
        workspace_id: str,
        part_id: str,
        content: bytes,
        *,
        expected_sha256: str,
    ) -> str:
        self.published.append((workspace_id, part_id, bytes(content), expected_sha256))
        return expected_sha256


class _FailingSecondPassStream(BytesIO):
    def __init__(self, content: bytes) -> None:
        super().__init__(content)
        self._rewinds = 0

    def seek(self, offset: int, whence: int = 0) -> int:
        if offset == 0 and whence == 0:
            self._rewinds += 1
        return super().seek(offset, whence)

    def read(self, size: int = -1) -> bytes:
        if self._rewinds >= 3:
            raise RuntimeError("private parser sentinel")
        return super().read(size)


def test_long_pdf_plans_begin_middle_end_before_remaining_pages(artifact_store):
    parts = _prepare_pdf(artifact_store, _pdf_bytes(pages=357))

    assert [part.ordinal for part in parts[:3]] == [0, 7, 14]
    assert sorted(part.ordinal for part in parts) == list(range(15))
    assert all(part.size_bytes <= 10 * 1024 * 1024 for part in parts)


def test_each_part_has_stable_hash_locator_and_idempotency_key(artifact_store):
    content = _pdf_bytes(pages=50)

    first = _prepare_pdf(artifact_store, content)
    second = _prepare_pdf(artifact_store, content)

    assert [(item.locator, item.part_sha256, item.idempotency_key) for item in first] == [
        (item.locator, item.part_sha256, item.idempotency_key) for item in second
    ]
    assert [part.locator for part in first] == ["pages 1-24", "pages 25-48", "pages 49-50"]
    assert all(part.part_id.startswith("part_") for part in first)


def test_prepared_request_is_frozen_and_forbids_native_paths():
    content = b"safe"
    request = _request(content, relative_path="notes/safe.txt", format_category="text")

    with pytest.raises(ValidationError):
        request.relative_path = "changed.txt"
    with pytest.raises(ValidationError):
        PreparedPartRequest(
            **request.model_dump(),
            native_path=Path("native.txt"),
        )


def test_prepared_request_carries_typed_archive_preview_authority():
    content = b"archive"
    preview = ManifestEntry(
        entry_id="member_preview_00000000000000001",
        workspace_id=WORKSPACE_ID,
        relative_path="bundle.zip/member.txt",
        item_kind="archive_member",
        format_category="text",
        size_bytes=4,
        sha256=hashlib.sha256(b"safe").hexdigest(),
        state=SourceState.APPROVED,
        included=True,
        archive_parent_entry_id=ENTRY_ID,
        archive_member_path="member.txt",
    )

    request = _request(
        content,
        relative_path="bundle.zip",
        format_category="archive",
        archive_preview_entries=(preview,),
    )

    assert request.archive_preview_entries == (preview,)


def test_prepared_request_rejects_archive_preview_from_another_parent():
    content = b"archive"
    preview = ManifestEntry(
        entry_id="member_preview_00000000000000001",
        workspace_id=WORKSPACE_ID,
        relative_path="bundle.zip/member.txt",
        item_kind="archive_member",
        size_bytes=4,
        state=SourceState.APPROVED,
        included=True,
        archive_parent_entry_id="other_entry_0000000000000000001",
        archive_member_path="member.txt",
    )

    with pytest.raises(ValidationError, match="archive preview authority"):
        _request(
            content,
            relative_path="bundle.zip",
            format_category="archive",
            archive_preview_entries=(preview,),
        )


def test_source_hash_mismatch_fails_before_artifact_publication():
    content = b"private sentinel source"
    store = _RecordingArtifactStore()
    request = _request(
        content,
        relative_path="notes.txt",
        format_category="text",
        source_sha256="0" * 64,
    )

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(store).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == ("source_hash_mismatch", "source")
    assert "private sentinel source" not in str(caught.value)
    assert "private sentinel source" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert store.published == []


def test_source_size_mismatch_fails_before_artifact_publication():
    content = b"source"
    store = _RecordingArtifactStore()
    request = _request(
        content,
        relative_path="notes.txt",
        format_category="text",
        source_size_bytes=len(content) + 1,
    )

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(store).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == ("source_size_mismatch", "source")
    assert store.published == []


def test_nonzero_stream_cursor_is_rewound_without_taking_ownership(artifact_store):
    content = "alpha\n中文 boundary\nomega\n".encode()
    stream = BytesIO(content)
    stream.seek(7)
    request = _request(content, relative_path="syllabus/notes.txt", format_category="text")

    parts = SourcePartPreparer(
        artifact_store,
        scan_policy=ScanPolicy(hash_chunk_bytes=2),
    ).prepare(request, stream)

    assert stream.closed is False
    assert parts[0].priority == 0
    assert parts[0].scheduling_class == "syllabus"
    with artifact_store.open_part(WORKSPACE_ID, parts[0].part_id) as opened:
        assert opened.read() == content


def test_native_path_is_not_a_preparation_stream():
    content = b"safe"
    request = _request(content, relative_path="notes.txt", format_category="text")

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(_RecordingArtifactStore()).prepare(request, Path("notes.txt"))

    assert (caught.value.code, caught.value.locator) == ("source_stream_invalid", "source")


def test_second_pass_stream_error_is_stable_and_redacted():
    content = b'{"safe": true}\n'
    request = _request(
        content,
        relative_path="data.json",
        format_category="structured_data",
    )
    store = _RecordingArtifactStore()

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(store).prepare(request, _FailingSecondPassStream(content))

    assert (caught.value.code, caught.value.locator) == ("source_stream_invalid", "source")
    assert "private parser sentinel" not in str(caught.value)
    assert "private parser sentinel" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert store.published == []


def test_pdf_groups_shrink_to_stay_below_the_part_limit(artifact_store):
    content = _pdf_with_payloads((1_500, 1_500))
    request = _request(content, relative_path="paper.pdf", format_category="pdf")
    preparer = SourcePartPreparer(
        artifact_store,
        policy=EvidencePolicy(pdf_pages_per_part=2, max_part_bytes=2_500),
    )

    parts = preparer.prepare(request, BytesIO(content))

    assert [part.locator for part in parts] == ["pages 1-1", "pages 2-2"]
    assert [part.ordinal for part in parts] == [0, 1]
    assert all(part.size_bytes <= 2_500 for part in parts)


@pytest.mark.parametrize(
    ("content", "expected_code"),
    [
        (b"%PDF private corrupt sentinel", "pdf_corrupt"),
        (_encrypted_pdf_bytes(), "pdf_encrypted"),
        (_pdf_bytes(pages=0), "pdf_empty"),
    ],
)
def test_invalid_pdf_fails_safely_without_publication(content, expected_code):
    store = _RecordingArtifactStore()
    request = _request(content, relative_path="paper.pdf", format_category="pdf")

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(store).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == (expected_code, "source")
    assert "private corrupt sentinel" not in str(caught.value)
    assert "private corrupt sentinel" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert store.published == []


def test_indivisible_oversized_pdf_page_fails_without_publication():
    content = _pdf_with_payloads((5_000,))
    store = _RecordingArtifactStore()
    request = _request(content, relative_path="paper.pdf", format_category="pdf")

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(
            store,
            policy=EvidencePolicy(max_part_bytes=2_048),
        ).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == ("pdf_page_too_large", "page 1")
    assert store.published == []


def test_text_splits_only_at_lines_and_records_overlap(artifact_store):
    content = "".join(f"line {index}: {'x' * 390}\n" for index in range(6)).encode()
    request = _request(content, relative_path="notes.md", format_category="text")

    parts = SourcePartPreparer(
        artifact_store,
        policy=EvidencePolicy(max_part_bytes=1_024),
    ).prepare(request, BytesIO(content))

    assert len(parts) > 1
    assert parts[0].locator == "lines 1-2"
    assert "overlap" in parts[1].locator
    for part in parts:
        with artifact_store.open_part(WORKSPACE_ID, part.part_id) as opened:
            prepared = opened.read()
        assert prepared.endswith(b"\n")
        assert hashlib.sha256(prepared).hexdigest() == part.part_sha256
        assert len(prepared) <= 1_024


def test_invalid_utf8_text_uses_bounded_replacement_fallback(artifact_store):
    content = b"alpha\nbad\xffline\nomega\n"
    request = _request(content, relative_path="notes.txt", format_category="text")

    parts = SourcePartPreparer(artifact_store).prepare(request, BytesIO(content))

    with artifact_store.open_part(WORKSPACE_ID, parts[0].part_id) as opened:
        prepared = opened.read()
    assert b"bad\xef\xbf\xbdline" in prepared


def test_html_is_published_as_inert_text_without_fetching(artifact_store, monkeypatch):
    content = b'<script>doBadThing()</script><a href="https://invalid.test/private">course</a>\n'
    request = _request(content, relative_path="course.html", format_category="text")
    fetched: list[str] = []

    def fail_fetch(url, *args, **kwargs):
        fetched.append(str(url))
        raise AssertionError("HTML preparation must never fetch URLs")

    monkeypatch.setattr(urllib.request, "urlopen", fail_fetch)
    parts = SourcePartPreparer(artifact_store).prepare(request, BytesIO(content))

    assert fetched == []
    with artifact_store.open_part(WORKSPACE_ID, parts[0].part_id) as opened:
        assert opened.read() == content
    assert parts[0].media_type == "text/html"


def test_csv_preserves_complete_quoted_rows_with_embedded_newlines(artifact_store):
    content = (
        "name,value\n" + '"alpha\nbeta ' + ("x" * 700) + '",1\n' + '"gamma ' + ("y" * 700) + '",2\n'
    ).encode()
    request = _request(content, relative_path="marks.csv", format_category="tabular")

    parts = SourcePartPreparer(
        artifact_store,
        policy=EvidencePolicy(max_part_bytes=1_024),
    ).prepare(request, BytesIO(content))

    prepared = bytearray()
    for part in sorted(parts, key=lambda item: item.ordinal):
        with artifact_store.open_part(WORKSPACE_ID, part.part_id) as opened:
            prepared.extend(opened.read())
    assert bytes(prepared) == content
    assert [part.locator for part in parts] == ["rows 1-2", "rows 3-3"]


def test_malformed_csv_error_is_stable_and_redacted():
    content = b'name,value\n"private sentinel,1\n'
    store = _RecordingArtifactStore()
    request = _request(content, relative_path="marks.csv", format_category="tabular")

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(store).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == ("tabular_malformed", "source")
    assert "private sentinel" not in str(caught.value)
    assert "private sentinel" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert store.published == []


@pytest.mark.parametrize(
    ("relative_path", "content"),
    [
        ("broken.json", b'{"private sentinel": [}'),
        ("broken.yaml", b"private sentinel: [unterminated"),
    ],
)
def test_malformed_structured_data_error_is_stable_and_redacted(relative_path, content):
    store = _RecordingArtifactStore()
    request = _request(content, relative_path=relative_path, format_category="structured_data")

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(store).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == ("structured_data_malformed", "source")
    assert "private sentinel" not in str(caught.value)
    assert "private sentinel" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert store.published == []


def test_static_image_magic_must_match_the_declared_extension():
    content = _image_bytes("GIF")
    store = _RecordingArtifactStore()
    request = _request(content, relative_path="scan.png", format_category="image")

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(store).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == ("image_magic_mismatch", "image")
    assert store.published == []


def test_image_bomb_dimensions_are_rejected_from_headers_before_decode():
    content = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (100_000).to_bytes(4, "big") * 2
    store = _RecordingArtifactStore()
    request = _request(content, relative_path="bomb.png", format_category="image")

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(store).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == ("image_dimensions_exceeded", "image")
    assert store.published == []


def test_animated_gif_selects_beginning_middle_end_and_records_omissions(artifact_store):
    content = _image_bytes("GIF", frames=7)
    request = _request(content, relative_path="animation.gif", format_category="image")

    parts = SourcePartPreparer(artifact_store).prepare(request, BytesIO(content))

    assert [part.locator for part in parts] == [
        "frame 1 of 7; omitted 4",
        "frame 4 of 7; omitted 4",
        "frame 7 of 7; omitted 4",
    ]
    assert [part.ordinal for part in parts] == [0, 1, 2]
    assert all(part.media_type == "image/png" for part in parts)
    for part in parts:
        with artifact_store.open_part(WORKSPACE_ID, part.part_id) as opened:
            prepared = opened.read()
        assert hashlib.sha256(prepared).hexdigest() == part.part_sha256


def test_animated_gif_fails_safely_when_pillow_is_unavailable(monkeypatch):
    content = _image_bytes("GIF", frames=2)
    store = _RecordingArtifactStore()
    request = _request(content, relative_path="animation.gif", format_category="image")
    monkeypatch.setattr(preparation_module, "Image", None, raising=False)

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(store).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == ("image_frames_unavailable", "image")
    assert store.published == []


def test_gif_with_excessive_frame_count_is_rejected_before_publication():
    content = _image_bytes("GIF", frames=257)
    store = _RecordingArtifactStore()
    request = _request(content, relative_path="animation.gif", format_category="image")

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(store).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == ("image_frame_count_exceeded", "image")
    assert store.published == []
