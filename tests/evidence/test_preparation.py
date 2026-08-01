from __future__ import annotations

import csv
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
from exam_predictor.evidence.artifacts import ArtifactBoundaryError, EvidenceArtifactStore
from exam_predictor.evidence.policy import EvidencePolicy
from exam_predictor.evidence.preparation import (
    ArchivePreviewAuthority,
    PreparedPartRequest,
    SourcePartPreparer,
    SourcePreparationError,
)
from exam_predictor.evidence.store import EvidenceStore
from exam_predictor.workspace.models import ManifestEntry, ScanPolicy, SourceState
from exam_predictor.workspace.policy import DEFAULT_SCAN_POLICY


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


def _archive_preview_entry(**overrides) -> ManifestEntry:
    values = {
        "entry_id": "member_preview_00000000000000001",
        "workspace_id": WORKSPACE_ID,
        "relative_path": "bundle.zip",
        "item_kind": "archive_member",
        "format_category": "text",
        "size_bytes": 4,
        "sha256": None,
        "state": SourceState.PENDING_APPROVAL,
        "included": False,
        "inclusion_reason": "archive_preview",
        "archive_parent_entry_id": ENTRY_ID,
        "archive_member_path": "member.txt",
        "archive_member_index": 1,
        "archive_member_crc32": 0,
        "archive_member_compressed_bytes": 4,
    }
    values.update(overrides)
    return ManifestEntry(**values)


def _archive_preview_authority(
    entry: ManifestEntry | None = None,
    **overrides,
) -> ArchivePreviewAuthority:
    values = {
        "workspace_id": WORKSPACE_ID,
        "revision_id": REVISION_ID,
        "parent_entry_id": ENTRY_ID,
        "parent_source_sha256": hashlib.sha256(b"archive").hexdigest(),
        "entry": entry or _archive_preview_entry(),
        "approved": True,
    }
    values.update(overrides)
    return ArchivePreviewAuthority(**values)


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


class _FailingAuthoritativeReadStream(BytesIO):
    def __init__(self, content: bytes) -> None:
        super().__init__(content)
        self._rewinds = 0

    def seek(self, offset: int, whence: int = 0) -> int:
        if offset == 0 and whence == 0:
            self._rewinds += 1
        return super().seek(offset, whence)

    def read(self, size: int = -1) -> bytes:
        if self._rewinds >= 1:
            raise RuntimeError("private parser sentinel")
        return super().read(size)


class _SubstitutesAfterHashStream(BytesIO):
    def __init__(self, original: bytes, substitute: bytes) -> None:
        if len(original) != len(substitute):
            raise ValueError("substitution fixture requires equal lengths")
        super().__init__(original)
        self._substitute = substitute
        self._rewinds = 0

    def seek(self, offset: int, whence: int = 0) -> int:
        if offset == 0 and whence == 0:
            self._rewinds += 1
            if self._rewinds == 3:
                super().seek(0)
                super().write(self._substitute)
        return super().seek(offset, whence)


class _FaultyStagedStream(BytesIO):
    def __init__(
        self,
        content: bytes,
        *,
        fault: str | None = None,
        close_fails: bool = False,
    ) -> None:
        super().__init__(content)
        self._fault = fault
        self._close_fails = close_fails

    def seek(self, offset: int, whence: int = 0) -> int:
        if self._fault == "seek":
            raise RuntimeError("private staged seek sentinel")
        return super().seek(offset, whence)

    def read(self, size: int = -1) -> bytes:
        if self._fault == "read":
            raise RuntimeError("private staged read sentinel")
        return super().read(size)

    def close(self) -> None:
        if self._close_fails:
            raise RuntimeError("private cleanup sentinel")
        super().close()


def _inject_staged_stream(
    monkeypatch,
    preparer: SourcePartPreparer,
    stream: BytesIO,
    content: bytes,
) -> None:
    part = preparation_module._StagedPart(
        ordinal=0,
        locator="lines 1-1",
        media_type="text/plain",
        stream=stream,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    monkeypatch.setattr(preparer, "_prepare_text", lambda *_args: [part])


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


def test_identical_parts_from_two_workspaces_remain_distinct_in_evidence_store(
    artifact_store,
    tmp_path,
):
    content = b"same source\n"
    preparer = SourcePartPreparer(artifact_store)
    request_a = _request(
        content,
        relative_path="notes.txt",
        format_category="text",
        workspace_id="workspace_identity_000000000000001",
        revision_id="revision_identity_000000000000001",
        entry_id="entry_identity_00000000000000001",
    )
    request_b = _request(
        content,
        relative_path="notes.txt",
        format_category="text",
        workspace_id="workspace_identity_000000000000002",
        revision_id="revision_identity_000000000000002",
        entry_id="entry_identity_00000000000000002",
    )

    part_a = preparer.prepare(request_a, BytesIO(content))[0]
    part_b = preparer.prepare(request_b, BytesIO(content))[0]

    assert part_a.part_id != part_b.part_id
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    try:
        store.upsert_part_plans((part_a, part_b))
        assert store.get_part(part_a.part_id).workspace_id == request_a.workspace_id
        assert store.get_part(part_b.part_id).workspace_id == request_b.workspace_id
    finally:
        store.close()


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
    preview = _archive_preview_authority()

    request = _request(
        content,
        relative_path="bundle.zip",
        format_category="archive",
        archive_previews=(preview,),
    )

    assert request.archive_previews == (preview,)
    assert request.archive_previews[0].entry.included is False


def test_prepared_request_rejects_raw_archive_preview_without_authority_wrapper():
    content = b"archive"
    preview = _archive_preview_entry()

    with pytest.raises(ValidationError):
        _request(
            content,
            relative_path="bundle.zip",
            format_category="archive",
            archive_previews=(preview,),
        )


def test_archive_preview_authority_requires_explicit_approval():
    common = {
        "workspace_id": WORKSPACE_ID,
        "revision_id": REVISION_ID,
        "parent_entry_id": ENTRY_ID,
        "entry": _archive_preview_entry(),
    }

    with pytest.raises(ValidationError):
        ArchivePreviewAuthority(**common)
    with pytest.raises(ValidationError):
        ArchivePreviewAuthority(**common, approved=False)


@pytest.mark.parametrize(
    "authority",
    [
        _archive_preview_authority(
            entry=_archive_preview_entry(workspace_id="other_workspace"),
            workspace_id="other_workspace",
        ),
        _archive_preview_authority(revision_id="other_revision"),
        _archive_preview_authority(
            entry=_archive_preview_entry(archive_parent_entry_id="other_parent"),
            parent_entry_id="other_parent",
        ),
    ],
    ids=["workspace", "revision", "parent"],
)
def test_prepared_request_rejects_archive_authority_context_mismatch(authority):
    with pytest.raises(ValidationError, match="archive preview authority"):
        _request(
            b"archive",
            relative_path="bundle.zip",
            format_category="archive",
            archive_previews=(authority,),
        )


@pytest.mark.parametrize(
    "authority",
    [
        _archive_preview_authority(
            entry=_archive_preview_entry(relative_path="other.zip"),
        ),
        _archive_preview_authority(parent_source_sha256="0" * 64),
    ],
    ids=["parent-relative-path", "parent-source-hash"],
)
def test_prepared_request_rejects_archive_authority_source_mismatch(authority):
    with pytest.raises(ValidationError, match="archive preview authority"):
        _request(
            b"archive",
            relative_path="bundle.zip",
            format_category="archive",
            archive_previews=(authority,),
        )


@pytest.mark.parametrize(
    "entry_overrides",
    [
        {"state": SourceState.FAILED, "failure_code": "archive_traversal"},
        {"state": SourceState.EXCLUDED},
        {"item_kind": "file"},
        {"failure_code": "archive_corrupt"},
        {"inclusion_reason": "user_excluded"},
        {"included": True},
        {"archive_member_path": "../escape.txt"},
        {"archive_member_path": ""},
        {"archive_member_path": "folder/control\n.txt"},
        {"archive_member_path": "folder/control\u0085.txt"},
        {"archive_member_path": "x" * (DEFAULT_SCAN_POLICY.max_path_chars + 1)},
    ],
    ids=[
        "failed",
        "excluded",
        "kind",
        "failure-code",
        "reason",
        "included",
        "traversal",
        "empty-path",
        "control-character",
        "c1-control-character",
        "oversized-path",
    ],
)
def test_archive_preview_authority_rejects_unsafe_raw_entries(entry_overrides):
    with pytest.raises(ValidationError, match="archive preview entry"):
        _archive_preview_authority(_archive_preview_entry(**entry_overrides))


def test_prepared_request_rejects_duplicate_archive_member_authority():
    first = _archive_preview_authority()
    second = _archive_preview_authority(_archive_preview_entry(entry_id="member_preview_00000000000000002"))

    with pytest.raises(ValidationError, match="archive preview authority"):
        _request(
            b"archive",
            relative_path="bundle.zip",
            format_category="archive",
            archive_previews=(first, second),
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


def test_mutable_caller_stream_cannot_substitute_bytes_after_hash(artifact_store):
    original = b"alpha\n"
    substitute = b"omega\n"
    request = _request(
        original,
        relative_path="notes.txt",
        format_category="text",
    )
    stream = _SubstitutesAfterHashStream(original, substitute)

    parts = SourcePartPreparer(artifact_store).prepare(request, stream)

    with artifact_store.open_part(WORKSPACE_ID, parts[0].part_id) as opened:
        prepared = opened.read()
    assert prepared == original
    assert hashlib.sha256(prepared).hexdigest() == request.source_sha256


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


def test_authoritative_stream_error_is_stable_and_has_no_exception_chain():
    content = b'{"safe": true}\n'
    request = _request(
        content,
        relative_path="data.json",
        format_category="structured_data",
    )
    store = _RecordingArtifactStore()

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(store).prepare(request, _FailingAuthoritativeReadStream(content))

    assert (caught.value.code, caught.value.locator) == ("source_stream_invalid", "source")
    assert "private parser sentinel" not in str(caught.value)
    assert "private parser sentinel" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert store.published == []


def test_authoritative_spool_creation_error_is_stable_and_has_no_exception_chain(monkeypatch):
    content = b"safe\n"
    request = _request(content, relative_path="notes.txt", format_category="text")

    def fail_spool(*args, **kwargs):
        raise RuntimeError("private spool sentinel")

    monkeypatch.setattr(preparation_module, "SpooledTemporaryFile", fail_spool)
    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(_RecordingArtifactStore()).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == (
        "source_spool_unavailable",
        "source",
    )
    assert "private spool sentinel" not in str(caught.value)
    assert "private spool sentinel" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_pdf_parser_error_is_stable_and_has_no_exception_chain(monkeypatch):
    content = _pdf_bytes(pages=1)
    request = _request(content, relative_path="paper.pdf", format_category="pdf")

    def fail_parser(*args, **kwargs):
        raise RuntimeError("private pdf sentinel")

    monkeypatch.setattr(preparation_module, "PdfReader", fail_parser)
    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(_RecordingArtifactStore()).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == ("pdf_corrupt", "source")
    assert "private pdf sentinel" not in str(caught.value)
    assert "private pdf sentinel" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_image_parser_error_is_stable_and_has_no_exception_chain(monkeypatch):
    content = _image_bytes("PNG")
    request = _request(content, relative_path="scan.png", format_category="image")

    def fail_parser(*args, **kwargs):
        raise RuntimeError("private image sentinel")

    monkeypatch.setattr(preparation_module.Image, "open", fail_parser)
    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(_RecordingArtifactStore()).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == ("image_corrupt", "image")
    assert "private image sentinel" not in str(caught.value)
    assert "private image sentinel" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_json_parser_error_is_stable_and_has_no_exception_chain(monkeypatch):
    content = b'{"safe": true}\n'
    request = _request(content, relative_path="data.json", format_category="structured_data")

    def fail_parser(*args, **kwargs):
        raise RuntimeError("private json sentinel")

    monkeypatch.setattr(preparation_module.json, "loads", fail_parser)
    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(_RecordingArtifactStore()).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == (
        "structured_data_malformed",
        "source",
    )
    assert "private json sentinel" not in str(caught.value)
    assert "private json sentinel" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_text_replacement_failure_discards_strict_decoder_exception_context():
    content = b"private decoder sentinel \xff" + b"x" * 2_048
    request = _request(content, relative_path="notes.txt", format_category="text")
    policy = EvidencePolicy(max_part_bytes=1_024)

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(_RecordingArtifactStore(), policy=policy).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == (
        "text_line_too_large",
        "line 1",
    )
    assert "private decoder sentinel" not in str(caught.value)
    assert "private decoder sentinel" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize("fault", ["seek", "read"])
def test_staged_publish_stream_fault_is_stable_and_has_no_exception_chain(
    monkeypatch,
    fault,
):
    content = b"safe\n"
    store = _RecordingArtifactStore()
    preparer = SourcePartPreparer(store)
    staged_stream = _FaultyStagedStream(content, fault=fault)
    _inject_staged_stream(monkeypatch, preparer, staged_stream, content)

    with pytest.raises(SourcePreparationError) as caught:
        preparer.prepare(
            _request(content, relative_path="notes.txt", format_category="text"),
            BytesIO(content),
        )

    assert (caught.value.code, caught.value.locator) == (
        "prepared_part_unavailable",
        "lines 1-1",
    )
    assert "private staged" not in str(caught.value)
    assert "private staged" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert store.published == []


def test_artifact_store_raw_fault_has_no_exception_chain():
    class FailingArtifactStore:
        def publish_part(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("private artifact store sentinel")

    content = b"safe\n"
    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(FailingArtifactStore()).prepare(
            _request(content, relative_path="notes.txt", format_category="text"),
            BytesIO(content),
        )

    assert (caught.value.code, caught.value.locator) == (
        "artifact_publish_failed",
        "lines 1-1",
    )
    assert "private artifact store sentinel" not in str(caught.value)
    assert "private artifact store sentinel" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_artifact_boundary_code_is_preserved_without_raw_exception_context():
    class BoundaryFailingArtifactStore:
        def publish_part(self, *args, **kwargs):
            del args, kwargs
            try:
                raise RuntimeError("private artifact boundary sentinel")
            except RuntimeError:
                raise ArtifactBoundaryError("artifact_identity_changed")

    content = b"safe\n"
    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(BoundaryFailingArtifactStore()).prepare(
            _request(content, relative_path="notes.txt", format_category="text"),
            BytesIO(content),
        )

    assert (caught.value.code, caught.value.locator) == (
        "artifact_identity_changed",
        "lines 1-1",
    )
    assert "private artifact boundary sentinel" not in str(caught.value)
    assert "private artifact boundary sentinel" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_cleanup_fault_cannot_override_selected_staged_stream_error(monkeypatch):
    content = b"safe\n"
    store = _RecordingArtifactStore()
    preparer = SourcePartPreparer(store)
    staged_stream = _FaultyStagedStream(content, fault="seek", close_fails=True)
    _inject_staged_stream(monkeypatch, preparer, staged_stream, content)

    with pytest.raises(SourcePreparationError) as caught:
        preparer.prepare(
            _request(content, relative_path="notes.txt", format_category="text"),
            BytesIO(content),
        )

    assert (caught.value.code, caught.value.locator) == (
        "prepared_part_unavailable",
        "lines 1-1",
    )
    assert "private cleanup sentinel" not in str(caught.value)
    assert "private cleanup sentinel" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
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


def test_csv_accepts_140k_field_without_mutating_process_global_limit(artifact_store):
    content = ("value\n" + ("x" * 140_000) + "\n").encode()
    request = _request(content, relative_path="large.csv", format_category="tabular")
    original_limit = csv.field_size_limit()

    parts = SourcePartPreparer(
        artifact_store,
        policy=EvidencePolicy(max_part_bytes=150_000),
    ).prepare(request, BytesIO(content))

    assert csv.field_size_limit() == original_limit
    with artifact_store.open_part(WORKSPACE_ID, parts[0].part_id) as opened:
        assert opened.read() == content


def test_csv_rejects_oversized_multiline_row_before_publication():
    content = ('value\n"' + ("a" * 700) + "\n" + ("b" * 700) + '",1\n').encode()
    request = _request(content, relative_path="large.csv", format_category="tabular")
    store = _RecordingArtifactStore()

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(
            store,
            policy=EvidencePolicy(max_part_bytes=1_024),
        ).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == (
        "tabular_row_too_large",
        "row 2",
    )
    assert store.published == []


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


@pytest.mark.parametrize(
    ("relative_path", "content", "loader_owner", "loader_name"),
    [
        pytest.param(
            "flat.json",
            ("[" + ",".join("0" for _ in range(100_001)) + "]").encode(),
            "json",
            "loads",
            id="flat-json",
        ),
        pytest.param(
            "deep.json",
            (("[" * 65) + "0" + ("]" * 65)).encode(),
            "json",
            "loads",
            id="deep-json",
        ),
        pytest.param(
            "string.json",
            ('{"value":"' + ("x" * (1024 * 1024 + 1)) + '"}').encode(),
            "json",
            "loads",
            id="large-string-json",
        ),
        pytest.param(
            "flat.yaml",
            ("[" + ",".join("0" for _ in range(100_001)) + "]").encode(),
            "yaml",
            "safe_load",
            id="flat-yaml",
        ),
        pytest.param(
            "deep.yaml",
            (("[" * 65) + "0" + ("]" * 65)).encode(),
            "yaml",
            "safe_load",
            id="deep-yaml",
        ),
    ],
)
def test_structured_data_budgets_reject_before_materializing_document(
    relative_path,
    content,
    loader_owner,
    loader_name,
    monkeypatch,
):
    materialized = False

    def forbidden_materialization(*args, **kwargs):
        nonlocal materialized
        materialized = True
        raise AssertionError("bounded preflight must run before document materialization")

    owner = preparation_module.json if loader_owner == "json" else preparation_module.yaml
    monkeypatch.setattr(owner, loader_name, forbidden_materialization)
    store = _RecordingArtifactStore()
    request = _request(
        content,
        relative_path=relative_path,
        format_category="structured_data",
    )

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(store).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == (
        "structured_data_limit_exceeded",
        "source",
    )
    assert materialized is False
    assert store.published == []


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_rejects_nonstandard_numeric_constants(constant):
    content = f'{{"value":{constant}}}\n'.encode()
    request = _request(
        content,
        relative_path="constants.json",
        format_category="structured_data",
    )
    store = _RecordingArtifactStore()

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(store).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == (
        "structured_data_malformed",
        "source",
    )
    assert store.published == []


def test_json_rejects_exponent_overflow_after_parsing():
    content = b'{"value":1e1000000}\n'
    request = _request(
        content,
        relative_path="overflow.json",
        format_category="structured_data",
    )
    store = _RecordingArtifactStore()

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(store).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == (
        "structured_data_malformed",
        "source",
    )
    assert store.published == []


@pytest.mark.parametrize(
    "scalar",
    [".nan", ".NaN", ".NAN", ".inf", ".Inf", ".INF", "+.inf", "-.Inf"],
)
def test_yaml_rejects_non_finite_numeric_variants_after_parsing(scalar):
    content = f"value: {scalar}\n".encode()
    request = _request(
        content,
        relative_path="non-finite.yaml",
        format_category="structured_data",
    )
    store = _RecordingArtifactStore()

    with pytest.raises(SourcePreparationError) as caught:
        SourcePartPreparer(store).prepare(request, BytesIO(content))

    assert (caught.value.code, caught.value.locator) == (
        "structured_data_malformed",
        "source",
    )
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
