from __future__ import annotations

import io
import stat
import struct
import zipfile

import exam_predictor.workspace.archive as archive_module
import pytest
from exam_predictor.workspace.archive import (
    ARCHIVE_ABSOLUTE_PATH,
    ARCHIVE_DEPTH_LIMIT,
    ARCHIVE_ENCRYPTED,
    ARCHIVE_LINK,
    ARCHIVE_MEMBER_LIMIT,
    ARCHIVE_RATIO_LIMIT,
    ARCHIVE_SIZE_LIMIT,
    ARCHIVE_TRAVERSAL,
    ArchiveInspector,
)
from exam_predictor.workspace.models import SourceState
from exam_predictor.workspace.policy import DEFAULT_SCAN_POLICY


def _archive(*members: tuple[str | zipfile.ZipInfo, bytes], compression=zipfile.ZIP_STORED) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        for name, content in members:
            archive.writestr(name, content)
    return stream.getvalue()


def _mark_first_entry_encrypted(raw_archive: bytes) -> bytes:
    mutable = bytearray(raw_archive)
    local_offset = mutable.index(b"PK\x03\x04")
    central_offset = mutable.index(b"PK\x01\x02")
    local_flags = struct.unpack_from("<H", mutable, local_offset + 6)[0]
    central_flags = struct.unpack_from("<H", mutable, central_offset + 8)[0]
    struct.pack_into("<H", mutable, local_offset + 6, local_flags | 0x1)
    struct.pack_into("<H", mutable, central_offset + 8, central_flags | 0x1)
    return bytes(mutable)


def _forge_eocd_member_count(raw_archive: bytes, member_count: int) -> bytes:
    mutable = bytearray(raw_archive)
    end_offset = mutable.rindex(b"PK\x05\x06")
    struct.pack_into("<H", mutable, end_offset + 8, member_count)
    struct.pack_into("<H", mutable, end_offset + 10, member_count)
    return bytes(mutable)


def _by_path(members, path):
    return next(member for member in members if member.display_path == path)


def test_archive_inspection_returns_metadata_without_extracting(tmp_path):
    archive_path = tmp_path / "materials.zip"
    archive_path.write_bytes(
        _archive(
            ("notes/week1.txt", b"safe"),
            ("../escape.txt", b"secret"),
            ("/absolute.txt", b"secret"),
        )
    )

    with archive_path.open("rb") as archive_stream:
        members = ArchiveInspector().inspect(archive_stream, parent_entry_id="archive-1")

    assert {member.display_path for member in members} >= {
        "notes/week1.txt",
        "../escape.txt",
    }
    assert not (tmp_path / "notes").exists()
    assert _by_path(members, "notes/week1.txt").state is SourceState.PENDING_APPROVAL
    assert _by_path(members, "../escape.txt").failure_code == ARCHIVE_TRAVERSAL
    assert _by_path(members, "/absolute.txt").failure_code == ARCHIVE_ABSOLUTE_PATH


def test_archive_rejects_drive_paths_before_other_member_handling():
    members = ArchiveInspector().inspect(
        io.BytesIO(_archive(("C:/private.txt", b"secret"))), parent_entry_id="archive-1"
    )

    assert members[0].state is SourceState.FAILED
    assert members[0].failure_code == ARCHIVE_ABSOLUTE_PATH


def test_archive_rejects_unix_symlink_metadata():
    link = zipfile.ZipInfo("linked-notes.txt")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16

    members = ArchiveInspector().inspect(
        io.BytesIO(_archive((link, b"../private.txt"))), parent_entry_id="archive-1"
    )

    assert members[0].state is SourceState.FAILED
    assert members[0].failure_code == ARCHIVE_LINK


def test_archive_rejects_encrypted_flag_without_opening_member_content():
    encrypted_metadata = _mark_first_entry_encrypted(_archive(("notes.txt", b"not encrypted")))

    members = ArchiveInspector().inspect(
        io.BytesIO(encrypted_metadata), parent_entry_id="archive-1"
    )

    assert members[0].state is SourceState.FAILED
    assert members[0].failure_code == ARCHIVE_ENCRYPTED


def test_archive_member_limit_returns_one_bounded_archive_failure():
    policy = DEFAULT_SCAN_POLICY.model_copy(update={"max_archive_members": 1})

    members = ArchiveInspector(policy).inspect(
        io.BytesIO(_archive(("one.txt", b"1"), ("two.txt", b"2"))),
        parent_entry_id="archive-1",
    )

    assert len(members) == 1
    assert members[0].state is SourceState.FAILED
    assert members[0].failure_code == ARCHIVE_MEMBER_LIMIT


def test_archive_member_limit_bounds_retained_overflow_metadata():
    policy = DEFAULT_SCAN_POLICY.model_copy(update={"max_archive_members": 1})

    members = ArchiveInspector(policy).inspect(
        io.BytesIO(
            _archive(
                ("one.txt", b"1"),
                ("two.txt", b"2"),
                ("three.txt", b"3"),
            )
        ),
        parent_entry_id="archive-1",
    )

    assert len(members) == 1
    assert members[-1].failure_code == ARCHIVE_MEMBER_LIMIT


def test_archive_member_limit_is_checked_before_zipfile_materializes_entries(
    monkeypatch,
):
    policy = DEFAULT_SCAN_POLICY.model_copy(update={"max_archive_members": 1})
    raw_archive = _archive(("one.txt", b"1"), ("two.txt", b"2"))

    class UnexpectedZipFile:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("ZipFile must not be constructed for declared overflow")

    monkeypatch.setattr(archive_module.zipfile, "ZipFile", UnexpectedZipFile)

    members = ArchiveInspector(policy).inspect(
        io.BytesIO(raw_archive),
        parent_entry_id="archive-1",
    )

    assert len(members) == 1
    assert members[0].failure_code == ARCHIVE_MEMBER_LIMIT


def test_archive_counts_actual_central_records_before_constructing_zipfile(monkeypatch):
    policy = DEFAULT_SCAN_POLICY.model_copy(update={"max_archive_members": 1})
    raw_archive = _forge_eocd_member_count(
        _archive(("one.txt", b"1"), ("two.txt", b"2")),
        member_count=1,
    )

    class UnexpectedZipFile:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("forged member overflow must not construct ZipFile")

    monkeypatch.setattr(archive_module.zipfile, "ZipFile", UnexpectedZipFile)

    members = ArchiveInspector(policy).inspect(
        io.BytesIO(raw_archive),
        parent_entry_id="archive-1",
    )

    assert len(members) == 1
    assert members[0].state is SourceState.FAILED
    assert members[0].failure_code == ARCHIVE_MEMBER_LIMIT


def test_archive_expanded_size_limit_is_enforced_cumulatively():
    policy = DEFAULT_SCAN_POLICY.model_copy(update={"max_archive_expanded_bytes": 5})

    members = ArchiveInspector(policy).inspect(
        io.BytesIO(_archive(("one.txt", b"1234"), ("two.txt", b"5678"))),
        parent_entry_id="archive-1",
    )

    assert members[0].state is SourceState.PENDING_APPROVAL
    assert members[1].state is SourceState.FAILED
    assert members[1].failure_code == ARCHIVE_SIZE_LIMIT


def test_archive_expansion_ratio_is_enforced_from_metadata():
    policy = DEFAULT_SCAN_POLICY.model_copy(update={"max_archive_ratio": 2.0})

    members = ArchiveInspector(policy).inspect(
        io.BytesIO(
            _archive(("compressed.txt", b"a" * 1_000), compression=zipfile.ZIP_DEFLATED)
        ),
        parent_entry_id="archive-1",
    )

    assert members[0].state is SourceState.FAILED
    assert members[0].failure_code == ARCHIVE_RATIO_LIMIT


def test_archive_member_depth_limit_is_enforced_from_metadata():
    policy = DEFAULT_SCAN_POLICY.model_copy(update={"max_depth": 2})

    members = ArchiveInspector(policy).inspect(
        io.BytesIO(_archive(("one/two/notes.txt", b"safe"))),
        parent_entry_id="archive-1",
    )

    assert members[0].state is SourceState.FAILED
    assert members[0].failure_code == ARCHIVE_DEPTH_LIMIT


def test_archive_member_metadata_is_immutable_and_preserves_parent_identity():
    members = ArchiveInspector().inspect(
        io.BytesIO(_archive(("folder/", b""), ("folder/notes.txt", b"safe"))),
        parent_entry_id="archive-1",
    )

    assert members[0].item_kind == "folder"
    assert members[1].item_kind == "file"
    assert all(member.parent_entry_id == "archive-1" for member in members)
    assert (members[1].member_index, members[1].crc32, members[1].compressed_bytes) == (
        2,
        0x1FA4288F,
        4,
    )
    assert isinstance(members, tuple)


def test_archive_rejects_duplicate_normalized_paths_even_when_sizes_match():
    with pytest.warns(UserWarning, match="Duplicate name"):
        content = _archive(("same.txt", b"safe"), ("same.txt", b"evil"))

    members = ArchiveInspector().inspect(
        io.BytesIO(content),
        parent_entry_id="archive-1",
    )

    assert len(members) == 2
    assert members[1].state is SourceState.FAILED
    assert members[1].failure_code == "archive_path_collision"
