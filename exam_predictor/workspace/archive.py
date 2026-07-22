from __future__ import annotations

import re
import stat
import struct
import zipfile
from pathlib import PurePosixPath
from typing import BinaryIO, Sequence

from exam_predictor.workspace.models import ArchiveMember, ScanPolicy, SourceState
from exam_predictor.workspace.policy import DEFAULT_SCAN_POLICY


ARCHIVE_TRAVERSAL = "archive_traversal"
ARCHIVE_ABSOLUTE_PATH = "archive_absolute_path"
ARCHIVE_LINK = "archive_link"
ARCHIVE_ENCRYPTED = "archive_encrypted"
ARCHIVE_MEMBER_LIMIT = "archive_member_limit"
ARCHIVE_SIZE_LIMIT = "archive_size_limit"
ARCHIVE_RATIO_LIMIT = "archive_ratio_limit"
ARCHIVE_DEPTH_LIMIT = "archive_depth_limit"
_END_OF_CENTRAL_DIRECTORY = b"PK\x05\x06"
_END_OF_CENTRAL_DIRECTORY_SIZE = 22
_MAX_ZIP_COMMENT_BYTES = 65_535
_CENTRAL_DIRECTORY_HEADER = b"PK\x01\x02"
_CENTRAL_DIRECTORY_HEADER_SIZE = 46


def _normalized_member_path(info: zipfile.ZipInfo) -> tuple[str, PurePosixPath]:
    raw = info.filename.replace("\\", "/")
    return raw, PurePosixPath(raw)


def _member_failure(info: zipfile.ZipInfo, policy: ScanPolicy) -> str | None:
    raw, member_path = _normalized_member_path(info)
    if member_path.is_absolute() or re.match(r"^[A-Za-z]:", raw):
        return ARCHIVE_ABSOLUTE_PATH
    if ".." in member_path.parts:
        return ARCHIVE_TRAVERSAL
    unix_kind = (info.external_attr >> 16) & 0o170000
    if unix_kind == stat.S_IFLNK:
        return ARCHIVE_LINK
    if info.flag_bits & 0x1:
        return ARCHIVE_ENCRYPTED
    if len(member_path.parts) > policy.max_depth:
        return ARCHIVE_DEPTH_LIMIT
    if info.file_size / max(info.compress_size, 1) > policy.max_archive_ratio:
        return ARCHIVE_RATIO_LIMIT
    return None


def _declared_central_directory(
    archive_stream: BinaryIO,
) -> tuple[int, int, int, int]:
    original_position = archive_stream.tell()
    try:
        archive_stream.seek(0, 2)
        archive_size = archive_stream.tell()
        tail_size = min(
            archive_size,
            _END_OF_CENTRAL_DIRECTORY_SIZE + _MAX_ZIP_COMMENT_BYTES,
        )
        archive_stream.seek(archive_size - tail_size)
        tail = archive_stream.read(tail_size)
        search_end = len(tail)
        while True:
            offset = tail.rfind(_END_OF_CENTRAL_DIRECTORY, 0, search_end)
            if offset < 0:
                raise zipfile.BadZipFile("end of central directory not found")
            if offset + _END_OF_CENTRAL_DIRECTORY_SIZE <= len(tail):
                fields = struct.unpack_from("<4s4H2LH", tail, offset)
                comment_bytes = fields[7]
                if offset + _END_OF_CENTRAL_DIRECTORY_SIZE + comment_bytes == len(tail):
                    entries_total = fields[4]
                    central_directory_bytes = fields[5]
                    central_directory_offset = fields[6]
                    eocd_offset = archive_size - tail_size + offset
                    return (
                        entries_total,
                        central_directory_bytes,
                        central_directory_offset,
                        eocd_offset,
                    )
            search_end = offset
    finally:
        archive_stream.seek(original_position)


def _count_central_directory_records(
    archive_stream: BinaryIO,
    *,
    central_directory_bytes: int,
    central_directory_offset: int,
    eocd_offset: int,
    max_members: int,
) -> int:
    original_position = archive_stream.tell()
    try:
        prefix_bytes = eocd_offset - central_directory_bytes - central_directory_offset
        actual_offset = central_directory_offset + prefix_bytes
        if prefix_bytes < 0 or actual_offset < 0:
            raise zipfile.BadZipFile("invalid central directory offset")
        if actual_offset + central_directory_bytes > eocd_offset:
            raise zipfile.BadZipFile("central directory overlaps end record")

        archive_stream.seek(actual_offset)
        remaining = central_directory_bytes
        member_count = 0
        while remaining:
            if remaining < _CENTRAL_DIRECTORY_HEADER_SIZE:
                raise zipfile.BadZipFile("truncated central directory")
            header = archive_stream.read(_CENTRAL_DIRECTORY_HEADER_SIZE)
            if len(header) != _CENTRAL_DIRECTORY_HEADER_SIZE:
                raise zipfile.BadZipFile("truncated central directory")
            fields = struct.unpack("<4s6H3L5H2L", header)
            if fields[0] != _CENTRAL_DIRECTORY_HEADER:
                raise zipfile.BadZipFile("invalid central directory signature")
            variable_bytes = fields[10] + fields[11] + fields[12]
            record_bytes = _CENTRAL_DIRECTORY_HEADER_SIZE + variable_bytes
            if record_bytes > remaining:
                raise zipfile.BadZipFile("invalid central directory record length")
            member_count += 1
            if member_count > max_members:
                return member_count
            archive_stream.seek(variable_bytes, 1)
            remaining -= record_bytes
        return member_count
    finally:
        archive_stream.seek(original_position)


def _archive_limit_member(parent_entry_id: str, failure_code: str) -> ArchiveMember:
    return ArchiveMember(
        parent_entry_id=parent_entry_id,
        display_path="[archive limit exceeded]",
        item_kind="archive_limit",
        size_bytes=0,
        compressed_bytes=0,
        state=SourceState.FAILED,
        failure_code=failure_code,
    )


class ArchiveInspector:
    def __init__(self, policy: ScanPolicy = DEFAULT_SCAN_POLICY) -> None:
        self._policy = policy

    def inspect(
        self, archive_stream: BinaryIO, *, parent_entry_id: str
    ) -> Sequence[ArchiveMember]:
        (
            declared_members,
            central_directory_bytes,
            central_directory_offset,
            eocd_offset,
        ) = _declared_central_directory(archive_stream)
        max_central_directory_bytes = max(
            4_096,
            self._policy.max_archive_members
            * (46 + self._policy.max_path_chars * 4 + 1_024),
        )
        if (
            declared_members > self._policy.max_archive_members
            or declared_members == 0xFFFF
            or central_directory_bytes == 0xFFFFFFFF
            or central_directory_bytes > max_central_directory_bytes
        ):
            return (_archive_limit_member(parent_entry_id, ARCHIVE_MEMBER_LIMIT),)

        actual_members = _count_central_directory_records(
            archive_stream,
            central_directory_bytes=central_directory_bytes,
            central_directory_offset=central_directory_offset,
            eocd_offset=eocd_offset,
            max_members=self._policy.max_archive_members,
        )
        if actual_members > self._policy.max_archive_members:
            return (_archive_limit_member(parent_entry_id, ARCHIVE_MEMBER_LIMIT),)
        if actual_members != declared_members:
            raise zipfile.BadZipFile("central directory member count mismatch")

        members: list[ArchiveMember] = []
        expanded_bytes = 0
        with zipfile.ZipFile(archive_stream, "r") as archive:
            for member_count, info in enumerate(archive.infolist(), start=1):
                expanded_bytes += info.file_size
                aggregate_limit_exceeded = (
                    member_count > self._policy.max_archive_members
                    or expanded_bytes > self._policy.max_archive_expanded_bytes
                )
                failure_code = _member_failure(info, self._policy)
                if failure_code is None and member_count > self._policy.max_archive_members:
                    failure_code = ARCHIVE_MEMBER_LIMIT
                if failure_code is None and expanded_bytes > self._policy.max_archive_expanded_bytes:
                    failure_code = ARCHIVE_SIZE_LIMIT

                _, member_path = _normalized_member_path(info)
                members.append(
                    ArchiveMember(
                        parent_entry_id=parent_entry_id,
                        display_path=member_path.as_posix(),
                        item_kind="folder" if info.is_dir() else "file",
                        size_bytes=info.file_size,
                        compressed_bytes=info.compress_size,
                        state=(SourceState.FAILED if failure_code else SourceState.PENDING_APPROVAL),
                        failure_code=failure_code,
                    )
                )
                if aggregate_limit_exceeded:
                    break
        return tuple(members)
