from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from io import BytesIO
from pathlib import Path
import zipfile

from exam_predictor.evidence.artifacts import EvidenceArtifactStore
from exam_predictor.evidence.preparation import (
    ArchivePreviewAuthority,
    PreparedPartRequest,
    SourcePartPreparer,
)
from exam_predictor.workspace.models import (
    SourceMode,
    WorkspaceJob,
    WorkspaceJobStatus,
    WorkspaceRecord,
    WorkspaceState,
)
from exam_predictor.workspace.scanner import WorkspaceScanner
from exam_predictor.workspace.store import WorkspaceStore


WORKSPACE_ID = "workspace_archive_pipeline_000000001"
JOB_ID = "job_archive_pipeline_000000000001"


def test_scanned_archive_identity_survives_storage_and_selects_the_exact_member(
    tmp_path: Path,
):
    source_root = tmp_path / "course"
    source_root.mkdir()
    archive_path = source_root / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("first.txt", b"AAAA")
        archive.writestr("second.txt", b"BBBB")
    archive_bytes = archive_path.read_bytes()

    execution = WorkspaceScanner().scan_with_identity(WORKSPACE_ID, source_root)
    parent = next(
        entry
        for entry in execution.result.entries
        if entry.relative_path == "bundle.zip" and entry.archive_member_path is None
    )
    scanned_member = next(
        entry
        for entry in execution.result.entries
        if entry.archive_member_path == "second.txt"
    )
    with zipfile.ZipFile(BytesIO(archive_bytes), "r") as archive:
        expected = archive.infolist()[1]
    assert (
        scanned_member.archive_member_index,
        scanned_member.archive_member_crc32,
        scanned_member.archive_member_compressed_bytes,
    ) == (2, expected.CRC, expected.compress_size)

    now = datetime.now(UTC)
    database_path = tmp_path / "workspace.sqlite3"
    store = WorkspaceStore(database_path)
    try:
        store.create_workspace(
            WorkspaceRecord(
                workspace_id=WORKSPACE_ID,
                display_name="Archive pipeline",
                source_mode=SourceMode.NATIVE_FOLDER,
                canonical_root=execution.canonical_root,
                root_device=execution.root_device,
                root_file_id=execution.root_file_id,
                state=WorkspaceState.SCANNING,
                created_at=now,
                updated_at=now,
            )
        )
        store.create_job(
            WorkspaceJob(
                job_id=JOB_ID,
                workspace_id=WORKSPACE_ID,
                job_kind="scan",
                status=WorkspaceJobStatus.QUEUED,
                idempotency_key="archive-pipeline-request",
                created_at=now,
            ),
            "archive-pipeline-request",
        )
        store.start_job(JOB_ID)
        revision = store.commit_scan(WORKSPACE_ID, execution.result, JOB_ID)
    finally:
        store.close()

    reopened = WorkspaceStore(database_path)
    try:
        persisted = reopened.get_manifest(WORKSPACE_ID, revision.revision_id)
        persisted_parent = next(
            entry for entry in persisted.entries if entry.entry_id == parent.entry_id
        )
        persisted_member = next(
            entry for entry in persisted.entries if entry.entry_id == scanned_member.entry_id
        )
    finally:
        reopened.close()
    assert (
        persisted_member.archive_member_index,
        persisted_member.archive_member_crc32,
        persisted_member.archive_member_compressed_bytes,
    ) == (2, expected.CRC, expected.compress_size)

    authority = ArchivePreviewAuthority(
        workspace_id=WORKSPACE_ID,
        revision_id=revision.revision_id,
        parent_entry_id=persisted_parent.entry_id,
        parent_source_sha256=persisted_parent.sha256,
        entry=persisted_member,
        approved=True,
    )
    request = PreparedPartRequest(
        workspace_id=WORKSPACE_ID,
        revision_id=revision.revision_id,
        entry_id=persisted_parent.entry_id,
        relative_path=persisted_parent.relative_path,
        format_category=persisted_parent.format_category,
        source_size_bytes=len(archive_bytes),
        source_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        archive_previews=(authority,),
    )
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifacts = EvidenceArtifactStore(artifact_root)
    try:
        plans = SourcePartPreparer(artifacts).prepare(request, BytesIO(archive_bytes))
        assert len(plans) == 1
        with artifacts.open_part(WORKSPACE_ID, plans[0].part_id) as stream:
            assert stream.read() == b"BBBB"
    finally:
        artifacts.close()
