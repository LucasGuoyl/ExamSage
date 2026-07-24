from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier

import pytest

from exam_predictor.workspace.models import (
    ScanPolicy,
    SourceMode,
    SourceState,
    WorkspaceJob,
    WorkspaceJobStatus,
    WorkspaceRecord,
    WorkspaceState,
)
from exam_predictor.workspace.scanner import WorkspaceScanner
from exam_predictor.workspace.store import WorkspaceStore
from exam_predictor.workspace.transmission import (
    SourceAuthorizationError,
    WorkspaceTransmissionGate,
)


NOW = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path):
    value = WorkspaceStore(tmp_path / "workspace.sqlite3")
    try:
        yield value
    finally:
        value.close()


def _approved_workspace(
    store: WorkspaceStore,
    tmp_path: Path,
    *,
    workspace_id: str = "workspace-1",
    sources: dict[str, bytes] | None = None,
    approve: bool = True,
) -> tuple[Path, tuple[str, ...]]:
    root = tmp_path / workspace_id
    root.mkdir()
    for relative_path, content in (sources or {"notes.txt": b"approved revision"}).items():
        source = root / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
    execution = WorkspaceScanner().scan_with_identity(workspace_id, root)
    store.create_workspace(
        WorkspaceRecord(
            workspace_id=workspace_id,
            display_name="Course",
            source_mode=SourceMode.NATIVE_FOLDER,
            canonical_root=execution.canonical_root,
            root_device=execution.root_device,
            root_file_id=execution.root_file_id,
            state=WorkspaceState.SCANNING,
            created_at=NOW,
            updated_at=NOW,
        )
    )
    job = WorkspaceJob(
        job_id=f"scan-{workspace_id}",
        workspace_id=workspace_id,
        job_kind="scan",
        status=WorkspaceJobStatus.QUEUED,
        idempotency_key=f"request-{workspace_id}",
        created_at=NOW,
    )
    store.create_job(job, job.idempotency_key)
    store.start_job(job.job_id)
    revision = store.commit_scan(workspace_id, execution.result, job.job_id)
    if approve:
        store.approve(workspace_id, revision.revision_id, revision.policy_version)
    return root, tuple(entry.entry_id for entry in revision.entries if entry.included)


def test_valid_approved_source_is_path_safe_and_reads_once(
    store: WorkspaceStore, tmp_path: Path
):
    root, entry_ids = _approved_workspace(store, tmp_path)
    entry_id = entry_ids[0]
    gate = WorkspaceTransmissionGate(store)

    approved = gate.authorize("workspace-1", [entry_id])

    assert len(approved) == 1
    descriptor = approved[0]
    assert descriptor.workspace_id == "workspace-1"
    assert descriptor.entry_id == entry_id
    assert descriptor.relative_path == "notes.txt"
    serialized = descriptor.model_dump_json()
    assert str(tmp_path) not in serialized
    assert descriptor.read_token.get_secret_value() not in serialized
    assert json.loads(serialized)["read_token"] == "**********"
    with gate.open_approved(descriptor.read_token) as handle:
        assert handle.read() == b"approved revision"
    with pytest.raises(SourceAuthorizationError) as caught:
        with gate.open_approved(descriptor.read_token):
            pass
    assert caught.value.code == "read_token_invalid"
    assert str(tmp_path) not in str(caught.value)
    assert store.get_workspace("workspace-1").last_access_verified_at is not None


@pytest.mark.parametrize(
    ("requested", "expected_code"),
    [
        ([], "source_request_invalid"),
        (["duplicate", "duplicate"], "source_request_invalid"),
        (["missing"], "source_not_approved"),
    ],
)
def test_rejects_invalid_or_unapproved_requests_without_tokens(
    store: WorkspaceStore,
    tmp_path: Path,
    requested: list[str],
    expected_code: str,
):
    _, entry_ids = _approved_workspace(store, tmp_path)
    if requested == ["duplicate", "duplicate"]:
        requested = [entry_ids[0], entry_ids[0]]
    gate = WorkspaceTransmissionGate(store)

    with pytest.raises(SourceAuthorizationError) as caught:
        gate.authorize("workspace-1", requested)

    assert caught.value.code == expected_code
    assert gate._read_grants == {}


def test_rejects_a_workspace_without_approval(store: WorkspaceStore, tmp_path: Path):
    _, entry_ids = _approved_workspace(store, tmp_path, approve=False)

    with pytest.raises(SourceAuthorizationError) as caught:
        WorkspaceTransmissionGate(store).authorize("workspace-1", [entry_ids[0]])

    assert caught.value.code == "source_approval_required"


def test_rejects_stale_policy_and_approval_hash_mismatch(
    store: WorkspaceStore, tmp_path: Path
):
    _, entry_ids = _approved_workspace(store, tmp_path)
    entry_id = entry_ids[0]
    stale_gate = WorkspaceTransmissionGate(
        store,
        ScanPolicy(policy_version="workspace-v2"),
    )

    with pytest.raises(SourceAuthorizationError) as stale:
        stale_gate.authorize("workspace-1", [entry_id])
    assert stale.value.code == "source_approval_stale_policy"

    with store._transaction() as connection:
        connection.execute(
            "UPDATE approvals SET approved_entries_json = ? WHERE workspace_id = ?",
            (
                json.dumps([{"entry_id": entry_id, "sha256": "0" * 64}]),
                "workspace-1",
            ),
        )
    with pytest.raises(SourceAuthorizationError) as mismatch:
        WorkspaceTransmissionGate(store).authorize("workspace-1", [entry_id])
    assert mismatch.value.code == "source_approval_mismatch"


def test_archive_preview_member_is_never_authorized(
    store: WorkspaceStore, tmp_path: Path
):
    import zipfile

    root = tmp_path / "workspace-1"
    root.mkdir()
    archive = root / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("inside.txt", b"preview")
    execution = WorkspaceScanner().scan_with_identity("workspace-1", root)
    store.create_workspace(
        WorkspaceRecord(
            workspace_id="workspace-1",
            display_name="Course",
            source_mode=SourceMode.NATIVE_FOLDER,
            canonical_root=execution.canonical_root,
            root_device=execution.root_device,
            root_file_id=execution.root_file_id,
            state=WorkspaceState.SCANNING,
            created_at=NOW,
            updated_at=NOW,
        )
    )
    job = WorkspaceJob(
        job_id="scan-workspace-1",
        workspace_id="workspace-1",
        job_kind="scan",
        status=WorkspaceJobStatus.QUEUED,
        idempotency_key="request-workspace-1",
        created_at=NOW,
    )
    store.create_job(job, job.idempotency_key)
    store.start_job(job.job_id)
    revision = store.commit_scan("workspace-1", execution.result, job.job_id)
    member = next(
        entry for entry in revision.entries if entry.archive_parent_entry_id is not None
    )
    parent = next(entry for entry in revision.entries if entry.entry_id != member.entry_id)
    with store._transaction() as connection:
        connection.execute(
            """UPDATE manifest_entries
               SET included = 1, state = ?, sha256 = ?
               WHERE revision_id = ? AND entry_id = ?""",
            (
                SourceState.PENDING_APPROVAL.value,
                parent.sha256,
                revision.revision_id,
                member.entry_id,
            ),
        )
    store.approve("workspace-1", revision.revision_id, revision.policy_version)

    with pytest.raises(SourceAuthorizationError) as caught:
        WorkspaceTransmissionGate(store).authorize("workspace-1", [member.entry_id])

    assert caught.value.code == "source_not_approved"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda path: path.write_bytes(b"larger changed revision"),
        lambda path: path.touch(),
        lambda path: (path.unlink(), path.write_bytes(b"approved revision")),
    ],
    ids=["size-and-hash", "mtime-only", "identity"],
)
def test_authorize_marks_mutated_sources_changed_before_provider_use(
    store: WorkspaceStore,
    tmp_path: Path,
    mutation,
):
    root, entry_ids = _approved_workspace(store, tmp_path)
    entry_id = entry_ids[0]
    original_approval = store.get_approval("workspace-1")
    source = root / "notes.txt"
    provider_calls = 0
    mutation(source)

    with pytest.raises(SourceAuthorizationError) as caught:
        WorkspaceTransmissionGate(store).authorize("workspace-1", [entry_id])
        provider_calls += 1

    assert caught.value.code == "approved_source_changed"
    assert provider_calls == 0
    assert store.get_workspace("workspace-1").state is WorkspaceState.NEEDS_ATTENTION
    assert store.get_approval("workspace-1") == original_approval


def test_authorize_is_all_or_nothing_for_a_changed_batch(
    store: WorkspaceStore, tmp_path: Path
):
    root, entry_ids = _approved_workspace(
        store,
        tmp_path,
        sources={"first.txt": b"first", "second.txt": b"second"},
    )
    (root / "second.txt").write_bytes(b"changed")
    gate = WorkspaceTransmissionGate(store)

    with pytest.raises(SourceAuthorizationError) as caught:
        gate.authorize("workspace-1", list(entry_ids))

    assert caught.value.code == "approved_source_changed"
    assert gate._read_grants == {}


def test_tokens_expire_and_restart_invalidates_them(
    store: WorkspaceStore, tmp_path: Path
):
    _, entry_ids = _approved_workspace(store, tmp_path)
    expiring = WorkspaceTransmissionGate(store, token_ttl_seconds=0)
    expired = expiring.authorize("workspace-1", [entry_ids[0]])[0]

    with pytest.raises(SourceAuthorizationError) as caught:
        with expiring.open_approved(expired.read_token):
            pass
    assert caught.value.code == "read_token_expired"

    first_gate = WorkspaceTransmissionGate(store)
    token = first_gate.authorize("workspace-1", [entry_ids[0]])[0].read_token
    with pytest.raises(SourceAuthorizationError) as restarted:
        with WorkspaceTransmissionGate(store).open_approved(token):
            pass
    assert restarted.value.code == "read_token_invalid"


def test_two_racing_consumers_have_exactly_one_success(
    store: WorkspaceStore, tmp_path: Path
):
    _, entry_ids = _approved_workspace(store, tmp_path)
    gate = WorkspaceTransmissionGate(store)
    token = gate.authorize("workspace-1", [entry_ids[0]])[0].read_token
    barrier = Barrier(2)

    def consume():
        barrier.wait()
        try:
            with gate.open_approved(token) as source:
                return source.read()
        except SourceAuthorizationError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: consume(), range(2)))

    assert outcomes.count(b"approved revision") == 1
    assert outcomes.count("read_token_invalid") == 1


def test_root_substitution_marks_attention_without_rewriting_the_grant(
    store: WorkspaceStore, tmp_path: Path
):
    root, entry_ids = _approved_workspace(store, tmp_path)
    before = store.get_workspace("workspace-1")
    displaced = tmp_path / "displaced"
    root.rename(displaced)
    root.mkdir()
    (root / "notes.txt").write_bytes(b"approved revision")

    with pytest.raises(SourceAuthorizationError) as caught:
        WorkspaceTransmissionGate(store).authorize("workspace-1", [entry_ids[0]])

    after = store.get_workspace("workspace-1")
    assert caught.value.code == "source_root_identity_changed"
    assert after.state is WorkspaceState.NEEDS_ATTENTION
    assert (after.root_device, after.root_file_id) == (
        before.root_device,
        before.root_file_id,
    )


def test_post_authorize_same_metadata_mutation_fails_on_the_same_open_handle(
    store: WorkspaceStore, tmp_path: Path
):
    root, entry_ids = _approved_workspace(store, tmp_path)
    source = root / "notes.txt"
    gate = WorkspaceTransmissionGate(store)
    descriptor = gate.authorize("workspace-1", [entry_ids[0]])[0]
    original = source.stat()
    source.write_bytes(b"changed! revision")
    os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))

    with pytest.raises(SourceAuthorizationError) as caught:
        with gate.open_approved(descriptor.read_token):
            pass

    assert caught.value.code == "approved_source_changed"
    assert store.get_workspace("workspace-1").state is WorkspaceState.NEEDS_ATTENTION
    with pytest.raises(SourceAuthorizationError) as consumed:
        with gate.open_approved(descriptor.read_token):
            pass
    assert consumed.value.code == "read_token_invalid"


def test_missing_root_fails_without_leaking_a_path(
    store: WorkspaceStore, tmp_path: Path
):
    root, entry_ids = _approved_workspace(store, tmp_path)
    shutil.rmtree(root)

    with pytest.raises(SourceAuthorizationError) as caught:
        WorkspaceTransmissionGate(store).authorize("workspace-1", [entry_ids[0]])

    assert caught.value.code == "source_root_invalid"
    assert str(tmp_path) not in str(caught.value)
    assert caught.value.__cause__ is None
