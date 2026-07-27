from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

import pytest

from exam_predictor.evidence.artifacts import (
    ArtifactBoundaryError,
    ArtifactCleanupState,
    EvidenceArtifactStore,
)


WORKSPACE_ID = "workspace_0123456789"
PART_ID = "part_0123456789ab"
UNIT_ID = "unit_0123456789ab"
SNAPSHOT_ID = "snapshot_01234567"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _publish_part(store: EvidenceArtifactStore, content: bytes = b"course evidence") -> str:
    digest = _sha256(content)
    assert (
        store.publish_part(
            WORKSPACE_ID,
            PART_ID,
            content,
            expected_sha256=digest,
        )
        == digest
    )
    return digest


def _directory_link_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError as symlink_error:
        if os.name != "nt":
            pytest.skip(f"directory links are unavailable: {symlink_error}")
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/j", str(link), str(target)],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip("directory links are unavailable")


def test_replaced_evidence_directory_is_rejected_by_its_anchored_identity(tmp_path):
    store = EvidenceArtifactStore(tmp_path / "data")
    _publish_part(store)
    workspace = tmp_path / "data" / "workspaces" / WORKSPACE_ID
    evidence = workspace / "evidence"
    original = workspace / "original-evidence"
    outside = tmp_path / "outside"
    outside.mkdir()
    evidence.rename(original)
    _directory_link_or_skip(evidence, outside)

    with pytest.raises(ArtifactBoundaryError) as caught:
        with store.open_part(WORKSPACE_ID, PART_ID):
            pytest.fail("a replaced evidence directory must not be opened")

    assert caught.value.code == "artifact_identity_changed"
    assert not list(outside.iterdir())


def test_publish_part_atomically_reopens_with_the_expected_sha256(tmp_path):
    store = EvidenceArtifactStore(tmp_path / "data")
    content = b"\x00known multimodal evidence\xff"
    expected = _publish_part(store, content)

    with store.open_part(WORKSPACE_ID, PART_ID) as source:
        assert hashlib.sha256(source.read()).hexdigest() == expected

    part = tmp_path / "data" / "workspaces" / WORKSPACE_ID / "evidence" / "parts" / PART_ID
    assert part.read_bytes() == content
    assert not list(part.parent.glob(".examsage-artifact-*.tmp"))


def test_delete_workspace_refuses_a_native_source_path_without_touching_it(tmp_path):
    source = tmp_path / "native-course" / "notes.txt"
    source.parent.mkdir()
    original = b"native source bytes must never be touched"
    source.write_bytes(original)
    store = EvidenceArtifactStore(tmp_path / "data")

    with pytest.raises(ArtifactBoundaryError) as caught:
        store.delete_workspace(str(source.parent))

    assert caught.value.code == "artifact_identifier_invalid"
    assert source.read_bytes() == original


@pytest.mark.parametrize("artifact_type,artifact_id", [("units", UNIT_ID), ("snapshots", SNAPSHOT_ID)])
def test_publish_json_uses_only_the_fixed_owned_layout(tmp_path, artifact_type, artifact_id):
    store = EvidenceArtifactStore(tmp_path / "data")
    document = {"z": [2, 1], "a": "evidence"}
    encoded = b'{"a":"evidence","z":[2,1]}'

    digest = store.publish_json(
        WORKSPACE_ID,
        artifact_type,
        artifact_id,
        document,
        expected_sha256=_sha256(encoded),
    )

    assert digest == _sha256(encoded)
    assert store.read_json(WORKSPACE_ID, artifact_type, artifact_id) == document
    expected_path = (
        tmp_path / "data" / "workspaces" / WORKSPACE_ID / "evidence" / artifact_type / f"{artifact_id}.json"
    )
    assert expected_path.read_bytes() == encoded


def test_hash_mismatch_never_publishes_or_leaves_a_temporary_file(tmp_path):
    store = EvidenceArtifactStore(tmp_path / "data")

    with pytest.raises(ArtifactBoundaryError) as caught:
        store.publish_part(
            WORKSPACE_ID,
            PART_ID,
            b"content",
            expected_sha256="0" * 64,
        )

    assert caught.value.code == "artifact_hash_mismatch"
    parts = tmp_path / "data" / "workspaces" / WORKSPACE_ID / "evidence" / "parts"
    assert not (parts / PART_ID).exists()
    assert not list(parts.glob(".examsage-artifact-*.tmp"))


def test_open_part_rejects_a_hardlink_substitution(tmp_path):
    store = EvidenceArtifactStore(tmp_path / "data")
    _publish_part(store)
    part = tmp_path / "data" / "workspaces" / WORKSPACE_ID / "evidence" / "parts" / PART_ID
    external = tmp_path / "external-copy"
    part.rename(external)
    os.link(external, part)

    with pytest.raises(ArtifactBoundaryError) as caught:
        with store.open_part(WORKSPACE_ID, PART_ID):
            pytest.fail("a hard-linked artifact must not be opened")

    assert caught.value.code == "artifact_identity_changed"
    assert external.read_bytes() == b"course evidence"


def test_delete_workspace_refuses_a_hardlinked_artifact(tmp_path):
    store = EvidenceArtifactStore(tmp_path / "data")
    _publish_part(store)
    part = tmp_path / "data" / "workspaces" / WORKSPACE_ID / "evidence" / "parts" / PART_ID
    external = tmp_path / "external-copy"
    os.link(part, external)

    with pytest.raises(ArtifactBoundaryError) as caught:
        store.delete_workspace(WORKSPACE_ID)

    assert caught.value.code == "artifact_identity_changed"
    assert part.read_bytes() == b"course evidence"
    assert external.read_bytes() == b"course evidence"


class _SharingViolationOnce:
    def __init__(self) -> None:
        self.failed = False

    def remove_owned_tree(self, data_root: Path, claim) -> None:
        if not self.failed:
            self.failed = True
            error = PermissionError("sharing violation")
            error.winerror = 32
            raise error
        from exam_predictor.workspace.browser_intake import OwnedTreeRemover

        OwnedTreeRemover(data_root)(claim)


class _RetargetBeforeReplace:
    def __init__(self, parts: Path, outside: Path) -> None:
        self.parts = parts
        self.outside = outside

    def atomic_replace(self, source, destination, *, directory_fd=None):
        original = self.parts.with_name("original-parts")
        self.parts.rename(original)
        _directory_link_or_skip(self.parts, self.outside)
        os.replace(
            source,
            destination,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )


def test_publish_reports_identity_change_when_parent_is_retargeted_at_replace(
    tmp_path,
):
    data_root = tmp_path / "data"
    warmup = EvidenceArtifactStore(data_root)
    parts = data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts"
    warmup.publish_part(
        WORKSPACE_ID,
        PART_ID,
        b"first",
        expected_sha256=_sha256(b"first"),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    store = EvidenceArtifactStore(
        data_root,
        filesystem_ops=_RetargetBeforeReplace(parts, outside),
    )

    with pytest.raises(ArtifactBoundaryError) as caught:
        store.publish_part(
            WORKSPACE_ID,
            PART_ID,
            b"second",
            expected_sha256=_sha256(b"second"),
        )

    assert caught.value.code == "artifact_identity_changed"
    assert not list(outside.iterdir())


def test_sharing_violation_is_cleanup_pending_and_restart_can_retry(tmp_path):
    operations = _SharingViolationOnce()
    data_root = tmp_path / "data"
    store = EvidenceArtifactStore(data_root, filesystem_ops=operations)
    _publish_part(store)

    assert store.delete_workspace(WORKSPACE_ID) is ArtifactCleanupState.CLEANUP_PENDING
    assert (
        data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts" / PART_ID
    ).read_bytes() == b"course evidence"

    restarted = EvidenceArtifactStore(data_root, filesystem_ops=operations)
    assert restarted.delete_workspace(WORKSPACE_ID) is ArtifactCleanupState.DELETED
    assert not (data_root / "workspaces" / WORKSPACE_ID / "evidence").exists()


def test_restart_reanchors_and_reads_an_existing_artifact(tmp_path):
    data_root = tmp_path / "data"
    first = EvidenceArtifactStore(data_root)
    expected = _publish_part(first)

    restarted = EvidenceArtifactStore(data_root)
    with restarted.open_part(WORKSPACE_ID, PART_ID) as source:
        assert hashlib.sha256(source.read()).hexdigest() == expected


@pytest.mark.skipif(os.name != "nt", reason="Windows no-delete-share handles")
def test_windows_open_handle_blocks_directory_retargeting(tmp_path):
    store = EvidenceArtifactStore(tmp_path / "data")
    _publish_part(store)
    evidence = tmp_path / "data" / "workspaces" / WORKSPACE_ID / "evidence"

    with store.open_part(WORKSPACE_ID, PART_ID):
        with pytest.raises(OSError):
            evidence.rename(evidence.with_name("retargeted-evidence"))


@pytest.mark.parametrize(
    "invalid_identifier",
    ["short", "a" * 129, "../outside________", "identifier.with.dot"],
)
def test_every_identifier_uses_the_exact_bounded_pattern(tmp_path, invalid_identifier):
    store = EvidenceArtifactStore(tmp_path / "data")

    with pytest.raises(ArtifactBoundaryError) as caught:
        store.publish_part(
            WORKSPACE_ID,
            invalid_identifier,
            b"content",
            expected_sha256=_sha256(b"content"),
        )

    assert caught.value.code == "artifact_identifier_invalid"
