from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from exam_predictor.evidence.artifacts import (
    ArtifactBoundaryError,
    ArtifactCleanupState,
    EvidenceArtifactStore,
    NativeArtifactFilesystemOps,
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


@pytest.mark.parametrize("document", [[], "text", 1, None])
def test_publish_json_rejects_nonobject_documents(tmp_path, document):
    store = EvidenceArtifactStore(tmp_path / "data")

    with pytest.raises(ArtifactBoundaryError) as caught:
        store.publish_json(
            WORKSPACE_ID,
            "units",
            UNIT_ID,
            document,
            expected_sha256=_sha256(b"irrelevant"),
        )

    assert caught.value.code == "artifact_json_invalid"


def test_publish_json_rejects_oversized_and_recursive_documents(tmp_path):
    store = EvidenceArtifactStore(tmp_path / "data")
    recursive: dict[str, object] = {}
    recursive["self"] = recursive

    for document, code in [
        ({"content": "x" * (8 * 1024 * 1024 + 1)}, "artifact_json_too_large"),
        (recursive, "artifact_json_invalid"),
    ]:
        with pytest.raises(ArtifactBoundaryError) as caught:
            store.publish_json(
                WORKSPACE_ID,
                "units",
                UNIT_ID,
                document,
                expected_sha256=_sha256(b"irrelevant"),
            )
        assert caught.value.code == code


def test_publish_and_read_json_reject_excessive_depth_and_malformed_bytes(tmp_path, monkeypatch):
    store = EvidenceArtifactStore(tmp_path / "data")
    deep: object = "leaf"
    for _ in range(70):
        deep = {"child": deep}

    with pytest.raises(ArtifactBoundaryError) as caught:
        store.publish_json(
            WORKSPACE_ID,
            "units",
            UNIT_ID,
            deep,
            expected_sha256=_sha256(b"irrelevant"),
        )
    assert caught.value.code == "artifact_json_too_deep"

    document = {"content": "valid"}
    encoded = b'{"content":"valid"}'
    store.publish_json(
        WORKSPACE_ID,
        "units",
        UNIT_ID,
        document,
        expected_sha256=_sha256(encoded),
    )
    path = tmp_path / "data" / "workspaces" / WORKSPACE_ID / "evidence" / "units" / f"{UNIT_ID}.json"
    path.write_bytes(b"{malformed")
    claim = store._ownership[WORKSPACE_ID]["artifacts"][f"units/{UNIT_ID}.json"]
    claim["identity"] = [path.stat().st_dev, path.stat().st_ino]
    claim["sha256"] = _sha256(b"{malformed")
    monkeypatch.setattr(store, "_load_or_create_ownership", lambda *args, **kwargs: None)

    with pytest.raises(ArtifactBoundaryError) as caught:
        store.read_json(WORKSPACE_ID, "units", UNIT_ID)
    assert caught.value.code == "artifact_json_invalid"


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


def test_delete_workspace_refuses_an_unknown_owned_subtree(tmp_path):
    data_root = tmp_path / "data"
    store = EvidenceArtifactStore(data_root)
    _publish_part(store)
    unknown = data_root / "workspaces" / WORKSPACE_ID / "evidence" / "unknown"
    unknown.mkdir()
    marker = unknown / "native.bin"
    marker.write_bytes(b"must remain")

    result = store.delete_workspace(WORKSPACE_ID)

    assert result is ArtifactCleanupState.CLEANUP_PENDING
    assert marker.read_bytes() == b"must remain"
    assert (
        data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts" / PART_ID
    ).read_bytes() == b"course evidence"


class _HardlinkSwapDuringRemove(NativeArtifactFilesystemOps):
    def __init__(self, part: Path, external: Path) -> None:
        self.part = part
        self.external = external

    def before_remove(self, data_root, claim) -> None:
        del data_root, claim
        self.part.unlink()
        os.link(self.external, self.part)


def test_delete_rechecks_hardlink_identity_after_judgement_before_delete(tmp_path):
    data_root = tmp_path / "data"
    initial = EvidenceArtifactStore(data_root)
    _publish_part(initial)
    part = data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts" / PART_ID
    external = tmp_path / "native-source"
    external.write_bytes(b"native bytes")
    store = EvidenceArtifactStore(
        data_root,
        filesystem_ops=_HardlinkSwapDuringRemove(part, external),
    )

    if os.name == "nt":
        assert store.delete_workspace(WORKSPACE_ID) is ArtifactCleanupState.CLEANUP_PENDING
        assert part.read_bytes() == b"course evidence"
    else:
        with pytest.raises(ArtifactBoundaryError) as caught:
            store.delete_workspace(WORKSPACE_ID)
        assert caught.value.code == "artifact_identity_changed"
        assert part.read_bytes() == b"native bytes"
    assert external.read_bytes() == b"native bytes"


class _AlwaysSharingViolation(NativeArtifactFilesystemOps):
    def remove_owned_tree(self, data_root: Path, claim) -> None:
        del data_root, claim
        error = PermissionError("sharing violation")
        error.winerror = 32
        raise error


class _RetargetBeforeReplace:
    def __init__(self, parts: Path, outside: Path) -> None:
        self.parts = parts
        self.outside = outside
        self.native = NativeArtifactFilesystemOps()

    def atomic_replace(
        self,
        source,
        destination,
        *,
        directory_fd,
        expected_identity,
        expected_parent_identity,
        expected_sha256,
    ):
        original = self.parts.with_name("original-parts")
        self.parts.rename(original)
        _directory_link_or_skip(self.parts, self.outside)
        return self.native.atomic_replace(
            source,
            destination,
            directory_fd=directory_fd,
            expected_identity=expected_identity,
            expected_parent_identity=expected_parent_identity,
            expected_sha256=expected_sha256,
        )


class _MutateTempBeforeReplace(NativeArtifactFilesystemOps):
    def atomic_replace(
        self,
        source,
        destination,
        *,
        directory_fd,
        expected_identity,
        expected_parent_identity,
        expected_sha256,
    ):
        flags = os.O_WRONLY | os.O_TRUNC
        descriptor = (
            os.open(source, flags) if directory_fd is None else os.open(source, flags, dir_fd=directory_fd)
        )
        try:
            os.write(descriptor, b"tampered-in-place")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return super().atomic_replace(
            source,
            destination,
            directory_fd=directory_fd,
            expected_identity=expected_identity,
            expected_parent_identity=expected_parent_identity,
            expected_sha256=expected_sha256,
        )


class _WrongPostDigest(NativeArtifactFilesystemOps):
    def atomic_replace(self, *args, **kwargs):
        result = super().atomic_replace(*args, **kwargs)
        identity = kwargs["expected_identity"] if result is None else result[0]
        return identity, "0" * 64


def test_temp_same_inode_mutation_is_rejected_before_replace(tmp_path):
    data_root = tmp_path / "data"
    original = EvidenceArtifactStore(data_root)
    _publish_part(original, b"original")
    store = EvidenceArtifactStore(data_root, filesystem_ops=_MutateTempBeforeReplace())

    with pytest.raises(ArtifactBoundaryError) as caught:
        store.publish_part(
            WORKSPACE_ID,
            PART_ID,
            b"replacement",
            expected_sha256=_sha256(b"replacement"),
        )

    assert caught.value.code == "artifact_hash_mismatch"
    part = data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts" / PART_ID
    assert part.read_bytes() == b"original"


def test_post_replace_hash_mismatch_removes_the_owned_target(tmp_path):
    data_root = tmp_path / "data"
    store = EvidenceArtifactStore(data_root, filesystem_ops=_WrongPostDigest())

    with pytest.raises(ArtifactBoundaryError) as caught:
        _publish_part(store, b"content")

    assert caught.value.code == "artifact_hash_mismatch"
    target = data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts" / PART_ID
    assert not target.exists()


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
    data_root = tmp_path / "data"
    store = EvidenceArtifactStore(data_root, filesystem_ops=_AlwaysSharingViolation())
    _publish_part(store)

    assert store.delete_workspace(WORKSPACE_ID) is ArtifactCleanupState.CLEANUP_PENDING
    assert (
        data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts" / PART_ID
    ).read_bytes() == b"course evidence"

    restarted = EvidenceArtifactStore(data_root, filesystem_ops=NativeArtifactFilesystemOps())
    assert restarted.delete_workspace(WORKSPACE_ID) is ArtifactCleanupState.DELETED
    assert not (data_root / "workspaces" / WORKSPACE_ID / "evidence").exists()


def test_restart_reanchors_and_reads_an_existing_artifact(tmp_path):
    data_root = tmp_path / "data"
    first = EvidenceArtifactStore(data_root)
    expected = _publish_part(first)

    restarted = EvidenceArtifactStore(data_root)
    with restarted.open_part(WORKSPACE_ID, PART_ID) as source:
        assert hashlib.sha256(source.read()).hexdigest() == expected


def test_restart_refuses_a_copied_evidence_tree_with_a_copied_marker(tmp_path):
    data_root = tmp_path / "data"
    first = EvidenceArtifactStore(data_root)
    _publish_part(first)
    workspace = data_root / "workspaces" / WORKSPACE_ID
    evidence = workspace / "evidence"
    original = workspace / "original-evidence"
    copied = workspace / "copied-evidence"
    shutil.copytree(evidence, copied)
    evidence.rename(original)
    copied.rename(evidence)

    restarted = EvidenceArtifactStore(data_root)
    with pytest.raises(ArtifactBoundaryError) as caught:
        with restarted.open_part(WORKSPACE_ID, PART_ID):
            pytest.fail("a copied ownership marker must not re-claim a tree")

    assert caught.value.code == "artifact_identity_changed"
    assert (original / "parts" / PART_ID).read_bytes() == b"course evidence"


def test_restart_refuses_a_single_link_file_replacement(tmp_path):
    data_root = tmp_path / "data"
    first = EvidenceArtifactStore(data_root)
    _publish_part(first)
    part = data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts" / PART_ID
    original = part.with_name("original-part")
    part.rename(original)
    part.write_bytes(original.read_bytes())
    assert part.stat().st_nlink == 1

    restarted = EvidenceArtifactStore(data_root)
    with pytest.raises(ArtifactBoundaryError) as caught:
        with restarted.open_part(WORKSPACE_ID, PART_ID):
            pytest.fail("a replacement inode must not be adopted after restart")

    assert caught.value.code == "artifact_identity_changed"
    assert original.read_bytes() == b"course evidence"


def test_constructor_rejects_a_link_in_the_root_ancestor_chain(tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    _directory_link_or_skip(linked_parent, real_parent)

    with pytest.raises(ArtifactBoundaryError) as caught:
        EvidenceArtifactStore(linked_parent / "data")

    assert caught.value.code == "artifact_identity_changed"
    assert not (real_parent / "data").exists()


def test_delete_does_not_report_deleted_when_lstat_is_denied(tmp_path, monkeypatch):
    import exam_predictor.evidence.artifacts as artifacts_module

    data_root = tmp_path / "data"
    store = EvidenceArtifactStore(data_root)
    _publish_part(store)
    evidence = data_root / "workspaces" / WORKSPACE_ID / "evidence"
    original_lstat = artifacts_module.os.lstat

    def deny_evidence(path, *args, **kwargs):
        if Path(path) == evidence:
            raise PermissionError("denied")
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(artifacts_module.os, "lstat", deny_evidence)

    assert store.delete_workspace(WORKSPACE_ID) is ArtifactCleanupState.CLEANUP_PENDING
    assert (evidence / "parts" / PART_ID).read_bytes() == b"course evidence"


@pytest.mark.skipif(os.name != "nt", reason="Windows no-delete-share handles")
def test_windows_open_handle_blocks_directory_retargeting(tmp_path):
    store = EvidenceArtifactStore(tmp_path / "data")
    _publish_part(store)
    evidence = tmp_path / "data" / "workspaces" / WORKSPACE_ID / "evidence"

    with store.open_part(WORKSPACE_ID, PART_ID):
        with pytest.raises(OSError):
            evidence.rename(evidence.with_name("retargeted-evidence"))


class _RecordingDurabilityOps(NativeArtifactFilesystemOps):
    def __init__(self) -> None:
        self.events: list[str] = []

    def durability_event(self, event: str) -> None:
        self.events.append(event)


@pytest.mark.skipif(os.name != "nt", reason="Windows write-through durability")
def test_windows_publish_orders_write_through_and_flushes(tmp_path):
    operations = _RecordingDurabilityOps()
    store = EvidenceArtifactStore(tmp_path / "data", filesystem_ops=operations)

    _publish_part(store)

    assert operations.events
    for index, event in enumerate(operations.events):
        if event != "rename_write_through":
            continue
        assert operations.events[index + 1] == "final_file_flushed"
        assert operations.events[index + 2] in {
            "parent_directory_flushed",
            "parent_flush_unavailable",
        }


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
