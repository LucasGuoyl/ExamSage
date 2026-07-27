from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
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
from exam_predictor.evidence.models import (
    EvidenceCitation,
    EvidenceUnit,
    SnapshotStatus,
    StudyMapSnapshot,
)
from exam_predictor.workspace.filesystem import OwnedFilesystemError


_RawEvidenceArtifactStore = EvidenceArtifactStore


def EvidenceArtifactStore(root, *args, **kwargs):
    Path(root).mkdir(parents=True, exist_ok=True)
    return _RawEvidenceArtifactStore(root, *args, **kwargs)


WORKSPACE_ID = "workspace_0123456789"
PART_ID = "part_0123456789ab"
UNIT_ID = "unit_0123456789ab"
SNAPSHOT_ID = "snapshot_01234567"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _unit() -> EvidenceUnit:
    return EvidenceUnit(
        evidence_unit_id=UNIT_ID,
        source_part_id=PART_ID,
        content="bounded evidence",
        citations=(
            EvidenceCitation(
                citation_id="citation_0123456",
                evidence_unit_id=UNIT_ID,
                source_part_id=PART_ID,
                relative_path="notes/week-1.txt",
                locator="page 1",
            ),
        ),
    )


def _snapshot() -> StudyMapSnapshot:
    return StudyMapSnapshot(
        snapshot_id=SNAPSHOT_ID,
        workspace_id=WORKSPACE_ID,
        revision_id="revision_0123456",
        status=SnapshotStatus.COMPLETE,
        nodes=(),
        coverage=None,
        evidence_unit_ids=(UNIT_ID,),
        created_at=datetime(2026, 7, 27, tzinfo=UTC),
    )


def _canonical_json(document) -> bytes:
    return json.dumps(
        document.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


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


@pytest.mark.parametrize(
    "artifact_type,artifact_id,document",
    [("units", UNIT_ID, _unit()), ("snapshots", SNAPSHOT_ID, _snapshot())],
)
def test_publish_json_uses_only_the_fixed_owned_layout(tmp_path, artifact_type, artifact_id, document):
    store = EvidenceArtifactStore(tmp_path / "data")
    encoded = _canonical_json(document)

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

    document = _unit()
    encoded = _canonical_json(document)
    store.publish_json(
        WORKSPACE_ID,
        "units",
        UNIT_ID,
        document,
        expected_sha256=_sha256(encoded),
    )
    original_loads = json.loads

    def malformed_artifact_loads(value, *args, **kwargs):
        if value == encoded.decode("utf-8"):
            raise json.JSONDecodeError("private input", value, 0)
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr("exam_predictor.evidence.artifacts.json.loads", malformed_artifact_loads)

    with pytest.raises(ArtifactBoundaryError) as caught:
        store.read_json(WORKSPACE_ID, "units", UNIT_ID)
    assert caught.value.code == "artifact_json_invalid"


def test_json_schema_and_unicode_failures_are_normalized(tmp_path):
    store = EvidenceArtifactStore(tmp_path / "data")
    for document in (
        _snapshot().model_dump(mode="json"),
        {**_unit().model_dump(mode="json"), "content": "\ud800"},
    ):
        with pytest.raises(ArtifactBoundaryError) as caught:
            store.publish_json(
                WORKSPACE_ID,
                "units",
                UNIT_ID,
                document,
                expected_sha256=_sha256(b"irrelevant"),
            )
        assert caught.value.code == "artifact_json_invalid"
        assert str(caught.value) == "artifact_json_invalid"


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
        super().__init__()
        self.part = part
        self.external = external
        self.swapped = False

    def before_mutation(self, operation, parent, source_name) -> None:
        del operation, parent
        if self.swapped or source_name != PART_ID:
            return
        self.swapped = True
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

    with pytest.raises(ArtifactBoundaryError) as caught:
        store.delete_workspace(WORKSPACE_ID)
    assert caught.value.code == "artifact_identity_changed"
    assert part.read_bytes() == b"native bytes"
    assert external.read_bytes() == b"native bytes"


class _RegularSwapDuringRemove(NativeArtifactFilesystemOps):
    def __init__(self, part: Path) -> None:
        super().__init__()
        self.part = part
        self.swapped = False

    def before_mutation(self, operation, parent, source_name) -> None:
        del operation, parent
        if self.swapped or source_name != PART_ID:
            return
        self.swapped = True
        self.part.unlink()
        self.part.write_bytes(b"not an artifact")


def test_delete_never_removes_regular_substitution_after_judgement(tmp_path):
    data_root = tmp_path / "data"
    initial = EvidenceArtifactStore(data_root)
    _publish_part(initial)
    part = data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts" / PART_ID
    store = EvidenceArtifactStore(
        data_root,
        filesystem_ops=_RegularSwapDuringRemove(part),
    )

    with pytest.raises(ArtifactBoundaryError) as caught:
        store.delete_workspace(WORKSPACE_ID)

    assert caught.value.code == "artifact_identity_changed"
    assert part.read_bytes() == b"not an artifact"


class _AlwaysSharingViolation(NativeArtifactFilesystemOps):
    def delete_claimed_file(self, *args, **kwargs) -> None:
        del args, kwargs
        error = PermissionError("sharing violation")
        error.winerror = 32
        raise error


class _RetargetBeforeReplace(NativeArtifactFilesystemOps):
    def __init__(self, parts: Path, outside: Path) -> None:
        super().__init__()
        self.parts = parts
        self.outside = outside
        self.retargeted = False

    def before_mutation(self, operation, parent, source_name) -> None:
        del operation, parent
        if self.retargeted or not source_name.startswith(".artifact-"):
            return
        self.retargeted = True
        original = self.parts.with_name("original-parts")
        try:
            self.parts.rename(original)
        except OSError:
            raise OwnedFilesystemError("owned_identity_changed") from None
        _directory_link_or_skip(self.parts, self.outside)


class _MutateTempBeforeReplace(NativeArtifactFilesystemOps):
    def __init__(self) -> None:
        super().__init__()
        self.mutated = False

    def replace_open_file(self, parent, source, source_name, destination_name, **kwargs):
        if not self.mutated and source_name.startswith(".artifact-"):
            self.mutated = True
            os.lseek(source.descriptor, 0, os.SEEK_SET)
            os.ftruncate(source.descriptor, 0)
            os.write(source.descriptor, b"tampered-in-place")
            os.fsync(source.descriptor)
        return super().replace_open_file(parent, source, source_name, destination_name, **kwargs)


class _CorruptTargetAfterReplace(NativeArtifactFilesystemOps):
    def __init__(self) -> None:
        super().__init__()
        self.corrupted = False

    def replace_open_file(self, parent, source, source_name, destination_name, **kwargs):
        result = super().replace_open_file(parent, source, source_name, destination_name, **kwargs)
        if not self.corrupted and destination_name == PART_ID:
            self.corrupted = True
            os.lseek(source.descriptor, 0, os.SEEK_SET)
            os.ftruncate(source.descriptor, 0)
            os.write(source.descriptor, b"corrupt")
            os.fsync(source.descriptor)
        return result


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
    store = EvidenceArtifactStore(data_root, filesystem_ops=_CorruptTargetAfterReplace())

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


def test_marker_persists_complete_authoritative_claim(tmp_path):
    data_root = tmp_path / "data"
    store = EvidenceArtifactStore(data_root)
    digest = _publish_part(store)
    part = data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts" / PART_ID
    part_stat = part.stat(follow_symlinks=False)
    marker = json.loads((part.parents[1] / ".artifact-ownership.json").read_text(encoding="utf-8"))

    assert marker["artifacts"][f"parts/{PART_ID}"] == {
        "slot": f"parts/{PART_ID}",
        "kind": "part",
        "id": PART_ID,
        "device_id": str(part_stat.st_dev),
        "file_id": str(part_stat.st_ino),
        "sha256": digest,
        "size": len(b"course evidence"),
    }
    assert marker["pending"] is None


def test_unknown_file_in_fixed_collection_is_never_deleted(tmp_path):
    data_root = tmp_path / "data"
    store = EvidenceArtifactStore(data_root)
    _publish_part(store)
    parts = data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts"
    unknown = parts / "not_an_artifact_01"
    unknown.write_bytes(b"native bytes")

    assert store.delete_workspace(WORKSPACE_ID) is ArtifactCleanupState.CLEANUP_PENDING
    assert unknown.read_bytes() == b"native bytes"
    assert (parts / PART_ID).read_bytes() == b"course evidence"


class _RetargetEvidenceBeforeMarker(NativeArtifactFilesystemOps):
    def __init__(self, outside: Path) -> None:
        super().__init__()
        self.outside = outside
        self.retargeted = False

    def before_mutation(self, operation, parent, source_name) -> None:
        del operation
        if self.retargeted or not source_name.startswith(".ownership-"):
            return
        self.retargeted = True
        original = parent.path.with_name("original-evidence")
        try:
            parent.path.rename(original)
        except OSError:
            raise OwnedFilesystemError("owned_identity_changed") from None
        _directory_link_or_skip(parent.path, self.outside)


def test_bootstrap_retarget_never_writes_marker_to_replacement_tree(tmp_path):
    data_root = tmp_path / "data"
    outside = tmp_path / "outside"
    outside.mkdir()
    store = EvidenceArtifactStore(
        data_root,
        filesystem_ops=_RetargetEvidenceBeforeMarker(outside),
    )

    with pytest.raises(ArtifactBoundaryError):
        _publish_part(store)

    assert not (outside / ".artifact-ownership.json").exists()
    assert not list(outside.iterdir())


class _SimulatedCrash(BaseException):
    pass


class _JournalFaultFilesystem(NativeArtifactFilesystemOps):
    def __init__(self) -> None:
        super().__init__()
        self.phase: str | None = None
        self.marker_count = 0

    def arm(self, phase: str) -> None:
        self.phase = phase
        self.marker_count = 0

    def replace_open_file(self, parent, source, source_name, destination_name, **kwargs):
        if self.phase and destination_name == ".artifact-ownership.json":
            self.marker_count += 1
            if self.phase == "fail_commit" and self.marker_count == 2:
                raise OwnedFilesystemError("owned_operation_failed")
        result = super().replace_open_file(parent, source, source_name, destination_name, **kwargs)
        if (
            self.phase == "pending"
            and self.marker_count == 1
            and destination_name == ".artifact-ownership.json"
        ):
            raise _SimulatedCrash()
        if self.phase == "backup" and destination_name.endswith(".backup"):
            raise _SimulatedCrash()
        if self.phase == "target" and destination_name == PART_ID:
            raise _SimulatedCrash()
        if (
            self.phase == "committed_marker"
            and self.marker_count == 2
            and destination_name == ".artifact-ownership.json"
        ):
            raise _SimulatedCrash()
        if (
            self.phase == "final_clear"
            and self.marker_count == 3
            and destination_name == ".artifact-ownership.json"
        ):
            raise _SimulatedCrash()
        return result

    def delete_claimed_file(self, parent, source_name, **kwargs) -> None:
        super().delete_claimed_file(parent, source_name, **kwargs)
        if self.phase == "backup_delete" and source_name.endswith(".backup"):
            raise _SimulatedCrash()


@pytest.mark.parametrize(
    "phase,expected",
    [
        ("pending", b"old"),
        ("backup", b"old"),
        ("target", b"old"),
        ("committed_marker", b"new"),
        ("backup_delete", b"new"),
        ("final_clear", b"new"),
    ],
)
def test_restart_recovers_every_persisted_publish_phase(tmp_path, phase, expected):
    data_root = tmp_path / "data"
    operations = _JournalFaultFilesystem()
    store = EvidenceArtifactStore(data_root, filesystem_ops=operations)
    _publish_part(store, b"old")
    operations.arm(phase)

    with pytest.raises(_SimulatedCrash):
        _publish_part(store, b"new")
    store.close()

    restarted = EvidenceArtifactStore(data_root)
    with restarted.open_part(WORKSPACE_ID, PART_ID) as source:
        assert source.read() == expected
    marker = json.loads(
        (data_root / "workspaces" / WORKSPACE_ID / "evidence" / ".artifact-ownership.json").read_text(
            encoding="utf-8"
        )
    )
    assert marker["pending"] is None


def test_marker_commit_failure_rolls_back_existing_target(tmp_path):
    data_root = tmp_path / "data"
    operations = _JournalFaultFilesystem()
    store = EvidenceArtifactStore(data_root, filesystem_ops=operations)
    _publish_part(store, b"old")
    operations.arm("fail_commit")

    with pytest.raises(ArtifactBoundaryError):
        _publish_part(store, b"new")
    store.close()

    restarted = EvidenceArtifactStore(data_root)
    with restarted.open_part(WORKSPACE_ID, PART_ID) as source:
        assert source.read() == b"old"


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


def test_restart_delete_refuses_same_bytes_replacement_inode(tmp_path):
    data_root = tmp_path / "data"
    first = EvidenceArtifactStore(data_root)
    _publish_part(first)
    part = data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts" / PART_ID
    original = part.with_name("original-claimed-part")
    part.rename(original)
    part.write_bytes(original.read_bytes())

    restarted = EvidenceArtifactStore(data_root)
    with pytest.raises(ArtifactBoundaryError) as caught:
        restarted.delete_workspace(WORKSPACE_ID)

    assert caught.value.code == "artifact_identity_changed"
    assert part.read_bytes() == b"course evidence"
    assert original.read_bytes() == b"course evidence"


def test_constructor_rejects_missing_or_linked_trusted_root(tmp_path):
    missing = tmp_path / "missing-root"
    with pytest.raises(ArtifactBoundaryError) as caught:
        _RawEvidenceArtifactStore(missing)
    assert caught.value.code == "artifact_root_missing"
    assert not missing.exists()

    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    _directory_link_or_skip(linked_root, real_root)

    with pytest.raises(ArtifactBoundaryError) as caught:
        _RawEvidenceArtifactStore(linked_root)

    assert caught.value.code == "artifact_root_invalid"
    assert not (real_root / "workspaces").exists()


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
