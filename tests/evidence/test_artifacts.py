from __future__ import annotations

from collections import UserDict
from contextlib import contextmanager
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
from unittest.mock import patch

import pytest

import exam_predictor.evidence.artifacts as artifacts_module
from exam_predictor.evidence.artifacts import (
    ArtifactBoundaryError,
    ArtifactCleanupState,
    EvidenceArtifactStore,
)
from exam_predictor.evidence.models import (
    EvidenceCitation,
    EvidenceUnit,
    SnapshotStatus,
    StudyMapSnapshot,
)
from exam_predictor.workspace.filesystem import OwnedArtifactFilesystem, OwnedFilesystemError


_RawEvidenceArtifactStore = EvidenceArtifactStore
NativeArtifactFilesystemOps = OwnedArtifactFilesystem


def EvidenceArtifactStore(root, *args, **kwargs):
    Path(root).mkdir(parents=True, exist_ok=True)
    return _RawEvidenceArtifactStore(root, *args, **kwargs)


def _store_with_filesystem(root, filesystem):
    Path(root).mkdir(parents=True, exist_ok=True)
    with patch.object(artifacts_module, "OwnedArtifactFilesystem", return_value=filesystem):
        return _RawEvidenceArtifactStore(root)


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


def test_public_constructor_cannot_inject_filesystem_operations(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with pytest.raises(TypeError):
        _RawEvidenceArtifactStore(root, filesystem_ops=NativeArtifactFilesystemOps())


def test_bootstrap_refuses_an_existing_unregistered_evidence_directory(tmp_path):
    data_root = tmp_path / "data"
    evidence = data_root / "workspaces" / WORKSPACE_ID / "evidence"
    evidence.mkdir(parents=True)
    store = _RawEvidenceArtifactStore(data_root)

    with pytest.raises(ArtifactBoundaryError) as caught:
        _publish_part(store)

    assert caught.value.code == "artifact_identity_changed"
    assert list(evidence.iterdir()) == []


def test_external_registry_rejects_a_forged_complete_evidence_replacement(tmp_path):
    data_root = tmp_path / "data"
    first = EvidenceArtifactStore(data_root)
    _publish_part(first)
    first.close()
    workspace = data_root / "workspaces" / WORKSPACE_ID
    evidence = workspace / "evidence"
    original = workspace / "original-evidence"
    evidence.rename(original)
    shutil.copytree(original, evidence)

    marker_path = evidence / ".artifact-ownership.json"
    part = evidence / "parts" / PART_ID
    part_stat = part.stat(follow_symlinks=False)
    marker_path.write_text("{}", encoding="utf-8")
    marker_stat = marker_path.stat(follow_symlinks=False)
    marker = {
        "version": 2,
        "root": list((data_root.stat().st_dev, data_root.stat().st_ino)),
        "workspace": list((workspace.stat().st_dev, workspace.stat().st_ino)),
        "evidence": list((evidence.stat().st_dev, evidence.stat().st_ino)),
        "marker": list((marker_stat.st_dev, marker_stat.st_ino)),
        "artifacts": {
            f"parts/{PART_ID}": {
                "slot": f"parts/{PART_ID}",
                "kind": "part",
                "id": PART_ID,
                "device_id": str(part_stat.st_dev),
                "file_id": str(part_stat.st_ino),
                "sha256": _sha256(b"course evidence"),
                "size": len(b"course evidence"),
            }
        },
        "pending": None,
    }
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    restarted = EvidenceArtifactStore(data_root)
    with pytest.raises(ArtifactBoundaryError) as caught:
        with restarted.open_part(WORKSPACE_ID, PART_ID):
            pytest.fail("a forged in-tree marker cannot replace the external authority")

    assert caught.value.code == "artifact_identity_changed"
    assert (original / "parts" / PART_ID).read_bytes() == b"course evidence"


def test_registry_is_external_and_internal_marker_is_absent(tmp_path):
    data_root = tmp_path / "data"
    store = EvidenceArtifactStore(data_root)
    digest = _publish_part(store)
    evidence = data_root / "workspaces" / WORKSPACE_ID / "evidence"
    registry_path = data_root / ".evidence-artifact-registry.sqlite3"

    assert registry_path.is_file()
    assert not (evidence / ".artifact-ownership.json").exists()
    with sqlite3.connect(registry_path) as observer:
        row = observer.execute(
            """
            SELECT device_id, file_id, sha256, size
            FROM artifacts WHERE workspace_id = ? AND slot = ?
            """,
            (WORKSPACE_ID, f"parts/{PART_ID}"),
        ).fetchone()
    part_stat = (evidence / "parts" / PART_ID).stat(follow_symlinks=False)
    assert row == (
        str(part_stat.st_dev),
        str(part_stat.st_ino),
        digest,
        len(b"course evidence"),
    )


def test_publish_part_atomically_reopens_with_the_expected_sha256(tmp_path):
    store = EvidenceArtifactStore(tmp_path / "data")
    content = b"\x00known multimodal evidence\xff"
    expected = _publish_part(store, content)

    with store.open_part(WORKSPACE_ID, PART_ID) as source:
        assert hashlib.sha256(source.read()).hexdigest() == expected

    part = tmp_path / "data" / "workspaces" / WORKSPACE_ID / "evidence" / "parts" / PART_ID
    assert part.read_bytes() == content
    assert not list(part.parent.glob(".examsage-artifact-*.tmp"))


def test_open_part_exposes_only_a_read_only_os_descriptor(tmp_path):
    data_root = tmp_path / "data"
    store = EvidenceArtifactStore(data_root)
    _publish_part(store, b"read only")

    with store.open_part(WORKSPACE_ID, PART_ID) as source:
        with pytest.raises(OSError):
            os.write(source.fileno(), b"mutate")

    assert (
        data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts" / PART_ID
    ).read_bytes() == b"read only"


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


def test_json_rejects_dict_subclasses_and_wrong_pydantic_model_types(tmp_path):
    store = EvidenceArtifactStore(tmp_path / "data")

    class DerivedDict(dict):
        pass

    for non_exact_mapping in (
        DerivedDict(_unit().model_dump(mode="json")),
        UserDict(_unit().model_dump(mode="json")),
    ):
        with pytest.raises(ArtifactBoundaryError) as caught:
            store.publish_json(
                WORKSPACE_ID,
                "units",
                UNIT_ID,
                non_exact_mapping,
                expected_sha256=_sha256(_canonical_json(_unit())),
            )
        assert caught.value.code == "artifact_json_invalid"

    with pytest.raises(ArtifactBoundaryError) as caught:
        store.publish_json(
            WORKSPACE_ID,
            "units",
            UNIT_ID,
            _snapshot(),
            expected_sha256=_sha256(b"irrelevant"),
        )
    assert caught.value.code == "artifact_json_invalid"


def test_json_runtime_failures_are_mapped_to_the_safe_boundary_code(tmp_path, monkeypatch):
    store = EvidenceArtifactStore(tmp_path / "data")

    def fail_validation(*args, **kwargs):
        raise RuntimeError("validator internals must not cross the boundary")

    monkeypatch.setattr(EvidenceUnit, "model_validate", fail_validation)
    with pytest.raises(ArtifactBoundaryError) as caught:
        store.publish_json(
            WORKSPACE_ID,
            "units",
            UNIT_ID,
            _unit().model_dump(mode="json"),
            expected_sha256=_sha256(b"irrelevant"),
        )

    assert caught.value.code == "artifact_json_invalid"


def test_json_iterencode_stops_as_soon_as_utf8_limit_is_exceeded(tmp_path, monkeypatch):
    store = EvidenceArtifactStore(tmp_path / "data")
    consumed: list[str] = []

    def oversized_chunks(self, document, _one_shot=False):
        del self, document, _one_shot
        consumed.append("first")
        yield "123456789"
        consumed.append("second")
        raise RuntimeError("the encoder consumed beyond the established limit")

    monkeypatch.setattr(artifacts_module, "_MAX_JSON_BYTES", 8)
    monkeypatch.setattr(json.JSONEncoder, "iterencode", oversized_chunks)
    with pytest.raises(ArtifactBoundaryError) as caught:
        store.publish_json(
            WORKSPACE_ID,
            "units",
            UNIT_ID,
            _unit(),
            expected_sha256=_sha256(b"irrelevant"),
        )

    assert caught.value.code == "artifact_json_too_large"
    assert consumed == ["first"]


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
    store = _store_with_filesystem(data_root, _HardlinkSwapDuringRemove(part, external))

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
    store = _store_with_filesystem(data_root, _RegularSwapDuringRemove(part))

    with pytest.raises(ArtifactBoundaryError) as caught:
        store.delete_workspace(WORKSPACE_ID)

    assert caught.value.code == "artifact_identity_changed"
    assert part.read_bytes() == b"not an artifact"


class _AddUnknownDuringRemove(NativeArtifactFilesystemOps):
    def __init__(self) -> None:
        super().__init__()
        self.added = False

    def before_mutation(self, operation, parent, source_name) -> None:
        del operation
        if self.added or source_name != PART_ID:
            return
        self.added = True
        (parent.path / "unknown_native_entry").write_bytes(b"native")


def test_delete_rescans_after_quarantine_and_preserves_claim_when_unknown_appears(tmp_path):
    data_root = tmp_path / "data"
    initial = EvidenceArtifactStore(data_root)
    _publish_part(initial)
    store = _store_with_filesystem(data_root, _AddUnknownDuringRemove())

    assert store.delete_workspace(WORKSPACE_ID) is ArtifactCleanupState.CLEANUP_PENDING
    parts = data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts"
    assert (parts / "unknown_native_entry").read_bytes() == b"native"
    quarantines = list(parts.glob(".artifact-delete-*.quarantine"))
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes() == b"course evidence"


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


class _SwapTargetAfterReplace(NativeArtifactFilesystemOps):
    def __init__(self) -> None:
        super().__init__()
        self.swapped = False

    def replace_open_file(self, parent, source, source_name, destination_name, **kwargs):
        result = super().replace_open_file(parent, source, source_name, destination_name, **kwargs)
        if not self.swapped and destination_name == PART_ID:
            self.swapped = True
            target = parent.path / destination_name
            target.rename(parent.path / ".attacker-stole-installed-target")
            target.write_bytes(b"attacker target")
        return result


def test_temp_same_inode_mutation_is_rejected_before_replace(tmp_path):
    data_root = tmp_path / "data"
    original = EvidenceArtifactStore(data_root)
    _publish_part(original, b"original")
    store = _store_with_filesystem(data_root, _MutateTempBeforeReplace())

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


def test_post_replace_hash_mismatch_keeps_registry_pending_and_old_backup(tmp_path):
    data_root = tmp_path / "data"
    original = EvidenceArtifactStore(data_root)
    _publish_part(original, b"original")
    store = _store_with_filesystem(data_root, _CorruptTargetAfterReplace())

    with pytest.raises(ArtifactBoundaryError) as caught:
        _publish_part(store, b"content")

    assert caught.value.code == "artifact_hash_mismatch"
    target = data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts" / PART_ID
    assert target.read_bytes() == b"corrupt"
    backups = list(target.parent.glob(".artifact-*.backup"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"original"
    with sqlite3.connect(data_root / ".evidence-artifact-registry.sqlite3") as observer:
        assert observer.execute(
            "SELECT phase FROM publish_journal WHERE workspace_id = ?",
            (WORKSPACE_ID,),
        ).fetchone() == ("backup",)
    with pytest.raises(ArtifactBoundaryError):
        with store.open_part(WORKSPACE_ID, PART_ID):
            pytest.fail("a pending corrupted target must not be opened")


def test_wrong_installed_target_keeps_old_backup_and_blocks_open(tmp_path):
    data_root = tmp_path / "data"
    original = EvidenceArtifactStore(data_root)
    _publish_part(original, b"original")
    store = _store_with_filesystem(data_root, _SwapTargetAfterReplace())

    with pytest.raises(ArtifactBoundaryError) as caught:
        _publish_part(store, b"replacement")

    assert caught.value.code == "artifact_identity_changed"
    parts = data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts"
    assert (parts / PART_ID).read_bytes() == b"attacker target"
    backups = list(parts.glob(".artifact-*.backup"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"original"
    with sqlite3.connect(data_root / ".evidence-artifact-registry.sqlite3") as observer:
        assert observer.execute(
            "SELECT phase FROM publish_journal WHERE workspace_id = ?",
            (WORKSPACE_ID,),
        ).fetchone() == ("backup",)
    with pytest.raises(ArtifactBoundaryError):
        with store.open_part(WORKSPACE_ID, PART_ID):
            pytest.fail("an unclaimed installed target must never be opened")


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
    store = _store_with_filesystem(data_root, _RetargetBeforeReplace(parts, outside))

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
    store = _store_with_filesystem(data_root, _AlwaysSharingViolation())
    _publish_part(store)

    assert store.delete_workspace(WORKSPACE_ID) is ArtifactCleanupState.CLEANUP_PENDING
    parts = data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts"
    quarantines = list(parts.glob(".artifact-delete-*.quarantine"))
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes() == b"course evidence"

    restarted = _store_with_filesystem(data_root, NativeArtifactFilesystemOps())
    assert restarted.delete_workspace(WORKSPACE_ID) is ArtifactCleanupState.DELETED
    assert not (data_root / "workspaces" / WORKSPACE_ID / "evidence").exists()


def test_restart_reanchors_and_reads_an_existing_artifact(tmp_path):
    data_root = tmp_path / "data"
    first = EvidenceArtifactStore(data_root)
    expected = _publish_part(first)

    restarted = EvidenceArtifactStore(data_root)
    with restarted.open_part(WORKSPACE_ID, PART_ID) as source:
        assert hashlib.sha256(source.read()).hexdigest() == expected


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


class _RetargetWorkspaceBeforeEvidence(NativeArtifactFilesystemOps):
    def __init__(self, outside: Path) -> None:
        super().__init__()
        self.outside = outside
        self.retargeted = False

    @contextmanager
    def create_new_child_directory(self, parent, name, **kwargs):
        if not self.retargeted and name == "evidence":
            self.retargeted = True
            original = parent.path.with_name("original-workspace")
            try:
                parent.path.rename(original)
            except OSError:
                raise OwnedFilesystemError("owned_identity_changed") from None
            _directory_link_or_skip(parent.path, self.outside)
        with super().create_new_child_directory(parent, name, **kwargs) as child:
            yield child


def test_bootstrap_retarget_never_writes_into_a_replacement_workspace(tmp_path):
    data_root = tmp_path / "data"
    outside = tmp_path / "outside"
    outside.mkdir()
    store = _store_with_filesystem(data_root, _RetargetWorkspaceBeforeEvidence(outside))

    with pytest.raises(ArtifactBoundaryError):
        _publish_part(store)

    assert not list(outside.iterdir())


class _SimulatedCrash(BaseException):
    pass


def _crash_after_registry_phase(store, phase):
    registry = store._registry
    assert registry is not None
    if phase == "prepared":
        method_name = "prepare_publish"
    elif phase in {"backup", "installed"}:
        method_name = "advance_publish"
    elif phase == "committed":
        method_name = "commit_publish"
    elif phase == "cleared":
        method_name = "clear_publish"
    else:
        raise AssertionError(phase)
    original = getattr(registry, method_name)

    def crashing(*args, **kwargs):
        result = original(*args, **kwargs)
        if method_name != "advance_publish" or kwargs.get("new_phase") == phase:
            raise _SimulatedCrash()
        return result

    setattr(registry, method_name, crashing)


def _crash_after_backup_delete(store):
    original = store._filesystem.delete_claimed_file

    def crashing(parent, source_name, **kwargs):
        result = original(parent, source_name, **kwargs)
        if source_name.endswith(".backup"):
            raise _SimulatedCrash()
        return result

    store._filesystem.delete_claimed_file = crashing


@pytest.mark.parametrize(
    "phase,expected",
    [
        ("prepared", b"old"),
        ("backup", b"old"),
        ("installed", b"new"),
        ("committed", b"new"),
        ("backup_delete", b"new"),
        ("cleared", b"new"),
    ],
)
def test_restart_recovers_every_registry_publish_phase(tmp_path, phase, expected):
    data_root = tmp_path / "data"
    store = EvidenceArtifactStore(data_root)
    _publish_part(store, b"old")
    if phase == "backup_delete":
        _crash_after_backup_delete(store)
    else:
        _crash_after_registry_phase(store, phase)

    with pytest.raises(_SimulatedCrash):
        _publish_part(store, b"new")
    store.close()

    restarted = EvidenceArtifactStore(data_root)
    with restarted.open_part(WORKSPACE_ID, PART_ID) as source:
        assert source.read() == expected
    with sqlite3.connect(data_root / ".evidence-artifact-registry.sqlite3") as observer:
        assert (
            observer.execute(
                "SELECT 1 FROM publish_journal WHERE workspace_id = ?",
                (WORKSPACE_ID,),
            ).fetchone()
            is None
        )


def test_registry_commit_failure_keeps_pending_and_restart_completes_install(tmp_path):
    data_root = tmp_path / "data"
    store = EvidenceArtifactStore(data_root)
    _publish_part(store, b"old")
    registry = store._registry
    assert registry is not None

    def fail_commit(*args, **kwargs):
        raise RuntimeError("registry commit unavailable")

    registry.commit_publish = fail_commit

    with pytest.raises(ArtifactBoundaryError):
        _publish_part(store, b"new")
    with pytest.raises(ArtifactBoundaryError):
        with store.open_part(WORKSPACE_ID, PART_ID):
            pytest.fail("pending registry commit must block reads")
    store.close()

    restarted = EvidenceArtifactStore(data_root)
    with restarted.open_part(WORKSPACE_ID, PART_ID) as source:
        assert source.read() == b"new"


def test_restart_refuses_a_copied_evidence_tree(tmp_path):
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
            pytest.fail("a copied tree cannot replace the registry-claimed evidence identity")

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


def test_delete_absent_registry_workspace_is_idempotent_and_touches_nothing(tmp_path):
    data_root = tmp_path / "data"
    store = EvidenceArtifactStore(data_root)
    unregistered = data_root / "workspaces" / WORKSPACE_ID / "evidence"
    unregistered.mkdir(parents=True)
    native = unregistered / "native.bin"
    native.write_bytes(b"native")

    assert store.delete_workspace(WORKSPACE_ID) is ArtifactCleanupState.DELETED
    assert store.delete_workspace(WORKSPACE_ID) is ArtifactCleanupState.DELETED
    assert native.read_bytes() == b"native"


def _arm_delete_crash(store, phase):
    registry = store._registry
    assert registry is not None
    if phase == "planned":
        original = registry.plan_delete_quarantine

        def crashing(*args, **kwargs):
            original(*args, **kwargs)
            raise _SimulatedCrash()

        registry.plan_delete_quarantine = crashing
        return
    if phase in {"quarantined", "removed"}:
        original = registry.advance_delete_item

        def crashing(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get("new_phase") == phase:
                raise _SimulatedCrash()
            return result

        registry.advance_delete_item = crashing
        return
    if phase == "moved":
        original = store._filesystem.move_claimed_file

        def crashing(parent, source_name, destination_name, **kwargs):
            result = original(parent, source_name, destination_name, **kwargs)
            if destination_name.endswith(".quarantine"):
                raise _SimulatedCrash()
            return result

        store._filesystem.move_claimed_file = crashing
        return
    if phase == "deleted":
        original = store._filesystem.delete_claimed_file

        def crashing(parent, source_name, **kwargs):
            result = original(parent, source_name, **kwargs)
            if source_name.endswith(".quarantine"):
                raise _SimulatedCrash()
            return result

        store._filesystem.delete_claimed_file = crashing
        return
    if phase == "evidence_removed":
        original = store._filesystem.remove_empty_directory

        def crashing(parent, name, **kwargs):
            result = original(parent, name, **kwargs)
            if name == "evidence":
                raise _SimulatedCrash()
            return result

        store._filesystem.remove_empty_directory = crashing
        return
    raise AssertionError(phase)


@pytest.mark.parametrize(
    "phase",
    ["planned", "moved", "quarantined", "deleted", "removed", "evidence_removed"],
)
def test_restart_continues_every_registry_delete_phase(tmp_path, phase):
    data_root = tmp_path / "data"
    store = EvidenceArtifactStore(data_root)
    _publish_part(store)
    _arm_delete_crash(store, phase)

    with pytest.raises(_SimulatedCrash):
        store.delete_workspace(WORKSPACE_ID)
    store.close()

    restarted = EvidenceArtifactStore(data_root)
    assert restarted.delete_workspace(WORKSPACE_ID) is ArtifactCleanupState.DELETED
    assert not (data_root / "workspaces" / WORKSPACE_ID / "evidence").exists()
    with sqlite3.connect(data_root / ".evidence-artifact-registry.sqlite3") as observer:
        assert (
            observer.execute(
                "SELECT 1 FROM workspaces WHERE workspace_id = ?",
                (WORKSPACE_ID,),
            ).fetchone()
            is None
        )


def test_restart_refuses_a_wrong_delete_quarantine_identity(tmp_path):
    data_root = tmp_path / "data"
    store = EvidenceArtifactStore(data_root)
    _publish_part(store)
    _arm_delete_crash(store, "moved")

    with pytest.raises(_SimulatedCrash):
        store.delete_workspace(WORKSPACE_ID)
    store.close()
    parts = data_root / "workspaces" / WORKSPACE_ID / "evidence" / "parts"
    quarantine = next(parts.glob(".artifact-delete-*.quarantine"))
    original = quarantine.with_name("owned-original-quarantine")
    quarantine.rename(original)
    quarantine.write_bytes(b"attacker replacement")

    restarted = EvidenceArtifactStore(data_root)
    with pytest.raises(ArtifactBoundaryError) as caught:
        restarted.delete_workspace(WORKSPACE_ID)

    assert caught.value.code == "artifact_identity_changed"
    assert quarantine.read_bytes() == b"attacker replacement"
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
