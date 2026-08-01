from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import subprocess
import sys

import pytest

import exam_predictor.evidence.registry as registry_module
from exam_predictor.evidence.registry import (
    ArtifactClaim,
    EvidenceArtifactRegistry,
    PublishIntent,
    PublishJournal,
    RegistryError,
)
from exam_predictor.workspace.filesystem import OwnedArtifactFilesystem


WORKSPACE_ID = "workspace_0123456789"
GENERATION = "a" * 32
REGISTRY_NAME = ".evidence-artifact-registry.log"
LEGACY_REGISTRY_NAME = ".evidence-artifact-registry.sqlite3"
COLLECTION_IDENTITIES = {
    "parts": (21, 31),
    "units": (22, 32),
    "snapshots": (23, 33),
}
OLD_CLAIM = ArtifactClaim(
    workspace_id=WORKSPACE_ID,
    slot="parts/part_0123456789ab",
    collection="parts",
    kind="part",
    artifact_id="part_0123456789ab",
    identity=(41, 51),
    sha256="1" * 64,
    size=11,
    generation=GENERATION,
)
NEW_CLAIM = ArtifactClaim(
    workspace_id=WORKSPACE_ID,
    slot="parts/part_0123456789ab",
    collection="parts",
    kind="part",
    artifact_id="part_0123456789ab",
    identity=(42, 52),
    sha256="2" * 64,
    size=12,
    generation=GENERATION,
)


@contextmanager
def _open_registry(root):
    filesystem = OwnedArtifactFilesystem()
    with filesystem.anchor_directory(root) as root_anchor:
        with EvidenceArtifactRegistry(root_anchor, filesystem) as registry:
            yield registry


def _active_workspace(registry: EvidenceArtifactRegistry) -> None:
    assert registry.reserve_workspace(WORKSPACE_ID, generation=GENERATION) is True
    registry.finalize_workspace(
        WORKSPACE_ID,
        expected_generation=GENERATION,
        workspace_identity=(11, 12),
        evidence_identity=(13, 14),
        collection_identities=COLLECTION_IDENTITIES,
    )


def _publish_journal(old_claim=OLD_CLAIM) -> PublishJournal:
    return PublishJournal(
        workspace_id=WORKSPACE_ID,
        slot=NEW_CLAIM.slot,
        phase="prepared",
        target_name="part_0123456789ab",
        temporary_name=".artifact-new.tmp",
        backup_name=".artifact-old.backup",
        new_claim=NEW_CLAIM,
        old_claim=old_claim,
        generation=GENERATION,
    )


def _publish_intent(old_claim=OLD_CLAIM) -> PublishIntent:
    return PublishIntent(
        workspace_id=WORKSPACE_ID,
        slot=NEW_CLAIM.slot,
        target_name="part_0123456789ab",
        temporary_name=".artifact-new.tmp",
        backup_name=".artifact-old.backup",
        collection=NEW_CLAIM.collection,
        kind=NEW_CLAIM.kind,
        artifact_id=NEW_CLAIM.artifact_id,
        expected_sha256=NEW_CLAIM.sha256,
        expected_size=NEW_CLAIM.size,
        old_claim=old_claim,
        generation=GENERATION,
    )


def _publish_token(temporary_name: str = ".artifact-new.tmp") -> dict[str, str]:
    return {
        "expected_generation": GENERATION,
        "expected_temporary_name": temporary_name,
    }


def _delete_token(generation: str = GENERATION) -> dict[str, str]:
    return {"expected_generation": generation}


def test_registry_uses_fixed_pinned_handle_native_log(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        assert registry.path == root / REGISTRY_NAME
        pinned_identity = registry.identity
        named = registry.path.stat(follow_symlinks=False)
        assert pinned_identity == (named.st_dev, named.st_ino)
        _active_workspace(registry)

    persisted = (root / REGISTRY_NAME).read_bytes()
    assert persisted.startswith(b"EXAMSAGE-EVIDENCE-REGISTRY-LOG")
    assert str(root).encode() not in persisted
    assert not (root / f"{REGISTRY_NAME}-wal").exists()
    assert not (root / f"{REGISTRY_NAME}-shm").exists()


def test_registry_file_is_pinned_or_replacement_is_detected_before_next_read(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        displaced = root / "displaced-registry"
        try:
            registry.path.rename(displaced)
        except PermissionError:
            assert os.name == "nt"
            assert registry.get_workspace(WORKSPACE_ID) is None
            return
        registry.path.write_bytes(b"replacement")

        with pytest.raises(RegistryError) as caught:
            registry.get_workspace(WORKSPACE_ID)
        assert caught.value.code == "registry_identity_changed"
        assert registry.path.read_bytes() == b"replacement"
        assert displaced.read_bytes().startswith(b"EXAMSAGE-EVIDENCE-REGISTRY-LOG")


def test_workspace_bootstrap_is_reserve_then_compare_and_swap_finalize(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        assert registry.reserve_workspace(WORKSPACE_ID) is True
        assert registry.reserve_workspace(WORKSPACE_ID) is False
        reserved = registry.get_workspace(WORKSPACE_ID)
        assert reserved is not None
        assert reserved.phase == "reserved"
        assert reserved.workspace_identity is None
        assert reserved.evidence_identity is None

        active = registry.finalize_workspace(
            WORKSPACE_ID,
            expected_generation=reserved.generation,
            workspace_identity=(11, 12),
            evidence_identity=(13, 14),
            collection_identities=COLLECTION_IDENTITIES,
        )
        assert active.phase == "active"
        assert active.workspace_identity == (11, 12)
        assert active.evidence_identity == (13, 14)
        assert registry.get_collection_identities(WORKSPACE_ID) == COLLECTION_IDENTITIES

        with pytest.raises(RegistryError) as caught:
            registry.finalize_workspace(
                WORKSPACE_ID,
                expected_generation=active.generation,
                workspace_identity=(91, 92),
                evidence_identity=(93, 94),
                collection_identities=COLLECTION_IDENTITIES,
            )
        assert caught.value.code == "registry_state_conflict"
        assert registry.get_workspace(WORKSPACE_ID) == active


def test_stale_workspace_generation_cannot_finalize_a_new_reservation(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    generation_a = "a" * 32
    generation_b = "b" * 32

    with _open_registry(root) as stale:
        assert stale.reserve_workspace(WORKSPACE_ID, generation=generation_a) is True
        with _open_registry(root) as current:
            assert current.retire_reserved_workspace(
                WORKSPACE_ID,
                expected_generation=generation_a,
            ) is True
            assert current.reserve_workspace(WORKSPACE_ID, generation=generation_b) is True

        with pytest.raises(RegistryError) as caught:
            stale.finalize_workspace(
                WORKSPACE_ID,
                expected_generation=generation_a,
                workspace_identity=(11, 12),
                evidence_identity=(13, 14),
                collection_identities=COLLECTION_IDENTITIES,
            )

        assert caught.value.code == "registry_state_conflict"
        reserved = stale.get_workspace(WORKSPACE_ID)
        assert reserved is not None
        assert reserved.phase == "reserved"
        assert reserved.generation == generation_b


def test_crash_during_genesis_does_not_leave_a_permanently_corrupt_registry(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    script = f"""
import os
from pathlib import Path
from exam_predictor.evidence.registry import EvidenceArtifactRegistry
from exam_predictor.workspace.filesystem import OwnedArtifactFilesystem

original = OwnedArtifactFilesystem.replace_open_file
def crash(self, *args, **kwargs):
    os._exit(77)
OwnedArtifactFilesystem.replace_open_file = crash
filesystem = OwnedArtifactFilesystem()
with filesystem.anchor_directory(Path({str(root)!r})) as anchor:
    EvidenceArtifactRegistry(anchor, filesystem)
"""
    completed = subprocess.run([sys.executable, "-c", script], check=False)
    assert completed.returncode == 77

    with _open_registry(root) as restarted:
        assert restarted.get_workspace(WORKSPACE_ID) is None


def test_reserved_workspace_can_be_retired_without_adopting_a_tree(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        assert registry.reserve_workspace(WORKSPACE_ID) is True
        reserved = registry.get_workspace(WORKSPACE_ID)
        assert reserved is not None
        registry.retire_reserved_workspace(
            WORKSPACE_ID,
            expected_generation=reserved.generation,
        )
        assert registry.get_workspace(WORKSPACE_ID) is None


def test_publish_intent_is_durable_before_temporary_bytes_are_written(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        _active_workspace(registry)
        intent = _publish_intent(old_claim=None)
        registry.reserve_publish(intent)

    with _open_registry(root) as restarted:
        assert restarted.get_publish_intent(WORKSPACE_ID) == intent
        assert restarted.get_publish_journal(WORKSPACE_ID) is None
        restarted.clear_publish_intent(WORKSPACE_ID, **_publish_token())
        assert restarted.get_publish_intent(WORKSPACE_ID) is None


def test_publish_journal_persists_normalized_claims_and_commits_atomically(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        _active_workspace(registry)
        registry.reserve_publish(_publish_intent(old_claim=None))
        registry.prepare_publish(_publish_journal(old_claim=None))

    with _open_registry(root) as restarted:
        assert restarted.get_publish_journal(WORKSPACE_ID) == _publish_journal(old_claim=None)
        restarted.advance_publish(
            WORKSPACE_ID,
            expected_phase="prepared",
            new_phase="backup",
            **_publish_token(),
        )
        restarted.advance_publish(
            WORKSPACE_ID,
            expected_phase="backup",
            new_phase="installed",
            **_publish_token(),
        )
        restarted.commit_publish(WORKSPACE_ID, NEW_CLAIM, **_publish_token())

        assert restarted.get_artifact(WORKSPACE_ID, NEW_CLAIM.slot) == NEW_CLAIM
        committed = restarted.get_publish_journal(WORKSPACE_ID)
        assert committed is not None
        assert committed.phase == "committed"
        restarted.clear_publish(WORKSPACE_ID, **_publish_token())
        assert restarted.get_publish_journal(WORKSPACE_ID) is None
        assert restarted.get_artifact(WORKSPACE_ID, NEW_CLAIM.slot) == NEW_CLAIM


def test_delete_journal_survives_restart_and_workspace_clear_cascades(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        _active_workspace(registry)
        registry.reserve_publish(_publish_intent(old_claim=None))
        registry.prepare_publish(_publish_journal(old_claim=None))
        registry.advance_publish(
            WORKSPACE_ID,
            expected_phase="prepared",
            new_phase="backup",
            **_publish_token(),
        )
        registry.advance_publish(
            WORKSPACE_ID,
            expected_phase="backup",
            new_phase="installed",
            **_publish_token(),
        )
        registry.commit_publish(WORKSPACE_ID, NEW_CLAIM, **_publish_token())
        registry.clear_publish(WORKSPACE_ID, **_publish_token())
        items = registry.begin_delete(WORKSPACE_ID, **_delete_token())
        assert len(items) == 1
        assert items[0].phase == "planned"
        assert items[0].claim == NEW_CLAIM

    with _open_registry(root) as restarted:
        item = restarted.get_delete_items(WORKSPACE_ID, **_delete_token())[0]
        restarted.advance_delete_item(
            WORKSPACE_ID,
            item.slot,
            **_delete_token(),
            expected_phase="planned",
            new_phase="quarantined",
            quarantine_name=".owned-quarantine-fixed",
        )
        restarted.advance_delete_item(
            WORKSPACE_ID,
            item.slot,
            **_delete_token(),
            expected_phase="quarantined",
            new_phase="removed",
            quarantine_name=".owned-quarantine-fixed",
        )
        restarted.clear_deleted_workspace(WORKSPACE_ID, **_delete_token())

        assert restarted.get_workspace(WORKSPACE_ID) is None
        assert restarted.get_collection_identities(WORKSPACE_ID) == {}
        assert restarted.get_artifact(WORKSPACE_ID, NEW_CLAIM.slot) is None
        assert restarted.get_delete_items(WORKSPACE_ID, **_delete_token()) == ()


def test_publish_abort_clears_only_the_expected_uncommitted_phase(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        _active_workspace(registry)
        registry.reserve_publish(_publish_intent(old_claim=None))
        registry.prepare_publish(_publish_journal(old_claim=None))
        registry.abort_publish(
            WORKSPACE_ID,
            expected_phases={"prepared", "backup"},
            **_publish_token(),
        )
        assert registry.get_publish_journal(WORKSPACE_ID) is None
        assert registry.get_artifact(WORKSPACE_ID, NEW_CLAIM.slot) is None

        with pytest.raises(RegistryError) as caught:
            registry.abort_publish(
                WORKSPACE_ID,
                expected_phases={"prepared"},
                **_publish_token(),
            )
        assert caught.value.code == "registry_state_conflict"


def test_delete_quarantine_name_is_durable_before_the_filesystem_move(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        _active_workspace(registry)
        registry.reserve_publish(_publish_intent(old_claim=None))
        registry.prepare_publish(_publish_journal(old_claim=None))
        registry.advance_publish(
            WORKSPACE_ID,
            expected_phase="prepared",
            new_phase="backup",
            **_publish_token(),
        )
        registry.advance_publish(
            WORKSPACE_ID,
            expected_phase="backup",
            new_phase="installed",
            **_publish_token(),
        )
        registry.commit_publish(WORKSPACE_ID, NEW_CLAIM, **_publish_token())
        registry.clear_publish(WORKSPACE_ID, **_publish_token())
        registry.begin_delete(WORKSPACE_ID, **_delete_token())
        registry.plan_delete_quarantine(
            WORKSPACE_ID,
            NEW_CLAIM.slot,
            ".owned-quarantine-before-move",
            **_delete_token(),
        )

    with _open_registry(root) as restarted:
        item = restarted.get_delete_items(WORKSPACE_ID, **_delete_token())[0]
        assert item.phase == "planned"
        assert item.quarantine_name == ".owned-quarantine-before-move"


def test_registry_replay_truncates_only_an_incomplete_final_record(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        _active_workspace(registry)
    path = root / REGISTRY_NAME
    committed_size = path.stat().st_size
    with path.open("ab") as target:
        target.write(b'{"sequence":999,"incomplete"')
        target.flush()
        os.fsync(target.fileno())

    with _open_registry(root) as restarted:
        assert restarted.get_workspace(WORKSPACE_ID).phase == "active"
    assert path.stat().st_size == committed_size


def test_registry_replay_rejects_a_corrupted_complete_record(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        _active_workspace(registry)
    path = root / REGISTRY_NAME
    content = bytearray(path.read_bytes())
    final_newline = len(content) - 1
    record_start = content.rfind(b"\n", 0, final_newline) + 1
    corrupt_at = next(
        index for index in range(record_start, final_newline) if content[index] in b"0123456789abcdef"
    )
    content[corrupt_at] = ord("f") if content[corrupt_at] != ord("f") else ord("e")
    path.write_bytes(content)

    with pytest.raises(RegistryError) as caught:
        with _open_registry(root):
            pytest.fail("a complete corrupted registry record must not be replayed")
    assert caught.value.code == "registry_corrupt"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("sequence", 999),
        ("previous_sha256", "f" * 64),
    ],
)
def test_registry_replay_rejects_rechecksummed_schema_sequence_and_chain_corruption(
    tmp_path,
    field,
    value,
):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        _active_workspace(registry)
    path = root / REGISTRY_NAME
    content = path.read_bytes()
    final_newline = len(content) - 1
    record_start = content.rfind(b"\n", 0, final_newline) + 1
    body, _digest = content[record_start:final_newline].rsplit(b"\t", 1)
    document = json.loads(body)
    actual_previous = bytes.fromhex(document["previous_sha256"])
    document[field] = value
    changed_body = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    changed_digest = hashlib.sha256(actual_previous + changed_body).hexdigest().encode("ascii")
    path.write_bytes(content[:record_start] + changed_body + b"\t" + changed_digest + b"\n")

    with pytest.raises(RegistryError) as caught:
        with _open_registry(root):
            pytest.fail("rechecksummed structural corruption must not be replayed")
    assert caught.value.code == "registry_corrupt"


def test_registry_serializes_concurrent_worker_mutations(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    workspace_ids = tuple(f"workspace_{index:016d}" for index in range(24))

    with _open_registry(root) as registry:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = tuple(executor.map(registry.reserve_workspace, workspace_ids))
        assert results == (True,) * len(workspace_ids)

    with _open_registry(root) as restarted:
        assert all(
            restarted.get_workspace(workspace_id).phase == "reserved" for workspace_id in workspace_ids
        )


def test_two_live_registry_instances_share_one_serialized_tail(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    first_workspace = "workspace_live_0001"
    second_workspace = "workspace_live_0002"

    with _open_registry(root) as first, _open_registry(root) as second:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = (
                executor.submit(first.reserve_workspace, first_workspace),
                executor.submit(second.reserve_workspace, second_workspace),
            )
            assert tuple(future.result() for future in futures) == (True, True)
        assert first.get_workspace(second_workspace).phase == "reserved"
        assert second.get_workspace(first_workspace).phase == "reserved"

    with _open_registry(root) as restarted:
        assert restarted.get_workspace(first_workspace).phase == "reserved"
        assert restarted.get_workspace(second_workspace).phase == "reserved"


def test_registry_refuses_an_append_that_would_cross_its_replay_limit(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    path = root / REGISTRY_NAME

    with _open_registry(root) as registry:
        original_limit = registry_module._MAX_LOG_BYTES
        monkeypatch.setattr(registry_module, "_MAX_LOG_BYTES", path.stat().st_size + 1)
        with pytest.raises(RegistryError) as caught:
            registry.reserve_workspace(WORKSPACE_ID)
        assert caught.value.code == "registry_operation_failed"
        monkeypatch.setattr(registry_module, "_MAX_LOG_BYTES", original_limit)
        assert registry.get_workspace(WORKSPACE_ID) is None

    with _open_registry(root) as restarted:
        assert restarted.get_workspace(WORKSPACE_ID) is None


def test_registry_corruption_errors_expose_only_a_stable_safe_code(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root):
        pass
    secret = "private-provider-token-must-not-escape"
    with (root / REGISTRY_NAME).open("ab") as target:
        target.write(secret.encode() + b"\t" + b"0" * 64 + b"\n")

    with pytest.raises(RegistryError) as caught:
        with _open_registry(root):
            pytest.fail("malformed complete records must fail closed")
    assert caught.value.code == "registry_corrupt"
    assert str(caught.value) == "registry_corrupt"
    assert secret not in str(caught.value)
    assert str(root) not in str(caught.value)


def test_unreleased_legacy_sqlite_registry_is_never_adopted(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    legacy = root / LEGACY_REGISTRY_NAME
    legacy.write_bytes(b"SQLite format 3\x00unreleased authority")

    with pytest.raises(RegistryError) as caught:
        with _open_registry(root):
            pytest.fail("legacy pathname state must fail closed")
    assert caught.value.code == "registry_legacy_state"
    assert legacy.read_bytes() == b"SQLite format 3\x00unreleased authority"
    assert not (root / REGISTRY_NAME).exists()


@pytest.mark.parametrize(
    "forged",
    [
        b"",
        b"EXAMSAGE-EVIDENCE-REGISTRY",
        b"EXAMSAGE-EVIDENCE-REGISTRY-LOG-v1\n",
    ],
)
def test_existing_uninitialized_registry_log_is_never_adopted(tmp_path, forged):
    root = tmp_path / "data"
    root.mkdir()
    path = root / REGISTRY_NAME
    path.write_bytes(forged)

    with pytest.raises(RegistryError) as caught:
        with _open_registry(root):
            pytest.fail("an existing uninitialized authority must fail closed")
    assert caught.value.code == "registry_corrupt"
    assert path.read_bytes() == forged


@pytest.mark.parametrize("replacement", [b"", registry_module._LOG_MAGIC])
def test_live_registry_rejects_same_inode_rollback_below_acknowledged_tail(
    tmp_path,
    replacement,
):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        assert registry.reserve_workspace(WORKSPACE_ID) is True
        source = registry._file
        assert source is not None
        os.ftruncate(source.descriptor, 0)
        os.lseek(source.descriptor, 0, os.SEEK_SET)
        if replacement:
            os.write(source.descriptor, replacement)
        os.fsync(source.descriptor)

        with pytest.raises(RegistryError) as caught:
            registry.get_workspace(WORKSPACE_ID)

        assert caught.value.code == "registry_corrupt"
        assert os.fstat(source.descriptor).st_size == len(replacement)


def test_stale_registry_cannot_retire_a_new_reservation_generation(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as initial:
        assert initial.reserve_workspace(WORKSPACE_ID) is True

    with _open_registry(root) as stale, _open_registry(root) as current:
        assert current.retire_recovered_workspace(WORKSPACE_ID) is True
        assert current.reserve_workspace(WORKSPACE_ID) is True
        replacement = current.get_workspace(WORKSPACE_ID)
        assert replacement is not None

        assert stale.retire_recovered_workspace(WORKSPACE_ID) is False
        assert current.get_workspace(WORKSPACE_ID) == replacement


def test_every_registry_relation_carries_the_workspace_generation(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        _publish_registry_claim(registry)
        workspace = registry.get_workspace(WORKSPACE_ID)
        artifact = registry.get_artifact(WORKSPACE_ID, NEW_CLAIM.slot)
        assert workspace is not None
        assert artifact is not None
        assert artifact.generation == workspace.generation

        registry.reserve_publish(_publish_intent(old_claim=NEW_CLAIM))
        intent = registry.get_publish_intent(WORKSPACE_ID)
        assert intent is not None
        assert intent.generation == workspace.generation
        registry.clear_publish_intent(WORKSPACE_ID, **_publish_token())

        items = registry.begin_delete(WORKSPACE_ID, **_delete_token())
        assert items
        assert all(item.generation == workspace.generation for item in items)


def _publish_registry_claim(registry: EvidenceArtifactRegistry) -> None:
    _active_workspace(registry)
    registry.reserve_publish(_publish_intent(old_claim=None))
    registry.prepare_publish(_publish_journal(old_claim=None))
    registry.advance_publish(
        WORKSPACE_ID,
        expected_phase="prepared",
        new_phase="backup",
        **_publish_token(),
    )
    registry.advance_publish(
        WORKSPACE_ID,
        expected_phase="backup",
        new_phase="installed",
        **_publish_token(),
    )
    registry.commit_publish(WORKSPACE_ID, NEW_CLAIM, **_publish_token())
    registry.clear_publish(WORKSPACE_ID, **_publish_token())


def test_stale_publish_token_cannot_advance_a_new_operation(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    next_temporary = ".artifact-next.tmp"
    next_backup = ".artifact-next.backup"

    with _open_registry(root) as registry:
        _active_workspace(registry)
        registry.reserve_publish(_publish_intent(old_claim=None))
        registry.prepare_publish(_publish_journal(old_claim=None))
        registry.abort_publish(
            WORKSPACE_ID,
            expected_phases={"prepared"},
            **_publish_token(),
        )

        next_intent = replace(
            _publish_intent(old_claim=None),
            temporary_name=next_temporary,
            backup_name=next_backup,
        )
        next_journal = replace(
            _publish_journal(old_claim=None),
            temporary_name=next_temporary,
            backup_name=next_backup,
        )
        registry.reserve_publish(next_intent)
        registry.prepare_publish(next_journal)

        with pytest.raises(RegistryError) as caught:
            registry.advance_publish(
                WORKSPACE_ID,
                expected_phase="prepared",
                new_phase="backup",
                **_publish_token(),
            )

        assert caught.value.code == "registry_state_conflict"
        assert registry.get_publish_journal(WORKSPACE_ID) == next_journal


def test_stale_delete_generation_cannot_begin_a_new_workspace_deletion(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    generation_a = "a" * 32
    generation_b = "b" * 32

    with _open_registry(root) as registry:
        _active_workspace(registry)
        assert registry.begin_delete(
            WORKSPACE_ID,
            expected_generation=generation_a,
        ) == ()
        registry.clear_deleted_workspace(
            WORKSPACE_ID,
            expected_generation=generation_a,
        )
        assert registry.reserve_workspace(WORKSPACE_ID, generation=generation_b)
        registry.finalize_workspace(
            WORKSPACE_ID,
            expected_generation=generation_b,
            workspace_identity=(61, 62),
            evidence_identity=(63, 64),
            collection_identities=COLLECTION_IDENTITIES,
        )

        with pytest.raises(RegistryError) as caught:
            registry.begin_delete(
                WORKSPACE_ID,
                expected_generation=generation_a,
            )

        assert caught.value.code == "registry_state_conflict"
        current = registry.get_workspace(WORKSPACE_ID)
        assert current is not None
        assert current.phase == "active"
        assert current.generation == generation_b


def _rewrite_final_state(root, mutate) -> None:
    path = root / REGISTRY_NAME
    content = path.read_bytes()
    final_newline = len(content) - 1
    record_start = content.rfind(b"\n", 0, final_newline) + 1
    body, _digest = content[record_start:final_newline].rsplit(b"\t", 1)
    document = json.loads(body)
    mutate(document["state"])
    changed_body = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    previous = bytes.fromhex(document["previous_sha256"])
    changed_digest = hashlib.sha256(previous + changed_body).hexdigest().encode("ascii")
    path.write_bytes(content[:record_start] + changed_body + b"\t" + changed_digest + b"\n")


@pytest.mark.parametrize(
    "case",
    [
        "reserved_has_collections",
        "active_has_delete_rows",
        "artifact_route_mismatch",
        "publish_old_claim_mismatch",
        "delete_claim_mismatch",
    ],
)
def test_registry_replay_rejects_rechecksummed_relationally_impossible_state(
    tmp_path,
    case,
):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        if case == "reserved_has_collections":
            _active_workspace(registry)
        else:
            _publish_registry_claim(registry)
            if case == "publish_old_claim_mismatch":
                registry.reserve_publish(_publish_intent(old_claim=NEW_CLAIM))
            elif case in {"active_has_delete_rows", "delete_claim_mismatch"}:
                registry.begin_delete(WORKSPACE_ID, **_delete_token())

    def corrupt(state):
        workspace = state["workspaces"][0]
        if case == "reserved_has_collections":
            workspace["phase"] = "reserved"
            workspace["workspace_identity"] = None
            workspace["evidence_identity"] = None
        elif case == "active_has_delete_rows":
            workspace["phase"] = "active"
        elif case == "artifact_route_mismatch":
            state["artifacts"][0]["artifact_id"] = "different_0123456789"
        elif case == "publish_old_claim_mismatch":
            state["publish_intents"][0]["old_claim"]["identity"] = ["999", "1000"]
        elif case == "delete_claim_mismatch":
            state["delete_items"][0]["claim"]["identity"] = ["999", "1000"]
        else:
            raise AssertionError(case)

    _rewrite_final_state(root, corrupt)

    with pytest.raises(RegistryError) as caught:
        with _open_registry(root):
            pytest.fail("semantically impossible authority must fail closed")
    assert caught.value.code == "registry_corrupt"
