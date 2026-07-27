from __future__ import annotations

import sqlite3
from contextlib import contextmanager

import pytest

from exam_predictor.evidence.registry import (
    ArtifactClaim,
    EvidenceArtifactRegistry,
    PublishJournal,
    RegistryError,
)
from exam_predictor.workspace.filesystem import OwnedArtifactFilesystem


WORKSPACE_ID = "workspace_0123456789"
REGISTRY_NAME = ".evidence-artifact-registry.sqlite3"
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
)


@contextmanager
def _open_registry(root):
    filesystem = OwnedArtifactFilesystem()
    with filesystem.anchor_directory(root) as root_anchor:
        with EvidenceArtifactRegistry(root_anchor, filesystem) as registry:
            yield registry


def _active_workspace(registry: EvidenceArtifactRegistry) -> None:
    assert registry.reserve_workspace(WORKSPACE_ID) is True
    registry.finalize_workspace(
        WORKSPACE_ID,
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
    )


def test_registry_uses_fixed_pinned_file_and_required_sqlite_pragmas(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        assert registry.path == root / REGISTRY_NAME
        assert registry.pragmas() == {
            "foreign_keys": 1,
            "journal_mode": "wal",
            "synchronous": 2,
        }
        pinned_identity = registry.identity
        named = registry.path.stat(follow_symlinks=False)
        assert pinned_identity == (named.st_dev, named.st_ino)

        with sqlite3.connect(registry.path) as observer:
            tables = {
                row[0] for row in observer.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            assert {
                "metadata",
                "workspaces",
                "collections",
                "artifacts",
                "publish_journal",
                "delete_journal",
            } <= tables


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
                workspace_identity=(91, 92),
                evidence_identity=(93, 94),
                collection_identities=COLLECTION_IDENTITIES,
            )
        assert caught.value.code == "registry_state_conflict"
        assert registry.get_workspace(WORKSPACE_ID) == active


def test_publish_journal_persists_normalized_claims_and_commits_atomically(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        _active_workspace(registry)
        registry.prepare_publish(_publish_journal(old_claim=None))

    with _open_registry(root) as restarted:
        assert restarted.get_publish_journal(WORKSPACE_ID) == _publish_journal(old_claim=None)
        restarted.advance_publish(WORKSPACE_ID, expected_phase="prepared", new_phase="backup")
        restarted.advance_publish(WORKSPACE_ID, expected_phase="backup", new_phase="installed")
        restarted.commit_publish(WORKSPACE_ID, NEW_CLAIM)

        assert restarted.get_artifact(WORKSPACE_ID, NEW_CLAIM.slot) == NEW_CLAIM
        committed = restarted.get_publish_journal(WORKSPACE_ID)
        assert committed is not None
        assert committed.phase == "committed"
        restarted.clear_publish(WORKSPACE_ID)
        assert restarted.get_publish_journal(WORKSPACE_ID) is None

        with sqlite3.connect(restarted.path) as observer:
            columns = {row[1] for row in observer.execute("PRAGMA table_info(publish_journal)")}
            assert "payload" not in columns
            assert {
                "new_device_id",
                "new_file_id",
                "new_sha256",
                "new_size",
                "old_device_id",
                "old_file_id",
                "old_sha256",
                "old_size",
            } <= columns


def test_delete_journal_survives_restart_and_workspace_clear_cascades(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    with _open_registry(root) as registry:
        _active_workspace(registry)
        registry.prepare_publish(_publish_journal(old_claim=None))
        registry.advance_publish(WORKSPACE_ID, expected_phase="prepared", new_phase="backup")
        registry.advance_publish(WORKSPACE_ID, expected_phase="backup", new_phase="installed")
        registry.commit_publish(WORKSPACE_ID, NEW_CLAIM)
        registry.clear_publish(WORKSPACE_ID)
        items = registry.begin_delete(WORKSPACE_ID)
        assert len(items) == 1
        assert items[0].phase == "planned"
        assert items[0].claim == NEW_CLAIM

    with _open_registry(root) as restarted:
        item = restarted.get_delete_items(WORKSPACE_ID)[0]
        restarted.advance_delete_item(
            WORKSPACE_ID,
            item.slot,
            expected_phase="planned",
            new_phase="quarantined",
            quarantine_name=".owned-quarantine-fixed",
        )
        restarted.advance_delete_item(
            WORKSPACE_ID,
            item.slot,
            expected_phase="quarantined",
            new_phase="removed",
            quarantine_name=".owned-quarantine-fixed",
        )
        restarted.clear_deleted_workspace(WORKSPACE_ID)

        assert restarted.get_workspace(WORKSPACE_ID) is None
        assert restarted.get_collection_identities(WORKSPACE_ID) == {}
        assert restarted.get_artifact(WORKSPACE_ID, NEW_CLAIM.slot) is None
        assert restarted.get_delete_items(WORKSPACE_ID) == ()
