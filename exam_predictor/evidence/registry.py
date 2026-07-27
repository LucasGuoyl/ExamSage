from __future__ import annotations

import contextlib
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from exam_predictor.workspace.filesystem import (
    OwnedArtifactFilesystem,
    OwnedDirectoryAnchor,
    OwnedFilesystemError,
    OwnedMutationFile,
)


REGISTRY_NAME = ".evidence-artifact-registry.sqlite3"
_SCHEMA_VERSION = 1
_COLLECTIONS = frozenset({"parts", "units", "snapshots"})

Identity = tuple[int, int]


class RegistryError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class WorkspaceClaim:
    workspace_id: str
    phase: str
    workspace_identity: Identity | None
    evidence_identity: Identity | None


@dataclass(frozen=True)
class ArtifactClaim:
    workspace_id: str
    slot: str
    collection: str
    kind: str
    artifact_id: str
    identity: Identity
    sha256: str
    size: int


@dataclass(frozen=True)
class PublishJournal:
    workspace_id: str
    slot: str
    phase: str
    target_name: str
    temporary_name: str
    backup_name: str
    new_claim: ArtifactClaim
    old_claim: ArtifactClaim | None


@dataclass(frozen=True)
class DeleteJournalItem:
    workspace_id: str
    slot: str
    phase: str
    quarantine_name: str | None
    claim: ArtifactClaim


class EvidenceArtifactRegistry:
    """Pinned SQLite authority for evidence-owned identities and journals."""

    def __init__(
        self,
        root: OwnedDirectoryAnchor,
        filesystem: OwnedArtifactFilesystem,
    ) -> None:
        self._root = root
        self._filesystem = filesystem
        self._file_context = filesystem.open_or_create_mutation_file(
            root,
            REGISTRY_NAME,
            expected_parent_identity=root.identity,
        )
        self._file: OwnedMutationFile | None = None
        self._connection: sqlite3.Connection | None = None
        self._closed = False
        try:
            self._file = self._file_context.__enter__()
            self._connection = sqlite3.connect(
                self.path,
                isolation_level=None,
                timeout=10.0,
            )
            self._connection.row_factory = sqlite3.Row
            self._configure()
            self._initialize_schema()
        except Exception:
            self.close()
            raise

    @property
    def path(self) -> Path:
        return self._root.path / REGISTRY_NAME

    @property
    def identity(self) -> Identity:
        if self._file is None:
            raise RegistryError("registry_closed")
        return self._file.identity

    def __enter__(self) -> EvidenceArtifactRegistry:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._connection is not None:
            with contextlib.suppress(sqlite3.Error):
                self._connection.close()
            self._connection = None
        if self._file is not None:
            self._file_context.__exit__(None, None, None)
            self._file = None

    def pragmas(self) -> dict[str, int | str]:
        connection = self._require_connection()
        try:
            return {
                "foreign_keys": int(connection.execute("PRAGMA foreign_keys").fetchone()[0]),
                "journal_mode": str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
                "synchronous": int(connection.execute("PRAGMA synchronous").fetchone()[0]),
            }
        except sqlite3.Error:
            raise RegistryError("registry_operation_failed") from None

    def reserve_workspace(self, workspace_id: str) -> bool:
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO workspaces(
                    workspace_id, phase, workspace_device_id, workspace_file_id,
                    evidence_device_id, evidence_file_id
                ) VALUES (?, 'reserved', NULL, NULL, NULL, NULL)
                ON CONFLICT(workspace_id) DO NOTHING
                """,
                (workspace_id,),
            )
            return cursor.rowcount == 1

    def finalize_workspace(
        self,
        workspace_id: str,
        *,
        workspace_identity: Identity,
        evidence_identity: Identity,
        collection_identities: dict[str, Identity],
    ) -> WorkspaceClaim:
        if set(collection_identities) != _COLLECTIONS:
            raise RegistryError("registry_claim_invalid")
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE workspaces
                SET phase = 'active',
                    workspace_device_id = ?, workspace_file_id = ?,
                    evidence_device_id = ?, evidence_file_id = ?
                WHERE workspace_id = ? AND phase = 'reserved'
                  AND workspace_device_id IS NULL AND workspace_file_id IS NULL
                  AND evidence_device_id IS NULL AND evidence_file_id IS NULL
                """,
                (
                    str(workspace_identity[0]),
                    str(workspace_identity[1]),
                    str(evidence_identity[0]),
                    str(evidence_identity[1]),
                    workspace_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RegistryError("registry_state_conflict")
            connection.executemany(
                """
                INSERT INTO collections(
                    workspace_id, collection_name, device_id, file_id
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (workspace_id, name, str(identity[0]), str(identity[1]))
                    for name, identity in sorted(collection_identities.items())
                ],
            )
        claim = self.get_workspace(workspace_id)
        if claim is None:
            raise RegistryError("registry_operation_failed")
        return claim

    def get_workspace(self, workspace_id: str) -> WorkspaceClaim | None:
        row = self._fetchone(
            """
            SELECT workspace_id, phase, workspace_device_id, workspace_file_id,
                   evidence_device_id, evidence_file_id
            FROM workspaces WHERE workspace_id = ?
            """,
            (workspace_id,),
        )
        if row is None:
            return None
        return WorkspaceClaim(
            workspace_id=str(row["workspace_id"]),
            phase=str(row["phase"]),
            workspace_identity=self._optional_identity(row["workspace_device_id"], row["workspace_file_id"]),
            evidence_identity=self._optional_identity(row["evidence_device_id"], row["evidence_file_id"]),
        )

    def get_collection_identities(self, workspace_id: str) -> dict[str, Identity]:
        rows = self._fetchall(
            """
            SELECT collection_name, device_id, file_id
            FROM collections WHERE workspace_id = ? ORDER BY collection_name
            """,
            (workspace_id,),
        )
        return {str(row["collection_name"]): self._identity(row["device_id"], row["file_id"]) for row in rows}

    def get_artifact(self, workspace_id: str, slot: str) -> ArtifactClaim | None:
        row = self._fetchone(
            "SELECT * FROM artifacts WHERE workspace_id = ? AND slot = ?",
            (workspace_id, slot),
        )
        return None if row is None else self._artifact_from_row(row)

    def get_artifacts(self, workspace_id: str) -> tuple[ArtifactClaim, ...]:
        rows = self._fetchall(
            "SELECT * FROM artifacts WHERE workspace_id = ? ORDER BY slot",
            (workspace_id,),
        )
        return tuple(self._artifact_from_row(row) for row in rows)

    def prepare_publish(self, journal: PublishJournal) -> None:
        self._validate_publish_journal(journal)
        with self._transaction() as connection:
            workspace = connection.execute(
                "SELECT phase FROM workspaces WHERE workspace_id = ?",
                (journal.workspace_id,),
            ).fetchone()
            if workspace is None or workspace["phase"] != "active":
                raise RegistryError("registry_state_conflict")
            current_row = connection.execute(
                "SELECT * FROM artifacts WHERE workspace_id = ? AND slot = ?",
                (journal.workspace_id, journal.slot),
            ).fetchone()
            current = None if current_row is None else self._artifact_from_row(current_row)
            if current != journal.old_claim:
                raise RegistryError("registry_state_conflict")
            values = self._publish_values(journal)
            try:
                connection.execute(
                    """
                    INSERT INTO publish_journal(
                        workspace_id, slot, phase, target_name, temporary_name, backup_name,
                        new_collection, new_kind, new_artifact_id,
                        new_device_id, new_file_id, new_sha256, new_size,
                        old_collection, old_kind, old_artifact_id,
                        old_device_id, old_file_id, old_sha256, old_size
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    values,
                )
            except sqlite3.IntegrityError:
                raise RegistryError("registry_state_conflict") from None

    def get_publish_journal(self, workspace_id: str) -> PublishJournal | None:
        row = self._fetchone(
            "SELECT * FROM publish_journal WHERE workspace_id = ?",
            (workspace_id,),
        )
        return None if row is None else self._publish_from_row(row)

    def advance_publish(self, workspace_id: str, *, expected_phase: str, new_phase: str) -> None:
        allowed = {
            ("prepared", "backup"),
            ("backup", "installed"),
        }
        if (expected_phase, new_phase) not in allowed:
            raise RegistryError("registry_claim_invalid")
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE publish_journal SET phase = ?
                WHERE workspace_id = ? AND phase = ?
                """,
                (new_phase, workspace_id, expected_phase),
            )
            if cursor.rowcount != 1:
                raise RegistryError("registry_state_conflict")

    def commit_publish(self, workspace_id: str, claim: ArtifactClaim) -> None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM publish_journal WHERE workspace_id = ? AND phase = 'installed'",
                (workspace_id,),
            ).fetchone()
            if row is None:
                raise RegistryError("registry_state_conflict")
            journal = self._publish_from_row(row)
            if journal.new_claim != claim:
                raise RegistryError("registry_state_conflict")
            connection.execute(
                """
                INSERT INTO artifacts(
                    workspace_id, slot, collection_name, kind, artifact_id,
                    device_id, file_id, sha256, size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id, slot) DO UPDATE SET
                    collection_name = excluded.collection_name,
                    kind = excluded.kind,
                    artifact_id = excluded.artifact_id,
                    device_id = excluded.device_id,
                    file_id = excluded.file_id,
                    sha256 = excluded.sha256,
                    size = excluded.size
                """,
                self._artifact_values(claim),
            )
            connection.execute(
                "UPDATE publish_journal SET phase = 'committed' WHERE workspace_id = ?",
                (workspace_id,),
            )

    def clear_publish(self, workspace_id: str) -> None:
        with self._transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM publish_journal WHERE workspace_id = ? AND phase = 'committed'",
                (workspace_id,),
            )
            if cursor.rowcount != 1:
                raise RegistryError("registry_state_conflict")

    def abort_publish(self, workspace_id: str, *, expected_phases: set[str]) -> None:
        if not expected_phases or not expected_phases <= {"prepared", "backup"}:
            raise RegistryError("registry_claim_invalid")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT phase FROM publish_journal WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
            if row is None or str(row["phase"]) not in expected_phases:
                raise RegistryError("registry_state_conflict")
            connection.execute(
                "DELETE FROM publish_journal WHERE workspace_id = ?",
                (workspace_id,),
            )

    def begin_delete(self, workspace_id: str) -> tuple[DeleteJournalItem, ...]:
        with self._transaction() as connection:
            workspace = connection.execute(
                "SELECT phase FROM workspaces WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
            if workspace is None:
                return ()
            if workspace["phase"] == "deleting":
                pass
            elif workspace["phase"] == "active":
                if connection.execute(
                    "SELECT 1 FROM publish_journal WHERE workspace_id = ?",
                    (workspace_id,),
                ).fetchone():
                    raise RegistryError("registry_pending")
                connection.execute(
                    "UPDATE workspaces SET phase = 'deleting' WHERE workspace_id = ?",
                    (workspace_id,),
                )
                connection.execute(
                    """
                    INSERT INTO delete_journal(
                        workspace_id, slot, phase, quarantine_name, collection_name,
                        kind, artifact_id, device_id, file_id, sha256, size
                    )
                    SELECT workspace_id, slot, 'planned', NULL, collection_name,
                           kind, artifact_id, device_id, file_id, sha256, size
                    FROM artifacts WHERE workspace_id = ?
                    """,
                    (workspace_id,),
                )
            else:
                raise RegistryError("registry_state_conflict")
        return self.get_delete_items(workspace_id)

    def get_delete_items(self, workspace_id: str) -> tuple[DeleteJournalItem, ...]:
        rows = self._fetchall(
            "SELECT * FROM delete_journal WHERE workspace_id = ? ORDER BY slot",
            (workspace_id,),
        )
        return tuple(self._delete_from_row(row) for row in rows)

    def plan_delete_quarantine(
        self,
        workspace_id: str,
        slot: str,
        quarantine_name: str,
    ) -> None:
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE delete_journal SET quarantine_name = ?
                WHERE workspace_id = ? AND slot = ? AND phase = 'planned'
                  AND (quarantine_name IS NULL OR quarantine_name = ?)
                """,
                (quarantine_name, workspace_id, slot, quarantine_name),
            )
            if cursor.rowcount != 1:
                raise RegistryError("registry_state_conflict")

    def advance_delete_item(
        self,
        workspace_id: str,
        slot: str,
        *,
        expected_phase: str,
        new_phase: str,
        quarantine_name: str,
    ) -> None:
        allowed = {("planned", "quarantined"), ("quarantined", "removed")}
        if (expected_phase, new_phase) not in allowed:
            raise RegistryError("registry_claim_invalid")
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE delete_journal SET phase = ?, quarantine_name = ?
                WHERE workspace_id = ? AND slot = ? AND phase = ?
                  AND (quarantine_name IS NULL OR quarantine_name = ?)
                """,
                (
                    new_phase,
                    quarantine_name,
                    workspace_id,
                    slot,
                    expected_phase,
                    quarantine_name,
                ),
            )
            if cursor.rowcount != 1:
                raise RegistryError("registry_state_conflict")

    def clear_deleted_workspace(self, workspace_id: str) -> None:
        with self._transaction() as connection:
            workspace = connection.execute(
                "SELECT phase FROM workspaces WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
            if workspace is None:
                return
            remaining = connection.execute(
                """
                SELECT 1 FROM delete_journal
                WHERE workspace_id = ? AND phase != 'removed' LIMIT 1
                """,
                (workspace_id,),
            ).fetchone()
            if workspace["phase"] != "deleting" or remaining is not None:
                raise RegistryError("registry_state_conflict")
            connection.execute("DELETE FROM workspaces WHERE workspace_id = ?", (workspace_id,))

    def _configure(self) -> None:
        connection = self._require_connection()
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
            connection.execute("PRAGMA synchronous = FULL")
            if mode != "wal" or self.pragmas() != {
                "foreign_keys": 1,
                "journal_mode": "wal",
                "synchronous": 2,
            }:
                raise RegistryError("registry_configuration_failed")
        except RegistryError:
            raise
        except sqlite3.Error:
            raise RegistryError("registry_configuration_failed") from None

    def _initialize_schema(self) -> None:
        connection = self._require_connection()
        try:
            connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS metadata(
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    schema_version INTEGER NOT NULL CHECK(schema_version = 1),
                    root_device_id TEXT NOT NULL,
                    root_file_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workspaces(
                    workspace_id TEXT PRIMARY KEY,
                    phase TEXT NOT NULL CHECK(phase IN ('reserved', 'active', 'deleting')),
                    workspace_device_id TEXT,
                    workspace_file_id TEXT,
                    evidence_device_id TEXT,
                    evidence_file_id TEXT
                );
                CREATE TABLE IF NOT EXISTS collections(
                    workspace_id TEXT NOT NULL,
                    collection_name TEXT NOT NULL CHECK(collection_name IN ('parts', 'units', 'snapshots')),
                    device_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    PRIMARY KEY(workspace_id, collection_name),
                    FOREIGN KEY(workspace_id) REFERENCES workspaces(workspace_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS artifacts(
                    workspace_id TEXT NOT NULL,
                    slot TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size INTEGER NOT NULL CHECK(size >= 0),
                    PRIMARY KEY(workspace_id, slot),
                    FOREIGN KEY(workspace_id, collection_name)
                        REFERENCES collections(workspace_id, collection_name) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS publish_journal(
                    workspace_id TEXT PRIMARY KEY,
                    slot TEXT NOT NULL,
                    phase TEXT NOT NULL CHECK(phase IN ('prepared', 'backup', 'installed', 'committed')),
                    target_name TEXT NOT NULL,
                    temporary_name TEXT NOT NULL,
                    backup_name TEXT NOT NULL,
                    new_collection TEXT NOT NULL,
                    new_kind TEXT NOT NULL,
                    new_artifact_id TEXT NOT NULL,
                    new_device_id TEXT NOT NULL,
                    new_file_id TEXT NOT NULL,
                    new_sha256 TEXT NOT NULL,
                    new_size INTEGER NOT NULL CHECK(new_size >= 0),
                    old_collection TEXT,
                    old_kind TEXT,
                    old_artifact_id TEXT,
                    old_device_id TEXT,
                    old_file_id TEXT,
                    old_sha256 TEXT,
                    old_size INTEGER CHECK(old_size >= 0),
                    FOREIGN KEY(workspace_id) REFERENCES workspaces(workspace_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS delete_journal(
                    workspace_id TEXT NOT NULL,
                    slot TEXT NOT NULL,
                    phase TEXT NOT NULL CHECK(phase IN ('planned', 'quarantined', 'removed')),
                    quarantine_name TEXT,
                    collection_name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size INTEGER NOT NULL CHECK(size >= 0),
                    PRIMARY KEY(workspace_id, slot),
                    FOREIGN KEY(workspace_id, slot)
                        REFERENCES artifacts(workspace_id, slot) ON DELETE CASCADE
                );
                COMMIT;
                """
            )
            with self._transaction() as transaction:
                row = transaction.execute("SELECT * FROM metadata WHERE singleton = 1").fetchone()
                root_identity = self._root.identity
                if row is None:
                    transaction.execute(
                        """
                        INSERT INTO metadata(singleton, schema_version, root_device_id, root_file_id)
                        VALUES (1, ?, ?, ?)
                        """,
                        (_SCHEMA_VERSION, str(root_identity[0]), str(root_identity[1])),
                    )
                elif (
                    int(row["schema_version"]) != _SCHEMA_VERSION
                    or self._identity(row["root_device_id"], row["root_file_id"]) != root_identity
                ):
                    raise RegistryError("registry_identity_changed")
        except RegistryError:
            raise
        except sqlite3.Error:
            with contextlib.suppress(sqlite3.Error):
                connection.execute("ROLLBACK")
            raise RegistryError("registry_operation_failed") from None

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._require_connection()
        self._verify_pinned()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
            self._verify_pinned()
        except RegistryError:
            with contextlib.suppress(sqlite3.Error):
                connection.execute("ROLLBACK")
            raise
        except sqlite3.Error:
            with contextlib.suppress(sqlite3.Error):
                connection.execute("ROLLBACK")
            raise RegistryError("registry_operation_failed") from None
        except Exception:
            with contextlib.suppress(sqlite3.Error):
                connection.execute("ROLLBACK")
            raise

    def _fetchone(self, query: str, parameters: tuple[object, ...]) -> sqlite3.Row | None:
        self._verify_pinned()
        try:
            return self._require_connection().execute(query, parameters).fetchone()
        except sqlite3.Error:
            raise RegistryError("registry_operation_failed") from None

    def _fetchall(self, query: str, parameters: tuple[object, ...]) -> list[sqlite3.Row]:
        self._verify_pinned()
        try:
            return list(self._require_connection().execute(query, parameters).fetchall())
        except sqlite3.Error:
            raise RegistryError("registry_operation_failed") from None

    def _verify_pinned(self) -> None:
        if self._file is None:
            raise RegistryError("registry_closed")
        try:
            if not self._filesystem.name_has_identity(
                self._root,
                REGISTRY_NAME,
                expected_parent_identity=self._root.identity,
                expected_source_identity=self._file.identity,
            ):
                raise RegistryError("registry_identity_changed")
        except OwnedFilesystemError:
            raise RegistryError("registry_identity_changed") from None

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RegistryError("registry_closed")
        return self._connection

    @staticmethod
    def _identity(device_id: object, file_id: object) -> Identity:
        return int(str(device_id)), int(str(file_id))

    @classmethod
    def _optional_identity(cls, device_id: object, file_id: object) -> Identity | None:
        if device_id is None and file_id is None:
            return None
        if device_id is None or file_id is None:
            raise RegistryError("registry_claim_invalid")
        return cls._identity(device_id, file_id)

    @classmethod
    def _artifact_from_row(cls, row: sqlite3.Row) -> ArtifactClaim:
        collection_key = "collection_name" if "collection_name" in row.keys() else "new_collection"
        kind_key = "kind" if "kind" in row.keys() else "new_kind"
        id_key = "artifact_id" if "artifact_id" in row.keys() else "new_artifact_id"
        device_key = "device_id" if "device_id" in row.keys() else "new_device_id"
        file_key = "file_id" if "file_id" in row.keys() else "new_file_id"
        sha_key = "sha256" if "sha256" in row.keys() else "new_sha256"
        size_key = "size" if "size" in row.keys() else "new_size"
        return ArtifactClaim(
            workspace_id=str(row["workspace_id"]),
            slot=str(row["slot"]),
            collection=str(row[collection_key]),
            kind=str(row[kind_key]),
            artifact_id=str(row[id_key]),
            identity=cls._identity(row[device_key], row[file_key]),
            sha256=str(row[sha_key]),
            size=int(row[size_key]),
        )

    @staticmethod
    def _artifact_values(claim: ArtifactClaim) -> tuple[object, ...]:
        return (
            claim.workspace_id,
            claim.slot,
            claim.collection,
            claim.kind,
            claim.artifact_id,
            str(claim.identity[0]),
            str(claim.identity[1]),
            claim.sha256,
            claim.size,
        )

    @classmethod
    def _publish_from_row(cls, row: sqlite3.Row) -> PublishJournal:
        new_claim = ArtifactClaim(
            workspace_id=str(row["workspace_id"]),
            slot=str(row["slot"]),
            collection=str(row["new_collection"]),
            kind=str(row["new_kind"]),
            artifact_id=str(row["new_artifact_id"]),
            identity=cls._identity(row["new_device_id"], row["new_file_id"]),
            sha256=str(row["new_sha256"]),
            size=int(row["new_size"]),
        )
        old_claim = None
        if row["old_device_id"] is not None:
            old_claim = ArtifactClaim(
                workspace_id=str(row["workspace_id"]),
                slot=str(row["slot"]),
                collection=str(row["old_collection"]),
                kind=str(row["old_kind"]),
                artifact_id=str(row["old_artifact_id"]),
                identity=cls._identity(row["old_device_id"], row["old_file_id"]),
                sha256=str(row["old_sha256"]),
                size=int(row["old_size"]),
            )
        return PublishJournal(
            workspace_id=str(row["workspace_id"]),
            slot=str(row["slot"]),
            phase=str(row["phase"]),
            target_name=str(row["target_name"]),
            temporary_name=str(row["temporary_name"]),
            backup_name=str(row["backup_name"]),
            new_claim=new_claim,
            old_claim=old_claim,
        )

    @staticmethod
    def _publish_values(journal: PublishJournal) -> tuple[object, ...]:
        new = journal.new_claim
        old = journal.old_claim
        old_values: tuple[object, ...]
        if old is None:
            old_values = (None, None, None, None, None, None, None)
        else:
            old_values = (
                old.collection,
                old.kind,
                old.artifact_id,
                str(old.identity[0]),
                str(old.identity[1]),
                old.sha256,
                old.size,
            )
        return (
            journal.workspace_id,
            journal.slot,
            journal.phase,
            journal.target_name,
            journal.temporary_name,
            journal.backup_name,
            new.collection,
            new.kind,
            new.artifact_id,
            str(new.identity[0]),
            str(new.identity[1]),
            new.sha256,
            new.size,
            *old_values,
        )

    @classmethod
    def _delete_from_row(cls, row: sqlite3.Row) -> DeleteJournalItem:
        claim = ArtifactClaim(
            workspace_id=str(row["workspace_id"]),
            slot=str(row["slot"]),
            collection=str(row["collection_name"]),
            kind=str(row["kind"]),
            artifact_id=str(row["artifact_id"]),
            identity=cls._identity(row["device_id"], row["file_id"]),
            sha256=str(row["sha256"]),
            size=int(row["size"]),
        )
        return DeleteJournalItem(
            workspace_id=claim.workspace_id,
            slot=claim.slot,
            phase=str(row["phase"]),
            quarantine_name=None if row["quarantine_name"] is None else str(row["quarantine_name"]),
            claim=claim,
        )

    @staticmethod
    def _validate_publish_journal(journal: PublishJournal) -> None:
        if (
            journal.phase != "prepared"
            or journal.workspace_id != journal.new_claim.workspace_id
            or journal.slot != journal.new_claim.slot
            or (
                journal.old_claim is not None
                and (
                    journal.old_claim.workspace_id != journal.workspace_id
                    or journal.old_claim.slot != journal.slot
                )
            )
        ):
            raise RegistryError("registry_claim_invalid")
