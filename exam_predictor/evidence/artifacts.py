from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import secrets
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from exam_predictor.evidence.models import EvidenceUnit, StudyMapSnapshot
from exam_predictor.evidence.registry import (
    ArtifactClaim,
    DeleteJournalItem,
    EvidenceArtifactRegistry,
    PublishIntent,
    PublishJournal,
    RegistryError,
    WorkspaceClaim,
)
from exam_predictor.workspace.filesystem import (
    OwnedArtifactFilesystem,
    OwnedDirectoryAnchor,
    OwnedFilesystemError,
    OwnedTemporaryFile,
)


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JSON_TYPES = frozenset({"units", "snapshots"})
_COLLECTIONS = ("parts", "units", "snapshots")
_KIND_BY_COLLECTION = {"parts": "part", "units": "unit", "snapshots": "snapshot"}
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 100_000
_MAX_JSON_STRING_BYTES = 1024 * 1024

Identity = tuple[int, int]


class ArtifactBoundaryError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ArtifactCleanupState(StrEnum):
    DELETED = "deleted"
    CLEANUP_PENDING = "cleanup_pending"


@dataclass(frozen=True)
class _WorkspaceTree:
    claim: WorkspaceClaim
    workspace: OwnedDirectoryAnchor
    evidence: OwnedDirectoryAnchor
    collections: dict[str, OwnedDirectoryAnchor]


class EvidenceArtifactStore:
    """Registry-authoritative evidence artifacts below a trusted existing root."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root).absolute()
        self._filesystem = OwnedArtifactFilesystem()
        self._root_context = self._filesystem.anchor_directory(self._root)
        self._root_anchor: OwnedDirectoryAnchor | None = None
        self._registry: EvidenceArtifactRegistry | None = None
        self._closed = False
        try:
            self._root_anchor = self._root_context.__enter__()
        except OwnedFilesystemError as error:
            code = "artifact_root_missing" if error.code == "owned_root_missing" else "artifact_root_invalid"
            raise ArtifactBoundaryError(code) from None
        try:
            self._registry = EvidenceArtifactRegistry(self._root_anchor, self._filesystem)
            with self._filesystem.create_child_directory(
                self._root_anchor,
                "workspaces",
                expected_parent_identity=self._root_anchor.identity,
            ):
                pass
        except Exception:
            self.close()
            raise ArtifactBoundaryError("artifact_root_invalid") from None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._registry is not None:
            self._registry.close()
            self._registry = None
        if self._root_anchor is not None:
            self._root_context.__exit__(None, None, None)
            self._root_anchor = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def publish_part(
        self,
        workspace_id: str,
        part_id: str,
        content: bytes | bytearray | memoryview,
        *,
        expected_sha256: str,
    ) -> str:
        self._validate_identifier(workspace_id)
        self._validate_identifier(part_id)
        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise ArtifactBoundaryError("artifact_content_invalid")
        try:
            return self._publish_bytes(
                workspace_id,
                "parts",
                part_id,
                bytes(content),
                expected_sha256,
                suffix="",
            )
        except ArtifactBoundaryError:
            raise
        except RegistryError as error:
            self._raise_registry_boundary(error)
        except OwnedFilesystemError as error:
            self._raise_filesystem_boundary(error)
        except Exception:
            raise ArtifactBoundaryError("artifact_publish_failed") from None

    @contextmanager
    def open_part(self, workspace_id: str, part_id: str) -> Iterator[BinaryIO]:
        self._validate_identifier(workspace_id)
        self._validate_identifier(part_id)
        try:
            with self._ready_tree(workspace_id, create=False) as tree:
                claim = self._claim(workspace_id, "parts", part_id)
                collection = tree.collections["parts"]
                with self._filesystem.open_claimed_file(
                    collection,
                    part_id,
                    expected_parent_identity=collection.identity,
                    expected_source_identity=claim.identity,
                    expected_sha256=claim.sha256,
                    expected_size=claim.size,
                ) as opened:
                    with os.fdopen(os.dup(opened.descriptor), "rb", buffering=0) as source:
                        yield source
                    self._filesystem.anchor_identity(collection)
        except ArtifactBoundaryError:
            raise
        except (RegistryError, OwnedFilesystemError):
            raise ArtifactBoundaryError("artifact_identity_changed") from None
        except Exception:
            raise ArtifactBoundaryError("artifact_identity_changed") from None

    def publish_json(
        self,
        workspace_id: str,
        artifact_type: str,
        artifact_id: str,
        document: Any,
        *,
        expected_sha256: str,
    ) -> str:
        self._validate_identifier(workspace_id)
        self._validate_json_type(artifact_type)
        self._validate_identifier(artifact_id)
        encoded = self._encode_json(workspace_id, artifact_type, artifact_id, document)
        try:
            return self._publish_bytes(
                workspace_id,
                artifact_type,
                artifact_id,
                encoded,
                expected_sha256,
                suffix=".json",
            )
        except ArtifactBoundaryError:
            raise
        except RegistryError as error:
            self._raise_registry_boundary(error)
        except OwnedFilesystemError as error:
            self._raise_filesystem_boundary(error)
        except Exception:
            raise ArtifactBoundaryError("artifact_publish_failed") from None

    def read_json(
        self,
        workspace_id: str,
        artifact_type: str,
        artifact_id: str,
    ) -> EvidenceUnit | StudyMapSnapshot:
        self._validate_identifier(workspace_id)
        self._validate_json_type(artifact_type)
        self._validate_identifier(artifact_id)
        filename = f"{artifact_id}.json"
        try:
            with self._ready_tree(workspace_id, create=False) as tree:
                claim = self._claim(workspace_id, artifact_type, filename)
                collection = tree.collections[artifact_type]
                with self._filesystem.open_claimed_file(
                    collection,
                    filename,
                    expected_parent_identity=collection.identity,
                    expected_source_identity=claim.identity,
                    expected_sha256=claim.sha256,
                    expected_size=claim.size,
                ) as opened:
                    if claim.size > _MAX_JSON_BYTES:
                        raise ArtifactBoundaryError("artifact_json_too_large")
                    os.lseek(opened.descriptor, 0, os.SEEK_SET)
                    content = os.read(opened.descriptor, _MAX_JSON_BYTES + 1)
        except ArtifactBoundaryError:
            raise
        except (RegistryError, OwnedFilesystemError):
            raise ArtifactBoundaryError("artifact_identity_changed") from None
        except Exception:
            raise ArtifactBoundaryError("artifact_identity_changed") from None
        return self._decode_json(workspace_id, artifact_type, artifact_id, content)

    def delete_workspace(self, workspace_id: str) -> ArtifactCleanupState:
        self._validate_identifier(workspace_id)
        try:
            registry = self._require_registry()
            claim = registry.get_workspace(workspace_id)
            if claim is None:
                return ArtifactCleanupState.DELETED
            if claim.phase == "reserved":
                if registry.retire_recovered_workspace(workspace_id):
                    return ArtifactCleanupState.DELETED
                return ArtifactCleanupState.CLEANUP_PENDING
            if registry.get_publish_intent(workspace_id) is not None:
                with self._open_tree(
                    workspace_id,
                    allowed_phases={"active"},
                    expected_generation=claim.generation,
                ) as tree:
                    self._recover_publish_intent(tree)
                if registry.get_publish_intent(workspace_id) is not None:
                    return ArtifactCleanupState.CLEANUP_PENDING
            if registry.get_publish_journal(workspace_id) is not None:
                with self._open_tree(
                    workspace_id,
                    allowed_phases={"active"},
                    expected_generation=claim.generation,
                ) as tree:
                    self._recover_publish(tree)
                if registry.get_publish_journal(workspace_id) is not None:
                    return ArtifactCleanupState.CLEANUP_PENDING
            claim = registry.get_workspace(workspace_id)
            if claim is None:
                return ArtifactCleanupState.DELETED
            deletion_generation = claim.generation
            if claim.phase == "active":
                registry.begin_delete(
                    workspace_id,
                    expected_generation=claim.generation,
                )
            elif claim.phase != "deleting":
                return ArtifactCleanupState.CLEANUP_PENDING
            items = registry.get_delete_items(
                workspace_id,
                expected_generation=deletion_generation,
            )
            if any(item.phase != "removed" for item in items):
                with self._open_tree(
                    workspace_id,
                    allowed_phases={"deleting"},
                    expected_generation=deletion_generation,
                ) as tree:
                    if not self._delete_layout_is_known(tree, items):
                        return ArtifactCleanupState.CLEANUP_PENDING
                    self._delete_claimed_items(tree, items)
            items = registry.get_delete_items(
                workspace_id,
                expected_generation=deletion_generation,
            )
            if any(item.phase != "removed" for item in items):
                return ArtifactCleanupState.CLEANUP_PENDING
            self._remove_deleted_tree(
                workspace_id,
                expected_generation=deletion_generation,
            )
            return ArtifactCleanupState.DELETED
        except ArtifactBoundaryError:
            raise
        except RegistryError as error:
            if error.code in {"registry_identity_changed", "registry_claim_invalid"}:
                raise ArtifactBoundaryError("artifact_identity_changed") from None
            return ArtifactCleanupState.CLEANUP_PENDING
        except OwnedFilesystemError as error:
            if error.code == "owned_identity_changed":
                raise ArtifactBoundaryError("artifact_identity_changed") from None
            return ArtifactCleanupState.CLEANUP_PENDING
        except Exception:
            return ArtifactCleanupState.CLEANUP_PENDING

    def _publish_bytes(
        self,
        workspace_id: str,
        artifact_type: str,
        artifact_id: str,
        content: bytes,
        expected_sha256: str,
        *,
        suffix: str,
    ) -> str:
        self._validate_expected_hash(expected_sha256)
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ArtifactBoundaryError("artifact_hash_mismatch")
        filename = f"{artifact_id}{suffix}"
        slot = f"{artifact_type}/{filename}"
        with self._ready_tree(workspace_id, create=True) as tree:
            registry = self._require_registry()
            collection = tree.collections[artifact_type]
            old_claim = registry.get_artifact(workspace_id, slot)
            if old_claim is None:
                if self._filesystem.name_exists(
                    collection,
                    filename,
                    expected_parent_identity=collection.identity,
                ):
                    raise ArtifactBoundaryError("artifact_identity_changed")
            else:
                self._verify_claimed_name(collection, filename, old_claim)
            temporary_name = f".artifact-{secrets.token_hex(16)}.tmp"
            backup_name = f".artifact-{secrets.token_hex(16)}.backup"
            intent = PublishIntent(
                workspace_id=workspace_id,
                slot=slot,
                target_name=filename,
                temporary_name=temporary_name,
                backup_name=backup_name,
                collection=artifact_type,
                kind=_KIND_BY_COLLECTION[artifact_type],
                artifact_id=artifact_id,
                expected_sha256=expected_sha256,
                expected_size=len(content),
                old_claim=old_claim,
                generation=tree.claim.generation,
            )
            registry.reserve_publish(intent)
            try:
                with self._filesystem.create_temporary_file(
                    collection,
                    temporary_name,
                    expected_parent_identity=collection.identity,
                ) as temporary:
                    registry.claim_publish_temporary(
                        workspace_id,
                        temporary.identity,
                        expected_generation=tree.claim.generation,
                        expected_temporary_name=temporary_name,
                    )
                    self._write_descriptor(temporary.descriptor, content)
                    digest, size = self._filesystem.hash_open_file(temporary)
                    if digest != expected_sha256 or size != len(content):
                        raise ArtifactBoundaryError("artifact_hash_mismatch")
                    new_claim = ArtifactClaim(
                        workspace_id=workspace_id,
                        slot=slot,
                        collection=artifact_type,
                        kind=_KIND_BY_COLLECTION[artifact_type],
                        artifact_id=artifact_id,
                        identity=temporary.identity,
                        sha256=digest,
                        size=size,
                        generation=tree.claim.generation,
                    )
                    journal = PublishJournal(
                        workspace_id=workspace_id,
                        slot=slot,
                        phase="prepared",
                        target_name=filename,
                        temporary_name=temporary_name,
                        backup_name=backup_name,
                        new_claim=new_claim,
                        old_claim=old_claim,
                        generation=tree.claim.generation,
                    )
                    registry.prepare_publish(journal)
                    self._install_publish(tree, journal, temporary)
            except BaseException as error:
                if isinstance(error, Exception):
                    with contextlib.suppress(Exception):
                        self._recover_publish_intent(tree)
                    with contextlib.suppress(Exception):
                        self._recover_publish(tree)
                raise
        return expected_sha256

    def _install_publish(
        self,
        tree: _WorkspaceTree,
        journal: PublishJournal,
        temporary: OwnedTemporaryFile,
    ) -> None:
        registry = self._require_registry()
        collection = tree.collections[journal.new_claim.collection]
        old_claim = journal.old_claim
        if old_claim is not None:
            self._filesystem.move_claimed_file(
                collection,
                journal.target_name,
                journal.backup_name,
                expected_parent_identity=collection.identity,
                expected_source_identity=old_claim.identity,
                expected_sha256=old_claim.sha256,
                expected_size=old_claim.size,
                replace_existing=False,
            )
        registry.advance_publish(
            journal.workspace_id,
            expected_generation=journal.generation,
            expected_temporary_name=journal.temporary_name,
            expected_phase="prepared",
            new_phase="backup",
        )
        self._filesystem.replace_open_file(
            collection,
            temporary,
            journal.temporary_name,
            journal.target_name,
            expected_parent_identity=collection.identity,
            expected_source_identity=temporary.identity,
            expected_sha256=journal.new_claim.sha256,
            expected_size=journal.new_claim.size,
            replace_existing=False,
        )
        self._verify_claimed_name(collection, journal.target_name, journal.new_claim)
        registry.advance_publish(
            journal.workspace_id,
            expected_generation=journal.generation,
            expected_temporary_name=journal.temporary_name,
            expected_phase="backup",
            new_phase="installed",
        )
        registry.commit_publish(
            journal.workspace_id,
            journal.new_claim,
            expected_generation=journal.generation,
            expected_temporary_name=journal.temporary_name,
        )
        if old_claim is not None:
            self._delete_if_claimed(collection, journal.backup_name, old_claim)
        registry.clear_publish(
            journal.workspace_id,
            expected_generation=journal.generation,
            expected_temporary_name=journal.temporary_name,
        )

    def _recover_publish(self, tree: _WorkspaceTree) -> None:
        registry = self._require_registry()
        journal = registry.get_publish_journal(tree.claim.workspace_id)
        if journal is None:
            return
        collection = tree.collections[journal.new_claim.collection]
        if journal.phase in {"prepared", "backup"}:
            self._rollback_publish(collection, journal)
            registry.abort_publish(
                journal.workspace_id,
                expected_generation=journal.generation,
                expected_temporary_name=journal.temporary_name,
                expected_phases={"prepared", "backup"},
            )
            return
        if journal.phase == "installed":
            self._require_name_claim(collection, journal.target_name, journal.new_claim)
            self._delete_named_claim_if_present(
                collection,
                journal.temporary_name,
                journal.new_claim,
            )
            self._validate_optional_backup(collection, journal)
            registry.commit_publish(
                journal.workspace_id,
                journal.new_claim,
                expected_generation=journal.generation,
                expected_temporary_name=journal.temporary_name,
            )
            journal = registry.get_publish_journal(journal.workspace_id)
            if journal is None:
                raise RegistryError("registry_state_conflict")
        if journal.phase != "committed":
            raise RegistryError("registry_state_conflict")
        self._require_name_claim(collection, journal.target_name, journal.new_claim)
        committed = registry.get_artifact(journal.workspace_id, journal.slot)
        if committed != journal.new_claim:
            raise RegistryError("registry_state_conflict")
        self._delete_named_claim_if_present(
            collection,
            journal.temporary_name,
            journal.new_claim,
        )
        if journal.old_claim is not None:
            self._delete_named_claim_if_present(
                collection,
                journal.backup_name,
                journal.old_claim,
            )
        elif self._filesystem.name_exists(
            collection,
            journal.backup_name,
            expected_parent_identity=collection.identity,
        ):
            raise ArtifactBoundaryError("artifact_identity_changed")
        registry.clear_publish(
            journal.workspace_id,
            expected_generation=journal.generation,
            expected_temporary_name=journal.temporary_name,
        )

    def _recover_publish_intent(self, tree: _WorkspaceTree) -> None:
        registry = self._require_registry()
        intent = registry.get_publish_intent(tree.claim.workspace_id)
        if intent is None:
            return
        collection = tree.collections[intent.collection]
        temporary_state = self._filesystem.classify_named_identity(
            collection,
            intent.temporary_name,
            expected_parent_identity=collection.identity,
            expected_source_identity=intent.temporary_identity,
        )
        backup_state = self._filesystem.classify_named_identity(
            collection,
            intent.backup_name,
            expected_parent_identity=collection.identity,
            expected_source_identity=intent.temporary_identity,
        )
        if intent.temporary_identity is None:
            if temporary_state != "absent" or backup_state != "absent":
                raise ArtifactBoundaryError("artifact_identity_changed")
            registry.clear_publish_intent(
                intent.workspace_id,
                expected_generation=intent.generation,
                expected_temporary_name=intent.temporary_name,
            )
            return
        if temporary_state == "foreign" or backup_state == "foreign":
            raise ArtifactBoundaryError("artifact_identity_changed")
        if temporary_state == "owned" and backup_state == "owned":
            raise ArtifactBoundaryError("artifact_identity_changed")
        if temporary_state == "owned":
            self._filesystem.delete_reserved_file(
                collection,
                intent.temporary_name,
                expected_parent_identity=collection.identity,
                expected_source_identity=intent.temporary_identity,
                quarantine_name=intent.backup_name,
            )
        elif backup_state == "owned":
            self._filesystem.delete_reserved_file(
                collection,
                intent.backup_name,
                expected_parent_identity=collection.identity,
                expected_source_identity=intent.temporary_identity,
                quarantine_name=intent.backup_name,
            )
        registry.clear_publish_intent(
            intent.workspace_id,
            expected_generation=intent.generation,
            expected_temporary_name=intent.temporary_name,
        )

    def _rollback_publish(
        self,
        collection: OwnedDirectoryAnchor,
        journal: PublishJournal,
    ) -> None:
        target = self._classify_name(
            collection,
            journal.target_name,
            tuple(claim for claim in (journal.old_claim, journal.new_claim) if claim is not None),
        )
        temporary = self._classify_name(
            collection,
            journal.temporary_name,
            (journal.new_claim,),
        )
        backup_claims = () if journal.old_claim is None else (journal.old_claim,)
        backup = self._classify_name(collection, journal.backup_name, backup_claims)
        if journal.old_claim is None:
            if target is journal.new_claim:
                self._delete_if_claimed(collection, journal.target_name, journal.new_claim)
            if temporary is journal.new_claim:
                self._delete_if_claimed(collection, journal.temporary_name, journal.new_claim)
            if backup is not None:
                raise ArtifactBoundaryError("artifact_identity_changed")
            return
        old_claim = journal.old_claim
        if target is journal.new_claim:
            self._delete_if_claimed(collection, journal.target_name, journal.new_claim)
            target = None
        if temporary is journal.new_claim:
            self._delete_if_claimed(collection, journal.temporary_name, journal.new_claim)
        if target is old_claim:
            if backup is not None:
                raise ArtifactBoundaryError("artifact_identity_changed")
            return
        if target is not None or backup is not old_claim:
            raise ArtifactBoundaryError("artifact_operation_pending")
        self._filesystem.move_claimed_file(
            collection,
            journal.backup_name,
            journal.target_name,
            expected_parent_identity=collection.identity,
            expected_source_identity=old_claim.identity,
            expected_sha256=old_claim.sha256,
            expected_size=old_claim.size,
            replace_existing=False,
        )

    def _validate_optional_backup(
        self,
        collection: OwnedDirectoryAnchor,
        journal: PublishJournal,
    ) -> None:
        if journal.old_claim is None:
            if self._filesystem.name_exists(
                collection,
                journal.backup_name,
                expected_parent_identity=collection.identity,
            ):
                raise ArtifactBoundaryError("artifact_identity_changed")
            return
        self._require_name_claim(collection, journal.backup_name, journal.old_claim)

    @contextmanager
    def _ready_tree(self, workspace_id: str, *, create: bool) -> Iterator[_WorkspaceTree]:
        registry = self._require_registry()
        claim = registry.get_workspace(workspace_id)
        if claim is not None and claim.phase == "reserved":
            if registry.retire_recovered_workspace(workspace_id):
                claim = None
            else:
                raise ArtifactBoundaryError("artifact_operation_pending")
        if claim is None and create:
            self._bootstrap_workspace(workspace_id)
            claim = registry.get_workspace(workspace_id)
        if claim is None:
            raise ArtifactBoundaryError("artifact_identity_changed")
        if claim.phase != "active":
            raise ArtifactBoundaryError("artifact_operation_pending")
        if (
            registry.get_publish_intent(workspace_id) is not None
            or registry.get_publish_journal(workspace_id) is not None
        ):
            with self._open_tree(
                workspace_id,
                allowed_phases={"active"},
                expected_generation=claim.generation,
            ) as recovery_tree:
                self._recover_publish_intent(recovery_tree)
                self._recover_publish(recovery_tree)
            if (
                registry.get_publish_intent(workspace_id) is not None
                or registry.get_publish_journal(workspace_id) is not None
            ):
                raise ArtifactBoundaryError("artifact_operation_pending")
        with self._open_tree(
            workspace_id,
            allowed_phases={"active"},
            expected_generation=claim.generation,
        ) as tree:
            yield tree

    def _bootstrap_workspace(self, workspace_id: str) -> None:
        registry = self._require_registry()
        generation = secrets.token_hex(16)
        if not registry.reserve_workspace(workspace_id, generation=generation):
            claim = registry.get_workspace(workspace_id)
            if claim is None or claim.phase != "active":
                raise ArtifactBoundaryError("artifact_operation_pending")
            return
        try:
            root = self._require_root()
            with self._filesystem.anchor_child_directory(
                root,
                "workspaces",
                expected_parent_identity=root.identity,
            ) as workspaces:
                with self._filesystem.create_child_directory(
                    workspaces,
                    workspace_id,
                    expected_parent_identity=workspaces.identity,
                ) as workspace:
                    if self._filesystem.name_exists(
                        workspace,
                        "evidence",
                        expected_parent_identity=workspace.identity,
                    ):
                        raise ArtifactBoundaryError("artifact_identity_changed")
                    with self._filesystem.create_new_child_directory(
                        workspace,
                        "evidence",
                        expected_parent_identity=workspace.identity,
                    ) as evidence:
                        identities: dict[str, Identity] = {}
                        with ExitStack() as stack:
                            for name in _COLLECTIONS:
                                child = stack.enter_context(
                                    self._filesystem.create_new_child_directory(
                                        evidence,
                                        name,
                                        expected_parent_identity=evidence.identity,
                                    )
                                )
                                identities[name] = child.identity
                            registry.finalize_workspace(
                                workspace_id,
                                expected_generation=generation,
                                workspace_identity=workspace.identity,
                                evidence_identity=evidence.identity,
                                collection_identities=identities,
                            )
        except Exception:
            with contextlib.suppress(RegistryError):
                registry.retire_reserved_workspace(
                    workspace_id,
                    expected_generation=generation,
                )
            raise

    @contextmanager
    def _open_tree(
        self,
        workspace_id: str,
        *,
        allowed_phases: set[str],
        expected_generation: str,
    ) -> Iterator[_WorkspaceTree]:
        registry = self._require_registry()
        claim = registry.get_workspace(workspace_id)
        if (
            claim is None
            or claim.phase not in allowed_phases
            or claim.generation != expected_generation
            or claim.workspace_identity is None
            or claim.evidence_identity is None
        ):
            raise RegistryError("registry_state_conflict")
        collection_identities = registry.get_collection_identities(workspace_id)
        if set(collection_identities) != set(_COLLECTIONS):
            raise RegistryError("registry_claim_invalid")
        root = self._require_root()
        with ExitStack() as stack:
            workspaces = stack.enter_context(
                self._filesystem.anchor_child_directory(
                    root,
                    "workspaces",
                    expected_parent_identity=root.identity,
                )
            )
            workspace = stack.enter_context(
                self._filesystem.anchor_child_directory(
                    workspaces,
                    workspace_id,
                    expected_parent_identity=workspaces.identity,
                )
            )
            if workspace.identity != claim.workspace_identity:
                raise OwnedFilesystemError("owned_identity_changed")
            evidence = stack.enter_context(
                self._filesystem.anchor_child_directory(
                    workspace,
                    "evidence",
                    expected_parent_identity=workspace.identity,
                )
            )
            if evidence.identity != claim.evidence_identity:
                raise OwnedFilesystemError("owned_identity_changed")
            collections: dict[str, OwnedDirectoryAnchor] = {}
            for name in _COLLECTIONS:
                collection = stack.enter_context(
                    self._filesystem.anchor_child_directory(
                        evidence,
                        name,
                        expected_parent_identity=evidence.identity,
                    )
                )
                if collection.identity != collection_identities[name]:
                    raise OwnedFilesystemError("owned_identity_changed")
                collections[name] = collection
            yield _WorkspaceTree(claim, workspace, evidence, collections)

    def _delete_layout_is_known(
        self,
        tree: _WorkspaceTree,
        items: tuple[DeleteJournalItem, ...],
    ) -> bool:
        if set(self._filesystem.list_names(tree.evidence)) != set(_COLLECTIONS):
            return False
        by_collection: dict[str, set[str]] = {name: set() for name in _COLLECTIONS}
        for item in items:
            filename = item.slot.split("/", 1)[1]
            collection = tree.collections[item.claim.collection]
            if item.phase == "planned":
                by_collection[item.claim.collection].add(filename)
                if self._filesystem.name_exists(
                    collection,
                    filename,
                    expected_parent_identity=collection.identity,
                ):
                    self._classify_name(collection, filename, (item.claim,))
                if item.quarantine_name is not None:
                    by_collection[item.claim.collection].add(item.quarantine_name)
                    if self._filesystem.name_exists(
                        collection,
                        item.quarantine_name,
                        expected_parent_identity=collection.identity,
                    ):
                        self._classify_name(collection, item.quarantine_name, (item.claim,))
            elif item.phase == "quarantined" and item.quarantine_name is not None:
                by_collection[item.claim.collection].add(item.quarantine_name)
                if self._filesystem.name_exists(
                    collection,
                    item.quarantine_name,
                    expected_parent_identity=collection.identity,
                ):
                    self._classify_name(collection, item.quarantine_name, (item.claim,))
        return all(
            set(self._filesystem.list_names(tree.collections[name])) <= expected
            for name, expected in by_collection.items()
        )

    def _delete_claimed_items(
        self,
        tree: _WorkspaceTree,
        items: tuple[DeleteJournalItem, ...],
    ) -> None:
        registry = self._require_registry()
        for original in items:
            item = original
            collection = tree.collections[item.claim.collection]
            filename = item.slot.split("/", 1)[1]
            if item.phase == "planned":
                quarantine = item.quarantine_name
                if quarantine is None:
                    quarantine = f".artifact-delete-{secrets.token_hex(16)}.quarantine"
                    registry.plan_delete_quarantine(
                        item.workspace_id,
                        item.slot,
                        quarantine,
                        expected_generation=item.generation,
                    )
                    item = next(
                        candidate
                        for candidate in registry.get_delete_items(
                            item.workspace_id,
                            expected_generation=item.generation,
                        )
                        if candidate.slot == item.slot
                    )
                source = self._classify_name(collection, filename, (item.claim,))
                quarantined = self._classify_name(collection, quarantine, (item.claim,))
                if source is item.claim and quarantined is None:
                    self._filesystem.move_claimed_file(
                        collection,
                        filename,
                        quarantine,
                        expected_parent_identity=collection.identity,
                        expected_source_identity=item.claim.identity,
                        expected_sha256=item.claim.sha256,
                        expected_size=item.claim.size,
                        replace_existing=False,
                    )
                elif source is not None or quarantined is not item.claim:
                    raise ArtifactBoundaryError("artifact_identity_changed")
                if not self._delete_layout_is_known(
                    tree,
                    registry.get_delete_items(
                        item.workspace_id,
                        expected_generation=item.generation,
                    ),
                ):
                    raise RegistryError("registry_pending")
                registry.advance_delete_item(
                    item.workspace_id,
                    item.slot,
                    expected_generation=item.generation,
                    expected_phase="planned",
                    new_phase="quarantined",
                    quarantine_name=quarantine,
                )
                item = next(
                    candidate
                    for candidate in registry.get_delete_items(
                        item.workspace_id,
                        expected_generation=item.generation,
                    )
                    if candidate.slot == item.slot
                )
            if item.phase == "quarantined":
                quarantine = item.quarantine_name
                if quarantine is None:
                    raise RegistryError("registry_claim_invalid")
                if not self._delete_layout_is_known(
                    tree,
                    registry.get_delete_items(
                        item.workspace_id,
                        expected_generation=item.generation,
                    ),
                ):
                    raise RegistryError("registry_pending")
                if self._filesystem.name_exists(
                    collection,
                    filename,
                    expected_parent_identity=collection.identity,
                ):
                    raise ArtifactBoundaryError("artifact_identity_changed")
                if self._filesystem.name_exists(
                    collection,
                    quarantine,
                    expected_parent_identity=collection.identity,
                ):
                    self._delete_if_claimed(collection, quarantine, item.claim)
                registry.advance_delete_item(
                    item.workspace_id,
                    item.slot,
                    expected_generation=item.generation,
                    expected_phase="quarantined",
                    new_phase="removed",
                    quarantine_name=quarantine,
                )

    def _remove_deleted_tree(
        self,
        workspace_id: str,
        *,
        expected_generation: str,
    ) -> None:
        registry = self._require_registry()
        claim = registry.get_workspace(workspace_id)
        if claim is None:
            return
        if (
            claim.generation != expected_generation
            or claim.workspace_identity is None
            or claim.evidence_identity is None
        ):
            raise RegistryError("registry_claim_invalid")
        collection_identities = registry.get_collection_identities(workspace_id)
        root = self._require_root()
        with self._filesystem.anchor_child_directory(
            root,
            "workspaces",
            expected_parent_identity=root.identity,
        ) as workspaces:
            with self._filesystem.anchor_child_directory(
                workspaces,
                workspace_id,
                expected_parent_identity=workspaces.identity,
            ) as workspace:
                if workspace.identity != claim.workspace_identity:
                    raise OwnedFilesystemError("owned_identity_changed")
                evidence_is_visible = self._filesystem.name_exists(
                    workspace,
                    "evidence",
                    expected_parent_identity=workspace.identity,
                )
                if evidence_is_visible:
                    with self._filesystem.anchor_child_directory(
                        workspace,
                        "evidence",
                        expected_parent_identity=workspace.identity,
                    ) as evidence:
                        self._verify_empty_evidence_tree(
                            evidence,
                            claim.evidence_identity,
                            collection_identities,
                        )
                quarantine = self._filesystem.quarantine_directory_tree(
                    workspace,
                    "evidence",
                    expected_parent_identity=workspace.identity,
                    expected_child_identity=claim.evidence_identity,
                )
                if quarantine is None:
                    if self._filesystem.directory_names_with_identity(
                        workspace,
                        expected_parent_identity=workspace.identity,
                        expected_child_identity=claim.evidence_identity,
                    ):
                        raise OwnedFilesystemError("owned_identity_changed")
                    raise RegistryError("registry_pending")
                with self._filesystem.anchor_child_directory(
                    workspace,
                    quarantine,
                    expected_parent_identity=workspace.identity,
                ) as evidence:
                    self._verify_empty_evidence_tree(
                        evidence,
                        claim.evidence_identity,
                        collection_identities,
                    )
                owned_names = self._filesystem.directory_names_with_identity(
                    workspace,
                    expected_parent_identity=workspace.identity,
                    expected_child_identity=claim.evidence_identity,
                )
                if owned_names != (quarantine,):
                    raise OwnedFilesystemError("owned_identity_changed")
        registry.clear_deleted_workspace(
            workspace_id,
            expected_generation=claim.generation,
        )

    def _verify_empty_evidence_tree(
        self,
        evidence: OwnedDirectoryAnchor,
        expected_evidence_identity: tuple[int, int],
        collection_identities: dict[str, tuple[int, int]],
    ) -> None:
        if (
            evidence.identity != expected_evidence_identity
            or set(self._filesystem.list_names(evidence)) != set(_COLLECTIONS)
        ):
            raise OwnedFilesystemError("owned_identity_changed")
        for name in _COLLECTIONS:
            with self._filesystem.anchor_child_directory(
                evidence,
                name,
                expected_parent_identity=evidence.identity,
            ) as collection:
                if (
                    collection.identity != collection_identities[name]
                    or self._filesystem.list_names(collection)
                ):
                    raise OwnedFilesystemError("owned_identity_changed")

    def _claim(self, workspace_id: str, collection: str, filename: str) -> ArtifactClaim:
        claim = self._require_registry().get_artifact(
            workspace_id,
            f"{collection}/{filename}",
        )
        if claim is None:
            raise ArtifactBoundaryError("artifact_identity_changed")
        return claim

    def _verify_claimed_name(
        self,
        collection: OwnedDirectoryAnchor,
        name: str,
        claim: ArtifactClaim,
    ) -> None:
        try:
            with self._filesystem.open_claimed_file(
                collection,
                name,
                expected_parent_identity=collection.identity,
                expected_source_identity=claim.identity,
                expected_sha256=claim.sha256,
                expected_size=claim.size,
            ):
                pass
        except OwnedFilesystemError as error:
            if error.code == "owned_content_changed":
                raise ArtifactBoundaryError("artifact_hash_mismatch") from None
            raise ArtifactBoundaryError("artifact_identity_changed") from None

    def _classify_name(
        self,
        collection: OwnedDirectoryAnchor,
        name: str,
        claims: tuple[ArtifactClaim, ...],
    ) -> ArtifactClaim | None:
        if not self._filesystem.name_exists(
            collection,
            name,
            expected_parent_identity=collection.identity,
        ):
            return None
        for claim in claims:
            if self._filesystem.name_has_identity(
                collection,
                name,
                expected_parent_identity=collection.identity,
                expected_source_identity=claim.identity,
            ):
                self._verify_claimed_name(collection, name, claim)
                return claim
        raise ArtifactBoundaryError("artifact_identity_changed")

    def _require_name_claim(
        self,
        collection: OwnedDirectoryAnchor,
        name: str,
        claim: ArtifactClaim,
    ) -> None:
        if self._classify_name(collection, name, (claim,)) is not claim:
            raise ArtifactBoundaryError("artifact_operation_pending")

    def _delete_named_claim_if_present(
        self,
        collection: OwnedDirectoryAnchor,
        name: str,
        claim: ArtifactClaim,
    ) -> None:
        classified = self._classify_name(collection, name, (claim,))
        if classified is claim:
            self._delete_if_claimed(collection, name, claim)

    def _delete_if_claimed(
        self,
        collection: OwnedDirectoryAnchor,
        name: str,
        claim: ArtifactClaim,
    ) -> None:
        self._filesystem.delete_claimed_file(
            collection,
            name,
            expected_parent_identity=collection.identity,
            expected_source_identity=claim.identity,
            expected_sha256=claim.sha256,
            expected_size=claim.size,
            quarantine_name=name,
        )

    def _encode_json(
        self,
        workspace_id: str,
        artifact_type: str,
        artifact_id: str,
        document: Any,
    ) -> bytes:
        try:
            model_type = EvidenceUnit if artifact_type == "units" else StudyMapSnapshot
            if type(document) is model_type:
                model = model_type.model_validate(document.model_dump(mode="python"))
            elif type(document) is dict:
                self._validate_json_object(document)
                model = model_type.model_validate(document)
            else:
                raise ValueError("document type")
            if artifact_type == "units" and model.evidence_unit_id != artifact_id:
                raise ValueError("identifier mismatch")
            if artifact_type == "snapshots" and (
                model.snapshot_id != artifact_id or model.workspace_id != workspace_id
            ):
                raise ValueError("identifier mismatch")
            canonical = model.model_dump(mode="json")
            if type(canonical) is not dict:
                raise ValueError("model serialization")
            self._validate_json_object(canonical)
            return self._canonical_bytes(canonical, _MAX_JSON_BYTES)
        except ArtifactBoundaryError:
            raise
        except Exception:
            raise ArtifactBoundaryError("artifact_json_invalid") from None

    def _decode_json(
        self,
        workspace_id: str,
        artifact_type: str,
        artifact_id: str,
        content: bytes,
    ) -> EvidenceUnit | StudyMapSnapshot:
        try:
            if len(content) > _MAX_JSON_BYTES:
                raise ArtifactBoundaryError("artifact_json_too_large")
            document = json.loads(content.decode("utf-8"))
            if type(document) is not dict:
                raise ValueError("document type")
            self._validate_json_object(document)
            model_type = EvidenceUnit if artifact_type == "units" else StudyMapSnapshot
            model = model_type.model_validate(document)
            if artifact_type == "units" and model.evidence_unit_id != artifact_id:
                raise ValueError("identifier mismatch")
            if artifact_type == "snapshots" and (
                model.snapshot_id != artifact_id or model.workspace_id != workspace_id
            ):
                raise ValueError("identifier mismatch")
            return model
        except ArtifactBoundaryError:
            raise
        except Exception:
            raise ArtifactBoundaryError("artifact_json_invalid") from None

    @staticmethod
    def _canonical_bytes(document: Any, maximum: int) -> bytes:
        try:
            encoder = json.JSONEncoder(
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            encoded = bytearray()
            for chunk in encoder.iterencode(document):
                piece = chunk.encode("utf-8")
                if len(encoded) + len(piece) > maximum:
                    raise ArtifactBoundaryError("artifact_json_too_large")
                encoded.extend(piece)
            return bytes(encoded)
        except ArtifactBoundaryError:
            raise
        except Exception:
            raise ArtifactBoundaryError("artifact_json_invalid") from None

    @staticmethod
    def _validate_json_object(document: Any) -> None:
        if type(document) is not dict:
            raise ArtifactBoundaryError("artifact_json_invalid")
        stack: list[tuple[Any, int]] = [(document, 0)]
        seen: set[int] = set()
        nodes = 0
        while stack:
            value, depth = stack.pop()
            nodes += 1
            if nodes > _MAX_JSON_NODES:
                raise ArtifactBoundaryError("artifact_json_too_large")
            if depth > _MAX_JSON_DEPTH:
                raise ArtifactBoundaryError("artifact_json_too_deep")
            if type(value) is dict:
                identity = id(value)
                if identity in seen:
                    raise ArtifactBoundaryError("artifact_json_invalid")
                seen.add(identity)
                for key, child in value.items():
                    if type(key) is not str:
                        raise ArtifactBoundaryError("artifact_json_invalid")
                    if len(key.encode("utf-8")) > _MAX_JSON_STRING_BYTES:
                        raise ArtifactBoundaryError("artifact_json_too_large")
                    stack.append((child, depth + 1))
                continue
            if type(value) in {list, tuple}:
                identity = id(value)
                if identity in seen:
                    raise ArtifactBoundaryError("artifact_json_invalid")
                seen.add(identity)
                stack.extend((child, depth + 1) for child in value)
                continue
            if type(value) is str:
                if len(value.encode("utf-8")) > _MAX_JSON_STRING_BYTES:
                    raise ArtifactBoundaryError("artifact_json_too_large")
                continue
            if value is None or type(value) in {bool, int}:
                continue
            if type(value) is float and math.isfinite(value):
                continue
            raise ArtifactBoundaryError("artifact_json_invalid")

    @staticmethod
    def _write_descriptor(descriptor: int, content: bytes) -> None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short artifact write")
            view = view[written:]
        os.ftruncate(descriptor, len(content))
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)

    def _require_root(self) -> OwnedDirectoryAnchor:
        if self._root_anchor is None:
            raise ArtifactBoundaryError("artifact_store_closed")
        return self._root_anchor

    def _require_registry(self) -> EvidenceArtifactRegistry:
        if self._registry is None:
            raise ArtifactBoundaryError("artifact_store_closed")
        return self._registry

    @staticmethod
    def _validate_identifier(value: str) -> None:
        if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
            raise ArtifactBoundaryError("artifact_identifier_invalid")

    @staticmethod
    def _validate_json_type(artifact_type: str) -> None:
        if artifact_type not in _JSON_TYPES:
            raise ArtifactBoundaryError("artifact_type_invalid")

    @staticmethod
    def _validate_expected_hash(expected_sha256: str) -> None:
        if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
            raise ArtifactBoundaryError("artifact_hash_invalid")

    @staticmethod
    def _raise_registry_boundary(error: RegistryError) -> None:
        if error.code in {"registry_identity_changed", "registry_claim_invalid"}:
            raise ArtifactBoundaryError("artifact_identity_changed") from None
        if error.code in {"registry_pending", "registry_state_conflict"}:
            raise ArtifactBoundaryError("artifact_operation_pending") from None
        raise ArtifactBoundaryError("artifact_publish_failed") from None

    @staticmethod
    def _raise_filesystem_boundary(error: OwnedFilesystemError) -> None:
        if error.code == "owned_content_changed":
            raise ArtifactBoundaryError("artifact_hash_mismatch") from None
        if error.code in {"owned_identity_changed", "owned_destination_exists"}:
            raise ArtifactBoundaryError("artifact_identity_changed") from None
        raise ArtifactBoundaryError("artifact_publish_failed") from None
