from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import secrets
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from exam_predictor.evidence.models import EvidenceUnit, StudyMapSnapshot
from exam_predictor.workspace.filesystem import (
    OwnedArtifactFilesystem,
    OwnedDirectoryAnchor,
    OwnedFilesystemError,
    OwnedOpenFile,
)


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JSON_TYPES = frozenset({"units", "snapshots"})
_COLLECTIONS = ("parts", "units", "snapshots")
_KIND_BY_COLLECTION = {"parts": "part", "units": "unit", "snapshots": "snapshot"}
_OWNERSHIP_NAME = ".artifact-ownership.json"
_OWNERSHIP_VERSION = 2
_MAX_MARKER_BYTES = 1024 * 1024
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 100_000
_MAX_JSON_STRING_BYTES = 1024 * 1024

Identity = tuple[int, int]
Claim = dict[str, Any]


class ArtifactBoundaryError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ArtifactCleanupState(StrEnum):
    DELETED = "deleted"
    CLEANUP_PENDING = "cleanup_pending"


# Backward-compatible public name; all mutation mechanics now live in the shared adapter.
NativeArtifactFilesystemOps = OwnedArtifactFilesystem
ArtifactFilesystemOps = OwnedArtifactFilesystem


class EvidenceArtifactStore:
    """Marker-authoritative evidence artifacts below a trusted existing data root."""

    def __init__(
        self,
        root: Path,
        *,
        filesystem_ops: OwnedArtifactFilesystem | None = None,
    ) -> None:
        self._root = Path(root).absolute()
        self._filesystem = filesystem_ops or OwnedArtifactFilesystem()
        self._root_context = self._filesystem.anchor_directory(self._root)
        try:
            self._root_anchor = self._root_context.__enter__()
        except OwnedFilesystemError as error:
            code = "artifact_root_missing" if error.code == "owned_root_missing" else "artifact_root_invalid"
            raise ArtifactBoundaryError(code) from None
        self._closed = False
        try:
            with self._filesystem.create_child_directory(
                self._root_anchor,
                "workspaces",
                expected_parent_identity=self._root_anchor.identity,
            ):
                pass
        except OwnedFilesystemError:
            self.close()
            raise ArtifactBoundaryError("artifact_root_invalid") from None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._root_context.__exit__(None, None, None)

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
        except OwnedFilesystemError as error:
            self._raise_filesystem_boundary(error)
        except OSError:
            raise ArtifactBoundaryError("artifact_publish_failed") from None

    @contextmanager
    def open_part(self, workspace_id: str, part_id: str) -> Iterator[BinaryIO]:
        self._validate_identifier(workspace_id)
        self._validate_identifier(part_id)
        with self._collection(workspace_id, "parts", create=False) as (
            workspace,
            evidence,
            collection,
            ownership,
        ):
            del workspace, evidence
            claim = self._claim(ownership, "parts", part_id)
            try:
                with self._filesystem.open_claimed_file(
                    collection,
                    part_id,
                    expected_parent_identity=collection.identity,
                    expected_source_identity=self._claim_identity(claim),
                    expected_sha256=claim["sha256"],
                    expected_size=claim["size"],
                ) as opened:
                    with os.fdopen(os.dup(opened.descriptor), "rb", buffering=0) as source:
                        yield source
                    self._filesystem.anchor_identity(collection)
            except OwnedFilesystemError:
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
        except OwnedFilesystemError as error:
            self._raise_filesystem_boundary(error)
        except OSError:
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
        with self._collection(workspace_id, artifact_type, create=False) as (
            workspace,
            evidence,
            collection,
            ownership,
        ):
            del workspace, evidence
            claim = self._claim(ownership, artifact_type, filename)
            try:
                with self._filesystem.open_claimed_file(
                    collection,
                    filename,
                    expected_parent_identity=collection.identity,
                    expected_source_identity=self._claim_identity(claim),
                    expected_sha256=claim["sha256"],
                    expected_size=claim["size"],
                ) as opened:
                    if claim["size"] > _MAX_JSON_BYTES:
                        raise ArtifactBoundaryError("artifact_json_too_large")
                    os.lseek(opened.descriptor, 0, os.SEEK_SET)
                    content = os.read(opened.descriptor, _MAX_JSON_BYTES + 1)
            except ArtifactBoundaryError:
                raise
            except OwnedFilesystemError:
                raise ArtifactBoundaryError("artifact_identity_changed") from None
        return self._decode_json(workspace_id, artifact_type, artifact_id, content)

    def delete_workspace(self, workspace_id: str) -> ArtifactCleanupState:
        self._validate_identifier(workspace_id)
        collection_identities: dict[str, Identity] = {}
        evidence_identity: Identity | None = None
        try:
            with self._evidence(workspace_id, create=False) as (
                workspace,
                evidence,
                ownership,
                marker_identity,
                marker_bytes,
            ):
                evidence_identity = evidence.identity
                root_names = set(self._filesystem.list_names(evidence))
                if not root_names <= {_OWNERSHIP_NAME, *_COLLECTIONS}:
                    return ArtifactCleanupState.CLEANUP_PENDING
                claims = ownership["artifacts"]
                opened_collections: dict[str, OwnedDirectoryAnchor] = {}
                unknown_entry = False
                with contextlib.ExitStack() as stack:
                    for collection_name in _COLLECTIONS:
                        expected_names = {
                            slot.split("/", 1)[1] for slot in claims if slot.startswith(f"{collection_name}/")
                        }
                        if collection_name not in root_names:
                            if expected_names:
                                raise ArtifactBoundaryError("artifact_identity_changed")
                            continue
                        collection = stack.enter_context(
                            self._filesystem.anchor_child_directory(
                                evidence,
                                collection_name,
                                expected_parent_identity=evidence.identity,
                            )
                        )
                        opened_collections[collection_name] = collection
                        collection_identities[collection_name] = collection.identity
                        if set(self._filesystem.list_names(collection)) != expected_names:
                            unknown_entry = True
                    for slot, claim in sorted(claims.items()):
                        collection_name, filename = slot.split("/", 1)
                        self._verify_claimed_name(opened_collections[collection_name], filename, claim)
                    if unknown_entry:
                        return ArtifactCleanupState.CLEANUP_PENDING
                    for slot, claim in sorted(claims.items()):
                        collection_name, filename = slot.split("/", 1)
                        collection = opened_collections[collection_name]
                        self._filesystem.delete_claimed_file(
                            collection,
                            filename,
                            expected_parent_identity=collection.identity,
                            expected_source_identity=self._claim_identity(claim),
                            expected_sha256=claim["sha256"],
                            expected_size=claim["size"],
                        )
                for collection_name, identity in collection_identities.items():
                    self._filesystem.remove_empty_directory(
                        evidence,
                        collection_name,
                        expected_parent_identity=evidence.identity,
                        expected_child_identity=identity,
                    )
                self._filesystem.delete_claimed_file(
                    evidence,
                    _OWNERSHIP_NAME,
                    expected_parent_identity=evidence.identity,
                    expected_source_identity=marker_identity,
                    expected_sha256=hashlib.sha256(marker_bytes).hexdigest(),
                    expected_size=len(marker_bytes),
                )
            if evidence_identity is None:
                return ArtifactCleanupState.DELETED
            with self._workspace(workspace_id, create=False) as workspace:
                self._filesystem.remove_empty_directory(
                    workspace,
                    "evidence",
                    expected_parent_identity=workspace.identity,
                    expected_child_identity=evidence_identity,
                )
            return ArtifactCleanupState.DELETED
        except ArtifactBoundaryError:
            raise
        except OwnedFilesystemError as error:
            if error.code in {"owned_not_found", "owned_root_missing"}:
                return ArtifactCleanupState.DELETED
            if error.code == "owned_identity_changed":
                raise ArtifactBoundaryError("artifact_identity_changed") from None
            return ArtifactCleanupState.CLEANUP_PENDING
        except OSError:
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
        with self._collection(workspace_id, artifact_type, create=True) as (
            workspace,
            evidence,
            collection,
            ownership,
        ):
            slot = f"{artifact_type}/{filename}"
            old_claim = ownership["artifacts"].get(slot)
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
            with self._filesystem.create_temporary_file(
                collection,
                temporary_name,
                expected_parent_identity=collection.identity,
            ) as temporary:
                self._write_descriptor(temporary.descriptor, content)
                digest, size = self._filesystem.hash_open_file(temporary)
                if digest != expected_sha256 or size != len(content):
                    raise ArtifactBoundaryError("artifact_hash_mismatch")
                new_claim = self._make_claim(
                    slot,
                    artifact_type,
                    artifact_id,
                    temporary.identity,
                    digest,
                    size,
                )
                pending = {
                    "phase": "prepared",
                    "slot": slot,
                    "target": filename,
                    "temporary": temporary_name,
                    "backup": backup_name,
                    "new": new_claim,
                    "old": old_claim,
                }
                self._write_marker(workspace, evidence, ownership["artifacts"], pending)
                try:
                    if old_claim is not None:
                        self._filesystem.move_claimed_file(
                            collection,
                            filename,
                            backup_name,
                            expected_parent_identity=collection.identity,
                            expected_source_identity=self._claim_identity(old_claim),
                            expected_sha256=old_claim["sha256"],
                            expected_size=old_claim["size"],
                            replace_existing=False,
                        )
                    self._filesystem.replace_open_file(
                        collection,
                        temporary,
                        temporary_name,
                        filename,
                        expected_parent_identity=collection.identity,
                        expected_source_identity=temporary.identity,
                        expected_sha256=digest,
                        expected_size=size,
                        replace_existing=True,
                    )
                    self._verify_claimed_name(collection, filename, new_claim)
                    new_artifacts = dict(ownership["artifacts"])
                    new_artifacts[slot] = new_claim
                    committed = dict(pending)
                    committed["phase"] = "committed"
                    self._write_marker(workspace, evidence, new_artifacts, committed)
                except BaseException as error:
                    if isinstance(error, Exception):
                        self._rollback_pending(
                            workspace,
                            evidence,
                            collection,
                            ownership["artifacts"],
                            pending,
                            temporary=temporary,
                        )
                    raise
                if old_claim is not None:
                    self._delete_if_claimed(collection, backup_name, old_claim)
                self._write_marker(workspace, evidence, new_artifacts, None)
        return expected_sha256

    def _rollback_pending(
        self,
        workspace: OwnedDirectoryAnchor,
        evidence: OwnedDirectoryAnchor,
        collection: OwnedDirectoryAnchor,
        old_artifacts: dict[str, Claim],
        pending: dict[str, Any],
        *,
        temporary: OwnedOpenFile | None = None,
    ) -> None:
        new_claim = pending["new"]
        old_claim = pending["old"]
        target_exists = self._filesystem.name_exists(
            collection,
            pending["target"],
            expected_parent_identity=collection.identity,
        )
        if target_exists:
            if temporary is not None and self._filesystem.name_has_identity(
                collection,
                pending["target"],
                expected_parent_identity=collection.identity,
                expected_source_identity=temporary.identity,
            ):
                self._filesystem.delete_open_file(
                    collection,
                    temporary,
                    pending["target"],
                    expected_parent_identity=collection.identity,
                    expected_source_identity=temporary.identity,
                )
            elif self._matches_claim(collection, pending["target"], new_claim):
                if temporary is not None:
                    self._filesystem.delete_open_file(
                        collection,
                        temporary,
                        pending["target"],
                        expected_parent_identity=collection.identity,
                        expected_source_identity=temporary.identity,
                    )
                else:
                    self._delete_if_claimed(collection, pending["target"], new_claim)
            elif old_claim is None or not self._matches_claim(collection, pending["target"], old_claim):
                raise ArtifactBoundaryError("artifact_identity_changed")
        if old_claim is not None and self._is_claimed_name(collection, pending["backup"], old_claim):
            if self._filesystem.name_exists(
                collection,
                pending["target"],
                expected_parent_identity=collection.identity,
            ):
                raise ArtifactBoundaryError("artifact_identity_changed")
            self._filesystem.move_claimed_file(
                collection,
                pending["backup"],
                pending["target"],
                expected_parent_identity=collection.identity,
                expected_source_identity=self._claim_identity(old_claim),
                expected_sha256=old_claim["sha256"],
                expected_size=old_claim["size"],
                replace_existing=False,
            )
        if temporary is not None and self._filesystem.name_exists(
            collection,
            pending["temporary"],
            expected_parent_identity=collection.identity,
        ):
            self._filesystem.delete_open_file(
                collection,
                temporary,
                pending["temporary"],
                expected_parent_identity=collection.identity,
                expected_source_identity=temporary.identity,
            )
        else:
            self._delete_if_claimed(collection, pending["temporary"], new_claim)
        self._write_marker(workspace, evidence, old_artifacts, None)

    def _recover_pending(
        self,
        workspace: OwnedDirectoryAnchor,
        evidence: OwnedDirectoryAnchor,
        ownership: dict[str, Any],
    ) -> dict[str, Any]:
        pending = ownership["pending"]
        if pending is None:
            return ownership
        collection_name = pending["slot"].split("/", 1)[0]
        with self._filesystem.anchor_child_directory(
            evidence,
            collection_name,
            expected_parent_identity=evidence.identity,
        ) as collection:
            if pending["phase"] == "committed":
                if not self._is_claimed_name(collection, pending["target"], pending["new"]):
                    raise ArtifactBoundaryError("artifact_identity_changed")
                if pending["old"] is not None:
                    self._delete_if_claimed(collection, pending["backup"], pending["old"])
                self._delete_if_claimed(collection, pending["temporary"], pending["new"])
                return self._write_marker(workspace, evidence, ownership["artifacts"], None)
            old_artifacts = dict(ownership["artifacts"])
            self._rollback_pending(workspace, evidence, collection, old_artifacts, pending)
            return self._load_marker(workspace, evidence, recover=False)[0]

    @contextmanager
    def _workspace(self, workspace_id: str, *, create: bool) -> Iterator[OwnedDirectoryAnchor]:
        with contextlib.ExitStack() as stack:
            workspaces = stack.enter_context(
                self._filesystem.anchor_child_directory(
                    self._root_anchor,
                    "workspaces",
                    expected_parent_identity=self._root_anchor.identity,
                )
            )
            workspace = self._enter_child(stack, workspaces, workspace_id, create=create)
            yield workspace

    @contextmanager
    def _evidence(
        self, workspace_id: str, *, create: bool
    ) -> Iterator[tuple[OwnedDirectoryAnchor, OwnedDirectoryAnchor, dict[str, Any], Identity, bytes]]:
        with contextlib.ExitStack() as stack:
            workspace = stack.enter_context(self._workspace(workspace_id, create=create))
            existed = self._filesystem.name_exists(
                workspace,
                "evidence",
                expected_parent_identity=workspace.identity,
            )
            evidence = self._enter_child(stack, workspace, "evidence", create=create)
            if not existed:
                ownership = self._write_marker(workspace, evidence, {}, None)
            ownership, marker_identity, marker_bytes = self._load_marker(workspace, evidence, recover=True)
            yield workspace, evidence, ownership, marker_identity, marker_bytes

    @contextmanager
    def _collection(
        self, workspace_id: str, artifact_type: str, *, create: bool
    ) -> Iterator[
        tuple[
            OwnedDirectoryAnchor,
            OwnedDirectoryAnchor,
            OwnedDirectoryAnchor,
            dict[str, Any],
        ]
    ]:
        with contextlib.ExitStack() as stack:
            workspace, evidence, ownership, _, _ = stack.enter_context(
                self._evidence(workspace_id, create=create)
            )
            collection = self._enter_child(stack, evidence, artifact_type, create=create)
            yield workspace, evidence, collection, ownership

    def _enter_child(
        self,
        stack: contextlib.ExitStack,
        parent: OwnedDirectoryAnchor,
        name: str,
        *,
        create: bool,
    ) -> OwnedDirectoryAnchor:
        try:
            return stack.enter_context(
                self._filesystem.anchor_child_directory(
                    parent,
                    name,
                    expected_parent_identity=parent.identity,
                )
            )
        except OwnedFilesystemError as error:
            if error.code != "owned_not_found" or not create:
                code = (
                    "artifact_not_found" if error.code == "owned_not_found" else "artifact_identity_changed"
                )
                raise ArtifactBoundaryError(code) from None
            return stack.enter_context(
                self._filesystem.create_child_directory(
                    parent,
                    name,
                    expected_parent_identity=parent.identity,
                )
            )

    def _load_marker(
        self,
        workspace: OwnedDirectoryAnchor,
        evidence: OwnedDirectoryAnchor,
        *,
        recover: bool,
    ) -> tuple[dict[str, Any], Identity, bytes]:
        try:
            marker_identity, content = self._filesystem.read_named_file(
                evidence,
                _OWNERSHIP_NAME,
                expected_parent_identity=evidence.identity,
                maximum_bytes=_MAX_MARKER_BYTES,
            )
        except OwnedFilesystemError as error:
            if error.code == "owned_content_changed":
                raise ArtifactBoundaryError("artifact_hash_mismatch") from None
            raise ArtifactBoundaryError("artifact_identity_changed") from None
        if len(content) > _MAX_MARKER_BYTES:
            raise ArtifactBoundaryError("artifact_identity_changed")
        try:
            payload = json.loads(content.decode("utf-8"))
            self._validate_marker(payload, workspace.identity, evidence.identity, marker_identity)
        except ArtifactBoundaryError:
            raise
        except (
            TypeError,
            ValueError,
            UnicodeError,
            RecursionError,
            OwnedFilesystemError,
        ):
            raise ArtifactBoundaryError("artifact_identity_changed") from None
        if recover and payload["pending"] is not None:
            payload = self._recover_pending(workspace, evidence, payload)
            return self._load_marker(workspace, evidence, recover=False)
        return payload, marker_identity, content

    def _write_marker(
        self,
        workspace: OwnedDirectoryAnchor,
        evidence: OwnedDirectoryAnchor,
        artifacts: dict[str, Claim],
        pending: dict[str, Any] | None,
    ) -> dict[str, Any]:
        temporary_name = f".ownership-{secrets.token_hex(16)}.tmp"
        temporary: OwnedOpenFile | None = None
        digest = ""
        size = 0
        try:
            with self._filesystem.create_temporary_file(
                evidence,
                temporary_name,
                expected_parent_identity=evidence.identity,
            ) as temporary:
                payload = {
                    "version": _OWNERSHIP_VERSION,
                    "root": list(self._root_anchor.identity),
                    "workspace": list(workspace.identity),
                    "evidence": list(evidence.identity),
                    "marker": list(temporary.identity),
                    "artifacts": artifacts,
                    "pending": pending,
                }
                encoded = self._canonical_bytes(payload, _MAX_MARKER_BYTES)
                self._write_descriptor(temporary.descriptor, encoded)
                digest, size = self._filesystem.hash_open_file(temporary)
                self._filesystem.replace_open_file(
                    evidence,
                    temporary,
                    temporary_name,
                    _OWNERSHIP_NAME,
                    expected_parent_identity=evidence.identity,
                    expected_source_identity=temporary.identity,
                    expected_sha256=digest,
                    expected_size=size,
                    replace_existing=True,
                )
                return payload
        except ArtifactBoundaryError:
            raise
        except (OwnedFilesystemError, OSError):
            if temporary is not None and digest:
                with contextlib.suppress(OwnedFilesystemError):
                    self._filesystem.delete_claimed_file(
                        evidence,
                        temporary_name,
                        expected_parent_identity=evidence.identity,
                        expected_source_identity=temporary.identity,
                        expected_sha256=digest,
                        expected_size=size,
                    )
            raise ArtifactBoundaryError("artifact_publish_failed") from None

    def _validate_marker(
        self,
        payload: Any,
        workspace_identity: Identity,
        evidence_identity: Identity,
        marker_identity: Identity,
    ) -> None:
        if not isinstance(payload, dict) or set(payload) != {
            "version",
            "root",
            "workspace",
            "evidence",
            "marker",
            "artifacts",
            "pending",
        }:
            raise ArtifactBoundaryError("artifact_identity_changed")
        expected = {
            "version": _OWNERSHIP_VERSION,
            "root": list(self._root_anchor.identity),
            "workspace": list(workspace_identity),
            "evidence": list(evidence_identity),
            "marker": list(marker_identity),
        }
        if any(payload[key] != value for key, value in expected.items()):
            raise ArtifactBoundaryError("artifact_identity_changed")
        artifacts = payload["artifacts"]
        if not isinstance(artifacts, dict) or len(artifacts) > _MAX_JSON_NODES:
            raise ArtifactBoundaryError("artifact_identity_changed")
        for slot, claim in artifacts.items():
            self._validate_claim(slot, claim)
        pending = payload["pending"]
        if pending is not None:
            self._validate_pending(pending)

    def _validate_pending(self, pending: Any) -> None:
        if not isinstance(pending, dict) or set(pending) != {
            "phase",
            "slot",
            "target",
            "temporary",
            "backup",
            "new",
            "old",
        }:
            raise ArtifactBoundaryError("artifact_identity_changed")
        if pending["phase"] not in {"prepared", "committed"}:
            raise ArtifactBoundaryError("artifact_identity_changed")
        self._validate_claim(pending["slot"], pending["new"])
        if pending["old"] is not None:
            self._validate_claim(pending["slot"], pending["old"])
        for key in ("target", "temporary", "backup"):
            OwnedArtifactFilesystem._validate_name(pending[key])

    @staticmethod
    def _validate_claim(slot: str, claim: Any) -> None:
        if (
            not isinstance(slot, str)
            or not isinstance(claim, dict)
            or set(claim)
            != {
                "slot",
                "kind",
                "id",
                "device_id",
                "file_id",
                "sha256",
                "size",
            }
        ):
            raise ArtifactBoundaryError("artifact_identity_changed")
        collection, separator, filename = slot.partition("/")
        expected_kind = _KIND_BY_COLLECTION.get(collection)
        artifact_id = claim.get("id")
        expected_filename = artifact_id if collection == "parts" else f"{artifact_id}.json"
        if (
            not separator
            or claim.get("slot") != slot
            or claim.get("kind") != expected_kind
            or not isinstance(artifact_id, str)
            or _IDENTIFIER.fullmatch(artifact_id) is None
            or filename != expected_filename
            or not isinstance(claim.get("device_id"), str)
            or not claim["device_id"].isdigit()
            or not isinstance(claim.get("file_id"), str)
            or not claim["file_id"].isdigit()
            or not isinstance(claim.get("sha256"), str)
            or _SHA256.fullmatch(claim["sha256"]) is None
            or not isinstance(claim.get("size"), int)
            or isinstance(claim["size"], bool)
            or claim["size"] < 0
        ):
            raise ArtifactBoundaryError("artifact_identity_changed")

    @staticmethod
    def _make_claim(
        slot: str,
        artifact_type: str,
        artifact_id: str,
        identity: Identity,
        digest: str,
        size: int,
    ) -> Claim:
        return {
            "slot": slot,
            "kind": _KIND_BY_COLLECTION[artifact_type],
            "id": artifact_id,
            "device_id": str(identity[0]),
            "file_id": str(identity[1]),
            "sha256": digest,
            "size": size,
        }

    @staticmethod
    def _claim_identity(claim: Claim) -> Identity:
        return int(claim["device_id"]), int(claim["file_id"])

    def _claim(self, ownership: dict[str, Any], artifact_type: str, filename: str) -> Claim:
        claim = ownership["artifacts"].get(f"{artifact_type}/{filename}")
        if not isinstance(claim, dict):
            raise ArtifactBoundaryError("artifact_identity_changed")
        return claim

    def _verify_claimed_name(self, collection: OwnedDirectoryAnchor, name: str, claim: Claim) -> None:
        try:
            with self._filesystem.open_claimed_file(
                collection,
                name,
                expected_parent_identity=collection.identity,
                expected_source_identity=self._claim_identity(claim),
                expected_sha256=claim["sha256"],
                expected_size=claim["size"],
            ):
                pass
        except OwnedFilesystemError as error:
            if error.code == "owned_content_changed":
                raise ArtifactBoundaryError("artifact_hash_mismatch") from None
            raise ArtifactBoundaryError("artifact_identity_changed") from None

    def _is_claimed_name(self, collection: OwnedDirectoryAnchor, name: str, claim: Claim) -> bool:
        if not self._filesystem.name_exists(collection, name, expected_parent_identity=collection.identity):
            return False
        self._verify_claimed_name(collection, name, claim)
        return True

    def _matches_claim(self, collection: OwnedDirectoryAnchor, name: str, claim: Claim) -> bool:
        if not self._filesystem.name_exists(collection, name, expected_parent_identity=collection.identity):
            return False
        try:
            self._verify_claimed_name(collection, name, claim)
        except ArtifactBoundaryError:
            return False
        return True

    def _delete_if_claimed(self, collection: OwnedDirectoryAnchor, name: str, claim: Claim) -> None:
        if not self._is_claimed_name(collection, name, claim):
            return
        self._filesystem.delete_claimed_file(
            collection,
            name,
            expected_parent_identity=collection.identity,
            expected_source_identity=self._claim_identity(claim),
            expected_sha256=claim["sha256"],
            expected_size=claim["size"],
        )

    def _encode_json(self, workspace_id: str, artifact_type: str, artifact_id: str, document: Any) -> bytes:
        try:
            bounded = (
                document.model_dump(mode="json")
                if isinstance(document, (EvidenceUnit, StudyMapSnapshot))
                else document
            )
            self._validate_json_object(bounded)
            model_type = EvidenceUnit if artifact_type == "units" else StudyMapSnapshot
            model = model_type.model_validate(document)
            if artifact_type == "units" and model.evidence_unit_id != artifact_id:
                raise ValueError("identifier mismatch")
            if artifact_type == "snapshots" and (
                model.snapshot_id != artifact_id or model.workspace_id != workspace_id
            ):
                raise ValueError("identifier mismatch")
            canonical = model.model_dump(mode="json")
            self._validate_json_object(canonical)
            return self._canonical_bytes(canonical, _MAX_JSON_BYTES)
        except ArtifactBoundaryError:
            raise
        except (TypeError, ValueError, UnicodeError, RecursionError):
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
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise ArtifactBoundaryError("artifact_json_invalid") from None

    @staticmethod
    def _canonical_bytes(document: Any, maximum: int) -> bytes:
        try:
            encoded = json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise ArtifactBoundaryError("artifact_json_invalid") from None
        if len(encoded) > maximum:
            code = "artifact_marker_full" if maximum == _MAX_MARKER_BYTES else "artifact_json_too_large"
            raise ArtifactBoundaryError(code)
        return encoded

    @staticmethod
    def _validate_json_object(document: Any) -> None:
        if not isinstance(document, dict):
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
            if isinstance(value, dict):
                identity = id(value)
                if identity in seen:
                    raise ArtifactBoundaryError("artifact_json_invalid")
                seen.add(identity)
                for key, child in value.items():
                    if not isinstance(key, str):
                        raise ArtifactBoundaryError("artifact_json_invalid")
                    if len(key.encode("utf-8")) > _MAX_JSON_STRING_BYTES:
                        raise ArtifactBoundaryError("artifact_json_too_large")
                    stack.append((child, depth + 1))
                continue
            if isinstance(value, (list, tuple)):
                identity = id(value)
                if identity in seen:
                    raise ArtifactBoundaryError("artifact_json_invalid")
                seen.add(identity)
                stack.extend((child, depth + 1) for child in value)
                continue
            if isinstance(value, str):
                if len(value.encode("utf-8")) > _MAX_JSON_STRING_BYTES:
                    raise ArtifactBoundaryError("artifact_json_too_large")
                continue
            if value is None or isinstance(value, (bool, int)):
                continue
            if isinstance(value, float) and math.isfinite(value):
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
    def _raise_filesystem_boundary(error: OwnedFilesystemError) -> None:
        if error.code == "owned_content_changed":
            raise ArtifactBoundaryError("artifact_hash_mismatch") from None
        if error.code == "owned_identity_changed":
            raise ArtifactBoundaryError("artifact_identity_changed") from None
        raise ArtifactBoundaryError("artifact_publish_failed") from None
