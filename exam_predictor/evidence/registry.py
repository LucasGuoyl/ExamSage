from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import re
import secrets
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, TypeVar

from exam_predictor.workspace.filesystem import (
    OwnedArtifactFilesystem,
    OwnedDirectoryAnchor,
    OwnedFilesystemError,
    OwnedMutationFile,
)


REGISTRY_NAME = ".evidence-artifact-registry.log"
LEGACY_REGISTRY_NAME = ".evidence-artifact-registry.sqlite3"
_LOG_MAGIC = b"EXAMSAGE-EVIDENCE-REGISTRY-LOG-v1\n"
_SCHEMA_VERSION = 1
_COLLECTIONS = frozenset({"parts", "units", "snapshots"})
_KIND_BY_COLLECTION = {"parts": "part", "units": "unit", "snapshots": "snapshot"}
_PUBLISH_PHASES = frozenset({"prepared", "backup", "installed", "committed"})
_DELETE_PHASES = frozenset({"planned", "quarantined", "removed"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GENERATION = re.compile(r"^[0-9a-f]{32}$")
_MAX_RECORD_BYTES = 16 * 1024 * 1024
_MAX_LOG_BYTES = 256 * 1024 * 1024

Identity = tuple[int, int]
_T = TypeVar("_T")
_REGISTRY_LOCKS_GUARD = threading.Lock()
_REGISTRY_LOCKS: dict[Identity, threading.RLock] = {}


def _shared_registry_lock(root_identity: Identity) -> threading.RLock:
    with _REGISTRY_LOCKS_GUARD:
        return _REGISTRY_LOCKS.setdefault(root_identity, threading.RLock())


class RegistryError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class WorkspaceClaim:
    workspace_id: str
    phase: str
    generation: str
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
    generation: str = ""


@dataclass(frozen=True)
class PublishIntent:
    workspace_id: str
    slot: str
    target_name: str
    temporary_name: str
    backup_name: str
    collection: str
    kind: str
    artifact_id: str
    expected_sha256: str
    expected_size: int
    old_claim: ArtifactClaim | None
    temporary_identity: Identity | None = None
    generation: str = ""


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
    generation: str = ""


@dataclass(frozen=True)
class DeleteJournalItem:
    workspace_id: str
    slot: str
    phase: str
    quarantine_name: str | None
    claim: ArtifactClaim
    generation: str = ""


@dataclass
class _RegistryState:
    workspaces: dict[str, WorkspaceClaim] = field(default_factory=dict)
    collections: dict[str, dict[str, Identity]] = field(default_factory=dict)
    artifacts: dict[tuple[str, str], ArtifactClaim] = field(default_factory=dict)
    publish_intents: dict[str, PublishIntent] = field(default_factory=dict)
    publish_journals: dict[str, PublishJournal] = field(default_factory=dict)
    delete_items: dict[tuple[str, str], DeleteJournalItem] = field(default_factory=dict)


class EvidenceArtifactRegistry:
    """Single-Worker registry persisted only through one pinned no-follow handle."""

    def __init__(
        self,
        root: OwnedDirectoryAnchor,
        filesystem: OwnedArtifactFilesystem,
    ) -> None:
        self._root = root
        self._filesystem = filesystem
        self._lock = _shared_registry_lock(root.identity)
        self._file_context = None
        self._file: OwnedMutationFile | None = None
        self._closed = False
        self._sequence = 0
        self._previous_digest = bytes(32)
        self._minimum_sequence = 0
        self._minimum_length = 0
        self._minimum_digest = bytes(32)
        self._state = _RegistryState()
        self._recovered_reservations: dict[str, str] = {}
        try:
            with self._lock:
                self._initialize()
        except Exception:
            self.close()
            raise

    def _initialize(self) -> None:
        if self._filesystem.name_exists(
            self._root,
            LEGACY_REGISTRY_NAME,
            expected_parent_identity=self._root.identity,
        ):
            raise RegistryError("registry_legacy_state")
        if not self._filesystem.name_exists(
            self._root,
            REGISTRY_NAME,
            expected_parent_identity=self._root.identity,
        ):
            self._install_initial_registry()
        try:
            self._file_context = self._filesystem.open_mutation_file(
                self._root,
                REGISTRY_NAME,
                expected_parent_identity=self._root.identity,
            )
            self._file = self._file_context.__enter__()
        except OwnedFilesystemError as error:
            raise RegistryError("registry_operation_failed") from error
        self._load()
        self._recovered_reservations = {
            workspace_id: claim.generation
            for workspace_id, claim in self._state.workspaces.items()
            if claim.phase == "reserved"
        }
        if self._sequence == 0:
            raise RegistryError("registry_corrupt")
        self._verify_pinned()

    def _install_initial_registry(self) -> None:
        content = self._initial_registry_bytes()
        temporary_name = f".evidence-registry-{secrets.token_hex(16)}.tmp"
        digest = hashlib.sha256(content).hexdigest()
        try:
            with self._filesystem.create_temporary_file(
                self._root,
                temporary_name,
                expected_parent_identity=self._root.identity,
            ) as temporary:
                self._write_all(temporary.descriptor, content)
                os.fsync(temporary.descriptor)
                self._filesystem.replace_open_file(
                    self._root,
                    temporary,
                    temporary_name,
                    REGISTRY_NAME,
                    expected_parent_identity=self._root.identity,
                    expected_source_identity=temporary.identity,
                    expected_sha256=digest,
                    expected_size=len(content),
                    replace_existing=False,
                )
        except OwnedFilesystemError as error:
            if error.code == "owned_destination_exists":
                return
            raise RegistryError("registry_operation_failed") from error
        except OSError:
            raise RegistryError("registry_operation_failed") from None

    def _initial_registry_bytes(self) -> bytes:
        return _LOG_MAGIC + self._encode_state_record(
            _RegistryState(),
            sequence=1,
            previous_digest=bytes(32),
        )[0]

    @property
    def path(self) -> Path:
        return self._root.path / REGISTRY_NAME

    @property
    def identity(self) -> Identity:
        with self._lock:
            return self._require_file().identity

    def __enter__(self) -> EvidenceArtifactRegistry:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._file is not None and self._file_context is not None:
                self._file_context.__exit__(None, None, None)
            self._file = None
            self._file_context = None

    def reserve_workspace(self, workspace_id: str, *, generation: str | None = None) -> bool:
        generation = secrets.token_hex(16) if generation is None else generation
        if _GENERATION.fullmatch(generation) is None:
            raise RegistryError("registry_claim_invalid")

        def mutate(state: _RegistryState) -> tuple[bool, bool]:
            if workspace_id in state.workspaces:
                return False, False
            state.workspaces[workspace_id] = WorkspaceClaim(
                workspace_id,
                "reserved",
                generation,
                None,
                None,
            )
            return True, True

        return self._mutate(mutate)

    def retire_reserved_workspace(
        self,
        workspace_id: str,
        *,
        expected_generation: str,
    ) -> bool:
        self._validate_generation(expected_generation)

        def mutate(state: _RegistryState) -> tuple[bool, bool]:
            claim = state.workspaces.get(workspace_id)
            if claim is None:
                return False, False
            if claim.generation != expected_generation:
                raise RegistryError("registry_state_conflict")
            if (
                claim.phase != "reserved"
                or workspace_id in state.collections
                or any(key[0] == workspace_id for key in state.artifacts)
                or workspace_id in state.publish_intents
                or workspace_id in state.publish_journals
                or any(key[0] == workspace_id for key in state.delete_items)
            ):
                raise RegistryError("registry_state_conflict")
            del state.workspaces[workspace_id]
            return True, True

        with self._lock:
            retired = self._mutate(mutate)
            self._recovered_reservations.pop(workspace_id, None)
            return retired

    def retire_recovered_workspace(self, workspace_id: str) -> bool:
        with self._lock:
            generation = self._recovered_reservations.get(workspace_id)
            if generation is None:
                return False
            try:
                return self.retire_reserved_workspace(
                    workspace_id,
                    expected_generation=generation,
                )
            except RegistryError as error:
                if error.code == "registry_state_conflict":
                    return False
                raise

    def finalize_workspace(
        self,
        workspace_id: str,
        *,
        expected_generation: str,
        workspace_identity: Identity,
        evidence_identity: Identity,
        collection_identities: dict[str, Identity],
    ) -> WorkspaceClaim:
        if _GENERATION.fullmatch(expected_generation) is None or set(collection_identities) != _COLLECTIONS:
            raise RegistryError("registry_claim_invalid")

        def mutate(state: _RegistryState) -> tuple[bool, WorkspaceClaim]:
            current = state.workspaces.get(workspace_id)
            if (
                current is None
                or current.phase != "reserved"
                or current.generation != expected_generation
            ):
                raise RegistryError("registry_state_conflict")
            claim = WorkspaceClaim(
                workspace_id,
                "active",
                current.generation,
                workspace_identity,
                evidence_identity,
            )
            state.workspaces[workspace_id] = claim
            state.collections[workspace_id] = dict(collection_identities)
            return True, claim

        with self._lock:
            claim = self._mutate(mutate)
            self._recovered_reservations.pop(workspace_id, None)
            return claim

    def get_workspace(self, workspace_id: str) -> WorkspaceClaim | None:
        return self._read(lambda state: state.workspaces.get(workspace_id))

    def get_collection_identities(self, workspace_id: str) -> dict[str, Identity]:
        return self._read(lambda state: dict(state.collections.get(workspace_id, {})))

    def get_artifact(self, workspace_id: str, slot: str) -> ArtifactClaim | None:
        return self._read(lambda state: state.artifacts.get((workspace_id, slot)))

    def get_artifacts(self, workspace_id: str) -> tuple[ArtifactClaim, ...]:
        return self._read(
            lambda state: tuple(
                claim
                for (candidate_workspace, _slot), claim in sorted(state.artifacts.items())
                if candidate_workspace == workspace_id
            )
        )

    def reserve_publish(self, intent: PublishIntent) -> None:
        self._validate_publish_intent(intent)

        def mutate(state: _RegistryState) -> tuple[bool, None]:
            workspace = state.workspaces.get(intent.workspace_id)
            if workspace is None or workspace.phase != "active" or intent.generation != workspace.generation:
                raise RegistryError("registry_state_conflict")
            if (
                state.artifacts.get((intent.workspace_id, intent.slot)) != intent.old_claim
                or intent.workspace_id in state.publish_intents
                or intent.workspace_id in state.publish_journals
            ):
                raise RegistryError("registry_state_conflict")
            state.publish_intents[intent.workspace_id] = intent
            return True, None

        self._mutate(mutate)

    def get_publish_intent(self, workspace_id: str) -> PublishIntent | None:
        return self._read(lambda state: state.publish_intents.get(workspace_id))

    def claim_publish_temporary(
        self,
        workspace_id: str,
        identity: Identity,
        *,
        expected_generation: str,
        expected_temporary_name: str,
    ) -> None:
        if not self._valid_identity(identity):
            raise RegistryError("registry_claim_invalid")
        self._validate_publish_token(expected_generation, expected_temporary_name)

        def mutate(state: _RegistryState) -> tuple[bool, None]:
            intent = state.publish_intents.get(workspace_id)
            if (
                intent is None
                or intent.temporary_identity is not None
                or intent.generation != expected_generation
                or intent.temporary_name != expected_temporary_name
            ):
                raise RegistryError("registry_state_conflict")
            state.publish_intents[workspace_id] = replace(intent, temporary_identity=identity)
            return True, None

        self._mutate(mutate)

    def clear_publish_intent(
        self,
        workspace_id: str,
        *,
        expected_generation: str,
        expected_temporary_name: str,
    ) -> None:
        self._validate_publish_token(expected_generation, expected_temporary_name)

        def mutate(state: _RegistryState) -> tuple[bool, None]:
            intent = state.publish_intents.get(workspace_id)
            if (
                intent is None
                or intent.generation != expected_generation
                or intent.temporary_name != expected_temporary_name
            ):
                raise RegistryError("registry_state_conflict")
            del state.publish_intents[workspace_id]
            return True, None

        self._mutate(mutate)

    def prepare_publish(self, journal: PublishJournal) -> None:
        self._validate_publish_journal(journal)
        expected_intent = self._intent_for_journal(journal)

        def mutate(state: _RegistryState) -> tuple[bool, None]:
            intent = state.publish_intents.get(journal.workspace_id)
            if (
                intent is None
                or intent.temporary_identity not in {None, journal.new_claim.identity}
                or replace(intent, temporary_identity=journal.new_claim.identity) != expected_intent
            ):
                raise RegistryError("registry_state_conflict")
            if state.artifacts.get((journal.workspace_id, journal.slot)) != journal.old_claim:
                raise RegistryError("registry_state_conflict")
            del state.publish_intents[journal.workspace_id]
            state.publish_journals[journal.workspace_id] = journal
            return True, None

        self._mutate(mutate)

    def get_publish_journal(self, workspace_id: str) -> PublishJournal | None:
        return self._read(lambda state: state.publish_journals.get(workspace_id))

    def advance_publish(
        self,
        workspace_id: str,
        *,
        expected_generation: str,
        expected_temporary_name: str,
        expected_phase: str,
        new_phase: str,
    ) -> None:
        if (expected_phase, new_phase) not in {("prepared", "backup"), ("backup", "installed")}:
            raise RegistryError("registry_claim_invalid")
        self._validate_publish_token(expected_generation, expected_temporary_name)

        def mutate(state: _RegistryState) -> tuple[bool, None]:
            journal = state.publish_journals.get(workspace_id)
            if (
                journal is None
                or journal.generation != expected_generation
                or journal.temporary_name != expected_temporary_name
                or journal.phase != expected_phase
            ):
                raise RegistryError("registry_state_conflict")
            state.publish_journals[workspace_id] = replace(journal, phase=new_phase)
            return True, None

        self._mutate(mutate)

    def commit_publish(
        self,
        workspace_id: str,
        claim: ArtifactClaim,
        *,
        expected_generation: str,
        expected_temporary_name: str,
    ) -> None:
        self._validate_publish_token(expected_generation, expected_temporary_name)

        def mutate(state: _RegistryState) -> tuple[bool, None]:
            journal = state.publish_journals.get(workspace_id)
            if (
                journal is None
                or journal.generation != expected_generation
                or journal.temporary_name != expected_temporary_name
                or journal.phase != "installed"
                or journal.new_claim != claim
            ):
                raise RegistryError("registry_state_conflict")
            state.artifacts[(workspace_id, claim.slot)] = claim
            state.publish_journals[workspace_id] = replace(journal, phase="committed")
            return True, None

        self._mutate(mutate)

    def clear_publish(
        self,
        workspace_id: str,
        *,
        expected_generation: str,
        expected_temporary_name: str,
    ) -> None:
        self._validate_publish_token(expected_generation, expected_temporary_name)

        def mutate(state: _RegistryState) -> tuple[bool, None]:
            journal = state.publish_journals.get(workspace_id)
            if (
                journal is None
                or journal.generation != expected_generation
                or journal.temporary_name != expected_temporary_name
                or journal.phase != "committed"
            ):
                raise RegistryError("registry_state_conflict")
            del state.publish_journals[workspace_id]
            return True, None

        self._mutate(mutate)

    def abort_publish(
        self,
        workspace_id: str,
        *,
        expected_generation: str,
        expected_temporary_name: str,
        expected_phases: set[str],
    ) -> None:
        if not expected_phases or not expected_phases <= {"prepared", "backup"}:
            raise RegistryError("registry_claim_invalid")
        self._validate_publish_token(expected_generation, expected_temporary_name)

        def mutate(state: _RegistryState) -> tuple[bool, None]:
            journal = state.publish_journals.get(workspace_id)
            if (
                journal is None
                or journal.generation != expected_generation
                or journal.temporary_name != expected_temporary_name
                or journal.phase not in expected_phases
            ):
                raise RegistryError("registry_state_conflict")
            del state.publish_journals[workspace_id]
            return True, None

        self._mutate(mutate)

    def begin_delete(
        self,
        workspace_id: str,
        *,
        expected_generation: str,
    ) -> tuple[DeleteJournalItem, ...]:
        self._validate_generation(expected_generation)

        def mutate(state: _RegistryState) -> tuple[bool, tuple[DeleteJournalItem, ...]]:
            workspace = state.workspaces.get(workspace_id)
            if workspace is None:
                return False, ()
            if workspace.generation != expected_generation:
                raise RegistryError("registry_state_conflict")
            if workspace.phase == "deleting":
                return False, self._delete_items(state, workspace_id)
            if workspace.phase != "active":
                raise RegistryError("registry_state_conflict")
            if workspace_id in state.publish_intents or workspace_id in state.publish_journals:
                raise RegistryError("registry_pending")
            state.workspaces[workspace_id] = replace(workspace, phase="deleting")
            for (candidate_workspace, slot), claim in tuple(state.artifacts.items()):
                if candidate_workspace == workspace_id:
                    state.delete_items[(workspace_id, slot)] = DeleteJournalItem(
                        workspace_id,
                        slot,
                        "planned",
                        None,
                        claim,
                        workspace.generation,
                    )
            return True, self._delete_items(state, workspace_id)

        return self._mutate(mutate)

    def get_delete_items(
        self,
        workspace_id: str,
        *,
        expected_generation: str,
    ) -> tuple[DeleteJournalItem, ...]:
        self._validate_generation(expected_generation)

        def read(state: _RegistryState) -> tuple[DeleteJournalItem, ...]:
            workspace = state.workspaces.get(workspace_id)
            if workspace is None:
                return ()
            if workspace.generation != expected_generation:
                raise RegistryError("registry_state_conflict")
            return self._delete_items(state, workspace_id)

        return self._read(read)

    def plan_delete_quarantine(
        self,
        workspace_id: str,
        slot: str,
        quarantine_name: str,
        *,
        expected_generation: str,
    ) -> None:
        self._validate_generation(expected_generation)

        def mutate(state: _RegistryState) -> tuple[bool, None]:
            key = (workspace_id, slot)
            item = state.delete_items.get(key)
            if (
                item is None
                or item.generation != expected_generation
                or item.phase != "planned"
                or item.quarantine_name not in {None, quarantine_name}
            ):
                raise RegistryError("registry_state_conflict")
            state.delete_items[key] = replace(item, quarantine_name=quarantine_name)
            return True, None

        self._mutate(mutate)

    def advance_delete_item(
        self,
        workspace_id: str,
        slot: str,
        *,
        expected_generation: str,
        expected_phase: str,
        new_phase: str,
        quarantine_name: str,
    ) -> None:
        if (expected_phase, new_phase) not in {
            ("planned", "quarantined"),
            ("quarantined", "removed"),
        }:
            raise RegistryError("registry_claim_invalid")
        self._validate_generation(expected_generation)

        def mutate(state: _RegistryState) -> tuple[bool, None]:
            key = (workspace_id, slot)
            item = state.delete_items.get(key)
            if (
                item is None
                or item.generation != expected_generation
                or item.phase != expected_phase
                or item.quarantine_name not in {None, quarantine_name}
            ):
                raise RegistryError("registry_state_conflict")
            state.delete_items[key] = replace(
                item,
                phase=new_phase,
                quarantine_name=quarantine_name,
            )
            return True, None

        self._mutate(mutate)

    def clear_deleted_workspace(
        self,
        workspace_id: str,
        *,
        expected_generation: str,
    ) -> None:
        self._validate_generation(expected_generation)

        def mutate(state: _RegistryState) -> tuple[bool, None]:
            workspace = state.workspaces.get(workspace_id)
            if workspace is None:
                return False, None
            items = self._delete_items(state, workspace_id)
            if (
                workspace.generation != expected_generation
                or workspace.phase != "deleting"
                or any(item.phase != "removed" for item in items)
            ):
                raise RegistryError("registry_state_conflict")
            del state.workspaces[workspace_id]
            state.collections.pop(workspace_id, None)
            state.publish_intents.pop(workspace_id, None)
            state.publish_journals.pop(workspace_id, None)
            for key in tuple(state.artifacts):
                if key[0] == workspace_id:
                    del state.artifacts[key]
            for key in tuple(state.delete_items):
                if key[0] == workspace_id:
                    del state.delete_items[key]
            return True, None

        self._mutate(mutate)

    def _read(self, reader: Callable[[_RegistryState], _T]) -> _T:
        with self._lock:
            self._load()
            self._verify_pinned()
            return reader(self._state)

    def _mutate(self, mutator: Callable[[_RegistryState], tuple[bool, _T]]) -> _T:
        with self._lock:
            self._load()
            self._verify_pinned()
            candidate = copy.deepcopy(self._state)
            changed, result = mutator(candidate)
            if changed:
                self._validate_state(candidate)
                self._append_state(candidate)
                self._state = candidate
            self._verify_pinned()
            return result

    def _load(self) -> None:
        source = self._require_file()
        try:
            opened = os.fstat(source.descriptor)
            if opened.st_size > _MAX_LOG_BYTES:
                raise RegistryError("registry_corrupt")
            os.lseek(source.descriptor, 0, os.SEEK_SET)
            content = bytearray()
            while chunk := os.read(source.descriptor, 1024 * 1024):
                content.extend(chunk)
        except RegistryError:
            raise
        except OSError:
            raise RegistryError("registry_operation_failed") from None
        raw = bytes(content)
        if not raw:
            if source.created and self._minimum_sequence == 0 and self._minimum_length == 0:
                self._rewrite_magic()
                return
            raise RegistryError("registry_corrupt")
        if len(raw) < len(_LOG_MAGIC):
            raise RegistryError("registry_corrupt")
        if not raw.startswith(_LOG_MAGIC):
            raise RegistryError("registry_corrupt")
        cursor = len(_LOG_MAGIC)
        last_good = cursor
        state = _RegistryState()
        previous = bytes(32)
        sequence = 0
        incomplete_tail = False
        while cursor < len(raw):
            newline = raw.find(b"\n", cursor)
            if newline < 0:
                incomplete_tail = True
                break
            line = raw[cursor:newline]
            if not line or len(line) > _MAX_RECORD_BYTES:
                raise RegistryError("registry_corrupt")
            state, sequence, previous = self._decode_record(line, sequence, previous)
            cursor = newline + 1
            last_good = cursor
        if (
            sequence < self._minimum_sequence
            or last_good < self._minimum_length
            or (
                sequence == self._minimum_sequence
                and self._minimum_sequence > 0
                and previous != self._minimum_digest
            )
        ):
            raise RegistryError("registry_corrupt")
        if incomplete_tail:
            self._truncate(last_good)
        self._state = state
        self._sequence = sequence
        self._previous_digest = previous
        if sequence > self._minimum_sequence:
            self._minimum_sequence = sequence
            self._minimum_digest = previous
        self._minimum_length = max(self._minimum_length, last_good)

    def _decode_record(
        self,
        line: bytes,
        previous_sequence: int,
        previous_digest: bytes,
    ) -> tuple[_RegistryState, int, bytes]:
        try:
            body, separator, encoded_digest = line.rpartition(b"\t")
            if not separator or _SHA256.fullmatch(encoded_digest.decode("ascii")) is None:
                raise ValueError("record framing")
            digest = bytes.fromhex(encoded_digest.decode("ascii"))
            if digest != hashlib.sha256(previous_digest + body).digest():
                raise ValueError("record checksum")
            document = json.loads(body.decode("utf-8"))
            if self._canonical(document) != body:
                raise ValueError("record normalization")
            self._require_keys(
                document,
                {"schema_version", "sequence", "previous_sha256", "root_identity", "state"},
            )
            sequence = self._require_int(document["sequence"])
            if (
                self._require_int(document["schema_version"]) != _SCHEMA_VERSION
                or sequence != previous_sequence + 1
                or self._require_string(document["previous_sha256"]) != previous_digest.hex()
                or self._parse_identity(document["root_identity"]) != self._root.identity
            ):
                raise ValueError("record chain")
            state = self._parse_state(document["state"])
            self._validate_state(state)
            return state, sequence, digest
        except (RegistryError, ValueError, TypeError, UnicodeError, json.JSONDecodeError):
            raise RegistryError("registry_corrupt") from None

    def _append_state(self, state: _RegistryState) -> None:
        source = self._require_file()
        sequence = self._sequence + 1
        record, digest = self._encode_state_record(
            state,
            sequence=sequence,
            previous_digest=self._previous_digest,
        )
        start = os.lseek(source.descriptor, 0, os.SEEK_END)
        if start + len(record) > _MAX_LOG_BYTES:
            raise RegistryError("registry_operation_failed")
        try:
            self._write_all(source.descriptor, record)
            os.fsync(source.descriptor)
        except OSError:
            with contextlib.suppress(OSError):
                os.ftruncate(source.descriptor, start)
                os.fsync(source.descriptor)
            raise RegistryError("registry_operation_failed") from None
        self._sequence = sequence
        self._previous_digest = digest
        self._minimum_sequence = sequence
        self._minimum_length = start + len(record)
        self._minimum_digest = digest

    def _encode_state_record(
        self,
        state: _RegistryState,
        *,
        sequence: int,
        previous_digest: bytes,
    ) -> tuple[bytes, bytes]:
        document = {
            "previous_sha256": previous_digest.hex(),
            "root_identity": self._identity_json(self._root.identity),
            "schema_version": _SCHEMA_VERSION,
            "sequence": sequence,
            "state": self._state_json(state),
        }
        body = self._canonical(document)
        digest = hashlib.sha256(previous_digest + body).digest()
        record = body + b"\t" + digest.hex().encode("ascii") + b"\n"
        if len(record) > _MAX_RECORD_BYTES:
            raise RegistryError("registry_operation_failed")
        return record, digest

    def _rewrite_magic(self) -> None:
        source = self._require_file()
        try:
            os.lseek(source.descriptor, 0, os.SEEK_SET)
            os.ftruncate(source.descriptor, 0)
            self._write_all(source.descriptor, _LOG_MAGIC)
            os.fsync(source.descriptor)
        except OSError:
            raise RegistryError("registry_operation_failed") from None
        self._minimum_length = len(_LOG_MAGIC)

    def _truncate(self, size: int) -> None:
        try:
            os.ftruncate(self._require_file().descriptor, size)
            os.fsync(self._require_file().descriptor)
        except OSError:
            raise RegistryError("registry_operation_failed") from None

    def _verify_pinned(self) -> None:
        source = self._require_file()
        try:
            opened = os.fstat(source.descriptor)
            if (opened.st_dev, opened.st_ino) != source.identity:
                raise RegistryError("registry_identity_changed")
            if self._filesystem.name_exists(
                self._root,
                LEGACY_REGISTRY_NAME,
                expected_parent_identity=self._root.identity,
            ):
                raise RegistryError("registry_identity_changed")
            if not self._filesystem.name_has_identity(
                self._root,
                REGISTRY_NAME,
                expected_parent_identity=self._root.identity,
                expected_source_identity=source.identity,
            ):
                raise RegistryError("registry_identity_changed")
        except RegistryError:
            raise
        except (OSError, OwnedFilesystemError):
            raise RegistryError("registry_identity_changed") from None

    def _require_file(self) -> OwnedMutationFile:
        if self._closed or self._file is None:
            raise RegistryError("registry_closed")
        return self._file

    @staticmethod
    def _write_all(descriptor: int, content: bytes) -> None:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short registry write")
            view = view[written:]

    @classmethod
    def _validate_publish_token(
        cls,
        expected_generation: str,
        expected_temporary_name: str,
    ) -> None:
        if (
            _GENERATION.fullmatch(expected_generation) is None
            or not cls._valid_name(expected_temporary_name)
        ):
            raise RegistryError("registry_claim_invalid")

    @staticmethod
    def _validate_generation(expected_generation: str) -> None:
        if _GENERATION.fullmatch(expected_generation) is None:
            raise RegistryError("registry_claim_invalid")

    @staticmethod
    def _delete_items(state: _RegistryState, workspace_id: str) -> tuple[DeleteJournalItem, ...]:
        return tuple(
            item
            for (candidate_workspace, _slot), item in sorted(state.delete_items.items())
            if candidate_workspace == workspace_id
        )

    @classmethod
    def _validate_publish_intent(cls, intent: PublishIntent) -> None:
        strings = (
            intent.workspace_id,
            intent.slot,
            intent.target_name,
            intent.temporary_name,
            intent.backup_name,
            intent.collection,
            intent.kind,
            intent.artifact_id,
            intent.expected_sha256,
            intent.generation,
        )
        if (
            any(type(value) is not str for value in strings)
            or type(intent.expected_size) is not int
            or intent.expected_size < 0
            or _SHA256.fullmatch(intent.expected_sha256) is None
            or _GENERATION.fullmatch(intent.generation) is None
            or intent.collection not in _COLLECTIONS
            or intent.slot != f"{intent.collection}/{intent.target_name}"
            or not all(
                cls._valid_name(name)
                for name in (intent.target_name, intent.temporary_name, intent.backup_name)
            )
            or (intent.temporary_identity is not None and not cls._valid_identity(intent.temporary_identity))
            or (
                intent.old_claim is not None
                and (
                    intent.old_claim.workspace_id != intent.workspace_id
                    or intent.old_claim.slot != intent.slot
                    or intent.old_claim.generation != intent.generation
                )
            )
        ):
            raise RegistryError("registry_claim_invalid")

    @classmethod
    def _validate_publish_journal(cls, journal: PublishJournal) -> None:
        if (
            journal.phase != "prepared"
            or journal.workspace_id != journal.new_claim.workspace_id
            or journal.slot != journal.new_claim.slot
            or journal.generation != journal.new_claim.generation
            or _GENERATION.fullmatch(journal.generation) is None
            or journal.slot != f"{journal.new_claim.collection}/{journal.target_name}"
            or (
                journal.old_claim is not None
                and (
                    journal.old_claim.workspace_id != journal.workspace_id
                    or journal.old_claim.slot != journal.slot
                    or journal.old_claim.generation != journal.generation
                )
            )
        ):
            raise RegistryError("registry_claim_invalid")
        cls._validate_artifact_claim(journal.new_claim)
        if journal.old_claim is not None:
            cls._validate_artifact_claim(journal.old_claim)

    @staticmethod
    def _intent_for_journal(journal: PublishJournal) -> PublishIntent:
        claim = journal.new_claim
        return PublishIntent(
            workspace_id=journal.workspace_id,
            slot=journal.slot,
            target_name=journal.target_name,
            temporary_name=journal.temporary_name,
            backup_name=journal.backup_name,
            collection=claim.collection,
            kind=claim.kind,
            artifact_id=claim.artifact_id,
            expected_sha256=claim.sha256,
            expected_size=claim.size,
            old_claim=journal.old_claim,
            temporary_identity=claim.identity,
            generation=journal.generation,
        )

    @classmethod
    def _validate_state(cls, state: _RegistryState) -> None:
        for workspace_id, claim in state.workspaces.items():
            if (
                claim.workspace_id != workspace_id
                or not cls._valid_name(workspace_id)
                or claim.phase not in {"reserved", "active", "deleting"}
            ):
                raise RegistryError("registry_corrupt")
            if _GENERATION.fullmatch(claim.generation) is None:
                raise RegistryError("registry_corrupt")
            artifact_keys = {key for key in state.artifacts if key[0] == workspace_id}
            delete_keys = {key for key in state.delete_items if key[0] == workspace_id}
            has_publish = workspace_id in state.publish_intents or workspace_id in state.publish_journals
            if claim.phase == "reserved":
                if (
                    claim.workspace_identity is not None
                    or claim.evidence_identity is not None
                    or workspace_id in state.collections
                    or artifact_keys
                    or delete_keys
                    or has_publish
                ):
                    raise RegistryError("registry_corrupt")
                continue
            if (
                claim.workspace_identity is None
                or claim.evidence_identity is None
                or set(state.collections.get(workspace_id, {})) != _COLLECTIONS
            ):
                raise RegistryError("registry_corrupt")
            if claim.phase == "active" and delete_keys:
                raise RegistryError("registry_corrupt")
            if claim.phase == "deleting" and (has_publish or delete_keys != artifact_keys):
                raise RegistryError("registry_corrupt")
        for workspace_id, identities in state.collections.items():
            workspace = state.workspaces.get(workspace_id)
            if (
                workspace is None
                or workspace.phase == "reserved"
                or set(identities) != _COLLECTIONS
                or any(not cls._valid_identity(identity) for identity in identities.values())
            ):
                raise RegistryError("registry_corrupt")
        for key, claim in state.artifacts.items():
            workspace = state.workspaces.get(claim.workspace_id)
            if (
                key != (claim.workspace_id, claim.slot)
                or workspace is None
                or workspace.phase == "reserved"
                or claim.generation != workspace.generation
                or not cls._claim_route_valid(claim)
            ):
                raise RegistryError("registry_corrupt")
            cls._validate_artifact_claim(claim)
            if claim.collection not in state.collections.get(claim.workspace_id, {}):
                raise RegistryError("registry_corrupt")
        for workspace_id, intent in state.publish_intents.items():
            workspace = state.workspaces.get(workspace_id)
            if (
                workspace_id != intent.workspace_id
                or workspace_id in state.publish_journals
                or workspace is None
                or workspace.phase != "active"
                or intent.generation != workspace.generation
                or state.artifacts.get((workspace_id, intent.slot)) != intent.old_claim
                or intent.kind != _KIND_BY_COLLECTION.get(intent.collection)
                or intent.artifact_id != cls._artifact_id_for_target(intent.collection, intent.target_name)
                or len({intent.target_name, intent.temporary_name, intent.backup_name}) != 3
            ):
                raise RegistryError("registry_corrupt")
            cls._validate_publish_intent(intent)
        for workspace_id, journal in state.publish_journals.items():
            workspace = state.workspaces.get(workspace_id)
            current = state.artifacts.get((workspace_id, journal.slot))
            expected = journal.new_claim if journal.phase == "committed" else journal.old_claim
            if (
                workspace_id != journal.workspace_id
                or journal.phase not in _PUBLISH_PHASES
                or workspace is None
                or workspace.phase != "active"
                or journal.generation != workspace.generation
                or current != expected
                or len({journal.target_name, journal.temporary_name, journal.backup_name}) != 3
            ):
                raise RegistryError("registry_corrupt")
            shape = replace(journal, phase="prepared")
            cls._validate_publish_journal(shape)
        for key, item in state.delete_items.items():
            workspace = state.workspaces.get(item.workspace_id)
            if (
                key != (item.workspace_id, item.slot)
                or item.phase not in _DELETE_PHASES
                or workspace is None
                or workspace.phase != "deleting"
                or item.generation != workspace.generation
                or item.claim.generation != item.generation
                or state.artifacts.get(key) != item.claim
                or (item.phase in {"quarantined", "removed"} and item.quarantine_name is None)
                or (item.quarantine_name is not None and not cls._valid_name(item.quarantine_name))
            ):
                raise RegistryError("registry_corrupt")
            cls._validate_artifact_claim(item.claim)

    @staticmethod
    def _artifact_id_for_target(collection: str, target_name: str) -> str | None:
        if collection == "parts":
            return target_name
        if collection in {"units", "snapshots"} and target_name.endswith(".json"):
            return target_name[:-5]
        return None

    @classmethod
    def _claim_route_valid(cls, claim: ArtifactClaim) -> bool:
        target_name = claim.slot.split("/", 1)[1] if "/" in claim.slot else ""
        return (
            claim.collection in _COLLECTIONS
            and claim.slot == f"{claim.collection}/{target_name}"
            and claim.kind == _KIND_BY_COLLECTION[claim.collection]
            and claim.artifact_id == cls._artifact_id_for_target(claim.collection, target_name)
        )

    @staticmethod
    def _validate_artifact_claim(claim: ArtifactClaim) -> None:
        strings = (
            claim.workspace_id,
            claim.slot,
            claim.collection,
            claim.kind,
            claim.artifact_id,
            claim.sha256,
            claim.generation,
        )
        if any(type(value) is not str for value in strings):
            raise RegistryError("registry_claim_invalid")
        slot_parts = claim.slot.split("/")
        if (
            claim.collection not in _COLLECTIONS
            or len(slot_parts) != 2
            or slot_parts[0] != claim.collection
            or not EvidenceArtifactRegistry._valid_name(slot_parts[1])
            or _SHA256.fullmatch(claim.sha256) is None
            or _GENERATION.fullmatch(claim.generation) is None
            or type(claim.size) is not int
            or claim.size < 0
            or type(claim.identity) is not tuple
            or len(claim.identity) != 2
            or any(type(value) is not int or value < 0 for value in claim.identity)
        ):
            raise RegistryError("registry_claim_invalid")

    @staticmethod
    def _valid_name(value: str) -> bool:
        return (
            type(value) is str
            and value not in {"", ".", ".."}
            and "/" not in value
            and "\\" not in value
            and ":" not in value
            and "\x00" not in value
        )

    @staticmethod
    def _valid_identity(value: Any) -> bool:
        return (
            type(value) is tuple
            and len(value) == 2
            and all(type(item) is int and item >= 0 for item in value)
        )

    @classmethod
    def _state_json(cls, state: _RegistryState) -> dict[str, Any]:
        return {
            "artifacts": [cls._artifact_json(claim) for _key, claim in sorted(state.artifacts.items())],
            "collections": [
                {
                    "collection": name,
                    "identity": cls._identity_json(identity),
                    "workspace_id": workspace_id,
                }
                for workspace_id, identities in sorted(state.collections.items())
                for name, identity in sorted(identities.items())
            ],
            "delete_items": [
                {
                    "claim": cls._artifact_json(item.claim),
                    "generation": item.generation,
                    "phase": item.phase,
                    "quarantine_name": item.quarantine_name,
                    "slot": item.slot,
                    "workspace_id": item.workspace_id,
                }
                for _key, item in sorted(state.delete_items.items())
            ],
            "publish_intents": [
                cls._intent_json(intent) for _key, intent in sorted(state.publish_intents.items())
            ],
            "publish_journals": [
                cls._journal_json(journal) for _key, journal in sorted(state.publish_journals.items())
            ],
            "workspaces": [
                {
                    "evidence_identity": cls._optional_identity_json(claim.evidence_identity),
                    "generation": claim.generation,
                    "phase": claim.phase,
                    "workspace_id": claim.workspace_id,
                    "workspace_identity": cls._optional_identity_json(claim.workspace_identity),
                }
                for _key, claim in sorted(state.workspaces.items())
            ],
        }

    @classmethod
    def _parse_state(cls, value: Any) -> _RegistryState:
        cls._require_keys(
            value,
            {"artifacts", "collections", "delete_items", "publish_intents", "publish_journals", "workspaces"},
        )
        state = _RegistryState()
        for item in cls._require_list(value["workspaces"]):
            cls._require_keys(
                item,
                {"evidence_identity", "generation", "phase", "workspace_id", "workspace_identity"},
            )
            workspace_id = cls._require_string(item["workspace_id"])
            if workspace_id in state.workspaces:
                raise ValueError("duplicate workspace")
            state.workspaces[workspace_id] = WorkspaceClaim(
                workspace_id,
                cls._require_string(item["phase"]),
                cls._require_string(item["generation"]),
                cls._parse_optional_identity(item["workspace_identity"]),
                cls._parse_optional_identity(item["evidence_identity"]),
            )
        for item in cls._require_list(value["collections"]):
            cls._require_keys(item, {"collection", "identity", "workspace_id"})
            workspace_id = cls._require_string(item["workspace_id"])
            name = cls._require_string(item["collection"])
            identities = state.collections.setdefault(workspace_id, {})
            if name in identities:
                raise ValueError("duplicate collection")
            identities[name] = cls._parse_identity(item["identity"])
        for item in cls._require_list(value["artifacts"]):
            claim = cls._parse_artifact(item)
            key = (claim.workspace_id, claim.slot)
            if key in state.artifacts:
                raise ValueError("duplicate artifact")
            state.artifacts[key] = claim
        for item in cls._require_list(value["publish_intents"]):
            intent = cls._parse_intent(item)
            if intent.workspace_id in state.publish_intents:
                raise ValueError("duplicate intent")
            state.publish_intents[intent.workspace_id] = intent
        for item in cls._require_list(value["publish_journals"]):
            journal = cls._parse_journal(item)
            if journal.workspace_id in state.publish_journals:
                raise ValueError("duplicate journal")
            state.publish_journals[journal.workspace_id] = journal
        for item in cls._require_list(value["delete_items"]):
            cls._require_keys(
                item,
                {"claim", "generation", "phase", "quarantine_name", "slot", "workspace_id"},
            )
            claim = cls._parse_artifact(item["claim"])
            delete_item = DeleteJournalItem(
                cls._require_string(item["workspace_id"]),
                cls._require_string(item["slot"]),
                cls._require_string(item["phase"]),
                cls._optional_string(item["quarantine_name"]),
                claim,
                cls._require_string(item["generation"]),
            )
            key = (delete_item.workspace_id, delete_item.slot)
            if key in state.delete_items:
                raise ValueError("duplicate delete")
            state.delete_items[key] = delete_item
        return state

    @classmethod
    def _artifact_json(cls, claim: ArtifactClaim) -> dict[str, Any]:
        return {
            "artifact_id": claim.artifact_id,
            "collection": claim.collection,
            "generation": claim.generation,
            "identity": cls._identity_json(claim.identity),
            "kind": claim.kind,
            "sha256": claim.sha256,
            "size": claim.size,
            "slot": claim.slot,
            "workspace_id": claim.workspace_id,
        }

    @classmethod
    def _parse_artifact(cls, value: Any) -> ArtifactClaim:
        cls._require_keys(
            value,
            {
                "artifact_id",
                "collection",
                "generation",
                "identity",
                "kind",
                "sha256",
                "size",
                "slot",
                "workspace_id",
            },
        )
        return ArtifactClaim(
            cls._require_string(value["workspace_id"]),
            cls._require_string(value["slot"]),
            cls._require_string(value["collection"]),
            cls._require_string(value["kind"]),
            cls._require_string(value["artifact_id"]),
            cls._parse_identity(value["identity"]),
            cls._require_string(value["sha256"]),
            cls._require_int(value["size"]),
            cls._require_string(value["generation"]),
        )

    @classmethod
    def _intent_json(cls, intent: PublishIntent) -> dict[str, Any]:
        return {
            "artifact_id": intent.artifact_id,
            "backup_name": intent.backup_name,
            "collection": intent.collection,
            "expected_sha256": intent.expected_sha256,
            "expected_size": intent.expected_size,
            "generation": intent.generation,
            "kind": intent.kind,
            "old_claim": None if intent.old_claim is None else cls._artifact_json(intent.old_claim),
            "slot": intent.slot,
            "target_name": intent.target_name,
            "temporary_name": intent.temporary_name,
            "temporary_identity": cls._optional_identity_json(intent.temporary_identity),
            "workspace_id": intent.workspace_id,
        }

    @classmethod
    def _parse_intent(cls, value: Any) -> PublishIntent:
        cls._require_keys(
            value,
            {
                "artifact_id",
                "backup_name",
                "collection",
                "expected_sha256",
                "expected_size",
                "generation",
                "kind",
                "old_claim",
                "slot",
                "target_name",
                "temporary_name",
                "temporary_identity",
                "workspace_id",
            },
        )
        old = None if value["old_claim"] is None else cls._parse_artifact(value["old_claim"])
        return PublishIntent(
            cls._require_string(value["workspace_id"]),
            cls._require_string(value["slot"]),
            cls._require_string(value["target_name"]),
            cls._require_string(value["temporary_name"]),
            cls._require_string(value["backup_name"]),
            cls._require_string(value["collection"]),
            cls._require_string(value["kind"]),
            cls._require_string(value["artifact_id"]),
            cls._require_string(value["expected_sha256"]),
            cls._require_int(value["expected_size"]),
            old,
            cls._parse_optional_identity(value["temporary_identity"]),
            cls._require_string(value["generation"]),
        )

    @classmethod
    def _journal_json(cls, journal: PublishJournal) -> dict[str, Any]:
        return {
            "backup_name": journal.backup_name,
            "generation": journal.generation,
            "new_claim": cls._artifact_json(journal.new_claim),
            "old_claim": None if journal.old_claim is None else cls._artifact_json(journal.old_claim),
            "phase": journal.phase,
            "slot": journal.slot,
            "target_name": journal.target_name,
            "temporary_name": journal.temporary_name,
            "workspace_id": journal.workspace_id,
        }

    @classmethod
    def _parse_journal(cls, value: Any) -> PublishJournal:
        cls._require_keys(
            value,
            {
                "backup_name",
                "generation",
                "new_claim",
                "old_claim",
                "phase",
                "slot",
                "target_name",
                "temporary_name",
                "workspace_id",
            },
        )
        old = None if value["old_claim"] is None else cls._parse_artifact(value["old_claim"])
        return PublishJournal(
            cls._require_string(value["workspace_id"]),
            cls._require_string(value["slot"]),
            cls._require_string(value["phase"]),
            cls._require_string(value["target_name"]),
            cls._require_string(value["temporary_name"]),
            cls._require_string(value["backup_name"]),
            cls._parse_artifact(value["new_claim"]),
            old,
            cls._require_string(value["generation"]),
        )

    @staticmethod
    def _identity_json(identity: Identity) -> list[str]:
        return [str(identity[0]), str(identity[1])]

    @classmethod
    def _optional_identity_json(cls, identity: Identity | None) -> list[str] | None:
        return None if identity is None else cls._identity_json(identity)

    @classmethod
    def _parse_identity(cls, value: Any) -> Identity:
        values = cls._require_list(value)
        if len(values) != 2:
            raise ValueError("identity")
        return int(cls._require_string(values[0])), int(cls._require_string(values[1]))

    @classmethod
    def _parse_optional_identity(cls, value: Any) -> Identity | None:
        return None if value is None else cls._parse_identity(value)

    @staticmethod
    def _canonical(value: Any) -> bytes:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")

    @staticmethod
    def _require_keys(value: Any, keys: set[str]) -> None:
        if type(value) is not dict or set(value) != keys:
            raise ValueError("object keys")

    @staticmethod
    def _require_list(value: Any) -> list[Any]:
        if type(value) is not list:
            raise ValueError("list")
        return value

    @staticmethod
    def _require_string(value: Any) -> str:
        if type(value) is not str:
            raise ValueError("string")
        return value

    @classmethod
    def _optional_string(cls, value: Any) -> str | None:
        return None if value is None else cls._require_string(value)

    @staticmethod
    def _require_int(value: Any) -> int:
        if type(value) is not int:
            raise ValueError("integer")
        return value
