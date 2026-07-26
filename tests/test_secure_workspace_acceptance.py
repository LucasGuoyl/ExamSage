from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import httpx
import pytest
from fastapi import FastAPI

from exam_predictor.runtime.coordinator import RuntimeCoordinator
from exam_predictor.runtime.provider_sessions import ProviderSessionRegistry
from exam_predictor.runtime.store import RuntimeStore
from exam_predictor.worker.api import WorkerSettings, create_worker_app
from exam_predictor.workspace.browser_intake import BrowserIntakeWriter
from exam_predictor.workspace.models import WorkspaceJobStatus, WorkspaceState
from exam_predictor.workspace.scanner import WorkspaceScanner
from exam_predictor.workspace.service import WorkspaceService
from exam_predictor.workspace.store import WorkspaceStore
from exam_predictor.workspace.transmission import (
    SourceAuthorizationError,
    WorkspaceTransmissionGate,
)


WORKER_TOKEN = "secure-workspace-acceptance-token"
AUTH = {"X-ExamSage-Token": WORKER_TOKEN}
pytestmark = pytest.mark.anyio

SecretSurface = Literal[
    "database bytes",
    "checkpoint bytes",
    "workspace events",
    "HTTP responses",
    "captured logs",
    "exception chains",
    "temporary artifacts",
    "Git diff stdout",
    "Git diff stderr",
]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _assert_secret_absent(
    secret: str,
    surface: SecretSurface,
    payload: bytes | str,
) -> None:
    detected = (
        secret.encode() in payload
        if isinstance(payload, bytes)
        else secret in payload
    )
    if detected:
        pytest.fail(
            f"Secret sentinel detected in {surface}.",
            pytrace=False,
        )


def _assert_fake_vault_is_only_secret_container(
    vault: FakeVault,
    profile_id: str,
    secret: str,
) -> None:
    if tuple(vault.secrets) != (profile_id,):
        pytest.fail("Fake vault profile set was not the expected test container.", pytrace=False)
    stored = vault.secrets.get(profile_id)
    if not isinstance(stored, str) or not secrets.compare_digest(stored, secret):
        pytest.fail("Fake vault did not retain the expected test credential.", pytrace=False)


def _snapshot_source_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@dataclass
class FakePicker:
    selections: list[Path | None]

    def choose_folder(self) -> Path | None:
        if not self.selections:
            raise AssertionError("The native picker was invoked unexpectedly.")
        return self.selections.pop(0)


@dataclass
class FakeVault:
    secrets: dict[str, str] = field(default_factory=dict)

    def save(self, profile_id: str, api_key: str) -> None:
        self.secrets[profile_id] = api_key

    def load(self, profile_id: str) -> str | None:
        return self.secrets.get(profile_id)

    def exists(self, profile_id: str) -> bool:
        return profile_id in self.secrets

    def delete(self, profile_id: str) -> None:
        self.secrets.pop(profile_id, None)


class FakeProvider:
    name = "acceptance-fake"
    capabilities = SimpleNamespace(chat=True, vision=False, web_search=False)
    models = SimpleNamespace(fast="acceptance-fast", balanced="acceptance-balanced")

    def __init__(self, factory: FakeProviderFactory) -> None:
        self._factory = factory

    def accept_approved_sources(self, sources: tuple[bytes, ...]) -> None:
        assert sources
        self._factory.source_calls += 1


@dataclass
class FakeProviderFactory:
    vault: FakeVault
    creations: int = 0
    source_calls: int = 0

    def __call__(self, config: dict[str, object]) -> FakeProvider:
        supplied = config.get("api_key")
        if not isinstance(supplied, str) or not supplied:
            pytest.fail("Provider factory did not receive a test credential.", pytrace=False)
        if self.vault.secrets:
            matched = any(
                secrets.compare_digest(supplied, stored)
                for stored in self.vault.secrets.values()
            )
            if not matched:
                pytest.fail("Provider factory received an unexpected test credential.", pytrace=False)
        self.creations += 1
        return FakeProvider(self)


@dataclass(frozen=True)
class WorkerComposition:
    app: FastAPI
    workspace_store: WorkspaceStore
    runtime: RuntimeCoordinator
    sessions: ProviderSessionRegistry


def _compose_worker(
    data_dir: Path,
    picker: FakePicker,
    vault: FakeVault,
    provider_factory: FakeProviderFactory,
) -> WorkerComposition:
    workspace_store = WorkspaceStore(data_dir / "workspace.sqlite3")
    sessions = ProviderSessionRegistry(factory=provider_factory)
    runtime = RuntimeCoordinator(
        store=RuntimeStore(data_dir / "agent-runtime.sqlite3"),
        provider_sessions=sessions,
        checkpoints_path=data_dir / "agent-checkpoints.sqlite3",
        vault=vault,
        workspace_repository=workspace_store,
    )
    workspace_service = WorkspaceService(
        store=workspace_store,
        scanner=WorkspaceScanner(),
        picker=picker,
        browser_intake=BrowserIntakeWriter(data_dir / "workspaces"),
        run_guard=runtime,
        close_store_on_shutdown=True,
    )
    settings = WorkerSettings(
        port=8765,
        token=WORKER_TOKEN,
        data_dir=data_dir,
    )
    app = create_worker_app(
        settings,
        runtime=runtime,
        workspace_store=workspace_store,
        workspace_service=workspace_service,
    )
    return WorkerComposition(app, workspace_store, runtime, sessions)


async def _request(
    client: httpx.AsyncClient,
    http_bodies: list[bytes],
    method: str,
    path: str,
    **kwargs: object,
) -> httpx.Response:
    headers = dict(AUTH)
    headers.update(kwargs.pop("headers", {}))
    response = await client.request(method, path, headers=headers, **kwargs)
    http_bodies.append(response.content)
    return response


async def _wait_for_job(
    client: httpx.AsyncClient,
    http_bodies: list[bytes],
    job_id: str,
) -> dict[str, object]:
    deadline = time.monotonic() + 5
    latest: dict[str, object] = {}
    while time.monotonic() < deadline:
        response = await _request(
            client,
            http_bodies,
            "GET",
            f"/v1/workspace-jobs/{job_id}",
        )
        assert response.status_code == 200
        latest = response.json()
        if latest["status"] in {
            WorkspaceJobStatus.SUCCEEDED.value,
            WorkspaceJobStatus.FAILED.value,
            WorkspaceJobStatus.CANCELLED.value,
        }:
            return latest
        await asyncio.sleep(0.01)
    raise AssertionError(f"Workspace job did not reach a terminal state: {latest}")


def _transmit(
    gate: WorkspaceTransmissionGate,
    provider: FakeProvider,
    workspace_id: str,
    entry_ids: list[str],
) -> None:
    approved = gate.authorize(workspace_id, entry_ids)
    contents: list[bytes] = []
    for source in approved:
        with gate.open_approved(source.read_token) as handle:
            contents.append(handle.read())
    provider.accept_approved_sources(tuple(contents))


def _exception_chain(error: BaseException) -> bytes:
    parts: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.extend((type(current).__name__, str(current), repr(current)))
        current = current.__cause__ or current.__context__
    return "\n".join(parts).encode()


async def test_secret_audit_failure_message_is_fixed_and_nonrevealing() -> None:
    test_sentinel = "focused-secret-audit-sentinel"
    with pytest.raises(pytest.fail.Exception) as caught:
        _assert_secret_absent(
            test_sentinel,
            "HTTP responses",
            f"contaminated={test_sentinel}",
        )

    expected = "Secret sentinel detected in HTTP responses."
    if not secrets.compare_digest(str(caught.value), expected):
        pytest.fail("Secret audit helper did not emit its fixed category message.", pytrace=False)


async def test_secure_course_workspace_vertical_slice(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    repository_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    native_root = tmp_path / "mixed-course"
    native_root.mkdir()
    sources = {
        native_root / "course" / "lecture.txt": b"limits and continuity",
        native_root / "course" / "private-notes.md": b"keep this source local",
        native_root / "course" / "recording.mp4": b"unsupported recording",
    }
    for path, content in sources.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    initial_snapshot = _snapshot_source_files(native_root)

    fake_secret = f"acceptance-{secrets.token_urlsafe(32)}"
    vault = FakeVault()
    provider_factory = FakeProviderFactory(vault)
    http_bodies: list[bytes] = []
    event_artifacts: list[bytes] = []
    exception_artifacts: list[bytes] = []

    first = _compose_worker(
        data_dir,
        FakePicker([native_root]),
        vault,
        provider_factory,
    )
    transport = httpx.ASGITransport(app=first.app)
    async with (
        first.app.router.lifespan_context(first.app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        connected = await _request(
            client,
            http_bodies,
            "POST",
            "/v1/providers/connect",
            json={
                "profile": {"profile_id": "primary", "provider": "gemini"},
                "api_key": fake_secret,
            },
        )
        assert connected.status_code == 200
        assert connected.json()["credential_saved"] is True

        selected = await _request(
            client,
            http_bodies,
            "POST",
            "/v1/workspaces/select-folder",
            headers={"Idempotency-Key": "acceptance-select"},
        )
        assert selected.status_code == 202
        job = await _wait_for_job(client, http_bodies, selected.json()["job_id"])
        assert job["status"] == WorkspaceJobStatus.SUCCEEDED.value
        workspace_id = str(job["workspace_id"])

        events = await _request(
            client,
            http_bodies,
            "GET",
            f"/v1/workspace-jobs/{job['job_id']}/events",
        )
        assert events.status_code == 200
        event_artifacts.append(events.content)
        assert events.json()[-1]["event_type"] == "approval_required"

        manifest_response = await _request(
            client,
            http_bodies,
            "GET",
            f"/v1/workspaces/{workspace_id}/manifest",
            params={"limit": 500},
        )
        assert manifest_response.status_code == 200
        manifest = manifest_response.json()
        assert manifest["total"] == 3
        by_path = {item["relative_path"]: item for item in manifest["items"]}
        assert {
            path: (item["state"], item["included"], item["inclusion_reason"])
            for path, item in by_path.items()
        } == {
            "course/lecture.txt": ("pending_approval", True, None),
            "course/private-notes.md": ("pending_approval", True, None),
            "course/recording.mp4": ("excluded", False, "unsupported_format"),
        }
        visible_reasons = tuple(
            item["safe_message"]
            or item["failure_code"]
            or item["inclusion_reason"]
            or str(item["state"]).replace("_", " ")
            for item in manifest["items"]
        )
        assert all(reason.strip() for reason in visible_reasons)
        assert str(native_root) not in "\n".join(visible_reasons)
        workspace_response = await _request(
            client,
            http_bodies,
            "GET",
            f"/v1/workspaces/{workspace_id}",
        )
        assert workspace_response.status_code == 200
        draft_revision_id = workspace_response.json()["current_draft_revision_id"]
        assert isinstance(draft_revision_id, str)

        gate = WorkspaceTransmissionGate(first.workspace_store)
        lecture = by_path["course/lecture.txt"]
        with pytest.raises(SourceAuthorizationError) as unapproved:
            gate.authorize(workspace_id, [lecture["entry_id"]])
        assert unapproved.value.code == "source_approval_required"
        exception_artifacts.append(_exception_chain(unapproved.value))

        excluded = await _request(
            client,
            http_bodies,
            "PATCH",
            (
                f"/v1/workspaces/{workspace_id}/entries/"
                f"{by_path['course/private-notes.md']['entry_id']}"
            ),
            json={
                "revision_id": draft_revision_id,
                "included": False,
                "subtree": False,
            },
        )
        assert excluded.status_code == 200
        draft = excluded.json()
        excluded_by_path = {item["relative_path"]: item for item in draft["entries"]}
        assert excluded_by_path["course/private-notes.md"]["state"] == "excluded"
        assert excluded_by_path["course/private-notes.md"]["inclusion_reason"] == "user_excluded"

        approval_response = await _request(
            client,
            http_bodies,
            "POST",
            f"/v1/workspaces/{workspace_id}/approval",
            json={"revision_id": draft["revision_id"]},
        )
        assert approval_response.status_code == 200
        approval = approval_response.json()
        expected_hash = hashlib.sha256(
            initial_snapshot["course/lecture.txt"]
        ).hexdigest()
        assert approval["entries"] == [
            {"entry_id": lecture["entry_id"], "sha256": expected_hash}
        ]

        for blocked_path in ("course/private-notes.md", "course/recording.mp4"):
            with pytest.raises(SourceAuthorizationError) as blocked:
                gate.authorize(workspace_id, [by_path[blocked_path]["entry_id"]])
            assert blocked.value.code == "source_not_approved"
            exception_artifacts.append(_exception_chain(blocked.value))

        provider = first.sessions.get_provider("primary")
        assert isinstance(provider, FakeProvider)
        _transmit(gate, provider, workspace_id, [lecture["entry_id"]])
        assert provider_factory.source_calls == 1

        approved_path = native_root / "course" / "lecture.txt"
        pre_mutation_snapshot = _snapshot_source_files(native_root)
        assert pre_mutation_snapshot == initial_snapshot
        approved_path.write_bytes(b"mutated after exact-hash approval")
        calls_before_denial = provider_factory.source_calls
        with pytest.raises(SourceAuthorizationError) as changed:
            _transmit(gate, provider, workspace_id, [lecture["entry_id"]])
        assert changed.value.code == "approved_source_changed"
        assert provider_factory.source_calls == calls_before_denial
        exception_artifacts.append(_exception_chain(changed.value))

        approved_path.write_bytes(initial_snapshot["course/lecture.txt"])
        pre_delete_snapshot = _snapshot_source_files(native_root)
        assert pre_delete_snapshot == initial_snapshot

    restarted = _compose_worker(
        data_dir,
        FakePicker([]),
        vault,
        provider_factory,
    )
    transport = httpx.ASGITransport(app=restarted.app)
    async with (
        restarted.app.router.lifespan_context(restarted.app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        workspaces = await _request(client, http_bodies, "GET", "/v1/workspaces")
        assert workspaces.status_code == 200
        assert [item["workspace_id"] for item in workspaces.json()] == [workspace_id]

        detail = await _request(
            client,
            http_bodies,
            "GET",
            f"/v1/workspaces/{workspace_id}",
        )
        assert detail.status_code == 200
        assert detail.json()["state"] == WorkspaceState.NEEDS_ATTENTION.value
        restored_approval = restarted.workspace_store.get_approval(workspace_id)
        assert restored_approval is not None
        assert restored_approval.revision_id == approval["revision_id"]
        assert [(item.entry_id, item.sha256) for item in restored_approval.entries] == [
            (lecture["entry_id"], expected_hash)
        ]

        saved = await _request(client, http_bodies, "GET", "/v1/providers/saved")
        assert saved.status_code == 200
        assert [item["profile"]["profile_id"] for item in saved.json()] == ["primary"]
        assert restarted.sessions.has_provider("primary")
        assert provider_factory.creations == 2

        deleted = await _request(
            client,
            http_bodies,
            "DELETE",
            f"/v1/workspaces/{workspace_id}",
        )
        assert deleted.status_code == 204
        assert (await _request(client, http_bodies, "GET", "/v1/workspaces")).json() == []

    assert _snapshot_source_files(native_root) == pre_delete_snapshot == initial_snapshot
    _assert_fake_vault_is_only_secret_container(vault, "primary", fake_secret)

    database_paths = sorted(data_dir.glob("*.sqlite3*"))
    assert data_dir / "workspace.sqlite3" in database_paths
    assert data_dir / "agent-runtime.sqlite3" in database_paths
    checkpoint_paths = sorted(data_dir.glob("agent-checkpoints.sqlite3*"))
    assert data_dir / "agent-checkpoints.sqlite3" in checkpoint_paths
    for path in database_paths:
        _assert_secret_absent(fake_secret, "database bytes", path.read_bytes())
    for path in checkpoint_paths:
        _assert_secret_absent(fake_secret, "checkpoint bytes", path.read_bytes())
    for payload in event_artifacts:
        _assert_secret_absent(fake_secret, "workspace events", payload)
    for payload in http_bodies:
        _assert_secret_absent(fake_secret, "HTTP responses", payload)
    _assert_secret_absent(fake_secret, "captured logs", caplog.text)
    for payload in exception_artifacts:
        _assert_secret_absent(fake_secret, "exception chains", payload)

    artifact_paths = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    assert artifact_paths
    for path in artifact_paths:
        _assert_secret_absent(fake_secret, "temporary artifacts", path.read_bytes())
    git_diff = subprocess.run(
        ["git", "-C", str(repository_root), "diff", "--binary", "--no-ext-diff"],
        check=True,
        capture_output=True,
    )
    _assert_secret_absent(fake_secret, "Git diff stdout", git_diff.stdout)
    _assert_secret_absent(fake_secret, "Git diff stderr", git_diff.stderr)
