from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from exam_predictor.runtime.models import ProviderProfile, SavedProviderProfile
from exam_predictor.workspace.models import (
    ApprovalRecord,
    ManifestEntry,
    ManifestRevision,
    SourceMode,
    SourceState,
    WorkspaceJob,
    WorkspaceJobStatus,
    WorkspaceRecord,
    WorkspaceState,
)
from exam_predictor.workspace.service import WorkspaceOperationError
from exam_predictor.worker.api import WorkerSettings, create_worker_app


TOKEN = "workspace-worker-token"
NOW = datetime(2026, 7, 21, tzinfo=UTC)
API_SECRET = "workspace-api-secret"
ABSOLUTE_PATH = "C:/private/workspace/source"
pytestmark = pytest.mark.anyio


def _job() -> WorkspaceJob:
    return WorkspaceJob(
        job_id="job-1",
        workspace_id="workspace-1",
        job_kind="scan",
        status=WorkspaceJobStatus.QUEUED,
        idempotency_key="repeatable",
        created_at=NOW,
    )


def _entry(
    entry_id: str = "entry-1",
    relative_path: str = "week-1/notes.pdf",
    state: SourceState = SourceState.PENDING_APPROVAL,
    course: str = "week-1",
) -> ManifestEntry:
    return ManifestEntry(
        entry_id=entry_id,
        workspace_id="workspace-1",
        relative_path=relative_path,
        item_kind="file",
        size_bytes=12,
        state=state,
        included=True,
        proposed_course_group=course,
    )


class FakeWorkspaceStore:
    def __init__(self) -> None:
        self.workspace = WorkspaceRecord(
            workspace_id="workspace-1",
            display_name="Course notes",
            source_mode=SourceMode.BROWSER_SNAPSHOT,
            canonical_root=Path(ABSOLUTE_PATH),
            state=WorkspaceState.APPROVAL_REQUIRED,
            updated_at=NOW,
            current_draft_revision_id="revision-1",
            created_at=NOW,
        )
        self.entries = (_entry(),)

    def list_workspaces(self):
        return [self.workspace]

    def get_workspace(self, workspace_id: str):
        return self.workspace if workspace_id == "workspace-1" else None

    def get_manifest(self, workspace_id: str, revision_id: str | None = None):
        if workspace_id != "workspace-1":
            raise WorkspaceOperationError("workspace_not_found")
        return type(
            "Revision",
            (),
            {"entries": self.entries, "revision_id": revision_id or "revision-1"},
        )()

    def get_manifest_entries(self, workspace_id: str):
        if workspace_id != "workspace-1":
            raise WorkspaceOperationError("workspace_not_found")
        return self.entries

    def get_job(self, job_id: str):
        if job_id != "job-1":
            raise WorkspaceOperationError("workspace_job_not_found")
        return _job()

    def list_job_events(self, job_id: str, after_sequence: int = 0):
        if job_id != "job-1":
            raise WorkspaceOperationError("workspace_job_not_found")
        return []


class FakeWorkspaceService:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.errors: dict[str, Exception] = {}
        self.select_returns_job = False
        self._jobs_by_key: dict[tuple[str, ...], WorkspaceJob] = {}

    def _job_for(self, *key: str) -> WorkspaceJob:
        return self._jobs_by_key.setdefault(
            key, _job().model_copy(update={"idempotency_key": key[-1]})
        )

    def _raise_if_configured(self, operation: str) -> None:
        error = self.errors.get(operation)
        if error is not None:
            raise error

    def select_folder(self, key: str):
        self.calls.append(("select", key))
        self._raise_if_configured("select")
        return self._job_for("select", key) if self.select_returns_job else None

    def create_browser_snapshot(self, name: str, files, key: str):
        self.calls.append(("upload", name, tuple(files), key))
        self._raise_if_configured("upload")
        return self._job_for("upload", name, key)

    def rescan(self, workspace_id: str, key: str):
        self.calls.append(("rescan", workspace_id, key))
        self._raise_if_configured("rescan")
        return self._job_for("rescan", workspace_id, key)

    def set_inclusion(self, workspace_id: str, revision_id: str, entry_id: str, included: bool, subtree: bool):
        self.calls.append(("include", workspace_id, revision_id, entry_id, included, subtree))
        self._raise_if_configured("include")
        return ManifestRevision(
            revision_id=revision_id,
            workspace_id=workspace_id,
            policy_version="workspace-v1",
            entries=(_entry(),),
            created_at=NOW,
        )

    def approve(self, workspace_id: str, revision_id: str):
        self.calls.append(("approve", workspace_id, revision_id))
        self._raise_if_configured("approve")
        return ApprovalRecord(
            approval_id="approval-1",
            workspace_id=workspace_id,
            revision_id=revision_id,
            entries=(),
            policy_version="workspace-v1",
            approved_at=NOW,
        )

    def delete_workspace(self, workspace_id: str):
        self.calls.append(("delete", workspace_id))
        self._raise_if_configured("delete")

    def delete_all_workspaces(self):
        self.calls.append(("delete-all",))
        self._raise_if_configured("delete-all")
        return ()

    def start(self):
        return None

    def shutdown(self):
        return None


class FakeRuntime:
    def __init__(self) -> None:
        self.saved_profiles: list[SavedProviderProfile] = []
        self.store = type("Store", (), {})()
        self.store.list_saved_provider_profiles = lambda: self.saved_profiles
        self.forget_error: Exception | None = None

    def start(self):
        return None

    def shutdown(self):
        return None

    def forget_provider_credential(self, profile_id: str):
        if self.forget_error is not None:
            raise self.forget_error
        if profile_id == "active":
            raise WorkspaceOperationError("provider_in_use")


@pytest.fixture
def workspace_service():
    return FakeWorkspaceService()


@pytest.fixture
def workspace_store():
    return FakeWorkspaceStore()


@pytest.fixture
def runtime():
    return FakeRuntime()


@pytest.fixture
async def client(tmp_path, workspace_service, workspace_store, runtime):
    app = create_worker_app(
        WorkerSettings(port=8765, token=TOKEN, data_dir=tmp_path),
        runtime=runtime,
        workspace_store=workspace_store,
        workspace_service=workspace_service,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as test_client:
            yield test_client


@pytest.fixture
def auth_headers():
    return {"X-ExamSage-Token": TOKEN}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/v1/workspaces/select-folder"),
        ("post", "/v1/workspaces/workspace-1/rescan"),
        ("post", "/v1/workspaces/workspace-1/approval"),
        ("patch", "/v1/workspaces/workspace-1/entries/entry-1"),
        ("delete", "/v1/workspaces/workspace-1"),
        ("delete", "/v1/workspaces"),
        ("delete", "/v1/providers/primary/credential"),
    ],
)
async def test_workspace_mutations_reject_unauthenticated_malformed_bodies_before_parsing(
    client, workspace_service, method, path
):
    response = await client.request(
        method.upper(),
        path,
        content=b"not-json" * 100_000,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 401
    assert workspace_service.calls == []


async def test_browser_snapshot_rejects_unauthenticated_multipart_before_parsing(
    client, workspace_service
):
    response = await client.post(
        "/v1/workspaces/browser-snapshot",
        content=b"not-a-valid-multipart-body" * 100_000,
        headers={"content-type": "multipart/form-data; boundary=broken"},
    )

    assert response.status_code == 401
    assert workspace_service.calls == []


async def test_workspace_routes_return_safe_models_and_delegate_mutations(
    client, auth_headers, workspace_service
):
    listed = await client.get("/v1/workspaces", headers=auth_headers)
    detail = await client.get("/v1/workspaces/workspace-1", headers=auth_headers)
    manifest = await client.get(
        "/v1/workspaces/workspace-1/manifest",
        params={"state": "pending_approval", "course": "week-1", "offset": 0, "limit": 1},
        headers=auth_headers,
    )
    select = await client.post(
        "/v1/workspaces/select-folder",
        headers={**auth_headers, "Idempotency-Key": "repeatable"},
    )
    rescan = await client.post(
        "/v1/workspaces/workspace-1/rescan",
        headers={**auth_headers, "Idempotency-Key": "repeatable"},
    )
    include = await client.patch(
        "/v1/workspaces/workspace-1/entries/entry-1",
        json={"revision_id": "revision-1", "included": False, "subtree": True},
        headers=auth_headers,
    )
    approval = await client.post(
        "/v1/workspaces/workspace-1/approval",
        json={"revision_id": "revision-1"},
        headers=auth_headers,
    )
    delete_all = await client.request(
        "DELETE", "/v1/workspaces", json={"confirmation": "DELETE ALL"}, headers=auth_headers
    )

    assert listed.status_code == detail.status_code == 200
    assert listed.json()[0]["counts"] == {"pending_approval": 1}
    assert "canonical_root" not in repr(listed.json())
    assert ABSOLUTE_PATH not in listed.text
    assert manifest.status_code == 200
    assert manifest.json()["total"] == 1
    assert manifest.json()["items"][0]["relative_path"] == "week-1/notes.pdf"
    assert ABSOLUTE_PATH not in detail.text
    assert "canonical_root" not in repr(detail.json())
    assert select.status_code == 204
    assert rescan.status_code == 202
    assert include.status_code == approval.status_code == 200
    assert delete_all.status_code == 204
    assert ("include", "workspace-1", "revision-1", "entry-1", False, True) in workspace_service.calls


async def test_select_folder_accepts_and_reuses_an_idempotency_key(
    client, auth_headers, workspace_service
):
    workspace_service.select_returns_job = True

    first = await client.post(
        "/v1/workspaces/select-folder",
        headers={**auth_headers, "Idempotency-Key": "select-once"},
    )
    repeated = await client.post(
        "/v1/workspaces/select-folder",
        headers={**auth_headers, "Idempotency-Key": "select-once"},
    )

    assert first.status_code == repeated.status_code == 202
    assert first.json()["job_id"] == repeated.json()["job_id"] == "job-1"
    assert first.json()["idempotency_key"] == "select-once"


async def test_rescan_reuses_the_same_idempotency_job(client, auth_headers):
    first = await client.post(
        "/v1/workspaces/workspace-1/rescan",
        headers={**auth_headers, "Idempotency-Key": "rescan-once"},
    )
    repeated = await client.post(
        "/v1/workspaces/workspace-1/rescan",
        headers={**auth_headers, "Idempotency-Key": "rescan-once"},
    )

    assert first.status_code == repeated.status_code == 202
    assert first.json()["job_id"] == repeated.json()["job_id"] == "job-1"
    assert repeated.json()["idempotency_key"] == "rescan-once"


async def test_manifest_applies_filters_and_returns_bounded_pages(
    client, auth_headers, workspace_store
):
    workspace_store.entries = (
        _entry("entry-1", "week-1/notes.pdf", SourceState.PENDING_APPROVAL, "week-1"),
        _entry("entry-2", "week-1/slides.pdf", SourceState.APPROVED, "week-1"),
        _entry("entry-3", "week-2/notes.pdf", SourceState.PENDING_APPROVAL, "week-2"),
    )

    filtered = await client.get(
        "/v1/workspaces/workspace-1/manifest",
        params={"state": "pending_approval", "course": "week-1", "offset": 0, "limit": 1},
        headers=auth_headers,
    )
    page = await client.get(
        "/v1/workspaces/workspace-1/manifest",
        params={"offset": 1, "limit": 1},
        headers=auth_headers,
    )

    assert filtered.status_code == page.status_code == 200
    assert [entry["entry_id"] for entry in filtered.json()["items"]] == ["entry-1"]
    assert filtered.json()["total"] == 1
    assert [entry["entry_id"] for entry in page.json()["items"]] == ["entry-2"]
    assert page.json()["total"] == 3
    assert page.json()["counts"] == {"pending_approval": 2, "approved": 1}


async def test_workspace_conflicts_have_stable_codes_and_no_sensitive_context(
    client, auth_headers, workspace_service
):
    workspace_service.errors["approve"] = WorkspaceOperationError("stale_manifest")
    stale = await client.post(
        "/v1/workspaces/workspace-1/approval",
        json={"revision_id": "revision-1"},
        headers=auth_headers,
    )
    workspace_service.errors["delete"] = WorkspaceOperationError("workspace_operation_active")
    active = await client.delete("/v1/workspaces/workspace-1", headers=auth_headers)

    assert stale.status_code == active.status_code == 409
    assert stale.json() == {"detail": "stale_manifest"}
    assert active.json() == {"detail": "active_workspace_operation"}
    assert ABSOLUTE_PATH not in stale.text + active.text
    assert API_SECRET not in stale.text + active.text


async def test_workspace_invalid_input_redacts_service_path_and_secret(
    client, auth_headers, workspace_service
):
    workspace_service.errors["upload"] = RuntimeError(
        f"Rejected {ABSOLUTE_PATH} using {API_SECRET}"
    )

    response = await client.post(
        "/v1/workspaces/browser-snapshot",
        data={"display_name": "Course notes", "idempotency_key": "upload-once"},
        files={"files": (ABSOLUTE_PATH, b"notes")},
        headers=auth_headers,
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_workspace_input"}
    assert ABSOLUTE_PATH not in response.text
    assert API_SECRET not in response.text


async def test_workspace_body_validation_and_delete_all_confirmation_are_safe(
    client, auth_headers
):
    malformed_entry = await client.patch(
        "/v1/workspaces/workspace-1/entries/entry-1",
        json={"included": False},
        headers=auth_headers,
    )
    invalid_confirmation = await client.request(
        "DELETE",
        "/v1/workspaces",
        json={"confirmation": "erase everything"},
        headers=auth_headers,
    )

    assert malformed_entry.status_code == invalid_confirmation.status_code == 422
    assert malformed_entry.json() == invalid_confirmation.json() == {"detail": "Invalid request."}
    assert API_SECRET not in malformed_entry.text + invalid_confirmation.text


async def test_saved_providers_and_forget_conflict_are_public_and_stable(
    client, auth_headers, runtime
):
    runtime.saved_profiles = [
        SavedProviderProfile(
            profile=ProviderProfile(profile_id="primary", provider="openai"),
            capabilities={"chat": True, "embeddings": False},
            credential_expected=True,
            reconnect_required=False,
            updated_at=NOW,
        )
    ]

    listed = await client.get("/v1/providers/saved", headers=auth_headers)
    conflict = await client.delete("/v1/providers/active/credential", headers=auth_headers)

    assert listed.status_code == 200
    assert listed.json()[0]["profile"] == {"profile_id": "primary", "provider": "openai", "base_url": None, "fast_model": None, "balanced_model": None, "reasoning_model": None, "embedding_model": None}
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "provider_in_use"}
    assert API_SECRET not in listed.text + conflict.text


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/v1/workspaces/missing", 404),
        ("/v1/workspaces/workspace-1/manifest?limit=501", 422),
        ("/v1/workspaces", 422),
    ],
)
async def test_workspace_routes_have_stable_not_found_and_validation_responses(
    client, auth_headers, path, expected
):
    response = await client.delete(path, headers=auth_headers) if path == "/v1/workspaces" else await client.get(path, headers=auth_headers)
    assert response.status_code == expected
