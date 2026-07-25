from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from exam_predictor.workspace.models import (
    ApprovalRecord,
    ManifestEntry,
    ManifestRevision,
    SourceMode,
    SourceState,
    WorkspaceDetail,
    WorkspaceJob,
    WorkspaceJobStatus,
    WorkspaceState,
)
from exam_predictor.workspace.service import WorkspaceOperationError
from exam_predictor.worker.api import WorkerSettings, create_worker_app


TOKEN = "workspace-worker-token"
NOW = datetime(2026, 7, 21, tzinfo=UTC)
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


def _entry() -> ManifestEntry:
    return ManifestEntry(
        entry_id="entry-1",
        workspace_id="workspace-1",
        relative_path="week-1/notes.pdf",
        item_kind="file",
        size_bytes=12,
        state=SourceState.PENDING_APPROVAL,
        included=True,
        proposed_course_group="week-1",
    )


class FakeWorkspaceStore:
    def __init__(self) -> None:
        self.workspace = WorkspaceDetail(
            workspace_id="workspace-1",
            display_name="Course notes",
            source_mode=SourceMode.BROWSER_SNAPSHOT,
            state=WorkspaceState.APPROVAL_REQUIRED,
            counts={SourceState.PENDING_APPROVAL: 1},
            updated_at=NOW,
            current_draft_revision_id="revision-1",
            created_at=NOW,
        )
        self.entry = _entry()

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
            {"entries": (self.entry,), "revision_id": revision_id or "revision-1"},
        )()

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

    def select_folder(self, key: str):
        self.calls.append(("select", key))
        return None

    def create_browser_snapshot(self, name: str, files, key: str):
        self.calls.append(("upload", name, tuple(files), key))
        return _job()

    def rescan(self, workspace_id: str, key: str):
        self.calls.append(("rescan", workspace_id, key))
        return _job()

    def set_inclusion(self, workspace_id: str, revision_id: str, entry_id: str, included: bool, subtree: bool):
        self.calls.append(("include", workspace_id, revision_id, entry_id, included, subtree))
        return ManifestRevision(
            revision_id=revision_id,
            workspace_id=workspace_id,
            policy_version="workspace-v1",
            entries=(_entry(),),
            created_at=NOW,
        )

    def approve(self, workspace_id: str, revision_id: str):
        self.calls.append(("approve", workspace_id, revision_id))
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

    def delete_all_workspaces(self):
        self.calls.append(("delete-all",))
        return ()

    def start(self):
        return None

    def shutdown(self):
        return None


class FakeRuntime:
    def __init__(self) -> None:
        self.store = type(
            "Store", (), {"list_saved_provider_profiles": lambda _self: []}
        )()

    def start(self):
        return None

    def shutdown(self):
        return None

    def forget_provider_credential(self, profile_id: str):
        if profile_id == "active":
            raise WorkspaceOperationError("provider_in_use")


@pytest.fixture
def workspace_service():
    return FakeWorkspaceService()


@pytest.fixture
async def client(tmp_path, workspace_service):
    app = create_worker_app(
        WorkerSettings(port=8765, token=TOKEN, data_dir=tmp_path),
        runtime=FakeRuntime(),
        workspace_store=FakeWorkspaceStore(),
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
    assert manifest.status_code == 200
    assert manifest.json()["total"] == 1
    assert manifest.json()["items"][0]["relative_path"] == "week-1/notes.pdf"
    assert "canonical_root" not in repr(detail.json())
    assert select.status_code == 204
    assert rescan.status_code == 202
    assert include.status_code == approval.status_code == 200
    assert delete_all.status_code == 204
    assert ("include", "workspace-1", "revision-1", "entry-1", False, True) in workspace_service.calls


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
