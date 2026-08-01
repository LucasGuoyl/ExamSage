from __future__ import annotations

from datetime import UTC, datetime
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from exam_predictor.evidence.models import (
    CoverageItem,
    CoverageSummary,
    EvidenceStatus,
    KnowledgeNode,
    SnapshotStatus,
    StudyMapSnapshot,
)
from exam_predictor.evidence.service import EvidenceInspection, EvidenceServiceError
from exam_predictor.runtime.client import WorkerClient
from exam_predictor.runtime.models import RunStatus
import exam_predictor.worker.api as worker_api
from exam_predictor.worker.api import WorkerSettings, create_worker_app
from exam_predictor.workspace.models import WorkspaceJobStatus
from exam_predictor.workspace.transmission import SourceAuthorizationError


TOKEN = "evidence-worker-token"
WORKSPACE_ID = "8d6f8d1f9ed34b3f9228dcd3cb6290c4"
CLIENT_WORKSPACE_ID = "workspace/evidence-one"
NOW = datetime(2026, 7, 27, tzinfo=UTC)
pytestmark = pytest.mark.anyio


def _coverage() -> CoverageSummary:
    item = CoverageItem(
        topic="week-1/notes.pdf",
        covered=True,
        entry_id="entry-1",
        relative_path="week-1/notes.pdf",
        approved_bytes=12,
        planned_part_count=1,
        processed_part_count=1,
        processed_locators=("pages 1-2",),
    )
    return CoverageSummary(
        items=(item,),
        covered_count=1,
        total_count=1,
        approved_bytes=12,
        part_total_count=1,
        part_processed_count=1,
    )


def _snapshot(coverage: CoverageSummary) -> StudyMapSnapshot:
    return StudyMapSnapshot(
        snapshot_id="snapshot-1",
        workspace_id=WORKSPACE_ID,
        revision_id="revision-1",
        status=SnapshotStatus.COMPLETE,
        nodes=(
            KnowledgeNode(
                node_id="node-1",
                title="Limits",
                focus_score=0.8,
                confidence=0.9,
                evidence_unit_ids=("unit-1",),
            ),
        ),
        coverage=coverage,
        evidence_unit_ids=("unit-1",),
        created_at=NOW,
    )


class FakeEvidenceService:
    def __init__(self) -> None:
        self.coverage = _coverage()
        self.snapshot = _snapshot(self.coverage)
        self.calls: list[str] = []
        self.error: Exception | None = None

    def inspect(self, workspace_id: str) -> EvidenceInspection:
        self.calls.append(workspace_id)
        if self.error is not None:
            raise self.error
        return EvidenceInspection(
            workspace_id=workspace_id,
            revision_id="revision-1",
            approval_id="approval-1",
            approval_required=False,
            approved_source_count=1,
            approved_bytes=12,
            coverage=self.coverage,
            snapshot=self.snapshot,
        )


class FakeRuntime:
    def __init__(self) -> None:
        self.store = SimpleNamespace(list_saved_provider_profiles=lambda: [])

    def start(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class FakeWorkspaceService:
    def start(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


@pytest.fixture
async def evidence_client(tmp_path):
    service = FakeEvidenceService()
    app = create_worker_app(
        WorkerSettings(port=8765, token=TOKEN, data_dir=tmp_path),
        runtime=FakeRuntime(),
        workspace_store=SimpleNamespace(),
        workspace_service=FakeWorkspaceService(),
        evidence_service=service,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            yield client, service


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_evidence_routes_return_typed_public_models(evidence_client):
    client, service = evidence_client
    headers = {"X-ExamSage-Token": TOKEN}

    coverage = await client.get(
        f"/v1/workspaces/{WORKSPACE_ID}/evidence/coverage",
        headers=headers,
    )
    snapshot = await client.get(
        f"/v1/workspaces/{WORKSPACE_ID}/evidence/snapshots/current",
        headers=headers,
    )
    status = await client.get(
        f"/v1/workspaces/{WORKSPACE_ID}/evidence/status",
        headers=headers,
    )

    assert coverage.status_code == snapshot.status_code == status.status_code == 200
    assert CoverageSummary.model_validate(coverage.json()) == service.coverage
    assert StudyMapSnapshot.model_validate(snapshot.json()) == service.snapshot
    assert EvidenceStatus.model_validate(status.json()) == EvidenceStatus(
        workspace_id=WORKSPACE_ID,
        revision_id="revision-1",
        approval_required=False,
        prior_approval_exists=False,
        approved_source_count=1,
        approved_bytes=12,
    )
    assert service.calls == [WORKSPACE_ID, WORKSPACE_ID, WORKSPACE_ID]
    serialized = coverage.text + snapshot.text + status.text
    assert "canonical_root" not in serialized
    assert "content_bytes" not in serialized


@pytest.mark.parametrize(
    "path",
    [
        "/v1/workspaces/private%2Fid/evidence/coverage",
        "/v1/workspaces/private%2Fid/evidence/snapshots/current",
        "/v1/workspaces/private%2Fid/evidence/status",
    ],
)
async def test_evidence_routes_authenticate_before_service_access(
    evidence_client,
    path: str,
):
    client, service = evidence_client

    response = await client.get(path)

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized."}
    assert service.calls == []


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (EvidenceServiceError("workspace_not_found"), 404, "workspace_not_found"),
        (EvidenceServiceError("evidence_not_ready"), 409, "evidence_not_ready"),
        (
            SourceAuthorizationError(
                "approved_source_changed",
                WORKSPACE_ID,
                "entry-private",
            ),
            409,
            "source_approval_revoked",
        ),
        (RuntimeError("C:/private/course secret source text"), 503, "evidence_unavailable"),
    ],
)
async def test_evidence_route_errors_are_stable_and_redacted(
    evidence_client,
    error: Exception,
    expected_status: int,
    expected_code: str,
):
    client, service = evidence_client
    service.error = error

    response = await client.get(
        f"/v1/workspaces/{WORKSPACE_ID}/evidence/coverage",
        headers={"X-ExamSage-Token": TOKEN},
    )

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_code}
    assert "private" not in response.text
    assert "source text" not in response.text


async def test_evidence_routes_report_missing_public_state(evidence_client):
    client, service = evidence_client
    service.coverage = None
    service.snapshot = None

    coverage = await client.get(
        f"/v1/workspaces/{WORKSPACE_ID}/evidence/coverage",
        headers={"X-ExamSage-Token": TOKEN},
    )
    snapshot = await client.get(
        f"/v1/workspaces/{WORKSPACE_ID}/evidence/snapshots/current",
        headers={"X-ExamSage-Token": TOKEN},
    )

    assert coverage.status_code == snapshot.status_code == 409
    assert coverage.json() == snapshot.json() == {"detail": "evidence_not_ready"}


def test_worker_client_returns_typed_evidence_models_and_quotes_workspace_id():
    coverage = _coverage()
    snapshot = _snapshot(coverage)
    status = EvidenceStatus(
        workspace_id=CLIENT_WORKSPACE_ID,
        revision_id="revision-1",
        approval_required=False,
        prior_approval_exists=True,
        approved_source_count=1,
        approved_bytes=12,
    )
    raw_paths: list[bytes] = []

    def respond(request: httpx.Request) -> httpx.Response:
        raw_paths.append(request.url.raw_path)
        if request.url.path.endswith("/evidence/coverage"):
            return httpx.Response(200, json=coverage.model_dump(mode="json"))
        if request.url.path.endswith("/evidence/status"):
            return httpx.Response(200, json=status.model_dump(mode="json"))
        if request.url.path.endswith("/evidence/snapshots/current"):
            return httpx.Response(200, json=snapshot.model_dump(mode="json"))
        raise AssertionError(request.url.path)

    client = WorkerClient(
        "http://127.0.0.1:8765",
        TOKEN,
        transport=httpx.MockTransport(respond),
    )
    try:
        assert client.get_evidence_coverage(CLIENT_WORKSPACE_ID) == coverage
        assert client.get_evidence_status(CLIENT_WORKSPACE_ID) == status
        assert client.get_current_evidence_snapshot(CLIENT_WORKSPACE_ID) == snapshot
    finally:
        client.close()

    assert all(b"workspace%2Fevidence-one" in path for path in raw_paths)


async def test_default_worker_composition_reaches_real_evidence_service(
    tmp_path,
):
    source_root = tmp_path / "course"
    source_root.mkdir()
    source_bytes = b"Limits describe behavior near a point."
    (source_root / "notes.txt").write_bytes(source_bytes)

    class FakeVault:
        def __init__(self) -> None:
            self.secrets: dict[str, str] = {}

        def save(self, profile_id: str, secret: str) -> None:
            self.secrets[profile_id] = secret

        def load(self, profile_id: str) -> str | None:
            return self.secrets.get(profile_id)

        def exists(self, profile_id: str) -> bool:
            return profile_id in self.secrets

        def delete(self, profile_id: str) -> None:
            self.secrets.pop(profile_id, None)

    class FakeTypes:
        class Part:
            @staticmethod
            def from_bytes(*, data, mime_type):
                return SimpleNamespace(data=data, mime_type=mime_type)

        class HttpOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class GenerateContentConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

    class FakeMediaModels:
        def __init__(self, owner) -> None:
            self.owner = owner

        def generate_content(self, *, model, contents, config):
            del model, config
            prefix = contents[0]
            part = contents[1]
            self.owner.analyzed_bytes.append(part.data)
            locator = next(
                line.removeprefix("Locator: ")
                for line in prefix.splitlines()
                if line.startswith("Locator: ")
            )
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "locator": locator,
                        "detected_language": "en",
                        "material_role": "course notes",
                        "headings": ["Limits"],
                        "concepts": ["limit"],
                        "definitions": [],
                        "formulas": [],
                        "procedures": [],
                        "examples": [],
                        "assessment_items": [],
                        "visual_descriptions": [],
                        "ocr_text": [],
                        "limitations": [],
                        "warnings": [],
                        "prompt_injection_indicators": [],
                    }
                )
            )

    class FakeProvider:
        name = "gemini"
        capabilities = SimpleNamespace(
            chat=True,
            vision=True,
            file_understanding=True,
            embeddings=False,
            web_search=False,
            citations=True,
            ephemeral_requests=True,
        )
        models = SimpleNamespace(
            fast="fake-fast",
            balanced="fake-balanced",
            reasoning="fake-reasoning",
            embedding="fake-embedding",
        )
        inline_file_limit_bytes = 1024 * 1024

        def __init__(self) -> None:
            self.analyzed_bytes: list[bytes] = []
            self._genai = SimpleNamespace(types=FakeTypes)
            self.client = SimpleNamespace(models=FakeMediaModels(self))

        def create_chat_completion(self, **kwargs):
            system = kwargs["messages"][0]["content"]
            if system.startswith("Choose exactly one ExamSage kernel tool"):
                content = json.dumps(
                    {
                        "tool": "build_study_map",
                        "arguments": {},
                        "reason": "Build approved course evidence.",
                    }
                )
            elif system.startswith("Synthesize a cited ExamSage study map"):
                payload = json.loads(kwargs["messages"][1]["content"])
                unit_ids = [
                    item["evidence_unit_id"] for item in payload["evidence"]
                ]
                content = json.dumps(
                    {
                        "course_groups": [],
                        "nodes": [
                            {
                                "node_id": "node-limits",
                                "title": "Limits",
                                "focus_score": 0.8,
                                "confidence": 0.9,
                                "evidence_unit_ids": unit_ids,
                                "parent_node_id": None,
                                "prerequisite_node_ids": [],
                                "course_group_id": None,
                            }
                        ],
                        "limitations": [],
                        "evidence_unit_ids": unit_ids,
                    }
                )
            else:
                content = "Evidence-grounded answer."
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    provider = FakeProvider()
    app = create_worker_app(
        WorkerSettings(port=8765, token=TOKEN, data_dir=tmp_path / "data"),
        provider_factory=lambda _config: provider,
        credential_vault=FakeVault(),
        folder_picker=SimpleNamespace(choose_folder=lambda: source_root),
    )
    headers = {"X-ExamSage-Token": TOKEN}

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            connected = await client.post(
                "/v1/providers/connect",
                headers=headers,
                json={
                    "profile": {"profile_id": "primary", "provider": "gemini"},
                    "api_key": "fake-provider-secret",
                },
            )
            selected = await client.post(
                "/v1/workspaces/select-folder",
                headers={**headers, "Idempotency-Key": "select-course"},
            )
            assert connected.status_code == 200
            assert selected.status_code == 202
            job_id = selected.json()["job_id"]
            for _ in range(300):
                job = app.state.workspace_store.get_job(job_id)
                if job.status is WorkspaceJobStatus.SUCCEEDED:
                    break
                await asyncio.sleep(0.01)
            assert job.status is WorkspaceJobStatus.SUCCEEDED
            revision = app.state.workspace_store.get_manifest(job.workspace_id)
            approved = await client.post(
                f"/v1/workspaces/{job.workspace_id}/approval",
                headers=headers,
                json={"revision_id": revision.revision_id},
            )
            assert approved.status_code == 200

            submitted = await client.post(
                "/v1/threads/ignored/messages",
                headers=headers,
                json={
                    "provider_profile_id": "primary",
                    "workspace_id": job.workspace_id,
                    "message": "Build my study map.",
                },
            )
            assert submitted.status_code == 202
            run_id = submitted.json()["run_id"]
            for _ in range(500):
                run = app.state.runtime.store.get_run(run_id)
                if run.status in {RunStatus.COMPLETED, RunStatus.FAILED}:
                    break
                await asyncio.sleep(0.01)
            assert run.status is RunStatus.COMPLETED, app.state.runtime.store.list_events(
                run_id
            )

            coverage = await client.get(
                f"/v1/workspaces/{job.workspace_id}/evidence/coverage",
                headers=headers,
            )
            snapshot = await client.get(
                f"/v1/workspaces/{job.workspace_id}/evidence/snapshots/current",
                headers=headers,
            )

    assert app.state.runtime.evidence_service is app.state.evidence_service
    assert coverage.status_code == snapshot.status_code == 200
    assert coverage.json()["part_processed_count"] == 1
    assert snapshot.json()["status"] == "complete"
    assert provider.analyzed_bytes == [source_bytes]
    evidence_store, artifact_store = app.state.evidence_resources
    assert evidence_store._closed is True
    assert artifact_store._closed is True


async def test_shutdown_timeout_keeps_live_runtime_dependencies_open(
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name
            self.closed = False

        def close(self) -> None:
            self.closed = True
            calls.append(f"{self.name}.close")

    evidence_store = Resource("evidence-store")
    artifact_store = Resource("artifact-store")
    evidence_service = SimpleNamespace(
        start=lambda: (),
        delete_workspace_evidence=lambda _workspace_id: None,
        inspect=lambda _workspace_id: None,
    )

    class StuckRuntime:
        def __init__(self, **kwargs) -> None:
            self.store = kwargs["store"]
            self.provider_sessions = kwargs["provider_sessions"]
            self.controls = SimpleNamespace()
            self.evidence_service = None

        def start(self) -> None:
            calls.append("runtime.start")

        def shutdown(self, *, timeout: float) -> None:
            assert timeout > 90
            calls.append("runtime.shutdown")
            raise TimeoutError("runtime still active")

    class WorkspaceLifecycle:
        def start(self) -> None:
            calls.append("workspace.start")

        def shutdown(self) -> None:
            calls.append("workspace.shutdown")

    monkeypatch.setattr(worker_api, "RuntimeCoordinator", StuckRuntime)
    monkeypatch.setattr(
        worker_api,
        "build_evidence_service",
        lambda **_kwargs: (evidence_service, evidence_store, artifact_store),
    )
    app = create_worker_app(
        WorkerSettings(port=8765, token=TOKEN, data_dir=tmp_path),
        workspace_store=SimpleNamespace(),
        workspace_service=WorkspaceLifecycle(),
        credential_vault=SimpleNamespace(),
    )

    with pytest.raises(TimeoutError, match="runtime still active"):
        async with app.router.lifespan_context(app):
            pass

    assert calls == ["runtime.start", "workspace.start", "runtime.shutdown"]
    assert evidence_store.closed is False
    assert artifact_store.closed is False
