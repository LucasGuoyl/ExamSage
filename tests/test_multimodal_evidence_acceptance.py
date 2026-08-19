from __future__ import annotations

import asyncio
from collections import Counter
from io import BytesIO
import hashlib
import json
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
import zipfile

import httpx
from PIL import Image, ImageDraw
import pytest
from reportlab.pdfgen import canvas

from exam_predictor.agent import ExamSageAgent
from exam_predictor.evidence.artifacts import EvidenceArtifactStore
from exam_predictor.evidence.models import CoverageSummary, SnapshotStatus
from exam_predictor.runtime.client import WorkerClient
from exam_predictor.runtime.models import RunStatus
from exam_predictor.ui.evidence_view import EvidencePhase, build_evidence_view_model
from exam_predictor.worker.api import WorkerSettings, create_worker_app
from exam_predictor.workspace.models import WorkspaceJobStatus


TOKEN = "multimodal-acceptance-token"
AUTH = {"X-ExamSage-Token": TOKEN}
RAW_SOURCE_MARKER = "raw-source-only-7f39b2e8"
FRONTIER_START_TIMEOUT_SECONDS = 60
pytestmark = pytest.mark.anyio


class SyncASGITransport(httpx.BaseTransport):
    """Let the production sync UI client consume the in-process Worker."""

    def __init__(self, app) -> None:
        self._app = app

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        async def dispatch() -> httpx.Response:
            async with httpx.ASGITransport(app=self._app) as transport:
                response = await transport.handle_async_request(request)
                content = await response.aread()
                return httpx.Response(
                    response.status_code,
                    headers=response.headers,
                    content=content,
                    request=request,
                    extensions=response.extensions,
                )

        return asyncio.run(dispatch())


def _ui_client_projection(app, workspace_id: str):
    client = WorkerClient(
        "http://127.0.0.1:8765",
        TOKEN,
        transport=SyncASGITransport(app),
    )
    try:
        status = client.get_evidence_status(workspace_id)
        coverage = client.get_evidence_coverage(workspace_id)
        snapshot = client.get_current_evidence_snapshot(workspace_id)
    finally:
        client.close()
    assert status.approval_required is False
    return build_evidence_view_model(coverage, snapshot)


class FakeClock:
    def __init__(self) -> None:
        self._value = 0.0
        self._lock = Lock()

    def tick(self, seconds: float = 1.0) -> float:
        with self._lock:
            self._value += seconds
            return self._value

    @property
    def value(self) -> float:
        with self._lock:
            return self._value


class ProviderHarness:
    def __init__(self) -> None:
        self.clock = FakeClock()
        self.first_activity_at: float | None = None
        self.second_frontier_started = Event()
        self.release_second_frontier = Event()
        self.block_after_two_parts = True
        self._lock = Lock()
        self.analysis_started = 0
        self.completed_part_keys: list[str] = []
        self.provider_instances: list[FakeProvider] = []

    def factory(self, config: dict[str, object]) -> "FakeProvider":
        assert config["api_key"] == "test-only-provider-secret"
        provider = FakeProvider(self)
        self.provider_instances.append(provider)
        return provider

    def analyze(self, prefix: str, content: bytes) -> str:
        locator = next(
            line.removeprefix("Locator: ")
            for line in prefix.splitlines()
            if line.startswith("Locator: ")
        )
        with self._lock:
            self.analysis_started += 1
            ordinal = self.analysis_started
            now = self.clock.tick()
            if self.first_activity_at is None:
                self.first_activity_at = now
        if self.block_after_two_parts and ordinal > 2:
            self.second_frontier_started.set()
            if not self.release_second_frontier.wait(timeout=10):
                raise TimeoutError("acceptance provider frontier was not released")
        key = hashlib.sha256(
            prefix.encode("utf-8") + b"\0" + content
        ).hexdigest()
        with self._lock:
            self.completed_part_keys.append(key)
        return json.dumps(
            {
                "locator": locator,
                "detected_language": "en",
                "material_role": "synthetic course evidence",
                "headings": ["Limits", "Continuity"],
                "concepts": ["limits", "continuity"],
                "definitions": [
                    {
                        "term": "continuity",
                        "explanation": "Continuity depends on a limit.",
                    }
                ],
                "formulas": ["lim x->a f(x) = f(a)"],
                "procedures": ["Evaluate the limit before testing continuity."],
                "examples": ["A removable discontinuity."],
                "assessment_items": ["Compute a limit and justify continuity."],
                "visual_descriptions": ["A labelled graph approaching a point."],
                "ocr_text": [],
                "limitations": [],
                "warnings": [],
                "prompt_injection_indicators": [],
            }
        )

    def synthesize(self, payload: dict[str, object]) -> str:
        self.clock.tick()
        evidence = payload["evidence"]
        assert isinstance(evidence, list) and evidence
        unit_ids = [item["evidence_unit_id"] for item in evidence]
        return json.dumps(
            {
                "course_groups": [
                    {
                        "group_id": "calculus",
                        "title": "Synthetic Calculus",
                        "confidence": 0.95,
                        "evidence_unit_ids": unit_ids,
                    }
                ],
                "nodes": [
                    {
                        "node_id": "limits",
                        "title": "Limits",
                        "focus_score": 0.9,
                        "confidence": 0.95,
                        "evidence_unit_ids": unit_ids,
                        "parent_node_id": None,
                        "prerequisite_node_ids": [],
                        "course_group_id": "calculus",
                    },
                    {
                        "node_id": "continuity",
                        "title": "Continuity",
                        "focus_score": 0.8,
                        "confidence": 0.9,
                        "evidence_unit_ids": unit_ids,
                        "parent_node_id": None,
                        "prerequisite_node_ids": ["limits"],
                        "course_group_id": "calculus",
                    },
                ],
                "limitations": [],
                "evidence_unit_ids": unit_ids,
            }
        )


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
    def __init__(self, harness: ProviderHarness) -> None:
        self.harness = harness

    def generate_content(self, *, model, contents, config):
        del model, config
        return SimpleNamespace(
            text=self.harness.analyze(contents[0], contents[1].data)
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
        fast="acceptance-fast",
        balanced="acceptance-balanced",
        reasoning="acceptance-reasoning",
        embedding="acceptance-embedding",
    )
    inline_file_limit_bytes = 48 * 1024 * 1024

    def __init__(self, harness: ProviderHarness) -> None:
        self.harness = harness
        self._genai = SimpleNamespace(types=FakeTypes)
        self.client = SimpleNamespace(models=FakeMediaModels(harness))

    def create_chat_completion(self, **kwargs):
        messages = kwargs["messages"]
        system = messages[0]["content"]
        if system.startswith("Choose exactly one ExamSage kernel tool"):
            content = json.dumps(
                {
                    "tool": "build_study_map",
                    "arguments": {},
                    "reason": "Build the approved synthetic course study map.",
                }
            )
        elif system.startswith("Synthesize a cited ExamSage study map"):
            content = self.harness.synthesize(json.loads(messages[1]["content"]))
        else:
            content = "Evidence-grounded synthetic course answer."
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _archive(files: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def _scan_png() -> bytes:
    image = Image.new("RGB", (640, 180), "white")
    drawing = ImageDraw.Draw(image)
    drawing.text(
        (20, 40),
        "SCANNED NOTE: limits are prerequisites for continuity",
        fill="black",
    )
    drawing.line((50, 140, 300, 70, 560, 90), fill="navy", width=4)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _pptx(image: bytes) -> bytes:
    return _archive(
        {
            "ppt/presentation.xml": b"""
              <p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                <p:sldIdLst><p:sldId id="1" r:id="rId1"/></p:sldIdLst>
              </p:presentation>""",
            "ppt/_rels/presentation.xml.rels": b"""
              <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                <Relationship Id="rId1" Target="slides/slide1.xml"/>
              </Relationships>""",
            "ppt/slides/slide1.xml": b"""
              <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                <a:t>Labelled continuity graph: lim x-&gt;a f(x) = f(a)</a:t>
                <a:blip r:embed="rImg1"/>
              </p:sld>""",
            "ppt/slides/_rels/slide1.xml.rels": b"""
              <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                <Relationship Id="rImg1" Target="../media/graph.png"
                  Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"/>
              </Relationships>""",
            "ppt/media/graph.png": image,
        }
    )


def _xlsx() -> bytes:
    return _archive(
        {
            "xl/workbook.xml": b"""
              <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                <sheets><sheet name="Relationships" sheetId="1" r:id="rId1"/></sheets>
              </workbook>""",
            "xl/_rels/workbook.xml.rels": b"""
              <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                <Relationship Id="rId1" Target="worksheets/sheet1.xml"/>
              </Relationships>""",
            "xl/worksheets/sheet1.xml": b"""
              <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
                <sheetData><row r="1">
                  <c r="A1" t="inlineStr"><is><t>Continuity score</t></is></c>
                  <c r="B1"><f>SUM(1,2)</f><v>3</v></c>
                </row></sheetData>
              </worksheet>""",
        }
    )


def _build_reference_pack(root: Path) -> dict[str, bytes]:
    root.mkdir()
    (root / "syllabus.md").write_text(
        "# Synthetic Calculus\n"
        "Limits are prerequisites for continuity.\n"
        f"Private fixture marker: {RAW_SOURCE_MARKER}.\n",
        encoding="utf-8",
    )
    (root / "past-paper.json").write_text(
        json.dumps(
            {
                "question": "Evaluate the limit, then test continuity.",
                "marks": 10,
            }
        ),
        encoding="utf-8",
    )
    scan = _scan_png()
    (root / "lecture-graph.pptx").write_bytes(_pptx(scan))
    pdf_path = root / "course-notes-120-pages.pdf"
    document = canvas.Canvas(str(pdf_path))
    for page in range(1, 121):
        document.drawString(
            72,
            720,
            f"Synthetic page {page}: limits support continuity proofs.",
        )
        document.showPage()
    document.save()
    (root / "relationship-table.xlsx").write_bytes(_xlsx())
    (root / "scan-note.png").write_bytes(scan)
    with zipfile.ZipFile(root / "safe-preview.zip", "w") as archive:
        archive.writestr(
            "preview.txt",
            "Synthetic optional preview: limits and continuity.",
        )
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.iterdir())
        if path.is_file()
    }


async def _wait_job(app, job_id: str):
    for _ in range(1_000):
        job = app.state.workspace_store.get_job(job_id)
        if job.status in {WorkspaceJobStatus.SUCCEEDED, WorkspaceJobStatus.FAILED}:
            return job
        await asyncio.sleep(0.01)
    raise AssertionError("workspace job did not settle")


async def _wait_run(app, run_id: str, expected: RunStatus):
    for _ in range(2_000):
        run = app.state.runtime.store.get_run(run_id)
        if run.status is expected:
            return run
        if run.status is RunStatus.FAILED and expected is not RunStatus.FAILED:
            raise AssertionError(app.state.runtime.store.list_events(run_id))
        await asyncio.sleep(0.01)
    raise AssertionError(f"run did not reach {expected.value}")


async def _connect(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/providers/connect",
        headers=AUTH,
        json={
            "profile": {"profile_id": "primary", "provider": "gemini"},
            "api_key": "test-only-provider-secret",
        },
    )
    assert response.status_code == 200


async def _submit(client: httpx.AsyncClient, workspace_id: str) -> str:
    response = await client.post(
        "/v1/threads/synthetic-course/messages",
        headers=AUTH,
        json={
            "provider_profile_id": "primary",
            "workspace_id": workspace_id,
            "message": "Build a complete cited study map for this course.",
        },
    )
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


async def _current_snapshot(client: httpx.AsyncClient, workspace_id: str):
    response = await client.get(
        f"/v1/workspaces/{workspace_id}/evidence/snapshots/current",
        headers=AUTH,
    )
    return response


async def test_progressive_multimodal_evidence_survives_full_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source_root = tmp_path / "reference-course"
    original_sources = _build_reference_pack(source_root)
    # Keep the acceptance root below the legacy Win32 MAX_PATH boundary while
    # still exercising the production handle-bound artifact implementation.
    data_dir = tmp_path / "d"
    harness = ProviderHarness()
    vault = FakeVault()
    artifact_errors: list[str] = []

    def forbid_legacy_build(*_args, **_kwargs):
        raise AssertionError("Agent evidence must never invoke legacy build_course")

    monkeypatch.setattr(ExamSageAgent, "build_course", forbid_legacy_build)
    publish_json = EvidenceArtifactStore.publish_json

    def observe_artifact_errors(self, *args, **kwargs):
        try:
            return publish_json(self, *args, **kwargs)
        except Exception as error:
            artifact_errors.append(getattr(error, "code", type(error).__name__))
            raise

    monkeypatch.setattr(EvidenceArtifactStore, "publish_json", observe_artifact_errors)

    settings = WorkerSettings(port=8765, token=TOKEN, data_dir=data_dir)
    app = create_worker_app(
        settings,
        provider_factory=harness.factory,
        credential_vault=vault,
        folder_picker=SimpleNamespace(choose_folder=lambda: source_root),
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            selected = await client.post(
                "/v1/workspaces/select-folder",
                headers={**AUTH, "Idempotency-Key": "reference-select"},
            )
            assert selected.status_code == 202
            job = await _wait_job(app, selected.json()["job_id"])
            assert job.status is WorkspaceJobStatus.SUCCEEDED
            workspace_id = job.workspace_id

            manifest_response = await client.get(
                f"/v1/workspaces/{workspace_id}/manifest",
                headers=AUTH,
            )
            manifest = manifest_response.json()
            archive_entry = next(
                item
                for item in manifest["items"]
                if item["relative_path"] == "safe-preview.zip"
                and item["item_kind"] == "file"
            )
            excluded = await client.patch(
                f"/v1/workspaces/{workspace_id}/entries/{archive_entry['entry_id']}",
                headers=AUTH,
                json={
                    "revision_id": app.state.workspace_store.get_manifest(
                        workspace_id
                    ).revision_id,
                    "included": False,
                },
            )
            assert excluded.status_code == 200
            revision_id = excluded.json()["revision_id"]
            approved = await client.post(
                f"/v1/workspaces/{workspace_id}/approval",
                headers=AUTH,
                json={"revision_id": revision_id},
            )
            assert approved.status_code == 200
            assert len(approved.json()["entries"]) == 6
            await _connect(client)

            submitted_at = harness.clock.value
            run_id = await _submit(client, workspace_id)
            assert await asyncio.to_thread(
                harness.second_frontier_started.wait,
                FRONTIER_START_TIMEOUT_SECONDS,
            ), (
                app.state.runtime.store.get_run(run_id),
                app.state.runtime.store.list_events(run_id),
                artifact_errors,
            )
            for _ in range(1_000):
                initial = await _current_snapshot(client, workspace_id)
                if initial.status_code == 200:
                    break
                await asyncio.sleep(0.01)
            assert initial.status_code == 200
            initial_payload = initial.json()
            assert initial_payload["status"] == SnapshotStatus.INITIAL.value
            initial_at = harness.clock.value
            assert harness.first_activity_at is not None
            assert harness.first_activity_at - submitted_at <= 15
            assert initial_at - submitted_at <= 180

            stopped = await client.post(
                f"/v1/runs/{run_id}/stop",
                headers=AUTH,
            )
            assert stopped.status_code == 200
            harness.release_second_frontier.set()
            await _wait_run(app, run_id, RunStatus.PAUSED)

            assert {
                path.relative_to(source_root).as_posix(): path.read_bytes()
                for path in source_root.iterdir()
                if path.is_file()
            } == original_sources

    harness.block_after_two_parts = False
    restarted = create_worker_app(
        settings,
        provider_factory=harness.factory,
        credential_vault=vault,
        folder_picker=SimpleNamespace(choose_folder=lambda: source_root),
    )
    async with restarted.router.lifespan_context(restarted):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=restarted),
            base_url="http://testserver",
        ) as client:
            assert restarted.state.runtime.store.get_run(run_id).status is RunStatus.PAUSED
            await _connect(client)
            resumed = await client.post(f"/v1/runs/{run_id}/resume", headers=AUTH)
            assert resumed.status_code == 200
            await _wait_run(restarted, run_id, RunStatus.COMPLETED)

            coverage_response = await client.get(
                f"/v1/workspaces/{workspace_id}/evidence/coverage",
                headers=AUTH,
            )
            snapshot_response = await _current_snapshot(client, workspace_id)
            coverage = CoverageSummary.model_validate(coverage_response.json())
            snapshot = snapshot_response.json()
            assert snapshot["status"] == SnapshotStatus.COMPLETE.value
            assert coverage.covered_count == 6
            assert coverage.total_count > coverage.covered_count
            assert coverage.excluded_count == coverage.total_count - 6
            excluded_item = next(
                item
                for item in coverage.items
                if item.relative_path == "safe-preview.zip"
            )
            assert excluded_item.excluded is True
            assert excluded_item.planned_part_count == 0
            assert excluded_item.approved_bytes == 0
            assert excluded_item.influenced_current_snapshot is False
            assert coverage.part_processed_count == coverage.part_total_count
            assert coverage.part_processed_count >= 11
            assert {item["node_id"] for item in snapshot["nodes"]} == {
                "limits",
                "continuity",
            }
            continuity = next(
                item for item in snapshot["nodes"] if item["node_id"] == "continuity"
            )
            assert continuity["prerequisite_node_ids"] == ["limits"]
            assert Counter(harness.completed_part_keys).most_common(1)[0][1] == 1
            ui_projection = await asyncio.to_thread(
                _ui_client_projection,
                restarted,
                workspace_id,
            )
            assert ui_projection.phase is EvidencePhase.COMPLETE
            assert ui_projection.covered_source_count == 6
            assert ui_projection.excluded_source_count == coverage.excluded_count
            assert ui_projection.processed_part_count == coverage.part_total_count

            run_events_response = await client.get(
                f"/v1/runs/{run_id}/events",
                headers=AUTH,
            )
            assert run_events_response.status_code == 200
            run_events = run_events_response.json()
            provider_operations = [
                event
                for event in run_events
                if event["payload"].get("evidence_event")
                == "provider_operation_started"
            ]
            assert any(
                event["payload"].get("operation") == "source_part"
                for event in provider_operations
            )
            assert any(
                event["payload"].get("operation") == "study_map_synthesis"
                for event in provider_operations
            )
            assert any(
                event["payload"].get("operation") == "kernel_planner"
                for event in provider_operations
            )
            public_projection = json.dumps(
                {
                    "events": run_events,
                    "snapshot": snapshot,
                    "coverage": coverage.model_dump(mode="json"),
                },
                sort_keys=True,
            ).encode()
            for forbidden in (
                TOKEN.encode(),
                b"test-only-provider-secret",
                RAW_SOURCE_MARKER.encode(),
            ):
                assert forbidden not in public_projection

            for durable_file in data_dir.rglob("*"):
                if (
                    not durable_file.is_file()
                    or "evidence-artifacts" in durable_file.parts
                ):
                    continue
                durable_bytes = durable_file.read_bytes()
                assert TOKEN.encode() not in durable_bytes
                assert b"test-only-provider-secret" not in durable_bytes
                assert RAW_SOURCE_MARKER.encode() not in durable_bytes

            calls_before_change = len(harness.completed_part_keys)
            changed_syllabus = (
                "# Synthetic Calculus\nLimits remain prerequisites for continuity.\n"
            ).encode()
            (source_root / "syllabus.md").write_bytes(changed_syllabus)
            rescanned = await client.post(
                f"/v1/workspaces/{workspace_id}/rescan",
                headers={**AUTH, "Idempotency-Key": "reference-rescan"},
            )
            assert rescanned.status_code == 202
            rescan_job = await _wait_job(restarted, rescanned.json()["job_id"])
            assert rescan_job.status is WorkspaceJobStatus.SUCCEEDED

            status = await client.get(
                f"/v1/workspaces/{workspace_id}/evidence/status",
                headers=AUTH,
            )
            assert status.status_code == 200
            assert status.json()["approval_required"] is True
            changed_revision = restarted.state.workspace_store.get_manifest(
                workspace_id
            )
            changed_entry = next(
                item
                for item in changed_revision.entries
                if item.relative_path == "syllabus.md"
            )
            acknowledged = await client.patch(
                f"/v1/workspaces/{workspace_id}/entries/{changed_entry.entry_id}",
                headers=AUTH,
                json={
                    "revision_id": changed_revision.revision_id,
                    "included": True,
                },
            )
            assert acknowledged.status_code == 200
            changed_revision_id = acknowledged.json()["revision_id"]
            reapproved = await client.post(
                f"/v1/workspaces/{workspace_id}/approval",
                headers=AUTH,
                json={"revision_id": changed_revision_id},
            )
            assert reapproved.status_code == 200

            changed_run_id = await _submit(client, workspace_id)
            await _wait_run(restarted, changed_run_id, RunStatus.COMPLETED)
            changed_snapshot = await _current_snapshot(client, workspace_id)
            assert changed_snapshot.status_code == 200
            assert changed_snapshot.json()["revision_id"] == changed_revision_id
            assert len(harness.completed_part_keys) == calls_before_change + 1
            assert Counter(harness.completed_part_keys).most_common(1)[0][1] == 1

            deleted = await client.delete(
                f"/v1/workspaces/{workspace_id}",
                headers=AUTH,
            )
            assert deleted.status_code == 204
            missing = await client.get(
                f"/v1/workspaces/{workspace_id}",
                headers=AUTH,
            )
            assert missing.status_code == 404

    expected_after_change = dict(original_sources)
    expected_after_change["syllabus.md"] = changed_syllabus
    assert {
        path.relative_to(source_root).as_posix(): path.read_bytes()
        for path in source_root.iterdir()
        if path.is_file()
    } == expected_after_change
    assert artifact_errors == []
