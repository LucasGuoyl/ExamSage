"""Opt-in live benchmark for one approved synthetic ExamSage course."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from pypdf import PdfReader

from exam_predictor.runtime.client import WorkerClient, WorkerClientError
from exam_predictor.runtime.models import (
    AgentEvent,
    EventType,
    RunStatus,
    SubmitMessageRequest,
)
from exam_predictor.workspace.models import (
    EntryInclusionRequest,
    ManifestPage,
    WorkspaceJobStatus,
)
from scripts.check_secret_patterns import sensitive_report_fields


@dataclass(frozen=True)
class FixtureSummary:
    root: Path
    source_count: int
    total_bytes: int
    pdf_pages: int
    files: tuple[Path, ...]


@dataclass(frozen=True)
class BenchmarkObservation:
    provider: str
    models: tuple[str, ...]
    provider_calls: int
    retries: int
    time_to_activity_seconds: float | None
    time_to_initial_map_seconds: float | None
    time_to_final_map_seconds: float
    coverage_sources: tuple[int, int]
    coverage_parts: tuple[int, int]
    safe_error_codes: tuple[str, ...]


@dataclass
class ProviderOperationLedger:
    """Count durable, logical provider-operation start receipts."""

    logical_calls: int = 0
    source_part_attempts: Counter[str] = field(default_factory=Counter)

    def observe(self, events: Sequence[AgentEvent]) -> None:
        for event in events:
            payload = event.payload
            if payload.get("evidence_event") != "provider_operation_started":
                continue
            self.logical_calls += 1
            if payload.get("operation") != "source_part":
                continue
            source_part_id = payload.get("source_part_id")
            if isinstance(source_part_id, str) and source_part_id:
                self.source_part_attempts[source_part_id] += 1

    @property
    def source_part_retries(self) -> int:
        return sum(max(0, count - 1) for count in self.source_part_attempts.values())


def validate_synthetic_fixture(root: Path) -> FixtureSummary:
    root = Path(root).resolve(strict=True)
    readme = root / "README.md"
    try:
        declaration = readme.read_text(encoding="utf-8")
    except OSError:
        raise ValueError("Synthetic fixture declaration is unavailable.") from None
    if (
        "ExamSage synthetic reference course" not in declaration
        or "CC0-1.0" not in declaration
    ):
        raise ValueError("Fixture must declare the ExamSage synthetic CC0 reference pack.")
    files: list[Path] = []
    total_bytes = 0
    pdf_pages = 0
    for candidate in sorted(root.rglob("*")):
        if candidate == readme:
            continue
        if candidate.is_symlink():
            raise ValueError("Synthetic fixture cannot contain links.")
        if not candidate.is_file():
            continue
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        files.append(resolved)
        total_bytes += resolved.stat().st_size
        if resolved.suffix.casefold() == ".pdf":
            try:
                pdf_pages += len(PdfReader(str(resolved)).pages)
            except Exception:
                raise ValueError("Synthetic fixture contains an invalid PDF.") from None
    if not files:
        raise ValueError("Synthetic fixture contains no course sources.")
    return FixtureSummary(root, len(files), total_bytes, pdf_pages, tuple(files))


def build_benchmark_report(
    *,
    provider_profile: str,
    fixture: FixtureSummary,
    observation: BenchmarkObservation,
) -> dict[str, Any]:
    if not observation.models:
        raise ValueError("Benchmark requires at least one explicit model name.")
    report: dict[str, Any] = {
        "schema_version": "examsage-initial-map-benchmark-v1",
        "recorded_at": datetime.now(UTC).isoformat(),
        "machine": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "provider": {
            "profile_id": provider_profile,
            "name": observation.provider,
            "models": list(observation.models),
            "models_basis": "explicit saved profile routes",
        },
        "fixture": {
            "source_count": fixture.source_count,
            "bytes": fixture.total_bytes,
            "pdf_pages": fixture.pdf_pages,
        },
        "calls": {
            "logical_provider_operations": observation.provider_calls,
            "source_part_retries": observation.retries,
            "measurement": (
                "durable provider-operation start receipts; provider SDK internal "
                "transport retries are not exposed"
            ),
        },
        "timings_seconds": {
            "first_activity": observation.time_to_activity_seconds,
            "initial_map": observation.time_to_initial_map_seconds,
            "final_map": observation.time_to_final_map_seconds,
        },
        "coverage": {
            "sources": {
                "covered": observation.coverage_sources[0],
                "total": observation.coverage_sources[1],
            },
            "parts": {
                "processed": observation.coverage_parts[0],
                "total": observation.coverage_parts[1],
            },
        },
        "estimated_usage": {
            "approved_source_bytes": fixture.total_bytes,
            "processed_source_parts": observation.coverage_parts[0],
            "billing_estimate": None,
        },
        "safe_error_codes": list(observation.safe_error_codes),
    }
    if sensitive_report_fields(report):
        raise ValueError("Benchmark report contains a forbidden field.")
    return report


def _ci_enabled() -> bool:
    value = os.environ.get("CI", "").strip().casefold()
    return value not in {"", "0", "false", "no"}


def _wait_for_job(client: WorkerClient, job_id: str, deadline: float):
    while time.monotonic() < deadline:
        job = client.get_job(job_id)
        if job.status is WorkspaceJobStatus.SUCCEEDED:
            return job
        if job.status is WorkspaceJobStatus.FAILED:
            raise RuntimeError(job.safe_error_code or "workspace_scan_failed")
        time.sleep(0.1)
    raise TimeoutError("workspace_scan_timeout")


def _exclude_archives(
    client: WorkerClient,
    workspace_id: str,
    manifest: ManifestPage,
    revision_id: str,
) -> str:
    """Exclude reference archives from the approved benchmark corpus."""
    current_revision_id = revision_id
    for item in manifest.items:
        if item.item_kind != "file" or not item.relative_path.casefold().endswith(".zip"):
            continue
        revision = client.set_entry_inclusion(
            workspace_id,
            item.entry_id,
            EntryInclusionRequest(
                revision_id=current_revision_id,
                included=False,
            ),
        )
        current_revision_id = revision.revision_id
    return current_revision_id


def _run_live(
    *,
    provider_profile: str,
    fixture: FixtureSummary,
    timeout_seconds: float,
) -> BenchmarkObservation:
    worker_url = os.environ.get("EXAMSAGE_WORKER_URL", "")
    worker_token = os.environ.get("EXAMSAGE_WORKER_TOKEN", "")
    if not worker_url or not worker_token:
        raise RuntimeError("worker_environment_missing")
    client = WorkerClient(worker_url, worker_token)
    started = time.monotonic()
    deadline = started + timeout_seconds
    try:
        saved = next(
            (
                item
                for item in client.list_saved_providers()
                if item.profile.profile_id == provider_profile
            ),
            None,
        )
        if saved is None or saved.reconnect_required:
            raise RuntimeError("provider_profile_not_connected")
        uploads = {
            item.relative_to(fixture.root).as_posix(): item for item in fixture.files
        }
        job = client.upload_directory(
            "ExamSage synthetic benchmark",
            uploads,
            f"benchmark-{uuid4().hex}",
        )
        job = _wait_for_job(client, job.job_id, deadline)
        manifest = client.get_manifest(job.workspace_id, limit=500)
        revision_ids = {item.workspace_id for item in manifest.items}
        if revision_ids != {job.workspace_id}:
            raise RuntimeError("fixture_manifest_invalid")
        workspace = client.get_workspace(job.workspace_id)
        revision_id = workspace.current_draft_revision_id
        if revision_id is None:
            raise RuntimeError("fixture_revision_missing")
        revision_id = _exclude_archives(
            client,
            job.workspace_id,
            manifest,
            revision_id,
        )
        client.approve_workspace(job.workspace_id, revision_id)
        submission = client.submit_message(
            SubmitMessageRequest(
                thread_id=f"benchmark-{uuid4().hex}",
                provider_profile_id=provider_profile,
                workspace_id=job.workspace_id,
                message="Build a complete cited study map for this synthetic course.",
            )
        )
        first_activity: float | None = None
        initial_map: float | None = None
        last_sequence = 0
        operation_ledger = ProviderOperationLedger()
        safe_codes: set[str] = set()
        while time.monotonic() < deadline:
            now = time.monotonic()
            events = client.events_after(submission.run_id, after=last_sequence)
            if events:
                last_sequence = max(item.sequence for item in events)
                operation_ledger.observe(events)
            if first_activity is None and any(
                item.event_type not in {EventType.QUEUED, EventType.STARTED}
                for item in events
            ):
                first_activity = now - started
            for event in events:
                code = event.payload.get("code")
                if isinstance(code, str) and code:
                    safe_codes.add(code)
            try:
                snapshot = client.get_current_evidence_snapshot(job.workspace_id)
            except WorkerClientError:
                snapshot = None
            if snapshot is not None and initial_map is None:
                initial_map = now - started
            run = client.get_run(submission.run_id)
            if run.status is RunStatus.COMPLETED:
                if operation_ledger.logical_calls == 0:
                    raise RuntimeError("benchmark_operation_receipts_missing")
                coverage = client.get_evidence_coverage(job.workspace_id)
                models = tuple(
                    value
                    for value in (
                        saved.profile.fast_model,
                        saved.profile.balanced_model,
                        saved.profile.reasoning_model,
                    )
                    if value
                )
                if not models:
                    raise RuntimeError("benchmark_models_not_explicit")
                return BenchmarkObservation(
                    provider=saved.profile.provider,
                    models=models,
                    provider_calls=operation_ledger.logical_calls,
                    retries=operation_ledger.source_part_retries,
                    time_to_activity_seconds=first_activity,
                    time_to_initial_map_seconds=initial_map,
                    time_to_final_map_seconds=now - started,
                    coverage_sources=(coverage.covered_count, coverage.total_count),
                    coverage_parts=(
                        coverage.part_processed_count,
                        coverage.part_total_count,
                    ),
                    safe_error_codes=tuple(sorted(safe_codes)),
                )
            if run.status in {RunStatus.FAILED, RunStatus.PAUSED}:
                raise RuntimeError(run.error or run.status.value)
            time.sleep(0.25)
        raise TimeoutError("benchmark_timeout")
    finally:
        client.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--provider-profile")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    arguments = parser.parse_args(argv)
    if (
        not arguments.live
        or not arguments.provider_profile
        or arguments.fixture is None
        or arguments.timeout_seconds <= 0
    ):
        print(
            "Live benchmark requires --live, --provider-profile, and --fixture.",
            file=sys.stderr,
        )
        return 2
    if _ci_enabled():
        print("Live benchmark is disabled in CI.", file=sys.stderr)
        return 2
    try:
        fixture = validate_synthetic_fixture(arguments.fixture)
        observation = _run_live(
            provider_profile=arguments.provider_profile,
            fixture=fixture,
            timeout_seconds=arguments.timeout_seconds,
        )
        report = build_benchmark_report(
            provider_profile=arguments.provider_profile,
            fixture=fixture,
            observation=observation,
        )
        serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if arguments.output is None:
            print(serialized)
        else:
            with arguments.output.open("x", encoding="utf-8", newline="\n") as output:
                output.write(serialized + "\n")
    except Exception as error:
        print(f"Benchmark failed safely: {type(error).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
