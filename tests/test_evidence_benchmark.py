from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from exam_predictor.legacy_intake import (
    LegacyIntakeError,
    acquire_legacy_intake_lease,
    cleanup_legacy_intake,
    diagnose_legacy_intake,
)
from scripts.benchmark_initial_map import (
    BenchmarkObservation,
    ProviderOperationLedger,
    _exclude_archives,
    build_benchmark_report,
    main as benchmark_main,
    validate_synthetic_fixture,
)
from scripts.check_secret_patterns import sensitive_report_fields


SESSION_ID = "a" * 32


def test_benchmark_requires_live_opt_in_and_refuses_ci(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "README.md").write_text(
        "# ExamSage synthetic reference course\nLicense: CC0-1.0\n",
        encoding="utf-8",
    )
    (fixture / "syllabus.md").write_text("Synthetic limits", encoding="utf-8")

    assert benchmark_main([]) == 2
    monkeypatch.setenv("CI", "true")
    assert benchmark_main(
        [
            "--live",
            "--provider-profile",
            "primary",
            "--fixture",
            str(fixture),
        ]
    ) == 2


def test_benchmark_report_is_reproducible_and_secret_safe(tmp_path: Path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "README.md").write_text(
        "# ExamSage synthetic reference course\nLicense: CC0-1.0\n",
        encoding="utf-8",
    )
    (fixture / "syllabus.md").write_bytes(b"limits")
    fixture_summary = validate_synthetic_fixture(fixture)
    observation = BenchmarkObservation(
        provider="gemini",
        models=("fast-model", "balanced-model"),
        provider_calls=7,
        retries=1,
        time_to_activity_seconds=1.25,
        time_to_initial_map_seconds=8.5,
        time_to_final_map_seconds=22.0,
        coverage_sources=(1, 1),
        coverage_parts=(6, 6),
        safe_error_codes=(),
    )

    report = build_benchmark_report(
        provider_profile="primary",
        fixture=fixture_summary,
        observation=observation,
    )
    serialized = json.dumps(report, sort_keys=True)

    assert report["fixture"]["source_count"] == 1
    assert report["fixture"]["bytes"] == 6
    assert report["timings_seconds"]["initial_map"] == 8.5
    assert report["coverage"]["parts"] == {"processed": 6, "total": 6}
    assert sensitive_report_fields(report) == ()
    assert "api_key" not in serialized
    assert "source_content" not in serialized


def test_benchmark_rejects_secret_patterns_in_dynamic_report_values(tmp_path: Path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "README.md").write_text(
        "# ExamSage synthetic reference course\nLicense: CC0-1.0\n",
        encoding="utf-8",
    )
    (fixture / "syllabus.md").write_bytes(b"limits")

    with pytest.raises(ValueError, match="forbidden"):
        build_benchmark_report(
            provider_profile="primary",
            fixture=validate_synthetic_fixture(fixture),
            observation=BenchmarkObservation(
                provider="gemini",
                models=("sk-" + "proj-this-must-never-be-persisted",),
                provider_calls=1,
                retries=0,
                time_to_activity_seconds=1.0,
                time_to_initial_map_seconds=2.0,
                time_to_final_map_seconds=3.0,
                coverage_sources=(1, 1),
                coverage_parts=(1, 1),
                safe_error_codes=(),
            ),
        )


def test_provider_operation_ledger_counts_logical_calls_and_repeated_parts():
    ledger = ProviderOperationLedger()
    ledger.observe(
        [
            SimpleNamespace(
                payload={
                    "evidence_event": "provider_operation_started",
                    "operation": "source_part",
                    "source_part_id": "part-1",
                }
            ),
            SimpleNamespace(
                payload={
                    "evidence_event": "provider_operation_started",
                    "operation": "study_map_synthesis",
                }
            ),
            SimpleNamespace(payload={"evidence_event": "coverage_updated"}),
        ]
    )
    ledger.observe(
        [
            SimpleNamespace(
                payload={
                    "evidence_event": "provider_operation_started",
                    "operation": "source_part",
                    "source_part_id": "part-1",
                }
            )
        ]
    )

    assert ledger.logical_calls == 3
    assert ledger.source_part_retries == 1


def test_benchmark_report_requires_explicit_model_names(tmp_path: Path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "README.md").write_text(
        "# ExamSage synthetic reference course\nLicense: CC0-1.0\n",
        encoding="utf-8",
    )
    (fixture / "syllabus.md").write_bytes(b"limits")

    with pytest.raises(ValueError, match="explicit model"):
        build_benchmark_report(
            provider_profile="primary",
            fixture=validate_synthetic_fixture(fixture),
            observation=BenchmarkObservation(
                provider="gemini",
                models=(),
                provider_calls=1,
                retries=0,
                time_to_activity_seconds=1.0,
                time_to_initial_map_seconds=2.0,
                time_to_final_map_seconds=3.0,
                coverage_sources=(1, 1),
                coverage_parts=(1, 1),
                safe_error_codes=(),
            ),
        )


def test_benchmark_excludes_archives_before_approval():
    calls = []

    class Client:
        def set_entry_inclusion(self, workspace_id, entry_id, request):
            calls.append((workspace_id, entry_id, request))
            return SimpleNamespace(revision_id=f"revision-{len(calls) + 1}")

    manifest = SimpleNamespace(
        items=(
            SimpleNamespace(
                relative_path="notes.md",
                item_kind="file",
                entry_id="notes",
            ),
            SimpleNamespace(
                relative_path="safe-preview.zip",
                item_kind="file",
                entry_id="archive",
            ),
        )
    )

    revision_id = _exclude_archives(Client(), "workspace-1", manifest, "revision-1")

    assert revision_id == "revision-2"
    assert [(workspace_id, entry_id) for workspace_id, entry_id, _ in calls] == [
        ("workspace-1", "archive")
    ]
    assert calls[0][2].revision_id == "revision-1"
    assert calls[0][2].included is False


def test_legacy_intake_diagnostic_counts_only_verified_sessions(tmp_path: Path):
    data_root = tmp_path / "examsage-data"
    session = data_root / "intake" / SESSION_ID
    session.mkdir(parents=True)
    (session / "one.pdf").write_bytes(b"123")
    nested = session / "nested"
    nested.mkdir()
    (nested / "two.png").write_bytes(b"4567")
    unknown = data_root / "intake" / "do-not-touch"
    unknown.mkdir()
    (unknown / "native-course.txt").write_bytes(b"native")

    summary = diagnose_legacy_intake(data_root)

    assert summary.session_count == 1
    assert summary.file_count == 2
    assert summary.total_bytes == 7
    assert summary.unknown_entry_count == 1


def test_legacy_intake_cleanup_refuses_active_session(tmp_path: Path):
    data_root = tmp_path / "examsage-data"
    session = data_root / "intake" / SESSION_ID
    session.mkdir(parents=True)
    (session / "source.pdf").write_bytes(b"course")

    with pytest.raises(LegacyIntakeError) as caught:
        cleanup_legacy_intake(
            data_root,
            session_ids=(SESSION_ID,),
            active_session_ids=(SESSION_ID,),
        )

    assert caught.value.code == "legacy_intake_active"
    assert session.is_dir()


def test_legacy_intake_cleanup_refuses_a_session_locked_by_another_process(
    tmp_path: Path,
):
    data_root = tmp_path / "examsage-data"
    session = data_root / "intake" / SESSION_ID
    session.mkdir(parents=True)
    (session / "source.pdf").write_bytes(b"course")
    lease = acquire_legacy_intake_lease(data_root, SESSION_ID)
    try:
        with pytest.raises(LegacyIntakeError) as caught:
            cleanup_legacy_intake(data_root, session_ids=(SESSION_ID,))

        assert caught.value.code == "legacy_intake_active"
        assert session.is_dir()
    finally:
        lease.close()

    result = cleanup_legacy_intake(data_root, session_ids=(SESSION_ID,))
    assert result.deleted_session_ids == (SESSION_ID,)


def test_legacy_intake_lease_is_enforced_in_a_separate_process(tmp_path: Path):
    data_root = tmp_path / "examsage-data"
    session = data_root / "intake" / SESSION_ID
    session.mkdir(parents=True)
    (session / "source.pdf").write_bytes(b"course")
    lease = acquire_legacy_intake_lease(data_root, SESSION_ID)
    program = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from exam_predictor.legacy_intake import LegacyIntakeError, cleanup_legacy_intake",
            "try:",
            "    cleanup_legacy_intake(Path(sys.argv[1]), session_ids=(sys.argv[2],))",
            "except LegacyIntakeError as error:",
            "    print(error.code)",
        )
    )
    try:
        child = subprocess.run(
            [sys.executable, "-c", program, str(data_root), SESSION_ID],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert child.stdout.strip() == "legacy_intake_active"
        assert session.is_dir()
    finally:
        lease.close()


def test_active_lease_survives_session_directory_name_substitution(tmp_path: Path):
    data_root = tmp_path / "examsage-data"
    session = data_root / "intake" / SESSION_ID
    session.mkdir(parents=True)
    (session / "active.pdf").write_bytes(b"active")
    displaced = data_root / "intake" / "displaced-active"
    lease = acquire_legacy_intake_lease(data_root, SESSION_ID)
    session.rename(displaced)
    session.mkdir()
    (session / "replacement.pdf").write_bytes(b"replacement")
    try:
        with pytest.raises(LegacyIntakeError) as caught:
            cleanup_legacy_intake(data_root, session_ids=(SESSION_ID,))

        assert caught.value.code == "legacy_intake_active"
        assert (displaced / "active.pdf").read_bytes() == b"active"
        assert (session / "replacement.pdf").read_bytes() == b"replacement"
    finally:
        lease.close()


def test_cleanup_claim_blocks_new_lease_until_atomic_isolation(tmp_path: Path):
    data_root = tmp_path / "examsage-data"
    session = data_root / "intake" / SESSION_ID
    session.mkdir(parents=True)
    (session / "source.pdf").write_bytes(b"copy")
    observed_codes = []

    def try_to_reactivate(_session_id: str, _path: Path) -> None:
        try:
            acquire_legacy_intake_lease(data_root, SESSION_ID)
        except LegacyIntakeError as error:
            observed_codes.append(error.code)

    result = cleanup_legacy_intake(
        data_root,
        session_ids=(SESSION_ID,),
        before_remove=try_to_reactivate,
    )

    assert result.deleted_session_ids == (SESSION_ID,)
    assert observed_codes == ["legacy_intake_active"]


def test_isolated_legacy_cleanup_recovers_after_process_interruption(tmp_path: Path):
    data_root = tmp_path / "examsage-data"
    session = data_root / "intake" / SESSION_ID
    session.mkdir(parents=True)
    (session / "source.pdf").write_bytes(b"copy")

    def crash_after_isolation(_session_id: str, isolated: Path) -> None:
        assert isolated.name.startswith(".owned-directory-")
        raise RuntimeError("simulated process interruption")

    with pytest.raises(RuntimeError, match="simulated"):
        cleanup_legacy_intake(
            data_root,
            session_ids=(SESSION_ID,),
            after_isolate=crash_after_isolation,
        )

    assert not session.exists()
    interrupted = diagnose_legacy_intake(data_root)
    assert interrupted.session_ids == (SESSION_ID,)
    assert interrupted.total_bytes == 4

    recovered = cleanup_legacy_intake(data_root, session_ids=(SESSION_ID,))

    assert recovered.deleted_session_ids == (SESSION_ID,)
    assert recovered.deleted_bytes == 4
    assert diagnose_legacy_intake(data_root).session_count == 0


@pytest.mark.skipif(os.name != "nt", reason="Windows lease file hardening")
def test_windows_legacy_lease_rejects_hardlinked_lock_file(tmp_path: Path):
    data_root = tmp_path / "examsage-data"
    session = data_root / "intake" / SESSION_ID
    session.mkdir(parents=True)
    external = tmp_path / "external.bin"
    external.write_bytes(b"unchanged")
    lock_path = data_root / "intake" / f".examsage-active-{SESSION_ID}.lock"
    os.link(external, lock_path)

    with pytest.raises(LegacyIntakeError) as caught:
        acquire_legacy_intake_lease(data_root, SESSION_ID)

    assert caught.value.code == "legacy_intake_unverified"
    assert external.read_bytes() == b"unchanged"


@pytest.mark.skipif(os.name != "nt", reason="Windows pinned-name semantics")
def test_windows_active_legacy_lease_name_cannot_be_replaced(tmp_path: Path):
    data_root = tmp_path / "examsage-data"
    session = data_root / "intake" / SESSION_ID
    session.mkdir(parents=True)
    lease = acquire_legacy_intake_lease(data_root, SESSION_ID)
    try:
        with pytest.raises(OSError):
            lock_path = data_root / "intake" / f".examsage-active-{SESSION_ID}.lock"
            lock_path.rename(data_root / "intake" / ".examsage-active.replaced")
    finally:
        lease.close()


def test_legacy_intake_cleanup_rechecks_identity_before_deletion(tmp_path: Path):
    data_root = tmp_path / "examsage-data"
    session = data_root / "intake" / SESSION_ID
    session.mkdir(parents=True)
    (session / "source.pdf").write_bytes(b"original")
    displaced = data_root / "intake" / "displaced-original"

    def substitute(_session_id: str, path: Path) -> None:
        path.rename(displaced)
        path.mkdir()
        (path / "replacement.pdf").write_bytes(b"replacement")

    with pytest.raises(LegacyIntakeError) as caught:
        cleanup_legacy_intake(
            data_root,
            session_ids=(SESSION_ID,),
            before_remove=substitute,
        )

    assert caught.value.code == "legacy_intake_identity_changed"
    assert (displaced / "source.pdf").read_bytes() == b"original"
    assert (session / "replacement.pdf").read_bytes() == b"replacement"


def test_legacy_intake_cleanup_deletes_only_selected_verified_copy(tmp_path: Path):
    data_root = tmp_path / "examsage-data"
    session = data_root / "intake" / SESSION_ID
    session.mkdir(parents=True)
    (session / "source.pdf").write_bytes(b"copy")
    unknown = data_root / "intake" / "native-course"
    unknown.mkdir()
    (unknown / "source.pdf").write_bytes(b"native")

    result = cleanup_legacy_intake(data_root, session_ids=(SESSION_ID,))

    assert result.deleted_session_ids == (SESSION_ID,)
    assert result.deleted_bytes == 4
    assert not session.exists()
    assert (unknown / "source.pdf").read_bytes() == b"native"
