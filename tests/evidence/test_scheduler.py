from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Condition

from exam_predictor.evidence.artifacts import EvidenceArtifactStore
from exam_predictor.evidence.models import (
    EvidenceCitation,
    EvidenceUnit,
    PartState,
    SourcePartPlan,
)
from exam_predictor.evidence.policy import EvidencePolicy
from exam_predictor.evidence.providers import (
    EvidencePartResult,
    EvidenceProviderError,
    EvidenceRouteIdentity,
)
from exam_predictor.evidence.scheduler import EvidenceScheduler, SchedulerStatus
from exam_predictor.evidence.store import EvidenceStore
from exam_predictor.runtime.control import RunControlRegistry


NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)
WORKSPACE_ID = "workspace-scheduler"
REVISION_ID = "revision-scheduler"


def _plan(
    part_id: str,
    *,
    priority: int,
    ordinal: int = 0,
    revision_id: str = REVISION_ID,
    scheduling_class: str = "reference",
) -> SourcePartPlan:
    part_id = f"part_{part_id}_000000000000"
    content = f"content:{part_id}".encode()
    import hashlib

    digest = hashlib.sha256(content).hexdigest()
    return SourcePartPlan(
        part_id=part_id,
        workspace_id=WORKSPACE_ID,
        revision_id=revision_id,
        entry_id=f"entry-{part_id}",
        relative_path=f"course/{part_id}.txt",
        source_sha256="a" * 64,
        part_sha256=digest,
        ordinal=ordinal,
        locator="lines 1-1",
        media_type="text/plain",
        size_bytes=len(content),
        scheduling_class=scheduling_class,
        priority=priority,
        state=PartState.PREPARED,
        idempotency_key=f"prepare:{part_id}",
    )


def _publish_artifacts(artifacts: EvidenceArtifactStore, plans) -> None:
    import hashlib

    for plan in plans:
        content = next(
            candidate
            for candidate in (f"content:{item.part_id}".encode() for item in plans)
            if hashlib.sha256(candidate).hexdigest() == plan.part_sha256
        )
        artifacts.publish_part(
            plan.workspace_id,
            plan.part_id,
            content,
            expected_sha256=plan.part_sha256,
        )


def _validate(result: EvidencePartResult, plan: SourcePartPlan) -> EvidenceUnit:
    unit_id = f"unit-{plan.part_id}"
    return EvidenceUnit(
        evidence_unit_id=unit_id,
        source_part_id=plan.part_id,
        content=f"Validated {result.source_part_id}",
        citations=(
            EvidenceCitation(
                citation_id=f"citation-{plan.part_id}",
                evidence_unit_id=unit_id,
                source_part_id=plan.part_id,
                relative_path=plan.relative_path,
                locator=plan.locator,
            ),
        ),
    )


class _ConcurrentProvider:
    def __init__(self) -> None:
        self._condition = Condition()
        self.active = 0
        self.maximum_active = 0
        self.calls: list[str] = []

    def route_identity(self, model_route: str) -> EvidenceRouteIdentity:
        return EvidenceRouteIdentity(
            provider="openai",
            model_route=model_route,
            model_id="test-model",
        )

    def analyze_source_part(self, request) -> EvidencePartResult:
        with self._condition:
            self.calls.append(request.source_part_id)
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self._condition.notify_all()
            self._condition.wait_for(lambda: self.active >= 2, timeout=1)
            self.active -= 1
            self._condition.notify_all()
        return EvidencePartResult(
            source_part_id=request.source_part_id,
            locator=request.locator,
            provider="openai",
            model_id="test-model",
            prompt_version="source-analysis-v1",
            raw_output="{}",
        )


class _ImmediateProvider(_ConcurrentProvider):
    def analyze_source_part(self, request) -> EvidencePartResult:
        self.calls.append(request.source_part_id)
        return EvidencePartResult(
            source_part_id=request.source_part_id,
            locator=request.locator,
            provider="openai",
            model_id="test-model",
            prompt_version="source-analysis-v1",
            raw_output="{}",
        )


class _FakeClock:
    def __init__(self) -> None:
        self.current = NOW
        self.monotonic_value = 0.0
        self.waits: list[float] = []
        self.after_wait = None

    def wall(self) -> datetime:
        return self.current

    def monotonic(self) -> float:
        return self.monotonic_value

    def wait(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.current += timedelta(seconds=seconds)
        self.monotonic_value += seconds
        if self.after_wait is not None:
            self.after_wait()


def _scheduler(
    tmp_path: Path,
    provider,
    plans: tuple[SourcePartPlan, ...],
    *,
    policy: EvidencePolicy = EvidencePolicy(),
    controls: RunControlRegistry | None = None,
    emit=None,
    clock: _FakeClock | None = None,
):
    store = EvidenceStore(tmp_path / "evidence.sqlite3")
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    artifacts = EvidenceArtifactStore(artifacts_root)
    store.upsert_part_plans(plans)
    _publish_artifacts(artifacts, plans)
    scheduler = EvidenceScheduler(
        store,
        artifacts,
        provider,
        _validate,
        controls or RunControlRegistry(),
        emit=emit,
        policy=policy,
        wall_clock=(clock.wall if clock else None),
        monotonic_clock=(clock.monotonic if clock else None),
        wait=(clock.wait if clock else None),
    )
    return scheduler, store, artifacts


def test_scheduler_bounds_concurrency_and_preserves_claim_priority(tmp_path: Path):
    plans = (
        _plan("syllabus", priority=0, scheduling_class="syllabus"),
        _plan("assessment", priority=1, scheduling_class="assessment"),
        _plan("reference-start", priority=3, ordinal=0),
        _plan("reference-middle", priority=3, ordinal=1),
        _plan("reference-end", priority=3, ordinal=2),
        _plan("supplement", priority=4, scheduling_class="supplemental"),
    )
    provider = _ConcurrentProvider()
    scheduler, store, artifacts = _scheduler(tmp_path, provider, plans)
    try:
        outcome = scheduler.run_to_completion("run-1", WORKSPACE_ID, REVISION_ID)
    finally:
        artifacts.close()
        store.close()

    assert outcome.status is SchedulerStatus.COMPLETE
    assert provider.maximum_active == 2
    assert outcome.processed_part_ids[:5] == tuple(plan.part_id for plan in plans[:5])


def test_retry_after_is_honored_and_third_route_failure_pauses(tmp_path: Path):
    class UnavailableProvider(_ImmediateProvider):
        def analyze_source_part(self, request):
            self.calls.append(request.source_part_id)
            raise EvidenceProviderError(
                "provider_unavailable",
                retryable=True,
                retry_after_seconds=2,
            )

    clock = _FakeClock()
    provider = UnavailableProvider()
    plan = _plan("retry", priority=0)
    scheduler, store, artifacts = _scheduler(
        tmp_path,
        provider,
        (plan,),
        policy=EvidencePolicy(multimodal_concurrency=1),
        clock=clock,
    )
    try:
        outcome = scheduler.run_to_completion("run-1", WORKSPACE_ID, REVISION_ID)
        persisted = store.get_part(plan.part_id)
    finally:
        artifacts.close()
        store.close()

    assert outcome.status is SchedulerStatus.PAUSED
    assert provider.calls == [plan.part_id] * 3
    assert sum(clock.waits) == 4.0
    assert max(clock.waits) <= 0.25
    assert persisted.state is PartState.RETRY_WAIT


def test_stop_after_publication_prevents_next_call_and_resume_skips_completed(tmp_path: Path):
    controls = RunControlRegistry()
    provider = _ImmediateProvider()
    plans = (_plan("first", priority=0), _plan("second", priority=1))
    stopped_once = False

    def emit(event_type, _payload):
        nonlocal stopped_once
        if event_type == "part_processed" and not stopped_once:
            stopped_once = True
            controls.request_stop("run-1")

    scheduler, store, artifacts = _scheduler(
        tmp_path,
        provider,
        plans,
        policy=EvidencePolicy(multimodal_concurrency=1),
        controls=controls,
        emit=emit,
    )
    try:
        paused = scheduler.run_to_completion("run-1", WORKSPACE_ID, REVISION_ID)
        controls.clear_stop("run-1")
        resumed = scheduler.run_to_completion("run-1", WORKSPACE_ID, REVISION_ID)
    finally:
        artifacts.close()
        store.close()

    assert paused.status is SchedulerStatus.PAUSED
    assert resumed.status is SchedulerStatus.COMPLETE
    assert provider.calls == [plans[0].part_id, plans[1].part_id]


def test_unchanged_part_uses_validated_cache_across_revision_without_provider_call(
    tmp_path: Path,
):
    provider = _ImmediateProvider()
    first = _plan("first-revision", priority=0, revision_id="revision-one")
    scheduler, store, artifacts = _scheduler(tmp_path, provider, (first,))
    try:
        first_outcome = scheduler.run_to_completion(
            "run-1", WORKSPACE_ID, "revision-one"
        )
        second = _plan("second-revision", priority=0, revision_id="revision-two").model_copy(
            update={
                "entry_id": first.entry_id,
                "relative_path": first.relative_path,
                "part_sha256": first.part_sha256,
                "size_bytes": first.size_bytes,
            }
        )
        store.upsert_part_plans((second,))
        artifacts.publish_part(
            WORKSPACE_ID,
            second.part_id,
            f"content:{first.part_id}".encode(),
            expected_sha256=second.part_sha256,
        )
        second_outcome = scheduler.run_to_completion(
            "run-2", WORKSPACE_ID, "revision-two"
        )
        persisted = store.get_part(second.part_id)
    finally:
        artifacts.close()
        store.close()

    assert first_outcome.status is SchedulerStatus.COMPLETE
    assert second_outcome.status is SchedulerStatus.COMPLETE
    assert provider.calls == [first.part_id]
    assert persisted.state is PartState.PROCESSED


def test_cache_key_separates_equal_bytes_at_different_relative_paths(tmp_path: Path):
    provider = _ImmediateProvider()
    first = _plan("same-bytes-first", priority=0)
    second = _plan("same-bytes-second", priority=1).model_copy(
        update={
            "source_sha256": first.source_sha256,
            "part_sha256": first.part_sha256,
            "size_bytes": first.size_bytes,
            "locator": first.locator,
        }
    )
    scheduler, store, artifacts = _scheduler(
        tmp_path,
        provider,
        (first, second),
        policy=EvidencePolicy(multimodal_concurrency=1),
    )
    try:
        outcome = scheduler.run_to_completion("run-1", WORKSPACE_ID, REVISION_ID)
    finally:
        artifacts.close()
        store.close()

    assert outcome.status is SchedulerStatus.COMPLETE
    assert provider.calls == [first.part_id, second.part_id]


def test_retry_beyond_job_deadline_remains_resumable_without_waiting(tmp_path: Path):
    class SlowRetryProvider(_ImmediateProvider):
        def analyze_source_part(self, request):
            self.calls.append(request.source_part_id)
            raise EvidenceProviderError(
                "provider_unavailable",
                retryable=True,
                retry_after_seconds=61,
            )

    clock = _FakeClock()
    plan = _plan("deadline-retry", priority=0)
    scheduler, store, artifacts = _scheduler(
        tmp_path,
        SlowRetryProvider(),
        (plan,),
        policy=EvidencePolicy(multimodal_concurrency=1, tool_deadline_seconds=60),
        clock=clock,
    )
    try:
        outcome = scheduler.run_to_completion("run-1", WORKSPACE_ID, REVISION_ID)
        persisted = store.get_part(plan.part_id)
    finally:
        artifacts.close()
        store.close()

    assert outcome.status is SchedulerStatus.PAUSED
    assert persisted.state is PartState.RETRY_WAIT
    assert clock.waits == []


def test_retry_wait_observes_stop_without_waiting_the_full_delay(tmp_path: Path):
    class RetryProvider(_ImmediateProvider):
        def analyze_source_part(self, request):
            self.calls.append(request.source_part_id)
            raise EvidenceProviderError(
                "provider_unavailable",
                retryable=True,
                retry_after_seconds=2,
            )

    controls = RunControlRegistry()
    clock = _FakeClock()
    clock.after_wait = lambda: controls.request_stop("run-1")
    plan = _plan("stopped-retry", priority=0)
    provider = RetryProvider()
    scheduler, store, artifacts = _scheduler(
        tmp_path,
        provider,
        (plan,),
        policy=EvidencePolicy(multimodal_concurrency=1),
        controls=controls,
        clock=clock,
    )
    try:
        outcome = scheduler.run_to_completion("run-1", WORKSPACE_ID, REVISION_ID)
        persisted = store.get_part(plan.part_id)
    finally:
        artifacts.close()
        store.close()

    assert outcome.status is SchedulerStatus.PAUSED
    assert provider.calls == [plan.part_id]
    assert clock.waits == [0.25]
    assert persisted.state is PartState.RETRY_WAIT


def test_event_emitter_failure_never_leaves_a_claim_running(tmp_path: Path):
    plan = _plan("event-failure", priority=0)

    def broken_emitter(_event_type, _payload):
        raise RuntimeError("private event sink sentinel")

    scheduler, store, artifacts = _scheduler(
        tmp_path,
        _ImmediateProvider(),
        (plan,),
        policy=EvidencePolicy(multimodal_concurrency=1),
        emit=broken_emitter,
    )
    try:
        outcome = scheduler.run_to_completion("run-1", WORKSPACE_ID, REVISION_ID)
        persisted = store.get_part(plan.part_id)
    finally:
        artifacts.close()
        store.close()

    assert outcome.status is SchedulerStatus.COMPLETE
    assert persisted.state is PartState.PROCESSED


def test_provider_credential_failure_pauses_and_resume_rechecks_route(tmp_path: Path):
    class ReconnectedProvider(_ImmediateProvider):
        def __init__(self) -> None:
            super().__init__()
            self.connected = False

        def route_identity(self, model_route: str) -> EvidenceRouteIdentity:
            if not self.connected:
                self.connected = True
                raise EvidenceProviderError(
                    "provider_credentials_invalid",
                    retryable=False,
                )
            return super().route_identity(model_route)

    plan = _plan("credentials", priority=0)
    provider = ReconnectedProvider()
    scheduler, store, artifacts = _scheduler(
        tmp_path,
        provider,
        (plan,),
        policy=EvidencePolicy(multimodal_concurrency=1),
    )
    try:
        paused = scheduler.run_to_completion("run-1", WORKSPACE_ID, REVISION_ID)
        resumed = scheduler.run_to_completion("run-2", WORKSPACE_ID, REVISION_ID)
        persisted = store.get_part(plan.part_id)
    finally:
        artifacts.close()
        store.close()

    assert paused.status is SchedulerStatus.PAUSED
    assert resumed.status is SchedulerStatus.COMPLETE
    assert provider.calls == [plan.part_id]
    assert persisted.state is PartState.PROCESSED


def test_scheduler_never_adopts_a_token_from_a_recovered_new_owner(tmp_path: Path):
    class ReclaimingProvider(_ImmediateProvider):
        def __init__(self) -> None:
            super().__init__()
            self.store = None
            self.new_claim = None

        def route_identity(self, model_route: str) -> EvidenceRouteIdentity:
            assert self.store is not None
            assert self.store.recover_unfinished()
            [self.new_claim] = self.store.claim_parts_with_tokens(
                WORKSPACE_ID,
                REVISION_ID,
                limit=1,
                now=datetime(2030, 1, 1, tzinfo=UTC),
            )
            return super().route_identity(model_route)

    plan = _plan("ownership-fence", priority=0)
    provider = ReclaimingProvider()
    scheduler, store, artifacts = _scheduler(
        tmp_path,
        provider,
        (plan,),
        policy=EvidencePolicy(multimodal_concurrency=1),
    )
    provider.store = store
    try:
        outcome = scheduler.run_frontier("run-1", WORKSPACE_ID, REVISION_ID)
        assert provider.new_claim is not None
        assert store.publication_token(plan.part_id) == provider.new_claim.claim_token
        persisted = store.get_part(plan.part_id)
    finally:
        artifacts.close()
        store.close()

    assert outcome.status is SchedulerStatus.PAUSED
    assert provider.calls == []
    assert persisted.state is PartState.RUNNING


def test_stop_after_artifact_read_prevents_provider_call(tmp_path: Path):
    controls = RunControlRegistry()
    provider = _ImmediateProvider()
    plan = _plan("stop-after-read", priority=0)
    scheduler, store, artifacts = _scheduler(
        tmp_path,
        provider,
        (plan,),
        policy=EvidencePolicy(multimodal_concurrency=1),
        controls=controls,
    )
    original_request = scheduler._request

    def request_then_stop(part, deadline):
        request = original_request(part, deadline)
        controls.request_stop("run-1")
        return request

    scheduler._request = request_then_stop
    try:
        outcome = scheduler.run_frontier("run-1", WORKSPACE_ID, REVISION_ID)
        persisted = store.get_part(plan.part_id)
    finally:
        artifacts.close()
        store.close()

    assert outcome.status is SchedulerStatus.PAUSED
    assert provider.calls == []
    assert persisted.state is PartState.RETRY_WAIT


def test_publish_failure_after_retry_pauses_the_rotated_claim(tmp_path: Path):
    class RetryOnceProvider(_ImmediateProvider):
        def analyze_source_part(self, request):
            self.calls.append(request.source_part_id)
            if len(self.calls) == 1:
                raise EvidenceProviderError(
                    "provider_unavailable",
                    retryable=True,
                    retry_after_seconds=0,
                )
            return EvidencePartResult(
                source_part_id=request.source_part_id,
                locator=request.locator,
                provider="openai",
                model_id="test-model",
                prompt_version="source-analysis-v1",
                raw_output="{}",
            )

    clock = _FakeClock()
    provider = RetryOnceProvider()
    plan = _plan("rotated-publish", priority=0)
    scheduler, store, artifacts = _scheduler(
        tmp_path,
        provider,
        (plan,),
        policy=EvidencePolicy(multimodal_concurrency=1),
        clock=clock,
    )

    def fail_publish(*args, **kwargs):
        raise RuntimeError("forced publication failure")

    store.publish_evidence = fail_publish
    try:
        outcome = scheduler.run_frontier("run-1", WORKSPACE_ID, REVISION_ID)
        persisted = store.get_part(plan.part_id)
    finally:
        artifacts.close()
        store.close()

    assert outcome.status is SchedulerStatus.PAUSED
    assert provider.calls == [plan.part_id, plan.part_id]
    assert persisted.state is PartState.RETRY_WAIT
