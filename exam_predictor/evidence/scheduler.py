"""Bounded, resumable scheduling for prepared evidence parts."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
import hashlib
import json
import time
from typing import ContextManager, Protocol

from exam_predictor.evidence.artifacts import EvidenceArtifactStore
from exam_predictor.evidence.models import EvidenceUnit, PartState, SourcePartPlan
from exam_predictor.evidence.policy import EvidencePolicy
from exam_predictor.evidence.providers import (
    AnalyzeSourcePartRequest,
    EvidencePartResult,
    EvidenceProviderError,
    EvidenceRouteIdentity,
)
from exam_predictor.evidence.store import EvidencePartClaim, EvidenceStore
from exam_predictor.runtime.control import RunControlRegistry


class SchedulerStatus(StrEnum):
    COMPLETE = "complete"
    PAUSED = "paused"


@dataclass(frozen=True)
class SchedulerOutcome:
    status: SchedulerStatus
    processed_part_ids: tuple[str, ...] = ()
    failed_part_ids: tuple[str, ...] = ()
    pending_count: int = 0


class SchedulerProvider(Protocol):
    def route_identity(self, model_route: str) -> EvidenceRouteIdentity: ...

    def analyze_source_part(self, request: AnalyzeSourcePartRequest) -> EvidencePartResult: ...


EvidenceValidator = Callable[[EvidencePartResult, SourcePartPlan], EvidenceUnit]
EventEmitter = Callable[[str, dict[str, object]], None]
PartAuthorizationGuard = Callable[[SourcePartPlan], ContextManager[object]]


class EvidenceAuthorizationError(RuntimeError):
    pass


class _EvidencePublicationError(RuntimeError):
    pass


@dataclass(frozen=True)
class _PartOutcome:
    part_id: str
    state: PartState


class EvidenceScheduler:
    """Run only a bounded claimed frontier and persist every safe transition."""

    def __init__(
        self,
        store: EvidenceStore,
        artifact_store: EvidenceArtifactStore,
        provider: SchedulerProvider,
        validator: EvidenceValidator,
        controls: RunControlRegistry,
        *,
        emit: EventEmitter | None = None,
        policy: EvidencePolicy = EvidencePolicy(),
        wall_clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        wait: Callable[[float], None] | None = None,
    ) -> None:
        self._store = store
        self._artifact_store = artifact_store
        self._provider = provider
        self._validator = validator
        self._controls = controls
        self._emit_callback = emit or (lambda _event_type, _payload: None)
        self._policy = policy
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._wait = wait or time.sleep

    def run_frontier(
        self,
        run_id: str,
        workspace_id: str,
        revision_id: str,
        *,
        deadline: float | None = None,
        authorize_part: PartAuthorizationGuard | None = None,
    ) -> SchedulerOutcome:
        boundary = deadline
        if boundary is None:
            boundary = self._monotonic_clock() + self._policy.tool_deadline_seconds
        if self._must_pause(run_id, boundary):
            return self._outcome(SchedulerStatus.PAUSED, workspace_id, revision_id)
        claimed = self._store.claim_parts_with_tokens(
            workspace_id,
            revision_id,
            limit=self._policy.multimodal_concurrency,
            now=self._wall_clock(),
        )
        if not claimed:
            counts = self._store.part_state_counts(workspace_id, revision_id)
            pending = self._pending_count(counts)
            status = SchedulerStatus.COMPLETE if pending == 0 else SchedulerStatus.PAUSED
            return SchedulerOutcome(status=status, pending_count=pending)

        for claim in claimed:
            self._emit("part_planned", claim.plan, {"state": PartState.RUNNING.value})
        futures: dict[str, Future[_PartOutcome]] = {}
        with ThreadPoolExecutor(
            max_workers=self._policy.multimodal_concurrency,
            thread_name_prefix="evidence-part",
        ) as executor:
            for claim in claimed:
                futures[claim.plan.part_id] = executor.submit(
                    self._run_part_guarded,
                    run_id,
                    claim,
                    boundary,
                    authorize_part,
                )
            results = tuple(
                futures[claim.plan.part_id].result() for claim in claimed
            )

        processed = tuple(
            result.part_id for result in results if result.state is PartState.PROCESSED
        )
        failed = tuple(
            result.part_id for result in results if result.state is PartState.FAILED
        )
        counts = self._store.part_state_counts(workspace_id, revision_id)
        pending = self._pending_count(counts)
        self._emit_event(
            "coverage_updated",
            {
                "workspace_id": workspace_id,
                "revision_id": revision_id,
                "processed_count": counts.get(PartState.PROCESSED, 0),
                "pending_count": pending,
                "failed_count": counts.get(PartState.FAILED, 0),
            },
        )
        if processed:
            self._emit_event(
                "initial_map_due",
                {
                    "workspace_id": workspace_id,
                    "revision_id": revision_id,
                    "processed_count": counts.get(PartState.PROCESSED, 0),
                    "pending_count": pending,
                },
            )
        retrying = any(result.state is PartState.RETRY_WAIT for result in results)
        status = (
            SchedulerStatus.PAUSED
            if self._must_pause(run_id, boundary) or failed or retrying
            else SchedulerStatus.COMPLETE
        )
        return SchedulerOutcome(
            status=status,
            processed_part_ids=processed,
            failed_part_ids=failed,
            pending_count=pending,
        )

    def run_to_completion(
        self,
        run_id: str,
        workspace_id: str,
        revision_id: str,
    ) -> SchedulerOutcome:
        deadline = self._monotonic_clock() + self._policy.tool_deadline_seconds
        processed: list[str] = []
        failed: list[str] = []
        while True:
            outcome = self.run_frontier(
                run_id,
                workspace_id,
                revision_id,
                deadline=deadline,
            )
            processed.extend(outcome.processed_part_ids)
            failed.extend(outcome.failed_part_ids)
            if outcome.status is SchedulerStatus.PAUSED:
                return SchedulerOutcome(
                    status=SchedulerStatus.PAUSED,
                    processed_part_ids=tuple(processed),
                    failed_part_ids=tuple(failed),
                    pending_count=outcome.pending_count,
                )
            if outcome.pending_count == 0:
                return SchedulerOutcome(
                    status=SchedulerStatus.COMPLETE,
                    processed_part_ids=tuple(processed),
                    failed_part_ids=tuple(failed),
                    pending_count=0,
                )

    def _run_part_guarded(
        self,
        run_id: str,
        claim: EvidencePartClaim,
        deadline: float,
        authorize_part: PartAuthorizationGuard | None,
    ) -> _PartOutcome:
        token_holder = [claim.claim_token]
        try:
            return self._run_part(
                run_id,
                claim,
                deadline,
                token_holder,
                authorize_part,
            )
        except Exception:
            self._pause_claim(claim.plan, token_holder[0])
            return _PartOutcome(claim.plan.part_id, PartState.RETRY_WAIT)

    def _run_part(
        self,
        run_id: str,
        claim: EvidencePartClaim,
        deadline: float,
        token_holder: list[str],
        authorize_part: PartAuthorizationGuard | None,
    ) -> _PartOutcome:
        part = claim.plan
        claim_token = claim.claim_token
        if self._must_pause(run_id, deadline):
            self._pause_claim(part, claim_token)
            return _PartOutcome(part.part_id, PartState.RETRY_WAIT)
        try:
            route = self._provider.route_identity("balanced")
        except EvidenceProviderError as error:
            return self._record_recoverable_route_error(part, claim_token, error)
        cache_key = self._cache_key(part, route)
        with self._authorization_context(authorize_part, part):
            cached = self._store.reuse_cached_evidence(
                part.part_id,
                cache_key=cache_key,
                claim_token=claim_token,
                completed_at=self._wall_clock(),
            )
        if cached is not None:
            self._emit("part_processed", part, {"cache_hit": True})
            return _PartOutcome(part.part_id, PartState.PROCESSED)

        attempt = self._store.attempt_count(part.part_id) + 1
        route_attempt = 1
        while route_attempt <= self._policy.max_attempts_per_route:
            if self._must_pause(run_id, deadline):
                self._pause_claim(part, claim_token)
                return _PartOutcome(part.part_id, PartState.RETRY_WAIT)
            started_at = self._wall_clock()
            self._emit(
                "part_started",
                part,
                {"attempt": attempt, "model_route": route.model_route},
            )
            try:
                with self._authorization_context(authorize_part, part):
                    request = self._request(part, deadline)
                    if self._must_pause(run_id, deadline):
                        self._pause_claim(part, claim_token)
                        return _PartOutcome(part.part_id, PartState.RETRY_WAIT)
                    provider_result = self._provider.analyze_source_part(request)
                    unit = self._validator(provider_result, part)
                    if self._must_pause(run_id, deadline):
                        self._store.record_attempt(
                            part.part_id,
                            attempt=attempt,
                            route=self._route_key(route),
                            outcome=PartState.RETRY_WAIT,
                            started_at=started_at,
                            finished_at=self._wall_clock(),
                            safe_error_code="provider_result_unpublished",
                            next_attempt_at=self._wall_clock(),
                            claim_token=claim_token,
                        )
                        return _PartOutcome(part.part_id, PartState.RETRY_WAIT)
                    try:
                        self._store.publish_evidence(
                            part.part_id,
                            unit,
                            cache_key=cache_key,
                            claim_token=claim_token,
                            completed_at=self._wall_clock(),
                        )
                    except Exception as error:
                        raise _EvidencePublicationError from error
            except EvidenceAuthorizationError:
                raise
            except _EvidencePublicationError:
                raise
            except EvidenceProviderError as error:
                finished_at = self._wall_clock()
                if (
                    error.retryable
                    and route_attempt < self._policy.max_attempts_per_route
                ):
                    delay = self._retry_delay(error, route_attempt)
                    next_attempt_at = finished_at + timedelta(seconds=delay)
                    self._store.record_attempt(
                        part.part_id,
                        attempt=attempt,
                        route=self._route_key(route),
                        outcome=PartState.RETRY_WAIT,
                        started_at=started_at,
                        finished_at=finished_at,
                        safe_error_code=error.code,
                        next_attempt_at=next_attempt_at,
                        claim_token=claim_token,
                    )
                    self._emit(
                        "part_retrying",
                        part,
                        {
                            "attempt": attempt,
                            "safe_error_code": error.code,
                            "retry_after_seconds": delay,
                        },
                    )
                    if not self._wait_for_retry(run_id, delay, deadline):
                        return _PartOutcome(part.part_id, PartState.RETRY_WAIT)
                    attempt += 1
                    route_attempt += 1
                    resumed = self._store.resume_claim(
                        part.part_id,
                        previous_claim_token=claim_token,
                        attempt=attempt,
                        now=self._wall_clock(),
                    )
                    claim_token = resumed.claim_token
                    token_holder[0] = claim_token
                    continue
                if error.retryable:
                    self._store.record_attempt(
                        part.part_id,
                        attempt=attempt,
                        route=self._route_key(route),
                        outcome=PartState.RETRY_WAIT,
                        started_at=started_at,
                        finished_at=finished_at,
                        safe_error_code=error.code,
                        next_attempt_at=finished_at
                        + timedelta(seconds=self._retry_delay(error, route_attempt)),
                        claim_token=claim_token,
                    )
                    self._emit(
                        "part_retrying",
                        part,
                        {"attempt": attempt, "safe_error_code": error.code},
                    )
                    return _PartOutcome(part.part_id, PartState.RETRY_WAIT)
                return self._record_recoverable_route_error(
                    part,
                    claim_token,
                    error,
                    attempt=attempt,
                    route=route,
                    started_at=started_at,
                    finished_at=finished_at,
                )
            except Exception:
                self._record_failed(
                    part,
                    claim_token,
                    attempt,
                    route,
                    "evidence_validation_failed",
                    started_at,
                    self._wall_clock(),
                )
                return _PartOutcome(part.part_id, PartState.FAILED)

            self._emit(
                "part_processed",
                part,
                {"attempt": attempt, "cache_hit": False},
            )
            return _PartOutcome(part.part_id, PartState.PROCESSED)
        return _PartOutcome(part.part_id, PartState.FAILED)

    @staticmethod
    def _authorization_context(
        authorize_part: PartAuthorizationGuard | None,
        part: SourcePartPlan,
    ) -> ContextManager[object]:
        return nullcontext() if authorize_part is None else authorize_part(part)

    def _request(
        self,
        part: SourcePartPlan,
        deadline: float,
    ) -> AnalyzeSourcePartRequest:
        with self._artifact_store.open_part(part.workspace_id, part.part_id) as stream:
            content = stream.read(self._policy.max_part_bytes + 1)
        return AnalyzeSourcePartRequest(
            source_part_id=part.part_id,
            relative_path=part.relative_path,
            locator=part.locator,
            media_type=part.media_type,
            content_bytes=content,
            content_size_bytes=part.size_bytes,
            content_sha256=part.part_sha256,
            model_route="balanced",
            deadline_seconds=min(
                self._policy.provider_timeout_seconds,
                max(0.001, deadline - self._monotonic_clock()),
            ),
        )

    def _record_recoverable_route_error(
        self,
        part: SourcePartPlan,
        claim_token: str,
        error: EvidenceProviderError,
        *,
        attempt: int | None = None,
        route: EvidenceRouteIdentity | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> _PartOutcome:
        attempt_number = attempt or (self._store.attempt_count(part.part_id) + 1)
        started = started_at or self._wall_clock()
        finished = finished_at or started
        self._store.record_attempt(
            part.part_id,
            attempt=attempt_number,
            route=("provider:route-unavailable" if route is None else self._route_key(route)),
            outcome=PartState.RETRY_WAIT,
            started_at=started,
            finished_at=finished,
            safe_error_code=error.code,
            next_attempt_at=finished,
            claim_token=claim_token,
        )
        self._emit(
            "part_failed",
            part,
            {
                "attempt": attempt_number,
                "safe_error_code": error.code,
                "next_action": "resume",
            },
        )
        return _PartOutcome(part.part_id, PartState.RETRY_WAIT)

    def _record_failed(
        self,
        part: SourcePartPlan,
        claim_token: str,
        attempt: int,
        route: EvidenceRouteIdentity,
        code: str,
        started_at: datetime,
        finished_at: datetime,
    ) -> None:
        self._store.record_attempt(
            part.part_id,
            attempt=attempt,
            route=self._route_key(route),
            outcome=PartState.FAILED,
            started_at=started_at,
            finished_at=finished_at,
            safe_error_code=code,
            claim_token=claim_token,
        )
        self._emit(
            "part_failed",
            part,
            {"attempt": attempt, "safe_error_code": code},
        )

    def _pause_claim(self, part: SourcePartPlan, claim_token: str) -> None:
        try:
            self._store.pause_claim(
                part.part_id,
                claim_token=claim_token,
                paused_at=self._wall_clock(),
            )
        except (KeyError, ValueError):
            pass

    def _retry_delay(
        self,
        error: EvidenceProviderError,
        attempt: int,
    ) -> float:
        exponential = self._policy.retry_backoff_seconds * (2 ** (attempt - 1))
        delay = (
            float(error.retry_after_seconds)
            if error.retry_after_seconds is not None
            else exponential
        )
        return max(0.0, delay)

    def _wait_for_retry(self, run_id: str, delay: float, deadline: float) -> bool:
        target = self._monotonic_clock() + delay
        if target > deadline:
            return False
        while self._monotonic_clock() < target:
            if self._must_pause(run_id, deadline):
                return False
            remaining_retry = target - self._monotonic_clock()
            remaining_job = deadline - self._monotonic_clock()
            if remaining_job <= 0:
                return False
            self._wait(min(0.25, remaining_retry, remaining_job))
        return not self._must_pause(run_id, deadline)

    def _cache_key(
        self,
        part: SourcePartPlan,
        route: EvidenceRouteIdentity,
    ) -> str:
        payload = {
            "capability_fingerprint": route.capability_fingerprint,
            "locator": part.locator,
            "media_type": part.media_type,
            "model_id": route.model_id,
            "part_sha256": part.part_sha256,
            "policy_version": self._policy.policy_version,
            "prompt_version": self._policy.prompt_version,
            "provider": route.provider,
            "schema_version": self._policy.schema_version,
            "source_sha256": part.source_sha256,
            "relative_path": part.relative_path,
            "workspace_id": part.workspace_id,
        }
        return "evidence-cache-v1:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _route_key(route: EvidenceRouteIdentity) -> str:
        model_digest = hashlib.sha256(route.model_id.encode("utf-8")).hexdigest()[:16]
        return f"{route.provider}:{route.model_route}:{model_digest}"

    def _must_pause(self, run_id: str, deadline: float) -> bool:
        return self._controls.is_stop_requested(run_id) or self._monotonic_clock() >= deadline

    def _emit(
        self,
        event_type: str,
        part: SourcePartPlan,
        extra: dict[str, object],
    ) -> None:
        self._emit_event(
            event_type,
            {
                "workspace_id": part.workspace_id,
                "revision_id": part.revision_id,
                "part_id": part.part_id,
                "relative_path": part.relative_path,
                "locator": part.locator,
                **extra,
            },
        )

    def _emit_event(self, event_type: str, payload: dict[str, object]) -> None:
        try:
            self._emit_callback(event_type, payload)
        except Exception:
            pass

    def _outcome(
        self,
        status: SchedulerStatus,
        workspace_id: str,
        revision_id: str,
    ) -> SchedulerOutcome:
        counts = self._store.part_state_counts(workspace_id, revision_id)
        return SchedulerOutcome(
            status=status,
            pending_count=self._pending_count(counts),
        )

    @staticmethod
    def _pending_count(counts: dict[PartState, int]) -> int:
        terminal = {PartState.PROCESSED, PartState.FAILED, PartState.INVALIDATED}
        return sum(count for state, count in counts.items() if state not in terminal)
