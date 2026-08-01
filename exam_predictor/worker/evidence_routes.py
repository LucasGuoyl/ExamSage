from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, status

from exam_predictor.evidence.models import (
    CoverageSummary,
    EvidenceStatus,
    StudyMapSnapshot,
)
from exam_predictor.evidence.service import EvidenceServiceError
from exam_predictor.workspace.transmission import SourceAuthorizationError


@dataclass(frozen=True)
class EvidenceRouterDependencies:
    evidence_service: Any


def _failure(code: str, status_code: int) -> HTTPException:
    return HTTPException(status_code=status_code, detail=code)


def _domain_error(error: BaseException) -> HTTPException:
    if isinstance(error, SourceAuthorizationError):
        return _failure("source_approval_revoked", status.HTTP_409_CONFLICT)
    if isinstance(error, EvidenceServiceError):
        if error.code == "workspace_not_found":
            return _failure("workspace_not_found", status.HTTP_404_NOT_FOUND)
        if error.code in {
            "evidence_not_ready",
            "source_approval_revoked",
        }:
            return _failure(error.code, status.HTTP_409_CONFLICT)
    return _failure("evidence_unavailable", status.HTTP_503_SERVICE_UNAVAILABLE)


def build_evidence_router(dependencies: EvidenceRouterDependencies) -> APIRouter:
    """Expose safe, read-only evidence projections behind the outer token boundary."""

    router = APIRouter(prefix="/v1")

    def inspect(workspace_id: str):
        try:
            return dependencies.evidence_service.inspect(workspace_id)
        except Exception as error:
            raise _domain_error(error) from None

    @router.get(
        "/workspaces/{workspace_id}/evidence/status",
        response_model=EvidenceStatus,
    )
    def get_status(workspace_id: str) -> EvidenceStatus:
        inspection = inspect(workspace_id)
        return EvidenceStatus(
            workspace_id=inspection.workspace_id,
            revision_id=inspection.revision_id,
            approval_required=inspection.approval_required,
            prior_approval_exists=inspection.prior_approval_exists,
            approved_source_count=inspection.approved_source_count,
            approved_bytes=inspection.approved_bytes,
        )

    @router.get(
        "/workspaces/{workspace_id}/evidence/coverage",
        response_model=CoverageSummary,
    )
    def get_coverage(workspace_id: str) -> CoverageSummary:
        coverage = inspect(workspace_id).coverage
        if coverage is None:
            raise _failure("evidence_not_ready", status.HTTP_409_CONFLICT)
        return coverage

    @router.get(
        "/workspaces/{workspace_id}/evidence/snapshots/current",
        response_model=StudyMapSnapshot,
    )
    def get_current_snapshot(workspace_id: str) -> StudyMapSnapshot:
        snapshot = inspect(workspace_id).snapshot
        if snapshot is None:
            raise _failure("evidence_not_ready", status.HTTP_409_CONFLICT)
        return snapshot

    return router
