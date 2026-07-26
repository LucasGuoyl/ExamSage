from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Header, HTTPException, Query, UploadFile, status
from fastapi.responses import Response

from exam_predictor.runtime.coordinator import ProviderProfileInUseError
from exam_predictor.runtime.credential_vault import VaultUnavailableError
from exam_predictor.runtime.models import SavedProviderProfile
from exam_predictor.workspace.browser_intake import BrowserUpload
from exam_predictor.workspace.models import (
    ApprovalRecord,
    ApprovalRequest,
    DeleteAllWorkspacesRequest,
    EntryInclusionRequest,
    ManifestEntry,
    ManifestPage,
    ManifestRevision,
    SourceState,
    WorkspaceDetail,
    WorkspaceEvent,
    WorkspaceJob,
    WorkspaceRecord,
    WorkspaceSummary,
)
from exam_predictor.workspace.service import WorkspaceOperationError
from exam_predictor.workspace.store import (
    ActiveWorkspaceOperationError,
    ManifestNotFoundError,
    StaleManifestError,
    WorkspaceJobNotFoundError,
    WorkspaceNotFoundError,
)


@dataclass(frozen=True)
class WorkspaceRouterDependencies:
    workspace_service: Any
    workspace_store: Any
    runtime: Any


def _failure(code: str, status_code: int) -> HTTPException:
    return HTTPException(status_code=status_code, detail=code)


def _domain_error(error: BaseException) -> HTTPException:
    if isinstance(error, (WorkspaceNotFoundError, ManifestNotFoundError, WorkspaceJobNotFoundError)):
        return _failure("workspace_not_found", status.HTTP_404_NOT_FOUND)
    if isinstance(error, StaleManifestError):
        return _failure("stale_manifest", status.HTTP_409_CONFLICT)
    if isinstance(error, ActiveWorkspaceOperationError):
        return _failure("active_workspace_operation", status.HTTP_409_CONFLICT)
    if isinstance(error, ProviderProfileInUseError):
        return _failure("provider_in_use", status.HTTP_409_CONFLICT)
    if isinstance(error, VaultUnavailableError):
        return _failure("vault_unavailable", status.HTTP_503_SERVICE_UNAVAILABLE)
    if isinstance(error, WorkspaceOperationError):
        code = error.code
        if code in {"workspace_not_found", "manifest_not_found", "workspace_job_not_found"}:
            return _failure(code, status.HTTP_404_NOT_FOUND)
        if code in {"workspace_operation_active", "idempotency_conflict", "workspace_has_unsettled_runs", "provider_in_use"}:
            return _failure("active_workspace_operation" if code == "workspace_operation_active" else code, status.HTTP_409_CONFLICT)
        if code == "stale_manifest":
            return _failure(code, status.HTTP_409_CONFLICT)
        return _failure("invalid_workspace_input", status.HTTP_422_UNPROCESSABLE_CONTENT)
    return _failure("invalid_workspace_input", status.HTTP_422_UNPROCESSABLE_CONTENT)


def _summary_counts(entries: Sequence[ManifestEntry]) -> dict[SourceState, int]:
    return dict(Counter(entry.state for entry in entries))


def _workspace_summary(
    dependencies: WorkspaceRouterDependencies,
    workspace: WorkspaceRecord | WorkspaceSummary,
) -> WorkspaceSummary:
    if isinstance(workspace, WorkspaceSummary):
        return workspace
    try:
        entries = dependencies.workspace_store.get_manifest_entries(
            workspace.workspace_id
        )
    except (WorkspaceNotFoundError, ManifestNotFoundError):
        entries = ()
    return WorkspaceSummary(
        workspace_id=workspace.workspace_id,
        display_name=workspace.display_name,
        source_mode=workspace.source_mode,
        state=workspace.state,
        counts=_summary_counts(entries),
        updated_at=workspace.updated_at,
    )


def _workspace_detail(dependencies: WorkspaceRouterDependencies, workspace_id: str) -> WorkspaceDetail:
    workspace = dependencies.workspace_store.get_workspace(workspace_id)
    if workspace is None:
        raise _failure("workspace_not_found", status.HTTP_404_NOT_FOUND)
    if isinstance(workspace, WorkspaceDetail):
        return workspace
    try:
        entries = dependencies.workspace_store.get_manifest_entries(workspace_id)
    except (WorkspaceNotFoundError, ManifestNotFoundError):
        entries = ()
    return WorkspaceDetail(
        workspace_id=workspace.workspace_id,
        display_name=workspace.display_name,
        source_mode=workspace.source_mode,
        state=workspace.state,
        counts=_summary_counts(entries),
        updated_at=workspace.updated_at,
        current_draft_revision_id=workspace.current_draft_revision_id,
        current_approved_revision_id=workspace.current_approved_revision_id,
        created_at=workspace.created_at,
        last_scanned_at=workspace.last_scanned_at,
        last_access_verified_at=workspace.last_access_verified_at,
    )


def build_workspace_router(dependencies: WorkspaceRouterDependencies) -> APIRouter:
    """Build the token-agnostic workspace routes for the already-authenticated app."""
    router = APIRouter(prefix="/v1")

    @router.post("/workspaces/select-folder", response_model=WorkspaceJob, status_code=status.HTTP_202_ACCEPTED)
    def select_folder(idempotency_key: Annotated[str, Header(alias="Idempotency-Key")]) -> WorkspaceJob | Response:
        try:
            job = dependencies.workspace_service.select_folder(idempotency_key)
        except Exception as error:
            raise _domain_error(error) from None
        return job if job is not None else Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.post("/workspaces/browser-snapshot", response_model=WorkspaceJob, status_code=status.HTTP_202_ACCEPTED)
    async def browser_snapshot(
        display_name: Annotated[str, Form()],
        idempotency_key: Annotated[str, Form()],
        files: Annotated[list[UploadFile], File()],
    ) -> WorkspaceJob:
        uploads = tuple(
            BrowserUpload(
                relative_path=upload.filename or "",
                size_bytes=upload.size if isinstance(upload.size, int) else 0,
                stream=upload.file,
            )
            for upload in files
        )
        try:
            return dependencies.workspace_service.create_browser_snapshot(display_name, uploads, idempotency_key)
        except Exception as error:
            raise _domain_error(error) from None
        finally:
            for upload in files:
                await upload.close()

    @router.get("/workspaces", response_model=list[WorkspaceSummary])
    def list_workspaces() -> Sequence[WorkspaceSummary]:
        return tuple(
            _workspace_summary(dependencies, workspace)
            for workspace in dependencies.workspace_store.list_workspaces()
        )

    @router.get("/workspaces/{workspace_id}", response_model=WorkspaceDetail)
    def get_workspace(workspace_id: str) -> WorkspaceDetail:
        return _workspace_detail(dependencies, workspace_id)

    @router.get("/workspaces/{workspace_id}/manifest", response_model=ManifestPage)
    def get_manifest(
        workspace_id: str,
        state: SourceState | None = None,
        course: str | None = None,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> ManifestPage:
        try:
            revision = dependencies.workspace_store.get_manifest(workspace_id)
        except Exception as error:
            raise _domain_error(error) from None
        entries = tuple(
            entry for entry in revision.entries
            if (state is None or entry.state is state)
            and (course is None or entry.proposed_course_group == course)
        )
        return ManifestPage(
            items=entries[offset : offset + limit],
            total=len(entries),
            offset=offset,
            limit=limit,
            counts=_summary_counts(entries),
        )

    @router.post("/workspaces/{workspace_id}/rescan", response_model=WorkspaceJob, status_code=status.HTTP_202_ACCEPTED)
    def rescan(workspace_id: str, idempotency_key: Annotated[str, Header(alias="Idempotency-Key")]) -> WorkspaceJob:
        try:
            return dependencies.workspace_service.rescan(workspace_id, idempotency_key)
        except Exception as error:
            raise _domain_error(error) from None

    @router.post("/workspaces/{workspace_id}/approval", response_model=ApprovalRecord)
    def approve_workspace(workspace_id: str, request: ApprovalRequest) -> ApprovalRecord:
        try:
            return dependencies.workspace_service.approve(workspace_id, request.revision_id)
        except Exception as error:
            raise _domain_error(error) from None

    @router.patch("/workspaces/{workspace_id}/entries/{entry_id}", response_model=ManifestRevision)
    def set_entry_inclusion(workspace_id: str, entry_id: str, request: EntryInclusionRequest) -> ManifestRevision:
        try:
            return dependencies.workspace_service.set_inclusion(workspace_id, request.revision_id, entry_id, request.included, request.subtree)
        except Exception as error:
            raise _domain_error(error) from None

    @router.delete("/workspaces/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_workspace(workspace_id: str) -> Response:
        try:
            dependencies.workspace_service.delete_workspace(workspace_id)
        except Exception as error:
            raise _domain_error(error) from None
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.delete("/workspaces", status_code=status.HTTP_204_NO_CONTENT)
    def delete_all_workspaces(request: DeleteAllWorkspacesRequest) -> Response:
        try:
            conflicts = dependencies.workspace_service.delete_all_workspaces()
        except Exception as error:
            raise _domain_error(error) from None
        if conflicts:
            raise _failure("active_workspace_operation", status.HTTP_409_CONFLICT)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.get("/workspace-jobs/{job_id}", response_model=WorkspaceJob)
    def get_job(job_id: str) -> WorkspaceJob:
        try:
            return dependencies.workspace_store.get_job(job_id)
        except Exception as error:
            raise _domain_error(error) from None

    @router.get("/workspace-jobs/{job_id}/events", response_model=list[WorkspaceEvent])
    def workspace_events_after(job_id: str, after: Annotated[int, Query(ge=0)] = 0) -> Sequence[WorkspaceEvent]:
        try:
            dependencies.workspace_store.get_job(job_id)
            return dependencies.workspace_store.list_job_events(job_id, after_sequence=after)
        except Exception as error:
            raise _domain_error(error) from None

    @router.get("/providers/saved", response_model=list[SavedProviderProfile])
    def list_saved_providers() -> Sequence[SavedProviderProfile]:
        return dependencies.runtime.store.list_saved_provider_profiles()

    @router.delete("/providers/{profile_id}/credential", status_code=status.HTTP_204_NO_CONTENT)
    def forget_provider_credential(profile_id: str) -> Response:
        try:
            dependencies.runtime.forget_provider_credential(profile_id)
        except Exception as error:
            raise _domain_error(error) from None
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router
