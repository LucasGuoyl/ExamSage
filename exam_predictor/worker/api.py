from __future__ import annotations

import asyncio
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, SecretStr, field_validator, model_validator

from exam_predictor.runtime.coordinator import RuntimeCoordinator
from exam_predictor.runtime.models import (
    AgentEvent,
    ConnectProviderRequest,
    EventType,
    HealthResponse,
    ProviderDescriptor,
    RunSnapshot,
    RunStatus,
    SubmitMessageResponse,
)
from exam_predictor.runtime.provider_sessions import ProviderSessionRegistry
from exam_predictor.runtime.store import RuntimeStore


_MISSING_RUN_DETAIL = "Agent run was not found."
_PROVIDER_REQUIRED_DETAIL = (
    "Connect provider profile before starting or resuming this run."
)
_RUN_CONFLICT_DETAIL = "Run state conflicts with this operation."


class WorkerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int
    token: SecretStr
    data_dir: Path

    @field_validator("token")
    @classmethod
    def nonempty_token(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("Worker token must not be empty.")
        return value

    @model_validator(mode="after")
    def loopback_only(self) -> WorkerSettings:
        if self.host != "127.0.0.1":
            raise ValueError("ExamSage Agent Worker must bind to 127.0.0.1.")
        return self


class SubmitMessageBody(BaseModel):
    provider_profile_id: str
    message: str


def create_worker_app(
    settings: WorkerSettings,
    runtime: RuntimeCoordinator | None = None,
) -> FastAPI:
    if runtime is None:
        store = RuntimeStore(settings.data_dir / "agent-runtime.sqlite3")
        sessions = ProviderSessionRegistry()
        runtime = RuntimeCoordinator(
            store=store,
            provider_sessions=sessions,
            checkpoints_path=settings.data_dir / "agent-checkpoints.sqlite3",
        )

    expected_token = settings.token.get_secret_value().encode()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.start()
        try:
            yield
        finally:
            runtime.shutdown()

    app = FastAPI(
        title="ExamSage Agent Worker",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(_request, _exc) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": "Invalid request."},
        )

    def require_token(
        supplied: Annotated[str | None, Header(alias="X-ExamSage-Token")] = None,
    ) -> None:
        supplied_token = supplied.encode() if supplied is not None else b""
        if not secrets.compare_digest(supplied_token, expected_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized.",
            )

    def missing_run() -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_MISSING_RUN_DETAIL,
        )

    def provider_required() -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_PROVIDER_REQUIRED_DETAIL,
        )

    def run_conflict() -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_RUN_CONFLICT_DETAIL,
        )

    auth = [Depends(require_token)]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.post(
        "/v1/providers/connect",
        response_model=ProviderDescriptor,
        dependencies=auth,
    )
    def connect_provider(request: ConnectProviderRequest) -> ProviderDescriptor:
        try:
            return runtime.provider_sessions.connect(request)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provider connection failed.",
            ) from None

    @app.post(
        "/v1/threads/{thread_id}/messages",
        response_model=SubmitMessageResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=auth,
    )
    def submit_message(
        thread_id: str,
        body: SubmitMessageBody,
    ) -> SubmitMessageResponse:
        try:
            run = runtime.submit_message(
                thread_id,
                body.provider_profile_id,
                body.message,
            )
        except KeyError:
            raise provider_required() from None
        return SubmitMessageResponse(run_id=run.run_id, status=run.status)

    @app.get(
        "/v1/runs/{run_id}",
        response_model=RunSnapshot,
        dependencies=auth,
    )
    def get_run(run_id: str) -> RunSnapshot:
        try:
            return runtime.store.get_run(run_id)
        except KeyError:
            raise missing_run() from None

    @app.get(
        "/v1/runs/{run_id}/events",
        response_model=list[AgentEvent],
        dependencies=auth,
    )
    def list_events(
        run_id: str,
        after: int = Query(default=0, ge=0),
    ) -> list[AgentEvent]:
        try:
            runtime.store.get_run(run_id)
        except KeyError:
            raise missing_run() from None
        return runtime.store.list_events(run_id, after=after)

    @app.get(
        "/v1/runs/{run_id}/stream",
        dependencies=auth,
    )
    def stream_events(
        run_id: str,
        after: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        try:
            runtime.store.get_run(run_id)
        except KeyError:
            raise missing_run() from None

        async def generate():
            cursor = after
            delivered_types = {
                event.event_type
                for event in runtime.store.list_events(run_id)
                if event.sequence <= cursor
            }
            heartbeat_at = time.monotonic() + 15.0
            while True:
                events = runtime.store.list_events(run_id, after=cursor)
                for event in events:
                    cursor = event.sequence
                    delivered_types.add(event.event_type)
                    yield (
                        f"id: {event.sequence}\n"
                        f"event: {event.event_type.value}\n"
                        f"data: {event.model_dump_json()}\n\n"
                    )

                run = runtime.store.get_run(run_id)
                terminal_type = {
                    RunStatus.PAUSED: EventType.PAUSED,
                    RunStatus.COMPLETED: EventType.COMPLETED,
                    RunStatus.FAILED: EventType.FAILED,
                }.get(run.status)
                if (
                    terminal_type in delivered_types
                    and not runtime.store.list_events(run_id, after=cursor)
                ):
                    return

                if time.monotonic() >= heartbeat_at:
                    yield ": keep-alive\n\n"
                    heartbeat_at = time.monotonic() + 15.0
                await asyncio.sleep(0.25)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post(
        "/v1/runs/{run_id}/stop",
        response_model=RunSnapshot,
        dependencies=auth,
    )
    def stop(run_id: str) -> RunSnapshot:
        try:
            return runtime.stop(run_id)
        except KeyError:
            raise missing_run() from None
        except ValueError:
            raise run_conflict() from None

    @app.post(
        "/v1/runs/{run_id}/resume",
        response_model=RunSnapshot,
        dependencies=auth,
    )
    def resume(run_id: str) -> RunSnapshot:
        try:
            runtime.store.get_run(run_id)
        except KeyError:
            raise missing_run() from None
        try:
            return runtime.resume(run_id)
        except KeyError:
            raise provider_required() from None
        except ValueError:
            raise run_conflict() from None

    @app.post(
        "/v1/runtime/pause-all",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=auth,
    )
    def pause_all() -> None:
        runtime.pause_all()

    return app
