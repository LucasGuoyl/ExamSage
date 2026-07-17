from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    STOPPING = "stopping"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {self.COMPLETED, self.FAILED}

    @property
    def is_settled(self) -> bool:
        return self in {self.PAUSED, self.COMPLETED, self.FAILED}


class EventType(str, Enum):
    QUEUED = "queued"
    STARTED = "started"
    PROGRESS = "progress"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    MESSAGE = "message"
    STOP_REQUESTED = "stop_requested"
    PAUSED = "paused"
    RESUMED = "resumed"
    COMPLETED = "completed"
    FAILED = "failed"


def _strip_required(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("value must not be empty")
    return cleaned


class ProviderProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str
    provider: Literal["openai", "gemini", "custom"]
    base_url: str | None = None
    fast_model: str | None = None
    balanced_model: str | None = None
    reasoning_model: str | None = None
    embedding_model: str | None = None

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        return _strip_required(value)

    def provider_config(self) -> dict[str, str]:
        values = self.model_dump(exclude={"profile_id"}, exclude_none=True)
        return {key: value for key, value in values.items() if value != ""}


class ConnectProviderRequest(BaseModel):
    profile: ProviderProfile
    api_key: SecretStr


class ProviderDescriptor(BaseModel):
    profile: ProviderProfile
    capabilities: dict[str, bool]


class SubmitMessageRequest(BaseModel):
    thread_id: str
    provider_profile_id: str
    message: str

    @field_validator("thread_id", "provider_profile_id", "message")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _strip_required(value)


class SubmitMessageResponse(BaseModel):
    run_id: str
    status: RunStatus


class AgentEvent(BaseModel):
    sequence: int = 0
    run_id: str
    event_type: EventType
    stage: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class RunSnapshot(BaseModel):
    run_id: str
    thread_id: str
    provider_profile_id: str
    message: str
    status: RunStatus
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    agent_v2: bool = True
