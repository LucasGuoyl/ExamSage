from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from exam_predictor.security import validate_public_https_url


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


_PROVIDER_PROFILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


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
        cleaned = _strip_required(value)
        if _PROVIDER_PROFILE_ID.fullmatch(cleaned) is None:
            raise ValueError("Invalid provider profile ID.")
        return cleaned

    def provider_config(self) -> dict[str, str]:
        values = self.model_dump(exclude={"profile_id"}, exclude_none=True)
        return {key: value for key, value in values.items() if value != ""}


class ProviderConfigurationError(RuntimeError):
    pass


def validate_provider_profile(profile: ProviderProfile) -> ProviderProfile:
    if profile.provider != "custom":
        return profile
    invalid = profile.base_url is None
    validated_url: str | None = None
    if not invalid:
        try:
            validated_url = validate_public_https_url(profile.base_url or "")
            invalid = bool(urlsplit(validated_url).query)
        except ValueError:
            invalid = True
    if invalid or validated_url is None:
        raise ProviderConfigurationError("Provider configuration is invalid.") from None
    return profile.model_copy(update={"base_url": validated_url})


class ConnectProviderRequest(BaseModel):
    profile: ProviderProfile
    api_key: SecretStr


class ProviderDescriptor(BaseModel):
    profile: ProviderProfile
    capabilities: dict[str, bool]
    credential_saved: bool = False
    credential_warning: str | None = None


class SavedProviderProfile(BaseModel):
    profile: ProviderProfile
    capabilities: dict[str, bool]
    credential_expected: bool
    reconnect_required: bool
    updated_at: datetime


class SubmitMessageRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    provider_profile_id: str = Field(min_length=1, max_length=64)
    workspace_id: str | None = Field(default=None, min_length=32, max_length=36)
    message: str = Field(min_length=1, max_length=20_000)

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
    workspace_id: str | None = None
    message: str
    status: RunStatus
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    agent_v2: bool = True
