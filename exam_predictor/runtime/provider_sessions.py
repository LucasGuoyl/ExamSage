from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any

from exam_predictor.providers import BaseProvider, create_provider

from .models import ConnectProviderRequest, ProviderDescriptor


class ProviderSessionRegistry:
    def __init__(self, factory: Callable[[dict[str, Any]], BaseProvider] = create_provider):
        self._factory = factory
        self._providers: dict[str, BaseProvider] = {}
        self._lock = RLock()

    def connect(self, request: ConnectProviderRequest) -> ProviderDescriptor:
        config = request.profile.provider_config()
        secret = request.api_key.get_secret_value()
        config["api_key"] = secret
        config.pop("approved_max_usd", None)
        try:
            provider = self._factory(config)
        except Exception as exc:
            safe_message = str(exc).replace(secret, "[REDACTED]")
            raise RuntimeError(safe_message or "Provider connection failed.") from None
        with self._lock:
            self._providers[request.profile.profile_id] = provider
        capabilities = {
            name: bool(value)
            for name, value in vars(provider.capabilities).items()
        }
        return ProviderDescriptor(profile=request.profile, capabilities=capabilities)

    def get_provider(self, profile_id: str) -> BaseProvider:
        with self._lock:
            provider = self._providers.get(profile_id)
        if provider is None:
            raise KeyError(
                f"Connect provider profile '{profile_id}' before starting or resuming this run."
            )
        return provider
