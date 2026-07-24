from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any

from exam_predictor.providers import BaseProvider, create_provider

from .models import ConnectProviderRequest, ProviderDescriptor, ProviderProfile


class ProviderSessionRegistry:
    def __init__(self, factory: Callable[[dict[str, Any]], BaseProvider] = create_provider):
        self._factory = factory
        self._providers: dict[str, BaseProvider] = {}
        self._descriptors: dict[str, ProviderDescriptor] = {}
        self._lock = RLock()

    def connect(self, request: ConnectProviderRequest) -> ProviderDescriptor:
        secret = request.api_key.get_secret_value()
        return self._install(request.profile, secret)

    def restore(self, profile: ProviderProfile, api_key: str) -> ProviderDescriptor:
        return self._install(profile, api_key)

    def _install(self, profile: ProviderProfile, secret: str) -> ProviderDescriptor:
        config = profile.provider_config()
        if not secret.strip():
            raise ValueError(
                f"An API key is required to connect provider profile '{profile.profile_id}'."
            )
        config["api_key"] = secret
        config.pop("approved_max_usd", None)
        failed = False
        provider: BaseProvider | None = None
        capabilities: dict[str, bool] = {}
        try:
            provider = self._factory(config)
            capabilities = {
                name: bool(value)
                for name, value in vars(provider.capabilities).items()
            }
        except Exception:
            failed = True
        if failed:
            raise RuntimeError("Provider connection failed: [REDACTED]") from None
        if provider is None:
            raise RuntimeError("Provider connection failed.") from None
        descriptor = ProviderDescriptor(profile=profile, capabilities=capabilities)
        with self._lock:
            self._providers[profile.profile_id] = provider
            self._descriptors[profile.profile_id] = descriptor
        return descriptor

    def get_provider(self, profile_id: str) -> BaseProvider:
        with self._lock:
            provider = self._providers.get(profile_id)
        if provider is None:
            raise KeyError(
                f"Connect provider profile '{profile_id}' before starting or resuming this run."
            )
        return provider

    def has_provider(self, profile_id: str) -> bool:
        with self._lock:
            return profile_id in self._providers

    def disconnect(self, profile_id: str) -> None:
        with self._lock:
            self._providers.pop(profile_id, None)
            self._descriptors.pop(profile_id, None)

    def list_profiles(self) -> list[ProviderDescriptor]:
        with self._lock:
            return [
                self._descriptors[profile_id]
                for profile_id in sorted(self._descriptors)
            ]
