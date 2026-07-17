from types import SimpleNamespace

import pytest

from exam_predictor.runtime.models import ConnectProviderRequest, ProviderProfile
from exam_predictor.runtime.provider_sessions import ProviderSessionRegistry


class FakeProvider:
    name = "gemini"
    capabilities = SimpleNamespace(
        chat=True,
        vision=True,
        file_understanding=True,
        embeddings=True,
        web_search=True,
        citations=True,
        ephemeral_requests=True,
    )


def test_connect_keeps_key_in_memory_and_returns_only_capabilities():
    seen: list[dict] = []

    def factory(config: dict):
        seen.append(config)
        return FakeProvider()

    registry = ProviderSessionRegistry(factory=factory)
    descriptor = registry.connect(ConnectProviderRequest(
        profile=ProviderProfile(profile_id="primary", provider="gemini"),
        api_key="secret-key",
    ))

    assert seen == [{"provider": "gemini", "api_key": "secret-key"}]
    assert registry.get_provider("primary").name == "gemini"
    assert "secret-key" not in descriptor.model_dump_json()
    assert descriptor.capabilities["web_search"] is True


def test_unknown_provider_profile_is_actionable():
    registry = ProviderSessionRegistry(factory=lambda config: FakeProvider())
    with pytest.raises(KeyError, match="Connect provider profile 'missing'"):
        registry.get_provider("missing")


def test_provider_connection_error_redacts_the_key():
    def failing_factory(config: dict):
        raise RuntimeError(f"rejected {config['api_key']}")

    registry = ProviderSessionRegistry(factory=failing_factory)
    with pytest.raises(RuntimeError) as captured:
        registry.connect(ConnectProviderRequest(
            profile=ProviderProfile(profile_id="primary", provider="gemini"),
            api_key="secret-key",
        ))
    assert "secret-key" not in str(captured.value)
    assert "[REDACTED]" in str(captured.value)


def test_provider_connection_error_redacts_the_full_exception_chain():
    def failing_factory(config: dict):
        raise RuntimeError(f"rejected {config['api_key']}")

    registry = ProviderSessionRegistry(factory=failing_factory)
    with pytest.raises(RuntimeError) as captured:
        registry.connect(ConnectProviderRequest(
            profile=ProviderProfile(profile_id="primary", provider="gemini"),
            api_key="secret-key",
        ))

    chain: list[BaseException] = []
    current: BaseException | None = captured.value
    while current is not None:
        chain.append(current)
        current = current.__cause__ or current.__context__

    assert all("secret-key" not in str(error) for error in chain)


@pytest.mark.parametrize("api_key", ["", "   "])
def test_empty_api_key_is_rejected_before_provider_factory(api_key: str):
    seen: list[dict] = []

    def factory(config: dict):
        seen.append(config)
        return FakeProvider()

    registry = ProviderSessionRegistry(factory=factory)
    with pytest.raises(ValueError, match="API key is required to connect provider profile 'primary'"):
        registry.connect(ConnectProviderRequest(
            profile=ProviderProfile(profile_id="primary", provider="gemini"),
            api_key=api_key,
        ))

    assert seen == []
