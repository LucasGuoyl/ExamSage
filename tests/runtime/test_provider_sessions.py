import json
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


def test_restore_uses_the_same_secret_safe_factory_path_and_lists_descriptors():
    seen: list[dict] = []

    def factory(config: dict):
        seen.append(config)
        return FakeProvider()

    registry = ProviderSessionRegistry(factory=factory)
    profile = ProviderProfile(profile_id="restored", provider="gemini")

    descriptor = registry.restore(profile, "restored-secret")

    assert seen == [{"provider": "gemini", "api_key": "restored-secret"}]
    assert registry.list_profiles() == [descriptor]
    assert "restored-secret" not in repr(registry.list_profiles())
    registry.disconnect("restored")
    registry.disconnect("restored")
    assert registry.list_profiles() == []
    assert registry.has_provider("restored") is False


def test_restore_failure_discards_the_provider_exception_chain():
    sentinel = "restored-secret"

    def failing_factory(config: dict):
        raise RuntimeError(f"rejected {config['api_key']}")

    registry = ProviderSessionRegistry(factory=failing_factory)
    with pytest.raises(RuntimeError) as captured:
        registry.restore(
            ProviderProfile(profile_id="restored", provider="gemini"),
            sentinel,
        )

    assert sentinel not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert registry.list_profiles() == []


def test_provider_capability_error_is_bounded_and_cause_free():
    sentinel = "secret-provider-path-C:/private/course"

    class FailingProvider:
        @property
        def capabilities(self):
            raise RuntimeError(sentinel)

    registry = ProviderSessionRegistry(factory=lambda config: FailingProvider())
    with pytest.raises(RuntimeError, match="Provider connection failed") as captured:
        registry.restore(
            ProviderProfile(profile_id="restored", provider="gemini"),
            "provider-key",
        )

    assert sentinel not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.parametrize(
    "base_url",
    [
        "http://models.example/v1",
        "https://127.0.0.1/v1",
        "https://userinfo-sentinel:password@models.example/v1",
        "https://models.example/v1?api_key=query-sentinel",
    ],
)
def test_custom_provider_url_errors_never_echo_untrusted_input(base_url: str):
    registry = ProviderSessionRegistry(factory=lambda config: FakeProvider())

    with pytest.raises(RuntimeError, match="^Provider configuration is invalid\\.$") as captured:
        registry.connect(
            ConnectProviderRequest(
                profile=ProviderProfile(
                    profile_id="custom",
                    provider="custom",
                    base_url=base_url,
                ),
                api_key="provider-key",
            )
        )

    error = captured.value
    errors = getattr(error, "errors", lambda: [])()
    json_value = getattr(error, "json", lambda: "{}")()
    surfaces = "\n".join(
        [
            str(error),
            repr(error),
            json.dumps(errors),
            json_value,
            json.dumps({"detail": str(error)}),
        ]
    )
    assert "userinfo-sentinel" not in surfaces
    assert "query-sentinel" not in surfaces
    assert error.__cause__ is None
    assert error.__context__ is None


def test_custom_provider_safe_base_url_is_normalized_before_factory():
    seen: list[dict] = []
    registry = ProviderSessionRegistry(
        factory=lambda config: seen.append(config) or FakeProvider()
    )

    descriptor = registry.connect(
        ConnectProviderRequest(
            profile=ProviderProfile(
                profile_id="custom",
                provider="custom",
                base_url="  https://models.example/v1  ",
            ),
            api_key="provider-key",
        )
    )

    assert seen[0]["base_url"] == "https://models.example/v1"
    assert descriptor.profile.base_url == "https://models.example/v1"


@pytest.mark.parametrize(
    "profile_id",
    ["profile/name", "profile id", "provider:primary", "a" * 65],
)
def test_provider_profile_rejects_ids_that_cannot_be_vault_accounts(profile_id: str):
    with pytest.raises(ValueError, match="provider profile ID"):
        ProviderProfile(profile_id=profile_id, provider="gemini")
