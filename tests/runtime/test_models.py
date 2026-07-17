from exam_predictor.runtime.models import (
    ConnectProviderRequest,
    EventType,
    ProviderProfile,
    RunStatus,
    SubmitMessageRequest,
)


def test_provider_request_never_serializes_the_real_key():
    request = ConnectProviderRequest(
        profile=ProviderProfile(profile_id="primary", provider="gemini"),
        api_key="gemini-secret-value",
    )
    assert request.api_key.get_secret_value() == "gemini-secret-value"
    assert "gemini-secret-value" not in request.model_dump_json()


def test_provider_profile_builds_non_secret_provider_config():
    profile = ProviderProfile(
        profile_id="primary",
        provider="custom",
        base_url="https://models.example/v1",
        balanced_model="study-model",
    )
    assert profile.provider_config() == {
        "provider": "custom",
        "base_url": "https://models.example/v1",
        "balanced_model": "study-model",
    }


def test_run_status_distinguishes_settled_and_terminal_states():
    assert RunStatus.PAUSED.is_settled
    assert not RunStatus.PAUSED.is_terminal
    assert RunStatus.COMPLETED.is_terminal
    assert RunStatus.FAILED.is_terminal
    assert not RunStatus.RUNNING.is_settled


def test_submit_message_normalizes_required_fields():
    request = SubmitMessageRequest(
        thread_id=" calculus ",
        provider_profile_id=" primary ",
        message=" Explain limits. ",
    )
    assert request.thread_id == "calculus"
    assert request.provider_profile_id == "primary"
    assert request.message == "Explain limits."
    assert EventType.TOOL_STARTED.value == "tool_started"
