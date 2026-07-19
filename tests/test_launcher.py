from __future__ import annotations

import signal
import subprocess
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts import launch_app
from scripts.launch_app import (
    allocate_loopback_port,
    build_child_environment,
    request_pause,
    run_application,
    wait_for_worker,
    worker_command,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKEN = "top-secret-worker-token"


class FakeSocket:
    def __init__(self) -> None:
        self.bound: tuple[str, int] | None = None

    def __enter__(self) -> FakeSocket:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def bind(self, address: tuple[str, int]) -> None:
        self.bound = address

    def getsockname(self) -> tuple[str, int]:
        return ("127.0.0.1", 43123)


class FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None


class FakeProcess:
    def __init__(
        self,
        name: str,
        actions: list[str],
        *,
        wait_result: int = 0,
        wait_interrupt: bool = False,
        stubborn: bool = False,
        already_exited: bool = False,
        terminate_error: BaseException | None = None,
        wait_after_kill_error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.actions = actions
        self.wait_result = wait_result
        self.wait_interrupt = wait_interrupt
        self.stubborn = stubborn
        self.terminate_error = terminate_error
        self.wait_after_kill_error = wait_after_kill_error
        self.killed = False
        self.returncode: int | None = wait_result if already_exited else None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.actions.append(f"wait:{self.name}:{timeout}")
        if self.wait_interrupt and timeout is None:
            raise KeyboardInterrupt
        if timeout is None:
            self.returncode = self.wait_result
            return self.wait_result
        if self.killed and self.wait_after_kill_error is not None:
            raise self.wait_after_kill_error
        if self.stubborn and self.returncode is None:
            raise subprocess.TimeoutExpired(self.name, timeout)
        if self.returncode is None:
            self.returncode = -15
        return self.returncode

    def terminate(self) -> None:
        self.actions.append(f"terminate:{self.name}")
        if self.terminate_error is not None:
            raise self.terminate_error

    def kill(self) -> None:
        self.actions.append(f"kill:{self.name}")
        self.killed = True
        self.returncode = -9


class SequencedPollProcess(FakeProcess):
    def __init__(
        self,
        name: str,
        actions: list[str],
        poll_results: list[int | None],
    ) -> None:
        super().__init__(name, actions)
        self.poll_results = iter(poll_results)

    def poll(self) -> int | None:
        try:
            self.returncode = next(self.poll_results)
        except StopIteration:
            pass
        self.actions.append(f"poll:{self.name}:{self.returncode}")
        return self.returncode


class SignalRecorder:
    def __init__(self) -> None:
        self.previous = object()
        self.calls: list[tuple[signal.Signals, object]] = []

    def getsignal(self, signum: signal.Signals) -> object:
        assert signum == signal.SIGTERM
        return self.previous

    def signal(self, signum: signal.Signals, handler: object) -> None:
        self.calls.append((signum, handler))


def configure_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMSAGE_AGENT_V2", "1")
    monkeypatch.setattr(launch_app, "allocate_loopback_port", lambda: 43123)
    monkeypatch.setattr(launch_app.secrets, "token_urlsafe", lambda size: TOKEN)


def assert_exception_chain_excludes(error: BaseException, secret: str) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        assert secret not in str(current)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


def test_worker_token_is_in_environment_not_process_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMSAGE_AGENT_V2", "0")

    environment = build_child_environment("http://127.0.0.1:8765", TOKEN)
    command = worker_command(sys.executable, 8765)

    assert environment["EXAMSAGE_WORKER_TOKEN"] == TOKEN
    assert environment["EXAMSAGE_WORKER_URL"] == "http://127.0.0.1:8765"
    assert environment["EXAMSAGE_AGENT_V2"] == "1"
    assert TOKEN not in " ".join(command)


def test_port_allocation_binds_to_ipv4_loopback() -> None:
    fake = FakeSocket()

    port = allocate_loopback_port(socket_factory=lambda *args: fake)

    assert fake.bound == ("127.0.0.1", 0)
    assert port == 43123


def test_wait_for_worker_retries_until_public_health_is_ready() -> None:
    calls: list[tuple[str, float]] = []
    responses = iter([OSError("not ready"), FakeResponse(503), FakeResponse(200)])

    def health_get(url: str, timeout: float) -> FakeResponse:
        calls.append((url, timeout))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    wait_for_worker(
        "http://127.0.0.1:43123",
        timeout=1,
        health_get=health_get,
        sleep=lambda seconds: None,
    )

    assert calls == [("http://127.0.0.1:43123/health", 1.0)] * 3


def test_readiness_timeout_is_actionable_and_does_not_echo_failure() -> None:
    def unavailable(url: str, timeout: float) -> object:
        raise OSError(TOKEN)

    with pytest.raises(RuntimeError, match="Worker did not become ready") as error:
        wait_for_worker(
            "http://127.0.0.1:43123",
            timeout=0,
            health_get=unavailable,
            sleep=lambda seconds: None,
        )

    assert TOKEN not in str(error.value)
    assert_exception_chain_excludes(error.value, TOKEN)


def test_worker_exit_before_first_health_fails_without_starting_streamlit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_agent(monkeypatch)
    actions: list[str] = []
    commands: list[list[str]] = []

    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        commands.append(command)
        actions.append("start:worker")
        return FakeProcess("worker", actions, wait_result=23, already_exited=True)

    def readiness(url: str, *, worker_poll: Callable[[], int | None]) -> None:
        def unexpected_health(url: str, timeout: float) -> object:
            pytest.fail("health must not be requested after the Worker has exited")

        wait_for_worker(
            url,
            timeout=10,
            health_get=unexpected_health,
            sleep=lambda seconds: actions.append("sleep"),
            worker_poll=worker_poll,
        )

    with pytest.raises(RuntimeError, match=r"exited before becoming ready.*23") as error:
        run_application(
            popen=popen,
            readiness=readiness,
            pause_request=lambda url, token: actions.append("pause"),
        )

    assert_exception_chain_excludes(error.value, TOKEN)
    assert len(commands) == 1
    assert actions == ["start:worker", "pause"]


def test_worker_exit_between_health_retries_fails_without_starting_streamlit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_agent(monkeypatch)
    actions: list[str] = []
    commands: list[list[str]] = []

    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        commands.append(command)
        actions.append("start:worker")
        return SequencedPollProcess("worker", actions, [None, 17])

    def readiness(url: str, *, worker_poll: Callable[[], int | None]) -> None:
        def unavailable(url: str, timeout: float) -> object:
            actions.append("health")
            raise OSError(TOKEN)

        wait_for_worker(
            url,
            timeout=10,
            health_get=unavailable,
            sleep=lambda seconds: actions.append("sleep"),
            worker_poll=worker_poll,
        )

    with pytest.raises(RuntimeError, match=r"exited before becoming ready.*17") as error:
        run_application(
            popen=popen,
            readiness=readiness,
            pause_request=lambda url, token: actions.append("pause"),
        )

    assert_exception_chain_excludes(error.value, TOKEN)
    assert len(commands) == 1
    assert actions.index("health") < actions.index("sleep") < actions.index("poll:worker:17")
    assert actions.index("pause") > actions.index("poll:worker:17")
    assert "terminate:worker" not in actions


def test_pause_request_uses_exact_token_on_authenticated_runtime_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def post(url: str, **kwargs: object) -> FakeResponse:
        calls.append((url, kwargs))
        return FakeResponse(200)

    monkeypatch.setattr(launch_app.httpx, "post", post)

    request_pause("http://127.0.0.1:43123", TOKEN)

    assert calls == [
        (
            "http://127.0.0.1:43123/v1/runtime/pause-all",
            {"headers": {"X-ExamSage-Token": TOKEN}, "timeout": 2.0},
        )
    ]


def test_legacy_mode_starts_only_streamlit_with_inherited_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXAMSAGE_AGENT_V2", "0")
    actions: list[str] = []
    calls: list[tuple[list[str], dict[str, object]]] = []

    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        calls.append((command, kwargs))
        return FakeProcess("streamlit", actions, wait_result=7)

    result = run_application(popen=popen)

    assert result.worker_started is False
    assert result.streamlit_started is True
    assert result.exit_code == 7
    assert len(calls) == 1
    assert calls[0][0] == [sys.executable, "-m", "streamlit", "run", "app.py"]
    assert calls[0][1]["cwd"] == PROJECT_ROOT
    assert calls[0][1]["env"]["EXAMSAGE_AGENT_V2"] == "0"


def test_legacy_keyboard_interrupt_terminates_streamlit_and_returns_130(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXAMSAGE_AGENT_V2", raising=False)
    actions: list[str] = []

    result = run_application(
        popen=lambda command, **kwargs: FakeProcess("streamlit", actions, wait_interrupt=True)
    )

    assert result.exit_code == 130
    assert actions.index("terminate:streamlit") < actions.index("wait:streamlit:3.0")


def test_agent_waits_for_worker_then_starts_streamlit_and_pauses_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_agent(monkeypatch)
    actions: list[str] = []
    calls: list[tuple[list[str], dict[str, object]]] = []

    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        name = "worker" if "exam_predictor.worker.main" in command else "streamlit"
        actions.append(f"start:{name}")
        calls.append((command, kwargs))
        return FakeProcess(name, actions)

    result = run_application(
        popen=popen,
        readiness=lambda url, *, worker_poll: actions.append(f"ready:{url}"),
        pause_request=lambda url, token: actions.append(f"pause:{url}:{token}"),
    )

    assert result.worker_started and result.streamlit_started
    assert result.exit_code == 0
    assert actions.index("ready:http://127.0.0.1:43123") < actions.index("start:streamlit")
    assert actions.index(f"pause:http://127.0.0.1:43123:{TOKEN}") < actions.index("terminate:worker")
    assert all(TOKEN not in " ".join(command) for command in result.commands)
    assert calls[0][1]["cwd"] == PROJECT_ROOT
    assert calls[1][1]["cwd"] == PROJECT_ROOT
    assert calls[0][1]["env"]["EXAMSAGE_WORKER_TOKEN"] == TOKEN
    assert calls[1][1]["env"]["EXAMSAGE_WORKER_TOKEN"] == TOKEN


def test_readiness_failure_cleans_worker_without_starting_streamlit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_agent(monkeypatch)
    actions: list[str] = []
    calls: list[list[str]] = []
    health_calls = 0

    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        calls.append(command)
        return FakeProcess("worker", actions)

    def fail_readiness(url: str, *, worker_poll: Callable[[], int | None]) -> None:
        def unavailable(url: str, timeout: float) -> object:
            nonlocal health_calls
            health_calls += 1
            raise OSError(TOKEN)

        wait_for_worker(
            url,
            timeout=0,
            health_get=unavailable,
            sleep=lambda seconds: None,
            worker_poll=worker_poll,
        )

    with pytest.raises(RuntimeError, match="Worker did not become ready within 0 seconds") as error:
        run_application(
            popen=popen,
            readiness=fail_readiness,
            pause_request=lambda url, token: actions.append("pause"),
        )

    assert_exception_chain_excludes(error.value, TOKEN)
    assert len(calls) == 1
    assert health_calls == 1
    assert actions.index("pause") < actions.index("terminate:worker")


def test_keyboard_interrupt_returns_130_and_cleans_both_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_agent(monkeypatch)
    actions: list[str] = []

    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        name = "worker" if "exam_predictor.worker.main" in command else "streamlit"
        return FakeProcess(name, actions, wait_interrupt=name == "streamlit")

    result = run_application(
        popen=popen,
        readiness=lambda url, *, worker_poll: None,
        pause_request=lambda url, token: actions.append("pause"),
    )

    assert result.exit_code == 130
    assert actions.index("pause") < actions.index("terminate:worker")
    assert "terminate:streamlit" in actions


def test_sigterm_handler_converts_shutdown_to_cleanup_and_is_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_agent(monkeypatch)
    recorder = SignalRecorder()
    monkeypatch.setattr(launch_app.signal, "getsignal", recorder.getsignal)
    monkeypatch.setattr(launch_app.signal, "signal", recorder.signal)
    actions: list[str] = []

    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        name = "worker" if "exam_predictor.worker.main" in command else "streamlit"
        return FakeProcess(name, actions)

    def readiness(url: str, *, worker_poll: Callable[[], int | None]) -> None:
        installed_handler = recorder.calls[0][1]
        installed_handler(signal.SIGTERM, None)

    result = run_application(
        popen=popen,
        readiness=readiness,
        pause_request=lambda url, token: actions.append("pause"),
    )

    assert result.exit_code == 130
    assert recorder.calls[-1] == (signal.SIGTERM, recorder.previous)
    assert actions.index("pause") < actions.index("terminate:worker")


@pytest.mark.parametrize("failure_target", ["worker", "streamlit"])
def test_child_start_failure_is_actionable_and_cleans_started_worker(
    monkeypatch: pytest.MonkeyPatch,
    failure_target: str,
) -> None:
    configure_agent(monkeypatch)
    actions: list[str] = []

    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        name = "worker" if "exam_predictor.worker.main" in command else "streamlit"
        if name == failure_target:
            raise OSError(TOKEN)
        return FakeProcess(name, actions)

    with pytest.raises(RuntimeError, match=f"Could not start the local ExamSage {failure_target}") as error:
        run_application(
            popen=popen,
            readiness=lambda url, *, worker_poll: None,
            pause_request=lambda url, token: actions.append("pause"),
        )

    assert TOKEN not in str(error.value)
    assert_exception_chain_excludes(error.value, TOKEN)
    rendered_traceback = "".join(
        traceback.format_exception(type(error.value), error.value, error.value.__traceback__)
    )
    assert TOKEN not in rendered_traceback
    if failure_target == "streamlit":
        assert actions.index("pause") < actions.index("terminate:worker")
    else:
        assert actions == []


def test_cleanup_escalates_after_bounded_wait_and_never_touches_exited_streamlit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_agent(monkeypatch)
    actions: list[str] = []

    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        name = "worker" if "exam_predictor.worker.main" in command else "streamlit"
        return FakeProcess(
            name,
            actions,
            stubborn=name == "worker",
            already_exited=name == "streamlit",
        )

    result = run_application(
        popen=popen,
        readiness=lambda url, *, worker_poll: None,
        pause_request=lambda url, token: actions.append("pause"),
    )

    assert result.exit_code == 0
    assert actions.index("pause") < actions.index("terminate:worker") < actions.index("kill:worker")
    assert "terminate:streamlit" not in actions
    assert "kill:streamlit" not in actions
    assert "wait:worker:3.0" in actions


def test_pause_failure_still_terminates_worker_and_does_not_leak_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_agent(monkeypatch)
    actions: list[str] = []

    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        name = "worker" if "exam_predictor.worker.main" in command else "streamlit"
        return FakeProcess(name, actions)

    def pause_request(url: str, token: str) -> None:
        actions.append("pause")
        raise OSError(TOKEN)

    result = run_application(
        popen=popen,
        readiness=lambda url, *, worker_poll: None,
        pause_request=pause_request,
    )

    assert result.exit_code == 0
    assert actions.index("pause") < actions.index("terminate:worker")


def test_cleanup_failures_do_not_skip_other_child_or_signal_restoration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_agent(monkeypatch)
    recorder = SignalRecorder()
    monkeypatch.setattr(launch_app.signal, "getsignal", recorder.getsignal)
    monkeypatch.setattr(launch_app.signal, "signal", recorder.signal)
    actions: list[str] = []

    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        name = "worker" if "exam_predictor.worker.main" in command else "streamlit"
        if name == "worker":
            return FakeProcess(
                name,
                actions,
                stubborn=True,
                wait_after_kill_error=OSError(TOKEN),
            )
        return FakeProcess(
            name,
            actions,
            wait_interrupt=True,
            terminate_error=OSError(TOKEN),
        )

    result = run_application(
        popen=popen,
        readiness=lambda url, *, worker_poll: None,
        pause_request=lambda url, token: actions.append("pause"),
    )

    assert result.exit_code == 130
    assert actions.index("pause") < actions.index("terminate:worker")
    assert "terminate:streamlit" in actions
    assert "kill:worker" in actions
    assert recorder.calls[-1] == (signal.SIGTERM, recorder.previous)
    captured = capsys.readouterr()
    assert TOKEN not in captured.out
    assert TOKEN not in captured.err


def test_worker_creation_flag_is_windows_only_and_streamlit_stays_console_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_agent(monkeypatch)
    monkeypatch.setattr(launch_app.sys, "platform", "win32")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        calls.append((command, kwargs))
        name = "worker" if "exam_predictor.worker.main" in command else "streamlit"
        return FakeProcess(name, [])

    run_application(
        popen=popen,
        readiness=lambda url, *, worker_poll: None,
        pause_request=lambda url, token: None,
    )

    assert calls[0][1]["creationflags"] == subprocess.CREATE_NO_WINDOW
    assert "creationflags" not in calls[1][1]


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_non_windows_processes_receive_no_windows_creation_flags(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
) -> None:
    configure_agent(monkeypatch)
    monkeypatch.setattr(launch_app.sys, "platform", platform)
    calls: list[dict[str, object]] = []

    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        calls.append(kwargs)
        name = "worker" if "exam_predictor.worker.main" in command else "streamlit"
        return FakeProcess(name, [])

    run_application(
        popen=popen,
        readiness=lambda url, *, worker_poll: None,
        pause_request=lambda url, token: None,
    )

    assert all("creationflags" not in kwargs for kwargs in calls)


def test_platform_launchers_delegate_only_final_command_without_enabling_agent() -> None:
    windows = (PROJECT_ROOT / "launch_windows.bat").read_text(encoding="utf-8")
    macos = (PROJECT_ROOT / "launch_macos.command").read_text(encoding="utf-8")

    assert '".venv\\Scripts\\python.exe" scripts\\launch_app.py' in windows
    assert '".venv/bin/python" scripts/launch_app.py' in macos
    assert "pip install -r requirements.txt" in windows
    assert "pip install -r requirements.txt" in macos
    assert "EXAMSAGE_AGENT_V2" not in windows
    assert "EXAMSAGE_AGENT_V2" not in macos
    assert "-m streamlit run app.py" not in windows
    assert "-m streamlit run app.py" not in macos
