from __future__ import annotations

import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx


@dataclass
class LauncherResult:
    worker_started: bool
    streamlit_started: bool
    exit_code: int
    commands: list[list[str]] = field(default_factory=list)


def worker_command(python: str, port: int) -> list[str]:
    return [
        python,
        "-m",
        "exam_predictor.worker.main",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def streamlit_command(python: str) -> list[str]:
    return [python, "-m", "streamlit", "run", "app.py"]


def build_child_environment(worker_url: str, token: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["EXAMSAGE_WORKER_URL"] = worker_url
    environment["EXAMSAGE_WORKER_TOKEN"] = token
    environment["EXAMSAGE_AGENT_V2"] = "1"
    return environment


def allocate_loopback_port(
    socket_factory: Callable[..., socket.socket] = socket.socket,
) -> int:
    with socket_factory(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_for_worker(
    worker_url: str,
    *,
    timeout: float = 15.0,
    health_get: Callable[..., object] = httpx.get,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            response = health_get(f"{worker_url}/health", timeout=1.0)
            if getattr(response, "status_code", 0) == 200:
                return
        except (OSError, httpx.HTTPError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"The local ExamSage Agent Worker did not become ready within {timeout:g} seconds."
            )
        sleep(0.1)


def request_pause(worker_url: str, token: str) -> None:
    httpx.post(
        f"{worker_url}/v1/runtime/pause-all",
        headers={"X-ExamSage-Token": token},
        timeout=2.0,
    ).raise_for_status()


def terminate_child(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            return
        process.wait(timeout=3.0)


def _cleanup_children(
    *,
    worker_process: subprocess.Popen[bytes] | None,
    streamlit_process: subprocess.Popen[bytes] | None,
    worker_url: str | None,
    token: str | None,
    pause_request: Callable[[str, str], None],
) -> None:
    if worker_process is not None and worker_url is not None and token is not None:
        try:
            pause_request(worker_url, token)
        except BaseException:
            pass
    try:
        terminate_child(streamlit_process)
    finally:
        terminate_child(worker_process)


def run_application(
    *,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    readiness: Callable[[str], None] = wait_for_worker,
    pause_request: Callable[[str, str], None] = request_pause,
) -> LauncherResult:
    project_root = Path(__file__).resolve().parents[1]
    python = sys.executable
    agent_mode = os.environ.get("EXAMSAGE_AGENT_V2", "0") == "1"
    commands: list[list[str]] = []
    worker_process: subprocess.Popen[bytes] | None = None
    streamlit_process: subprocess.Popen[bytes] | None = None
    worker_url: str | None = None
    token: str | None = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def on_sigterm(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_sigterm)
    try:
        if not agent_mode:
            streamlit = streamlit_command(python)
            commands.append(streamlit)
            try:
                streamlit_process = popen(streamlit, cwd=project_root, env=os.environ.copy())
            except OSError:
                raise RuntimeError("Could not start the local ExamSage streamlit process.") from None
            try:
                exit_code = int(streamlit_process.wait())
            except KeyboardInterrupt:
                exit_code = 130
            finally:
                terminate_child(streamlit_process)
            return LauncherResult(False, True, exit_code, commands)

        port = allocate_loopback_port()
        worker_url = f"http://127.0.0.1:{port}"
        token = secrets.token_urlsafe(32)
        environment = build_child_environment(worker_url, token)

        worker = worker_command(python, port)
        commands.append(worker)
        worker_options: dict[str, object] = {"cwd": project_root, "env": environment}
        if sys.platform == "win32":
            worker_options["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            worker_process = popen(worker, **worker_options)
        except OSError:
            raise RuntimeError("Could not start the local ExamSage worker process.") from None

        readiness(worker_url)

        streamlit = streamlit_command(python)
        commands.append(streamlit)
        try:
            streamlit_process = popen(streamlit, cwd=project_root, env=environment)
        except OSError:
            raise RuntimeError("Could not start the local ExamSage streamlit process.") from None
        try:
            exit_code = int(streamlit_process.wait())
        except KeyboardInterrupt:
            exit_code = 130
        return LauncherResult(True, True, exit_code, commands)
    except KeyboardInterrupt:
        return LauncherResult(worker_process is not None, streamlit_process is not None, 130, commands)
    finally:
        if agent_mode:
            _cleanup_children(
                worker_process=worker_process,
                streamlit_process=streamlit_process,
                worker_url=worker_url,
                token=token,
                pause_request=pause_request,
            )
        signal.signal(signal.SIGTERM, previous_sigterm)


def main() -> None:
    raise SystemExit(run_application().exit_code)


if __name__ == "__main__":
    main()
