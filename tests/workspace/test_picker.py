from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from exam_predictor.workspace.picker import FolderPickerError, SubprocessFolderPicker
from exam_predictor.workspace import picker_helper


def test_picker_keeps_selected_path_out_of_argv(monkeypatch, tmp_path):
    selected = tmp_path / "Private Course"
    selected.mkdir()
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(str(selected)).encode(),
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = SubprocessFolderPicker(Path(sys.executable), timeout_seconds=12.5).choose_folder()

    assert result == selected.resolve()
    assert observed["command"] == [
        str(Path(sys.executable)),
        "-m",
        "exam_predictor.workspace.picker_helper",
    ]
    assert str(selected) not in " ".join(observed["command"])
    assert observed["kwargs"] == {
        "capture_output": True,
        "stdin": subprocess.DEVNULL,
        "check": False,
        "timeout": 12.5,
        "shell": False,
    }


def test_picker_returns_none_when_the_helper_reports_cancel(monkeypatch):
    def fake_run(command, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(command, 0, stdout=b"null", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert SubprocessFolderPicker(Path(sys.executable)).choose_folder() is None


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected_code"),
    [
        (3, '"C:/private/course"', "folder_picker_helper_failed"),
        (0, "not-json", "folder_picker_response_invalid"),
        (0, "123", "folder_picker_response_invalid"),
        (0, "{}", "folder_picker_response_invalid"),
        (0, '""', "folder_picker_response_invalid"),
    ],
)
def test_picker_rejects_failed_or_invalid_helper_responses(
    monkeypatch, returncode, stdout, expected_code
):
    secret = "C:/private/course"

    def fake_run(command, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout=stdout.encode(),
            stderr=f"helper diagnostics mention {secret}".encode(),
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(FolderPickerError) as caught:
        SubprocessFolderPicker(Path(sys.executable)).choose_folder()

    assert caught.value.code == expected_code
    assert str(caught.value) == expected_code
    assert secret not in str(caught.value)


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_picker_rejects_a_selection_that_is_not_an_existing_directory(
    monkeypatch, tmp_path, kind
):
    selected = tmp_path / "private-selection"
    if kind == "file":
        selected.write_bytes(b"not a directory")

    def fake_run(command, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(str(selected)).encode(),
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(FolderPickerError) as caught:
        SubprocessFolderPicker(Path(sys.executable)).choose_folder()

    assert caught.value.code == "folder_picker_selection_invalid"
    assert str(selected) not in str(caught.value)


def test_picker_maps_a_timeout_to_a_safe_error(monkeypatch):
    def fake_run(command, **kwargs):
        del kwargs
        raise subprocess.TimeoutExpired(command, 1.0, output="C:/private/course")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(FolderPickerError) as caught:
        SubprocessFolderPicker(Path(sys.executable), timeout_seconds=1.0).choose_folder()

    assert caught.value.code == "folder_picker_timeout"
    assert "C:/private/course" not in str(caught.value)
    assert caught.value.__suppress_context__ is True


def test_picker_maps_process_start_failure_to_a_safe_error(monkeypatch):
    secret = "C:/private/python.exe"

    def fake_run(command, **kwargs):
        del command, kwargs
        raise OSError(f"cannot start {secret}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(FolderPickerError) as caught:
        SubprocessFolderPicker(Path(secret)).choose_folder()

    assert caught.value.code == "folder_picker_unavailable"
    assert secret not in str(caught.value)
    assert caught.value.__suppress_context__ is True


def test_picker_maps_invalid_utf8_stdout_to_a_safe_error(monkeypatch):
    def fake_run(command, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(command, 0, stdout=b"\xffprivate", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(FolderPickerError) as caught:
        SubprocessFolderPicker(Path(sys.executable)).choose_folder()

    assert caught.value.code == "folder_picker_response_invalid"
    assert "private" not in str(caught.value)
    assert caught.value.__suppress_context__ is True


@pytest.mark.parametrize(
    ("selection", "expected_output"),
    [("C:/课程\nWeek 1", '"C:/课程\\nWeek 1"'), ("", "null")],
)
def test_picker_helper_writes_one_json_value_and_always_destroys_root(
    monkeypatch, capsys, selection, expected_output
):
    events: list[object] = []

    class FakeRoot:
        def withdraw(self):
            events.append("withdraw")

        def destroy(self):
            events.append("destroy")

    monkeypatch.setattr(picker_helper.tkinter, "Tk", lambda: FakeRoot())
    monkeypatch.setattr(
        picker_helper.filedialog,
        "askdirectory",
        lambda **kwargs: events.append(("askdirectory", kwargs)) or selection,
    )

    picker_helper.main()

    assert capsys.readouterr().out == expected_output
    assert events == [
        "withdraw",
        ("askdirectory", {"mustexist": True}),
        "destroy",
    ]


def test_picker_helper_destroys_root_if_dialog_fails(monkeypatch):
    events: list[str] = []

    class FakeRoot:
        def withdraw(self):
            events.append("withdraw")

        def destroy(self):
            events.append("destroy")

    def fail(**kwargs):
        del kwargs
        raise RuntimeError("dialog failed")

    monkeypatch.setattr(picker_helper.tkinter, "Tk", lambda: FakeRoot())
    monkeypatch.setattr(picker_helper.filedialog, "askdirectory", fail)

    with pytest.raises(RuntimeError, match="dialog failed"):
        picker_helper.main()

    assert events == ["withdraw", "destroy"]
