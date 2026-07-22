from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Protocol


class FolderPicker(Protocol):
    def choose_folder(self) -> Path | None:
        """Return the selected folder or None when the user cancels."""


class FolderPickerError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class SubprocessFolderPicker:
    def __init__(
        self, python_executable: Path, timeout_seconds: float = 300.0
    ) -> None:
        self._python_executable = python_executable
        self._timeout_seconds = timeout_seconds

    def choose_folder(self) -> Path | None:
        """Run the picker helper with captured stdio and validate its JSON response."""
        command = [
            str(self._python_executable),
            "-m",
            "exam_predictor.workspace.picker_helper",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                check=False,
                timeout=self._timeout_seconds,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            raise FolderPickerError("folder_picker_timeout") from None
        except OSError:
            raise FolderPickerError("folder_picker_unavailable") from None

        if completed.returncode != 0:
            raise FolderPickerError("folder_picker_helper_failed")

        try:
            response = completed.stdout.decode("utf-8")
            selected = json.loads(response)
        except (UnicodeError, json.JSONDecodeError, TypeError):
            raise FolderPickerError("folder_picker_response_invalid") from None

        if selected is None:
            return None
        if not isinstance(selected, str) or not selected:
            raise FolderPickerError("folder_picker_response_invalid")

        try:
            canonical = Path(selected).resolve(strict=True)
            if not canonical.is_dir():
                raise OSError
        except (OSError, RuntimeError):
            raise FolderPickerError("folder_picker_selection_invalid") from None
        return canonical
