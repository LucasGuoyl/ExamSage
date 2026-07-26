from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Protocol

from exam_predictor.workspace.filesystem import (
    FILE_ATTRIBUTE_REPARSE_POINT,
    SecureFileOpener,
    SecureOpenError,
)


class FolderPicker(Protocol):
    def choose_folder(self) -> Path | None:
        """Return the selected folder or None when the user cancels."""


class FolderPickerError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _reject_linked_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.relative_to(current).parts:
        current /= part
        metadata = current.stat(follow_symlinks=False)
        attributes = getattr(metadata, "st_file_attributes", 0)
        if stat.S_ISLNK(metadata.st_mode) or attributes & FILE_ATTRIBUTE_REPARSE_POINT:
            raise SecureOpenError("source_link_or_reparse")


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
            lexical = Path(selected).absolute()
            anchor = Path(lexical.anchor)
            relative = lexical.relative_to(anchor)
            relative_path = (
                PurePosixPath(*relative.parts) if relative.parts else None
            )
            opener = SecureFileOpener()
            _reject_linked_components(lexical)
            if os.name == "nt":
                with opener.anchor_root(lexical) as root_anchor:
                    canonical = root_anchor.canonical_root
            else:
                with opener.anchor_directory(anchor, relative_path) as directory_fd:
                    with opener.anchor_root(lexical) as root_anchor:
                        if directory_fd is not None:
                            metadata = os.fstat(directory_fd)
                            if root_anchor.identity != (
                                metadata.st_dev,
                                metadata.st_ino,
                            ):
                                raise SecureOpenError("source_root_invalid")
                        canonical = root_anchor.canonical_root
        except (OSError, RuntimeError, ValueError):
            raise FolderPickerError("folder_picker_selection_invalid") from None
        return canonical
