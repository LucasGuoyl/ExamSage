"""Create a timestamped local backup without API keys or transient uploads."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


data_dir = Path(os.environ.get("EXAMSAGE_DATA_DIR", Path.home() / ".examsage"))
backup_dir = Path.home() / "ExamSage Backups"
backup_dir.mkdir(parents=True, exist_ok=True)
target = backup_dir / f"examsage-{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"

if not data_dir.exists():
    raise SystemExit(f"No ExamSage data found at {data_dir}")

with ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
    for path in data_dir.rglob("*"):
        relative = path.relative_to(data_dir)
        if not path.is_file() or "intake" in relative.parts:
            continue
        if path.suffix.lower() in {".key", ".env"}:
            continue
        archive.write(path, relative)
print(f"Backup created: {target}")
print("API keys are session-only and are not included.")
