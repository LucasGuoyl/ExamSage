from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from .api import WorkerSettings, create_worker_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExamSage local Agent Worker")
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("EXAMSAGE_WORKER_TOKEN", "")
    if not token.strip():
        raise SystemExit("EXAMSAGE_WORKER_TOKEN is required.")
    data_dir = Path(
        os.environ.get("EXAMSAGE_DATA_DIR", Path.home() / ".examsage")
    )
    settings = WorkerSettings(
        host=args.host,
        port=args.port,
        token=token,
        data_dir=data_dir,
    )
    uvicorn.run(
        create_worker_app(settings),
        host=settings.host,
        port=settings.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
