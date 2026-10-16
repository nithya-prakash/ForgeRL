"""Local MLflow helpers (SQLite backend, no server required)."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

from factory_scheduler.config import ROOT

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@contextmanager
def tracked_run(
    experiment: str, run_name: str, params: dict[str, Any], tags: dict[str, str] | None = None
) -> Iterator[mlflow.ActiveRun]:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"git_commit": git_commit(), **(tags or {})})
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        yield run


def log_file(path: Path, artifact_dir: str | None = None) -> None:
    mlflow.log_artifact(str(path), artifact_dir)
