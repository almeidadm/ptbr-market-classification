"""Convenções de I/O para dados, artefatos e metadados de _runs_."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def data_root() -> Path:
    return Path(os.environ.get("PTBR_DATA_ROOT", "data"))


def artifacts_root() -> Path:
    return Path(os.environ.get("PTBR_ARTIFACTS_ROOT", "artifacts"))


def raw_dir() -> Path:
    return data_root() / "raw"


def splits_dir() -> Path:
    return artifacts_root() / "splits"


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_split_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
