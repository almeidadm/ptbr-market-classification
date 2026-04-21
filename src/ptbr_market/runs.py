"""Convenções de I/O para dados, artefatos e metadados de _runs_.

Cada run é um diretório em `artifacts/runs/` com o padrão
`{YYYYMMDD-HHMMSS}__{gen}__{model}__{variant}/` e três arquivos:
`metadata.json` (schema completo), `metrics.json` (cópia rápida) e
`predictions.csv` (`index, y_true, y_pred, y_score`).
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALLOWED_GENERATIONS: tuple[str, ...] = ("gen1", "gen2", "gen3", "ensemble")

REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "run_id",
    "git_commit",
    "started_at",
    "finished_at",
    "generation",
    "model",
    "variant",
    "config",
    "split",
    "threshold",
    "metrics",
    "efficiency",
)

_SLUG_BAD = re.compile(r"[^a-z0-9.+-]+")


def data_root() -> Path:
    return Path(os.environ.get("PTBR_DATA_ROOT", "data"))


def artifacts_root() -> Path:
    return Path(os.environ.get("PTBR_ARTIFACTS_ROOT", "artifacts"))


def raw_dir() -> Path:
    return data_root() / "raw"


def splits_dir() -> Path:
    return artifacts_root() / "splits"


def runs_dir() -> Path:
    return artifacts_root() / "runs"


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


def _slugify(s: str) -> str:
    """Normaliza para `[a-z0-9.+-]` (kebab-case). Caracteres fora da classe viram `-`."""
    return _SLUG_BAD.sub("-", s.strip().lower()).strip("-")


def new_run_dir(
    generation: str,
    model: str,
    variant: str,
    now: datetime | None = None,
) -> Path:
    """Cria e retorna `artifacts/runs/{YYYYMMDD-HHMMSS}__{gen}__{model}__{variant}/`.

    `model` e `variant` são slugificados; `generation` precisa ser literal
    (`gen1`, `gen2`, `gen3` ou `ensemble`) para manter a nomenclatura do
    artigo estável.
    """
    if generation not in ALLOWED_GENERATIONS:
        raise ValueError(
            f"generation deve ser um de {ALLOWED_GENERATIONS}, recebido {generation!r}."
        )
    model_slug = _slugify(model)
    variant_slug = _slugify(variant)
    if not model_slug or not variant_slug:
        raise ValueError(
            "model e variant precisam produzir slugs não-vazios; recebido"
            f" model={model!r}→{model_slug!r}, variant={variant!r}→{variant_slug!r}"
        )

    stamp = (now if now is not None else datetime.now()).strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}__{generation}__{model_slug}__{variant_slug}"
    path = runs_dir() / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_split_metadata(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


def write_run_metadata(run_dir: Path, payload: dict[str, Any]) -> None:
    """Grava `metadata.json` validando chaves obrigatórias do schema do projeto."""
    missing = sorted(set(REQUIRED_METADATA_KEYS) - set(payload))
    if missing:
        raise ValueError(f"metadata.json incompleto: faltando chaves {missing!r}.")
    _write_json(run_dir / "metadata.json", payload)


def write_metrics(run_dir: Path, metrics: dict[str, Any]) -> None:
    """Grava `metrics.json` — cópia redundante do bloco `metrics` do metadata."""
    _write_json(run_dir / "metrics.json", metrics)


def write_predictions(
    run_dir: Path,
    indices: Iterable[int],
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float],
) -> None:
    """Grava `predictions.csv` com colunas `index, y_true, y_pred, y_score`."""
    idx = list(indices)
    yt = list(y_true)
    yp = list(y_pred)
    ys = list(y_score)
    n = len(idx)
    if not (len(yt) == n == len(yp) == len(ys)):
        raise ValueError(
            f"Tamanhos incompatíveis em predictions: index={n}, y_true={len(yt)},"
            f" y_pred={len(yp)}, y_score={len(ys)}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "y_true", "y_pred", "y_score"])
        writer.writerows(zip(idx, yt, yp, ys, strict=True))
