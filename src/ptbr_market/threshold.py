"""Ajuste e congelamento do limiar de decisão.

Invariante metodológica (ver CLAUDE.md): o threshold é ajustado **apenas**
na partição de validação via `fit_threshold`, e o objeto `ThresholdDecision`
que ele retorna é imutável (`frozen=True`). `apply_threshold` consome a
decisão mas não a reajusta — não existe API pública para re-ajustar em
teste, por construção.

Objetivo fixado: F1 (`f1_minority` por default ou `f1_macro`). PR-AUC e
ROC-AUC não dependem de threshold e portanto não são objetivos válidos.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.metrics import f1_score

ALLOWED_OBJECTIVES: tuple[str, ...] = ("f1_minority", "f1_macro")
GRID_START = 0.05
GRID_END = 0.95
GRID_N = 91  # 0.05, 0.06, ..., 0.95

Objective = Literal["f1_minority", "f1_macro"]


def _default_grid() -> tuple[float, ...]:
    values = np.round(np.linspace(GRID_START, GRID_END, GRID_N), 2)
    return tuple(float(v) for v in values)


@dataclass(frozen=True, slots=True)
class ThresholdDecision:
    """Decisão congelada de threshold, ajustada em validação.

    - `value`: threshold escolhido (ponto da `grid`).
    - `fitted_on`: sempre `"val"`; redundante mas torna o contrato explícito
      no `metadata.json` de cada run.
    - `objective`: métrica otimizada.
    - `grid`: grade varrida, preservada para auditoria/reprodutibilidade.
    """

    value: float
    fitted_on: Literal["val"]
    objective: Objective
    grid: tuple[float, ...]


def fit_threshold(
    y_val_true: Iterable[Any],
    y_val_score: Iterable[Any],
    objective: Objective = "f1_minority",
) -> ThresholdDecision:
    """Varre a grade e escolhe o threshold que maximiza o F1 pedido.

    Em caso de empate no F1, retorna o **menor** threshold — convenção
    conservadora que favorece recall da classe minoritária em empates
    numéricos (esperado em grades densas sobre validações pequenas).
    """
    if objective not in ALLOWED_OBJECTIVES:
        raise ValueError(
            f"objective deve ser um de {ALLOWED_OBJECTIVES}, recebido"
            f" {objective!r}. PR-AUC e ROC-AUC não dependem de threshold e"
            " não são objetivos válidos aqui."
        )

    y_true = np.asarray(list(y_val_true) if not hasattr(y_val_true, "__array__") else y_val_true)
    y_score = np.asarray(
        list(y_val_score) if not hasattr(y_val_score, "__array__") else y_val_score,
        dtype=np.float64,
    )

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_val_true e y_val_score devem ser 1D.")
    if len(y_true) != len(y_score):
        raise ValueError(
            f"Tamanhos incompatíveis: y_val_true={len(y_true)},"
            f" y_val_score={len(y_score)}"
        )
    if len(y_true) == 0:
        raise ValueError("Inputs vazios.")
    unique_labels = np.unique(y_true)
    if not np.isin(unique_labels, (0, 1)).all():
        raise ValueError(
            f"y_val_true deve conter apenas 0 e 1, recebido {unique_labels.tolist()!r}"
        )
    if len(unique_labels) < 2:
        raise ValueError(
            "y_val_true contém apenas uma classe — ajuste de threshold degenerado."
        )

    grid = _default_grid()
    best_value = grid[0]
    best_f1 = -1.0

    for t in grid:
        y_pred = (y_score >= t).astype(np.int8)
        if objective == "f1_minority":
            f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        else:
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_value = t

    return ThresholdDecision(
        value=float(best_value),
        fitted_on="val",
        objective=objective,
        grid=grid,
    )


def apply_threshold(
    y_score: Iterable[Any],
    decision: ThresholdDecision,
) -> np.ndarray:
    """Aplica `decision.value` a `y_score` retornando 0/1 com convenção `>=`.

    Não modifica `decision` nem o array de entrada.
    """
    raw = list(y_score) if not hasattr(y_score, "__array__") else y_score
    y = np.asarray(raw, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("y_score deve ser 1D.")
    return (y >= decision.value).astype(np.int8)
