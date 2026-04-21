"""Métricas de avaliação para classificação binária (mercado vs. outros).

A classe positiva é sempre 1 (`mercado`). PR-AUC é a métrica principal do
projeto — em um corpus com ~12% de positivos, ROC-AUC infla por conta da
massa de verdadeiros negativos e não reflete a qualidade da detecção.

McNemar é o teste de significância modelo-vs-modelo adotado no plano
(`plano_base.md`).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

POSITIVE_LABEL = 1
MCNEMAR_EXACT_THRESHOLD = 25

METRIC_KEYS: tuple[str, ...] = (
    "pr_auc",
    "roc_auc",
    "f1_macro",
    "f1_minority",
    "precision_minority",
    "recall_minority",
    "confusion_matrix",
)


def _as_binary_array(arr: Iterable[Any], name: str) -> np.ndarray:
    a = np.asarray(list(arr) if not hasattr(arr, "__array__") else arr)
    if a.ndim != 1:
        raise ValueError(f"{name} deve ser 1D, recebido shape={a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} está vazio")
    unique = np.unique(a)
    if not np.all(np.isin(unique, (0, 1))):
        raise ValueError(f"{name} deve conter apenas 0 e 1, recebido valores {unique.tolist()!r}")
    return a.astype(np.int8)


def _as_score_array(arr: Iterable[Any], name: str) -> np.ndarray:
    a = np.asarray(list(arr) if not hasattr(arr, "__array__") else arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} deve ser 1D, recebido shape={a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} está vazio")
    return a


def compute_metrics(
    y_true: Iterable[Any],
    y_score: Iterable[Any],
    y_pred: Iterable[Any],
) -> dict[str, Any]:
    """Calcula as métricas reportadas em cada run, na ordem de `METRIC_KEYS`.

    - `y_true`, `y_pred`: 0/1 (1 = mercado).
    - `y_score`: pontuação para a classe positiva (probabilidade calibrada
      ou saída monotônica de `decision_function` — PR-AUC/ROC-AUC são
      invariantes a transformações monotônicas).

    A matriz de confusão é `[[TN, FP], [FN, TP]]`.
    """
    y_true_arr = _as_binary_array(y_true, "y_true")
    y_pred_arr = _as_binary_array(y_pred, "y_pred")
    y_score_arr = _as_score_array(y_score, "y_score")

    n = len(y_true_arr)
    if not (len(y_pred_arr) == n == len(y_score_arr)):
        raise ValueError(
            f"Tamanhos incompatíveis: y_true={n}, y_score={len(y_score_arr)},"
            f" y_pred={len(y_pred_arr)}"
        )

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])

    return {
        "pr_auc": float(average_precision_score(y_true_arr, y_score_arr)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)),
        "f1_macro": float(f1_score(y_true_arr, y_pred_arr, average="macro")),
        "f1_minority": float(f1_score(y_true_arr, y_pred_arr, pos_label=POSITIVE_LABEL)),
        "precision_minority": float(
            precision_score(
                y_true_arr, y_pred_arr, pos_label=POSITIVE_LABEL, zero_division=0
            )
        ),
        "recall_minority": float(
            recall_score(
                y_true_arr, y_pred_arr, pos_label=POSITIVE_LABEL, zero_division=0
            )
        ),
        "confusion_matrix": cm.tolist(),
    }


def mcnemar_test(
    y_true: Iterable[Any],
    y_pred_a: Iterable[Any],
    y_pred_b: Iterable[Any],
    exact_threshold: int = MCNEMAR_EXACT_THRESHOLD,
) -> dict[str, float | int]:
    """Teste de McNemar para diferença entre dois classificadores binários.

    Conta discordâncias:

    - `b`: A acerta, B erra.
    - `c`: A erra, B acerta.

    Para `b + c < exact_threshold`, usa teste binomial exato com p=0.5.
    Caso contrário, usa chi-quadrado (df=1) com correção de continuidade:
    `(|b - c| - 1)^2 / (b + c)`.

    Quando `b + c == 0` (modelos idênticos), retorna `p_value=1.0` sem
    invocar scipy.
    """
    y_true_arr = _as_binary_array(y_true, "y_true")
    y_a = _as_binary_array(y_pred_a, "y_pred_a")
    y_b = _as_binary_array(y_pred_b, "y_pred_b")

    n = len(y_true_arr)
    if not (len(y_a) == n == len(y_b)):
        raise ValueError(
            f"Tamanhos incompatíveis: y_true={n}, y_pred_a={len(y_a)},"
            f" y_pred_b={len(y_b)}"
        )

    correct_a = y_a == y_true_arr
    correct_b = y_b == y_true_arr
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    disagreements = b + c

    if disagreements == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}

    if disagreements < exact_threshold:
        result = binomtest(min(b, c), disagreements, p=0.5, alternative="two-sided")
        return {
            "statistic": float(min(b, c)),
            "p_value": float(result.pvalue),
            "b": b,
            "c": c,
        }

    statistic = (abs(b - c) - 1) ** 2 / disagreements
    return {
        "statistic": float(statistic),
        "p_value": float(chi2.sf(statistic, df=1)),
        "b": b,
        "c": c,
    }
