"""Testes de evaluation.compute_metrics e evaluation.mcnemar_test."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from ptbr_market import evaluation

# -------- compute_metrics ---------------------------------------------------


def test_compute_metrics_keys_and_order() -> None:
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.6, 0.9]
    y_pred = [0, 0, 1, 1]
    metrics = evaluation.compute_metrics(y_true, y_score, y_pred)
    assert tuple(metrics) == evaluation.METRIC_KEYS


def test_compute_metrics_matches_sklearn_on_toy_case() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_score = np.array([0.1, 0.4, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.55, 0.05])
    y_pred = (y_score >= 0.5).astype(int)

    metrics = evaluation.compute_metrics(y_true, y_score, y_pred)

    assert metrics["pr_auc"] == pytest.approx(average_precision_score(y_true, y_score))
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y_true, y_score))
    assert metrics["f1_macro"] == pytest.approx(f1_score(y_true, y_pred, average="macro"))
    assert metrics["f1_minority"] == pytest.approx(f1_score(y_true, y_pred, pos_label=1))


def test_compute_metrics_confusion_matrix_layout() -> None:
    # 3 TN, 1 FP, 2 FN, 4 TP
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.6, 0.4, 0.45, 0.7, 0.8, 0.9, 0.95]
    metrics = evaluation.compute_metrics(y_true, y_score, y_pred)
    assert metrics["confusion_matrix"] == [[3, 1], [2, 4]]


def test_compute_metrics_f1_minority_differs_from_macro_on_imbalanced() -> None:
    # ~10% positivos, modelo recupera só parte deles.
    rng = np.random.default_rng(1)
    y_true = np.concatenate([np.zeros(90), np.ones(10)]).astype(int)
    y_score = np.concatenate(
        [rng.uniform(0.0, 0.5, size=90), rng.uniform(0.3, 1.0, size=10)]
    )
    y_pred = (y_score >= 0.5).astype(int)

    metrics = evaluation.compute_metrics(y_true, y_score, y_pred)
    assert metrics["f1_minority"] != metrics["f1_macro"]


def test_compute_metrics_pr_auc_invariant_under_monotonic_scaling() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.15, 0.6])
    y_pred = (y_score >= 0.5).astype(int)

    baseline = evaluation.compute_metrics(y_true, y_score, y_pred)
    scaled = evaluation.compute_metrics(y_true, 2.5 * y_score + 7, y_pred)
    assert scaled["pr_auc"] == pytest.approx(baseline["pr_auc"])
    assert scaled["roc_auc"] == pytest.approx(baseline["roc_auc"])


def test_compute_metrics_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="y_true"):
        evaluation.compute_metrics([0, 1, 2], [0.1, 0.5, 0.9], [0, 1, 1])


def test_compute_metrics_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="incompatíveis"):
        evaluation.compute_metrics([0, 1, 1], [0.1, 0.5], [0, 1, 1])


def test_compute_metrics_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError):
        evaluation.compute_metrics([], [], [])


def test_compute_metrics_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="1D"):
        evaluation.compute_metrics([[0, 1], [1, 0]], [0.5, 0.5], [0, 1])


# -------- mcnemar_test ------------------------------------------------------


def test_mcnemar_identical_models_returns_p_one() -> None:
    y_true = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    result = evaluation.mcnemar_test(y_true, y_true, y_true)
    assert result["b"] == 0
    assert result["c"] == 0
    assert result["statistic"] == 0.0
    assert result["p_value"] == 1.0


def test_mcnemar_counts_b_and_c_correctly_hand_case() -> None:
    # b = A acerta & B erra: posições 1, 6 → b=2
    # c = A erra & B acerta: posições 2, 4, 7, 9 → c=4
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_a = [1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
    y_b = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    result = evaluation.mcnemar_test(y_true, y_a, y_b)
    assert result["b"] == 2
    assert result["c"] == 4
    # b + c = 6 < 25 → caminho exato binomial, statistic = min(b, c) = 2.
    expected = binomtest(2, 6, p=0.5, alternative="two-sided").pvalue
    assert result["p_value"] == pytest.approx(expected)
    assert result["statistic"] == 2.0


def test_mcnemar_dispatches_to_chi_squared_for_large_disagreement() -> None:
    # Construir um caso com b + c = 40 para forçar o caminho chi-quadrado.
    rng = np.random.default_rng(1)
    n_samples = 200
    y_true = rng.integers(0, 2, size=n_samples)
    y_a = y_true.copy()
    y_b = y_true.copy()
    # 25 posições: A acerta, B erra.
    y_b[:25] = 1 - y_true[:25]
    # 15 posições: A erra, B acerta.
    y_a[25:40] = 1 - y_true[25:40]

    result = evaluation.mcnemar_test(y_true, y_a, y_b)
    assert result["b"] == 25
    assert result["c"] == 15
    expected_stat = (abs(25 - 15) - 1) ** 2 / 40
    assert result["statistic"] == pytest.approx(expected_stat)
    assert result["p_value"] == pytest.approx(chi2.sf(expected_stat, df=1))


def test_mcnemar_threshold_boundary() -> None:
    # Force n = 24 (< 25, caminho exato) mesmo com "muitas" discordâncias.
    y_true = [1] * 30
    y_a = [1] * 18 + [0] * 12
    y_b = [1] * 18 + [1] * 12  # A erra 12, B acerta 12 → c=12, b=0
    # Espera caminho exato: b+c = 12 < 25.
    result = evaluation.mcnemar_test(y_true, y_a, y_b)
    assert (result["b"], result["c"]) == (0, 12)
    expected = binomtest(0, 12, p=0.5, alternative="two-sided").pvalue
    assert result["p_value"] == pytest.approx(expected)


def test_mcnemar_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="incompatíveis"):
        evaluation.mcnemar_test([0, 1, 1], [0, 1], [0, 1, 1])


def test_mcnemar_accepts_numpy_and_list_inputs() -> None:
    y_true_list = [0, 1, 0, 1, 1, 0]
    y_a_np = np.array([0, 1, 0, 0, 1, 1])
    y_b_np = np.array([0, 0, 0, 1, 1, 0])
    result = evaluation.mcnemar_test(y_true_list, y_a_np, y_b_np)
    assert set(result) == {"statistic", "p_value", "b", "c"}
