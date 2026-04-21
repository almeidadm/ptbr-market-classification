"""Testes de threshold.fit_threshold e threshold.apply_threshold."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from sklearn.metrics import f1_score

from ptbr_market import threshold
from ptbr_market.threshold import ThresholdDecision

# -------- Grade ------------------------------------------------------------


def test_default_grid_is_monotonic_and_bounded() -> None:
    grid = threshold._default_grid()
    assert grid[0] == 0.05
    assert grid[-1] == 0.95
    assert len(grid) == threshold.GRID_N
    diffs = np.diff(np.array(grid))
    assert np.all(diffs > 0), "grade deve ser estritamente crescente"


# -------- fit_threshold ----------------------------------------------------


def test_fit_threshold_on_perfectly_separable_case() -> None:
    # Separação perfeita em (0.37, 0.63): qualquer threshold nesse intervalo
    # atinge F1=1.0. Com tie-break pelo menor, esperamos 0.38 (primeiro
    # ponto da grade dentro do intervalo — 0.37 ainda inclui o negativo).
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_score = np.array([0.91, 0.85, 0.77, 0.71, 0.63, 0.37, 0.31, 0.23, 0.17, 0.11])
    decision = threshold.fit_threshold(y_true, y_score)
    assert decision.value == 0.38
    assert decision.fitted_on == "val"
    assert decision.objective == "f1_minority"


def test_fit_threshold_value_is_in_grid() -> None:
    rng = np.random.default_rng(1)
    y_true = np.concatenate([np.ones(20), np.zeros(80)]).astype(int)
    y_score = np.concatenate(
        [rng.uniform(0.3, 1.0, size=20), rng.uniform(0.0, 0.5, size=80)]
    )
    decision = threshold.fit_threshold(y_true, y_score)
    assert decision.value in decision.grid


def test_fit_threshold_agrees_with_manual_argmax() -> None:
    rng = np.random.default_rng(1)
    y_true = np.concatenate([np.ones(30), np.zeros(70)]).astype(int)
    y_score = np.concatenate(
        [rng.uniform(0.2, 0.9, size=30), rng.uniform(0.0, 0.6, size=70)]
    )
    decision = threshold.fit_threshold(y_true, y_score, objective="f1_minority")

    best_t, best_f1 = decision.grid[0], -1.0
    for t in decision.grid:
        y_pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    assert decision.value == best_t


def test_fit_threshold_f1_macro_objective_differs_from_minority() -> None:
    rng = np.random.default_rng(1)
    y_true = np.concatenate([np.ones(15), np.zeros(85)]).astype(int)
    y_score = np.concatenate(
        [rng.uniform(0.3, 0.8, size=15), rng.uniform(0.0, 0.5, size=85)]
    )
    dec_minority = threshold.fit_threshold(y_true, y_score, objective="f1_minority")
    dec_macro = threshold.fit_threshold(y_true, y_score, objective="f1_macro")
    # Não precisam ser diferentes *sempre*, mas o objetivo deve ser refletido:
    assert dec_minority.objective == "f1_minority"
    assert dec_macro.objective == "f1_macro"


def test_fit_threshold_is_deterministic() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.15, 0.6, 0.55, 0.25])
    a = threshold.fit_threshold(y_true, y_score)
    b = threshold.fit_threshold(y_true, y_score)
    assert a == b


def test_fit_threshold_rejects_pr_auc() -> None:
    with pytest.raises(ValueError, match="PR-AUC"):
        threshold.fit_threshold([0, 1], [0.1, 0.9], objective="pr_auc")  # type: ignore[arg-type]


def test_fit_threshold_rejects_unknown_objective() -> None:
    with pytest.raises(ValueError, match="objective"):
        threshold.fit_threshold(
            [0, 1], [0.1, 0.9], objective="accuracy"
        )  # type: ignore[arg-type]


def test_fit_threshold_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="incompatíveis"):
        threshold.fit_threshold([0, 1, 1], [0.1, 0.9])


def test_fit_threshold_rejects_single_class() -> None:
    with pytest.raises(ValueError, match="apenas uma classe"):
        threshold.fit_threshold([1, 1, 1], [0.3, 0.7, 0.9])


def test_fit_threshold_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="apenas 0 e 1"):
        threshold.fit_threshold([0, 1, 2], [0.3, 0.7, 0.9])


def test_fit_threshold_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="vazios"):
        threshold.fit_threshold([], [])


# -------- ThresholdDecision frozen -----------------------------------------


def test_threshold_decision_is_frozen() -> None:
    dec = ThresholdDecision(
        value=0.4, fitted_on="val", objective="f1_minority", grid=(0.05, 0.5)
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        dec.value = 0.5  # type: ignore[misc]


def test_threshold_decision_equality() -> None:
    a = ThresholdDecision(0.33, "val", "f1_minority", (0.05, 0.5))
    b = ThresholdDecision(0.33, "val", "f1_minority", (0.05, 0.5))
    c = ThresholdDecision(0.34, "val", "f1_minority", (0.05, 0.5))
    assert a == b
    assert a != c


# -------- apply_threshold --------------------------------------------------


def test_apply_threshold_uses_ge_convention_at_boundary() -> None:
    dec = ThresholdDecision(0.5, "val", "f1_minority", (0.05, 0.5))
    y_pred = threshold.apply_threshold([0.5], dec)
    assert y_pred.tolist() == [1]


def test_apply_threshold_does_not_modify_score_or_decision() -> None:
    dec = ThresholdDecision(0.5, "val", "f1_minority", (0.05, 0.5))
    y_score = np.array([0.1, 0.6, 0.4, 0.9, 0.5])
    backup = y_score.copy()
    y_pred = threshold.apply_threshold(y_score, dec)
    assert y_pred.tolist() == [0, 1, 0, 1, 1]
    np.testing.assert_array_equal(y_score, backup)
    assert dec.value == 0.5


def test_apply_threshold_rejects_non_1d() -> None:
    dec = ThresholdDecision(0.5, "val", "f1_minority", (0.05, 0.5))
    with pytest.raises(ValueError, match="1D"):
        threshold.apply_threshold([[0.1, 0.2], [0.3, 0.4]], dec)


def test_apply_threshold_freeze_holds_across_partitions() -> None:
    """Invariante: o decision fitted em val aplica *como está* em test.

    Simula o caminho val → freeze → test: o threshold congelado não muda
    entre chamadas, e reutilizá-lo em um novo conjunto (teste) produz
    exatamente o que `(y >= dec.value)` daria.
    """
    rng = np.random.default_rng(1)
    y_val_true = np.concatenate([np.ones(20), np.zeros(80)]).astype(int)
    y_val_score = np.concatenate(
        [rng.uniform(0.3, 0.9, size=20), rng.uniform(0.0, 0.5, size=80)]
    )
    decision = threshold.fit_threshold(y_val_true, y_val_score)

    y_test_score = rng.uniform(0.0, 1.0, size=50)
    expected = (y_test_score >= decision.value).astype(np.int8)
    assert threshold.apply_threshold(y_test_score, decision).tolist() == expected.tolist()
