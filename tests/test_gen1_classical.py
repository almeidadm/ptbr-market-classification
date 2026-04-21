"""Testes dos builders de classificadores Gen 1."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import is_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from ptbr_market.gen1_classical import GEN1_CLASSIFIERS, SEED, build_classifier


def _toy_sparse_data(
    n_pos: int = 40,
    n_neg: int = 60,
    n_features: int = 20,
    seed: int = 1,
) -> tuple[csr_matrix, np.ndarray]:
    """Features inteiras não-negativas (compatíveis com MultinomialNB/ComplementNB)."""
    rng = np.random.default_rng(seed)
    X_pos = rng.integers(0, 5, size=(n_pos, n_features))
    X_neg = rng.integers(0, 3, size=(n_neg, n_features))
    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    return csr_matrix(X), y


def test_gen1_classifiers_is_stable_tuple() -> None:
    assert isinstance(GEN1_CLASSIFIERS, tuple)
    assert GEN1_CLASSIFIERS == ("linearsvc", "logreg", "multinomialnb", "complementnb")


def test_seed_constant_is_1() -> None:
    assert SEED == 1


@pytest.mark.parametrize("name", GEN1_CLASSIFIERS)
def test_build_classifier_is_sklearn_estimator(name: str) -> None:
    clf = build_classifier(name)
    assert is_classifier(clf)


def test_linearsvc_is_calibrated() -> None:
    clf = build_classifier("linearsvc")
    assert isinstance(clf, CalibratedClassifierCV)


def test_logreg_has_expected_type() -> None:
    assert isinstance(build_classifier("logreg"), LogisticRegression)


def test_multinomialnb_has_expected_type() -> None:
    assert isinstance(build_classifier("multinomialnb"), MultinomialNB)


def test_complementnb_has_expected_type() -> None:
    assert isinstance(build_classifier("complementnb"), ComplementNB)


@pytest.mark.parametrize("name", GEN1_CLASSIFIERS)
def test_build_classifier_fits_and_predicts_proba(name: str) -> None:
    X, y = _toy_sparse_data()
    clf = build_classifier(name)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("name", GEN1_CLASSIFIERS)
def test_build_classifier_is_deterministic(name: str) -> None:
    X, y = _toy_sparse_data()
    p1 = build_classifier(name).fit(X, y).predict_proba(X)
    p2 = build_classifier(name).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(p1, p2)


def test_build_classifier_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Classificador desconhecido"):
        build_classifier("xgboost")


def test_build_classifier_unknown_name_lists_allowed() -> None:
    with pytest.raises(ValueError) as exc:
        build_classifier("unknown")
    for name in GEN1_CLASSIFIERS:
        assert name in str(exc.value)
