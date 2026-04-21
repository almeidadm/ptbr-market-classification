"""Testes de representations.build_tfidf, build_bow, load_fasttext_vectors,
fasttext_average."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from ptbr_market import representations
from ptbr_market.representations import (
    BOW_DEFAULTS,
    FASTTEXT_DIM,
    TFIDF_DEFAULTS,
    build_bow,
    build_tfidf,
    fasttext_average,
    load_fasttext_vectors,
)

TRAIN_TEXTS: list[str] = [
    "petrobras anunciar lucro recorde trimestre",
    "empresa subir lucro ano passado",
    "bolsa fechar alta forte semana",
    "economia brasileiro crescer trimestre",
    "inflação atingir pico histórico país",
    "investidor comprar ação bolsa forte",
    "lucro empresa subir alta semana",
    "petrobras anunciar dividendos acionista",
]

VAL_TEXTS: list[str] = [
    "novo pregão bolsa valores",  # "novo", "pregão", "valores" são OOV do train
    "petrobras lucro recorde ano",
]


# -------- TF-IDF -----------------------------------------------------------


def test_build_tfidf_returns_vectorizer_and_sparse_matrix() -> None:
    vec, X = build_tfidf(TRAIN_TEXTS, min_df=1, max_df=1.0)
    assert issparse(X)
    assert isinstance(X, csr_matrix)
    assert X.shape[0] == len(TRAIN_TEXTS)
    assert X.shape[1] == len(vec.vocabulary_)


def test_build_tfidf_default_dtype_is_float32() -> None:
    vec, X = build_tfidf(TRAIN_TEXTS, min_df=1, max_df=1.0)
    assert X.dtype == np.float32


def test_build_tfidf_default_ngram_is_1_2() -> None:
    assert TFIDF_DEFAULTS["ngram_range"] == (1, 2)


def test_build_tfidf_fit_only_on_train_vocab() -> None:
    vec, _ = build_tfidf(TRAIN_TEXTS, min_df=1, max_df=1.0)
    vocab = set(vec.vocabulary_)
    X_val = vec.transform(VAL_TEXTS)
    assert X_val.shape == (len(VAL_TEXTS), len(vocab))
    assert "pregão" not in vocab  # OOV do train, não deve virar feature


def test_build_tfidf_respects_overrides() -> None:
    vec, _ = build_tfidf(TRAIN_TEXTS, ngram_range=(1, 1), min_df=1, max_df=1.0)
    assert vec.ngram_range == (1, 1)


def test_build_tfidf_sublinear_tf_default_true() -> None:
    assert TFIDF_DEFAULTS["sublinear_tf"] is True


def test_build_tfidf_rows_not_all_zero() -> None:
    _, X = build_tfidf(TRAIN_TEXTS, min_df=1, max_df=1.0)
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    assert (row_sums > 0).all()


# -------- BoW --------------------------------------------------------------


def test_build_bow_returns_count_matrix() -> None:
    vec, X = build_bow(TRAIN_TEXTS, min_df=1, max_df=1.0)
    assert issparse(X)
    assert X.shape[0] == len(TRAIN_TEXTS)
    # Contagens são inteiros não-negativos.
    assert X.min() >= 0
    assert np.issubdtype(X.dtype, np.integer)


def test_build_bow_default_ngram_is_1_1() -> None:
    assert BOW_DEFAULTS["ngram_range"] == (1, 1)


def test_build_bow_fit_only_on_train() -> None:
    vec, _ = build_bow(TRAIN_TEXTS, min_df=1, max_df=1.0)
    X_val = vec.transform(VAL_TEXTS)
    assert X_val.shape[1] == len(vec.vocabulary_)


# -------- fastText loader --------------------------------------------------


def _write_fake_vec_file(path: Path, tokens_vectors: dict[str, list[float]]) -> None:
    """Grava um .vec fake (cabeçalho + 'token v1 v2 ...') com dimensão FASTTEXT_DIM."""
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(tokens_vectors)} {FASTTEXT_DIM}\n")
        for tok, vec in tokens_vectors.items():
            assert len(vec) == FASTTEXT_DIM
            f.write(f"{tok} " + " ".join(f"{v:.6f}" for v in vec) + "\n")


@pytest.fixture
def fake_fasttext_file(tmp_path: Path) -> Path:
    path = tmp_path / "fake.vec"
    rng = np.random.default_rng(1)
    tokens = {
        "petrobras": rng.normal(size=FASTTEXT_DIM).tolist(),
        "lucro": rng.normal(size=FASTTEXT_DIM).tolist(),
        "bolsa": rng.normal(size=FASTTEXT_DIM).tolist(),
    }
    _write_fake_vec_file(path, tokens)
    return path


def test_load_fasttext_reads_fake_file(fake_fasttext_file: Path) -> None:
    vectors = load_fasttext_vectors(fake_fasttext_file)
    assert set(vectors) == {"petrobras", "lucro", "bolsa"}
    for v in vectors.values():
        assert v.shape == (FASTTEXT_DIM,)
        assert v.dtype == np.float32


def test_load_fasttext_filters_by_vocabulary(fake_fasttext_file: Path) -> None:
    vectors = load_fasttext_vectors(fake_fasttext_file, vocabulary=["lucro", "xyzunk"])
    assert set(vectors) == {"lucro"}


def test_load_fasttext_rejects_wrong_dimension(tmp_path: Path) -> None:
    bad = tmp_path / "bad.vec"
    with bad.open("w", encoding="utf-8") as f:
        f.write("1 50\n")
        f.write("foo " + " ".join(["0.1"] * 50) + "\n")
    with pytest.raises(ValueError, match="dimensão"):
        load_fasttext_vectors(bad)


def test_load_fasttext_missing_file_raises_with_download_hint(tmp_path: Path) -> None:
    missing = tmp_path / "nope.vec"
    with pytest.raises(FileNotFoundError, match="fasttext"):
        load_fasttext_vectors(missing)


# -------- fasttext_average -------------------------------------------------


def _synthetic_vectors() -> dict[str, np.ndarray]:
    return {
        "petrobras": np.ones(FASTTEXT_DIM, dtype=np.float32),
        "lucro": np.full(FASTTEXT_DIM, 2.0, dtype=np.float32),
        "bolsa": np.full(FASTTEXT_DIM, 3.0, dtype=np.float32),
    }


def test_fasttext_average_basic_shape_and_dtype() -> None:
    vectors = _synthetic_vectors()
    out = fasttext_average(["petrobras lucro", "bolsa"], vectors)
    assert out.shape == (2, FASTTEXT_DIM)
    assert out.dtype == np.float32


def test_fasttext_average_means_known_vectors() -> None:
    vectors = _synthetic_vectors()
    out = fasttext_average(["petrobras lucro"], vectors)
    np.testing.assert_allclose(out[0], np.full(FASTTEXT_DIM, 1.5, dtype=np.float32))


def test_fasttext_average_ignores_oov_tokens() -> None:
    vectors = _synthetic_vectors()
    out = fasttext_average(["petrobras xpto"], vectors)
    np.testing.assert_allclose(out[0], np.ones(FASTTEXT_DIM, dtype=np.float32))


def test_fasttext_average_empty_text_returns_zero() -> None:
    vectors = _synthetic_vectors()
    out = fasttext_average(["", "petrobras"], vectors)
    np.testing.assert_array_equal(out[0], np.zeros(FASTTEXT_DIM, dtype=np.float32))
    np.testing.assert_allclose(out[1], np.ones(FASTTEXT_DIM, dtype=np.float32))


def test_fasttext_average_all_oov_returns_zero() -> None:
    vectors = _synthetic_vectors()
    out = fasttext_average(["foo bar"], vectors)
    np.testing.assert_array_equal(out[0], np.zeros(FASTTEXT_DIM, dtype=np.float32))


def test_fasttext_average_is_deterministic() -> None:
    vectors = _synthetic_vectors()
    a = fasttext_average(["petrobras lucro bolsa"], vectors)
    b = fasttext_average(["petrobras lucro bolsa"], vectors)
    np.testing.assert_array_equal(a, b)


# -------- Caminho integrado: load + average -------------------------------


def test_load_then_average_roundtrip(fake_fasttext_file: Path) -> None:
    vectors = load_fasttext_vectors(fake_fasttext_file)
    out = fasttext_average(["petrobras lucro desconhecido", ""], vectors)
    assert out.shape == (2, FASTTEXT_DIM)
    np.testing.assert_array_equal(out[1], np.zeros(FASTTEXT_DIM, dtype=np.float32))
    expected = (vectors["petrobras"] + vectors["lucro"]) / 2.0
    np.testing.assert_allclose(out[0], expected, atol=1e-6)


# -------- fasttext_path defaults ------------------------------------------


def test_fasttext_path_respects_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PTBR_DATA_ROOT", str(tmp_path))
    p = representations.fasttext_path()
    assert p == tmp_path / "raw" / "cc.pt.300.vec"
