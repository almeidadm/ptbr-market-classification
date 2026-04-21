"""Testes de preprocessing.mask_financial_entities, preprocess_raw e preprocess_aggressive.

Testes que exercitam o SpaCy `pt_core_news_lg` são marcados como `slow` e
usam `skip_if_no_spacy_lg` para degradar com mensagem útil se o modelo
não estiver presente.
"""

from __future__ import annotations

import pytest

from ptbr_market import preprocessing
from ptbr_market.preprocessing import (
    MONEY_PLACEHOLDER,
    PERCENT_PLACEHOLDER,
    mask_financial_entities,
    preprocess_aggressive,
    preprocess_raw,
)

# -------- mask_financial_entities -----------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("R$ 10 bi", MONEY_PLACEHOLDER),
        ("R$10", MONEY_PLACEHOLDER),
        ("R$ 1.234,56", MONEY_PLACEHOLDER),
        ("US$ 2,5 milhões", MONEY_PLACEHOLDER),
        ("US$ 500 mil", MONEY_PLACEHOLDER),
        ("$100", MONEY_PLACEHOLDER),
        ("R$ 3 bilhões", MONEY_PLACEHOLDER),
        ("R$ 3 trilhão", MONEY_PLACEHOLDER),
    ],
)
def test_mask_money_exact(raw: str, expected: str) -> None:
    assert mask_financial_entities(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("5%", PERCENT_PLACEHOLDER),
        ("5,2%", PERCENT_PLACEHOLDER),
        ("0.5%", PERCENT_PLACEHOLDER),
        ("100 %", PERCENT_PLACEHOLDER),
    ],
)
def test_mask_percent_exact(raw: str, expected: str) -> None:
    assert mask_financial_entities(raw) == expected


def test_mask_in_sentence_money_and_percent() -> None:
    text = "A receita subiu R$ 10 bi, alta de 5,2% no ano."
    out = mask_financial_entities(text)
    assert MONEY_PLACEHOLDER in out
    assert PERCENT_PLACEHOLDER in out
    assert "R$" not in out
    assert "%" not in out


def test_mask_does_not_touch_plain_numbers_or_years() -> None:
    text = "Em 2023, 10 empresas com 50 funcionários cada."
    assert mask_financial_entities(text) == text


def test_mask_handles_multiple_money_occurrences() -> None:
    text = "Entre R$ 1 mi e R$ 10 mi, escolhemos o primeiro."
    out = mask_financial_entities(text)
    assert out.count(MONEY_PLACEHOLDER) == 2


def test_mask_is_idempotent() -> None:
    text = "Alta de 5% e lucro de R$ 2 bi."
    once = mask_financial_entities(text)
    twice = mask_financial_entities(once)
    assert once == twice


# -------- preprocess_raw --------------------------------------------------


def test_preprocess_raw_preserves_case_and_punctuation() -> None:
    out = preprocess_raw(["Petrobras Anuncia Dividendos!"])
    assert out == ["Petrobras Anuncia Dividendos!"]


def test_preprocess_raw_handles_none_and_empty() -> None:
    out = preprocess_raw([None, "", "  ", "ok"])
    assert out == ["", "", "", "ok"]


def test_preprocess_raw_strips_outer_whitespace_only() -> None:
    out = preprocess_raw(["   Bom dia   "])
    assert out == ["Bom dia"]


def test_preprocess_raw_normalizes_nfc() -> None:
    decomposed = "café"  # "café" em NFD
    composed = "café"
    assert decomposed != composed
    out = preprocess_raw([decomposed])
    assert out == [composed]


def test_preprocess_raw_mask_entities_off_by_default() -> None:
    out = preprocess_raw(["Alta de 5% no ano."])
    assert out == ["Alta de 5% no ano."]


def test_preprocess_raw_mask_entities_when_requested() -> None:
    out = preprocess_raw(["Alta de 5% no ano."], mask_entities=True)
    assert out == [f"Alta de {PERCENT_PLACEHOLDER} no ano."]


def test_preprocess_raw_accepts_iterator() -> None:
    out = preprocess_raw(iter(["a", "b"]))
    assert out == ["a", "b"]


# -------- preprocess_aggressive (SpaCy) -----------------------------------


@pytest.mark.slow
def test_preprocess_aggressive_lowercases_and_filters_stopwords(
    skip_if_no_spacy_lg: None,
) -> None:
    out = preprocess_aggressive(["A Petrobras anunciou um lucro recorde."])
    assert out[0] == out[0].lower()
    assert "a " not in f" {out[0]} "
    assert "petrobras" in out[0]
    assert "lucro" in out[0]


@pytest.mark.slow
def test_preprocess_aggressive_removes_punctuation(
    skip_if_no_spacy_lg: None,
) -> None:
    out = preprocess_aggressive(["Vendas! Lucros? Sim."])
    assert "!" not in out[0]
    assert "?" not in out[0]
    assert "." not in out[0]


@pytest.mark.slow
def test_preprocess_aggressive_lemmatizes(skip_if_no_spacy_lg: None) -> None:
    out = preprocess_aggressive(["As empresas estavam crescendo rapidamente."])
    tokens = out[0].split()
    assert "empresa" in tokens
    assert "empresas" not in tokens


@pytest.mark.slow
def test_preprocess_aggressive_masks_by_default(
    skip_if_no_spacy_lg: None,
) -> None:
    out = preprocess_aggressive(["Lucro de R$ 10 bi e alta de 5%."])
    assert "valor_monetario" in out[0]
    assert "percentual" in out[0]


@pytest.mark.slow
def test_preprocess_aggressive_mask_entities_off(
    skip_if_no_spacy_lg: None,
) -> None:
    out = preprocess_aggressive(
        ["Lucro de R$ 10 bi."], mask_entities=False
    )
    assert "valor_monetario" not in out[0]


@pytest.mark.slow
def test_preprocess_aggressive_handles_none_and_empty(
    skip_if_no_spacy_lg: None,
) -> None:
    out = preprocess_aggressive([None, "", "ok"])
    assert out[0] == ""
    assert out[1] == ""
    assert "ok" in out[2]


@pytest.mark.slow
def test_preprocess_aggressive_shrinks_vocabulary_vs_raw(
    skip_if_no_spacy_lg: None,
) -> None:
    texts = [
        "A Petrobras anunciou um lucro recorde no trimestre.",
        "Os lucros da empresa subiram fortemente.",
    ]
    raw_out = preprocess_raw(texts)
    agg_out = preprocess_aggressive(texts)
    raw_vocab = set(" ".join(raw_out).lower().split())
    agg_vocab = set(" ".join(agg_out).split())
    assert len(agg_vocab) < len(raw_vocab)


@pytest.mark.slow
def test_preprocess_aggressive_is_deterministic(
    skip_if_no_spacy_lg: None,
) -> None:
    texts = ["Petrobras fechou em alta de 5% hoje."]
    a = preprocess_aggressive(texts)
    b = preprocess_aggressive(texts)
    assert a == b


# -------- SpaCy loader -----------------------------------------------------


@pytest.mark.slow
def test_load_spacy_is_cached(skip_if_no_spacy_lg: None) -> None:
    nlp_a = preprocessing._load_spacy()
    nlp_b = preprocessing._load_spacy()
    assert nlp_a is nlp_b


def test_load_spacy_raises_useful_error_when_model_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Se o modelo não está no ambiente, a mensagem deve mencionar `spacy download`."""
    import spacy

    preprocessing._load_spacy.cache_clear()

    def _fake_load(*args: object, **kwargs: object) -> object:
        raise OSError("model not found")

    monkeypatch.setattr(spacy, "load", _fake_load)
    with pytest.raises(RuntimeError, match="spacy download"):
        preprocessing._load_spacy()
    preprocessing._load_spacy.cache_clear()
