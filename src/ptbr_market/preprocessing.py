"""Pré-processamento bimodal (Gen 1 agressivo / Gen 2–3 cru) e máscara de
entidades financeiras.

Convenções (ver CLAUDE.md):

- `mask_financial_entities` substitui valores monetários por
  `[VALOR_MONETARIO]` e percentuais por `[PERCENTUAL]`. Esses literais
  são fixados no CLAUDE.md e devem permanecer assim.
- `preprocess_raw` (Gen 2/3): Unicode NFC + strip, preserva caixa,
  pontuação e sintaxe. `mask_entities` default é **False**.
- `preprocess_aggressive` (Gen 1): baixa caixa, filtra stopwords e
  pontuação, lematiza via SpaCy `pt_core_news_lg`. `mask_entities`
  default é **True** (decisão D1 fixada em 2026-04-21).

`None` e string vazia viram `""` em ambos os caminhos; a decisão de
filtrar documentos vazios é do chamador (decisão D5).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy.language  # pragma: no cover

SPACY_MODEL = "pt_core_news_lg"

MONEY_PLACEHOLDER = "[VALOR_MONETARIO]"
PERCENT_PLACEHOLDER = "[PERCENTUAL]"

ALLOWED_MODES: tuple[str, ...] = ("raw", "aggressive")

_MONEY_RE = re.compile(
    r"(?:R\$|US\$|\$)\s*\d+(?:[.,]\d+)*"
    r"(?:\s+(?:milh[õo]es|milh[ãa]o|bilh[õo]es|bilh[ãa]o|trilh[õo]es|trilh[ãa]o"
    r"|mil|tri|bi|mi)\b)?",
    re.IGNORECASE,
)

_PERCENT_RE = re.compile(r"\d+(?:[.,]\d+)*\s*%")


def mask_financial_entities(text: str) -> str:
    """Substitui valores monetários e percentuais pelos placeholders do projeto.

    Exemplos:

    - `"R$ 10 bi"` → `"[VALOR_MONETARIO]"`
    - `"alta de 5,2%"` → `"alta de [PERCENTUAL]"`

    A ordem (monetário antes de percentual) é irrelevante porque os
    padrões são disjuntos em textos reais de notícias.
    """
    out = _MONEY_RE.sub(MONEY_PLACEHOLDER, text)
    out = _PERCENT_RE.sub(PERCENT_PLACEHOLDER, out)
    return out


def _coerce(text: str | None) -> str:
    return "" if text is None else str(text)


def preprocess_raw(
    texts: Iterable[str | None],
    mask_entities: bool = False,
) -> list[str]:
    """Passagem quase-crua: Unicode NFC + strip. Preserva o resto.

    Para Geração 2 (BERT) e Geração 3 (LLMs), onde números e caixa
    importam para a atenção/tokenizer.
    """
    output: list[str] = []
    for t in texts:
        s = unicodedata.normalize("NFC", _coerce(t)).strip()
        if mask_entities:
            s = mask_financial_entities(s)
        output.append(s)
    return output


@lru_cache(maxsize=1)
def _load_spacy() -> spacy.language.Language:
    """Carrega o modelo SpaCy uma única vez por processo.

    Desabilita `parser` e `ner` (não usados) para acelerar o pipeline.
    Dá mensagem útil se o modelo não estiver instalado.
    """
    try:
        import spacy
    except ImportError as e:  # pragma: no cover - spacy é dep declarada
        raise RuntimeError(
            "spacy não está instalado; rode `uv sync`."
        ) from e
    try:
        return spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    except OSError as e:
        raise RuntimeError(
            f"Modelo SpaCy {SPACY_MODEL!r} não encontrado no ambiente atual."
            " Instale com:\n"
            f"    uv run python -m spacy download {SPACY_MODEL}"
        ) from e


def preprocess_aggressive(
    texts: Iterable[str | None],
    mask_entities: bool = True,
    batch_size: int = 256,
) -> list[str]:
    """Pipeline agressivo para Geração 1: baixa caixa, lematiza, remove
    stopwords e pontuação via SpaCy `pt_core_news_lg`.

    `mask_entities=True` por default (decisão D1). Em modo agressivo, os
    colchetes do placeholder são classificados como pontuação e removidos;
    o token central (`valor_monetario`, `percentual`) é preservado como
    lema estável.
    """
    nlp = _load_spacy()

    prepared: list[str] = []
    for t in texts:
        s = unicodedata.normalize("NFC", _coerce(t))
        if mask_entities:
            s = mask_financial_entities(s)
        prepared.append(s.lower())

    output: list[str] = []
    for doc in nlp.pipe(prepared, batch_size=batch_size):
        tokens = [
            tok.lemma_.lower()
            for tok in doc
            if not tok.is_stop
            and not tok.is_punct
            and not tok.is_space
            and tok.lemma_.strip()
        ]
        output.append(" ".join(tokens))
    return output


def preprocess_split_cached(
    split_name: str,
    mode: str,
    texts: Sequence[str | None],
    *,
    force: bool = False,
) -> list[str]:
    """Pré-processa `texts` para `(split_name, mode)` com cache em parquet.

    Cache em `artifacts/preprocessed/{split_name}__{mode}.parquet` (coluna
    única `text`). Se o cache existe e `force=False`, é lido direto; caso
    contrário, `preprocess_raw` ou `preprocess_aggressive` é chamado
    (dependendo de `mode`) e o resultado é persistido.

    O comprimento do cache deve bater com `len(texts)` — um mismatch indica
    que os splits mudaram e o cache está obsoleto; nesse caso, levanta
    `ValueError` e o chamador deve passar `force=True`.
    """
    import pandas as pd  # import local — lazy para manter o módulo leve em imports

    from ptbr_market import runs

    if mode not in ALLOWED_MODES:
        raise ValueError(
            f"mode deve ser um de {ALLOWED_MODES}, recebido {mode!r}."
        )

    cache = runs.preprocessed_path(split_name, mode)
    texts_list = list(texts)

    if cache.exists() and not force:
        df = pd.read_parquet(cache)
        if len(df) != len(texts_list):
            raise ValueError(
                f"Cache {cache} tem {len(df)} linhas, mas {split_name!r} tem"
                f" {len(texts_list)}. Splits mudaram — rode com force=True."
            )
        return df["text"].astype(str).tolist()

    if mode == "aggressive":
        result = preprocess_aggressive(texts_list)
    else:  # "raw"
        result = preprocess_raw(texts_list)

    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": result}).to_parquet(cache, index=False)
    return result
