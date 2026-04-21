"""Representações textuais para Geração 1: BoW, TF-IDF e fastText-avg.

Convenções:

- Todos os _builders_ recebem texto **já pré-processado** (o chamador
  escolhe `preprocess_raw` ou `preprocess_aggressive`). Esta camada não
  normaliza nada.
- `fit` acontece no texto de treino; `transform` é aplicado a val/test.
  Isso deve ser respeitado pelo chamador — nunca refit em val/test
  (CLAUDE.md, constraint de OOT).
- Vetores fastText (`cc.pt.300.vec`) ficam em `data/raw/` (fora do LFS
  por tamanho). `load_fasttext_vectors` aceita `vocabulary` para
  carregar só os tokens relevantes e economizar RAM em Colab.
- `fasttext_average`: média dos vetores dos tokens presentes; OOV são
  ignorados; documento sem token ou com todos OOV vira vetor zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ptbr_market import runs

FASTTEXT_DIM = 300
FASTTEXT_FILENAME = "cc.pt.300.vec"

TFIDF_DEFAULTS: dict[str, Any] = {
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 0.95,
    "sublinear_tf": True,
    "dtype": np.float32,
}

BOW_DEFAULTS: dict[str, Any] = {
    "ngram_range": (1, 1),
    "min_df": 5,
    "max_df": 0.95,
}


def fasttext_path() -> Path:
    """Local canônico de `cc.pt.300.vec` (relativo a `PTBR_DATA_ROOT`)."""
    return runs.raw_dir() / FASTTEXT_FILENAME


def build_tfidf(
    train_texts: Sequence[str], **overrides: Any
) -> tuple[TfidfVectorizer, csr_matrix]:
    """Ajusta `TfidfVectorizer` em `train_texts` e retorna (vectorizer, X_train).

    Defaults vanilla em `TFIDF_DEFAULTS` — sobrescreva via kwargs conforme
    a variante do experimento.
    """
    params = {**TFIDF_DEFAULTS, **overrides}
    vec = TfidfVectorizer(**params)
    X = vec.fit_transform(train_texts)
    return vec, X


def build_bow(
    train_texts: Sequence[str], **overrides: Any
) -> tuple[CountVectorizer, csr_matrix]:
    """Ajusta `CountVectorizer` (BoW) em `train_texts`."""
    params = {**BOW_DEFAULTS, **overrides}
    vec = CountVectorizer(**params)
    X = vec.fit_transform(train_texts)
    return vec, X


def load_fasttext_vectors(
    path: Path | None = None,
    vocabulary: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Lê vetores fastText no formato `.vec` (`cc.pt.300.vec`).

    Cabeçalho: `<n_tokens> <dim>`. Linhas: `token v1 v2 ... v300`.

    Se `vocabulary` for dado, apenas esses tokens são mantidos em memória
    — essencial para Colab, já que o arquivo completo tem ~2M tokens.
    """
    p = Path(path) if path is not None else fasttext_path()
    if not p.exists():
        raise FileNotFoundError(
            f"fastText não encontrado em {p}. Baixe com:\n"
            "    curl -L https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz"
            " -o data/raw/cc.pt.300.vec.gz\n"
            "    gunzip data/raw/cc.pt.300.vec.gz"
        )
    wanted = set(vocabulary) if vocabulary is not None else None
    vectors: dict[str, np.ndarray] = {}
    with p.open(encoding="utf-8", errors="ignore") as f:
        header = f.readline().split()
        if len(header) == 2:
            dim = int(header[1])
            if dim != FASTTEXT_DIM:
                raise ValueError(
                    f"fastText esperado de dimensão {FASTTEXT_DIM}, encontrei {dim}."
                )
        else:
            f.seek(0)
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) < FASTTEXT_DIM + 1:
                continue
            token = parts[0]
            if wanted is not None and token not in wanted:
                continue
            vectors[token] = np.asarray(parts[1 : FASTTEXT_DIM + 1], dtype=np.float32)
            if wanted is not None and len(vectors) == len(wanted):
                break
    return vectors


def fasttext_average(
    texts: Sequence[str], vectors: dict[str, np.ndarray]
) -> np.ndarray:
    """Média dos vetores fastText por documento.

    Convenção: tokens OOV são ignorados. Documento vazio ou com todos
    tokens OOV vira vetor zero de dimensão `FASTTEXT_DIM`.
    Assume texto já tokenizado por whitespace (`text.split()`).
    """
    texts_list = list(texts)
    out = np.zeros((len(texts_list), FASTTEXT_DIM), dtype=np.float32)
    for i, text in enumerate(texts_list):
        if not text:
            continue
        found = [vectors[t] for t in text.split() if t in vectors]
        if found:
            out[i] = np.mean(found, axis=0, dtype=np.float32)
    return out
