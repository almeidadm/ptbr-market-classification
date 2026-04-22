"""Alvo de treino: modos binário e multiclasse + esquemas de colapso.

Todas as gerações do projeto avaliam binário (`mercado` vs. `outros`),
mas o **treino** pode opcionalmente decompor a classe negativa via um
`collapse_scheme` (ver `plano_base.md`, seção "Decomposição da Classe
Negativa"). A pergunta científica é se essa decomposição ajuda a
fronteira binária — foi respondida em Gen 1 (12 de 13 pares mc8 vencem
binary), e é reaplicada em Gen 2/Gen 3 para fechar o argumento.

Contrato:

- `POSITIVE_CATEGORY_LABEL = "mercado"` é invariante do projeto — é a
  classe positiva do `label` binário original.
- Em modo `multiclass`, os rótulos de **train** são strings da coluna
  `category` colapsadas pelo `scheme`. Val/test continuam usando o
  rótulo binário (`label` ∈ {0,1}) para threshold e avaliação.
- `encode_multiclass` envolve `LabelEncoder` para produzir códigos
  inteiros determinísticos (ordem alfabética das classes do scheme);
  o código da classe positiva entra no `metadata.json` para que a
  extração de `predict_proba[:, mercado]` seja auditável.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

POSITIVE_CATEGORY_LABEL: str = "mercado"
ALLOWED_TARGET_MODES: tuple[str, ...] = ("binary", "multiclass")

# `keep` lista as categorias mantidas como classes próprias no treino
# multiclasse; tudo fora cai em `sink`. `mercado` precisa estar em
# `keep` — a avaliação binária extrai a coluna da classe positiva.
COLLAPSE_SCHEMES: dict[str, dict[str, Any]] = {
    # Top-7 categorias do corpus FolhaSP (≈ 80.7% do train) + "outros".
    "top7_plus_other": {
        "keep": (
            "poder",
            "colunas",
            "mercado",
            "esporte",
            "mundo",
            "cotidiano",
            "ilustrada",
        ),
        "sink": "outros",
    },
}


def collapse_categories(categories: pd.Series, scheme: str) -> np.ndarray:
    """Aplica `scheme` sobre a série de categorias → array de strings."""
    if scheme not in COLLAPSE_SCHEMES:
        raise ValueError(
            f"collapse_scheme desconhecido: {scheme!r}. Use um de"
            f" {tuple(COLLAPSE_SCHEMES)}."
        )
    spec = COLLAPSE_SCHEMES[scheme]
    keep = set(spec["keep"])
    sink = spec["sink"]
    return np.where(categories.isin(keep), categories.astype(object), sink).astype(
        object
    )


def derive_multiclass_labels(
    splits: dict[str, pd.DataFrame],
    scheme: str,
) -> tuple[np.ndarray, int]:
    """Deriva rótulos multiclasse (strings) para train e conta as classes.

    Val/test continuam usando o rótulo binário original — só train recebe
    a decomposição.
    """
    if scheme not in COLLAPSE_SCHEMES:
        raise ValueError(
            f"collapse_scheme desconhecido: {scheme!r}. Use um de"
            f" {tuple(COLLAPSE_SCHEMES)}."
        )
    spec = COLLAPSE_SCHEMES[scheme]
    if POSITIVE_CATEGORY_LABEL not in spec["keep"]:
        raise ValueError(
            f"Esquema {scheme!r} não preserva a classe positiva"
            f" {POSITIVE_CATEGORY_LABEL!r} em keep."
        )
    if "category" not in splits["train"].columns:
        raise ValueError(
            "splits['train'] precisa da coluna 'category' para"
            " target_mode='multiclass'."
        )
    y_train = collapse_categories(splits["train"]["category"], scheme)
    num_classes = len(spec["keep"]) + 1  # +1 para o sink
    return y_train, num_classes


def encode_multiclass(
    y_strings: np.ndarray,
) -> tuple[np.ndarray, int, dict[str, int]]:
    """Codifica rótulos string em inteiros via `LabelEncoder`.

    Retorna `(y_encoded, positive_class_code, class_encoding)` — onde
    `positive_class_code` é o índice da classe `mercado` no softmax/proba
    do classificador, e `class_encoding` é o mapeamento serializável para
    o `metadata.json`.
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_strings)
    classes = list(encoder.classes_)
    if POSITIVE_CATEGORY_LABEL not in classes:
        raise ValueError(
            f"Classe positiva {POSITIVE_CATEGORY_LABEL!r} ausente após"
            f" encoding; classes encontradas: {classes!r}."
        )
    positive_class_code = int(
        encoder.transform([POSITIVE_CATEGORY_LABEL])[0]
    )
    class_encoding = {
        str(cls): int(code)
        for cls, code in zip(
            encoder.classes_,
            encoder.transform(encoder.classes_),
            strict=True,
        )
    }
    return y_encoded, positive_class_code, class_encoding


def target_variant_tag(
    target_mode: str,
    collapse_scheme: str | None,
) -> str:
    """Sufixo canônico para o `variant` dos runs: `bin` ou `mc{K}`."""
    if target_mode == "binary":
        return "bin"
    if target_mode == "multiclass":
        if collapse_scheme is None:
            raise ValueError(
                "target_mode='multiclass' requer collapse_scheme."
            )
        spec = COLLAPSE_SCHEMES[collapse_scheme]
        k = len(spec["keep"]) + 1
        return f"mc{k}"
    raise ValueError(
        f"target_mode deve ser um de {ALLOWED_TARGET_MODES}, recebido"
        f" {target_mode!r}."
    )
