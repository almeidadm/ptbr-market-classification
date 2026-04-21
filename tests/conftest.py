"""Fixtures compartilhadas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ptbr_market import runs


@pytest.fixture
def tiny_corpus() -> pd.DataFrame:
    """Corpus sintético de 30 linhas calibrado para testar a regra de desempate.

    Distribuição:
      dia 1–7: 3 linhas cada (21 linhas)
      dia 8:   4 linhas (dia de fronteira train/val em i_train=24)
      dia 9:   3 linhas (dia de fronteira val/test em i_val=27)
      dia 10:  2 linhas

    Com train_frac=0.8 e val_frac=0.1 o split esperado é:
      train = 25 linhas (todas as do dia 8 ficam no train)
      val   = 3 linhas (dia 9 inteiro)
      test  = 2 linhas (dia 10)
    """
    rows: list[dict[str, object]] = []
    daily_counts = [3, 3, 3, 3, 3, 3, 3, 4, 3, 2]
    positives = {(1, 0), (5, 0), (8, 2), (9, 1), (10, 0)}
    for d, count in enumerate(daily_counts, start=1):
        for j in range(count):
            category = "mercado" if (d, j) in positives else "esporte"
            rows.append(
                {
                    "title": f"titulo d{d}_{j}",
                    "text": f"texto d{d}_{j}",
                    "date": pd.Timestamp(f"2020-01-{d:02d}"),
                    "category": category,
                    "link": f"https://example.com/d{d}_{j}",
                    "label": 1 if category == "mercado" else 0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def real_corpus_available() -> bool:
    return bool(list(runs.raw_dir().glob("folhasp*.parquet")))


@pytest.fixture
def skip_if_no_real_corpus(real_corpus_available: bool) -> None:
    if not real_corpus_available:
        pytest.skip("Corpus real (data/raw/folhasp*.parquet) não disponível.")


@pytest.fixture
def real_data_root() -> Path:
    return runs.data_root()
