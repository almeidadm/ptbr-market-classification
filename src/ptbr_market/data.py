"""Carregamento do corpus FolhaSP e split Out-of-Time 80/10/10.

Invariantes metodológicas (ver CLAUDE.md e docs/estrutura_repositorio.md):

- Split estritamente cronológico: ordenado por (date, link) ascendente.
- Regra de desempate: todos os artigos de um dia de fronteira ficam na
  partição mais antiga (evita vazamento intra-dia entre train/val/test).
- Determinismo absoluto: `mergesort` + sem aleatoriedade. Rodar duas vezes
  produz partições byte-idênticas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ptbr_market import runs

POSITIVE_CATEGORY = "mercado"
LABEL_COLUMN = "label"
OUTPUT_COLUMNS = ["title", "text", "date", "category", "link", LABEL_COLUMN]
TIE_RULE = "boundary_day_stays_in_older_partition"


def load_corpus(data_root: Path | None = None) -> pd.DataFrame:
    """Lê os parquets de `data/raw/folhasp*.parquet`, concatena e deriva o rótulo binário.

    Não faz filtragem (nulos, duplicatas) nem split — é responsabilidade de
    módulos posteriores. Retorna o corpus ordenado por `(date, link)`.
    """
    root = Path(data_root) if data_root is not None else runs.raw_dir()
    files = sorted(root.glob("folhasp*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"Nenhum parquet encontrado em {root}. Ver data/README.md para"
            " instruções de obtenção/regeneração do corpus."
        )

    frames = [pd.read_parquet(f, engine="pyarrow") for f in files]
    df = pd.concat(frames, ignore_index=True)

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise TypeError(
            f"Coluna 'date' deve ser datetime, recebido {df['date'].dtype}."
            " Regenere os parquets com scripts/convert_csv_to_parquet.py."
        )

    df[LABEL_COLUMN] = (df["category"] == POSITIVE_CATEGORY).astype("int8")
    df = df.sort_values(["date", "link"], kind="mergesort").reset_index(drop=True)
    return df[OUTPUT_COLUMNS]


def split_out_of_time(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split Out-of-Time 80/10/10 ordenado cronologicamente.

    Regra de desempate `boundary_day_stays_in_older_partition`: o corte
    acontece entre DIAS, nunca no meio de um mesmo dia — todos os artigos
    da data fronteira ficam na partição mais antiga.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("train_frac e val_frac devem estar em (0, 1).")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac deve ser < 1 para sobrar espaço ao teste.")

    ordered = df.sort_values(["date", "link"], kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    if n < 3:
        raise ValueError(f"Corpus muito pequeno para split ({n} linhas).")

    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))

    boundary_train_val = ordered.loc[i_train, "date"]
    boundary_val_test = ordered.loc[i_val, "date"]

    dates = ordered["date"]
    train = ordered.loc[dates <= boundary_train_val].reset_index(drop=True)
    val = ordered.loc[(dates > boundary_train_val) & (dates <= boundary_val_test)].reset_index(
        drop=True
    )
    test = ordered.loc[dates > boundary_val_test].reset_index(drop=True)

    if len(train) + len(val) + len(test) != n:
        raise AssertionError("Split perdeu ou duplicou linhas — invariante violada.")
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            "Alguma partição ficou vazia após o split. Corpus provavelmente"
            " tem poucas datas distintas para o train_frac/val_frac pedidos."
        )
    if train["date"].max() >= val["date"].min() or val["date"].max() >= test["date"].min():
        raise AssertionError("Disjunção temporal violada — bug na regra de desempate.")
    if min(train[LABEL_COLUMN].sum(), val[LABEL_COLUMN].sum(), test[LABEL_COLUMN].sum()) == 0:
        raise ValueError(
            "Alguma partição ficou sem exemplo da classe positiva —"
            " corpus desbalanceado demais para split temporal."
        )

    return train, val, test


def describe_split(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict[str, object]:
    def _summary(name: str, part: pd.DataFrame) -> dict[str, object]:
        return {
            "n": int(len(part)),
            "window": [
                part["date"].min().date().isoformat(),
                part["date"].max().date().isoformat(),
            ],
            "pos_count": int(part[LABEL_COLUMN].sum()),
            "pos_ratio": round(float(part[LABEL_COLUMN].mean()), 6),
        }

    total = len(train) + len(val) + len(test)
    return {
        "tie_rule": TIE_RULE,
        "total_n": total,
        "train": _summary("train", train),
        "val": _summary("val", val),
        "test": _summary("test", test),
    }
