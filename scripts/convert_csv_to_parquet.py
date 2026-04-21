"""Converte o CSV bruto do Kaggle (FolhaSP/UOL) em parquet versionado.

One-shot. Rodado uma única vez para gerar `data/raw/folhasp.parquet` a
partir do CSV original disponível em:

    marlesson/news-of-the-site-folhauol (Kaggle)

O CSV não é versionado — o parquet compacto, sim (via git-lfs).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT = Path("data/raw/folhasp.parquet")
SUBCATEGORY_DROP = "subcategory"  # 94% nulo, irrelevante para classificação binária.


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(csv_path: Path, parquet_path: Path) -> dict[str, object]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV fonte não encontrado: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    expected = {"title", "text", "date", "category", "link"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if SUBCATEGORY_DROP in df.columns:
        df = df.drop(columns=[SUBCATEGORY_DROP])

    df = df.sort_values(["date", "link"], kind="mergesort").reset_index(drop=True)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        parquet_path,
        engine="pyarrow",
        compression="zstd",
        compression_level=3,
        index=False,
    )

    return {
        "csv_sha256": sha256_of_file(csv_path),
        "parquet_sha256": sha256_of_file(parquet_path),
        "n_rows": int(len(df)),
        "date_min": df["date"].min().date().isoformat(),
        "date_max": df["date"].max().date().isoformat(),
        "n_mercado": int((df["category"] == "mercado").sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Caminho para articles.csv do Kaggle")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Arquivo parquet de saída (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)

    info = convert(args.csv, args.output)
    print(f"Gerado: {args.output}")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
