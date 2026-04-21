"""Materializa o split Out-of-Time 80/10/10 em `artifacts/splits/`.

Lê `data/raw/folhasp*.parquet`, gera `train.parquet`, `val.parquet`,
`test.parquet` + `metadata.json`. Idempotente: rodar duas vezes produz
parquets byte-idênticos.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

from ptbr_market import data, runs


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def print_description(desc: dict[str, object]) -> None:
    print(f"tie_rule: {desc['tie_rule']}")
    print(f"total:    {desc['total_n']:>7d}")
    for name in ("train", "val", "test"):
        p = desc[name]
        pct = p["n"] / desc["total_n"] * 100
        print(
            f"{name:<5s} {p['n']:>7d} ({pct:5.2f}%)  "
            f"[{p['window'][0]} → {p['window'][1]}]  "
            f"pos={p['pos_count']} ({p['pos_ratio'] * 100:.2f}%)"
        )


def write_partition(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        path,
        engine="pyarrow",
        compression="zstd",
        compression_level=3,
        index=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--artifacts-root", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Imprime estatísticas sem escrever arquivos",
    )
    args = parser.parse_args(argv)

    raw_dir = (args.data_root / "raw") if args.data_root else runs.raw_dir()
    splits_dir = (args.artifacts_root / "splits") if args.artifacts_root else runs.splits_dir()

    df = data.load_corpus(raw_dir)
    train, val, test = data.split_out_of_time(df)
    desc = data.describe_split(train, val, test)

    print_description(desc)

    if args.dry_run:
        return 0

    paths = {
        "train": splits_dir / "train.parquet",
        "val": splits_dir / "val.parquet",
        "test": splits_dir / "test.parquet",
    }
    write_partition(train, paths["train"])
    write_partition(val, paths["val"])
    write_partition(test, paths["test"])

    source_files = sorted(raw_dir.glob("folhasp*.parquet"))
    metadata = {
        "generated_at": runs.utc_now_iso(),
        "git_commit": runs.git_commit(),
        "source_parquets": [
            {"name": p.name, "sha256": sha256_of_file(p)} for p in source_files
        ],
        "split": desc,
        "partitions": {
            name: {"path": str(path), "sha256": sha256_of_file(path)}
            for name, path in paths.items()
        },
    }
    runs.write_split_metadata(splits_dir / "metadata.json", metadata)

    print()
    print(f"Artefatos em {splits_dir}:")
    for path in [*paths.values(), splits_dir / "metadata.json"]:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
