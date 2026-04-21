"""Roda classificadores Gen 1 sobre os splits OOT em `artifacts/splits/`.

Uso típico:

    uv run python scripts/run_gen1.py
    uv run python scripts/run_gen1.py --models linearsvc,logreg
    uv run python scripts/run_gen1.py --representation bow --variant bow-lemmatized
    uv run python scripts/run_gen1.py --no-cache                 # força re-preprocess
    uv run python scripts/run_gen1.py --preprocess-mode raw      # útil só para depurar

Cada modelo produz um run em `artifacts/runs/{stamp}__gen1__{model}__{variant}/`.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from ptbr_market import gen1_classical, gen1_pipeline, preprocessing, runs

DEFAULT_VARIANT = "tfidf-lemmatized"
DEFAULT_MODE = "aggressive"
DEFAULT_REPRESENTATION = "tfidf"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        default=",".join(gen1_classical.GEN1_CLASSIFIERS),
        help=(
            "Lista de modelos separada por vírgulas."
            f" Default: {','.join(gen1_classical.GEN1_CLASSIFIERS)}."
        ),
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        help=f"Rótulo da variante no nome do run (default: {DEFAULT_VARIANT}).",
    )
    parser.add_argument(
        "--representation",
        default=DEFAULT_REPRESENTATION,
        choices=gen1_pipeline.ALLOWED_REPRESENTATIONS,
    )
    parser.add_argument(
        "--preprocess-mode",
        default=DEFAULT_MODE,
        choices=preprocessing.ALLOWED_MODES,
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Regenera o cache de pré-processamento em disco.",
    )
    return parser.parse_args(argv)


def load_splits() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(runs.splits_dir() / f"{name}.parquet")
        for name in ("train", "val", "test")
    }


def build_split_meta_block() -> dict:
    meta = json.loads((runs.splits_dir() / "metadata.json").read_text())
    return {
        "train_window": meta["split"]["train"]["window"],
        "val_window": meta["split"]["val"]["window"],
        "test_window": meta["split"]["test"]["window"],
        "n_train": meta["split"]["train"]["n"],
        "n_val": meta["split"]["val"]["n"],
        "n_test": meta["split"]["test"]["n"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print("Carregando splits…", file=sys.stderr)
    splits = load_splits()
    split_meta_block = build_split_meta_block()
    for name, df in splits.items():
        print(f"  {name}: {len(df)} linhas", file=sys.stderr)

    print(f"Executando {len(models)} modelo(s): {models}", file=sys.stderr)
    run_dirs = gen1_pipeline.run_gen1_experiment(
        splits=splits,
        split_meta_block=split_meta_block,
        models=models,
        variant=args.variant,
        preprocess_mode=args.preprocess_mode,
        representation=args.representation,
        use_cache=not args.no_cache,
    )

    for run_dir in run_dirs:
        metrics = json.loads((run_dir / "metrics.json").read_text())
        print(
            f"✓ {run_dir.name}"
            f"  PR-AUC={metrics['pr_auc']:.4f}"
            f"  F1-min={metrics['f1_minority']:.4f}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
