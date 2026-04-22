"""Roda classificadores Gen 1 sobre os splits OOT em `artifacts/splits/`.

Uso típico:

    uv run python scripts/run_gen1.py
    uv run python scripts/run_gen1.py --models linearsvc,logreg
    uv run python scripts/run_gen1.py --representation bow
    uv run python scripts/run_gen1.py --no-cache                 # força re-preprocess
    uv run python scripts/run_gen1.py --preprocess-mode raw      # útil só para depurar

    # Modo multiclasse (decomposição da classe negativa, avaliação binária):
    uv run python scripts/run_gen1.py --target-mode multiclass \\
        --collapse-scheme top7_plus_other

Cada modelo produz um run em `artifacts/runs/{stamp}__gen1__{model}__{variant}/`.
O `--variant` é derivado de `(representation, preprocess-mode, target)` se
não for passado explicitamente (ex.: `tfidf-aggr-bin`, `bow-aggr-mc8`).
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from ptbr_market import gen1_classical, gen1_pipeline, preprocessing, runs

DEFAULT_VARIANT: str | None = None  # None = auto-deriva
DEFAULT_MODE = "aggressive"
DEFAULT_REPRESENTATION = "tfidf"
DEFAULT_TARGET_MODE = "binary"

_PREPROC_SHORT = {"aggressive": "aggr", "raw": "raw"}


def _derive_variant(
    representation: str,
    preprocess_mode: str,
    target_mode: str,
    collapse_scheme: str | None,
) -> str:
    preproc_short = _PREPROC_SHORT.get(preprocess_mode, preprocess_mode)
    if target_mode == "binary":
        target_short = "bin"
    else:
        spec = gen1_pipeline.COLLAPSE_SCHEMES[collapse_scheme]
        k = len(spec["keep"]) + 1
        target_short = f"mc{k}"
    return f"{representation}-{preproc_short}-{target_short}"


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
        help=(
            "Rótulo da variante no nome do run. Se omitido, deriva de"
            " (representation, preprocess-mode, target-mode) — ex.:"
            " `tfidf-aggr-bin`, `bow-aggr-mc8`."
        ),
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
        "--target-mode",
        default=DEFAULT_TARGET_MODE,
        choices=gen1_pipeline.ALLOWED_TARGET_MODES,
        help=(
            "`binary` (default): treina com label ∈ {0,1}."
            " `multiclass`: decompõe a classe negativa via --collapse-scheme,"
            " avaliação permanece binária."
        ),
    )
    parser.add_argument(
        "--collapse-scheme",
        default=None,
        choices=tuple(gen1_pipeline.COLLAPSE_SCHEMES),
        help="Esquema de colapso para target-mode=multiclass.",
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

    if args.target_mode == "multiclass" and args.collapse_scheme is None:
        print(
            "erro: --target-mode=multiclass requer --collapse-scheme",
            file=sys.stderr,
        )
        return 2
    if args.target_mode == "binary" and args.collapse_scheme is not None:
        print(
            "erro: --collapse-scheme só é válido com --target-mode=multiclass",
            file=sys.stderr,
        )
        return 2

    variant = args.variant or _derive_variant(
        args.representation,
        args.preprocess_mode,
        args.target_mode,
        args.collapse_scheme,
    )

    print("Carregando splits…", file=sys.stderr)
    splits = load_splits()
    split_meta_block = build_split_meta_block()
    for name, df in splits.items():
        print(f"  {name}: {len(df)} linhas", file=sys.stderr)

    print(
        f"Executando {len(models)} modelo(s): {models}"
        f"  [target={args.target_mode}"
        + (f", scheme={args.collapse_scheme}" if args.collapse_scheme else "")
        + f", variant={variant}]",
        file=sys.stderr,
    )
    run_dirs = gen1_pipeline.run_gen1_experiment(
        splits=splits,
        split_meta_block=split_meta_block,
        models=models,
        variant=variant,
        preprocess_mode=args.preprocess_mode,
        representation=args.representation,
        use_cache=not args.no_cache,
        target_mode=args.target_mode,
        collapse_scheme=args.collapse_scheme,
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
