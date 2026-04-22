"""Fine-tune de um encoder BERT (Gen 2) sobre os splits OoT em `artifacts/splits/`.

Uso típico (Colab com GPU):

    uv run python scripts/run_gen2.py --model bertimbau-base
    uv run python scripts/run_gen2.py --model finbert-ptbr --epochs 2
    uv run python scripts/run_gen2.py --model bertimbau-large --batch-size 8 --grad-accum 4

    # Modo multiclasse (decomposição top7+outros, avaliação binária):
    uv run python scripts/run_gen2.py --model bertimbau-base \\
        --target-mode multiclass --collapse-scheme top7_plus_other

O run é persistido em `artifacts/runs/{stamp}__gen2__{slug}__{variant}/`
com o mesmo contrato do Gen 1 (metadata.json + metrics.json + predictions.csv).
O `--variant` é derivado como `raw-ml{max_length}-{bin|mc8}` se omitido
(ex.: `raw-ml256-bin`, `raw-ml256-mc8`).

Requisitos: `uv sync --group gen2` (torch, transformers, accelerate,
datasets). GPU fortemente recomendado — em CPU, o treino leva horas.
"""

from __future__ import annotations

import argparse
import json
import sys

from ptbr_market import gen2_bert, runs, targets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=tuple(gen2_bert.GEN2_MODELS),
        help="Slug do modelo Gen 2 (ver gen2_bert.GEN2_MODELS).",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help=(
            "Rótulo da variante. Se omitido, deriva"
            " `raw-ml{max_length}-{bin|mc8}`."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=gen2_bert.DEFAULT_MAX_LENGTH,
        help=f"Comprimento máximo em tokens. Default: {gen2_bert.DEFAULT_MAX_LENGTH}.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=gen2_bert.DEFAULT_EPOCHS,
        help=f"Número de épocas. Default: {gen2_bert.DEFAULT_EPOCHS}.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=gen2_bert.DEFAULT_LEARNING_RATE,
        help=f"Learning rate. Default: {gen2_bert.DEFAULT_LEARNING_RATE}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Sobrescreve batch_size do registry (útil em T4 apertado).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Sobrescreve gradient_accumulation_steps do registry.",
    )
    parser.add_argument(
        "--target-mode",
        default="binary",
        choices=targets.ALLOWED_TARGET_MODES,
        help=(
            "`binary` (default): treina com label ∈ {0,1}."
            " `multiclass`: decompõe a classe negativa via --collapse-scheme,"
            " avaliação permanece binária."
        ),
    )
    parser.add_argument(
        "--collapse-scheme",
        default=None,
        choices=tuple(targets.COLLAPSE_SCHEMES),
        help="Esquema de colapso para target-mode=multiclass.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

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

    print("Carregando splits…", file=sys.stderr)
    splits = runs.load_splits()
    split_meta_block = runs.build_split_meta_block()
    for name, df in splits.items():
        print(f"  {name}: {len(df)} linhas", file=sys.stderr)

    print(
        f"Fine-tuning {args.model} (max_length={args.max_length},"
        f" epochs={args.epochs}, lr={args.learning_rate},"
        f" target={args.target_mode}"
        + (f", scheme={args.collapse_scheme}" if args.collapse_scheme else "")
        + ")",
        file=sys.stderr,
    )

    run_dir = gen2_bert.run_gen2_experiment(
        splits=splits,
        split_meta_block=split_meta_block,
        model_slug=args.model,
        variant=args.variant,
        max_length=args.max_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        target_mode=args.target_mode,
        collapse_scheme=args.collapse_scheme,
    )

    metrics = json.loads((run_dir / "metrics.json").read_text())
    meta = json.loads((run_dir / "metadata.json").read_text())
    print(
        f"✓ {run_dir.name}"
        f"  PR-AUC={metrics['pr_auc']:.4f}"
        f"  F1-min={metrics['f1_minority']:.4f}"
        f"  lat={meta['efficiency']['latency_ms_per_1k']:.1f} ms/1k"
        f"  vram={meta['efficiency']['vram_peak_mb']:.0f} MB",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
