"""Executa um LLM Gen 3 (zero-shot via Ollama) sobre os splits OoT.

Uso típico (Colab com GPU T4/L4 + Ollama rodando local):

    uv run python scripts/run_gen3.py --model llama3.1-8b
    uv run python scripts/run_gen3.py --model qwen2.5-7b --target-mode multiclass \\
        --collapse-scheme top7_plus_other

    # Retomar um run parcial (ex.: Colab caiu no meio):
    uv run python scripts/run_gen3.py --model llama3.1-8b \\
        --resume-run-id 20260422-120000__gen3__llama3.1-8b__zs-v1-bin

    # Smoke test rápido (100 amostras do test):
    uv run python scripts/run_gen3.py --model tucano-2b4 --limit 100

O run é persistido em `artifacts/runs/{stamp}__gen3__{slug}__{variant}/`
com o mesmo contrato de Gen 1/Gen 2 (metadata.json + metrics.json +
predictions.csv). `--variant` é derivado como `zs-v1-{bin|mc8}` se omitido.

Requisitos: `uv sync --group gen3`; um daemon Ollama rodando
(`ollama serve`) e o modelo puxado (`ollama pull <tag>`). Para Bode e
Tucano, use `bootstrap_gen3.ipynb` — eles precisam ser importados via
`.gguf` + Modelfile (`ollama create`).
"""

from __future__ import annotations

import argparse
import json
import sys

from ptbr_market import gen3_llm, runs, targets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=tuple(gen3_llm.GEN3_MODELS),
        help="Slug do modelo Gen 3 (ver gen3_llm.GEN3_MODELS).",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Rótulo da variante. Se omitido, deriva `zs-v1-{bin|mc8}`.",
    )
    parser.add_argument(
        "--target-mode",
        default="binary",
        choices=targets.ALLOWED_TARGET_MODES,
        help="`binary` (default) ou `multiclass` (com --collapse-scheme).",
    )
    parser.add_argument(
        "--collapse-scheme",
        default=None,
        choices=tuple(targets.COLLAPSE_SCHEMES),
        help="Esquema de colapso para target-mode=multiclass.",
    )
    parser.add_argument(
        "--prompt-version",
        default=gen3_llm.PROMPT_VERSION,
        help=f"Versão do prompt. Default: {gen3_llm.PROMPT_VERSION}.",
    )
    parser.add_argument(
        "--text-max-chars",
        type=int,
        default=gen3_llm.DEFAULT_TEXT_MAX_CHARS,
        help=(
            f"Limite de chars do corpo da notícia (head). Default:"
            f" {gen3_llm.DEFAULT_TEXT_MAX_CHARS}."
        ),
    )
    parser.add_argument(
        "--ollama-host",
        default=gen3_llm.OLLAMA_DEFAULT_HOST,
        help=f"Host do daemon Ollama. Default: {gen3_llm.OLLAMA_DEFAULT_HOST}.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Sobrescreve num_ctx do spec (default por modelo).",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=gen3_llm.DEFAULT_NUM_PREDICT,
        help=f"Tokens gerados por resposta. Default: {gen3_llm.DEFAULT_NUM_PREDICT}.",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=gen3_llm.DEFAULT_TOP_LOGPROBS,
        help=f"top-k dos logprobs. Default: {gen3_llm.DEFAULT_TOP_LOGPROBS}.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=gen3_llm.DEFAULT_TIMEOUT_S,
        help=f"Timeout por requisição em segundos. Default: {gen3_llm.DEFAULT_TIMEOUT_S}.",
    )
    parser.add_argument(
        "--resume-run-id",
        default=None,
        help=(
            "Nome do diretório do run (sem o caminho) a retomar. O"
            " predictions.csv existente é respeitado e só índices faltantes"
            " são chamados no LLM."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita nº de amostras (smoke-test; produz run incompleto).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Frequência (em amostras) de logs de progresso. 0 desativa.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.target_mode == "multiclass" and args.collapse_scheme is None:
        print("erro: --target-mode=multiclass requer --collapse-scheme", file=sys.stderr)
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

    resume_dir = None
    if args.resume_run_id:
        resume_dir = runs.runs_dir() / args.resume_run_id
        if not resume_dir.is_dir():
            print(f"erro: resume-run-id não existe: {resume_dir}", file=sys.stderr)
            return 2

    print(
        f"Zero-shot {args.model} (target={args.target_mode}"
        + (f", scheme={args.collapse_scheme}" if args.collapse_scheme else "")
        + f", host={args.ollama_host})",
        file=sys.stderr,
    )

    run_dir = gen3_llm.run_gen3_experiment(
        splits=splits,
        split_meta_block=split_meta_block,
        model_slug=args.model,
        variant=args.variant,
        target_mode=args.target_mode,
        collapse_scheme=args.collapse_scheme,
        prompt_version=args.prompt_version,
        text_max_chars=args.text_max_chars,
        ollama_host=args.ollama_host,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        top_logprobs=args.top_logprobs,
        timeout_s=args.timeout_s,
        resume_run_dir=resume_dir,
        limit=args.limit,
        progress_every=args.progress_every,
    )

    metrics = json.loads((run_dir / "metrics.json").read_text())
    meta = json.loads((run_dir / "metadata.json").read_text())
    print(
        f"✓ {run_dir.name}"
        f"  PR-AUC={metrics['pr_auc']:.4f}"
        f"  F1-min={metrics['f1_minority']:.4f}"
        f"  lat={meta['efficiency']['latency_ms_per_1k']:.1f} ms/1k"
        f"  vram={meta['efficiency']['vram_peak_mb']:.0f} MB"
        f"  n={meta['efficiency']['n_processed']}/{meta['efficiency']['n_total']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
