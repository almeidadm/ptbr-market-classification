"""Agrega os runs Gen 3 em `artifacts/runs/` e produz relatórios pareados.

Lê `metadata.json` + `predictions.csv` de cada `*__gen3__*` e grava:

- `artifacts/reports/gen3_summary.csv`: uma linha por run Gen 3 com
  métricas, latência, VRAM e config (tag do Ollama, hash do prompt,
  contagem de fallback para hard label).
- `artifacts/reports/gen3_paired.csv`: binary vs mc8 lado a lado para
  cada LLM Gen 3 (quando ambas as variantes existirem).
- `artifacts/reports/gen3_intra_mcnemar.csv`: McNemar pareado bin vs
  mc8 dentro de Gen 3 — responde à pergunta "o prompt multiclasse ajuda
  o LLM a separar melhor a fronteira binária?".
- `artifacts/reports/gen3_vs_gen1_champion.csv`: McNemar pareado de
  cada LLM Gen 3 contra o campeão Gen 1 (default: maior F1-min em
  variants `*-aggr-*`).
- `artifacts/reports/gen3_vs_gen2_champion.csv` (se houver Gen 2):
  McNemar contra o campeão Gen 2 — a comparação-chave do artigo.

Runs Gen 3 incompletos (parse_failure em 100% dos casos, ou `n_processed
< n_total`) entram no sumário mas podem aparecer como ruído no McNemar;
filtre manualmente se necessário.

Uso:

    uv run python scripts/gen3_report.py
    uv run python scripts/gen3_report.py --champion-variant tfidf-aggr-mc8 \\
                                         --champion-model linearsvc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ptbr_market.evaluation import mcnemar_test

RUNS_DIR = Path("artifacts/runs")
REPORT_DIR = Path("artifacts/reports")


def _load_gen3_runs(runs_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for d in sorted(runs_dir.iterdir()):
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("generation") != "gen3":
            continue
        target = meta.get("config", {}).get("target", {})
        mode = target.get("mode", "binary")
        if mode == "multiclass":
            mode_tag = f"mc{target.get('num_classes')}"
        else:
            mode_tag = "bin"
        scoring = meta["config"].get("scoring", {})
        method_counts = scoring.get("method_counts", {})
        prompt = meta["config"].get("prompt", {})
        eff = meta["efficiency"]
        records.append(
            {
                "run_id": meta["run_id"],
                "model": meta["model"],
                "variant": meta["variant"],
                "target_mode": mode,
                "target_tag": mode_tag,
                "collapse_scheme": target.get("collapse_scheme"),
                "num_classes": target.get("num_classes", 2),
                "ollama_model_tag": meta["config"].get("ollama_model_tag"),
                "bucket": meta["config"].get("bucket"),
                "num_ctx": meta["config"].get("num_ctx"),
                "prompt_version": prompt.get("version"),
                "prompt_hash": prompt.get("hash"),
                "threshold": meta["threshold"]["value"],
                "threshold_applicable": meta["threshold"].get("applicable", False),
                "pr_auc": meta["metrics"]["pr_auc"],
                "roc_auc": meta["metrics"]["roc_auc"],
                "f1_macro": meta["metrics"]["f1_macro"],
                "f1_minority": meta["metrics"]["f1_minority"],
                "precision_minority": meta["metrics"]["precision_minority"],
                "recall_minority": meta["metrics"]["recall_minority"],
                "latency_ms_per_1k": eff["latency_ms_per_1k"],
                "vram_peak_mb": eff.get("vram_peak_mb", 0),
                "n_processed": eff.get("n_processed", 0),
                "n_total": eff.get("n_total", 0),
                "n_logprobs": method_counts.get("logprobs", 0),
                "n_hard_fallback": method_counts.get("hard", 0),
                "n_parse_failure": method_counts.get("parse_failure", 0),
                "run_dir": str(d),
            }
        )
    return pd.DataFrame.from_records(records)


def _pair_bin_mc8(df: pd.DataFrame) -> pd.DataFrame:
    binary = df[df["target_tag"] == "bin"].set_index("model")
    mc8 = df[df["target_tag"] == "mc8"].set_index("model")
    metric_cols = [
        "f1_minority",
        "pr_auc",
        "precision_minority",
        "recall_minority",
        "latency_ms_per_1k",
        "vram_peak_mb",
        "n_parse_failure",
        "n_hard_fallback",
        "run_dir",
    ]
    joined = binary[metric_cols].join(
        mc8[metric_cols], lsuffix="_bin", rsuffix="_mc8", how="inner"
    )
    joined["delta_f1_minority"] = (
        joined["f1_minority_mc8"] - joined["f1_minority_bin"]
    )
    joined["delta_pr_auc"] = joined["pr_auc_mc8"] - joined["pr_auc_bin"]
    return joined.reset_index().sort_values("model")


def _intra_mcnemar_rows(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in paired.iterrows():
        pred_bin = pd.read_csv(Path(row["run_dir_bin"]) / "predictions.csv")
        pred_mc8 = pd.read_csv(Path(row["run_dir_mc8"]) / "predictions.csv")
        merged = pred_bin[["index", "y_true", "y_pred"]].merge(
            pred_mc8[["index", "y_true", "y_pred"]],
            on="index",
            suffixes=("_bin", "_mc8"),
        )
        if not (merged["y_true_bin"] == merged["y_true_mc8"]).all():
            raise RuntimeError(
                f"y_true diverge entre bin e mc8 para {row['model']}"
            )
        result = mcnemar_test(
            merged["y_true_bin"].to_numpy(),
            merged["y_pred_bin"].to_numpy(),
            merged["y_pred_mc8"].to_numpy(),
        )
        rows.append(
            {
                "model": row["model"],
                "n_test": len(merged),
                "b_bin_only_correct": result["b"],
                "c_mc8_only_correct": result["c"],
                "mcnemar_statistic": result["statistic"],
                "p_value": result["p_value"],
                "winner": (
                    "mc8" if result["c"] > result["b"] else
                    "binary" if result["b"] > result["c"] else "tie"
                ),
                "significant_p_lt_0_05": result["p_value"] < 0.05,
            }
        )
    return pd.DataFrame.from_records(rows)


def _find_gen1_champion(runs_dir: Path) -> Path | None:
    best_f1 = -1.0
    best_dir: Path | None = None
    for d in runs_dir.iterdir():
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("generation") != "gen1":
            continue
        if "-aggr-" not in meta.get("variant", ""):
            continue
        f1 = meta["metrics"]["f1_minority"]
        if f1 > best_f1:
            best_f1 = f1
            best_dir = d
    return best_dir


def _find_gen2_champion(runs_dir: Path) -> Path | None:
    best_f1 = -1.0
    best_dir: Path | None = None
    for d in runs_dir.iterdir():
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("generation") != "gen2":
            continue
        f1 = meta["metrics"]["f1_minority"]
        if f1 > best_f1:
            best_f1 = f1
            best_dir = d
    return best_dir


def _find_run(runs_dir: Path, generation: str, variant: str, model: str) -> Path:
    matches = [
        d for d in runs_dir.iterdir()
        if d.is_dir() and f"__{generation}__{model}__{variant}" in d.name
    ]
    if not matches:
        raise RuntimeError(
            f"Run {generation} não encontrado: variant={variant!r}, model={model!r}."
        )
    matches.sort()
    return matches[-1]


def _mcnemar_vs(
    gen3_df: pd.DataFrame,
    other_dir: Path,
    other_tag: str,
) -> pd.DataFrame:
    other_meta = json.loads((other_dir / "metadata.json").read_text())
    other_pred = pd.read_csv(other_dir / "predictions.csv")
    rows: list[dict] = []
    for _, r in gen3_df.iterrows():
        gen3_pred = pd.read_csv(Path(r["run_dir"]) / "predictions.csv")
        merged = other_pred[["index", "y_true", "y_pred"]].merge(
            gen3_pred[["index", "y_true", "y_pred"]],
            on="index",
            suffixes=(f"_{other_tag}", "_gen3"),
        )
        if merged.empty:
            continue
        if not (merged[f"y_true_{other_tag}"] == merged["y_true_gen3"]).all():
            raise RuntimeError(
                f"y_true diverge entre {other_tag} e {r['run_id']}"
            )
        result = mcnemar_test(
            merged[f"y_true_{other_tag}"].to_numpy(),
            merged[f"y_pred_{other_tag}"].to_numpy(),
            merged["y_pred_gen3"].to_numpy(),
        )
        rows.append(
            {
                "gen3_model": r["model"],
                "gen3_variant": r["variant"],
                "gen3_target_tag": r["target_tag"],
                f"{other_tag}": other_meta["model"] + "__" + other_meta["variant"],
                "n_test": len(merged),
                f"b_{other_tag}_only_correct": result["b"],
                "c_gen3_only_correct": result["c"],
                "mcnemar_statistic": result["statistic"],
                "p_value": result["p_value"],
                "winner": (
                    "gen3" if result["c"] > result["b"] else
                    other_tag if result["b"] > result["c"] else "tie"
                ),
                "significant_p_lt_0_05": result["p_value"] < 0.05,
                "gen3_f1_minority": r["f1_minority"],
                f"{other_tag}_f1_minority": other_meta["metrics"]["f1_minority"],
                "delta_f1_minority": r["f1_minority"] - other_meta["metrics"]["f1_minority"],
            }
        )
    return pd.DataFrame.from_records(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--champion-variant",
        default=None,
        help="Variant do campeão Gen 1 (override). Default: auto.",
    )
    parser.add_argument(
        "--champion-model",
        default=None,
        help="Model do campeão Gen 1 (override). Default: auto.",
    )
    parser.add_argument(
        "--gen2-champion-variant",
        default=None,
        help="Variant do campeão Gen 2 (override). Default: auto.",
    )
    parser.add_argument(
        "--gen2-champion-model",
        default=None,
        help="Model do campeão Gen 2 (override). Default: auto.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    gen3 = _load_gen3_runs(RUNS_DIR)
    if gen3.empty:
        print("Nenhum run Gen 3 encontrado em artifacts/runs/.")
        return 1
    gen3 = gen3.sort_values(["model", "target_tag"])
    gen3.to_csv(REPORT_DIR / "gen3_summary.csv", index=False)

    paired = _pair_bin_mc8(gen3)
    paired.to_csv(REPORT_DIR / "gen3_paired.csv", index=False)
    intra_mcnemar = _intra_mcnemar_rows(paired) if not paired.empty else pd.DataFrame()
    intra_mcnemar.to_csv(REPORT_DIR / "gen3_intra_mcnemar.csv", index=False)

    # vs Gen 1 champion
    if args.champion_variant and args.champion_model:
        gen1_dir = _find_run(RUNS_DIR, "gen1", args.champion_variant, args.champion_model)
    elif args.champion_variant or args.champion_model:
        print("erro: --champion-variant e --champion-model devem vir juntos.")
        return 2
    else:
        gen1_dir = _find_gen1_champion(RUNS_DIR)

    if gen1_dir is not None:
        gen1_meta = json.loads((gen1_dir / "metadata.json").read_text())
        print(
            f"Campeão Gen 1: {gen1_meta['model']} / {gen1_meta['variant']}"
            f"  (F1-min={gen1_meta['metrics']['f1_minority']:.4f})"
        )
        vs_gen1 = _mcnemar_vs(gen3, gen1_dir, "gen1")
        vs_gen1.to_csv(REPORT_DIR / "gen3_vs_gen1_champion.csv", index=False)
    else:
        print("Aviso: nenhum campeão Gen 1 encontrado; pulando comparação.")
        vs_gen1 = pd.DataFrame()

    # vs Gen 2 champion
    if args.gen2_champion_variant and args.gen2_champion_model:
        gen2_dir = _find_run(
            RUNS_DIR, "gen2", args.gen2_champion_variant, args.gen2_champion_model
        )
    elif args.gen2_champion_variant or args.gen2_champion_model:
        print("erro: --gen2-champion-variant e --gen2-champion-model devem vir juntos.")
        return 2
    else:
        gen2_dir = _find_gen2_champion(RUNS_DIR)

    if gen2_dir is not None:
        gen2_meta = json.loads((gen2_dir / "metadata.json").read_text())
        print(
            f"Campeão Gen 2: {gen2_meta['model']} / {gen2_meta['variant']}"
            f"  (F1-min={gen2_meta['metrics']['f1_minority']:.4f})"
        )
        vs_gen2 = _mcnemar_vs(gen3, gen2_dir, "gen2")
        vs_gen2.to_csv(REPORT_DIR / "gen3_vs_gen2_champion.csv", index=False)
    else:
        print("Aviso: nenhum run Gen 2 encontrado; pulando comparação.")
        vs_gen2 = pd.DataFrame()

    print(f"\nRuns Gen 3: {len(gen3)} | pares bin×mc8: {len(paired)}")
    print(f"Escrito em {REPORT_DIR}/\n")
    if not intra_mcnemar.empty:
        print("McNemar intra-Gen 3 (binary vs mc8):")
        print(
            intra_mcnemar[
                [
                    "model",
                    "b_bin_only_correct",
                    "c_mc8_only_correct",
                    "p_value",
                    "winner",
                    "significant_p_lt_0_05",
                ]
            ].to_string(index=False)
        )
        print()
    if not vs_gen1.empty:
        print("McNemar vs campeão Gen 1:")
        print(
            vs_gen1[
                [
                    "gen3_model",
                    "gen3_target_tag",
                    "gen3_f1_minority",
                    "delta_f1_minority",
                    "p_value",
                    "winner",
                    "significant_p_lt_0_05",
                ]
            ].to_string(index=False)
        )
        print()
    if not vs_gen2.empty:
        print("McNemar vs campeão Gen 2:")
        print(
            vs_gen2[
                [
                    "gen3_model",
                    "gen3_target_tag",
                    "gen3_f1_minority",
                    "delta_f1_minority",
                    "p_value",
                    "winner",
                    "significant_p_lt_0_05",
                ]
            ].to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
