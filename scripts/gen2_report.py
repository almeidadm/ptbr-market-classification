"""Agrega os runs Gen 2 em `artifacts/runs/` e produz relatórios pareados.

Lê `metadata.json` + `predictions.csv` de cada `*__gen2__*` e grava:

- `artifacts/reports/gen2_summary.csv`: uma linha por run Gen 2 com
  métricas, latência, VRAM e config.
- `artifacts/reports/gen2_paired.csv`: binary vs mc8 lado a lado para
  cada modelo Gen 2 (quando ambas as variantes existirem).
- `artifacts/reports/gen2_intra_mcnemar.csv`: McNemar pareado bin vs
  mc8 dentro de Gen 2 — responde à pergunta "a decomposição da classe
  negativa ajudou também nos encoders?".
- `artifacts/reports/gen2_vs_gen1_champion.csv`: McNemar pareado de
  cada modelo Gen 2 contra o campeão Gen 1 (default:
  `tfidf+linearsvc+mc8`, selecionado automaticamente via maior F1-min
  em `*-aggr-*` do Gen 1).

Uso:

    uv run python scripts/gen2_report.py
    uv run python scripts/gen2_report.py --champion-variant tfidf-aggr-mc8 \\
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


def _load_gen2_runs(runs_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for d in sorted(runs_dir.iterdir()):
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("generation") != "gen2":
            continue
        target = meta.get("config", {}).get("target", {})
        mode = target.get("mode", "binary")
        if mode == "multiclass":
            mode_tag = f"mc{target.get('num_classes')}"
        else:
            mode_tag = "bin"
        records.append(
            {
                "run_id": meta["run_id"],
                "model": meta["model"],
                "variant": meta["variant"],
                "target_mode": mode,
                "target_tag": mode_tag,
                "collapse_scheme": target.get("collapse_scheme"),
                "num_classes": target.get("num_classes", 2),
                "hf_id": meta["config"].get("hf_id"),
                "bucket": meta["config"].get("bucket"),
                "max_length": meta["config"].get("max_length"),
                "epochs": meta["config"].get("epochs"),
                "threshold": meta["threshold"]["value"],
                "pr_auc": meta["metrics"]["pr_auc"],
                "roc_auc": meta["metrics"]["roc_auc"],
                "f1_macro": meta["metrics"]["f1_macro"],
                "f1_minority": meta["metrics"]["f1_minority"],
                "precision_minority": meta["metrics"]["precision_minority"],
                "recall_minority": meta["metrics"]["recall_minority"],
                "latency_ms_per_1k": meta["efficiency"]["latency_ms_per_1k"],
                "vram_peak_mb": meta["efficiency"].get("vram_peak_mb", 0),
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
        "threshold",
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


def _find_gen1_champion(runs_dir: Path) -> Path:
    """Seleciona o run Gen 1 com maior F1-min entre os `*-aggr-*`.

    Mantém a análise auto-consistente mesmo se novos runs Gen 1 forem
    adicionados — sempre compara contra o melhor disponível.
    """
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
    if best_dir is None:
        raise RuntimeError(
            f"Nenhum run Gen 1 com variant `*-aggr-*` encontrado em {runs_dir}."
        )
    return best_dir


def _find_gen1_run(runs_dir: Path, variant: str, model: str) -> Path:
    matches = [
        d for d in runs_dir.iterdir()
        if d.is_dir() and f"__gen1__{model}__{variant}" in d.name
    ]
    if not matches:
        raise RuntimeError(
            f"Run Gen 1 não encontrado: variant={variant!r}, model={model!r}."
        )
    if len(matches) > 1:
        matches.sort()
    return matches[-1]


def _mcnemar_vs_champion(
    gen2_df: pd.DataFrame,
    champion_dir: Path,
) -> pd.DataFrame:
    champ_meta = json.loads((champion_dir / "metadata.json").read_text())
    champ_pred = pd.read_csv(champion_dir / "predictions.csv")
    rows: list[dict] = []
    for _, r in gen2_df.iterrows():
        gen2_pred = pd.read_csv(Path(r["run_dir"]) / "predictions.csv")
        merged = champ_pred[["index", "y_true", "y_pred"]].merge(
            gen2_pred[["index", "y_true", "y_pred"]],
            on="index",
            suffixes=("_champ", "_gen2"),
        )
        if not (merged["y_true_champ"] == merged["y_true_gen2"]).all():
            raise RuntimeError(
                f"y_true diverge entre campeão e {r['run_id']}"
            )
        result = mcnemar_test(
            merged["y_true_champ"].to_numpy(),
            merged["y_pred_champ"].to_numpy(),
            merged["y_pred_gen2"].to_numpy(),
        )
        rows.append(
            {
                "gen2_model": r["model"],
                "gen2_variant": r["variant"],
                "gen2_target_tag": r["target_tag"],
                "champion": champ_meta["model"] + "__" + champ_meta["variant"],
                "n_test": len(merged),
                "b_champ_only_correct": result["b"],
                "c_gen2_only_correct": result["c"],
                "mcnemar_statistic": result["statistic"],
                "p_value": result["p_value"],
                "winner": (
                    "gen2" if result["c"] > result["b"] else
                    "champion" if result["b"] > result["c"] else "tie"
                ),
                "significant_p_lt_0_05": result["p_value"] < 0.05,
                "gen2_f1_minority": r["f1_minority"],
                "champion_f1_minority": champ_meta["metrics"]["f1_minority"],
                "delta_f1_minority": r["f1_minority"] - champ_meta["metrics"]["f1_minority"],
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
        help="Variant do campeão Gen 1. Se omitido, auto-seleciona maior F1-min.",
    )
    parser.add_argument(
        "--champion-model",
        default=None,
        help="Model do campeão Gen 1 (requer também --champion-variant).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    gen2 = _load_gen2_runs(RUNS_DIR)
    if gen2.empty:
        print("Nenhum run Gen 2 encontrado em artifacts/runs/.")
        return 1
    gen2 = gen2.sort_values(["model", "target_tag"])
    gen2.to_csv(REPORT_DIR / "gen2_summary.csv", index=False)

    paired = _pair_bin_mc8(gen2)
    paired.to_csv(REPORT_DIR / "gen2_paired.csv", index=False)
    intra_mcnemar = _intra_mcnemar_rows(paired) if not paired.empty else pd.DataFrame()
    intra_mcnemar.to_csv(REPORT_DIR / "gen2_intra_mcnemar.csv", index=False)

    if args.champion_variant and args.champion_model:
        champion_dir = _find_gen1_run(RUNS_DIR, args.champion_variant, args.champion_model)
    elif args.champion_variant or args.champion_model:
        print(
            "erro: --champion-variant e --champion-model devem vir juntos."
        )
        return 2
    else:
        champion_dir = _find_gen1_champion(RUNS_DIR)

    champ_meta = json.loads((champion_dir / "metadata.json").read_text())
    print(
        f"Campeão Gen 1: {champ_meta['model']} / {champ_meta['variant']}"
        f"  (F1-min={champ_meta['metrics']['f1_minority']:.4f},"
        f"   PR-AUC={champ_meta['metrics']['pr_auc']:.4f})"
    )

    mcnemar_df = _mcnemar_vs_champion(gen2, champion_dir)
    mcnemar_df.to_csv(REPORT_DIR / "gen2_vs_gen1_champion.csv", index=False)

    print(f"\nRuns Gen 2: {len(gen2)} | pares bin×mc8: {len(paired)}")
    print(f"Escrito em {REPORT_DIR}/\n")
    if not intra_mcnemar.empty:
        print("McNemar intra-Gen 2 (binary vs mc8):")
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
    print("McNemar vs campeão Gen 1:")
    print(
        mcnemar_df[
            [
                "gen2_model",
                "gen2_target_tag",
                "gen2_f1_minority",
                "delta_f1_minority",
                "b_champ_only_correct",
                "c_gen2_only_correct",
                "p_value",
                "winner",
                "significant_p_lt_0_05",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
