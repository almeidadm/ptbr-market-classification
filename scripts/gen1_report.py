"""Agrega os runs Gen 1 em `artifacts/runs/` e produz relatórios pareados.

Lê cada `metadata.json` em `artifacts/runs/*__gen1__*__*-aggr-*` e a
respectiva `predictions.csv`, e grava:

- `artifacts/reports/gen1_summary.csv`: uma linha por run com métricas,
  latência e config.
- `artifacts/reports/gen1_paired.csv`: binary vs mc8 lado a lado para
  cada par `(representation, model)`.
- `artifacts/reports/gen1_mcnemar.csv`: teste de McNemar pareado
  comparando as predições binárias (threshold-aplicado) dos dois modos.

Uso:

    uv run python scripts/gen1_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ptbr_market.evaluation import mcnemar_test

RUNS_DIR = Path("artifacts/runs")
REPORT_DIR = Path("artifacts/reports")


def _load_runs(runs_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for run_dir in sorted(runs_dir.iterdir()):
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        variant = meta.get("variant", "")
        # Somente a matriz final (aggressive, bin|mc8).
        if "-aggr-" not in variant:
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
                "representation": meta["config"]["representation"],
                "preprocess": meta["config"]["preprocess"],
                "target_mode": mode,
                "target_tag": mode_tag,
                "collapse_scheme": target.get("collapse_scheme"),
                "num_classes": target.get("num_classes", 2),
                "threshold": meta["threshold"]["value"],
                "pr_auc": meta["metrics"]["pr_auc"],
                "roc_auc": meta["metrics"]["roc_auc"],
                "f1_macro": meta["metrics"]["f1_macro"],
                "f1_minority": meta["metrics"]["f1_minority"],
                "precision_minority": meta["metrics"]["precision_minority"],
                "recall_minority": meta["metrics"]["recall_minority"],
                "latency_ms_per_1k": meta["efficiency"]["latency_ms_per_1k"],
                "n_test": meta["split"]["n_test"],
                "run_dir": str(run_dir),
            }
        )
    return pd.DataFrame.from_records(records)


def _pair(df: pd.DataFrame) -> pd.DataFrame:
    binary = df[df["target_tag"] == "bin"].set_index(["representation", "model"])
    mc8 = df[df["target_tag"] == "mc8"].set_index(["representation", "model"])
    metric_cols = [
        "f1_minority",
        "pr_auc",
        "precision_minority",
        "recall_minority",
        "latency_ms_per_1k",
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
    return joined.reset_index().sort_values(["representation", "model"])


def _mcnemar_rows(paired: pd.DataFrame) -> pd.DataFrame:
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
                f"y_true diverge entre binary e mc8 para {row['representation']}+{row['model']}"
            )
        result = mcnemar_test(
            merged["y_true_bin"].to_numpy(),
            merged["y_pred_bin"].to_numpy(),
            merged["y_pred_mc8"].to_numpy(),
        )
        rows.append(
            {
                "representation": row["representation"],
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


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = _load_runs(RUNS_DIR)
    if summary.empty:
        print("Nenhum run aggr-bin/aggr-mc8 encontrado em artifacts/runs/.")
        return 1
    summary = summary.sort_values(["representation", "model", "target_tag"])
    summary.to_csv(REPORT_DIR / "gen1_summary.csv", index=False)

    paired = _pair(summary)
    paired.to_csv(REPORT_DIR / "gen1_paired.csv", index=False)

    mcnemar = _mcnemar_rows(paired)
    mcnemar.to_csv(REPORT_DIR / "gen1_mcnemar.csv", index=False)

    print(f"Runs: {len(summary)} | pares: {len(paired)}")
    print(f"Escrito em {REPORT_DIR}/")
    print(
        mcnemar[
            [
                "representation",
                "model",
                "b_bin_only_correct",
                "c_mc8_only_correct",
                "p_value",
                "winner",
                "significant_p_lt_0_05",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
