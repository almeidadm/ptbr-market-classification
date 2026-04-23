"""EDA dos splits OOT 80/10/10 do FolhaSP.

Gera as tabelas e figuras usadas em `docs/splits_eda.md`. Rodar com:

    uv run python scripts/eda_splits.py

Idempotente: regrava stdout estruturado e sobrescreve PNGs em
`docs/figures/splits_eda/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO / "artifacts" / "splits"
FIG_DIR = REPO / "docs" / "figures" / "splits_eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_ORDER = ("train", "val", "test")
SPLIT_COLORS = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}


def load_splits() -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(SPLITS_DIR / f"{name}.parquet") for name in SPLIT_ORDER}


def per_split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    total_n = sum(len(df) for df in splits.values())
    total_pos = sum(int(df["label"].sum()) for df in splits.values())
    for name, df in splits.items():
        n = len(df)
        pos = int(df["label"].sum())
        rows.append(
            {
                "split": name,
                "n": n,
                "share_total": n / total_n,
                "pos_count": pos,
                "pos_ratio": pos / n,
                "date_min": df["date"].min().date().isoformat(),
                "date_max": df["date"].max().date().isoformat(),
                "days_span": (df["date"].max() - df["date"].min()).days + 1,
                "n_null_text": int(df["text"].isna().sum()),
            }
        )
    rows.append(
        {
            "split": "total",
            "n": total_n,
            "share_total": 1.0,
            "pos_count": total_pos,
            "pos_ratio": total_pos / total_n,
            "date_min": "—",
            "date_max": "—",
            "days_span": None,
            "n_null_text": sum(int(df["text"].isna().sum()) for df in splits.values()),
        }
    )
    return pd.DataFrame(rows)


def category_distribution(splits: dict[str, pd.DataFrame], top_k: int = 15) -> pd.DataFrame:
    all_cats = sorted({c for df in splits.values() for c in df["category"].unique()})
    rows = []
    for cat in all_cats:
        row = {"category": cat}
        total = 0
        for name, df in splits.items():
            count = int((df["category"] == cat).sum())
            row[f"{name}_n"] = count
            row[f"{name}_share"] = count / len(df)
            total += count
        row["total_n"] = total
        rows.append(row)
    df_cat = pd.DataFrame(rows).sort_values("total_n", ascending=False)
    return df_cat.head(top_k).reset_index(drop=True)


def category_drift(splits: dict[str, pd.DataFrame], min_count: int = 500) -> pd.DataFrame:
    """Para cada categoria com >=min_count no total, mede drift de share entre splits."""
    all_cats = sorted({c for df in splits.values() for c in df["category"].unique()})
    rows = []
    for cat in all_cats:
        counts = {name: int((df["category"] == cat).sum()) for name, df in splits.items()}
        total = sum(counts.values())
        if total < min_count:
            continue
        shares = {name: counts[name] / len(splits[name]) for name in SPLIT_ORDER}
        row = {"category": cat, "total_n": total}
        for name in SPLIT_ORDER:
            row[f"{name}_share_pct"] = shares[name] * 100
        row["drift_test_minus_train_pp"] = (shares["test"] - shares["train"]) * 100
        rows.append(row)
    df = pd.DataFrame(rows)
    df["abs_drift"] = df["drift_test_minus_train_pp"].abs()
    return df.sort_values("abs_drift", ascending=False).reset_index(drop=True)


def text_length_stats(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in splits.items():
        title_chars = df["title"].fillna("").str.len()
        text_chars = df["text"].fillna("").str.len()
        text_tokens = df["text"].fillna("").str.split().str.len()
        rows.append(
            {
                "split": name,
                "title_chars_mean": title_chars.mean(),
                "title_chars_median": title_chars.median(),
                "text_chars_mean": text_chars.mean(),
                "text_chars_median": text_chars.median(),
                "text_chars_p95": text_chars.quantile(0.95),
                "text_tokens_mean": text_tokens.mean(),
                "text_tokens_median": text_tokens.median(),
                "text_tokens_p95": text_tokens.quantile(0.95),
            }
        )
    return pd.DataFrame(rows)


def monthly_volume(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for name, df in splits.items():
        g = df.assign(ym=df["date"].dt.to_period("M").astype(str)).groupby("ym").agg(
            n=("label", "size"), pos=("label", "sum")
        )
        g["split"] = name
        g["pos_ratio"] = g["pos"] / g["n"]
        frames.append(g.reset_index())
    return pd.concat(frames, ignore_index=True)


def weekday_distribution(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in splits.items():
        counts = df["date"].dt.dayofweek.value_counts().sort_index()
        for wd, c in counts.items():
            rows.append({"split": name, "weekday": int(wd), "n": int(c), "share": c / len(df)})
    return pd.DataFrame(rows)


def positive_category_audit(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in splits.items():
        pos = df[df["label"] == 1]
        cats = pos["category"].value_counts()
        rows.append(
            {
                "split": name,
                "n_positive": len(pos),
                "positive_category_value": cats.index[0] if len(cats) else "—",
                "n_positive_cat": int(cats.iloc[0]) if len(cats) else 0,
                "all_positive_are_mercado": (pos["category"] == "mercado").all() if len(pos) else True,
            }
        )
    return pd.DataFrame(rows)


def plot_monthly_volume(monthly: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    all_ym = sorted(monthly["ym"].unique())
    for name in SPLIT_ORDER:
        sub = monthly[monthly["split"] == name].set_index("ym").reindex(all_ym)
        ax1.plot(all_ym, sub["n"], marker="o", label=name, color=SPLIT_COLORS[name])
        ax2.plot(all_ym, sub["pos_ratio"] * 100, marker="o", label=name, color=SPLIT_COLORS[name])
    ax1.set_ylabel("Artigos por mês")
    ax1.set_title("Volume mensal por split (FolhaSP 2015-2017)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.axhline(12.55, linestyle="--", color="gray", alpha=0.6, label="prevalência global 12,55%")
    ax2.set_ylabel("% positivos (mercado)")
    ax2.set_xlabel("Mês")
    ax2.set_title("Prevalência da classe positiva por mês")
    ax2.legend()
    ax2.grid(alpha=0.3)
    for ax in (ax1, ax2):
        ticks = [ym for i, ym in enumerate(all_ym) if i % 3 == 0]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monthly_volume.png", dpi=130)
    plt.close(fig)


def plot_class_balance(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    per_split = summary[summary["split"] != "total"]
    x = np.arange(len(per_split))
    neg = per_split["n"] - per_split["pos_count"]
    ax.bar(x, per_split["pos_count"], color="#d62728", label="mercado (1)")
    ax.bar(x, neg, bottom=per_split["pos_count"], color="#7f7f7f", alpha=0.5, label="outros (0)")
    for i, row in per_split.reset_index(drop=True).iterrows():
        ax.text(i, row["n"] * 0.5, f"{row['pos_ratio'] * 100:.2f}%", ha="center", color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(per_split["split"])
    ax.set_ylabel("Nº artigos")
    ax.set_title("Distribuição de classe por split (contagem absoluta)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_balance.png", dpi=130)
    plt.close(fig)


def plot_text_length(splits: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [splits[name]["text"].fillna("").str.split().str.len().clip(upper=2000) for name in SPLIT_ORDER]
    ax.boxplot(data, labels=list(SPLIT_ORDER), showfliers=False)
    ax.set_ylabel("Tokens por artigo (clipped em 2000)")
    ax.set_title("Distribuição do comprimento do texto por split")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "text_length.png", dpi=130)
    plt.close(fig)


def plot_category_drift(drift_df: pd.DataFrame, top_k: int = 10) -> None:
    top = drift_df.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(top))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top["drift_test_minus_train_pp"]]
    ax.barh(y, top["drift_test_minus_train_pp"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(top["category"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Drift de share test − train (p.p.)")
    ax.set_title(f"Top {top_k} categorias com maior drift temporal (|share test − share train|)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "category_drift.png", dpi=130)
    plt.close(fig)


def plot_weekday(weekday: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    x = np.arange(7)
    width = 0.25
    for i, name in enumerate(SPLIT_ORDER):
        sub = weekday[weekday["split"] == name].set_index("weekday").reindex(range(7))
        ax.bar(x + (i - 1) * width, sub["share"] * 100, width, label=name, color=SPLIT_COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% dos artigos do split")
    ax.set_title("Distribuição por dia da semana")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weekday.png", dpi=130)
    plt.close(fig)


def main() -> None:
    print("== Carregando splits ==")
    splits = load_splits()

    summary = per_split_summary(splits)
    print("\n== Resumo por split ==")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    cat_top = category_distribution(splits, top_k=15)
    print("\n== Top 15 categorias (contagem total) ==")
    print(cat_top.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    drift = category_drift(splits, min_count=500)
    print("\n== Drift de share por categoria (test − train, p.p.) ==")
    print(drift.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    tl = text_length_stats(splits)
    print("\n== Estatísticas de comprimento do texto ==")
    print(tl.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    monthly = monthly_volume(splits)
    print("\n== Volume mensal (primeiros 6 e últimos 6 meses) ==")
    print(pd.concat([monthly.head(6), monthly.tail(6)]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    wd = weekday_distribution(splits)
    print("\n== Distribuição por dia da semana ==")
    print(wd.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    audit = positive_category_audit(splits)
    print("\n== Auditoria da classe positiva (deve ser 100% 'mercado') ==")
    print(audit.to_string(index=False))

    print("\n== Gerando figuras ==")
    plot_monthly_volume(monthly)
    plot_class_balance(summary)
    plot_text_length(splits)
    plot_category_drift(drift)
    plot_weekday(wd)

    print(f"Figuras salvas em {FIG_DIR.relative_to(REPO)}")

    out = {
        "summary": summary.to_dict(orient="records"),
        "category_drift_top10": drift.head(10).to_dict(orient="records"),
        "text_length": tl.to_dict(orient="records"),
        "positive_audit": audit.to_dict(orient="records"),
    }
    (REPO / "docs" / "figures" / "splits_eda" / "stats.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    print("JSON de referência salvo em docs/figures/splits_eda/stats.json")


if __name__ == "__main__":
    main()
