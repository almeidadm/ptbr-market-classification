"""Orquestração de um experimento Gen 1 fim-a-fim.

Responsabilidades:

- Aplicar o pré-processamento escolhido (`raw` ou `aggressive`) com cache.
- Construir a representação (`tfidf` ou `bow`) ajustada **apenas no train**.
- Para cada classificador listado: treinar, pontuar val/test, ajustar
  _threshold_ no val (F1 da minoritária) e aplicá-lo no test.
- Medir latência de inferência (ms por 1000 artigos, sobre o val).
- Persistir o run em `artifacts/runs/{run_id}/` (metadata.json, metrics.json,
  predictions.csv).

Esta função é invocada tanto pelo script `scripts/run_gen1.py` (CLI)
quanto por testes de integração — por isso recebe DataFrames já
carregados em memória, não caminhos de parquet.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from ptbr_market import (
    evaluation,
    gen1_classical,
    preprocessing,
    representations,
    runs,
    threshold,
)

REQUIRED_SPLITS: tuple[str, ...] = ("train", "val", "test")
ALLOWED_REPRESENTATIONS: tuple[str, ...] = ("tfidf", "bow", "fasttext")

# Modelos que exigem features não-negativas — incompatíveis com médias
# fastText (que produzem valores negativos).
NON_NEGATIVE_ONLY_MODELS: tuple[str, ...] = ("multinomialnb", "complementnb")
DENSE_NEGATIVE_REPRESENTATIONS: tuple[str, ...] = ("fasttext",)


def _preprocess_all(
    splits: dict[str, pd.DataFrame],
    mode: str,
    force: bool,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name in REQUIRED_SPLITS:
        out[name] = preprocessing.preprocess_split_cached(
            name, mode, splits[name]["text"].tolist(), force=force
        )
    return out


def _collect_tokens(texts: Iterable[str]) -> set[str]:
    vocab: set[str] = set()
    for t in texts:
        vocab.update(t.split())
    return vocab


def _build_fasttext_features(
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
    params: dict[str, Any],
) -> tuple[Any, Any, Any]:
    path = params.get("path")
    vocab = _collect_tokens(train_texts) | _collect_tokens(val_texts) | _collect_tokens(
        test_texts
    )
    vectors = representations.load_fasttext_vectors(path=path, vocabulary=vocab)
    return (
        representations.fasttext_average(train_texts, vectors),
        representations.fasttext_average(val_texts, vectors),
        representations.fasttext_average(test_texts, vectors),
    )


def _build_representation(
    name: str,
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
    params: dict[str, Any] | None,
) -> tuple[Any, Any, Any]:
    params = params or {}
    if name == "tfidf":
        vec, X_train = representations.build_tfidf(train_texts, **params)
        return X_train, vec.transform(val_texts), vec.transform(test_texts)
    if name == "bow":
        vec, X_train = representations.build_bow(train_texts, **params)
        return X_train, vec.transform(val_texts), vec.transform(test_texts)
    if name == "fasttext":
        return _build_fasttext_features(train_texts, val_texts, test_texts, params)
    raise ValueError(
        f"representation deve ser um de {ALLOWED_REPRESENTATIONS}, recebido {name!r}."
    )


def _measure_latency_ms_per_1k(clf: Any, X_val: Any) -> float:
    n = X_val.shape[0]
    if n == 0:
        return 0.0
    t0 = time.perf_counter()
    clf.predict_proba(X_val)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms * 1000.0 / n


def _run_one_model(
    name: str,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    X_test: Any,
    y_test: Any,
    test_indices: list[int],
    variant: str,
    config: dict[str, Any],
    split_meta_block: dict[str, Any],
) -> Path:
    started_at = runs.utc_now_iso()
    clf = gen1_classical.build_classifier(name)
    clf.fit(X_train, y_train)

    val_score = clf.predict_proba(X_val)[:, 1]
    test_score = clf.predict_proba(X_test)[:, 1]

    decision = threshold.fit_threshold(y_val, val_score)
    y_pred_test = threshold.apply_threshold(test_score, decision)

    metrics = evaluation.compute_metrics(y_test, test_score, y_pred_test)
    latency = _measure_latency_ms_per_1k(clf, X_val)
    finished_at = runs.utc_now_iso()

    run_dir = runs.new_run_dir("gen1", name, variant)
    runs.write_predictions(
        run_dir,
        indices=test_indices,
        y_true=y_test.tolist(),
        y_pred=y_pred_test.tolist(),
        y_score=test_score.tolist(),
    )
    runs.write_metrics(run_dir, metrics)
    runs.write_run_metadata(
        run_dir,
        {
            "run_id": run_dir.name,
            "git_commit": runs.git_commit(),
            "started_at": started_at,
            "finished_at": finished_at,
            "generation": "gen1",
            "model": name,
            "variant": variant,
            "config": config,
            "split": split_meta_block,
            "threshold": {
                "fitted_on": decision.fitted_on,
                "value": decision.value,
                "objective": decision.objective,
            },
            "metrics": metrics,
            "efficiency": {"latency_ms_per_1k": latency, "vram_peak_mb": 0},
        },
    )
    return run_dir


def run_gen1_experiment(
    splits: dict[str, pd.DataFrame],
    split_meta_block: dict[str, Any],
    models: Iterable[str],
    variant: str = "tfidf-lemmatized",
    preprocess_mode: str = "aggressive",
    representation: str = "tfidf",
    representation_params: dict[str, Any] | None = None,
    use_cache: bool = True,
) -> list[Path]:
    """Executa um experimento Gen 1 fim-a-fim e retorna os diretórios de run criados.

    `splits` deve conter chaves `"train"`, `"val"`, `"test"`, cada uma
    apontando para um DataFrame com colunas `text` e `label` (e idealmente
    `date`, mas não é exigida aqui — o metadata das janelas vem de
    `split_meta_block`).
    """
    missing = [s for s in REQUIRED_SPLITS if s not in splits]
    if missing:
        raise ValueError(f"splits faltando: {missing!r}")

    model_list = list(models)
    for m in model_list:
        if m not in gen1_classical.GEN1_CLASSIFIERS:
            raise ValueError(
                f"Modelo desconhecido: {m!r}. Use um de"
                f" {gen1_classical.GEN1_CLASSIFIERS}."
            )

    if representation in DENSE_NEGATIVE_REPRESENTATIONS:
        bad = [m for m in model_list if m in NON_NEGATIVE_ONLY_MODELS]
        if bad:
            raise ValueError(
                f"{representation!r} produz features com valores negativos;"
                f" incompatível com {bad!r}. Use linearsvc ou logreg."
            )

    texts = _preprocess_all(splits, preprocess_mode, force=not use_cache)
    X_train, X_val, X_test = _build_representation(
        representation,
        texts["train"],
        texts["val"],
        texts["test"],
        representation_params,
    )

    y_train = splits["train"]["label"].to_numpy()
    y_val = splits["val"]["label"].to_numpy()
    y_test = splits["test"]["label"].to_numpy()

    config = {
        "seed": gen1_classical.SEED,
        "mask_entities": preprocess_mode == "aggressive",
        "preprocess": preprocess_mode,
        "representation": representation,
        "representation_params": representation_params or {},
    }

    run_dirs: list[Path] = []
    for name in model_list:
        run_dir = _run_one_model(
            name,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            test_indices=splits["test"].index.tolist(),
            variant=variant,
            config=config,
            split_meta_block=split_meta_block,
        )
        run_dirs.append(run_dir)
    return run_dirs
