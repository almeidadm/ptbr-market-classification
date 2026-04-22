"""Orquestração de um experimento Gen 1 fim-a-fim.

Responsabilidades:

- Aplicar o pré-processamento escolhido (`raw` ou `aggressive`) com cache.
- Construir a representação (`tfidf`, `bow` ou `fasttext`) ajustada
  **apenas no train**.
- Para cada classificador listado: treinar (binário ou multiclasse),
  pontuar val/test extraindo a probabilidade da classe `mercado`, ajustar
  _threshold_ no val (F1 da minoritária sobre o rótulo binário) e
  aplicá-lo no test.
- Medir latência de inferência (ms por 1000 artigos, sobre o val).
- Persistir o run em `artifacts/runs/{run_id}/` (metadata.json, metrics.json,
  predictions.csv).

Esta função é invocada tanto pelo script `scripts/run_gen1.py` (CLI)
quanto por testes de integração — por isso recebe DataFrames já
carregados em memória, não caminhos de parquet.

Target modes
------------
- `binary` (default): treina com `splits[*]["label"]` ∈ {0,1}; a classe
  positiva é `1` (mercado).
- `multiclass`: decompõe a classe negativa (plano_base.md, seção
  "Decomposição da Classe Negativa"). Treina com `splits[*]["category"]`
  colapsada pelo `collapse_scheme`, e usa `predict_proba[:, idx("mercado")]`
  como _score_ binário. Avaliação e _threshold_ continuam binários
  (`label` ∈ {0,1}) — o alvo do paper não muda.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ptbr_market import (
    evaluation,
    gen1_classical,
    preprocessing,
    representations,
    runs,
    targets,
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


def _positive_class_index(clf: Any, positive_class_label: Any) -> int:
    classes = list(clf.classes_)
    try:
        return classes.index(positive_class_label)
    except ValueError as exc:
        raise ValueError(
            f"Classe positiva {positive_class_label!r} não está em classes_"
            f" do classificador: {classes!r}. Verifique se o collapse_scheme"
            " preservou 'mercado' em train."
        ) from exc


def _run_one_model(
    name: str,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val_binary: Any,
    X_test: Any,
    y_test_binary: Any,
    positive_class_label: Any,
    test_indices: list[int],
    variant: str,
    config: dict[str, Any],
    split_meta_block: dict[str, Any],
) -> Path:
    started_at = runs.utc_now_iso()
    clf = gen1_classical.build_classifier(name)
    clf.fit(X_train, y_train)

    pos_idx = _positive_class_index(clf, positive_class_label)
    val_score = clf.predict_proba(X_val)[:, pos_idx]
    test_score = clf.predict_proba(X_test)[:, pos_idx]

    decision = threshold.fit_threshold(y_val_binary, val_score)
    y_pred_test = threshold.apply_threshold(test_score, decision)

    metrics = evaluation.compute_metrics(y_test_binary, test_score, y_pred_test)
    latency = _measure_latency_ms_per_1k(clf, X_val)
    finished_at = runs.utc_now_iso()

    run_dir = runs.new_run_dir("gen1", name, variant)
    runs.write_predictions(
        run_dir,
        indices=test_indices,
        y_true=y_test_binary.tolist(),
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
    target_mode: Literal["binary", "multiclass"] = "binary",
    collapse_scheme: str | None = None,
) -> list[Path]:
    """Executa um experimento Gen 1 fim-a-fim e retorna os diretórios de run criados.

    `splits` deve conter chaves `"train"`, `"val"`, `"test"`, cada uma
    apontando para um DataFrame com colunas `text` e `label`. Para
    `target_mode="multiclass"` também é exigida a coluna `category` em
    `train`. Val/test usam apenas `label` (avaliação é sempre binária).
    """
    missing = [s for s in REQUIRED_SPLITS if s not in splits]
    if missing:
        raise ValueError(f"splits faltando: {missing!r}")

    if target_mode not in targets.ALLOWED_TARGET_MODES:
        raise ValueError(
            f"target_mode deve ser um de {targets.ALLOWED_TARGET_MODES},"
            f" recebido {target_mode!r}."
        )

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

    if target_mode == "binary":
        if collapse_scheme is not None:
            raise ValueError(
                "collapse_scheme só é usado quando target_mode='multiclass'."
            )
        y_train_fit = splits["train"]["label"].to_numpy()
        positive_class_label: Any = 1
        target_block: dict[str, Any] = {
            "mode": "binary",
            "num_classes": 2,
            "positive_class_label": 1,
            "collapse_scheme": None,
        }
    else:
        if collapse_scheme is None:
            raise ValueError(
                "target_mode='multiclass' requer collapse_scheme (ex.:"
                f" {tuple(targets.COLLAPSE_SCHEMES)!r})."
            )
        y_train_strings, num_classes = targets.derive_multiclass_labels(
            splits, collapse_scheme
        )
        y_train_fit, positive_class_code, class_encoding = targets.encode_multiclass(
            y_train_strings
        )
        positive_class_label = positive_class_code
        target_block = {
            "mode": "multiclass",
            "num_classes": num_classes,
            "positive_class_label": targets.POSITIVE_CATEGORY_LABEL,
            "positive_class_code": positive_class_code,
            "collapse_scheme": collapse_scheme,
            "class_encoding": class_encoding,
        }

    y_val_binary = splits["val"]["label"].to_numpy()
    y_test_binary = splits["test"]["label"].to_numpy()

    texts = _preprocess_all(splits, preprocess_mode, force=not use_cache)
    X_train, X_val, X_test = _build_representation(
        representation,
        texts["train"],
        texts["val"],
        texts["test"],
        representation_params,
    )

    config = {
        "seed": gen1_classical.SEED,
        "mask_entities": preprocess_mode == "aggressive",
        "preprocess": preprocess_mode,
        "representation": representation,
        "representation_params": representation_params or {},
        "target": target_block,
    }

    run_dirs: list[Path] = []
    for name in model_list:
        run_dir = _run_one_model(
            name,
            X_train,
            y_train_fit,
            X_val,
            y_val_binary,
            X_test,
            y_test_binary,
            positive_class_label=positive_class_label,
            test_indices=splits["test"].index.tolist(),
            variant=variant,
            config=config,
            split_meta_block=split_meta_block,
        )
        run_dirs.append(run_dir)
    return run_dirs
