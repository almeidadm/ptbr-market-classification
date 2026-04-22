"""Fine-tuning de encoders BERT (Geração 2).

Pipeline único e reutilizável para rodar um encoder PT-BR/Multilíngue/Domain
sobre os mesmos splits OoT do Gen 1, mantendo o contrato de artefatos
(`metadata.json`, `metrics.json`, `predictions.csv`) para permitir McNemar
pareado contra o campeão Gen 1 sem mexer em `scripts/*report.py`.

Convenções:

- **Pré-processamento `raw`** (decisão D6 fixada em 2026-04-22): sem máscara
  de entidades; o tokenizer BPE/WordPiece já compõe `R$ 10 bi` em subtokens
  que a atenção combina, e substituir por `[VALOR_MONETARIO]` quebra essa
  composição (o literal vira `[`, `VALOR`, `_`, `MONETARIO`, `]`).
- **Latência inclui tokenização.** Mede o pipeline real de inferência.
- **VRAM pico** é lido de `torch.cuda.max_memory_allocated()` — reset antes
  do fit, leitura após o último predict.
- **Threshold no val, congelado no test** — mesmo contrato do Gen 1, via
  `ptbr_market.threshold.fit_threshold`/`apply_threshold`.
- **`seed=1`** em `torch`, `numpy`, `random`, `transformers.set_seed` — a
  reprodutibilidade bit-exact não é garantida em GPU mas a variância fica
  em ~0,3 pp de F1 entre execuções.

Este módulo só faz sentido com os extras `gen2` instalados
(`uv sync --group gen2` ou `pip install torch transformers accelerate
datasets`). As importações ficam no topo — falhar cedo se faltar
dependência.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ptbr_market import evaluation, preprocessing, runs, targets, threshold

SEED = 1
DEFAULT_MAX_LENGTH = 256
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-5


@dataclass(frozen=True, slots=True)
class Gen2ModelSpec:
    """Parâmetros específicos de um encoder BERT na matriz Gen 2.

    `batch_size` e `grad_accum` são calibrados para T4 (16 GB) com fp16
    e `max_length=256`; pode-se sobrescrever via CLI se rodar em L4/A100.
    """

    slug: str
    hf_id: str
    bucket: str  # "domain" | "ptbr" | "multilingual" | "efficiency"
    batch_size: int
    grad_accum: int = 1


GEN2_MODELS: dict[str, Gen2ModelSpec] = {
    "bertimbau-base": Gen2ModelSpec(
        slug="bertimbau-base",
        hf_id="neuralmind/bert-base-portuguese-cased",
        bucket="ptbr",
        batch_size=16,
    ),
    "finbert-ptbr": Gen2ModelSpec(
        slug="finbert-ptbr",
        hf_id="lucas-leme/FinBERT-PT-BR",
        bucket="domain",
        batch_size=16,
    ),
    "distilbertimbau": Gen2ModelSpec(
        slug="distilbertimbau",
        hf_id="adalbertojunior/distilbert-portuguese-cased",
        bucket="efficiency",
        batch_size=32,
    ),
    "bertimbau-large": Gen2ModelSpec(
        slug="bertimbau-large",
        hf_id="neuralmind/bert-large-portuguese-cased",
        bucket="ptbr",
        batch_size=8,
        grad_accum=4,
    ),
    "xlmr-base": Gen2ModelSpec(
        slug="xlmr-base",
        hf_id="FacebookAI/xlm-roberta-base",
        bucket="multilingual",
        batch_size=16,
    ),
    "deb3rta-base": Gen2ModelSpec(
        slug="deb3rta-base",
        hf_id="higopires/DeB3RTa-base",
        bucket="domain",
        batch_size=16,
    ),
}


def _seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


class _TextDataset(torch.utils.data.Dataset):
    """Dataset PyTorch mínimo — tokenização sob demanda (lazy).

    Tokenizar todo o corpus de uma vez consome ~6 GB de RAM no train (133k
    linhas × max_length=256 × 3 tensores int64); tokenizar por batch no
    `__getitem__` troca CPU por RAM e converge igual.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _softmax_positive_score(logits: np.ndarray, positive_class_code: int) -> np.ndarray:
    """Extrai `P(classe=mercado)` via softmax sobre logits (n, num_labels).

    Em binário, `positive_class_code=1`. Em multiclasse, é o código da
    classe `mercado` gravado no metadata via `targets.encode_multiclass`.
    """
    t = torch.from_numpy(logits)
    probs = torch.softmax(t, dim=-1).numpy()
    return probs[:, positive_class_code]


def _predict_scores(
    trainer: Trainer,
    dataset: _TextDataset,
    positive_class_code: int,
) -> np.ndarray:
    out = trainer.predict(dataset)
    return _softmax_positive_score(out.predictions, positive_class_code)


def _measure_latency_ms_per_1k(
    trainer: Trainer,
    texts: list[str],
    tokenizer: Any,
    max_length: int,
) -> float:
    """Latência inferência sobre val, incluindo tokenização.

    Constrói um dataset novo a partir dos textos crus (para medir tokenização
    também, conforme decisão) e cronometra `trainer.predict`.
    """
    n = len(texts)
    if n == 0:
        return 0.0
    t0 = time.perf_counter()
    dataset = _TextDataset(texts, labels=None, tokenizer=tokenizer, max_length=max_length)
    trainer.predict(dataset)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms * 1000.0 / n


def _peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


def run_gen2_experiment(
    splits: dict[str, pd.DataFrame],
    split_meta_block: dict[str, Any],
    model_slug: str,
    variant: str | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int | None = None,
    grad_accum: int | None = None,
    output_dir: Path | None = None,
    target_mode: Literal["binary", "multiclass"] = "binary",
    collapse_scheme: str | None = None,
) -> Path:
    """Fine-tuna um encoder BERT e persiste o run no padrão Gen 2.

    Retorna o `run_dir` criado em `artifacts/runs/{stamp}__gen2__{slug}__{variant}/`.

    `output_dir` é onde o HF Trainer escreve checkpoints intermediários
    durante o treino (padrão: `artifacts/tmp/gen2_trainer/{slug}/`). Não é
    o `run_dir` final — esse sai do `runs.new_run_dir`.

    `target_mode="multiclass"` reproduz a decomposição Gen 1: treina com
    `num_labels = len(keep)+1` sobre a categoria colapsada; val/test
    continuam binários (extrai `P(mercado)` via softmax e aplica
    threshold sobre o rótulo binário original).
    """
    if model_slug not in GEN2_MODELS:
        raise ValueError(
            f"Modelo Gen 2 desconhecido: {model_slug!r}. Use um de"
            f" {tuple(GEN2_MODELS)}."
        )
    if target_mode not in targets.ALLOWED_TARGET_MODES:
        raise ValueError(
            f"target_mode deve ser um de {targets.ALLOWED_TARGET_MODES},"
            f" recebido {target_mode!r}."
        )
    if target_mode == "binary" and collapse_scheme is not None:
        raise ValueError(
            "collapse_scheme só é usado quando target_mode='multiclass'."
        )
    if target_mode == "multiclass" and collapse_scheme is None:
        raise ValueError(
            "target_mode='multiclass' requer collapse_scheme (ex.:"
            f" {tuple(targets.COLLAPSE_SCHEMES)!r})."
        )

    spec = GEN2_MODELS[model_slug]
    effective_batch = batch_size or spec.batch_size
    effective_grad_accum = grad_accum or spec.grad_accum
    target_tag = targets.target_variant_tag(target_mode, collapse_scheme)
    variant = variant or f"raw-ml{max_length}-{target_tag}"

    _seed_all(SEED)

    started_at = runs.utc_now_iso()

    train_texts = preprocessing.preprocess_raw(splits["train"]["text"].tolist())
    val_texts = preprocessing.preprocess_raw(splits["val"]["text"].tolist())
    test_texts = preprocessing.preprocess_raw(splits["test"]["text"].tolist())

    y_val_binary = splits["val"]["label"].to_numpy()
    y_test_binary = splits["test"]["label"].to_numpy()

    if target_mode == "binary":
        y_train_fit = splits["train"]["label"].astype(int).to_numpy()
        num_labels = 2
        positive_class_code = 1
        target_block: dict[str, Any] = {
            "mode": "binary",
            "num_classes": 2,
            "positive_class_label": 1,
            "collapse_scheme": None,
        }
    else:
        y_train_strings, num_classes = targets.derive_multiclass_labels(
            splits, collapse_scheme
        )
        y_train_fit, positive_class_code, class_encoding = targets.encode_multiclass(
            y_train_strings
        )
        num_labels = num_classes
        target_block = {
            "mode": "multiclass",
            "num_classes": num_classes,
            "positive_class_label": targets.POSITIVE_CATEGORY_LABEL,
            "positive_class_code": positive_class_code,
            "collapse_scheme": collapse_scheme,
            "class_encoding": class_encoding,
        }

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.hf_id, num_labels=num_labels
    )

    train_ds = _TextDataset(train_texts, y_train_fit.tolist(), tokenizer, max_length)
    val_ds = _TextDataset(val_texts, y_val_binary.tolist(), tokenizer, max_length)
    test_ds = _TextDataset(test_texts, y_test_binary.tolist(), tokenizer, max_length)

    trainer_out = output_dir or (runs.artifacts_root() / "tmp" / "gen2_trainer" / spec.slug)
    trainer_out.mkdir(parents=True, exist_ok=True)

    fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(trainer_out),
        num_train_epochs=epochs,
        per_device_train_batch_size=effective_batch,
        per_device_eval_batch_size=effective_batch * 2,
        gradient_accumulation_steps=effective_grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        # Avalia-se externamente: em multiclass, val_ds carrega labels
        # binários que não batem com o espaço multiclass do modelo, e o
        # eval do Trainer computaria uma loss enganosa. O ajuste de
        # threshold + métricas reais ocorrem no predict() pós-treino.
        eval_strategy="no",
        save_strategy="no",
        fp16=fp16,
        seed=SEED,
        data_seed=SEED,
        report_to=[],
        dataloader_num_workers=2,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )
    trainer.train()

    val_score = _predict_scores(trainer, val_ds, positive_class_code)
    test_score = _predict_scores(trainer, test_ds, positive_class_code)

    decision = threshold.fit_threshold(y_val_binary, val_score)
    y_pred_test = threshold.apply_threshold(test_score, decision)

    metrics = evaluation.compute_metrics(y_test_binary, test_score, y_pred_test)
    latency = _measure_latency_ms_per_1k(trainer, val_texts, tokenizer, max_length)
    vram_peak = _peak_vram_mb()
    finished_at = runs.utc_now_iso()

    run_dir = runs.new_run_dir("gen2", spec.slug, variant)
    runs.write_predictions(
        run_dir,
        indices=splits["test"].index.tolist(),
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
            "generation": "gen2",
            "model": spec.slug,
            "variant": variant,
            "config": {
                "seed": SEED,
                "hf_id": spec.hf_id,
                "bucket": spec.bucket,
                "max_length": max_length,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": effective_batch,
                "grad_accum": effective_grad_accum,
                "fp16": fp16,
                "preprocess": "raw",
                "mask_entities": False,
                "target": target_block,
            },
            "split": split_meta_block,
            "threshold": {
                "fitted_on": decision.fitted_on,
                "value": decision.value,
                "objective": decision.objective,
            },
            "metrics": metrics,
            "efficiency": {
                "latency_ms_per_1k": latency,
                "latency_includes_tokenization": True,
                "vram_peak_mb": vram_peak,
            },
        },
    )
    return run_dir
