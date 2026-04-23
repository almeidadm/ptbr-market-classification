"""Zero-shot de LLMs servidos via Ollama (Geração 3).

Pipeline único e resumível para avaliar LLMs como classificadores zero-shot
sobre os splits OoT do projeto, mantendo o contrato de artefatos
(`metadata.json`, `metrics.json`, `predictions.csv`) para McNemar pareado
contra Gen 1/Gen 2 sem mexer em `scripts/*report.py`.

Convenções (decisões locked 2026-04-22):

- **Pré-processamento `raw`** (igual Gen 2): sem máscara de entidades; o
  tokenizer do LLM já lida com `R$ 10 bi` e a semântica importa na
  resposta. Título + texto são concatenados no prompt via placeholders.
- **Prompt fechado, zero-shot, single-word output** (ver
  `prompts/gen3_v1_{bin,mc8}.txt`). Uma única versão `v1` por variante;
  `compute_prompt_hash` grava SHA-256 no `metadata.config` para auditoria.
- **Scoring via logprobs** com fallback por hard label: `/v1/chat/completions`
  OpenAI-compat com `logprobs: true, top_logprobs: 20`. `y_score = softmax`
  sobre os logprobs dos tokens das classes alvo; quando nenhum token-alvo
  aparece no top-k, cai para `y_score ∈ {0.0, 1.0}` a partir do hard label
  parseado. A contagem de quedas fica em `config.scoring.method_counts`.
- **Threshold calibrado em val, congelado em test** (mesma mecânica de
  Gen 1/Gen 2). Gen 3 roda inferência sobre `splits["val"]` antes do test
  para obter `y_score_val`; `threshold.fit_threshold` varre a grade
  `[0,05, 0,95]` passo 0,01 com objetivo `f1_minority` e o valor
  escolhido é aplicado ao `y_score_test` via `apply_threshold` para
  materializar `y_pred`. `metadata.threshold.applicable = True`,
  `fitted_on = "val"`.
- **Latência wall-clock medida sobre val**, agregada para `ms/1k` —
  alinhada com Gen 1/Gen 2 (a passada de val já é obrigatória para o
  fit do threshold, então medir lá é gratuito). VRAM pico via
  amostragem `nvidia-smi` em thread daemon — o processo Ollama fica em
  PID separado, então `torch.cuda.*` não enxerga esse consumo.
- **Resumabilidade append-only.** `predictions_val.csv` (val) e
  `predictions.csv` (test) são abertos em modo append com
  `flush + os.fsync` por linha; reabrir o `run_dir` via
  `--resume-run-id` pula os `index` já processados em cada split
  independentemente. Metadata só é gravado no final; runs incompletos
  deixam os CSVs legíveis mas sem `metadata.json` → fácil identificar
  e retomar.
- **`seed=1`** onde faz sentido (ordem do dataset, `temperature=0.0` na
  API). A decodificação determinística do Ollama não é bit-exact entre
  versões do servidor — `config.ollama_model_tag` fixa a tag para auditoria.

Só faz sentido com os extras `gen3` instalados
(`uv sync --group gen3` — `requests`, `huggingface-hub`). O Ollama
em si é um binário externo — quem lida com o setup é
`notebooks/colab/bootstrap_gen3.ipynb`.
"""

from __future__ import annotations

import csv
import hashlib
import os
import random
import re
import subprocess
import threading
import time
import unicodedata
from collections.abc import Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests

from ptbr_market import evaluation, preprocessing, runs, targets, threshold as thr_mod

SEED = 1
DEFAULT_NUM_CTX = 2048
DEFAULT_NUM_PREDICT = 1
DEFAULT_TOP_LOGPROBS = 20
DEFAULT_TEXT_MAX_CHARS = 4000
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_VRAM_SAMPLE_INTERVAL_S = 1.0

OLLAMA_DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

PROMPT_VERSION = "v1"

# Classes aceitas por variante. A ordem fixa o tie-break do parser para
# prefix-match (primeiro rótulo que "bate" é o escolhido).
_BIN_LABELS: tuple[str, ...] = ("mercado", "outros")
_MC8_LABELS: tuple[str, ...] = (
    "mercado",
    "poder",
    "colunas",
    "esporte",
    "mundo",
    "cotidiano",
    "ilustrada",
    "outros",
)


@dataclass(frozen=True, slots=True)
class HFGGUFSource:
    """Modelo GGUF hospedado no HuggingFace para importação no Ollama.

    Dois modos:

    - `convert_from_safetensors=False` (default): `repo_id` aponta para um
      repositório que já hospeda `.gguf` prontos. `filename` é o nome do
      arquivo a baixar via `hf_hub_download`.
    - `convert_from_safetensors=True`: `repo_id` aponta para o repositório
      FP16 oficial (formato HuggingFace/safetensors). `filename` é o nome
      de saída esperado **após conversão local** para Q4_K_M GGUF via
      `llama.cpp` (`convert_hf_to_gguf.py` + `llama-quantize`).

    Modelos nativos do Ollama (Llama 3.1, Qwen 2.5, Gemma 2) não precisam
    disso — vêm da registry oficial.
    """

    repo_id: str
    filename: str
    convert_from_safetensors: bool = False
    modelfile_template: str = (
        'FROM ./{filename}\nPARAMETER temperature 0\nPARAMETER stop "\\n"\n'
    )


@dataclass(frozen=True, slots=True)
class Gen3ModelSpec:
    """Parâmetros de um LLM na matriz Gen 3.

    `ollama_model_tag` é a tag que o `ollama pull`/`ollama create` deixa
    no daemon e que o cliente HTTP consome em `model`. `num_ctx` pode ser
    inflado para modelos com janela maior (ex.: Qwen 2.5 aceita 32k).
    `bucket` alinha com o agrupamento do `plano_base.md`.
    """

    slug: str
    ollama_model_tag: str
    bucket: str  # "global-native" | "ptbr-gguf"
    num_ctx: int = DEFAULT_NUM_CTX
    gguf_source: HFGGUFSource | None = None


GEN3_MODELS: dict[str, Gen3ModelSpec] = {
    "llama3.1-8b": Gen3ModelSpec(
        slug="llama3.1-8b",
        ollama_model_tag="llama3.1:8b-instruct-q4_K_M",
        bucket="global-native",
    ),
    "qwen2.5-7b": Gen3ModelSpec(
        slug="qwen2.5-7b",
        ollama_model_tag="qwen2.5:7b-instruct-q4_K_M",
        bucket="global-native",
    ),
    "qwen2.5-14b": Gen3ModelSpec(
        slug="qwen2.5-14b",
        ollama_model_tag="qwen2.5:14b-instruct-q4_K_M",
        bucket="global-native",
    ),
    "gemma2-9b": Gen3ModelSpec(
        slug="gemma2-9b",
        ollama_model_tag="gemma2:9b-instruct-q4_K_M",
        bucket="global-native",
    ),
    "bode-7b": Gen3ModelSpec(
        slug="bode-7b",
        ollama_model_tag="bode-7b-alpaca:q4_k_m",
        bucket="ptbr-gguf",
        gguf_source=HFGGUFSource(
            repo_id="recogna-nlp/bode-7b-alpaca-pt-br-gguf",
            filename="bode-7b-alpaca-q4_k_m.gguf",
        ),
    ),
    "tucano-2b4": Gen3ModelSpec(
        slug="tucano-2b4",
        ollama_model_tag="tucano-2b4-instruct:q4_k_m",
        bucket="ptbr-gguf",
        gguf_source=HFGGUFSource(
            repo_id="TucanoBR/Tucano-2b4-Instruct",
            filename="tucano-2b4-instruct-q4_k_m.gguf",
            convert_from_safetensors=True,
        ),
    ),
}


# --- Prompts e parsing -----------------------------------------------------


def _prompts_dir() -> Path:
    return Path(
        os.environ.get(
            "PTBR_PROMPTS_ROOT",
            str(Path(__file__).resolve().parent.parent.parent / "prompts"),
        )
    )


def load_prompt(
    target_tag: Literal["bin", "mc8"],
    version: str = PROMPT_VERSION,
) -> tuple[str, str]:
    """Lê `prompts/gen3_{version}_{target_tag}.txt` e retorna `(system, user_template)`.

    O arquivo usa marcadores `[SYSTEM]` e `[USER]` em linhas próprias. O
    template do usuário contém os placeholders `{title}` e `{text}`.
    """
    path = _prompts_dir() / f"gen3_{version}_{target_tag}.txt"
    raw = path.read_text(encoding="utf-8")
    if "[SYSTEM]" not in raw or "[USER]" not in raw:
        raise ValueError(
            f"Prompt {path} precisa conter marcadores [SYSTEM] e [USER]."
        )
    system_part, user_part = raw.split("[USER]", 1)
    system_part = system_part.replace("[SYSTEM]", "", 1).strip()
    user_template = user_part.strip()
    if "{title}" not in user_template or "{text}" not in user_template:
        raise ValueError(
            f"Prompt {path}: template do usuário precisa de {{title}} e {{text}}."
        )
    return system_part, user_template


def compute_prompt_hash(system: str, user_template: str) -> str:
    """SHA-256 (primeiros 16 chars) do par `(system, template)` — vai no metadata."""
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(b"\0")
    h.update(user_template.encode("utf-8"))
    return h.hexdigest()[:16]


def render_user_prompt(
    template: str,
    title: str | None,
    text: str | None,
    max_chars: int = DEFAULT_TEXT_MAX_CHARS,
) -> str:
    """Preenche o template com título + texto truncado no começo (head).

    Truncar o head preserva o lead da matéria — que concentra o sinal em
    notícias (lide). Unicode NFC para bater com `preprocess_raw`.
    """
    t_title = unicodedata.normalize("NFC", "" if title is None else str(title)).strip()
    t_text = unicodedata.normalize("NFC", "" if text is None else str(text)).strip()
    if max_chars > 0 and len(t_text) > max_chars:
        t_text = t_text[:max_chars]
    return template.format(title=t_title, text=t_text)


_LABEL_NORMALIZE_RE = re.compile(r"[^a-zà-ÿ]", re.IGNORECASE)


def _normalize_label_text(s: str) -> str:
    nfc = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return _LABEL_NORMALIZE_RE.sub("", nfc).lower()


def parse_response(
    raw_text: str,
    allowed_labels: Iterable[str],
    positive_label: str = "mercado",
) -> tuple[int, str | None]:
    """Converte a resposta bruta do LLM em `(y_pred_binario, matched_label)`.

    Estratégia: primeiro token "palavra" da resposta, normalizado (sem
    acentos, minúsculas, só letras), comparado aos rótulos permitidos. Se
    nenhum rótulo bate, retorna `(0, None)` — conta como erro de parsing
    e vira negativo para não inflar recall.

    `y_pred` é 1 apenas quando o rótulo parseado é o `positive_label`.
    """
    allowed = tuple(allowed_labels)
    stripped = raw_text.strip()
    if not stripped:
        return 0, None
    first = stripped.split()[0]
    norm = _normalize_label_text(first)
    if not norm:
        return 0, None
    pos_norm = _normalize_label_text(positive_label)
    for label in allowed:
        if norm.startswith(_normalize_label_text(label)):
            y = 1 if label == positive_label else 0
            return y, label
    # Fallback: se a resposta inteira contém um dos rótulos como substring,
    # aceita (cobre respostas tipo "Classe: mercado").
    whole_norm = _normalize_label_text(stripped)
    for label in allowed:
        key = _normalize_label_text(label)
        if key and key in whole_norm:
            y = 1 if label == positive_label else 0
            return y, label
    if pos_norm and pos_norm in whole_norm:
        return 1, positive_label
    return 0, None


def extract_score_from_logprobs(
    top_logprobs: list[dict[str, Any]] | None,
    positive_label: str,
    negative_labels: Iterable[str],
) -> float | None:
    """Calcula `P(positivo) ≈ exp(lp_pos) / Σ exp(lp_i)` a partir do top-k.

    `top_logprobs` tem o formato OpenAI-compat do Ollama:
    `[{"token": "mercado", "logprob": -0.12}, {"token": "outros", "logprob": -2.1}, ...]`
    — para o primeiro token gerado. Quando nenhum token do conjunto-alvo
    aparece, retorna `None` (o chamador cai no hard label).
    """
    if not top_logprobs:
        return None
    positive_norm = _normalize_label_text(positive_label)
    negative_norms = tuple(_normalize_label_text(n) for n in negative_labels if n)
    all_norms = (positive_norm, *negative_norms)

    best_lps: dict[str, float] = {}
    for entry in top_logprobs:
        tok = entry.get("token") or entry.get("bytes") or ""
        lp = entry.get("logprob")
        if lp is None:
            continue
        tok_norm = _normalize_label_text(str(tok))
        if not tok_norm:
            continue
        for key in all_norms:
            if not key:
                continue
            if tok_norm.startswith(key) or key.startswith(tok_norm):
                # Mantém o maior logprob para esse rótulo (primeira ocorrência
                # já é o maior, porque Ollama devolve top-k ordenado, mas
                # guardamos defensivamente).
                current = best_lps.get(key)
                if current is None or lp > current:
                    best_lps[key] = float(lp)
                break

    if positive_norm not in best_lps:
        return None

    lps = np.array(list(best_lps.values()), dtype=np.float64)
    lps -= lps.max()  # estabilização numérica
    exps = np.exp(lps)
    denom = float(exps.sum())
    if denom <= 0:
        return None
    keys = list(best_lps.keys())
    idx = keys.index(positive_norm)
    return float(exps[idx] / denom)


# --- Cliente Ollama --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClassifyResult:
    """Saída por amostra do cliente. `score_source` é `logprobs` ou `hard`."""

    y_pred: int
    y_score: float
    matched_label: str | None
    score_source: Literal["logprobs", "hard", "parse_failure"]
    raw_text: str
    latency_s: float


class OllamaClient:
    """Cliente fino sobre o endpoint OpenAI-compat do Ollama.

    Usa `/v1/chat/completions` com `logprobs` — a API nativa
    (`/api/chat`) também tem logprobs, mas o shape é menos estável
    entre versões do servidor.
    """

    def __init__(
        self,
        model_tag: str,
        host: str = OLLAMA_DEFAULT_HOST,
        num_ctx: int = DEFAULT_NUM_CTX,
        num_predict: int = DEFAULT_NUM_PREDICT,
        top_logprobs: int = DEFAULT_TOP_LOGPROBS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.model_tag = model_tag
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.top_logprobs = top_logprobs
        self.timeout_s = timeout_s
        self._endpoint = f"{self.host}/v1/chat/completions"

    def _payload(self, system: str, user: str) -> dict[str, Any]:
        return {
            "model": self.model_tag,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": self.num_predict,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            # Campo nativo do Ollama aceito pelo endpoint OpenAI-compat.
            # Quando ignorado, o servidor usa o default do modelfile (2048).
            "options": {"num_ctx": self.num_ctx, "seed": SEED},
        }

    def warmup(self, timeout_s: float = 300.0) -> float:
        """Força o load do modelo na VRAM via uma chamada minimal.

        A primeira inferência paga ~30-90 s (load dos pesos no T4 para
        modelos 7B-8B Q4_K_M; mais para Qwen 14B). O `timeout_s` padrão de
        `classify_one` (60 s) é insuficiente para esse cold-start, então o
        warmup roda separado com timeout grande, antes do loop de avaliação,
        e fora da contagem de `latencies_s`. Retorna o tempo de warmup em
        segundos para log/diagnóstico.
        """
        payload = self._payload("ok", "ok")
        payload["max_tokens"] = 1
        t0 = time.perf_counter()
        requests.post(self._endpoint, json=payload, timeout=timeout_s).raise_for_status()
        return time.perf_counter() - t0

    def classify_one(
        self,
        system: str,
        user: str,
        allowed_labels: Iterable[str],
        positive_label: str = "mercado",
    ) -> ClassifyResult:
        """Uma chamada → uma classificação. Não levanta em erro do servidor:
        propaga `HTTPError` para o chamador, que decide se retenta."""
        allowed = tuple(allowed_labels)
        negatives = tuple(lbl for lbl in allowed if lbl != positive_label)

        t0 = time.perf_counter()
        resp = requests.post(
            self._endpoint, json=self._payload(system, user), timeout=self.timeout_s
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.perf_counter() - t0

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        raw_text = str(message.get("content") or "")

        top_lps: list[dict[str, Any]] | None = None
        lp_block = choice.get("logprobs") or {}
        content_block = lp_block.get("content") or []
        if content_block:
            top_lps = content_block[0].get("top_logprobs") or []

        y_pred, matched = parse_response(raw_text, allowed, positive_label)

        score = extract_score_from_logprobs(top_lps, positive_label, negatives)
        if score is not None:
            source: Literal["logprobs", "hard", "parse_failure"] = "logprobs"
            y_score = score
        elif matched is not None:
            source = "hard"
            y_score = float(y_pred)
        else:
            source = "parse_failure"
            y_score = 0.0

        return ClassifyResult(
            y_pred=y_pred,
            y_score=y_score,
            matched_label=matched,
            score_source=source,
            raw_text=raw_text,
            latency_s=elapsed,
        )


# --- VRAM sampler ---------------------------------------------------------


class VRAMSampler(AbstractContextManager[Any]):
    """Amostra o VRAM usado na GPU 0 via `nvidia-smi` em thread daemon.

    Por que não `torch.cuda.max_memory_allocated()`: o Ollama serve o modelo
    em outro processo; `torch` no processo-cliente vê ~0 MB. `nvidia-smi`
    lê do driver e captura o consumo real do daemon.

    Quando `nvidia-smi` não está disponível (ex.: rodando em CPU), retorna
    `peak_mb = 0.0` sem falhar.
    """

    def __init__(self, interval_s: float = DEFAULT_VRAM_SAMPLE_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self.peak_mb: float = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._available = self._probe()

    @staticmethod
    def _probe() -> bool:
        try:
            subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                check=True,
                timeout=2,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
        return True

    def _sample_once(self) -> float | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
        line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        try:
            return float(line)
        except ValueError:
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = self._sample_once()
            if v is not None and v > self.peak_mb:
                self.peak_mb = v
            self._stop.wait(self.interval_s)

    def __enter__(self) -> VRAMSampler:
        if self._available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 1.0)


# --- Runner resumível -----------------------------------------------------


_PRED_HEADER: tuple[str, ...] = ("index", "y_true", "y_pred", "y_score")
_VAL_PRED_FILENAME = "predictions_val.csv"
_TEST_PRED_FILENAME = "predictions.csv"


def _read_processed_indices(path: Path) -> set[int]:
    """Lê `predictions.csv` existente e devolve os `index` já processados."""
    if not path.exists():
        return set()
    done: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add(int(row["index"]))
            except (KeyError, TypeError, ValueError):
                continue
    return done


def _append_prediction_row(
    fp: Any, writer: csv.writer, row: tuple[int, int, int, float]
) -> None:
    writer.writerow(row)
    fp.flush()
    os.fsync(fp.fileno())


@dataclass(slots=True)
class _RunStats:
    n_total: int = 0
    n_processed: int = 0
    n_skipped_resume: int = 0
    latencies_s: list[float] = field(default_factory=list)
    method_counts: dict[str, int] = field(
        default_factory=lambda: {"logprobs": 0, "hard": 0, "parse_failure": 0}
    )
    label_counts: dict[str, int] = field(default_factory=dict)


def _texts_for_prompt(split: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Retorna (titles, texts) pré-normalizados NFC + strip, sem máscara."""
    titles = preprocessing.preprocess_raw(
        split["title"].tolist() if "title" in split.columns else ["" for _ in range(len(split))]
    )
    texts = preprocessing.preprocess_raw(split["text"].tolist())
    return titles, texts


# Alias mantido por compatibilidade com código que já importava o nome antigo.
_test_texts_for_prompt = _texts_for_prompt


def _load_accumulated_rows(path: Path) -> list[tuple[int, int, int, float]]:
    """Lê um `predictions*.csv` existente e devolve as linhas como tuplas."""
    rows: list[tuple[int, int, int, float]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    (
                        int(row["index"]),
                        int(row["y_true"]),
                        int(row["y_pred"]),
                        float(row["y_score"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _process_split(
    *,
    client: Any,
    preds_path: Path,
    titles: list[str],
    texts: list[str],
    indices: list[Any],
    y_true_col: list[int],
    n_items: int,
    already_done: set[int],
    stats: "_RunStats",
    system_prompt: str,
    user_template: str,
    text_max_chars: int,
    allowed_labels: tuple[str, ...],
    positive_label: str,
    vram: "VRAMSampler",
    progress_every: int,
    split_label: str,
    spec_slug: str,
    variant: str,
    decision: "thr_mod.ThresholdDecision | None" = None,
) -> list[tuple[int, int, int, float]]:
    """Roda o cliente LLM sobre um split, grava predições append-only e devolve
    todas as linhas (incluindo as reaproveitadas de um run anterior).

    Se `decision` for não-None, `y_pred` gravado é derivado de
    `apply_threshold(y_score, decision)`; caso contrário, é a saída
    hard-label do parser (usado na passada de val, antes de saber a
    threshold).
    """
    accumulated = _load_accumulated_rows(preds_path)

    file_mode = "a" if preds_path.exists() else "w"
    with preds_path.open(file_mode, encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        if file_mode == "w":
            writer.writerow(list(_PRED_HEADER))
            fp.flush()
            os.fsync(fp.fileno())

        for i in range(n_items):
            idx = indices[i]
            if idx in already_done:
                continue
            title_i = titles[i]
            text_i = texts[i]
            user_prompt = render_user_prompt(
                user_template, title_i, text_i, text_max_chars
            )
            result = client.classify_one(
                system=system_prompt,
                user=user_prompt,
                allowed_labels=allowed_labels,
                positive_label=positive_label,
            )
            if decision is not None:
                y_pred = int(1 if float(result.y_score) >= decision.value else 0)
            else:
                y_pred = int(result.y_pred)
            row = (int(idx), int(y_true_col[i]), y_pred, float(result.y_score))
            _append_prediction_row(fp, writer, row)
            accumulated.append(row)
            stats.n_processed += 1
            stats.latencies_s.append(result.latency_s)
            stats.method_counts[result.score_source] += 1
            if result.matched_label is not None:
                stats.label_counts[result.matched_label] = (
                    stats.label_counts.get(result.matched_label, 0) + 1
                )
            if progress_every and stats.n_processed % progress_every == 0:
                elapsed = sum(stats.latencies_s)
                avg = elapsed / max(stats.n_processed, 1)
                print(
                    f"[gen3] {spec_slug} {variant} {split_label}: "
                    f"{stats.n_processed}/{n_items - len(already_done)} "
                    f"(avg {avg:.2f}s/req, VRAM peak {vram.peak_mb:.0f} MB)",
                    flush=True,
                )
    return accumulated


def run_gen3_experiment(
    splits: dict[str, pd.DataFrame],
    split_meta_block: dict[str, Any],
    model_slug: str,
    variant: str | None = None,
    target_mode: Literal["binary", "multiclass"] = "binary",
    collapse_scheme: str | None = None,
    prompt_version: str = PROMPT_VERSION,
    text_max_chars: int = DEFAULT_TEXT_MAX_CHARS,
    ollama_host: str = OLLAMA_DEFAULT_HOST,
    num_ctx: int | None = None,
    num_predict: int = DEFAULT_NUM_PREDICT,
    top_logprobs: int = DEFAULT_TOP_LOGPROBS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    resume_run_dir: Path | None = None,
    limit: int | None = None,
    progress_every: int = 200,
    client: OllamaClient | None = None,
) -> Path:
    """Avalia um LLM zero-shot sobre `splits['val']` + `splits['test']` e
    persiste o run Gen 3.

    Retorna o `run_dir` criado em `artifacts/runs/{stamp}__gen3__{slug}__{variant}/`
    (ou `resume_run_dir` se fornecido).

    Fluxo:

    1. Inferência completa sobre `splits["val"]` → grava
       `predictions_val.csv` (append-only, resumível).
    2. `threshold.fit_threshold` sobre `(y_val_true, y_score_val)` com
       objetivo `f1_minority` → `ThresholdDecision` congelada.
    3. Inferência sobre `splits["test"]` → grava `predictions.csv` com
       `y_pred = apply_threshold(y_score, decision)`.
    4. Métricas (PR-AUC, F1-min, etc.) computadas só no test; latência
       agregada sobre as requisições de val (alinhada com Gen 1/Gen 2,
       que também medem na passada de val).

    Em `target_mode="binary"` o prompt limita a saída a `mercado|outros`;
    em `"multiclass"` limita à lista do `collapse_scheme` (atualmente
    sempre `top7_plus_other` → 8 rótulos). Em ambos os casos, a avaliação
    continua binária: `y_true` vem do rótulo binário do split, e o
    `y_score` do LLM é renormalizado entre os tokens-alvo (PA-004).

    `limit` é só para smoke-test — afeta só a fase de test (val é sempre
    processado integralmente, caso contrário o fit de threshold não faz
    sentido). Produz um run incompleto se `limit < len(test)`.

    `client` injetável para mock em testes; quando `None`, instancia um
    `OllamaClient` padrão sobre `ollama_host` e a tag do spec.
    """
    if model_slug not in GEN3_MODELS:
        raise ValueError(
            f"Modelo Gen 3 desconhecido: {model_slug!r}. Use um de"
            f" {tuple(GEN3_MODELS)}."
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

    spec = GEN3_MODELS[model_slug]
    effective_num_ctx = num_ctx or spec.num_ctx
    if target_mode == "binary":
        target_tag: Literal["bin", "mc8"] = "bin"
    else:
        target_tag = targets.target_variant_tag(  # type: ignore[assignment]
            target_mode, collapse_scheme
        )
    variant = variant or f"zs-{prompt_version}-{target_tag}"

    random.seed(SEED)
    np.random.seed(SEED)

    system_prompt, user_template = load_prompt(target_tag, prompt_version)
    prompt_hash = compute_prompt_hash(system_prompt, user_template)

    if target_mode == "binary":
        allowed_labels = _BIN_LABELS
    else:
        allowed_labels = _MC8_LABELS
    positive_label = targets.POSITIVE_CATEGORY_LABEL

    val = splits["val"]
    test = splits["test"]
    val_titles, val_texts = _texts_for_prompt(val)
    test_titles, test_texts = _texts_for_prompt(test)
    y_val_true = val["label"].astype(int).tolist()
    y_test_true = test["label"].astype(int).tolist()
    val_indices = val.index.tolist()
    test_indices = test.index.tolist()

    # `limit` é smoke-test: afeta só a fase de test. Val é sempre
    # processado integralmente para garantir o fit do threshold.
    n_val = len(val)
    n_test = len(test)
    if limit is not None:
        n_test = min(n_test, int(limit))

    if resume_run_dir is not None:
        run_dir = Path(resume_run_dir)
        if not run_dir.is_dir():
            raise ValueError(f"resume_run_dir não existe: {run_dir!r}")
        started_at = runs.utc_now_iso()
    else:
        run_dir = runs.new_run_dir("gen3", spec.slug, variant)
        started_at = runs.utc_now_iso()

    val_preds_path = run_dir / _VAL_PRED_FILENAME
    test_preds_path = run_dir / _TEST_PRED_FILENAME
    val_already_done = _read_processed_indices(val_preds_path)
    test_already_done = _read_processed_indices(test_preds_path)

    if client is None:
        client = OllamaClient(
            model_tag=spec.ollama_model_tag,
            host=ollama_host,
            num_ctx=effective_num_ctx,
            num_predict=num_predict,
            top_logprobs=top_logprobs,
            timeout_s=timeout_s,
        )

    if hasattr(client, "warmup"):
        print(
            f"  [warmup] carregando {spec.ollama_model_tag} na VRAM"
            " (até 300s, fora da contagem de latência)...",
            flush=True,
        )
        warm_s = client.warmup()
        print(f"  [warmup] OK em {warm_s:.1f}s", flush=True)

    stats_val = _RunStats(n_total=n_val, n_skipped_resume=len(val_already_done))
    stats_test = _RunStats(n_total=n_test, n_skipped_resume=len(test_already_done))

    with VRAMSampler() as vram:
        # Passada 1 — val. y_pred gravado é a hard-label do parser (a
        # threshold ainda não existe); o fit a seguir consome só y_score.
        val_rows = _process_split(
            client=client,
            preds_path=val_preds_path,
            titles=val_titles,
            texts=val_texts,
            indices=val_indices,
            y_true_col=y_val_true,
            n_items=n_val,
            already_done=val_already_done,
            stats=stats_val,
            system_prompt=system_prompt,
            user_template=user_template,
            text_max_chars=text_max_chars,
            allowed_labels=allowed_labels,
            positive_label=positive_label,
            vram=vram,
            progress_every=progress_every,
            split_label="val",
            spec_slug=spec.slug,
            variant=variant,
            decision=None,
        )

        if not val_rows:
            raise RuntimeError(
                "Nenhuma predição de val foi gerada; impossível calibrar"
                " threshold. Verifique conectividade com Ollama."
            )

        val_rows_sorted = sorted(val_rows, key=lambda r: r[0])
        y_val_true_sorted = [r[1] for r in val_rows_sorted]
        y_val_score = [r[3] for r in val_rows_sorted]
        decision = thr_mod.fit_threshold(
            y_val_true_sorted,
            y_val_score,
            objective="f1_minority",
        )
        print(
            f"[gen3] threshold calibrado em val: value={decision.value:.2f}"
            f" objective={decision.objective}",
            flush=True,
        )

        # Passada 2 — test. y_pred gravado já usa a threshold congelada.
        test_rows = _process_split(
            client=client,
            preds_path=test_preds_path,
            titles=test_titles,
            texts=test_texts,
            indices=test_indices,
            y_true_col=y_test_true,
            n_items=n_test,
            already_done=test_already_done,
            stats=stats_test,
            system_prompt=system_prompt,
            user_template=user_template,
            text_max_chars=text_max_chars,
            allowed_labels=allowed_labels,
            positive_label=positive_label,
            vram=vram,
            progress_every=progress_every,
            split_label="test",
            spec_slug=spec.slug,
            variant=variant,
            decision=decision,
        )
        vram_peak_mb = vram.peak_mb

    finished_at = runs.utc_now_iso()

    if not test_rows:
        raise RuntimeError(
            "Nenhuma predição de test foi gerada; nada a avaliar. Verifique"
            " conectividade com Ollama ou o valor de `limit`."
        )

    test_rows_sorted = sorted(test_rows, key=lambda r: r[0])
    y_test_true_sorted = [r[1] for r in test_rows_sorted]
    y_test_score = [r[3] for r in test_rows_sorted]
    # Reaplica a threshold atual a TODOS os y_score do test — garante
    # consistência se o run resumiu de uma versão antiga do predictions.csv.
    y_test_pred = thr_mod.apply_threshold(y_test_score, decision).tolist()

    metrics = evaluation.compute_metrics(
        y_test_true_sorted, y_test_score, y_test_pred
    )

    # Latência medida sobre val (passada obrigatória e temporalmente
    # adjacente ao test) — alinha com Gen 1/Gen 2, que também medem lá.
    total_val_latency_s = sum(stats_val.latencies_s)
    latency_ms_per_1k = (
        (total_val_latency_s * 1000.0 * 1000.0) / max(stats_val.n_processed, 1)
        if stats_val.n_processed > 0
        else 0.0
    )

    combined_method_counts = {
        k: stats_val.method_counts[k] + stats_test.method_counts[k]
        for k in stats_val.method_counts
    }
    combined_label_counts: dict[str, int] = {}
    for sc in (stats_val, stats_test):
        for k, v in sc.label_counts.items():
            combined_label_counts[k] = combined_label_counts.get(k, 0) + v

    target_block: dict[str, Any]
    if target_mode == "binary":
        target_block = {
            "mode": "binary",
            "num_classes": 2,
            "positive_class_label": targets.POSITIVE_CATEGORY_LABEL,
            "collapse_scheme": None,
            "allowed_labels": list(_BIN_LABELS),
        }
    else:
        target_block = {
            "mode": "multiclass",
            "num_classes": len(_MC8_LABELS),
            "positive_class_label": targets.POSITIVE_CATEGORY_LABEL,
            "collapse_scheme": collapse_scheme,
            "allowed_labels": list(_MC8_LABELS),
        }

    runs.write_metrics(run_dir, metrics)
    runs.write_run_metadata(
        run_dir,
        {
            "run_id": run_dir.name,
            "git_commit": runs.git_commit(),
            "started_at": started_at,
            "finished_at": finished_at,
            "generation": "gen3",
            "model": spec.slug,
            "variant": variant,
            "config": {
                "seed": SEED,
                "ollama_model_tag": spec.ollama_model_tag,
                "ollama_host": ollama_host,
                "bucket": spec.bucket,
                "num_ctx": effective_num_ctx,
                "num_predict": num_predict,
                "top_logprobs": top_logprobs,
                "temperature": 0.0,
                "preprocess": "raw",
                "mask_entities": False,
                "text_max_chars": text_max_chars,
                "prompt": {
                    "version": prompt_version,
                    "target_tag": target_tag,
                    "hash": prompt_hash,
                },
                "scoring": {
                    "method": "logprobs_with_hard_fallback",
                    "method_counts": combined_method_counts,
                    "label_counts": combined_label_counts,
                    "method_counts_val": dict(stats_val.method_counts),
                    "method_counts_test": dict(stats_test.method_counts),
                },
                "resume": {
                    "resumed": resume_run_dir is not None,
                    "n_skipped": stats_test.n_skipped_resume,
                    "n_processed_this_run": stats_test.n_processed,
                    "n_skipped_val": stats_val.n_skipped_resume,
                    "n_processed_val_this_run": stats_val.n_processed,
                },
                "target": target_block,
                "gguf_source": (
                    {
                        "repo_id": spec.gguf_source.repo_id,
                        "filename": spec.gguf_source.filename,
                    }
                    if spec.gguf_source is not None
                    else None
                ),
            },
            "split": split_meta_block,
            # Threshold calibrado sobre `y_score_val` via
            # `ptbr_market.threshold.fit_threshold` (grade [0,05, 0,95]
            # passo 0,01, objetivo f1_minority). Valor congelado e
            # aplicado ao test — mesma mecânica de Gen 1/Gen 2. O campo
            # `applicable` permanece no schema Gen 3 para continuidade
            # com runs legacy; `applicable=True` indica calibração
            # efetiva.
            "threshold": {
                "applicable": True,
                "fitted_on": decision.fitted_on,
                "value": decision.value,
                "objective": decision.objective,
            },
            "metrics": metrics,
            "efficiency": {
                "latency_ms_per_1k": latency_ms_per_1k,
                "latency_measured_on": "val",
                "latency_includes_tokenization": True,
                "vram_peak_mb": vram_peak_mb,
                "n_processed": stats_val.n_processed + stats_test.n_processed,
                "n_total": n_val + n_test,
                "n_processed_val": stats_val.n_processed,
                "n_total_val": n_val,
                "n_processed_test": stats_test.n_processed,
                "n_total_test": n_test,
            },
        },
    )
    return run_dir
