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
- **Sem threshold.** Gen 3 grava `threshold.applicable = False` — o recorte
  é fixo em 0.5 (ou pelo hard label quando não há logprob). A decisão é
  reportada como "zero-shot cru" no artigo; a comparação com Gen 1/Gen 2
  usa PR-AUC (invariante a threshold) e F1-minority sobre o recorte padrão.
- **Latência wall-clock por requisição**, agregada para `ms/1k`. VRAM pico
  via amostragem `nvidia-smi` em thread daemon — o processo Ollama fica em
  PID separado, então `torch.cuda.*` não enxerga esse consumo.
- **Resumabilidade append-only.** `predictions.csv` é aberto em modo
  append com `flush + os.fsync` por linha; reabrir o `run_dir` via
  `--resume-run-id` pula os `index` já processados. Metadata só é gravado
  no final (ou em `--dry-run`); runs incompletos deixam `predictions.csv`
  legível mas sem `metadata.json` → fácil identificar e retomar.
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

from ptbr_market import evaluation, preprocessing, runs, targets

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

    `repo_id` + `filename` endereçam um snapshot único. Modelos nativos do
    Ollama (Llama 3.1, Qwen 2.5, Gemma 2) não precisam disso — vêm da
    registry oficial.
    """

    repo_id: str
    filename: str
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
            repo_id="tensorblock/Tucano-2b4-Instruct-GGUF",
            filename="Tucano-2b4-Instruct-Q4_K_M.gguf",
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


def _test_texts_for_prompt(split: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Retorna (titles, texts) pré-normalizados NFC + strip, sem máscara."""
    titles = preprocessing.preprocess_raw(
        split["title"].tolist() if "title" in split.columns else ["" for _ in range(len(split))]
    )
    texts = preprocessing.preprocess_raw(split["text"].tolist())
    return titles, texts


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
    """Avalia um LLM zero-shot sobre `splits['test']` e persiste o run Gen 3.

    Retorna o `run_dir` criado em `artifacts/runs/{stamp}__gen3__{slug}__{variant}/`
    (ou `resume_run_dir` se fornecido).

    Em `target_mode="binary"` o prompt limita a saída a `mercado|outros`;
    em `"multiclass"` limita à lista do `collapse_scheme` (atualmente
    sempre `top7_plus_other` → 8 rótulos). Em ambos os casos, a avaliação
    continua binária: `y_true` vem do rótulo binário do split, e o
    `y_pred` derivado do LLM é 1 sse a classe escolhida for `mercado`.

    `limit` é só para smoke-test — produz um run incompleto se < len(test).

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

    test = splits["test"]
    titles, texts = _test_texts_for_prompt(test)
    y_true = test["label"].astype(int).tolist()
    test_indices = test.index.tolist()

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

    preds_path = run_dir / "predictions.csv"
    already_done = _read_processed_indices(preds_path)

    # Carrega o que já foi gravado para poder agregar métricas no final sem
    # reler o CSV duas vezes. Cada entrada: (index, y_true, y_pred, y_score).
    accumulated_rows: list[tuple[int, int, int, float]] = []
    if already_done:
        with preds_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    accumulated_rows.append(
                        (
                            int(row["index"]),
                            int(row["y_true"]),
                            int(row["y_pred"]),
                            float(row["y_score"]),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue

    if client is None:
        client = OllamaClient(
            model_tag=spec.ollama_model_tag,
            host=ollama_host,
            num_ctx=effective_num_ctx,
            num_predict=num_predict,
            top_logprobs=top_logprobs,
            timeout_s=timeout_s,
        )

    stats = _RunStats(n_total=n_test, n_skipped_resume=len(already_done))

    file_mode = "a" if preds_path.exists() else "w"
    with VRAMSampler() as vram:
        with preds_path.open(file_mode, encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            if file_mode == "w":
                writer.writerow(list(_PRED_HEADER))
                fp.flush()
                os.fsync(fp.fileno())

            for i in range(n_test):
                idx = test_indices[i]
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
                row = (int(idx), int(y_true[i]), int(result.y_pred), float(result.y_score))
                _append_prediction_row(fp, writer, row)
                accumulated_rows.append(row)
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
                        f"[gen3] {spec.slug} {variant}: "
                        f"{stats.n_processed}/{n_test - stats.n_skipped_resume} "
                        f"(avg {avg:.2f}s/req, VRAM peak {vram.peak_mb:.0f} MB)",
                        flush=True,
                    )

        vram_peak_mb = vram.peak_mb

    finished_at = runs.utc_now_iso()

    # Agrega predictions e calcula métricas/efficiency sobre tudo que está
    # gravado (inclui o que vinha do resume).
    accumulated_rows.sort(key=lambda r: r[0])
    y_true_col = [r[1] for r in accumulated_rows]
    y_pred_col = [r[2] for r in accumulated_rows]
    y_score_col = [r[3] for r in accumulated_rows]

    if not accumulated_rows:
        raise RuntimeError(
            "Nenhuma predição foi gerada; nada a avaliar. Verifique conectividade"
            " com Ollama ou o valor de `limit`."
        )

    metrics = evaluation.compute_metrics(y_true_col, y_score_col, y_pred_col)

    total_latency_s = sum(stats.latencies_s)
    latency_ms_per_1k = (
        (total_latency_s * 1000.0 * 1000.0) / max(stats.n_processed, 1)
        if stats.n_processed > 0
        else 0.0
    )

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
                    "method_counts": stats.method_counts,
                    "label_counts": stats.label_counts,
                },
                "resume": {
                    "resumed": resume_run_dir is not None,
                    "n_skipped": stats.n_skipped_resume,
                    "n_processed_this_run": stats.n_processed,
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
            # Gen 3 é zero-shot cru: não há calibração em val. O cutoff fica
            # fixo em 0.5 (ou no hard label) e aparece aqui apenas para
            # manter o schema. Comparações com Gen 1/Gen 2 usam PR-AUC
            # (invariante) e F1-minority sobre esse recorte.
            "threshold": {
                "applicable": False,
                "fitted_on": None,
                "value": 0.5,
                "objective": None,
            },
            "metrics": metrics,
            "efficiency": {
                "latency_ms_per_1k": latency_ms_per_1k,
                "latency_includes_tokenization": True,
                "vram_peak_mb": vram_peak_mb,
                "n_processed": stats.n_processed,
                "n_total": stats.n_total,
            },
        },
    )
    return run_dir
