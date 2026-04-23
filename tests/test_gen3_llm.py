"""Testes do pipeline Gen 3 (zero-shot LLM via Ollama).

O cliente real (`OllamaClient`) só é exercitado com mock — o pipeline
pergunta ao cliente via `client.classify_one(...)` e o injetável em
`run_gen3_experiment` permite substituí-lo por um fake determinístico.
Smoke com modelo real fica em Colab via `scripts/run_gen3.py`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptbr_market import gen3_llm, runs, targets
from ptbr_market.gen3_llm import ClassifyResult

# --- Fixtures -------------------------------------------------------------


@pytest.fixture
def tiny_binary_splits() -> dict[str, pd.DataFrame]:
    """Splits mínimos com sinal binário óbvio pelo primeiro token do texto.

    O fake client usa a palavra 'mercado' em títulos positivos para decidir
    — não há chamada HTTP real.
    """
    market_titles = [
        "Bolsa de valores em alta",
        "Petrobras lucra forte",
        "Banco Central corta Selic",
    ]
    market_texts = [
        "A bolsa fechou em alta forte hoje, puxada por ações da Petrobras.",
        "A Petrobras anunciou lucro recorde no trimestre, surpreendendo analistas.",
        "O Banco Central decidiu cortar a taxa Selic na reunião desta quarta.",
    ]
    other_titles = [
        "Time vence no futebol",
        "Filme estreia no cinema",
        "Show lota estádio",
    ]
    other_texts = [
        "O time venceu o jogo no fim de semana por dois a zero no estádio.",
        "O filme estreia nesta quinta-feira nos cinemas de todo o Brasil.",
        "O show da banda lotou o estádio na noite de sábado, com ingressos esgotados.",
    ]

    def make(n_pos: int, n_neg: int, start_idx: int) -> pd.DataFrame:
        # Intercala pos/neg para que qualquer fatiamento inicial (ex.: via
        # --limit para smoke) contenha as duas classes — caso contrário
        # `compute_metrics` (ROC-AUC) quebra em y_true de classe única.
        titles_list: list[str] = []
        texts_list: list[str] = []
        labels_list: list[int] = []
        for i in range(max(n_pos, n_neg)):
            if i < n_pos:
                titles_list.append(market_titles[i])
                texts_list.append(market_texts[i])
                labels_list.append(1)
            if i < n_neg:
                titles_list.append(other_titles[i])
                texts_list.append(other_texts[i])
                labels_list.append(0)
        df = pd.DataFrame(
            {"title": titles_list, "text": texts_list, "label": labels_list}
        )
        df.index = range(start_idx, start_idx + len(df))
        return df

    return {
        "train": make(3, 3, 0),
        "val": make(2, 2, 100),
        "test": make(3, 3, 200),
    }


@pytest.fixture
def split_meta_block() -> dict:
    return {
        "train_window": ["2020-01-01", "2020-01-10"],
        "val_window": ["2020-01-11", "2020-01-16"],
        "test_window": ["2020-01-17", "2020-01-22"],
        "n_train": 6,
        "n_val": 4,
        "n_test": 6,
    }


@pytest.fixture
def isolated_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PTBR_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


class FakeOllamaClient:
    """Cliente fake: responde 'mercado' se o prompt do usuário mencionar
    'bolsa|petrobras|selic', caso contrário 'outros'. Emite logprobs falsos
    para ativar o caminho de scoring por logprobs.
    """

    def __init__(self, force_hard: bool = False, force_parse_failure: bool = False) -> None:
        self.calls = 0
        self.force_hard = force_hard
        self.force_parse_failure = force_parse_failure

    def classify_one(
        self,
        system: str,
        user: str,
        allowed_labels,
        positive_label: str = "mercado",
    ) -> ClassifyResult:
        self.calls += 1
        low = user.lower()
        is_market = any(kw in low for kw in ("bolsa", "petrobras", "selic", "banco central"))

        if self.force_parse_failure:
            return ClassifyResult(
                y_pred=0,
                y_score=0.0,
                matched_label=None,
                score_source="parse_failure",
                raw_text="???",
                latency_s=0.01,
            )

        if is_market:
            raw = "mercado"
            y_pred = 1
            matched = "mercado"
            score = 0.9 if not self.force_hard else 1.0
        else:
            raw = "outros"
            y_pred = 0
            matched = "outros"
            score = 0.1 if not self.force_hard else 0.0
        return ClassifyResult(
            y_pred=y_pred,
            y_score=score,
            matched_label=matched,
            score_source="hard" if self.force_hard else "logprobs",
            raw_text=raw,
            latency_s=0.01,
        )


# --- Registry & constantes ------------------------------------------------


def test_gen3_models_registry_exposes_expected_slugs() -> None:
    expected = {
        "llama3.1-8b",
        "qwen2.5-7b",
        "qwen2.5-14b",
        "gemma2-9b",
        "bode-7b",
        "tucano-2b4",
    }
    assert expected.issubset(set(gen3_llm.GEN3_MODELS))


def test_gen3_ptbr_models_have_gguf_source() -> None:
    for slug in ("bode-7b", "tucano-2b4"):
        spec = gen3_llm.GEN3_MODELS[slug]
        assert spec.bucket == "ptbr-gguf"
        assert spec.gguf_source is not None
        assert "/" in spec.gguf_source.repo_id
        assert spec.gguf_source.filename.lower().endswith(".gguf")


def test_gen3_native_models_have_no_gguf_source() -> None:
    for slug in ("llama3.1-8b", "qwen2.5-7b", "qwen2.5-14b", "gemma2-9b"):
        spec = gen3_llm.GEN3_MODELS[slug]
        assert spec.bucket == "global-native"
        assert spec.gguf_source is None


def test_seed_constant_is_1() -> None:
    assert gen3_llm.SEED == 1


# --- Prompts --------------------------------------------------------------


def test_load_prompt_bin_and_mc8() -> None:
    sys_bin, user_bin = gen3_llm.load_prompt("bin")
    sys_mc8, user_mc8 = gen3_llm.load_prompt("mc8")
    assert "mercado" in sys_bin.lower()
    assert "{title}" in user_bin and "{text}" in user_bin
    assert "{title}" in user_mc8 and "{text}" in user_mc8
    # mc8 menciona as 8 categorias
    for label in gen3_llm._MC8_LABELS:
        assert label in sys_mc8.lower()


def test_compute_prompt_hash_is_stable_and_sensitive() -> None:
    h1 = gen3_llm.compute_prompt_hash("abc", "xyz")
    h2 = gen3_llm.compute_prompt_hash("abc", "xyz")
    h3 = gen3_llm.compute_prompt_hash("abc", "xyz ")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16


def test_render_user_prompt_truncates_text_head() -> None:
    template = "T: {title}\n\nX: {text}"
    long = "a" * 10_000
    out = gen3_llm.render_user_prompt(template, "tit", long, max_chars=100)
    assert "T: tit" in out
    assert out.count("a") == 100


def test_render_user_prompt_handles_none() -> None:
    out = gen3_llm.render_user_prompt("T:{title} X:{text}", None, None)
    assert out == "T: X:"


# --- Parser ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected_pred,expected_label",
    [
        ("mercado", 1, "mercado"),
        ("outros", 0, "outros"),
        ("MERCADO.", 1, "mercado"),
        ("  mercado\n", 1, "mercado"),
        ("Classe: mercado", 1, "mercado"),
        ("Mercádo", 1, "mercado"),  # com acento que não existe
        ("algo", 0, None),
        ("", 0, None),
    ],
)
def test_parse_response_binary(raw: str, expected_pred: int, expected_label: str | None) -> None:
    y, matched = gen3_llm.parse_response(raw, gen3_llm._BIN_LABELS, "mercado")
    assert y == expected_pred
    assert matched == expected_label


def test_parse_response_multiclass_only_mercado_is_positive() -> None:
    for lbl in gen3_llm._MC8_LABELS:
        y, matched = gen3_llm.parse_response(lbl, gen3_llm._MC8_LABELS, "mercado")
        assert matched == lbl
        assert y == (1 if lbl == "mercado" else 0)


# --- Logprobs -------------------------------------------------------------


def test_extract_score_from_logprobs_binary_case() -> None:
    top = [
        {"token": "mercado", "logprob": math.log(0.8)},
        {"token": "outros", "logprob": math.log(0.2)},
        {"token": "foo", "logprob": math.log(0.001)},
    ]
    score = gen3_llm.extract_score_from_logprobs(top, "mercado", ("outros",))
    assert score == pytest.approx(0.8, rel=1e-3)


def test_extract_score_from_logprobs_missing_positive_returns_none() -> None:
    top = [
        {"token": "outros", "logprob": math.log(0.9)},
        {"token": "foo", "logprob": math.log(0.1)},
    ]
    score = gen3_llm.extract_score_from_logprobs(top, "mercado", ("outros",))
    assert score is None


def test_extract_score_from_logprobs_empty_returns_none() -> None:
    assert gen3_llm.extract_score_from_logprobs(None, "mercado", ("outros",)) is None
    assert gen3_llm.extract_score_from_logprobs([], "mercado", ("outros",)) is None


# --- Runner end-to-end com fake client -----------------------------------


def test_run_gen3_experiment_writes_complete_run_dir(
    tiny_binary_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    fake = FakeOllamaClient()
    run_dir = gen3_llm.run_gen3_experiment(
        splits=tiny_binary_splits,
        split_meta_block=split_meta_block,
        model_slug="llama3.1-8b",
        client=fake,
    )
    assert run_dir.is_dir()
    assert run_dir.parent == isolated_artifacts / "runs"
    assert "__gen3__llama3.1-8b__zs-v1-bin" in run_dir.name
    assert (run_dir / "metadata.json").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "predictions.csv").is_file()
    assert (run_dir / "predictions_val.csv").is_file()
    # Agora o pipeline roda val + test (val para calibrar threshold).
    expected_calls = len(tiny_binary_splits["val"]) + len(tiny_binary_splits["test"])
    assert fake.calls == expected_calls


def test_run_gen3_metadata_schema_and_config(
    tiny_binary_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    fake = FakeOllamaClient()
    run_dir = gen3_llm.run_gen3_experiment(
        splits=tiny_binary_splits,
        split_meta_block=split_meta_block,
        model_slug="llama3.1-8b",
        client=fake,
    )
    meta = json.loads((run_dir / "metadata.json").read_text())
    for key in runs.REQUIRED_METADATA_KEYS:
        assert key in meta, f"metadata faltando chave obrigatória {key!r}"
    assert meta["generation"] == "gen3"
    assert meta["model"] == "llama3.1-8b"
    assert meta["variant"] == "zs-v1-bin"
    assert meta["config"]["preprocess"] == "raw"
    assert meta["config"]["mask_entities"] is False
    assert meta["config"]["temperature"] == 0.0
    assert meta["config"]["prompt"]["version"] == "v1"
    assert meta["config"]["prompt"]["target_tag"] == "bin"
    assert len(meta["config"]["prompt"]["hash"]) == 16
    assert meta["config"]["scoring"]["method"] == "logprobs_with_hard_fallback"
    # method_counts combina val + test
    assert meta["config"]["scoring"]["method_counts"]["logprobs"] == fake.calls
    assert meta["config"]["target"]["mode"] == "binary"
    # Threshold calibrado em val (mesma mecânica de Gen 1/Gen 2)
    assert meta["threshold"]["applicable"] is True
    assert meta["threshold"]["fitted_on"] == "val"
    assert meta["threshold"]["objective"] == "f1_minority"
    assert 0.05 <= meta["threshold"]["value"] <= 0.95


def test_run_gen3_predictions_are_binary_and_aligned(
    tiny_binary_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    fake = FakeOllamaClient()
    run_dir = gen3_llm.run_gen3_experiment(
        splits=tiny_binary_splits,
        split_meta_block=split_meta_block,
        model_slug="llama3.1-8b",
        client=fake,
    )
    preds = pd.read_csv(run_dir / "predictions.csv")
    assert list(preds.columns) == ["index", "y_true", "y_pred", "y_score"]
    assert len(preds) == len(tiny_binary_splits["test"])
    preds_sorted = preds.sort_values("index").reset_index(drop=True)
    np.testing.assert_array_equal(
        preds_sorted["index"].to_numpy(),
        tiny_binary_splits["test"].index.to_numpy(),
    )
    np.testing.assert_array_equal(
        preds_sorted["y_true"].to_numpy(),
        tiny_binary_splits["test"]["label"].to_numpy(),
    )
    assert preds_sorted["y_pred"].isin([0, 1]).all()
    # Fake sempre acerta — F1 perfeito no tiny corpus
    np.testing.assert_array_equal(
        preds_sorted["y_pred"].to_numpy(),
        tiny_binary_splits["test"]["label"].to_numpy(),
    )


def test_run_gen3_multiclass_variant_slug(
    tiny_binary_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    fake = FakeOllamaClient()
    run_dir = gen3_llm.run_gen3_experiment(
        splits=tiny_binary_splits,
        split_meta_block=split_meta_block,
        model_slug="llama3.1-8b",
        target_mode="multiclass",
        collapse_scheme="top7_plus_other",
        client=fake,
    )
    assert "__gen3__llama3.1-8b__zs-v1-mc8" in run_dir.name
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["variant"] == "zs-v1-mc8"
    assert meta["config"]["target"]["mode"] == "multiclass"
    assert meta["config"]["target"]["num_classes"] == 8
    assert meta["config"]["target"]["collapse_scheme"] == "top7_plus_other"
    assert meta["config"]["prompt"]["target_tag"] == "mc8"


def test_run_gen3_resume_skips_processed_indices(
    tiny_binary_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    """Run parcial via `limit`, depois retoma sobre o mesmo run_dir.

    `limit` só afeta a fase de test (val é sempre processado integralmente
    para permitir fit do threshold).
    """
    fake1 = FakeOllamaClient()
    n_val = len(tiny_binary_splits["val"])
    # Passe 1: val completo (4) + 3 primeiras de test (limit=3) = 7 chamadas.
    run_dir = gen3_llm.run_gen3_experiment(
        splits=tiny_binary_splits,
        split_meta_block=split_meta_block,
        model_slug="llama3.1-8b",
        limit=3,
        client=fake1,
    )
    assert fake1.calls == n_val + 3
    # Passe 2: retoma o mesmo run_dir sem limit. Val já processado → 0
    # calls; test tem 3 done, 3 faltando → 3 calls. Total: 3.
    fake2 = FakeOllamaClient()
    run_dir2 = gen3_llm.run_gen3_experiment(
        splits=tiny_binary_splits,
        split_meta_block=split_meta_block,
        model_slug="llama3.1-8b",
        resume_run_dir=run_dir,
        client=fake2,
    )
    assert run_dir2 == run_dir
    assert fake2.calls == 3  # só as que faltavam de test; val já estava feito
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["config"]["resume"]["resumed"] is True
    assert meta["config"]["resume"]["n_skipped"] == 3
    assert meta["config"]["resume"]["n_processed_this_run"] == 3
    # Val completamente pulado na 2ª passada
    assert meta["config"]["resume"]["n_skipped_val"] == n_val
    assert meta["config"]["resume"]["n_processed_val_this_run"] == 0
    preds = pd.read_csv(run_dir / "predictions.csv")
    assert len(preds) == len(tiny_binary_splits["test"])
    val_preds = pd.read_csv(run_dir / "predictions_val.csv")
    assert len(val_preds) == n_val


def test_run_gen3_rejects_unknown_model(
    tiny_binary_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    with pytest.raises(ValueError, match="Modelo Gen 3 desconhecido"):
        gen3_llm.run_gen3_experiment(
            splits=tiny_binary_splits,
            split_meta_block=split_meta_block,
            model_slug="inexistente-xl",
            client=FakeOllamaClient(),
        )


def test_run_gen3_multiclass_requires_scheme(
    tiny_binary_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    with pytest.raises(ValueError, match="requer collapse_scheme"):
        gen3_llm.run_gen3_experiment(
            splits=tiny_binary_splits,
            split_meta_block=split_meta_block,
            model_slug="llama3.1-8b",
            target_mode="multiclass",
            client=FakeOllamaClient(),
        )


def test_run_gen3_binary_rejects_scheme(
    tiny_binary_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    with pytest.raises(ValueError, match="só é usado quando"):
        gen3_llm.run_gen3_experiment(
            splits=tiny_binary_splits,
            split_meta_block=split_meta_block,
            model_slug="llama3.1-8b",
            target_mode="binary",
            collapse_scheme="top7_plus_other",
            client=FakeOllamaClient(),
        )


def test_run_gen3_counts_parse_failures(
    tiny_binary_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    fake = FakeOllamaClient(force_parse_failure=True)
    run_dir = gen3_llm.run_gen3_experiment(
        splits=tiny_binary_splits,
        split_meta_block=split_meta_block,
        model_slug="llama3.1-8b",
        client=fake,
    )
    meta = json.loads((run_dir / "metadata.json").read_text())
    counts = meta["config"]["scoring"]["method_counts"]
    assert counts["parse_failure"] == fake.calls
    assert counts["logprobs"] == 0


def test_run_gen3_efficiency_block(
    tiny_binary_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    fake = FakeOllamaClient()
    run_dir = gen3_llm.run_gen3_experiment(
        splits=tiny_binary_splits,
        split_meta_block=split_meta_block,
        model_slug="llama3.1-8b",
        client=fake,
    )
    meta = json.loads((run_dir / "metadata.json").read_text())
    eff = meta["efficiency"]
    assert eff["latency_includes_tokenization"] is True
    assert eff["latency_measured_on"] == "val"
    assert eff["latency_ms_per_1k"] > 0
    # n_processed combinado (val + test) bate com o total de chamadas
    assert eff["n_processed"] == fake.calls
    assert eff["n_processed_val"] == len(tiny_binary_splits["val"])
    assert eff["n_processed_test"] == len(tiny_binary_splits["test"])
    assert "vram_peak_mb" in eff


def test_gen3_model_specs_have_valid_tags() -> None:
    for slug, spec in gen3_llm.GEN3_MODELS.items():
        assert spec.slug == slug
        assert ":" in spec.ollama_model_tag, f"tag inválida: {spec.ollama_model_tag!r}"
        assert spec.bucket in {"global-native", "ptbr-gguf"}
        assert spec.num_ctx > 0


def test_target_tag_alignment_with_targets_module() -> None:
    # Garantia: o sufixo `bin|mc8` do variant bate com targets.target_variant_tag
    assert targets.target_variant_tag("binary", None) == "bin"
    assert targets.target_variant_tag("multiclass", "top7_plus_other") == "mc8"
