"""Smoke tests do pipeline Gen 2 end-to-end com tiny-random-bert."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptbr_market import gen2_bert, runs

TINY_MODEL_ID = "hf-internal-testing/tiny-random-bert"


@pytest.fixture
def tiny_binary_splits() -> dict[str, pd.DataFrame]:
    """Corpus mínimo com sinal binário claro. Fixa índices para validar
    alinhamento de `predictions.csv`."""
    market = [
        "a bolsa de valores fechou em alta forte hoje",
        "a petrobras anunciou lucro recorde no trimestre",
        "investidores compraram ações no mercado financeiro",
        "o real subiu frente ao dolar na sessão",
        "banco central cortou a taxa selic nesta reunião",
    ]
    other = [
        "time venceu o jogo no fim de semana passado",
        "o filme estreia nos cinemas nesta quinta",
        "evento cultural atrai publico na capital paulista",
        "equipe conquistou medalha no campeonato nacional",
        "show da banda lotou o estadio na noite",
    ]

    def make(n_pos: int, n_neg: int) -> pd.DataFrame:
        texts = market[:n_pos] + other[:n_neg]
        labels = [1] * n_pos + [0] * n_neg
        return pd.DataFrame({"text": texts, "label": labels}).reset_index(drop=True)

    return {
        "train": make(5, 5),
        "val": make(3, 3),
        "test": make(3, 3),
    }


@pytest.fixture
def split_meta_block() -> dict:
    return {
        "train_window": ["2020-01-01", "2020-01-10"],
        "val_window": ["2020-01-11", "2020-01-16"],
        "test_window": ["2020-01-17", "2020-01-22"],
        "n_train": 10,
        "n_val": 6,
        "n_test": 6,
    }


@pytest.fixture
def isolated_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setenv("PTBR_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def registered_tiny_model(monkeypatch: pytest.MonkeyPatch) -> str:
    """Injeta um spec de teste no GEN2_MODELS apontando para tiny-random-bert.

    Usa monkeypatch sobre o dict para não vazar para outros testes.
    """
    slug = "tiny-test-bert"
    spec = gen2_bert.Gen2ModelSpec(
        slug=slug,
        hf_id=TINY_MODEL_ID,
        bucket="ptbr",
        batch_size=2,
    )
    patched = dict(gen2_bert.GEN2_MODELS)
    patched[slug] = spec
    monkeypatch.setattr(gen2_bert, "GEN2_MODELS", patched)
    return slug


def test_gen2_models_registry_exposes_expected_slugs() -> None:
    expected = {
        "bertimbau-base",
        "finbert-ptbr",
        "distilbertimbau",
        "bertimbau-large",
        "xlmr-base",
        "deb3rta-base",
    }
    assert expected.issubset(set(gen2_bert.GEN2_MODELS))


def test_gen2_model_specs_have_nonempty_hf_ids() -> None:
    for slug, spec in gen2_bert.GEN2_MODELS.items():
        assert spec.slug == slug
        assert "/" in spec.hf_id, f"hf_id inválido para {slug}: {spec.hf_id!r}"
        assert spec.bucket in {"domain", "ptbr", "multilingual", "efficiency"}


def test_seed_constant_is_1() -> None:
    assert gen2_bert.SEED == 1


@pytest.mark.slow
def test_run_gen2_experiment_writes_complete_run_dir(
    tiny_binary_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
    registered_tiny_model: str,
) -> None:
    run_dir = gen2_bert.run_gen2_experiment(
        splits=tiny_binary_splits,
        split_meta_block=split_meta_block,
        model_slug=registered_tiny_model,
        max_length=32,
        epochs=1,
        batch_size=2,
    )
    assert run_dir.is_dir()
    assert run_dir.parent == isolated_artifacts / "runs"
    assert run_dir.name.startswith(tuple(f"2{y}" for y in range(0, 10)))
    assert f"__gen2__{registered_tiny_model}__raw-ml32" in run_dir.name
    assert (run_dir / "metadata.json").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "predictions.csv").is_file()


@pytest.mark.slow
def test_run_gen2_experiment_metadata_schema(
    tiny_binary_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
    registered_tiny_model: str,
) -> None:
    run_dir = gen2_bert.run_gen2_experiment(
        splits=tiny_binary_splits,
        split_meta_block=split_meta_block,
        model_slug=registered_tiny_model,
        max_length=32,
        epochs=1,
        batch_size=2,
    )
    meta = json.loads((run_dir / "metadata.json").read_text())
    for key in runs.REQUIRED_METADATA_KEYS:
        assert key in meta, f"metadata faltando chave obrigatória {key!r}"
    assert meta["generation"] == "gen2"
    assert meta["model"] == registered_tiny_model
    assert meta["variant"] == "raw-ml32"
    assert meta["config"]["preprocess"] == "raw"
    assert meta["config"]["mask_entities"] is False
    assert meta["config"]["target"]["mode"] == "binary"
    assert meta["config"]["hf_id"] == TINY_MODEL_ID
    assert meta["efficiency"]["latency_includes_tokenization"] is True
    assert meta["threshold"]["fitted_on"] == "val"
    assert 0.05 <= meta["threshold"]["value"] <= 0.95


@pytest.mark.slow
def test_run_gen2_experiment_predictions_aligned_with_test_index(
    tiny_binary_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
    registered_tiny_model: str,
) -> None:
    run_dir = gen2_bert.run_gen2_experiment(
        splits=tiny_binary_splits,
        split_meta_block=split_meta_block,
        model_slug=registered_tiny_model,
        max_length=32,
        epochs=1,
        batch_size=2,
    )
    preds = pd.read_csv(run_dir / "predictions.csv")
    assert list(preds.columns) == ["index", "y_true", "y_pred", "y_score"]
    assert len(preds) == len(tiny_binary_splits["test"])
    np.testing.assert_array_equal(
        preds["index"].to_numpy(),
        tiny_binary_splits["test"].index.to_numpy(),
    )
    np.testing.assert_array_equal(
        preds["y_true"].to_numpy(),
        tiny_binary_splits["test"]["label"].to_numpy(),
    )
    assert preds["y_pred"].isin([0, 1]).all()
    assert ((preds["y_score"] >= 0) & (preds["y_score"] <= 1)).all()


def test_run_gen2_rejects_unknown_model(
    tiny_binary_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    with pytest.raises(ValueError, match="Modelo Gen 2 desconhecido"):
        gen2_bert.run_gen2_experiment(
            splits=tiny_binary_splits,
            split_meta_block=split_meta_block,
            model_slug="bertimbau-inexistente",
        )


@pytest.mark.slow
def test_run_gen2_variant_autoderives_from_max_length(
    tiny_binary_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
    registered_tiny_model: str,
) -> None:
    run_dir = gen2_bert.run_gen2_experiment(
        splits=tiny_binary_splits,
        split_meta_block=split_meta_block,
        model_slug=registered_tiny_model,
        max_length=16,
        epochs=1,
        batch_size=2,
    )
    assert f"__gen2__{registered_tiny_model}__raw-ml16" in run_dir.name
