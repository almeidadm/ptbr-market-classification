"""Testes de integração do pipeline Gen 1 end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptbr_market import gen1_pipeline, runs
from ptbr_market.representations import FASTTEXT_DIM


@pytest.fixture
def tiny_market_splits() -> dict[str, pd.DataFrame]:
    """Corpus sintético com sinal claro: textos 'de mercado' contêm palavras-chave
    financeiras; textos 'outros' contêm vocabulário neutro. Tamanhos suficientes
    para calibração 3-fold (≥ 3 positivos em cada split)."""
    rng = np.random.default_rng(1)
    market_templates = [
        "a bolsa de valores fechou em alta forte hoje",
        "a petrobras anunciou lucro recorde no trimestre",
        "investidores compraram ações no mercado financeiro",
        "o real subiu frente ao dolar na sessão",
        "a inflação atingiu o topo do intervalo da meta",
        "banco central cortou a taxa selic nesta reunião",
        "empresa divulgou balanço com alta de receita",
        "o ibovespa subiu com o bom humor do mercado",
    ]
    other_templates = [
        "time venceu o jogo no fim de semana passado",
        "o filme estreia nos cinemas nesta quinta",
        "evento cultural atrai publico na capital paulista",
        "equipe conquistou medalha no campeonato nacional",
        "show da banda lotou o estadio na noite",
        "festival de cinema começa no proximo mes",
        "novela tem recorde de audiencia no horario",
        "partida teve placar apertado no segundo tempo",
    ]

    def make(n_pos: int, n_neg: int, date_offset: int) -> pd.DataFrame:
        texts: list[str] = []
        labels: list[int] = []
        dates: list[pd.Timestamp] = []
        for _ in range(n_pos):
            texts.append(rng.choice(market_templates))
            labels.append(1)
        for _ in range(n_neg):
            texts.append(rng.choice(other_templates))
            labels.append(0)
        start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=date_offset)
        for i in range(len(texts)):
            dates.append(start + pd.Timedelta(days=i))
        df = pd.DataFrame({"text": texts, "label": labels, "date": dates})
        return df.reset_index(drop=True)

    return {
        "train": make(n_pos=15, n_neg=45, date_offset=0),
        "val": make(n_pos=5, n_neg=15, date_offset=60),
        "test": make(n_pos=5, n_neg=15, date_offset=80),
    }


@pytest.fixture
def split_meta_block() -> dict:
    return {
        "train_window": ["2020-01-01", "2020-03-01"],
        "val_window": ["2020-03-02", "2020-03-22"],
        "test_window": ["2020-03-23", "2020-04-11"],
        "n_train": 60,
        "n_val": 20,
        "n_test": 20,
    }


@pytest.fixture
def isolated_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setenv("PTBR_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


def test_run_gen1_experiment_writes_complete_run_dir(
    tiny_market_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    run_dirs = gen1_pipeline.run_gen1_experiment(
        splits=tiny_market_splits,
        split_meta_block=split_meta_block,
        models=["logreg"],
        variant="test-tiny",
        preprocess_mode="raw",
        representation="tfidf",
        representation_params={"min_df": 1, "max_df": 1.0, "ngram_range": (1, 1)},
    )
    assert len(run_dirs) == 1
    rd = run_dirs[0]
    assert rd.is_dir()
    assert rd.parent == isolated_artifacts / "runs"
    assert rd.name.endswith("__gen1__logreg__test-tiny")
    # Três arquivos obrigatórios existem:
    assert (rd / "metadata.json").is_file()
    assert (rd / "metrics.json").is_file()
    assert (rd / "predictions.csv").is_file()


def test_run_gen1_experiment_metadata_has_required_keys(
    tiny_market_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    run_dirs = gen1_pipeline.run_gen1_experiment(
        splits=tiny_market_splits,
        split_meta_block=split_meta_block,
        models=["logreg"],
        variant="test-tiny",
        preprocess_mode="raw",
        representation_params={"min_df": 1, "max_df": 1.0},
    )
    meta = json.loads((run_dirs[0] / "metadata.json").read_text())
    for key in runs.REQUIRED_METADATA_KEYS:
        assert key in meta, f"chave {key!r} ausente em metadata.json"
    assert meta["generation"] == "gen1"
    assert meta["model"] == "logreg"
    assert meta["variant"] == "test-tiny"
    assert meta["threshold"]["fitted_on"] == "val"
    assert meta["threshold"]["objective"] == "f1_minority"
    assert 0.05 <= meta["threshold"]["value"] <= 0.95
    assert meta["config"]["seed"] == 1
    assert meta["config"]["preprocess"] == "raw"
    assert meta["config"]["mask_entities"] is False  # raw mode não mascara
    assert meta["efficiency"]["latency_ms_per_1k"] >= 0.0


def test_run_gen1_experiment_metrics_match_metadata(
    tiny_market_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    run_dirs = gen1_pipeline.run_gen1_experiment(
        splits=tiny_market_splits,
        split_meta_block=split_meta_block,
        models=["logreg"],
        variant="test",
        preprocess_mode="raw",
        representation_params={"min_df": 1, "max_df": 1.0},
    )
    meta = json.loads((run_dirs[0] / "metadata.json").read_text())
    metrics = json.loads((run_dirs[0] / "metrics.json").read_text())
    assert metrics == meta["metrics"]


def test_run_gen1_experiment_predictions_shape(
    tiny_market_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    run_dirs = gen1_pipeline.run_gen1_experiment(
        splits=tiny_market_splits,
        split_meta_block=split_meta_block,
        models=["logreg"],
        variant="test",
        preprocess_mode="raw",
        representation_params={"min_df": 1, "max_df": 1.0},
    )
    df = pd.read_csv(run_dirs[0] / "predictions.csv")
    assert list(df.columns) == ["index", "y_true", "y_pred", "y_score"]
    assert len(df) == len(tiny_market_splits["test"])
    assert set(df["y_true"].unique()) <= {0, 1}
    assert set(df["y_pred"].unique()) <= {0, 1}
    assert (df["y_score"] >= 0.0).all() and (df["y_score"] <= 1.0).all()


def test_run_gen1_experiment_learns_some_signal(
    tiny_market_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    """Em um corpus sintético com sinal limpo, PR-AUC deve ser bem acima do
    baseline (prevalência ≈ 25%)."""
    run_dirs = gen1_pipeline.run_gen1_experiment(
        splits=tiny_market_splits,
        split_meta_block=split_meta_block,
        models=["logreg"],
        variant="test",
        preprocess_mode="raw",
        representation_params={"min_df": 1, "max_df": 1.0},
    )
    metrics = json.loads((run_dirs[0] / "metrics.json").read_text())
    assert metrics["pr_auc"] > 0.6, metrics


def test_run_gen1_experiment_rejects_unknown_model(
    tiny_market_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    with pytest.raises(ValueError, match="desconhecido"):
        gen1_pipeline.run_gen1_experiment(
            splits=tiny_market_splits,
            split_meta_block=split_meta_block,
            models=["xgboost"],
            variant="test",
            preprocess_mode="raw",
        )


def test_run_gen1_experiment_rejects_missing_split(
    tiny_market_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    bad = {k: v for k, v in tiny_market_splits.items() if k != "test"}
    with pytest.raises(ValueError, match="splits faltando"):
        gen1_pipeline.run_gen1_experiment(
            splits=bad,
            split_meta_block=split_meta_block,
            models=["logreg"],
            variant="test",
            preprocess_mode="raw",
        )


@pytest.fixture
def fake_fasttext_in_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_market_splits: dict[str, pd.DataFrame],
) -> Path:
    """Cria um `cc.pt.300.vec` fake em `data/raw/` cobrindo o vocabulário do
    corpus sintético, com vetores determinísticos: tokens de 'mercado' somam
    ~+1 em cada dimensão e tokens 'outros' ~-1, dando sinal linearmente
    separável para LogReg."""
    monkeypatch.setenv("PTBR_DATA_ROOT", str(tmp_path / "data"))
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)

    market_tokens = {
        "bolsa", "valores", "alta", "forte", "petrobras", "lucro", "recorde",
        "trimestre", "investidores", "ações", "mercado", "financeiro", "real",
        "subiu", "dolar", "sessão", "inflação", "atingiu", "topo", "intervalo",
        "meta", "banco", "central", "cortou", "taxa", "selic", "reunião",
        "empresa", "divulgou", "balanço", "receita", "ibovespa", "humor",
        "hoje", "fechou", "compraram", "nesta", "frente", "ao",
        "ano", "de", "a", "o", "com", "em", "no", "do", "anunciou", "da",
    }
    rng = np.random.default_rng(0)
    tokens_vectors: dict[str, np.ndarray] = {}

    def _noise() -> np.ndarray:
        return rng.normal(scale=0.05, size=FASTTEXT_DIM).astype(np.float32)

    # Colete todos os tokens do corpus sintético.
    all_tokens: set[str] = set()
    for df in tiny_market_splits.values():
        for t in df["text"]:
            all_tokens.update(t.split())
    for tok in all_tokens:
        base = np.ones(FASTTEXT_DIM, dtype=np.float32) if tok in market_tokens else (
            -np.ones(FASTTEXT_DIM, dtype=np.float32)
        )
        tokens_vectors[tok] = base + _noise()

    path = raw / "cc.pt.300.vec"
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(tokens_vectors)} {FASTTEXT_DIM}\n")
        for tok, vec in tokens_vectors.items():
            f.write(f"{tok} " + " ".join(f"{v:.6f}" for v in vec) + "\n")
    return path


def test_run_gen1_experiment_with_fasttext(
    tiny_market_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
    fake_fasttext_in_data_root: Path,
) -> None:
    run_dirs = gen1_pipeline.run_gen1_experiment(
        splits=tiny_market_splits,
        split_meta_block=split_meta_block,
        models=["logreg"],
        variant="fasttext-raw",
        preprocess_mode="raw",
        representation="fasttext",
    )
    assert len(run_dirs) == 1
    rd = run_dirs[0]
    assert (rd / "metadata.json").is_file()
    meta = json.loads((rd / "metadata.json").read_text())
    assert meta["config"]["representation"] == "fasttext"
    # Sinal sintético é linearmente separável — PR-AUC deve ser alto.
    metrics = json.loads((rd / "metrics.json").read_text())
    assert metrics["pr_auc"] > 0.8, metrics


def test_run_gen1_experiment_rejects_fasttext_with_naive_bayes(
    tiny_market_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    with pytest.raises(ValueError, match="features com valores negativos"):
        gen1_pipeline.run_gen1_experiment(
            splits=tiny_market_splits,
            split_meta_block=split_meta_block,
            models=["multinomialnb"],
            variant="test",
            preprocess_mode="raw",
            representation="fasttext",
        )


def test_run_gen1_experiment_rejects_unknown_representation(
    tiny_market_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    with pytest.raises(ValueError, match="representation deve ser"):
        gen1_pipeline.run_gen1_experiment(
            splits=tiny_market_splits,
            split_meta_block=split_meta_block,
            models=["logreg"],
            variant="test",
            preprocess_mode="raw",
            representation="word2vec",
        )


def test_run_gen1_experiment_produces_one_run_per_model(
    tiny_market_splits: dict[str, pd.DataFrame],
    split_meta_block: dict,
    isolated_artifacts: Path,
) -> None:
    run_dirs = gen1_pipeline.run_gen1_experiment(
        splits=tiny_market_splits,
        split_meta_block=split_meta_block,
        models=["logreg", "multinomialnb"],
        variant="test",
        preprocess_mode="raw",
        representation_params={"min_df": 1, "max_df": 1.0},
    )
    assert len(run_dirs) == 2
    assert run_dirs[0].name.endswith("__logreg__test")
    assert run_dirs[1].name.endswith("__multinomialnb__test")


# -------- CLI smoke tests --------------------------------------------------


def test_script_parse_args_defaults_match_expected() -> None:
    """Smoke test do script CLI: defaults batem com as constantes."""
    import importlib.util

    script_path = Path(__file__).parent.parent / "scripts" / "run_gen1.py"
    spec = importlib.util.spec_from_file_location("run_gen1", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.parse_args([])
    assert args.variant == module.DEFAULT_VARIANT
    assert args.representation == module.DEFAULT_REPRESENTATION
    assert args.preprocess_mode == module.DEFAULT_MODE
    assert args.no_cache is False


def test_script_parse_args_honors_overrides() -> None:
    import importlib.util

    script_path = Path(__file__).parent.parent / "scripts" / "run_gen1.py"
    spec = importlib.util.spec_from_file_location("run_gen1", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.parse_args(
        ["--models", "logreg", "--variant", "v1", "--no-cache"]
    )
    assert args.models == "logreg"
    assert args.variant == "v1"
    assert args.no_cache is True
