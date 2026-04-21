"""Testes de runs.new_run_dir, write_run_metadata, write_metrics, write_predictions e _slugify."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from ptbr_market import runs

# -------- _slugify ---------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("LinearSVC", "linearsvc"),
        ("BERTimbau/base", "bertimbau-base"),
        ("  Foo  Bar! ", "foo-bar"),
        ("a_b", "a-b"),
        ("tfidf-lemmatized", "tfidf-lemmatized"),
        ("model+v2", "model+v2"),
        ("news 1.2.3", "news-1.2.3"),
        ("---edge---", "edge"),
    ],
)
def test_slugify(raw: str, expected: str) -> None:
    assert runs._slugify(raw) == expected


def test_slugify_returns_empty_for_garbage() -> None:
    assert runs._slugify("!!!") == ""
    assert runs._slugify("   ") == ""
    assert runs._slugify("") == ""


# -------- new_run_dir ------------------------------------------------------


@pytest.fixture
def artifacts_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setenv("PTBR_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


def test_new_run_dir_format(artifacts_tmp: Path) -> None:
    now = datetime(2026, 4, 21, 14, 30, 5)
    path = runs.new_run_dir("gen1", "LinearSVC", "tfidf-lemmatized", now=now)
    expected = artifacts_tmp / "runs" / "20260421-143005__gen1__linearsvc__tfidf-lemmatized"
    assert path == expected
    assert path.is_dir()


def test_new_run_dir_slugifies_model_and_variant(artifacts_tmp: Path) -> None:
    now = datetime(2026, 4, 21, 14, 30, 5)
    path = runs.new_run_dir("gen2", "BERTimbau/Large", "raw preprocessing", now=now)
    assert path.name == "20260421-143005__gen2__bertimbau-large__raw-preprocessing"


def test_new_run_dir_rejects_unknown_generation(artifacts_tmp: Path) -> None:
    with pytest.raises(ValueError, match="generation"):
        runs.new_run_dir("gen99", "foo", "bar")


def test_new_run_dir_rejects_empty_slug(artifacts_tmp: Path) -> None:
    with pytest.raises(ValueError, match="slugs não-vazios"):
        runs.new_run_dir("gen1", "!!!", "variant")
    with pytest.raises(ValueError, match="slugs não-vazios"):
        runs.new_run_dir("gen1", "model", "   ")


def test_new_run_dir_uses_now_by_default(artifacts_tmp: Path) -> None:
    path = runs.new_run_dir("gen1", "m", "v")
    assert re.match(r"\d{8}-\d{6}__gen1__m__v", path.name)


def test_new_run_dir_is_idempotent_with_fixed_now(artifacts_tmp: Path) -> None:
    now = datetime(2026, 4, 21, 14, 30, 5)
    a = runs.new_run_dir("gen1", "m", "v", now=now)
    b = runs.new_run_dir("gen1", "m", "v", now=now)
    assert a == b
    assert a.is_dir()


# -------- write_run_metadata ----------------------------------------------


def _valid_payload(run_id: str = "run-1") -> dict:
    return {
        "run_id": run_id,
        "git_commit": "abc1234",
        "started_at": "2026-04-21T14:30:05-03:00",
        "finished_at": "2026-04-21T14:31:47-03:00",
        "generation": "gen1",
        "model": "linearsvc",
        "variant": "tfidf",
        "config": {"seed": 1, "mask_entities": True},
        "split": {
            "train_window": ["2015-01-01", "2017-01-07"],
            "val_window": ["2017-01-08", "2017-05-23"],
            "test_window": ["2017-05-24", "2017-10-01"],
            "n_train": 133725,
            "n_val": 16701,
            "n_test": 16627,
        },
        "threshold": {"fitted_on": "val", "value": 0.33, "objective": "f1_minority"},
        "metrics": {"pr_auc": 0.5, "roc_auc": 0.9, "f1_macro": 0.6, "f1_minority": 0.4},
        "efficiency": {"latency_ms_per_1k": 120.5, "vram_peak_mb": 0},
    }


def test_write_run_metadata_roundtrip(tmp_path: Path) -> None:
    payload = _valid_payload("roundtrip")
    runs.write_run_metadata(tmp_path, payload)
    read_back = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert read_back == payload


def test_write_run_metadata_rejects_missing_keys(tmp_path: Path) -> None:
    payload = _valid_payload()
    del payload["config"]
    del payload["efficiency"]
    with pytest.raises(ValueError, match="config.*efficiency|efficiency.*config"):
        runs.write_run_metadata(tmp_path, payload)


def test_write_run_metadata_sorts_top_level_keys(tmp_path: Path) -> None:
    payload = _valid_payload()
    runs.write_run_metadata(tmp_path, payload)
    with (tmp_path / "metadata.json").open(encoding="utf-8") as f:
        on_disk = json.load(f)
    assert list(on_disk.keys()) == sorted(payload.keys())


def test_write_run_metadata_creates_missing_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "run"
    runs.write_run_metadata(target, _valid_payload())
    assert (target / "metadata.json").is_file()


# -------- write_metrics ----------------------------------------------------


def test_write_metrics_roundtrip(tmp_path: Path) -> None:
    metrics = {
        "pr_auc": 0.67,
        "f1_minority": 0.45,
        "confusion_matrix": [[80, 5], [10, 5]],
    }
    runs.write_metrics(tmp_path, metrics)
    read_back = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert read_back == metrics


# -------- write_predictions ------------------------------------------------


def test_write_predictions_shape_and_headers(tmp_path: Path) -> None:
    runs.write_predictions(
        tmp_path,
        indices=[0, 1, 2],
        y_true=[0, 1, 1],
        y_pred=[0, 1, 0],
        y_score=[0.1, 0.9, 0.4],
    )
    with (tmp_path / "predictions.csv").open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["index", "y_true", "y_pred", "y_score"]
    assert rows[1:] == [
        ["0", "0", "0", "0.1"],
        ["1", "1", "1", "0.9"],
        ["2", "1", "0", "0.4"],
    ]


def test_write_predictions_rejects_length_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incompatíveis"):
        runs.write_predictions(
            tmp_path,
            indices=[0, 1],
            y_true=[0, 1],
            y_pred=[0],
            y_score=[0.1, 0.9],
        )


def test_write_predictions_creates_run_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper"
    runs.write_predictions(target, [0], [1], [1], [0.9])
    assert (target / "predictions.csv").is_file()


# -------- preprocessed_path -----------------------------------------------


def test_preprocessed_path_respects_artifacts_root(artifacts_tmp: Path) -> None:
    p = runs.preprocessed_path("train", "aggressive")
    assert p == artifacts_tmp / "preprocessed" / "train__aggressive.parquet"


def test_preprocessed_path_varies_by_split_and_mode(artifacts_tmp: Path) -> None:
    assert runs.preprocessed_path("train", "raw") != runs.preprocessed_path(
        "val", "raw"
    )
    assert runs.preprocessed_path("train", "raw") != runs.preprocessed_path(
        "train", "aggressive"
    )
