"""Testes do carregamento e do split Out-of-Time."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from ptbr_market import data

# -------- Unitários (corpus sintético) -------------------------------------


def test_split_temporal_disjoint(tiny_corpus: pd.DataFrame) -> None:
    train, val, test = data.split_out_of_time(tiny_corpus)
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()


def test_split_conservation(tiny_corpus: pd.DataFrame) -> None:
    train, val, test = data.split_out_of_time(tiny_corpus)
    assert len(train) + len(val) + len(test) == len(tiny_corpus)
    union_links = set(train["link"]) | set(val["link"]) | set(test["link"])
    assert union_links == set(tiny_corpus["link"])


def test_split_is_deterministic(tiny_corpus: pd.DataFrame) -> None:
    first = data.split_out_of_time(tiny_corpus)
    shuffled = tiny_corpus.sample(frac=1.0, random_state=7).reset_index(drop=True)
    second = data.split_out_of_time(shuffled)
    for a, b in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(a, b, check_like=False)


def test_split_boundary_day_rule(tiny_corpus: pd.DataFrame) -> None:
    train, val, test = data.split_out_of_time(tiny_corpus)
    day8 = pd.Timestamp("2020-01-08")
    day9 = pd.Timestamp("2020-01-09")
    day10 = pd.Timestamp("2020-01-10")

    assert (train["date"] == day8).sum() == 4, "dia 8 inteiro deve ficar em train"
    assert (val["date"] == day8).sum() == 0
    assert (val["date"] == day9).sum() == 3, "dia 9 inteiro deve ficar em val"
    assert (test["date"] == day9).sum() == 0
    assert (test["date"] == day10).sum() == 2


def test_split_expected_sizes(tiny_corpus: pd.DataFrame) -> None:
    train, val, test = data.split_out_of_time(tiny_corpus)
    assert (len(train), len(val), len(test)) == (25, 3, 2)


def test_split_has_positive_in_every_partition(tiny_corpus: pd.DataFrame) -> None:
    train, val, test = data.split_out_of_time(tiny_corpus)
    assert train["label"].sum() > 0
    assert val["label"].sum() > 0
    assert test["label"].sum() > 0


def test_split_empty_partition_raises() -> None:
    df = pd.DataFrame(
        {
            "title": ["a", "b", "c"],
            "text": ["x", "y", "z"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "category": ["mercado", "esporte", "mercado"],
            "link": ["a", "b", "c"],
            "label": [1, 0, 1],
        }
    )
    with pytest.raises(ValueError):
        data.split_out_of_time(df)


def test_split_invalid_fractions_raise(tiny_corpus: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        data.split_out_of_time(tiny_corpus, train_frac=0.95, val_frac=0.1)
    with pytest.raises(ValueError):
        data.split_out_of_time(tiny_corpus, train_frac=0.0, val_frac=0.1)


def test_describe_split_schema(tiny_corpus: pd.DataFrame) -> None:
    train, val, test = data.split_out_of_time(tiny_corpus)
    desc = data.describe_split(train, val, test)
    assert desc["tie_rule"] == data.TIE_RULE
    assert desc["total_n"] == len(tiny_corpus)
    for name in ("train", "val", "test"):
        part = desc[name]
        assert set(part) == {"n", "window", "pos_count", "pos_ratio"}
        assert len(part["window"]) == 2


# -------- Integração (corpus real) ------------------------------------------


@pytest.mark.integration
def test_real_corpus_loads(skip_if_no_real_corpus: None) -> None:  # noqa: ARG001
    df = data.load_corpus()
    assert len(df) == 167_053
    assert (df["category"] == "mercado").sum() == 20_970
    assert df["date"].min().date().isoformat() == "2015-01-01"
    assert df["date"].max().date().isoformat() == "2017-10-01"


@pytest.mark.integration
def test_real_split_proportions(skip_if_no_real_corpus: None) -> None:  # noqa: ARG001
    df = data.load_corpus()
    train, val, test = data.split_out_of_time(df)
    n = len(df)
    assert abs(len(train) / n - 0.80) < 0.005
    assert abs(len(val) / n - 0.10) < 0.005
    assert abs(len(test) / n - 0.10) < 0.005


@pytest.mark.integration
def test_real_split_windows_exact(skip_if_no_real_corpus: None) -> None:  # noqa: ARG001
    df = data.load_corpus()
    train, val, test = data.split_out_of_time(df)
    assert train["date"].min().date().isoformat() == "2015-01-01"
    assert train["date"].max().date().isoformat() == "2017-01-07"
    assert val["date"].min().date().isoformat() == "2017-01-08"
    assert val["date"].max().date().isoformat() == "2017-05-23"
    assert test["date"].min().date().isoformat() == "2017-05-24"
    assert test["date"].max().date().isoformat() == "2017-10-01"
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()


@pytest.mark.integration
def test_real_split_positives(skip_if_no_real_corpus: None) -> None:  # noqa: ARG001
    df = data.load_corpus()
    train, val, test = data.split_out_of_time(df)
    assert train["label"].sum() > 1_000
    assert val["label"].sum() > 1_000
    assert test["label"].sum() > 1_000


@pytest.mark.integration
def test_build_splits_script_produces_parquets(
    skip_if_no_real_corpus: None,  # noqa: ARG001
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_splits.py",
            "--artifacts-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    for name in ("train.parquet", "val.parquet", "test.parquet", "metadata.json"):
        assert (tmp_path / "splits" / name).is_file(), f"{name} não foi gerado"
    assert "train" in result.stdout
