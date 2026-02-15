"""Tests for data loader: CSV parsing, column normalization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_csv, _resolve_columns, load_data


def _write_csv(path, content):
    Path(path).write_text(content, encoding="utf-8")


def test_load_csv_standard_columns(tmp_path):
    csv_file = tmp_path / "test.csv"
    _write_csv(csv_file, "Date,Open,High,Low,Close,Volume\n2024-01-01,100,105,95,102,1000\n2024-01-02,102,107,100,105,1200\n")

    df = load_csv(csv_file)
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "close" in df.columns
    assert len(df) == 2
    assert df["close"].iloc[0] == 102


def test_load_csv_lowercase_columns(tmp_path):
    csv_file = tmp_path / "test.csv"
    _write_csv(csv_file, "timestamp,open,high,low,close,volume\n2024-01-01,100,105,95,102,1000\n")

    df = load_csv(csv_file)
    assert len(df) == 1
    assert df["close"].iloc[0] == 102


def test_load_csv_tab_separated(tmp_path):
    csv_file = tmp_path / "test.tsv"
    _write_csv(csv_file, "Date\tOpen\tHigh\tLow\tClose\tVolume\n2024-01-01\t100\t105\t95\t102\t1000\n")

    df = load_csv(csv_file)
    assert len(df) == 1
    assert df["close"].iloc[0] == 102


def test_load_csv_missing_columns(tmp_path):
    csv_file = tmp_path / "bad.csv"
    _write_csv(csv_file, "foo,bar\n1,2\n")

    with pytest.raises(ValueError, match="Missing"):
        load_csv(csv_file)


def test_load_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_csv("/nonexistent/path.csv")


def test_resolve_columns_adj_close():
    df = pd.DataFrame({
        "Date": ["2024-01-01"],
        "Open": [100], "High": [105], "Low": [95],
        "Adj Close": [102], "Volume": [1000],
    })
    result = _resolve_columns(df)
    assert "close" in result.columns
    assert result["close"].iloc[0] == 102


def test_load_data_csv_path(tmp_path):
    csv_file = tmp_path / "data.csv"
    _write_csv(csv_file, "Date,Open,High,Low,Close,Volume\n2024-01-01,100,105,95,102,1000\n2024-01-02,102,107,100,105,1200\n")

    df = load_data(str(csv_file))
    assert len(df) == 2
