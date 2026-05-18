"""CLI tests using typer's test runner."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from frugal_ml.cli import app

runner = CliRunner()


def _write_parquet(path: Path, n: int = 20) -> pd.DataFrame:
    df = pd.DataFrame({"text": [f"row {i}" for i in range(n)]})
    df.to_parquet(path, index=False)
    return df


@pytest.fixture
def mock_pipeline(monkeypatch):
    """Patch ml_filter and ml_classify so CLI tests need no real API calls."""
    rng = np.random.default_rng(0)

    def fake_filter(df, **kwargs):
        return np.array([True, False] * (len(df) // 2) + [True] * (len(df) % 2))

    def fake_classify(df, classes, **kwargs):
        return np.array([classes[i % len(classes)] for i in range(len(df))])

    monkeypatch.setattr("frugal_ml.ml_filter", fake_filter)
    monkeypatch.setattr("frugal_ml.ml_classify", fake_classify)


def test_filter_command(tmp_path, mock_pipeline):
    input_path = tmp_path / "input.parquet"
    out_path = tmp_path / "out.parquet"
    _write_parquet(input_path)

    result = runner.invoke(
        app,
        [
            "filter",
            str(input_path),
            "--prompt", "Is this relevant?",
            "--text-col", "text",
            "--out", str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    out_df = pd.read_parquet(out_path)
    assert len(out_df) > 0
    assert "_frugal_mask" in out_df.columns


def test_classify_command(tmp_path, mock_pipeline):
    input_path = tmp_path / "input.parquet"
    out_path = tmp_path / "out.parquet"
    _write_parquet(input_path)

    result = runner.invoke(
        app,
        [
            "classify",
            str(input_path),
            "--prompt", "Classify this.",
            "--text-col", "text",
            "--classes", "a,b,c",
            "--out", str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    out_df = pd.read_parquet(out_path)
    assert "label" in out_df.columns


def test_embed_command(tmp_path, monkeypatch):
    import litellm

    def fake_embedding(model, input, **kwargs):
        resp = MagicMock()
        resp.data = [{"embedding": [0.1] * 8} for _ in input]
        return resp

    monkeypatch.setattr("litellm.embedding", fake_embedding)

    input_path = tmp_path / "input.parquet"
    out_path = tmp_path / "out.npy"
    _write_parquet(input_path, n=5)

    result = runner.invoke(
        app,
        [
            "embed",
            str(input_path),
            "--text-col", "text",
            "--model", "fake-model",
            "--out", str(out_path),
            "--cache-dir", str(tmp_path / "cache"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    arr = np.load(str(out_path))
    assert arr.shape == (5, 8)


def test_cache_clear_command(tmp_path):
    result = runner.invoke(
        app,
        ["cache", "clear", "--cache-dir", str(tmp_path / "cache")],
    )
    assert result.exit_code == 0, result.output
    assert "Cleared" in result.output


def test_filter_missing_prompt_errors(tmp_path):
    input_path = tmp_path / "input.parquet"
    _write_parquet(input_path)
    result = runner.invoke(
        app,
        ["filter", str(input_path), "--text-col", "text", "--out", str(tmp_path / "out.parquet")],
    )
    assert result.exit_code != 0
