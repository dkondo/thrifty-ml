"""Fully mocked benchmark tests — no network, no API keys needed."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

EXPECTED_METRIC_KEYS = {
    "f1_proxy_vs_llm",
    "f1_proxy_vs_gold_remainder",
    "f1_llm_vs_gold_sample",
    "relative_accuracy",
    "use_proxy",
    "fallback",
    "tokens_actual",
    "tokens_projected_full_llm",
    "token_reduction_x",
    "wall_fit_s",
    "wall_predict_s",
    "projected_full_llm_wall_s",
    "speedup_x",
    "n_rows",
    "sample_size",
    "llm",
    "embedding_model",
    "proxy",
    "git_sha",
}


def _fixture_df(n: int = 50) -> pd.DataFrame:
    """Balanced 50/50 DataFrame that stands in for the IMDB dataset."""
    return pd.DataFrame({
        "text": [f"positive text {i}" if i < n // 2 else f"negative text {i}" for i in range(n)],
        "gold": [1] * (n // 2) + [0] * (n // 2),
    })


def _fake_embedding(model, input, **kwargs):
    """Linearly separable embeddings: positive texts cluster at +1, negatives at -1."""
    rng = np.random.default_rng(42)
    dim = 32
    resp = MagicMock()
    resp.data = []
    for text in input:
        loc = 1.0 if "positive" in text else -1.0
        resp.data.append({"embedding": rng.normal(loc=loc, scale=0.3, size=dim).tolist()})
    return resp


async def _fake_acompletion_consistent(model, messages, **kwargs):
    """Returns True for 'positive', False otherwise — proxy should learn this easily."""
    content = messages[0]["content"]
    label = "positive" in content.lower() and "negative" not in content.lower()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = f'{{"label": {str(label).lower()}}}'
    resp.usage = None
    return resp


async def _fake_acompletion_random(model, messages, **kwargs):
    """Returns random labels — proxy should fail to learn and fall back."""
    import hashlib
    content = messages[0]["content"]
    label = int(hashlib.md5(content.encode()).hexdigest(), 16) % 3 == 0  # biased random
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = f'{{"label": {str(label).lower()}}}'
    resp.usage = None
    return resp


# ---------------------------------------------------------------------------
# test_benchmark_happy_path
# ---------------------------------------------------------------------------

def test_benchmark_happy_path(monkeypatch, tmp_path):
    monkeypatch.setattr("litellm.embedding", _fake_embedding)
    monkeypatch.setattr("litellm.acompletion", _fake_acompletion_consistent)
    monkeypatch.setattr("benchmarks.imdb.data.load_imdb", lambda rows, seed: _fixture_df())

    from benchmarks.imdb.run import benchmark, write_json

    out = tmp_path / "results.json"
    metrics = benchmark(
        llm="fake/model",
        embedding_model="fake/embed",
        proxy="lr",
        sample_size=10,
        rows=None,
        seed=42,
        cache_dir=tmp_path / "cache",
        max_concurrency=2,
    )

    assert EXPECTED_METRIC_KEYS.issubset(metrics.keys()), (
        f"Missing keys: {EXPECTED_METRIC_KEYS - metrics.keys()}"
    )
    assert metrics["f1_proxy_vs_llm"] > 0.8
    assert metrics["tokens_projected_full_llm"] >= metrics["tokens_actual"]

    write_json(metrics, out)
    assert out.exists()
    with open(out) as f:
        loaded = json.load(f)
    # NaN is not valid JSON; check non-NaN keys round-trip cleanly
    for k in metrics:
        if isinstance(metrics[k], float) and (metrics[k] != metrics[k]):  # NaN check
            continue
        assert loaded[k] == metrics[k], f"Mismatch on key {k!r}"


# ---------------------------------------------------------------------------
# test_benchmark_fallback
# ---------------------------------------------------------------------------

def test_benchmark_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr("litellm.embedding", _fake_embedding)
    monkeypatch.setattr("litellm.acompletion", _fake_acompletion_consistent)
    monkeypatch.setattr("benchmarks.imdb.data.load_imdb", lambda rows, seed: _fixture_df())

    # Force use_proxy=False by patching the evaluator
    from thrifty_ml.evaluator import EvalResult

    original_evaluate = None

    def mock_evaluate(proxy, X_train, y_train, X_holdout, y_holdout, *args, **kwargs):
        proxy.fit(X_train, y_train)  # still fit so proxy_model is usable if needed
        return EvalResult(proxy_f1=0.0, llm_f1=1.0, use_proxy=False, holdout_size=len(y_holdout))

    monkeypatch.setattr("thrifty_ml.engine.evaluate", mock_evaluate)

    acompletion_calls: list[str] = []
    original_acompletion = _fake_acompletion_consistent

    async def tracked_acompletion(model, messages, **kwargs):
        acompletion_calls.append(messages[0]["content"])
        return await original_acompletion(model, messages, **kwargs)

    monkeypatch.setattr("litellm.acompletion", tracked_acompletion)

    from benchmarks.imdb.run import benchmark

    metrics = benchmark(
        llm="fake/model",
        embedding_model="fake/embed",
        proxy="lr",
        sample_size=10,
        rows=None,
        seed=42,
        cache_dir=tmp_path / "cache",
        max_concurrency=2,
    )

    assert metrics["fallback"] is True
    assert metrics["use_proxy"] is False

    # The remainder is 50 - 10 = 40 rows; acompletion must have been called for them.
    # Sample labels (10) come from Engine.fit; remainder labels (40) from the fallback path.
    # Cache hits from the trailing sample-rereads cost 0 extra calls.
    assert len(acompletion_calls) >= 40, (
        f"Expected ≥40 acompletion calls for the remainder; got {len(acompletion_calls)}"
    )


# ---------------------------------------------------------------------------
# test_meter_captures_usage
# ---------------------------------------------------------------------------

def test_meter_captures_usage(monkeypatch):
    import asyncio
    import litellm
    from benchmarks.imdb.instrumentation import instrument

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 2
    usage.total_tokens = 12

    resp = MagicMock()
    resp.usage = usage

    async def fake_acompletion(*args, **kwargs):
        return resp

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with instrument() as meter:
        asyncio.run(litellm.acompletion(model="fake", messages=[]))

    assert meter.prompt_tokens == 10
    assert meter.completion_tokens == 2
    assert meter.total_tokens == 12
    assert meter.n_llm_calls == 1
    assert meter.llm_wall_s >= 0.0


# ---------------------------------------------------------------------------
# test_meter_restores_callbacks
# ---------------------------------------------------------------------------

def test_meter_restores_callbacks():
    import litellm
    from benchmarks.imdb.instrumentation import instrument

    original_acompletion = litellm.acompletion

    try:
        with instrument():
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert litellm.acompletion is original_acompletion, (
        "instrument() did not restore litellm.acompletion after exception"
    )


# ---------------------------------------------------------------------------
# test_load_imdb_balance
# ---------------------------------------------------------------------------

def test_load_imdb_balance(monkeypatch):
    import sys
    import types

    rows_data = {
        "text": [f"text {i}" for i in range(100)],
        "label": [i % 2 for i in range(100)],
    }

    class FakeDataset:
        def to_pandas(self):
            return pd.DataFrame(rows_data)

    fake_ds = types.ModuleType("datasets")
    fake_ds.load_dataset = lambda *a, **kw: FakeDataset()
    monkeypatch.setitem(sys.modules, "datasets", fake_ds)

    from benchmarks.imdb.data import load_imdb

    df = load_imdb(rows=10, seed=0)
    assert len(df) == 10
    assert (df["gold"] == 0).sum() == 5
    assert (df["gold"] == 1).sum() == 5
