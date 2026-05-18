"""Mock-API integration tests — CI-safe, no real API keys needed.

Real API tests are marked @pytest.mark.integration and skipped in CI.
"""
import asyncio
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _fake_embedding_response(texts: list[str], dim: int = 64) -> MagicMock:
    rng = np.random.default_rng(abs(hash(str(texts))) % (2**32))
    response = MagicMock()
    response.data = [
        {"embedding": rng.normal(size=dim).tolist()} for _ in texts
    ]
    return response


def _make_df(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({"text": [f"sample text {i}" for i in range(n)]})


@pytest.fixture
def mock_litellm(monkeypatch):
    """Patch litellm so no real API calls are made."""
    call_counts = {"embed": 0, "label": 0}

    def fake_embedding(model, input, **kwargs):
        call_counts["embed"] += 1
        # Deterministic: positive texts get cluster A, others cluster B
        rng = np.random.default_rng(42)
        dim = 64
        response = MagicMock()
        response.data = []
        for text in input:
            vec = rng.normal(loc=(1.0 if "positive" in text else -1.0), scale=0.3, size=dim)
            response.data.append({"embedding": vec.tolist()})
        return response

    async def fake_acompletion(model, messages, **kwargs):
        call_counts["label"] += 1
        content = messages[0]["content"]
        # "negative text" → False; "positive text" → True; others → hash parity
        if "negative text" in content.lower():
            label = False
        elif "positive text" in content.lower():
            label = True
        else:
            # Hash the content so different texts get different labels deterministically
            label = int(hashlib.md5(content.encode()).hexdigest(), 16) % 2 == 0
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = f'{{"label": {str(label).lower()}}}'
        return response

    monkeypatch.setattr("litellm.embedding", fake_embedding)
    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    return call_counts


def test_ml_filter_full_pipeline(mock_litellm, tmp_path):
    from thrifty_ml import ml_filter

    texts = (
        ["this is positive text"] * 50
        + ["this is negative text"] * 50
    )
    df = pd.DataFrame({"text": texts})

    mask = ml_filter(
        df,
        prompt="Is this text positive?",
        text_column="text",
        llm="fake/model",
        embedding_model="fake/embed",
        proxy="lr",
        sample_size=60,
        cache_dir=tmp_path / "cache",
        seed=42,
    )

    assert mask.shape == (100,)
    assert mask.dtype == bool
    # Most positives should pass, most negatives should not
    assert mask[:50].sum() > 30
    assert mask[50:].sum() < 20


def test_cache_prevents_second_embed_call(mock_litellm, tmp_path):
    """Second run with same args and cache_dir should not re-embed."""
    from thrifty_ml import ml_filter

    df = _make_df(50)
    kwargs = dict(
        prompt="Is this relevant?",
        text_column="text",
        llm="fake/model",
        embedding_model="fake/embed",
        proxy="lr",
        sample_size=30,
        cache_dir=tmp_path / "cache",
        seed=0,
    )

    ml_filter(df, **kwargs)
    first_embed_count = mock_litellm["embed"]

    # Second call: embeddings should all be cache hits
    ml_filter(df, **kwargs)
    assert mock_litellm["embed"] == first_embed_count, (
        "embed was called again despite cache — expected 0 new calls"
    )


def test_ml_filter_output_length_matches_input(mock_litellm, tmp_path):
    from thrifty_ml import ml_filter

    df = _make_df(80)
    mask = ml_filter(
        df,
        prompt="test",
        text_column="text",
        llm="fake",
        embedding_model="fake",
        proxy="lr",
        sample_size=40,
        cache_dir=tmp_path / "cache",
        seed=1,
    )
    assert len(mask) == 80


def test_ml_classify_unknown_label(monkeypatch, tmp_path):
    """LLM returning a value outside classes → __unknown__."""
    import json

    async def fake_acompletion(model, messages, **kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = '{"label": "garbage"}'
        return resp

    def fake_embedding(model, input, **kwargs):
        rng = np.random.default_rng(0)
        resp = MagicMock()
        resp.data = [{"embedding": rng.normal(size=32).tolist()} for _ in input]
        return resp

    monkeypatch.setattr("litellm.embedding", fake_embedding)
    monkeypatch.setattr("litellm.acompletion", fake_acompletion)

    from thrifty_ml import ml_classify

    df = pd.DataFrame({"text": ["a", "b", "c", "d", "e"] * 4})
    labels = ml_classify(
        df,
        prompt="classify",
        text_column="text",
        llm="fake",
        embedding_model="fake",
        classes=["cat", "dog"],
        sample_size=10,
        cache_dir=tmp_path / "cache",
        seed=0,
    )
    assert "__unknown__" in labels


def test_sample_size_exceeds_df_returns_llm_labels_directly(mock_litellm, tmp_path):
    """When sample_size >= len(df), entire df is labeled by LLM directly."""
    import warnings
    from thrifty_ml import ml_filter

    df = pd.DataFrame({"text": ["positive text"] * 10})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mask = ml_filter(
            df,
            prompt="positive?",
            text_column="text",
            llm="fake",
            embedding_model="fake",
            proxy="lr",
            sample_size=100,  # > len(df)
            cache_dir=tmp_path / "cache",
        )
    assert mask.shape == (10,)
    assert any("sample_size" in str(warning.message) for warning in w)


def test_custom_embedding_backend(monkeypatch, tmp_path):
    """Users can pass an EmbeddingBackend subclass instead of a model string."""
    import hashlib
    from thrifty_ml.embeddings import EmbeddingBackend
    from thrifty_ml import ml_filter

    class DeterministicBackend(EmbeddingBackend):
        """Returns a deterministic embedding based on text content."""

        @property
        def model_id(self) -> str:
            return "deterministic-test-backend"

        def embed(self, texts: list[str]) -> np.ndarray:
            out = []
            for t in texts:
                seed = int(hashlib.md5(t.encode()).hexdigest(), 16) % (2**32)
                rng = np.random.default_rng(seed)
                loc = 1.0 if "positive" in t else -1.0
                out.append(rng.normal(loc=loc, scale=0.3, size=32).astype(np.float32))
            return np.array(out, dtype=np.float32)

    async def fake_acompletion(model, messages, **kwargs):
        from unittest.mock import MagicMock
        content = messages[0]["content"]
        label = "positive" in content.lower() and "negative" not in content.lower()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = f'{{"label": {str(label).lower()}}}'
        return resp

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)

    texts = ["this is positive text"] * 30 + ["this is negative text"] * 30
    df = pd.DataFrame({"text": texts})

    mask = ml_filter(
        df,
        prompt="Is this positive?",
        text_column="text",
        llm="fake/model",
        embedding_model=DeterministicBackend(),
        proxy="lr",
        sample_size=40,
        cache_dir=tmp_path / "cache",
        seed=0,
    )

    assert mask.shape == (60,)
    assert mask.dtype == bool
    assert mask[:30].sum() > 20
    assert mask[30:].sum() < 10


@pytest.mark.integration
def test_real_api_ml_filter():
    """Requires ANTHROPIC_API_KEY and OPENAI_API_KEY. Not run in CI."""
    import os
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Real API keys not available")

    from thrifty_ml import ml_filter

    texts = (
        ["I loved this movie, brilliant cinematography!"] * 20
        + ["The plot was boring and predictable."] * 20
    )
    df = pd.DataFrame({"text": texts})
    mask = ml_filter(
        df,
        prompt="Is this a positive review?",
        text_column="text",
        llm="anthropic/claude-haiku-4-5",
        embedding_model="text-embedding-3-small",
        sample_size=20,
        seed=42,
    )
    assert mask.shape == (40,)
    # Expect most positives labeled True
    assert mask[:20].sum() > 15
    assert mask[20:].sum() < 5
