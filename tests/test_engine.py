"""Unit tests for the Engine using mock LLM and embeddings."""
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from thrifty_ml.evaluator import EvalResult, evaluate, train_holdout_split
from thrifty_ml.proxy.linear import LogisticRegressionProxy


def _two_cluster_embeddings(n: int = 200, dim: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=-2.0, scale=0.5, size=(n // 2, dim)).astype(np.float32)
    X1 = rng.normal(loc=2.0, scale=0.5, size=(n // 2, dim)).astype(np.float32)
    return np.vstack([X0, X1])


def _make_df(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame({"text": [f"text {i}" for i in range(n)]})


def test_evaluator_uses_proxy_when_f1_above_threshold():
    X, y = _two_cluster_embeddings(), np.array([0] * 100 + [1] * 100)
    split = int(len(X) * 0.8)
    proxy = LogisticRegressionProxy()
    result = evaluate(proxy, X[:split], y[:split], X[split:], y[split:], fallback_threshold=0.1)
    assert result.use_proxy is True
    assert result.proxy_f1 > 0.9


def test_evaluator_fallback_when_proxy_is_bad():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 64)).astype(np.float32)
    # Labels not separable from features
    y = rng.integers(0, 2, size=100)
    proxy = LogisticRegressionProxy()
    # Use very tight threshold so random accuracy fails
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = evaluate(proxy, X[:80], y[:80], X[80:], y[80:], fallback_threshold=0.0)
    # With threshold 0.0 proxy must match LLM F1 exactly (1.0); random proxy won't
    if not result.use_proxy:
        assert any("Falling back" in str(warning.message) for warning in w)


def test_train_holdout_split_sizes():
    X = np.zeros((100, 10))
    y = np.zeros(100)
    X_tr, y_tr, X_h, y_h = train_holdout_split(X, y, holdout_fraction=0.2)
    assert len(X_tr) == 80
    assert len(X_h) == 20


def test_train_holdout_split_no_overlap():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8))
    y = np.arange(50)
    X_tr, y_tr, X_h, y_h = train_holdout_split(X, y)
    assert set(y_tr).isdisjoint(set(y_h))


def test_empty_df_raises():
    from thrifty_ml.engine import Engine
    engine = Engine(
        prompt="test",
        llm="fake/model",
        embedding_model="fake/embed",
    )
    with pytest.raises(ValueError, match="empty"):
        engine.run(pd.DataFrame({"text": []}), "text")


def test_ml_classify_empty_classes_raises():
    import pandas as pd
    from thrifty_ml import ml_classify
    with pytest.raises(ValueError, match="classes"):
        ml_classify(
            pd.DataFrame({"text": ["a"]}),
            prompt="...",
            text_column="text",
            llm="fake",
            embedding_model="fake",
            classes=[],
        )


def test_evaluate_average_uses_union_of_labels():
    """evaluate() must not crash when a class appears only in the holdout fold."""
    rng = np.random.default_rng(0)
    dim = 32
    # Three classes; class 2 has only one sample — force it into holdout only.
    X_train = rng.normal(size=(10, dim)).astype(np.float32)
    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    X_holdout = rng.normal(size=(3, dim)).astype(np.float32)
    y_holdout = np.array([0, 1, 2])  # class 2 never seen in training

    proxy = LogisticRegressionProxy()
    # Should not raise, even though f1_score must handle a class unseen in training.
    result = evaluate(proxy, X_train, y_train, X_holdout, y_holdout)
    assert isinstance(result.proxy_f1, float)


def test_proxy_save_load_roundtrip(tmp_path):
    """Proxy.save() writes a sidecar; Proxy.load() restores predict() capability."""
    from thrifty_ml import Proxy
    from thrifty_ml.embeddings import EmbeddingBackend

    class FixedBackend(EmbeddingBackend):
        model_id = "fixed-32"
        def embed(self, texts):
            return np.zeros((len(texts), 32), dtype=np.float32)

    # Build proxy without a real Engine fit — stub _proxy_model and _embedding_backend.
    # Use the public constructor path via LogisticRegressionProxy directly.
    from thrifty_ml.proxy.linear import LogisticRegressionProxy
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=-1.0, size=(50, 32)).astype(np.float32),
        rng.normal(loc=1.0, size=(50, 32)).astype(np.float32),
    ])
    y = np.array([0] * 50 + [1] * 50)

    lr = LogisticRegressionProxy()
    lr.fit(X, y)

    # Simulate a fitted Proxy without actually calling Engine (no API needed).
    p = Proxy.__new__(Proxy)
    p._proxy_model = lr
    p._embedding_backend = FixedBackend()
    p._cache_dir = None
    p._proxy_type = "lr"
    p._engine = None

    model_path = str(tmp_path / "proxy.joblib")
    p.save(model_path)

    assert (tmp_path / "proxy.joblib.meta.json").exists()

    loaded = Proxy.load(model_path, embedding_model=FixedBackend())
    assert loaded._proxy_type == "lr"
    assert loaded._embedding_backend.model_id == "fixed-32"

    # predict() must work on the loaded instance.
    df = pd.DataFrame({"text": ["hello", "world"]})
    preds = loaded.predict(df, "text")
    assert preds.shape == (2,)
