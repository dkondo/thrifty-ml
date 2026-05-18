"""Proxy model tests using synthetic Gaussian clusters."""
import numpy as np
import pytest
from sklearn.metrics import f1_score

from frugal_ml.proxy.linear import LinearSVCProxy, LogisticRegressionProxy


def _make_clusters(n: int = 500, dim: int = 384, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=-1.0, scale=0.5, size=(n // 2, dim)).astype(np.float32)
    X1 = rng.normal(loc=1.0, scale=0.5, size=(n // 2, dim)).astype(np.float32)
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


@pytest.fixture
def cluster_data():
    return _make_clusters()


def test_logistic_regression_f1(cluster_data):
    X, y = cluster_data
    split = len(X) * 4 // 5
    proxy = LogisticRegressionProxy()
    proxy.fit(X[:split], y[:split])
    preds = proxy.predict(X[split:])
    score = f1_score(y[split:], preds)
    assert score > 0.95, f"LR F1={score:.3f} < 0.95"


def test_linear_svc_f1(cluster_data):
    X, y = cluster_data
    split = len(X) * 4 // 5
    proxy = LinearSVCProxy()
    proxy.fit(X[:split], y[:split])
    preds = proxy.predict(X[split:])
    score = f1_score(y[split:], preds)
    assert score > 0.95, f"LinearSVC F1={score:.3f} < 0.95"


def test_lgbm_f1(cluster_data):
    try:
        import lightgbm  # noqa: F401
    except (ImportError, OSError):
        pytest.skip("lightgbm not available (missing libomp or not installed)")
    from frugal_ml.proxy.trees import LightGBMProxy
    X, y = cluster_data
    split = len(X) * 4 // 5
    proxy = LightGBMProxy()
    proxy.fit(X[:split], y[:split])
    preds = proxy.predict(X[split:])
    score = f1_score(y[split:], preds)
    assert score > 0.95, f"LightGBM F1={score:.3f} < 0.95"


def test_lgbm_import_error_without_package():
    """LightGBMProxy raises ImportError with helpful message if lightgbm not loadable."""
    import sys
    lgb = sys.modules.pop("lightgbm", None)
    # Also clear any submodules that may be cached
    to_remove = [k for k in sys.modules if k.startswith("lightgbm")]
    for k in to_remove:
        sys.modules.pop(k, None)
    try:
        # Reload to clear cached imports within the module
        import importlib
        import frugal_ml.proxy.trees as trees_mod
        importlib.reload(trees_mod)
        from frugal_ml.proxy.trees import LightGBMProxy
        with pytest.raises(ImportError, match="pip install frugal-ml"):
            LightGBMProxy()
    finally:
        if lgb is not None:
            sys.modules["lightgbm"] = lgb


def test_lr_supports_proba():
    proxy = LogisticRegressionProxy()
    assert proxy.supports_proba is True
    X, y = _make_clusters(n=100, dim=32)
    proxy.fit(X, y)
    proba = proxy.predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_svc_no_proba():
    proxy = LinearSVCProxy()
    assert proxy.supports_proba is False
    with pytest.raises(NotImplementedError):
        proxy.predict_proba(np.zeros((5, 32)))


def test_lr_save_load(tmp_path):
    import joblib
    X, y = _make_clusters(n=100, dim=32)
    proxy = LogisticRegressionProxy()
    proxy.fit(X, y)
    path = str(tmp_path / "model.joblib")
    proxy.save(path)
    loaded = LogisticRegressionProxy.load(path)
    assert np.array_equal(proxy.predict(X), loaded.predict(X))
