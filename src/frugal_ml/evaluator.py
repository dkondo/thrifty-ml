from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from frugal_ml.proxy.base import ProxyModel


@dataclass
class EvalResult:
    proxy_f1: float
    llm_f1: float
    use_proxy: bool
    holdout_size: int


def evaluate(
    proxy: ProxyModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    fallback_threshold: float = 0.1,
    holdout_fraction: float = 0.2,
) -> EvalResult:
    """Fit proxy on train split, evaluate on holdout vs LLM labels."""
    if len(np.unique(y_train)) < 2:
        # Single-class sample — proxy can't learn; fall back to LLM
        warnings.warn(
            "Labeled sample contains only one class. Cannot train proxy. "
            "Falling back to LLM for prediction.",
            UserWarning,
            stacklevel=2,
        )
        return EvalResult(
            proxy_f1=0.0,
            llm_f1=1.0,
            use_proxy=False,
            holdout_size=len(y_holdout),
        )

    proxy.fit(X_train, y_train)

    # Base average on the union of train+holdout labels so a class that
    # only appears in the holdout fold doesn't trigger "binary" mode and
    # then crash f1_score when y_holdout has >2 unique values.
    all_classes = np.unique(np.concatenate([y_train, y_holdout]))
    average = "binary" if len(all_classes) <= 2 else "macro"

    proxy_preds = proxy.predict(X_holdout)
    proxy_f1 = float(f1_score(y_holdout, proxy_preds, average=average, zero_division=0))

    # LLM F1 is trivially 1.0 when we treat LLM labels as ground truth.
    # In practice the LLM labels ARE our ground truth — so we use 1.0 as the
    # reference, meaning the proxy just needs to be within tau of perfect mimicry.
    llm_f1 = 1.0

    use_proxy = proxy_f1 >= llm_f1 - fallback_threshold

    if not use_proxy:
        warnings.warn(
            f"Proxy F1={proxy_f1:.3f} is below threshold "
            f"(LLM F1={llm_f1:.3f} - τ={fallback_threshold}). "
            "Falling back to LLM for prediction.",
            UserWarning,
            stacklevel=2,
        )

    return EvalResult(
        proxy_f1=proxy_f1,
        llm_f1=llm_f1,
        use_proxy=use_proxy,
        holdout_size=len(y_holdout),
    )


def train_holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    holdout_fraction: float = 0.2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split labeled sample into train/holdout for proxy evaluation."""
    n_classes = len(np.unique(y))
    n = len(y)
    n_holdout = max(1, int(n * holdout_fraction))
    # Use stratified split when possible (≥2 samples per class).
    # Falls back to random permutation when the sample is too small to stratify.
    min_class_count = int(np.min(np.unique(y, return_counts=True)[1]))
    if n_classes >= 2 and min_class_count >= 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n_holdout, random_state=seed)
        train_idx, holdout_idx = next(sss.split(X, y))
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        holdout_idx, train_idx = idx[:n_holdout], idx[n_holdout:]
    return X[train_idx], y[train_idx], X[holdout_idx], y[holdout_idx]
