from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from frugal_ml.proxy.base import ProxyModel


class LogisticRegressionProxy(ProxyModel):
    def __init__(self, max_iter: int = 1000, C: float = 1.0, **kwargs):
        self._model = LogisticRegression(
            class_weight="balanced", max_iter=max_iter, C=C, **kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def supports_proba(self) -> bool:
        return True


class LinearSVCProxy(ProxyModel):
    def __init__(self, C: float = 1.0, max_iter: int = 2000, **kwargs):
        self._model = LinearSVC(
            class_weight="balanced", C=C, max_iter=max_iter, **kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
