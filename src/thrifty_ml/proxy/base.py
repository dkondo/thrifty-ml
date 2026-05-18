from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ProxyModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} does not support predict_proba")

    @property
    def supports_proba(self) -> bool:
        return False

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "ProxyModel":
        import joblib
        return joblib.load(path)
