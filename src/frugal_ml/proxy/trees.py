from __future__ import annotations

import numpy as np

from frugal_ml.proxy.base import ProxyModel


class LightGBMProxy(ProxyModel):
    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.1, **kwargs):
        try:
            import lightgbm as lgb
        except (ImportError, OSError) as exc:
            raise ImportError(
                "LightGBM is not installed or cannot load its native library. "
                "Install it with: pip install frugal-ml[lgbm]"
            ) from exc
        self._lgb = lgb
        self._params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            is_unbalance=True,
            verbosity=-1,
            **kwargs,
        )
        self._model: "lgb.LGBMClassifier | None" = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model = self._lgb.LGBMClassifier(**self._params)
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Must call fit() before predict()")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Must call fit() before predict_proba()")
        return self._model.predict_proba(X)

    @property
    def supports_proba(self) -> bool:
        return True

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("Must call fit() before save()")
        self._model.booster_.save_model(path)

    @classmethod
    def load(cls, path: str) -> "LightGBMProxy":
        try:
            import lightgbm as lgb
        except (ImportError, OSError) as exc:
            raise ImportError(
                "LightGBM is not installed or cannot load its native library. "
                "Install it with: pip install frugal-ml[lgbm]"
            ) from exc
        proxy = cls.__new__(cls)
        proxy._lgb = lgb
        proxy._params = {}
        proxy._model = lgb.LGBMClassifier()
        proxy._model._Booster = lgb.Booster(model_file=path)
        return proxy
