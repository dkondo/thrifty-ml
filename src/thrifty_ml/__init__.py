from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from thrifty_ml.embeddings import EmbeddingBackend, LiteLLMEmbeddingBackend
from thrifty_ml.engine import Engine
from thrifty_ml.proxy.base import ProxyModel

try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    __version__ = _pkg_version("thrifty-ml")
except Exception:
    __version__ = "dev"

__all__ = [
    "ml_filter",
    "ml_classify",
    "Proxy",
    "EmbeddingBackend",
    "LiteLLMEmbeddingBackend",
    "__version__",
]


def ml_filter(
    df: pd.DataFrame,
    prompt: str,
    text_column: str,
    llm: str,
    embedding_model: str | EmbeddingBackend,
    proxy: str = "lr",
    sample_size: int = 1000,
    fallback_threshold: float = 0.1,
    max_concurrency: int = 8,
    cache_dir: Path | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Return a boolean mask over df rows based on prompt.

    Args:
        df: Input DataFrame.
        prompt: Natural-language filter condition (e.g. "Is this review positive?").
        text_column: Column name containing the text to evaluate.
        llm: LiteLLM model string (e.g. "anthropic/claude-haiku-4-5").
        embedding_model: LiteLLM embedding model string.
        proxy: Proxy model type — "lr" | "svc" | "lgbm".
        sample_size: Number of rows to LLM-label for proxy training.
        fallback_threshold: τ — if proxy F1 < 1.0 - τ, fall back to full LLM.
        max_concurrency: Max simultaneous LLM API calls.
        cache_dir: Override default cache directory (~/.cache/thrifty_ml/).
        seed: Random seed for sampling reproducibility.

    Returns:
        Boolean numpy array of shape (len(df),).

    Note:
        Proxy F1 is evaluated against LLM labels on a holdout split, not against
        human ground truth. Accuracy relative to human labels may differ.
    """
    engine = Engine(
        prompt=prompt,
        llm=llm,
        embedding_model=embedding_model,
        proxy=proxy,
        sample_size=sample_size,
        fallback_threshold=fallback_threshold,
        max_concurrency=max_concurrency,
        cache_dir=cache_dir,
        seed=seed,
    )
    result = engine.run(df, text_column)
    return result.astype(bool)


def ml_classify(
    df: pd.DataFrame,
    prompt: str,
    text_column: str,
    llm: str,
    embedding_model: str | EmbeddingBackend,
    classes: list[str],
    proxy: str = "lr",
    sample_size: int = 1000,
    fallback_threshold: float = 0.1,
    max_concurrency: int = 8,
    cache_dir: Path | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Return class labels over df rows based on prompt and classes list.

    If the LLM returns a value outside `classes`, that row is labeled "__unknown__".

    Note:
        Proxy F1 is evaluated against LLM labels, not human ground truth.
    """
    if not classes:
        raise ValueError("classes must be a non-empty list.")
    engine = Engine(
        prompt=prompt,
        llm=llm,
        embedding_model=embedding_model,
        proxy=proxy,
        sample_size=sample_size,
        fallback_threshold=fallback_threshold,
        max_concurrency=max_concurrency,
        cache_dir=cache_dir,
        seed=seed,
        classes=classes,
    )
    return engine.run(df, text_column)


class Proxy:
    """Offline-mode proxy: fit on training data, predict on new data without LLM calls."""

    def __init__(
        self,
        prompt: str,
        llm: str,
        embedding_model: str | EmbeddingBackend,
        model: str = "lr",
        sample_size: int = 1000,
        fallback_threshold: float = 0.1,
        max_concurrency: int = 8,
        cache_dir: Path | None = None,
        seed: int | None = None,
        classes: list[str] | None = None,
    ):
        self._engine = Engine(
            prompt=prompt,
            llm=llm,
            embedding_model=embedding_model,
            proxy=model,
            sample_size=sample_size,
            fallback_threshold=fallback_threshold,
            max_concurrency=max_concurrency,
            cache_dir=cache_dir,
            seed=seed,
            classes=classes,
        )
        # Stored directly so predict() and load() work without a live Engine.
        self._embedding_backend: EmbeddingBackend = self._engine.embedding_backend
        self._cache_dir: Path | None = cache_dir
        self._proxy_type: str = model
        self._proxy_model: ProxyModel | None = None

    def fit(self, df: pd.DataFrame, text_column: str) -> "Proxy":
        self._proxy_model, self._eval_result = self._engine.fit(df, text_column)
        return self

    def predict(self, df: pd.DataFrame, text_column: str) -> np.ndarray:
        if self._proxy_model is None:
            raise RuntimeError("Call fit() before predict().")
        from thrifty_ml.embeddings import embed_texts
        texts = df[text_column].tolist()
        X = embed_texts(texts, self._embedding_backend, self._cache_dir)
        return self._proxy_model.predict(X)

    def save(self, path: str) -> None:
        if self._proxy_model is None:
            raise RuntimeError("Call fit() before save().")
        self._proxy_model.save(path)
        # Sidecar so load() knows the proxy type and embedding model without
        # needing caller-supplied arguments.
        meta = {
            "proxy_type": self._proxy_type,
            "embedding_model": self._embedding_backend.model_id,
        }
        Path(path + ".meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(
        cls,
        path: str,
        embedding_model: str | EmbeddingBackend | None = None,
    ) -> "Proxy":
        """Load a saved proxy.

        ``embedding_model`` is required only when the ``.meta.json`` sidecar
        written by ``save()`` is absent (e.g. files saved by an older version).
        """
        meta_path = Path(path + ".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            proxy_type = meta.get("proxy_type", "lr")
            saved_model_id = meta.get("embedding_model", "")
        else:
            proxy_type = "lr"
            saved_model_id = ""

        # Resolve embedding backend: explicit arg wins over sidecar.
        embed = embedding_model or saved_model_id
        if isinstance(embed, str):
            if not embed:
                raise ValueError(
                    "embedding_model is required when loading a proxy saved without "
                    "a .meta.json sidecar."
                )
            backend: EmbeddingBackend = LiteLLMEmbeddingBackend(embed)
        else:
            backend = embed

        # Dispatch to the proxy type's own load() so each format is handled correctly
        # (joblib for sklearn, LightGBM booster text for lgbm).
        if proxy_type == "lgbm":
            from thrifty_ml.proxy.trees import LightGBMProxy
            proxy_model: ProxyModel = LightGBMProxy.load(path)
        elif proxy_type == "svc":
            from thrifty_ml.proxy.linear import LinearSVCProxy
            proxy_model = LinearSVCProxy.load(path)
        else:
            from thrifty_ml.proxy.linear import LogisticRegressionProxy
            proxy_model = LogisticRegressionProxy.load(path)

        instance = cls.__new__(cls)
        instance._proxy_model = proxy_model
        instance._embedding_backend = backend
        instance._cache_dir = None
        instance._proxy_type = proxy_type
        instance._engine = None  # not available after load; predict() doesn't need it
        return instance
