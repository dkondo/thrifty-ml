from __future__ import annotations

import asyncio
import warnings
from pathlib import Path
from typing import Any, Coroutine

import numpy as np
import pandas as pd

from frugal_ml import embeddings as _emb
from frugal_ml import llm as _llm
from frugal_ml.embeddings import EmbeddingBackend, LiteLLMEmbeddingBackend
from frugal_ml.evaluator import evaluate, train_holdout_split
from frugal_ml.proxy.base import ProxyModel
from frugal_ml.proxy.linear import LogisticRegressionProxy, LinearSVCProxy
from frugal_ml.sampling import random_sample

def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine safely whether or not an event loop is already running.

    _run_async() raises RuntimeError inside Jupyter / async frameworks.
    When a loop is already running we use nest_asyncio if available, otherwise
    we schedule the coroutine on that loop and block until it completes.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro)

    # Running loop detected (Jupyter, FastAPI, etc.)
    try:
        import nest_asyncio  # noqa: PLC0415
        nest_asyncio.apply(loop)
        return loop.run_until_complete(coro)
    except ImportError:
        pass

    # nest_asyncio not available — run in a new thread with its own loop so we
    # don't block the existing one.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


_PROXY_REGISTRY: dict[str, type[ProxyModel]] = {
    "lr": LogisticRegressionProxy,
    "svc": LinearSVCProxy,
}


def _get_proxy(name: str) -> ProxyModel:
    if name == "lgbm":
        from frugal_ml.proxy.trees import LightGBMProxy
        return LightGBMProxy()
    if name not in _PROXY_REGISTRY:
        raise ValueError(f"Unknown proxy '{name}'. Choose from: lr, svc, lgbm")
    return _PROXY_REGISTRY[name]()


class Engine:
    def __init__(
        self,
        prompt: str,
        llm: str,
        embedding_model: str | EmbeddingBackend,
        proxy: str = "lr",
        sample_size: int = 1000,
        fallback_threshold: float = 0.1,
        max_concurrency: int = 8,
        holdout_fraction: float = 0.2,
        cache_dir: Path | None = None,
        seed: int | None = None,
        classes: list[str] | None = None,
    ):
        self.prompt = prompt
        self.llm = llm
        if isinstance(embedding_model, str):
            self.embedding_backend: EmbeddingBackend = LiteLLMEmbeddingBackend(embedding_model)
        else:
            self.embedding_backend = embedding_model
        self.embedding_model = self.embedding_backend.model_id
        self.proxy_name = proxy
        self.sample_size = sample_size
        self.fallback_threshold = fallback_threshold
        self.max_concurrency = max_concurrency
        self.holdout_fraction = holdout_fraction
        self.cache_dir = cache_dir
        self.seed = seed
        self.classes = classes

    def run(self, df: pd.DataFrame, text_column: str) -> np.ndarray:
        if len(df) == 0:
            raise ValueError("Input DataFrame is empty.")

        texts = df[text_column].tolist()

        embeddings = _emb.embed_texts(texts, self.embedding_backend, self.cache_dir)

        sample_df, remainder_df = random_sample(df, self.sample_size, self.seed)

        sample_texts = sample_df[text_column].tolist()
        labels_raw = _run_async(
            _llm.label_texts(
                sample_texts,
                self.prompt,
                self.llm,
                classes=self.classes,
                max_concurrency=self.max_concurrency,
                cache_dir=self.cache_dir,
            )
        )

        if remainder_df.empty:
            # Entire dataset was sampled — return LLM labels directly
            return np.array(labels_raw)

        sample_idx = sample_df.index
        y_sample = np.array(labels_raw)
        X_sample = embeddings[df.index.get_indexer(sample_idx)]

        X_train, y_train, X_holdout, y_holdout = train_holdout_split(
            X_sample, y_sample, self.holdout_fraction, self.seed
        )

        proxy_model = _get_proxy(self.proxy_name)
        eval_result = evaluate(
            proxy_model, X_train, y_train, X_holdout, y_holdout, self.fallback_threshold
        )

        remainder_idx = remainder_df.index
        X_remainder = embeddings[df.index.get_indexer(remainder_idx)]

        if eval_result.use_proxy:
            remainder_preds = proxy_model.predict(X_remainder)
        else:
            remainder_preds = np.array(
                _run_async(
                    _llm.label_texts(
                        remainder_df[text_column].tolist(),
                        self.prompt,
                        self.llm,
                        classes=self.classes,
                        max_concurrency=self.max_concurrency,
                        cache_dir=self.cache_dir,
                    )
                )
            )

        result = np.empty(len(df), dtype=object)
        result[df.index.get_indexer(sample_idx)] = y_sample
        result[df.index.get_indexer(remainder_idx)] = remainder_preds
        return result

    def fit(self, df: pd.DataFrame, text_column: str) -> tuple[ProxyModel, "EvalResult"]:
        """Offline mode: fit proxy and return it without predicting remainder."""
        from frugal_ml.evaluator import EvalResult

        if len(df) == 0:
            raise ValueError("Input DataFrame is empty.")

        texts = df[text_column].tolist()
        embeddings = _emb.embed_texts(texts, self.embedding_backend, self.cache_dir)

        sample_df, _ = random_sample(df, self.sample_size, self.seed)
        sample_texts = sample_df[text_column].tolist()
        labels_raw = _run_async(
            _llm.label_texts(
                sample_texts,
                self.prompt,
                self.llm,
                classes=self.classes,
                max_concurrency=self.max_concurrency,
                cache_dir=self.cache_dir,
            )
        )

        sample_idx = sample_df.index
        y_sample = np.array(labels_raw)
        X_sample = embeddings[df.index.get_indexer(sample_idx)]

        X_train, y_train, X_holdout, y_holdout = train_holdout_split(
            X_sample, y_sample, self.holdout_fraction, self.seed
        )

        proxy_model = _get_proxy(self.proxy_name)
        eval_result = evaluate(
            proxy_model, X_train, y_train, X_holdout, y_holdout, self.fallback_threshold
        )

        return proxy_model, eval_result
