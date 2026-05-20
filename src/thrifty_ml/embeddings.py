from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from thrifty_ml import cache as _cache

_CHUNK_SIZE = 512
_LARGE_DF_THRESHOLD = 250_000


class EmbeddingBackend(ABC):
    """ABC for embedding backends.

    Implement this to bring your own embeddings — sentence-transformers,
    pre-computed vectors, custom API wrappers, etc.  The ``model_id``
    property is used as the diskcache key, so keep it stable across runs.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable identifier used as the cache key."""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dim)."""
        ...


class LiteLLMEmbeddingBackend(EmbeddingBackend):
    """Embedding backend that delegates to any provider via LiteLLM."""

    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def model_id(self) -> str:
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        import litellm

        results: list[np.ndarray] = []
        for start in range(0, len(texts), _CHUNK_SIZE):
            chunk = texts[start : start + _CHUNK_SIZE]
            response = litellm.embedding(model=self._model, input=chunk)
            for item in response.data:
                results.append(np.array(item["embedding"], dtype=np.float32))
        return np.array(results, dtype=np.float32)


def embed_texts(
    texts: list[str],
    backend: EmbeddingBackend,
    cache_dir: Path | None = None,
) -> np.ndarray:
    if len(texts) > _LARGE_DF_THRESHOLD:
        warnings.warn(
            f"Embedding {len(texts)} texts. At typical embedding dimensions this "
            "may require several GB of RAM. Consider chunking your input.",
            UserWarning,
            stacklevel=2,
        )

    model_id = backend.model_id
    cached: list[np.ndarray | None] = [
        _cache.get_embedding(t, model_id, cache_dir) for t in texts
    ]

    miss_indices = [i for i, v in enumerate(cached) if v is None]
    miss_texts = [texts[i] for i in miss_indices]

    if miss_texts:
        new_vecs = backend.embed(miss_texts)
        for i, idx in enumerate(miss_indices):
            vec = new_vecs[i]
            _cache.set_embedding(texts[idx], model_id, vec, cache_dir)
            cached[idx] = vec

    return np.array(cached, dtype=np.float32)
