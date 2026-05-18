from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import diskcache
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    _VERSION = _pkg_version("frugal-ml")
except PackageNotFoundError:
    _VERSION = "dev"

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "frugal_ml"
_SHARDS = 8

_CACHES: dict[Path, diskcache.FanoutCache] = {}


def _get_cache(cache_dir: Path) -> diskcache.FanoutCache:
    if cache_dir not in _CACHES:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _CACHES[cache_dir] = diskcache.FanoutCache(
            str(cache_dir), shards=_SHARDS, timeout=60
        )
    return _CACHES[cache_dir]


def _vkey(*parts: str) -> str:
    return f"frugal_ml/{_VERSION}/" + "/".join(parts)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def get_embedding(
    text: str, model_id: str, cache_dir: Path | None = None
) -> Any | None:
    cache = _get_cache(cache_dir or _DEFAULT_CACHE_DIR)
    return cache.get(_vkey("emb", _sha256(text), model_id))


def set_embedding(
    text: str, model_id: str, value: Any, cache_dir: Path | None = None
) -> None:
    cache = _get_cache(cache_dir or _DEFAULT_CACHE_DIR)
    cache.set(_vkey("emb", _sha256(text), model_id), value)


def _classes_key(classes: list[str] | None) -> str:
    if classes is None:
        return "binary"
    return _sha256(",".join(sorted(classes)))


def get_label(
    text: str,
    prompt: str,
    model_id: str,
    cache_dir: Path | None = None,
    classes: list[str] | None = None,
) -> Any | None:
    cache = _get_cache(cache_dir or _DEFAULT_CACHE_DIR)
    return cache.get(_vkey("lbl", _sha256(text), _sha256(prompt), model_id, _classes_key(classes)))


def set_label(
    text: str,
    prompt: str,
    model_id: str,
    value: Any,
    cache_dir: Path | None = None,
    classes: list[str] | None = None,
) -> None:
    cache = _get_cache(cache_dir or _DEFAULT_CACHE_DIR)
    cache.set(_vkey("lbl", _sha256(text), _sha256(prompt), model_id, _classes_key(classes)), value)


def clear_cache(cache_dir: Path | None = None) -> int:
    target = cache_dir or _DEFAULT_CACHE_DIR
    if target in _CACHES:
        del _CACHES[target]
    cache = _get_cache(target)
    count = len(cache)
    cache.clear()
    return count
