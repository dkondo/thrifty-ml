from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")

_DEFAULT_RETRY_DELAYS = (1.0, 2.0, 4.0)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


async def _litellm_call_with_retry(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    retry_delays: tuple[float, ...] = _DEFAULT_RETRY_DELAYS,
    **kwargs: Any,
) -> T:
    last_exc: Exception | None = None
    for attempt, delay in enumerate((*retry_delays, None)):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            status = getattr(exc, "status_code", None)
            if status not in _RETRYABLE_STATUS_CODES and not _is_rate_limit(exc):
                raise
            if delay is None:
                break
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "rate limit" in msg or "ratelimit" in msg or "429" in msg


def make_semaphore(max_concurrency: int) -> asyncio.Semaphore:
    return asyncio.Semaphore(max_concurrency)
