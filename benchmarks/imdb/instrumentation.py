from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

import litellm


@dataclass
class Meter:
    n_llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_wall_s: float = 0.0
    n_embed_calls: int = 0
    embed_tokens: int = 0
    embed_wall_s: float = 0.0


@contextmanager
def instrument(
    on_llm_call: Callable[[], None] | None = None,
    on_embed_call: Callable[[int], None] | None = None,
) -> Iterator[Meter]:
    """Wrap litellm.acompletion and litellm.embedding to capture token usage and timing.

    Optionally accepts progress hooks:
      on_llm_call()  — called after each successful acompletion
      on_embed_call(n)  — called after each successful embedding batch with n=len(texts)
    """
    meter = Meter()
    _orig_acompletion = litellm.acompletion
    _orig_embedding = litellm.embedding

    async def _wrapped_acompletion(*args, **kwargs):
        t0 = time.perf_counter()
        response = await _orig_acompletion(*args, **kwargs)
        elapsed = time.perf_counter() - t0

        meter.n_llm_calls += 1
        meter.llm_wall_s += elapsed
        usage = getattr(response, "usage", None)
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0
            tt = getattr(usage, "total_tokens", 0) or 0
            if tt == 0:
                tt = pt + ct
            meter.prompt_tokens += pt
            meter.completion_tokens += ct
            meter.total_tokens += tt
        else:
            try:
                model = kwargs.get("model", "") or (args[0] if args else "")
                messages = kwargs.get("messages", [])
                pt = litellm.token_counter(model=model, messages=messages)
                choices = getattr(response, "choices", [])
                content = (
                    getattr(getattr(choices[0], "message", None), "content", "")
                    if choices else ""
                ) or ""
                ct = litellm.token_counter(model=model, text=content)
                meter.prompt_tokens += pt
                meter.completion_tokens += ct
                meter.total_tokens += pt + ct
            except Exception:
                pass

        if on_llm_call is not None:
            on_llm_call()
        return response

    def _wrapped_embedding(*args, **kwargs):
        t0 = time.perf_counter()
        response = _orig_embedding(*args, **kwargs)
        elapsed = time.perf_counter() - t0

        meter.n_embed_calls += 1
        meter.embed_wall_s += elapsed
        texts = kwargs.get("input", []) or (args[1] if len(args) > 1 else [])
        n = len(texts)
        usage = getattr(response, "usage", None)
        if usage is not None:
            meter.embed_tokens += getattr(usage, "prompt_tokens", 0) or 0

        if on_embed_call is not None:
            on_embed_call(n)
        return response

    litellm.acompletion = _wrapped_acompletion
    litellm.embedding = _wrapped_embedding
    try:
        yield meter
    finally:
        litellm.acompletion = _orig_acompletion
        litellm.embedding = _orig_embedding
