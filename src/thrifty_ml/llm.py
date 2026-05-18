from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

import litellm

from thrifty_ml._utils import _litellm_call_with_retry, make_semaphore
from thrifty_ml import cache as _cache

litellm.suppress_debug_info = True

_BINARY_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "boolean"}},
    "required": ["label"],
}

_MULTICLASS_SCHEMA_TEMPLATE = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": []}
    },
    "required": ["label"],
}

_UNKNOWN_LABEL = "__unknown__"


def _binary_prompt(user_prompt: str, text: str) -> str:
    return (
        f"{user_prompt}\n\nText: {text}\n\n"
        'Reply with JSON: {"label": true} or {"label": false}. No explanation.'
    )


def _multiclass_prompt(user_prompt: str, text: str, classes: list[str]) -> str:
    classes_str = ", ".join(classes)
    return (
        f"{user_prompt}\n\nText: {text}\n\n"
        f'Reply with JSON: {{"label": "<one of: {classes_str}>"}}'
        ". Only use one of the listed labels. No explanation."
    )


async def _label_one(
    text: str,
    prompt: str,
    model: str,
    classes: list[str] | None,
    semaphore: asyncio.Semaphore,
    cache_dir: Path | None,
) -> bool | str:
    cached = _cache.get_label(text, prompt, model, cache_dir, classes=classes)
    if cached is not None:
        return cached

    full_prompt = (
        _binary_prompt(prompt, text)
        if classes is None
        else _multiclass_prompt(prompt, text, classes)
    )

    async def _call() -> bool | str:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"
        # Some models wrap the JSON in markdown fences despite json_object mode.
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)
        label_val = parsed.get("label")

        if classes is None:
            if isinstance(label_val, bool):
                return label_val
            if isinstance(label_val, str):
                return label_val.lower() in ("true", "yes", "1")
            return bool(label_val)
        else:
            if label_val not in classes:
                return _UNKNOWN_LABEL
            return str(label_val)

    async with semaphore:
        result = await _litellm_call_with_retry(_call)

    _cache.set_label(text, prompt, model, result, cache_dir, classes=classes)
    return result


async def label_texts(
    texts: list[str],
    prompt: str,
    model: str,
    classes: list[str] | None = None,
    max_concurrency: int = 8,
    cache_dir: Path | None = None,
) -> list[bool | str]:
    semaphore = make_semaphore(max_concurrency)
    tasks = [
        _label_one(text, prompt, model, classes, semaphore, cache_dir)
        for text in texts
    ]
    return list(await asyncio.gather(*tasks))
