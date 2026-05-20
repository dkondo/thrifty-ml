"""IMDB benchmark: reproduce proxy-model paper results.

Usage:
    python -m benchmarks.imdb.run                        # full 50k-row run
    python -m benchmarks.imdb.run --rows 500 --sample-size 100  # smoke run
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, TextIO

import numpy as np
import typer
from sklearn.metrics import f1_score
from tqdm import tqdm

from benchmarks.imdb.data import load_imdb
from benchmarks.imdb.instrumentation import instrument
from thrifty_ml.embeddings import embed_texts
from thrifty_ml.engine import Engine
from thrifty_ml.llm import label_texts
from thrifty_ml.sampling import random_sample

app = typer.Typer(add_completion=False)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def benchmark(
    llm: str,
    embedding_model: str,
    proxy: str,
    sample_size: int,
    rows: int | None,
    seed: int,
    cache_dir: Path | None,
    max_concurrency: int,
) -> dict:
    typer.echo("Loading IMDB dataset...")
    df = load_imdb(rows, seed).reset_index(drop=True)
    n_rows = len(df)

    embed_bar = tqdm(total=n_rows, desc="Embedding   ", unit="text", leave=True)
    llm_bar = tqdm(total=sample_size, desc="LLM labels  ", unit="call", leave=True)

    # Mutable reference so the fallback path can swap in a different bar.
    _llm_cb: list = [lambda: llm_bar.update(1)]

    with instrument(
        on_llm_call=lambda: _llm_cb[0](),
        on_embed_call=lambda n: embed_bar.update(n),
    ) as meter:
        t0 = time.perf_counter()
        engine = Engine(
            prompt="Is this movie review positive?",
            llm=llm,
            embedding_model=embedding_model,
            proxy=proxy,
            sample_size=sample_size,
            cache_dir=cache_dir,
            seed=seed,
            max_concurrency=max_concurrency,
        )
        proxy_model, eval_result = engine.fit(df, "text")
        fit_time = time.perf_counter() - t0

        embed_bar.close()
        llm_bar.close()

        # Recover the same sample/remainder split used inside fit (same seed)
        sample_df, remainder_df = random_sample(df, sample_size, seed)

        t1 = time.perf_counter()
        if eval_result.use_proxy:
            typer.echo("Proxy accepted — predicting remainder with classifier...")
            X_rem = embed_texts(remainder_df["text"].tolist(), engine.embedding_backend, cache_dir)
            remainder_preds = proxy_model.predict(X_rem).astype(int)
        else:
            n_rem = len(remainder_df)
            typer.echo(f"Proxy fell back (F1={eval_result.proxy_f1:.3f} < threshold) — LLM-labeling {n_rem} remainder rows...")
            with tqdm(total=n_rem, desc="Fallback LLM", unit="call", leave=True) as fb_bar:
                _llm_cb[0] = lambda: fb_bar.update(1)
                remainder_preds = np.array(
                    asyncio.run(
                        label_texts(
                            remainder_df["text"].tolist(),
                            engine.prompt,
                            engine.llm,
                            max_concurrency=max_concurrency,
                            cache_dir=cache_dir,
                        )
                    )
                ).astype(int)
        predict_time = time.perf_counter() - t1

    # Sample LLM labels reread from cache — zero API cost
    typer.echo("Rereading sample labels from cache...")
    sample_llm_labels = np.array(
        asyncio.run(
            label_texts(
                sample_df["text"].tolist(),
                engine.prompt,
                engine.llm,
                max_concurrency=max_concurrency,
                cache_dir=cache_dir,
            )
        )
    ).astype(int)

    f1_proxy_vs_gold = float(f1_score(remainder_df["gold"], remainder_preds, zero_division=0))
    f1_llm_vs_gold = float(f1_score(sample_df["gold"], sample_llm_labels, zero_division=0))
    tokens_actual = meter.total_tokens
    tokens_projected = tokens_actual * n_rows / sample_size if sample_size > 0 else 0.0

    return {
        "f1_proxy_vs_llm": eval_result.proxy_f1,
        "f1_proxy_vs_gold_remainder": f1_proxy_vs_gold,
        "f1_llm_vs_gold_sample": f1_llm_vs_gold,
        "relative_accuracy": f1_proxy_vs_gold / f1_llm_vs_gold if f1_llm_vs_gold else float("nan"),
        "use_proxy": eval_result.use_proxy,
        "fallback": not eval_result.use_proxy,
        "tokens_actual": tokens_actual,
        "tokens_projected_full_llm": tokens_projected,
        "token_reduction_x": tokens_projected / max(tokens_actual, 1),
        "wall_fit_s": fit_time,
        "wall_predict_s": predict_time,
        "projected_full_llm_wall_s": meter.llm_wall_s * n_rows / sample_size,
        "speedup_x": (meter.llm_wall_s * n_rows / sample_size) / max(fit_time + predict_time, 1e-9),
        "n_rows": n_rows,
        "sample_size": sample_size,
        "llm": llm,
        "embedding_model": embedding_model,
        "proxy": proxy,
        "git_sha": _git_sha(),
    }


def write_markdown(metrics: dict, file: TextIO = sys.stdout) -> None:
    header = (
        "| Dataset | Prompt | F1 (proxy vs LLM) | F1 (LLM vs gold) "
        "| F1 (proxy vs gold) | Relative accuracy | Token reduction | Speedup | Fallback |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    row = (
        f"| IMDB"
        f"| Is this movie review positive?"
        f"| {metrics['f1_proxy_vs_llm']:.3f}"
        f"| {metrics['f1_llm_vs_gold_sample']:.3f}"
        f"| {metrics['f1_proxy_vs_gold_remainder']:.3f}"
        f"| {metrics.get('relative_accuracy', float('nan')):.3f}"
        f"| {metrics['token_reduction_x']:.1f}×"
        f"| {metrics['speedup_x']:.1f}×"
        f"| {metrics['fallback']}"
        f" |"
    )
    file.write("\n".join([header, sep, row]) + "\n")


def write_json(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


@app.command()
def main(
    llm: str = typer.Option("openai/gpt-4o-mini", help="LiteLLM model string"),
    embedding_model: str = typer.Option("openai/text-embedding-3-large", help="LiteLLM embedding model"),
    proxy: str = typer.Option("lr", help="Proxy type: lr, svc, lgbm"),
    sample_size: int = typer.Option(1000, help="Rows to label with the LLM"),
    rows: Optional[int] = typer.Option(None, help="Total rows (None = full 50k)"),
    seed: int = typer.Option(42, help="Random seed"),
    cache_dir: Optional[Path] = typer.Option(None, help="Cache directory"),
    out: Path = typer.Option(Path("benchmarks/imdb/results.json"), help="JSON output path"),
    max_concurrency: int = typer.Option(8, help="Max simultaneous LLM calls"),
) -> None:
    metrics = benchmark(
        llm=llm,
        embedding_model=embedding_model,
        proxy=proxy,
        sample_size=sample_size,
        rows=rows,
        seed=seed,
        cache_dir=cache_dir,
        max_concurrency=max_concurrency,
    )
    write_markdown(metrics)
    write_json(metrics, out)
    typer.echo(f"\nFull metrics written to {out}")


if __name__ == "__main__":
    app()
