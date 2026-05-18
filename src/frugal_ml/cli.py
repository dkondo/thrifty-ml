from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="frugal-ml: proxy-model library CLI", no_args_is_help=True)


def _load_df(input_path: Path) -> "pd.DataFrame":
    import pandas as pd
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    elif suffix == ".csv":
        return pd.read_csv(input_path)
    elif suffix == ".json" or suffix == ".jsonl":
        return pd.read_json(input_path, lines=(suffix == ".jsonl"))
    raise typer.BadParameter(f"Unsupported file format: {suffix}. Use .parquet, .csv, or .jsonl")


def _save_df(df: "pd.DataFrame", out: Path) -> None:
    if out.suffix.lower() == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)


@app.command()
def filter(
    input: Path = typer.Argument(..., help="Input file (.parquet or .csv)"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Filter condition prompt"),
    text_col: str = typer.Option(..., "--text-col", "-t", help="Text column name"),
    out: Path = typer.Option(..., "--out", "-o", help="Output file path"),
    llm: str = typer.Option("anthropic/claude-haiku-4-5", "--llm"),
    embedding_model: str = typer.Option("text-embedding-3-small", "--embedding-model"),
    proxy: str = typer.Option("lr", "--proxy", help="lr | svc | lgbm"),
    sample_size: int = typer.Option(1000, "--sample-size"),
    fallback_threshold: float = typer.Option(0.1, "--fallback-threshold"),
    max_concurrency: int = typer.Option(8, "--max-concurrency"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    """Filter rows based on a natural-language prompt."""
    import numpy as np
    from frugal_ml import ml_filter

    df = _load_df(input)
    typer.echo(f"Loaded {len(df)} rows from {input}")

    mask = ml_filter(
        df,
        prompt=prompt,
        text_column=text_col,
        llm=llm,
        embedding_model=embedding_model,
        proxy=proxy,
        sample_size=sample_size,
        fallback_threshold=fallback_threshold,
        max_concurrency=max_concurrency,
        cache_dir=cache_dir,
        seed=seed,
    )

    result = df[mask].copy()
    result["_frugal_mask"] = True
    _save_df(result, out)
    typer.echo(f"Filtered to {mask.sum()} / {len(df)} rows → {out}")


@app.command()
def classify(
    input: Path = typer.Argument(...),
    prompt: str = typer.Option(..., "--prompt", "-p"),
    text_col: str = typer.Option(..., "--text-col", "-t"),
    classes: str = typer.Option(..., "--classes", help="Comma-separated class list"),
    out: Path = typer.Option(..., "--out", "-o"),
    llm: str = typer.Option("anthropic/claude-haiku-4-5", "--llm"),
    embedding_model: str = typer.Option("text-embedding-3-small", "--embedding-model"),
    proxy: str = typer.Option("lr", "--proxy"),
    sample_size: int = typer.Option(1000, "--sample-size"),
    fallback_threshold: float = typer.Option(0.1, "--fallback-threshold"),
    max_concurrency: int = typer.Option(8, "--max-concurrency"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    """Classify rows into named classes using a natural-language prompt."""
    from frugal_ml import ml_classify

    class_list = [c.strip() for c in classes.split(",")]
    df = _load_df(input)
    typer.echo(f"Loaded {len(df)} rows, classifying into: {class_list}")

    labels = ml_classify(
        df,
        prompt=prompt,
        text_column=text_col,
        llm=llm,
        embedding_model=embedding_model,
        classes=class_list,
        proxy=proxy,
        sample_size=sample_size,
        fallback_threshold=fallback_threshold,
        max_concurrency=max_concurrency,
        cache_dir=cache_dir,
        seed=seed,
    )

    result = df.copy()
    result["label"] = labels
    _save_df(result, out)
    typer.echo(f"Labels written to {out}")


@app.command()
def embed(
    input: Path = typer.Argument(...),
    text_col: str = typer.Option(..., "--text-col", "-t"),
    model: str = typer.Option("text-embedding-3-small", "--model"),
    out: Path = typer.Option(..., "--out", "-o"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
) -> None:
    """Embed a text column and save as .npy."""
    import numpy as np
    from frugal_ml.embeddings import embed_texts, LiteLLMEmbeddingBackend

    df = _load_df(input)
    typer.echo(f"Embedding {len(df)} texts with {model}...")

    vectors = embed_texts(df[text_col].tolist(), LiteLLMEmbeddingBackend(model), cache_dir)
    np.save(str(out), vectors)
    typer.echo(f"Saved {vectors.shape} array to {out}")


@app.command()
def label(
    input: Path = typer.Argument(...),
    prompt: str = typer.Option(..., "--prompt", "-p"),
    text_col: str = typer.Option(..., "--text-col", "-t"),
    out: Path = typer.Option(..., "--out", "-o"),
    llm: str = typer.Option("anthropic/claude-haiku-4-5", "--llm"),
    sample_size: int = typer.Option(1000, "--sample-size"),
    max_concurrency: int = typer.Option(8, "--max-concurrency"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    """LLM-label a sample of rows and save labels."""
    import asyncio
    from frugal_ml import llm as _llm
    from frugal_ml.sampling import random_sample

    df = _load_df(input)
    sample_df, _ = random_sample(df, sample_size, seed)
    texts = sample_df[text_col].tolist()

    typer.echo(f"Labeling {len(texts)} rows with {llm}...")
    labels = asyncio.run(
        _llm.label_texts(
            texts, prompt, llm, max_concurrency=max_concurrency, cache_dir=cache_dir
        )
    )

    result = sample_df.copy()
    result["label"] = labels
    _save_df(result, out)
    typer.echo(f"Labels written to {out}")


cache_app = typer.Typer(help="Cache management")
app.add_typer(cache_app, name="cache")


@cache_app.command("clear")
def cache_clear(
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
) -> None:
    """Clear the embedding and label cache."""
    from frugal_ml.cache import clear_cache
    count = clear_cache(cache_dir)
    typer.echo(f"Cleared {count} cache entries.")


if __name__ == "__main__":
    app()
