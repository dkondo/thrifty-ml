from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def random_sample(
    df: pd.DataFrame,
    n: int,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sample, remainder) DataFrames.

    If n >= len(df), all rows are returned as the sample and remainder is empty.
    """
    if len(df) == 0:
        raise ValueError("Cannot sample from an empty DataFrame.")

    if n >= len(df):
        warnings.warn(
            f"sample_size={n} >= len(df)={len(df)}. Using all rows as the labeled "
            "sample. The proxy will not be used — LLM labels are returned directly.",
            UserWarning,
            stacklevel=3,
        )
        return df.copy(), df.iloc[:0].copy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=False)
    mask = np.zeros(len(df), dtype=bool)
    mask[idx] = True
    return df.iloc[mask].copy(), df.iloc[~mask].copy()
