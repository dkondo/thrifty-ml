from __future__ import annotations

import pandas as pd


def load_imdb(rows: int | None, seed: int) -> pd.DataFrame:
    import datasets  # optional dep: pip install thrifty-ml[benchmark]

    ds = datasets.load_dataset("stanfordnlp/imdb", split="train+test")
    df = ds.to_pandas().rename(columns={"label": "gold"})[["text", "gold"]]
    if rows is not None:
        df = df.groupby("gold", group_keys=False).sample(rows // 2, random_state=seed)
    return df.reset_index(drop=True)
