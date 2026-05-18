import warnings

import pandas as pd
import pytest

from frugal_ml.sampling import random_sample


def _df(n: int) -> pd.DataFrame:
    return pd.DataFrame({"text": [f"row {i}" for i in range(n)]})


def test_basic_split():
    df = _df(100)
    sample, remainder = random_sample(df, 20)
    assert len(sample) == 20
    assert len(remainder) == 80
    assert len(sample) + len(remainder) == len(df)


def test_no_overlap():
    df = _df(100)
    sample, remainder = random_sample(df, 20)
    assert set(sample.index).isdisjoint(set(remainder.index))


def test_reproducible_with_seed():
    df = _df(200)
    s1, _ = random_sample(df, 50, seed=42)
    s2, _ = random_sample(df, 50, seed=42)
    assert list(s1.index) == list(s2.index)


def test_different_seeds():
    df = _df(200)
    s1, _ = random_sample(df, 50, seed=1)
    s2, _ = random_sample(df, 50, seed=2)
    assert list(s1.index) != list(s2.index)


def test_sample_size_equals_len_warns_and_returns_all():
    df = _df(50)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sample, remainder = random_sample(df, 50)
    assert len(sample) == 50
    assert len(remainder) == 0
    assert any("sample_size" in str(warning.message) for warning in w)


def test_sample_size_greater_than_len_warns():
    df = _df(30)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sample, remainder = random_sample(df, 100)
    assert len(sample) == 30
    assert len(remainder) == 0
    assert len(w) >= 1


def test_sample_size_one():
    df = _df(50)
    sample, remainder = random_sample(df, 1)
    assert len(sample) == 1
    assert len(remainder) == 49


def test_empty_df_raises():
    df = _df(0)
    with pytest.raises(ValueError, match="empty"):
        random_sample(df, 10)
