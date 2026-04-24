import pandas as pd

from alpha_lab.utils.cache import (
    cached_parquet,
    read_csv,
    read_parquet,
    write_csv,
    write_parquet,
)


def test_parquet_round_trip(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    path = tmp_path / "x.parquet"
    write_parquet(df, path)
    loaded = read_parquet(path)
    pd.testing.assert_frame_equal(df, loaded)


def test_csv_round_trip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "x.csv"
    write_csv(df, path)
    loaded = read_csv(path)
    pd.testing.assert_frame_equal(df, loaded)


def test_cached_parquet_builds_then_loads(tmp_path):
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return pd.DataFrame({"a": [1, 2, 3]})

    first = cached_parquet("demo", build, cache_dir=tmp_path)
    second = cached_parquet("demo", build, cache_dir=tmp_path)
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(first, second)

    third = cached_parquet("demo", build, cache_dir=tmp_path, refresh=True)
    assert calls["n"] == 2
    pd.testing.assert_frame_equal(first, third)
