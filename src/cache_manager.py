"""Cache manager with mtime-based invalidation for parquet and pickle files."""

import pickle
from pathlib import Path

import polars as pl


def is_cache_fresh(cache_path: Path, *source_paths: Path) -> bool:
    """Check if cache file exists and is newer than all source files."""
    if not cache_path.exists():
        return False
    cache_mtime = cache_path.stat().st_mtime
    for src in source_paths:
        if src.exists() and src.stat().st_mtime > cache_mtime:
            return False
    return True


def read_parquet(cache_path: Path) -> pl.DataFrame:
    """Read a Polars DataFrame from parquet."""
    return pl.read_parquet(cache_path)


def write_parquet(df: pl.DataFrame, cache_path: Path) -> None:
    """Write a Polars DataFrame to parquet."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_path)


def read_pickle(cache_path: Path):
    """Read a Python object from pickle."""
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def write_pickle(obj, cache_path: Path) -> None:
    """Write a Python object to pickle."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def cached_parquet(cache_path: Path, source_paths: list[Path], builder_fn):
    """Return cached DataFrame or build it using builder_fn, then cache."""
    if is_cache_fresh(cache_path, *source_paths):
        return read_parquet(cache_path)
    df = builder_fn()
    write_parquet(df, cache_path)
    return df


def cached_pickle(cache_path: Path, source_paths: list[Path], builder_fn):
    """Return cached object or build it using builder_fn, then cache."""
    if is_cache_fresh(cache_path, *source_paths):
        return read_pickle(cache_path)
    obj = builder_fn()
    write_pickle(obj, cache_path)
    return obj
