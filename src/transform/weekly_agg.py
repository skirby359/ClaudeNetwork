"""Weekly aggregation using pure Polars."""

import polars as pl

from src.config import AppConfig
from src.cache_manager import cached_parquet


def compute_weekly_stats(message_fact: pl.DataFrame, edge_fact: pl.DataFrame) -> pl.DataFrame:
    """Compute weekly aggregation using pure Polars (no DuckDB)."""
    weekly = (
        message_fact.group_by("week_id")
        .agg([
            pl.col("timestamp").min().alias("week_start"),
            pl.len().alias("msg_count"),
            pl.col("n_recipients").sum().alias("recipient_impressions"),
            pl.col("size_bytes").sum().alias("total_bytes"),
            pl.col("from_email").n_unique().alias("unique_senders"),
            pl.col("size_bytes").mean().alias("avg_size_bytes"),
            (pl.col("is_after_hours").sum().cast(pl.Float64) / pl.len()).alias("after_hours_rate"),
            (pl.col("is_weekend").sum().cast(pl.Float64) / pl.len()).alias("weekend_rate"),
        ])
        .sort("week_start")
    )

    recip_per_week = (
        edge_fact.group_by("week_id")
        .agg(pl.col("to_email").n_unique().alias("unique_recipients"))
    )

    weekly = weekly.join(recip_per_week, on="week_id", how="left")
    weekly = weekly.with_columns(pl.col("unique_recipients").fill_null(0))

    return weekly


def build_weekly_agg(
    message_fact: pl.DataFrame,
    edge_fact: pl.DataFrame,
    config: AppConfig,
) -> pl.DataFrame:
    """Build weekly aggregation table (with file caching)."""
    cache_path = config.cache_path(config.weekly_agg_file)
    source_paths = [config.cache_path(config.message_fact_file)]
    return cached_parquet(cache_path, source_paths, lambda: compute_weekly_stats(message_fact, edge_fact))
