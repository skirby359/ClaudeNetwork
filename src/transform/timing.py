"""Time feature extraction and timing metrics."""

import polars as pl

from src.config import AppConfig
from src.cache_manager import cached_parquet


def build_timing_metrics(message_fact: pl.DataFrame, config: AppConfig) -> pl.DataFrame:
    """Build timing metrics: hourly/daily distributions, after-hours patterns."""
    cache_path = config.cache_path(config.timing_metrics_file)
    source_paths = [config.cache_path(config.message_fact_file)]

    def _build():
        # Hourly distribution by day_of_week
        hourly = (
            message_fact.group_by(["hour", "day_of_week"])
            .agg([
                pl.len().alias("msg_count"),
                pl.col("size_bytes").mean().alias("avg_size"),
                pl.col("n_recipients").mean().alias("avg_recipients"),
                pl.col("is_after_hours").mean().alias("after_hours_rate"),
            ])
            .sort(["day_of_week", "hour"])
        )

        return hourly

    return cached_parquet(cache_path, source_paths, _build)
