"""Broadcast metrics: recipient count distributions, blast detection."""

import polars as pl

from src.config import AppConfig
from src.cache_manager import cached_parquet


def compute_broadcast_stats(message_fact: pl.DataFrame) -> pl.DataFrame:
    """Compute broadcast/blast metrics per sender (no file caching)."""
    sender_stats = (
        message_fact.group_by("from_email")
        .agg([
            pl.len().alias("total_msgs"),
            pl.col("n_recipients").mean().alias("avg_recipients"),
            pl.col("n_recipients").median().alias("median_recipients"),
            pl.col("n_recipients").max().alias("max_recipients"),
            (pl.col("n_recipients") > 5).sum().alias("blast_count"),
            (pl.col("n_recipients") > 10).sum().alias("large_blast_count"),
            pl.col("size_bytes").sum().alias("total_bytes_sent"),
            (pl.col("n_recipients") * pl.col("size_bytes")).sum().alias("total_impressions_bytes"),
        ])
    )

    sender_stats = sender_stats.with_columns([
        (pl.col("blast_count") / pl.col("total_msgs")).alias("blast_rate"),
        (pl.col("large_blast_count") / pl.col("total_msgs")).alias("large_blast_rate"),
    ])

    return sender_stats.sort("total_impressions_bytes", descending=True)


def build_broadcast_metrics(message_fact: pl.DataFrame, config: AppConfig) -> pl.DataFrame:
    """Build broadcast/blast metrics per sender (with file caching)."""
    cache_path = config.cache_path(config.broadcast_metrics_file)
    source_paths = [config.cache_path(config.message_fact_file)]
    return cached_parquet(cache_path, source_paths, lambda: compute_broadcast_stats(message_fact))
