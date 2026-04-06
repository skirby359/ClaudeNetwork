"""Email size forensics: classification, templates, anomalies."""

import polars as pl
import numpy as np


DEFAULT_THRESHOLDS = {
    "tiny": 1024,           # < 1 KB
    "small": 10240,         # < 10 KB
    "medium": 102400,       # < 100 KB
    "large": 1048576,       # < 1 MB
    # >= 1 MB = "huge"
}


def classify_by_size(
    message_fact: pl.DataFrame,
    thresholds: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Add a size_class column to message_fact."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    return message_fact.with_columns(
        pl.when(pl.col("size_bytes") < thresholds["tiny"]).then(pl.lit("tiny"))
        .when(pl.col("size_bytes") < thresholds["small"]).then(pl.lit("small"))
        .when(pl.col("size_bytes") < thresholds["medium"]).then(pl.lit("medium"))
        .when(pl.col("size_bytes") < thresholds["large"]).then(pl.lit("large"))
        .otherwise(pl.lit("huge"))
        .alias("size_class")
    )


def detect_size_templates(message_fact: pl.DataFrame, min_occurrences: int = 20) -> pl.DataFrame:
    """Detect messages with repeated exact sizes (likely auto-generated).

    Groups messages by exact size_bytes and flags frequent sizes.
    """
    size_counts = (
        message_fact.group_by("size_bytes")
        .agg([
            pl.len().alias("occurrence_count"),
            pl.col("from_email").n_unique().alias("unique_senders"),
            pl.col("from_email").first().alias("example_sender"),
        ])
        .filter(pl.col("occurrence_count") >= min_occurrences)
        .sort("occurrence_count", descending=True)
    )

    return size_counts


def compute_sender_size_profile(message_fact: pl.DataFrame) -> pl.DataFrame:
    """Per-sender size statistics: avg, stddev, dominant class."""
    classified = classify_by_size(message_fact)

    # Find dominant class per sender
    class_counts = (
        classified.group_by(["from_email", "size_class"])
        .agg(pl.len().alias("class_count"))
    )
    dominant = (
        class_counts.sort(["from_email", "class_count"], descending=[False, True])
        .group_by("from_email")
        .agg(pl.col("size_class").first().alias("dominant_class"))
    )

    # Per-sender size stats
    sender_stats = (
        message_fact.group_by("from_email")
        .agg([
            pl.col("size_bytes").mean().alias("avg_size"),
            pl.col("size_bytes").std().alias("size_stddev"),
            pl.col("size_bytes").median().alias("median_size"),
            pl.col("size_bytes").max().alias("max_size"),
            pl.len().alias("msg_count"),
        ])
    )

    sender_stats = sender_stats.join(dominant, on="from_email", how="left")
    return sender_stats.sort("avg_size", descending=True)


def detect_size_anomalies(message_fact: pl.DataFrame, z_threshold: float = 3.0) -> pl.DataFrame:
    """Detect messages with unusual size for their sender.

    Computes per-sender z-score of message size and flags outliers.
    """
    # Only consider senders with enough messages
    sender_stats = (
        message_fact.group_by("from_email")
        .agg([
            pl.col("size_bytes").mean().alias("sender_mean"),
            pl.col("size_bytes").std().alias("sender_std"),
            pl.len().alias("sender_count"),
        ])
        .filter(pl.col("sender_count") >= 5)
    )

    with_stats = message_fact.join(sender_stats, on="from_email", how="inner")

    # Compute z-score
    with_stats = with_stats.with_columns(
        (
            (pl.col("size_bytes").cast(pl.Float64) - pl.col("sender_mean"))
            / pl.col("sender_std").clip(lower_bound=1.0)
        ).alias("size_zscore")
    )

    anomalies = with_stats.filter(pl.col("size_zscore").abs() > z_threshold)
    return anomalies.select([
        "msg_id", "timestamp", "from_email", "size_bytes",
        "sender_mean", "sender_std", "size_zscore",
    ]).sort("size_zscore", descending=True)
