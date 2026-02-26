"""Statistical outlier detection for email patterns."""

import polars as pl
import numpy as np
from scipy import stats


def detect_volume_anomalies(weekly_agg: pl.DataFrame, z_threshold: float = 2.5) -> pl.DataFrame:
    """Detect weeks with anomalous message volume using z-scores."""
    counts = weekly_agg["msg_count"].to_numpy().astype(float)

    if len(counts) < 4:
        return weekly_agg.with_columns([
            pl.lit(0.0).alias("volume_zscore"),
            pl.lit(False).alias("is_volume_anomaly"),
        ])

    z_scores = stats.zscore(counts)

    return weekly_agg.with_columns([
        pl.Series("volume_zscore", z_scores),
        pl.Series("is_volume_anomaly", np.abs(z_scores) > z_threshold),
    ])


def detect_sender_anomalies(
    edge_fact: pl.DataFrame,
    person_dim: pl.DataFrame,
    z_threshold: float = 2.5,
) -> pl.DataFrame:
    """Detect senders with anomalous behavior (volume, timing, recipient patterns)."""
    sender_stats = (
        edge_fact.group_by("from_email")
        .agg([
            pl.len().alias("total_sent"),
            pl.col("to_email").n_unique().alias("unique_recipients"),
            pl.col("is_after_hours").mean().alias("after_hours_rate"),
            pl.col("is_weekend").mean().alias("weekend_rate"),
        ])
    )

    # Z-scores for each metric
    for col_name in ["total_sent", "unique_recipients", "after_hours_rate", "weekend_rate"]:
        values = sender_stats[col_name].to_numpy().astype(float)
        if len(values) > 3:
            z = stats.zscore(values)
            sender_stats = sender_stats.with_columns(
                pl.Series(f"{col_name}_zscore", z)
            )
        else:
            sender_stats = sender_stats.with_columns(
                pl.lit(0.0).alias(f"{col_name}_zscore")
            )

    # Flag anomalies: anyone with any z-score exceeding threshold
    sender_stats = sender_stats.with_columns(
        (
            (pl.col("total_sent_zscore").abs() > z_threshold)
            | (pl.col("unique_recipients_zscore").abs() > z_threshold)
            | (pl.col("after_hours_rate_zscore").abs() > z_threshold)
            | (pl.col("weekend_rate_zscore").abs() > z_threshold)
        ).alias("is_anomaly")
    )

    anomalies = sender_stats.filter(pl.col("is_anomaly"))
    return anomalies.sort("total_sent", descending=True)


def compute_anomaly_summary(
    weekly_agg: pl.DataFrame,
    edge_fact: pl.DataFrame,
    person_dim: pl.DataFrame,
) -> dict:
    """Compute a summary of all detected anomalies."""
    vol_anomalies = detect_volume_anomalies(weekly_agg)
    sender_anomalies = detect_sender_anomalies(edge_fact, person_dim)

    anomalous_weeks = vol_anomalies.filter(pl.col("is_volume_anomaly"))

    return {
        "weekly_anomalies": anomalous_weeks,
        "sender_anomalies": sender_anomalies,
        "n_anomalous_weeks": len(anomalous_weeks),
        "n_anomalous_senders": len(sender_anomalies),
    }
