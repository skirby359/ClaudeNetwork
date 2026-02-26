"""Volume analytics: trends, rolling averages, Gini coefficient, concentration."""

import numpy as np
import polars as pl


def gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient of a distribution (0=equal, 1=concentrated)."""
    values = np.sort(values.astype(float))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))


def compute_volume_trends(weekly_agg: pl.DataFrame) -> pl.DataFrame:
    """Add rolling averages and week-over-week change to weekly aggregation."""
    df = weekly_agg.sort("week_start")

    df = df.with_columns([
        pl.col("msg_count").rolling_mean(window_size=4).alias("msg_count_4wk_avg"),
        pl.col("recipient_impressions").rolling_mean(window_size=4).alias("impressions_4wk_avg"),
        pl.col("total_bytes").rolling_mean(window_size=4).alias("bytes_4wk_avg"),
    ])

    # Week-over-week change
    df = df.with_columns([
        (pl.col("msg_count") - pl.col("msg_count").shift(1)).alias("msg_count_wow_change"),
        ((pl.col("msg_count") - pl.col("msg_count").shift(1))
         / pl.col("msg_count").shift(1) * 100).alias("msg_count_wow_pct"),
    ])

    return df


def compute_sender_concentration(edge_fact: pl.DataFrame) -> dict:
    """Compute concentration metrics for senders."""
    sent_counts = edge_fact.group_by("from_email").agg(
        pl.len().alias("count")
    ).sort("count", descending=True)

    counts = sent_counts["count"].to_numpy()
    total = counts.sum()

    # Top-N share
    top_5_share = counts[:5].sum() / total if total > 0 else 0
    top_10_share = counts[:10].sum() / total if total > 0 else 0
    top_20_share = counts[:20].sum() / total if total > 0 else 0

    return {
        "gini": gini_coefficient(counts),
        "top_5_share": float(top_5_share),
        "top_10_share": float(top_10_share),
        "top_20_share": float(top_20_share),
        "total_senders": len(counts),
        "total_edges": int(total),
        "top_senders": sent_counts.head(20),
    }


def compute_top_n(edge_fact: pl.DataFrame, n: int = 20) -> dict:
    """Get top-N senders and receivers."""
    top_senders = (
        edge_fact.group_by("from_email")
        .agg(pl.len().alias("sent_count"))
        .sort("sent_count", descending=True)
        .head(n)
    )

    top_receivers = (
        edge_fact.group_by("to_email")
        .agg(pl.len().alias("received_count"))
        .sort("received_count", descending=True)
        .head(n)
    )

    return {"top_senders": top_senders, "top_receivers": top_receivers}
