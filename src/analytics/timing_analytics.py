"""Timing analytics: heatmaps, after-hours trends, burstiness, ping-pong."""

import polars as pl


def compute_hour_day_heatmap(message_fact: pl.DataFrame) -> pl.DataFrame:
    """Build a heatmap matrix of message counts by hour and day of week."""
    heatmap = (
        message_fact.group_by(["hour", "day_of_week"])
        .agg(pl.len().alias("msg_count"))
        .sort(["day_of_week", "hour"])
    )
    return heatmap


def compute_after_hours_by_week(message_fact: pl.DataFrame) -> pl.DataFrame:
    """Compute after-hours messaging rate over time."""
    return (
        message_fact.group_by("week_id")
        .agg([
            pl.len().alias("total_msgs"),
            pl.col("is_after_hours").sum().alias("after_hours_count"),
            pl.col("is_weekend").sum().alias("weekend_count"),
            pl.col("is_after_hours").mean().alias("after_hours_rate"),
            pl.col("is_weekend").mean().alias("weekend_rate"),
            pl.col("timestamp").min().alias("week_start"),
        ])
        .sort("week_start")
    )


def compute_burstiness(message_fact: pl.DataFrame, top_n: int = 50) -> pl.DataFrame:
    """Compute burstiness metric per sender using vectorized Polars operations.

    Burstiness B = (σ - μ) / (σ + μ) where σ, μ are the std and mean of
    inter-message intervals. B ∈ [-1, 1]: B>0 = bursty, B<0 = periodic, B≈0 = random.
    """
    top_senders = (
        message_fact.group_by("from_email")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(top_n)["from_email"]
    )

    filtered = (
        message_fact.filter(pl.col("from_email").is_in(top_senders.to_list()))
        .sort(["from_email", "timestamp"])
    )

    # Compute inter-message intervals using diff() over each sender group
    with_intervals = filtered.with_columns(
        pl.col("timestamp").diff().over("from_email").dt.total_seconds().alias("interval_seconds")
    )

    # Filter to positive intervals (drop first per group which is null, and zero intervals)
    positive_intervals = with_intervals.filter(
        pl.col("interval_seconds").is_not_null() & (pl.col("interval_seconds") > 0)
    )

    # Aggregate: mean, std, count per sender
    result = (
        positive_intervals.group_by("from_email")
        .agg([
            pl.col("interval_seconds").mean().alias("mean_interval"),
            pl.col("interval_seconds").std().alias("std_interval"),
            pl.len().alias("interval_count"),
        ])
        .filter(pl.col("interval_count") >= 2)
    )

    # Compute burstiness and convert to hours
    result = result.with_columns([
        (
            (pl.col("std_interval") - pl.col("mean_interval"))
            / (pl.col("std_interval") + pl.col("mean_interval"))
        ).alias("burstiness"),
        (pl.col("mean_interval") / 3600).alias("mean_interval_hours"),
        (pl.col("std_interval") / 3600).alias("std_interval_hours"),
    ])

    # Add original message counts
    msg_counts = (
        filtered.group_by("from_email")
        .agg(pl.len().alias("msg_count"))
    )
    result = result.join(msg_counts, on="from_email", how="left")

    return (
        result.select(["from_email", "burstiness", "mean_interval_hours", "std_interval_hours", "msg_count"])
        .sort("burstiness", descending=True)
    )


def compute_ping_pong(edge_fact: pl.DataFrame, min_exchanges: int = 5) -> pl.DataFrame:
    """Find rapid back-and-forth (ping-pong) communication pairs.

    Looks for pairs where messages alternate direction within short time windows.
    """
    # Get conversations sorted by time
    convos = (
        edge_fact.select(["from_email", "to_email", "timestamp"])
        .sort("timestamp")
    )

    # For each directed pair, count messages
    pair_counts = (
        convos.group_by(["from_email", "to_email"])
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") >= min_exchanges)
    )

    # Find bidirectional pairs with enough activity
    forward = pair_counts.rename({"from_email": "a", "to_email": "b", "count": "a_to_b"})
    reverse = pair_counts.rename({"from_email": "b_r", "to_email": "a_r", "count": "b_to_a"})

    bidirectional = forward.join(
        reverse,
        left_on=["a", "b"],
        right_on=["b_r", "a_r"],
        how="inner",
    )

    # Keep unique pairs
    bidirectional = bidirectional.filter(pl.col("a") < pl.col("b"))

    bidirectional = bidirectional.with_columns([
        (pl.col("a_to_b") + pl.col("b_to_a")).alias("total_exchanges"),
        (pl.col("a_to_b").cast(pl.Float64) / (pl.col("a_to_b") + pl.col("b_to_a"))).alias("balance_ratio"),
    ])

    return bidirectional.sort("total_exchanges", descending=True)
