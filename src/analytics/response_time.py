"""Response time analysis: reply detection, response speed metrics."""

import polars as pl


def compute_reply_times(edge_fact: pl.DataFrame) -> pl.DataFrame:
    """Detect reply pairs and compute response times.

    Uses an asof join to find the next B->A message after each A->B message
    within 24 hours, avoiding the N×M memory explosion of a full cross-join.

    Returns a DataFrame with person_a, person_b, median_reply_seconds, and counts.
    """
    if len(edge_fact) == 0:
        return pl.DataFrame({
            "person_a": [], "person_b": [],
            "median_reply_seconds": [], "p95_reply_seconds": [],
            "reply_count": [],
        })

    # Get directed messages sorted by time
    msgs = (
        edge_fact.select(["from_email", "to_email", "timestamp"])
        .sort("timestamp")
    )

    # Forward: A->B messages (the originals)
    forward = msgs.rename({
        "from_email": "sender",
        "to_email": "receiver",
        "timestamp": "send_time",
    }).with_row_index("fwd_idx")

    # Reverse: B->A messages (the potential replies), keyed by (original_sender, replier)
    reverse = msgs.rename({
        "from_email": "replier",
        "to_email": "original_sender",
        "timestamp": "reply_time",
    })

    # Use join_asof to find the NEXT reverse message after each forward message
    # for the same pair. This is O(n log n) instead of O(n×m).
    # join_asof requires sorting on the "by" key + the asof column.
    forward_sorted = forward.sort(["sender", "receiver", "send_time"])
    reverse_sorted = reverse.sort(["original_sender", "replier", "reply_time"])

    joined = forward_sorted.join_asof(
        reverse_sorted,
        left_on="send_time",
        right_on="reply_time",
        by_left=["sender", "receiver"],
        by_right=["original_sender", "replier"],
        strategy="forward",  # find the next reply AFTER send_time
    )

    # Filter: reply must exist and be within 24 hours
    joined = joined.filter(
        pl.col("reply_time").is_not_null()
        & ((pl.col("reply_time") - pl.col("send_time")).dt.total_seconds() <= 86400)
    )

    # Compute reply seconds
    joined = joined.with_columns(
        (pl.col("reply_time") - pl.col("send_time")).dt.total_seconds().alias("reply_seconds")
    )

    # Aggregate per pair
    pair_stats = (
        joined.group_by(["sender", "receiver"])
        .agg([
            pl.col("reply_seconds").median().alias("median_reply_seconds"),
            pl.col("reply_seconds").quantile(0.95).alias("p95_reply_seconds"),
            pl.len().alias("reply_count"),
        ])
    ).rename({"sender": "person_a", "receiver": "person_b"})

    return pair_stats.sort("reply_count", descending=True)


def compute_person_response_stats(reply_times: pl.DataFrame) -> pl.DataFrame:
    """Per-person response stats: how fast does each person reply?"""
    # As replier (person_b is the one replying)
    replier_stats = (
        reply_times.group_by("person_b")
        .agg([
            pl.col("median_reply_seconds").median().alias("median_response_sec"),
            pl.col("reply_count").sum().alias("total_replies"),
        ])
        .rename({"person_b": "email"})
    )
    return replier_stats.sort("median_response_sec")


def compute_department_response_stats(
    reply_times: pl.DataFrame,
    person_dim: pl.DataFrame,
) -> pl.DataFrame:
    """Group response times by domain (department proxy)."""
    # Join domain onto person_b
    with_domain = reply_times.join(
        person_dim.select(["email", "domain"]),
        left_on="person_b",
        right_on="email",
        how="left",
    )

    dept_stats = (
        with_domain.group_by("domain")
        .agg([
            pl.col("median_reply_seconds").median().alias("median_response_sec"),
            pl.col("reply_count").sum().alias("total_replies"),
            pl.col("person_b").n_unique().alias("n_people"),
        ])
        .sort("median_response_sec")
    )
    return dept_stats
