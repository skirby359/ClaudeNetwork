"""Data quality metrics: zero-size messages, missing names, parse errors."""

import polars as pl


def compute_quality_metrics(message_fact: pl.DataFrame) -> dict:
    """Compute data quality metrics from the message_fact table."""
    total = len(message_fact)
    if total == 0:
        return {
            "total_messages": 0,
            "zero_size_count": 0,
            "zero_size_pct": 0.0,
            "missing_name_count": 0,
            "missing_name_pct": 0.0,
            "single_recipient_count": 0,
            "single_recipient_pct": 0.0,
            "duplicate_msg_ids": 0,
        }

    zero_size = len(message_fact.filter(pl.col("size_bytes") == 0))
    missing_name = len(message_fact.filter(
        (pl.col("from_name") == "") | pl.col("from_name").is_null()
    ))
    single_recip = len(message_fact.filter(pl.col("n_recipients") == 1))
    dup_ids = total - message_fact["msg_id"].n_unique()

    return {
        "total_messages": total,
        "zero_size_count": zero_size,
        "zero_size_pct": zero_size / total,
        "missing_name_count": missing_name,
        "missing_name_pct": missing_name / total,
        "single_recipient_count": single_recip,
        "single_recipient_pct": single_recip / total,
        "duplicate_msg_ids": dup_ids,
    }


def compute_per_file_stats(stats_list: list[dict]) -> pl.DataFrame:
    """Convert per-file ingestion stats into a Polars DataFrame."""
    if not stats_list:
        return pl.DataFrame({"file": [], "rows": [], "errors": [], "cached": []})
    return pl.DataFrame(stats_list)
