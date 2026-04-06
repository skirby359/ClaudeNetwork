"""Orchestrates full ingestion: CSV → message_fact.parquet."""

from datetime import datetime
from pathlib import Path
from typing import Callable

import polars as pl

from src.config import AppConfig, DatasetConfig
from src.cache_manager import is_cache_fresh, write_parquet
from src.ingest.csv_parser import parse_csv
from src.ingest.size_parser import parse_size
from src.ingest.email_parser import parse_email_address, parse_recipients
from src.ingest.normalizer import normalize_email, normalize_name


# Module-level storage for per-file ingestion stats
_last_ingestion_stats: list[dict] = []


def get_last_ingestion_stats() -> list[dict]:
    """Return per-file error/row counts from the most recent ingestion run."""
    return list(_last_ingestion_stats)


def _parse_timestamp(date_str: str, fmt: str) -> datetime | None:
    """Parse a date string, returning None on failure."""
    try:
        return datetime.strptime(date_str.strip(), fmt)
    except (ValueError, AttributeError):
        return None


def _ingest_single_csv(csv_path: Path, dataset: DatasetConfig, start_msg_id: int) -> tuple[pl.DataFrame, int, int]:
    """Parse a single CSV file. Returns (DataFrame, next_msg_id, error_count).

    Email parsing is per-row (inherently sequential), but time-derived columns
    are computed via vectorized Polars expressions after the loop.
    """
    records = []
    msg_id = start_msg_id
    parse_errors = 0

    for row in parse_csv(csv_path):
        ts = _parse_timestamp(row["date"], dataset.date_format)
        if ts is None:
            parse_errors += 1
            continue

        size_bytes = parse_size(row["size"])
        if size_bytes is None:
            size_bytes = 0

        from_name, from_email = parse_email_address(row["from_raw"])
        if not from_email:
            parse_errors += 1
            continue

        from_email = normalize_email(from_email)
        from_name = normalize_name(from_name)

        recipients = parse_recipients(row["to_raw"])
        if not recipients:
            parse_errors += 1
            continue

        to_emails = []
        to_names = []
        for rname, remail in recipients:
            remail = normalize_email(remail)
            rname = normalize_name(rname)
            if remail:
                to_emails.append(remail)
                to_names.append(rname)

        if not to_emails:
            parse_errors += 1
            continue

        records.append({
            "msg_id": msg_id,
            "timestamp": ts,
            "size_bytes": size_bytes,
            "from_email": from_email,
            "from_name": from_name,
            "to_emails": to_emails,
            "to_names": to_names,
            "n_recipients": len(to_emails),
        })
        msg_id += 1

    if not records:
        return pl.DataFrame(), msg_id, parse_errors

    df = pl.DataFrame(records)

    # Compute time-derived columns using vectorized Polars expressions
    after_hours_start = dataset.after_hours_start
    after_hours_end = dataset.after_hours_end
    weekend_days = dataset.weekend_days

    df = df.with_columns([
        pl.col("timestamp").dt.strftime("%G-W%V").alias("week_id"),
        pl.col("timestamp").dt.hour().alias("hour"),
        (pl.col("timestamp").dt.weekday() - 1).cast(pl.Int32).alias("day_of_week"),
    ])
    df = df.with_columns([
        ((pl.col("hour") >= after_hours_start) | (pl.col("hour") < after_hours_end)).alias("is_after_hours"),
        pl.col("day_of_week").is_in(weekend_days).alias("is_weekend"),
    ])

    return df, msg_id, parse_errors


def run_ingestion(
    config: AppConfig = None,
    dataset: DatasetConfig = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> pl.DataFrame:
    """Run the full ingestion pipeline across all CSV files.

    Auto-discovers CSVs from dataset.csv_paths, parses each, and combines
    into a single message_fact.parquet. Uses per-file chunk caching:
    only re-parses CSVs whose chunk cache is stale or missing.

    Args:
        config: Application config.
        dataset: Dataset config.
        progress_callback: Optional callback(fraction, filename) for progress reporting.
    """
    global _last_ingestion_stats

    if config is None:
        config = AppConfig()
    if dataset is None:
        dataset = config.default_dataset

    cache_path = config.cache_path(config.message_fact_file)
    csv_paths = dataset.csv_paths

    if not csv_paths:
        print("No CSV files found in data directory")
        return pl.DataFrame()

    # Check if the combined cache is fresh against ALL source CSVs
    if is_cache_fresh(cache_path, *csv_paths, config.data_dir):
        return pl.read_parquet(cache_path)

    dfs = []
    stats = []
    next_id = 0
    total_files = len(csv_paths)

    for i, csv_path in enumerate(csv_paths):
        chunk_cache = config.csv_cache_path(csv_path)

        if progress_callback:
            progress_callback(i / total_files, csv_path.name)

        # Per-file chunk caching: skip re-parse if chunk is fresh
        if is_cache_fresh(chunk_cache, csv_path):
            chunk_df = pl.read_parquet(chunk_cache)
            if len(chunk_df) > 0:
                # Re-number msg_ids to be contiguous
                chunk_df = chunk_df.with_columns(
                    (pl.lit(next_id) + pl.arange(0, pl.len())).cast(pl.Int64).alias("msg_id")
                )
                next_id += len(chunk_df)
                dfs.append(chunk_df)
            stats.append({"file": csv_path.name, "rows": len(chunk_df), "errors": 0, "cached": True})
            print(f"  {csv_path.name}: {len(chunk_df)} messages (cached)")
            continue

        print(f"Ingesting {csv_path.name}...")
        chunk_df, next_id, errors = _ingest_single_csv(csv_path, dataset, next_id)

        if len(chunk_df) > 0:
            dfs.append(chunk_df)
            write_parquet(chunk_df, chunk_cache)

        stats.append({"file": csv_path.name, "rows": len(chunk_df), "errors": errors, "cached": False})
        print(f"  {len(chunk_df)} messages, {errors} errors")

    if progress_callback:
        progress_callback(1.0, "done")

    _last_ingestion_stats = stats

    print(f"Ingestion complete: {next_id} total messages, {sum(s['errors'] for s in stats)} total errors skipped")

    if not dfs:
        return pl.DataFrame()

    df = pl.concat(dfs)
    write_parquet(df, cache_path)
    return df
