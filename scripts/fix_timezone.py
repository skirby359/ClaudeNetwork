"""Convert stored UTC timestamps in the Graph extract to a local timezone and
recompute the time-derived columns (hour, day_of_week, is_after_hours,
is_weekend, week_id).

The Graph extract stored naive UTC timestamps; the analytics assume local time,
so after-hours/weekend/heatmap metrics were shifted. This converts the existing
parquet + cache in place (equivalent to re-extracting with the timezone fix, but
without re-pulling from Graph).

Run: python scripts/fix_timezone.py America/Los_Angeles
"""

import sys
sys.path.insert(0, ".")

import polars as pl
from src.config import AppConfig

AFTER_HOURS_START = 18
AFTER_HOURS_END = 7
WEEKEND_DAYS = [5, 6]


def convert(df: pl.DataFrame, tz: str) -> pl.DataFrame:
    # naive UTC -> aware UTC -> local -> naive local
    df = df.with_columns(
        pl.col("timestamp")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(tz)
        .dt.replace_time_zone(None)
        .alias("timestamp")
    )
    # Recompute derived columns from the now-local timestamp
    df = df.with_columns([
        pl.col("timestamp").dt.strftime("%G-W%V").alias("week_id"),
        pl.col("timestamp").dt.hour().alias("hour"),
        (pl.col("timestamp").dt.weekday() - 1).cast(pl.Int32).alias("day_of_week"),
    ])
    df = df.with_columns([
        ((pl.col("hour") >= AFTER_HOURS_START) | (pl.col("hour") < AFTER_HOURS_END)).alias("is_after_hours"),
        pl.col("day_of_week").is_in(WEEKEND_DAYS).alias("is_weekend"),
    ])
    return df


def main():
    tz = sys.argv[1] if len(sys.argv) > 1 else "America/Los_Angeles"
    config = AppConfig()
    targets = [
        config.data_dir / "microsoft365_messages.parquet",
        config.cache_path(config.message_fact_file),
    ]
    for path in targets:
        if not path.exists():
            print(f"skip (missing): {path}")
            continue
        df = pl.read_parquet(path)
        before_ah = df["is_after_hours"].mean() * 100
        df = convert(df, tz)
        after_ah = df["is_after_hours"].mean() * 100
        df.write_parquet(path)
        print(f"{path.name}: after-hours {before_ah:.1f}% -> {after_ah:.1f}%  (tz={tz})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
