"""Full Microsoft Graph extraction run.

Pulls ALL email metadata from ALL mailboxes in the configured tenant (full
history), writes it to data/microsoft365_messages.parquet, and installs it as
the active message_fact cache so the dashboard renders it.

Credentials are read from .env.local (MS_TENANT_ID / MS_APP_ID / MS_APP_SECRET).
No secrets are hardcoded here.

The existing message_fact.parquet cache (Spokane demo data) is backed up before
being overwritten; it is also fully regenerable from the CSVs in data/.

Run: python scripts/full_graph_run.py
"""

import os
import sys
import time
import shutil
from pathlib import Path

sys.path.insert(0, ".")

import polars as pl

from src.config import AppConfig, DatasetConfig
from src.cache_manager import write_parquet
from src.ingest.msgraph import GraphConfig, run_graph_ingestion


def load_env_local():
    env_path = Path(".env.local")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def main():
    load_env_local()
    tenant = os.environ.get("MS_TENANT_ID", "")
    app_id = os.environ.get("MS_APP_ID", "")
    secret = os.environ.get("MS_APP_SECRET", "")
    if not (tenant and app_id and secret):
        print("Missing MS_TENANT_ID / MS_APP_ID / MS_APP_SECRET in .env.local")
        return 1

    config = AppConfig()
    cfg = GraphConfig(tenant_id=tenant, app_id=app_id, app_secret=secret)

    # Graph delivers UTC; convert to the org's local timezone so time-of-day,
    # after-hours and weekend analytics are correct. Set MS_TIMEZONE (IANA name).
    tz = os.environ.get("MS_TIMEZONE", "")
    dataset = DatasetConfig(name="graph", timezone=tz or None)
    if tz:
        print(f"Timezone: {tz}")
    else:
        print("WARNING: MS_TIMEZONE not set — timestamps kept as UTC; "
              "after-hours/weekend metrics will be wrong if the org isn't on UTC.")

    # Back up the existing (Spokane) message_fact cache before overwriting.
    cache_path = config.cache_path(config.message_fact_file)
    if cache_path.exists():
        backup = cache_path.with_suffix(".spokane_backup.parquet")
        if not backup.exists():
            shutil.copy2(cache_path, backup)
            print(f"Backed up existing cache -> {backup.name}")

    def progress(frac, text):
        print(f"  [{frac*100:5.1f}%] {text}")

    print("Starting full Graph extraction (all mailboxes, all history)...")
    t0 = time.perf_counter()
    df = run_graph_ingestion(
        cfg,
        user_ids=None,   # all licensed mailboxes
        since=None,      # full history
        max_per_user=50000,
        progress_callback=progress,
        dataset_config=dataset,
    )
    elapsed = time.perf_counter() - t0

    if len(df) == 0:
        print("No messages retrieved.")
        return 1

    # Durable Graph artifact in data/ (NOT a CSV, so the CSV pipeline ignores it).
    out_path = config.data_dir / "microsoft365_messages.parquet"
    config.data_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(df, out_path)

    # Install as the active message_fact cache LAST, so its mtime is newest and
    # is_cache_fresh() serves it instead of rebuilding Spokane from the CSVs.
    write_parquet(df, cache_path)

    # ---- Summary ----
    print("\n=== Extraction complete ===")
    print(f"Elapsed        : {elapsed:.1f}s  ({len(df)/elapsed:,.0f} deduped msgs/sec)")
    print(f"Total messages : {len(df):,} (after cross-mailbox de-duplication)")
    print(f"Date range     : {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Unique senders : {df['from_email'].n_unique():,}")
    print(f"Written to     : {out_path}")
    print(f"Active cache   : {cache_path}")

    print("\nTop senders by message count:")
    top = (
        df.group_by("from_email")
        .len()
        .sort("len", descending=True)
        .head(15)
    )
    for row in top.iter_rows(named=True):
        print(f"  {row['len']:6,}  {row['from_email']}")

    detected = AppConfig.detect_internal_domains(df["from_email"].unique().to_list())
    print(f"\nAuto-detected internal domains: {detected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
