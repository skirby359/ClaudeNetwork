"""Headless smoke test of the full analytics pipeline on the active dataset.

Builds every core table and runs key analytics with the Conroy internal-domain
settings, catching exceptions — so we find crashes (e.g. from size_bytes=0 or
heavily-automated senders) BEFORE launching Streamlit.

Run: python scripts/smoke_pipeline.py
"""

import sys
import traceback

sys.path.insert(0, ".")

import polars as pl

from src.config import AppConfig, DatasetConfig
from src.transform.fact_tables import build_edge_fact, build_person_dim
from src.transform.weekly_agg import build_weekly_agg
from src.transform.timing import build_timing_metrics
from src.transform.broadcast import build_broadcast_metrics
from src.analytics.network import build_network_graph, compute_graph_metrics
from src.analytics.size_forensics import (
    detect_size_templates,
    compute_sender_size_profile,
    detect_size_anomalies,
)

results = []


def step(name, fn):
    try:
        out = fn()
        n = len(out) if hasattr(out, "__len__") else "ok"
        results.append((name, "OK", n))
        return out
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        traceback.print_exc()
        return None


def main():
    config = AppConfig()
    dataset = DatasetConfig(
        name="conroy",
        internal_domains=["conroy4congress.com", "conroy4congress.onmicrosoft.com"],
    )

    mf = pl.read_parquet(config.cache_path(config.message_fact_file))
    print(f"message_fact: {len(mf):,} rows, cols={mf.columns}")

    ef = step("edge_fact", lambda: build_edge_fact(mf, config))
    if ef is not None:
        step("person_dim", lambda: build_person_dim(ef, mf, config, dataset=dataset))
        step("weekly_agg", lambda: build_weekly_agg(mf, ef, config))
        G = step("network_graph", lambda: build_network_graph(ef, config))
        if G is not None:
            step("graph_metrics", lambda: compute_graph_metrics(G, config))
    step("timing_metrics", lambda: build_timing_metrics(mf, config))
    step("broadcast_metrics", lambda: build_broadcast_metrics(mf, config))
    # Size analytics on size_bytes=0 data — most likely to divide-by-zero / empty
    step("size_templates", lambda: detect_size_templates(mf))
    step("size_sender_profile", lambda: compute_sender_size_profile(mf))
    step("size_anomalies", lambda: detect_size_anomalies(mf))

    print("\n=== Smoke results ===")
    ok = sum(1 for _, s, _ in results if s == "OK")
    for name, status, detail in results:
        print(f"  [{status:4}] {name:18} {detail}")
    print(f"\n{ok}/{len(results)} steps OK")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
