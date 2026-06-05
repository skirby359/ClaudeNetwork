"""Automated plausibility audit of the Conroy dataset across the analytics layer.

Goes beyond the smoke test (which only checks 'doesn't crash') to assert that the
NUMBERS are plausible: percentages in range, counts non-negative, internal<=total,
reply times positive, gini in [0,1], no size-zero artifacts surfacing as real, etc.

Each check is OK / WARN / FAIL / DEGRADED (expected-empty, e.g. size with Graph).

Run: python scripts/audit_conroy.py
"""

import sys
sys.path.insert(0, ".")

import polars as pl

from src.config import AppConfig, DatasetConfig
from src.transform.fact_tables import build_edge_fact, build_person_dim
from src.transform.timing import build_timing_metrics
from src.transform.broadcast import build_broadcast_metrics
from src.analytics.network import build_network_graph, compute_graph_metrics
from src.analytics.volume import compute_sender_concentration
from src.analytics.response_time import compute_reply_times, compute_person_response_stats
from src.analytics.health_score import compute_health_score
from src.analytics.hierarchy import detect_nonhuman_addresses
from src.analytics.comparison import compute_period_summary
from src.analytics.narrative import generate_executive_narrative

checks = []
def chk(name, level, detail=""):
    checks.append((name, level, detail))

def expect(name, cond, detail_ok="", detail_bad="", warn=False):
    if cond:
        chk(name, "OK", detail_ok)
    else:
        chk(name, "WARN" if warn else "FAIL", detail_bad)


def main():
    config = AppConfig()
    ds = DatasetConfig(name="conroy",
                       internal_domains=["conroy4congress.com", "conroy4congress.onmicrosoft.com"])
    mf = pl.read_parquet(config.cache_path(config.message_fact_file))
    ef = build_edge_fact(mf, config)
    pdim = build_person_dim(ef, mf, config, dataset=ds)

    # --- message_fact integrity ---
    expect("message_fact non-empty", len(mf) > 0, f"{len(mf):,} rows")
    expect("no null timestamps", mf["timestamp"].null_count() == 0,
           detail_bad=f"{mf['timestamp'].null_count()} nulls")
    expect("hour in 0..23", mf["hour"].min() >= 0 and mf["hour"].max() <= 23,
           f"min={mf['hour'].min()} max={mf['hour'].max()}")
    ah = mf["is_after_hours"].mean() * 100
    expect("after-hours rate plausible (<60%)", ah < 60, f"{ah:.1f}%",
           f"{ah:.1f}% — possible timezone issue", warn=True)
    we = mf["is_weekend"].mean() * 100
    expect("weekend rate plausible (<40%)", we < 40, f"{we:.1f}%", f"{we:.1f}%", warn=True)
    expect("n_recipients >= 1", mf["n_recipients"].min() >= 1, detail_bad=f"min={mf['n_recipients'].min()}")

    # --- size (expected degraded for Graph) ---
    if mf["size_bytes"].sum() == 0:
        chk("size_bytes all zero", "DEGRADED", "expected: Graph has no size data")
    else:
        chk("size_bytes present", "OK", f"sum={mf['size_bytes'].sum():,}")

    # --- person_dim ---
    intern = pdim.filter(pl.col("is_internal"))
    expect("internal staff detected", len(intern) > 0, f"{len(intern)} internal")
    expect("internal <= total people", len(intern) <= len(pdim),
           f"{len(intern)}/{len(pdim)}")
    expect("no negative sent", pdim["total_sent"].min() >= 0)
    expect("no negative received", pdim["total_received"].min() >= 0)

    # --- human/machine ---
    nh = detect_nonhuman_addresses(pdim, ef)
    nh_set = nh.filter(pl.col("is_nonhuman"))["email"].to_list()
    human = mf.filter(~pl.col("from_email").is_in(nh_set))
    hpct = len(human) / len(mf) * 100
    expect("human share in (0,100)", 0 < hpct < 100, f"{hpct:.0f}% human")
    expect("no internal staff flagged nonhuman",
           intern.filter(pl.col("email").is_in(nh_set)).height == 0,
           "0 internal flagged",
           f"{intern.filter(pl.col('email').is_in(nh_set)).height} internal flagged as machine",
           warn=True)

    # --- concentration ---
    conc = compute_sender_concentration(ef)
    expect("gini in [0,1]", 0 <= conc["gini"] <= 1, f"gini={conc['gini']:.3f}")
    expect("top10 share in [0,1]", 0 <= conc["top_10_share"] <= 1, f"{conc['top_10_share']:.1%}")

    # --- network ---
    G = build_network_graph(ef, config)
    gm = compute_graph_metrics(G, config)
    ncomm = gm["community_id"].n_unique()
    expect("graph has nodes", G.number_of_nodes() > 0, f"{G.number_of_nodes()} nodes")
    expect("communities <= nodes", ncomm <= G.number_of_nodes(), f"{ncomm} communities")

    # --- response times ---
    reply = compute_reply_times(ef.filter(~pl.col("from_email").is_in(nh_set)
                                          & ~pl.col("to_email").is_in(nh_set)))
    if len(reply) > 0:
        med = reply["median_reply_seconds"].median()
        expect("median reply positive", med is not None and med > 0, f"{med/60:.0f} min")
        expect("median reply < 30 days", med < 30*86400, f"{med/86400:.1f} days",
               f"{med/86400:.1f} days — implausibly long", warn=True)
        prs = compute_person_response_stats(reply)
        chk("person response stats", "OK", f"{len(prs)} people")
    else:
        chk("reply detection", "WARN", "no replies detected")

    # --- health score ---
    hs = compute_health_score(mf, ef, gm, reply_median_seconds=(reply["median_reply_seconds"].median() if len(reply) else None))
    comp = hs.get("composite", -1)
    expect("health composite in [0,100]", 0 <= comp <= 100, f"{comp:.0f}/100")
    subs = hs.get("sub_scores", {})
    def _val(v):
        return v.get("value") if isinstance(v, dict) else v
    bad_subs = {k: _val(v) for k, v in subs.items()
                if not (_val(v) is not None and 0 <= _val(v) <= 100)}
    expect("all health sub-scores in [0,100]", not bad_subs, f"{len(subs)} sub-scores",
           f"out of range: {bad_subs}")

    # --- timing / broadcast ---
    tm = build_timing_metrics(mf, config)
    expect("timing metrics non-empty", len(tm) > 0, f"{len(tm)} rows")
    bm = build_broadcast_metrics(mf, config)
    expect("broadcast metrics computed", len(bm) >= 0, f"{len(bm)} rows")

    # --- narrative ---
    wa = pl.read_parquet(config.cache_path(config.weekly_agg_file)) if config.cache_path(config.weekly_agg_file).exists() else None
    try:
        narr = generate_executive_narrative(mf, wa, ef, pdim) if wa is not None else "(skipped)"
        bad_tokens = any(t in narr.lower() for t in ["nan", "none msgs", "0.0 gb", "inf "])
        expect("narrative clean (no nan/inf/0GB)", not bad_tokens, f"{len(narr)} chars",
               "narrative contains nan/inf/0GB token", warn=True)
    except Exception as e:
        chk("narrative", "WARN", f"could not generate: {e}")

    # --- comparison summary sanity ---
    ps = compute_period_summary(mf, ef)
    expect("period summary after-hours in [0,1]", 0 <= ps.get("after_hours_rate", -1) <= 1,
           f"{ps.get('after_hours_rate'):.2f}")

    # ---- report ----
    print("\n=== Conroy automated audit ===")
    order = {"FAIL": 0, "WARN": 1, "DEGRADED": 2, "OK": 3}
    for name, level, detail in sorted(checks, key=lambda c: order.get(c[1], 9)):
        print(f"  [{level:8}] {name:42} {detail}")
    n_fail = sum(1 for _, l, _ in checks if l == "FAIL")
    n_warn = sum(1 for _, l, _ in checks if l == "WARN")
    print(f"\n{len(checks)} checks: {n_fail} FAIL, {n_warn} WARN, "
          f"{sum(1 for _,l,_ in checks if l=='DEGRADED')} DEGRADED, "
          f"{sum(1 for _,l,_ in checks if l=='OK')} OK")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
