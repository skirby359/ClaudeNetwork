"""Network analytics: graph build, centrality, communities, dyads."""

import networkx as nx
import community as community_louvain
import polars as pl

from src.config import AppConfig
from src.cache_manager import cached_parquet, cached_pickle


# ---------------------------------------------------------------------------
# Pure computation functions (no file caching)
# ---------------------------------------------------------------------------

def build_graph(edge_fact: pl.DataFrame) -> nx.DiGraph:
    """Build a directed weighted graph from edge_fact."""
    G = nx.DiGraph()
    edge_weights = (
        edge_fact.group_by(["from_email", "to_email"])
        .agg([
            pl.len().alias("weight"),
            pl.col("size_bytes").sum().alias("total_bytes"),
        ])
    )
    froms = edge_weights["from_email"].to_list()
    tos = edge_weights["to_email"].to_list()
    weights = edge_weights["weight"].to_list()
    bytes_list = edge_weights["total_bytes"].to_list()
    for f, t, w, b in zip(froms, tos, weights, bytes_list):
        G.add_edge(f, t, weight=w, total_bytes=b)
    return G


def compute_node_metrics(G: nx.DiGraph) -> pl.DataFrame:
    """Compute per-node graph metrics from a NetworkX graph."""
    in_degree = dict(G.in_degree(weight="weight"))
    out_degree = dict(G.out_degree(weight="weight"))
    pagerank = nx.pagerank(G, weight="weight")

    if G.number_of_nodes() > 5000:
        betweenness = nx.betweenness_centrality(G, weight="weight", k=500)
    else:
        betweenness = nx.betweenness_centrality(G, weight="weight")

    G_undirected = G.to_undirected()
    partition = community_louvain.best_partition(G_undirected, weight="weight")

    records = []
    for node in G.nodes():
        records.append({
            "email": node,
            "in_degree": in_degree.get(node, 0),
            "out_degree": out_degree.get(node, 0),
            "betweenness_centrality": betweenness.get(node, 0.0),
            "community_id": partition.get(node, -1),
            "pagerank": pagerank.get(node, 0.0),
        })
    return pl.DataFrame(records)


def compute_dyads(edge_fact: pl.DataFrame) -> pl.DataFrame:
    """Compute dyad analysis from edge_fact."""
    pairs = (
        edge_fact.group_by(["from_email", "to_email"])
        .agg([
            pl.len().alias("msg_count"),
            pl.col("size_bytes").sum().alias("total_bytes"),
        ])
    )

    pairs = pairs.with_columns([
        pl.min_horizontal("from_email", "to_email").alias("person_a"),
        pl.max_horizontal("from_email", "to_email").alias("person_b"),
    ])

    a_to_b = (
        pairs.filter(pl.col("from_email") == pl.col("person_a"))
        .select([
            pl.col("person_a"), pl.col("person_b"),
            pl.col("msg_count").alias("a_to_b_count"),
            pl.col("total_bytes").alias("a_to_b_bytes"),
        ])
    )

    b_to_a = (
        pairs.filter(pl.col("from_email") == pl.col("person_b"))
        .select([
            pl.col("person_a"), pl.col("person_b"),
            pl.col("msg_count").alias("b_to_a_count"),
            pl.col("total_bytes").alias("b_to_a_bytes"),
        ])
    )

    dyads = a_to_b.join(b_to_a, on=["person_a", "person_b"], how="full", coalesce=True)
    dyads = dyads.with_columns([
        pl.col("a_to_b_count").fill_null(0).cast(pl.Int64),
        pl.col("b_to_a_count").fill_null(0).cast(pl.Int64),
        pl.col("a_to_b_bytes").fill_null(0).cast(pl.Int64),
        pl.col("b_to_a_bytes").fill_null(0).cast(pl.Int64),
    ])

    dyads = dyads.filter(pl.col("person_a") != pl.col("person_b"))

    dyads = dyads.with_columns([
        (pl.col("a_to_b_count") + pl.col("b_to_a_count")).alias("total_pair_msgs"),
        (pl.col("a_to_b_bytes") + pl.col("b_to_a_bytes")).alias("total_pair_bytes"),
        (
            (pl.col("a_to_b_count") - pl.col("b_to_a_count")).abs().cast(pl.Float64)
            / (pl.col("a_to_b_count") + pl.col("b_to_a_count")).cast(pl.Float64)
        ).alias("asymmetry_ratio"),
    ])

    dyads = dyads.rename({"person_a": "from_email", "person_b": "to_email"})
    return dyads.sort("total_pair_msgs", descending=True)


# ---------------------------------------------------------------------------
# File-cached wrappers (used for full-dataset precomputation)
# ---------------------------------------------------------------------------

def build_network_graph(edge_fact: pl.DataFrame, config: AppConfig) -> nx.DiGraph:
    """Build graph with file caching."""
    cache_path = config.cache_path(config.network_graph_file)
    source_paths = [config.cache_path(config.edge_fact_file)]
    return cached_pickle(cache_path, source_paths, lambda: build_graph(edge_fact))


def compute_graph_metrics(G: nx.DiGraph, config: AppConfig) -> pl.DataFrame:
    """Compute graph metrics with file caching."""
    cache_path = config.cache_path(config.graph_metrics_file)
    source_paths = [config.cache_path(config.network_graph_file)]
    return cached_parquet(cache_path, source_paths, lambda: compute_node_metrics(G))


def compute_dyad_analysis(edge_fact: pl.DataFrame, config: AppConfig) -> pl.DataFrame:
    """Compute dyad analysis with file caching."""
    cache_path = config.cache_path(config.dyad_analysis_file)
    source_paths = [config.cache_path(config.edge_fact_file)]
    return cached_parquet(cache_path, source_paths, lambda: compute_dyads(edge_fact))
