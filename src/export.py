"""Export utilities for downloading DataFrames as CSV, Excel, PowerPoint, GraphML, JSON."""

import io
import json

import streamlit as st
import polars as pl
import networkx as nx

from src.anonymize import anon_df


def download_csv_button(df: pl.DataFrame, filename: str, label: str = "Download CSV") -> None:
    """Render a Streamlit download button for a Polars DataFrame as CSV."""
    csv_bytes = anon_df(df).write_csv().encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


def download_excel_button(df: pl.DataFrame, filename: str, label: str = "Download Excel") -> None:
    """Render a Streamlit download button for a Polars DataFrame as Excel."""
    buf = io.BytesIO()
    anon_df(df).to_pandas().to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def download_graphml_button(
    G: nx.DiGraph,
    graph_metrics: pl.DataFrame,
    filename: str = "network.graphml",
    label: str = "Download GraphML",
) -> None:
    """Export network graph as GraphML with node attributes."""
    # Enrich graph with node attributes from graph_metrics
    G_export = G.copy()
    metrics_dict = {}
    for row in graph_metrics.iter_rows(named=True):
        metrics_dict[row["email"]] = row

    for node in G_export.nodes():
        m = metrics_dict.get(node, {})
        G_export.nodes[node]["betweenness"] = m.get("betweenness_centrality", 0.0)
        G_export.nodes[node]["pagerank"] = m.get("pagerank", 0.0)
        G_export.nodes[node]["community"] = m.get("community_id", -1)
        G_export.nodes[node]["community_label"] = m.get("community_label", "")
        G_export.nodes[node]["in_degree"] = m.get("in_degree", 0)
        G_export.nodes[node]["out_degree"] = m.get("out_degree", 0)

    buf = io.BytesIO()
    nx.write_graphml(G_export, buf)
    buf.seek(0)

    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=filename,
        mime="application/xml",
    )


def download_network_json_button(
    G: nx.DiGraph,
    graph_metrics: pl.DataFrame,
    filename: str = "network.json",
    label: str = "Download JSON",
) -> None:
    """Export network graph as JSON (nodes + edges with attributes)."""
    metrics_dict = {}
    for row in graph_metrics.iter_rows(named=True):
        metrics_dict[row["email"]] = row

    nodes = []
    for node in G.nodes():
        m = metrics_dict.get(node, {})
        nodes.append({
            "id": node,
            "betweenness": m.get("betweenness_centrality", 0.0),
            "pagerank": m.get("pagerank", 0.0),
            "community": m.get("community_id", -1),
            "community_label": m.get("community_label", ""),
            "in_degree": m.get("in_degree", 0),
            "out_degree": m.get("out_degree", 0),
        })

    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "weight": data.get("weight", 1),
            "total_bytes": data.get("total_bytes", 0),
        })

    network_data = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "directed": True,
        },
    }

    json_bytes = json.dumps(network_data, indent=2, default=str).encode("utf-8")

    st.download_button(
        label=label,
        data=json_bytes,
        file_name=filename,
        mime="application/json",
    )
