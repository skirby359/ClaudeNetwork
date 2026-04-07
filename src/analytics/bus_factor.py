"""Key-Person Dependency / Bus Factor: articulation points, team risk, succession."""

import networkx as nx
import polars as pl


def find_articulation_points(G: nx.DiGraph) -> list[str]:
    """Find articulation points (cut vertices) in the undirected projection.

    These are people whose removal would disconnect the network.
    """
    G_und = G.to_undirected()
    return list(nx.articulation_points(G_und))


def compute_team_bus_factor(
    teams: pl.DataFrame,
    G: nx.DiGraph,
) -> pl.DataFrame:
    """For each team, compute bus factor = minimum removals to disconnect the team.

    Iteratively removes members (highest betweenness first) until the team's
    subgraph becomes disconnected from the rest of the network. The count of
    removals needed = bus factor.
    """
    if len(teams) == 0:
        return pl.DataFrame({
            "manager": [], "team_size": [], "bus_factor": [],
            "critical_members": [], "risk_level": [],
        })

    G_und = G.to_undirected()

    # Pre-compute betweenness for removal ordering
    if G.number_of_nodes() > 5000:
        betweenness = nx.betweenness_centrality(G, weight="weight", k=500)
    else:
        betweenness = nx.betweenness_centrality(G, weight="weight")

    records = []
    for row in teams.iter_rows(named=True):
        manager = row["manager"]
        team_members_raw = row["team_members"]
        if isinstance(team_members_raw, list):
            team_members = set(team_members_raw) | {manager}
        else:
            team_members = {manager}

        # Only consider members actually in the graph
        team_in_graph = team_members & set(G_und.nodes())
        if len(team_in_graph) < 2:
            records.append({
                "manager": manager,
                "team_size": row["team_size"],
                "bus_factor": 0,
                "critical_members": [],
                "risk_level": "critical",
            })
            continue

        # Simulate removals: remove highest-betweenness team member first
        G_sim = G_und.copy()
        ordered = sorted(team_in_graph, key=lambda n: betweenness.get(n, 0), reverse=True)
        removed = []
        bus_factor = len(ordered)  # default: takes removing everyone to disconnect

        for person in ordered:
            if person not in G_sim:
                continue
            G_sim.remove_node(person)
            removed.append(person)

            # Check if remaining team members are still connected to each other
            remaining = team_in_graph - set(removed)
            if len(remaining) < 2:
                bus_factor = len(removed)
                break

            # Use component membership for O(V+E) check instead of O(n*(V+E))
            remaining_list = list(remaining)
            ref = remaining_list[0]
            if ref not in G_sim:
                bus_factor = len(removed)
                break

            ref_component = set(nx.node_connected_component(G_sim, ref))
            if not all(other in ref_component for other in remaining_list[1:]):
                bus_factor = len(removed)
                break

        risk = "critical" if bus_factor <= 1 else ("warning" if bus_factor <= 2 else "ok")

        records.append({
            "manager": manager,
            "team_size": row["team_size"],
            "bus_factor": bus_factor,
            "critical_members": removed[:bus_factor],
            "risk_level": risk,
        })

    return pl.DataFrame(records).sort("bus_factor")


def compute_succession_readiness(
    G: nx.DiGraph,
    articulation_points: list[str],
) -> pl.DataFrame:
    """For each articulation point, find the best successor candidate.

    Successor = person with highest contact network overlap who could fill the role.
    Overlap = |contacts(A) intersect contacts(B)| / |contacts(A)|
    """
    if not articulation_points:
        return pl.DataFrame({
            "critical_person": [], "successor_candidate": [],
            "contact_overlap_pct": [], "shared_contacts": [],
            "unique_to_critical": [], "readiness_score": [],
        })

    G_und = G.to_undirected()

    records = []
    for person in articulation_points:
        if person not in G_und:
            continue

        contacts_a = set(G_und.neighbors(person))
        if not contacts_a:
            continue

        best_overlap = 0.0
        best_candidate = None
        best_shared = 0

        # Check each contact as a potential successor
        for candidate in contacts_a:
            contacts_b = set(G_und.neighbors(candidate))
            shared = len(contacts_a & contacts_b)
            overlap = shared / len(contacts_a)

            if overlap > best_overlap:
                best_overlap = overlap
                best_candidate = candidate
                best_shared = shared

        if best_candidate:
            records.append({
                "critical_person": person,
                "successor_candidate": best_candidate,
                "contact_overlap_pct": round(best_overlap * 100, 1),
                "shared_contacts": best_shared,
                "unique_to_critical": len(contacts_a) - best_shared,
                "readiness_score": round(best_overlap * 100, 0),
            })

    if not records:
        return pl.DataFrame({
            "critical_person": [], "successor_candidate": [],
            "contact_overlap_pct": [], "shared_contacts": [],
            "unique_to_critical": [], "readiness_score": [],
        })

    return pl.DataFrame(records).sort("readiness_score", descending=True)


def compute_dependency_risk_matrix(
    G: nx.DiGraph,
    graph_metrics: pl.DataFrame,
    articulation_points: list[str],
) -> pl.DataFrame:
    """Combined risk view: articulation points, centrality, bridging role."""
    if len(graph_metrics) == 0:
        return pl.DataFrame({
            "email": [], "is_articulation_point": [],
            "betweenness": [], "pagerank": [],
            "communities_bridged": [], "risk_score": [],
        })

    ap_set = set(articulation_points)

    risk = graph_metrics.select([
        "email", "betweenness_centrality", "pagerank", "community_id",
    ]).rename({"betweenness_centrality": "betweenness"})

    risk = risk.with_columns(
        pl.col("email").is_in(list(ap_set)).alias("is_articulation_point")
    )

    # Count communities bridged per person
    G_und = G.to_undirected()
    comm_lookup = dict(zip(
        graph_metrics["email"].to_list(),
        graph_metrics["community_id"].to_list(),
    ))

    bridge_counts = {}
    for node in G_und.nodes():
        if node not in comm_lookup:
            continue
        home = comm_lookup[node]
        ext_comms = set()
        for nb in G_und.neighbors(node):
            nc = comm_lookup.get(nb)
            if nc is not None and nc != home:
                ext_comms.add(nc)
        bridge_counts[node] = len(ext_comms) + 1 if ext_comms else 1

    bridge_df = pl.DataFrame({
        "email": list(bridge_counts.keys()),
        "communities_bridged": list(bridge_counts.values()),
    })

    risk = risk.join(bridge_df, on="email", how="left")
    risk = risk.with_columns(
        pl.col("communities_bridged").fill_null(1)
    )

    # Composite risk score: weighted combination
    # Normalize each factor to 0-1, then combine
    max_betw = risk["betweenness"].max()
    max_pr = risk["pagerank"].max()
    max_bridge = risk["communities_bridged"].max()

    risk = risk.with_columns(
        (
            (pl.col("is_articulation_point").cast(pl.Float64) * 40)
            + (pl.col("betweenness") / max(max_betw, 1e-10) * 30)
            + (pl.col("communities_bridged") / max(max_bridge, 1) * 20)
            + (pl.col("pagerank") / max(max_pr, 1e-10) * 10)
        ).alias("risk_score")
    )

    return risk.sort("risk_score", descending=True)
