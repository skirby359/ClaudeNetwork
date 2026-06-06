"""Organizational hierarchy inference from email patterns."""

import re
import polars as pl


# ---------------------------------------------------------------------------
# Nonhuman address detection
# ---------------------------------------------------------------------------

# Regex patterns that strongly suggest a nonhuman/automated address
NONHUMAN_PATTERNS = [
    r"^(noreply|no[-_.]?reply|donotreply|do[-_.]?not[-_.]?reply)",
    r"^(postmaster|mailer[-_.]?daemon|system[-_.]?administrator)",
    r"^(automail|auto[-_.]?notify|auto[-_.]?response)",
    r"(copier|scanner|ricoh|canon|ikon|xerox|konica)",
    r"^(sql\.|rightfax|fax|eventlog)",
    r"(flatfileprocess|importerror|_error@)",
    r"^(microsoftexchange|exchange329)",
    r"^(hiplink|sourcefire|blueteam|p25radio)",
    r"^(scomcml|adminisd|rsnadmin)",
    # Hosting/system accounts. Anchored to a complete token (next char must be a
    # delimiter, digit, @, or end) so surnames like "Cronin" aren't misflagged.
    r"^(cpanel|whm|hostmaster|webmaster|root|cron|bounce|daemon)([-_.@0-9]|$)",
]

_NONHUMAN_RE = re.compile("|".join(NONHUMAN_PATTERNS), re.IGNORECASE)

# Type classification patterns for automated senders
_TYPE_PATTERNS = [
    (r"(copier|scanner|ricoh|canon|ikon|xerox|konica|mfp|printer)", "Copier/Scanner"),
    (r"(fax|rightfax)", "Fax"),
    (r"(noreply|no[-_.]?reply|donotreply|do[-_.]?not[-_.]?reply)", "Notification"),
    (r"(alert|hiplink|sourcefire|blueteam|scom|eventlog|p25radio)", "Alert/Monitoring"),
    (r"(postmaster|mailer[-_.]?daemon|microsoftexchange|exchange329)", "Mail Infrastructure"),
    (r"(automail|auto[-_.]?notify|auto[-_.]?response)", "Auto-Response"),
    (r"(sql\.|flatfileprocess|importerror|_error@|adminisd|rsnadmin|cpanel|whm|hostmaster|cron|bounce|daemon)", "System Process"),
]
_TYPE_COMPILED = [(re.compile(p, re.IGNORECASE), t) for p, t in _TYPE_PATTERNS]


def is_likely_nonhuman(email: str) -> bool:
    """Check if an email address looks like a machine/system account."""
    return bool(_NONHUMAN_RE.search(email))


def classify_nonhuman_type(email: str) -> str:
    """Classify an automated address into a category."""
    for pattern, type_name in _TYPE_COMPILED:
        if pattern.search(email):
            return type_name
    return "Other Automated"


def detect_nonhuman_addresses(person_dim: pl.DataFrame, edge_fact: pl.DataFrame) -> pl.DataFrame:
    """Flag addresses that are likely nonhuman.

    Uses two signals:
    1. Regex pattern matching on the address itself (noreply, copier, etc.).
    2. High-volume one-way *senders*: >95% send ratio with 100+ messages sent.

    Note: a high *receive* ratio is deliberately NOT treated as a machine signal.
    An address that mostly receives is usually a human on distribution lists (a
    donor or constituent who gets blasts but rarely replies), not an automated
    system — flagging those produced many false positives on real client data.
    Named system accounts that only receive are still caught by pattern matching.
    """
    emails = person_dim["email"].to_list()

    # Pattern-based detection
    pattern_flags = [is_likely_nonhuman(e) for e in emails]

    # Ratio-based detection: compute per-person send ratio
    sent_counts = (
        edge_fact.group_by("from_email")
        .agg(pl.len().alias("sent"))
        .rename({"from_email": "email"})
    )
    recv_counts = (
        edge_fact.group_by("to_email")
        .agg(pl.len().alias("received"))
        .rename({"to_email": "email"})
    )
    ratios = sent_counts.join(recv_counts, on="email", how="full", coalesce=True)
    ratios = ratios.with_columns([
        pl.col("sent").fill_null(0),
        pl.col("received").fill_null(0),
    ])
    ratios = ratios.with_columns(
        (pl.col("sent").cast(pl.Float64) / (pl.col("sent") + pl.col("received")).cast(pl.Float64))
        .alias("send_ratio")
    )
    # High-volume one-way sender = likely machine. Only the high-SEND-ratio side
    # is a machine signal (blasters that never receive); mostly-receiving is not.
    ratio_flags = ratios.with_columns(
        (
            (pl.col("send_ratio") > 0.95)
            & (pl.col("sent") > 100)
        ).alias("ratio_nonhuman")
    ).select(["email", "send_ratio", "ratio_nonhuman"])

    # Combine
    result = person_dim.with_columns(
        pl.Series("pattern_nonhuman", pattern_flags)
    )
    result = result.join(ratio_flags, on="email", how="left")
    result = result.with_columns(
        (pl.col("pattern_nonhuman") | pl.col("ratio_nonhuman").fill_null(False)).alias("is_nonhuman")
    )

    return result


# ---------------------------------------------------------------------------
# Hierarchy scoring (original algorithm, kept for the scatter plot)
# ---------------------------------------------------------------------------

def compute_hierarchy_score(edge_fact: pl.DataFrame, person_dim: pl.DataFrame) -> pl.DataFrame:
    """Score each person's leadership likelihood.

    Score = (unique_recipients / unique_senders_to_them) + recipient_diversity.
    High score = sends to many, receives from few, sends broadcasts.
    """
    sent_stats = (
        edge_fact.group_by("from_email")
        .agg([
            pl.col("to_email").n_unique().alias("unique_recipients"),
            pl.len().alias("total_sent"),
            (pl.col("to_email").n_unique().cast(pl.Float64) / pl.len()).alias("recipient_diversity"),
        ])
        .rename({"from_email": "email"})
    )

    recv_stats = (
        edge_fact.group_by("to_email")
        .agg([
            pl.col("from_email").n_unique().alias("unique_senders_to"),
            pl.len().alias("total_received"),
        ])
        .rename({"to_email": "email"})
    )

    scores = sent_stats.join(recv_stats, on="email", how="full", coalesce=True)
    scores = scores.with_columns([
        pl.col("unique_recipients").fill_null(0),
        pl.col("total_sent").fill_null(0),
        pl.col("unique_senders_to").fill_null(1),
        pl.col("total_received").fill_null(0),
    ])
    scores = scores.with_columns(
        (
            pl.col("unique_recipients").cast(pl.Float64)
            / pl.col("unique_senders_to").cast(pl.Float64).clip(lower_bound=1)
            + pl.col("recipient_diversity").fill_null(0.0)
        ).alias("hierarchy_score")
    )
    scores = scores.join(
        person_dim.select(["email", "domain", "is_internal", "display_name"]),
        on="email",
        how="left",
    )
    return scores.sort("hierarchy_score", descending=True)


# ---------------------------------------------------------------------------
# Reciprocal team detection (better algorithm for real manager-report pairs)
# ---------------------------------------------------------------------------

def infer_reciprocal_teams(
    edge_fact: pl.DataFrame,
    person_dim: pl.DataFrame,
    min_msgs_per_direction: int = 5,
    min_team_size: int = 3,
    exclude_emails: set[str] | None = None,
) -> pl.DataFrame:
    """Find mutual-communication groups using bidirectional communication.

    IMPORTANT: this measures *mutual* (two-way) communication, which is
    SYMMETRIC and has no direction — it does NOT infer who supervises whom.
    If A<->B communicate mutually, A appears in B's group and B in A's group.
    Do not present the output as a manager/report org chart. (For a directional
    leadership signal, use compute_hierarchy_score instead.)

    Method:
    1. Finds all bidirectional pairs (A->B >= min AND B->A >= min).
    2. For each person, counts how many mutual contacts they have.
    3. Ranks by number of mutual contacts.

    This naturally filters out copiers/bots (which only send, never receive).

    Returns DataFrame with columns: manager (the person — kept for back-compat),
    team_size (mutual-contact count), total_sent_to_team, total_recv_from_team,
    team_members (their mutual contacts).
    """
    ef = edge_fact
    if exclude_emails:
        ef = ef.filter(
            ~pl.col("from_email").is_in(list(exclude_emails))
            & ~pl.col("to_email").is_in(list(exclude_emails))
        )

    # Forward counts
    fwd = (
        ef.group_by(["from_email", "to_email"])
        .agg(pl.len().alias("fwd_count"))
        .filter(pl.col("fwd_count") >= min_msgs_per_direction)
    )
    # Reverse counts
    rev = (
        ef.group_by(["from_email", "to_email"])
        .agg(pl.len().alias("rev_count"))
        .filter(pl.col("rev_count") >= min_msgs_per_direction)
        .rename({"from_email": "r_from", "to_email": "r_to"})
    )

    # Join to find reciprocal pairs
    recip = fwd.join(
        rev,
        left_on=["from_email", "to_email"],
        right_on=["r_to", "r_from"],
        how="inner",
    )

    # Aggregate per potential manager
    teams = (
        recip.group_by("from_email")
        .agg([
            pl.col("to_email").alias("team_members"),
            pl.col("fwd_count").sum().alias("total_sent_to_team"),
            pl.col("rev_count").sum().alias("total_recv_from_team"),
            pl.len().alias("team_size"),
        ])
        .filter(pl.col("team_size") >= min_team_size)
        .rename({"from_email": "manager"})
        .sort("team_size", descending=True)
    )

    # Add display name
    teams = teams.join(
        person_dim.select(["email", "display_name"]).rename({"email": "manager"}),
        on="manager",
        how="left",
    )

    return teams


def build_reporting_pairs_from_teams(teams: pl.DataFrame) -> pl.DataFrame:
    """Explode mutual contacts into person/contact pairs for the treemap.

    These pairs are undirected mutual relationships, NOT manager->report edges;
    a pair (A, B) also appears as (B, A). Column names manager/report are kept
    for back-compat but mean person/mutual-contact.
    """
    if len(teams) == 0:
        return pl.DataFrame({"manager": [], "report": [], "msg_count": []})

    # Explode team_members list and pair with sent counts
    # We need per-pair counts, so re-derive from the teams DataFrame
    exploded = teams.select(["manager", "team_members", "total_sent_to_team"]).explode("team_members")
    exploded = exploded.rename({"team_members": "report"})
    # Approximate per-report count by dividing evenly (exact counts lost in agg)
    exploded = exploded.with_columns(
        (pl.col("total_sent_to_team") / pl.len().over("manager")).cast(pl.Int64).alias("msg_count")
    )
    return exploded.select(["manager", "report", "msg_count"])


# ---------------------------------------------------------------------------
# Legacy wrappers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def infer_calculated_hierarchy(
    edge_fact: pl.DataFrame,
    hierarchy_scores: pl.DataFrame,
    min_msgs: int = 5,
) -> pl.DataFrame:
    """Infer a directional reporting tree from communication patterns.

    ALGORITHMIC ESTIMATE — NOT an official org chart. Method: each person is
    linked to the higher-"authority" person they most communicate with. Authority
    = hierarchy_score (sends to many, receives from few). For each person we pick,
    among their contacts whose score is STRICTLY higher (and with >= min_msgs
    exchanged), the highest-scoring one as their inferred supervisor. Strict
    ordering guarantees the result is acyclic (a tree); the top-scoring person(s)
    become roots.

    Returns: email, display_name, inferred_manager, hierarchy_score, msg_weight,
    total_volume.
    """
    from collections import defaultdict

    schema = {
        "email": pl.String, "display_name": pl.String, "inferred_manager": pl.String,
        "hierarchy_score": pl.Float64, "msg_weight": pl.Int64, "total_volume": pl.Int64,
    }
    if len(hierarchy_scores) == 0 or len(edge_fact) == 0:
        return pl.DataFrame(schema=schema)

    hs = hierarchy_scores
    score = dict(zip(hs["email"].to_list(), hs["hierarchy_score"].to_list()))
    name = (dict(zip(hs["email"].to_list(), hs["display_name"].to_list()))
            if "display_name" in hs.columns else {})
    sent = dict(zip(hs["email"].to_list(), hs["total_sent"].fill_null(0).to_list())) if "total_sent" in hs.columns else {}
    recv = dict(zip(hs["email"].to_list(), hs["total_received"].fill_null(0).to_list())) if "total_received" in hs.columns else {}

    # Undirected contact weights (messages exchanged either direction).
    pair_w = defaultdict(int)
    contacts = defaultdict(set)
    pairs = edge_fact.group_by(["from_email", "to_email"]).agg(pl.len().alias("w"))
    for r in pairs.iter_rows(named=True):
        a, b, w = r["from_email"], r["to_email"], r["w"]
        if a == b or a not in score or b not in score:
            continue
        key = (a, b) if a < b else (b, a)
        pair_w[key] += w
        contacts[a].add(b)
        contacts[b].add(a)

    rows = []
    for p in score:
        candidates = []
        for c in contacts.get(p, ()):
            if score[c] > score[p]:  # strictly higher authority -> potential manager
                w = pair_w[(p, c) if p < c else (c, p)]
                if w >= min_msgs:
                    candidates.append((score[c], w, c))
        if candidates:
            candidates.sort(reverse=True)  # highest score, then most messages
            mgr, mw = candidates[0][2], candidates[0][1]
        else:
            mgr, mw = "", 0
        rows.append({
            "email": p,
            "display_name": name.get(p, "") or "",
            "inferred_manager": mgr,
            "hierarchy_score": float(score[p]),
            "msg_weight": int(mw),
            "total_volume": int(sent.get(p, 0) + recv.get(p, 0)),
        })

    return pl.DataFrame(rows, schema=schema).sort("hierarchy_score", descending=True)


def infer_reporting_pairs(
    edge_fact: pl.DataFrame,
    hierarchy_scores: pl.DataFrame,
    top_n: int = 50,
) -> pl.DataFrame:
    """Infer manager-report pairs from high-hierarchy senders (legacy algorithm)."""
    managers = hierarchy_scores.head(top_n)["email"].to_list()
    manager_edges = (
        edge_fact.filter(pl.col("from_email").is_in(managers))
        .group_by(["from_email", "to_email"])
        .agg(pl.len().alias("msg_count"))
        .sort(["from_email", "msg_count"], descending=[False, True])
    )
    pairs = manager_edges.filter(pl.col("msg_count") >= 10)
    pairs = pairs.rename({"from_email": "manager", "to_email": "report"})
    return pairs


def build_hierarchy_tree(reporting_pairs: pl.DataFrame) -> list[tuple[str, str]]:
    """Build a list of (parent, child) tuples for tree visualization."""
    if len(reporting_pairs) == 0:
        return []
    return list(zip(
        reporting_pairs["manager"].to_list(),
        reporting_pairs["report"].to_list(),
    ))
