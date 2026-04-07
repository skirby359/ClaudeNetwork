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
    (r"(sql\.|flatfileprocess|importerror|_error@|adminisd|rsnadmin)", "System Process"),
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
    1. Regex pattern matching on the address itself.
    2. Extreme send ratio: >95% sends or <5% sends with 100+ messages.
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
    # Extreme ratio + high volume = likely machine
    ratio_flags = ratios.with_columns(
        (
            ((pl.col("send_ratio") > 0.95) | (pl.col("send_ratio") < 0.05))
            & ((pl.col("sent") + pl.col("received")) > 100)
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
    """Infer manager-team relationships using bidirectional communication.

    A real manager both sends to AND receives from their reports. This method:
    1. Finds all bidirectional pairs (A->B >= min AND B->A >= min).
    2. For each person, counts how many reciprocal contacts they have.
    3. Ranks by reciprocal team size.

    This naturally filters out copiers/bots (which only send, never receive).

    Returns DataFrame with columns: manager, team_size, total_sent_to_team,
    total_recv_from_team, team_members.
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
    """Explode team members into individual manager-report pairs for treemap."""
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
