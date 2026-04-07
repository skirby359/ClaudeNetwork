"""Engagement profile management: save/load per-client settings."""

import json
from datetime import date, datetime
from pathlib import Path

import polars as pl


PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"


def _ensure_dir():
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def list_profiles() -> list[str]:
    """List available engagement profile names."""
    _ensure_dir()
    return sorted(
        p.stem for p in PROFILES_DIR.glob("*.json")
    )


def save_profile(name: str, settings: dict, department_df: pl.DataFrame | None = None):
    """Save an engagement profile to disk.

    settings dict should contain:
        internal_domains, date_format, column_mapping, exclude_nonhuman,
        key_dates, alert_rules, org_name
    """
    _ensure_dir()

    # Serialize dates in key_dates
    key_dates = settings.get("key_dates", [])
    serialized_dates = []
    for kd in key_dates:
        d = kd.get("date")
        if isinstance(d, (date, datetime)):
            d = d.isoformat()
        serialized_dates.append({"label": kd.get("label", ""), "date": d})
    settings["key_dates"] = serialized_dates

    # Save main JSON
    profile_path = PROFILES_DIR / f"{name}.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, default=str)

    # Save department mapping as separate CSV
    if department_df is not None and len(department_df) > 0:
        dept_path = PROFILES_DIR / f"{name}_departments.csv"
        department_df.write_csv(dept_path)


def load_profile(name: str) -> tuple[dict, pl.DataFrame | None]:
    """Load an engagement profile. Returns (settings_dict, department_df_or_None)."""
    profile_path = PROFILES_DIR / f"{name}.json"
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile '{name}' not found")

    with open(profile_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    # Deserialize dates
    key_dates = settings.get("key_dates", [])
    for kd in key_dates:
        d = kd.get("date")
        if isinstance(d, str):
            try:
                kd["date"] = date.fromisoformat(d)
            except (ValueError, TypeError):
                pass

    # Load department mapping if exists
    dept_path = PROFILES_DIR / f"{name}_departments.csv"
    dept_df = None
    if dept_path.exists():
        try:
            dept_df = pl.read_csv(dept_path)
        except Exception:
            pass

    return settings, dept_df


def delete_profile(name: str):
    """Delete an engagement profile."""
    profile_path = PROFILES_DIR / f"{name}.json"
    dept_path = PROFILES_DIR / f"{name}_departments.csv"
    if profile_path.exists():
        profile_path.unlink()
    if dept_path.exists():
        dept_path.unlink()


def collect_current_settings(session_state) -> dict:
    """Collect current session state into a saveable settings dict."""
    return {
        "internal_domains": session_state.get("_internal_domains", []),
        "date_format": session_state.get("_date_format", "%m/%d/%Y %H:%M"),
        "column_mapping": session_state.get("_column_mapping", {
            "date": "Date", "size": "Size", "from": "From", "to": "To",
        }),
        "exclude_nonhuman": session_state.get("exclude_nonhuman", True),
        "key_dates": session_state.get("_compliance_key_dates", []),
        "alert_rules": session_state.get("_alert_rules", default_alert_rules()),
        "org_name": session_state.get("_org_name", "Organization"),
    }


def apply_profile_to_session(settings: dict, dept_df: pl.DataFrame | None, session_state):
    """Apply a loaded profile's settings to session state."""
    if "internal_domains" in settings:
        session_state._internal_domains = settings["internal_domains"]
    if "date_format" in settings:
        session_state._date_format = settings["date_format"]
    if "column_mapping" in settings:
        session_state._column_mapping = settings["column_mapping"]
    if "exclude_nonhuman" in settings:
        session_state.exclude_nonhuman = settings["exclude_nonhuman"]
    if "key_dates" in settings:
        session_state._compliance_key_dates = settings["key_dates"]
    if "alert_rules" in settings:
        session_state._alert_rules = settings["alert_rules"]
    if "org_name" in settings:
        session_state._org_name = settings["org_name"]
    if dept_df is not None:
        session_state._department_mapping = dept_df


# ---------------------------------------------------------------------------
# Alert rules
# ---------------------------------------------------------------------------

def default_alert_rules() -> list[dict]:
    """Default set of alert rules."""
    return [
        {
            "name": "High after-hours rate",
            "metric": "after_hours_pct",
            "operator": ">",
            "threshold": 40.0,
            "severity": "warning",
            "description": "Person sends >40% of messages outside business hours",
        },
        {
            "name": "Bus factor critical",
            "metric": "bus_factor",
            "operator": "<=",
            "threshold": 1,
            "severity": "critical",
            "description": "Team has bus factor of 1 or less",
        },
        {
            "name": "Communication blackout",
            "metric": "blackout_hours",
            "operator": ">",
            "threshold": 72.0,
            "severity": "warning",
            "description": "Active sender goes silent for >72 hours",
        },
        {
            "name": "External contact spike",
            "metric": "external_spike_z",
            "operator": ">",
            "threshold": 2.5,
            "severity": "warning",
            "description": "External contact count exceeds 2.5 standard deviations",
        },
        {
            "name": "High concentration",
            "metric": "gini",
            "operator": ">",
            "threshold": 0.7,
            "severity": "info",
            "description": "Communication volume is highly concentrated",
        },
        {
            "name": "Low health score",
            "metric": "health_composite",
            "operator": "<",
            "threshold": 50.0,
            "severity": "critical",
            "description": "Overall organizational health score below 50",
        },
    ]


def evaluate_alerts(
    alert_rules: list[dict],
    message_fact: pl.DataFrame,
    edge_fact: pl.DataFrame,
    graph_metrics: pl.DataFrame,
    health_score: dict | None = None,
    blackouts: pl.DataFrame | None = None,
    team_bus_factor: pl.DataFrame | None = None,
) -> list[dict]:
    """Evaluate alert rules against current data. Returns list of triggered alerts."""
    triggered = []

    for rule in alert_rules:
        metric = rule["metric"]
        op = rule["operator"]
        threshold = rule["threshold"]

        try:
            findings = _check_rule(
                metric, op, threshold, rule,
                message_fact, edge_fact, graph_metrics,
                health_score, blackouts, team_bus_factor,
            )
            triggered.extend(findings)
        except Exception:
            continue

    return triggered


def _check_rule(
    metric, op, threshold, rule,
    message_fact, edge_fact, graph_metrics,
    health_score, blackouts, team_bus_factor,
) -> list[dict]:
    """Check a single alert rule. Returns list of triggered alert dicts."""
    findings = []

    if metric == "after_hours_pct" and len(message_fact) > 0:
        # Per-person after-hours rate
        person_ah = (
            message_fact.group_by("from_email")
            .agg([
                pl.col("is_after_hours").mean().alias("ah_rate"),
                pl.len().alias("msg_count"),
            ])
            .filter(pl.col("msg_count") >= 10)
        )
        flagged = _apply_op(person_ah, "ah_rate", op, threshold / 100.0)
        for row in flagged.iter_rows(named=True):
            findings.append({
                **rule,
                "entity": row["from_email"],
                "value": round(row["ah_rate"] * 100, 1),
                "detail": f"{row['ah_rate']*100:.1f}% after-hours ({row['msg_count']} msgs)",
            })

    elif metric == "bus_factor" and team_bus_factor is not None and len(team_bus_factor) > 0:
        flagged = _apply_op(team_bus_factor, "bus_factor", op, threshold)
        for row in flagged.iter_rows(named=True):
            findings.append({
                **rule,
                "entity": row["manager"],
                "value": row["bus_factor"],
                "detail": f"Team bus factor: {row['bus_factor']} (team size: {row['team_size']})",
            })

    elif metric == "blackout_hours" and blackouts is not None and len(blackouts) > 0:
        flagged = _apply_op(blackouts, "gap_hours", op, threshold)
        for row in flagged.iter_rows(named=True):
            findings.append({
                **rule,
                "entity": row["from_email"],
                "value": round(row["gap_hours"], 1),
                "detail": f"{row['gap_hours']:.0f}h gap (avg {row['avg_weekly_volume']:.0f} msgs/week)",
            })

    elif metric == "external_spike_z":
        # Handled by compliance module, just check if spikes exist
        pass

    elif metric == "gini" and len(edge_fact) > 0:
        from src.analytics.volume import gini_coefficient
        import numpy as np
        sender_counts = (
            edge_fact.group_by("from_email")
            .agg(pl.len().alias("count"))
        )["count"].to_numpy()
        gini = gini_coefficient(sender_counts)
        if _compare(gini, op, threshold):
            findings.append({
                **rule,
                "entity": "Organization",
                "value": round(gini, 3),
                "detail": f"Gini coefficient: {gini:.3f}",
            })

    elif metric == "health_composite" and health_score is not None:
        composite = health_score.get("composite", 0)
        if _compare(composite, op, threshold):
            findings.append({
                **rule,
                "entity": "Organization",
                "value": round(composite, 1),
                "detail": f"Health score: {composite:.0f}/100",
            })

    return findings


def _apply_op(df: pl.DataFrame, col: str, op: str, threshold: float) -> pl.DataFrame:
    """Filter a DataFrame by operator comparison."""
    if op == ">":
        return df.filter(pl.col(col) > threshold)
    elif op == ">=":
        return df.filter(pl.col(col) >= threshold)
    elif op == "<":
        return df.filter(pl.col(col) < threshold)
    elif op == "<=":
        return df.filter(pl.col(col) <= threshold)
    elif op == "==":
        return df.filter(pl.col(col) == threshold)
    return df


def _compare(value: float, op: str, threshold: float) -> bool:
    """Compare a single value with operator."""
    if op == ">":
        return value > threshold
    elif op == ">=":
        return value >= threshold
    elif op == "<":
        return value < threshold
    elif op == "<=":
        return value <= threshold
    elif op == "==":
        return value == threshold
    return False
