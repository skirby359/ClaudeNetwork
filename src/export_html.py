"""HTML report export: one-click branded report with key findings."""

import io
import base64
from datetime import date

import polars as pl
import plotly.express as px
import plotly.graph_objects as go


def _fig_to_img_tag(fig, width: int = 700, height: int = 350) -> str:
    """Convert a Plotly figure to an inline <img> tag using base64-encoded PNG."""
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height)
        b64 = base64.b64encode(img_bytes).decode()
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;" />'
    except Exception:
        # Kaleido not available or figure rendering failed — skip chart
        return '<p style="color:#888;"><em>Chart rendering unavailable (install kaleido)</em></p>'


def _metric_card(label: str, value: str, color: str = "#4e79a7") -> str:
    return f"""
    <div style="display:inline-block;text-align:center;padding:15px 25px;margin:5px;
                border:1px solid #e0e0e0;border-radius:8px;min-width:140px;">
        <div style="font-size:28px;font-weight:bold;color:{color};">{value}</div>
        <div style="font-size:12px;color:#666;margin-top:4px;">{label}</div>
    </div>"""


def _section(title: str, content: str) -> str:
    return f"""
    <div style="margin-top:30px;">
        <h2 style="color:#333;border-bottom:2px solid #4e79a7;padding-bottom:5px;">{title}</h2>
        {content}
    </div>"""


def _table_html(headers: list[str], rows: list[list[str]], max_rows: int = 20) -> str:
    header_cells = "".join(
        f'<th style="background:#4e79a7;color:white;padding:8px 12px;text-align:left;">{h}</th>'
        for h in headers
    )
    body_rows = []
    for i, row in enumerate(rows[:max_rows]):
        bg = "#f8f8f8" if i % 2 == 0 else "#ffffff"
        cells = "".join(
            f'<td style="padding:6px 12px;border-bottom:1px solid #eee;">{v}</td>'
            for v in row
        )
        body_rows.append(f'<tr style="background:{bg};">{cells}</tr>')

    note = ""
    if len(rows) > max_rows:
        note = f'<p style="color:#888;font-size:11px;">Showing {max_rows} of {len(rows)} rows</p>'

    return f"""
    <table style="border-collapse:collapse;width:100%;font-size:13px;margin:10px 0;">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{"".join(body_rows)}</tbody>
    </table>{note}"""


def generate_html_report(
    message_fact: pl.DataFrame,
    edge_fact: pl.DataFrame,
    person_dim: pl.DataFrame,
    graph_metrics: pl.DataFrame,
    health_score: dict | None = None,
    narrative: str = "",
    bridges: pl.DataFrame | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    org_name: str = "Organization",
) -> str:
    """Generate a complete HTML report as a string."""
    date_range = ""
    if start_date and end_date:
        date_range = f"{start_date.strftime('%b %d, %Y')} &mdash; {end_date.strftime('%b %d, %Y')}"

    total_msgs = len(message_fact)
    unique_people = len(person_dim)
    unique_senders = message_fact["from_email"].n_unique() if total_msgs > 0 else 0
    n_communities = graph_metrics["community_id"].n_unique() if len(graph_metrics) > 0 else 0

    # --- Build sections ---
    sections = []

    # KPI row
    kpi_html = '<div style="text-align:center;margin:20px 0;">'
    kpi_html += _metric_card("Total Messages", f"{total_msgs:,}")
    kpi_html += _metric_card("Unique People", f"{unique_people:,}")
    kpi_html += _metric_card("Active Senders", f"{unique_senders:,}")
    kpi_html += _metric_card("Communities", f"{n_communities:,}")
    kpi_html += '</div>'
    sections.append(kpi_html)

    # Health score
    if health_score:
        composite = health_score["composite"]
        color = "#59a14f" if composite >= 70 else ("#f28e2b" if composite >= 50 else "#e15759")
        verdict = "Healthy" if composite >= 70 else ("Moderate" if composite >= 50 else "Needs Attention")

        health_html = f"""
        <div style="text-align:center;margin:20px 0;">
            <div style="font-size:64px;font-weight:bold;color:{color};">{composite:.0f}</div>
            <div style="font-size:20px;color:{color};">{verdict}</div>
            <div style="font-size:12px;color:#888;">out of 100</div>
        </div>
        <table style="width:100%;font-size:13px;margin:10px 0;">"""
        for key, score in health_score["sub_scores"].items():
            val = score["value"]
            bar_color = "#4e79a7" if val >= 60 else "#f28e2b" if val >= 40 else "#e15759"
            pct = min(val, 100)
            health_html += f"""
            <tr>
                <td style="width:30%;padding:4px;">{score['label']}</td>
                <td style="width:50%;padding:4px;">
                    <div style="background:#eee;border-radius:4px;height:18px;width:100%;">
                        <div style="background:{bar_color};border-radius:4px;height:18px;width:{pct}%;"></div>
                    </div>
                </td>
                <td style="width:20%;padding:4px;text-align:right;">{val:.0f}/100</td>
            </tr>"""
        health_html += "</table>"
        sections.append(_section("Organizational Health Score", health_html))

    # Narrative
    if narrative:
        narrative_html = narrative.replace("\n\n", "</p><p>").replace("**", "")
        sections.append(_section("Executive Narrative", f"<p>{narrative_html}</p>"))

    # Work patterns
    if total_msgs > 0:
        ah_rate = float(message_fact["is_after_hours"].mean()) * 100
        we_rate = float(message_fact["is_weekend"].mean()) * 100
        avg_size_kb = float(message_fact["size_bytes"].mean()) / 1024

        work_html = '<div style="text-align:center;">'
        work_html += _metric_card("After-Hours Rate", f"{ah_rate:.1f}%", "#f28e2b")
        work_html += _metric_card("Weekend Rate", f"{we_rate:.1f}%", "#f28e2b")
        work_html += _metric_card("Avg Message Size", f"{avg_size_kb:.1f} KB")
        work_html += '</div>'

        # Hourly distribution chart
        hourly = (
            message_fact.group_by("hour")
            .agg(pl.len().alias("count"))
            .sort("hour")
        )
        if len(hourly) > 0:
            fig_hourly = px.bar(
                hourly.to_pandas(), x="hour", y="count",
                title="Messages by Hour of Day",
                labels={"hour": "Hour", "count": "Messages"},
                color_discrete_sequence=["#4e79a7"],
            )
            fig_hourly.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40))
            work_html += _fig_to_img_tag(fig_hourly)

        sections.append(_section("Work Patterns", work_html))

    # Top senders
    if total_msgs > 0:
        top_senders = (
            message_fact.group_by("from_email")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(15)
        )
        rows = [[r["from_email"], f"{r['count']:,}"] for r in top_senders.iter_rows(named=True)]
        sections.append(_section("Top Senders", _table_html(["Email", "Messages Sent"], rows)))

    # Network — top connectors
    if len(graph_metrics) > 0:
        top_conn = graph_metrics.sort("betweenness_centrality", descending=True).head(10)
        rows = [
            [r["email"], f"{r['betweenness_centrality']:.4f}",
             f"{r['pagerank']:.4f}", str(r.get("community_label", r.get("community_id", "")))]
            for r in top_conn.iter_rows(named=True)
        ]
        sections.append(_section(
            "Key Connectors",
            _table_html(["Person", "Connector Score", "Importance", "Community"], rows),
        ))

    # Communities
    if len(graph_metrics) > 0 and "community_label" in graph_metrics.columns:
        comm_summary = (
            graph_metrics.group_by("community_id")
            .agg([pl.len().alias("members"), pl.col("community_label").first().alias("label")])
            .sort("members", descending=True)
            .head(15)
        )
        rows = [
            [str(r.get("label", f"Group {r['community_id']}")), f"{r['members']:,}"]
            for r in comm_summary.iter_rows(named=True)
        ]

        # Community size distribution chart
        fig_comm = px.bar(
            comm_summary.to_pandas(), x="label", y="members",
            title="Largest Communication Communities",
            labels={"label": "Community", "members": "Members"},
            color_discrete_sequence=["#4e79a7"],
        )
        fig_comm.update_layout(height=300, xaxis_tickangle=-45, margin=dict(l=40, r=20, t=40, b=80))

        comm_html = _fig_to_img_tag(fig_comm)
        comm_html += _table_html(["Community", "Members"], rows)
        sections.append(_section("Communication Communities", comm_html))

    # Bridges
    if bridges is not None and len(bridges) > 0:
        top_b = bridges.head(10)
        rows = [[r["email"], str(r["communities_bridged"])] for r in top_b.iter_rows(named=True)]
        sections.append(_section(
            "Bridge People",
            "<p>People whose contacts span multiple communities. Losing a bridge can isolate groups.</p>"
            + _table_html(["Person", "Communities Bridged"], rows),
        ))

    # Department breakdown
    if "department" in person_dim.columns:
        dept = (
            person_dim.group_by("department")
            .agg(pl.len().alias("people"))
            .sort("people", descending=True)
            .head(15)
        )
        if len(dept) > 1:
            rows = [[r["department"], f"{r['people']:,}"] for r in dept.iter_rows(named=True)]
            sections.append(_section("Department Breakdown", _table_html(["Department", "People"], rows)))

    # Assemble full HTML
    body = "\n".join(sections)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Email Communication Analysis — {org_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 30px;
            color: #333;
            line-height: 1.5;
        }}
        h1 {{ color: #4e79a7; margin-bottom: 5px; }}
        h2 {{ font-size: 18px; margin-top: 25px; }}
        p {{ margin: 8px 0; }}
        @media print {{
            body {{ padding: 10px; }}
            div {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <h1>Email Communication Analysis</h1>
    <p style="color:#666;font-size:14px;">{org_name} &mdash; {date_range}</p>
    <hr style="border:1px solid #4e79a7;">
    {body}
    <hr style="border:1px solid #ddd;margin-top:40px;">
    <p style="color:#888;font-size:11px;">
        Generated by Email Metadata Analytics Platform.
        Analysis based on email metadata only (Date, From, To, Size).
        Message bodies are never accessed or stored.
    </p>
</body>
</html>"""

    return html
