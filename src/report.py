"""HTML report generation for executive summary export.

Generates a self-contained HTML report with embedded charts (as base64 images)
that can be saved as PDF via browser print or downloaded directly.
"""

import base64
import datetime as dt
import io

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from src.anonymize import anon, anon_df, is_anonymized


def _fig_to_base64(fig: go.Figure, width: int = 800, height: int = 400) -> str:
    """Convert a Plotly figure to a base64-encoded PNG."""
    img_bytes = fig.to_image(format="png", width=width, height=height)
    return base64.b64encode(img_bytes).decode("utf-8")


def _format_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def generate_executive_report(
    message_fact: pl.DataFrame,
    edge_fact: pl.DataFrame,
    person_dim: pl.DataFrame,
    weekly_agg: pl.DataFrame,
    graph_metrics: pl.DataFrame,
    narrative: str,
    start_date: dt.date,
    end_date: dt.date,
    org_name: str = "Organization",
) -> str:
    """Generate a self-contained HTML executive report.

    Returns HTML string. Can be downloaded directly or printed to PDF.
    """
    total_msgs = len(message_fact)
    total_people = len(person_dim)
    internal_count = len(person_dim.filter(pl.col("is_internal"))) if "is_internal" in person_dim.columns else 0
    external_count = total_people - internal_count
    total_bytes = int(message_fact["size_bytes"].sum()) if total_msgs > 0 else 0
    n_communities = graph_metrics["community_id"].n_unique() if len(graph_metrics) > 0 else 0

    # After-hours stats
    ah_rate = float(message_fact["is_after_hours"].mean()) if total_msgs > 0 else 0
    we_rate = float(message_fact["is_weekend"].mean()) if total_msgs > 0 else 0

    # Top senders
    top_senders_df = (
        edge_fact.group_by("from_email")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(10)
    )
    top_senders_df = anon_df(top_senders_df)

    # Generate charts
    charts = {}

    # Weekly volume chart
    if len(weekly_agg) > 0:
        fig_vol = px.bar(
            weekly_agg.to_pandas(), x="week_start", y="msg_count",
            title="Weekly Message Volume",
        )
        fig_vol.update_layout(
            margin=dict(l=40, r=20, t=40, b=40),
            font=dict(size=11),
        )
        try:
            charts["weekly_volume"] = _fig_to_base64(fig_vol)
        except Exception:
            charts["weekly_volume"] = None

    # Top senders chart
    if len(top_senders_df) > 0:
        fig_senders = px.bar(
            top_senders_df.to_pandas(), x="from_email", y="count",
            title="Top 10 Senders",
        )
        fig_senders.update_layout(
            margin=dict(l=40, r=20, t=40, b=80),
            xaxis_tickangle=-45,
            font=dict(size=11),
        )
        try:
            charts["top_senders"] = _fig_to_base64(fig_senders)
        except Exception:
            charts["top_senders"] = None

    # Community distribution
    if len(graph_metrics) > 0:
        comm_sizes = (
            graph_metrics.group_by("community_id")
            .agg(pl.len().alias("members"))
            .sort("members", descending=True)
            .head(15)
        )
        fig_comm = px.bar(
            comm_sizes.to_pandas(), x="community_id", y="members",
            title="Top 15 Communities by Size",
        )
        fig_comm.update_layout(
            margin=dict(l=40, r=20, t=40, b=40),
            font=dict(size=11),
        )
        try:
            charts["communities"] = _fig_to_base64(fig_comm)
        except Exception:
            charts["communities"] = None

    # Top senders table HTML
    top_senders_rows = ""
    for row in top_senders_df.iter_rows(named=True):
        top_senders_rows += f"<tr><td>{row['from_email']}</td><td>{row['count']:,}</td></tr>\n"

    # Build chart HTML
    def chart_html(key: str, alt: str = "") -> str:
        if charts.get(key):
            return f'<img src="data:image/png;base64,{charts[key]}" style="width:100%;max-width:800px;" alt="{alt}">'
        return f"<p><em>[Chart not available — install kaleido: pip install kaleido]</em></p>"

    anon_notice = ""
    if is_anonymized():
        anon_notice = '<p style="color:#c00;font-weight:bold;">Note: Email addresses in this report have been anonymized.</p>'

    report_date = dt.datetime.now().strftime("%B %d, %Y")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Email Analytics Report — {org_name}</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; color: #333; line-height: 1.6; }}
  h1 {{ color: #1a3a5c; border-bottom: 3px solid #1a3a5c; padding-bottom: 10px; }}
  h2 {{ color: #2d5f8a; margin-top: 30px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }}
  .kpi {{ background: #f0f4f8; border-radius: 8px; padding: 15px; text-align: center; }}
  .kpi .value {{ font-size: 28px; font-weight: bold; color: #1a3a5c; }}
  .kpi .label {{ font-size: 13px; color: #666; margin-top: 5px; }}
  .chart {{ margin: 20px 0; text-align: center; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f0f4f8; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .narrative {{ background: #f8f9fa; border-left: 4px solid #2d5f8a; padding: 15px 20px; margin: 20px 0; }}
  .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #999; font-size: 12px; }}
  @media print {{
    body {{ margin: 20px; }}
    .no-print {{ display: none; }}
  }}
</style>
</head>
<body>

<h1>Email Communication Analysis Report</h1>
<p><strong>{org_name}</strong> &mdash; {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}</p>
<p>Report generated: {report_date}</p>
{anon_notice}

<h2>Key Metrics</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="value">{total_msgs:,}</div><div class="label">Total Messages</div></div>
  <div class="kpi"><div class="value">{total_people:,}</div><div class="label">Unique People</div></div>
  <div class="kpi"><div class="value">{internal_count:,}</div><div class="label">Internal</div></div>
  <div class="kpi"><div class="value">{external_count:,}</div><div class="label">External</div></div>
  <div class="kpi"><div class="value">{_format_bytes(total_bytes)}</div><div class="label">Total Data Volume</div></div>
  <div class="kpi"><div class="value">{n_communities}</div><div class="label">Communities Detected</div></div>
  <div class="kpi"><div class="value">{ah_rate:.1%}</div><div class="label">After-Hours Rate</div></div>
  <div class="kpi"><div class="value">{we_rate:.1%}</div><div class="label">Weekend Rate</div></div>
</div>

<h2>Executive Narrative</h2>
<div class="narrative">
{_markdown_to_html(narrative)}
</div>

<h2>Weekly Message Volume</h2>
<div class="chart">{chart_html("weekly_volume", "Weekly message volume chart")}</div>

<h2>Top 10 Senders</h2>
<div class="chart">{chart_html("top_senders", "Top 10 senders chart")}</div>
<table>
  <thead><tr><th>Email</th><th>Messages Sent</th></tr></thead>
  <tbody>{top_senders_rows}</tbody>
</table>

<h2>Community Structure</h2>
<div class="chart">{chart_html("communities", "Community sizes chart")}</div>
<p>The network contains <strong>{n_communities} communities</strong> detected via Louvain modularity optimization.
Communities represent clusters of people who communicate more frequently with each other than with outsiders.</p>

<div class="footer">
  <p>Generated by Email Metadata Analytics Platform. Analysis based on email header metadata only (Date, Size, From, To). No email body content was accessed.</p>
</div>

</body>
</html>"""

    return html


def _markdown_to_html(md: str) -> str:
    """Minimal markdown to HTML conversion for the narrative."""
    import re
    lines = md.split("\n")
    html_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            html_lines.append("<br>")
            continue
        # Bold
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        html_lines.append(f"<p>{line}</p>")
    return "\n".join(html_lines)
