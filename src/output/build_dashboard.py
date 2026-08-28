"""
Build a single, self-contained interactive HTML dashboard from the district
risk scores (src/output/risk_score.py's output). This is the human-facing
front end for the project: a groundwater board official (or a recruiter)
should be able to open dashboard/index.html directly -- no server, no
Python, no code -- and explore the same ranking as docs/RISK_ASSESSMENT.md,
interactively.

Uses Plotly (via CDN) for the map and chart, plus a small amount of
hand-written vanilla JS for the sortable/filterable table -- no framework,
no build step, so the output is one plain HTML file.

Output: dashboard/index.html
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
RISK_CSV = ROOT / "data" / "processed" / "district_risk_scores.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
OUT_HTML = ROOT / "dashboard" / "index.html"


def build_map_figure(df: pd.DataFrame) -> go.Figure:
    """Plain Cartesian scatter (lon/lat as x/y), NOT Plotly's Scattergeo.
    Scattergeo needs to fetch world-map topojson from an external CDN at
    render time even when plotly.js itself is embedded inline -- verified
    by hand (it rendered as a blank panel in an offline test). A plain
    scatter has zero external dependencies and, since India's well
    locations alone trace a recognizable outline (already confirmed in
    reports/figures/risk_map.png), needs no basemap to be legible."""
    hover = (
        df["district"] + ", " + df["state"]
        + "<br>Risk score: " + df["risk_score"].round(1).astype(str)
        + "<br>Current depth: " + df["current_depth"].round(2).astype(str) + " m"
        + "<br>Trend: " + df["measured_trend"].round(3).astype(str) + " m/yr"
        + "<br>Wells: " + df["n_wells"].astype(str)
        + df["known_model_bias_flag"].map({True: "<br><b>* known model-bias district</b>", False: ""})
    )
    fig = go.Figure(
        go.Scatter(
            x=df["lon"], y=df["lat"],
            text=hover, hoverinfo="text",
            mode="markers",
            marker=dict(
                size=df["n_wells"].clip(upper=60) ** 0.5 * 4.5,
                color=df["risk_score"], colorscale="YlOrRd", cmin=0, cmax=100,
                colorbar=dict(title="Risk score"), line=dict(width=0.5, color="black"), opacity=0.85,
            ),
        )
    )
    fig.update_xaxes(title="Longitude", showgrid=False, zeroline=False)
    fig.update_yaxes(title="Latitude", showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title="District groundwater risk -- bubble position = wells' average location (hover for detail, scroll to zoom)",
        height=650, margin=dict(l=0, r=0, t=40, b=0), plot_bgcolor="#fafaf8",
    )
    return fig


def build_bar_figure(df: pd.DataFrame) -> go.Figure:
    top = df.head(40).sort_values("risk_score")
    labels = top["district"] + ", " + top["state"] + top["known_model_bias_flag"].map({True: " *", False: ""})
    fig = go.Figure(
        go.Bar(
            x=top["risk_score"], y=labels, orientation="h",
            marker=dict(color=top["risk_score"], colorscale="YlOrRd", cmin=0, cmax=100),
            hovertext=(
                "Current depth: " + top["current_depth"].round(2).astype(str) + " m<br>"
                + "Trend: " + top["measured_trend"].round(3).astype(str) + " m/yr<br>"
                + "Wells: " + top["n_wells"].astype(str)
            ),
            hoverinfo="text+x",
        )
    )
    fig.update_layout(
        title="Top 40 priority districts (use the map above to explore the rest)",
        xaxis_title="Risk score (0-100)", height=850, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


TABLE_COLUMNS = ["rank", "state", "district", "n_wells", "current_depth", "measured_trend", "risk_score", "known_model_bias_flag"]
TABLE_HEADERS = ["Rank", "State", "District", "Wells", "Current depth (m)", "Trend (m/yr)", "Risk score", "Flag"]
TABLE_TYPES = ["num", "str", "str", "num", "num", "num", "num", "str"]


def build_table_rows(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.iterrows():
        flag = "&#9888;" if r["known_model_bias_flag"] else ""
        rows.append(
            "<tr>"
            f"<td>{int(r['rank'])}</td><td>{r['state']}</td><td>{r['district']}</td>"
            f"<td>{int(r['n_wells'])}</td><td>{r['current_depth']:.2f}</td>"
            f"<td>{r['measured_trend']:+.3f}</td><td>{r['risk_score']:.1f}</td><td>{flag}</td>"
            "</tr>"
        )
    return "\n".join(rows)


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Groundwater Depletion Early-Warning Dashboard</title>
<style>
  :root {{
    --bg: #fafaf8; --panel: #ffffff; --text: #1f2328; --muted: #5b6069;
    --accent: #b23a1f; --border: #e3e1db;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0 0 60px 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  header {{ padding: 32px 24px 20px; max-width: 1100px; margin: 0 auto; }}
  h1 {{ margin: 0 0 6px; font-size: 1.6rem; }}
  .subtitle {{ color: var(--muted); font-size: 0.95rem; max-width: 760px; line-height: 1.5; }}
  .caveat {{
    margin-top: 14px; padding: 12px 16px; background: #fff4ec; border: 1px solid #f0d3bd;
    border-radius: 8px; font-size: 0.88rem; color: #6b3a1c; max-width: 760px; line-height: 1.5;
  }}
  .stats {{ display: flex; gap: 16px; margin-top: 20px; flex-wrap: wrap; }}
  .stat {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px 18px; min-width: 130px;
  }}
  .stat .num {{ font-size: 1.4rem; font-weight: 600; }}
  .stat .label {{ font-size: 0.78rem; color: var(--muted); }}
  section {{ max-width: 1100px; margin: 28px auto 0; padding: 0 24px; }}
  .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 12px; }}
  h2 {{ font-size: 1.05rem; margin: 0 0 12px; }}
  .controls {{ display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; align-items: center; }}
  input[type=text], select {{
    padding: 7px 10px; border: 1px solid var(--border); border-radius: 6px; font-size: 0.88rem;
  }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }}
  th {{ cursor: pointer; user-select: none; background: #f4f2ec; position: sticky; top: 0; }}
  th:hover {{ background: #ece9e0; }}
  tbody tr:hover {{ background: #fbf6ef; }}
  .table-wrap {{ max-height: 520px; overflow-y: auto; border-radius: 8px; }}
  .flag-note {{ font-size: 0.78rem; color: var(--muted); margin-top: 8px; }}
  footer {{ max-width: 1100px; margin: 40px auto 0; padding: 0 24px; color: var(--muted); font-size: 0.82rem; line-height: 1.6; }}
  footer a {{ color: var(--accent); }}
</style>
</head>
<body>

<header>
  <h1>Groundwater Depletion Early-Warning Dashboard</h1>
  <div class="subtitle">
    District-level groundwater risk ranking built from real Central Ground Water Board (CGWB)
    monitoring-well data. Risk score combines measured severity, measured historical trend, and a
    machine-learning forecast &mdash; see the methodology note below.
  </div>
  <div class="caveat">
    <strong>This is a methodology demonstration, not a live assessment.</strong> The underlying CGWB
    data runs through Jan 2017, so this ranks districts as of the mid-2010s, not present-day
    conditions. Full methodology, data sources, and every modelling decision are documented in the
    project's <a href="https://github.com/Tamanna1012/groundwater-early-warning-system" target="_blank">GitHub repository</a>.
  </div>
  <div class="stats">
    <div class="stat"><div class="num">{n_districts}</div><div class="label">districts ranked</div></div>
    <div class="stat"><div class="num">{n_wells}</div><div class="label">monitoring wells</div></div>
    <div class="stat"><div class="num">{n_flagged}</div><div class="label">flagged model-bias districts</div></div>
    <div class="stat"><div class="num">{top_district}</div><div class="label">#1 priority district</div></div>
  </div>
</header>

<section>
  <div class="panel">{map_div}</div>
</section>

<section>
  <div class="panel">{bar_div}</div>
</section>

<section>
  <div class="panel">
    <h2>All {n_districts} ranked districts</h2>
    <div class="controls">
      <input type="text" id="searchBox" placeholder="Search district or state..." oninput="filterTable()">
      <select id="stateFilter" onchange="filterTable()"><option value="">All states</option>{state_options}</select>
      <span style="font-size:0.8rem;color:var(--muted)">Click a column header to sort</span>
    </div>
    <div class="table-wrap">
      <table id="districtTable">
        <thead><tr>{table_headers}</tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
    <div class="flag-note">&#9888; = known model-bias district (docs/EVALUATION.md / docs/EXPLAINABILITY.md): treat its risk score as a floor, not a ceiling.</div>
  </div>
</section>

<footer>
  Groundwater Depletion Early-Warning System &mdash; a real-problem, real-data data science project.
  See <a href="https://github.com/Tamanna1012/groundwater-early-warning-system">the repository</a> for
  the full pipeline: problem statement, data sources, EDA, feature engineering, baseline and ML models,
  backtested evaluation, explainability, and this risk assessment. Not an official CGWB product.
</footer>

<script>
const columnTypes = {column_types};
let sortState = {{ col: 6, dir: -1 }};

function sortTable(colIndex) {{
  const table = document.getElementById("districtTable");
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);
  const dir = (sortState.col === colIndex) ? -sortState.dir : -1;
  sortState = {{ col: colIndex, dir: dir }};
  const type = columnTypes[colIndex];
  rows.sort((a, b) => {{
    let av = a.cells[colIndex].innerText, bv = b.cells[colIndex].innerText;
    if (type === "num") {{ av = parseFloat(av) || 0; bv = parseFloat(bv) || 0; return dir * (av - bv); }}
    return dir * av.localeCompare(bv);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

function filterTable() {{
  const q = document.getElementById("searchBox").value.toLowerCase();
  const state = document.getElementById("stateFilter").value;
  const rows = document.getElementById("districtTable").tBodies[0].rows;
  for (const row of rows) {{
    const text = row.innerText.toLowerCase();
    const rowState = row.cells[1].innerText;
    const matchesText = text.includes(q);
    const matchesState = !state || rowState === state;
    row.style.display = (matchesText && matchesState) ? "" : "none";
  }}
}}

document.querySelectorAll("#districtTable th").forEach((th, i) => {{
  th.addEventListener("click", () => sortTable(i));
}});
</script>

</body>
</html>
"""


def main() -> None:
    df = pd.read_csv(RISK_CSV)

    features_df = pd.read_csv(FEATURES_PATH, usecols=["well_id"])
    n_wells_total = features_df["well_id"].nunique()

    map_fig = build_map_figure(df)
    bar_fig = build_bar_figure(df)
    # Plotly.js is embedded INLINE (not loaded from a CDN) so the dashboard
    # is genuinely self-contained: it opens correctly with no internet
    # connection, and isn't at the mercy of a CDN being reachable or
    # changing later. Costs ~4MB of file size; worth it for a file meant to
    # be opened by double-clicking, offline, indefinitely into the future.
    map_div = map_fig.to_html(full_html=False, include_plotlyjs="inline", div_id="mapDiv")
    bar_div = bar_fig.to_html(full_html=False, include_plotlyjs=False, div_id="barDiv")

    states = sorted(df["state"].unique())
    state_options = "".join(f'<option value="{s}">{s}</option>' for s in states)

    table_headers = "".join(f"<th>{h}</th>" for h in TABLE_HEADERS)
    table_rows = build_table_rows(df)

    top = df.iloc[0]

    html = HTML_TEMPLATE.format(
        n_districts=len(df),
        n_wells=f"{n_wells_total:,}",
        n_flagged=int(df["known_model_bias_flag"].sum()),
        top_district=f"{top['district']}, {top['state']}",
        map_div=map_div,
        bar_div=bar_div,
        state_options=state_options,
        table_headers=table_headers,
        table_rows=table_rows,
        column_types=json.dumps(TABLE_TYPES),
    )

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html)
    print(f"Wrote {OUT_HTML} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
