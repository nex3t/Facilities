"""
charts.py — Facilities Portfolio Optimization Dashboard
City of Chicago — Department of Public Services

Plotly figure factory functions, one per chart in the dashboard.
Every function accepts plain DataFrames / scalars and returns a go.Figure.
No Streamlit calls — figures are rendered by the caller via st.plotly_chart().
"""

import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

from config import (
    COLORS, PILLAR_COLORS, PILLAR_LABELS, PILLAR_DEFAULTS,
    PLOTLY_BASE, _AXIS_BASE, _LEGEND_BASE, _MARGIN_BASE,
    BS_DOMAIN_RATIO, FO_DOMAIN_RATIO,
    CHART_BASE_HEIGHT, CHART_HEIGHT_PER_ROW,
    SANKEY_LINK_OPACITY_HIGH, SANKEY_LINK_OPACITY_LOW,
    BUDGET_LINE_COLORS, BUDGET_LINE_FILL_COLORS, BUDGET_LINE_LABELS,
)
from calculations import fmt, sigmoid


def _apply_base(fig: go.Figure, *, legend: dict | None = None,
                margin: dict | None = None, **layout_kw) -> go.Figure:
    """Apply PLOTLY_BASE + axis styles + optional legend/margin to a figure."""
    kw = dict(**PLOTLY_BASE, **layout_kw)
    kw["legend"] = {**_LEGEND_BASE, **(legend or {})}
    kw["margin"] = {**_MARGIN_BASE, **(margin or {})}
    fig.update_layout(**kw)
    fig.update_xaxes(**_AXIS_BASE)
    fig.update_yaxes(**_AXIS_BASE)
    return fig


# ============================================================================
# TAB 1 — OVERVIEW
# ============================================================================

def make_cost_evolution_chart(
    yearly_base: pd.DataFrame,
    hist_costs: pd.DataFrame,
    yearly_by_line: pd.DataFrame | None = None,
) -> go.Figure:
    """Stacked area chart: DFF Cost Evolution (historical 2016-2024 + projection 2025-2033).

    v7 mode (yearly_by_line provided with 3 budget lines):
        Shows Facilities / Equipment / Fleet stacked areas using ACFR historical actuals
        and simulation projections.  Total DFF ≈ $333M in 2024.

    Legacy mode (yearly_by_line is None or has only Facilities):
        Shows Building Systems / Facilities Operations split (original v6 behaviour).

    Args:
        yearly_base:     Yearly aggregated totals (columns: year, Total, BS, FO).
        hist_costs:      Historical costs with Year, DFF_Total_Actual.
                         v7 also has Facilities_Actual, Equipment_Actual, Fleet_Actual.
        yearly_by_line:  Optional. Output of aggregate_yearly_by_line() —
                         columns: year, budget_line, baseline, l1, l2, final.

    Returns:
        Plotly Figure ready for st.plotly_chart().
    """
    if yearly_base.empty:
        logger.warning("make_cost_evolution_chart() called with empty yearly_base — returning empty figure.")
        return go.Figure()

    # ── Decide rendering mode ─────────────────────────────────────────────────
    use_budget_lines = (
        yearly_by_line is not None
        and not yearly_by_line.empty
        and "budget_line" in yearly_by_line.columns
        and yearly_by_line["budget_line"].nunique() > 1
    )

    if use_budget_lines:
        return _make_evolution_budget_lines(yearly_base, hist_costs, yearly_by_line)

    # ── Legacy BS/FO mode ─────────────────────────────────────────────────────
    hist_plot = pd.DataFrame()
    if len(hist_costs) > 0 and "DFF_Total_Actual" in hist_costs.columns:
        sim_2024  = yearly_base[yearly_base["year"] == yearly_base["year"].min()]["Total"].values
        hist_2024 = hist_costs[hist_costs["Year"] == hist_costs["Year"].max()]["DFF_Total_Actual"].values
        if len(sim_2024) > 0 and len(hist_2024) > 0 and hist_2024[0] > 0:
            scale     = sim_2024[0] / hist_2024[0]
            hist_plot = hist_costs.copy()
            hist_plot["Total_Scaled"] = hist_costs["DFF_Total_Actual"] * scale
            hist_plot = hist_plot[hist_plot["Year"] >= 2021].sort_values("Year")

    if len(hist_plot) > 0:
        hist_years = hist_plot["Year"].tolist()
        hist_bs    = (hist_plot["Total_Scaled"] * BS_DOMAIN_RATIO / 1e6).tolist()
        hist_fo    = (hist_plot["Total_Scaled"] * FO_DOMAIN_RATIO / 1e6).tolist()
        hist_total = (hist_plot["Total_Scaled"] / 1e6).tolist()
        proj_years = yearly_base["year"].tolist()
        proj_bs    = (yearly_base["BS"] / 1e6).tolist()
        proj_fo    = (yearly_base["FO"] / 1e6).tolist()
        proj_total = (yearly_base["Total"] / 1e6).tolist()
        anchor  = proj_years[0]
        h_mask  = [y < anchor for y in hist_years]
        all_years = [y for y, m in zip(hist_years, h_mask) if m] + proj_years
        all_bs    = [v for v, m in zip(hist_bs,    h_mask) if m] + proj_bs
        all_fo    = [v for v, m in zip(hist_fo,    h_mask) if m] + proj_fo
        all_total = [v for v, m in zip(hist_total, h_mask) if m] + proj_total
        split_i   = sum(1 for m in h_mask if m)
    else:
        all_years = yearly_base["year"].tolist()
        all_bs    = (yearly_base["BS"] / 1e6).tolist()
        all_fo    = (yearly_base["FO"] / 1e6).tolist()
        all_total = (yearly_base["Total"] / 1e6).tolist()
        split_i   = 0

    BS_AREA = "rgba(167,139,250,0.40)"
    FO_AREA = "rgba(45,212,191,0.30)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=all_years, y=all_bs, name="Building Systems",
        mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
        stackgroup="cost", fillcolor=BS_AREA,
        hovertemplate="<b>Building Systems</b><br>%{x}: $%{y:.0f}M<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=all_years, y=all_fo, name="Facilities Operations",
        mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
        stackgroup="cost", fillcolor=FO_AREA,
        hovertemplate="<b>Facilities Operations</b><br>%{x}: $%{y:.0f}M<extra></extra>",
    ))

    actual_x = all_years[:split_i + 1]
    actual_y = all_total[:split_i + 1]
    fig.add_trace(go.Scatter(
        x=actual_x, y=actual_y, name="Actuals",
        mode="lines+markers+text", line=dict(color="#ffffff", width=3),
        marker=dict(size=7, color="#ffffff", line=dict(color=COLORS["bg"], width=1.5)),
        text=[f"${v:.1f}B" for v in actual_y], textposition="top center",
        textfont=dict(size=9, color="#64748b", family="'Inter', sans-serif"),
        hovertemplate="<b>Actual</b><br>%{x}: $%{y:.0f}M<extra></extra>",
    ))
    proj_x = all_years[split_i:]
    proj_y = all_total[split_i:]
    fig.add_trace(go.Scatter(
        x=proj_x, y=proj_y, name="Projection",
        mode="lines+markers+text", line=dict(color="#1e40af", width=2.5, dash="dash"),
        marker=dict(size=7, color="#1e40af", symbol="circle-open",
                    line=dict(color="#ffd166", width=2)),
        text=[f"${v:.1f}B" for v in proj_y], textposition="top center",
        textfont=dict(size=9, color="#64748b", family="'Inter', sans-serif"),
        hovertemplate="<b>Projection</b><br>%{x}: $%{y:.0f}M<extra></extra>",
    ))
    fig.add_vline(x=2024, line_dash="dot", line_color="rgba(71,85,105,0.25)", line_width=1)
    fig.add_annotation(
        x=2024, y=1.0, yref="paper", text="Actual ◀ | ▶ Projection",
        showarrow=False, xanchor="center", yanchor="bottom",
        font=dict(size=9, color="#64748b"), bgcolor="rgba(255,255,255,0.90)", borderpad=4,
    )
    _apply_base(fig, legend={"y": 1.10}, margin=dict(l=55, r=20, t=65, b=44),
                height=400, yaxis_title="Cost ($ Millions)", hovermode="x unified")
    fig.update_xaxes(**_AXIS_BASE, dtick=1)
    return fig


def _make_evolution_budget_lines(
    yearly_base: pd.DataFrame,
    hist_costs: pd.DataFrame,
    yearly_by_line: pd.DataFrame,
) -> go.Figure:
    """v7 multi-budget-line cost evolution chart (Facilities + Equipment + Fleet).

    Historical (2016-2024): uses Facilities_Actual, Equipment_Actual, Fleet_Actual
    from hist_costs if available; otherwise splits DFF_Total_Actual proportionally.
    Projection (2024-2033): uses yearly_by_line baselines.
    """
    BL_ORDER = ["Facilities", "Equipment", "Fleet"]

    # ── Build projection arrays per budget line ───────────────────────────────
    proj_yr = sorted(yearly_by_line["year"].unique().tolist())
    proj_vals: dict[str, list[float]] = {}
    for bl in BL_ORDER:
        sub = yearly_by_line[yearly_by_line["budget_line"] == bl].set_index("year")
        proj_vals[bl] = [sub.loc[y, "baseline"] / 1e6 if y in sub.index else 0.0
                         for y in proj_yr]

    # ── Build historical arrays ───────────────────────────────────────────────
    hist_vals: dict[str, list[float]] = {}
    hist_yr: list[int] = []
    split_i = 0

    hc = hist_costs.copy()
    anchor = proj_yr[0]

    has_per_line = (
        "Facilities_Actual" in hc.columns
        and "Equipment_Actual" in hc.columns
        and "Fleet_Actual" in hc.columns
    )

    if len(hc) > 0:
        hc = hc.sort_values("Year")
        hist_yrs_all = hc["Year"].tolist()
        h_mask       = [y < anchor for y in hist_yrs_all]
        hist_yr      = [y for y, m in zip(hist_yrs_all, h_mask) if m and y >= 2021]
        split_i      = len(hist_yr)

        if has_per_line:
            for bl in BL_ORDER:
                col = f"{bl}_Actual"
                hist_vals[bl] = [float(hc.loc[hc["Year"] == y, col].iloc[0]) / 1e6
                                 if y in hc["Year"].values else 0.0
                                 for y in hist_yr]
        elif "DFF_Total_Actual" in hc.columns:
            # Proportional split using 2024 simulation shares
            total_proj_2024 = sum(vals[0] for vals in proj_vals.values()) or 1.0
            for bl in BL_ORDER:
                share = (proj_vals[bl][0] if proj_vals[bl] else 0) / total_proj_2024
                dff   = [float(hc.loc[hc["Year"] == y, "DFF_Total_Actual"].iloc[0]) / 1e6
                         if y in hc["Year"].values else 0.0
                         for y in hist_yr]
                hist_vals[bl] = [v * share for v in dff]
        else:
            for bl in BL_ORDER:
                hist_vals[bl] = [0.0] * len(hist_yr)

    # ── Combine historical + projection per budget line ───────────────────────
    # Include 2024 anchor point from projection in historical segment for smooth join
    all_yr_dict: dict[str, list] = {}
    for bl in BL_ORDER:
        h = hist_vals.get(bl, [])
        p = proj_vals.get(bl, [])
        all_yr_dict[bl] = h + p

    all_years = hist_yr + proj_yr

    # Total for overlay lines
    all_total = [
        sum(all_yr_dict[bl][i] for bl in BL_ORDER)
        for i in range(len(all_years))
    ]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = go.Figure()

    for bl in BL_ORDER:
        label     = BUDGET_LINE_LABELS.get(bl, bl)
        fill_clr  = BUDGET_LINE_FILL_COLORS.get(bl, "rgba(150,150,150,0.3)")
        fig.add_trace(go.Scatter(
            x=all_years, y=all_yr_dict[bl],
            name=label,
            mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
            stackgroup="cost", fillcolor=fill_clr,
            hovertemplate=f"<b>{label}</b><br>%{{x}}: $%{{y:.0f}}M<extra></extra>",
        ))

    # Actuals overlay (historical segment)
    if split_i > 0:
        act_x = all_years[:split_i + 1]
        act_y = all_total[:split_i + 1]
        fig.add_trace(go.Scatter(
            x=act_x, y=act_y, name="DFF Actuals (ACFR)",
            mode="lines+markers+text", line=dict(color="#334155", width=2),
            marker=dict(size=6, color="#334155", line=dict(color=COLORS["bg"], width=1.5)),
            text=[f"${v:.0f}M" for v in act_y], textposition="top center",
            textfont=dict(size=8, color="#64748b", family="'Inter', sans-serif"),
            hovertemplate="<b>DFF Actual</b><br>%{x}: $%{y:.0f}M<extra></extra>",
        ))

    # Projection overlay
    proj_x = all_years[split_i:]
    proj_y = all_total[split_i:]
    fig.add_trace(go.Scatter(
        x=proj_x, y=proj_y, name="DFF Projection",
        mode="lines+markers+text", line=dict(color="#1e40af", width=2, dash="dash"),
        marker=dict(size=6, color="#1e40af", symbol="circle-open",
                    line=dict(color="#ffd166", width=2)),
        text=[f"${v:.0f}M" for v in proj_y], textposition="top center",
        textfont=dict(size=8, color="#64748b", family="'Inter', sans-serif"),
        hovertemplate="<b>DFF Projection</b><br>%{x}: $%{y:.0f}M<extra></extra>",
    ))

    fig.add_vline(x=2024, line_dash="dot", line_color="rgba(71,85,105,0.25)", line_width=1)
    fig.add_annotation(
        x=2024, y=1.0, yref="paper", text="Actual ◀ | ▶ Projection",
        showarrow=False, xanchor="center", yanchor="bottom",
        font=dict(size=9, color="#64748b"), bgcolor="rgba(255,255,255,0.90)", borderpad=4,
    )

    _apply_base(fig,
                legend={"y": 1.10},
                margin=dict(l=55, r=20, t=65, b=44),
                height=400,
                yaxis_title="Cost ($ Millions)",
                hovermode="x unified",
                title=dict(text="Total DFF Cost — Facilities · Equipment · Fleet",
                           font=dict(size=12, color="#94a3b8"), x=0.01, xanchor="left"))
    fig.update_xaxes(**_AXIS_BASE, dtick=1)
    return fig


def make_dept_cost_bar_chart(dept_agg: pd.DataFrame) -> go.Figure:
    """Horizontal stacked bar: annual base cost by department (BS + FO split).

    Args:
        dept_agg: DataFrame with columns Department, BS, FO (dollar values).
                  Should be pre-sorted ascending by Total for chart readability.

    Returns:
        Plotly Figure.
    """
    if dept_agg.empty:
        logger.warning("make_dept_cost_bar_chart() called with empty dept_agg — returning empty figure.")
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=dept_agg["Department"], x=dept_agg["BS"] / 1e6,
        name="Building Systems", orientation="h",
        marker_color="rgba(255,107,0,0.85)",
        text=dept_agg["BS"].apply(fmt), textposition="inside",
        textfont=dict(size=8, color="rgba(255,255,255,0.7)"),
    ))
    fig.add_trace(go.Bar(
        y=dept_agg["Department"], x=dept_agg["FO"] / 1e6,
        name="Facilities Ops", orientation="h",
        marker_color="rgba(0,194,224,0.80)",
        text=dept_agg["FO"].apply(fmt), textposition="inside",
        textfont=dict(size=8, color="rgba(255,255,255,0.7)"),
    ))
    _apply_base(fig,
                legend={"y": -0.08},
                height=max(CHART_BASE_HEIGHT, len(dept_agg) * CHART_HEIGHT_PER_ROW),
                barmode="stack",
                xaxis_title="Annual Cost ($M)",
                yaxis_title="")
    return fig


def make_avg_cost_per_building_chart(dept_bld_n: pd.DataFrame) -> go.Figure:
    """Horizontal bar: average annual cost per building by department.

    Args:
        dept_bld_n: DataFrame with columns Department and AvgAnn (dollar values).
                    Should be pre-sorted ascending by AvgAnn.

    Returns:
        Plotly Figure.
    """
    if dept_bld_n.empty:
        logger.warning("make_avg_cost_per_building_chart() called with empty dept_bld_n — returning empty figure.")
        return go.Figure()
    fig = go.Figure(go.Bar(
        y=dept_bld_n["Department"], x=dept_bld_n["AvgAnn"] / 1e6,
        orientation="h", marker_color=COLORS["green"],
        text=dept_bld_n["AvgAnn"].apply(fmt),
        textposition="outside", textfont=dict(size=9, color="#94a3b8"),
    ))
    _apply_base(fig,
                height=max(CHART_BASE_HEIGHT, len(dept_bld_n) * CHART_HEIGHT_PER_ROW),
                xaxis_title="Annual Cost / Building ($M)",
                yaxis_title="",
                showlegend=False)
    fig.update_yaxes(**_AXIS_BASE, showticklabels=False)
    return fig


def make_nav_chart(nav_yr: pd.DataFrame) -> go.Figure:
    """Dual-axis chart: Portfolio NAV over time (line) + Depreciation (bars).

    Args:
        nav_yr: DataFrame with columns year, NAV (dollars), Dep (dollars).

    Returns:
        Plotly Figure with secondary y-axis.
    """
    if nav_yr.empty:
        logger.warning("make_nav_chart() called with empty nav_yr — returning empty figure.")
        return go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=nav_yr["year"], y=nav_yr["NAV"] / 1e9,
        name="Net Asset Value ($B)", mode="lines+markers",
        line=dict(color=COLORS["cyan"], width=2.5),
        marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(0,194,224,0.08)",
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=nav_yr["year"], y=nav_yr["Dep"] / 1e6,
        name="Depreciation ($M)", marker_color="rgba(248,113,113,0.4)",
        text=nav_yr["Dep"].apply(lambda x: f"${x/1e6:.0f}M"),
        textposition="auto", textfont=dict(size=8),
    ), secondary_y=True)
    _apply_base(fig,
                height=CHART_BASE_HEIGHT,
                hovermode="x unified",
                showlegend=False)
    fig.update_yaxes(title_text="NAV ($ Billions)", secondary_y=False,
                     gridcolor="rgba(255,255,255,0.04)", color="#94a3b8")
    fig.update_yaxes(title_text="Depreciation ($M)", secondary_y=True, color="#94a3b8")
    return fig


# ============================================================================
# TAB 2 — COST SIMULATION
# ============================================================================

def make_cost_trajectory_chart(
    yearly: pd.DataFrame,
    hist_costs: pd.DataFrame,
) -> go.Figure:
    """Line chart: Baseline vs Optimized cost trajectory (2016-2033).

    Shows four series:
    1. Actuals — solid white, historical 2016-2024.
    2. Trend   — solid orange, baseline no-optimization projection.
    3. After L1 — dotted yellow, cost after pillar savings only.
    4. Optimized — solid green, cost after both L1 + L2.

    Args:
        yearly:     Yearly aggregate with columns year, baseline, l1, l2, final.
        hist_costs: Historical costs DataFrame. Pass empty DataFrame to skip.

    Returns:
        Plotly Figure.
    """
    hist_tl = pd.DataFrame()
    if len(hist_costs) > 0 and "DFF_Total_Actual" in hist_costs.columns:
        sim_2024_base = yearly[yearly["year"] == yearly["year"].min()]["baseline"].values
        hist_2024_val = hist_costs[hist_costs["Year"] == hist_costs["Year"].max()]["DFF_Total_Actual"].values
        if len(sim_2024_base) > 0 and len(hist_2024_val) > 0 and hist_2024_val[0] > 0:
            tl_scale = sim_2024_base[0] / hist_2024_val[0]
            hist_tl  = hist_costs[["Year", "DFF_Total_Actual"]].copy()
            hist_tl["Total_Scaled"] = hist_tl["DFF_Total_Actual"] * tl_scale
            hist_tl  = hist_tl[hist_tl["Year"] >= 2021].sort_values("Year")

    fig = go.Figure()

    if len(hist_tl) > 0:
        fig.add_trace(go.Scatter(
            x=hist_tl["Year"], y=hist_tl["Total_Scaled"] / 1e6,
            name="Actuals",
            mode="lines+markers+text",
            line=dict(color="#334155", width=2),
            marker=dict(size=6, color="#334155", line=dict(color=COLORS["bg"], width=1.5)),
            text=hist_tl["Total_Scaled"].apply(lambda v: f"${v/1e9:.1f}B"),
            textposition="top center",
            textfont=dict(size=8, color="#64748b", family="'Inter', sans-serif"),
            hovertemplate="<b>Actual</b><br>%{x}: $%{y:.0f}M<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["baseline"] / 1e6,
        name="Trend",
        mode="lines+markers+text",
        line=dict(color=COLORS["orange"], width=2.5),
        marker=dict(size=6, color=COLORS["orange"], symbol="circle-open",
                    line=dict(color=COLORS["orange"], width=2)),
        text=yearly["baseline"].apply(lambda v: f"${v/1e9:.1f}B"),
        textposition="top center",
        textfont=dict(size=9, color="#64748b", family="'Inter', sans-serif"),
        hovertemplate="<b>Trend (No Optimization)</b><br>%{x}: $%{y:.0f}M<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=yearly["year"], y=(yearly["baseline"] - yearly["l1"]) / 1e6,
        name="AI Savings",
        mode="lines",
        line=dict(color=COLORS["yellow"], width=2, dash="dot"),
        hovertemplate="<b>After L1 Pillars</b><br>%{x}: $%{y:.0f}M<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["final"] / 1e6,
        name="Optimized",
        mode="lines+markers+text",
        line=dict(color=COLORS["green"], width=3),
        marker=dict(size=7, color=COLORS["green"]),
        fill="tonexty", fillcolor="rgba(45,212,191,0.07)",
        text=yearly["final"].apply(lambda v: f"${v/1e9:.1f}B"),
        textposition="bottom center",
        textfont=dict(size=9, color=COLORS["green"], family="'Inter', sans-serif"),
        hovertemplate="<b>Optimized</b><br>%{x}: $%{y:.0f}M<extra></extra>",
    ))

    if len(hist_tl) > 0:
        fig.add_vline(x=2024, line_dash="dot",
                      line_color="rgba(71,85,105,0.30)", line_width=1)
        fig.add_annotation(
            x=2024, y=1.0, yref="paper",
            text="Actual ◀ | ▶ Projection",
            showarrow=False, xanchor="center", yanchor="bottom",
            font=dict(size=9, color="#64748b"),
            bgcolor="rgba(255,255,255,0.90)", borderpad=4,
        )

    _apply_base(fig,
                legend={"y": 1.10},
                margin=dict(l=55, r=20, t=65, b=44),
                height=400,
                yaxis_title="Cost ($ Millions)",
                hovermode="x unified")
    fig.update_xaxes(**_AXIS_BASE, dtick=1)
    return fig


def make_waterfall_chart(
    last: pd.Series,
    pil_last: pd.DataFrame,
) -> go.Figure:
    """Vertical waterfall: Baseline → per-pillar savings → AI → Optimized Cost.

    Args:
        last:     Single-row Series from yearly aggregate for the last year.
                  Must have keys: baseline, l1, l2, final (dollar values).
        pil_last: Pillar savings for the last year with columns pillar, label, total.

    Returns:
        Plotly Figure.
    """
    wf_base    = float(last["baseline"]) / 1e6 if last is not None else 0
    wf_final   = float(last["final"]) / 1e6    if last is not None else 0
    wf_labels  = ["Baseline"]
    wf_vals    = [wf_base]
    wf_colors  = ["rgba(255,107,0,0.85)"]
    wf_measure = ["absolute"]

    for k in PILLAR_DEFAULTS:
        row = pil_last[pil_last["pillar"] == k]
        sav = float(row["total"].values[0]) / 1e6 if len(row) > 0 else 0
        if sav > 0:
            wf_labels.append(PILLAR_LABELS[k])
            wf_vals.append(-sav)
            wf_colors.append(PILLAR_COLORS[k])
            wf_measure.append("relative")

    ai_sav = float(last["l2"]) / 1e6 if last is not None else 0
    if ai_sav > 0:
        wf_labels.append("AI Optimization")
        wf_vals.append(-ai_sav)
        wf_colors.append(COLORS["indigo"])
        wf_measure.append("relative")

    wf_labels.append("Optimized Cost")
    wf_vals.append(wf_final)
    wf_colors.append("rgba(45,212,191,0.85)")
    wf_measure.append("total")

    wf_text = [
        f"${abs(v):.0f}M" if m in ("absolute", "total") else f"-${abs(v):.0f}M"
        for m, v in zip(wf_measure, wf_vals)
    ]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=wf_measure,
        x=wf_labels,
        y=wf_vals,
        text=wf_text,
        textposition="outside",
        textfont=dict(size=9, color="#94a3b8"),
        connector=dict(line=dict(color=COLORS["border"], width=1, dash="dot")),
        decreasing=dict(marker=dict(color=COLORS["green"])),
        increasing=dict(marker=dict(color=COLORS["orange"])),
        totals=dict(marker=dict(color=COLORS["green"])),
    ))
    _apply_base(fig)
    return fig


# ============================================================================
# EFFICIENCY TRAJECTORY
# ============================================================================

def make_efficiency_trajectory_chart(
    yearly: pd.DataFrame,
    hist_costs: pd.DataFrame | None = None,
) -> go.Figure:
    """Cost trajectory: historical actuals + 3 forward-projection lines.

    Historical (2016-2024, ACFR)
        Solid slate line with dots — what was actually spent.
        Scaled so its last point connects seamlessly to the projection baseline.

    Projection lines (first sim year onwards):
    - Trend (dashed gold)   : baseline cost — status quo, no optimization.
    - Efficiency (cyan)     : baseline − L1  — pillar programs only; diverges
                              downward from 2025 as adoption matures.
    - AI-Optimized (teal)   : final (L1 + L2) — full AI amplification.

    Shaded bands show the L1 savings layer (gold) and AI layer (indigo).
    A dotted vline marks the historical/projection boundary.
    """
    if yearly.empty:
        return go.Figure()

    df = yearly.copy().sort_values("year")
    df["l1_cost_m"]  = ((df["baseline"] - df["l1"]).clip(lower=0)) / 1e6
    df["final_m"]    = df["final"]    / 1e6
    df["baseline_m"] = df["baseline"] / 1e6
    proj_years  = df["year"].tolist()
    proj_start  = int(df["year"].min())

    fig = go.Figure()

    # ── Historical actuals ────────────────────────────────────────────────────
    hist_drawn = False
    if hist_costs is not None and len(hist_costs) > 0 and "DFF_Total_Actual" in hist_costs.columns:
        hist = hist_costs.copy().sort_values("Year")
        # Include up to and including the first projection year so the lines join
        hist = hist[(hist["Year"] >= 2021) & (hist["Year"] <= proj_start)]
        if not hist.empty:
            anchor_hist = float(hist[hist["Year"] == hist["Year"].max()]["DFF_Total_Actual"].values[0])
            anchor_proj = float(df[df["year"] == proj_start]["baseline"].values[0]) if proj_start in proj_years else df["baseline_m"].iloc[0] * 1e6
            scale = anchor_proj / anchor_hist if anchor_hist > 0 else 1.0
            hist["scaled_m"] = hist["DFF_Total_Actual"] * scale / 1e6
            h_years = hist["Year"].tolist()
            h_vals  = hist["scaled_m"].tolist()

            fig.add_trace(go.Scatter(
                x=h_years, y=h_vals,
                name="Historical Actuals (ACFR)",
                mode="lines+markers",
                line=dict(color=COLORS["muted"], width=2),
                marker=dict(size=5, color=COLORS["muted"], symbol="circle",
                            line=dict(color=COLORS["bg"], width=1)),
                hovertemplate="<b>Historical</b>: $%{y:.1f}M (%{x})<extra></extra>",
            ))
            hist_drawn = True

    # ── Fill bands — projection period only (ghost traces, not in legend) ─────
    # L1 savings band: between Trend and Efficiency
    fig.add_trace(go.Scatter(
        x=proj_years, y=df["baseline_m"],
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=proj_years, y=df["l1_cost_m"],
        fill="tonexty", fillcolor="rgba(214,179,90,0.10)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    # AI savings band: between Efficiency and AI-Optimized
    fig.add_trace(go.Scatter(
        x=proj_years, y=df["l1_cost_m"],
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=proj_years, y=df["final_m"],
        fill="tonexty", fillcolor="rgba(99,102,241,0.10)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # ── Line 1: Trend ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=proj_years, y=df["baseline_m"],
        name="Trend — Status Quo",
        mode="lines",
        line=dict(color=COLORS["orange"], width=2.5, dash="dash"),
        hovertemplate="<b>Trend</b>: $%{y:.1f}M<extra></extra>",
    ))

    # ── Line 2: L1 Efficiency ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=proj_years, y=df["l1_cost_m"],
        name="Efficiency — L1 Programs",
        mode="lines+markers",
        line=dict(color=COLORS["cyan"], width=2.5),
        marker=dict(size=6, color=COLORS["cyan"],
                    line=dict(color=COLORS["bg"], width=1)),
        hovertemplate="<b>L1 Efficiency</b>: $%{y:.1f}M<extra></extra>",
    ))

    # ── Line 3: AI-Optimized ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=proj_years, y=df["final_m"],
        name="AI-Optimized — L1 + L2",
        mode="lines+markers",
        line=dict(color=COLORS["green"], width=3),
        marker=dict(size=7, color=COLORS["green"],
                    line=dict(color=COLORS["bg"], width=1.5)),
        hovertemplate="<b>AI-Optimized</b>: $%{y:.1f}M<extra></extra>",
    ))

    # ── Vertical dividers ─────────────────────────────────────────────────────
    if hist_drawn:
        fig.add_vline(
            x=proj_start,
            line=dict(color="rgba(255,255,255,0.18)", width=1.5, dash="dot"),
            annotation_text="Actual  |  Projection",
            annotation_position="top",
            annotation_font=dict(size=9, color="#475569"),
        )

    diverge_yr = max(proj_start + 1, 2025)
    if diverge_yr in proj_years:
        fig.add_vline(
            x=diverge_yr,
            line=dict(color="rgba(255,255,255,0.10)", width=1, dash="dot"),
            annotation_text="Divergence",
            annotation_position="top right",
            annotation_font=dict(size=9, color="#475569"),
        )

    # ── End-of-series cost labels ─────────────────────────────────────────────
    last_yr  = df["year"].max()
    last_row = df[df["year"] == last_yr].iloc[0]
    for y_val, color in [
        (last_row["baseline_m"], COLORS["orange"]),
        (last_row["l1_cost_m"],  COLORS["cyan"]),
        (last_row["final_m"],    COLORS["green"]),
    ]:
        fig.add_annotation(
            x=last_yr, y=y_val,
            text=f"${y_val:.0f}M",
            xanchor="left", xshift=10,
            showarrow=False,
            font=dict(size=10, color=color, family="'Inter', sans-serif"),
        )

    _apply_base(
        fig,
        legend={"y": 1.08},
        margin=dict(l=58, r=90, t=52, b=48),
        height=400,
        yaxis_title="Annual Portfolio Cost ($M)",
        hovermode="x unified",
    )
    fig.update_xaxes(dtick=1)
    return fig


# ============================================================================
# TAB 3 — AI & SAVINGS
# ============================================================================

def make_prescription_efficiency_chart(
    yearly: pd.DataFrame,
    pil_yr: pd.DataFrame,
) -> go.Figure:
    """Side-by-side layout: Left (2/3) — efficiency % line; Right (1/3) — two
    summary bars (Baseline vs Optimized) with a red efficiency delta badge.

    Definitions
    -----------
    Optimization Efficiency % = (L1 + L2) / Annual Baseline × 100
    Optimized Cost            = Baseline − L1 − L2  (= final_override)

    Left panel  — Ascending line: efficiency % from ~0% in 2024 rising as programs
                  and AI adoption mature.
    Right panel — Two tall bars (avg annual):
                    • Baseline  (gold)  — what we spend without optimization
                    • Optimized (teal)  — net cost after L1 + L2 savings
                  A red downward-arrow badge between the bars shows average
                  efficiency % (savings / baseline).

    Args:
        yearly: Yearly aggregate with columns year, baseline, l1, l2, final.
        pil_yr: Kept for API compatibility; not used in this layout.

    Returns:
        Plotly Figure (make_subplots, 1 row × 2 cols).
    """
    if yearly.empty:
        return go.Figure()

    df = yearly.copy().sort_values("year")
    df["savings"] = df["l1"] + df["l2"]
    df["eff_pct"] = (
        df["savings"] / df["baseline"].replace(0, float("nan")) * 100
    ).fillna(0).clip(0)

    # Average annual figures across selected year range
    avg_baseline_m = df["baseline"].mean() / 1e6
    avg_final_m    = df["final"].mean()    / 1e6      # optimized (net) cost
    avg_eff_pct    = df["eff_pct"].mean()
    avg_savings_m  = avg_baseline_m - avg_final_m

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=["Optimization Efficiency % (L1 + L2)", "Avg. Annual Cost ($M)"],
        horizontal_spacing=0.10,
    )

    # ── Left panel: efficiency line (ascending) ────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["eff_pct"],
        name="Optimization Efficiency",
        mode="lines+markers+text",
        line=dict(color=COLORS["green"], width=3),
        marker=dict(size=8, color=COLORS["green"],
                    line=dict(color=COLORS["bg"], width=1.5)),
        text=[f"{v:.1f}%" for v in df["eff_pct"]],
        textposition="top center",
        textfont=dict(size=9, color=COLORS["green"], family="'Inter', sans-serif"),
        fill="tozeroy",
        fillcolor="rgba(45,212,191,0.08)",
        hovertemplate="<b>Optimization Efficiency</b><br>%{x}: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)

    # ── Right panel: two summary bars ─────────────────────────────────────────
    bar_x      = ["Baseline", "Optimized"]
    bar_y      = [avg_baseline_m, avg_final_m]
    bar_colors = [COLORS["orange"], COLORS["green"]]
    bar_labels = [f"<b>${avg_baseline_m:.1f}M</b>", f"<b>${avg_final_m:.1f}M</b>"]

    fig.add_trace(go.Bar(
        x=bar_x,
        y=bar_y,
        marker=dict(
            color=bar_colors,
            opacity=0.85,
            line=dict(color=["rgba(214,179,90,0.6)", "rgba(45,212,191,0.6)"], width=1.5),
        ),
        text=bar_labels,
        textposition="outside",
        textfont=dict(size=12, color=COLORS["text"], family="'Inter', sans-serif"),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>$%{y:.1f}M<extra></extra>",
        width=0.5,
    ), row=1, col=2)

    # Dashed reference line at baseline level
    fig.add_hline(
        y=avg_baseline_m,
        line_dash="dot",
        line_color="rgba(214,179,90,0.35)",
        line_width=1.5,
        row=1, col=2,
    )

    # Red downward-arrow badge — sits above the Optimized bar pointing down
    badge_y = avg_baseline_m + (avg_baseline_m - avg_final_m) * 0.15
    fig.add_annotation(
        x="Optimized",
        y=badge_y,
        xref="x2", yref="y2",
        ax=0, ay=-44,
        axref="pixel", ayref="pixel",
        text=(
            f"<b>▼ {avg_eff_pct:.1f}%</b><br>"
            f"<span style='font-size:9px'>${avg_savings_m:.1f}M saved</span>"
        ),
        font=dict(size=12, color=COLORS["red"], family="'Inter', sans-serif"),
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS["red"],
        arrowwidth=2,
        arrowsize=0.9,
        bgcolor="rgba(239,68,68,0.13)",
        bordercolor=COLORS["red"],
        borderwidth=1,
        borderpad=6,
        align="center",
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    kw = dict(**PLOTLY_BASE)
    kw["legend"]    = {**_LEGEND_BASE, "y": 1.08, "x": 0}
    kw["margin"]    = dict(l=55, r=28, t=68, b=52)
    kw["height"]    = 500
    kw["barmode"]   = "group"
    kw["hovermode"] = "x unified"
    fig.update_layout(**kw)

    fig.update_xaxes(**_AXIS_BASE, dtick=1, row=1, col=1)
    fig.update_xaxes(**_AXIS_BASE, row=1, col=2)
    fig.update_yaxes(**_AXIS_BASE, ticksuffix="%", row=1, col=1)
    fig.update_yaxes(
        **_AXIS_BASE, tickprefix="$", ticksuffix="M",
        range=[0, avg_baseline_m * 1.28],   # headroom for badge and outside labels
        row=1, col=2,
    )

    # Style only subplot title annotations (yref="paper")
    for ann in fig.layout.annotations:
        if getattr(ann, "yref", None) == "paper":
            ann.update(font=dict(size=11, color=COLORS["muted"]))

    return fig


def make_pillar_savings_lines_chart(
    pil_yr: pd.DataFrame,
    yearly: pd.DataFrame,
) -> go.Figure:
    """Multi-line chart: annual savings by pillar (L1) + AI line (L2).

    Args:
        pil_yr:  Pillar-year aggregate with columns year, pillar, l1, l2.
        yearly:  Yearly aggregate with columns year, l2.

    Returns:
        Plotly Figure.
    """
    if pil_yr.empty:
        logger.warning("make_pillar_savings_lines_chart() called with empty pil_yr — returning empty figure.")
        return go.Figure()
    fig = go.Figure()

    for key in PILLAR_DEFAULTS:
        pil_data = pil_yr[pil_yr["pillar"] == key].sort_values("year")
        if len(pil_data) == 0 or pil_data["l1"].sum() == 0:
            continue
        y_m = pil_data["l1"] / 1e6
        fig.add_trace(go.Scatter(
            x=pil_data["year"],
            y=y_m,
            name=PILLAR_LABELS[key],
            mode="lines+markers+text",
            line=dict(color=PILLAR_COLORS[key], width=2),
            marker=dict(size=5, color=PILLAR_COLORS[key]),
            text=[f"${v:.1f}M" for v in y_m],
            textposition="top center",
            textfont=dict(size=8, color=PILLAR_COLORS[key], family="'Inter', sans-serif"),
            hovertemplate=f"<b>{PILLAR_LABELS[key]}</b><br>%{{x}}: $%{{y:.1f}}M<extra></extra>",
        ))

    _apply_base(fig,
                legend={"orientation": "h", "y": 1.12, "x": 0,
                        "font": dict(size=9, color="#64748b"), "tracegroupgap": 2},
                margin=dict(l=55, r=65, t=75, b=44),
                height=480,
                yaxis_title="Annual Savings ($ Millions)",
                hovermode="x unified")
    fig.update_xaxes(**_AXIS_BASE, dtick=1)
    return fig


def make_ai_donut_chart(
    pillar_ai: pd.DataFrame,
    total_ai: float,
) -> go.Figure:
    """Donut (pie with hole) chart: AI savings share by pillar category.

    Args:
        pillar_ai: DataFrame with columns pillar and l2_scaled (dollar values).
        total_ai:  Total AI savings (dollar) used for the center annotation.

    Returns:
        Plotly Figure.
    """
    pie_labels  = [PILLAR_LABELS[p] for p in pillar_ai["pillar"]]
    pie_values  = pillar_ai["l2_scaled"] / 1e6
    pie_colors  = [PILLAR_COLORS.get(p, COLORS["purple"]) for p in pillar_ai["pillar"]]
    pie_amounts = pillar_ai["l2_scaled"].apply(fmt)

    fig = go.Figure(go.Pie(
        labels=pie_labels,
        values=pie_values,
        hole=0.56,
        marker=dict(colors=pie_colors, line=dict(color=COLORS["bg"], width=3)),
        textinfo="label+percent",
        textfont=dict(size=10, color="#eef2f7", family="'Inter', sans-serif"),
        textposition="outside",
        direction="clockwise",
        sort=False,
        customdata=pie_amounts,
        hovertemplate=(
            "<b>%{label}</b><br>"
            "AI Savings: $%{value:.0f}M<br>"
            "Total: %{customdata}<br>"
            "Share: %{percent}<extra></extra>"
        ),
    ))
    fig.update_layout(
        **PLOTLY_BASE, height=400,
        showlegend=False,
        margin=dict(l=60, r=60, t=30, b=30),
        annotations=[dict(
            text=f"<b>{fmt(total_ai)}</b><br>Total AI",
            x=0.5, y=0.5,
            font=dict(size=13, color="#eef2f7", family="'Inter', sans-serif"),
            showarrow=False,
            align="center",
        )],
    )
    return fig


# ============================================================================
# TAB 4 — COMPONENTS
# ============================================================================

def make_sankey_chart(
    sk_nodes: list[str],
    sk_src: list[int],
    sk_tgt: list[int],
    sk_val: list[float],
    node_colors: list[str],
    link_colors: list[str],
) -> go.Figure:
    """Sankey diagram: savings flow from Domain → Pillar → Component.

    Args:
        sk_nodes:    Ordered list of node labels.
        sk_src:      Source node indices for each link.
        sk_tgt:      Target node indices for each link.
        sk_val:      Flow value ($M) for each link.
        node_colors: RGBA color string for each node (same order as sk_nodes).
        link_colors: RGBA color string for each link (same order as sk_src/tgt/val).

    Returns:
        Plotly Figure.
    """
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=14,
            thickness=20,
            line=dict(color=COLORS["bg"], width=0.5),
            label=sk_nodes,
            color=node_colors,
            hovertemplate="<b>%{label}</b><br>Total: $%{value:.0f}M<extra></extra>",
        ),
        link=dict(
            source=sk_src,
            target=sk_tgt,
            value=sk_val,
            color=link_colors,
            hovertemplate="<b>%{source.label} → %{target.label}</b><br>$%{value:.1f}M<extra></extra>",
        ),
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        height=640,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_traces(
        textfont=dict(size=11, color="#c4cdd8", family="'DM Sans', sans-serif"),
    )
    return fig


def make_top5_component_bar_chart(
    df_stack: pd.DataFrame,
    short_order: list[str],
    sorted_pillars: list[str],
    short_names: dict[str, str],
) -> go.Figure:
    """Stacked vertical bar: top 5 components broken down by pillar savings.

    Args:
        df_stack:      DataFrame with columns Component, Pillar, Savings (dollars).
        short_order:   Component short names in display order (x-axis order).
        sorted_pillars: Pillar keys sorted by descending total savings.
        short_names:   Mapping of full component name → short display name.

    Returns:
        Plotly Figure.
    """
    if df_stack.empty:
        logger.warning("make_top5_component_bar_chart() called with empty df_stack — returning empty figure.")
        return go.Figure()
    fig = go.Figure()
    for p_key in sorted_pillars:
        p_data = df_stack[df_stack["Pillar"] == p_key].copy()
        if p_data["Savings"].sum() == 0:
            continue
        p_data["ShortName"] = p_data["Component"].map(
            lambda x: short_names.get(x, x[:14]))
        p_data = p_data.set_index("ShortName").reindex(short_order).reset_index()
        fig.add_trace(go.Bar(
            x=p_data["ShortName"],
            y=p_data["Savings"] / 1e6,
            name=PILLAR_LABELS[p_key],
            marker_color=PILLAR_COLORS[p_key],
            marker_line=dict(color=COLORS["bg"], width=0.5),
            text=p_data["Savings"].apply(
                lambda v: f"${v/1e6:.1f}M" if pd.notna(v) and v > 0 else ""),
            textposition="inside", textangle=0,
            textfont=dict(size=8, color="rgba(255,255,255,0.95)",
                          family="'Inter', sans-serif"),
            insidetextanchor="middle", constraintext="none",
            hovertemplate=f"<b>{PILLAR_LABELS[p_key]}</b><br>%{{x}}: $%{{y:.1f}}M<extra></extra>",
        ))
    _apply_base(fig,
                margin=dict(l=50, r=10, t=75, b=55),
                height=480,
                barmode="stack",
                yaxis_title="Annual Savings ($ M)",
                xaxis_title="",
                showlegend=False)
    fig.update_xaxes(**{**_AXIS_BASE, "tickangle": -35, "tickfont": dict(size=9, color="#334155", family="'Inter', sans-serif")})
    return fig


def make_component_treemap_chart(comp_costs: pd.DataFrame) -> go.Figure:
    """Treemap: Domain → Component hierarchy sized by savings, colored categorically.

    Building Systems components use warm tones (orange/amber/coral).
    Facilities Operations components use cool tones (teal/cyan/green/lime).
    Domain nodes use their anchor color; root node is neutral.

    Args:
        comp_costs: DataFrame with columns Component, Domain, Savings (dollars),
                    Savings_Pct (0–1), Base_Annual (dollars).

    Returns:
        Plotly Figure with go.Treemap.
    """
    if comp_costs.empty or comp_costs["Savings"].sum() == 0:
        logger.warning("make_component_treemap_chart() called with no savings data.")
        return go.Figure()

    DOMAIN_SHORT = {
        "Building Systems & Assets": "Building Systems",
        "Facilities Operations":     "Facilities Operations",
    }

    # Distinct palettes per domain — enough slots for up to 12 components each
    BS_PALETTE = [
        "#d6b35a", "#c89f4a", "#eab308", "#b8973a", "#f59e0b",
        "#d97706", "#ca8a04", "#a16207", "#fbbf24", "#e0a020",
        "#f0c040", "#dba830",
    ]
    FO_PALETTE = [
        "#2dd4bf", "#60a5fa", "#34d399", "#4ade80", "#22d3ee",
        "#38bdf8", "#6ee7b7", "#a3e635", "#86efac", "#67e8f9",
        "#14b8a6", "#0ea5e9",
    ]

    df = comp_costs.copy()
    df["Domain_Short"] = df["Domain"].map(DOMAIN_SHORT).fillna(df["Domain"])
    df = df[df["Savings"] > 0].copy()

    # Assign a distinct color to each component within its domain palette
    bs_comps = df[df["Domain_Short"] == "Building Systems"]["Component"].tolist()
    fo_comps = df[df["Domain_Short"] == "Facilities Operations"]["Component"].tolist()
    comp_color: dict[str, str] = {}
    for i, c in enumerate(bs_comps):
        comp_color[c] = BS_PALETTE[i % len(BS_PALETTE)]
    for i, c in enumerate(fo_comps):
        comp_color[c] = FO_PALETTE[i % len(FO_PALETTE)]

    # Build node lists: root + domains + components
    ids, labels, parents, values, colors, custom = [], [], [], [], [], []

    # Root
    ids.append("Portfolio");     labels.append("Portfolio")
    parents.append("");          values.append(df["Savings"].sum())
    colors.append("#1a2540");    custom.append(["", "", df["Savings"].sum(), 0.0])

    # Domain nodes
    domain_anchor = {"Building Systems": "#d6b35a", "Facilities Operations": "#2dd4bf"}
    for dom in ["Building Systems", "Facilities Operations"]:
        sub = df[df["Domain_Short"] == dom]
        if sub.empty:
            continue
        ids.append(dom);         labels.append(dom)
        parents.append("Portfolio"); values.append(sub["Savings"].sum())
        colors.append(domain_anchor[dom])
        custom.append([dom, "", sub["Savings"].sum(), sub["Savings_Pct"].mean()])

    # Component nodes
    for _, row in df.iterrows():
        ids.append(row["Component"]);    labels.append(row["Component"])
        parents.append(row["Domain_Short"])
        values.append(row["Savings"])
        colors.append(comp_color.get(row["Component"], "#8a99b0"))
        custom.append([row["Component"], row["Domain_Short"],
                       row["Savings"], row["Savings_Pct"]])

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        customdata=custom,
        marker=dict(
            colors=colors,
            showscale=False,
            line=dict(width=1.5, color=COLORS["bg"]),
        ),
        texttemplate=(
            "<b>%{label}</b><br>"
            "$%{customdata[2]:.0f}M<br>"
            "%{customdata[3]:.1%}"
        ),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Domain: %{customdata[1]}<br>"
            "Savings: $%{customdata[2]:.0f}M<br>"
            "Reduction: %{customdata[3]:.1%}"
            "<extra></extra>"
        ),
        textfont=dict(size=11, color="#ffffff", family="'DM Sans', sans-serif"),
        pathbar=dict(
            visible=True,
            thickness=18,
            textfont=dict(size=10, color="#94a3b8"),
        ),
        tiling=dict(packing="squarify", pad=3),
    ))

    fig.update_layout(
        **PLOTLY_BASE,
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


# ============================================================================
# ASSET MANAGEMENT DASHBOARD CHARTS
# ============================================================================

_AGE_CLASS_COLORS = {
    "Historic": "#a78bfa",  # violet
    "Old":      "#d6b35a",  # gold
    "Mid":      "#eab308",  # yellow
    "New":      "#2dd4bf",  # teal
}

_BUCKET_COLORS = {
    "Critical": "#ef4444",  # red
    "Aging":    "#d6b35a",  # gold
    "Mature":   "#eab308",  # yellow
    "New":      "#2dd4bf",  # teal
}


def make_ahi_scatter_chart(pa_db: pd.DataFrame) -> go.Figure:
    """Scatter: Asset Health Index vs Cost Intensity, sized by GAV, colored by Age_Class.

    Quadrant lines divide the space into four decision zones:
    High AHI + Low Cost = Keep & Invest | Low AHI + High Cost = Sell / Replace.

    Args:
        pa_db: Portfolio_Assessment joined with dim_building.
               Required cols: AHI, Cost_Intensity, Gross_Asset_Value,
               Age_Class, Department, Asset_ID, Sell_Quadrant_5y.

    Returns:
        Plotly Figure.
    """
    if pa_db.empty:
        logger.warning("make_ahi_scatter_chart() called with empty DataFrame.")
        return go.Figure()

    df = pa_db[pa_db["Cost_Intensity"] < 1e10].copy()
    ahi_mid = 40.0
    ci_mid  = float(df["Cost_Intensity"].median())

    fig = go.Figure()

    for (x0, x1, y0, y1, color, label) in [
        (ahi_mid, 80,  0,      ci_mid, "rgba(45,212,191,0.06)", "Keep & Invest"),
        (0,  ahi_mid,  0,      ci_mid, "rgba(255,209,102,0.04)","Defer / Monitor"),
        (ahi_mid, 80,  ci_mid, 11,     "rgba(0,194,224,0.04)",  "Monitor Costs"),
        (0,  ahi_mid,  ci_mid, 11,     "rgba(255,77,109,0.04)", "Sell / Replace"),
    ]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=color, line_width=0, layer="below")
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=label,
                           showarrow=False,
                           font=dict(size=9, color="rgba(255,255,255,0.18)"),
                           xanchor="center", yanchor="middle")

    fig.add_vline(x=ahi_mid, line_dash="dot",
                  line_color="rgba(255,255,255,0.12)", line_width=1)
    fig.add_hline(y=ci_mid,  line_dash="dot",
                  line_color="rgba(255,255,255,0.12)", line_width=1)

    for age_class, color in _AGE_CLASS_COLORS.items():
        sub = df[df["Age_Class"] == age_class]
        if sub.empty:
            continue
        max_gav = df["Gross_Asset_Value"].max()
        fig.add_trace(go.Scatter(
            x=sub["AHI"],
            y=sub["Cost_Intensity"],
            mode="markers",
            name=age_class,
            marker=dict(
                size=(sub["Gross_Asset_Value"] / max_gav * 28 + 6).clip(6, 34),
                color=color,
                opacity=0.82,
                line=dict(color=COLORS["bg"], width=0.8),
            ),
            customdata=sub[["Asset_ID", "Department", "Gross_Asset_Value",
                             "AHI", "Cost_Intensity", "Sell_Quadrant_5y"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Dept: %{customdata[1]}<br>"
                "AHI: %{customdata[3]:.1f} / 80<br>"
                "Cost Intensity: %{customdata[4]:.2f}<br>"
                "GAV: $%{customdata[2]:,.0f}<br>"
                "Disposition: %{customdata[5]}"
                "<extra></extra>"
            ),
        ))

    _apply_base(fig, height=440,
                xaxis_title="Asset Health Index (0 = worst, 80 = best)",
                yaxis_title="Cost Intensity (OpEx / sqft)",
                hovermode="closest")
    fig.update_xaxes(**_AXIS_BASE, range=[0, 82])
    fig.update_yaxes(**_AXIS_BASE, range=[0, 11])
    return fig


def make_lifecycle_bubble_chart(db: pd.DataFrame) -> go.Figure:
    """Bubble: Current Age (x) vs Remaining Useful Life (y), sized by GAV.

    Args:
        db: dim_building enriched with remaining_life_yrs and life_bucket columns.

    Returns:
        Plotly Figure.
    """
    if db.empty:
        logger.warning("make_lifecycle_bubble_chart() called with empty DataFrame.")
        return go.Figure()

    fig = go.Figure()
    max_gav = db["Gross_Asset_Value_2024"].max()

    for bucket in ["Critical", "Aging", "Mature", "New"]:
        sub = db[db["life_bucket"] == bucket]
        if sub.empty:
            continue
        color = _BUCKET_COLORS[bucket]
        fig.add_trace(go.Scatter(
            x=sub["Age_Years_Current"],
            y=sub["remaining_life_yrs"],
            mode="markers",
            name=bucket,
            marker=dict(
                size=(sub["Gross_Asset_Value_2024"] / max_gav * 30 + 6).clip(6, 36),
                color=color,
                opacity=0.80,
                line=dict(color=COLORS["bg"], width=0.8),
            ),
            customdata=sub[["asset_id", "Department", "Facility_Type",
                             "Age_Years_Current", "remaining_life_yrs",
                             "Gross_Asset_Value_2024"]].values,
            hovertemplate=(
                "<b>%{customdata[2]}</b><br>"
                "Dept: %{customdata[1]}<br>"
                "Age: %{customdata[3]:.1f} yrs | Remaining: %{customdata[4]:.1f} yrs<br>"
                "GAV: $%{customdata[5]:,.0f}"
                "<extra></extra>"
            ),
        ))

    max_life = float(db["Life_Years_Total"].max())
    fig.add_shape(type="line", x0=0, y0=max_life, x1=max_life, y1=0,
                  line=dict(color="rgba(255,255,255,0.07)", dash="dot", width=1))
    fig.add_annotation(x=max_life * 0.65, y=max_life * 0.12,
                       text="End of Life boundary", showarrow=False,
                       font=dict(size=9, color="rgba(255,255,255,0.18)"), textangle=-40)

    _apply_base(fig, height=420,
                xaxis_title="Current Age (years)",
                yaxis_title="Remaining Useful Life (years)",
                hovermode="closest")
    fig.update_xaxes(**_AXIS_BASE)
    fig.update_yaxes(**_AXIS_BASE)
    return fig


def make_replacement_timeline_chart(repl_df: pd.DataFrame) -> go.Figure:
    """Stacked bar: buildings replaced per year by department.

    Args:
        repl_df: fact_savings[replaced==True] joined with dim_building.
                 Required cols: year, Department, asset_id, Gross_Asset_Value_2024.

    Returns:
        Plotly Figure.
    """
    if repl_df.empty:
        logger.warning("make_replacement_timeline_chart() called with empty DataFrame.")
        return go.Figure()

    dept_yr = (repl_df.groupby(["year", "Department"])
               .agg(n=("asset_id", "count"), gav=("Gross_Asset_Value_2024", "sum"))
               .reset_index())

    top_depts = (dept_yr.groupby("Department")["n"].sum()
                 .sort_values(ascending=False).head(6).index.tolist())
    palette   = [COLORS["orange"], COLORS["green"], COLORS["cyan"],
                 COLORS["purple"], COLORS["yellow"], COLORS["red"]]

    fig = go.Figure()
    for i, dept in enumerate(top_depts):
        sub   = dept_yr[dept_yr["Department"] == dept]
        short = (dept.replace("Chicago ", "")
                     .replace("Department of ", "")
                     .replace(" Department", ""))
        fig.add_trace(go.Bar(
            x=sub["year"], y=sub["n"],
            name=short,
            marker_color=palette[i % len(palette)],
            customdata=sub[["gav"]].values,
            hovertemplate=(
                f"<b>{short}</b><br>"
                "Year: %{x}<br>Buildings: %{y}<br>"
                "GAV: $%{customdata[0]:,.0f}<extra></extra>"
            ),
        ))

    _apply_base(fig, height=320,
                barmode="stack",
                yaxis_title="Buildings Replaced",
                xaxis_title="")
    fig.update_xaxes(**_AXIS_BASE, dtick=1)
    fig.update_yaxes(**_AXIS_BASE)
    return fig


def make_am_savings_bucket_chart(am_yr: pd.DataFrame) -> go.Figure:
    """Stacked area: Asset Management L1 savings per year by lifecycle bucket.

    Args:
        am_yr: DataFrame with columns year, life_bucket, l1 (dollars).

    Returns:
        Plotly Figure.
    """
    if am_yr.empty:
        logger.warning("make_am_savings_bucket_chart() called with empty DataFrame.")
        return go.Figure()

    fig = go.Figure()
    fill_colors = {
        "Critical": "rgba(255,77,109,0.55)",
        "Aging":    "rgba(255,107,0,0.50)",
        "Mature":   "rgba(255,209,102,0.48)",
        "New":      "rgba(45,212,191,0.40)",
    }

    for bucket in ["Critical", "Aging", "Mature", "New"]:
        sub = am_yr[am_yr["life_bucket"] == bucket].sort_values("year")
        if sub.empty or sub["l1"].sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["l1"] / 1e6,
            name=bucket,
            mode="lines",
            stackgroup="bucket",
            fillcolor=fill_colors[bucket],
            line=dict(color=_BUCKET_COLORS[bucket], width=1.5),
            hovertemplate=f"<b>{bucket}</b><br>%{{x}}: $%{{y:.0f}}M<extra></extra>",
        ))

    _apply_base(fig, height=350,
                yaxis_title="AM Savings ($ Millions)",
                hovermode="x unified")
    fig.update_xaxes(**_AXIS_BASE, dtick=1)
    fig.update_yaxes(**_AXIS_BASE)
    return fig


def make_am_component_chart(cpm_am: pd.DataFrame) -> go.Figure:
    """Horizontal bar: Asset Management weight per component, colored by domain.

    Args:
        cpm_am: Component_Pillar_Map filtered for Pillar_Key == Asset_Management.
                Required cols: Component, Domain, Weight.

    Returns:
        Plotly Figure.
    """
    if cpm_am.empty:
        logger.warning("make_am_component_chart() called with empty DataFrame.")
        return go.Figure()

    df     = cpm_am.sort_values("Weight", ascending=True)
    colors = [COLORS["orange"] if "Building" in d else COLORS["cyan"]
              for d in df["Domain"]]

    fig = go.Figure(go.Bar(
        y=df["Component"],
        x=df["Weight"],
        orientation="h",
        marker_color=colors,
        marker_line=dict(color=COLORS["bg"], width=0.5),
        text=df["Weight"].apply(lambda w: f"{w:.0%}"),
        textposition="outside",
        textfont=dict(size=9, color="#94a3b8"),
        hovertemplate="<b>%{y}</b><br>AM Weight: %{x:.0%}<extra></extra>",
    ))

    _apply_base(fig,
                height=max(CHART_BASE_HEIGHT, len(df) * CHART_HEIGHT_PER_ROW),
                xaxis_title="Asset Management Influence Weight",
                yaxis_title="",
                showlegend=False)
    fig.update_xaxes(**_AXIS_BASE, tickformat=".0%", range=[0, 1.15])
    fig.update_yaxes(**_AXIS_BASE)
    return fig


# ============================================================================
# COMPONENT × PILLAR NETWORK GRAPH
# ============================================================================

def make_network_chart(
    comp_df: pd.DataFrame,
    cpm_df: pd.DataFrame,
    pillar_overrides: dict[str, float],
    view_mode: str = "all",
) -> go.Figure:
    """Force-directed network: Component nodes connected to Pillar nodes.

    Three view modes:
        "all"  — Components (BS + FO) + all active Pillars, coloured by domain.
        "bs"   — Only Building Systems components + their pillars.
        "fo"   — Only Facilities Operations components + their pillars.

    Node sizes are proportional to the component's share of facility cost (or
    fixed for pillar diamonds). Edges are weighted by the pillar assignment
    weight and coloured by the pillar's accent colour.

    Args:
        comp_df:          Components sheet — columns: Component, Domain,
                          Pct_of_Facility_Cost. Caller should also add
                          base_annual, savings, savings_pct for richer hovers.
        cpm_df:           Component_Pillar_Map sheet — columns: Component,
                          Domain, Pillar_Key, Weight.
        pillar_overrides: Dict pillar_key → active rate (0 = disabled).
        view_mode:        "all" | "bs" | "fo"

    Returns:
        Plotly Figure using go.Scatter for nodes and edges.

    Raises:
        ModuleNotFoundError: If the `networkx` package is not installed.
    """
    import networkx as nx

    if comp_df is None or comp_df.empty:
        logger.warning("make_network_chart() received empty comp_df.")
        return go.Figure()

    BS_DOMAIN = "Building Systems & Assets"
    FO_DOMAIN = "Facilities Operations"

    # ── 1. Filter components by view mode ────────────────────────────────────
    if view_mode == "bs":
        comps_use = comp_df[comp_df["Domain"] == BS_DOMAIN].copy()
    elif view_mode == "fo":
        comps_use = comp_df[comp_df["Domain"] == FO_DOMAIN].copy()
    else:
        comps_use = comp_df.copy()

    if comps_use.empty:
        return go.Figure()

    active_pillars = {k for k, r in pillar_overrides.items() if r > 0}

    # ── 2. Build networkx graph ───────────────────────────────────────────────
    G = nx.Graph()

    for _, row in comps_use.iterrows():
        G.add_node(
            row["Component"],
            ntype="component",
            domain="BS" if row["Domain"] == BS_DOMAIN else "FO",
            cost=float(row.get("base_annual", row["Pct_of_Facility_Cost"] * 1e9)),
            savings=float(row.get("savings", 0)),
            pct=float(row.get("savings_pct", 0)),
        )

    comp_names = set(comps_use["Component"].tolist())
    relevant_pillars = set(
        cpm_df[cpm_df["Component"].isin(comp_names)]["Pillar_Key"].unique()
    )
    if active_pillars:
        relevant_pillars &= active_pillars

    for pkey in relevant_pillars:
        G.add_node(pkey, ntype="pillar", domain="pillar", cost=0, savings=0, pct=0)

    for _, row in cpm_df.iterrows():
        comp = row["Component"]
        pkey = row["Pillar_Key"]
        if comp not in comp_names or pkey not in relevant_pillars:
            continue
        G.add_edge(comp, pkey, weight=float(row["Weight"]), pillar=pkey)

    if G.number_of_nodes() == 0:
        return go.Figure()

    # ── 3. Spring layout ──────────────────────────────────────────────────────
    pos = nx.spring_layout(G, k=2.8, iterations=120, seed=42, weight="weight")

    # ── 4. Build Plotly traces ────────────────────────────────────────────────
    fig = go.Figure()

    # Edge traces — one per pillar colour
    edges_by_pillar: dict[str, list] = {}
    for u, v, data in G.edges(data=True):
        pk = data.get("pillar", "")
        edges_by_pillar.setdefault(pk, []).append((u, v, data["weight"]))

    for pkey, edge_list in edges_by_pillar.items():
        ex, ey = [], []
        for u, v, _ in edge_list:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            ex += [x0, x1, None]
            ey += [y0, y1, None]
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(color=PILLAR_COLORS.get(pkey, "#5a6a82"), width=1.4),
            opacity=0.35, hoverinfo="skip", showlegend=False,
        ))

    # Component nodes (squares, split by domain for legend)
    for domain, domain_label, marker_color in [
        ("BS", "Building Systems", COLORS["orange"]),
        ("FO", "Facilities Ops",   COLORS["green"]),
    ]:
        nodes_d = [n for n, d in G.nodes(data=True)
                   if d["ntype"] == "component" and d["domain"] == domain]
        if not nodes_d:
            continue
        nx_vals, ny_vals, n_text, n_hover, n_sizes = [], [], [], [], []
        for n in nodes_d:
            nx_vals.append(pos[n][0]); ny_vals.append(pos[n][1])
            d       = G.nodes[n]
            cost_m  = d["cost"] / 1e6
            sav_m   = d["savings"] / 1e6
            short   = n if len(n) <= 22 else n[:20] + "…"
            n_text.append(short)
            n_hover.append(
                f"<b>{n}</b><br>"
                f"Domain: {'Building Systems' if domain == 'BS' else 'Facilities Ops'}<br>"
                f"Annual Cost: ${cost_m:.0f}M<br>"
                f"Proj. Savings: ${sav_m:.1f}M<br>"
                f"Reduction: {d['pct']:.1%}"
            )
            n_sizes.append(max(16, min(52, cost_m ** 0.45)))

        fig.add_trace(go.Scatter(
            x=nx_vals, y=ny_vals,
            mode="markers+text",
            name=domain_label,
            marker=dict(symbol="square", size=n_sizes, color=marker_color,
                        opacity=0.82, line=dict(color=COLORS["bg"], width=1.5)),
            text=n_text,
            textposition="top center",
            textfont=dict(size=8, color=marker_color, family="'Inter', sans-serif"),
            hovertext=n_hover, hoverinfo="text",
        ))

    # Pillar nodes (diamonds)
    pillar_nodes = [n for n, d in G.nodes(data=True) if d["ntype"] == "pillar"]
    if pillar_nodes:
        px_vals, py_vals, p_text, p_hover, p_colors = [], [], [], [], []
        for n in pillar_nodes:
            px_vals.append(pos[n][0]); py_vals.append(pos[n][1])
            col  = PILLAR_COLORS.get(n, COLORS["purple"])
            rate = pillar_overrides.get(n, 0)
            p_colors.append(col)
            p_text.append(PILLAR_LABELS.get(n, n))
            p_hover.append(
                f"<b>{PILLAR_LABELS.get(n, n)}</b><br>"
                f"Type: L1 Optimization Pillar<br>"
                f"Active Rate: {rate:.1%}<br>"
                f"Connected components: {G.degree(n)}"
            )
        fig.add_trace(go.Scatter(
            x=px_vals, y=py_vals,
            mode="markers+text",
            name="Optimization Pillars",
            marker=dict(symbol="diamond", size=30, color=p_colors,
                        opacity=0.90, line=dict(color=COLORS["bg"], width=2)),
            text=p_text,
            textposition="bottom center",
            textfont=dict(size=9, color="#eef2f7", family="'DM Sans', sans-serif"),
            hovertext=p_hover, hoverinfo="text",
        ))

    # ── 5. Layout ─────────────────────────────────────────────────────────────
    x_vals = [v[0] for v in pos.values()]
    y_vals = [v[1] for v in pos.values()]
    fig.update_layout(
        **PLOTLY_BASE,
        height=680,
        showlegend=True,
        legend={**_LEGEND_BASE, "y": 1.04},
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[min(x_vals) - 0.3, max(x_vals) + 0.3]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[min(y_vals) - 0.3, max(y_vals) + 0.3]),
        hovermode="closest",
    )
    return fig


# ============================================================================
# EXECUTIVE DASHBOARD CHARTS  (asset_management_dashboard.py)
# ============================================================================

def make_portfolio_status_chart(db: pd.DataFrame) -> go.Figure:
    """Horizontal stacked bar: building count by lifecycle bucket per department.

    Shows which departments have the most urgent asset risk.
    Sorted so the most critical departments appear at the top.

    Args:
        db: dim_building enriched with life_bucket and Department columns.

    Returns:
        Plotly Figure.
    """
    if db.empty:
        logger.warning("make_portfolio_status_chart() called with empty DataFrame.")
        return go.Figure()

    dept_bucket = (
        db.groupby(["Department", "life_bucket"])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=["Critical", "Aging", "Mature", "New"], fill_value=0)
    )
    dept_bucket = dept_bucket.sort_values("Critical", ascending=True)

    dept_labels = [
        d.replace("Chicago ", "").replace("Department of ", "").replace(" Department", "")
        for d in dept_bucket.index
    ]

    fig = go.Figure()
    for bucket, color in [
        ("Critical", _BUCKET_COLORS["Critical"]),
        ("Aging",    _BUCKET_COLORS["Aging"]),
        ("Mature",   _BUCKET_COLORS["Mature"]),
        ("New",      _BUCKET_COLORS["New"]),
    ]:
        counts = dept_bucket[bucket].tolist() if bucket in dept_bucket.columns else [0] * len(dept_bucket)
        fig.add_trace(go.Bar(
            y=dept_labels,
            x=counts,
            name=bucket,
            orientation="h",
            marker_color=color,
            marker_line=dict(color=COLORS["bg"], width=0.5),
            hovertemplate=f"<b>%{{y}}</b><br>{bucket}: %{{x}} edificios<extra></extra>",
        ))

    _apply_base(fig,
                height=max(CHART_BASE_HEIGHT, len(dept_bucket) * CHART_HEIGHT_PER_ROW),
                barmode="stack",
                xaxis_title="Número de edificios",
                yaxis_title="")
    fig.update_xaxes(**_AXIS_BASE, dtick=5)
    fig.update_yaxes(**_AXIS_BASE)
    return fig


def make_cost_projection_chart(
    baseline_yr: pd.DataFrame,
    am_sav_yr: pd.DataFrame,
) -> go.Figure:
    """Two-line chart: portfolio cost without action vs. with AM optimization.

    The gap between the lines is shaded and annotated with the annual saving
    at the last projection year.

    Args:
        baseline_yr: DataFrame with columns year, baseline (total portfolio cost).
        am_sav_yr:   DataFrame with columns year, l1 (total AM savings per year,
                     already aggregated across lifecycle buckets).

    Returns:
        Plotly Figure.
    """
    if baseline_yr.empty:
        logger.warning("make_cost_projection_chart() called with empty baseline_yr.")
        return go.Figure()

    merged = (baseline_yr
              .merge(am_sav_yr.rename(columns={"l1": "am_sav"}), on="year", how="left")
              .fillna({"am_sav": 0}))
    merged["optimized"] = (merged["baseline"] - merged["am_sav"]).clip(lower=0)

    year_last = int(merged["year"].max())
    gap_last  = float(merged.loc[merged["year"] == year_last, "am_sav"].iloc[0])
    base_last = float(merged.loc[merged["year"] == year_last, "baseline"].iloc[0])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=merged["year"], y=merged["baseline"] / 1e6,
        name="Sin optimización (tendencia)",
        mode="lines+markers",
        line=dict(color=COLORS["red"], width=2.5, dash="dot"),
        marker=dict(size=5, color=COLORS["red"], symbol="circle-open"),
        hovertemplate="<b>Sin optimización</b><br>%{x}: $%{y:.1f}M<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=merged["year"], y=merged["optimized"] / 1e6,
        name="Con optimización AM",
        mode="lines+markers",
        line=dict(color=COLORS["green"], width=3),
        marker=dict(size=6, color=COLORS["green"]),
        fill="tonexty",
        fillcolor="rgba(45,212,191,0.07)",
        hovertemplate="<b>Con optimización AM</b><br>%{x}: $%{y:.1f}M<extra></extra>",
    ))

    fig.add_annotation(
        x=year_last, y=base_last / 1e6,
        text=f"▼ {fmt(gap_last)} anuales",
        showarrow=True, arrowhead=2, arrowcolor=COLORS["green"], arrowwidth=1.5,
        font=dict(size=11, color=COLORS["green"], family="'Inter', sans-serif"),
        bgcolor="rgba(255,255,255,0.92)", bordercolor=COLORS["green"], borderwidth=1,
        ax=30, ay=-50,
    )

    _apply_base(fig,
                height=400,
                yaxis_title="Costo operativo anual ($ Millones)",
                hovermode="x unified",
                legend={"y": 1.10})
    fig.update_xaxes(**_AXIS_BASE, dtick=1)
    return fig


def make_savings_scenarios_chart(
    baseline_yr: pd.DataFrame,
    selected_rate: float,
    k: float = 1.0,
    x0: float = 5.0,
    start_year: int = 2024,
) -> go.Figure:
    """Area+lines chart: cumulative AM savings for the selected rate vs. reference scenarios.

    Reference scenario lines (thin, dashed) give context; the selected rate is
    shown as a thick solid line with a shaded area.

    Args:
        baseline_yr:   DataFrame with columns year, baseline (total portfolio cost).
        selected_rate: User-selected AM improvement rate (0.0–0.25).
        k:             Sigmoid steepness parameter.
        x0:            Sigmoid inflection point (year of 50% adoption).
        start_year:    First year of projection (used to compute elapsed time).

    Returns:
        Plotly Figure.
    """
    if baseline_yr.empty:
        logger.warning("make_savings_scenarios_chart() called with empty baseline_yr.")
        return go.Figure()

    def _cumulative(rate: float) -> pd.DataFrame:
        rows, cumsum = [], 0.0
        for _, row in baseline_yr.sort_values("year").iterrows():
            elapsed  = row["year"] - start_year
            annual   = row["baseline"] * rate * sigmoid(elapsed, k, x0)
            cumsum  += annual
            rows.append({"year": row["year"], "cumulative": cumsum})
        return pd.DataFrame(rows)

    fig = go.Figure()

    # Reference scenarios — thin dashed lines for context
    ref_scenarios = [
        ("Mínimo (2%)",    0.02, "rgba(255,77,109,0.55)"),
        ("Moderado (10%)", 0.10, "rgba(255,209,102,0.65)"),
        ("Óptimo (20%)",   0.20, "rgba(45,212,191,0.55)"),
    ]
    for label, rate, color in ref_scenarios:
        df_s = _cumulative(rate)
        fig.add_trace(go.Scatter(
            x=df_s["year"], y=df_s["cumulative"] / 1e6,
            name=label,
            mode="lines",
            line=dict(color=color, width=1.5, dash="dot"),
            hovertemplate=f"<b>{label}</b><br>%{{x}}: acumulado $%{{y:.0f}}M<extra></extra>",
        ))

    # Selected rate — thick solid + shaded area
    df_sel    = _cumulative(selected_rate)
    cum_total = float(df_sel["cumulative"].iloc[-1])
    year_last = int(df_sel["year"].iloc[-1])

    fig.add_trace(go.Scatter(
        x=df_sel["year"], y=df_sel["cumulative"] / 1e6,
        name=f"Tu escenario ({selected_rate:.0%})",
        mode="lines+markers",
        line=dict(color=COLORS["cyan"], width=3),
        marker=dict(size=7, color=COLORS["cyan"]),
        fill="tozeroy",
        fillcolor="rgba(0,194,224,0.08)",
        hovertemplate="<b>Escenario seleccionado</b><br>%{x}: acumulado $%{y:.0f}M<extra></extra>",
    ))

    fig.add_annotation(
        x=year_last, y=cum_total / 1e6,
        text=f"<b>{fmt(cum_total)}</b><br>acumulado",
        showarrow=True, arrowhead=2, arrowcolor=COLORS["cyan"], arrowwidth=1.5,
        font=dict(size=11, color=COLORS["cyan"], family="'Inter', sans-serif"),
        bgcolor="rgba(255,255,255,0.92)", bordercolor=COLORS["cyan"], borderwidth=1,
        ax=0, ay=-55,
    )

    _apply_base(fig,
                height=390,
                yaxis_title="Ahorros acumulados ($ Millones)",
                hovermode="x unified",
                legend={"y": 1.10})
    fig.update_xaxes(**_AXIS_BASE, dtick=1)
    return fig
