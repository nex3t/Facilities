"""
mro_benchmark.py — MRO Analytics Capability Benchmark
City of Chicago — Department of Procurement Services

Standalone module following the vendor_dashboard.py pattern.
Contains all benchmark data (static), Plotly chart functions,
and the render_mro_benchmark_tab() entrypoint called from
facilities_dashboard_v3.py.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    COLORS,
    PLOTLY_BASE,
    _AXIS_BASE,
    _LEGEND_BASE,
    _MARGIN_BASE,
)
from ui import kcard, section
from score_engine import compute_all_scores, compute_5yr_overpayment

# ── Blue-grey institutional palette (internal constants) ─────────────────────
_C = {
    # Surfaces
    "card":        "#ffffff",
    "card_alt":    "#f1f5f9",       # header row of matrix
    "border":      "rgba(71,85,105,0.18)",
    "border_soft": "rgba(71,85,105,0.09)",
    # Text
    "text":        "#0f172a",
    "text_sub":    "#1e3a5f",
    "text_muted":  "#475569",
    "text_faint":  "#94a3b8",
    # Score heat — green → amber → red
    "score_hi_bg":  "#dcfce7", "score_hi_fg":  "#1a7a4a",   # ≥8.5  green
    "score_med_bg": "#dbeafe", "score_med_fg": "#1d4ed8",   # ≥7.0  blue
    "score_ok_bg":  "#fef9c3", "score_ok_fg":  "#92400e",   # ≥5.5  amber
    "score_lo_bg":  "#ffedd5", "score_lo_fg":  "#9a3412",   # ≥3.5  orange
    "score_cr_bg":  "#fee2e2", "score_cr_fg":  "#991b1b",   # <3.5  red
    # Gap bars — reference palette
    "gap_hi":  "#dc2626",   # gap > 5  — red
    "gap_med": "#d97706",   # gap 3-5  — amber
    "gap_lo":  "#1a7a4a",   # gap < 3  — green
    # Progress bars in info panel
    "bar_hi":  "#1a7a4a",
    "bar_med": "#d97706",
    "bar_lo":  "#dc2626",
    "bar_cr":  "#7c3aed",
    # Roadmap phases — reference palette progression
    "phase1": "#dc2626",    # red (urgent)
    "phase2": "#d97706",    # amber (mid)
    "phase3": "#1a7a4a",    # green (achieved)
    # Chicago callout — fuchsia/pink matching reference
    "chicago_accent": "#c026d3",
    "chicago_bg":     "rgba(232,121,249,0.08)",
    "chicago_border": "rgba(232,121,249,0.30)",
    "chicago_dot":    "#e879f9",
}


def _fmt_kpi(val: float) -> str:
    if val >= 1e9:
        return f"${val / 1e9:.1f}B"
    if val >= 1e6:
        return f"${val / 1e6:.1f}M"
    if val >= 1e3:
        return f"${val / 1e3:.0f}K"
    return f"${val:,.0f}"


def _build_meta(dims, chi_scores, al_scores):
    avg     = round(sum(chi_scores) / len(chi_scores), 1)
    al_avg  = sum(al_scores) / len(al_scores)
    gap     = round(al_avg - avg, 1)
    risk_lvl = (
        "CRITICAL" if avg < 3.5 else
        "HIGH"     if avg < 5.5 else
        "MODERATE" if avg < 7.5 else
        "LOW"
    )
    gaps_by_dim = sorted(
        zip(dims, [a - c for a, c in zip(al_scores, chi_scores)], chi_scores),
        key=lambda x: -x[1],
    )
    priority = (
        f"{gaps_by_dim[0][0]} ({gaps_by_dim[0][2]:.1f}) "
        f"& {gaps_by_dim[1][0]} ({gaps_by_dim[1][2]:.1f}) most urgent"
        if len(gaps_by_dim) >= 2
        else f"{gaps_by_dim[0][0]} ({gaps_by_dim[0][2]:.1f}) most urgent"
    )
    return {
        "avg": str(avg), "gap": f"{gap:.1f} pts below GSA / Leading Municipal",
        "risk": risk_lvl, "priority": priority,
        "ci": list(chi_scores), "al": list(al_scores),
    }


# ============================================================================
# STATIC BENCHMARK DATA
# ============================================================================

PARTICIPANTS: dict[str, dict] = {
    "max":     {"label": "Maximum Performance",                       "color": "#16a34a", "dash": "solid",  "width": 2.0},
    "digital": {"label": "Algorithmically & Digitally Native",        "color": "#4ade80", "dash": "solid",  "width": 2.0},
    "alpha":   {"label": "Alpha Best Performer (GSA / Leading Municipal)", "color": "#3b82f6", "dash": "solid",  "width": 2.5},
    "bench":   {"label": "Benchmark Performance",                     "color": "#f59e0b", "dash": "dash",   "width": 2.0},
    "late":    {"label": "Late Adopters",                             "color": "#ef4444", "dash": "dot",    "width": 1.8},
    "chicago": {"label": "Observed — City of Chicago",                "color": "#e879f9", "dash": "solid",  "width": 3.0},
}

PERF_DIMS: list[str] = [
    "Budget", "Payments", "Purchase Orders", "Vendor",
    "Category", "Component", "Return", "Risk",
]
PERF_SCORES: dict[str, list[float]] = {
    "max":     [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    "digital": [ 9.5,  9.5,  9.0,  9.5,  9.0,  8.5,  8.5,  9.0],
    "alpha":   [ 9.0,  9.0,  9.5,  9.5,  9.0,  9.5,  8.5,  9.0],
    "bench":   [ 7.0,  7.0,  5.5,  6.5,  5.5,  5.0,  4.5,  5.5],
    "late":    [ 4.0,  4.5,  2.5,  3.0,  2.5,  1.5,  1.5,  2.5],
    "chicago": [ 6.5,  7.5,  3.5,  5.5,  4.0,  2.5,  2.0,  3.5],
}
PERF_META = {
    "avg": "4.4", "gap": "4.7 pts below GSA / Leading Municipal", "risk": "HIGH",
    "priority": "Purchase Orders (3.5) & Component visibility (2.5) most urgent",
    "ci": [6.5, 7.5, 3.5, 5.5, 4.0, 2.5, 2.0, 3.5],
    "al": [9.0, 9.0, 9.5, 9.5, 9.0, 9.5, 8.5, 9.0],
}

PRED_DIMS: list[str] = ["Return Analytics", "Investment Analytics", "Risk Analytics"]
PRED_SCORES: dict[str, list[float]] = {
    "max":     [10.0, 10.0, 10.0],
    "digital": [ 8.5,  9.0,  8.5],
    "alpha":   [ 9.5,  9.0,  9.5],
    "bench":   [ 4.5,  5.0,  4.5],
    "late":    [ 1.5,  2.0,  1.5],
    "chicago": [ 2.5,  3.5,  3.0],
}
PRED_META = {
    "avg": "3.0", "gap": "6.2 pts below GSA / Leading Municipal", "risk": "CRITICAL",
    "priority": "No predictive analytics — condition-based maintenance model required",
    "ci": [2.5, 3.5, 3.0], "al": [9.5, 9.0, 9.5],
}

PRES_DIMS: list[str] = ["Financial Optimization", "Work Modernization", "Risk Mitigation"]
PRES_SCORES: dict[str, list[float]] = {
    "max":     [10.0, 10.0, 10.0],
    "digital": [ 8.5,  8.0,  8.5],
    "alpha":   [ 9.0,  9.5,  9.0],
    "bench":   [ 4.5,  4.0,  4.5],
    "late":    [ 1.5,  1.0,  1.5],
    "chicago": [ 2.5,  2.0,  2.5],
}
PRES_META = {
    "avg": "2.3", "gap": "6.6 pts below GSA / Leading Municipal", "risk": "CRITICAL",
    "priority": "Work modernization critical — adopt JIT & continuous improvement approach",
    "ci": [2.5, 2.0, 2.5], "al": [9.0, 9.5, 9.0],
}

ROADMAPS: dict[str, list[dict]] = {
    "performance": [
        {"phase": "Phase 1", "time": "0–6 mo",   "color": _C["phase1"], "score": "4.4→5.5",
         "actions": ["Centralize MRO purchase orders across departments",
                     "Launch vendor master — reduce to strategic suppliers",
                     "Implement commodity coding NAICS + equipment class",
                     "Deploy spend analytics vs. equipment lifecycle model"]},
        {"phase": "Phase 2", "time": "6–18 mo",  "color": _C["phase2"], "score": "5.5→6.5",
         "actions": ["Category management with lifecycle scoring",
                     "Component-level spend visibility",
                     "Cooperative contract enrollment across depts",
                     "Supplier performance scorecards + review cycles"]},
        {"phase": "Phase 3", "time": "18–36 mo", "color": _C["phase3"], "score": "6.5→7.5",
         "actions": ["ROI tracking per MRO category",
                     "Risk monitoring & mitigation framework",
                     "Dynamic market benchmarking quarterly",
                     "Target Alpha best performer level (9.2 avg)"]},
    ],
    "prediction": [
        {"phase": "Phase 1", "time": "0–6 mo",   "color": _C["phase1"], "score": "3.0→4.5",
         "actions": ["Build MRO spend forecasting model",
                     "Historical baseline 2021–2025 analysis",
                     "Category demand trends by equipment type",
                     "Adopt condition-based demand sensing"]},
        {"phase": "Phase 2", "time": "6–18 mo",  "color": _C["phase2"], "score": "4.5→6.0",
         "actions": ["Equipment lifecycle cost modeling",
                     "Vendor risk scoring & dual-source mandates",
                     "Predictive maintenance pilot — fleet & HVAC",
                     "Asset telemetry integration for city fleet"]},
        {"phase": "Phase 3", "time": "18–36 mo", "color": _C["phase3"], "score": "6.0→8.0",
         "actions": ["Full IoT asset telemetry integration",
                     "AI-driven parts demand forecasting",
                     "Risk analytics — supply chain visibility",
                     "Target Alpha best performer level (9.3 avg)"]},
    ],
    "prescription": [
        {"phase": "Phase 1", "time": "0–6 mo",   "color": _C["phase1"], "score": "2.3→3.5",
         "actions": ["Adopt JIT for high-velocity MRO items",
                     "Workshop: eliminate emergency PO waste",
                     "Centralize supplier contracts city-wide",
                     "Standardize parts catalog — reduce SKUs"]},
        {"phase": "Phase 2", "time": "6–18 mo",  "color": _C["phase2"], "score": "3.5→5.5",
         "actions": ["Planned maintenance replaces reactive spend",
                     "CI program per MRO category",
                     "Dual-source for critical MRO suppliers",
                     "Same-day parts availability framework"]},
        {"phase": "Phase 3", "time": "18–36 mo", "color": _C["phase3"], "score": "5.5→7.5",
         "actions": ["Full planned maintenance MRO workflow",
                     "Optimization via continuous improvement",
                     "Autonomous risk triggers at component level",
                     "Target Alpha best performer level (9.2 avg)"]},
    ],
}


# ============================================================================
# LIVE SCORE DERIVATION
# ============================================================================

def derive_vendor_score_from_df(vendor_df: pd.DataFrame) -> float:
    total_spend = vendor_df["fragmented_spend"].sum()
    if total_spend <= 0:
        return PERF_SCORES["chicago"][3]
    wtd_pidx = (
        (vendor_df["price_index"] * vendor_df["fragmented_spend"]).sum() / total_spend
    )
    score = 10.0 * max(0.0, 1.0 - (wtd_pidx - 1.0) / 0.32)
    return round(min(max(score, 0.0), 10.0), 1)


# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def _apply_base(fig: go.Figure, **layout_kw) -> go.Figure:
    kw = {**PLOTLY_BASE, **layout_kw}
    kw.setdefault("legend", {**_LEGEND_BASE})
    kw.setdefault("margin", {**_MARGIN_BASE})
    fig.update_layout(**kw)
    fig.update_xaxes(**_AXIS_BASE)
    fig.update_yaxes(**_AXIS_BASE)
    return fig


def make_participant_line_chart(dims, scores) -> go.Figure:
    fig = go.Figure()
    for key, meta in PARTICIPANTS.items():
        if key not in scores:
            continue
        fig.add_trace(go.Scatter(
            x=dims, y=scores[key],
            name=meta["label"],
            mode="lines+markers",
            line=dict(color=meta["color"], width=meta["width"], dash=meta["dash"]),
            marker=dict(
                size=10 if key == "chicago" else 7,
                color=meta["color"],
                line=dict(color="#ffffff", width=2),
            ),
            hovertemplate=f"<b>{meta['label']}</b><br>%{{x}}: <b>%{{y:.1f}}</b> / 10<extra></extra>",
        ))
    _apply_base(
        fig, height=420,
        yaxis=dict(**_AXIS_BASE, range=[0, 10.5], title="Score (0–10)", dtick=2),
        xaxis=dict(**_AXIS_BASE),
        legend={**_LEGEND_BASE, "orientation": "h", "y": -0.32, "x": 0,
                "xanchor": "left", "yanchor": "top",
                "font": dict(size=10, color=_C["text_muted"]),
                "traceorder": "normal", "itemwidth": 40},
        margin=dict(l=52, r=16, t=20, b=130),
    )
    return fig


def make_score_matrix_df(dims, scores) -> pd.DataFrame:
    rows = []
    for key, meta in PARTICIPANTS.items():
        if key not in scores:
            continue
        row = {"Participant": meta["label"]}
        vals = scores[key]
        for i, dim in enumerate(dims):
            row[dim] = vals[i]
        row["AVG"] = round(sum(vals) / len(vals), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def make_gap_bar_chart(dims, ci_scores, al_scores) -> go.Figure:
    gaps   = [round(al - ci, 1) for al, ci in zip(al_scores, ci_scores)]
    colors = [
        _C["gap_hi"] if g > 5 else _C["gap_med"] if g > 3 else _C["gap_lo"]
        for g in gaps
    ]
    fig = go.Figure(go.Bar(
        x=gaps, y=dims, orientation="h",
        marker_color=colors,
        marker=dict(opacity=0.80, line=dict(color="#ffffff", width=1)),
        text=[f"−{g:.1f} pts" for g in gaps],
        textposition="outside",
        textfont=dict(size=11, color=_C["text_sub"], family="'Inter', sans-serif"),
        hovertemplate="<b>%{y}</b><br>Gap to Alpha: −%{x:.1f} pts<extra></extra>",
        width=0.55,
    ))
    _apply_base(
        fig, height=max(240, len(dims) * 42),
        xaxis=dict(**_AXIS_BASE, title="Score gap (pts)",
                   range=[0, max(gaps) * 1.25 if gaps else 10]),
        yaxis=dict(**_AXIS_BASE),
        showlegend=False,
        margin=dict(l=160, r=80, t=16, b=40),
    )
    return fig


def make_roadmap_chart(tab_key: str) -> go.Figure:
    phases = ROADMAPS.get(tab_key, [])
    if not phases:
        return go.Figure()
    time_map = {"0–6 mo": (0, 6), "6–18 mo": (6, 18), "18–36 mo": (18, 36)}
    fig = go.Figure()
    for ph in phases:
        start, end = time_map.get(ph["time"], (0, 6))
        actions_str = "<br>· ".join(ph["actions"])
        fig.add_trace(go.Bar(
            name=f"{ph['phase']} ({ph['time']})",
            x=[end - start], y=[ph["phase"]],
            base=[start], orientation="h",
            marker=dict(color=ph["color"], opacity=0.80,
                        line=dict(color="#ffffff", width=1.5)),
            text=f"Score: {ph['score']}",
            textposition="inside",
            textfont=dict(size=11, color="#ffffff", family="'Inter', sans-serif"),
            hovertemplate=(
                f"<b>{ph['phase']}</b> — {ph['time']}<br>"
                f"Score target: {ph['score']}<br>"
                f"· {actions_str}<extra></extra>"
            ),
        ))
    _apply_base(
        fig, height=200,
        xaxis=dict(**_AXIS_BASE, title="Implementation timeline (months)",
                   tickvals=[0, 6, 12, 18, 24, 30, 36],
                   ticktext=["0", "6m", "12m", "18m", "24m", "30m", "36m"]),
        yaxis=dict(**_AXIS_BASE, categoryorder="array",
                   categoryarray=[p["phase"] for p in reversed(phases)]),
        showlegend=False,
        margin=dict(l=72, r=20, t=12, b=44),
        barmode="stack",
    )
    return fig


# ============================================================================
# SCORE CONFIGS BUILDER
# ============================================================================

def _build_mro_configs(vendor_df, fact_pillar, fact_savings, portfolio_asmt,
                       hist_costs, cpm_df, live_vendor_score=True, signals=None):
    _empty = pd.DataFrame()
    fv  = vendor_df      if not vendor_df.empty      else _empty
    fp  = fact_pillar    if not fact_pillar.empty     else _empty
    fs  = fact_savings   if not fact_savings.empty    else _empty
    pa  = portfolio_asmt if not portfolio_asmt.empty  else _empty
    hc  = hist_costs     if not hist_costs.empty      else _empty
    cpm = cpm_df         if not cpm_df.empty          else _empty

    use_engine = not (fv.empty and fp.empty and fs.empty and pa.empty)
    computed = (
        compute_all_scores(fv, pa, fp, fs, hc, cpm, signals=signals)
        if use_engine else {
            "performance":  list(PERF_SCORES["chicago"]),
            "prediction":   list(PRED_SCORES["chicago"]),
            "prescription": list(PRES_SCORES["chicago"]),
        }
    )

    perf_scores = {k: list(v) for k, v in PERF_SCORES.items()}
    pred_scores = {k: list(v) for k, v in PRED_SCORES.items()}
    pres_scores = {k: list(v) for k, v in PRES_SCORES.items()}
    perf_scores["chicago"] = computed["performance"]
    pred_scores["chicago"] = computed["prediction"]
    pres_scores["chicago"] = computed["prescription"]

    if live_vendor_score and not fv.empty:
        perf_scores["chicago"][3] = derive_vendor_score_from_df(fv)

    overpayment_str = _fmt_kpi(compute_5yr_overpayment(fv)) if not fv.empty else "$748M"

    configs = [
        {"key": "performance",  "label": "📊 Performance",
         "dims": PERF_DIMS, "scores": perf_scores,
         "meta": _build_meta(PERF_DIMS, perf_scores["chicago"], list(PERF_SCORES["alpha"])),
         "sub_title": "MRO spend efficiency across 8 procurement dimensions"},
        {"key": "prediction",   "label": "🔮 Prediction",
         "dims": PRED_DIMS, "scores": pred_scores,
         "meta": _build_meta(PRED_DIMS, pred_scores["chicago"], list(PRED_SCORES["alpha"])),
         "sub_title": "Analytical forecasting & investment intelligence capability"},
        {"key": "prescription", "label": "💊 Prescription",
         "dims": PRES_DIMS, "scores": pres_scores,
         "meta": _build_meta(PRES_DIMS, pres_scores["chicago"], list(PRES_SCORES["alpha"])),
         "sub_title": "Recommended optimization levers for MRO spend reduction"},
    ]
    return configs, overpayment_str, fv


# ============================================================================
# STREAMLIT ENTRYPOINTS
# ============================================================================

def render_mro_single_tab(tab_key, vendor_df=None, live_vendor_score=True,
                           fact_pillar=None, fact_savings=None,
                           portfolio_asmt=None, hist_costs=None, cpm_df=None,
                           signals=None):
    _e = pd.DataFrame()
    configs, overpayment_str, fv = _build_mro_configs(
        vendor_df      = vendor_df      if vendor_df      is not None else _e,
        fact_pillar    = fact_pillar    if fact_pillar     is not None else _e,
        fact_savings   = fact_savings   if fact_savings    is not None else _e,
        portfolio_asmt = portfolio_asmt if portfolio_asmt  is not None else _e,
        hist_costs     = hist_costs     if hist_costs      is not None else _e,
        cpm_df         = cpm_df         if cpm_df          is not None else _e,
        live_vendor_score=live_vendor_score,
        signals=signals,
    )
    cfg = next((c for c in configs if c["key"] == tab_key), configs[0])
    _render_analysis_subtab(cfg, fv if not fv.empty else None, live_vendor_score, overpayment_str)


def render_mro_benchmark_tab(vendor_df=None, live_vendor_score=True,
                              fact_pillar=None, fact_savings=None,
                              portfolio_asmt=None, hist_costs=None, cpm_df=None):
    _e = pd.DataFrame()
    configs, overpayment_str, fv = _build_mro_configs(
        vendor_df      = vendor_df      if vendor_df      is not None else _e,
        fact_pillar    = fact_pillar    if fact_pillar     is not None else _e,
        fact_savings   = fact_savings   if fact_savings    is not None else _e,
        portfolio_asmt = portfolio_asmt if portfolio_asmt  is not None else _e,
        hist_costs     = hist_costs     if hist_costs      is not None else _e,
        cpm_df         = cpm_df         if cpm_df          is not None else _e,
        live_vendor_score=live_vendor_score,
    )
    sub_tabs = st.tabs([c["label"] for c in configs])
    for sub_tab, cfg in zip(sub_tabs, configs):
        with sub_tab:
            _render_analysis_subtab(cfg, fv if not fv.empty else None, live_vendor_score, overpayment_str)


# ============================================================================
# INTERNAL RENDER HELPER
# ============================================================================

def _render_analysis_subtab(cfg, vendor_df, live_vendor_score, overpayment_kpi="$748M"):
    meta    = cfg["meta"]
    dims    = cfg["dims"]
    scores  = cfg["scores"]
    tab_key = cfg["key"]

    risk_color = COLORS["red"] if meta["risk"] == "CRITICAL" else COLORS["yellow"]

    # ── 1. KPI cards ──────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(kcard("City of Chicago Score", meta["avg"],
            "out of 10.0 maximum", COLORS["purple"]), unsafe_allow_html=True)
    with k2:
        st.markdown(kcard("Gap vs Alpha Performer", meta["gap"],
            "GSA / Leading Municipal benchmark", COLORS["red"]), unsafe_allow_html=True)
    with k3:
        st.markdown(kcard("Risk Level", meta["risk"],
            "requires immediate action", risk_color), unsafe_allow_html=True)
    with k4:
        st.markdown(kcard("5-Yr MRO Overpayment", overpayment_kpi,
            "vs. market-equivalent pricing · facilities scope", COLORS["red"]),
            unsafe_allow_html=True)
    with k5:
        st.markdown(kcard("Top Priority", "→", meta["priority"], COLORS["green"]),
            unsafe_allow_html=True)

    # Live vendor score badge
    if tab_key == "performance" and live_vendor_score and vendor_df is not None and not vendor_df.empty:
        live_s    = scores["chicago"][3]
        delta     = live_s - PERF_SCORES["chicago"][3]
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        st.markdown(
            f"<div style='display:inline-block;"
            f"background:{_C['chicago_bg']};border:1px solid {_C['chicago_border']};"
            f"border-radius:6px;padding:5px 14px;font-size:11px;"
            f"color:{_C['chicago_accent']};margin-bottom:10px;'>"
            f"🔵 Live — Vendor dimension score: <b>{live_s:.1f}</b> "
            f"({delta_str} vs static baseline) · derived from active vendor_df filters"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── 2. Line chart ─────────────────────────────────────────────────────────
    section("Market Participant Comparison")
    st.markdown(
        f"<div style='font-size:11px;color:{_C['text_muted']};margin-bottom:8px;'>"
        f"{cfg['sub_title']}</div>",
        unsafe_allow_html=True,
    )
    chi_avg = round(sum(scores["chicago"]) / len(scores["chicago"]), 1)
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:12px;"
        f"background:linear-gradient(90deg,rgba(232,121,249,0.12),rgba(232,121,249,0.04));"
        f"border:1.5px solid {_C['chicago_border']};"
        f"border-radius:8px;padding:8px 16px;width:fit-content;'>"
        f"<span style='width:10px;height:10px;border-radius:50%;"
        f"background:{_C['chicago_dot']};display:inline-block;"
        f"box-shadow:0 0 6px {_C['chicago_dot']};'></span>"
        f"<span style='color:{_C['chicago_accent']};font-size:12px;font-weight:700;'>"
        f"City of Chicago — Observed Score: <strong>{chi_avg} avg</strong></span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(make_participant_line_chart(dims, scores), use_container_width=True)

    # ── 3. Score Matrix ───────────────────────────────────────────────────────
    section("Score Matrix")
    matrix_df = make_score_matrix_df(dims, scores)

    def _score_color(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if v >= 8.5:
            bg, fg = _C["score_hi_bg"],  _C["score_hi_fg"]
        elif v >= 7.0:
            bg, fg = _C["score_med_bg"], _C["score_med_fg"]
        elif v >= 5.5:
            bg, fg = _C["score_ok_bg"],  _C["score_ok_fg"]
        elif v >= 3.5:
            bg, fg = _C["score_lo_bg"],  _C["score_lo_fg"]
        else:
            bg, fg = _C["score_cr_bg"],  _C["score_cr_fg"]
        return f"background-color:{bg};color:{fg};font-weight:700;font-family:'Inter',sans-serif;"

    score_cols = dims + ["AVG"]
    styled = (
        matrix_df.style
        .applymap(_score_color, subset=score_cols)
        .format({col: "{:.1f}" for col in score_cols})
        .set_properties(
            subset=["Participant"],
            **{"font-weight": "600", "white-space": "nowrap",
               "color": _C["text"], "background-color": _C["card_alt"]},
        )
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── 4. Gap Analysis ───────────────────────────────────────────────────────
    section("Gap Analysis — City of Chicago vs. Alpha Best Performer")
    col_gap, col_info = st.columns([60, 40])

    with col_gap:
        st.plotly_chart(
            make_gap_bar_chart(dims, meta["ci"], meta["al"]),
            use_container_width=True,
        )

    with col_info:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        for i, dim in enumerate(dims):
            ci  = meta["ci"][i]
            al  = meta["al"][i]
            gap = al - ci
            pct = (ci / 10.0) * 100
            bar_color = (
                _C["bar_hi"]  if pct >= 70 else
                _C["bar_med"] if pct >= 50 else
                _C["bar_lo"]  if pct >= 35 else
                _C["bar_cr"]
            )
            st.markdown(
                f"<div style='margin-bottom:10px;background:{_C['card']};"
                f"border:1px solid {_C['border']};border-radius:8px;padding:10px 13px;"
                f"box-shadow:0 1px 3px rgba(30,64,175,0.06);'>"
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:center;margin-bottom:5px;'>"
                f"<span style='color:{_C['text_muted']};font-size:9px;"
                f"text-transform:uppercase;letter-spacing:1px;'>{dim}</span>"
                f"<span style='color:{_C['chicago_accent']};font-weight:800;"
                f"font-size:16px;font-family:\"Inter\",sans-serif;'>{ci:.1f}</span></div>"
                f"<div style='height:4px;background:rgba(30,64,175,0.10);"
                f"border-radius:2px;margin-bottom:5px;overflow:hidden;'>"
                f"<div style='height:100%;width:{pct:.0f}%;background:{bar_color};"
                f"border-radius:2px;'></div></div>"
                f"<div style='display:flex;justify-content:space-between;font-size:9px;'>"
                f"<span style='color:{_C['text_faint']};'>Gap to Alpha</span>"
                f"<span style='color:{_C['gap_hi']};font-weight:700;"
                f"font-family:\"Inter\",sans-serif;'>−{gap:.1f}</span></div></div>",
                unsafe_allow_html=True,
            )

    # ── 5. Roadmap ────────────────────────────────────────────────────────────
    roadmap_titles = {
        "performance":  "Performance Improvement Roadmap",
        "prediction":   "Predictive Analytics Capability Roadmap",
        "prescription": "Prescriptive Optimization Roadmap",
    }
    section(roadmap_titles.get(tab_key, "Roadmap"))
    st.plotly_chart(make_roadmap_chart(tab_key), use_container_width=True)

    phases     = ROADMAPS.get(tab_key, [])
    phase_cols = st.columns(len(phases))
    for col, ph in zip(phase_cols, phases):
        with col:
            actions_html = "".join(
                f"<div style='display:flex;gap:6px;margin-bottom:5px;"
                f"align-items:flex-start;'>"
                f"<span style='color:{ph['color']};font-size:12px;"
                f"flex-shrink:0;margin-top:1px;'>›</span>"
                f"<span style='color:{_C['text_sub']};font-size:11px;'>{a}</span></div>"
                for a in ph["actions"]
            )
            st.markdown(
                f"<div style='background:{_C['card']};"
                f"border:1px solid {ph['color']}55;border-radius:10px;"
                f"padding:14px;height:100%;"
                f"box-shadow:0 1px 4px rgba(30,64,175,0.07);'>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:8px;'>"
                f"<span style='color:{ph['color']};font-weight:700;font-size:13px;'>"
                f"{ph['phase']}</span>"
                f"<span style='color:{_C['text_muted']};font-size:10px;"
                f"font-family:\"Inter\",sans-serif;'>{ph['time']}</span></div>"
                f"<div style='color:{_C['text_sub']};font-size:11px;font-weight:700;"
                f"margin-bottom:9px;font-family:\"Inter\",sans-serif;'>"
                f"Score: {ph['score']}</div>"
                f"{actions_html}</div>",
                unsafe_allow_html=True,
            )
