"""
ui.py — Facilities Portfolio Optimization Dashboard
City of Chicago — Department of Procurement Services

Streamlit UI components: global CSS injection, KPI card HTML,
section title helper, page header, and sidebar controls.

Signals upgrade:
- render_sidebar now optionally returns score_signals (dict | None) that can
  be used by score_engine to compute Payments/Component scores from data.
"""

import html as _html
import pandas as pd
import streamlit as st

from config import COLORS, PILLAR_DEFAULTS, PILLAR_LABELS


# ============================================================================
# CSS
# ============================================================================

def inject_css() -> None:
    # Small f-string only for CSS variables (safe: braces doubled)
    css_vars = f"""
<style>
:root {{
  --q16-bg: {COLORS['bg']};
  --q16-sidebar: {COLORS['sidebar']};
  --q16-card: {COLORS['card']};
  --q16-border: {COLORS['border']};
  --q16-text: {COLORS['text']};
  --q16-muted: {COLORS['muted']};
  --q16-navy: {COLORS['navy']};

  --q16-primary: {COLORS.get('cyan', '#2563eb')};
  --q16-success: {COLORS.get('green', '#059669')};
  --q16-warning: {COLORS.get('orange', '#d97706')};
  --q16-danger:  {COLORS.get('red', '#dc2626')};
}}
"""

    # Rest is plain string (no f-string parsing -> braces are safe)
    css_rest = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp {
    background-color: var(--q16-bg);
    color: var(--q16-text);
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

/* Header */
header[data-testid="stHeader"]{
    background-color: var(--q16-bg) !important;
    border-bottom: 1px solid var(--q16-border) !important;
}

/* ── Sidebar base ── */
section[data-testid="stSidebar"]{
    background-color: #d8e2ee !important;
    border-right: 2px solid rgba(30,58,95,0.18) !important;
    padding-top: 14px;
}
section[data-testid="stSidebar"] > div{
    padding-top: 12px !important;
}

/* ── ALL text inside sidebar → dark, no exceptions ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #0f172a !important;
}

/* ── Markdown headings (## ⚙️ Layer 1 etc.) ── */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #1e3a5f !important;
    font-size: 13px !important;
    font-weight: 800 !important;
    letter-spacing: 0.3px !important;
    margin: 8px 0 4px 0 !important;
}

/* ── Checkbox label ── */
section[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] p,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] span {
    color: #0f172a !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}

/* ── Selectbox: selected value + dropdown text ── */
section[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-baseweb="select"] div {
    color: #0f172a !important;
}

/* ── Slider: label + min/max values + current value ── */
section[data-testid="stSidebar"] [data-testid="stSlider"] label,
section[data-testid="stSidebar"] [data-testid="stSlider"] p,
section[data-testid="stSidebar"] [data-testid="stSlider"] span,
section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMax"],
section[data-testid="stSidebar"] [data-baseweb="slider"] [aria-valuetext] {
    color: #334155 !important;
    font-size: 11px !important;
}

/* ── Number input text ── */
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
section[data-testid="stSidebar"] [data-testid="stNumberInput"] label,
section[data-testid="stSidebar"] [data-testid="stNumberInput"] p {
    color: #0f172a !important;
}

/* ── Expander: summary header + body text ── */
section[data-testid="stSidebar"] details summary,
section[data-testid="stSidebar"] details summary p,
section[data-testid="stSidebar"] details summary span,
section[data-testid="stSidebar"] details summary svg {
    color: #1e3a5f !important;
    font-weight: 700 !important;
    font-size: 12px !important;
}
section[data-testid="stSidebar"] details > div p,
section[data-testid="stSidebar"] details > div span,
section[data-testid="stSidebar"] details > div label {
    color: #0f172a !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
    gap: 4px;
    background-color: rgba(255,255,255,0.80);
    border-radius: 10px;
    padding: 6px;
    border: 1px solid var(--q16-border);
}

.stTabs [data-baseweb="tab"]{
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 700;
    font-size: 13px;
    color: var(--q16-muted);
}

.stTabs [aria-selected="true"]{
    background-color: var(--q16-navy) !important;
    color: #ffffff !important;
}

/* KPI card (generic) */
.kcard{
    background: var(--q16-card);
    border: 1px solid var(--q16-border);
    border-radius: 12px;
    padding: 10px 16px;
    box-shadow: 0 8px 22px rgba(15,23,42,0.08);
}

h1, h2, h3, h4{
    color: var(--q16-text);
    font-weight: 800;
}

.section-title{
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: var(--q16-muted);
    margin-bottom: 10px;
    margin-top: 6px;
}

/* ===========================
   Sidebar — Filter Panel Card
   =========================== */
.q16-filter-card{
    background: #ffffff;
    border: 1px solid rgba(30,58,95,0.20);
    border-radius: 14px;
    padding: 14px 14px 10px 14px;
    box-shadow: 0 4px 18px rgba(15,23,42,0.10);
}

.q16-filter-title{
    display:flex;
    align-items:center;
    gap:10px;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #1e3a5f;
    margin: 0 0 8px 0;
}

.q16-filter-subtitle{
    font-size: 11px;
    color: #475569;
    margin: 2px 0 12px 0;
}

.q16-filter-divider{
    height: 1px;
    background: rgba(30,58,95,0.15);
    margin: 14px 0;
}

/* ── Selectbox container ── */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div{
    background: #ffffff !important;
    border: 1px solid rgba(100,116,139,0.45) !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 6px rgba(15,23,42,0.06) !important;
    color: #0f172a !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within{
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.20), 0 1px 6px rgba(15,23,42,0.06) !important;
    border-color: #2563eb !important;
}

/* ── Number + Text inputs ── */
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stTextInput input{
    background: #ffffff !important;
    border: 1px solid rgba(100,116,139,0.45) !important;
    border-radius: 10px !important;
    color: #0f172a !important;
    box-shadow: 0 1px 4px rgba(15,23,42,0.05) !important;
}
section[data-testid="stSidebar"] .stNumberInput input:focus,
section[data-testid="stSidebar"] .stTextInput input:focus{
    outline: none !important;
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.18) !important;
}

/* ── Slider track + thumb ── */
section[data-testid="stSidebar"] [data-testid="stSlider"] > div{ padding-top: 2px; }
section[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSliderTrack"]{
    height: 5px !important;
    border-radius: 999px !important;
    background: rgba(30,58,95,0.15) !important;
}
section[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"]{
    width: 18px !important;
    height: 18px !important;
    border-radius: 999px !important;
    background: #1e3a5f !important;
    border: 2px solid #ffffff !important;
    box-shadow: 0 3px 10px rgba(30,58,95,0.35) !important;
}

/* ── Expander card ── */
section[data-testid="stSidebar"] details{
    background: #f1f5f9;
    border: 1px solid rgba(30,58,95,0.15);
    border-radius: 10px;
    padding: 6px 10px;
    margin-bottom: 4px;
}

/* ── Reset button ── */
section[data-testid="stSidebar"] .stButton button{
    width: 100%;
    border-radius: 10px;
    border: none !important;
    background: linear-gradient(160deg, #1e3a8a, #1e3a5f) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 12px !important;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 14px rgba(30,58,142,0.30) !important;
    padding: 8px 0 !important;
}
section[data-testid="stSidebar"] .stButton button:hover{
    background: linear-gradient(160deg, #2563eb, #1e3a8a) !important;
    box-shadow: 0 6px 18px rgba(37,99,235,0.35) !important;
}
section[data-testid="stSidebar"] .stButton button:active{ transform: translateY(1px); }

/* Plotly container card */
div[data-testid="stPlotlyChart"]{
    background: #ffffff;
    border: 1px solid var(--q16-border);
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0 10px 26px rgba(15,23,42,0.08);
}
.stDataFrame{ border-radius: 10px; overflow: hidden; }

</style>
"""
    st.markdown(css_vars + css_rest, unsafe_allow_html=True)


# ============================================================================
# COMPONENTS
# ============================================================================

def kcard(label: str, value: str, sub: str = "", color: str = "#059669") -> str:
    e_label = _html.escape(str(label))
    e_value = _html.escape(str(value))
    e_sub   = _html.escape(str(sub))
    return f"""
    <div class='kcard' style='border-top: 2px solid {color};'>
        <div style='font-size:9px;letter-spacing:2px;color:{COLORS['muted']};font-weight:800;
             text-transform:uppercase;margin-bottom:3px;'>{e_label}</div>
        <div style='font-size:20px;font-weight:900;color:{color};
             font-family:"DM Mono",monospace;line-height:1.1;'>{e_value}</div>
        <div style='font-size:10px;color:{COLORS['muted']};margin-top:2px;'>{e_sub}</div>
    </div>"""


def section(title: str) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


def render_header(sel_yrs: tuple[int, int], selected_dept: str) -> None:
    safe_dept  = _html.escape(selected_dept)
    dept_label = f"· {safe_dept}" if selected_dept != "All" else "· All Departments"
    st.markdown(f"""
<div style='display:flex;align-items:center;gap:16px;margin-bottom:16px;padding-bottom:14px;
     border-bottom:1px solid {COLORS["border"]};'>
    <div style='width:44px;height:44px;border-radius:10px;
         background:linear-gradient(135deg,{COLORS["green"]},{COLORS["cyan"]});
         display:flex;align-items:center;justify-content:center;
         font-size:16px;font-weight:900;color:{COLORS["bg"]};flex-shrink:0;'>DPS</div>
    <div>
        <div style='font-size:18px;font-weight:900;letter-spacing:-0.5px;color:{COLORS["text"]};'>
            Facilities Portfolio Optimization Model</div>
        <div style='font-size:11px;color:{COLORS['muted']};margin-top:1px;'>
            City of Chicago · Department of Procurement Services · {sel_yrs[0]}–{sel_yrs[1]}
            {dept_label}</div>
    </div>
</div>
""", unsafe_allow_html=True)


def render_footer(
    l1_blended: float,
    ai_rate: float,
    adoption_years: int,
    data_file: str,
    data_mtime: str | None = None,
) -> None:
    freshness = f" · Data updated: {data_mtime}" if data_mtime else ""
    st.markdown(f"""
<div style='text-align:center;padding:28px 0 10px;color:{COLORS["muted"]};font-size:11px;
     border-top:1px solid {COLORS["border"]};margin-top:32px;'>
    Facilities Portfolio Optimization Model v5 · Star Schema Architecture ·
    L1 Pillars: {l1_blended:.1%} · L2 AI: {ai_rate:.1%} · AI Adoption: {adoption_years} yrs ·
    Source: {data_file}{freshness}
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar(
    dim_building: pd.DataFrame,
    fact_savings: pd.DataFrame,
) -> tuple[str, tuple[int, int], dict[str, float], float, int, dict | None]:

    st.sidebar.markdown("## 🎛️ Filters")
    st.sidebar.markdown(
        "<div class='q16-filter-card'>"
        "<div class='q16-filter-title'>🎛️ Filters</div>"
        "<div class='q16-filter-subtitle'>Set scope and scenario parameters</div>",
        unsafe_allow_html=True,
    )

    departments   = ["All"] + sorted(dim_building["Department"].dropna().unique().tolist())
    selected_dept = st.sidebar.selectbox("Department", departments)

    min_yr = int(fact_savings["year"].min())
    max_yr = int(fact_savings["year"].max())
    sel_yrs = st.sidebar.slider("Year Range", min_yr, max_yr, (min_yr, max_yr))

    # ── Layer 1 — Pillar controls ─────────────────────────────────────────────
    st.sidebar.markdown("<div class='q16-filter-divider'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("## ⚙️ Layer 1 — Pillar Rates")
    pillars_enabled = st.sidebar.checkbox("Enable Pillar Savings", value=True)

    pillar_overrides: dict[str, float] = {}
    if pillars_enabled:
        for key, cfg_p in PILLAR_DEFAULTS.items():
            with st.sidebar.expander(PILLAR_LABELS[key], expanded=False):
                enabled = st.checkbox("Enable", value=True, key=f"p_en_{key}")
                if enabled:
                    val = st.slider(
                        "Rate (%)",
                        int(cfg_p["r_min"] * 100),
                        int(cfg_p["r_max"] * 100),
                        int(cfg_p["r_p"]  * 100),
                        key=f"p_rate_{key}",
                    )
                    pillar_overrides[key] = val / 100
                else:
                    pillar_overrides[key] = 0.0
    else:
        pillar_overrides = {key: 0.0 for key in PILLAR_DEFAULTS}

    # ── Layer 2 — AI controls ─────────────────────────────────────────────────
    st.sidebar.markdown("<div class='q16-filter-divider'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("## 🤖 Layer 2 — AI")
    ai_enabled     = st.sidebar.checkbox("Enable AI Savings", value=True)
    ai_rate        = st.sidebar.slider("AI Blended Rate (%)", 0, 25, 10) / 100 if ai_enabled else 0.0
    adoption_years = st.sidebar.slider("AI Adoption (years)", 1, 15, 7)

    # ── Advanced Signals (optional) — for data-driven Payments/Component ───────
    st.sidebar.markdown("<div class='q16-filter-divider'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("## 🔬 Advanced Signals (optional)")
    use_signals = st.sidebar.checkbox("Use data-driven Payments & Component", value=False)

    score_signals: dict | None
    if use_signals:
        score_signals = {}
        with st.sidebar.expander("Payments signals", expanded=False):
            score_signals["invoice_cycle_days"] = float(
                st.sidebar.number_input("Invoice cycle time (days)", 1.0, 60.0, 14.0, 1.0)
            )
            score_signals["touchless_rate"] = float(
                st.sidebar.slider("Touchless processing rate", 0.0, 1.0, 0.35, 0.05)
            )
            score_signals["discount_capture_rate"] = float(
                st.sidebar.slider("Discount capture rate", 0.0, 0.20, 0.05, 0.01)
            )

        with st.sidebar.expander("Component signals", expanded=False):
            score_signals["component_visibility"] = float(
                st.sidebar.slider("Component/SKU visibility", 0.0, 1.0, 0.20, 0.05)
            )
            score_signals["catalog_coverage"] = float(
                st.sidebar.slider("Catalog coverage", 0.0, 1.0, 0.30, 0.05)
            )
            score_signals["inventory_turnover"] = float(
                st.sidebar.number_input("Inventory turnover (turns/year)", 0.0, 12.0, 1.5, 0.5)
            )
    else:
        score_signals = None

    st.sidebar.markdown("<div class='q16-filter-divider'></div>", unsafe_allow_html=True)
    if st.sidebar.button("↺ Reset"):
        st.rerun()

    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    return selected_dept, sel_yrs, pillar_overrides, ai_rate, adoption_years, score_signals
