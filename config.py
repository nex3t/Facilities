"""
config.py — Facilities Portfolio Optimization Dashboard
City of Chicago — Department of Public Services

All application-wide constants: data source, theme colors, pillar defaults,
chart styling, and business-logic ratios. Import from here instead of
hardcoding values in other modules.
"""

from pathlib import Path

# ============================================================================
# DATA SOURCE
# ============================================================================
DATA_FILE = Path(__file__).parent / "asset_model_outputs.xlsx"

# ============================================================================
# BUSINESS LOGIC RATIOS
# (previously hardcoded inline — centralised here for traceability)
# ============================================================================

# Domain split: share of building costs attributed to each domain
BS_DOMAIN_RATIO = 0.55   # Building Services domain
FO_DOMAIN_RATIO = 0.45   # Facilities Operations domain

# Use-case cost factors (relative to baseline cost)
UC1_COST_FACTOR = 0.00   # UC1 — no incremental AI cost
UC2_COST_FACTOR = 0.50   # UC2 — moderate AI uplift
FC3_COST_FACTOR = 0.164  # FC3+ — advanced AI / full portfolio

# Sankey diagram link opacity (rgba alpha channels)
SANKEY_LINK_OPACITY_HIGH = "0.88"
SANKEY_LINK_OPACITY_LOW  = "0.18"

# ============================================================================
# CHART LAYOUT DEFAULTS
# ============================================================================
CHART_BASE_HEIGHT     = 380   # minimum chart height in pixels
CHART_HEIGHT_PER_ROW  = 28    # additional pixels per data row (bar charts)

# ============================================================================
# THEME — COLORS
# ============================================================================
COLORS = {
    # Light institutional — blue/grey base
    # NOTE: tuned for stronger contrast + clearer panel separation.
    "bg":      "#eef2f6",
    "sidebar": "#e5ebf1",
    "card":    "#ffffff",
    "border":  "rgba(100,116,139,0.35)",
    "text":    "#0f172a",
    "muted":   "#64748b",
    "navy":    "#1e3a5f",
    # Accents — from reference chart palette
    "green":   "#1a7a4a",   # dark green (Maximum Performance)
    "orange":  "#d97706",   # amber/gold (Benchmark)
    "cyan":    "#3b82f6",   # blue (Alpha / Digital)
    "red":     "#dc2626",   # red (Late Adopters)
    "yellow":  "#d97706",   # amber (same as orange)
    "purple":  "#7c3aed",   # violet (Top Performance)
    "indigo":  "#1e3a8a",   # deep blue
    "lime":    "#c026d3",   # magenta (Chicago observed)
}

# ============================================================================
# THEME — PILLAR COLORS & LABELS
# ============================================================================
PILLAR_COLORS = {
    "Work_Modernization":   "#1a7a4a",  # dark green
    "Demand_Management":    "#3b82f6",  # blue
    "Asset_Management":     "#7c3aed",  # violet
    "Vendor_Management":    "#d97706",  # amber/gold
    "Payment_Management":   "#dc2626",  # red
    "Early_Pay_Management": "#c026d3",  # magenta
}

PILLAR_LABELS = {
    "Work_Modernization":   "Work Modernization",
    "Demand_Management":    "Demand Management",
    "Asset_Management":     "Asset Management",
    "Vendor_Management":    "Vendor Management",
    "Payment_Management":   "Payment Management",
    "Early_Pay_Management": "Early Pay Mgmt",
}

# ============================================================================
# PLOTLY BASE STYLES
# ============================================================================
_AXIS_BASE = dict(
    gridcolor="rgba(100,116,139,0.18)",
    zerolinecolor="rgba(100,116,139,0.26)",
    linecolor="rgba(100,116,139,0.30)",
    tickfont=dict(color="#334155", size=11, family="'Inter', sans-serif"),
    title_font=dict(color="#1e293b", size=12, family="'Inter', sans-serif"),
    tickcolor="#475569",
)

_LEGEND_BASE = dict(
    bgcolor="rgba(255,255,255,0.95)",
    bordercolor="rgba(100,116,139,0.30)",
    borderwidth=1,
    font=dict(size=10, color="#334155"),
    orientation="h",
    y=1.08,
    x=0,
    xanchor="left",
    yanchor="bottom",
    itemsizing="constant",
    tracegroupgap=4,
)

_HOVER_BASE = dict(
    bgcolor="#ffffff",
    bordercolor="rgba(100,116,139,0.35)",
    font=dict(color="#0f172a", size=12),
)

_MARGIN_BASE = dict(l=58, r=22, t=52, b=48)

PLOTLY_BASE = dict(
    # Render figures on white “cards” so they read as contained panels.
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#334155", family="'Inter', sans-serif", size=11),
    hoverlabel=_HOVER_BASE,
)

# ============================================================================
# PILLAR DEFAULTS
# Each pillar entry: name, default rate (r_p), min/max bounds, sigmoid params.
# ============================================================================
PILLAR_DEFAULTS: dict[str, dict] = {
    "Work_Modernization": {
        "name": "Work Modernization",
        "r_p": 0.12, "r_min": 0.02, "r_max": 0.25,
        "k": 1.5, "x0": 3.0,
    },
    "Demand_Management": {
        "name": "Demand Management",
        "r_p": 0.07, "r_min": 0.01, "r_max": 0.20,
        "k": 1.2, "x0": 4.0,
    },
    "Asset_Management": {
        "name": "Asset Management",
        "r_p": 0.10, "r_min": 0.02, "r_max": 0.25,
        "k": 1.0, "x0": 5.0,
    },
    "Vendor_Management": {
        "name": "Vendor Management",
        "r_p": 0.04, "r_min": 0.01, "r_max": 0.15,
        "k": 0.8, "x0": 4.0,
    },
    "Payment_Management": {
        "name": "Payment Management",
        "r_p": 0.025, "r_min": 0.01, "r_max": 0.12,
        "k": 1.8, "x0": 4.0,   # matures 2028 — requires modernised procurement first
    },
    "Early_Pay_Management": {
        "name": "Early Pay Management",
        "r_p": 0.005, "r_min": 0.00, "r_max": 0.08,
        "k": 2.0, "x0": 5.0,   # matures 2029 — depends on established payment processes
    },
}

# Pillars whose L1 savings feed the AI (L2) amplification base.
# Financial / a-posteriori pillars (Vendor, Payment, Early Pay) reduce costs
# but are not amplified by AI — their savings are policy-driven, not process-driven.
OPERATIONAL_PILLARS: frozenset[str] = frozenset({
    "Work_Modernization",
    "Demand_Management",
    "Asset_Management",
})

# ============================================================================
# BUDGET LINE COLORS & LABELS  (v7 — Equipment + Fleet)
# ============================================================================
BUDGET_LINE_COLORS: dict[str, str] = {
    "Facilities": "#1a7a4a",   # dark green
    "Equipment":  "#3b82f6",   # blue
    "Fleet":      "#7c3aed",   # violet
}

BUDGET_LINE_FILL_COLORS: dict[str, str] = {
    "Facilities": "rgba(26,122,74,0.12)",
    "Equipment":  "rgba(59,130,246,0.12)",
    "Fleet":      "rgba(124,58,237,0.12)",
}

BUDGET_LINE_LABELS: dict[str, str] = {
    "Facilities": "Facilities O&M",
    "Equipment":  "Equipment",
    "Fleet":      "Fleet",
}

# ============================================================================
# Use-case scaling for AI scenario analysis
UC_SCALE: dict[str, float] = {
    "UC1": UC1_COST_FACTOR,
    "UC2": UC2_COST_FACTOR,
    "UC3": 1.00,
}
