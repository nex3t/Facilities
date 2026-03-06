"""
Facilities Portfolio Dashboard v2 — City of Chicago · DPS
Four top-level tabs:
    📊 Performance   — MRO procurement performance (8 dimensions)
    🔮 Prediction    — MRO predictive analytics capability (3 dimensions)
    💊 Prescription  — MRO optimization levers (3 dimensions)
    ⚡ Efficiency    — Pillar & AI savings + component breakdown

Usage:
    streamlit run facilities_dashboard_v2.py
"""

import logging
import os
import datetime
import streamlit as st
import pandas as pd

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from calculations import (
    recalc_savings, recalc_per_pillar,
    aggregate_yearly, aggregate_yearly_by_line, fmt,
)
from ui import inject_css, kcard, section, render_header, render_footer, render_sidebar
from charts import (
    make_efficiency_trajectory_chart,
    make_pillar_savings_lines_chart,
    make_component_treemap_chart,
    make_top5_component_bar_chart,
)
from data import (
    load_data,
    get_dept_assets,
    filter_fact_savings,
    filter_fact_pillar,
    join_building_info,
    filter_portfolio_asmt,
)
from config import (
    DATA_FILE,
    COLORS, PILLAR_DEFAULTS,
    UC2_COST_FACTOR, FC3_COST_FACTOR,
    BUDGET_LINE_LABELS,
)
from vendor_synth import CATEGORY_LABELS
from mro_benchmark import render_mro_single_tab

st.set_page_config(
    page_title="Facilities Portfolio — City of Chicago",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()


# ============================================================================
# DATA LOADING
# ============================================================================
try:
    (fact_savings, fact_pillar, dim_building, dim_year,
     portfolio_asmt, pillar_cfg_df, comp_df, hist_costs, vendor_df) = load_data()
except FileNotFoundError:
    st.error(f"❌ File not found: `{DATA_FILE}`")
    st.stop()
except ValueError as exc:
    st.error(f"❌ Data schema error: {exc}")
    st.stop()


# ============================================================================
# PILLAR CONFIG — apply Excel overrides
# ============================================================================
if len(pillar_cfg_df) > 0 and "Pillar_Key" in pillar_cfg_df.columns:
    for _, row in pillar_cfg_df.iterrows():
        key = str(row["Pillar_Key"])
        if key not in PILLAR_DEFAULTS:
            continue
        if "Rate_Default" in pillar_cfg_df.columns and pd.notna(row.get("Rate_Default")):
            PILLAR_DEFAULTS[key]["r_p"] = float(row["Rate_Default"])
        elif "Rate_r_p" in pillar_cfg_df.columns and pd.notna(row.get("Rate_r_p")):
            PILLAR_DEFAULTS[key]["r_p"] = float(row["Rate_r_p"])
        if "k_p" in pillar_cfg_df.columns and pd.notna(row.get("k_p")):
            PILLAR_DEFAULTS[key]["k"] = float(row["k_p"])
        if "x0_p" in pillar_cfg_df.columns and pd.notna(row.get("x0_p")):
            PILLAR_DEFAULTS[key]["x0"] = float(row["x0_p"])
        if "Rate_Low" in pillar_cfg_df.columns and pd.notna(row.get("Rate_Low")):
            PILLAR_DEFAULTS[key]["r_min"] = float(row["Rate_Low"])
        if "Rate_High" in pillar_cfg_df.columns and pd.notna(row.get("Rate_High")):
            PILLAR_DEFAULTS[key]["r_max"] = float(row["Rate_High"])


# ============================================================================
# SIDEBAR
# ============================================================================
selected_dept, sel_yrs, pillar_overrides, ai_rate, adoption_years, score_signals = render_sidebar(
    dim_building, fact_savings
)


# ============================================================================
# FILTER + RECALCULATE
# ============================================================================
dept_assets = get_dept_assets(dim_building, selected_dept)
fs     = filter_fact_savings(fact_savings, sel_yrs, dept_assets)
fp     = filter_fact_pillar(fact_pillar,   sel_yrs, dept_assets)
fs_bld = join_building_info(fs, dim_building)
pa_f   = filter_portfolio_asmt(portfolio_asmt, selected_dept)


@st.cache_data(show_spinner="Recalculating savings…")
def _cached_recalc(bld_df, overrides_tuple, ai_rate, adopt_yrs):
    return recalc_savings(None, bld_df, dict(overrides_tuple), ai_rate, adopt_yrs)

fs_recalc = _cached_recalc(
    fs_bld,
    tuple(sorted(pillar_overrides.items())),
    ai_rate,
    adoption_years,
)

if fs_recalc.empty:
    render_header(sel_yrs, selected_dept)
    st.warning("No data matches the current filters. Adjust the sidebar.")
    st.stop()

yearly  = aggregate_yearly(fs_recalc)
pil_yr  = recalc_per_pillar(fs_bld, pillar_overrides)
pil_yr["total"] = pil_yr["l1"]
last_yr = int(yearly["year"].max())
last    = yearly[yearly["year"] == last_yr].iloc[0]

n_yrs   = fs["year"].nunique()
_has_bl = "budget_line" in fs_recalc.columns


# ============================================================================
# COMPONENT COSTS (Facilities-only baseline)
# ============================================================================
_fs_fac = fs[fs["budget_line"] == "Facilities"] if _has_bl else fs
_fr_fac = fs_recalc[fs_recalc["budget_line"] == "Facilities"] if _has_bl else fs_recalc
_avg_ann_base = _fs_fac["baseline"].sum()       / max(n_yrs, 1)
_avg_ann_opt  = _fr_fac["final_override"].sum() / max(n_yrs, 1)

if len(comp_df) > 0:
    comp_costs = comp_df.copy()
    comp_costs["Base_Annual"] = 0.0
    comp_costs["Opt_Annual"]  = 0.0
    for _bl in comp_costs["budget_line"].unique():
        _bm = comp_costs["budget_line"] == _bl
        _fb = fs[fs["budget_line"] == _bl]       if _has_bl else fs
        _fr = fs_recalc[fs_recalc["budget_line"] == _bl] if _has_bl else fs_recalc
        _b  = _fb["baseline"].sum()       / max(n_yrs, 1)
        _o  = _fr["final_override"].sum() / max(n_yrs, 1)
        comp_costs.loc[_bm, "Base_Annual"] = comp_costs.loc[_bm, "Pct_of_Facility_Cost"] * _b
        comp_costs.loc[_bm, "Opt_Annual"]  = comp_costs.loc[_bm, "Pct_of_Facility_Cost"] * _o
    comp_costs["Savings"]     = comp_costs["Base_Annual"] - comp_costs["Opt_Annual"]
    comp_costs["Savings_Pct"] = comp_costs["Savings"] / comp_costs["Base_Annual"].replace(0, float("nan"))
    comp_costs["base_annual"] = comp_costs["Base_Annual"]
    comp_costs["savings"]     = comp_costs["Savings"]
    comp_costs["savings_pct"] = comp_costs["Savings_Pct"]
    try:
        cpm_df = pd.read_excel(DATA_FILE, sheet_name="Component_Pillar_Map")
    except Exception:
        cpm_df = pd.DataFrame(columns=["Component", "Domain", "Pillar_Key", "Weight"])
else:
    comp_costs = pd.DataFrame()
    cpm_df     = pd.DataFrame(columns=["Component", "Domain", "Pillar_Key", "Weight"])

# Facilities-only comp_costs for savings breakdown
comp_fac = (
    comp_costs[comp_costs["budget_line"] == "Facilities"].copy()
    if (len(comp_costs) > 0 and "budget_line" in comp_costs.columns)
    else comp_costs
)


# ============================================================================
# HEADER
# ============================================================================
render_header(sel_yrs, selected_dept)


# ============================================================================
# VENDOR DATA — filtered to sidebar year range
# ============================================================================
lo, hi = sel_yrs
vendor_df_filtered = vendor_df[vendor_df["year"].between(lo, hi)].copy()


# ============================================================================
# TABS
# ============================================================================
tab_perf, tab_pred, tab_pres, tab_eff = st.tabs([
    "📊 Performance",
    "🔮 Prediction",
    "💊 Prescription",
    "⚡ Efficiency",
])


# ── Performance ───────────────────────────────────────────────────────────────
with tab_perf:
    render_mro_single_tab(
        "performance",
        vendor_df      = vendor_df_filtered,
        live_vendor_score = True,
        fact_pillar    = fp,
        fact_savings   = fs_recalc,
        portfolio_asmt = pa_f,
        hist_costs     = hist_costs,
        cpm_df         = cpm_df,
        signals        = score_signals,
    )


# ── Prediction ────────────────────────────────────────────────────────────────
with tab_pred:
    render_mro_single_tab(
        "prediction",
        vendor_df      = vendor_df_filtered,
        live_vendor_score = True,
        fact_pillar    = fp,
        fact_savings   = fs_recalc,
        portfolio_asmt = pa_f,
        hist_costs     = hist_costs,
        cpm_df         = cpm_df,
        signals        = score_signals,
    )


# ── Prescription (MRO) ────────────────────────────────────────────────────────
with tab_pres:
    render_mro_single_tab(
        "prescription",
        vendor_df      = vendor_df_filtered,
        live_vendor_score = True,
        fact_pillar    = fp,
        fact_savings   = fs_recalc,
        portfolio_asmt = pa_f,
        hist_costs     = hist_costs,
        cpm_df         = cpm_df,
        signals        = score_signals,
    )


# ── Efficiency ────────────────────────────────────────────────────────────────
with tab_eff:
    # ── KPI cards ─────────────────────────────────────────────────────────────
    total_l1   = fs_recalc["l1_override"].sum()
    total_l2   = fs_recalc["l2_override"].sum()
    total_base = fs_recalc["baseline"].sum()
    total_sav  = total_l1 + total_l2
    pct_total  = total_sav / total_base if total_base > 0 else 0
    n_bld_t    = fs["asset_id"].nunique()

    # ── CAGR from historical costs (display only) ─────────────────────────────
    _cagr_str = "N/A"
    _cagr_yr0, _cagr_yr1 = 2021, 2024
    if len(hist_costs) >= 2 and "DFF_Total_Actual" in hist_costs.columns:
        _hc_s = hist_costs.sort_values("Year")
        _v0   = _hc_s[_hc_s["Year"] == _cagr_yr0]["DFF_Total_Actual"].values
        _v1   = _hc_s[_hc_s["Year"] == _cagr_yr1]["DFF_Total_Actual"].values
        if len(_v0) > 0 and len(_v1) > 0 and _v0[0] > 0:
            _cagr_str = f"{((_v1[0]/_v0[0])**(1/(_cagr_yr1-_cagr_yr0))-1):.1%}"

    # ── 6 KPI cards matching reference design ─────────────────────────────────
    st.markdown(f"""
    <style>
    .mro-kcard {{
        background:{COLORS['card']};border:1px solid {COLORS['border']};
        border-radius:12px;padding:16px 18px 14px 18px;min-height:108px;
    }}
    .mro-kcard .mk-icon  {{font-size:20px;margin-bottom:6px;display:block;}}
    .mro-kcard .mk-label {{font-size:9px;font-weight:700;letter-spacing:1.8px;
        text-transform:uppercase;color:{COLORS['muted']};margin-bottom:2px;}}
    .mro-kcard .mk-value {{font-size:24px;font-weight:800;
        font-family:'DM Mono',monospace;line-height:1.15;}}
    .mro-kcard .mk-sub   {{font-size:10px;color:{COLORS['muted']};margin-top:3px;}}
    </style>
    """, unsafe_allow_html=True)

    def _mkc(icon, label, value, sub, color):
        return f"""<div class='mro-kcard'>
            <span class='mk-icon'>{icon}</span>
            <div class='mk-label'>{label}</div>
            <div class='mk-value' style='color:{color};'>{value}</div>
            <div class='mk-sub'>{sub}</div>
        </div>"""

    _proj_yr   = int(yearly["year"].max())
    _opt_spend = fmt(float(yearly[yearly["year"] == _proj_yr]["final"].sum()))
    _base_spend = fmt(float(yearly[yearly["year"] == _proj_yr]["baseline"].sum()))
    _sav_5yr_l1 = fmt(fs_recalc[fs_recalc["year"].between(2026, min(2030, _proj_yr))]["l1_override"].sum())
    _sav_5yr_l2 = fmt(fs_recalc[fs_recalc["year"].between(2026, min(2030, _proj_yr))]["l2_override"].sum())
    _sav_5yr_tot = fmt(
        fs_recalc[fs_recalc["year"].between(2026, min(2030, _proj_yr))]["l1_override"].sum() +
        fs_recalc[fs_recalc["year"].between(2026, min(2030, _proj_yr))]["l2_override"].sum()
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(_mkc("📈", f"Chicago Spend CAGR {_cagr_yr0}–{_cagr_yr1}",
            _cagr_str, "Current trajectory — no intervention", COLORS["red"]),
            unsafe_allow_html=True)
    with k2:
        st.markdown(_mkc("⚙️", f"{_proj_yr} Baseline Spend",
            _base_spend, "No optimization applied", COLORS["muted"]),
            unsafe_allow_html=True)
    with k3:
        st.markdown(_mkc("🎯", f"{_proj_yr} Optimized Spend",
            _opt_spend, "With L1 pillars + L2 AI applied", COLORS["cyan"]),
            unsafe_allow_html=True)
    with k4:
        st.markdown(_mkc("💰", "5-Yr L1 Savings (2026–2030)",
            _sav_5yr_l1, "Pillar optimization savings", COLORS["orange"]),
            unsafe_allow_html=True)
    with k5:
        st.markdown(_mkc("💎", "5-Yr L2 Savings (2026–2030)",
            _sav_5yr_l2, "AI amplification savings", COLORS["purple"]),
            unsafe_allow_html=True)
    with k6:
        st.markdown(_mkc("🏛️", "5-Yr Combined Savings",
            _sav_5yr_tot, "L1 + L2 vs. baseline trend", COLORS["green"]),
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    section("Cost Trajectory — Historical Actuals + Optimization Scenarios")
    st.plotly_chart(
        make_efficiency_trajectory_chart(yearly, hist_costs),
        use_container_width=True,
        key="eff_trajectory",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Pillar savings trajectory | Top 5 components ───────────────────
    col_pil, col_top = st.columns([1, 1])

    with col_pil:
        section("Savings by Pillar vs AI — Annual Trajectory")
        st.plotly_chart(
            make_pillar_savings_lines_chart(pil_yr, yearly),
            use_container_width=True,
            key="eff_pillar_lines",
        )

    with col_top:
        if not comp_fac.empty:
            section("Top 5 Components — Annual Savings")
            cpm_weights = {
                (r["Component"], r["Pillar_Key"]): float(r["Weight"])
                for _, r in cpm_df.iterrows()
            }
            top5 = comp_fac.nlargest(5, "Savings")[["Component", "Domain", "Savings", "Base_Annual"]].copy()
            stacked = []
            for _, crow in top5.iterrows():
                for p_key, p_rate in pillar_overrides.items():
                    w = cpm_weights.get((crow["Component"], p_key), 0.0)
                    stacked.append({"Component": crow["Component"], "Pillar": p_key,
                                    "Savings": crow["Base_Annual"] * w * p_rate})
            df_stack = pd.DataFrame(stacked)
            short_names = {
                "HVAC Systems":                     "HVAC",
                "Electrical Systems & Distribution": "Electrical",
                "Roof & Building Envelope":          "Roof & Envelope",
                "Energy Management & Utilities":     "Energy Mgmt",
                "Plumbing & Water Systems":          "Plumbing",
            }
            comp_order    = df_stack.groupby("Component")["Savings"].sum().sort_values(ascending=False).index.tolist()
            short_order   = [short_names.get(c, c[:14]) for c in comp_order]
            pillar_totals = df_stack.groupby("Pillar")["Savings"].sum().sort_values(ascending=False)
            sorted_pillars = [p for p in pillar_totals.index if p in PILLAR_DEFAULTS]
            st.plotly_chart(
                make_top5_component_bar_chart(df_stack, short_order, sorted_pillars, short_names),
                use_container_width=True,
                key="eff_top5",
            )
        else:
            st.info("Components data not available.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Component savings table ────────────────────────────────────────────────
    if not comp_fac.empty:
        section("Component Savings Detail")
        tbl_comp = comp_fac[["Component", "Domain", "Base_Annual", "Opt_Annual", "Savings", "Savings_Pct"]].copy()
        tbl_comp = tbl_comp.sort_values("Savings", ascending=False)
        st.dataframe(
            pd.DataFrame({
                "Component":    tbl_comp["Component"],
                "Domain":       tbl_comp["Domain"],
                "Base Annual":  tbl_comp["Base_Annual"].apply(fmt),
                "Optimized":    tbl_comp["Opt_Annual"].apply(fmt),
                "Savings":      tbl_comp["Savings"].apply(fmt),
                "Savings %":    tbl_comp["Savings_Pct"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—"),
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AI-Adjusted Use Case table ─────────────────────────────────────────────
    section("AI-Adjusted Use Case Cost Factors")
    last_yr_rc    = fs_recalc[fs_recalc["year"] == last_yr]
    avg_cost      = total_base / max(n_bld_t, 1) / max(n_yrs, 1)

    n_uc1 = max(last_yr_rc[last_yr_rc["Use_Case"] == "UC1"]["asset_id"].nunique(), 1)
    n_uc2 = max(last_yr_rc[last_yr_rc["Use_Case"] == "UC2"]["asset_id"].nunique(), 1)
    n_fc  = max(last_yr_rc[~last_yr_rc["Use_Case"].isin(["UC1", "UC2"])]["asset_id"].nunique(), 1)

    port_uc1   = avg_cost
    port_uc2   = avg_cost * UC2_COST_FACTOR
    port_fc    = avg_cost * FC3_COST_FACTOR
    sil_total  = avg_cost * (n_uc1 + n_uc2 + n_fc)
    port_total = port_uc1 * n_uc1 + port_uc2 * n_uc2 + port_fc * n_fc
    port_sav   = (1 - port_total / sil_total) * 100 if sil_total > 0 else 0

    ai_uc1        = last_yr_rc[last_yr_rc["Use_Case"] == "UC1"]["final_override"].mean() if n_uc1 > 0 else 0
    ai_uc2        = last_yr_rc[last_yr_rc["Use_Case"] == "UC2"]["final_override"].mean() if n_uc2 > 0 else 0
    ai_fc         = last_yr_rc[~last_yr_rc["Use_Case"].isin(["UC1", "UC2"])]["final_override"].mean() if n_fc > 0 else 0
    ai_total      = last_yr_rc["final_override"].sum()
    base_total_ly = last_yr_rc["baseline"].sum()
    ai_sav        = (1 - ai_total / base_total_ly) * 100 if base_total_ly > 0 else 0

    uc_html = f"""
    <div style='background:rgba(255,255,255,0.02);border:1px solid {COLORS["border"]};
         border-radius:10px;overflow:hidden;font-size:12px;'>
        <table style='width:100%;border-collapse:collapse;'>
            <thead>
                <tr style='border-bottom:1px solid {COLORS["border"]};background:rgba(255,255,255,0.02);'>
                    <th style='text-align:left;padding:10px 14px;color:#64748b;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;'>Scenario</th>
                    <th style='text-align:right;padding:10px 10px;color:#64748b;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;'>UC1</th>
                    <th style='text-align:right;padding:10px 10px;color:#64748b;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;'>UC2</th>
                    <th style='text-align:right;padding:10px 10px;color:#64748b;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;'>FC3+</th>
                    <th style='text-align:right;padding:10px 10px;color:#64748b;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;'>Total</th>
                    <th style='text-align:right;padding:10px 14px;color:#64748b;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;'>Savings</th>
                </tr>
            </thead>
            <tbody>
                <tr style='border-bottom:1px solid rgba(255,255,255,0.03);'>
                    <td style='padding:9px 14px;color:{COLORS["red"]};font-weight:700;'>Siloed Baseline</td>
                    <td style='padding:9px 10px;text-align:right;color:#94a3b8;font-family:"DM Mono",monospace;'>{fmt(avg_cost)}</td>
                    <td style='padding:9px 10px;text-align:right;color:#94a3b8;font-family:"DM Mono",monospace;'>{fmt(avg_cost)}</td>
                    <td style='padding:9px 10px;text-align:right;color:#94a3b8;font-family:"DM Mono",monospace;'>{fmt(avg_cost)}</td>
                    <td style='padding:9px 10px;text-align:right;color:#94a3b8;font-family:"DM Mono",monospace;font-weight:700;'>{fmt(sil_total)}</td>
                    <td style='padding:9px 14px;text-align:right;color:#475569;'>—</td>
                </tr>
                <tr style='border-bottom:1px solid rgba(255,255,255,0.03);'>
                    <td style='padding:9px 14px;color:{COLORS["orange"]};font-weight:700;'>Portfolio Opt. (No AI)</td>
                    <td style='padding:9px 10px;text-align:right;color:#94a3b8;font-family:"DM Mono",monospace;'>{fmt(port_uc1)}</td>
                    <td style='padding:9px 10px;text-align:right;color:#94a3b8;font-family:"DM Mono",monospace;'>{fmt(port_uc2)}</td>
                    <td style='padding:9px 10px;text-align:right;color:#94a3b8;font-family:"DM Mono",monospace;'>{fmt(port_fc)}</td>
                    <td style='padding:9px 10px;text-align:right;color:#94a3b8;font-family:"DM Mono",monospace;font-weight:700;'>{fmt(port_total)}</td>
                    <td style='padding:9px 14px;text-align:right;color:{COLORS["green"]};font-weight:700;'>{port_sav:.1f}%</td>
                </tr>
                <tr>
                    <td style='padding:9px 14px;color:{COLORS["green"]};font-weight:700;'>AI-Enabled Optimization</td>
                    <td style='padding:9px 10px;text-align:right;color:#94a3b8;font-family:"DM Mono",monospace;'>{fmt(ai_uc1)}</td>
                    <td style='padding:9px 10px;text-align:right;color:#94a3b8;font-family:"DM Mono",monospace;'>{fmt(ai_uc2)}</td>
                    <td style='padding:9px 10px;text-align:right;color:#94a3b8;font-family:"DM Mono",monospace;'>{fmt(ai_fc)}</td>
                    <td style='padding:9px 10px;text-align:right;color:#94a3b8;font-family:"DM Mono",monospace;font-weight:700;'>{fmt(ai_total)}</td>
                    <td style='padding:9px 14px;text-align:right;color:{COLORS["green"]};font-weight:700;'>{ai_sav:.1f}%</td>
                </tr>
            </tbody>
        </table>
    </div>"""
    st.markdown(uc_html, unsafe_allow_html=True)


# ============================================================================
# FOOTER
# ============================================================================
l1_blended = fs_recalc["l1_override"].sum() / max(fs_recalc["baseline"].sum(), 1)
try:
    _mtime    = os.path.getmtime(DATA_FILE)
    data_mtime = datetime.datetime.fromtimestamp(_mtime).strftime("%Y-%m-%d %H:%M")
except OSError:
    data_mtime = None

render_footer(l1_blended, ai_rate, adoption_years, str(DATA_FILE), data_mtime)
