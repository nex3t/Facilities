"""
data.py — Facilities Portfolio Optimization Dashboard
City of Chicago — Department of Public Services

Data loading and filtering functions.
All functions are pure (no Streamlit calls) except load_data(), which uses
@st.cache_data and must be called from the main app module.

v6 changes vs v3:
  - Pillar_Config sheet now has columns: Pillar_Key, Pillar_Name, Rate_Low,
    Rate_High, Rate_Default, k_p, x0_p, Enabled, Description
    (previously only Pillar_Key, Rate_r_p)
  - dim_building has additional vendor and Q16 classification columns (transparent)
  - New optional sheets: fact_vendor, ACFR_Vendor_Actuals, Vendor_Calibration_Delta
"""

import logging
import pandas as pd
import streamlit as st

from config import DATA_FILE

logger = logging.getLogger(__name__)

# Columns from dim_building joined onto fact_savings for chart use
_BLD_JOIN_COLS = [
    "asset_id",
    "Department",
    "Use_Case",
    "Facility_Type",
    "Gross_Asset_Value_2024",
    "Life_Years_Total",
    "Age_Years_Current",
]

# Required columns that must exist in each sheet after loading
_REQUIRED_COLS: dict[str, list[str]] = {
    "fact_savings": ["asset_id", "year", "baseline"],
    "fact_pillar":  ["asset_id", "year", "pillar", "l1", "l2"],
    "dim_building": ["asset_id", "Department"],
    "dim_year":     ["year"],
}


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(show_spinner="Loading simulation data…")
def load_data() -> tuple[
    pd.DataFrame,  # fact_savings
    pd.DataFrame,  # fact_pillar
    pd.DataFrame,  # dim_building
    pd.DataFrame,  # dim_year
    pd.DataFrame,  # portfolio_assessment
    pd.DataFrame,  # pillar_cfg         (empty if sheet missing)
    pd.DataFrame,  # components         (empty if sheet missing)
    pd.DataFrame,  # hist_costs         (empty if sheet missing)
    pd.DataFrame,  # fact_vendor_named  (empty if sheet missing)
]:
    """Load all required and optional sheets from the star-schema Excel workbook.

    Required sheets (raises on missing):
        - fact_savings: Building-year cost baseline and savings data.
        - fact_pillar:  Pre-computed pillar-level savings breakdown.
        - dim_building: Building metadata (department, type, asset value, age).
        - dim_year:     Year dimension (used for joins and axis labelling).
        - Portfolio_Assessment: Portfolio-level KPI summary.

    Optional sheets (returns empty DataFrame with default columns if missing):
        - Pillar_Config:    Per-pillar rate overrides from the model run.
                            v6 columns: Pillar_Key, Rate_Default, k_p, x0_p, Enabled.
        - Components:       Component-level cost breakdown for the Sankey chart.
        - Historical_Costs: DFF historical cost actuals (2016–2024).

    Returns:
        Tuple of eight DataFrames in the order listed above.

    Raises:
        FileNotFoundError: If DATA_FILE does not exist.
        ValueError: If a required sheet is missing from the workbook.
    """
    def _read(sheet: str) -> pd.DataFrame:
        return pd.read_excel(DATA_FILE, sheet_name=sheet)

    def _read_optional(sheet: str, fallback_cols: list[str]) -> pd.DataFrame:
        try:
            return _read(sheet)
        except Exception as exc:
            logger.warning("Optional sheet '%s' not found — using empty fallback. (%s)", sheet, exc)
            return pd.DataFrame(columns=fallback_cols)

    # Required sheets — let exceptions propagate to caller
    fact_sav   = _read("fact_savings")
    fact_pil   = _read("fact_pillar")
    dim_bld    = _read("dim_building")
    dim_yr     = _read("dim_year")
    pa         = _read("Portfolio_Assessment")

    # Optional sheets
    # v6 Pillar_Config: Pillar_Key, Pillar_Name, Rate_Low, Rate_High,
    #                   Rate_Default, k_p, x0_p, Enabled, Description
    pillar_cfg     = _read_optional("Pillar_Config",    fallback_cols=["Pillar_Key", "Rate_Default"])
    comp_df        = _read_optional("Components",       fallback_cols=["Component", "Domain", "Pct_of_Facility_Cost"])
    hist_df        = _read_optional("Historical_Costs", fallback_cols=["Year", "DFF_Total_Actual"])
    vendor_named   = _read_optional(
        "fact_vendor_named_pbi",
        fallback_cols=["vendor_category", "year", "vendor_name", "category_label",
                       "price_index", "market_share_pct", "fragmented_spend",
                       "portfolio_spend", "ai_enabled_spend", "portfolio_savings_usd", "ai_savings_usd"],
    )

    # Plug-and-play: ensure budget_line column exists so Equipment/Fleet components
    # can be added to the Components sheet without any code changes.
    if len(comp_df) > 0 and "budget_line" not in comp_df.columns:
        comp_df["budget_line"] = "Facilities"

    _validate_required_cols(fact_sav,  "fact_savings")
    _validate_required_cols(fact_pil,  "fact_pillar")
    _validate_required_cols(dim_bld,   "dim_building")
    _validate_required_cols(dim_yr,    "dim_year")

    return fact_sav, fact_pil, dim_bld, dim_yr, pa, pillar_cfg, comp_df, hist_df, vendor_named


def _validate_required_cols(df: pd.DataFrame, sheet_name: str) -> None:
    """Raise ValueError if any required column is missing from a loaded sheet."""
    required = _REQUIRED_COLS.get(sheet_name, [])
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Sheet '{sheet_name}' is missing required columns: {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )


# ============================================================================
# FILTER HELPERS
# ============================================================================

def get_dept_assets(
    dim_building: pd.DataFrame,
    selected_dept: str,
) -> set[str] | None:
    """Return the set of asset_ids belonging to a department, or None for 'All'."""
    if selected_dept == "All":
        return None
    return set(dim_building[dim_building["Department"] == selected_dept]["asset_id"])


def filter_fact_savings(
    fact_savings: pd.DataFrame,
    sel_yrs: tuple[int, int],
    dept_assets: set[str] | None,
) -> pd.DataFrame:
    """Filter fact_savings by year range and optionally by department assets."""
    mask = fact_savings["year"].between(*sel_yrs)
    if dept_assets is not None:
        mask &= fact_savings["asset_id"].isin(dept_assets)
    return fact_savings[mask].copy()


def filter_fact_pillar(
    fact_pillar: pd.DataFrame,
    sel_yrs: tuple[int, int],
    dept_assets: set[str] | None,
) -> pd.DataFrame:
    """Filter fact_pillar by year range and optionally by department assets."""
    mask = fact_pillar["year"].between(*sel_yrs)
    if dept_assets is not None:
        mask &= fact_pillar["asset_id"].isin(dept_assets)
    return fact_pillar[mask].copy()


def join_building_info(
    fs: pd.DataFrame,
    dim_building: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join building metadata onto a filtered fact_savings DataFrame."""
    return fs.merge(
        dim_building[_BLD_JOIN_COLS],
        on="asset_id",
        how="left",
    )


def filter_portfolio_asmt(
    portfolio_asmt: pd.DataFrame,
    selected_dept: str,
) -> pd.DataFrame:
    """Filter Portfolio_Assessment to the selected department (or return all).

    v6 note: Portfolio_Assessment uses 'Department' column (same as v3).
    """
    if selected_dept == "All":
        return portfolio_asmt.copy()
    dept_col = "Department" if "Department" in portfolio_asmt.columns else "Asset_ID"
    if "Department" not in portfolio_asmt.columns:
        return portfolio_asmt.copy()
    return portfolio_asmt[portfolio_asmt["Department"] == selected_dept].copy()
