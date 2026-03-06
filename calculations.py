"""
calculations.py -- Facilities Portfolio Optimization Dashboard
City of Chicago -- Department of Public Services

Pure calculation functions: sigmoid adoption curve and L1/L2 savings model.
No Streamlit or Plotly imports -- these functions are fully testable in isolation.
"""

import logging
import math
import pandas as pd
from config import PILLAR_DEFAULTS, OPERATIONAL_PILLARS

logger = logging.getLogger(__name__)


# ============================================================================
# ADOPTION CURVE
# ============================================================================

def sigmoid(elapsed: float, k: float, x0: float) -> float:
    """Logistic (sigmoid) function used to model gradual pillar adoption.

    Maps elapsed implementation years to an adoption fraction in (0, 1).
    The curve rises slowly at first, then accelerates around x0, then flattens.

    Args:
        elapsed: Years since the implementation start year (can be negative
                 for pre-implementation periods, which gives values near 0).
        k:       Steepness of the curve. Higher values = sharper transition.
                 Typical range: 0.8 - 2.0.
        x0:      Inflection point -- the year at which adoption reaches ~50%.
                 Typical range: 2.0 - 5.0.

    Returns:
        Adoption fraction in the open interval (0.0, 1.0).

    Examples:
        >>> round(sigmoid(0, k=1.5, x0=3.0), 3)   # year 0 -- very low adoption
        0.011
        >>> round(sigmoid(3, k=1.5, x0=3.0), 3)   # year 3 -- inflection: ~50%
        0.5
        >>> round(sigmoid(8, k=1.5, x0=3.0), 3)   # year 8 -- near full adoption
        0.999
    """
    return 1.0 / (1.0 + math.exp(-k * (elapsed - x0)))


def linear_adoption_ramp(elapsed: float, adopt_yrs: int) -> float:
    """Linear ramp from 0 to 1 over `adopt_yrs` years, then capped at 1.

    Used for the AI (Layer 2) adoption schedule, which follows a simpler
    linear ramp rather than a sigmoid curve.

    Args:
        elapsed:   Years since implementation start year.
        adopt_yrs: Total years until full AI adoption (ramp duration).
                   Must be > 0; if 0 or negative, returns 1.0 immediately.

    Returns:
        Adoption fraction clamped to [0.0, 1.0].

    Examples:
        >>> linear_adoption_ramp(0, adopt_yrs=7)
        0.0
        >>> linear_adoption_ramp(3.5, adopt_yrs=7)
        0.5
        >>> linear_adoption_ramp(10, adopt_yrs=7)
        1.0
    """
    if adopt_yrs <= 0:
        return 1.0
    return min(max(elapsed / adopt_yrs, 0.0), 1.0)


# ============================================================================
# SAVINGS MODEL
# ============================================================================

def recalc_savings(
    df: pd.DataFrame,
    bld_df: pd.DataFrame,
    pillar_rates: dict[str, float],
    ai_rate_val: float,
    adopt_yrs: int,
    start_year: int = 2024,
    *,
    _facilities_only: bool = False,
) -> pd.DataFrame:
    """Apply pillar (L1) and AI (L2) savings overrides to the fact_savings dataset.

    This is the core recalculation engine that responds to sidebar controls.
    It computes two savings layers on top of the baseline cost for each
    building-year record:

    Layer 1 (L1) -- Pillar-based operational savings (multiplicative / diminishing-returns):
        For each enabled pillar, the per-pillar adoption fraction is:
            s_p = clip(r_p x sigmoid(elapsed, k, x0), 0, 0.95)
        Pillars are combined multiplicatively (residual product), preventing
        unrealistic linear stacking:
            combined = 1 − Π(1 − s_p)
            L1 = baseline x combined
        Total L1 is capped at the baseline cost (cannot save more than you spend).

    Layer 2 (L2) -- AI amplification of operational L1 only:
        L2 = L1_operational x ai_rate x linear_adoption_ramp.
        Only OPERATIONAL_PILLARS (Work_Modernization, Demand_Management,
        Asset_Management) feed the L2 base. Financial/a-posteriori pillars
        (Vendor, Payment, Early Pay) reduce costs but are not AI-amplifiable
        because their savings are policy-driven, not process-driven.
        L2 <= L1_operational <= L1_total.

    Final cost = (baseline − L1) − L2, clamped to >= 0.

    Args:
        df:           Filtered fact_savings DataFrame (not used directly;
                      kept for API consistency with filter helpers).
        bld_df:       fact_savings joined with dim_building. Must contain
                      columns: "year", "baseline", and any columns referenced
                      by downstream aggregations (e.g. "nav_end", "depreciation").
        pillar_rates: Dict mapping pillar key -> override rate (0.0 disables pillar).
                      Keys must match PILLAR_DEFAULTS in config.py.
        ai_rate_val:  Blended AI amplification rate (0.0 - 1.0).
                      0.0 disables Layer 2 entirely.
        adopt_yrs:    Number of years for full AI adoption (linear ramp duration).
        start_year:   Base year from which elapsed time is measured. Defaults to 2024.

    Returns:
        Copy of bld_df with three new columns appended:
            - "l1_override":    Total Layer 1 savings per building-year ($).
            - "l2_override":    Total Layer 2 (AI) savings per building-year ($).
            - "final_override": Net cost after both savings layers ($, >= 0).

    Notes:
        - Sigmoid parameters (k, x0) come from PILLAR_DEFAULTS in config.py.
        - If pillar_rates[key] <= 0, that pillar is skipped entirely.
        - l2_override is rounded to 2 decimal places to avoid float noise.
    """
    if bld_df is None or bld_df.empty:
        logger.warning("recalc_savings() received an empty DataFrame -- returning empty result.")
        return bld_df if bld_df is not None else pd.DataFrame()

    active_pillars = [k for k, r in pillar_rates.items() if r > 0]
    if not active_pillars:
        logger.info("recalc_savings() called with all pillar rates at 0 -- L1 will be zero.")
    if ai_rate_val == 0.0:
        logger.info("recalc_savings() called with ai_rate=0 -- L2 will be zero.")

    d = bld_df.copy()

    # ── v7: identify Facilities rows (recalculate) vs Equipment/Fleet (keep precomputed) ──
    # budget_line column added in v7. v6 data has no budget_line → all rows are Facilities.
    if "budget_line" in d.columns:
        fac_mask = d["budget_line"].isin(["Facilities"]) | d["budget_line"].isna()
    else:
        fac_mask = pd.Series(True, index=d.index)

    # Pre-populate ALL rows from stored simulation values (Equipment/Fleet keep these).
    # Columns l1_total / l2_total / final come from fact_savings via join_building_info.
    if "l1_total" in d.columns and "l2_total" in d.columns and "final" in d.columns:
        d["l1_override"]    = d["l1_total"].copy()
        d["l2_override"]    = d["l2_total"].copy()
        d["final_override"] = d["final"].copy()
    else:
        d["l1_override"]    = 0.0
        d["l2_override"]    = 0.0
        d["final_override"] = d["baseline"].copy()

    if not fac_mask.any():
        return d

    # ── Recalculate Facilities rows only ─────────────────────────────────────
    fac_idx    = d.index[fac_mask]
    elapsed_f  = d.loc[fac_idx, "year"] - start_year
    baseline_f = d.loc[fac_idx, "baseline"]

    residual_total = pd.Series(1.0, index=fac_idx)
    residual_op    = pd.Series(1.0, index=fac_idx)

    for key, r_p in pillar_rates.items():
        if r_p <= 0:
            continue
        k_p  = PILLAR_DEFAULTS[key]["k"]
        x0_p = PILLAR_DEFAULTS[key]["x0"]
        mu   = elapsed_f.apply(lambda e, k=k_p, x=x0_p: sigmoid(e, k, x))
        s    = (r_p * mu).clip(0.0, 0.95)
        residual_total *= (1.0 - s)
        if key in OPERATIONAL_PILLARS:
            residual_op *= (1.0 - s)

    l1    = (baseline_f * (1.0 - residual_total)).clip(upper=baseline_f)
    l1_op = (baseline_f * (1.0 - residual_op)).clip(upper=baseline_f)

    d.loc[fac_idx, "l1_override"] = l1

    adopt         = elapsed_f.apply(lambda e: linear_adoption_ramp(e, adopt_yrs))
    cost_after_l1 = (baseline_f - l1).clip(lower=0)
    l2            = (l1_op * ai_rate_val * adopt).round(2)

    d.loc[fac_idx, "l2_override"]    = l2
    d.loc[fac_idx, "final_override"] = (cost_after_l1 - l2).clip(lower=0)

    logger.debug(
        "recalc_savings() complete: %d rows (Facilities: %d), L1=%.2fM, L2=%.2fM",
        len(d), int(fac_mask.sum()),
        d["l1_override"].sum() / 1e6,
        d["l2_override"].sum() / 1e6,
    )
    return d


# ============================================================================
# PER-PILLAR RECALCULATION
# ============================================================================

def recalc_per_pillar(
    bld_df: pd.DataFrame,
    pillar_rates: dict[str, float],
    start_year: int = 2024,
) -> pd.DataFrame:
    """Compute per-pillar L1 savings by year, consistent with the multiplicative model.

    Each pillar's savings are first computed individually (baseline x r_p x sigmoid),
    then scaled proportionally so their sum equals the actual multiplicative L1 from
    recalc_savings(). This keeps the waterfall and trajectory charts arithmetically
    consistent with the total L1 shown in KPI cards and the savings table.

    Args:
        bld_df:       fact_savings joined with dim_building. Must contain
                      columns: "year", "baseline".
        pillar_rates: Dict mapping pillar key -> override rate (0.0 disables pillar).
        start_year:   Base year from which elapsed time is measured. Defaults to 2024.

    Returns:
        DataFrame with columns: year, pillar, l1.
        One row per (year, pillar) combination. Disabled pillars (rate <= 0) are omitted.
        Per-year sums match recalc_savings() l1_override totals exactly.
    """
    if bld_df is None or bld_df.empty:
        return pd.DataFrame(columns=["year", "pillar", "l1"])

    # v7: pillar breakdown is Facilities-only (sidebar rates apply to Facilities).
    if "budget_line" in bld_df.columns:
        bld_df = bld_df[
            bld_df["budget_line"].isin(["Facilities"]) | bld_df["budget_line"].isna()
        ].copy()
    if bld_df.empty:
        return pd.DataFrame(columns=["year", "pillar", "l1"])

    d       = bld_df[["year", "baseline"]].copy()
    elapsed = d["year"] - start_year

    # --- Pass 1: individual additive contributions + multiplicative residual ---
    indiv:    dict[str, pd.Series] = {}
    residual = pd.Series(1.0, index=d.index)

    for key, r_p in pillar_rates.items():
        if r_p <= 0:
            continue
        k_p  = PILLAR_DEFAULTS[key]["k"]
        x0_p = PILLAR_DEFAULTS[key]["x0"]
        mu   = elapsed.apply(lambda e, k=k_p, x=x0_p: sigmoid(e, k, x))
        s    = (r_p * mu).clip(0.0, 0.95)
        residual  *= (1.0 - s)
        indiv[key] = d["baseline"] * r_p * mu   # individual (for proportioning)

    if not indiv:
        return pd.DataFrame(columns=["year", "pillar", "l1"])

    # Multiplicative L1 -- the ground-truth total (same formula as recalc_savings)
    l1_multi  = (d["baseline"] * (1.0 - residual)).clip(upper=d["baseline"])
    indiv_sum = sum(indiv.values())              # element-wise sum of all individual contribs

    # --- Pass 2: scale each pillar proportionally to l1_multi ---
    # scale_factor = l1_multi / indiv_sum  (0 where no active pillars)
    scale = (l1_multi / indiv_sum.replace(0, float("nan"))).fillna(0.0)

    records = []
    for key, i_sav in indiv.items():
        d["_sav"] = i_sav * scale
        yr_sums   = d.groupby("year")["_sav"].sum().reset_index()
        yr_sums["pillar"] = key
        yr_sums.rename(columns={"_sav": "l1"}, inplace=True)
        records.append(yr_sums)

    d.drop(columns=["_sav"], inplace=True, errors="ignore")
    return pd.concat(records, ignore_index=True)[["year", "pillar", "l1"]]


# ============================================================================
# YEARLY AGGREGATION
# ============================================================================

def aggregate_yearly(fs_recalc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate building-level recalculated savings to yearly portfolio totals.

    Args:
        fs_recalc: Output of recalc_savings() -- building-year records with
                   l1_override, l2_override, final_override columns.

    Returns:
        DataFrame indexed by year with columns:
            baseline, l1, l2, final, nav_end, nav_begin, dep,
            total_sav, total_pct.
        total_pct is NaN for years where baseline == 0.
    """
    yearly = fs_recalc.groupby("year").agg(
        baseline = ("baseline",     "sum"),
        l1       = ("l1_override",  "sum"),
        l2       = ("l2_override",  "sum"),
        final    = ("final_override","sum"),
        nav_end  = ("nav_end",      "sum"),
        nav_begin= ("nav_begin",    "sum"),
        dep      = ("depreciation", "sum"),
    ).reset_index()

    yearly["total_sav"] = yearly["l1"] + yearly["l2"]
    yearly["total_pct"] = yearly["total_sav"] / yearly["baseline"].replace(0, float("nan"))

    return yearly


def aggregate_yearly_by_line(fs_recalc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate recalculated savings by (year, budget_line) for the overview chart.

    v7: returns one row per (year, budget_line) with baseline, l1, l2, final.
    v6 fallback (no budget_line column): returns the same shape with budget_line="Facilities".

    Args:
        fs_recalc: Output of recalc_savings() with l1_override, l2_override, final_override.

    Returns:
        DataFrame with columns: year, budget_line, baseline, l1, l2, final.
    """
    if "budget_line" not in fs_recalc.columns:
        df = aggregate_yearly(fs_recalc)
        df["budget_line"] = "Facilities"
        return df[["year", "budget_line", "baseline", "l1", "l2", "final"]].rename(
            columns={"l1": "l1", "l2": "l2", "final": "final"}
        )

    return (
        fs_recalc
        .groupby(["year", "budget_line"])
        .agg(
            baseline=("baseline",      "sum"),
            l1      =("l1_override",   "sum"),
            l2      =("l2_override",   "sum"),
            final   =("final_override","sum"),
        )
        .reset_index()
    )


def fmt(n: float) -> str:
    """Format a dollar amount as a compact human-readable string.

    Args:
        n: Numeric dollar amount (can be negative).

    Returns:
        String with $ prefix and B/M/K suffix as appropriate.

    Examples:
        >>> fmt(1_500_000_000)
        '$1.50B'
        >>> fmt(3_200_000)
        '$3.2M'
        >>> fmt(45_000)
        '$45.0K'
        >>> fmt(900)
        '$900'
    """
    if abs(n) >= 1e9:
        return f"${n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"${n / 1e6:.1f}M"
    if abs(n) >= 1e3:
        return f"${n / 1e3:.1f}K"
    return f"${n:.0f}"


# ============================================================================
# SELF-TEST  (run with: python calculations.py)
# ============================================================================

if __name__ == "__main__":
    import sys

    print("-- calculations.py self-test -----------------------------")

    # 1. Multiplicative combination never exceeds a single-pillar upper bound
    from config import PILLAR_DEFAULTS, OPERATIONAL_PILLARS
    rates_full = {k: v["r_p"] for k, v in PILLAR_DEFAULTS.items()}
    test_row = pd.DataFrame({
        "year":         [2024, 2026, 2028, 2030, 2033],
        "baseline":     [1_000_000.0] * 5,
        "nav_end":      [0.0] * 5,
        "nav_begin":    [0.0] * 5,
        "depreciation": [0.0] * 5,
    })
    result = recalc_savings(pd.DataFrame(), test_row, rates_full, ai_rate_val=0.10, adopt_yrs=7)
    max_l1_pct = (result["l1_override"] / result["baseline"]).max()
    max_total_sav_pct = (
        (result["l1_override"] + result["l2_override"]) / result["baseline"]
    ).max()
    assert max_l1_pct < 0.35, (
        f"FAIL: L1 reached {max_l1_pct:.1%} -- multiplicative model should stay well below 35%"
    )
    assert max_total_sav_pct < 0.45, (
        f"FAIL: L1+L2 reached {max_total_sav_pct:.1%} -- combined savings look too high"
    )
    assert (result["final_override"] >= 0).all(), "FAIL: final_override has negative values"
    print(f"  [OK] max L1 pct     = {max_l1_pct:.2%}  (< 35%)")
    print(f"  [OK] max total pct  = {max_total_sav_pct:.2%}  (< 45%)")
    print(f"  [OK] final_override >= 0 in all rows")

    # 2. Single pillar at r_p=0.08 should not exceed 8% at full adoption
    single_rate = {"Work_Modernization": 0.08, "Demand_Management": 0.0,
                   "Asset_Management": 0.0, "Vendor_Management": 0.0,
                   "Payment_Management": 0.0, "Early_Pay_Management": 0.0}
    r_single = recalc_savings(pd.DataFrame(), test_row, single_rate, ai_rate_val=0.0, adopt_yrs=7)
    max_single = (r_single["l1_override"] / r_single["baseline"]).max()
    assert max_single <= 0.08 + 1e-9, (
        f"FAIL: single pillar L1 = {max_single:.4%} > rate cap of 8%"
    )
    print(f"  [OK] single pillar L1 max = {max_single:.4%}  (<= r_p=8%)")

    # 3. Zero rates -> zero savings
    zero_rates = {k: 0.0 for k in PILLAR_DEFAULTS}
    r_zero = recalc_savings(pd.DataFrame(), test_row, zero_rates, ai_rate_val=0.0, adopt_yrs=7)
    assert r_zero["l1_override"].sum() == 0.0, "FAIL: expected zero L1 with all rates=0"
    assert r_zero["l2_override"].sum() == 0.0, "FAIL: expected zero L2 with all rates=0"
    print("  [OK] zero rates -> zero savings")

    print("-- All assertions passed ----------------------------------")
    sys.exit(0)
