"""
score_engine.py — MRO Analytics Capability Score Engine
City of Chicago — Department of Public Services

Derives MRO benchmark scores from simulation model outputs.
Replaces static PERF/PRED/PRES scores["chicago"] in mro_benchmark.py.

Tier classification per dimension:
  T1 — Fully computed : direct, calibrated formula — signal is structurally valid
  T3 — Static fallback: model signal insufficient; original expert score returned

Dimension → source → tier → expected score (vs static)
───────────────────────────────────────────────────────────────────────────────
PERFORMANCE (8 dims, aligned to PERF_DIMS)
  Budget          T1  Historical_Costs  — abs std of YoY% changes in DFF actuals
  Payments        T3  —                 — static 7.5  (no AP-process signal)
  Purchase Orders T1  fact_vendor       — inverse mean fragmentation_premium
  Vendor          —   vendor_df         — overridden live in mro_benchmark.py
  Category        T1  fact_vendor       — mean portfolio_savings_rate
  Component       T3  —                 — static 2.5  (CPM is model artifact)
  Return          T1  Portfolio_Asmt    — CV of VCR_5y (high spread = poor mgmt)
  Risk            T1  Portfolio_Asmt    — CV of AHI    (high spread = inconsistent)

PREDICTION (3 dims, aligned to PRED_DIMS)
  Return Analytics     T1  fact_savings        — first-year (l1+l2)/baseline rate
  Investment Analytics T1  Portfolio_Asmt      — pct portfolio with VCR_10y > 10
  Risk Analytics       T1  Portfolio_Asmt      — inverse mean C_Financial

PRESCRIPTION (3 dims, aligned to PRES_DIMS)
  Financial Optimization T3 —           — static 2.5  (yr-1 rate < 1% vs 2.5 target)
  Work Modernization     T1 fact_pillar — all-years WM savings rate / max achievable
  Risk Mitigation        T3 —           — static 2.5  (sell-flag rate ≠ op resilience)
"""

from __future__ import annotations

import pandas as pd

# ── Calibration constants ─────────────────────────────────────────────────────
# Anchored against asset_model_outputs_v6.xlsx observed ranges.
# Values confirmed against static expert scores (see tier notes above).

# Budget: abs std of YoY% DFF changes — at Chicago's ~5% std → score 6.7
_BUDGET_STD_WORST  = 0.15    # 15% YoY volatility → score 0

# Purchase Orders: avg fragmentation_premium
_PO_PREMIUM_WORST  = 0.44    # max observed → score 0 (at Chicago avg 0.31 → 3.5)

# Category: mean portfolio_savings_rate
_CAT_RATE_BEST     = 0.44    # max observed → score 10 (at Chicago avg 0.20 → 4.5)

# Return (Performance): CV of VCR_5y portfolio
_VCR5_CV_WORST     = 3.0     # CV at this level → score 0 (Chicago CV 2.2 → 2.7)

# Risk (Performance): CV of AHI across portfolio
_AHI_CV_WORST      = 0.70    # CV at this level → score 0 (Chicago CV 0.49 → 3.0)

# Return Analytics (Prediction): first-year (l1+l2)/baseline rate
_RETURN_ANALYTICS_BENCH = 0.020  # 2% in yr1 = strong baseline → score 10
                                  # (Chicago yr1 rate ≈ 0.49% → score 2.5)

# Investment Analytics (Prediction): pct portfolio with VCR_10y > 10
_VCR10_HIGH_PCT_BEST = 0.80  # 80% high-return assets → score 10
                              # (Chicago 24.9% → score 3.1)

# Work Modernization (Prescription): all-years WM savings rate / max rate
_WM_RATE_BEST      = 0.12    # Work Modernization default max rate (from config.py)


def _clamp(x: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return float(max(lo, min(hi, x)))

def _norm01_to_10(x: float) -> float:
    """Normalize a 0–1 metric to 0–10, clamped."""
    return _clamp(10.0 * float(x), 0.0, 10.0)

def _norm_minmax_to_10(x: float, best: float, worst: float, higher_is_better: bool) -> float:
    """Map a metric to 0–10 using best/worst anchors.
    - If higher_is_better: best > worst; else best < worst.
    """
    x = float(x)
    if best == worst:
        return 5.0
    if higher_is_better:
        t = (x - worst) / (best - worst)
    else:
        t = (worst - x) / (worst - best)
    return _clamp(10.0 * t, 0.0, 10.0)

def _get_signal(signals: dict | None, key: str):
    if not signals:
        return None
    v = signals.get(key)
    return None if v is None else v


# ============================================================================
# PERFORMANCE SCORES  (8 dims — aligned to PERF_DIMS in mro_benchmark.py)
# ============================================================================

def score_budget(hist_costs: pd.DataFrame) -> float:
    """T1 — Budget stability: absolute std of year-over-year % changes in
    DFF actuals. Stable, predictable spending → higher score.
    """
    col = "DFF_Total_Actual"
    if hist_costs.empty or col not in hist_costs.columns:
        return 6.5
    vals = hist_costs[col].dropna()
    if len(vals) < 3:
        return 6.5
    std_yoy = vals.pct_change().dropna().abs().std()
    return _clamp(10.0 * max(0.0, 1.0 - std_yoy / _BUDGET_STD_WORST))


def score_payments(
    fact_pillar: pd.DataFrame,   # noqa: ARG001 — kept for API consistency
    fact_savings: pd.DataFrame,  # noqa: ARG001
    signals: dict | None = None,
) -> float:
    """Payments capability score.

    Default (no signals): T3 static fallback 7.5 (kept for backwards-compatibility).

    Data-driven (T1-ish) when signals provided (any subset):
      - invoice_cycle_days (lower is better)   : typical 3–30
      - touchless_rate (0–1, higher better)    : % invoices processed w/o manual touches
      - discount_capture_rate (0–1, higher)    : early-pay discount capture rate

    The score is a weighted blend of the available signals.
    """
    inv_days = _get_signal(signals, "invoice_cycle_days")
    touch    = _get_signal(signals, "touchless_rate")
    disc     = _get_signal(signals, "discount_capture_rate")

    parts = []
    weights = []

    if inv_days is not None:
        # best=3 days, worst=30 days
        parts.append(_norm_minmax_to_10(inv_days, best=3.0, worst=30.0, higher_is_better=False))
        weights.append(0.45)

    if touch is not None:
        parts.append(_norm01_to_10(touch))
        weights.append(0.35)

    if disc is not None:
        # discount capture: best=15%, worst=0%
        parts.append(_norm_minmax_to_10(disc, best=0.15, worst=0.0, higher_is_better=True))
        weights.append(0.20)

    if not parts:
        return 7.5

    wsum = sum(weights)
    return float(sum(p*w for p, w in zip(parts, weights)) / (wsum if wsum else 1.0))

def score_purchase_orders(fact_vendor: pd.DataFrame) -> float:
    """T1 — PO consolidation: inverse of mean fragmentation premium.
    High premium (fragmented POs) → low score.
    At Chicago avg premium 0.31 → score ≈ 3.5.
    """
    if fact_vendor.empty or "fragmentation_premium" not in fact_vendor.columns:
        return 3.5
    avg = fact_vendor["fragmentation_premium"].mean()
    return _clamp(10.0 * max(0.0, 1.0 - avg / _PO_PREMIUM_WORST))


def score_category(fact_vendor: pd.DataFrame) -> float:
    """T1 — Category management: mean portfolio_savings_rate across all
    vendor categories. Measures how much savings potential is captured.
    At Chicago avg rate 0.20 → score ≈ 4.5.
    """
    if fact_vendor.empty or "portfolio_savings_rate" not in fact_vendor.columns:
        return 4.0
    return _clamp(10.0 * fact_vendor["portfolio_savings_rate"].mean() / _CAT_RATE_BEST)


def score_component(
    fact_pillar: pd.DataFrame,  # noqa: ARG001
    cpm_df: pd.DataFrame,       # noqa: ARG001
    signals: dict | None = None,
) -> float:
    """Component-level visibility score.

    Default: T3 static fallback 2.5 (kept for backwards-compatibility).

    Data-driven when signals exist (any subset):
      - component_visibility (0–1): % MRO spend with component/SKU/part ID
      - catalog_coverage     (0–1): % spend that maps to a managed catalog
      - inventory_turnover   (float): turns/year (higher is better; typical 0–8)

    Notes:
      This intentionally measures *visibility & operational readiness*, not model
      artifact completeness (CPM map).
    """
    vis  = _get_signal(signals, "component_visibility")
    cat  = _get_signal(signals, "catalog_coverage")
    turn = _get_signal(signals, "inventory_turnover")

    parts = []
    weights = []

    if vis is not None:
        parts.append(_norm01_to_10(vis))
        weights.append(0.50)

    if cat is not None:
        parts.append(_norm01_to_10(cat))
        weights.append(0.30)

    if turn is not None:
        # best=6 turns, worst=0.5 turns
        parts.append(_norm_minmax_to_10(turn, best=6.0, worst=0.5, higher_is_better=True))
        weights.append(0.20)

    if not parts:
        return 2.5

    wsum = sum(weights)
    return float(sum(p*w for p, w in zip(parts, weights)) / (wsum if wsum else 1.0))

def score_return_perf(portfolio_asmt: pd.DataFrame) -> float:
    """T1 — Return: coefficient of variation of VCR_5y across portfolio.
    High CV = wide spread in asset returns = poor systematic investment mgmt.
    At Chicago VCR_5y CV 2.2 → score ≈ 2.7.
    """
    col = "Value_Cost_Ratio_5y"
    if portfolio_asmt.empty or col not in portfolio_asmt.columns:
        return 2.0
    valid = portfolio_asmt[col].dropna()
    if valid.empty or valid.mean() <= 0:
        return 2.0
    cv = valid.std() / valid.mean()
    return _clamp(10.0 * max(0.0, 1.0 - cv / _VCR5_CV_WORST))


def score_risk_perf(portfolio_asmt: pd.DataFrame) -> float:
    """T1 — Risk: coefficient of variation of AHI across portfolio.
    High CV = highly uneven physical health = inconsistent risk coverage.
    At Chicago AHI CV 0.49 → score ≈ 3.0.
    """
    if portfolio_asmt.empty or "AHI" not in portfolio_asmt.columns:
        return 3.5
    ahi = portfolio_asmt["AHI"].dropna()
    if ahi.empty or ahi.mean() <= 0:
        return 3.5
    ahi_cv = ahi.std() / ahi.mean()
    return _clamp(10.0 * max(0.0, 1.0 - ahi_cv / _AHI_CV_WORST))


# ============================================================================
# PREDICTION SCORES  (3 dims — aligned to PRED_DIMS)
# ============================================================================

def score_return_analytics(fact_savings: pd.DataFrame) -> float:
    """T1 — Return analytics capability: first-year (l1_total + l2_total)
    savings rate as a fraction of baseline. Captures how much ROI improvement
    is identified and captured immediately at program inception.
    At Chicago yr-1 rate ≈ 0.49% → score ≈ 2.5.
    """
    if fact_savings.empty:
        return 2.5
    l1_col = "l1_total" if "l1_total" in fact_savings.columns else "l1_override"
    l2_col = "l2_total" if "l2_total" in fact_savings.columns else "l2_override"
    if l1_col not in fact_savings.columns:
        return 2.5

    yr1 = int(fact_savings["year"].min())
    fs1 = fact_savings[fact_savings["year"] == yr1]
    l2_sum = fs1[l2_col].sum() if l2_col in fs1.columns else 0.0
    opt_sum = fs1[l1_col].sum() + l2_sum
    base    = fs1["baseline"].sum()
    if base <= 0:
        return 2.5
    rate = opt_sum / base
    return _clamp(10.0 * rate / _RETURN_ANALYTICS_BENCH)


def score_investment_analytics(portfolio_asmt: pd.DataFrame) -> float:
    """T1 — Investment analytics: % of portfolio with Value_Cost_Ratio_10y > 10.
    Higher fraction of high-return assets → stronger long-horizon investment analysis.
    At Chicago 24.9% → score ≈ 3.1.
    """
    col = "Value_Cost_Ratio_10y"
    if portfolio_asmt.empty or col not in portfolio_asmt.columns:
        return 3.5
    valid = portfolio_asmt[col].dropna()
    if valid.empty:
        return 3.5
    pct_high = (valid > 10.0).mean()
    return _clamp(10.0 * pct_high / _VCR10_HIGH_PCT_BEST)


def score_risk_analytics(portfolio_asmt: pd.DataFrame) -> float:
    """T1 — Risk analytics: inverse of mean C_Financial criticality.
    High average financial criticality = poor risk identification & monitoring.
    At Chicago C_Financial mean 0.634 → score ≈ 3.7.
    """
    col = "C_Financial"
    if portfolio_asmt.empty or col not in portfolio_asmt.columns:
        return 3.0
    return _clamp(10.0 * (1.0 - portfolio_asmt[col].mean()))


# ============================================================================
# PRESCRIPTION SCORES  (3 dims — aligned to PRES_DIMS)
# ============================================================================

def score_financial_optimization(
    fact_pillar: pd.DataFrame,   # noqa: ARG001
    fact_savings: pd.DataFrame,  # noqa: ARG001
) -> float:
    """T3 — Year-1 combined Vendor+Payment+EarlyPay savings rate is < 1%,
    below the target 2.5. All-years average overshoots (sigmoid reaches full
    adoption by end of horizon). Returns static 2.5."""
    return 2.5


def score_work_modernization(
    fact_pillar: pd.DataFrame,
    fact_savings: pd.DataFrame,
) -> float:
    """T1 — Work Modernization: all-years WM L1 savings as fraction of
    all-years baseline, normalized against the max achievable rate.
    At Chicago all-years average ≈ 2.1% → score ≈ 1.8 (static 2.0).
    """
    if fact_pillar.empty or fact_savings.empty:
        return 2.0
    l1_wm = fact_pillar[fact_pillar["pillar"] == "Work_Modernization"]["l1"].sum()
    base   = fact_savings["baseline"].sum()
    if base <= 0:
        return 2.0
    rate = l1_wm / base
    return _clamp(10.0 * rate / _WM_RATE_BEST)


def score_risk_mitigation(
    portfolio_asmt: pd.DataFrame,  # noqa: ARG001
    fact_savings: pd.DataFrame,    # noqa: ARG001
) -> float:
    """T3 — Sell-flag rate reflects economic value, not operational resilience.
    No reliable risk-mitigation program signal in current model. Returns static 2.5."""
    return 2.5


# ============================================================================
# 5-YR MRO OVERPAYMENT KPI
# ============================================================================

def compute_5yr_overpayment(
    fact_vendor: pd.DataFrame,
    start_year: int = 2024,
) -> float:
    """Total portfolio_savings achievable in the 5-year window starting at
    start_year. Represents cumulative facilities-scope MRO overpayment vs
    market-optimized pricing.

    Returns value in dollars. Falls back to $748M static if data unavailable.
    """
    col = next(
        (c for c in ("portfolio_savings", "portfolio_savings_usd") if c in fact_vendor.columns),
        None,
    )
    if fact_vendor.empty or col is None:
        return 748_000_000.0
    mask = fact_vendor["year"].between(start_year, start_year + 4)
    return float(fact_vendor.loc[mask, col].sum())


# ============================================================================
# POWER BI EXPORT HELPERS
# ============================================================================

# Dimension name lists (mirrored from mro_benchmark.py)
PERF_DIMS: list[str] = [
    "Budget", "Payments", "Purchase Orders", "Vendor",
    "Category", "Component", "Return", "Risk",
]
PRED_DIMS: list[str] = ["Return Analytics", "Investment Analytics", "Risk Analytics"]
PRES_DIMS: list[str] = ["Financial Optimization", "Work Modernization", "Risk Mitigation"]

_TAB_DIMS: dict[str, list[str]] = {
    "performance":  PERF_DIMS,
    "prediction":   PRED_DIMS,
    "prescription": PRES_DIMS,
}

# Tier per dimension: T1=computed, T3=static fallback, live=vendor override
_DIM_TIERS: dict[str, str] = {
    "Budget": "T1", "Payments": "T3", "Purchase Orders": "T1",
    "Vendor": "live", "Category": "T1", "Component": "T3",
    "Return": "T1", "Risk": "T1",
    "Return Analytics": "T1", "Investment Analytics": "T1", "Risk Analytics": "T1",
    "Financial Optimization": "T3", "Work Modernization": "T1", "Risk Mitigation": "T3",
}

# Static scores for the 5 benchmark participants (from MRO_Analytics JSX)
_PARTICIPANT_SCORES: dict[str, dict[str, list[float]]] = {
    "performance": {
        "max":     [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        "digital": [ 9.5,  9.5,  9.0,  9.5,  9.0,  8.5,  8.5,  9.0],
        "alpha":   [ 9.0,  9.0,  9.5,  9.5,  9.0,  9.5,  8.5,  9.0],
        "bench":   [ 7.0,  7.0,  5.5,  6.5,  5.5,  5.0,  4.5,  5.5],
        "late":    [ 4.0,  4.5,  2.5,  3.0,  2.5,  1.5,  1.5,  2.5],
    },
    "prediction": {
        "max":     [10.0, 10.0, 10.0],
        "digital": [ 8.5,  9.0,  8.5],
        "alpha":   [ 9.5,  9.0,  9.5],
        "bench":   [ 4.5,  5.0,  4.5],
        "late":    [ 1.5,  2.0,  1.5],
    },
    "prescription": {
        "max":     [10.0, 10.0, 10.0],
        "digital": [ 8.5,  8.0,  8.5],
        "alpha":   [ 9.0,  9.5,  9.0],
        "bench":   [ 4.5,  4.0,  4.5],
        "late":    [ 1.5,  1.0,  1.5],
    },
}

_PARTICIPANT_LABELS: dict[str, str] = {
    "max":     "Maximum Performance",
    "digital": "Algorithmically & Digitally Native",
    "alpha":   "Alpha Best Performer (GSA / Leading Municipal)",
    "bench":   "Benchmark Performance",
    "late":    "Late Adopters",
    "chicago": "Observed - City of Chicago",
}


def derive_vendor_score(fact_vendor: pd.DataFrame, fallback: float = 5.5) -> float:
    """Derive the Vendor dimension score from fact_vendor price_index.
    Mirrors the logic in mro_benchmark.derive_vendor_score_from_df().
    price_index 1.00 (benchmark) -> score 10.
    price_index 1.32 (worst observed) -> score 0.
    """
    if fact_vendor.empty or "price_index" not in fact_vendor.columns:
        return fallback
    if "fragmented_spend" not in fact_vendor.columns or fact_vendor["fragmented_spend"].sum() <= 0:
        return fallback
    wtd_pidx = (
        (fact_vendor["price_index"] * fact_vendor["fragmented_spend"]).sum()
        / fact_vendor["fragmented_spend"].sum()
    )
    return _clamp(10.0 * max(0.0, 1.0 - (wtd_pidx - 1.0) / 0.32))


def build_mro_scores_df(
    fact_vendor:    pd.DataFrame,
    portfolio_asmt: pd.DataFrame,
    fact_pillar:    pd.DataFrame,
    fact_savings:   pd.DataFrame,
    hist_costs:     pd.DataFrame,
    cpm_df:         pd.DataFrame,
    vendor_score:   float | None = None,
) -> pd.DataFrame:
    """Return Chicago's computed MRO scores as a flat DataFrame.

    Columns: participant | label | tab | dimension | score | tier

    Suitable for direct Power BI import. All 14 dimensions included;
    tier column distinguishes computed (T1/live) from static fallbacks (T3).
    """
    if vendor_score is None:
        vendor_score = derive_vendor_score(fact_vendor)
    scores = compute_all_scores(
        fact_vendor, portfolio_asmt, fact_pillar, fact_savings, hist_costs, cpm_df
    )
    scores["performance"][3] = vendor_score  # apply live vendor score
    rows = []
    for tab, dims in _TAB_DIMS.items():
        for dim, score in zip(dims, scores[tab]):
            rows.append({
                "participant": "chicago",
                "label":       _PARTICIPANT_LABELS["chicago"],
                "tab":         tab.capitalize(),
                "dimension":   dim,
                "score":       round(score, 2),
                "tier":        _DIM_TIERS.get(dim, "T1"),
            })
    return pd.DataFrame(rows)


def build_participant_scores_df(
    fact_vendor:    pd.DataFrame,
    portfolio_asmt: pd.DataFrame,
    fact_pillar:    pd.DataFrame,
    fact_savings:   pd.DataFrame,
    hist_costs:     pd.DataFrame,
    cpm_df:         pd.DataFrame,
    vendor_score:   float | None = None,
) -> pd.DataFrame:
    """Return scores for ALL 6 participants (Chicago computed + 5 static benchmarks).

    Columns: participant | label | tab | dimension | score | tier

    Chicago rows use T1/T3/live tiers (computed). Benchmark rows use tier='static'.
    """
    chi_df = build_mro_scores_df(
        fact_vendor, portfolio_asmt, fact_pillar, fact_savings, hist_costs, cpm_df, vendor_score
    )
    rows = chi_df.to_dict("records")
    for tab, dims in _TAB_DIMS.items():
        tab_scores = _PARTICIPANT_SCORES.get(tab, {})
        for participant, scores_list in tab_scores.items():
            for dim, score in zip(dims, scores_list):
                rows.append({
                    "participant": participant,
                    "label":       _PARTICIPANT_LABELS.get(participant, participant),
                    "tab":         tab.capitalize(),
                    "dimension":   dim,
                    "score":       score,
                    "tier":        "static",
                })
    return pd.DataFrame(rows)


def build_summary_pbi(
    fact_savings: pd.DataFrame,
    fact_pillar:  pd.DataFrame,
    fact_vendor:  pd.DataFrame,
) -> pd.DataFrame:
    """Una fila por año con todos los KPIs agregados del portafolio completo.

    Columnas: year, n_buildings, baseline_total, l1_total, l2_total, final_total,
              total_savings, savings_rate, l1_rate, l2_rate,
              [un col por pillar], vendor_savings_total.
    """
    l1_col = "l1_total" if "l1_total" in fact_savings.columns else "l1_override"
    l2_col = "l2_total" if "l2_total" in fact_savings.columns else "l2_override"
    fin_col = "final"   if "final"    in fact_savings.columns else "final_override"

    yr = (
        fact_savings.groupby("year")
        .agg(
            n_buildings   = ("asset_id", "nunique"),
            baseline_total = ("baseline", "sum"),
            l1_total       = (l1_col,    "sum"),
            l2_total       = (l2_col,    "sum"),
            final_total    = (fin_col,   "sum"),
        )
        .reset_index()
    )
    yr["total_savings"] = yr["l1_total"] + yr["l2_total"]
    yr["savings_rate"]  = yr["total_savings"] / yr["baseline_total"].replace(0, float("nan"))
    yr["l1_rate"]       = yr["l1_total"] / yr["baseline_total"].replace(0, float("nan"))
    yr["l2_rate"]       = yr["l2_total"] / yr["baseline_total"].replace(0, float("nan"))

    # Pillar breakdown by year (Facilities only for comparability)
    fp_fac = fact_pillar[fact_pillar.get("budget_line", pd.Series("Facilities", index=fact_pillar.index)) == "Facilities"] \
        if "budget_line" in fact_pillar.columns else fact_pillar
    pil_yr = (
        fp_fac.groupby(["year", "pillar"])["l1"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    pil_yr.columns.name = None
    yr = yr.merge(pil_yr, on="year", how="left")

    # Vendor savings by year
    _sav_col = next(
        (c for c in ("portfolio_savings", "portfolio_savings_usd") if c in fact_vendor.columns),
        None,
    )
    if not fact_vendor.empty and _sav_col is not None:
        vend_yr = (
            fact_vendor.groupby("year")[_sav_col]
            .sum()
            .reset_index(name="vendor_savings_total")
        )
        yr = yr.merge(vend_yr, on="year", how="left")

    return yr


def build_dim_pillar_pbi() -> pd.DataFrame:
    """Catálogo de 6 pillars con rates, parámetros sigmoid y clasificación.

    Para usar en PBI como tabla de dimensión relacionada con fact_pillar[pillar].
    """
    rows = [
        {"pillar_key": "Work_Modernization",   "pillar_name": "Work Modernization",
         "layer": "L1-Operational", "ai_amplified": True,
         "rate_default": 0.120, "rate_low": 0.06, "rate_high": 0.18,
         "k": 1.5, "x0": 3.0,
         "description": "Digital work orders, IoT, predictive workflows"},
        {"pillar_key": "Demand_Management",    "pillar_name": "Demand Management",
         "layer": "L1-Operational", "ai_amplified": True,
         "rate_default": 0.070, "rate_low": 0.04, "rate_high": 0.10,
         "k": 1.2, "x0": 4.0,
         "description": "JIT procurement, demand sensing, SKU rationalization"},
        {"pillar_key": "Asset_Management",     "pillar_name": "Asset Management",
         "layer": "L1-Operational", "ai_amplified": True,
         "rate_default": 0.100, "rate_low": 0.05, "rate_high": 0.15,
         "k": 1.0, "x0": 5.0,
         "description": "AHI-based lifecycle optimization, condition monitoring"},
        {"pillar_key": "Vendor_Management",    "pillar_name": "Vendor Management",
         "layer": "L1-Financial",   "ai_amplified": False,
         "rate_default": 0.130, "rate_low": 0.08, "rate_high": 0.20,
         "k": 0.8, "x0": 4.0,
         "description": "Vendor consolidation, fragmentation premium (ACFR-calibrated)"},
        {"pillar_key": "Payment_Management",   "pillar_name": "Payment Management",
         "layer": "L1-Financial",   "ai_amplified": False,
         "rate_default": 0.025, "rate_low": 0.02, "rate_high": 0.06,
         "k": 1.8, "x0": 4.0,
         "description": "AP cycle optimization, invoice automation"},
        {"pillar_key": "Early_Pay_Management", "pillar_name": "Early Pay Management",
         "layer": "L1-Financial",   "ai_amplified": False,
         "rate_default": 0.005, "rate_low": 0.01, "rate_high": 0.04,
         "k": 2.0, "x0": 5.0,
         "description": "Dynamic discounting, supply chain finance"},
    ]
    return pd.DataFrame(rows)


def build_overpayment_df(
    fact_vendor: pd.DataFrame,
    min_year:    int = 2024,
    max_year:    int = 2030,
) -> pd.DataFrame:
    """Return facilities-scope 5yr MRO overpayment for rolling start years.

    Columns: start_year | end_year | overpayment_usd
    """
    rows = [
        {
            "start_year":      y,
            "end_year":        y + 4,
            "overpayment_usd": compute_5yr_overpayment(fact_vendor, start_year=y),
        }
        for y in range(min_year, max_year + 1)
    ]
    return pd.DataFrame(rows)


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def compute_all_scores(
    fact_vendor:    pd.DataFrame,
    portfolio_asmt: pd.DataFrame,
    fact_pillar:    pd.DataFrame,
    fact_savings:   pd.DataFrame,
    hist_costs:     pd.DataFrame,
    cpm_df:         pd.DataFrame,
    signals:        dict | None = None,
) -> dict[str, list[float]]:
    """Compute Chicago benchmark scores for all 3 MRO sub-tabs.

    Returns a dict with keys "performance", "prediction", "prescription",
    each containing a list of floats aligned to PERF_DIMS / PRED_DIMS /
    PRES_DIMS in mro_benchmark.py.

    Vendor (index 3 in "performance") is set to 5.5 as a placeholder;
    render_mro_benchmark_tab() overwrites it live from vendor_df price_index.

    Args:
        fact_vendor:    vendor_df filtered to the active sidebar year range.
        portfolio_asmt: Portfolio_Assessment (dept-filtered).
        fact_pillar:    fact_pillar filtered to the active year range + dept.
        fact_savings:   fact_savings (recalculated) filtered to year + dept.
        hist_costs:     Historical_Costs (full history — not time-filtered).
        cpm_df:         Component_Pillar_Map (static — not filtered).

    Computed vs static per sub-tab:
        performance:   Budget✓ Payments(s) PO✓ Vendor(live) Category✓
                       Component(s) Return✓ Risk✓
        prediction:    ReturnAnalytics✓ InvestmentAnalytics✓ RiskAnalytics✓
        prescription:  FinancialOpt(s) WorkMod✓ RiskMitigation(s)
        (s) = static fallback
    """
    return {
        "performance": [
            score_budget(hist_costs),                           # 0 Budget
            score_payments(fact_pillar, fact_savings, signals), # 1 Payments (signals → data-driven)
            score_purchase_orders(fact_vendor),                 # 2 Purchase Orders
            5.5,                                                # 3 Vendor (live override)
            score_category(fact_vendor),                        # 4 Category
            score_component(fact_pillar, cpm_df, signals),      # 5 Component (signals → data-driven)
            score_return_perf(portfolio_asmt),                  # 6 Return
            score_risk_perf(portfolio_asmt),                    # 7 Risk
        ],
        "prediction": [
            score_return_analytics(fact_savings),               # 0 Return Analytics
            score_investment_analytics(portfolio_asmt),         # 1 Investment Analytics
            score_risk_analytics(portfolio_asmt),               # 2 Risk Analytics
        ],
        "prescription": [
            score_financial_optimization(fact_pillar, fact_savings),  # 0 Fin Opt (static)
            score_work_modernization(fact_pillar, fact_savings),       # 1 Work Modernization
            score_risk_mitigation(portfolio_asmt, fact_savings),       # 2 Risk Mitigation (s)
        ],
    }
