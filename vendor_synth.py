"""
vendor_synth.py
Data module: loads fact_vendor from asset_model_outputs.xlsx,
generates synthetic (deterministic) vendor assignments, and computes metrics.
"""

import pathlib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
V6_PATH = pathlib.Path(__file__).parent / "asset_model_outputs.xlsx"

# ---------------------------------------------------------------------------
# Category metadata
# ---------------------------------------------------------------------------
CATEGORY_LABELS = {
    "facility_oandm":      "Facility O&M",
    "asset_contracts":     "Asset Contracts",
    "asset_parts":         "Asset Parts",
    "facility_materials":  "Facility Materials",
    "fleet_maint":         "Fleet Maintenance",
}

# (name, price_index) — sorted descending by price_index (most expensive first)
VENDOR_POOL: dict[str, list[tuple[str, float]]] = {
    "facility_oandm": [
        ("GCA Services",              1.32),  # regional FM/janitorial, high fragmentation premium
        ("Able Services",             1.28),  # ABM subsidiary, integrated FM
        ("ISS Facility Services",     1.22),  # global IFM provider
        ("Aramark Facility Services", 1.18),  # FM for government/municipal buildings
        ("Sodexo Government Services",1.15),  # FM for federal/municipal facilities
        ("ABM Industries",            1.10),  # integrated facility services, municipal contracts
        ("Envise",                    1.07),  # ex-EMCOR FM operations, mechanical O&M
        ("Cushman & Wakefield",       1.04),  # commercial FM
        ("JLL",                       1.01),  # Jones Lang LaSalle, global FM
        ("CBRE",                      1.00),  # global FM benchmark
    ],
    "asset_contracts": [
        ("National Air Balancing", 1.30),  # specialty HVAC balancing & commissioning
        ("EMCOR Group",            1.25),  # mechanical & electrical building services
        ("Lennox Intl",            1.21),  # HVAC equipment & service contracts
        ("Daikin Applied",         1.17),  # commercial HVAC systems
        ("Comfort Systems USA",    1.13),  # HVAC/mechanical service contractor (replaces York — same entity as JCI)
        ("Trane Technologies",     1.09),  # building climate systems
        ("Carrier Global",         1.06),  # HVAC & refrigeration service
        ("Honeywell",              1.03),  # building automation systems
        ("Siemens",                1.01),  # building technologies & BAS
        ("Johnson Controls",       1.00),  # BAS & fire/security — benchmark
    ],
    "asset_parts": [
        ("Motion Industries",      1.29),  # industrial MRO distributor
        ("Interline Brands",       1.24),  # FM-specific MRO (replaces Anixter — network/cable, not facility parts)
        ("Graybar Electric",       1.20),  # electrical distribution, Chicago presence
        ("Global Industrial",      1.16),  # industrial & facility supplies
        ("McMaster-Carr",          1.12),  # broad MRO catalog
        ("Zoro Tools",             1.08),  # online MRO distributor
        ("HD Supply",              1.05),  # facilities maintenance supplies
        ("Fastenal",               1.03),  # fasteners & MRO
        ("MSC Industrial",         1.01),  # metalworking & MRO
        ("Grainger",               1.00),  # broad-line MRO — benchmark
    ],
    "facility_materials": [
        ("Rexel USA",              1.28),  # electrical & facility materials (replaces Consolidated Electrical — too narrow)
        ("Noland Company",         1.23),  # HVAC/plumbing materials, appropriate scale (replaces Waxman — too small)
        ("F.W. Webb",              1.19),  # plumbing, HVAC & industrial PVF
        ("Johnstone Supply",       1.14),  # HVAC/R parts & materials
        ("Hajoca Corp",            1.10),  # plumbing & mechanical materials
        ("Ferguson Enterprises",   1.07),  # HVAC, plumbing & fire protection materials
        ("ABC Supply",             1.04),  # roofing & exterior building materials
        ("Lowe's Pro",             1.02),  # broad facility materials
        ("Home Depot Pro",         1.01),  # broad facility materials
        ("Maintenance Warehouse",  1.00),  # FM-specific materials benchmark (replaces W.W. Grainger — already in asset_parts)
    ],
    "fleet_maint": [
        ("Rush Truck Centers",    1.27),  # largest US truck dealer/service network
        ("Navistar Service",      1.22),  # International truck service (common in city fleets)
        ("Altec Industries",      1.18),  # aerial lift & utility vehicle maintenance
        ("Holman Fleet (ARI)",    1.14),  # full-service fleet management, government contracts (replaces Firestone — tires only)
        ("Wheels Inc.",           1.10),  # fleet management, Des Plaines IL — Chicago-area specialist (replaces Bridgestone — tires only)
        ("Fleet Pride",           1.06),  # heavy-duty truck parts & service (replaces Goodyear — tires only)
        ("NAPA AutoCare Fleet",   1.04),  # fleet service network
        ("Jiffy Lube Fleet",      1.02),  # preventive maintenance (oil, fluids)
        ("Penske",                1.01),  # fleet maintenance & logistics services
        ("Ryder",                 1.00),  # full-service maintenance — benchmark
    ],
}

# Dirichlet concentration parameter — top vendors get more share
_ALPHA = np.array([8, 6, 4, 3, 2, 2, 1.5, 1.5, 1, 1], dtype=float)

# Fragmentation premium by vendor category — reflects real market structure:
#   facility_oandm     : labor-intensive, local subcontractors → highest fragmentation
#   asset_contracts    : HVAC/MEP moderately fragmented
#   asset_parts        : MRO distribution somewhat consolidated
#   facility_materials : commodity materials, lower fragmentation
#   fleet_maint        : fleet management most consolidated (master contracts)
FRAGMENTATION_PREMIUM_BY_CAT: dict[str, float] = {
    "facility_oandm":      0.38,
    "asset_contracts":     0.31,
    "asset_parts":         0.27,
    "facility_materials":  0.22,
    "fleet_maint":         0.19,
}

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_vendor_fact(path=V6_PATH) -> pd.DataFrame:
    """Load the fact_vendor sheet (21 000 rows) from the Excel workbook."""
    return pd.read_excel(path, sheet_name="fact_vendor")


def generate_vendor_df(fact_vendor: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Aggregate fact_vendor to (vendor_category × year), then distribute spend
    across 10 synthetic vendors per category using Dirichlet market shares.

    Returns a DataFrame with columns:
        vendor_category, year, vendor_name, category_label,
        price_index, market_share_pct,
        fragmented_spend, portfolio_spend, ai_enabled_spend,
        portfolio_savings, ai_savings
    """
    rng = np.random.default_rng(seed)

    spend_cols = [
        "fragmented_spend",
        "portfolio_spend",
        "ai_enabled_spend",
        "portfolio_savings",
        "ai_savings",
    ]

    # --- aggregate to (category × year) ---
    agg = (
        fact_vendor
        .groupby(["vendor_category", "year"])[spend_cols]
        .sum()
        .reset_index()
    )

    rows = []
    for _, row in agg.iterrows():
        cat = row["vendor_category"]
        yr  = row["year"]

        pool = VENDOR_POOL.get(cat, [])
        if not pool:
            continue

        # Dirichlet draw: deterministic per (cat, year)
        rng_local = np.random.default_rng(seed + hash((cat, int(yr))) % (2**32))
        shares = rng_local.dirichlet(alpha=_ALPHA)

        for i, (vname, pidx) in enumerate(pool):
            s = shares[i]
            rows.append({
                "vendor_category":  cat,
                "year":             yr,
                "vendor_name":      vname,
                "category_label":   CATEGORY_LABELS.get(cat, cat),
                "price_index":      pidx,
                "market_share_pct": round(s * 100, 4),
                "fragmented_spend":  row["fragmented_spend"]  * s,
                "portfolio_spend":   row["portfolio_spend"]   * s,
                "ai_enabled_spend":  row["ai_enabled_spend"]  * s,
                "portfolio_savings": row["portfolio_savings"] * s,
                "ai_savings":        row["ai_savings"]        * s,
            })

    df = pd.DataFrame(rows)
    return df


def compute_hhi(vendor_df: pd.DataFrame, year_range: tuple[int, int]) -> pd.DataFrame:
    """
    Compute Herfindahl-Hirschman Index per vendor category over the given year range.
    HHI = Σ(market_share_i²) × 10 000  (shares in [0,1])

    Returns DataFrame with columns: vendor_category, hhi
    """
    lo, hi = year_range
    filtered = vendor_df[vendor_df["year"].between(lo, hi)]

    # Average share per vendor (across years)
    avg_shares = (
        filtered
        .groupby(["vendor_category", "vendor_name"])["market_share_pct"]
        .mean()
        .reset_index()
    )
    avg_shares["share_frac"] = avg_shares["market_share_pct"] / 100.0

    hhi = (
        avg_shares
        .groupby("vendor_category")
        .apply(lambda g: (g["share_frac"] ** 2).sum() * 10_000, include_groups=False)
        .reset_index(name="hhi")
    )
    return hhi


# ---------------------------------------------------------------------------
# Power BI export helpers
# ---------------------------------------------------------------------------

def build_fact_vendor_named_enriched(
    vendor_df: pd.DataFrame,
    fact_vendor: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enriquece vendor_df (output de generate_vendor_df) con columnas adicionales
    necesarias para Power BI:
        - fragmentation_premium  : premium por categoria desde FRAGMENTATION_PREMIUM_BY_CAT
                                   (varia por categoria, no es un promedio global)
        - portfolio_savings_rate : media por category × year desde fact_vendor
        - overpayment_usd        : spend × (price_index - 1.0)
                                   cuanto paga Chicago de mas vs el vendor benchmark

    Renombra portfolio_savings / ai_savings a portfolio_savings_usd / ai_savings_usd.

    Returns:
        DataFrame listo para exportar como fact_vendor_named_pbi.
    """
    # portfolio_savings_rate sigue viniendo de fact_vendor (es un output de la simulacion)
    fv_agg = (
        fact_vendor
        .groupby(["vendor_category", "year"])[["portfolio_savings_rate"]]
        .mean()
        .reset_index()
    )
    df = vendor_df.merge(fv_agg, on=["vendor_category", "year"], how="left")

    # fragmentation_premium: categoria-especifico, no promedio global de edificios
    df["fragmentation_premium"] = df["vendor_category"].map(FRAGMENTATION_PREMIUM_BY_CAT)

    # overpayment_usd: costo extra vs benchmark (price_index=1.00)
    # Interpretacion: cuanto mas paga Chicago por usar este vendor en lugar del mas barato
    df["overpayment_usd"] = df["fragmented_spend"] * (df["price_index"] - 1.0)

    df = df.rename(columns={
        "portfolio_savings": "portfolio_savings_usd",
        "ai_savings":        "ai_savings_usd",
    })
    for col in ["fragmented_spend", "portfolio_spend", "ai_enabled_spend",
                "portfolio_savings_usd", "ai_savings_usd", "overpayment_usd"]:
        if col in df.columns:
            df[col] = df[col].round(2)
    return df


def build_dim_vendor_benchmark(
    vendor_df_enriched: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye dim_vendor_benchmark: una fila por vendor con atributos estáticos
    para slicing en Power BI.

    Columnas clave:
        price_tier          — High (>=1.20) / Medium (1.10-1.19) / Benchmark (<1.10)
        price_premium_pct   — % que cobra el vendor sobre el precio de referencia
        spend_share_pct     — peso del vendor en el gasto total del portafolio
        score_drag          — contribución ponderada a la brecha del Vendor MRO score
        vendor_score_if_only— score hipotético si Chicago usara solo ese vendor

    Relacionar en PBI:
        dim_vendor_benchmark[vendor_name]     -> fact_vendor_named_pbi[vendor_name]
        dim_vendor_benchmark[vendor_category] -> fact_vendor_pbi[vendor_category]

    Args:
        vendor_df_enriched: output de build_fact_vendor_named_enriched().

    Returns:
        DataFrame de 50 filas (10 vendors × 5 categorías).
    """
    df = vendor_df_enriched

    dim = (
        df.groupby(["vendor_name", "vendor_category", "category_label", "price_index"])
        .agg(
            total_fragmented_spend   = ("fragmented_spend",       "sum"),
            total_portfolio_savings  = ("portfolio_savings_usd",  "sum"),
            total_ai_savings         = ("ai_savings_usd",         "sum"),
            total_overpayment        = ("overpayment_usd",        "sum"),
            avg_market_share_pct     = ("market_share_pct",       "mean"),
        )
        .reset_index()
        .sort_values(["vendor_category", "price_index"], ascending=[True, False])
    )

    def _price_tier(pi: float) -> str:
        if pi >= 1.20:  return "High (>=1.20)"
        if pi >= 1.10:  return "Medium (1.10-1.19)"
        return "Benchmark (<1.10)"

    dim["price_tier"]            = dim["price_index"].apply(_price_tier)
    dim["price_premium_pct"]     = ((dim["price_index"] - 1.0) * 100).round(2)
    total_spend                  = dim["total_fragmented_spend"].sum()
    dim["spend_share_pct"]       = (dim["total_fragmented_spend"] / total_spend * 100).round(4)
    # score_drag: cuántos puntos del Vendor MRO score (escala 0-10) arrastra este vendor
    # Derivado de: (price_index - 1.0) / 0.32 * 10 ponderado por spend_share
    dim["score_drag"]            = (
        (dim["price_index"] - 1.0) / 0.32 * 10 * dim["spend_share_pct"] / 100
    ).round(4)
    # vendor_score_if_only: score hipotético si Chicago solo usara ese vendor
    dim["vendor_score_if_only"]  = (
        ((1.0 - (dim["price_index"] - 1.0) / 0.32) * 10).clip(0, 10)
    ).round(2)

    for col in ["total_fragmented_spend", "total_portfolio_savings",
                "total_ai_savings", "total_overpayment"]:
        dim[col] = dim[col].round(0).astype(int)
    dim["avg_market_share_pct"] = dim["avg_market_share_pct"].round(3)

    return dim


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Loading fact_vendor from: {V6_PATH}")
    fv = load_vendor_fact()
    print(f"  Rows: {len(fv):,}  |  Columns: {list(fv.columns)}")

    print("\nGenerating vendor DataFrame...")
    vdf = generate_vendor_df(fv)
    print(f"  Rows: {len(vdf):,}")
    print(f"  Categories: {sorted(vdf['vendor_category'].unique())}")
    print(f"  Years: {sorted(vdf['year'].unique())}")
    print(f"  Vendors per category: {vdf.groupby('vendor_category')['vendor_name'].nunique().to_dict()}")

    yr_range = (int(vdf["year"].min()), int(vdf["year"].max()))
    hhi = compute_hhi(vdf, yr_range)
    print(f"\nHHI by category:\n{hhi.to_string(index=False)}")

    print("\n--- Spend totals ---")
    totals = vdf.groupby("vendor_category")[
        ["fragmented_spend", "portfolio_spend", "ai_enabled_spend",
         "portfolio_savings", "ai_savings"]
    ].sum()
    print(totals.to_string())
    print("\nDone.")
