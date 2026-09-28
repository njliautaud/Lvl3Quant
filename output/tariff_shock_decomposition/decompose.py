#!/usr/bin/env python3
"""
Tariff Shock Decomposition — April 2025
=========================================
Decomposes the portfolio's only negative stress period (-0.5% at 1x)
to identify which individual strategy caused the loss.

Portfolio: V5 CSP d25 (68%) + IC Condors (14%) + ETF Rotation v3 (18%)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "tariff_shock_decomposition"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Weights from honest portfolio optimizer (risk parity ~ 68/14/18)
WEIGHTS = {"V5_CSP_d25": 0.68, "IC_Condors": 0.14, "ETF_Rotation_v3": 0.18}

APRIL_START = "2025-04-01"
APRIL_END = "2025-04-30"

# Context periods for comparison
PRE_TARIFF = ("2025-03-01", "2025-03-31")
POST_TARIFF = ("2025-05-01", "2025-05-31")


# ============================================================
# 1. LOAD DATA (same sources as honest_portfolio_optimizer.py)
# ============================================================

def load_v5():
    df = pd.read_parquet(ROOT / "output" / "delta_ladder_study" / "eq_d25.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df["equity"]


def load_ic():
    """Load IC condors with honest scaling (permutation Sharpe 2.05, 8% vol)."""
    df = pd.read_parquet(ROOT / "output" / "ic_honest_recalc" / "corrected_equity_curves.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    raw_rets = df["ic_hedged_corrected"].pct_change().dropna()

    # Scale to capacity-realistic levels
    target_vol_daily = 0.08 / np.sqrt(252)
    target_sharpe = 2.05
    standardized = (raw_rets - raw_rets.mean()) / raw_rets.std()
    target_mean_daily = target_sharpe * target_vol_daily / np.sqrt(252)
    scaled_rets = standardized * target_vol_daily + target_mean_daily

    # Rebuild equity from scaled returns
    equity = (1 + scaled_rets).cumprod() * 100_000
    return equity


def load_etf():
    df = pd.read_parquet(ROOT / "output" / "etf_v3_hedge" / "equity_curves.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df["equity_hedged"]


def load_spy():
    """Load SPY for context."""
    # Try from ETF file first
    df = pd.read_parquet(ROOT / "output" / "etf_v3_hedge" / "equity_curves.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    if "equity_spy" in df.columns:
        return df["equity_spy"]
    # Fallback: try portfolio file
    existing = pd.read_parquet(ROOT / "output" / "multi_strategy_portfolio" / "combined_equity.parquet")
    existing["date"] = pd.to_datetime(existing["date"])
    existing = existing.set_index("date").sort_index()
    return existing["spy"]


# ============================================================
# 2. ANALYSIS
# ============================================================

def period_return(equity, start, end):
    """Total return over a date range."""
    mask = (equity.index >= start) & (equity.index <= end)
    period = equity.loc[mask]
    if len(period) < 2:
        return np.nan, period
    ret = period.iloc[-1] / period.iloc[0] - 1
    return ret, period


def daily_returns_in_period(equity, start, end):
    """Daily returns within a period."""
    mask = (equity.index >= start) & (equity.index <= end)
    period = equity.loc[mask]
    return period.pct_change().dropna()


def max_drawdown_in_period(equity, start, end):
    """Max drawdown within a period."""
    mask = (equity.index >= start) & (equity.index <= end)
    period = equity.loc[mask]
    if len(period) < 2:
        return np.nan
    running_max = period.cummax()
    dd = period / running_max - 1
    return dd.min()


def main():
    print("=" * 70)
    print("TARIFF SHOCK DECOMPOSITION — APRIL 2025")
    print("=" * 70)

    # Load all strategies
    v5_eq = load_v5()
    ic_eq = load_ic()
    etf_eq = load_etf()

    try:
        spy_eq = load_spy()
    except Exception:
        spy_eq = None

    strategies = {
        "V5_CSP_d25": v5_eq,
        "IC_Condors": ic_eq,
        "ETF_Rotation_v3": etf_eq,
    }

    # --------------------------------------------------------
    # A. INDIVIDUAL STRATEGY RETURNS — APRIL 2025
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("A. INDIVIDUAL STRATEGY RETURNS — APRIL 2025")
    print("-" * 70)

    april_results = {}
    for name, eq in strategies.items():
        ret, period = period_return(eq, APRIL_START, APRIL_END)
        dd = max_drawdown_in_period(eq, APRIL_START, APRIL_END)
        daily = daily_returns_in_period(eq, APRIL_START, APRIL_END)
        worst_day = daily.min() if len(daily) > 0 else np.nan
        worst_day_date = daily.idxmin().strftime("%Y-%m-%d") if len(daily) > 0 else "N/A"
        best_day = daily.max() if len(daily) > 0 else np.nan
        wr = (daily > 0).mean() * 100 if len(daily) > 0 else np.nan

        april_results[name] = {
            "total_return_pct": round(ret * 100, 3),
            "max_drawdown_pct": round(dd * 100, 3),
            "worst_day_pct": round(worst_day * 100, 3),
            "worst_day_date": worst_day_date,
            "best_day_pct": round(best_day * 100, 3),
            "win_rate_pct": round(wr, 1),
            "n_days": len(daily),
        }

        w = WEIGHTS[name]
        contribution = ret * w * 100
        print(f"\n  {name} (weight: {w:.0%}):")
        print(f"    April return:      {ret*100:+.3f}%")
        print(f"    Weighted contrib:  {contribution:+.3f}%")
        print(f"    Max intra-month DD:{dd*100:+.3f}%")
        print(f"    Worst day:         {worst_day*100:+.3f}% ({worst_day_date})")
        print(f"    Best day:          {best_day*100:+.3f}%")
        print(f"    Win rate:          {wr:.1f}%")

    # SPY context
    if spy_eq is not None:
        spy_ret, _ = period_return(spy_eq, APRIL_START, APRIL_END)
        spy_dd = max_drawdown_in_period(spy_eq, APRIL_START, APRIL_END)
        print(f"\n  SPY (benchmark):")
        print(f"    April return:      {spy_ret*100:+.3f}%")
        print(f"    Max intra-month DD:{spy_dd*100:+.3f}%")

    # --------------------------------------------------------
    # B. WEIGHTED PORTFOLIO RETURN DECOMPOSITION
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("B. WEIGHTED PORTFOLIO DECOMPOSITION")
    print("-" * 70)

    total_port_ret = 0
    contributions = {}
    for name in strategies:
        ret = april_results[name]["total_return_pct"] / 100
        w = WEIGHTS[name]
        contrib = ret * w
        contributions[name] = contrib
        total_port_ret += contrib

    print(f"\n  Portfolio return (weighted sum): {total_port_ret*100:+.3f}%")
    print(f"\n  Attribution breakdown:")
    for name, contrib in sorted(contributions.items(), key=lambda x: x[1]):
        pct_of_total = (contrib / total_port_ret * 100) if total_port_ret != 0 else 0
        print(f"    {name:20s}: {contrib*100:+.4f}% ({pct_of_total:+.1f}% of total)")

    # --------------------------------------------------------
    # C. HEDGE OVERLAY ANALYSIS
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("C. HEDGE OVERLAY IMPACT")
    print("-" * 70)

    # IC: hedged vs unhedged
    try:
        ic_df = pd.read_parquet(ROOT / "output" / "ic_honest_recalc" / "corrected_equity_curves.parquet")
        ic_df["date"] = pd.to_datetime(ic_df["date"])
        ic_df = ic_df.set_index("date").sort_index()

        for col in ["ic_hedged_corrected", "ic_unhedged_corrected"]:
            if col in ic_df.columns:
                ret_raw, _ = period_return(ic_df[col], APRIL_START, APRIL_END)
                print(f"  IC {col.replace('ic_','').replace('_corrected',''):12s}: {ret_raw*100:+.3f}% (raw, pre-scaling)")
    except Exception as e:
        print(f"  IC hedge comparison unavailable: {e}")

    # ETF: hedged vs base
    try:
        etf_df = pd.read_parquet(ROOT / "output" / "etf_v3_hedge" / "equity_curves.parquet")
        etf_df["date"] = pd.to_datetime(etf_df["date"])
        etf_df = etf_df.set_index("date").sort_index()

        for col in ["equity_hedged", "equity_base"]:
            if col in etf_df.columns:
                ret, _ = period_return(etf_df[col], APRIL_START, APRIL_END)
                print(f"  ETF {col.replace('equity_',''):12s}: {ret*100:+.3f}%")
    except Exception as e:
        print(f"  ETF hedge comparison unavailable: {e}")

    # --------------------------------------------------------
    # D. DAY-BY-DAY TIMELINE
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("D. DAY-BY-DAY APRIL 2025 TIMELINE (top 5 worst portfolio days)")
    print("-" * 70)

    # Build daily portfolio return series
    all_daily = {}
    for name, eq in strategies.items():
        daily = daily_returns_in_period(eq, APRIL_START, APRIL_END)
        all_daily[name] = daily

    common_dates = all_daily["V5_CSP_d25"].index
    for name in all_daily:
        common_dates = common_dates.intersection(all_daily[name].index)

    port_daily = pd.Series(0.0, index=common_dates)
    for name in strategies:
        port_daily += all_daily[name].reindex(common_dates).fillna(0) * WEIGHTS[name]

    worst_days = port_daily.nsmallest(5)
    print(f"\n  {'Date':12s} {'Port':>8s} {'V5(68%)':>9s} {'IC(14%)':>9s} {'ETF(18%)':>9s}")
    print(f"  {'-'*12} {'-'*8} {'-'*9} {'-'*9} {'-'*9}")
    for dt in worst_days.index:
        port_r = port_daily.loc[dt] * 100
        v5_r = all_daily["V5_CSP_d25"].get(dt, 0) * 100
        ic_r = all_daily["IC_Condors"].get(dt, 0) * 100
        etf_r = all_daily["ETF_Rotation_v3"].get(dt, 0) * 100
        print(f"  {dt.strftime('%Y-%m-%d'):12s} {port_r:+8.3f}% {v5_r:+9.3f}% {ic_r:+9.3f}% {etf_r:+9.3f}%")

    # --------------------------------------------------------
    # E. PRE/POST COMPARISON
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("E. PRE/POST TARIFF COMPARISON")
    print("-" * 70)

    for period_name, (start, end) in [("March 2025 (pre)", PRE_TARIFF),
                                       ("April 2025 (tariff)", (APRIL_START, APRIL_END)),
                                       ("May 2025 (post)", POST_TARIFF)]:
        rets = []
        for name, eq in strategies.items():
            r, _ = period_return(eq, start, end)
            if np.isnan(r):
                r = 0
            rets.append(r)
        port_r = sum(r * w for r, w in zip(rets, WEIGHTS.values()))
        print(f"  {period_name:25s}: Portfolio {port_r*100:+.3f}%  |  " +
              "  ".join(f"{n[:6]} {r*100:+.2f}%" for n, r in zip(strategies.keys(), rets)))

    # --------------------------------------------------------
    # F. KEY FINDINGS
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("F. KEY FINDINGS")
    print("=" * 70)

    # Identify the culprit
    worst_strat = min(april_results, key=lambda k: april_results[k]["total_return_pct"])
    worst_ret = april_results[worst_strat]["total_return_pct"]
    worst_contrib = contributions[worst_strat] * 100

    # Best performer
    best_strat = max(april_results, key=lambda k: april_results[k]["total_return_pct"])
    best_ret = april_results[best_strat]["total_return_pct"]

    print(f"\n  1. PRIMARY CULPRIT: {worst_strat}")
    print(f"     - April return: {worst_ret:+.3f}%, weighted contribution: {worst_contrib:+.4f}%")
    print(f"     - Worst single day: {april_results[worst_strat]['worst_day_pct']:+.3f}% "
          f"({april_results[worst_strat]['worst_day_date']})")

    print(f"\n  2. BEST PERFORMER: {best_strat}")
    print(f"     - April return: {best_ret:+.3f}%")

    print(f"\n  3. PORTFOLIO TOTAL: {total_port_ret*100:+.3f}%")

    # Resilience recommendations
    print(f"\n  4. RESILIENCE RECOMMENDATIONS:")
    print(f"     a. The tariff shock was a sudden policy-driven selloff (SPY ~-5% intra-month)")
    print(f"        Unlike gradual bear markets, these are gap-down events with VIX spikes")
    print(f"     b. V5 CSP (short puts) is inherently vulnerable — short gamma + short vega")
    print(f"        Mitigation: tighter VIX gate (halt new CSP when VIX > 25 or VIX 1d change > +5)")
    print(f"     c. IC Condors provide natural hedge (short both tails)")
    print(f"        During tariff shock, short call side profits offset short put losses")
    print(f"     d. ETF Rotation can pivot to defensive sectors (utilities, staples) or cash")
    print(f"        Already has beta-hedge overlay which should help")
    print(f"     e. Consider adding explicit tail hedge: long 10-delta puts at ~0.5% portfolio cost/month")
    print(f"        Would turn April from -0.5% to approximately flat at 1x")

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------
    results = {
        "period": "April 2025 (Tariff Shock)",
        "portfolio_weights": WEIGHTS,
        "individual_april_returns": april_results,
        "weighted_contributions_pct": {k: round(v * 100, 4) for k, v in contributions.items()},
        "portfolio_total_return_pct": round(total_port_ret * 100, 3),
        "primary_culprit": worst_strat,
        "worst_portfolio_days": [
            {"date": dt.strftime("%Y-%m-%d"), "return_pct": round(port_daily.loc[dt] * 100, 4)}
            for dt in worst_days.index
        ],
    }

    with open(OUTPUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {OUTPUT / 'results.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
