#!/usr/bin/env python3
"""
BPS Honest Sharpe Estimate — Vol-Skew Credit Haircut Analysis
==============================================================
Applies a vol-skew pricing haircut to the corrected V2 backtest (15% BA)
to estimate the "fully honest" Sharpe after accounting for BS model bias.

KNOWN BIASES:
1. Cost model — FIXED in bps_full_stack_v2.py (per-leg BA, correct formula)
2. Vol skew — THIS SCRIPT. Flat-IV BS ignores put skew, overestimates net
   credit. Live validation showed:
   - Tier 1 (large-cap): +7.4% median overestimate
   - Tier 2 (mid-cap): +39.4% median overestimate

APPROACH:
- Load trade-level results from V2 backtest at 15% BA
- Apply credit haircut to each trade's net_credit (reduce by X%)
- Recompute realized_pnl = haircut_credit - original_loss_component
- Rebuild equity curve from adjusted trade P&Ls
- Compute Sharpe at each haircut level
- Find break-even haircut (Sharpe = 0)

Output: output/bps_honest_sharpe/
"""

import json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
INPUT = ROOT / "output" / "bps_full_stack_v2"
OUTPUT = ROOT / "output" / "bps_honest_sharpe"
OUTPUT.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 100_000


def load_trades_and_equity():
    """Load trade-level results and equity curve from V2 backtest at 15% BA."""
    trades = pd.read_parquet(INPUT / "trades_BA_15pct.parquet")
    equity = pd.read_parquet(INPUT / "eq_BA_15pct.parquet")
    trades["open_date"] = pd.to_datetime(trades["open_date"])
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    equity["date"] = pd.to_datetime(equity["date"])
    return trades, equity


def apply_haircut(trades, haircut_pct, tiered=False,
                  tier1_haircut=None, tier2_haircut=None):
    """
    Apply credit haircut to trades.

    The haircut reduces net_credit, which means the trade collects less premium.
    The loss component (when stock breaches) stays the same — we only overestimated
    the credit, not the loss.

    For each trade:
      original_loss = net_credit - realized_pnl  (total costs + any assignment loss)
      adjusted_credit = net_credit * (1 - haircut)
      adjusted_pnl = adjusted_credit - original_loss
    """
    df = trades.copy()

    if tiered and tier1_haircut is not None and tier2_haircut is not None:
        # Different haircut by tier
        haircut = df["tier"].map({
            "tier1": tier1_haircut / 100.0,
            "tier2": tier2_haircut / 100.0,
        }).fillna(haircut_pct / 100.0)
    else:
        haircut = haircut_pct / 100.0

    # The loss component is everything the trade lost beyond the credit
    # original_loss = net_credit - realized_pnl
    # This includes: close costs, BA on close, assignment losses
    original_loss = df["net_credit"] - df["realized_pnl"]

    # Reduce the credit by the haircut fraction
    df["adjusted_credit"] = df["net_credit"] * (1.0 - haircut)

    # New P&L = adjusted credit - same loss component
    df["adjusted_pnl"] = df["adjusted_credit"] - original_loss

    return df


def build_equity_curve(trades_adj, starting_cap=STARTING_CAPITAL):
    """
    Rebuild equity curve from adjusted trade P&Ls.

    We allocate each trade's P&L to its close_date and accumulate.
    """
    daily_pnl = trades_adj.groupby("close_date")["adjusted_pnl"].sum().sort_index()

    # Build cumulative equity
    all_dates = sorted(daily_pnl.index)
    equity = starting_cap + daily_pnl.cumsum()

    eq_df = pd.DataFrame({
        "date": equity.index,
        "equity": equity.values,
    }).reset_index(drop=True)

    return eq_df


def compute_metrics(eq_df, label, starting_cap=STARTING_CAPITAL):
    """Compute Sharpe, Sortino, CAGR, max DD from equity curve."""
    eq = eq_df.copy().sort_values("date").reset_index(drop=True)
    eq["ret"] = eq["equity"].pct_change()
    rets = eq["ret"].dropna()

    if len(rets) < 10 or rets.std() == 0:
        return {"label": label, "sharpe": 0.0, "sortino": 0.0, "cagr_pct": 0.0,
                "max_dd_pct": 0.0, "final_equity": float(eq["equity"].iloc[-1]),
                "profit_factor": 0.0, "daily_wr_pct": 0.0}

    total_days = (eq["date"].iloc[-1] - eq["date"].iloc[0]).days
    total_years = max(total_days / 365.25, 0.01)
    total_return = eq["equity"].iloc[-1] / starting_cap
    cagr = (total_return ** (1 / total_years)) - 1 if total_return > 0 else -1.0

    sharpe = float(rets.mean() / rets.std() * np.sqrt(252))

    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std() * np.sqrt(252)) if len(downside) > 5 and downside.std() > 0 else 0.0

    eq["peak"] = eq["equity"].cummax()
    eq["dd"] = (eq["equity"] - eq["peak"]) / eq["peak"]
    max_dd = float(eq["dd"].min())

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    wr = float(len(rets[rets > 0]) / len(rets) * 100)
    wins = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = float(wins / losses) if losses > 0 else float("inf")

    # Per-year breakdown
    eq["year"] = eq["date"].dt.year
    per_year = {}
    for yr, grp in eq.groupby("year"):
        if len(grp) < 5:
            continue
        yr_ret = grp["equity"].iloc[-1] / grp["equity"].iloc[0] - 1
        yr_rets = grp["ret"].dropna()
        yr_sharpe = float(yr_rets.mean() / yr_rets.std() * np.sqrt(252)) if yr_rets.std() > 0 else 0.0
        yr_dd = float(((grp["equity"] / grp["equity"].cummax()) - 1).min())
        per_year[int(yr)] = {
            "return_pct": round(yr_ret * 100, 2),
            "sharpe": round(yr_sharpe, 2),
            "max_dd_pct": round(yr_dd * 100, 2),
        }

    return {
        "label": label,
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 2),
        "daily_wr_pct": round(wr, 1),
        "profit_factor": round(pf, 2),
        "final_equity": round(float(eq["equity"].iloc[-1]), 2),
        "total_return_pct": round((total_return - 1) * 100, 2),
        "per_year": per_year,
    }


def find_breakeven_haircut(trades, max_haircut=60.0):
    """
    Find the uniform haircut percentage where total P&L = 0.

    Using total P&L rather than Sharpe because once equity goes negative,
    the Sharpe ratio becomes nonsensical (negative denominators, bizarre returns).
    Total P&L crossing zero is the clean definition of break-even.
    """
    def total_pnl_at_haircut(h):
        original_loss = trades["net_credit"] - trades["realized_pnl"]
        adj_credit = trades["net_credit"] * (1 - h / 100.0)
        adj_pnl = adj_credit - original_loss
        return float(adj_pnl.sum())

    pnl_0 = total_pnl_at_haircut(0.0)
    pnl_max = total_pnl_at_haircut(max_haircut)

    if pnl_0 <= 0:
        return 0.0
    if pnl_max > 0:
        return max_haircut

    try:
        breakeven = brentq(total_pnl_at_haircut, 0.0, max_haircut, xtol=0.01)
        return round(breakeven, 1)
    except Exception:
        # Fallback: linear scan
        for h in np.arange(0, max_haircut + 0.5, 0.5):
            if total_pnl_at_haircut(h) <= 0:
                return round(h, 1)
        return max_haircut


def main():
    t0 = time.time()
    print("=" * 80)
    print("BPS HONEST SHARPE ESTIMATE")
    print("Vol-Skew Credit Haircut on V2 Backtest (15% BA)")
    print("=" * 80)

    # Load data
    trades, equity_orig = load_trades_and_equity()
    print(f"\nLoaded {len(trades)} trades from V2 backtest (15% BA)")
    print(f"  Tier 1: {(trades['tier'] == 'tier1').sum()} trades")
    print(f"  Tier 2: {(trades['tier'] == 'tier2').sum()} trades")
    print(f"  Original final equity: ${equity_orig['equity'].iloc[-1]:,.0f}")

    # ═══════════════════════════════════════════════════════════
    # 1. UNIFORM HAIRCUT SWEEP
    # ═══════════════════════════════════════════════════════════
    uniform_haircuts = [0, 5, 7.5, 10, 12.5, 15, 20, 25, 30]
    uniform_results = {}

    print(f"\n{'='*80}")
    print("UNIFORM CREDIT HAIRCUT SWEEP")
    print(f"{'='*80}")
    print(f"{'Haircut':>10} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'PF':>6} {'Final$':>14}")
    print("-" * 75)

    for h in uniform_haircuts:
        adj = apply_haircut(trades, h)
        eq = build_equity_curve(adj)
        m = compute_metrics(eq, f"haircut_{h}pct")
        uniform_results[h] = m

        print(f"{h:>9.1f}% {m['sharpe']:>8.2f} {m['sortino']:>8.2f} "
              f"{m['cagr_pct']:>7.1f}% {m['max_dd_pct']:>7.1f}% "
              f"{m['profit_factor']:>6.2f} ${m['final_equity']:>13,.0f}")

        # Save equity curve
        eq.to_parquet(OUTPUT / f"eq_haircut_{h}pct.parquet", index=False)

    # ═══════════════════════════════════════════════════════════
    # 2. TIERED HAIRCUT (live validation medians)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("TIERED HAIRCUT (from live chain validation)")
    print("  Tier 1: 7.4% (median BS overestimate for large-cap)")
    print("  Tier 2: 39.4% (median BS overestimate for mid-cap)")
    print(f"{'='*80}")

    adj_tiered = apply_haircut(trades, 0, tiered=True,
                                tier1_haircut=7.4, tier2_haircut=39.4)
    eq_tiered = build_equity_curve(adj_tiered)
    m_tiered = compute_metrics(eq_tiered, "tiered_7.4_39.4")

    print(f"\n  Sharpe:      {m_tiered['sharpe']:.2f}")
    print(f"  Sortino:     {m_tiered['sortino']:.2f}")
    print(f"  CAGR:        {m_tiered['cagr_pct']:.1f}%")
    print(f"  Max DD:      {m_tiered['max_dd_pct']:.1f}%")
    print(f"  PF:          {m_tiered['profit_factor']:.2f}")
    print(f"  Final $:     ${m_tiered['final_equity']:,.0f}")

    # Per-year for tiered
    print(f"\n  Per-Year Breakdown:")
    print(f"    {'Year':>6} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print(f"    {'-'*34}")
    for yr in sorted(m_tiered["per_year"].keys()):
        y = m_tiered["per_year"][yr]
        print(f"    {yr:>6} {y['return_pct']:>7.1f}% {y['sharpe']:>8.2f} {y['max_dd_pct']:>7.1f}%")

    eq_tiered.to_parquet(OUTPUT / "eq_tiered_7.4_39.4.parquet", index=False)

    # ═══════════════════════════════════════════════════════════
    # 3. TIERED WITH MEAN (conservative) — mean overestimates
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("TIERED HAIRCUT (CONSERVATIVE — using means)")
    print("  Tier 1: 12.5% (mean BS overestimate for large-cap)")
    print("  Tier 2: 43.9% (mean BS overestimate for mid-cap)")
    print(f"{'='*80}")

    adj_tiered_cons = apply_haircut(trades, 0, tiered=True,
                                     tier1_haircut=12.5, tier2_haircut=43.9)
    eq_tiered_cons = build_equity_curve(adj_tiered_cons)
    m_tiered_cons = compute_metrics(eq_tiered_cons, "tiered_12.5_43.9")

    print(f"\n  Sharpe:      {m_tiered_cons['sharpe']:.2f}")
    print(f"  Sortino:     {m_tiered_cons['sortino']:.2f}")
    print(f"  CAGR:        {m_tiered_cons['cagr_pct']:.1f}%")
    print(f"  Max DD:      {m_tiered_cons['max_dd_pct']:.1f}%")
    print(f"  PF:          {m_tiered_cons['profit_factor']:.2f}")
    print(f"  Final $:     ${m_tiered_cons['final_equity']:,.0f}")

    eq_tiered_cons.to_parquet(OUTPUT / "eq_tiered_12.5_43.9.parquet", index=False)

    # ═══════════════════════════════════════════════════════════
    # 4. FIND BREAK-EVEN HAIRCUT
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("BREAK-EVEN HAIRCUT (uniform haircut where Sharpe = 0)")
    print(f"{'='*80}")

    breakeven = find_breakeven_haircut(trades)
    print(f"\n  Break-even haircut: {breakeven:.1f}%")
    print(f"  (At this uniform haircut, total P&L = 0 — strategy stops making money)")

    # ═══════════════════════════════════════════════════════════
    # 5. TIER-1 ONLY ANALYSIS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("TIER-1 ONLY (large-cap only, 7.4% median haircut)")
    print(f"{'='*80}")

    trades_t1 = trades[trades["tier"] == "tier1"].copy()
    adj_t1 = apply_haircut(trades_t1, 7.4)
    eq_t1 = build_equity_curve(adj_t1)
    m_t1 = compute_metrics(eq_t1, "tier1_only_7.4pct")

    print(f"\n  Trades:      {len(trades_t1)}")
    print(f"  Sharpe:      {m_t1['sharpe']:.2f}")
    print(f"  Sortino:     {m_t1['sortino']:.2f}")
    print(f"  CAGR:        {m_t1['cagr_pct']:.1f}%")
    print(f"  Max DD:      {m_t1['max_dd_pct']:.1f}%")
    print(f"  PF:          {m_t1['profit_factor']:.2f}")
    print(f"  Final $:     ${m_t1['final_equity']:,.0f}")

    # ═══════════════════════════════════════════════════════════
    # 6. SENSITIVITY: WHAT IF TIER-2 IS EXCLUDED?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("IMPACT OF TIER-2 EXCLUSION")
    print(f"{'='*80}")

    # No haircut, tier1 only
    adj_t1_nohaircut = apply_haircut(trades_t1, 0)
    eq_t1_nh = build_equity_curve(adj_t1_nohaircut)
    m_t1_nh = compute_metrics(eq_t1_nh, "tier1_only_0pct")

    print(f"\n  {'Scenario':<35} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8}")
    print(f"  {'-'*65}")
    print(f"  {'Full universe, 0% haircut':<35} {uniform_results[0]['sharpe']:>8.2f} "
          f"{uniform_results[0]['cagr_pct']:>7.1f}% {uniform_results[0]['max_dd_pct']:>7.1f}%")
    print(f"  {'Tier-1 only, 0% haircut':<35} {m_t1_nh['sharpe']:>8.2f} "
          f"{m_t1_nh['cagr_pct']:>7.1f}% {m_t1_nh['max_dd_pct']:>7.1f}%")
    print(f"  {'Tier-1 only, 7.4% haircut':<35} {m_t1['sharpe']:>8.2f} "
          f"{m_t1['cagr_pct']:>7.1f}% {m_t1['max_dd_pct']:>7.1f}%")
    print(f"  {'Tiered (7.4% / 39.4%)':<35} {m_tiered['sharpe']:>8.2f} "
          f"{m_tiered['cagr_pct']:>7.1f}% {m_tiered['max_dd_pct']:>7.1f}%")
    print(f"  {'Tiered conservative (12.5%/43.9%)':<35} {m_tiered_cons['sharpe']:>8.2f} "
          f"{m_tiered_cons['cagr_pct']:>7.1f}% {m_tiered_cons['max_dd_pct']:>7.1f}%")

    # ═══════════════════════════════════════════════════════════
    # 7. SUMMARY
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("HONEST SHARPE SUMMARY")
    print(f"{'='*80}")
    print(f"\n  BASELINE (V2 backtest, 15% BA, no vol-skew adjustment):")
    print(f"    Sharpe = {uniform_results[0]['sharpe']:.2f}, CAGR = {uniform_results[0]['cagr_pct']:.1f}%")

    print(f"\n  BEST ESTIMATE (tiered haircut from live validation):")
    print(f"    Sharpe = {m_tiered['sharpe']:.2f}, CAGR = {m_tiered['cagr_pct']:.1f}%")
    print(f"    (Tier 1: 7.4% haircut, Tier 2: 39.4% haircut)")

    print(f"\n  CONSERVATIVE ESTIMATE (mean overestimates):")
    print(f"    Sharpe = {m_tiered_cons['sharpe']:.2f}, CAGR = {m_tiered_cons['cagr_pct']:.1f}%")
    print(f"    (Tier 1: 12.5% haircut, Tier 2: 43.9% haircut)")

    print(f"\n  UNIFORM 7.5% HAIRCUT (simple approach):")
    print(f"    Sharpe = {uniform_results[7.5]['sharpe']:.2f}, CAGR = {uniform_results[7.5]['cagr_pct']:.1f}%")

    print(f"\n  BREAK-EVEN: Uniform haircut of {breakeven:.1f}% zeroes total P&L")
    print(f"  SAFETY MARGIN: {breakeven:.1f}% break-even vs 7.4% median real-world overestimate")
    if breakeven > 0:
        print(f"                 = {breakeven / 7.4:.1f}x safety factor")

    # ═══════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = {
        "generated": pd.Timestamp.now().isoformat(),
        "description": "Vol-skew credit haircut applied to V2 backtest (15% BA, corrected cost model)",
        "baseline": {
            "note": "V2 backtest at 15% BA with corrected per-leg cost model, NO vol-skew adjustment",
            "sharpe": uniform_results[0]["sharpe"],
            "cagr_pct": uniform_results[0]["cagr_pct"],
            "sortino": uniform_results[0]["sortino"],
            "max_dd_pct": uniform_results[0]["max_dd_pct"],
        },
        "uniform_haircut_sweep": {
            str(h): uniform_results[h] for h in uniform_haircuts
        },
        "tiered_best_estimate": {
            "tier1_haircut_pct": 7.4,
            "tier2_haircut_pct": 39.4,
            "source": "median BS credit overestimate from live chain validation",
            "metrics": m_tiered,
        },
        "tiered_conservative": {
            "tier1_haircut_pct": 12.5,
            "tier2_haircut_pct": 43.9,
            "source": "mean BS credit overestimate from live chain validation",
            "metrics": m_tiered_cons,
        },
        "tier1_only_with_haircut": {
            "haircut_pct": 7.4,
            "n_trades": len(trades_t1),
            "metrics": m_t1,
        },
        "breakeven_uniform_haircut_pct": breakeven,
        "safety_factor": round(breakeven / 7.4, 1),
        "honest_sharpe_best_estimate": m_tiered["sharpe"],
        "honest_sharpe_conservative": m_tiered_cons["sharpe"],
        "honest_sharpe_uniform_7_5": uniform_results[7.5]["sharpe"],
    }

    results = convert(results)

    with open(OUTPUT / "honest_sharpe_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Results saved to {OUTPUT}")


if __name__ == "__main__":
    main()
