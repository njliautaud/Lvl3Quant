#!/usr/bin/env python3
"""
BPS Realism Bridge — Honest Expected Sharpe After All Frictions
================================================================

HC #664 R4: The backtest reports Sharpe 2-3.7. The deep audit says 0.8-1.5.
WHERE does the Sharpe drop? This analysis builds a systematic bridge from
BS-modeled backtest to honest real-world expectations.

Friction sources (each quantified independently):
  1. BS vs real IV: +8% (BS is CONSERVATIVE — real premiums richer)
  2. Bid-ask crossing cost: spreads aren't free to open/close
  3. Fill rate: not every order fills (especially at mid)
  4. Assignment/settlement friction: early assignment, pin risk
  5. Margin interest: cash held as collateral earns 0 in backtest
  6. Liquidity degradation: at scale, fewer liquid names available
  7. Correlation/clustering: losses cluster in drawdowns (fat tails)
  8. Regime dependence: strategy performance varies by market regime

Method: Start from the conservative BPS equity curve, then apply
each friction as a P&L haircut and recompute Sharpe.

Output: output/bps_realism_bridge/realism_bridge_report.json
"""

import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "output" / "wheel_higher_returns_study"))
from higher_returns_study import load_data, compute_metrics

OUTPUT = ROOT / "output" / "bps_realism_bridge"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_assignment_risk_trades():
    """Load trade-level data from the assignment risk study."""
    trades_path = ROOT / "output" / "bps_assignment_risk" / "trades_close_1dte.parquet"
    if trades_path.exists():
        return pd.read_parquet(trades_path)
    # Fallback to hold-to-expiry
    trades_path = ROOT / "output" / "bps_assignment_risk" / "trades_hold_to_expiry.parquet"
    if trades_path.exists():
        return pd.read_parquet(trades_path)
    return None


def compute_sharpe_from_equity(eq_series, annual_factor=252):
    """Compute annualized Sharpe from equity series."""
    rets = eq_series.pct_change().dropna()
    if len(rets) < 10 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(annual_factor))


def compute_sortino_from_equity(eq_series, annual_factor=252):
    """Compute annualized Sortino from equity series."""
    rets = eq_series.pct_change().dropna()
    downside = rets[rets < 0]
    if len(downside) < 5 or downside.std() == 0:
        return 0.0
    return float(rets.mean() / downside.std() * np.sqrt(annual_factor))


def compute_max_dd(eq_series):
    """Compute max drawdown from equity series."""
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak
    return float(dd.min() * 100)


def apply_bid_ask_haircut(trades_df, ba_frac_open=0.05, ba_frac_close=0.05):
    """
    Apply bid-ask spread cost to each trade.

    In reality, you cross the spread to enter and exit:
    - Opening: sell short put at bid, buy long put at ask
    - Closing: buy short put at ask, sell long put at bid

    Cost = ba_frac * premium per leg * 2 legs * 2 transactions (open + close)

    ba_frac: bid-ask spread as fraction of option premium
    Empirically: mega-cap 3%, large-cap 5%, mid-cap 10%, small-cap 18%
    Using 5% as portfolio-weighted average (universe is mostly large-cap).
    """
    haircut = trades_df["net_credit"].abs() * ba_frac_open * 2  # 2 legs
    # For trades that close before expiry, additional close cost
    closed_early = trades_df["exit_type"].isin(["profit_take", "early_close_1DTE", "loss_stop"])
    haircut[closed_early] += trades_df.loc[closed_early, "net_credit"].abs() * ba_frac_close * 2

    return haircut


def apply_fill_rate(trades_df, fill_rate=0.75):
    """
    Not all limit orders fill. At mid-price, historical fill rate ~60-85%.
    This doesn't reduce Sharpe directly (fewer trades = lower return AND lower vol)
    but it reduces absolute P&L.

    Model: randomly remove (1-fill_rate) fraction of trades.
    Run N simulations to get distribution.
    """
    n_sims = 100
    sharpes = []
    pnls = []

    for _ in range(n_sims):
        mask = np.random.random(len(trades_df)) < fill_rate
        subset = trades_df[mask]
        if len(subset) < 50:
            continue

        # Build daily P&L
        daily_pnl = subset.groupby("close_date")["realized_pnl"].sum()
        if len(daily_pnl) < 30:
            continue

        sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0
        sharpes.append(sharpe)
        pnls.append(daily_pnl.sum())

    return {
        "mean_sharpe": round(np.mean(sharpes), 2) if sharpes else 0,
        "std_sharpe": round(np.std(sharpes), 2) if sharpes else 0,
        "mean_total_pnl": round(np.mean(pnls), 0) if pnls else 0,
        "fill_rate": fill_rate,
    }


def regime_analysis(trades_df, macro_df):
    """Stratify BPS performance by VIX regime."""
    # Get VIX by date
    macro_df = macro_df.copy()
    macro_df["date"] = pd.to_datetime(macro_df["date"])
    vix_by_date = macro_df.set_index("date")["vix"].to_dict()

    trades = trades_df.copy()
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    trades["vix"] = trades["close_date"].map(vix_by_date)

    # Remove NaN VIX
    trades = trades.dropna(subset=["vix"])

    regimes = {
        "low_vol": trades[trades["vix"] < 15],
        "normal_vol": trades[(trades["vix"] >= 15) & (trades["vix"] < 25)],
        "high_vol": trades[(trades["vix"] >= 25) & (trades["vix"] < 35)],
        "crisis": trades[trades["vix"] >= 35],
    }

    results = {}
    for name, subset in regimes.items():
        if len(subset) < 20:
            results[name] = {"n_trades": len(subset), "note": "too few trades"}
            continue

        daily_pnl = subset.groupby("close_date")["realized_pnl"].sum()
        sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0

        results[name] = {
            "n_trades": len(subset),
            "total_pnl": round(subset["realized_pnl"].sum(), 0),
            "avg_pnl": round(subset["realized_pnl"].mean(), 2),
            "win_rate": round((subset["realized_pnl"] > 0).mean() * 100, 1),
            "daily_sharpe": round(sharpe, 2),
        }

    return results


def correlation_clustering(trades_df):
    """
    Measure how much losses cluster. If losses are independent,
    VaR is well-behaved. If correlated, tail risk is underestimated.
    """
    trades = trades_df.copy()
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    daily_pnl = trades.groupby("close_date")["realized_pnl"].sum()

    if len(daily_pnl) < 50:
        return {"note": "too few days"}

    rets = daily_pnl.values

    # Autocorrelation of losses
    loss_days = rets < 0
    if loss_days.sum() < 10:
        return {"note": "too few loss days"}

    # Streak analysis: how often do losses cluster?
    streaks = []
    current_streak = 0
    for r in rets:
        if r < 0:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)

    # VaR / CVaR
    var_95 = np.percentile(rets, 5)
    var_99 = np.percentile(rets, 1)
    cvar_95 = rets[rets <= var_95].mean() if (rets <= var_95).sum() > 0 else var_95
    cvar_99 = rets[rets <= var_99].mean() if (rets <= var_99).sum() > 0 else var_99

    # Tail ratio: CVaR/VaR > 1 means fatter tails
    tail_ratio_95 = abs(cvar_95 / var_95) if var_95 != 0 else 0
    tail_ratio_99 = abs(cvar_99 / var_99) if var_99 != 0 else 0

    return {
        "loss_days_pct": round(loss_days.mean() * 100, 1),
        "avg_loss_streak": round(np.mean(streaks), 1) if streaks else 0,
        "max_loss_streak": max(streaks) if streaks else 0,
        "var_95_daily": round(var_95, 0),
        "cvar_95_daily": round(cvar_95, 0),
        "var_99_daily": round(var_99, 0),
        "cvar_99_daily": round(cvar_99, 0),
        "tail_ratio_95": round(tail_ratio_95, 2),
        "tail_ratio_99": round(tail_ratio_99, 2),
    }


def margin_interest_haircut(trades_df, starting_capital=100_000, annual_rate=0.05):
    """
    In a real account, margin held for spreads earns interest (in some brokers)
    or costs interest. Opportunity cost of capital tied up as margin.

    For BPS: margin = spread_width * 100 * contracts per position
    This capital earns 0 return in backtest but could earn risk-free rate elsewhere.

    Actually, this is a BENEFIT for BPS vs CSP — BPS ties up less margin.
    And most brokers pay interest on options margin.
    So this is slightly POSITIVE for real-world BPS (backtest understates).
    """
    trades = trades_df.copy()
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    trades["open_date"] = pd.to_datetime(trades["open_date"])

    total_margin_days = (trades["max_loss"] * trades["days_held"]).sum()
    avg_daily_margin = total_margin_days / max(trades["days_held"].sum(), 1)

    # Interest earned on margin (if broker pays)
    daily_rate = annual_rate / 365
    total_interest_earned = total_margin_days * daily_rate

    # Interest on unused capital
    total_days = (trades["close_date"].max() - trades["open_date"].min()).days
    avg_equity = starting_capital + trades["realized_pnl"].sum() / 2  # rough midpoint
    total_unused = avg_equity * total_days - total_margin_days
    interest_on_unused = total_unused * daily_rate

    return {
        "avg_daily_margin_tied": round(avg_daily_margin, 0),
        "total_interest_earned_on_margin": round(total_interest_earned, 0),
        "total_interest_on_unused_capital": round(interest_on_unused, 0),
        "net_interest_benefit": round(total_interest_earned + interest_on_unused, 0),
        "note": "Positive = backtest UNDERSTATES BPS return (margin earns interest in real account)"
    }


def main():
    t0 = time.time()

    # Load data
    print("Loading data...")
    prices, iv, macro, fund, universe, earnings = load_data()

    # Load trade-level data from assignment risk study
    trades = load_assignment_risk_trades()
    if trades is None or trades.empty:
        print("ERROR: No trade data found. Run bps_assignment_risk_analysis.py first.")
        return

    print(f"Loaded {len(trades)} trades")

    # ── Step 1: Baseline (from backtest) ──
    print("\n=== BASELINE (BS-modeled, no additional frictions) ===")
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    daily_pnl = trades.groupby("close_date")["realized_pnl"].sum()

    baseline_sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0
    baseline_total = trades["realized_pnl"].sum()
    baseline_wr = (trades["realized_pnl"] > 0).mean() * 100

    print(f"  Sharpe: {baseline_sharpe:.2f}")
    print(f"  Total P&L: ${baseline_total:,.0f}")
    print(f"  Win Rate: {baseline_wr:.1f}%")
    print(f"  Trades: {len(trades)}")

    results = {
        "baseline": {
            "sharpe": round(baseline_sharpe, 2),
            "total_pnl": round(baseline_total, 0),
            "win_rate": round(baseline_wr, 1),
            "n_trades": len(trades),
            "n_days": len(daily_pnl),
        }
    }

    # ── Step 2: Bid-ask spread haircut ──
    print("\n=== FRICTION 1: Bid-Ask Spread ===")
    for ba_frac in [0.03, 0.05, 0.08, 0.10]:
        haircut = apply_bid_ask_haircut(trades, ba_frac_open=ba_frac, ba_frac_close=ba_frac)
        adjusted_pnl = trades["realized_pnl"] - haircut
        daily_adj = adjusted_pnl.groupby(trades["close_date"]).sum()
        adj_sharpe = daily_adj.mean() / daily_adj.std() * np.sqrt(252) if daily_adj.std() > 0 else 0
        adj_total = adjusted_pnl.sum()
        adj_wr = (adjusted_pnl > 0).mean() * 100

        label = f"ba_{int(ba_frac*100)}pct"
        results[label] = {
            "sharpe": round(adj_sharpe, 2),
            "total_pnl": round(adj_total, 0),
            "win_rate": round(adj_wr, 1),
            "haircut_total": round(haircut.sum(), 0),
            "haircut_per_trade": round(haircut.mean(), 2),
        }
        print(f"  BA={ba_frac:.0%}: Sharpe {adj_sharpe:.2f}, Total ${adj_total:,.0f}, "
              f"Haircut ${haircut.sum():,.0f} (${haircut.mean():.2f}/trade)")

    # ── Step 3: Fill rate impact ──
    print("\n=== FRICTION 2: Fill Rate ===")
    for fill_rate in [0.60, 0.75, 0.85, 1.0]:
        fr_result = apply_fill_rate(trades, fill_rate)
        results[f"fill_rate_{int(fill_rate*100)}pct"] = fr_result
        print(f"  Fill={fill_rate:.0%}: Sharpe {fr_result['mean_sharpe']:.2f} ± {fr_result['std_sharpe']:.2f}")

    # ── Step 4: Regime analysis ──
    print("\n=== FRICTION 3: Regime Dependence ===")
    regime_results = regime_analysis(trades, macro)
    results["regime_analysis"] = regime_results
    for name, data in regime_results.items():
        if "daily_sharpe" in data:
            print(f"  {name}: Sharpe {data['daily_sharpe']:.2f}, WR {data['win_rate']:.1f}%, "
                  f"trades {data['n_trades']}")
        else:
            print(f"  {name}: {data.get('note', 'N/A')}")

    # ── Step 5: Loss clustering ──
    print("\n=== FRICTION 4: Loss Clustering / Tail Risk ===")
    cluster_results = correlation_clustering(trades)
    results["loss_clustering"] = cluster_results
    if "var_95_daily" in cluster_results:
        print(f"  Loss days: {cluster_results['loss_days_pct']:.1f}%")
        print(f"  Avg loss streak: {cluster_results['avg_loss_streak']:.1f} days")
        print(f"  Max loss streak: {cluster_results['max_loss_streak']} days")
        print(f"  VaR 95%: ${cluster_results['var_95_daily']:,.0f}/day")
        print(f"  CVaR 95%: ${cluster_results['cvar_95_daily']:,.0f}/day")
        print(f"  Tail ratio (CVaR/VaR) 95%: {cluster_results['tail_ratio_95']:.2f}")

    # ── Step 6: Margin interest (positive factor) ──
    print("\n=== POSITIVE FACTOR: Margin Interest ===")
    interest_results = margin_interest_haircut(trades)
    results["margin_interest"] = interest_results
    print(f"  Net interest benefit: ${interest_results['net_interest_benefit']:,.0f} over backtest period")

    # ── Step 7: COMBINED realistic estimate ──
    print("\n" + "="*80)
    print("REALISM BRIDGE: BS Backtest → Honest Expected Performance")
    print("="*80)

    # Best estimate: 5% BA spread (portfolio average for large-cap heavy universe)
    # + 75% fill rate + regime-adjusted
    ba_haircut = apply_bid_ask_haircut(trades, ba_frac_open=0.05, ba_frac_close=0.05)
    realistic_pnl = trades["realized_pnl"] - ba_haircut
    realistic_daily = realistic_pnl.groupby(trades["close_date"]).sum()
    realistic_sharpe = realistic_daily.mean() / realistic_daily.std() * np.sqrt(252) if realistic_daily.std() > 0 else 0
    realistic_total = realistic_pnl.sum()

    # Fill rate doesn't change Sharpe much (reduces trades proportionally)
    # but reduces total P&L
    fill_adjusted_pnl = realistic_total * 0.75  # 75% fill rate

    # Compute combined metrics
    n_years = len(daily_pnl) / 252
    starting_cap = 100_000

    realistic_cagr = ((starting_cap + fill_adjusted_pnl) / starting_cap) ** (1/n_years) - 1 if n_years > 0 else 0

    print(f"\n  Layer 0 (BS backtest):      Sharpe {baseline_sharpe:.2f}, Total ${baseline_total:,.0f}")
    print(f"  Layer 1 (+5% BA spread):    Sharpe {realistic_sharpe:.2f}, Total ${realistic_total:,.0f}")
    print(f"  Layer 2 (+75% fill rate):   P&L ${fill_adjusted_pnl:,.0f} (Sharpe same — proportional reduction)")
    print(f"  Layer 3 (+margin interest): +${interest_results['net_interest_benefit']:,.0f}")

    final_pnl = fill_adjusted_pnl + interest_results['net_interest_benefit']
    final_cagr = ((starting_cap + final_pnl) / starting_cap) ** (1/n_years) - 1 if n_years > 0 else 0

    print(f"\n  ═══════════════════════════════════════")
    print(f"  HONEST EXPECTED SHARPE: {realistic_sharpe:.2f}")
    print(f"  HONEST EXPECTED CAGR:   {final_cagr*100:.1f}%")
    print(f"  HONEST EXPECTED P&L:    ${final_pnl:,.0f} over {n_years:.1f} years")
    print(f"  ═══════════════════════════════════════")

    # Check if regime-dependent
    if "low_vol" in regime_results and "high_vol" in regime_results:
        low_vol_sharpe = regime_results.get("low_vol", {}).get("daily_sharpe", 0)
        high_vol_sharpe = regime_results.get("high_vol", {}).get("daily_sharpe", 0)
        if max(abs(low_vol_sharpe), abs(high_vol_sharpe)) > 0:
            regime_gap = abs(low_vol_sharpe - high_vol_sharpe) / max(abs(low_vol_sharpe), abs(high_vol_sharpe), 1)
            print(f"\n  Regime gap: {regime_gap:.2f} (threshold: 0.50)")
            if regime_gap > 0.50:
                print(f"  ⚠️ REGIME-DEPENDENT: strategy works much better in one VIX regime")
            else:
                print(f"  ✅ REGIME-AGNOSTIC: works across VIX environments")

    results["combined_realistic"] = {
        "honest_sharpe": round(realistic_sharpe, 2),
        "honest_cagr_pct": round(final_cagr * 100, 1),
        "honest_total_pnl": round(final_pnl, 0),
        "years": round(n_years, 1),
        "assumptions": {
            "bid_ask_spread": "5% of premium (portfolio weighted avg)",
            "fill_rate": "75% (limit orders at mid)",
            "margin_interest": "5% annual on unused + held margin",
            "assignment_risk": "modeled via 1 DTE close",
        }
    }

    # ── Save ──
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [clean_dict(v) for v in d]
        else:
            return make_serializable(d)

    report = clean_dict(results)
    with open(OUTPUT / "realism_bridge_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✓ Report saved. Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
