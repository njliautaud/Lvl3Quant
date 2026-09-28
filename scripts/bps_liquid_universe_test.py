#!/usr/bin/env python3
"""
BPS Liquid Universe Test — Trade Only the Most Liquid Names
============================================================

HC #664 R4: Realism bridge showed bid-ask spread is THE dominant friction.
Mega-cap names have ~3% BA, mid-caps ~10%. If we restrict to top 22 liquid
names (all in our universe), we get 3% BA instead of 5% portfolio average.

Hypothesis: fewer but higher-quality trades → better honest Sharpe.

Tests:
  1. Full 70-ticker universe (current baseline)
  2. Top 22 liquid names only (>20M avg daily vol)
  3. Top 15 liquid names (>30M avg daily vol)
  4. Top 10 liquid names (>45M avg daily vol)

For each: run backtest, apply realistic BA haircut, compute honest Sharpe.
"""

import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "output" / "wheel_higher_returns_study"))
from higher_returns_study import (
    load_data, bs_price, bs_delta, strike_from_delta, trade_cost,
    compute_metrics, COST_PER_CONTRACT
)

# Import the enhanced BPS runner from assignment risk study
sys.path.insert(0, str(ROOT / "scripts"))
from bps_assignment_risk_analysis import run_bps_with_trade_tracking, analyze_assignment_risk

OUTPUT = ROOT / "output" / "bps_liquid_universe"
OUTPUT.mkdir(parents=True, exist_ok=True)


# Liquidity tiers and expected bid-ask spreads
LIQUIDITY_TIERS = {
    "top10_mega": {
        "min_avg_vol": 45e6,
        "expected_ba_pct": 0.025,  # 2.5% — mega-cap, very tight
        "description": "Top 10 mega-cap (>45M avg vol)"
    },
    "top15_large": {
        "min_avg_vol": 30e6,
        "expected_ba_pct": 0.03,  # 3% — large-cap
        "description": "Top 15 large-cap (>30M avg vol)"
    },
    "top22_liquid": {
        "min_avg_vol": 20e6,
        "expected_ba_pct": 0.035,  # 3.5% — mostly large-cap
        "description": "Top 22 liquid (>20M avg vol)"
    },
    "full_universe": {
        "min_avg_vol": 0,
        "expected_ba_pct": 0.05,  # 5% — portfolio average
        "description": "Full 70-ticker universe"
    },
}


def apply_ba_haircut_to_trades(trades_df, ba_frac):
    """Apply bid-ask spread haircut and return adjusted daily Sharpe."""
    if trades_df.empty:
        return {"sharpe": 0, "total_pnl": 0, "n_trades": 0}

    trades = trades_df.copy()
    trades["close_date"] = pd.to_datetime(trades["close_date"])

    # BA cost: crossing spread on both legs at open + close (for early-closed trades)
    haircut = trades["net_credit"].abs() * ba_frac * 2  # 2 legs at open
    closed_early = trades["exit_type"].isin(["profit_take", "early_close_1DTE", "loss_stop"])
    haircut[closed_early] += trades.loc[closed_early, "net_credit"].abs() * ba_frac * 2  # 2 legs at close

    adjusted_pnl = trades["realized_pnl"] - haircut
    daily = adjusted_pnl.groupby(trades["close_date"]).sum()

    sharpe = daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 10 and daily.std() > 0 else 0
    sortino_denom = daily[daily < 0].std()
    sortino = daily.mean() / sortino_denom * np.sqrt(252) if sortino_denom > 0 else 0

    # Max drawdown from daily P&L
    cum = daily.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    max_dd = dd.min()

    return {
        "honest_sharpe": round(sharpe, 2),
        "honest_sortino": round(sortino, 2),
        "total_pnl_after_ba": round(adjusted_pnl.sum(), 0),
        "ba_haircut_total": round(haircut.sum(), 0),
        "ba_haircut_per_trade": round(haircut.mean(), 2),
        "n_trades": len(trades),
        "n_days": len(daily),
        "win_rate_after_ba": round((adjusted_pnl > 0).mean() * 100, 1),
        "max_dd_pnl": round(max_dd, 0) if not np.isnan(max_dd) else 0,
    }


def main():
    t0 = time.time()
    print("Loading data...")
    prices, iv, macro, fund, universe, earnings = load_data()

    # Compute average daily volume for each ticker
    vol_by_ticker = prices.groupby("ticker")["volume"].mean().sort_values(ascending=False)
    universe_tickers = set(universe["ticker"].values)

    results = {}

    for tier_name, tier_config in LIQUIDITY_TIERS.items():
        min_vol = tier_config["min_avg_vol"]
        ba_pct = tier_config["expected_ba_pct"]
        desc = tier_config["description"]

        # Filter universe to liquid names
        if min_vol > 0:
            liquid_tickers = [tk for tk in vol_by_ticker.index
                              if vol_by_ticker[tk] >= min_vol and tk in universe_tickers]
        else:
            liquid_tickers = list(universe_tickers)

        print(f"\n{'='*60}")
        print(f"TIER: {desc} ({len(liquid_tickers)} tickers, BA={ba_pct:.1%})")
        print(f"{'='*60}")
        print(f"Tickers: {sorted(liquid_tickers)[:15]}{'...' if len(liquid_tickers) > 15 else ''}")

        # Create filtered universe DataFrame
        filtered_universe = universe[universe["ticker"].isin(liquid_tickers)]

        # Run BPS backtest with 1 DTE close
        r = run_bps_with_trade_tracking(
            prices, iv, macro, fund, filtered_universe, earnings,
            spread_width=15.0, put_delta=0.25, dte_target=10,
            profit_take=0.40, margin_cap=0.15, max_concurrent=40,
            per_name_pct=0.03, close_before_expiry_days=1,
            label=f"BPS_{tier_name}"
        )

        if r["trades"].empty:
            print(f"  No trades generated!")
            results[tier_name] = {"error": "no trades"}
            continue

        # Raw metrics
        raw_metrics = r["metrics"]
        print(f"  Raw: Sharpe {raw_metrics.get('sharpe')}, MaxDD {raw_metrics.get('max_dd_pct')}%, "
              f"WR {raw_metrics.get('win_rate_pct')}%, Trades {raw_metrics.get('n_trades')}")

        # Apply realistic BA haircut
        honest = apply_ba_haircut_to_trades(r["trades"], ba_pct)
        print(f"  Honest (BA={ba_pct:.1%}): Sharpe {honest['honest_sharpe']}, "
              f"P&L ${honest['total_pnl_after_ba']:,.0f}, "
              f"WR {honest['win_rate_after_ba']:.1f}%")

        # Also compute what happens at the next-worse BA tier for safety margin
        conservative_ba = ba_pct + 0.02
        conservative = apply_ba_haircut_to_trades(r["trades"], conservative_ba)
        print(f"  Conservative (BA={conservative_ba:.1%}): Sharpe {conservative['honest_sharpe']}")

        # Assignment risk summary
        ar = analyze_assignment_risk(r["trades"])
        breach_rate = ar.get("breach_analysis", {}).get("pct_of_all_trades", 0)
        print(f"  Breach rate: {breach_rate:.1f}%")

        # Per-ticker attribution
        trades = r["trades"].copy()
        ticker_pnl = trades.groupby("ticker")["realized_pnl"].agg(["sum", "count", "mean"])
        ticker_pnl = ticker_pnl.sort_values("sum", ascending=False)
        top5 = ticker_pnl.head(5)
        bot5 = ticker_pnl.tail(5)

        print(f"\n  Top 5 tickers: {list(top5.index)}")
        print(f"  Bottom 5 tickers: {list(bot5.index)}")

        results[tier_name] = {
            "description": desc,
            "n_tickers": len(liquid_tickers),
            "tickers": sorted(liquid_tickers),
            "expected_ba_pct": ba_pct,
            "raw_metrics": {k: v for k, v in raw_metrics.items() if not isinstance(v, (pd.DataFrame, pd.Series))},
            "honest_metrics": honest,
            "conservative_metrics": conservative,
            "breach_rate_pct": breach_rate,
            "ticker_attribution": {
                "top5": {tk: {"pnl": round(row["sum"], 0), "trades": int(row["count"]), "avg": round(row["mean"], 2)}
                         for tk, row in top5.iterrows()},
                "bottom5": {tk: {"pnl": round(row["sum"], 0), "trades": int(row["count"]), "avg": round(row["mean"], 2)}
                         for tk, row in bot5.iterrows()},
            }
        }

        # Save equity curve
        r["equity_curve"].to_parquet(OUTPUT / f"eq_{tier_name}.parquet", index=False)

    # ── Summary ──
    print("\n" + "="*80)
    print("LIQUIDITY-FILTERED BPS COMPARISON")
    print("="*80)
    print(f"{'Tier':<25} {'Tickers':>8} {'BA':>5} {'Raw Sharpe':>11} {'Honest Sharpe':>14} {'P&L':>10} {'WR':>6}")
    print("-"*80)
    for tier_name, data in results.items():
        if "error" in data:
            continue
        h = data["honest_metrics"]
        r = data["raw_metrics"]
        print(f"{data['description']:<25} {data['n_tickers']:>8} {data['expected_ba_pct']:>4.1%} "
              f"{r.get('sharpe', 'N/A'):>11} {h['honest_sharpe']:>14} "
              f"${h['total_pnl_after_ba']:>9,.0f} {h['win_rate_after_ba']:>5.1f}%")

    # ── Save ──
    def clean(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Timestamp): return str(obj)
        return obj

    def clean_dict(d):
        if isinstance(d, dict): return {k: clean_dict(v) for k, v in d.items()}
        if isinstance(d, list): return [clean_dict(v) for v in d]
        return clean(d)

    with open(OUTPUT / "liquid_universe_results.json", "w") as f:
        json.dump(clean_dict(results), f, indent=2, default=str)

    print(f"\n✓ Saved. Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
