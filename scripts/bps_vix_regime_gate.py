#!/usr/bin/env python3
"""
BPS VIX Regime Gating Study — HC #664 R4 Development
=====================================================

The realism bridge found critical regime dependence:
  - Low vol (<15):    Sharpe 9.34, WR 93.9%
  - Normal (15-25):   Sharpe 5.72, WR 89.6%
  - High vol (25-35): Sharpe -0.28, WR 81.5%
  - Crisis (>35):     Sharpe -9.23, WR 27.0%

Question: Can a simple VIX gate (don't open new positions when VIX > threshold)
improve honest risk-adjusted returns?

Test VIX gates at: 20, 22, 25, 28, 30, no-gate
Apply 5% BA cost (our honest baseline), then compare.

Output: output/bps_vix_gate/vix_gate_results.json
"""

import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "output" / "wheel_higher_returns_study"))

OUTPUT = ROOT / "output" / "bps_vix_gate"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_trades_and_macro():
    """Load trade-level data and macro data (VIX)."""
    trades_path = ROOT / "output" / "bps_assignment_risk" / "trades_close_1dte.parquet"
    trades = pd.read_parquet(trades_path)

    # Load macro for VIX
    from higher_returns_study import load_data
    prices, iv, macro, fund, universe, earnings = load_data()

    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])

    return trades, macro


def apply_ba_cost(trades_df, ba_frac=0.05):
    """Apply bid-ask spread cost to trades. Returns modified trades with adjusted PnL."""
    trades = trades_df.copy()

    # Cost to open: ba_frac * premium * 2 legs
    open_cost = trades["net_credit"].abs() * ba_frac * 2

    # Cost to close (for trades that close early): ba_frac * premium * 2 legs
    closed_early = trades["exit_type"].isin(["profit_take", "early_close_1DTE", "loss_stop"])
    close_cost = pd.Series(0.0, index=trades.index)
    close_cost[closed_early] = trades.loc[closed_early, "net_credit"].abs() * ba_frac * 2

    trades["ba_cost"] = open_cost + close_cost
    trades["honest_pnl"] = trades["realized_pnl"] - trades["ba_cost"]

    return trades


def compute_metrics(daily_pnl, starting_capital=100_000):
    """Compute risk-adjusted metrics from daily P&L series."""
    if len(daily_pnl) < 30:
        return {"note": "too few days", "n_days": len(daily_pnl)}

    # Daily returns
    equity = starting_capital + daily_pnl.cumsum()
    rets = equity.pct_change().dropna()

    if rets.std() == 0:
        return {"note": "zero variance", "n_days": len(daily_pnl)}

    sharpe = float(rets.mean() / rets.std() * np.sqrt(252))

    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std() * np.sqrt(252)) if len(downside) > 5 and downside.std() > 0 else 0.0

    # Max drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min() * 100)

    # Win rate (daily)
    daily_wr = float((daily_pnl > 0).sum() / len(daily_pnl) * 100)

    # Calmar
    total_return_ann = float(daily_pnl.sum() / starting_capital / (len(daily_pnl) / 252) * 100)
    calmar = abs(total_return_ann / max_dd) if max_dd != 0 else 0

    # Profit factor
    gross_profit = daily_pnl[daily_pnl > 0].sum()
    gross_loss = abs(daily_pnl[daily_pnl < 0].sum())
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    # Tail risk
    var_95 = float(np.percentile(daily_pnl.values, 5))
    cvar_95 = float(daily_pnl[daily_pnl <= var_95].mean()) if (daily_pnl <= var_95).sum() > 0 else var_95

    return {
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "profit_factor": round(pf, 2),
        "max_dd_pct": round(max_dd, 1),
        "calmar": round(calmar, 2),
        "daily_wr_pct": round(daily_wr, 1),
        "total_pnl": round(float(daily_pnl.sum()), 0),
        "cagr_pct": round(total_return_ann, 1),
        "n_trades": None,  # filled by caller
        "n_days": len(daily_pnl),
        "var_95": round(var_95, 0),
        "cvar_95": round(cvar_95, 0),
    }


def run_vix_gated_backtest(trades, macro, vix_threshold, ba_frac=0.05):
    """
    Run BPS backtest with VIX gate: don't OPEN positions when VIX > threshold.
    Positions already open are held to their normal exit.
    """
    trades = trades.copy()
    trades["open_date"] = pd.to_datetime(trades["open_date"])
    trades["close_date"] = pd.to_datetime(trades["close_date"])

    # Map VIX to open date
    vix_by_date = macro.set_index("date")["vix"].to_dict()
    trades["open_vix"] = trades["open_date"].map(vix_by_date)

    # Drop trades with no VIX data
    trades = trades.dropna(subset=["open_vix"])

    if vix_threshold is not None:
        # Gate: only open trades when VIX <= threshold
        trades_gated = trades[trades["open_vix"] <= vix_threshold].copy()
    else:
        trades_gated = trades.copy()

    if len(trades_gated) < 50:
        return {"note": f"too few trades after VIX gate at {vix_threshold}", "n_trades": len(trades_gated)}

    # Apply BA cost
    trades_gated = apply_ba_cost(trades_gated, ba_frac=ba_frac)

    # Build daily P&L
    daily_pnl = trades_gated.groupby("close_date")["honest_pnl"].sum()

    # Fill in zero-P&L trading days to get proper Sharpe
    all_dates = pd.date_range(daily_pnl.index.min(), daily_pnl.index.max(), freq='B')
    daily_pnl = daily_pnl.reindex(all_dates, fill_value=0.0)

    metrics = compute_metrics(daily_pnl)
    metrics["n_trades"] = len(trades_gated)
    metrics["vix_threshold"] = vix_threshold if vix_threshold else "no_gate"
    metrics["trades_filtered_pct"] = round((1 - len(trades_gated) / len(trades)) * 100, 1)
    metrics["avg_open_vix"] = round(float(trades_gated["open_vix"].mean()), 1)

    # Per-regime breakdown of remaining trades
    trades_gated["close_vix"] = trades_gated["close_date"].map(vix_by_date)
    regime_pnl = {}
    for rname, lo, hi in [("low", 0, 15), ("normal", 15, 25), ("high", 25, 35), ("crisis", 35, 200)]:
        mask = (trades_gated["close_vix"] >= lo) & (trades_gated["close_vix"] < hi)
        subset = trades_gated[mask]
        if len(subset) > 10:
            regime_pnl[rname] = {
                "n_trades": len(subset),
                "total_pnl": round(float(subset["honest_pnl"].sum()), 0),
                "avg_pnl": round(float(subset["honest_pnl"].mean()), 2),
                "win_rate": round(float((subset["honest_pnl"] > 0).mean() * 100), 1),
            }
    metrics["regime_breakdown"] = regime_pnl

    # Save equity curve
    equity = 100_000 + daily_pnl.cumsum()
    eq_df = pd.DataFrame({"date": equity.index, "equity": equity.values})
    gate_name = f"vix_{vix_threshold}" if vix_threshold else "no_gate"
    eq_df.to_parquet(OUTPUT / f"eq_{gate_name}.parquet", index=False)

    return metrics


def run_adaptive_vix_gate(trades, macro, ba_frac=0.05):
    """
    Adaptive VIX gate: instead of a fixed threshold, use VIX percentile.
    - When VIX is in top 20% of its 60-day rolling distribution → pause
    - This adapts to the current vol regime rather than a fixed number
    """
    trades = trades.copy()
    trades["open_date"] = pd.to_datetime(trades["open_date"])
    trades["close_date"] = pd.to_datetime(trades["close_date"])

    macro_sorted = macro.sort_values("date").copy()
    macro_sorted["vix_60d_pct"] = macro_sorted["vix"].rolling(60).rank(pct=True)
    vix_pct_by_date = macro_sorted.set_index("date")["vix_60d_pct"].to_dict()
    vix_by_date = macro_sorted.set_index("date")["vix"].to_dict()

    trades["open_vix_pct"] = trades["open_date"].map(vix_pct_by_date)
    trades["open_vix"] = trades["open_date"].map(vix_by_date)
    trades = trades.dropna(subset=["open_vix_pct"])

    results = {}
    for pct_gate in [0.70, 0.75, 0.80, 0.85, 0.90, 1.0]:
        gated = trades[trades["open_vix_pct"] <= pct_gate].copy()
        if len(gated) < 50:
            results[f"pct_{int(pct_gate*100)}"] = {"note": "too few trades"}
            continue

        gated = apply_ba_cost(gated, ba_frac=ba_frac)
        daily_pnl = gated.groupby("close_date")["honest_pnl"].sum()
        all_dates = pd.date_range(daily_pnl.index.min(), daily_pnl.index.max(), freq='B')
        daily_pnl = daily_pnl.reindex(all_dates, fill_value=0.0)

        m = compute_metrics(daily_pnl)
        m["n_trades"] = len(gated)
        m["pct_gate"] = pct_gate
        m["trades_filtered_pct"] = round((1 - len(gated) / len(trades)) * 100, 1)
        results[f"pct_{int(pct_gate*100)}"] = m

    return results


def run_vix_position_sizing(trades, macro, ba_frac=0.05):
    """
    Instead of binary gate, scale position size inversely with VIX.
    VIX < 15: full size (100%)
    VIX 15-20: 80%
    VIX 20-25: 50%
    VIX 25-30: 25%
    VIX > 30: 0% (fully gated)

    This is a softer approach — keeps trading in moderate vol but smaller.
    """
    trades = trades.copy()
    trades["open_date"] = pd.to_datetime(trades["open_date"])
    trades["close_date"] = pd.to_datetime(trades["close_date"])

    vix_by_date = macro.set_index("date")["vix"].to_dict()
    trades["open_vix"] = trades["open_date"].map(vix_by_date)
    trades = trades.dropna(subset=["open_vix"])

    # Size scalar based on VIX
    def size_scalar(vix):
        if vix < 15:
            return 1.0
        elif vix < 20:
            return 0.8
        elif vix < 25:
            return 0.5
        elif vix < 30:
            return 0.25
        else:
            return 0.0

    trades["size_scalar"] = trades["open_vix"].apply(size_scalar)
    trades = trades[trades["size_scalar"] > 0].copy()

    # Apply BA cost then scale PnL
    trades = apply_ba_cost(trades, ba_frac=ba_frac)
    trades["honest_pnl"] = trades["honest_pnl"] * trades["size_scalar"]

    daily_pnl = trades.groupby("close_date")["honest_pnl"].sum()
    all_dates = pd.date_range(daily_pnl.index.min(), daily_pnl.index.max(), freq='B')
    daily_pnl = daily_pnl.reindex(all_dates, fill_value=0.0)

    metrics = compute_metrics(daily_pnl)
    metrics["n_trades"] = len(trades)
    metrics["method"] = "vix_position_sizing"
    metrics["avg_size_scalar"] = round(float(trades["size_scalar"].mean()), 2)

    return metrics


def main():
    t0 = time.time()

    print("Loading trades and macro data...")
    trades, macro = load_trades_and_macro()
    print(f"  {len(trades)} trades loaded")

    results = {}

    # 1. Fixed VIX gates
    print("\n=== Fixed VIX Gate Tests ===")
    for threshold in [None, 30, 28, 25, 22, 20, 18]:
        label = f"vix_{threshold}" if threshold else "no_gate"
        print(f"  Testing VIX gate: {label}...")
        r = run_vix_gated_backtest(trades, macro, threshold)
        results[label] = r
        if "sharpe" in r:
            print(f"    Sharpe={r['sharpe']}, Sortino={r['sortino']}, PF={r['profit_factor']}, "
                  f"MaxDD={r['max_dd_pct']}%, N={r['n_trades']}, Filtered={r['trades_filtered_pct']}%")

    # 2. Adaptive VIX percentile gate
    print("\n=== Adaptive VIX Percentile Gate ===")
    adaptive = run_adaptive_vix_gate(trades, macro)
    results["adaptive_gates"] = adaptive
    for k, v in adaptive.items():
        if "sharpe" in v:
            print(f"  {k}: Sharpe={v['sharpe']}, Sortino={v['sortino']}, N={v['n_trades']}, Filtered={v['trades_filtered_pct']}%")

    # 3. VIX-scaled position sizing
    print("\n=== VIX-Scaled Position Sizing ===")
    sized = run_vix_position_sizing(trades, macro)
    results["vix_position_sizing"] = sized
    if "sharpe" in sized:
        print(f"  Sharpe={sized['sharpe']}, Sortino={sized['sortino']}, PF={sized['profit_factor']}, "
              f"MaxDD={sized['max_dd_pct']}%, AvgScale={sized['avg_size_scalar']}")

    # 4. Summary comparison
    print("\n" + "="*80)
    print("SUMMARY: VIX Gate Comparison (all with 5% BA cost)")
    print("="*80)
    print(f"{'Config':<25} {'Sharpe':>8} {'Sortino':>8} {'PF':>6} {'MaxDD':>8} {'CAGR%':>7} {'Trades':>7} {'Cut%':>6}")
    print("-"*80)

    for key in ["no_gate", "vix_30", "vix_28", "vix_25", "vix_22", "vix_20", "vix_18"]:
        r = results.get(key, {})
        if "sharpe" in r:
            print(f"{key:<25} {r['sharpe']:>8.2f} {r['sortino']:>8.2f} {r['profit_factor']:>6.2f} "
                  f"{r['max_dd_pct']:>7.1f}% {r.get('cagr_pct', 0):>6.1f}% {r['n_trades']:>7} {r['trades_filtered_pct']:>5.1f}%")

    sized = results.get("vix_position_sizing", {})
    if "sharpe" in sized:
        print(f"{'vix_scaling':<25} {sized['sharpe']:>8.2f} {sized['sortino']:>8.2f} {sized['profit_factor']:>6.2f} "
              f"{sized['max_dd_pct']:>7.1f}% {sized.get('cagr_pct', 0):>6.1f}% {sized['n_trades']:>7}   N/A")

    # Adaptive best
    best_adaptive_sharpe = 0
    best_adaptive_key = None
    for k, v in adaptive.items():
        if "sharpe" in v and v["sharpe"] > best_adaptive_sharpe:
            best_adaptive_sharpe = v["sharpe"]
            best_adaptive_key = k

    if best_adaptive_key:
        ba = adaptive[best_adaptive_key]
        print(f"{'adaptive_'+best_adaptive_key:<25} {ba['sharpe']:>8.2f} {ba['sortino']:>8.2f} {ba.get('profit_factor', 0):>6.2f} "
              f"{ba['max_dd_pct']:>7.1f}% {ba.get('cagr_pct', 0):>6.1f}% {ba['n_trades']:>7} {ba['trades_filtered_pct']:>5.1f}%")

    # Save results
    # Convert any numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    results_clean = convert_types(results)
    with open(OUTPUT / "vix_gate_results.json", "w") as f:
        json.dump(results_clean, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Results → {OUTPUT / 'vix_gate_results.json'}")


if __name__ == "__main__":
    main()
