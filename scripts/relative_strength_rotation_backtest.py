#!/usr/bin/env python3
"""
Relative Strength Sector Rotation Backtest
Academic basis: Jegadeesh & Titman (1993) momentum applied to sector ETFs.

Variants A-F tested with 5-gate validation:
  1. Sharpe > 0.5
  2. Permutation test p < 0.05 (100 iterations)
  3. Regime gap < 0.5
  4. MaxDD > -50%
  5. >= 20 trades

OOT: Jan 2022 - Jul 2026
Cost: $0 commission, 0.02% slippage per trade
Starting capital: $10,000
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────
SECTOR_ETFS = ["XLK", "XLC", "XLY", "XLE", "XLF", "XLV", "XLI", "XLB", "XLU", "XLRE", "XLP"]
BENCHMARK = "SPY"
START_DATE = "2019-06-01"  # extra history for lookback
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
INITIAL_CAPITAL = 10_000
SLIPPAGE_PCT = 0.0002  # 0.02%
N_PERMUTATIONS = 100
RISK_FREE_RATE = 0.0  # annualized, for Sharpe calc

OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/relative_strength_rotation_results.json")


def download_data():
    """Download adjusted close prices for all tickers."""
    tickers = SECTOR_ETFS + [BENCHMARK]
    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)
    # yfinance returns multi-level columns; extract Close
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
    prices = prices.dropna(how="all")
    # Forward fill small gaps (holidays etc)
    prices = prices.ffill().dropna()
    print(f"Data range: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} days")
    return prices


def compute_returns(prices, months=3):
    """Compute trailing total return over N months (approx 21 trading days per month)."""
    lookback = months * 21
    return prices.pct_change(lookback)


def compute_sma(prices, window=200):
    """Simple moving average."""
    return prices.rolling(window).mean()


def get_monthly_rebalance_dates(prices, start, end):
    """Get first trading day of each month in range."""
    idx = prices.loc[start:end].index
    monthly = idx.to_period("M").unique()
    dates = []
    for m in monthly:
        mask = idx.to_period("M") == m
        candidates = idx[mask]
        if len(candidates) > 0:
            dates.append(candidates[0])
    return dates


def apply_slippage(weight_changes, slippage_pct):
    """Compute total slippage cost from weight changes."""
    turnover = weight_changes.abs().sum()
    return turnover * slippage_pct


def run_backtest(prices, variant, spy_prices):
    """
    Run a single variant backtest. Returns daily equity curve and trade log.

    Variants:
    A: Top 3 by 3mo RS, monthly
    B: Top 2 by 3mo RS, monthly
    C: Top 3 by 1mo RS, monthly
    D: Top 3 by 3mo RS, skip negative absolute return
    E: Long top 3 / Short bottom 3
    F: Top 3 by 3mo RS + above 50-SMA
    """
    # Determine parameters
    if variant == "A":
        n_top, lookback_months, trend_filter, long_short, abs_filter = 3, 3, False, False, False
    elif variant == "B":
        n_top, lookback_months, trend_filter, long_short, abs_filter = 2, 3, False, False, False
    elif variant == "C":
        n_top, lookback_months, trend_filter, long_short, abs_filter = 3, 1, False, False, False
    elif variant == "D":
        n_top, lookback_months, trend_filter, long_short, abs_filter = 3, 3, False, False, True
    elif variant == "E":
        n_top, lookback_months, trend_filter, long_short, abs_filter = 3, 3, False, True, False
    elif variant == "F":
        n_top, lookback_months, trend_filter, long_short, abs_filter = 3, 3, True, False, False
    else:
        raise ValueError(f"Unknown variant: {variant}")

    sector_prices = prices[SECTOR_ETFS]
    rs = compute_returns(sector_prices, lookback_months)
    sma50 = compute_sma(sector_prices, 50)
    spy_sma200 = compute_sma(spy_prices, 200)

    rebal_dates = get_monthly_rebalance_dates(prices, OOT_START, OOT_END)

    # Daily returns for sectors
    daily_ret = sector_prices.pct_change()

    # Track portfolio
    equity = INITIAL_CAPITAL
    equity_curve = []
    current_weights = pd.Series(0.0, index=SECTOR_ETFS)
    trade_count = 0
    trade_log = []

    oot_idx = prices.loc[OOT_START:OOT_END].index

    for i, date in enumerate(oot_idx):
        # Check if rebalance day
        if date in rebal_dates:
            rs_today = rs.loc[date].dropna()
            if len(rs_today) < n_top:
                continue

            # Rank sectors
            ranked = rs_today.sort_values(ascending=False)

            # Determine long picks
            if abs_filter:
                # Only sectors with positive absolute return
                candidates = ranked[ranked > 0]
                long_picks = list(candidates.index[:n_top])
            elif trend_filter:
                # Only sectors above their 50-SMA
                above_sma = []
                for sym in ranked.index:
                    if date in sma50.index and sector_prices.loc[date, sym] > sma50.loc[date, sym]:
                        above_sma.append(sym)
                candidates = ranked.loc[[s for s in ranked.index if s in above_sma]]
                long_picks = list(candidates.index[:n_top])
            else:
                long_picks = list(ranked.index[:n_top])

            # Determine short picks (variant E only)
            short_picks = []
            if long_short:
                short_picks = list(ranked.index[-n_top:])

            # Regime hedge: half-size if SPY < 200-SMA
            size_mult = 1.0
            if date in spy_sma200.index and not pd.isna(spy_sma200.loc[date]):
                if spy_prices.loc[date] < spy_sma200.loc[date]:
                    size_mult = 0.5

            # Build new weights
            new_weights = pd.Series(0.0, index=SECTOR_ETFS)
            n_long = len(long_picks)
            n_short = len(short_picks)

            if n_long > 0:
                w_long = size_mult / n_long
                for s in long_picks:
                    new_weights[s] += w_long

            if n_short > 0:
                w_short = size_mult / n_short
                for s in short_picks:
                    new_weights[s] -= w_short

            # Slippage from turnover
            weight_changes = new_weights - current_weights
            slippage_cost = apply_slippage(weight_changes, SLIPPAGE_PCT)
            equity *= (1.0 - slippage_cost)

            if not new_weights.equals(current_weights):
                trade_count += 1
                trade_log.append({
                    "date": str(date.date()),
                    "longs": long_picks,
                    "shorts": short_picks if short_picks else None,
                    "size_mult": size_mult,
                    "turnover": float(weight_changes.abs().sum()),
                })

            current_weights = new_weights

        # Apply daily returns
        if i > 0:
            port_ret = (current_weights * daily_ret.loc[date]).sum()
            equity *= (1.0 + port_ret)

        equity_curve.append({"date": str(date.date()), "equity": equity})

    return equity_curve, trade_count, trade_log


def compute_metrics(equity_curve):
    """Compute performance metrics from equity curve."""
    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    daily_returns = df["equity"].pct_change().dropna()

    if len(daily_returns) < 2:
        return {"sharpe": 0, "sortino": 0, "max_dd": -1.0, "total_return": 0, "cagr": 0, "pf": 0}

    # Annualized Sharpe
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-9
    sortino = (daily_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    # Max Drawdown
    cum = (1 + daily_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Total return
    total_return = (df["equity"].iloc[-1] / df["equity"].iloc[0]) - 1

    # CAGR
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (df["equity"].iloc[-1] / df["equity"].iloc[0]) ** (1 / years) - 1 if years > 0 else 0

    # Profit Factor
    gains = daily_returns[daily_returns > 0].sum()
    losses = abs(daily_returns[daily_returns < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    # Win rate (daily)
    wr = (daily_returns > 0).mean()

    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "max_dd": round(float(max_dd), 4),
        "total_return": round(float(total_return), 4),
        "cagr": round(float(cagr), 4),
        "profit_factor": round(float(pf), 3),
        "win_rate_daily": round(float(wr), 4),
        "final_equity": round(float(df["equity"].iloc[-1]), 2),
    }


def compute_regime_metrics(equity_curve, spy_prices):
    """Split performance into bull (SPY > 200-SMA) and bear regimes."""
    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    daily_returns = df["equity"].pct_change().dropna()

    spy_sma200 = compute_sma(spy_prices, 200)

    bull_rets, bear_rets = [], []
    for date in daily_returns.index:
        if date in spy_sma200.index and not pd.isna(spy_sma200.loc[date]):
            if spy_prices.loc[date] >= spy_sma200.loc[date]:
                bull_rets.append(daily_returns.loc[date])
            else:
                bear_rets.append(daily_returns.loc[date])

    bull_rets = pd.Series(bull_rets) if bull_rets else pd.Series([0.0])
    bear_rets = pd.Series(bear_rets) if bear_rets else pd.Series([0.0])

    bull_sharpe = (bull_rets.mean() / bull_rets.std()) * np.sqrt(252) if bull_rets.std() > 0 else 0
    bear_sharpe = (bear_rets.mean() / bear_rets.std()) * np.sqrt(252) if bear_rets.std() > 0 else 0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        "bull_sharpe": round(float(bull_sharpe), 3),
        "bear_sharpe": round(float(bear_sharpe), 3),
        "regime_gap": round(float(regime_gap), 4),
        "bull_days": len(bull_rets),
        "bear_days": len(bear_rets),
    }


def permutation_test(equity_curve, prices, n_perms=100):
    """
    Permutation test: shuffle sector labels at each rebalance to test
    if ranking-based selection adds value vs random selection.
    Returns p-value.
    """
    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    actual_return = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1

    sector_prices = prices[SECTOR_ETFS]
    daily_ret = sector_prices.pct_change()
    oot_idx = prices.loc[OOT_START:OOT_END].index
    rebal_dates = set(get_monthly_rebalance_dates(prices, OOT_START, OOT_END))

    rng = np.random.RandomState(42)
    perm_returns = []

    for _ in range(n_perms):
        equity = INITIAL_CAPITAL
        weights = pd.Series(0.0, index=SECTOR_ETFS)

        for i, date in enumerate(oot_idx):
            if date in rebal_dates:
                # Random selection of 3 sectors
                picks = rng.choice(SECTOR_ETFS, size=3, replace=False)
                weights = pd.Series(0.0, index=SECTOR_ETFS)
                for s in picks:
                    weights[s] = 1.0 / 3
                equity *= (1.0 - SLIPPAGE_PCT * 2)  # approx turnover cost

            if i > 0:
                port_ret = (weights * daily_ret.loc[date]).sum()
                equity *= (1.0 + port_ret)

        perm_returns.append(equity / INITIAL_CAPITAL - 1)

    perm_returns = np.array(perm_returns)
    p_value = (perm_returns >= actual_return).mean()
    return round(float(p_value), 4)


def five_gate_validation(metrics, regime, trade_count, p_value):
    """Apply 5-gate validation."""
    gates = {
        "gate_1_sharpe_gt_05": metrics["sharpe"] > 0.5,
        "gate_2_perm_p_lt_05": p_value < 0.05,
        "gate_3_regime_gap_lt_05": regime["regime_gap"] < 0.5,
        "gate_4_maxdd_gt_neg50": metrics["max_dd"] > -0.50,
        "gate_5_trades_gte_20": trade_count >= 20,
    }
    gates["all_passed"] = all(gates.values())
    gates["p_value"] = p_value
    return gates


def main():
    prices = download_data()
    spy_prices = prices[BENCHMARK]

    variants = ["A", "B", "C", "D", "E", "F"]
    results = {}

    for v in variants:
        print(f"\n{'='*60}")
        print(f"Running Variant {v}...")
        print(f"{'='*60}")

        equity_curve, trade_count, trade_log = run_backtest(prices, v, spy_prices)
        metrics = compute_metrics(equity_curve)
        regime = compute_regime_metrics(equity_curve, spy_prices)

        print(f"  Total Return: {metrics['total_return']*100:.1f}%  |  Sharpe: {metrics['sharpe']}")
        print(f"  MaxDD: {metrics['max_dd']*100:.1f}%  |  Trades: {trade_count}")
        print(f"  Regime gap: {regime['regime_gap']:.3f}")

        print(f"  Running permutation test ({N_PERMUTATIONS} iterations)...")
        p_value = permutation_test(equity_curve, prices, N_PERMUTATIONS)
        print(f"  p-value: {p_value}")

        gates = five_gate_validation(metrics, regime, trade_count, p_value)
        print(f"  Gates: {'PASS' if gates['all_passed'] else 'FAIL'} - {gates}")

        results[f"variant_{v}"] = {
            "description": {
                "A": "Top 3 by 3-month RS, monthly rebalance",
                "B": "Top 2 by 3-month RS, monthly rebalance (concentrated)",
                "C": "Top 3 by 1-month RS (faster rotation)",
                "D": "Top 3 by 3-month RS, avoid negative absolute return",
                "E": "Long top 3 / Short bottom 3 (market neutral)",
                "F": "Top 3 by 3-month RS + above 50-SMA (trend filter)",
            }[v],
            "metrics": metrics,
            "regime": regime,
            "gates": gates,
            "trade_count": trade_count,
            "equity_curve_start": equity_curve[0],
            "equity_curve_end": equity_curve[-1],
            "sample_trades": trade_log[:5],  # first 5 rebalances
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<10} {'Sharpe':>8} {'Sortino':>8} {'Return':>8} {'MaxDD':>8} {'PF':>6} {'Trades':>7} {'Gates':>6}")
    print("-" * 65)
    for v in variants:
        r = results[f"variant_{v}"]
        m = r["metrics"]
        g = "PASS" if r["gates"]["all_passed"] else "FAIL"
        print(f"  {v:<8} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['total_return']*100:>7.1f}% {m['max_dd']*100:>7.1f}% {m['profit_factor']:>6.2f} {r['trade_count']:>7} {g:>6}")

    # Save results
    output = {
        "strategy": "Relative Strength Sector Rotation",
        "academic_basis": "Jegadeesh & Titman (1993) momentum effect applied to sector ETFs",
        "universe": SECTOR_ETFS,
        "oot_period": f"{OOT_START} to {OOT_END}",
        "initial_capital": INITIAL_CAPITAL,
        "slippage_pct": SLIPPAGE_PCT,
        "regime_hedge": "Half-size when SPY < 200-SMA",
        "run_timestamp": datetime.now().isoformat(),
        "variants": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
