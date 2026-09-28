#!/usr/bin/env python3
"""
Walk-Forward Trend-Following Backtest (Multi-Asset)
====================================================
Strategy: Long when price > 200-day SMA, flat (cash) when below.
Variant:  Golden-cross confirmation (50MA > 200MA AND price > 200MA).

Equal-weight across all assets that are "on". Monthly rebalance.
Sliding walk-forward: no expanding window.

Covers 2008-present to capture GFC, COVID, 2022 bear market.
Compares to SPY buy-and-hold and 60/40 SPY/AGG.

Usage:
  python growth/trend_backtest.py
"""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

UNIVERSE = {
    "US Equities": ["SPY", "QQQ", "IWM"],
    "International": ["EFA", "EEM"],
    "Bonds": ["TLT", "IEF", "AGG"],
    "Commodities": ["GLD", "DBA"],
    "Real Estate": ["VNQ"],
}

ALL_TICKERS = [t for tickers in UNIVERSE.values() for t in tickers]

START_DATE = "2007-01-01"   # extra lookback for 200MA warmup
BACKTEST_START = "2008-01-01"
END_DATE = None  # present

SMA_SLOW = 200
SMA_FAST = 50
TRANSACTION_COST_BPS = 5  # 0.05% per trade
REBALANCE_FREQ = "ME"     # month-end

OUTPUT_DIR = Path(__file__).parent / "output"


# ── Data Download ──────────────────────────────────────────────────────────────

def download_data(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted close prices for all tickers."""
    print(f"Downloading {len(tickers)} ETFs from {start}...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    # Drop any ticker with <200 rows (insufficient for 200MA)
    valid = prices.columns[prices.count() >= SMA_SLOW + 20]
    dropped = set(prices.columns) - set(valid)
    if dropped:
        print(f"  Dropped (insufficient data): {dropped}")
    prices = prices[valid]

    print(f"  Got {len(prices)} days for {len(prices.columns)} tickers")
    return prices


# ── Signal Generation ──────────────────────────────────────────────────────────

def compute_signals(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute two signal sets:
      1. Simple: price > 200MA
      2. Golden cross: price > 200MA AND 50MA > 200MA
    Returns DataFrames of 1/0 signals.
    """
    sma_slow = prices.rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()
    sma_fast = prices.rolling(SMA_FAST, min_periods=SMA_FAST).mean()

    sig_simple = (prices > sma_slow).astype(int)
    sig_golden = ((prices > sma_slow) & (sma_fast > sma_slow)).astype(int)

    return sig_simple, sig_golden


# ── Portfolio Construction ─────────────────────────────────────────────────────

def build_portfolio(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    cost_bps: float = TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    """
    Equal-weight portfolio with monthly rebalance.
    Returns daily portfolio returns series.
    """
    daily_returns = prices.pct_change()

    # Resample signals to month-end (use last trading day of month)
    monthly_signals = signals.resample(REBALANCE_FREQ).last()

    # Forward-fill monthly signals to daily frequency
    daily_signals = monthly_signals.reindex(signals.index).ffill()

    # Equal-weight: divide by number of "on" assets each day
    n_on = daily_signals.sum(axis=1).replace(0, np.nan)
    weights = daily_signals.div(n_on, axis=0).fillna(0)

    # Compute turnover for transaction costs
    weight_changes = weights.diff().abs().sum(axis=1)
    costs = weight_changes * (cost_bps / 10000)

    # Portfolio return = sum of weighted asset returns minus costs
    port_returns = (weights.shift(1) * daily_returns).sum(axis=1) - costs

    # Trim to backtest period
    port_returns = port_returns.loc[BACKTEST_START:]

    return port_returns


def build_benchmark_bh(prices: pd.DataFrame, ticker: str = "SPY") -> pd.Series:
    """Buy-and-hold benchmark."""
    ret = prices[ticker].pct_change().loc[BACKTEST_START:]
    return ret


def build_benchmark_6040(prices: pd.DataFrame) -> pd.Series:
    """60/40 SPY/AGG benchmark, rebalanced monthly."""
    spy_ret = prices["SPY"].pct_change()
    agg_ret = prices["AGG"].pct_change()

    # Monthly rebalance back to 60/40
    port_ret = 0.6 * spy_ret + 0.4 * agg_ret
    return port_ret.loc[BACKTEST_START:]


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(returns: pd.Series, name: str = "") -> dict:
    """Compute risk-adjusted performance metrics."""
    returns = returns.dropna()
    if len(returns) < 252:
        return {"name": name, "error": "insufficient data"}

    # Annualization
    ann_factor = 252

    # CAGR
    total_return = (1 + returns).prod()
    n_years = len(returns) / ann_factor
    cagr = total_return ** (1 / n_years) - 1

    # Volatility
    vol = returns.std() * np.sqrt(ann_factor)

    # Sharpe (rf=0 for simplicity)
    sharpe = (returns.mean() * ann_factor) / vol if vol > 0 else 0

    # Sortino
    downside = returns[returns < 0].std() * np.sqrt(ann_factor)
    sortino = (returns.mean() * ann_factor) / downside if downside > 0 else 0

    # Max Drawdown
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdowns = (cum - running_max) / running_max
    max_dd = drawdowns.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Monthly win rate
    monthly_ret = returns.resample("ME").sum()
    win_rate = (monthly_ret > 0).mean()

    # Max drawdown duration (days)
    is_dd = cum < running_max
    if is_dd.any():
        dd_groups = (~is_dd).cumsum()
        dd_durations = is_dd.groupby(dd_groups).sum()
        max_dd_duration = int(dd_durations.max())
    else:
        max_dd_duration = 0

    # Total return
    total_ret_pct = (total_return - 1) * 100

    return {
        "name": name,
        "cagr_pct": round(cagr * 100, 2),
        "total_return_pct": round(total_ret_pct, 2),
        "annual_vol_pct": round(vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_dd_duration_days": max_dd_duration,
        "calmar": round(calmar, 3),
        "monthly_win_rate_pct": round(win_rate * 100, 1),
        "n_years": round(n_years, 1),
        "n_months": len(monthly_ret),
    }


# ── Regime Analysis ───────────────────────────────────────────────────────────

def regime_analysis(
    returns: pd.Series,
    spy_returns: pd.Series,
    name: str = "",
) -> dict:
    """
    Stratify performance by SPY regime:
      Green month: SPY monthly return > 0
      Red month:   SPY monthly return <= 0
    """
    monthly_strat = returns.resample("ME").sum()
    monthly_spy = spy_returns.resample("ME").sum()

    # Align indices
    common = monthly_strat.index.intersection(monthly_spy.index)
    monthly_strat = monthly_strat.loc[common]
    monthly_spy = monthly_spy.loc[common]

    green_mask = monthly_spy > 0
    red_mask = monthly_spy <= 0

    green_ret = monthly_strat[green_mask]
    red_ret = monthly_strat[red_mask]

    def _monthly_sharpe(s):
        if len(s) < 2 or s.std() == 0:
            return 0
        return (s.mean() / s.std()) * np.sqrt(12)

    return {
        "name": name,
        "green_months": int(green_mask.sum()),
        "green_avg_ret_pct": round(green_ret.mean() * 100, 3) if len(green_ret) > 0 else 0,
        "green_sharpe": round(_monthly_sharpe(green_ret), 3),
        "red_months": int(red_mask.sum()),
        "red_avg_ret_pct": round(red_ret.mean() * 100, 3) if len(red_ret) > 0 else 0,
        "red_sharpe": round(_monthly_sharpe(red_ret), 3),
        "regime_sharpe_gap": round(
            abs(_monthly_sharpe(green_ret) - _monthly_sharpe(red_ret))
            / max(abs(_monthly_sharpe(green_ret)), abs(_monthly_sharpe(red_ret)), 0.001),
            3,
        ),
    }


# ── Drawdown Detail ───────────────────────────────────────────────────────────

def top_drawdowns(returns: pd.Series, n: int = 5) -> list[dict]:
    """Find the N worst drawdown periods."""
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max

    drawdowns_list = []
    dd_copy = dd.copy()

    for _ in range(n):
        if dd_copy.min() >= -0.001:
            break
        trough_idx = dd_copy.idxmin()
        trough_val = dd_copy.loc[trough_idx]

        # Find start (last peak before trough)
        pre_trough = cum.loc[:trough_idx]
        peak_idx = pre_trough.idxmax()

        # Find recovery (next time cum >= peak value)
        post_trough = cum.loc[trough_idx:]
        peak_val = cum.loc[peak_idx]
        recovered = post_trough[post_trough >= peak_val]
        recovery_idx = recovered.index[0] if len(recovered) > 0 else cum.index[-1]

        drawdowns_list.append({
            "start": str(peak_idx.date()),
            "trough": str(trough_idx.date()),
            "recovery": str(recovery_idx.date()) if len(recovered) > 0 else "ongoing",
            "depth_pct": round(trough_val * 100, 2),
            "duration_days": (recovery_idx - peak_idx).days,
        })

        # Zero out this drawdown period so we find the next one
        dd_copy.loc[peak_idx:recovery_idx] = 0

    return drawdowns_list


# ── Exposure Analysis ─────────────────────────────────────────────────────────

def exposure_analysis(signals: pd.DataFrame) -> dict:
    """Analyze what % of time each asset was 'on' and overall exposure."""
    monthly_signals = signals.resample(REBALANCE_FREQ).last()
    monthly_signals = monthly_signals.loc[BACKTEST_START:]

    per_asset = {}
    for col in monthly_signals.columns:
        pct_on = monthly_signals[col].mean() * 100
        per_asset[col] = round(pct_on, 1)

    n_on = monthly_signals.sum(axis=1)
    avg_assets_on = n_on.mean()
    pct_fully_invested = (n_on == len(monthly_signals.columns)).mean() * 100
    pct_all_cash = (n_on == 0).mean() * 100

    return {
        "per_asset_on_pct": per_asset,
        "avg_assets_on": round(avg_assets_on, 1),
        "total_assets": len(monthly_signals.columns),
        "pct_fully_invested": round(pct_fully_invested, 1),
        "pct_all_cash": round(pct_all_cash, 1),
    }


# ── Year-by-Year Breakdown ───────────────────────────────────────────────────

def yearly_returns(returns_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Build year-by-year return table for all strategies."""
    yearly = {}
    for name, ret in returns_dict.items():
        annual = ret.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        yearly[name] = annual * 100

    df = pd.DataFrame(yearly)
    df.index = df.index.year
    df.index.name = "Year"
    return df.round(2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  WALK-FORWARD TREND-FOLLOWING BACKTEST (Multi-Asset)")
    print("=" * 72)
    print()

    # Download data
    all_needed = list(set(ALL_TICKERS + ["SPY", "AGG"]))
    prices = download_data(all_needed, START_DATE, END_DATE)

    # Filter to tickers we actually got data for
    available_tickers = [t for t in ALL_TICKERS if t in prices.columns]
    missing = set(ALL_TICKERS) - set(available_tickers)
    if missing:
        print(f"  Missing tickers (rate-limited or no data): {missing}")
    print(f"  Using {len(available_tickers)} tickers: {available_tickers}")

    # Compute signals
    sig_simple, sig_golden = compute_signals(prices[available_tickers])

    # Build portfolios
    print("\nBuilding portfolios...")
    ret_simple = build_portfolio(prices[available_tickers], sig_simple)
    ret_golden = build_portfolio(prices[available_tickers], sig_golden)
    ret_spy_bh = build_benchmark_bh(prices)
    ret_6040 = build_benchmark_6040(prices)

    strategies = {
        "Trend 200MA": ret_simple,
        "Golden Cross": ret_golden,
        "SPY Buy&Hold": ret_spy_bh,
        "60/40 SPY/AGG": ret_6040,
    }

    # Compute metrics
    print("\n" + "=" * 72)
    print("  PERFORMANCE SUMMARY")
    print("=" * 72)

    metrics_list = []
    for name, ret in strategies.items():
        m = compute_metrics(ret, name)
        metrics_list.append(m)

    # Print summary table
    header = f"{'Strategy':<16} {'CAGR%':>7} {'Vol%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>8} {'Calmar':>7} {'WinR%':>7}"
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        if "error" in m:
            print(f"{m['name']:<16} {'ERROR':>7}")
            continue
        print(
            f"{m['name']:<16} "
            f"{m['cagr_pct']:>7.2f} "
            f"{m['annual_vol_pct']:>7.2f} "
            f"{m['sharpe']:>7.3f} "
            f"{m['sortino']:>8.3f} "
            f"{m['max_drawdown_pct']:>8.2f} "
            f"{m['calmar']:>7.3f} "
            f"{m['monthly_win_rate_pct']:>7.1f}"
        )

    # Regime analysis
    print("\n" + "=" * 72)
    print("  REGIME ANALYSIS (Green/Red months by SPY)")
    print("=" * 72)

    regime_list = []
    for name, ret in strategies.items():
        r = regime_analysis(ret, ret_spy_bh, name)
        regime_list.append(r)

    header_r = f"{'Strategy':<16} {'GrnMo':>6} {'GrnAvg%':>8} {'GrnSh':>7} {'RedMo':>6} {'RedAvg%':>8} {'RedSh':>7} {'Gap':>6}"
    print(header_r)
    print("-" * len(header_r))
    for r in regime_list:
        print(
            f"{r['name']:<16} "
            f"{r['green_months']:>6} "
            f"{r['green_avg_ret_pct']:>8.3f} "
            f"{r['green_sharpe']:>7.3f} "
            f"{r['red_months']:>6} "
            f"{r['red_avg_ret_pct']:>8.3f} "
            f"{r['red_sharpe']:>7.3f} "
            f"{r['regime_sharpe_gap']:>6.3f}"
        )

    # Year-by-year
    print("\n" + "=" * 72)
    print("  YEAR-BY-YEAR RETURNS (%)")
    print("=" * 72)

    yr = yearly_returns(strategies)
    print(yr.to_string())

    # Exposure analysis
    print("\n" + "=" * 72)
    print("  EXPOSURE ANALYSIS (200MA Strategy)")
    print("=" * 72)

    exp_simple = exposure_analysis(sig_simple)
    print(f"\n  Average assets 'on': {exp_simple['avg_assets_on']:.1f} / {exp_simple['total_assets']}")
    print(f"  Fully invested:     {exp_simple['pct_fully_invested']:.1f}%")
    print(f"  All cash:           {exp_simple['pct_all_cash']:.1f}%")
    print("\n  Per-asset time 'on':")
    for ticker, pct in sorted(exp_simple["per_asset_on_pct"].items(), key=lambda x: -x[1]):
        bar = "#" * int(pct / 2)
        print(f"    {ticker:<5} {pct:>5.1f}%  {bar}")

    # Top drawdowns
    print("\n" + "=" * 72)
    print("  TOP 5 DRAWDOWNS")
    print("=" * 72)

    for name in ["Trend 200MA", "SPY Buy&Hold"]:
        dds = top_drawdowns(strategies[name])
        print(f"\n  {name}:")
        for i, dd in enumerate(dds, 1):
            print(
                f"    #{i}: {dd['depth_pct']:>6.2f}%  "
                f"{dd['start']} -> {dd['trough']}  "
                f"Recovery: {dd['recovery']}  ({dd['duration_days']}d)"
            )

    # Key insight
    print("\n" + "=" * 72)
    print("  KEY INSIGHT")
    print("=" * 72)

    m_trend = next(m for m in metrics_list if m["name"] == "Trend 200MA")
    m_spy = next(m for m in metrics_list if m["name"] == "SPY Buy&Hold")

    dd_reduction = (1 - m_trend["max_drawdown_pct"] / m_spy["max_drawdown_pct"]) * 100
    sharpe_diff = m_trend["sharpe"] - m_spy["sharpe"]

    print(f"\n  Max drawdown reduction vs SPY: {dd_reduction:+.1f}%")
    print(f"  Sharpe improvement:           {sharpe_diff:+.3f}")
    print(f"  CAGR trade-off:               {m_trend['cagr_pct'] - m_spy['cagr_pct']:+.2f}%")
    print(f"\n  The trend strategy {'DOES' if dd_reduction > 30 else 'does NOT'} deliver meaningful crisis alpha.")
    print(f"  It {'improves' if sharpe_diff > 0 else 'trails'} risk-adjusted returns (Sharpe).")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datestamp = datetime.now().strftime("%Y%m%d")
    outfile = OUTPUT_DIR / f"trend_backtest_{datestamp}.json"

    results = {
        "run_date": datetime.now().isoformat(),
        "backtest_start": BACKTEST_START,
        "backtest_end": str(prices.index[-1].date()),
        "universe": {k: v for k, v in UNIVERSE.items()},
        "config": {
            "sma_slow": SMA_SLOW,
            "sma_fast": SMA_FAST,
            "rebalance_freq": "monthly",
            "transaction_cost_bps": TRANSACTION_COST_BPS,
        },
        "metrics": metrics_list,
        "regime_analysis": regime_list,
        "yearly_returns": yr.to_dict(),
        "exposure": exp_simple,
        "top_drawdowns": {
            "trend_200ma": top_drawdowns(strategies["Trend 200MA"]),
            "spy_buyhold": top_drawdowns(strategies["SPY Buy&Hold"]),
        },
    }

    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {outfile}")
    print()


if __name__ == "__main__":
    main()
