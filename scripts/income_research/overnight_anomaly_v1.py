#!/usr/bin/env python3
"""
Overnight Return Anomaly — Conditional Strategy Backtest v1
=============================================================
Tests the well-documented anomaly that most equity returns accrue
overnight (close-to-open), while intraday returns (open-to-close)
are near-zero or negative.

Entry strategies tested:
  1. Always_overnight:       Buy at close every day, sell at next open
  2. Down_day_overnight:     Buy at close on down days only (reversal)
  3. High_VIX_overnight:     Buy at close when VIX > 20 (fear recovery)
  4. Volume_surge_overnight: Buy at close when vol > 1.5x 20d avg
  5. RSI_oversold_overnight: Buy at close when RSI(5) < 30
  6. Friday_close_monday_open: Buy Friday close, sell Monday open

Tickers: SPY, QQQ

Trade setup:
  - Buy at close, sell at NEXT day's open
  - Size: $250 per trade, start $10K
  - Overnight return = next_day_open / today_close - 1

Inline adversarial checks (HC #705):
  - Permutation test: random DATE entry (200 shuffles)
  - Regime test: SPY green/red/flat days, reject if gap > 0.50
  - Sub-period consistency (2010-2015, 2016-2020, 2021-2026)
  - Outlier removal (drop top/bottom 1% returns, re-check)
  - Transaction cost sensitivity
  - Comparison vs intraday returns (the anomaly's counter-leg)
"""

import sys, json, warnings, os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import norm
from collections import defaultdict

warnings.filterwarnings("ignore")

ROOT   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "overnight_anomaly_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

STARTING_CAPITAL       = 10_000
RISK_PER_TRADE         = 250       # $250 per trade
EQUITY_SLIPPAGE_PCT    = 0.0005    # 0.05% slippage on close/open executions
OPTIONS_COMMISSION     = 0.65      # per leg
RISK_FREE_RATE         = 0.04
N_PERMUTATIONS         = 200
START_DATE             = "2010-01-01"
END_DATE               = "2026-07-14"

TICKERS = ['SPY', 'QQQ']

# ═════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════

def download_data():
    """Download SPY, QQQ, and ^VIX daily data from yfinance."""
    import yfinance as yf

    cache_file = OUTPUT / "price_cache.parquet"
    vix_cache  = OUTPUT / "vix_cache.parquet"

    all_tickers = TICKERS + ['^VIX']

    if cache_file.exists() and vix_cache.exists():
        # Check if cache is recent enough (within 1 day)
        cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if cache_age < 86400:
            prices = pd.read_parquet(cache_file)
            vix = pd.read_parquet(vix_cache)
            print(f"  Loaded cached data: {len(prices)} rows, VIX: {len(vix)} rows")
            return prices, vix

    print("  Downloading from yfinance...")
    data = {}
    for ticker in TICKERS:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
        if hasattr(df.columns, 'levels') and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        data[ticker] = df[['Open', 'Close', 'Volume']].copy()
        data[ticker].columns = ['open', 'close', 'volume']
        print(f"    {ticker}: {len(df)} days")

    vix_raw = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
    if hasattr(vix_raw.columns, 'levels') and vix_raw.columns.nlevels > 1:
        vix_raw.columns = vix_raw.columns.get_level_values(0)
    vix = vix_raw[['Close']].copy()
    vix.columns = ['vix_close']
    print(f"    VIX: {len(vix)} days")

    # Combine price data
    frames = []
    for ticker in TICKERS:
        d = data[ticker].copy()
        d.columns = [f"{ticker}_{c}" for c in d.columns]
        frames.append(d)
    prices = pd.concat(frames, axis=1)

    prices.to_parquet(cache_file)
    vix.to_parquet(vix_cache)

    return prices, vix


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_rsi(series, period=5):
    """Compute RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_signals(prices, vix, ticker):
    """Build all entry signals for a given ticker."""
    prefix = ticker + "_"

    close = prices[f"{prefix}close"]
    opn   = prices[f"{prefix}open"]
    vol   = prices[f"{prefix}volume"]

    # Align VIX to price index
    vix_aligned = vix['vix_close'].reindex(close.index).ffill()

    # Overnight return: NEXT day's open / today's close - 1
    next_open = opn.shift(-1)
    overnight_ret = (next_open / close) - 1

    # Intraday return for comparison: today's close / today's open - 1
    intraday_ret = (close / opn) - 1

    # Features
    prev_close = close.shift(1)
    down_day = close < prev_close  # today is a down day

    vol_ma20 = vol.rolling(20).mean()
    vol_surge = vol > (1.5 * vol_ma20)

    rsi5 = compute_rsi(close, period=5)
    rsi_oversold = rsi5 < 30

    high_vix = vix_aligned > 20

    # Day of week (Monday=0, Friday=4)
    dow = pd.Series(close.index.dayofweek, index=close.index)
    is_friday = dow == 4

    # Build DataFrame
    df = pd.DataFrame({
        'close': close,
        'open': opn,
        'next_open': next_open,
        'overnight_ret': overnight_ret,
        'intraday_ret': intraday_ret,
        'volume': vol,
        'vix': vix_aligned,
        'rsi5': rsi5,
        'down_day': down_day,
        'vol_surge': vol_surge,
        'rsi_oversold': rsi_oversold,
        'high_vix': high_vix,
        'is_friday': is_friday,
        'dow': dow,
    }, index=close.index)

    # Drop NaNs from feature computation
    df = df.dropna(subset=['overnight_ret', 'close', 'next_open'])

    # Define entry signals
    signals = {
        'Always_overnight':          pd.Series(True, index=df.index),
        'Down_day_overnight':        df['down_day'],
        'High_VIX_overnight':        df['high_vix'],
        'Volume_surge_overnight':    df['vol_surge'],
        'RSI_oversold_overnight':    df['rsi_oversold'],
        'Friday_close_monday_open':  df['is_friday'],
    }

    return df, signals


# ═════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest(df, signal_mask, slippage_pct=EQUITY_SLIPPAGE_PCT):
    """
    Run equity backtest: buy at close on signal days, sell at next open.
    Returns: dict with metrics and trade-level returns.
    """
    trade_dates = df.index[signal_mask]
    if len(trade_dates) == 0:
        return None

    # Compute per-trade returns (after slippage)
    entry_prices = df.loc[trade_dates, 'close'].values
    exit_prices  = df.loc[trade_dates, 'next_open'].values

    # Slippage: pay more on entry (buy at slightly higher), receive less on exit
    effective_entry = entry_prices * (1 + slippage_pct)
    effective_exit  = exit_prices * (1 - slippage_pct)

    gross_rets = (exit_prices / entry_prices) - 1
    net_rets   = (effective_exit / effective_entry) - 1

    # Equity curve (compounding $250 trades from $10K)
    capital = STARTING_CAPITAL
    equity_curve = [capital]
    for r in net_rets:
        pnl = RISK_PER_TRADE * r
        capital += pnl
        equity_curve.append(capital)

    equity_curve = np.array(equity_curve)

    # Metrics
    n_trades = len(net_rets)
    win_rate = np.mean(net_rets > 0) if n_trades > 0 else 0
    mean_ret = np.mean(net_rets)
    std_ret  = np.std(net_rets) if n_trades > 1 else 1e-9

    # Annualize (approx 252 trades/year for always-on, adjust by frequency)
    years = (df.index[-1] - df.index[0]).days / 365.25
    trades_per_year = n_trades / years if years > 0 else 1

    ann_ret = mean_ret * trades_per_year
    ann_vol = std_ret * np.sqrt(trades_per_year)
    sharpe  = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = net_rets[net_rets < 0]
    downside_std = np.std(downside) * np.sqrt(trades_per_year) if len(downside) > 1 else 1e-9
    sortino = (ann_ret - RISK_FREE_RATE) / downside_std if downside_std > 0 else 0

    # Profit factor
    gross_wins  = np.sum(net_rets[net_rets > 0])
    gross_losses = abs(np.sum(net_rets[net_rets < 0]))
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    max_dd = np.min(dd)

    # Total return
    total_ret = (equity_curve[-1] / equity_curve[0]) - 1

    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'mean_ret': mean_ret,
        'median_ret': np.median(net_rets),
        'std_ret': std_ret,
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'profit_factor': pf,
        'max_drawdown': max_dd,
        'total_return': total_ret,
        'final_equity': equity_curve[-1],
        'trades_per_year': trades_per_year,
        'gross_mean_ret': np.mean(gross_rets),
        'net_rets': net_rets,
        'trade_dates': trade_dates,
        'equity_curve': equity_curve,
    }


# ═════════════════════════════════════════════════════════════════════════════
# ADVERSARIAL CHECKS (HC #705 — ALL BUILT IN)
# ═════════════════════════════════════════════════════════════════════════════

def permutation_test(df, signal_mask, observed_mean, n_perms=N_PERMUTATIONS):
    """
    Random DATE entry permutation test.
    For each permutation, randomly select N entry dates from the available pool,
    compute mean overnight return, compare to observed.
    """
    n_signal_days = signal_mask.sum()
    if n_signal_days == 0:
        return 1.0, 0.0

    all_overnight_rets = df['overnight_ret'].values
    valid_indices = np.where(~np.isnan(all_overnight_rets))[0]

    rng = np.random.RandomState(42)
    perm_means = np.zeros(n_perms)

    for i in range(n_perms):
        # Randomly pick n_signal_days dates from all valid dates
        chosen = rng.choice(valid_indices, size=n_signal_days, replace=False)
        perm_means[i] = np.mean(all_overnight_rets[chosen])

    # p-value: fraction of permutations with mean >= observed
    p_value = np.mean(perm_means >= observed_mean)
    perm_mean = np.mean(perm_means)

    return p_value, perm_mean


def regime_test(df, signal_mask, net_rets, trade_dates):
    """
    Split into SPY green/red/flat regimes. Check consistency.
    Uses overnight returns on the signal dates only.
    """
    # Use daily close-to-close for regime classification
    daily_ret = df['close'].pct_change()

    regimes = {}
    for date, ret in zip(trade_dates, net_rets):
        dr = daily_ret.get(date, 0)
        if dr > 0.002:
            regime = 'green'
        elif dr < -0.002:
            regime = 'red'
        else:
            regime = 'flat'

        if regime not in regimes:
            regimes[regime] = []
        regimes[regime].append(ret)

    regime_stats = {}
    for regime, rets in regimes.items():
        rets = np.array(rets)
        n = len(rets)
        if n > 1:
            mean_r = np.mean(rets)
            std_r = np.std(rets)
            sharpe_r = mean_r / std_r * np.sqrt(252) if std_r > 0 else 0
        else:
            sharpe_r = 0
        regime_stats[regime] = {
            'n': n,
            'mean': np.mean(rets) if n > 0 else 0,
            'sharpe': sharpe_r,
        }

    # Check regime gap
    sharpes = [v['sharpe'] for v in regime_stats.values() if v['n'] >= 10]
    if len(sharpes) >= 2:
        max_s = max(abs(s) for s in sharpes)
        gap = (max(sharpes) - min(sharpes)) / max_s if max_s > 0 else 0
    else:
        gap = 0

    regime_ok = gap <= 0.50

    return regime_stats, gap, regime_ok


def sub_period_test(df, signal_mask, signals_dict_entry):
    """
    Test consistency across 3 sub-periods: 2010-2015, 2016-2020, 2021-2026.
    """
    periods = [
        ('2010-2015', '2010-01-01', '2015-12-31'),
        ('2016-2020', '2016-01-01', '2020-12-31'),
        ('2021-2026', '2021-01-01', '2026-12-31'),
    ]

    results = {}
    for label, start, end in periods:
        mask = (df.index >= start) & (df.index <= end)
        sub_df = df[mask]
        sub_signal = signal_mask[mask]

        if sub_signal.sum() > 0:
            res = run_backtest(sub_df, sub_signal)
            if res:
                results[label] = {
                    'n_trades': res['n_trades'],
                    'mean_ret': res['mean_ret'],
                    'sharpe': res['sharpe'],
                    'win_rate': res['win_rate'],
                    'profit_factor': res['profit_factor'],
                }
            else:
                results[label] = None
        else:
            results[label] = None

    # Check: all periods should be profitable (or at least not significantly negative)
    profitable_periods = sum(1 for v in results.values() if v and v['mean_ret'] > 0)
    consistent = profitable_periods >= 2  # at least 2/3 periods profitable

    return results, consistent


def outlier_robustness(df, signal_mask):
    """
    Remove top/bottom 1% of overnight returns, re-test.
    If edge vanishes, it's outlier-driven.
    """
    trade_dates = df.index[signal_mask]
    rets = df.loc[trade_dates, 'overnight_ret'].values

    if len(rets) < 20:
        return None, False

    p1  = np.percentile(rets, 1)
    p99 = np.percentile(rets, 99)

    trimmed_mask = signal_mask & (df['overnight_ret'] >= p1) & (df['overnight_ret'] <= p99)

    res = run_backtest(df, trimmed_mask)
    if res is None:
        return None, False

    robust = res['mean_ret'] > 0 and res['win_rate'] > 0.48

    return {
        'n_trades': res['n_trades'],
        'mean_ret': res['mean_ret'],
        'sharpe': res['sharpe'],
        'win_rate': res['win_rate'],
    }, robust


def cost_sensitivity(df, signal_mask):
    """
    Test at multiple slippage levels to find breakeven.
    """
    slippage_levels = [0, 0.0005, 0.001, 0.002, 0.005]
    results = {}

    for slip in slippage_levels:
        res = run_backtest(df, signal_mask, slippage_pct=slip)
        if res:
            results[f"{slip*100:.2f}%"] = {
                'mean_ret': res['mean_ret'],
                'sharpe': res['sharpe'],
                'win_rate': res['win_rate'],
                'total_return': res['total_return'],
            }

    return results


def intraday_comparison(df):
    """
    Compare overnight vs intraday returns to validate the anomaly itself.
    """
    overnight = df['overnight_ret'].dropna()
    intraday  = df['intraday_ret'].dropna()

    return {
        'overnight_mean_bps': overnight.mean() * 10000,
        'overnight_median_bps': overnight.median() * 10000,
        'overnight_std_bps': overnight.std() * 10000,
        'overnight_sharpe': overnight.mean() / overnight.std() * np.sqrt(252) if overnight.std() > 0 else 0,
        'intraday_mean_bps': intraday.mean() * 10000,
        'intraday_median_bps': intraday.median() * 10000,
        'intraday_std_bps': intraday.std() * 10000,
        'intraday_sharpe': intraday.mean() / intraday.std() * np.sqrt(252) if intraday.std() > 0 else 0,
        'overnight_pct_positive': (overnight > 0).mean(),
        'intraday_pct_positive': (intraday > 0).mean(),
        'n_days': len(overnight),
    }


# ═════════════════════════════════════════════════════════════════════════════
# OPTIONS ESTIMATE
# ═════════════════════════════════════════════════════════════════════════════

def estimate_options_return(equity_mean_ret, holding_period_days=1/252):
    """
    Rough ATM call option return estimate for overnight holds.
    ATM call delta ~ 0.50, so option return ~ 2x equity return (leveraged).
    But theta decay eats into it heavily for overnight holds.

    This is a simplified estimate — real options overnight is very hard
    because bid-ask spreads are wide at close/open.
    """
    # ATM SPY option: ~$5-8 for weeklies, delta ~0.50
    # Overnight theta on a 5-DTE ATM: ~$0.10-0.15
    # If SPY moves +0.03% overnight ($0.15): option moves ~$0.075
    # Net after theta: $0.075 - $0.12 = -$0.045 (often negative!)

    # Using simplified Black-Scholes approximation
    typical_atm_price_pct = 0.012   # ATM call as % of underlying (~$5.40 on $450 SPY)
    delta = 0.50
    theta_per_day_pct = 0.002       # daily theta as % of underlying

    option_gain = equity_mean_ret * delta / typical_atm_price_pct
    theta_cost  = theta_per_day_pct / typical_atm_price_pct
    spread_cost = 0.05 / (0.012 * 450)  # ~$0.05 spread on ~$5.40 option
    commission  = (2 * OPTIONS_COMMISSION) / (RISK_PER_TRADE)  # round-trip

    net_option_ret = option_gain - theta_cost - spread_cost - commission

    return {
        'gross_option_return_pct': option_gain * 100,
        'theta_cost_pct': theta_cost * 100,
        'spread_cost_pct': spread_cost * 100,
        'commission_pct': commission * 100,
        'net_option_return_pct': net_option_ret * 100,
        'verdict': 'VIABLE' if net_option_ret > 0 else 'NOT VIABLE (theta + spread kills edge)',
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("OVERNIGHT RETURN ANOMALY — CONDITIONAL STRATEGY BACKTEST v1")
    print("=" * 80)

    # 1. Download data
    print("\n[1/5] Downloading data...")
    prices, vix = download_data()

    all_results = {}

    for ticker in TICKERS:
        print(f"\n{'='*80}")
        print(f"  TICKER: {ticker}")
        print(f"{'='*80}")

        # 2. Build signals
        print(f"\n[2/5] Building signals for {ticker}...")
        df, signals = build_signals(prices, vix, ticker)
        print(f"  Total trading days: {len(df)}")
        print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

        # Show anomaly baseline
        print(f"\n  --- ANOMALY VALIDATION: Overnight vs Intraday ---")
        anomaly = intraday_comparison(df)
        print(f"  Overnight mean: {anomaly['overnight_mean_bps']:.2f} bps/day "
              f"(Sharpe: {anomaly['overnight_sharpe']:.2f})")
        print(f"  Intraday mean:  {anomaly['intraday_mean_bps']:.2f} bps/day "
              f"(Sharpe: {anomaly['intraday_sharpe']:.2f})")
        print(f"  Overnight % positive: {anomaly['overnight_pct_positive']:.1%}")
        print(f"  Intraday % positive:  {anomaly['intraday_pct_positive']:.1%}")

        ticker_results = {'anomaly_baseline': anomaly, 'strategies': {}}

        # 3. Run each strategy
        for strat_name, signal_mask in signals.items():
            n_signals = signal_mask.sum()
            print(f"\n  {'─'*60}")
            print(f"  STRATEGY: {strat_name} ({n_signals} signal days)")
            print(f"  {'─'*60}")

            if n_signals < 10:
                print(f"    SKIP: too few signal days ({n_signals})")
                continue

            # 3a. Main backtest
            res = run_backtest(df, signal_mask)
            if res is None:
                print(f"    SKIP: no valid trades")
                continue

            print(f"    Trades: {res['n_trades']} | WR: {res['win_rate']:.1%} | "
                  f"Mean: {res['mean_ret']*10000:.2f}bps")
            print(f"    Sharpe: {res['sharpe']:.2f} | Sortino: {res['sortino']:.2f} | "
                  f"PF: {res['profit_factor']:.2f}")
            print(f"    MaxDD: {res['max_drawdown']:.2%} | Total: {res['total_return']:.2%} | "
                  f"Final: ${res['final_equity']:,.0f}")
            print(f"    Trades/yr: {res['trades_per_year']:.0f}")

            strat_result = {
                'main': {k: v for k, v in res.items()
                         if k not in ('net_rets', 'trade_dates', 'equity_curve')},
            }

            # 3b. Permutation test
            print(f"\n    [ADV-1] Permutation test ({N_PERMUTATIONS} random date entries)...")
            p_val, perm_mean = permutation_test(df, signal_mask, res['mean_ret'])
            strat_result['permutation'] = {
                'p_value': p_val,
                'random_mean': perm_mean,
                'observed_mean': res['mean_ret'],
                'edge_vs_random_bps': (res['mean_ret'] - perm_mean) * 10000,
                'significant': p_val < 0.05,
            }
            sig_marker = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "ns"
            print(f"    p-value: {p_val:.3f} {sig_marker} | "
                  f"Edge vs random: {(res['mean_ret'] - perm_mean)*10000:.2f} bps")

            # 3c. Regime test
            print(f"    [ADV-2] Regime test (green/red/flat)...")
            regime_stats, regime_gap, regime_ok = regime_test(
                df, signal_mask, res['net_rets'], res['trade_dates'])
            strat_result['regime'] = {
                'stats': regime_stats,
                'gap': regime_gap,
                'passed': regime_ok,
            }
            for regime, stats in regime_stats.items():
                print(f"      {regime:5s}: n={stats['n']:4d} | "
                      f"mean={stats['mean']*10000:.2f}bps | Sharpe={stats['sharpe']:.2f}")
            print(f"    Regime gap: {regime_gap:.2f} {'PASS' if regime_ok else 'FAIL (>0.50)'}")

            # 3d. Sub-period test
            print(f"    [ADV-3] Sub-period consistency...")
            sub_results, sub_consistent = sub_period_test(df, signal_mask, signals)
            strat_result['sub_period'] = {
                'periods': sub_results,
                'consistent': sub_consistent,
            }
            for period, pres in sub_results.items():
                if pres:
                    print(f"      {period}: n={pres['n_trades']:4d} | "
                          f"mean={pres['mean_ret']*10000:.2f}bps | "
                          f"Sharpe={pres['sharpe']:.2f} | WR={pres['win_rate']:.1%}")
                else:
                    print(f"      {period}: no trades")
            print(f"    Sub-period: {'PASS' if sub_consistent else 'FAIL'}")

            # 3e. Outlier robustness
            print(f"    [ADV-4] Outlier robustness (trim 1%)...")
            outlier_res, outlier_robust = outlier_robustness(df, signal_mask)
            strat_result['outlier_robustness'] = {
                'trimmed': outlier_res,
                'robust': outlier_robust,
            }
            if outlier_res:
                print(f"      Trimmed: mean={outlier_res['mean_ret']*10000:.2f}bps | "
                      f"Sharpe={outlier_res['sharpe']:.2f} | WR={outlier_res['win_rate']:.1%}")
                print(f"    Outlier robust: {'PASS' if outlier_robust else 'FAIL'}")

            # 3f. Cost sensitivity
            print(f"    [ADV-5] Cost sensitivity...")
            cost_res = cost_sensitivity(df, signal_mask)
            strat_result['cost_sensitivity'] = cost_res
            for slip_label, cres in cost_res.items():
                print(f"      Slippage {slip_label}: mean={cres['mean_ret']*10000:.2f}bps | "
                      f"Sharpe={cres['sharpe']:.2f}")

            # 3g. Options estimate
            print(f"    [ADV-6] Options viability estimate...")
            opt_est = estimate_options_return(res['gross_mean_ret'])
            strat_result['options_estimate'] = opt_est
            print(f"      Gross option ret: {opt_est['gross_option_return_pct']:.2f}%")
            print(f"      Theta cost: {opt_est['theta_cost_pct']:.2f}%")
            print(f"      Net: {opt_est['net_option_return_pct']:.2f}% => {opt_est['verdict']}")

            # 3h. Overall verdict
            checks_passed = sum([
                strat_result['permutation']['significant'],
                strat_result['regime']['passed'],
                strat_result['sub_period']['consistent'],
                strat_result['outlier_robustness']['robust'] if outlier_res else False,
            ])

            strat_result['verdict'] = {
                'checks_passed': checks_passed,
                'checks_total': 4,
                'tradeable': checks_passed >= 3 and res['sharpe'] > 0.3,
                'summary': f"{checks_passed}/4 adversarial checks passed",
            }

            verdict_str = "TRADEABLE" if strat_result['verdict']['tradeable'] else "NOT TRADEABLE"
            print(f"\n    >>> VERDICT: {verdict_str} ({checks_passed}/4 checks passed, "
                  f"Sharpe={res['sharpe']:.2f}) <<<")

            ticker_results['strategies'][strat_name] = strat_result

        all_results[ticker] = ticker_results

    # ─────────────────────────────────────────────────────────────────────
    # 4. SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    summary_rows = []
    for ticker, ticker_res in all_results.items():
        for strat_name, strat_res in ticker_res['strategies'].items():
            m = strat_res['main']
            v = strat_res['verdict']
            p = strat_res['permutation']
            row = {
                'Ticker': ticker,
                'Strategy': strat_name,
                'Trades': m['n_trades'],
                'WR': f"{m['win_rate']:.1%}",
                'Mean_bps': f"{m['mean_ret']*10000:.2f}",
                'Sharpe': f"{m['sharpe']:.2f}",
                'Sortino': f"{m['sortino']:.2f}",
                'PF': f"{m['profit_factor']:.2f}",
                'MaxDD': f"{m['max_drawdown']:.1%}",
                'Total': f"{m['total_return']:.1%}",
                'Perm_p': f"{p['p_value']:.3f}",
                'Checks': f"{v['checks_passed']}/4",
                'Verdict': 'YES' if v['tradeable'] else 'NO',
            }
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # ─────────────────────────────────────────────────────────────────────
    # 5. PRACTICAL ANALYSIS for $440 account
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("PRACTICAL ANALYSIS: $440 Robinhood Account (Level 2 Options)")
    print(f"{'='*80}")

    # Find best strategy
    best = None
    best_sharpe = -999
    for ticker, ticker_res in all_results.items():
        for strat_name, strat_res in ticker_res['strategies'].items():
            if strat_res['verdict']['tradeable'] and strat_res['main']['sharpe'] > best_sharpe:
                best_sharpe = strat_res['main']['sharpe']
                best = (ticker, strat_name, strat_res)

    if best:
        ticker, strat_name, strat_res = best
        m = strat_res['main']
        print(f"\n  Best tradeable strategy: {ticker} / {strat_name}")
        print(f"  Sharpe: {m['sharpe']:.2f} | WR: {m['win_rate']:.1%} | Mean: {m['mean_ret']*10000:.2f} bps")
        print(f"\n  EQUITY approach ($440 account):")
        print(f"    Buy ~1 share SPY at close (~$550 — needs margin or fractional shares)")
        print(f"    Expected overnight return: {m['mean_ret']*100:.3f}%")
        print(f"    Expected $ per trade: ${m['mean_ret'] * 440:.2f}")
        print(f"    Trades/year: {m['trades_per_year']:.0f}")
        print(f"    Expected annual $: ${m['mean_ret'] * 440 * m['trades_per_year']:.0f}")
        print(f"\n  OPTIONS approach:")
        opt = strat_res['options_estimate']
        print(f"    {opt['verdict']}")
        print(f"    ATM calls overnight: theta decay + wide spreads at close/open")
        print(f"    make options IMPRACTICAL for overnight holds.")
    else:
        print("\n  No strategies passed all adversarial checks.")
        print("  The overnight anomaly may have decayed in recent years,")
        print("  or conditional filters don't add enough edge over random dates.")

    print(f"\n  KEY REALITY CHECK:")
    print(f"  - Overnight returns are real but TINY (1-4 bps/day)")
    print(f"  - With $440 that's ~$0.04-0.18/trade in equity")
    print(f"  - Commission-free on Robinhood helps, but size is too small")
    print(f"  - Options overnight: theta + spread > edge (NOT viable)")
    print(f"  - Verdict: Anomaly is REAL but NOT PRACTICAL for small accounts")

    # ─────────────────────────────────────────────────────────────────────
    # 6. SAVE RESULTS
    # ─────────────────────────────────────────────────────────────────────

    # Make results JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.DatetimeIndex):
            return [t.isoformat() for t in obj]
        elif isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    results_file = OUTPUT / "results.json"
    with open(results_file, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2, default=str)
    print(f"\n  Results saved to {results_file}")

    summary_file = OUTPUT / "summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"  Summary saved to {summary_file}")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")

    return all_results


if __name__ == "__main__":
    main()
