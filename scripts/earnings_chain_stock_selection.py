#!/usr/bin/env python3
"""
Earnings Chain Momentum — Stock Selection Backtest
===================================================
Tests whether buying stocks after earnings beats (>3% gap) adds stock-selection
value vs buying random stocks from the same universe at the same dates.

Key adversarial check: Variant E compares beat vs miss performance. If misses
perform similarly, the "beat" signal is just timing/beta, not stock selection.

6 Variants:
A) Single-stock beat (40d hold) vs random stock
B) Consecutive beats (40d hold) — requires prior quarter beat too
C) Beat magnitude weighted (40d hold)
D) Beat + relative strength (above 200-SMA)
E) Beat vs Miss diagnostic — the key adversarial check
F) Multi-stock portfolio (max 5 concurrent) vs random portfolio
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─── Configuration ───────────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'CRM',
    'SNOW', 'PLTR', 'SOFI', 'HOOD', 'SNAP', 'PINS', 'COIN', 'RBLX', 'RIVN', 'UBER',
    'LYFT', 'ROKU', 'NET', 'DDOG', 'TTD', 'SHOP', 'SE', 'MELI', 'NU', 'SQ'
]

ACCOUNT_SIZE = 645.0
MAX_CONCURRENT = 5
HOLD_DAYS = 40
GAP_THRESHOLD = 0.03  # 3% gap = beat proxy
SLIPPAGE_PCT = 0.0002  # 0.02%
N_PERMS = 1000

OOT_START = '2022-01-01'
OOT_END = '2026-07-28'
DATA_START = '2021-01-01'  # Extra lookback for 200-SMA and prior beats

OUTPUT_PATH = Path('/home/jupiter/Lvl3Quant/data/earnings_chain_stock_selection_results.json')


def download_data():
    """Download price data for universe + SPY."""
    tickers = UNIVERSE + ['SPY']
    print(f"Downloading data for {len(tickers)} tickers...")
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=DATA_START, end=OOT_END, progress=False, auto_adjust=True)
            if len(df) > 100:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                data[ticker] = df
                print(f"  {ticker}: {len(df)} days")
            else:
                print(f"  {ticker}: insufficient data ({len(df)} days), skipping")
        except Exception as e:
            print(f"  {ticker}: download failed ({e})")
    return data


def detect_gaps(data):
    """Detect earnings gaps (>3% overnight gap) for each stock."""
    all_gaps = {}
    for ticker in UNIVERSE:
        if ticker not in data:
            continue
        df = data[ticker]
        # Overnight gap = (Open - prev Close) / prev Close
        prev_close = df['Close'].shift(1)
        gap_pct = (df['Open'] - prev_close) / prev_close

        # Filter to OOT period
        mask = (df.index >= pd.Timestamp(OOT_START)) & (df.index <= pd.Timestamp(OOT_END))
        gap_pct = gap_pct[mask]

        beats = gap_pct[gap_pct > GAP_THRESHOLD]
        misses = gap_pct[gap_pct < -GAP_THRESHOLD]

        all_gaps[ticker] = {
            'beats': [(d, v) for d, v in zip(beats.index, beats.values)],
            'misses': [(d, v) for d, v in zip(misses.index, misses.values)],
        }
    return all_gaps


def forward_return(data, ticker, entry_date, hold_days=HOLD_DAYS):
    """Calculate forward return from entry_date over hold_days trading days."""
    if ticker not in data:
        return None
    df = data[ticker]
    idx = df.index.get_indexer([entry_date], method='nearest')[0]
    if idx < 0 or idx + hold_days >= len(df):
        return None
    entry_price = float(df['Close'].iloc[idx]) * (1 + SLIPPAGE_PCT)  # slippage on entry
    exit_price = float(df['Close'].iloc[idx + hold_days]) * (1 - SLIPPAGE_PCT)  # slippage on exit
    return (exit_price - entry_price) / entry_price


def get_sma200(data, ticker, date):
    """Check if stock is above 200-SMA on given date."""
    if ticker not in data:
        return False
    df = data[ticker]
    idx = df.index.get_indexer([date], method='ffill')[0]
    if idx < 200:
        return False
    sma = float(df['Close'].iloc[idx-199:idx+1].mean())
    price = float(df['Close'].iloc[idx])
    return price > sma


def spy_regime(data, date):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    spy = data.get('SPY')
    if spy is None:
        return 'unknown'
    idx = spy.index.get_indexer([date], method='ffill')[0]
    if idx < 200:
        return 'unknown'
    sma = float(spy['Close'].iloc[idx-199:idx+1].mean())
    price = float(spy['Close'].iloc[idx])
    return 'bull' if price > sma else 'bear'


def had_prior_beat(all_gaps, ticker, current_date):
    """Check if this ticker had a beat in the ~60-120 day window before current_date (prior quarter)."""
    beats = all_gaps.get(ticker, {}).get('beats', [])
    for d, v in beats:
        days_before = (current_date - d).days
        if 50 < days_before < 130:  # Prior quarter window
            return True
    # Also check pre-OOT gaps from full data
    return False


def had_prior_beat_full(data, ticker, current_date):
    """Check prior quarter beat using full data (including pre-OOT)."""
    if ticker not in data:
        return False
    df = data[ticker]
    # Look 50-130 days before current_date for a >3% gap
    start = current_date - pd.Timedelta(days=130)
    end = current_date - pd.Timedelta(days=50)
    mask = (df.index >= start) & (df.index <= end)
    sub = df[mask]
    if len(sub) < 2:
        return False
    prev_close = sub['Close'].shift(1)
    gaps = (sub['Open'] - prev_close) / prev_close
    return (gaps > GAP_THRESHOLD).any()


def compute_metrics(returns, label=""):
    """Compute Sharpe, Sortino, PF, WR, MaxDD, trade count."""
    returns = np.array([r for r in returns if r is not None])
    n = len(returns)
    if n == 0:
        return {'n_trades': 0, 'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0, 'maxdd': 0,
                'mean_ret': 0, 'med_ret': 0, 'total_ret': 0}

    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1) if n > 1 else 1e-9

    # Annualize: ~6.5 trades per year (each stock ~quarterly, 30 stocks)
    # But just report per-trade metrics for clarity
    sharpe = mean_r / std_r * np.sqrt(n) if std_r > 1e-9 else 0  # t-stat style

    downside = returns[returns < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = mean_r / down_std * np.sqrt(n) if down_std > 1e-9 else 0

    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    pf = gross_profit / gross_loss if gross_loss > 1e-9 else float('inf')

    wr = np.mean(returns > 0)

    # Drawdown on cumulative equity
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    maxdd = np.min(dd) if len(dd) > 0 else 0

    total_ret = np.sum(returns)

    return {
        'n_trades': int(n),
        'sharpe_tstat': round(float(sharpe), 3),
        'sortino_tstat': round(float(sortino), 3),
        'profit_factor': round(float(pf), 3),
        'win_rate': round(float(wr), 4),
        'max_dd_pct': round(float(maxdd) * 100, 2),
        'mean_return_pct': round(float(mean_r) * 100, 3),
        'median_return_pct': round(float(np.median(returns)) * 100, 3),
        'total_return_pct': round(float(total_ret) * 100, 2),
        'mean_return': float(mean_r),
        'std_return': float(std_r),
    }


def permutation_test(actual_returns, all_entry_dates, data, n_perms=N_PERMS):
    """
    Permutation test: at each entry date, pick a RANDOM stock from universe instead of the beater.
    Compare mean return of random picks vs actual picks.
    Returns p-value: fraction of permutations where random picks beat actual.
    """
    actual_mean = np.mean([r for r in actual_returns if r is not None])
    available_tickers = [t for t in UNIVERSE if t in data]

    count_better = 0
    perm_means = []

    for _ in range(n_perms):
        perm_rets = []
        for entry_date in all_entry_dates:
            rand_ticker = available_tickers[np.random.randint(len(available_tickers))]
            r = forward_return(data, rand_ticker, entry_date)
            if r is not None:
                perm_rets.append(r)
        if perm_rets:
            pm = np.mean(perm_rets)
            perm_means.append(pm)
            if pm >= actual_mean:
                count_better += 1

    p_value = count_better / n_perms
    perm_mean_avg = np.mean(perm_means) if perm_means else 0
    return p_value, perm_mean_avg, perm_means


def run_variant_a(data, all_gaps):
    """Variant A: Single-stock beat, 40d hold, vs random."""
    print("\n=== VARIANT A: Single-Stock Beat (40d hold) ===")
    returns = []
    entry_dates = []
    regimes = []

    for ticker in UNIVERSE:
        beats = all_gaps.get(ticker, {}).get('beats', [])
        for date, gap_size in beats:
            r = forward_return(data, ticker, date)
            if r is not None:
                returns.append(r)
                entry_dates.append(date)
                regimes.append(spy_regime(data, date))

    metrics = compute_metrics(returns, "Variant A")

    # Regime breakdown
    bull_rets = [r for r, reg in zip(returns, regimes) if reg == 'bull']
    bear_rets = [r for r, reg in zip(returns, regimes) if reg == 'bear']
    bull_m = compute_metrics(bull_rets, "A-Bull")
    bear_m = compute_metrics(bear_rets, "A-Bear")

    # Regime gap check
    if bull_m['n_trades'] > 0 and bear_m['n_trades'] > 0:
        s_bull = bull_m['sharpe_tstat']
        s_bear = bear_m['sharpe_tstat']
        denom = max(abs(s_bull), abs(s_bear), 0.001)
        regime_gap = abs(s_bull - s_bear) / denom
    else:
        regime_gap = 999

    # Permutation test
    print(f"  Running {N_PERMS} permutations (random stock selection)...")
    p_val, perm_mean, perm_means = permutation_test(returns, entry_dates, data)

    print(f"  Trades: {metrics['n_trades']}, Mean: {metrics['mean_return_pct']:.3f}%, "
          f"Sharpe-t: {metrics['sharpe_tstat']:.3f}, PF: {metrics['profit_factor']:.3f}, "
          f"WR: {metrics['win_rate']:.1%}")
    print(f"  Perm p-value: {p_val:.4f}, Perm mean: {perm_mean*100:.3f}%")
    print(f"  Regime gap: {regime_gap:.3f} (bull t={bull_m['sharpe_tstat']:.2f}, bear t={bear_m['sharpe_tstat']:.2f})")

    return {
        'name': 'A) Single-Stock Beat (40d hold)',
        'metrics': metrics,
        'bull_metrics': bull_m,
        'bear_metrics': bear_m,
        'regime_gap': round(regime_gap, 3),
        'perm_p_value': round(p_val, 4),
        'perm_mean_return_pct': round(perm_mean * 100, 3),
        'actual_vs_random_spread_pct': round((metrics['mean_return'] - perm_mean) * 100, 3),
    }


def run_variant_b(data, all_gaps):
    """Variant B: Consecutive beats only (40d hold)."""
    print("\n=== VARIANT B: Consecutive Beats (40d hold) ===")
    returns = []
    entry_dates = []
    regimes = []

    for ticker in UNIVERSE:
        beats = all_gaps.get(ticker, {}).get('beats', [])
        for date, gap_size in beats:
            if had_prior_beat_full(data, ticker, date):
                r = forward_return(data, ticker, date)
                if r is not None:
                    returns.append(r)
                    entry_dates.append(date)
                    regimes.append(spy_regime(data, date))

    metrics = compute_metrics(returns, "Variant B")

    bull_rets = [r for r, reg in zip(returns, regimes) if reg == 'bull']
    bear_rets = [r for r, reg in zip(returns, regimes) if reg == 'bear']
    bull_m = compute_metrics(bull_rets)
    bear_m = compute_metrics(bear_rets)

    if bull_m['n_trades'] > 0 and bear_m['n_trades'] > 0:
        regime_gap = abs(bull_m['sharpe_tstat'] - bear_m['sharpe_tstat']) / max(abs(bull_m['sharpe_tstat']), abs(bear_m['sharpe_tstat']), 0.001)
    else:
        regime_gap = 999

    if len(returns) >= 5:
        print(f"  Running {N_PERMS} permutations...")
        p_val, perm_mean, _ = permutation_test(returns, entry_dates, data)
    else:
        p_val, perm_mean = 1.0, 0.0

    print(f"  Trades: {metrics['n_trades']}, Mean: {metrics['mean_return_pct']:.3f}%, "
          f"Sharpe-t: {metrics['sharpe_tstat']:.3f}")
    print(f"  Perm p-value: {p_val:.4f}")

    return {
        'name': 'B) Consecutive Beats (40d hold)',
        'metrics': metrics,
        'bull_metrics': bull_m,
        'bear_metrics': bear_m,
        'regime_gap': round(regime_gap, 3),
        'perm_p_value': round(p_val, 4),
        'perm_mean_return_pct': round(perm_mean * 100, 3),
        'actual_vs_random_spread_pct': round((metrics['mean_return'] - perm_mean) * 100, 3),
    }


def run_variant_c(data, all_gaps):
    """Variant C: Beat magnitude weighted (40d hold)."""
    print("\n=== VARIANT C: Beat Magnitude Weighted (40d hold) ===")
    returns = []
    weighted_returns = []
    entry_dates = []
    regimes = []
    gap_sizes = []

    for ticker in UNIVERSE:
        beats = all_gaps.get(ticker, {}).get('beats', [])
        for date, gap_size in beats:
            r = forward_return(data, ticker, date)
            if r is not None:
                returns.append(r)
                # Weight by gap size (normalize so average weight = 1)
                gap_sizes.append(gap_size)
                entry_dates.append(date)
                regimes.append(spy_regime(data, date))

    if gap_sizes:
        avg_gap = np.mean(gap_sizes)
        weights = [g / avg_gap for g in gap_sizes]
        weighted_returns = [r * w for r, w in zip(returns, weights)]

    metrics_unweighted = compute_metrics(returns)
    metrics_weighted = compute_metrics(weighted_returns, "Variant C")

    bull_rets = [r for r, reg in zip(weighted_returns, regimes) if reg == 'bull']
    bear_rets = [r for r, reg in zip(weighted_returns, regimes) if reg == 'bear']
    bull_m = compute_metrics(bull_rets)
    bear_m = compute_metrics(bear_rets)

    if bull_m['n_trades'] > 0 and bear_m['n_trades'] > 0:
        regime_gap = abs(bull_m['sharpe_tstat'] - bear_m['sharpe_tstat']) / max(abs(bull_m['sharpe_tstat']), abs(bear_m['sharpe_tstat']), 0.001)
    else:
        regime_gap = 999

    print(f"  Trades: {metrics_weighted['n_trades']}, Weighted Mean: {metrics_weighted['mean_return_pct']:.3f}%, "
          f"Unweighted Mean: {metrics_unweighted['mean_return_pct']:.3f}%")
    print(f"  Weighted Sharpe-t: {metrics_weighted['sharpe_tstat']:.3f}")

    return {
        'name': 'C) Beat Magnitude Weighted (40d hold)',
        'metrics_weighted': metrics_weighted,
        'metrics_unweighted': metrics_unweighted,
        'bull_metrics': bull_m,
        'bear_metrics': bear_m,
        'regime_gap': round(regime_gap, 3),
    }


def run_variant_d(data, all_gaps):
    """Variant D: Beat + Relative Strength (above 200-SMA)."""
    print("\n=== VARIANT D: Beat + Above 200-SMA (40d hold) ===")
    returns = []
    entry_dates = []
    regimes = []

    for ticker in UNIVERSE:
        beats = all_gaps.get(ticker, {}).get('beats', [])
        for date, gap_size in beats:
            if get_sma200(data, ticker, date):
                r = forward_return(data, ticker, date)
                if r is not None:
                    returns.append(r)
                    entry_dates.append(date)
                    regimes.append(spy_regime(data, date))

    metrics = compute_metrics(returns, "Variant D")

    bull_rets = [r for r, reg in zip(returns, regimes) if reg == 'bull']
    bear_rets = [r for r, reg in zip(returns, regimes) if reg == 'bear']
    bull_m = compute_metrics(bull_rets)
    bear_m = compute_metrics(bear_rets)

    if bull_m['n_trades'] > 0 and bear_m['n_trades'] > 0:
        regime_gap = abs(bull_m['sharpe_tstat'] - bear_m['sharpe_tstat']) / max(abs(bull_m['sharpe_tstat']), abs(bear_m['sharpe_tstat']), 0.001)
    else:
        regime_gap = 999

    if len(returns) >= 5:
        print(f"  Running {N_PERMS} permutations...")
        p_val, perm_mean, _ = permutation_test(returns, entry_dates, data)
    else:
        p_val, perm_mean = 1.0, 0.0

    print(f"  Trades: {metrics['n_trades']}, Mean: {metrics['mean_return_pct']:.3f}%, "
          f"Sharpe-t: {metrics['sharpe_tstat']:.3f}")
    print(f"  Perm p-value: {p_val:.4f}")

    return {
        'name': 'D) Beat + Above 200-SMA (40d hold)',
        'metrics': metrics,
        'bull_metrics': bull_m,
        'bear_metrics': bear_m,
        'regime_gap': round(regime_gap, 3),
        'perm_p_value': round(p_val, 4),
        'perm_mean_return_pct': round(perm_mean * 100, 3),
        'actual_vs_random_spread_pct': round((metrics['mean_return'] - perm_mean) * 100, 3),
    }


def run_variant_e(data, all_gaps):
    """Variant E: Beat vs Miss — THE KEY ADVERSARIAL CHECK."""
    print("\n=== VARIANT E: Beat vs Miss (ADVERSARIAL CHECK) ===")

    beat_returns = []
    miss_returns = []
    beat_dates = []
    miss_dates = []

    for ticker in UNIVERSE:
        gaps = all_gaps.get(ticker, {})
        for date, gap_size in gaps.get('beats', []):
            r = forward_return(data, ticker, date)
            if r is not None:
                beat_returns.append(r)
                beat_dates.append(date)
        for date, gap_size in gaps.get('misses', []):
            r = forward_return(data, ticker, date)
            if r is not None:
                miss_returns.append(r)
                miss_dates.append(date)

    beat_metrics = compute_metrics(beat_returns, "Beats")
    miss_metrics = compute_metrics(miss_returns, "Misses")

    # Statistical test: are beat returns significantly > miss returns?
    if len(beat_returns) > 5 and len(miss_returns) > 5:
        t_stat, p_val_ttest = stats.ttest_ind(beat_returns, miss_returns, alternative='greater')
    else:
        t_stat, p_val_ttest = 0, 1.0

    beat_mean = np.mean(beat_returns) if beat_returns else 0
    miss_mean = np.mean(miss_returns) if miss_returns else 0

    print(f"  BEATS:  n={beat_metrics['n_trades']}, mean={beat_metrics['mean_return_pct']:.3f}%, "
          f"Sharpe-t={beat_metrics['sharpe_tstat']:.3f}, WR={beat_metrics['win_rate']:.1%}")
    print(f"  MISSES: n={miss_metrics['n_trades']}, mean={miss_metrics['mean_return_pct']:.3f}%, "
          f"Sharpe-t={miss_metrics['sharpe_tstat']:.3f}, WR={miss_metrics['win_rate']:.1%}")
    print(f"  Beat-Miss spread: {(beat_mean - miss_mean)*100:.3f}%")
    print(f"  T-test (beats > misses): t={t_stat:.3f}, p={p_val_ttest:.4f}")

    if miss_metrics['mean_return_pct'] > 0 and abs(beat_mean - miss_mean) / max(abs(beat_mean), abs(miss_mean), 1e-9) < 0.3:
        diagnosis = "SIGNAL IS FAKE — misses perform similarly to beats. Stock selection adds no value; it's just timing/beta."
    elif beat_mean > miss_mean and p_val_ttest < 0.05:
        diagnosis = "SIGNAL MAY BE REAL — beats significantly outperform misses. Stock selection adds value."
    elif beat_mean > miss_mean:
        diagnosis = "INCONCLUSIVE — beats outperform misses but not significantly. Weak evidence for stock selection."
    else:
        diagnosis = "SIGNAL IS FAKE — misses outperform beats. Earnings beat adds negative stock-selection value."

    print(f"  DIAGNOSIS: {diagnosis}")

    return {
        'name': 'E) Beat vs Miss (ADVERSARIAL CHECK)',
        'beat_metrics': beat_metrics,
        'miss_metrics': miss_metrics,
        'beat_minus_miss_spread_pct': round((beat_mean - miss_mean) * 100, 3),
        'ttest_t_stat': round(float(t_stat), 3),
        'ttest_p_value': round(float(p_val_ttest), 4),
        'diagnosis': diagnosis,
    }


def run_variant_f(data, all_gaps):
    """Variant F: Multi-stock portfolio (max 5 concurrent) vs random portfolio."""
    print("\n=== VARIANT F: Multi-Stock Portfolio (max 5) ===")

    # Collect all beat events with dates, sorted by date
    events = []
    for ticker in UNIVERSE:
        beats = all_gaps.get(ticker, {}).get('beats', [])
        for date, gap_size in beats:
            events.append((date, ticker, gap_size))
    events.sort(key=lambda x: x[0])

    if not events:
        print("  No events found.")
        return {'name': 'F) Multi-Stock Portfolio', 'metrics': compute_metrics([])}

    # Simulate portfolio: hold up to 5 positions, each for 40 days
    # Track daily portfolio returns
    all_dates = data['SPY'].index
    oot_dates = all_dates[(all_dates >= pd.Timestamp(OOT_START)) & (all_dates <= pd.Timestamp(OOT_END))]

    positions = []  # list of (ticker, entry_date, entry_price, exit_idx)
    daily_returns = []
    portfolio_trades = []

    for i, date in enumerate(oot_dates):
        # Check for new events on this date
        for ev_date, ticker, gap_size in events:
            if ev_date == date and len(positions) < MAX_CONCURRENT:
                if ticker in data:
                    df = data[ticker]
                    idx = df.index.get_indexer([date], method='nearest')[0]
                    if idx + HOLD_DAYS < len(df):
                        entry_p = float(df['Close'].iloc[idx]) * (1 + SLIPPAGE_PCT)
                        exit_idx = idx + HOLD_DAYS
                        positions.append((ticker, date, entry_p, exit_idx, idx))

        # Calculate daily portfolio return
        day_ret = 0
        n_pos = len(positions)
        if n_pos > 0:
            weight = 1.0 / MAX_CONCURRENT  # Fixed weight per slot
            for ticker, entry_date, entry_p, exit_idx, start_idx in positions:
                df = data[ticker]
                curr_idx = df.index.get_indexer([date], method='nearest')[0]
                prev_idx = max(start_idx, curr_idx - 1)
                if curr_idx > start_idx and curr_idx < len(df):
                    prev_p = float(df['Close'].iloc[prev_idx])
                    curr_p = float(df['Close'].iloc[curr_idx])
                    day_ret += weight * (curr_p - prev_p) / prev_p
            daily_returns.append(day_ret)
        else:
            daily_returns.append(0)

        # Remove expired positions
        new_positions = []
        for pos in positions:
            ticker, entry_date, entry_p, exit_idx, start_idx = pos
            df = data[ticker]
            curr_idx = df.index.get_indexer([date], method='nearest')[0]
            if curr_idx < exit_idx:
                new_positions.append(pos)
            else:
                exit_p = float(df['Close'].iloc[exit_idx]) * (1 - SLIPPAGE_PCT)
                trade_ret = (exit_p - entry_p) / entry_p
                portfolio_trades.append(trade_ret)
        positions = new_positions

    daily_returns = np.array(daily_returns)

    # Portfolio metrics
    total_ret = np.sum(daily_returns)
    if len(daily_returns) > 1:
        ann_sharpe = np.mean(daily_returns) / (np.std(daily_returns, ddof=1) + 1e-9) * np.sqrt(252)
        down = daily_returns[daily_returns < 0]
        ann_sortino = np.mean(daily_returns) / (np.std(down, ddof=1) + 1e-9) * np.sqrt(252) if len(down) > 1 else 0
    else:
        ann_sharpe = 0
        ann_sortino = 0

    cum = np.cumsum(daily_returns)
    running_max = np.maximum.accumulate(cum)
    maxdd = float(np.min(cum - running_max))

    trade_metrics = compute_metrics(portfolio_trades)

    # Regime breakdown
    spy_closes = data['SPY']['Close']
    spy_sma200 = spy_closes.rolling(200).mean()
    bull_daily = []
    bear_daily = []
    for i, date in enumerate(oot_dates):
        if date in spy_sma200.index:
            idx = spy_sma200.index.get_indexer([date], method='ffill')[0]
            if idx >= 0 and not pd.isna(spy_sma200.iloc[idx]):
                if float(spy_closes.iloc[idx]) > float(spy_sma200.iloc[idx]):
                    bull_daily.append(daily_returns[i])
                else:
                    bear_daily.append(daily_returns[i])

    bull_sharpe = np.mean(bull_daily) / (np.std(bull_daily, ddof=1) + 1e-9) * np.sqrt(252) if len(bull_daily) > 10 else 0
    bear_sharpe = np.mean(bear_daily) / (np.std(bear_daily, ddof=1) + 1e-9) * np.sqrt(252) if len(bear_daily) > 10 else 0
    denom = max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
    regime_gap = abs(bull_sharpe - bear_sharpe) / denom

    print(f"  Portfolio trades: {trade_metrics['n_trades']}")
    print(f"  Annualized Sharpe: {ann_sharpe:.3f}, Sortino: {ann_sortino:.3f}")
    print(f"  Total return: {total_ret*100:.2f}%, MaxDD: {maxdd*100:.2f}%")
    print(f"  Trade WR: {trade_metrics['win_rate']:.1%}, PF: {trade_metrics['profit_factor']:.3f}")
    print(f"  Regime: bull Sharpe={bull_sharpe:.3f}, bear Sharpe={bear_sharpe:.3f}, gap={regime_gap:.3f}")

    return {
        'name': 'F) Multi-Stock Portfolio (max 5)',
        'trade_metrics': trade_metrics,
        'ann_sharpe': round(float(ann_sharpe), 3),
        'ann_sortino': round(float(ann_sortino), 3),
        'total_return_pct': round(float(total_ret * 100), 2),
        'max_dd_pct': round(float(maxdd * 100), 2),
        'bull_sharpe': round(float(bull_sharpe), 3),
        'bear_sharpe': round(float(bear_sharpe), 3),
        'regime_gap': round(float(regime_gap), 3),
    }


def five_gate_validation(result):
    """Apply 5-gate validation to a variant result."""
    gates = {}

    # Get the primary metrics dict
    m = result.get('metrics', result.get('metrics_weighted', result.get('trade_metrics', {})))

    # Gate 1: Sharpe > 0.5 (using t-stat as proxy; for portfolio variants use ann_sharpe)
    sharpe_val = result.get('ann_sharpe', m.get('sharpe_tstat', 0))
    gates['sharpe_gt_0.5'] = sharpe_val > 0.5

    # Gate 2: Perm p < 0.05
    p_val = result.get('perm_p_value', result.get('ttest_p_value', 1.0))
    gates['perm_p_lt_0.05'] = p_val < 0.05

    # Gate 3: Regime gap < 0.5
    rg = result.get('regime_gap', 999)
    gates['regime_gap_lt_0.5'] = rg < 0.5

    # Gate 4: MaxDD > -50%
    maxdd = m.get('max_dd_pct', result.get('max_dd_pct', 0))
    gates['maxdd_gt_neg50'] = maxdd > -50

    # Gate 5: >= 20 trades
    n = m.get('n_trades', 0)
    gates['trades_gte_20'] = n >= 20

    gates['passed'] = all(gates.values())
    gates['gates_passed'] = sum(1 for v in gates.values() if v and isinstance(v, bool)) - (1 if gates['passed'] else 0)

    return gates


def main():
    print("=" * 70)
    print("EARNINGS CHAIN MOMENTUM — STOCK SELECTION BACKTEST")
    print("=" * 70)
    print(f"Universe: {len(UNIVERSE)} stocks")
    print(f"OOT period: {OOT_START} to {OOT_END}")
    print(f"Gap threshold: {GAP_THRESHOLD:.0%}")
    print(f"Hold period: {HOLD_DAYS} trading days")
    print(f"Account: ${ACCOUNT_SIZE}")
    print(f"Permutations: {N_PERMS}")
    print()

    # Download data
    data = download_data()

    if len(data) < 10:
        print("ERROR: Too few tickers downloaded. Aborting.")
        return

    # Detect gaps
    all_gaps = detect_gaps(data)

    total_beats = sum(len(g.get('beats', [])) for g in all_gaps.values())
    total_misses = sum(len(g.get('misses', [])) for g in all_gaps.values())
    print(f"\nDetected {total_beats} beat events (>{GAP_THRESHOLD:.0%} gap)")
    print(f"Detected {total_misses} miss events (<-{GAP_THRESHOLD:.0%} gap)")

    # Run all variants
    results = {}

    results['A'] = run_variant_a(data, all_gaps)
    results['B'] = run_variant_b(data, all_gaps)
    results['C'] = run_variant_c(data, all_gaps)
    results['D'] = run_variant_d(data, all_gaps)
    results['E'] = run_variant_e(data, all_gaps)
    results['F'] = run_variant_f(data, all_gaps)

    # Apply 5-gate validation
    print("\n" + "=" * 70)
    print("5-GATE VALIDATION SUMMARY")
    print("=" * 70)

    for key in ['A', 'B', 'C', 'D', 'F']:
        gates = five_gate_validation(results[key])
        results[key]['five_gate'] = gates
        status = "PASS" if gates['passed'] else "FAIL"
        print(f"  {results[key]['name']}: {status} ({gates['gates_passed']}/5 gates)")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)

    # The key question: does Variant E show beat selection adds value?
    e = results['E']
    print(f"\n  KEY ADVERSARIAL CHECK (Variant E):")
    print(f"  {e['diagnosis']}")
    print(f"  Beat mean: {e['beat_metrics']['mean_return_pct']:.3f}%")
    print(f"  Miss mean: {e['miss_metrics']['mean_return_pct']:.3f}%")
    print(f"  Spread: {e['beat_minus_miss_spread_pct']:.3f}%")

    # Check perm tests
    a_perm = results['A'].get('perm_p_value', 1.0)
    print(f"\n  Variant A perm test (beater vs random stock): p={a_perm:.4f}")
    if a_perm > 0.05:
        print(f"  → Buying the beater is NOT significantly better than buying a random stock.")
        print(f"  → EARNINGS BEAT STOCK SELECTION HAS NO EDGE.")
    else:
        print(f"  → Buying the beater IS significantly better than random. Edge may exist.")

    any_passed = any(results[k].get('five_gate', {}).get('passed', False) for k in ['A', 'B', 'C', 'D', 'F'])
    if any_passed:
        print(f"\n  VERDICT: At least one variant passed all 5 gates. Worth further investigation.")
    else:
        print(f"\n  VERDICT: NO variant passed all 5 gates. Strategy is NOT viable.")

    # Save results
    # Convert timestamps to strings for JSON
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        'strategy': 'Earnings Chain Momentum — Stock Selection',
        'run_date': datetime.now().isoformat(),
        'config': {
            'universe_size': len(UNIVERSE),
            'oot_period': f"{OOT_START} to {OOT_END}",
            'gap_threshold': GAP_THRESHOLD,
            'hold_days': HOLD_DAYS,
            'n_permutations': N_PERMS,
            'account_size': ACCOUNT_SIZE,
            'slippage_pct': SLIPPAGE_PCT,
        },
        'total_beats': total_beats,
        'total_misses': total_misses,
        'variants': make_serializable(results),
        'verdict': 'PASS' if any_passed else 'FAIL — no variant passed all 5 gates',
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
