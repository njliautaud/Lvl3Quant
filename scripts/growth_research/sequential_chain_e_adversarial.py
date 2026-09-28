#!/usr/bin/env python3
"""
Adversarial Backtest: Chain E — RSI Divergence Setup -> Bond Yield Drop Trigger
================================================================================
Baseline (5-gate pass): Sharpe 1.528, Sortino 2.257, WR 65.9%, PF 2.23,
                         135 trades, regime gap 0.413, perm p=0.020

6 Adversarial Tests:
  1. Re-implementation (reproduce Sharpe within +/-30%)
  2. Inverse Signal (bearish RSI convergence + yield rise)
  3. Random Timing Permutation (1000 perms, p < 0.05)
  4. Sub-Period Stability (4 equal periods, all positive Sharpe)
  5. Top-3 Ticker Removal (Sharpe drop < 50%)
  6. Parameter Sensitivity (108 combos, >=80% with Sharpe > 0.30)
"""

import os, sys, json, warnings, time, functools, pickle, itertools
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

# ─── Config ───
START_DATE = '2019-01-01'
END_DATE = '2026-07-01'
BACKTEST_START = '2020-01-01'
POS_SIZE = 300.0
MAX_CONCURRENT = 2
HOLD_DAYS = 21
PROFIT_TARGET = 0.10
STOP_LOSS = -0.15
SPREAD_COST_PCT = 0.001

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/sequential_signal_chains')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
    'MA', 'WMT', 'PEP', 'KO', 'TMO', 'ABT', 'BAC', 'XOM', 'CVX', 'CSCO',
]

MACRO_TICKERS = ['SPY', '^VIX', 'TLT', '^TNX']

# Baseline metrics from original run
BASELINE = {
    'sharpe': 1.528,
    'sortino': 2.257,
    'wr': 0.659,
    'pf': 2.227,
    'n_trades': 135,
    'regime_gap': 0.413,
    'perm_p': 0.020,
}

print("=" * 80)
print("ADVERSARIAL BACKTEST: Chain E")
print("RSI Divergence Setup -> Bond Yield Drop Trigger")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD (cached)
# ═══════════════════════════════════════════════════════════════════════
def download_data():
    import yfinance as yf
    cache_file = OUTPUT_DIR / '_adversarial_e_cache.pkl'
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded cache: {len(data['close'].columns)} tickers, {len(data['close'])} days")
        return data

    print(f"\n[DATA] Downloading {len(UNIVERSE)} stocks + {len(MACRO_TICKERS)} macro...")
    all_tickers = UNIVERSE + MACRO_TICKERS
    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False, threads=True)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
        high = raw['High']
        low = raw['Low']
        volume = raw['Volume']
    else:
        close = high = low = volume = raw

    for df in [close, high, low, volume]:
        if hasattr(df.columns, 'droplevel'):
            try:
                df.columns = df.columns.droplevel(1)
            except Exception:
                pass

    close = close.ffill().dropna(how='all')
    data = {
        'close': close,
        'high': high.reindex(close.index).ffill(),
        'low': low.reindex(close.index).ffill(),
        'volume': volume.reindex(close.index).ffill().fillna(0),
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Downloaded: {len(close.columns)} tickers, {len(close)} days")
    return data

# ═══════════════════════════════════════════════════════════════════════
# INDICATORS (re-implemented from scratch for Test 1)
# ═══════════════════════════════════════════════════════════════════════
def compute_rsi(series, period=14):
    """Wilder-style RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

# ═══════════════════════════════════════════════════════════════════════
# SIGNAL GENERATORS
# ═══════════════════════════════════════════════════════════════════════
def generate_rsi_divergence_setups(data, stock_tickers, lookback=14):
    """Bullish RSI divergence: price makes lower low but RSI makes higher low."""
    setups = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        rsi = compute_rsi(close, lookback)
        for i in range(lookback, len(close)):
            if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i - lookback]):
                continue
            price_lower_low = close.iloc[i] < close.iloc[i - lookback]
            rsi_higher_low = rsi.iloc[i] > rsi.iloc[i - lookback]
            if price_lower_low and rsi_higher_low:
                setups[(close.index[i], t)] = True
    return setups

def generate_rsi_convergence_setups(data, stock_tickers, lookback=14):
    """Bearish RSI convergence (INVERSE): price makes higher high but RSI makes lower high."""
    setups = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        rsi = compute_rsi(close, lookback)
        for i in range(lookback, len(close)):
            if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i - lookback]):
                continue
            price_higher_high = close.iloc[i] > close.iloc[i - lookback]
            rsi_lower_high = rsi.iloc[i] < rsi.iloc[i - lookback]
            if price_higher_high and rsi_lower_high:
                setups[(close.index[i], t)] = True
    return setups

def generate_yield_drop_trigger(data, threshold=0.05):
    """Macro trigger: 10Y yield drops > threshold over 5 days."""
    if '^TNX' not in data['close'].columns:
        tlt = data['close']['TLT']
        # Approximate: TLT up > threshold*20 pct as proxy
        change = tlt.pct_change(5)
        fire_days = change[change > threshold * 0.2].index
    else:
        tnx = data['close']['^TNX']
        yield_change = tnx.diff(5)
        fire_days = yield_change[yield_change < -threshold].index
    return set(fire_days)

def generate_yield_rise_trigger(data, threshold=0.05):
    """INVERSE trigger: 10Y yield RISES > threshold over 5 days."""
    if '^TNX' not in data['close'].columns:
        tlt = data['close']['TLT']
        change = tlt.pct_change(5)
        fire_days = change[change < -threshold * 0.2].index
    else:
        tnx = data['close']['^TNX']
        yield_change = tnx.diff(5)
        fire_days = yield_change[yield_change > threshold].index
    return set(fire_days)

# ═══════════════════════════════════════════════════════════════════════
# CHAIN MATCHER
# ═══════════════════════════════════════════════════════════════════════
def match_chain(stock_setups, macro_trigger_dates, stock_tickers, max_delay=5):
    """Match stock-specific setup + macro trigger within max_delay days."""
    setup_by_ticker = defaultdict(list)
    for (d, t) in stock_setups:
        setup_by_ticker[t].append(d)
    for t in setup_by_ticker:
        setup_by_ticker[t].sort()

    sorted_triggers = sorted(macro_trigger_dates)
    entries = {}

    for t in stock_tickers:
        if t not in setup_by_ticker:
            continue
        for sd in setup_by_ticker[t]:
            for td in sorted_triggers:
                delta = (td - sd).days
                if delta < 0:
                    continue
                if delta > max_delay:
                    break
                entries[(td, t)] = True
    return entries

# ═══════════════════════════════════════════════════════════════════════
# BACKTESTER
# ═══════════════════════════════════════════════════════════════════════
def run_backtest(signal_entries, data, stock_tickers, hold_days=HOLD_DAYS,
                 profit_target=PROFIT_TARGET, stop_loss=STOP_LOSS,
                 max_concurrent=MAX_CONCURRENT, pos_size=POS_SIZE):
    """Run backtest on signal entries {(date, ticker): True}."""
    close = data['close']
    spy_close = close['SPY']
    bt_start = pd.Timestamp(BACKTEST_START)
    entries = sorted([(d, t) for (d, t) in signal_entries if d >= bt_start], key=lambda x: x[0])

    if not entries:
        return None

    trades = []
    open_positions = []

    for entry_date, ticker in entries:
        open_positions = [p for p in open_positions if p[3] > entry_date]
        if len(open_positions) >= max_concurrent:
            continue

        try:
            entry_price = close.at[entry_date, ticker]
        except Exception:
            continue
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        future_dates = close.index[close.index > entry_date]
        if len(future_dates) == 0:
            continue

        exit_price = None
        exit_date = None
        exit_reason = 'hold_expiry'

        for fdate in future_dates[:hold_days]:
            try:
                price = close.at[fdate, ticker]
            except Exception:
                continue
            if pd.isna(price):
                continue
            ret = (price - entry_price) / entry_price
            if ret >= profit_target:
                exit_price = price
                exit_date = fdate
                exit_reason = 'profit_target'
                break
            elif ret <= stop_loss:
                exit_price = price
                exit_date = fdate
                exit_reason = 'stop_loss'
                break

        if exit_price is None:
            hold_end = min(hold_days, len(future_dates))
            if hold_end > 0:
                exit_date = future_dates[hold_end - 1]
                try:
                    exit_price = close.at[exit_date, ticker]
                except Exception:
                    continue
                if pd.isna(exit_price):
                    continue

        if exit_price is None:
            continue

        raw_ret = (exit_price - entry_price) / entry_price
        net_ret = raw_ret - SPREAD_COST_PCT
        pnl = pos_size * net_ret

        try:
            spy_idx = spy_close.index.get_loc(entry_date)
            if spy_idx > 0:
                regime = 'bull' if spy_close.iloc[spy_idx] > spy_close.iloc[spy_idx - 1] else 'bear'
            else:
                regime = 'unknown'
        except Exception:
            regime = 'unknown'

        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'ticker': ticker,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return': net_ret,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'regime': regime,
            'hold_days': (exit_date - entry_date).days,
        })
        open_positions.append((entry_date, ticker, entry_price, exit_date))

    if not trades:
        return None
    return pd.DataFrame(trades)

# ═══════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════
def compute_metrics(trades_df):
    """Compute Sharpe, Sortino, WR, PF from trades DataFrame."""
    if trades_df is None or len(trades_df) < 3:
        return {'sharpe': 0, 'sortino': 0, 'wr': 0, 'pf': 0, 'n_trades': 0, 'mean_ret': 0}

    rets = trades_df['return'].values
    n = len(rets)
    years = max(1, (trades_df['entry_date'].max() - trades_df['entry_date'].min()).days / 365.25)
    tpy = max(1, n / years)

    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1.0
    sharpe = (mean_ret / std_ret) * np.sqrt(tpy) if std_ret > 0 else 0

    downside = rets[rets < 0]
    if len(downside) >= 2:
        down_std = np.std(downside, ddof=1)
        sortino = (mean_ret / down_std) * np.sqrt(tpy) if down_std > 0 else 99.0
    else:
        sortino = 99.0

    wr = np.mean(rets > 0)
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = np.abs(np.sum(rets[rets < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'wr': round(wr, 3),
        'pf': round(pf, 3),
        'n_trades': n,
        'mean_ret': round(mean_ret * 100, 2),
    }

# ═══════════════════════════════════════════════════════════════════════
# TEST 1: RE-IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════
def test_reimplementation(data, stock_tickers):
    """Re-implement Chain E from scratch. Must reproduce Sharpe within +/-30%."""
    print("\n" + "─" * 70)
    print("TEST 1: RE-IMPLEMENTATION")
    print("─" * 70)

    setups = generate_rsi_divergence_setups(data, stock_tickers, lookback=14)
    print(f"  RSI divergence setups: {len(setups)}")

    triggers = generate_yield_drop_trigger(data, threshold=0.05)
    print(f"  Bond yield drop triggers: {len(triggers)} days")

    entries = match_chain(setups, triggers, stock_tickers, max_delay=5)
    print(f"  Matched entries: {len(entries)}")

    trades_df = run_backtest(entries, data, stock_tickers)
    metrics = compute_metrics(trades_df)

    sharpe_ratio = metrics['sharpe'] / BASELINE['sharpe'] if BASELINE['sharpe'] != 0 else 0
    within_30 = 0.70 <= sharpe_ratio <= 1.30
    passed = within_30

    print(f"  Re-impl Sharpe: {metrics['sharpe']:.3f} vs baseline {BASELINE['sharpe']:.3f} "
          f"(ratio: {sharpe_ratio:.2f})")
    print(f"  N trades: {metrics['n_trades']} vs baseline {BASELINE['n_trades']}")
    print(f"  WR: {metrics['wr']:.1%}, PF: {metrics['pf']:.2f}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'reimplementation',
        'passed': passed,
        'reimpl_sharpe': metrics['sharpe'],
        'baseline_sharpe': BASELINE['sharpe'],
        'ratio': round(sharpe_ratio, 3),
        'n_trades': metrics['n_trades'],
        'metrics': metrics,
    }, trades_df, entries

# ═══════════════════════════════════════════════════════════════════════
# TEST 2: INVERSE SIGNAL
# ═══════════════════════════════════════════════════════════════════════
def test_inverse_signal(data, stock_tickers, forward_sharpe):
    """Bearish RSI convergence + yield RISE. If inverse ratio > 0.50, edge is beta."""
    print("\n" + "─" * 70)
    print("TEST 2: INVERSE SIGNAL")
    print("─" * 70)

    setups = generate_rsi_convergence_setups(data, stock_tickers, lookback=14)
    print(f"  RSI convergence (bearish) setups: {len(setups)}")

    triggers = generate_yield_rise_trigger(data, threshold=0.05)
    print(f"  Bond yield RISE triggers: {len(triggers)} days")

    entries = match_chain(setups, triggers, stock_tickers, max_delay=5)
    print(f"  Inverse matched entries: {len(entries)}")

    trades_df = run_backtest(entries, data, stock_tickers)
    inv_metrics = compute_metrics(trades_df)

    # If inverse also makes money, ratio = inverse_sharpe / forward_sharpe
    inv_sharpe = max(inv_metrics['sharpe'], 0)
    ratio = inv_sharpe / forward_sharpe if forward_sharpe > 0 else 99
    passed = ratio < 0.50  # Inverse should NOT work well

    print(f"  Inverse Sharpe: {inv_metrics['sharpe']:.3f}")
    print(f"  Forward Sharpe: {forward_sharpe:.3f}")
    print(f"  Ratio (inv/fwd): {ratio:.3f} (must be < 0.50)")
    print(f"  Inverse N trades: {inv_metrics['n_trades']}, WR: {inv_metrics['wr']:.1%}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'inverse_signal',
        'passed': passed,
        'inverse_sharpe': inv_metrics['sharpe'],
        'forward_sharpe': round(forward_sharpe, 3),
        'ratio': round(ratio, 3),
        'inverse_metrics': inv_metrics,
    }

# ═══════════════════════════════════════════════════════════════════════
# TEST 3: RANDOM TIMING PERMUTATION
# ═══════════════════════════════════════════════════════════════════════
def test_random_timing(data, stock_tickers, real_entries, real_sharpe, n_perms=1000):
    """Generate random entries with same frequency. 1000 perms, p < 0.05."""
    print("\n" + "─" * 70)
    print("TEST 3: RANDOM TIMING PERMUTATION (1000 perms)")
    print("─" * 70)

    bt_start = pd.Timestamp(BACKTEST_START)
    valid_dates = data['close'].index[data['close'].index >= bt_start]
    valid_dates = valid_dates[:-HOLD_DAYS - 5] if len(valid_dates) > HOLD_DAYS + 5 else valid_dates

    n_signals = len(real_entries)
    beats = 0
    perm_sharpes = []

    t0 = time.time()
    for i in range(n_perms):
        rand_dates = np.random.choice(valid_dates, size=n_signals, replace=True)
        rand_tickers = np.random.choice(stock_tickers, size=n_signals, replace=True)
        rand_signals = {(d, t): True for d, t in zip(rand_dates, rand_tickers)}

        rand_trades = run_backtest(rand_signals, data, stock_tickers)
        if rand_trades is not None and len(rand_trades) >= 5:
            r_m = compute_metrics(rand_trades)
            r_sharpe = r_m['sharpe']
        else:
            r_sharpe = 0.0
        perm_sharpes.append(r_sharpe)
        if r_sharpe >= real_sharpe:
            beats += 1

        if (i + 1) % 200 == 0:
            print(f"    Perm {i+1}/{n_perms}... elapsed {time.time()-t0:.0f}s")

    perm_p = beats / n_perms
    passed = perm_p < 0.05

    perm_arr = np.array(perm_sharpes)
    print(f"  Real Sharpe: {real_sharpe:.3f}")
    print(f"  Random Sharpe: mean={np.mean(perm_arr):.3f}, "
          f"median={np.median(perm_arr):.3f}, p95={np.percentile(perm_arr, 95):.3f}")
    print(f"  p-value: {perm_p:.4f} ({beats}/{n_perms} random >= real)")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'random_timing_permutation',
        'passed': passed,
        'perm_p': round(perm_p, 4),
        'n_perms': n_perms,
        'real_sharpe': round(real_sharpe, 3),
        'random_sharpe_mean': round(float(np.mean(perm_arr)), 3),
        'random_sharpe_p95': round(float(np.percentile(perm_arr, 95)), 3),
    }

# ═══════════════════════════════════════════════════════════════════════
# TEST 4: SUB-PERIOD STABILITY
# ═══════════════════════════════════════════════════════════════════════
def test_subperiod_stability(trades_df):
    """Split into 4 equal sub-periods. All must have positive Sharpe."""
    print("\n" + "─" * 70)
    print("TEST 4: SUB-PERIOD STABILITY")
    print("─" * 70)

    if trades_df is None or len(trades_df) < 12:
        print("  Insufficient trades for sub-period analysis")
        return {'test': 'subperiod_stability', 'passed': False, 'reason': 'insufficient trades'}

    trades_sorted = trades_df.sort_values('entry_date').reset_index(drop=True)
    min_date = trades_sorted['entry_date'].min()
    max_date = trades_sorted['entry_date'].max()
    total_days = (max_date - min_date).days
    period_days = total_days / 4

    sub_results = []
    all_positive = True

    for p in range(4):
        p_start = min_date + pd.Timedelta(days=int(p * period_days))
        p_end = min_date + pd.Timedelta(days=int((p + 1) * period_days))
        sub = trades_sorted[
            (trades_sorted['entry_date'] >= p_start) & (trades_sorted['entry_date'] < p_end)
        ]

        if len(sub) < 3:
            sub_sharpe = 0.0
            n_sub = len(sub)
        else:
            m = compute_metrics(sub)
            sub_sharpe = m['sharpe']
            n_sub = m['n_trades']

        is_positive = sub_sharpe > 0
        if not is_positive:
            all_positive = False

        sub_results.append({
            'period': f"P{p+1} ({p_start.strftime('%Y-%m')} to {p_end.strftime('%Y-%m')})",
            'n_trades': n_sub,
            'sharpe': round(sub_sharpe, 3),
            'positive': is_positive,
        })

        status = "+" if is_positive else "NEGATIVE"
        print(f"  P{p+1} ({p_start.strftime('%Y-%m')} - {p_end.strftime('%Y-%m')}): "
              f"n={n_sub:3d}, Sharpe={sub_sharpe:+.3f} [{status}]")

    passed = all_positive
    print(f"  RESULT: {'PASS' if passed else 'FAIL'} "
          f"({sum(1 for s in sub_results if s['positive'])}/4 positive)")

    return {
        'test': 'subperiod_stability',
        'passed': passed,
        'sub_periods': sub_results,
        'n_positive': sum(1 for s in sub_results if s['positive']),
    }

# ═══════════════════════════════════════════════════════════════════════
# TEST 5: TOP-3 TICKER REMOVAL
# ═══════════════════════════════════════════════════════════════════════
def test_top3_removal(trades_df, data, stock_tickers, full_sharpe):
    """Remove 3 highest-PnL tickers. If Sharpe drops > 50%, edge is concentrated."""
    print("\n" + "─" * 70)
    print("TEST 5: TOP-3 TICKER REMOVAL")
    print("─" * 70)

    if trades_df is None or len(trades_df) < 10:
        print("  Insufficient trades")
        return {'test': 'top3_ticker_removal', 'passed': False, 'reason': 'insufficient trades'}

    ticker_pnl = trades_df.groupby('ticker')['pnl'].sum().sort_values(ascending=False)
    top3 = ticker_pnl.head(3).index.tolist()
    print(f"  Top 3 by PnL: {top3}")
    for t in top3:
        print(f"    {t}: ${ticker_pnl[t]:.2f} total PnL, "
              f"{len(trades_df[trades_df['ticker'] == t])} trades")

    reduced = trades_df[~trades_df['ticker'].isin(top3)]
    if len(reduced) < 5:
        print("  Too few trades remaining after removal")
        return {'test': 'top3_ticker_removal', 'passed': False, 'reason': 'too few remaining trades'}

    reduced_metrics = compute_metrics(reduced)
    reduced_sharpe = reduced_metrics['sharpe']

    drop_pct = 1.0 - (reduced_sharpe / full_sharpe) if full_sharpe > 0 else 1.0
    passed = drop_pct < 0.50

    print(f"  Full Sharpe: {full_sharpe:.3f}")
    print(f"  Reduced Sharpe: {reduced_sharpe:.3f} ({len(reduced)} trades)")
    print(f"  Drop: {drop_pct:.1%} (must be < 50%)")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'top3_ticker_removal',
        'passed': passed,
        'full_sharpe': round(full_sharpe, 3),
        'reduced_sharpe': reduced_sharpe,
        'drop_pct': round(drop_pct, 3),
        'removed_tickers': top3,
        'remaining_trades': len(reduced),
    }

# ═══════════════════════════════════════════════════════════════════════
# TEST 6: PARAMETER SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════
def test_parameter_sensitivity(data, stock_tickers):
    """Grid search: 3x3x3x4 = 108 combos. >= 80% must have Sharpe > 0.30."""
    print("\n" + "─" * 70)
    print("TEST 6: PARAMETER SENSITIVITY (108 combos)")
    print("─" * 70)

    rsi_lookbacks = [10, 14, 20]
    trigger_windows = [3, 5, 7]
    yield_thresholds = [0.03, 0.05, 0.10]
    hold_days_list = [10, 14, 21, 30]

    total = len(rsi_lookbacks) * len(trigger_windows) * len(yield_thresholds) * len(hold_days_list)
    print(f"  Total combinations: {total}")

    # Pre-compute setups for each RSI lookback
    print("  Pre-computing RSI divergence setups...")
    setup_cache = {}
    for lb in rsi_lookbacks:
        t0 = time.time()
        setup_cache[lb] = generate_rsi_divergence_setups(data, stock_tickers, lookback=lb)
        print(f"    lookback={lb}: {len(setup_cache[lb])} setups ({time.time()-t0:.1f}s)")

    # Pre-compute triggers for each yield threshold
    print("  Pre-computing yield triggers...")
    trigger_cache = {}
    for thresh in yield_thresholds:
        trigger_cache[thresh] = generate_yield_drop_trigger(data, threshold=thresh)
        print(f"    threshold={thresh}: {len(trigger_cache[thresh])} trigger days")

    results = []
    n_above_030 = 0
    count = 0
    t0 = time.time()

    for lb in rsi_lookbacks:
        for tw in trigger_windows:
            for yt in yield_thresholds:
                for hd in hold_days_list:
                    count += 1
                    setups = setup_cache[lb]
                    triggers = trigger_cache[yt]
                    entries = match_chain(setups, triggers, stock_tickers, max_delay=tw)

                    trades_df = run_backtest(entries, data, stock_tickers, hold_days=hd)
                    m = compute_metrics(trades_df)

                    above = m['sharpe'] > 0.30
                    if above:
                        n_above_030 += 1

                    results.append({
                        'rsi_lookback': lb,
                        'trigger_window': tw,
                        'yield_threshold': yt,
                        'hold_days': hd,
                        'sharpe': m['sharpe'],
                        'n_trades': m['n_trades'],
                        'wr': m['wr'],
                        'above_030': above,
                    })

                    if count % 27 == 0:
                        print(f"    {count}/{total} done... elapsed {time.time()-t0:.0f}s")

    pct_above = n_above_030 / total
    passed = pct_above >= 0.80

    sharpes = [r['sharpe'] for r in results]
    print(f"\n  Sharpe distribution across {total} combos:")
    print(f"    Min: {min(sharpes):.3f}, Median: {np.median(sharpes):.3f}, "
          f"Max: {max(sharpes):.3f}, Mean: {np.mean(sharpes):.3f}")
    print(f"  Above 0.30: {n_above_030}/{total} = {pct_above:.1%} (must be >= 80%)")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'parameter_sensitivity',
        'passed': passed,
        'n_combos': total,
        'n_above_030': n_above_030,
        'pct_above': round(pct_above, 3),
        'sharpe_min': round(min(sharpes), 3),
        'sharpe_median': round(float(np.median(sharpes)), 3),
        'sharpe_max': round(max(sharpes), 3),
        'sharpe_mean': round(float(np.mean(sharpes)), 3),
        'best_combo': max(results, key=lambda x: x['sharpe']),
        'worst_combo': min(results, key=lambda x: x['sharpe']),
    }

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    np.random.seed(42)
    data = download_data()
    stock_tickers = [t for t in UNIVERSE if t in data['close'].columns]
    print(f"  Universe: {len(stock_tickers)} tickers")

    all_results = []

    # ── Test 1: Re-implementation ──
    t0 = time.time()
    r1, trades_df, entries = test_reimplementation(data, stock_tickers)
    r1['elapsed_s'] = round(time.time() - t0, 1)
    all_results.append(r1)

    # Use re-implemented metrics as the forward reference
    reimpl_sharpe = r1['reimpl_sharpe']

    # ── Test 2: Inverse Signal ──
    t0 = time.time()
    r2 = test_inverse_signal(data, stock_tickers, reimpl_sharpe)
    r2['elapsed_s'] = round(time.time() - t0, 1)
    all_results.append(r2)

    # ── Test 3: Random Timing Permutation ──
    t0 = time.time()
    r3 = test_random_timing(data, stock_tickers, entries, reimpl_sharpe, n_perms=1000)
    r3['elapsed_s'] = round(time.time() - t0, 1)
    all_results.append(r3)

    # ── Test 4: Sub-Period Stability ──
    t0 = time.time()
    r4 = test_subperiod_stability(trades_df)
    r4['elapsed_s'] = round(time.time() - t0, 1)
    all_results.append(r4)

    # ── Test 5: Top-3 Ticker Removal ──
    t0 = time.time()
    r5 = test_top3_removal(trades_df, data, stock_tickers, reimpl_sharpe)
    r5['elapsed_s'] = round(time.time() - t0, 1)
    all_results.append(r5)

    # ── Test 6: Parameter Sensitivity ──
    t0 = time.time()
    r6 = test_parameter_sensitivity(data, stock_tickers)
    r6['elapsed_s'] = round(time.time() - t0, 1)
    all_results.append(r6)

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("ADVERSARIAL AUDIT SUMMARY: Chain E")
    print("RSI Divergence Setup -> Bond Yield Drop Trigger")
    print("=" * 80)

    n_passed = sum(1 for r in all_results if r['passed'])
    n_total = len(all_results)

    test_names = [
        "1. Re-implementation",
        "2. Inverse Signal",
        "3. Random Timing (1000 perms)",
        "4. Sub-Period Stability",
        "5. Top-3 Ticker Removal",
        "6. Parameter Sensitivity",
    ]

    for name, r in zip(test_names, all_results):
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  {name:35s} [{status}]")

    print(f"\n  OVERALL: {n_passed}/{n_total} PASS")
    if n_passed == n_total:
        print("  VERDICT: Chain E passes full adversarial audit. Edge appears ROBUST.")
    elif n_passed >= 4:
        print("  VERDICT: Chain E passes most tests. Edge is LIKELY REAL with caveats.")
    else:
        print("  VERDICT: Chain E FAILS adversarial audit. Edge may be spurious.")

    # Save results
    def sanitize(v):
        if isinstance(v, (pd.Timestamp,)):
            return str(v)
        if isinstance(v, (np.floating, np.float64, np.float32)):
            return float(v)
        if isinstance(v, (np.integer, np.int64, np.int32)):
            return int(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, dict):
            return {k: sanitize(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [sanitize(vv) for vv in v]
        return v

    output = {
        'run_date': datetime.now().isoformat(),
        'strategy': 'Chain E: RSI Divergence Setup -> Bond Yield Drop Trigger',
        'baseline': BASELINE,
        'n_passed': n_passed,
        'n_total': n_total,
        'verdict': 'ROBUST' if n_passed == n_total else ('LIKELY_REAL' if n_passed >= 4 else 'SPURIOUS'),
        'tests': [sanitize(r) for r in all_results],
    }

    out_file = OUTPUT_DIR / 'adversarial_e_results.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_file}")

    print("\n" + "=" * 80)
    print(f"DONE — {n_passed}/{n_total} adversarial tests passed")
    print("=" * 80)

    return output


if __name__ == '__main__':
    main()
