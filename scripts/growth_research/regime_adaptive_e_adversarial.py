#!/usr/bin/env python3
"""
Adversarial Backtest: Best-of-4 Adaptive (Variant E)
=====================================================
6 adversarial tests against the regime-adaptive meta-strategy that passed
5-gate with Sharpe 1.855, Sortino 2.724, WR 66.1%, PF 2.75, 127 trades.

Tests:
  1. Re-implementation (reproduce Sharpe within +/-30%)
  2. Inverse Signal (use WORST trailing strategy)
  3. Random Strategy Selection (1000 permutations)
  4. Sub-Period Stability (4 equal sub-periods, all positive Sharpe)
  5. Top-3 Ticker Removal (Sharpe must not drop >50%)
  6. Lookback Sensitivity (180 combos, >=80% must have Sharpe > 0.30)
"""

import os, sys, json, warnings, time, functools, pickle, itertools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/regime_adaptive')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
    'MA', 'WMT', 'PEP', 'KO', 'TMO', 'ABT', 'BAC', 'XOM', 'CVX', 'CSCO',
]

MACRO_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', '^TNX', 'HYG', 'LQD']

# Baseline metrics from the original run
BASELINE_SHARPE = 1.855
BASELINE_SORTINO = 2.724
BASELINE_WR = 0.661
BASELINE_PF = 2.75
BASELINE_N_TRADES = 127

print("=" * 80)
print("ADVERSARIAL BACKTEST: Best-of-4 Adaptive (Variant E)")
print("=" * 80)
print(f"Baseline: Sharpe={BASELINE_SHARPE}, Sortino={BASELINE_SORTINO}, "
      f"WR={BASELINE_WR:.1%}, PF={BASELINE_PF}, N={BASELINE_N_TRADES}")


# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

def download_data():
    import yfinance as yf
    cache_file = OUTPUT_DIR / '_regime_adaptive_cache.pkl'
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded cache: {len(data['close'].columns)} stocks, {len(data['close'])} days")
        return data

    print(f"\n  Downloading {len(UNIVERSE)} stocks + {len(MACRO_TICKERS)} macro tickers...")
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
            except:
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
# INDICATOR HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _realized_vol(returns, window=20):
    return returns.rolling(window).std() * np.sqrt(252) * 100


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL GENERATORS (exact replication from original)
# ═══════════════════════════════════════════════════════════════════════

def signal_base_mr(data, stock_tickers):
    """Base MR: RSI(14) < 35 AND stock > 5% below 20-SMA."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        if len(close) < 25:
            continue
        rsi = _rsi(close, 14)
        sma20 = close.rolling(20).mean()
        below_sma = (close - sma20) / sma20
        for i in range(20, len(close)):
            if pd.notna(rsi.iloc[i]) and rsi.iloc[i] < 35 and below_sma.iloc[i] < -0.05:
                signals[(close.index[i], t)] = True
    return signals


def signal_iv_rv_gap(data, stock_tickers):
    """IV-RV Gap: VIX > realized vol by 5pts AND dip AND RSI < 40."""
    spy_close = data['close']['SPY']
    spy_ret = spy_close.pct_change()
    vix = data['close']['^VIX']
    rv_20 = _realized_vol(spy_ret, 20)
    gap = vix - rv_20

    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        if len(close) < 25:
            continue
        rsi = _rsi(close, 14)
        sma20 = close.rolling(20).mean()
        below_sma = (close - sma20) / sma20
        for i in range(20, len(close)):
            day = close.index[i]
            if day not in gap.index:
                continue
            g = gap.get(day, np.nan)
            if pd.notna(g) and g > 5:
                if pd.notna(rsi.iloc[i]) and rsi.iloc[i] < 40 and below_sma.iloc[i] < -0.05:
                    signals[(day, t)] = True
    return signals


def signal_bond_yield(data, stock_tickers):
    """Bond Yield: ^TNX drops > 0.1% over 5 days AND dip."""
    if '^TNX' not in data['close'].columns:
        tlt = data['close']['TLT']
        yield_change = tlt.pct_change(5)
        fire_mask = yield_change > 0.02
    else:
        tnx = data['close']['^TNX']
        yield_change = tnx.diff(5)
        fire_mask = yield_change < -0.10

    fire_days = set(fire_mask[fire_mask == True].index)

    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        if len(close) < 25:
            continue
        sma20 = close.rolling(20).mean()
        below_sma = (close - sma20) / sma20
        for i in range(20, len(close)):
            day = close.index[i]
            if day in fire_days and below_sma.iloc[i] < -0.05:
                signals[(day, t)] = True
    return signals


def signal_liquidity(data, stock_tickers):
    """Liquidity: (H-L)/C < 60d avg AND dip AND RSI < 40."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns or t not in data['high'].columns:
            continue
        close = data['close'][t].dropna()
        high = data['high'][t].dropna()
        low = data['low'][t].dropna()
        idx = close.index.intersection(high.index).intersection(low.index)
        if len(idx) < 65:
            continue
        close_a = close.loc[idx]
        high_a = high.loc[idx]
        low_a = low.loc[idx]

        hl_spread = (high_a - low_a) / close_a
        avg_60 = hl_spread.rolling(60).mean()
        rsi = _rsi(close_a, 14)
        sma20 = close_a.rolling(20).mean()
        below_sma = (close_a - sma20) / sma20

        for i in range(60, len(idx)):
            if pd.isna(avg_60.iloc[i]) or pd.isna(rsi.iloc[i]):
                continue
            if hl_spread.iloc[i] < avg_60.iloc[i] and rsi.iloc[i] < 40 and below_sma.iloc[i] < -0.05:
                signals[(idx[i], t)] = True
    return signals


# ═══════════════════════════════════════════════════════════════════════
# BEST-OF-4 ADAPTIVE (Variant E) — exact replication
# ═══════════════════════════════════════════════════════════════════════

def _compute_per_signal_trades(all_sigs, data, stock_tickers, hold_days=HOLD_DAYS):
    """Precompute per-signal trade outcomes for rolling performance tracking."""
    close = data['close']
    signal_names = ['base_mr', 'iv_rv_gap', 'bond_yield', 'liquidity']
    per_signal_trades = {}

    for sig_name in signal_names:
        sig_dict = all_sigs.get(sig_name, {})
        trades = []
        for (d, t) in sorted(sig_dict.keys()):
            if t not in close.columns:
                continue
            try:
                entry_price = close.at[d, t]
            except:
                continue
            if pd.isna(entry_price) or entry_price <= 0:
                continue
            future = close.index[close.index > d]
            if len(future) < 1:
                continue
            hold_end = min(hold_days, len(future))
            try:
                exit_price = close.at[future[hold_end - 1], t]
            except:
                continue
            if pd.isna(exit_price):
                continue
            ret = (exit_price - entry_price) / entry_price - SPREAD_COST_PCT
            trades.append({'date': d, 'ticker': t, 'return': ret})
        per_signal_trades[sig_name] = pd.DataFrame(trades) if trades else pd.DataFrame(columns=['date', 'ticker', 'return'])

    return per_signal_trades


def build_best_of_4_adaptive(all_sigs, data, stock_tickers, trailing_window=60):
    """
    Variant E: On each day, pick the signal with best trailing Sharpe.
    """
    bt_start = pd.Timestamp(BACKTEST_START)
    close = data['close']
    signal_names = ['base_mr', 'iv_rv_gap', 'bond_yield', 'liquidity']

    per_signal_trades = _compute_per_signal_trades(all_sigs, data, stock_tickers)

    all_days = set()
    for sig_name in signal_names:
        for (d, t) in all_sigs.get(sig_name, {}):
            if d >= bt_start:
                all_days.add(d)

    result = {}
    selection_log = {}  # track which signal was selected each day

    for day in sorted(all_days):
        lookback_start = day - pd.Timedelta(days=trailing_window)

        best_sig = 'base_mr'
        best_sharpe = -999

        for sig_name in signal_names:
            df = per_signal_trades[sig_name]
            if len(df) == 0:
                continue
            recent = df[(df['date'] >= lookback_start) & (df['date'] < day)]
            if len(recent) < 3:
                continue
            rets = recent['return'].values
            mean_r = np.mean(rets)
            std_r = np.std(rets, ddof=1)
            sharpe = mean_r / std_r if std_r > 0 else 0
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_sig = sig_name

        selection_log[day] = best_sig

        sig_dict = all_sigs.get(best_sig, {})
        for (d, t) in sig_dict:
            if d == day:
                result[(day, t)] = True

    return result, selection_log


def build_worst_of_4_adaptive(all_sigs, data, stock_tickers, trailing_window=60):
    """
    Inverse: On each day, pick the signal with WORST trailing Sharpe.
    """
    bt_start = pd.Timestamp(BACKTEST_START)
    close = data['close']
    signal_names = ['base_mr', 'iv_rv_gap', 'bond_yield', 'liquidity']

    per_signal_trades = _compute_per_signal_trades(all_sigs, data, stock_tickers)

    all_days = set()
    for sig_name in signal_names:
        for (d, t) in all_sigs.get(sig_name, {}):
            if d >= bt_start:
                all_days.add(d)

    result = {}
    for day in sorted(all_days):
        lookback_start = day - pd.Timedelta(days=trailing_window)

        worst_sig = 'base_mr'
        worst_sharpe = 999

        for sig_name in signal_names:
            df = per_signal_trades[sig_name]
            if len(df) == 0:
                continue
            recent = df[(df['date'] >= lookback_start) & (df['date'] < day)]
            if len(recent) < 3:
                continue
            rets = recent['return'].values
            mean_r = np.mean(rets)
            std_r = np.std(rets, ddof=1)
            sharpe = mean_r / std_r if std_r > 0 else 0
            if sharpe < worst_sharpe:
                worst_sharpe = sharpe
                worst_sig = sig_name

        sig_dict = all_sigs.get(worst_sig, {})
        for (d, t) in sig_dict:
            if d == day:
                result[(day, t)] = True

    return result


def build_random_selection(all_sigs, data, stock_tickers, rng):
    """
    Random: On each day, randomly pick which of the 4 strategies to accept.
    """
    bt_start = pd.Timestamp(BACKTEST_START)
    signal_names = ['base_mr', 'iv_rv_gap', 'bond_yield', 'liquidity']

    all_days = set()
    for sig_name in signal_names:
        for (d, t) in all_sigs.get(sig_name, {}):
            if d >= bt_start:
                all_days.add(d)

    result = {}
    for day in sorted(all_days):
        chosen = signal_names[rng.integers(0, 4)]
        sig_dict = all_sigs.get(chosen, {})
        for (d, t) in sig_dict:
            if d == day:
                result[(day, t)] = True

    return result


# ═══════════════════════════════════════════════════════════════════════
# BACKTESTER
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(signal_entries, data, stock_tickers, hold_days=HOLD_DAYS,
                 profit_target=PROFIT_TARGET, stop_loss=STOP_LOSS,
                 exclude_tickers=None):
    """Run backtest on a set of signal entries {(date, ticker): True}."""
    close = data['close']
    spy_close = close['SPY']
    spy_sma200 = spy_close.rolling(200).mean()

    bt_start = pd.Timestamp(BACKTEST_START)
    entries = [(d, t) for (d, t) in signal_entries if d >= bt_start]
    if exclude_tickers:
        entries = [(d, t) for (d, t) in entries if t not in exclude_tickers]
    entries.sort(key=lambda x: x[0])

    if not entries:
        return None

    trades = []
    open_positions = []

    for entry_date, ticker in entries:
        open_positions = [p for p in open_positions if p[3] > entry_date]
        if len(open_positions) >= MAX_CONCURRENT:
            continue

        try:
            entry_price = close.at[entry_date, ticker]
        except:
            continue
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        future_dates = close.index[close.index > entry_date]
        if len(future_dates) == 0:
            continue

        exit_price = None
        exit_date = None
        exit_reason = 'hold_expiry'

        for j, fdate in enumerate(future_dates[:hold_days]):
            try:
                price = close.at[fdate, ticker]
            except:
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
                except:
                    continue
                if pd.isna(exit_price):
                    continue

        if exit_price is None:
            continue

        raw_ret = (exit_price - entry_price) / entry_price
        net_ret = raw_ret - SPREAD_COST_PCT
        pnl = POS_SIZE * net_ret

        try:
            spy_regime = 'bull' if spy_close.at[entry_date] > spy_sma200.at[entry_date] else 'bear'
        except:
            spy_regime = 'unknown'

        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'ticker': ticker,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return': net_ret,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'regime': spy_regime,
            'hold_days': (exit_date - entry_date).days,
        })
        open_positions.append((entry_date, ticker, entry_price, exit_date))

    if not trades:
        return None

    return pd.DataFrame(trades)


def compute_sharpe(trades_df):
    """Compute annualized Sharpe from trades DataFrame."""
    if trades_df is None or len(trades_df) < 5:
        return 0.0
    rets = trades_df['return'].values
    n = len(rets)
    date_range_years = max(0.5, (trades_df['entry_date'].max() - trades_df['entry_date'].min()).days / 365.25)
    tpy = max(1, n / date_range_years)
    mean_r = np.mean(rets)
    std_r = np.std(rets, ddof=1)
    if std_r <= 0:
        return 0.0
    return (mean_r / std_r) * np.sqrt(tpy)


def compute_metrics(trades_df):
    """Compute full metrics dict from trades DataFrame."""
    if trades_df is None or len(trades_df) < 3:
        return {'sharpe': 0.0, 'sortino': 0.0, 'wr': 0.0, 'pf': 0.0, 'n_trades': 0}

    rets = trades_df['return'].values
    n = len(rets)
    date_range_years = max(0.5, (trades_df['entry_date'].max() - trades_df['entry_date'].min()).days / 365.25)
    tpy = max(1, n / date_range_years)

    mean_r = np.mean(rets)
    std_r = np.std(rets, ddof=1)
    sharpe = (mean_r / std_r) * np.sqrt(tpy) if std_r > 0 else 0.0

    downside = rets[rets < 0]
    if len(downside) >= 2:
        down_std = np.std(downside, ddof=1)
        sortino = (mean_r / down_std) * np.sqrt(tpy) if down_std > 0 else 99.0
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
        'mean_ret_pct': round(mean_r * 100, 2),
    }


# ═══════════════════════════════════════════════════════════════════════
# ADVERSARIAL TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_1_reimplementation(all_sigs, data, stock_tickers):
    """Re-implement from scratch. Must reproduce Sharpe within +/-30%."""
    print("\n" + "─" * 70)
    print("TEST 1: Re-implementation (reproduce Sharpe within +/-30%)")
    print("─" * 70)

    signals, selection_log = build_best_of_4_adaptive(all_sigs, data, stock_tickers)
    trades_df = run_backtest(signals, data, stock_tickers)
    metrics = compute_metrics(trades_df)

    reproduced_sharpe = metrics['sharpe']
    ratio = reproduced_sharpe / BASELINE_SHARPE if BASELINE_SHARPE != 0 else 0
    within_range = 0.70 <= ratio <= 1.30

    print(f"  Reproduced: Sharpe={reproduced_sharpe:.3f}, N={metrics['n_trades']}, "
          f"WR={metrics['wr']:.1%}, PF={metrics['pf']:.2f}")
    print(f"  Baseline:   Sharpe={BASELINE_SHARPE:.3f}, N={BASELINE_N_TRADES}")
    print(f"  Ratio: {ratio:.3f} (need 0.70-1.30)")

    passed = within_range
    print(f"  => {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'reimplementation',
        'passed': passed,
        'reproduced_sharpe': reproduced_sharpe,
        'baseline_sharpe': BASELINE_SHARPE,
        'ratio': round(ratio, 3),
        'reproduced_n': metrics['n_trades'],
        'metrics': metrics,
    }


def test_2_inverse_signal(all_sigs, data, stock_tickers):
    """Use WORST-performing trailing strategy. If ratio > 0.50, adaptive adds nothing."""
    print("\n" + "─" * 70)
    print("TEST 2: Inverse Signal (worst trailing strategy)")
    print("─" * 70)

    # Get best-of-4 Sharpe first
    best_signals, _ = build_best_of_4_adaptive(all_sigs, data, stock_tickers)
    best_trades = run_backtest(best_signals, data, stock_tickers)
    best_sharpe = compute_sharpe(best_trades)

    # Worst-of-4
    worst_signals = build_worst_of_4_adaptive(all_sigs, data, stock_tickers)
    worst_trades = run_backtest(worst_signals, data, stock_tickers)
    worst_metrics = compute_metrics(worst_trades)
    worst_sharpe = worst_metrics['sharpe']

    ratio = worst_sharpe / best_sharpe if best_sharpe > 0 else 999

    print(f"  Best-of-4 Sharpe:  {best_sharpe:.3f}")
    print(f"  Worst-of-4 Sharpe: {worst_sharpe:.3f}")
    print(f"  Ratio (worst/best): {ratio:.3f} (must be < 0.50 to pass)")
    print(f"  Worst-of-4 metrics: N={worst_metrics['n_trades']}, "
          f"WR={worst_metrics['wr']:.1%}, PF={worst_metrics['pf']:.2f}")

    passed = ratio < 0.50
    print(f"  => {'PASS' if passed else 'FAIL'}: Adaptive selection {'DOES' if passed else 'does NOT'} add value")

    return {
        'test': 'inverse_signal',
        'passed': passed,
        'best_sharpe': round(best_sharpe, 3),
        'worst_sharpe': round(worst_sharpe, 3),
        'ratio': round(ratio, 3),
        'worst_metrics': worst_metrics,
    }


def test_3_random_selection(all_sigs, data, stock_tickers):
    """Random strategy selection, 1000 permutations. p must be < 0.05."""
    print("\n" + "─" * 70)
    print("TEST 3: Random Strategy Selection (1000 permutations)")
    print("─" * 70)

    # Adaptive Sharpe
    best_signals, _ = build_best_of_4_adaptive(all_sigs, data, stock_tickers)
    best_trades = run_backtest(best_signals, data, stock_tickers)
    adaptive_sharpe = compute_sharpe(best_trades)

    N_PERMS = 1000
    random_sharpes = []
    t0 = time.time()

    for i in range(N_PERMS):
        rng = np.random.default_rng(seed=i)
        rand_signals = build_random_selection(all_sigs, data, stock_tickers, rng)
        rand_trades = run_backtest(rand_signals, data, stock_tickers)
        rand_sharpe = compute_sharpe(rand_trades)
        random_sharpes.append(rand_sharpe)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"    Completed {i+1}/{N_PERMS} permutations ({elapsed:.0f}s)")

    random_sharpes = np.array(random_sharpes)
    p_value = np.mean(random_sharpes >= adaptive_sharpe)

    print(f"  Adaptive Sharpe: {adaptive_sharpe:.3f}")
    print(f"  Random mean Sharpe: {np.mean(random_sharpes):.3f} +/- {np.std(random_sharpes):.3f}")
    print(f"  Random median Sharpe: {np.median(random_sharpes):.3f}")
    print(f"  Random max Sharpe: {np.max(random_sharpes):.3f}")
    print(f"  p-value (random >= adaptive): {p_value:.4f} (must be < 0.05)")

    passed = p_value < 0.05
    print(f"  => {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'random_selection',
        'passed': passed,
        'adaptive_sharpe': round(adaptive_sharpe, 3),
        'random_mean_sharpe': round(float(np.mean(random_sharpes)), 3),
        'random_std_sharpe': round(float(np.std(random_sharpes)), 3),
        'random_max_sharpe': round(float(np.max(random_sharpes)), 3),
        'p_value': round(float(p_value), 4),
        'n_perms': N_PERMS,
    }


def test_4_sub_period_stability(all_sigs, data, stock_tickers):
    """Split into 4 equal sub-periods. All 4 must have positive Sharpe."""
    print("\n" + "─" * 70)
    print("TEST 4: Sub-Period Stability (4 equal periods, all positive Sharpe)")
    print("─" * 70)

    best_signals, _ = build_best_of_4_adaptive(all_sigs, data, stock_tickers)
    trades_df = run_backtest(best_signals, data, stock_tickers)

    if trades_df is None or len(trades_df) < 8:
        print("  FAIL: Insufficient trades for sub-period analysis")
        return {'test': 'sub_period_stability', 'passed': False, 'reason': 'insufficient trades'}

    # Sort by entry date and split into 4 equal time periods
    trades_df = trades_df.sort_values('entry_date')
    min_date = trades_df['entry_date'].min()
    max_date = trades_df['entry_date'].max()
    total_days = (max_date - min_date).days
    period_days = total_days / 4

    sub_results = []
    all_positive = True

    for p in range(4):
        period_start = min_date + timedelta(days=int(p * period_days))
        period_end = min_date + timedelta(days=int((p + 1) * period_days))
        if p == 3:
            period_end = max_date + timedelta(days=1)

        period_trades = trades_df[
            (trades_df['entry_date'] >= period_start) &
            (trades_df['entry_date'] < period_end)
        ]

        if len(period_trades) < 3:
            sharpe = 0.0
            n = len(period_trades)
        else:
            rets = period_trades['return'].values
            mean_r = np.mean(rets)
            std_r = np.std(rets, ddof=1)
            # Use raw Sharpe (not annualized) for sub-period since periods are short
            n = len(rets)
            period_years = max(0.25, (period_end - period_start).days / 365.25)
            tpy = max(1, n / period_years)
            sharpe = (mean_r / std_r) * np.sqrt(tpy) if std_r > 0 else 0.0

        positive = sharpe > 0
        if not positive:
            all_positive = False

        sub_results.append({
            'period': p + 1,
            'start': str(period_start.date()),
            'end': str(period_end.date()),
            'n_trades': n,
            'sharpe': round(sharpe, 3),
            'positive': positive,
        })

        status = "+" if positive else "NEGATIVE"
        print(f"  Period {p+1}: {period_start.date()} to {period_end.date()} | "
              f"N={n:3d} | Sharpe={sharpe:6.3f} | {status}")

    passed = all_positive
    print(f"  => {'PASS' if passed else 'FAIL'}: {'All' if passed else 'Not all'} 4 periods have positive Sharpe")

    return {
        'test': 'sub_period_stability',
        'passed': passed,
        'sub_periods': sub_results,
    }


def test_5_top3_ticker_removal(all_sigs, data, stock_tickers):
    """Remove 3 highest-PnL tickers. Sharpe must not drop > 50%."""
    print("\n" + "─" * 70)
    print("TEST 5: Top-3 Ticker Removal (Sharpe must not drop > 50%)")
    print("─" * 70)

    best_signals, _ = build_best_of_4_adaptive(all_sigs, data, stock_tickers)
    trades_df = run_backtest(best_signals, data, stock_tickers)
    full_sharpe = compute_sharpe(trades_df)
    full_metrics = compute_metrics(trades_df)

    if trades_df is None:
        print("  FAIL: No trades")
        return {'test': 'top3_ticker_removal', 'passed': False}

    # Find top 3 tickers by total PnL
    ticker_pnl = trades_df.groupby('ticker')['pnl'].sum().sort_values(ascending=False)
    top3 = list(ticker_pnl.head(3).index)
    top3_pnl = ticker_pnl.head(3).values

    print(f"  Full: Sharpe={full_sharpe:.3f}, N={full_metrics['n_trades']}")
    print(f"  Top 3 tickers by PnL: {top3}")
    print(f"  Top 3 PnL contributions: {[f'${v:.0f}' for v in top3_pnl]}")

    # Re-run backtest excluding top 3
    reduced_trades = run_backtest(best_signals, data, stock_tickers, exclude_tickers=set(top3))
    reduced_sharpe = compute_sharpe(reduced_trades)
    reduced_metrics = compute_metrics(reduced_trades)

    drop_pct = 1 - (reduced_sharpe / full_sharpe) if full_sharpe > 0 else 1.0

    print(f"  Reduced: Sharpe={reduced_sharpe:.3f}, N={reduced_metrics['n_trades']}")
    print(f"  Sharpe drop: {drop_pct:.1%} (must be < 50%)")

    passed = drop_pct < 0.50
    print(f"  => {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'top3_ticker_removal',
        'passed': passed,
        'full_sharpe': round(full_sharpe, 3),
        'reduced_sharpe': round(reduced_sharpe, 3),
        'drop_pct': round(drop_pct, 3),
        'removed_tickers': top3,
        'removed_pnl': [round(float(v), 2) for v in top3_pnl],
        'full_n': full_metrics['n_trades'],
        'reduced_n': reduced_metrics['n_trades'],
    }


def test_6_lookback_sensitivity(all_sigs, data, stock_tickers):
    """
    Grid search: trailing window x hold days x RSI threshold x dip threshold.
    5 x 4 x 3 x 3 = 180 combos. At least 80% must have Sharpe > 0.30.
    """
    print("\n" + "─" * 70)
    print("TEST 6: Lookback Sensitivity (180 combos, >=80% Sharpe > 0.30)")
    print("─" * 70)

    trailing_windows = [30, 45, 60, 90, 120]
    hold_days_list = [10, 14, 21, 30]
    rsi_thresholds = [30, 35, 40]
    dip_thresholds = [0.03, 0.05, 0.07]

    total_combos = len(trailing_windows) * len(hold_days_list) * len(rsi_thresholds) * len(dip_thresholds)
    print(f"  Total combinations: {total_combos}")

    # For RSI/dip threshold variations, we need to regenerate signals
    # Cache signal generation by (rsi_thresh, dip_thresh)
    signal_cache = {}

    def generate_signals_with_params(rsi_thresh, dip_thresh):
        key = (rsi_thresh, dip_thresh)
        if key in signal_cache:
            return signal_cache[key]

        sigs = {}

        # Base MR: RSI < rsi_thresh (but use 35 baseline mapping) AND > dip_thresh below 20-SMA
        base_mr = {}
        for t in stock_tickers:
            if t not in data['close'].columns:
                continue
            close = data['close'][t].dropna()
            if len(close) < 25:
                continue
            rsi = _rsi(close, 14)
            sma20 = close.rolling(20).mean()
            below_sma = (close - sma20) / sma20
            # Base MR uses its own RSI threshold (mapped from the parameter)
            # Original: RSI < 35. We vary the RSI threshold.
            for i in range(20, len(close)):
                if pd.notna(rsi.iloc[i]) and rsi.iloc[i] < rsi_thresh and below_sma.iloc[i] < -dip_thresh:
                    base_mr[(close.index[i], t)] = True
        sigs['base_mr'] = base_mr

        # IV-RV Gap: VIX > RV by 5pts + dip + RSI < rsi_thresh+5 (original was 40 when base was 35)
        spy_close = data['close']['SPY']
        spy_ret = spy_close.pct_change()
        vix = data['close']['^VIX']
        rv_20 = _realized_vol(spy_ret, 20)
        gap = vix - rv_20
        iv_rv_rsi_thresh = rsi_thresh + 5  # maintain +5 offset from base

        iv_rv = {}
        for t in stock_tickers:
            if t not in data['close'].columns:
                continue
            close = data['close'][t].dropna()
            if len(close) < 25:
                continue
            rsi = _rsi(close, 14)
            sma20 = close.rolling(20).mean()
            below_sma = (close - sma20) / sma20
            for i in range(20, len(close)):
                day = close.index[i]
                if day not in gap.index:
                    continue
                g = gap.get(day, np.nan)
                if pd.notna(g) and g > 5:
                    if pd.notna(rsi.iloc[i]) and rsi.iloc[i] < iv_rv_rsi_thresh and below_sma.iloc[i] < -dip_thresh:
                        iv_rv[(day, t)] = True
        sigs['iv_rv_gap'] = iv_rv

        # Bond Yield (no RSI filter in original, just dip threshold)
        if '^TNX' not in data['close'].columns:
            tlt = data['close']['TLT']
            yield_change = tlt.pct_change(5)
            fire_mask = yield_change > 0.02
        else:
            tnx = data['close']['^TNX']
            yield_change = tnx.diff(5)
            fire_mask = yield_change < -0.10
        fire_days = set(fire_mask[fire_mask == True].index)

        bond = {}
        for t in stock_tickers:
            if t not in data['close'].columns:
                continue
            close = data['close'][t].dropna()
            if len(close) < 25:
                continue
            sma20 = close.rolling(20).mean()
            below_sma = (close - sma20) / sma20
            for i in range(20, len(close)):
                day = close.index[i]
                if day in fire_days and below_sma.iloc[i] < -dip_thresh:
                    bond[(day, t)] = True
        sigs['bond_yield'] = bond

        # Liquidity (RSI < rsi_thresh+5 to match IV-RV offset)
        liq = {}
        for t in stock_tickers:
            if t not in data['close'].columns or t not in data['high'].columns:
                continue
            close = data['close'][t].dropna()
            high = data['high'][t].dropna()
            low = data['low'][t].dropna()
            idx = close.index.intersection(high.index).intersection(low.index)
            if len(idx) < 65:
                continue
            close_a = close.loc[idx]
            high_a = high.loc[idx]
            low_a = low.loc[idx]
            hl_spread = (high_a - low_a) / close_a
            avg_60 = hl_spread.rolling(60).mean()
            rsi = _rsi(close_a, 14)
            sma20 = close_a.rolling(20).mean()
            below_sma = (close_a - sma20) / sma20

            for i in range(60, len(idx)):
                if pd.isna(avg_60.iloc[i]) or pd.isna(rsi.iloc[i]):
                    continue
                if hl_spread.iloc[i] < avg_60.iloc[i] and rsi.iloc[i] < iv_rv_rsi_thresh and below_sma.iloc[i] < -dip_thresh:
                    liq[(idx[i], t)] = True
        sigs['liquidity'] = liq

        signal_cache[key] = sigs
        return sigs

    results_grid = []
    sharpe_values = []
    done = 0
    t0 = time.time()

    for trailing_w in trailing_windows:
        for hold_d in hold_days_list:
            for rsi_t in rsi_thresholds:
                for dip_t in dip_thresholds:
                    # Generate signals with these params
                    sigs = generate_signals_with_params(rsi_t, dip_t)

                    # Build adaptive with this trailing window
                    # Use _compute_per_signal_trades with custom hold_days
                    per_sig_trades = _compute_per_signal_trades(sigs, data, stock_tickers, hold_days=hold_d)

                    # Build best-of-4 with custom trailing window
                    bt_start = pd.Timestamp(BACKTEST_START)
                    signal_names = ['base_mr', 'iv_rv_gap', 'bond_yield', 'liquidity']
                    all_days = set()
                    for sig_name in signal_names:
                        for (d, t) in sigs.get(sig_name, {}):
                            if d >= bt_start:
                                all_days.add(d)

                    adaptive_signals = {}
                    for day in sorted(all_days):
                        lookback_start = day - pd.Timedelta(days=trailing_w)
                        best_sig = 'base_mr'
                        best_sharpe = -999
                        for sig_name in signal_names:
                            df = per_sig_trades[sig_name]
                            if len(df) == 0:
                                continue
                            recent = df[(df['date'] >= lookback_start) & (df['date'] < day)]
                            if len(recent) < 3:
                                continue
                            rets = recent['return'].values
                            mean_r = np.mean(rets)
                            std_r = np.std(rets, ddof=1)
                            sh = mean_r / std_r if std_r > 0 else 0
                            if sh > best_sharpe:
                                best_sharpe = sh
                                best_sig = sig_name

                        sig_dict = sigs.get(best_sig, {})
                        for (d, t) in sig_dict:
                            if d == day:
                                adaptive_signals[(day, t)] = True

                    # Backtest with custom hold_days
                    trades_df = run_backtest(adaptive_signals, data, stock_tickers, hold_days=hold_d)
                    sharpe = compute_sharpe(trades_df)
                    n_trades = len(trades_df) if trades_df is not None else 0

                    sharpe_values.append(sharpe)
                    results_grid.append({
                        'trailing_window': trailing_w,
                        'hold_days': hold_d,
                        'rsi_threshold': rsi_t,
                        'dip_threshold': dip_t,
                        'sharpe': round(sharpe, 3),
                        'n_trades': n_trades,
                    })

                    done += 1
                    if done % 36 == 0:
                        elapsed = time.time() - t0
                        print(f"    Completed {done}/{total_combos} ({elapsed:.0f}s)")

    sharpe_values = np.array(sharpe_values)
    n_above = np.sum(sharpe_values > 0.30)
    pct_above = n_above / total_combos

    print(f"\n  Total combos: {total_combos}")
    print(f"  Sharpe > 0.30: {n_above}/{total_combos} ({pct_above:.1%})")
    print(f"  Mean Sharpe: {np.mean(sharpe_values):.3f}")
    print(f"  Median Sharpe: {np.median(sharpe_values):.3f}")
    print(f"  Min Sharpe: {np.min(sharpe_values):.3f}")
    print(f"  Max Sharpe: {np.max(sharpe_values):.3f}")

    passed = pct_above >= 0.80
    print(f"  => {'PASS' if passed else 'FAIL'}: {pct_above:.1%} of combos above 0.30 (need >=80%)")

    return {
        'test': 'lookback_sensitivity',
        'passed': passed,
        'total_combos': total_combos,
        'n_above_030': int(n_above),
        'pct_above_030': round(float(pct_above), 3),
        'mean_sharpe': round(float(np.mean(sharpe_values)), 3),
        'median_sharpe': round(float(np.median(sharpe_values)), 3),
        'min_sharpe': round(float(np.min(sharpe_values)), 3),
        'max_sharpe': round(float(np.max(sharpe_values)), 3),
        'grid_results': results_grid,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # Download data
    print("\n[1] Loading data...")
    data = download_data()
    stock_tickers = [t for t in UNIVERSE if t in data['close'].columns]
    print(f"  Universe: {len(stock_tickers)} stocks")

    # Generate all 4 core signals
    print("\n[2] Generating core signals...")
    all_sigs = {}
    for name, gen_func in [
        ('base_mr', signal_base_mr),
        ('iv_rv_gap', signal_iv_rv_gap),
        ('bond_yield', signal_bond_yield),
        ('liquidity', signal_liquidity),
    ]:
        sigs = gen_func(data, stock_tickers)
        bt_count = sum(1 for (d, t) in sigs if d >= pd.Timestamp(BACKTEST_START))
        print(f"  {name:15s}: {bt_count:5d} entries")
        all_sigs[name] = sigs

    # Run 6 adversarial tests
    print("\n" + "=" * 80)
    print("RUNNING 6 ADVERSARIAL TESTS")
    print("=" * 80)

    results = {}

    results['test_1'] = test_1_reimplementation(all_sigs, data, stock_tickers)
    results['test_2'] = test_2_inverse_signal(all_sigs, data, stock_tickers)
    results['test_3'] = test_3_random_selection(all_sigs, data, stock_tickers)
    results['test_4'] = test_4_sub_period_stability(all_sigs, data, stock_tickers)
    results['test_5'] = test_5_top3_ticker_removal(all_sigs, data, stock_tickers)
    results['test_6'] = test_6_lookback_sensitivity(all_sigs, data, stock_tickers)

    # Summary
    print("\n" + "=" * 80)
    print("ADVERSARIAL TEST SUMMARY: Best-of-4 Adaptive (Variant E)")
    print("=" * 80)

    n_passed = 0
    for key in ['test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6']:
        r = results[key]
        status = "PASS" if r['passed'] else "FAIL"
        n_passed += int(r['passed'])
        test_name = r['test']
        print(f"  {key}: {test_name:25s} => {status}")

    print(f"\n  RESULT: {n_passed}/6 tests passed")

    if n_passed == 6:
        print("  VERDICT: FULL PASS - Strategy survives all adversarial tests")
    elif n_passed >= 4:
        print("  VERDICT: PARTIAL PASS - Strategy has some robustness concerns")
    else:
        print("  VERDICT: FAIL - Strategy does not survive adversarial testing")

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed:.0f}s")

    # Save results
    def _serialize(obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    output_data = {
        'run_date': datetime.now().isoformat(),
        'strategy': 'Best-of-4 Adaptive (Variant E)',
        'baseline': {
            'sharpe': BASELINE_SHARPE,
            'sortino': BASELINE_SORTINO,
            'wr': BASELINE_WR,
            'pf': BASELINE_PF,
            'n_trades': BASELINE_N_TRADES,
        },
        'tests': {},
        'n_passed': n_passed,
        'n_total': 6,
        'verdict': 'FULL PASS' if n_passed == 6 else ('PARTIAL PASS' if n_passed >= 4 else 'FAIL'),
        'runtime_seconds': round(elapsed, 1),
    }

    for key in ['test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6']:
        # Strip grid_results from test_6 to keep JSON manageable
        r = dict(results[key])
        if 'grid_results' in r:
            # Keep summary stats only
            grid = r.pop('grid_results')
            r['grid_sample'] = grid[:5]  # first 5 as sample
        output_data['tests'][key] = r

    output_file = OUTPUT_DIR / 'adversarial_e_results.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=_serialize)
    print(f"\nResults saved to {output_file}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    return output_data


if __name__ == '__main__':
    main()
