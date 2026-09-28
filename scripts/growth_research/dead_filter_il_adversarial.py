#!/usr/bin/env python3
"""
Adversarial Audit for Dead-Signal-Filter Combos I and L
========================================================
I) RSI Divergence + VIX Term Structure filter: Sharpe 1.761, 159 trades, perm p=0.010
L) RSI Divergence + Momentum Context filter: Sharpe 1.765, 153 trades, perm p=0.005

6 adversarial tests on EACH combo:
1. Re-implementation (Sharpe within +/-30% of original)
2. Inverse Signal (inverted filter ratio < 0.50)
3. Random Timing (1000 perms, p < 0.05)
4. Sub-Period Stability (4 equal periods, all positive)
5. Top-3 Ticker Removal (Sharpe drop < 50%)
6. Parameter Sensitivity (144-combo grid, >=80% with Sharpe > 0.30)
"""

import os, sys, json, warnings, time, functools, itertools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

# --- Config ---
START_DATE = '2019-01-01'
END_DATE = '2026-07-01'
BACKTEST_START = '2020-01-01'
POS_SIZE = 300.0
MAX_CONCURRENT = 2
HOLD_DAYS = 21
PROFIT_TARGET = 0.10
STOP_LOSS = -0.15
SPREAD_COST_PCT = 0.001
N_PERMS = 1000
REGIME_GAP_LIMIT = 0.50

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/dead_signal_filter')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
    'MA', 'WMT', 'PEP', 'KO', 'TMO', 'ABT', 'BAC', 'XOM', 'CVX', 'CSCO',
]
MACRO_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', '^TNX']

# Original results for re-implementation check
ORIGINAL = {
    'I': {'sharpe': 1.761, 'n_trades': 159},
    'L': {'sharpe': 1.765, 'n_trades': 153},
}

print("=" * 90)
print("ADVERSARIAL AUDIT: DEAD SIGNAL FILTER COMBOS I AND L")
print("=" * 90)

# =====================================================================
# 1. DATA
# =====================================================================
def download_data():
    import yfinance as yf
    cache_file = OUTPUT_DIR / '_dead_filter_cache.pkl'
    if cache_file.exists():
        import pickle
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded cache: {len(data['close'].columns)} stocks, {len(data['close'])} days")
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
            except:
                pass

    close = close.ffill().dropna(how='all')
    data = {
        'close': close,
        'high': high.reindex(close.index).ffill(),
        'low': low.reindex(close.index).ffill(),
        'volume': volume.reindex(close.index).ffill().fillna(0),
    }

    import pickle
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Downloaded: {len(close.columns)} tickers, {len(close)} days")
    return data


# =====================================================================
# 2. HELPERS
# =====================================================================
def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# =====================================================================
# 3. SIGNAL GENERATORS (parameterized)
# =====================================================================
def signal_rsi_divergence(data, stock_tickers, rsi_lookback=14, dip_pct=0.05):
    """RSI Divergence: Price lower low vs rsi_lookback days ago, RSI higher low, volume declining,
    stock > dip_pct below 20-SMA."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        rsi = _rsi(close, rsi_lookback)
        vol = data['volume'].get(t)
        if vol is None:
            continue
        vol = vol.reindex(close.index).fillna(0)
        sma20 = close.rolling(20).mean()
        drawdown = (close - sma20) / sma20

        lb = rsi_lookback
        for i in range(max(20, lb), len(close)):
            # Dip condition
            if drawdown.iloc[i] > -dip_pct:
                continue
            # Price lower low vs lb days ago
            if close.iloc[i] >= close.iloc[i - lb]:
                continue
            # RSI higher low (divergence)
            r_now = rsi.iloc[i]
            r_prev = rsi.iloc[i - lb]
            if pd.isna(r_now) or pd.isna(r_prev):
                continue
            if r_now <= r_prev:
                continue
            # Volume declining
            vol_now = vol.iloc[max(0, i-4):i+1].mean()
            vol_prev = vol.iloc[max(0, i-9):i-4].mean()
            if vol_prev > 0 and vol_now >= vol_prev:
                continue
            signals[(close.index[i], t)] = True
    return signals


def filter_vix_term_structure(data, threshold=1.05, invert=False):
    """Block if VIX/VIX3M > threshold (backwardation). If invert, block when <= threshold."""
    vix = data['close'].get('^VIX')
    vix3m = data['close'].get('^VIX3M')
    if vix is not None and vix3m is not None:
        ratio = vix / vix3m
        if invert:
            return ratio <= threshold  # block when NOT in backwardation
        return ratio > threshold  # block when in backwardation
    return pd.Series(False, index=data['close'].index)


def filter_momentum_context(data, threshold=-8.0, invert=False):
    """Block if SPY 20d return < threshold%. If invert, block when >= threshold."""
    spy_close = data['close']['SPY']
    spy_ret_20d = spy_close.pct_change(20) * 100
    if invert:
        return spy_ret_20d >= threshold  # block when SPY NOT crashed
    return spy_ret_20d < threshold  # block when SPY crashed


def apply_filter(signal_entries, filter_series):
    """Remove entries where filter_series is True (blocked)."""
    filtered = {}
    for (day, ticker) in signal_entries:
        try:
            is_blocked = filter_series.at[day]
        except (KeyError, ValueError):
            is_blocked = False
        if pd.isna(is_blocked):
            is_blocked = False
        if is_blocked:
            continue
        filtered[(day, ticker)] = True
    return filtered


# =====================================================================
# 4. BACKTESTER
# =====================================================================
def run_backtest(signal_entries, data, stock_tickers, hold_days=HOLD_DAYS,
                 profit_target=PROFIT_TARGET, stop_loss=STOP_LOSS):
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
            spy_sma = spy_close.rolling(200).mean()
            spy_regime = 'bull' if spy_close.at[entry_date] > spy_sma.at[entry_date] else 'bear'
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


def calc_sharpe(trades_df):
    """Calculate annualized Sharpe from trades df."""
    if trades_df is None or len(trades_df) < 5:
        return 0.0
    rets = trades_df['return'].values
    n = len(rets)
    years = max(1, (trades_df['entry_date'].max() - trades_df['entry_date'].min()).days / 365.25)
    tpy = max(1, n / years)
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1)
    return (mean_ret / std_ret) * np.sqrt(tpy) if std_ret > 0 else 0.0


def calc_metrics(trades_df):
    """Full metrics dict from trades df."""
    if trades_df is None or len(trades_df) < 5:
        return {'sharpe': 0, 'n_trades': 0, 'wr': 0, 'pf': 0, 'mdd': 0, 'regime_gap': 1.0}

    rets = trades_df['return'].values
    n = len(rets)
    years = max(1, (trades_df['entry_date'].max() - trades_df['entry_date'].min()).days / 365.25)
    tpy = max(1, n / years)
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1)
    sharpe = (mean_ret / std_ret) * np.sqrt(tpy) if std_ret > 0 else 0

    wr = np.mean(rets > 0)
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = np.abs(np.sum(rets[rets < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

    cum_pnl = np.cumsum(trades_df['pnl'].values)
    peak = np.maximum.accumulate(cum_pnl)
    mdd = np.min(cum_pnl - peak)

    # Regime gap
    bull = trades_df[trades_df['regime'] == 'bull']
    bear = trades_df[trades_df['regime'] == 'bear']
    if len(bull) >= 3 and len(bear) >= 3:
        bs = np.mean(bull['return']) / max(np.std(bull['return'], ddof=1), 1e-6)
        brs = np.mean(bear['return']) / max(np.std(bear['return'], ddof=1), 1e-6)
        regime_gap = abs(bs - brs) / max(abs(bs), abs(brs), 1e-6)
    else:
        regime_gap = 0.0

    # Sortino
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else std_ret
    sortino = (mean_ret / downside_std) * np.sqrt(tpy) if downside_std > 0 else 0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'n_trades': n,
        'wr': round(wr, 3),
        'pf': round(pf, 3),
        'mdd': round(mdd, 2),
        'regime_gap': round(regime_gap, 3),
        'mean_ret_pct': round(mean_ret * 100, 2),
        'avg_hold': round(trades_df['hold_days'].mean(), 1),
    }


# =====================================================================
# 5. ADVERSARIAL TESTS
# =====================================================================

def test_1_reimplementation(combo_label, original_sharpe, trades_df):
    """Re-implementation: reproduce Sharpe within +-30%."""
    sharpe = calc_sharpe(trades_df)
    n = len(trades_df) if trades_df is not None else 0
    lower = original_sharpe * 0.70
    upper = original_sharpe * 1.30
    passed = lower <= sharpe <= upper
    return {
        'test': '1_reimplementation',
        'combo': combo_label,
        'passed': passed,
        'sharpe': round(sharpe, 3),
        'n_trades': n,
        'original_sharpe': original_sharpe,
        'range': f"[{lower:.3f}, {upper:.3f}]",
        'detail': f"Sharpe={sharpe:.3f} vs original={original_sharpe:.3f}, range [{lower:.3f},{upper:.3f}]",
    }


def test_2_inverse_signal(combo_label, data, stock_tickers, rsi_signals,
                          filter_func, filter_kwargs, normal_sharpe):
    """Inverse filter: if inverted filter Sharpe / normal Sharpe > 0.50, filter has no value."""
    inv_series = filter_func(data, invert=True, **filter_kwargs)
    inv_signals = apply_filter(rsi_signals, inv_series)
    inv_trades = run_backtest(inv_signals, data, stock_tickers)
    inv_sharpe = calc_sharpe(inv_trades)
    inv_n = len(inv_trades) if inv_trades is not None else 0

    ratio = abs(inv_sharpe / normal_sharpe) if abs(normal_sharpe) > 0 else 999
    passed = ratio < 0.50
    return {
        'test': '2_inverse_signal',
        'combo': combo_label,
        'passed': passed,
        'inverse_sharpe': round(inv_sharpe, 3),
        'normal_sharpe': round(normal_sharpe, 3),
        'ratio': round(ratio, 3),
        'inv_n_trades': inv_n,
        'detail': f"Inverse Sharpe={inv_sharpe:.3f}, ratio={ratio:.3f} (need <0.50)",
    }


def test_3_random_timing(combo_label, data, stock_tickers, n_raw_signals, actual_sharpe):
    """1000 random timing permutations, p < 0.05."""
    bt_start = pd.Timestamp(BACKTEST_START)
    valid_dates = data['close'].index[data['close'].index >= bt_start]
    valid_dates = valid_dates[:-HOLD_DAYS - 5]

    perm_sharpes = []
    for _ in range(N_PERMS):
        rand_dates = np.random.choice(valid_dates, size=min(n_raw_signals, len(valid_dates)), replace=True)
        rand_tickers = np.random.choice(stock_tickers, size=min(n_raw_signals, len(valid_dates)), replace=True)
        rand_signals = {(d, t): True for d, t in zip(rand_dates, rand_tickers)}
        rand_trades = run_backtest(rand_signals, data, stock_tickers)
        perm_sharpes.append(calc_sharpe(rand_trades))

    perm_p = np.mean(np.array(perm_sharpes) >= actual_sharpe)
    passed = perm_p < 0.05
    return {
        'test': '3_random_timing',
        'combo': combo_label,
        'passed': passed,
        'perm_p': round(float(perm_p), 4),
        'actual_sharpe': round(actual_sharpe, 3),
        'perm_mean': round(float(np.mean(perm_sharpes)), 3),
        'perm_95th': round(float(np.percentile(perm_sharpes, 95)), 3),
        'n_perms': N_PERMS,
        'detail': f"p={perm_p:.4f} (need <0.05), actual Sharpe={actual_sharpe:.3f}, perm mean={np.mean(perm_sharpes):.3f}",
    }


def test_4_subperiod_stability(combo_label, trades_df):
    """4 equal sub-periods, all must have positive mean return."""
    if trades_df is None or len(trades_df) < 20:
        return {
            'test': '4_subperiod_stability', 'combo': combo_label, 'passed': False,
            'detail': 'Insufficient trades for 4 sub-periods',
            'period_sharpes': [], 'period_returns': [],
        }

    trades_sorted = trades_df.sort_values('entry_date')
    n = len(trades_sorted)
    chunk = n // 4
    period_sharpes = []
    period_returns = []
    all_positive = True

    for i in range(4):
        start = i * chunk
        end = (i + 1) * chunk if i < 3 else n
        sub = trades_sorted.iloc[start:end]
        sub_rets = sub['return'].values
        mean_r = np.mean(sub_rets)
        period_returns.append(round(float(mean_r * 100), 2))
        s = calc_sharpe(sub)
        period_sharpes.append(round(float(s), 3))
        if mean_r <= 0:
            all_positive = False

    return {
        'test': '4_subperiod_stability',
        'combo': combo_label,
        'passed': all_positive,
        'period_sharpes': period_sharpes,
        'period_returns': period_returns,
        'detail': f"Period Sharpes: {period_sharpes}, returns%: {period_returns}, all positive: {all_positive}",
    }


def test_5_top3_ticker_removal(combo_label, data, stock_tickers, filtered_signals, normal_sharpe):
    """Remove top-3 tickers by trade count, Sharpe drop must be < 50%."""
    # Count trades per ticker
    ticker_counts = {}
    for (day, t) in filtered_signals:
        ticker_counts[t] = ticker_counts.get(t, 0) + 1
    top3 = sorted(ticker_counts, key=ticker_counts.get, reverse=True)[:3]

    reduced_signals = {(d, t): True for (d, t) in filtered_signals if t not in top3}
    reduced_tickers = [t for t in stock_tickers if t not in top3]
    reduced_trades = run_backtest(reduced_signals, data, reduced_tickers)
    reduced_sharpe = calc_sharpe(reduced_trades)
    reduced_n = len(reduced_trades) if reduced_trades is not None else 0

    drop_pct = (normal_sharpe - reduced_sharpe) / abs(normal_sharpe) if abs(normal_sharpe) > 0 else 1.0
    passed = drop_pct < 0.50
    return {
        'test': '5_top3_ticker_removal',
        'combo': combo_label,
        'passed': passed,
        'top3_removed': top3,
        'normal_sharpe': round(normal_sharpe, 3),
        'reduced_sharpe': round(reduced_sharpe, 3),
        'drop_pct': round(float(drop_pct), 3),
        'reduced_n_trades': reduced_n,
        'detail': f"Removed {top3}: Sharpe {normal_sharpe:.3f} -> {reduced_sharpe:.3f}, drop={drop_pct:.1%} (need <50%)",
    }


def test_6_parameter_sensitivity(combo_label, data, stock_tickers, filter_func, filter_param_name,
                                 rsi_lookbacks, filter_thresholds, dip_pcts, hold_days_list):
    """Grid sweep: >=80% of combos must have Sharpe > 0.30."""
    total = 0
    above_threshold = 0
    all_sharpes = []

    for rsi_lb in rsi_lookbacks:
        for filt_thresh in filter_thresholds:
            for dip in dip_pcts:
                for hold in hold_days_list:
                    total += 1
                    dip_frac = dip / 100.0
                    rsi_sigs = signal_rsi_divergence(data, stock_tickers,
                                                     rsi_lookback=rsi_lb, dip_pct=dip_frac)
                    filt_series = filter_func(data, threshold=filt_thresh)
                    filtered = apply_filter(rsi_sigs, filt_series)
                    trades = run_backtest(filtered, data, stock_tickers,
                                          hold_days=hold, profit_target=PROFIT_TARGET,
                                          stop_loss=STOP_LOSS)
                    s = calc_sharpe(trades)
                    all_sharpes.append(s)
                    if s > 0.30:
                        above_threshold += 1

    pct_above = above_threshold / total if total > 0 else 0
    passed = pct_above >= 0.80
    return {
        'test': '6_parameter_sensitivity',
        'combo': combo_label,
        'passed': passed,
        'total_combos': total,
        'above_030': above_threshold,
        'pct_above': round(pct_above, 3),
        'median_sharpe': round(float(np.median(all_sharpes)), 3),
        'min_sharpe': round(float(np.min(all_sharpes)), 3),
        'max_sharpe': round(float(np.max(all_sharpes)), 3),
        'detail': f"{above_threshold}/{total} ({pct_above:.1%}) have Sharpe>0.30 (need >=80%)",
    }


# =====================================================================
# 6. MAIN
# =====================================================================
def main():
    t_start = time.time()
    data = download_data()
    stock_tickers = [t for t in UNIVERSE if t in data['close'].columns]
    print(f"  Universe: {len(stock_tickers)} tickers")

    # --- Generate base RSI divergence signals (default params) ---
    print("\n[1] Generating RSI Divergence signals (default params)...")
    rsi_signals = signal_rsi_divergence(data, stock_tickers, rsi_lookback=14, dip_pct=0.05)
    print(f"  RSI Divergence raw entries: {len(rsi_signals)}")

    # --- Build filtered signal sets ---
    print("\n[2] Building filtered signal sets...")

    # I) VIX Term Structure filter
    vts_filter = filter_vix_term_structure(data, threshold=1.05)
    signals_I = apply_filter(rsi_signals, vts_filter)
    trades_I = run_backtest(signals_I, data, stock_tickers)
    sharpe_I = calc_sharpe(trades_I)
    metrics_I = calc_metrics(trades_I)
    print(f"  I) RSI Div + VIX TermStr: {len(signals_I)} entries, {metrics_I['n_trades']} trades, Sharpe={sharpe_I:.3f}")

    # L) Momentum Context filter
    mc_filter = filter_momentum_context(data, threshold=-8.0)
    signals_L = apply_filter(rsi_signals, mc_filter)
    trades_L = run_backtest(signals_L, data, stock_tickers)
    sharpe_L = calc_sharpe(trades_L)
    metrics_L = calc_metrics(trades_L)
    print(f"  L) RSI Div + Momentum:    {len(signals_L)} entries, {metrics_L['n_trades']} trades, Sharpe={sharpe_L:.3f}")

    # =====================================================================
    # RUN 6 TESTS ON EACH COMBO
    # =====================================================================
    results = {}

    for combo_id, combo_label, filtered_signals, trades_df, sharpe_val, metrics, \
        filter_fn, filter_kwargs, filter_param_name, filter_thresholds in [
        ('I', 'RSI Div + VIX TermStr', signals_I, trades_I, sharpe_I, metrics_I,
         filter_vix_term_structure, {'threshold': 1.05}, 'vix_vix3m_threshold',
         [1.00, 1.05, 1.10, 1.15]),
        ('L', 'RSI Div + Momentum', signals_L, trades_L, sharpe_L, metrics_L,
         filter_momentum_context, {'threshold': -8.0}, 'spy_return_threshold',
         [-5.0, -8.0, -10.0, -12.0]),
    ]:
        print(f"\n{'='*90}")
        print(f"  ADVERSARIAL TESTS FOR COMBO {combo_id}: {combo_label}")
        print(f"{'='*90}")

        combo_results = []

        # Test 1: Re-implementation
        print(f"\n  [{combo_id}] Test 1: Re-implementation...")
        r1 = test_1_reimplementation(combo_label, ORIGINAL[combo_id]['sharpe'], trades_df)
        combo_results.append(r1)
        print(f"    {'PASS' if r1['passed'] else 'FAIL'}: {r1['detail']}")

        # Test 2: Inverse Signal
        print(f"  [{combo_id}] Test 2: Inverse Signal...")
        r2 = test_2_inverse_signal(combo_label, data, stock_tickers, rsi_signals,
                                    filter_fn, filter_kwargs, sharpe_val)
        combo_results.append(r2)
        print(f"    {'PASS' if r2['passed'] else 'FAIL'}: {r2['detail']}")

        # Test 3: Random Timing
        print(f"  [{combo_id}] Test 3: Random Timing ({N_PERMS} perms)...")
        r3 = test_3_random_timing(combo_label, data, stock_tickers, len(filtered_signals), sharpe_val)
        combo_results.append(r3)
        print(f"    {'PASS' if r3['passed'] else 'FAIL'}: {r3['detail']}")

        # Test 4: Sub-Period Stability
        print(f"  [{combo_id}] Test 4: Sub-Period Stability...")
        r4 = test_4_subperiod_stability(combo_label, trades_df)
        combo_results.append(r4)
        print(f"    {'PASS' if r4['passed'] else 'FAIL'}: {r4['detail']}")

        # Test 5: Top-3 Ticker Removal
        print(f"  [{combo_id}] Test 5: Top-3 Ticker Removal...")
        r5 = test_5_top3_ticker_removal(combo_label, data, stock_tickers, filtered_signals, sharpe_val)
        combo_results.append(r5)
        print(f"    {'PASS' if r5['passed'] else 'FAIL'}: {r5['detail']}")

        # Test 6: Parameter Sensitivity (144 combos)
        print(f"  [{combo_id}] Test 6: Parameter Sensitivity (144 combos)...")
        rsi_lookbacks = [10, 14, 20]
        dip_pcts = [3, 5, 7]
        hold_days_list = [10, 14, 21, 30]
        r6 = test_6_parameter_sensitivity(combo_label, data, stock_tickers,
                                           filter_fn, filter_param_name,
                                           rsi_lookbacks, filter_thresholds,
                                           dip_pcts, hold_days_list)
        combo_results.append(r6)
        print(f"    {'PASS' if r6['passed'] else 'FAIL'}: {r6['detail']}")

        n_passed = sum(1 for r in combo_results if r['passed'])
        results[combo_id] = {
            'combo_label': combo_label,
            'metrics': metrics,
            'tests': combo_results,
            'n_passed': n_passed,
            'n_total': 6,
            'verdict': 'PASS' if n_passed >= 5 else 'FAIL',
        }

    # =====================================================================
    # SUMMARY
    # =====================================================================
    elapsed = time.time() - t_start

    print(f"\n{'='*90}")
    print(f"ADVERSARIAL AUDIT SUMMARY")
    print(f"{'='*90}")

    for combo_id in ['I', 'L']:
        r = results[combo_id]
        print(f"\n  {combo_id}) {r['combo_label']}:")
        print(f"     Reproduced: Sharpe={r['metrics']['sharpe']:.3f}, N={r['metrics']['n_trades']}, "
              f"WR={r['metrics']['wr']:.1%}, PF={r['metrics']['pf']:.2f}")
        for t in r['tests']:
            status = 'PASS' if t['passed'] else 'FAIL'
            print(f"     [{status}] {t['test']}: {t['detail']}")
        print(f"     VERDICT: {r['n_passed']}/6 passed -> {r['verdict']}")

    print(f"\n  Total runtime: {elapsed:.0f}s")

    # Save
    def _sanitize(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _sanitize_deep(obj):
        if isinstance(obj, dict):
            return {k: _sanitize_deep(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_deep(v) for v in obj]
        return _sanitize(obj)

    output_data = {
        'run_date': datetime.now().isoformat(),
        'runtime_seconds': round(elapsed, 1),
        'config': {
            'universe_size': len(stock_tickers),
            'backtest_start': BACKTEST_START,
            'backtest_end': END_DATE,
            'pos_size': POS_SIZE,
            'max_concurrent': MAX_CONCURRENT,
            'hold_days': HOLD_DAYS,
            'profit_target': PROFIT_TARGET,
            'stop_loss': STOP_LOSS,
            'spread_cost': SPREAD_COST_PCT,
            'n_perms': N_PERMS,
        },
        'results': _sanitize_deep(results),
    }

    out_file = OUTPUT_DIR / 'adversarial_il_results.json'
    with open(out_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")
    print("=" * 90)
    print("DONE")


if __name__ == '__main__':
    main()
