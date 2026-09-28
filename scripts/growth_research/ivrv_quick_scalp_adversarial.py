#!/usr/bin/env python3
"""
Adversarial Validation: IV-RV Gap Quick Scalp Strategy
=======================================================
Entry: Buy quality megacap stocks when:
  - VIX > 20-day realized vol by 5+ points
  - Stock > 5% below 20-SMA
  - RSI < 40

Exit: +5% TP, -7% SL, 10-day max hold
Position: $300 fixed, max 2 concurrent
Cost: 0.1% round-trip
Baseline: Sharpe 1.542, Sortino 2.476, WR 60.9%, PF 1.742, 317 trades, p=0.002, regime gap 0.119

6-Test Adversarial Battery:
1. Re-implementation     — Sharpe within +/-30% of 1.542 => [1.079, 2.005]
2. Inverse Signal        — Buy when VIX < RV by 5+, stock ABOVE SMA, RSI > 60; ratio < 0.50
3. Random Timing         — 1000 perms, p < 0.05
4. Sub-Period Stability  — 4 equal periods, ALL positive returns
5. Top-3 Ticker Removal  — Sharpe drop < 50%
6. Parameter Sensitivity — 100+ combos, >=80% with Sharpe > 0.30
"""

import os, sys, json, warnings, time, functools, itertools, pickle
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
HOLD_DAYS = 10
PROFIT_TARGET = 0.05   # +5%
STOP_LOSS = -0.07      # -7%
SPREAD_COST_PCT = 0.001  # 0.1% round-trip
N_PERMS = 1000

# Baseline results for re-implementation check
BASELINE_SHARPE = 1.542
BASELINE_SORTINO = 2.476
BASELINE_WR = 0.609
BASELINE_PF = 1.742
BASELINE_N_TRADES = 317
BASELINE_REGIME_GAP = 0.119

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/ivrv_quick_scalp')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
    'MA', 'WMT', 'PEP', 'KO', 'TMO', 'ABT', 'BAC', 'XOM', 'CVX', 'CSCO',
]
MACRO_TICKERS = ['SPY', '^VIX']

print("=" * 90)
print("ADVERSARIAL VALIDATION: IV-RV GAP QUICK SCALP STRATEGY")
print(f"Baseline: Sharpe={BASELINE_SHARPE}, Sortino={BASELINE_SORTINO}, WR={BASELINE_WR:.1%}, "
      f"PF={BASELINE_PF}, N={BASELINE_N_TRADES}, regime_gap={BASELINE_REGIME_GAP}")
print("=" * 90)

# =====================================================================
# 1. DATA
# =====================================================================
def download_data():
    import yfinance as yf
    cache_file = OUTPUT_DIR / '_ivrv_scalp_cache.pkl'
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
        close = raw[['Close']] if 'Close' in raw else raw
        high = raw[['High']] if 'High' in raw else raw
        low = raw[['Low']] if 'Low' in raw else raw
        volume = raw[['Volume']] if 'Volume' in raw else raw

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


# =====================================================================
# 2. HELPERS
# =====================================================================
def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _realized_vol_20d(series):
    """20-day realized volatility as annualized standard deviation of returns (in vol points, scaled to match VIX)."""
    log_ret = np.log(series / series.shift(1))
    # Annualize: *sqrt(252), express as percentage points like VIX
    return log_ret.rolling(20).std() * np.sqrt(252) * 100


# =====================================================================
# 3. SIGNAL GENERATOR (parameterized)
# =====================================================================
def generate_signals(data, stock_tickers,
                     vix_rv_gap=5.0,
                     sma_drawdown=-0.05,
                     rsi_threshold=40,
                     rsi_period=14,
                     sma_period=20,
                     rv_period=20,
                     invert=False):
    """
    Generate IVRV Quick Scalp entry signals.

    Forward (normal): VIX > RV by gap+, stock < SMA by drawdown, RSI < rsi_threshold
    Inverse: VIX < RV by gap+, stock > SMA, RSI > (100 - rsi_threshold)
    """
    vix = data['close'].get('^VIX')
    if vix is None:
        print("  WARNING: ^VIX not found in data")
        return {}

    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        if len(close) < max(sma_period, rv_period, rsi_period) + 5:
            continue

        rsi = _rsi(close, rsi_period)
        sma = close.rolling(sma_period).mean()
        drawdown_from_sma = (close - sma) / sma
        rv = _realized_vol_20d(close)

        vix_aligned = vix.reindex(close.index).ffill()
        iv_rv_spread = vix_aligned - rv  # VIX minus realized vol

        for i in range(max(sma_period, rv_period, rsi_period) + 5, len(close)):
            date = close.index[i]

            spread = iv_rv_spread.iloc[i]
            dd = drawdown_from_sma.iloc[i]
            r = rsi.iloc[i]

            if pd.isna(spread) or pd.isna(dd) or pd.isna(r):
                continue

            if invert:
                # Inverse: VIX < RV by gap+, stock ABOVE SMA, RSI > (100 - rsi_threshold)
                cond_spread = spread <= -vix_rv_gap
                cond_sma = dd >= abs(sma_drawdown)   # above SMA by same magnitude
                cond_rsi = r >= (100 - rsi_threshold)  # overbought
                if cond_spread and cond_sma and cond_rsi:
                    signals[(date, t)] = True
            else:
                # Forward: VIX > RV by gap+, stock < SMA, RSI < threshold
                cond_spread = spread >= vix_rv_gap
                cond_sma = dd <= sma_drawdown
                cond_rsi = r <= rsi_threshold
                if cond_spread and cond_sma and cond_rsi:
                    signals[(date, t)] = True

    return signals


# =====================================================================
# 4. BACKTESTER
# =====================================================================
def run_backtest(signal_entries, data, stock_tickers,
                 hold_days=HOLD_DAYS, profit_target=PROFIT_TARGET,
                 stop_loss=STOP_LOSS, backtest_start=BACKTEST_START):
    close = data['close']
    spy_close = close.get('SPY')
    bt_start = pd.Timestamp(backtest_start)
    entries = sorted([(d, t) for (d, t) in signal_entries if d >= bt_start], key=lambda x: x[0])

    if not entries:
        return None

    trades = []
    open_positions = []  # list of exit_dates

    for entry_date, ticker in entries:
        # Prune expired positions
        open_positions = [ed for ed in open_positions if ed > entry_date]
        if len(open_positions) >= MAX_CONCURRENT:
            continue
        if ticker not in close.columns:
            continue
        try:
            entry_price = close.at[entry_date, ticker]
        except KeyError:
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
            except KeyError:
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
                except KeyError:
                    continue
                if pd.isna(exit_price):
                    continue

        if exit_price is None or exit_date is None:
            continue

        raw_ret = (exit_price - entry_price) / entry_price
        net_ret = raw_ret - SPREAD_COST_PCT
        pnl = POS_SIZE * net_ret

        # Regime classification (SPY vs 200-SMA)
        spy_regime = 'unknown'
        if spy_close is not None:
            try:
                sma200 = spy_close.rolling(200).mean()
                spy_val = spy_close.at[entry_date]
                sma_val = sma200.at[entry_date]
                if not pd.isna(spy_val) and not pd.isna(sma_val):
                    spy_regime = 'bull' if spy_val > sma_val else 'bear'
            except KeyError:
                pass

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
        open_positions.append(exit_date)

    if not trades:
        return None
    return pd.DataFrame(trades)


def calc_sharpe(trades_df):
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
    if trades_df is None or len(trades_df) < 5:
        return {'sharpe': 0, 'sortino': 0, 'n_trades': 0, 'wr': 0, 'pf': 0,
                'mdd': 0, 'regime_gap': 1.0, 'mean_ret_pct': 0, 'avg_hold': 0}

    rets = trades_df['return'].values
    n = len(rets)
    years = max(1, (trades_df['entry_date'].max() - trades_df['entry_date'].min()).days / 365.25)
    tpy = max(1, n / years)
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1)
    sharpe = (mean_ret / std_ret) * np.sqrt(tpy) if std_ret > 0 else 0.0

    wr = float(np.mean(rets > 0))
    gross_profit = float(np.sum(rets[rets > 0]))
    gross_loss = float(np.abs(np.sum(rets[rets < 0])))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

    cum_pnl = np.cumsum(trades_df['pnl'].values)
    peak = np.maximum.accumulate(cum_pnl)
    mdd = float(np.min(cum_pnl - peak))

    # Regime gap
    bull = trades_df[trades_df['regime'] == 'bull']
    bear = trades_df[trades_df['regime'] == 'bear']
    if len(bull) >= 3 and len(bear) >= 3:
        bs = float(np.mean(bull['return'])) / max(float(np.std(bull['return'], ddof=1)), 1e-6)
        brs = float(np.mean(bear['return'])) / max(float(np.std(bear['return'], ddof=1)), 1e-6)
        regime_gap = abs(bs - brs) / max(abs(bs), abs(brs), 1e-6)
    else:
        regime_gap = 0.0

    # Sortino
    downside = rets[rets < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else std_ret
    sortino = (mean_ret / downside_std) * np.sqrt(tpy) if downside_std > 0 else 0.0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'n_trades': n,
        'wr': round(wr, 3),
        'pf': round(pf, 3),
        'mdd': round(mdd, 2),
        'regime_gap': round(regime_gap, 3),
        'mean_ret_pct': round(mean_ret * 100, 2),
        'avg_hold': round(float(trades_df['hold_days'].mean()), 1),
    }


# =====================================================================
# 5. ADVERSARIAL TESTS
# =====================================================================

def test_1_reimplementation(trades_df):
    """Re-implementation: Sharpe within +/-30% of baseline 1.542."""
    sharpe = calc_sharpe(trades_df)
    n = len(trades_df) if trades_df is not None else 0
    lower = BASELINE_SHARPE * 0.70   # 1.079
    upper = BASELINE_SHARPE * 1.30   # 2.005
    passed = lower <= sharpe <= upper
    return {
        'test': 'T1_reimplementation',
        'passed': passed,
        'sharpe': round(sharpe, 3),
        'n_trades': n,
        'baseline_sharpe': BASELINE_SHARPE,
        'range': f"[{lower:.3f}, {upper:.3f}]",
        'detail': f"Sharpe={sharpe:.3f} vs baseline={BASELINE_SHARPE} => range [{lower:.3f},{upper:.3f}]",
    }


def test_2_inverse_signal(data, stock_tickers, forward_sharpe):
    """
    Inverse: Buy when VIX < RV by 5+, stock ABOVE 20-SMA by 5%, RSI > 60.
    Ratio = |inverse_sharpe / forward_sharpe| must be < 0.50.
    """
    print("    Generating inverse signals...")
    inv_signals = generate_signals(data, stock_tickers, invert=True,
                                   vix_rv_gap=5.0, sma_drawdown=-0.05, rsi_threshold=40)
    print(f"    Inverse signal count: {len(inv_signals)}")
    inv_trades = run_backtest(inv_signals, data, stock_tickers)
    inv_sharpe = calc_sharpe(inv_trades)
    inv_n = len(inv_trades) if inv_trades is not None else 0

    ratio = abs(inv_sharpe / forward_sharpe) if abs(forward_sharpe) > 1e-6 else 999.0
    passed = ratio < 0.50
    return {
        'test': 'T2_inverse_signal',
        'passed': passed,
        'inverse_sharpe': round(inv_sharpe, 3),
        'forward_sharpe': round(forward_sharpe, 3),
        'ratio': round(ratio, 3),
        'inv_n_trades': inv_n,
        'detail': f"Inverse Sharpe={inv_sharpe:.3f}, ratio={ratio:.3f} (need <0.50), N={inv_n}",
    }


def test_3_random_timing(data, stock_tickers, n_signals, actual_sharpe):
    """1000 random timing permutations. p-value must be < 0.05."""
    bt_start = pd.Timestamp(BACKTEST_START)
    valid_dates = data['close'].index[data['close'].index >= bt_start]
    valid_dates = valid_dates[:-HOLD_DAYS - 5]

    rng = np.random.RandomState(42)
    perm_sharpes = []
    signal_size = min(n_signals, len(valid_dates))

    for i in range(N_PERMS):
        rand_dates = rng.choice(valid_dates, size=signal_size, replace=True)
        rand_tickers = rng.choice(stock_tickers, size=signal_size, replace=True)
        rand_signals = {}
        for d, t in zip(rand_dates, rand_tickers):
            rand_signals[(d, t)] = True
        rand_trades = run_backtest(rand_signals, data, stock_tickers)
        perm_sharpes.append(calc_sharpe(rand_trades))

        if (i + 1) % 200 == 0:
            print(f"    Perm {i+1}/{N_PERMS} done, running mean={np.mean(perm_sharpes):.3f}")

    perm_arr = np.array(perm_sharpes)
    perm_p = float(np.mean(perm_arr >= actual_sharpe))
    passed = perm_p < 0.05
    return {
        'test': 'T3_random_timing',
        'passed': passed,
        'perm_p': round(perm_p, 4),
        'actual_sharpe': round(actual_sharpe, 3),
        'perm_mean': round(float(np.mean(perm_arr)), 3),
        'perm_std': round(float(np.std(perm_arr)), 3),
        'perm_95th': round(float(np.percentile(perm_arr, 95)), 3),
        'n_perms': N_PERMS,
        'detail': f"p={perm_p:.4f} (need <0.05), actual={actual_sharpe:.3f}, perm_mean={np.mean(perm_arr):.3f}",
    }


def test_4_subperiod_stability(trades_df):
    """4 equal time periods, ALL must have positive mean return."""
    if trades_df is None or len(trades_df) < 20:
        return {
            'test': 'T4_subperiod_stability', 'passed': False,
            'detail': f'Insufficient trades ({len(trades_df) if trades_df is not None else 0}) for 4 sub-periods',
            'period_sharpes': [], 'period_returns': [], 'period_n_trades': [],
        }

    trades_sorted = trades_df.sort_values('entry_date').reset_index(drop=True)
    n = len(trades_sorted)
    chunk = n // 4
    period_sharpes = []
    period_returns = []
    period_dates = []
    period_ns = []
    all_positive = True

    for i in range(4):
        start = i * chunk
        end = (i + 1) * chunk if i < 3 else n
        sub = trades_sorted.iloc[start:end]
        sub_rets = sub['return'].values
        mean_r = float(np.mean(sub_rets))
        period_returns.append(round(mean_r * 100, 2))
        s = calc_sharpe(sub)
        period_sharpes.append(round(float(s), 3))
        period_ns.append(len(sub))
        d_start = sub['entry_date'].min().strftime('%Y-%m') if len(sub) > 0 else '?'
        d_end = sub['entry_date'].max().strftime('%Y-%m') if len(sub) > 0 else '?'
        period_dates.append(f"{d_start}..{d_end}")
        if mean_r <= 0:
            all_positive = False

    return {
        'test': 'T4_subperiod_stability',
        'passed': all_positive,
        'period_sharpes': period_sharpes,
        'period_returns': period_returns,
        'period_dates': period_dates,
        'period_ns': period_ns,
        'detail': (f"Periods: {period_dates}\n"
                   f"       Returns%: {period_returns}, Sharpes: {period_sharpes}, all positive: {all_positive}"),
    }


def test_5_top3_ticker_removal(data, stock_tickers, all_signals, normal_sharpe):
    """Remove top-3 tickers by trade count, Sharpe drop must be < 50%."""
    # Count SIGNAL entries per ticker
    ticker_counts = {}
    for (day, t) in all_signals:
        ticker_counts[t] = ticker_counts.get(t, 0) + 1

    if not ticker_counts:
        return {
            'test': 'T5_top3_ticker_removal', 'passed': False,
            'detail': 'No signals to count tickers from',
        }

    top3 = sorted(ticker_counts, key=ticker_counts.get, reverse=True)[:3]
    reduced_signals = {(d, t): True for (d, t) in all_signals if t not in top3}
    reduced_tickers = [t for t in stock_tickers if t not in top3]
    reduced_trades = run_backtest(reduced_signals, data, reduced_tickers)
    reduced_sharpe = calc_sharpe(reduced_trades)
    reduced_n = len(reduced_trades) if reduced_trades is not None else 0

    drop_pct = (normal_sharpe - reduced_sharpe) / abs(normal_sharpe) if abs(normal_sharpe) > 1e-6 else 1.0
    passed = drop_pct < 0.50
    return {
        'test': 'T5_top3_ticker_removal',
        'passed': passed,
        'top3_removed': top3,
        'top3_counts': [ticker_counts.get(t, 0) for t in top3],
        'normal_sharpe': round(normal_sharpe, 3),
        'reduced_sharpe': round(reduced_sharpe, 3),
        'drop_pct': round(float(drop_pct), 3),
        'reduced_n_trades': reduced_n,
        'detail': (f"Removed {top3}: Sharpe {normal_sharpe:.3f} -> {reduced_sharpe:.3f}, "
                   f"drop={drop_pct:.1%} (need <50%), N={reduced_n}"),
    }


def test_6_parameter_sensitivity(data, stock_tickers):
    """
    Grid: RSI threshold [30,35,40,45], VIX-RV gap [3,5,7,10], SMA drawdown [-3%,-5%,-7%,-10%],
          TP [3%,5%,7%], SL [-5%,-7%,-10%], hold [5,10,15].
    Sample at least 100+ combos. >=80% must have Sharpe > 0.30.
    """
    rsi_thresholds = [30, 35, 40, 45]
    vix_rv_gaps = [3, 5, 7, 10]
    sma_drawdowns = [-0.03, -0.05, -0.07, -0.10]
    tps = [0.03, 0.05, 0.07]
    sls = [-0.05, -0.07, -0.10]
    holds = [5, 10, 15]

    # Full grid is 4*4*4*3*3*3 = 1728; sample 150 combos via deterministic random
    all_combos = list(itertools.product(rsi_thresholds, vix_rv_gaps, sma_drawdowns, tps, sls, holds))
    rng = np.random.RandomState(99)
    n_sample = min(150, len(all_combos))
    indices = rng.choice(len(all_combos), size=n_sample, replace=False)
    sampled = [all_combos[i] for i in indices]

    total = 0
    above_threshold = 0
    all_sharpes = []

    for idx, (rsi_t, gap, sma_dd, tp, sl, hold) in enumerate(sampled):
        signals = generate_signals(data, stock_tickers,
                                   vix_rv_gap=gap,
                                   sma_drawdown=sma_dd,
                                   rsi_threshold=rsi_t,
                                   invert=False)
        trades = run_backtest(signals, data, stock_tickers,
                              hold_days=hold, profit_target=tp, stop_loss=sl)
        s = calc_sharpe(trades)
        all_sharpes.append(s)
        total += 1
        if s > 0.30:
            above_threshold += 1

        if (idx + 1) % 25 == 0:
            print(f"    Sensitivity combo {idx+1}/{n_sample}, running: {above_threshold}/{total} above 0.30")

    pct_above = above_threshold / total if total > 0 else 0.0
    passed = pct_above >= 0.80
    return {
        'test': 'T6_parameter_sensitivity',
        'passed': passed,
        'total_combos': total,
        'above_030': above_threshold,
        'pct_above': round(pct_above, 3),
        'median_sharpe': round(float(np.median(all_sharpes)), 3),
        'mean_sharpe': round(float(np.mean(all_sharpes)), 3),
        'min_sharpe': round(float(np.min(all_sharpes)), 3),
        'max_sharpe': round(float(np.max(all_sharpes)), 3),
        'detail': (f"{above_threshold}/{total} ({pct_above:.1%}) have Sharpe>0.30 (need >=80%), "
                   f"median={np.median(all_sharpes):.3f}, mean={np.mean(all_sharpes):.3f}"),
    }


# =====================================================================
# 6. MAIN
# =====================================================================
def main():
    t_start = time.time()
    data = download_data()
    stock_tickers = [t for t in UNIVERSE if t in data['close'].columns]
    print(f"  Universe: {len(stock_tickers)} tickers")

    # --- Generate baseline forward signals ---
    print("\n[SIGNAL] Generating forward signals (default params)...")
    forward_signals = generate_signals(data, stock_tickers,
                                       vix_rv_gap=5.0,
                                       sma_drawdown=-0.05,
                                       rsi_threshold=40,
                                       invert=False)
    print(f"  Forward signal count: {len(forward_signals)}")

    # --- Run baseline backtest ---
    print("\n[BACKTEST] Running baseline backtest...")
    trades_df = run_backtest(forward_signals, data, stock_tickers)
    metrics = calc_metrics(trades_df)

    print(f"\n  BASELINE REIMPLEMENTATION METRICS:")
    print(f"  Sharpe:     {metrics['sharpe']} (target: {BASELINE_SHARPE})")
    print(f"  Sortino:    {metrics['sortino']} (target: {BASELINE_SORTINO})")
    print(f"  WR:         {metrics['wr']:.1%} (target: {BASELINE_WR:.1%})")
    print(f"  PF:         {metrics['pf']} (target: {BASELINE_PF})")
    print(f"  N Trades:   {metrics['n_trades']} (target: {BASELINE_N_TRADES})")
    print(f"  Regime Gap: {metrics['regime_gap']} (target: {BASELINE_REGIME_GAP})")
    print(f"  Mean Ret:   {metrics['mean_ret_pct']}%")
    print(f"  Avg Hold:   {metrics['avg_hold']} days")
    print(f"  Max DD:     ${metrics['mdd']:.2f}")

    forward_sharpe = metrics['sharpe']
    n_signals = len(forward_signals)

    # =====================================================================
    # RUN 6 ADVERSARIAL TESTS
    # =====================================================================
    all_results = []

    print(f"\n{'='*90}")
    print("ADVERSARIAL TEST 1: RE-IMPLEMENTATION")
    print(f"{'='*90}")
    r1 = test_1_reimplementation(trades_df)
    all_results.append(r1)
    print(f"  {'PASS' if r1['passed'] else 'FAIL'}: {r1['detail']}")

    print(f"\n{'='*90}")
    print("ADVERSARIAL TEST 2: INVERSE SIGNAL")
    print(f"{'='*90}")
    r2 = test_2_inverse_signal(data, stock_tickers, forward_sharpe)
    all_results.append(r2)
    print(f"  {'PASS' if r2['passed'] else 'FAIL'}: {r2['detail']}")

    print(f"\n{'='*90}")
    print(f"ADVERSARIAL TEST 3: RANDOM TIMING ({N_PERMS} permutations)")
    print(f"{'='*90}")
    r3 = test_3_random_timing(data, stock_tickers, n_signals, forward_sharpe)
    all_results.append(r3)
    print(f"  {'PASS' if r3['passed'] else 'FAIL'}: {r3['detail']}")

    print(f"\n{'='*90}")
    print("ADVERSARIAL TEST 4: SUB-PERIOD STABILITY")
    print(f"{'='*90}")
    r4 = test_4_subperiod_stability(trades_df)
    all_results.append(r4)
    print(f"  {'PASS' if r4['passed'] else 'FAIL'}: {r4['detail']}")

    print(f"\n{'='*90}")
    print("ADVERSARIAL TEST 5: TOP-3 TICKER REMOVAL")
    print(f"{'='*90}")
    r5 = test_5_top3_ticker_removal(data, stock_tickers, forward_signals, forward_sharpe)
    all_results.append(r5)
    print(f"  {'PASS' if r5['passed'] else 'FAIL'}: {r5['detail']}")

    print(f"\n{'='*90}")
    print("ADVERSARIAL TEST 6: PARAMETER SENSITIVITY (150 combos)")
    print(f"{'='*90}")
    r6 = test_6_parameter_sensitivity(data, stock_tickers)
    all_results.append(r6)
    print(f"  {'PASS' if r6['passed'] else 'FAIL'}: {r6['detail']}")

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    elapsed = time.time() - t_start
    n_passed = sum(1 for r in all_results if r['passed'])
    n_total = len(all_results)

    print(f"\n{'='*90}")
    print(f"FINAL ADVERSARIAL SUMMARY: {n_passed}/{n_total} PASSED")
    print(f"{'='*90}")
    for r in all_results:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  [{status}] {r['test']}: {r['detail']}")

    overall = n_passed >= 5
    print(f"\n  OVERALL VERDICT: {'VALIDATED' if overall else 'REJECTED'} "
          f"({n_passed}/{n_total} tests passed, need >= 5/6)")
    print(f"  Total runtime: {elapsed/60:.1f} min")

    # Save results JSON
    output = {
        'strategy': 'IVRV Quick Scalp',
        'run_timestamp': datetime.now().isoformat(),
        'baseline': {
            'sharpe': BASELINE_SHARPE,
            'sortino': BASELINE_SORTINO,
            'wr': BASELINE_WR,
            'pf': BASELINE_PF,
            'n_trades': BASELINE_N_TRADES,
            'regime_gap': BASELINE_REGIME_GAP,
        },
        'reimplemented_metrics': metrics,
        'tests': all_results,
        'summary': {
            'n_passed': n_passed,
            'n_total': n_total,
            'overall_verdict': 'VALIDATED' if overall else 'REJECTED',
            'elapsed_min': round(elapsed / 60, 1),
        }
    }

    out_file = OUTPUT_DIR / f'adversarial_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_file}")

    return output


if __name__ == '__main__':
    main()
