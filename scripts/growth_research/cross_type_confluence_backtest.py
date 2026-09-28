#!/usr/bin/env python3
"""
Cross-TYPE Confluence Backtest (HC #772 — Phase 2)
====================================================
Previous same-type confluence (combining dip signals) just filtered trades.
This tests CROSS-TYPE confluence — fundamentally different strategy types:

  Type 1: Stock-level mean reversion (RSI dip + price below SMA)
  Type 2: Market-level fear (VIX elevated vs 60d avg)
  Type 3: Cross-asset macro (10Y yield drop)
  Type 4: Price pattern (bullish RSI divergence)
  Type 5: Microstructure (high-low spread narrowing)

Strategies tested:
  A: FEAR + DIP (T2 + T1)
  B: FEAR + MACRO + DIP (T2 + T3 + T1)
  C: FEAR + DIVERGENCE (T2 + T4)
  D: MACRO + MICROSTRUCTURE (T3 + T5)
  E: FEAR + MICRO + DIP (T2 + T5 + T1)
  F: ALL TYPES (T2 + T3 + T4 + T5 + T1) — "Perfect Storm"

5-gate validation: Sharpe>0.3, WR>0.45, PF>1.0, MDD cap, regime gap<0.50, perm p<0.05
Confluence alpha: strategy Sharpe vs best individual component Sharpe.
"""

import os, sys, warnings, time, functools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

# ─── Config ───
START_DATE = '2019-01-01'  # lookback for indicators
END_DATE = '2026-07-01'
BACKTEST_START = '2020-01-01'
POS_SIZE = 300.0
MAX_CONCURRENT = 2
HOLD_DAYS = 21
PROFIT_TARGET = 0.10
STOP_LOSS = -0.15
SPREAD_COST_PCT = 0.001  # 10bps RT
N_PERMS = 200
REGIME_GAP_LIMIT = 0.50
CONFLUENCE_WINDOW = 5  # days — all conditions must be true within this window

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/cross_type_confluence')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'UNH', 'LLY', 'AVGO', 'AMD',
]

MACRO_TICKERS = ['SPY', '^VIX', 'TLT', '^TNX']

print("=" * 80)
print("CROSS-TYPE CONFLUENCE BACKTEST (HC #772 — Phase 2)")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════

def download_data():
    import yfinance as yf
    cache_file = OUTPUT_DIR / '_cross_type_cache.pkl'
    if cache_file.exists():
        import pickle
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded cache: {len(data['close'].columns)} tickers, {len(data['close'])} days")
        return data

    print(f"\n[1] Downloading {len(UNIVERSE)} stocks + {len(MACRO_TICKERS)} macro tickers...")
    all_tickers = list(set(UNIVERSE + MACRO_TICKERS))

    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False, threads=True)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
        high = raw['High']
        low = raw['Low']
        volume = raw['Volume']
    else:
        close = high = low = volume = raw

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

# ═══════════════════════════════════════════════════════════════════════
# 2. INDIVIDUAL TYPE SIGNAL GENERATORS
# ═══════════════════════════════════════════════════════════════════════

def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def gen_type1_dip(data, stock_tickers, rsi_thresh=35, sma_below_pct=0.05):
    """TYPE 1 — Stock-level mean reversion: RSI(14) < threshold AND price > X% below 20-SMA.
    Returns dict: {(date, ticker): True}"""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        if len(close) < 30:
            continue
        rsi = _rsi(close, 14)
        sma20 = close.rolling(20).mean()
        pct_below_sma = (close - sma20) / sma20

        for i in range(20, len(close)):
            if pd.isna(rsi.iloc[i]) or pd.isna(sma20.iloc[i]):
                continue
            if rsi.iloc[i] < rsi_thresh and pct_below_sma.iloc[i] < -sma_below_pct:
                signals[(close.index[i], t)] = True
    return signals


def gen_type2_fear(data, vix_mult=1.15, vix_avg_window=60):
    """TYPE 2 — Market-level fear: VIX > multiplier × 60d avg.
    Returns set of fear dates."""
    vix = data['close'].get('^VIX')
    if vix is None:
        return set()
    vix = vix.dropna()
    vix_avg = vix.rolling(vix_avg_window).mean()
    fear_dates = set()
    for i in range(vix_avg_window, len(vix)):
        if pd.isna(vix_avg.iloc[i]):
            continue
        if vix.iloc[i] > vix_mult * vix_avg.iloc[i]:
            fear_dates.add(vix.index[i])
    return fear_dates


def gen_type3_macro(data, yield_drop_thresh=0.05, yield_drop_thresh_big=0.10, lookback=5):
    """TYPE 3 — Cross-asset macro: 10Y yield dropped > threshold in `lookback` days.
    Returns set of macro-signal dates."""
    macro_dates = {}  # threshold -> set of dates
    for thresh in [yield_drop_thresh, yield_drop_thresh_big]:
        dates = set()
        if '^TNX' in data['close'].columns:
            tnx = data['close']['^TNX'].dropna()
            yield_change = tnx.diff(lookback)
            for i in range(lookback, len(tnx)):
                if pd.isna(yield_change.iloc[i]):
                    continue
                if yield_change.iloc[i] < -thresh:
                    dates.add(tnx.index[i])
        elif 'TLT' in data['close'].columns:
            tlt = data['close']['TLT'].dropna()
            tlt_chg = tlt.pct_change(lookback)
            tlt_thresh = thresh * 0.2  # rough proxy
            for i in range(lookback, len(tlt)):
                if pd.isna(tlt_chg.iloc[i]):
                    continue
                if tlt_chg.iloc[i] > tlt_thresh:
                    dates.add(tlt.index[i])
        macro_dates[thresh] = dates
    return macro_dates


def gen_type4_divergence(data, stock_tickers):
    """TYPE 4 — RSI bullish divergence: price lower low vs 10d ago, RSI higher low.
    Returns dict: {(date, ticker): True}"""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        if len(close) < 30:
            continue
        rsi = _rsi(close, 14)
        for i in range(20, len(close)):
            if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i - 10]):
                continue
            # Price lower low vs 10d ago
            if close.iloc[i] < close.iloc[i - 10]:
                # RSI higher low (bullish divergence — hidden strength)
                if rsi.iloc[i] > rsi.iloc[i - 10]:
                    signals[(close.index[i], t)] = True
    return signals


def gen_type5_microstructure(data, stock_tickers, spread_pct=0.60):
    """TYPE 5 — Microstructure: high-low spread < spread_pct × 60d avg (narrowing).
    Returns dict: {(date, ticker): True}"""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns or t not in data['high'].columns:
            continue
        high = data['high'][t].dropna()
        low = data['low'][t].dropna()
        close = data['close'][t].dropna()
        idx = high.index.intersection(low.index).intersection(close.index)
        if len(idx) < 65:
            continue
        high, low, close = high.loc[idx], low.loc[idx], close.loc[idx]
        hl_spread = (high - low) / close
        avg_60 = hl_spread.rolling(60).mean()
        for i in range(60, len(idx)):
            if pd.isna(avg_60.iloc[i]):
                continue
            if hl_spread.iloc[i] < avg_60.iloc[i] * spread_pct:
                signals[(idx[i], t)] = True
    return signals


# ═══════════════════════════════════════════════════════════════════════
# 3. CROSS-TYPE STRATEGY BUILDERS (with confluence window)
# ═══════════════════════════════════════════════════════════════════════

def _dates_within_window(target_date, date_set, window):
    """Check if any date in date_set is within `window` trading days of target_date."""
    for offset in range(-window, window + 1):
        check = target_date + pd.Timedelta(days=offset)
        if check in date_set:
            return True
    return False


def _signal_within_window(target_date, ticker, signal_dict, window):
    """Check if (date, ticker) appears in signal_dict within window days of target_date."""
    for offset in range(-window, window + 1):
        check = target_date + pd.Timedelta(days=offset)
        if (check, ticker) in signal_dict:
            return True
    return False


def build_strategy_a(type1_sigs, fear_dates, window=CONFLUENCE_WINDOW):
    """FEAR + DIP (Type 2 + Type 1)"""
    signals = {}
    for (date, ticker) in type1_sigs:
        if _dates_within_window(date, fear_dates, window):
            signals[(date, ticker)] = True
    return signals


def build_strategy_b(type1_sigs_rsi40, fear_dates, macro_dates_005, window=CONFLUENCE_WINDOW):
    """FEAR + MACRO + DIP (Type 2 + Type 3 + Type 1) — RSI < 40 for broader entry"""
    signals = {}
    for (date, ticker) in type1_sigs_rsi40:
        if _dates_within_window(date, fear_dates, window) and \
           _dates_within_window(date, macro_dates_005, window):
            signals[(date, ticker)] = True
    return signals


def build_strategy_c(type4_sigs, fear_dates, window=CONFLUENCE_WINDOW):
    """FEAR + DIVERGENCE (Type 2 + Type 4)"""
    signals = {}
    for (date, ticker) in type4_sigs:
        if _dates_within_window(date, fear_dates, window):
            signals[(date, ticker)] = True
    return signals


def build_strategy_d(type5_sigs, macro_dates_010, window=CONFLUENCE_WINDOW):
    """MACRO + MICROSTRUCTURE (Type 3 + Type 5) — yield drop > 0.1%"""
    signals = {}
    for (date, ticker) in type5_sigs:
        if _dates_within_window(date, macro_dates_010, window):
            signals[(date, ticker)] = True
    return signals


def build_strategy_e(type1_sigs_rsi40, fear_dates, type5_sigs, window=CONFLUENCE_WINDOW):
    """FEAR + MICRO + DIP (Type 2 + Type 5 + Type 1)"""
    signals = {}
    for (date, ticker) in type1_sigs_rsi40:
        if _dates_within_window(date, fear_dates, window) and \
           _signal_within_window(date, ticker, type5_sigs, window):
            signals[(date, ticker)] = True
    return signals


def build_strategy_f(type1_sigs, fear_dates, macro_dates_005, type4_sigs, type5_sigs, window=CONFLUENCE_WINDOW):
    """ALL TYPES — Perfect Storm (T2 + T3 + T4 + T5 + T1)"""
    signals = {}
    for (date, ticker) in type1_sigs:
        if _dates_within_window(date, fear_dates, window) and \
           _dates_within_window(date, macro_dates_005, window) and \
           _signal_within_window(date, ticker, type4_sigs, window) and \
           _signal_within_window(date, ticker, type5_sigs, window):
            signals[(date, ticker)] = True
    return signals


# ═══════════════════════════════════════════════════════════════════════
# 4. BACKTESTER
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(signal_entries, data, stock_tickers, hold_days=HOLD_DAYS,
                 profit_target=PROFIT_TARGET, stop_loss=STOP_LOSS,
                 exit_condition_fn=None):
    """Run backtest on signal entries {(date, ticker): True}.
    exit_condition_fn(date, data) -> bool can add conditional exit (e.g. VIX normalizes)."""
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
            # Check conditional exit
            if exit_condition_fn is not None and exit_condition_fn(fdate, data):
                exit_price = price
                exit_date = fdate
                exit_reason = 'condition_exit'
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

    return pd.DataFrame(trades) if trades else None


# ═══════════════════════════════════════════════════════════════════════
# 5. FIVE-GATE VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def validate_strategy(trades_df, label='', run_perm=True, all_signal_entries=None,
                      data=None, stock_tickers=None):
    """5-gate: Sharpe, WR, PF, MDD, regime gap + perm test."""
    if trades_df is None or len(trades_df) < 5:
        return {
            'label': label, 'n_trades': 0 if trades_df is None else len(trades_df),
            'sharpe': 0, 'sortino': 0, 'wr': 0, 'pf': 0, 'mdd': 0,
            'regime_gap': 1.0, 'perm_p': 1.0, 'passed': False,
            'reason': 'insufficient trades (<5)'
        }

    rets = trades_df['return'].values
    n = len(rets)

    # Sharpe
    trades_per_year = n / max(1, (trades_df['entry_date'].max() - trades_df['entry_date'].min()).days / 365.25)
    if trades_per_year < 1:
        trades_per_year = 1
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1.0
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else std_ret
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Win rate
    wr = np.mean(rets > 0)

    # Profit factor
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = np.abs(np.sum(rets[rets < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

    # Max drawdown
    cum_pnl = np.cumsum(trades_df['pnl'].values)
    peak = np.maximum.accumulate(cum_pnl)
    dd = cum_pnl - peak
    mdd = np.min(dd) if len(dd) > 0 else 0
    total_pnl = cum_pnl[-1] if len(cum_pnl) > 0 else 0

    # Regime gap
    bull = trades_df[trades_df['regime'] == 'bull']
    bear = trades_df[trades_df['regime'] == 'bear']
    if len(bull) >= 3 and len(bear) >= 3:
        bull_sr = np.mean(bull['return']) / max(np.std(bull['return'], ddof=1), 1e-6)
        bear_sr = np.mean(bear['return']) / max(np.std(bear['return'], ddof=1), 1e-6)
        max_abs = max(abs(bull_sr), abs(bear_sr), 1e-6)
        regime_gap = abs(bull_sr - bear_sr) / max_abs
    else:
        regime_gap = 0.0

    # Permutation test — random entry timing
    perm_p = 1.0
    if run_perm and n >= 5 and all_signal_entries is not None and data is not None and stock_tickers is not None:
        actual_sharpe = sharpe
        perm_sharpes = []
        bt_start = pd.Timestamp(BACKTEST_START)
        valid_dates = data['close'].index[data['close'].index >= bt_start]
        valid_dates = valid_dates[:-HOLD_DAYS - 5]

        for _ in range(N_PERMS):
            n_raw = len(all_signal_entries)
            rand_dates = np.random.choice(valid_dates, size=min(n_raw, len(valid_dates)), replace=True)
            rand_tickers = np.random.choice(stock_tickers, size=min(n_raw, len(valid_dates)), replace=True)
            rand_signals = {(d, t): True for d, t in zip(rand_dates, rand_tickers)}
            rand_trades = run_backtest(rand_signals, data, stock_tickers)
            if rand_trades is not None and len(rand_trades) >= 3:
                r = rand_trades['return'].values
                r_tpy = len(r) / max(1, (rand_trades['entry_date'].max() - rand_trades['entry_date'].min()).days / 365.25)
                if r_tpy < 1: r_tpy = 1
                r_sharpe = (np.mean(r) / max(np.std(r, ddof=1), 1e-6)) * np.sqrt(r_tpy)
            else:
                r_sharpe = 0.0
            perm_sharpes.append(r_sharpe)
        perm_p = np.mean(np.array(perm_sharpes) >= actual_sharpe)

    # 5-gate check
    gates = {
        'sharpe': sharpe > 0.3,
        'wr': wr > 0.45,
        'pf': pf > 1.0,
        'mdd': mdd > -POS_SIZE * 5,
        'regime_gap': regime_gap < REGIME_GAP_LIMIT,
    }
    perm_pass = perm_p < 0.05

    passed = all(gates.values()) and perm_pass
    failed_gates = [k for k, v in gates.items() if not v]
    if not perm_pass:
        failed_gates.append('perm_test')

    return {
        'label': label,
        'n_trades': n,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'wr': round(wr, 3),
        'pf': round(pf, 3),
        'mdd': round(mdd, 2),
        'total_pnl': round(total_pnl, 2),
        'regime_gap': round(regime_gap, 3),
        'perm_p': round(perm_p, 4),
        'passed': passed,
        'failed_gates': failed_gates,
        'mean_ret': round(mean_ret * 100, 2),
        'bull_n': len(bull),
        'bear_n': len(bear),
        'avg_hold': round(trades_df['hold_days'].mean(), 1) if n > 0 else 0,
        'reason': 'PASSED' if passed else f"FAILED: {', '.join(failed_gates)}",
    }


# ═══════════════════════════════════════════════════════════════════════
# 6. SOLO COMPONENT BACKTESTS (for confluence alpha measurement)
# ═══════════════════════════════════════════════════════════════════════

def solo_backtest_from_dates(date_set, data, stock_tickers, label):
    """Run a solo backtest where a market-level signal applies to all stocks."""
    signals = {}
    bt_start = pd.Timestamp(BACKTEST_START)
    for d in date_set:
        if d < bt_start:
            continue
        for t in stock_tickers:
            if t in data['close'].columns:
                try:
                    if pd.notna(data['close'].at[d, t]):
                        signals[(d, t)] = True
                except:
                    pass
    trades = run_backtest(signals, data, stock_tickers)
    return validate_strategy(trades, label=label, run_perm=False)


def solo_backtest_from_signals(signal_dict, data, stock_tickers, label):
    """Run solo backtest from a stock-level signal dict."""
    trades = run_backtest(signal_dict, data, stock_tickers)
    return validate_strategy(trades, label=label, run_perm=False)


# ═══════════════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    data = download_data()
    stock_tickers = [t for t in UNIVERSE if t in data['close'].columns]
    print(f"  Stock universe: {len(stock_tickers)} tickers")

    # ── Generate all type signals ──
    print("\n[2] Generating TYPE signals...")

    t0 = time.time()
    type1_rsi35 = gen_type1_dip(data, stock_tickers, rsi_thresh=35, sma_below_pct=0.05)
    print(f"  Type 1 (RSI<35, >5% below SMA): {len(type1_rsi35):,} raw signals [{time.time()-t0:.1f}s]")

    t0 = time.time()
    type1_rsi40 = gen_type1_dip(data, stock_tickers, rsi_thresh=40, sma_below_pct=0.05)
    print(f"  Type 1 (RSI<40, >5% below SMA): {len(type1_rsi40):,} raw signals [{time.time()-t0:.1f}s]")

    t0 = time.time()
    fear_dates = gen_type2_fear(data, vix_mult=1.15, vix_avg_window=60)
    print(f"  Type 2 (VIX fear): {len(fear_dates):,} fear days [{time.time()-t0:.1f}s]")

    t0 = time.time()
    macro_dates = gen_type3_macro(data, yield_drop_thresh=0.05, yield_drop_thresh_big=0.10, lookback=5)
    macro_005 = macro_dates[0.05]
    macro_010 = macro_dates[0.10]
    print(f"  Type 3 (macro yield drop >0.05): {len(macro_005):,} days [{time.time()-t0:.1f}s]")
    print(f"  Type 3 (macro yield drop >0.10): {len(macro_010):,} days")

    t0 = time.time()
    type4_diverg = gen_type4_divergence(data, stock_tickers)
    print(f"  Type 4 (RSI divergence): {len(type4_diverg):,} raw signals [{time.time()-t0:.1f}s]")

    t0 = time.time()
    type5_micro = gen_type5_microstructure(data, stock_tickers, spread_pct=0.60)
    print(f"  Type 5 (microstructure): {len(type5_micro):,} raw signals [{time.time()-t0:.1f}s]")

    # ── Build VIX normalizer for conditional exits ──
    vix = data['close'].get('^VIX')
    if vix is not None:
        vix = vix.dropna()
        vix_60avg = vix.rolling(60).mean()

        def vix_normalized(date, data_):
            """Exit condition: VIX has returned below 1.05 × 60d avg."""
            try:
                v = vix.at[date]
                va = vix_60avg.at[date]
                return v < 1.05 * va
            except:
                return False

        def stock_recovered_5pct_factory(entry_price):
            def checker(date, data_):
                return False  # Handled by profit target
            return checker
    else:
        vix_normalized = None

    # ── Build and test strategies ──
    print("\n[3] Building cross-type confluence strategies...")
    print("=" * 80)

    results = []
    strategy_signals = {}

    # Strategy A: FEAR + DIP (T2 + T1)
    print("\n── Strategy A: FEAR + DIP (Type 2 + Type 1) ──")
    sig_a = build_strategy_a(type1_rsi35, fear_dates)
    strategy_signals['A'] = sig_a
    print(f"  Confluence signals: {len(sig_a):,}")
    trades_a = run_backtest(sig_a, data, stock_tickers, exit_condition_fn=vix_normalized)
    val_a = validate_strategy(trades_a, label='A: FEAR+DIP', run_perm=True,
                              all_signal_entries=sig_a, data=data, stock_tickers=stock_tickers)
    results.append(val_a)
    _print_result(val_a)

    # Strategy B: FEAR + MACRO + DIP (T2 + T3 + T1)
    print("\n── Strategy B: FEAR + MACRO + DIP (Type 2 + Type 3 + Type 1) ──")
    sig_b = build_strategy_b(type1_rsi40, fear_dates, macro_005)
    strategy_signals['B'] = sig_b
    print(f"  Confluence signals: {len(sig_b):,}")
    trades_b = run_backtest(sig_b, data, stock_tickers, exit_condition_fn=vix_normalized)
    val_b = validate_strategy(trades_b, label='B: FEAR+MACRO+DIP', run_perm=True,
                              all_signal_entries=sig_b, data=data, stock_tickers=stock_tickers)
    results.append(val_b)
    _print_result(val_b)

    # Strategy C: FEAR + DIVERGENCE (T2 + T4)
    print("\n── Strategy C: FEAR + DIVERGENCE (Type 2 + Type 4) ──")
    sig_c = build_strategy_c(type4_diverg, fear_dates)
    strategy_signals['C'] = sig_c
    print(f"  Confluence signals: {len(sig_c):,}")
    trades_c = run_backtest(sig_c, data, stock_tickers, exit_condition_fn=vix_normalized)
    val_c = validate_strategy(trades_c, label='C: FEAR+DIVERGENCE', run_perm=True,
                              all_signal_entries=sig_c, data=data, stock_tickers=stock_tickers)
    results.append(val_c)
    _print_result(val_c)

    # Strategy D: MACRO + MICROSTRUCTURE (T3 + T5)
    print("\n── Strategy D: MACRO + MICROSTRUCTURE (Type 3 + Type 5) ──")
    sig_d = build_strategy_d(type5_micro, macro_010)
    strategy_signals['D'] = sig_d
    print(f"  Confluence signals: {len(sig_d):,}")
    trades_d = run_backtest(sig_d, data, stock_tickers, profit_target=0.10, stop_loss=-0.15)
    val_d = validate_strategy(trades_d, label='D: MACRO+MICRO', run_perm=True,
                              all_signal_entries=sig_d, data=data, stock_tickers=stock_tickers)
    results.append(val_d)
    _print_result(val_d)

    # Strategy E: FEAR + MICRO + DIP (T2 + T5 + T1)
    print("\n── Strategy E: FEAR + MICRO + DIP (Type 2 + Type 5 + Type 1) ──")
    sig_e = build_strategy_e(type1_rsi40, fear_dates, type5_micro)
    strategy_signals['E'] = sig_e
    print(f"  Confluence signals: {len(sig_e):,}")
    trades_e = run_backtest(sig_e, data, stock_tickers, exit_condition_fn=vix_normalized)
    val_e = validate_strategy(trades_e, label='E: FEAR+MICRO+DIP', run_perm=True,
                              all_signal_entries=sig_e, data=data, stock_tickers=stock_tickers)
    results.append(val_e)
    _print_result(val_e)

    # Strategy F: ALL TYPES — Perfect Storm
    print("\n── Strategy F: ALL TYPES — Perfect Storm (T2+T3+T4+T5+T1) ──")
    sig_f = build_strategy_f(type1_rsi35, fear_dates, macro_005, type4_diverg, type5_micro)
    strategy_signals['F'] = sig_f
    print(f"  Confluence signals: {len(sig_f):,}")
    trades_f = run_backtest(sig_f, data, stock_tickers, exit_condition_fn=vix_normalized)
    val_f = validate_strategy(trades_f, label='F: PERFECT STORM', run_perm=True,
                              all_signal_entries=sig_f, data=data, stock_tickers=stock_tickers)
    results.append(val_f)
    _print_result(val_f)

    # ═════════════════════════════════════════════════════════════════
    # CONFLUENCE ALPHA: Compare each strategy to its best solo component
    # ═════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("CONFLUENCE ALPHA ANALYSIS")
    print("=" * 80)

    # Solo component backtests
    print("\n  Running solo component backtests...")
    solo_type1_35 = solo_backtest_from_signals(type1_rsi35, data, stock_tickers, 'Solo T1 (RSI<35)')
    solo_type1_40 = solo_backtest_from_signals(type1_rsi40, data, stock_tickers, 'Solo T1 (RSI<40)')
    solo_type2 = solo_backtest_from_dates(fear_dates, data, stock_tickers, 'Solo T2 (VIX fear)')
    solo_type3_005 = solo_backtest_from_dates(macro_005, data, stock_tickers, 'Solo T3 (yield>0.05)')
    solo_type3_010 = solo_backtest_from_dates(macro_010, data, stock_tickers, 'Solo T3 (yield>0.10)')
    solo_type4 = solo_backtest_from_signals(type4_diverg, data, stock_tickers, 'Solo T4 (divergence)')
    solo_type5 = solo_backtest_from_signals(type5_micro, data, stock_tickers, 'Solo T5 (micro)')

    solo_map = {
        'A': [solo_type2, solo_type1_35],
        'B': [solo_type2, solo_type3_005, solo_type1_40],
        'C': [solo_type2, solo_type4],
        'D': [solo_type3_010, solo_type5],
        'E': [solo_type2, solo_type5, solo_type1_40],
        'F': [solo_type2, solo_type3_005, solo_type4, solo_type5, solo_type1_35],
    }

    print(f"\n  {'Strategy':<25} {'Strat Sharpe':>12} {'Best Solo':>12} {'Alpha':>10} {'Alpha?':>8}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")

    for r in results:
        strat_key = r['label'].split(':')[0].strip()
        solos = solo_map.get(strat_key, [])
        best_solo_sharpe = max([s['sharpe'] for s in solos]) if solos else 0
        best_solo_label = ''
        for s in solos:
            if s['sharpe'] == best_solo_sharpe:
                best_solo_label = s['label']
                break
        alpha = r['sharpe'] - best_solo_sharpe
        has_alpha = 'YES' if alpha > 0 else 'no'
        print(f"  {r['label']:<25} {r['sharpe']:>12.3f} {best_solo_sharpe:>12.3f} {alpha:>+10.3f} {has_alpha:>8}")
        r['best_solo_sharpe'] = best_solo_sharpe
        r['best_solo_label'] = best_solo_label
        r['confluence_alpha'] = round(alpha, 3)

    # ═════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("FINAL SUMMARY — CROSS-TYPE CONFLUENCE RESULTS")
    print("=" * 80)

    passed = [r for r in results if r['passed']]
    failed = [r for r in results if not r['passed']]

    print(f"\n  Strategies tested: {len(results)}")
    print(f"  Passed 5-gate: {len(passed)}")
    print(f"  Failed: {len(failed)}")

    if passed:
        print(f"\n  ── PASSED STRATEGIES ──")
        for r in sorted(passed, key=lambda x: x['sharpe'], reverse=True):
            print(f"  {r['label']:<25} Sharpe={r['sharpe']:.3f}  Sortino={r['sortino']:.3f}  "
                  f"WR={r['wr']:.1%}  PF={r['pf']:.2f}  N={r['n_trades']}  "
                  f"PnL=${r['total_pnl']:.0f}  MDD=${r['mdd']:.0f}  "
                  f"RegimeGap={r['regime_gap']:.3f}  Perm_p={r['perm_p']:.4f}  "
                  f"ConfAlpha={r['confluence_alpha']:+.3f}")

    if failed:
        print(f"\n  ── FAILED STRATEGIES ──")
        for r in sorted(failed, key=lambda x: x['sharpe'], reverse=True):
            print(f"  {r['label']:<25} Sharpe={r['sharpe']:.3f}  WR={r['wr']:.1%}  N={r['n_trades']}  "
                  f"Reason: {r['reason']}")

    # Cross-type insight
    alpha_strats = [r for r in results if r.get('confluence_alpha', 0) > 0]
    print(f"\n  Strategies with confluence alpha (Sharpe > best solo): {len(alpha_strats)}/{len(results)}")
    if alpha_strats:
        best = max(alpha_strats, key=lambda x: x['confluence_alpha'])
        print(f"  Best confluence alpha: {best['label']} (+{best['confluence_alpha']:.3f} Sharpe above best solo)")

    # Save results
    import json
    results_file = OUTPUT_DIR / 'cross_type_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_file}")

    return results


def _print_result(r):
    """Pretty-print a single strategy result."""
    status = "PASSED" if r['passed'] else "FAILED"
    print(f"  [{status}] {r['label']}: N={r['n_trades']}  Sharpe={r['sharpe']:.3f}  "
          f"Sortino={r['sortino']:.3f}  WR={r['wr']:.1%}  PF={r['pf']:.2f}  "
          f"MeanRet={r['mean_ret']:.2f}%  MDD=${r['mdd']:.0f}  "
          f"RegimeGap={r['regime_gap']:.3f}  Perm_p={r['perm_p']:.4f}")
    if not r['passed']:
        print(f"    → {r['reason']}")
    if r['n_trades'] > 0:
        print(f"    Bull/Bear: {r['bull_n']}/{r['bear_n']}  AvgHold: {r['avg_hold']:.1f}d")


if __name__ == '__main__':
    main()
