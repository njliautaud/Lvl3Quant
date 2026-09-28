#!/usr/bin/env python3
"""
Regime-Adaptive Strategy Switching Backtest
============================================
Meta-strategy: instead of running ALL validated signals all the time,
dynamically select WHICH signal to use based on current market regime.

Hypothesis: IV-RV Gap works better in high-vol, Bond Yield in rate-cut
environments, Liquidity Signal in low-vol regimes, etc.

6 variants tested:
  A) Vol-Regime Switch
  B) Rate-Regime Switch
  C) Risk-Regime Switch
  D) Combined Regime
  E) Best-of-4 Adaptive (trailing 60d Sharpe selection)
  F) Equal Weight All (CONTROL baseline)

5-gate validation: Sharpe, WR, PF, MDD, regime gap (<0.50), perm test (p<0.05)
"""

import os, sys, json, warnings, time, functools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats as sp_stats
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
N_PERMS = 200
REGIME_GAP_LIMIT = 0.50

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/regime_adaptive')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
    'MA', 'WMT', 'PEP', 'KO', 'TMO', 'ABT', 'BAC', 'XOM', 'CVX', 'CSCO',
]

MACRO_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', '^TNX', 'HYG', 'LQD']

print("=" * 80)
print("REGIME-ADAPTIVE STRATEGY SWITCHING BACKTEST")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════
def download_data():
    import yfinance as yf
    cache_file = OUTPUT_DIR / '_regime_adaptive_cache.pkl'
    if cache_file.exists():
        import pickle
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded cache: {len(data['close'].columns)} stocks, {len(data['close'])} days")
        return data

    print(f"\n[1] Downloading {len(UNIVERSE)} stocks + {len(MACRO_TICKERS)} macro tickers...")
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

    # Clean column names
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


# ═══════════════════════════════════════════════════════════════════════
# 2. INDICATOR HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _rsi(series, period=14):
    """RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _realized_vol(returns, window=20):
    """Annualized realized vol in percentage points."""
    return returns.rolling(window).std() * np.sqrt(252) * 100


# ═══════════════════════════════════════════════════════════════════════
# 3. SIGNAL GENERATORS (per-stock, per-day)
# ═══════════════════════════════════════════════════════════════════════
# Each returns dict of {(date, ticker): True}

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
    """IV-RV Gap: VIX > 20d realized vol (SPY) by 5+ pts AND stock > 5% below 20-SMA AND RSI < 40."""
    spy_close = data['close']['SPY']
    spy_ret = spy_close.pct_change()
    vix = data['close']['^VIX']
    rv_20 = _realized_vol(spy_ret, 20)
    gap = vix - rv_20

    # Precompute per-stock indicators
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
    """Bond Yield: 10Y yield drops > 0.1% over 5 days AND stock > 5% below 20-SMA."""
    if '^TNX' not in data['close'].columns:
        # Fallback: use TLT as proxy
        tlt = data['close']['TLT']
        yield_change = tlt.pct_change(5)
        fire_mask = yield_change > 0.02  # TLT up 2% ~ yields down ~0.1%
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
    """Liquidity Proxy: (High-Low)/Close < 60-day avg AND stock > 5% below 20-SMA AND RSI < 40."""
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
# 4. REGIME CLASSIFICATION (per-day)
# ═══════════════════════════════════════════════════════════════════════

def classify_regimes(data):
    """
    Returns a DataFrame indexed by date with regime columns:
      vol_regime: HIGH_VOL / LOW_VOL / NORMAL_VOL
      rate_regime: RATE_CUT / RATE_RISE
      risk_regime: RISK_OFF / RISK_ON
      spy_trend: bull / bear  (for validation, not strategy selection)
    """
    vix = data['close']['^VIX'].dropna()

    # TNX
    if '^TNX' in data['close'].columns:
        tnx = data['close']['^TNX'].dropna()
    else:
        tnx = None

    # HYG/LQD ratio
    hyg = data['close'].get('HYG')
    lqd = data['close'].get('LQD')

    spy = data['close']['SPY'].dropna()

    all_dates = data['close'].index
    regimes = pd.DataFrame(index=all_dates)

    # ── Vol regime ──
    regimes['vol_regime'] = 'NORMAL_VOL'
    for d in all_dates:
        if d in vix.index and pd.notna(vix.get(d)):
            v = vix.at[d]
            if v > 25:
                regimes.at[d, 'vol_regime'] = 'HIGH_VOL'
            elif v < 18:
                regimes.at[d, 'vol_regime'] = 'LOW_VOL'

    # ── Rate regime ──
    regimes['rate_regime'] = 'NEUTRAL'
    if tnx is not None and len(tnx) > 45:
        tnx_sma20 = tnx.rolling(20).mean()
        tnx_sma40 = tnx.rolling(40).mean()
        for d in all_dates:
            if d in tnx_sma20.index and d in tnx_sma40.index:
                s20 = tnx_sma20.get(d, np.nan)
                s40 = tnx_sma40.get(d, np.nan)
                if pd.notna(s20) and pd.notna(s40):
                    if s20 < s40:
                        regimes.at[d, 'rate_regime'] = 'RATE_CUT'
                    else:
                        regimes.at[d, 'rate_regime'] = 'RATE_RISE'

    # ── Risk regime ──
    regimes['risk_regime'] = 'NEUTRAL'
    if hyg is not None and lqd is not None:
        hyg_s = hyg.dropna()
        lqd_s = lqd.dropna()
        common = hyg_s.index.intersection(lqd_s.index)
        if len(common) > 45:
            ratio = hyg_s.loc[common] / lqd_s.loc[common]
            ratio_sma20 = ratio.rolling(20).mean()
            ratio_sma40 = ratio.rolling(40).mean()
            for d in all_dates:
                if d in ratio_sma20.index and d in ratio_sma40.index:
                    s20 = ratio_sma20.get(d, np.nan)
                    s40 = ratio_sma40.get(d, np.nan)
                    if pd.notna(s20) and pd.notna(s40):
                        if s20 < s40:
                            regimes.at[d, 'risk_regime'] = 'RISK_OFF'
                        else:
                            regimes.at[d, 'risk_regime'] = 'RISK_ON'

    # ── SPY trend for validation regime ──
    spy_sma200 = spy.rolling(200).mean()
    regimes['spy_trend'] = 'unknown'
    for d in all_dates:
        if d in spy.index and d in spy_sma200.index:
            s = spy.get(d, np.nan)
            m = spy_sma200.get(d, np.nan)
            if pd.notna(s) and pd.notna(m):
                regimes.at[d, 'spy_trend'] = 'bull' if s > m else 'bear'

    return regimes


# ═══════════════════════════════════════════════════════════════════════
# 5. ADAPTIVE VARIANT SIGNAL SELECTORS
# ═══════════════════════════════════════════════════════════════════════

def variant_a_vol_switch(all_sigs, regimes, day):
    """HIGH_VOL -> IV-RV Gap. LOW_VOL -> Liquidity. NORMAL -> Base MR."""
    vol = regimes.at[day, 'vol_regime'] if day in regimes.index else 'NORMAL_VOL'
    if vol == 'HIGH_VOL':
        return 'iv_rv_gap'
    elif vol == 'LOW_VOL':
        return 'liquidity'
    else:
        return 'base_mr'


def variant_b_rate_switch(all_sigs, regimes, day):
    """RATE_CUT -> Bond Yield. RATE_RISE -> IV-RV Gap. Otherwise -> Base MR."""
    rate = regimes.at[day, 'rate_regime'] if day in regimes.index else 'NEUTRAL'
    if rate == 'RATE_CUT':
        return 'bond_yield'
    elif rate == 'RATE_RISE':
        return 'iv_rv_gap'
    else:
        return 'base_mr'


def variant_c_risk_switch(all_sigs, regimes, day):
    """RISK_OFF -> IV-RV Gap. RISK_ON -> Liquidity. Otherwise -> Base MR."""
    risk = regimes.at[day, 'risk_regime'] if day in regimes.index else 'NEUTRAL'
    if risk == 'RISK_OFF':
        return 'iv_rv_gap'
    elif risk == 'RISK_ON':
        return 'liquidity'
    else:
        return 'base_mr'


def variant_d_combined(all_sigs, regimes, day):
    """
    HIGH_VOL + RATE_CUT -> both Bond Yield AND IV-RV must fire (returns tuple).
    HIGH_VOL + RATE_RISE -> IV-RV Gap only.
    LOW_VOL -> Liquidity only.
    Everything else -> Base MR.
    """
    vol = regimes.at[day, 'vol_regime'] if day in regimes.index else 'NORMAL_VOL'
    rate = regimes.at[day, 'rate_regime'] if day in regimes.index else 'NEUTRAL'
    if vol == 'HIGH_VOL' and rate == 'RATE_CUT':
        return ('bond_yield', 'iv_rv_gap')  # Both must fire
    elif vol == 'HIGH_VOL':
        return 'iv_rv_gap'
    elif vol == 'LOW_VOL':
        return 'liquidity'
    else:
        return 'base_mr'


# ═══════════════════════════════════════════════════════════════════════
# 6. BUILD ADAPTIVE SIGNAL SETS
# ═══════════════════════════════════════════════════════════════════════

def build_adaptive_signals(all_sigs, regimes, variant_func, label):
    """
    For each day in backtest period, determine which signal(s) are allowed,
    then collect matching entries from the signal dicts.
    """
    bt_start = pd.Timestamp(BACKTEST_START)
    result = {}

    # Collect all unique days from all signals
    all_days = set()
    for sig_name, sig_dict in all_sigs.items():
        for (d, t) in sig_dict:
            if d >= bt_start:
                all_days.add(d)

    for day in sorted(all_days):
        allowed = variant_func(all_sigs, regimes, day)

        if isinstance(allowed, tuple):
            # Combined: ALL must fire for a given ticker
            for t in UNIVERSE:
                if all((day, t) in all_sigs.get(sig, {}) for sig in allowed):
                    result[(day, t)] = True
        else:
            # Single signal allowed
            sig_dict = all_sigs.get(allowed, {})
            for (d, t) in sig_dict:
                if d == day:
                    result[(day, t)] = True

    return result


def build_best_of_4_adaptive(all_sigs, data, regimes, stock_tickers):
    """
    Variant E: On each day, pick the signal with best trailing 60-day Sharpe
    in the current regime.
    """
    bt_start = pd.Timestamp(BACKTEST_START)
    close = data['close']
    signal_names = ['base_mr', 'iv_rv_gap', 'bond_yield', 'liquidity']

    # Precompute per-signal trade outcomes for all historical entries
    # We need to know per-signal PnL for rolling performance tracking
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
            # Simple 21-day return for performance tracking
            future = close.index[close.index > d]
            if len(future) < 1:
                continue
            hold_end = min(HOLD_DAYS, len(future))
            try:
                exit_price = close.at[future[hold_end - 1], t]
            except:
                continue
            if pd.isna(exit_price):
                continue
            ret = (exit_price - entry_price) / entry_price - SPREAD_COST_PCT
            trades.append({'date': d, 'ticker': t, 'return': ret})
        per_signal_trades[sig_name] = pd.DataFrame(trades) if trades else pd.DataFrame(columns=['date', 'ticker', 'return'])

    # For each day, compute trailing 60-day Sharpe per signal
    all_days = set()
    for sig_name in signal_names:
        for (d, t) in all_sigs.get(sig_name, {}):
            if d >= bt_start:
                all_days.add(d)

    result = {}
    for day in sorted(all_days):
        lookback_start = day - pd.Timedelta(days=60)

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

        # Use the best signal for this day
        sig_dict = all_sigs.get(best_sig, {})
        for (d, t) in sig_dict:
            if d == day:
                result[(day, t)] = True

    return result


def build_equal_weight_all(all_sigs):
    """Variant F (CONTROL): Any signal fires -> trade."""
    bt_start = pd.Timestamp(BACKTEST_START)
    result = {}
    for sig_name, sig_dict in all_sigs.items():
        for (d, t) in sig_dict:
            if d >= bt_start:
                result[(d, t)] = True
    return result


# ═══════════════════════════════════════════════════════════════════════
# 7. BACKTESTER
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(signal_entries, data, stock_tickers, label=''):
    """Run backtest on a set of signal entries {(date, ticker): True}."""
    close = data['close']
    spy_close = close['SPY']
    spy_sma200 = spy_close.rolling(200).mean()

    bt_start = pd.Timestamp(BACKTEST_START)
    entries = [(d, t) for (d, t) in signal_entries if d >= bt_start]
    entries.sort(key=lambda x: x[0])

    if not entries:
        return None

    trades = []
    open_positions = []

    for entry_date, ticker in entries:
        # Check max concurrent
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

        for j, fdate in enumerate(future_dates[:HOLD_DAYS]):
            try:
                price = close.at[fdate, ticker]
            except:
                continue
            if pd.isna(price):
                continue
            ret = (price - entry_price) / entry_price
            if ret >= PROFIT_TARGET:
                exit_price = price
                exit_date = fdate
                exit_reason = 'profit_target'
                break
            elif ret <= STOP_LOSS:
                exit_price = price
                exit_date = fdate
                exit_reason = 'stop_loss'
                break

        if exit_price is None:
            hold_end = min(HOLD_DAYS, len(future_dates))
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


# ═══════════════════════════════════════════════════════════════════════
# 8. VALIDATION (5-gate + perm test)
# ═══════════════════════════════════════════════════════════════════════

def compute_sortino(rets, trades_per_year):
    """Sortino ratio."""
    mean_ret = np.mean(rets)
    downside = rets[rets < 0]
    if len(downside) < 2:
        return 99.0
    down_std = np.std(downside, ddof=1)
    if down_std <= 0:
        return 99.0
    return (mean_ret / down_std) * np.sqrt(trades_per_year)


def validate_strategy(trades_df, label='', run_perm=True, all_signal_entries=None,
                      data=None, stock_tickers=None):
    """5-gate validation: Sharpe, WR, PF, MDD, regime gap, perm test."""
    if trades_df is None or len(trades_df) < 10:
        return {
            'label': label, 'n_trades': 0 if trades_df is None else len(trades_df),
            'sharpe': 0, 'sortino': 0, 'wr': 0, 'pf': 0, 'mdd': 0,
            'regime_gap': 1.0, 'perm_p': 1.0, 'passed': False,
            'reason': 'insufficient trades',
        }

    rets = trades_df['return'].values
    n = len(rets)

    # Trades per year
    date_range_years = max(1, (trades_df['entry_date'].max() - trades_df['entry_date'].min()).days / 365.25)
    trades_per_year = n / date_range_years
    if trades_per_year < 1:
        trades_per_year = 1

    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1.0
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0
    sortino = compute_sortino(rets, trades_per_year)

    wr = np.mean(rets > 0)

    gross_profit = np.sum(rets[rets > 0])
    gross_loss = np.abs(np.sum(rets[rets < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

    # Max drawdown
    cum_pnl = np.cumsum(trades_df['pnl'].values)
    peak = np.maximum.accumulate(cum_pnl)
    dd = cum_pnl - peak
    mdd = np.min(dd) if len(dd) > 0 else 0

    # Regime gap (bull vs bear)
    bull_trades = trades_df[trades_df['regime'] == 'bull']
    bear_trades = trades_df[trades_df['regime'] == 'bear']
    if len(bull_trades) >= 3 and len(bear_trades) >= 3:
        bull_sharpe_raw = np.mean(bull_trades['return']) / max(np.std(bull_trades['return'], ddof=1), 1e-6)
        bear_sharpe_raw = np.mean(bear_trades['return']) / max(np.std(bear_trades['return'], ddof=1), 1e-6)
        max_abs = max(abs(bull_sharpe_raw), abs(bear_sharpe_raw), 1e-6)
        regime_gap = abs(bull_sharpe_raw - bear_sharpe_raw) / max_abs
    else:
        regime_gap = 0.0

    # Permutation test (random entry timing)
    perm_p = 1.0
    if run_perm and n >= 10 and all_signal_entries is not None and data is not None and stock_tickers is not None:
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
            if rand_trades is not None and len(rand_trades) >= 5:
                r_rets = rand_trades['return'].values
                r_tpy = len(r_rets) / max(1, (rand_trades['entry_date'].max() - rand_trades['entry_date'].min()).days / 365.25)
                if r_tpy < 1:
                    r_tpy = 1
                r_mean = np.mean(r_rets)
                r_std = np.std(r_rets, ddof=1)
                r_sharpe = (r_mean / r_std) * np.sqrt(r_tpy) if r_std > 0 else 0
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
        'regime_gap': round(regime_gap, 3),
        'perm_p': round(perm_p, 4),
        'passed': passed,
        'failed_gates': failed_gates,
        'mean_ret': round(mean_ret * 100, 2),
        'bull_n': len(bull_trades),
        'bear_n': len(bear_trades),
        'avg_hold': round(trades_df['hold_days'].mean(), 1),
        'reason': 'PASSED' if passed else f"FAILED: {', '.join(failed_gates)}",
    }


# ═══════════════════════════════════════════════════════════════════════
# 9. MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════

def main():
    data = download_data()
    stock_tickers = [t for t in UNIVERSE if t in data['close'].columns]
    print(f"  Stock universe: {len(stock_tickers)} tickers")

    # ── Classify regimes ──
    print("\n[2] Classifying market regimes...")
    regimes = classify_regimes(data)
    bt_regimes = regimes.loc[regimes.index >= BACKTEST_START]
    for col in ['vol_regime', 'rate_regime', 'risk_regime']:
        counts = bt_regimes[col].value_counts()
        print(f"  {col}: {dict(counts)}")

    # ── Generate all 4 core signals ──
    print("\n[3] Generating core signals...")
    signal_generators = {
        'base_mr': signal_base_mr,
        'iv_rv_gap': signal_iv_rv_gap,
        'bond_yield': signal_bond_yield,
        'liquidity': signal_liquidity,
    }

    all_sigs = {}
    for key, gen_func in signal_generators.items():
        t0 = time.time()
        sigs = gen_func(data, stock_tickers)
        elapsed = time.time() - t0
        # Filter to backtest period for counting
        bt_count = sum(1 for (d, t) in sigs if d >= pd.Timestamp(BACKTEST_START))
        print(f"  {key:15s}: {bt_count:5d} entries in backtest window ({elapsed:.1f}s)")
        all_sigs[key] = sigs

    # ── Run solo backtests (baselines) ──
    print("\n[4] Running SOLO backtests (baselines)...")
    solo_results = {}
    for key, sigs in all_sigs.items():
        trades_df = run_backtest(sigs, data, stock_tickers, label=key)
        result = validate_strategy(trades_df, label=f"SOLO: {key}", run_perm=True,
                                   all_signal_entries=sigs, data=data, stock_tickers=stock_tickers)
        solo_results[key] = result
        status = "PASS" if result['passed'] else "FAIL"
        print(f"  {key:15s} | n={result['n_trades']:4d} | Sharpe={result['sharpe']:6.3f} | "
              f"Sortino={result['sortino']:6.3f} | WR={result['wr']:.3f} | PF={result['pf']:.3f} | "
              f"perm_p={result['perm_p']:.4f} | {status}")

    # ── Build 6 adaptive variants ──
    print("\n[5] Building adaptive variant signal sets...")

    variants = {}

    # A) Vol-Regime Switch
    variants['A_vol_switch'] = build_adaptive_signals(
        all_sigs, regimes, variant_a_vol_switch, 'Vol-Regime Switch')

    # B) Rate-Regime Switch
    variants['B_rate_switch'] = build_adaptive_signals(
        all_sigs, regimes, variant_b_rate_switch, 'Rate-Regime Switch')

    # C) Risk-Regime Switch
    variants['C_risk_switch'] = build_adaptive_signals(
        all_sigs, regimes, variant_c_risk_switch, 'Risk-Regime Switch')

    # D) Combined Regime
    variants['D_combined'] = build_adaptive_signals(
        all_sigs, regimes, variant_d_combined, 'Combined Regime')

    # E) Best-of-4 Adaptive
    variants['E_best_of_4'] = build_best_of_4_adaptive(
        all_sigs, data, regimes, stock_tickers)

    # F) Equal Weight All (CONTROL)
    variants['F_equal_weight'] = build_equal_weight_all(all_sigs)

    for vname, vsigs in variants.items():
        bt_count = sum(1 for (d, t) in vsigs if d >= pd.Timestamp(BACKTEST_START))
        print(f"  {vname:20s}: {bt_count:5d} entries")

    # ── Run variant backtests ──
    print(f"\n[6] Running {len(variants)} adaptive variant backtests (with perm tests)...")
    variant_results = {}

    for vname, vsigs in variants.items():
        t0 = time.time()
        trades_df = run_backtest(vsigs, data, stock_tickers, label=vname)
        result = validate_strategy(trades_df, label=vname, run_perm=True,
                                   all_signal_entries=vsigs, data=data, stock_tickers=stock_tickers)
        elapsed = time.time() - t0
        variant_results[vname] = result
        status = "PASS" if result['passed'] else "FAIL"
        print(f"  {vname:20s} | n={result['n_trades']:4d} | Sharpe={result['sharpe']:6.3f} | "
              f"Sortino={result['sortino']:6.3f} | WR={result['wr']:.3f} | PF={result['pf']:.3f} | "
              f"gap={result['regime_gap']:.3f} | perm_p={result['perm_p']:.4f} | {status} ({elapsed:.0f}s)")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("RESULTS SUMMARY")
    print("=" * 120)

    # Solo baselines
    print("\n── SOLO SIGNAL BASELINES ──")
    print(f"{'Signal':15s} | {'N':>5s} | {'Sharpe':>7s} | {'Sortino':>7s} | {'WR':>5s} | {'PF':>6s} | "
          f"{'MDD':>8s} | {'RegGap':>6s} | {'Perm p':>7s} | {'Status':>6s}")
    print("-" * 105)
    for key, r in solo_results.items():
        status = "PASS" if r['passed'] else "FAIL"
        print(f"{key:15s} | {r['n_trades']:5d} | {r['sharpe']:7.3f} | {r['sortino']:7.3f} | "
              f"{r['wr']:.3f} | {r['pf']:6.3f} | {r['mdd']:8.2f} | {r['regime_gap']:6.3f} | "
              f"{r['perm_p']:7.4f} | {status:>6s}")

    # Adaptive variants
    baseline_sharpe = variant_results.get('F_equal_weight', {}).get('sharpe', 0)

    print(f"\n── ADAPTIVE VARIANTS (baseline F Sharpe = {baseline_sharpe:.3f}) ──")
    print(f"{'Variant':20s} | {'N':>5s} | {'Sharpe':>7s} | {'Sortino':>7s} | {'WR':>5s} | {'PF':>6s} | "
          f"{'MDD':>8s} | {'RegGap':>6s} | {'Perm p':>7s} | {'Alpha':>7s} | {'Status':>6s}")
    print("-" * 130)

    for vname in ['A_vol_switch', 'B_rate_switch', 'C_risk_switch',
                  'D_combined', 'E_best_of_4', 'F_equal_weight']:
        r = variant_results[vname]
        status = "PASS" if r['passed'] else "FAIL"
        adaptive_alpha = r['sharpe'] - baseline_sharpe if vname != 'F_equal_weight' else 0.0
        alpha_str = f"{adaptive_alpha:+7.3f}" if vname != 'F_equal_weight' else "  BASE"
        print(f"{vname:20s} | {r['n_trades']:5d} | {r['sharpe']:7.3f} | {r['sortino']:7.3f} | "
              f"{r['wr']:.3f} | {r['pf']:6.3f} | {r['mdd']:8.2f} | {r['regime_gap']:6.3f} | "
              f"{r['perm_p']:7.4f} | {alpha_str} | {status:>6s}")

    # Best variant
    best_variant = max(
        [(vname, r) for vname, r in variant_results.items() if vname != 'F_equal_weight'],
        key=lambda x: x[1]['sharpe']
    )
    best_name, best_r = best_variant
    adaptive_alpha = best_r['sharpe'] - baseline_sharpe

    print(f"\n── KEY FINDING ──")
    print(f"  Best adaptive variant: {best_name}")
    print(f"  Sharpe: {best_r['sharpe']:.3f} (vs baseline {baseline_sharpe:.3f})")
    print(f"  Adaptive alpha: {adaptive_alpha:+.3f}")
    print(f"  Sortino: {best_r['sortino']:.3f}, WR: {best_r['wr']:.1%}, PF: {best_r['pf']:.2f}")
    print(f"  Regime gap: {best_r['regime_gap']:.3f}, Perm p: {best_r['perm_p']:.4f}")
    if best_r['passed'] and adaptive_alpha > 0:
        print(f"  CONCLUSION: Regime-adaptive switching ADDS value over equal-weight baseline.")
    elif best_r['passed']:
        print(f"  CONCLUSION: Best variant passes 5-gate but no alpha over equal-weight baseline.")
    else:
        print(f"  CONCLUSION: No adaptive variant passes all 5 gates. Failed: {best_r.get('failed_gates', [])}")

    # Winners
    winners = [vname for vname, r in variant_results.items()
               if r['passed'] and vname != 'F_equal_weight']
    if winners:
        print(f"\n  Passing variants: {', '.join(winners)}")
    else:
        print(f"\n  No adaptive variant passed all 5 gates.")

    # ── Save results ──
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

    results_data = {
        'run_date': datetime.now().isoformat(),
        'config': {
            'universe_size': len(stock_tickers),
            'backtest_start': BACKTEST_START,
            'backtest_end': END_DATE,
            'pos_size': POS_SIZE,
            'max_concurrent': MAX_CONCURRENT,
            'hold_days': HOLD_DAYS,
            'profit_target': PROFIT_TARGET,
            'stop_loss': STOP_LOSS,
            'n_perms': N_PERMS,
            'regime_gap_limit': REGIME_GAP_LIMIT,
        },
        'solo_baselines': {k: {kk: _serialize(vv) for kk, vv in v.items()} for k, v in solo_results.items()},
        'variant_results': {k: {kk: _serialize(vv) for kk, vv in v.items()} for k, v in variant_results.items()},
        'baseline_sharpe': float(baseline_sharpe),
        'best_variant': best_name,
        'adaptive_alpha': float(adaptive_alpha),
        'winners': winners,
        'regime_distribution': {
            col: dict(bt_regimes[col].value_counts()) for col in ['vol_regime', 'rate_regime', 'risk_regime']
        },
    }

    results_file = OUTPUT_DIR / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=_serialize)
    print(f"\nResults saved to {results_file}")

    print("\n" + "=" * 80)
    print("DONE — Regime-Adaptive Strategy Switching Backtest Complete")
    print("=" * 80)

    return results_data


if __name__ == '__main__':
    main()
