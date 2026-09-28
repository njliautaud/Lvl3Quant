#!/usr/bin/env python3
"""
Cross-Signal Confluence Backtest Framework (HC #772)
=====================================================
Tests validated strategy PAIRS to find confluence alpha.
Dead solo signals may become strong when paired.

Priority pairs:
  1. IV-RV Gap + RSI Divergence
  2. IV-RV Gap + Liquidity Signal
  3. Bond Yield + RSI Divergence
  4. IV-RV Gap + Bond Yield
  5. RSI Divergence + Liquidity Signal
  6. IV-RV Gap + Consecutive Dip
  7-9. IV-RV Gap + dead-solo filters (VIX term structure, vol contraction, gap reversal)

5-gate validation: Sharpe, WR, PF, MDD, regime gap (<0.50), perm test (p<0.05)
Confluence alpha: pair Sharpe must exceed BOTH solo Sharpes.
"""

import os, sys, json, warnings, time, functools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

# ─── Config ───
START_DATE = '2019-01-01'  # extra lookback for indicators
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

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/cross_signal_confluence')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
    'MA', 'WMT', 'PEP', 'KO', 'TMO', 'ABT', 'BAC', 'XOM', 'CVX', 'CSCO',
]

MACRO_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', '^TNX', 'HYG', 'LQD']

print("=" * 80)
print("CROSS-SIGNAL CONFLUENCE BACKTEST (HC #772)")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════
def download_data():
    import yfinance as yf
    cache_file = OUTPUT_DIR / '_confluence_cache.pkl'
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
# 2. SIGNAL GENERATORS
# ═══════════════════════════════════════════════════════════════════════

def _rsi(series, period=14):
    """RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _realized_vol(returns, window=21):
    """Annualized realized vol."""
    return returns.rolling(window).std() * np.sqrt(252) * 100

def signal_iv_rv_gap(data, stock_tickers):
    """IV-RV Gap: buy when VIX > realized vol of SPY by 5+ points.
    Maps to all quality stocks on the day."""
    spy_close = data['close']['SPY']
    spy_ret = spy_close.pct_change()
    vix = data['close']['^VIX']
    rv_21 = _realized_vol(spy_ret, 21)
    gap = vix - rv_21
    # Signal fires for ALL stocks when gap > 5
    fire_days = gap[gap > 5].index
    signals = {}
    for day in fire_days:
        for t in stock_tickers:
            if t in data['close'].columns and pd.notna(data['close'].loc[day, t] if day in data['close'].index else np.nan):
                signals[(day, t)] = True
    return signals, 'IV-RV Gap'

def signal_rsi_divergence(data, stock_tickers):
    """RSI Divergence: price makes lower low but RSI makes higher low.
    Look back 10 days for swing lows."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        rsi = _rsi(close, 14)
        for i in range(20, len(close)):
            # Check if today is a local low area (close < 10d min + small margin)
            window_close = close.iloc[i-10:i+1]
            window_rsi = rsi.iloc[i-10:i+1]
            if len(window_close) < 11 or window_rsi.isna().any():
                continue
            # Price: current close < close 10 days ago (lower low)
            if close.iloc[i] < window_close.iloc[0]:
                # RSI: current RSI > RSI 10 days ago (higher low = divergence)
                if rsi.iloc[i] > window_rsi.iloc[0]:
                    # Also require RSI < 40 (oversold territory)
                    if rsi.iloc[i] < 40:
                        day = close.index[i]
                        signals[(day, t)] = True
    return signals, 'RSI Divergence'

def signal_bond_yield(data, stock_tickers):
    """Bond Yield Signal: buy when 10Y yield drops >0.1% in 5 days (flight-to-safety ending)."""
    if '^TNX' not in data['close'].columns:
        print("  WARNING: ^TNX not available, using TLT proxy")
        tlt = data['close']['TLT']
        yield_change = tlt.pct_change(5)  # TLT up = yields down
        fire_days = yield_change[yield_change > 0.02].index  # TLT up 2% in 5d ~ yields down ~0.1%
    else:
        tnx = data['close']['^TNX']
        yield_change = tnx.diff(5)
        fire_days = yield_change[yield_change < -0.10].index  # yield dropped > 0.1%

    signals = {}
    for day in fire_days:
        for t in stock_tickers:
            if t in data['close'].columns:
                try:
                    if pd.notna(data['close'].at[day, t]):
                        signals[(day, t)] = True
                except:
                    pass
    return signals, 'Bond Yield'

def signal_liquidity(data, stock_tickers):
    """Liquidity Signal: buy when HL spread narrows below 60d average."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns or t not in data['high'].columns:
            continue
        high = data['high'][t].dropna()
        low = data['low'][t].dropna()
        close = data['close'][t].dropna()
        # Align
        idx = high.index.intersection(low.index).intersection(close.index)
        if len(idx) < 65:
            continue
        high, low, close = high.loc[idx], low.loc[idx], close.loc[idx]
        hl_spread = (high - low) / close
        avg_60 = hl_spread.rolling(60).mean()
        # Signal: spread < avg (tighter = more liquid = smart money)
        narrow = hl_spread < avg_60 * 0.85  # 15% narrower than average
        # Also require price is in a dip (RSI < 40)
        rsi = _rsi(close, 14)
        for i in range(60, len(idx)):
            if narrow.iloc[i] and rsi.iloc[i] < 40:
                signals[(idx[i], t)] = True
    return signals, 'Liquidity Signal'

def signal_consecutive_dip(data, stock_tickers):
    """Consecutive Dip: 3+ red days with each day's loss bigger than the prior."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        ret = close.pct_change()
        for i in range(3, len(close)):
            # Check 3 consecutive red days with deepening losses
            r1, r2, r3 = ret.iloc[i-2], ret.iloc[i-1], ret.iloc[i]
            if r1 < 0 and r2 < 0 and r3 < 0:
                if r2 < r1 and r3 < r2:  # Each day worse
                    signals[(close.index[i], t)] = True
    return signals, 'Consecutive Dip'

def signal_base_mr(data, stock_tickers):
    """Base Mean Reversion: RSI < 30 and price >7% below 50d high."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        rsi = _rsi(close, 14)
        high_50 = close.rolling(50).max()
        drawdown = (close - high_50) / high_50
        for i in range(50, len(close)):
            if rsi.iloc[i] < 30 and drawdown.iloc[i] < -0.07:
                signals[(close.index[i], t)] = True
    return signals, 'Base MR'

# ─── Dead solo signals (as confluence filters) ───

def signal_vix_term_structure(data, stock_tickers):
    """VIX Term Structure: VIX > VIX3M (backwardation = fear)."""
    vix = data['close'].get('^VIX')
    vix3m = data['close'].get('^VIX3M')
    if vix is None or vix3m is None:
        return {}, 'VIX Term Structure'
    ratio = vix / vix3m
    fire_days = ratio[ratio > 1.0].index
    signals = {}
    for day in fire_days:
        for t in stock_tickers:
            if t in data['close'].columns:
                try:
                    if pd.notna(data['close'].at[day, t]):
                        signals[(day, t)] = True
                except:
                    pass
    return signals, 'VIX Term Structure'

def signal_vol_contraction(data, stock_tickers):
    """Vol Contraction / Keltner Squeeze: Bollinger inside Keltner = coiled spring."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        high = data['high'].get(t, close).dropna()
        low = data['low'].get(t, close).dropna()
        idx = close.index.intersection(high.index).intersection(low.index)
        if len(idx) < 25:
            continue
        close, high, low = close.loc[idx], high.loc[idx], low.loc[idx]
        # Bollinger width
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_width = (2 * std20) / sma20
        # ATR for Keltner
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr20 = tr.rolling(20).mean()
        kelt_width = (2 * 1.5 * atr20) / sma20
        # Squeeze: BB inside Keltner
        squeeze = bb_width < kelt_width
        # Also need RSI < 40 (dip context)
        rsi = _rsi(close, 14)
        for i in range(20, len(idx)):
            if squeeze.iloc[i] and rsi.iloc[i] < 40:
                signals[(idx[i], t)] = True
    return signals, 'Vol Contraction'

def signal_gap_reversal(data, stock_tickers):
    """Gap Reversal: gap down >2% that starts recovering (close > open proxy)."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        high = data['high'].get(t, close).dropna()
        low = data['low'].get(t, close).dropna()
        idx = close.index.intersection(high.index).intersection(low.index)
        if len(idx) < 5:
            continue
        close, high, low = close.loc[idx], high.loc[idx], low.loc[idx]
        for i in range(1, len(idx)):
            gap = (low.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
            if gap < -0.02:  # gapped down >2%
                # Reversal: close in upper half of day's range
                day_range = high.iloc[i] - low.iloc[i]
                if day_range > 0:
                    close_position = (close.iloc[i] - low.iloc[i]) / day_range
                    if close_position > 0.6:  # closed in upper 40% of range
                        signals[(idx[i], t)] = True
    return signals, 'Gap Reversal'

# ═══════════════════════════════════════════════════════════════════════
# 3. BACKTESTER
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(signal_entries, data, stock_tickers, label=''):
    """Run backtest on a set of signal entries {(date, ticker): True}."""
    close = data['close']
    spy_close = close['SPY']

    # Filter to backtest period
    bt_start = pd.Timestamp(BACKTEST_START)
    entries = [(d, t) for (d, t) in signal_entries if d >= bt_start]
    entries.sort(key=lambda x: x[0])

    if not entries:
        return None

    trades = []
    open_positions = []  # list of (entry_date, ticker, entry_price, exit_date)

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

        # Find exit
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
            # Exit at end of hold period
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

        # SPY regime on entry
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

# ═══════════════════════════════════════════════════════════════════════
# 4. VALIDATION (5-gate)
# ═══════════════════════════════════════════════════════════════════════

def validate_strategy(trades_df, label='', run_perm=True, all_signal_entries=None, data=None, stock_tickers=None):
    """5-gate validation: Sharpe, WR, PF, MDD, regime gap, perm test."""
    if trades_df is None or len(trades_df) < 10:
        return {
            'label': label, 'n_trades': 0 if trades_df is None else len(trades_df),
            'sharpe': 0, 'wr': 0, 'pf': 0, 'mdd': 0, 'regime_gap': 1.0,
            'perm_p': 1.0, 'passed': False, 'reason': 'insufficient trades'
        }

    rets = trades_df['return'].values
    n = len(rets)

    # Sharpe (annualized, assuming ~12 trades/year avg)
    trades_per_year = n / max(1, (trades_df['entry_date'].max() - trades_df['entry_date'].min()).days / 365.25)
    if trades_per_year < 1:
        trades_per_year = 1
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1.0
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Win rate
    wr = np.mean(rets > 0)

    # Profit factor
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = np.abs(np.sum(rets[rets < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

    # Max drawdown (cumulative PnL based)
    cum_pnl = np.cumsum(trades_df['pnl'].values)
    peak = np.maximum.accumulate(cum_pnl)
    dd = cum_pnl - peak
    mdd = np.min(dd) if len(dd) > 0 else 0

    # Regime gap
    bull_trades = trades_df[trades_df['regime'] == 'bull']
    bear_trades = trades_df[trades_df['regime'] == 'bear']
    if len(bull_trades) >= 3 and len(bear_trades) >= 3:
        bull_sharpe_raw = np.mean(bull_trades['return']) / max(np.std(bull_trades['return'], ddof=1), 1e-6)
        bear_sharpe_raw = np.mean(bear_trades['return']) / max(np.std(bear_trades['return'], ddof=1), 1e-6)
        max_abs = max(abs(bull_sharpe_raw), abs(bear_sharpe_raw), 1e-6)
        regime_gap = abs(bull_sharpe_raw - bear_sharpe_raw) / max_abs
    else:
        regime_gap = 0.0  # Can't assess, pass by default

    # Permutation test — RANDOM ENTRY TIMING
    # Shuffling returns doesn't test timing alpha (all quality stocks went up 2020-2026).
    # Instead: generate random entry dates, run same backtest logic, compare Sharpes.
    perm_p = 1.0
    if run_perm and n >= 10 and all_signal_entries is not None and data is not None and stock_tickers is not None:
        actual_sharpe = sharpe
        perm_sharpes = []
        # Get valid trading dates in backtest window
        bt_start = pd.Timestamp(BACKTEST_START)
        valid_dates = data['close'].index[data['close'].index >= bt_start]
        # Leave room for hold period
        valid_dates = valid_dates[:-HOLD_DAYS-5]

        for perm_i in range(N_PERMS):
            # Generate random entries: same number of entries, random dates/tickers
            n_raw = len(all_signal_entries)
            rand_dates = np.random.choice(valid_dates, size=min(n_raw, len(valid_dates)), replace=True)
            rand_tickers = np.random.choice(stock_tickers, size=min(n_raw, len(valid_dates)), replace=True)
            rand_signals = {(d, t): True for d, t in zip(rand_dates, rand_tickers)}

            rand_trades = run_backtest(rand_signals, data, stock_tickers)
            if rand_trades is not None and len(rand_trades) >= 5:
                r_rets = rand_trades['return'].values
                r_tpy = len(r_rets) / max(1, (rand_trades['entry_date'].max() - rand_trades['entry_date'].min()).days / 365.25)
                if r_tpy < 1: r_tpy = 1
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
        'mdd': mdd > -POS_SIZE * 5,  # Max 5 positions worth of drawdown
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
# 5. CONFLUENCE: INTERSECT TWO SIGNAL SETS
# ═══════════════════════════════════════════════════════════════════════

def confluence_signals(sig_a, sig_b, window=1):
    """Find entries where BOTH signals fire on the same day (or within `window` days).
    Returns intersection set of (date, ticker) tuples."""
    if window == 0:
        # Exact same day
        return {k: True for k in sig_a if k in sig_b}

    # Build date index for sig_b per ticker
    from collections import defaultdict
    b_by_ticker = defaultdict(set)
    for (d, t) in sig_b:
        b_by_ticker[t].add(d)

    result = {}
    for (d, t) in sig_a:
        if t not in b_by_ticker:
            continue
        # Check if sig_b fired within window days
        for offset in range(-window, window + 1):
            check_d = d + pd.Timedelta(days=offset)
            if check_d in b_by_ticker[t]:
                result[(d, t)] = True
                break
    return result

# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════

def main():
    data = download_data()
    stock_tickers = [t for t in UNIVERSE if t in data['close'].columns]
    print(f"  Stock universe: {len(stock_tickers)} tickers")

    # ── Generate all signals ──
    print("\n[2] Generating signals...")
    signal_generators = {
        'iv_rv_gap': signal_iv_rv_gap,
        'rsi_divergence': signal_rsi_divergence,
        'bond_yield': signal_bond_yield,
        'liquidity': signal_liquidity,
        'consecutive_dip': signal_consecutive_dip,
        'base_mr': signal_base_mr,
        'vix_term_structure': signal_vix_term_structure,
        'vol_contraction': signal_vol_contraction,
        'gap_reversal': signal_gap_reversal,
    }

    all_signals = {}
    for key, gen_func in signal_generators.items():
        t0 = time.time()
        sigs, name = gen_func(data, stock_tickers)
        elapsed = time.time() - t0
        print(f"  {name}: {len(sigs)} raw entries ({elapsed:.1f}s)")
        all_signals[key] = sigs

    # ── Run solo backtests first ──
    print("\n[3] Running SOLO backtests (baselines)...")
    solo_results = {}
    solo_trades = {}
    for key, sigs in all_signals.items():
        trades_df = run_backtest(sigs, data, stock_tickers, label=key)
        result = validate_strategy(trades_df, label=f"SOLO: {key}", run_perm=True,
                                   all_signal_entries=sigs, data=data, stock_tickers=stock_tickers)
        solo_results[key] = result
        solo_trades[key] = trades_df
        status = "PASS" if result['passed'] else "FAIL"
        print(f"  {key:25s} | n={result['n_trades']:4d} | Sharpe={result['sharpe']:6.3f} | "
              f"WR={result['wr']:.3f} | PF={result['pf']:.3f} | perm_p={result['perm_p']:.4f} | {status}")

    # ── Define confluence pairs ──
    confluence_pairs = [
        # Priority validated pairs
        ('iv_rv_gap', 'rsi_divergence', 'IV-RV + RSI Div'),
        ('iv_rv_gap', 'liquidity', 'IV-RV + Liquidity'),
        ('bond_yield', 'rsi_divergence', 'Bond Yield + RSI Div'),
        ('iv_rv_gap', 'bond_yield', 'IV-RV + Bond Yield'),
        ('rsi_divergence', 'liquidity', 'RSI Div + Liquidity'),
        ('iv_rv_gap', 'consecutive_dip', 'IV-RV + Consec Dip'),
        # Dead solo as filters on IV-RV Gap
        ('iv_rv_gap', 'vix_term_structure', 'IV-RV + VIX TermStr'),
        ('iv_rv_gap', 'vol_contraction', 'IV-RV + Vol Contract'),
        ('iv_rv_gap', 'gap_reversal', 'IV-RV + Gap Reversal'),
        # Bonus: base MR combinations
        ('base_mr', 'iv_rv_gap', 'Base MR + IV-RV'),
        ('base_mr', 'rsi_divergence', 'Base MR + RSI Div'),
        ('base_mr', 'liquidity', 'Base MR + Liquidity'),
    ]

    # ── Run confluence backtests ──
    print(f"\n[4] Running {len(confluence_pairs)} CONFLUENCE pair backtests...")
    confluence_results = []

    for sig_a_key, sig_b_key, pair_label in confluence_pairs:
        sig_a = all_signals[sig_a_key]
        sig_b = all_signals[sig_b_key]

        # Test both same-day and 1-day window
        for window in [0, 1]:
            window_label = f"{pair_label} (w={window}d)"
            intersected = confluence_signals(sig_a, sig_b, window=window)

            if len(intersected) < 5:
                print(f"  {window_label:40s} | SKIP (only {len(intersected)} entries)")
                confluence_results.append({
                    'label': window_label,
                    'sig_a': sig_a_key, 'sig_b': sig_b_key, 'window': window,
                    'n_trades': len(intersected), 'sharpe': 0, 'wr': 0, 'pf': 0,
                    'perm_p': 1.0, 'passed': False,
                    'confluence_alpha': False,
                    'reason': 'insufficient entries',
                })
                continue

            trades_df = run_backtest(intersected, data, stock_tickers, label=window_label)
            result = validate_strategy(trades_df, label=window_label, run_perm=True,
                                       all_signal_entries=intersected, data=data, stock_tickers=stock_tickers)

            # Check confluence alpha: pair Sharpe > BOTH solo Sharpes
            solo_a_sharpe = solo_results[sig_a_key]['sharpe']
            solo_b_sharpe = solo_results[sig_b_key]['sharpe']
            pair_sharpe = result['sharpe']
            confluence_alpha = pair_sharpe > max(solo_a_sharpe, solo_b_sharpe)

            result['sig_a'] = sig_a_key
            result['sig_b'] = sig_b_key
            result['window'] = window
            result['solo_a_sharpe'] = solo_a_sharpe
            result['solo_b_sharpe'] = solo_b_sharpe
            result['confluence_alpha'] = confluence_alpha
            result['sharpe_lift'] = round(pair_sharpe - max(solo_a_sharpe, solo_b_sharpe), 3)

            confluence_results.append(result)

            alpha_tag = "ALPHA" if confluence_alpha else "no alpha"
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  {window_label:40s} | n={result['n_trades']:4d} | Sharpe={pair_sharpe:6.3f} "
                  f"(vs {solo_a_sharpe:.3f}/{solo_b_sharpe:.3f}) | WR={result['wr']:.3f} | "
                  f"PF={result['pf']:.3f} | p={result['perm_p']:.4f} | {status} | {alpha_tag}")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    # Solo baselines
    print("\n── SOLO BASELINES ──")
    print(f"{'Signal':25s} | {'N':>5s} | {'Sharpe':>7s} | {'WR':>5s} | {'PF':>6s} | {'MDD':>8s} | {'RegGap':>6s} | {'Perm p':>7s} | {'Status':>6s}")
    print("-" * 100)
    for key, r in solo_results.items():
        status = "PASS" if r['passed'] else "FAIL"
        print(f"{key:25s} | {r['n_trades']:5d} | {r['sharpe']:7.3f} | {r['wr']:.3f} | {r['pf']:6.3f} | "
              f"{r['mdd']:8.2f} | {r['regime_gap']:6.3f} | {r['perm_p']:7.4f} | {status:>6s}")

    # Confluence pairs
    print(f"\n── CONFLUENCE PAIRS ──")
    print(f"{'Pair':40s} | {'N':>5s} | {'Sharpe':>7s} | {'SoloA':>6s} | {'SoloB':>6s} | {'Lift':>6s} | "
          f"{'WR':>5s} | {'PF':>6s} | {'Perm p':>7s} | {'Alpha':>5s} | {'Pass':>4s}")
    print("-" * 130)

    # Sort by Sharpe descending
    sorted_conf = sorted(confluence_results, key=lambda x: x.get('sharpe', 0), reverse=True)
    for r in sorted_conf:
        alpha_tag = "YES" if r.get('confluence_alpha', False) else "no"
        status = "PASS" if r.get('passed', False) else "FAIL"
        solo_a = r.get('solo_a_sharpe', 0)
        solo_b = r.get('solo_b_sharpe', 0)
        lift = r.get('sharpe_lift', 0)
        print(f"{r['label']:40s} | {r['n_trades']:5d} | {r.get('sharpe',0):7.3f} | {solo_a:6.3f} | {solo_b:6.3f} | "
              f"{lift:+6.3f} | {r.get('wr',0):.3f} | {r.get('pf',0):6.3f} | {r.get('perm_p',1):7.4f} | "
              f"{alpha_tag:>5s} | {status:>4s}")

    # Winners
    winners = [r for r in sorted_conf if r.get('passed', False) and r.get('confluence_alpha', False)]
    print(f"\n── WINNERS (passed 5-gate + confluence alpha) ──")
    if winners:
        for r in winners:
            print(f"  ** {r['label']}: Sharpe={r['sharpe']:.3f}, lift={r.get('sharpe_lift',0):+.3f}, "
                  f"WR={r['wr']:.1%}, PF={r['pf']:.2f}, perm_p={r['perm_p']:.4f}")
    else:
        print("  No pairs passed ALL gates with confluence alpha.")
        # Show best near-misses
        near = [r for r in sorted_conf if r.get('sharpe', 0) > 0.3 and r['n_trades'] >= 10]
        if near:
            print("  Best near-misses:")
            for r in near[:5]:
                print(f"    {r['label']}: Sharpe={r.get('sharpe',0):.3f}, WR={r.get('wr',0):.1%}, "
                      f"perm_p={r.get('perm_p',1):.4f}, failed={r.get('failed_gates', [])}")

    # Save results
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
        },
        'solo_results': {k: {kk: (vv if not isinstance(vv, (pd.Timestamp, np.floating, np.integer)) else str(vv)) for kk, vv in v.items()} for k, v in solo_results.items()},
        'confluence_results': [{k: (v if not isinstance(v, (pd.Timestamp, np.floating, np.integer, np.bool_)) else (str(v) if isinstance(v, pd.Timestamp) else float(v) if isinstance(v, (np.floating, np.integer)) else bool(v))) for k, v in r.items()} for r in sorted_conf],
        'winners': [{k: (v if not isinstance(v, (pd.Timestamp, np.floating, np.integer, np.bool_)) else (str(v) if isinstance(v, pd.Timestamp) else float(v) if isinstance(v, (np.floating, np.integer)) else bool(v))) for k, v in r.items()} for r in winners],
    }

    results_file = OUTPUT_DIR / 'confluence_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    print("\n" + "=" * 80)
    print("DONE — Cross-Signal Confluence Backtest Complete")
    print("=" * 80)

    return results_data


if __name__ == '__main__':
    main()
