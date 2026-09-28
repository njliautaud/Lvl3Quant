#!/usr/bin/env python3
"""
Cross-Signal Confluence v2 Backtest (HC #772 + HC #777)
========================================================
Tests UNTESTED confluence pairs from validated strategies.
V1 tested 12 pairs — this covers the remaining ~24 untested combos.

Focus: Strategies #9-#12 cross-pairings + wider windows (0,1,3,5,7d).

Also adds Vol Term Structure (#12) signal which was missing from v1
(v1 had VIX term structure — a dead solo signal, not the validated strategy).
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
START_DATE = '2019-01-01'
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

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/cross_signal_confluence_v2')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
    'MA', 'WMT', 'PEP', 'KO', 'TMO', 'ABT', 'BAC', 'XOM', 'CVX', 'CSCO',
]

MACRO_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', '^TNX', 'HYG', 'LQD']

print("=" * 80)
print("CROSS-SIGNAL CONFLUENCE v2 BACKTEST (HC #772 — Untested Pairs)")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD (reuse v1 cache if available)
# ═══════════════════════════════════════════════════════════════════════
def download_data():
    import yfinance as yf
    # Try v1 cache first
    v1_cache = Path('/home/jupiter/Lvl3Quant/output/growth_research/cross_signal_confluence/_confluence_cache.pkl')
    v2_cache = OUTPUT_DIR / '_confluence_v2_cache.pkl'

    for cache_file in [v2_cache, v1_cache]:
        if cache_file.exists():
            import pickle
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"  Loaded cache from {cache_file.name}: {len(data['close'].columns)} stocks, {len(data['close'])} days")
            # Save to v2 location if loaded from v1
            if cache_file == v1_cache and not v2_cache.exists():
                with open(v2_cache, 'wb') as f:
                    pickle.dump(data, f)
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

    for df in [close, high, low, volume]:
        if hasattr(df.columns, 'droplevel'):
            try: df.columns = df.columns.droplevel(1)
            except: pass

    close = close.ffill().dropna(how='all')
    data = {
        'close': close,
        'high': high.reindex(close.index).ffill(),
        'low': low.reindex(close.index).ffill(),
        'volume': volume.reindex(close.index).ffill().fillna(0),
    }
    import pickle
    with open(v2_cache, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Downloaded: {len(close.columns)} tickers, {len(close)} days")
    return data

# ═══════════════════════════════════════════════════════════════════════
# 2. SIGNAL GENERATORS (all 9 from v1 + Vol Term Structure #12)
# ═══════════════════════════════════════════════════════════════════════

def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _realized_vol(returns, window=21):
    return returns.rolling(window).std() * np.sqrt(252) * 100

def signal_iv_rv_gap(data, stock_tickers):
    """#10 IV-RV Gap: buy when VIX > realized vol of SPY by 5+ points."""
    spy_close = data['close']['SPY']
    spy_ret = spy_close.pct_change()
    vix = data['close']['^VIX']
    rv_21 = _realized_vol(spy_ret, 21)
    gap = vix - rv_21
    fire_days = gap[gap > 5].index
    signals = {}
    for day in fire_days:
        for t in stock_tickers:
            if t in data['close'].columns:
                try:
                    if pd.notna(data['close'].at[day, t]):
                        signals[(day, t)] = True
                except: pass
    return signals, 'IV-RV Gap (#10)'

def signal_rsi_divergence(data, stock_tickers):
    """#8 RSI Divergence: price lower low but RSI higher low, RSI<40."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns: continue
        close = data['close'][t].dropna()
        rsi = _rsi(close, 14)
        for i in range(20, len(close)):
            window_close = close.iloc[i-10:i+1]
            window_rsi = rsi.iloc[i-10:i+1]
            if len(window_close) < 11 or window_rsi.isna().any(): continue
            if close.iloc[i] < window_close.iloc[0]:
                if rsi.iloc[i] > window_rsi.iloc[0]:
                    if rsi.iloc[i] < 40:
                        signals[(close.index[i], t)] = True
    return signals, 'RSI Divergence (#8)'

def signal_bond_yield(data, stock_tickers):
    """#9 Bond Yield: buy when 10Y yield drops >0.1% in 5 days."""
    if '^TNX' not in data['close'].columns:
        tlt = data['close']['TLT']
        yield_change = tlt.pct_change(5)
        fire_days = yield_change[yield_change > 0.02].index
    else:
        tnx = data['close']['^TNX']
        yield_change = tnx.diff(5)
        fire_days = yield_change[yield_change < -0.10].index
    signals = {}
    for day in fire_days:
        for t in stock_tickers:
            if t in data['close'].columns:
                try:
                    if pd.notna(data['close'].at[day, t]):
                        signals[(day, t)] = True
                except: pass
    return signals, 'Bond Yield (#9)'

def signal_liquidity(data, stock_tickers):
    """#11 Liquidity: HL spread < 85% of 60d avg + RSI<40."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns or t not in data['high'].columns: continue
        high = data['high'][t].dropna()
        low = data['low'][t].dropna()
        close = data['close'][t].dropna()
        idx = high.index.intersection(low.index).intersection(close.index)
        if len(idx) < 65: continue
        high, low, close = high.loc[idx], low.loc[idx], close.loc[idx]
        hl_spread = (high - low) / close
        avg_60 = hl_spread.rolling(60).mean()
        narrow = hl_spread < avg_60 * 0.85
        rsi = _rsi(close, 14)
        for i in range(60, len(idx)):
            if narrow.iloc[i] and rsi.iloc[i] < 40:
                signals[(idx[i], t)] = True
    return signals, 'Liquidity (#11)'

def signal_vol_term_structure(data, stock_tickers):
    """#12 Vol Term Structure (VALIDATED): buy when VIX/VIX3M > 1.0 (backwardation=fear)
    AND stock is >5% below 20d SMA (dip context). This is the VALIDATED version,
    not the dead VIX-only signal."""
    vix = data['close'].get('^VIX')
    vix3m = data['close'].get('^VIX3M')
    if vix is None or vix3m is None:
        return {}, 'Vol Term Structure (#12)'
    ratio = vix / vix3m
    fear_days = set(ratio[ratio > 1.0].index)
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns: continue
        close = data['close'][t].dropna()
        sma20 = close.rolling(20).mean()
        drawdown = (close - sma20) / sma20
        for i in range(20, len(close)):
            day = close.index[i]
            if day in fear_days and drawdown.iloc[i] < -0.05:
                signals[(day, t)] = True
    return signals, 'Vol Term Structure (#12)'

def signal_consecutive_dip(data, stock_tickers):
    """Consecutive Dip: 3+ red days with each day's loss bigger than prior."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns: continue
        close = data['close'][t].dropna()
        ret = close.pct_change()
        for i in range(3, len(close)):
            r1, r2, r3 = ret.iloc[i-2], ret.iloc[i-1], ret.iloc[i]
            if r1 < 0 and r2 < 0 and r3 < 0:
                if r2 < r1 and r3 < r2:
                    signals[(close.index[i], t)] = True
    return signals, 'Consecutive Dip'

def signal_base_mr(data, stock_tickers):
    """Base MR: RSI<30 and price >7% below 50d high."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns: continue
        close = data['close'][t].dropna()
        rsi = _rsi(close, 14)
        high_50 = close.rolling(50).max()
        drawdown = (close - high_50) / high_50
        for i in range(50, len(close)):
            if rsi.iloc[i] < 30 and drawdown.iloc[i] < -0.07:
                signals[(close.index[i], t)] = True
    return signals, 'Base MR'

def signal_vix_term_structure_dead(data, stock_tickers):
    """DEAD solo: plain VIX backwardation without dip context."""
    vix = data['close'].get('^VIX')
    vix3m = data['close'].get('^VIX3M')
    if vix is None or vix3m is None:
        return {}, 'VIX TermStr (dead)'
    ratio = vix / vix3m
    fire_days = ratio[ratio > 1.0].index
    signals = {}
    for day in fire_days:
        for t in stock_tickers:
            if t in data['close'].columns:
                try:
                    if pd.notna(data['close'].at[day, t]):
                        signals[(day, t)] = True
                except: pass
    return signals, 'VIX TermStr (dead)'

def signal_vol_contraction(data, stock_tickers):
    """DEAD solo: Keltner squeeze + RSI<40."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns: continue
        close = data['close'][t].dropna()
        high = data['high'].get(t, close).dropna()
        low = data['low'].get(t, close).dropna()
        idx = close.index.intersection(high.index).intersection(low.index)
        if len(idx) < 25: continue
        close, high, low = close.loc[idx], high.loc[idx], low.loc[idx]
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_width = (2 * std20) / sma20
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr20 = tr.rolling(20).mean()
        kelt_width = (2 * 1.5 * atr20) / sma20
        squeeze = bb_width < kelt_width
        rsi = _rsi(close, 14)
        for i in range(20, len(idx)):
            if squeeze.iloc[i] and rsi.iloc[i] < 40:
                signals[(idx[i], t)] = True
    return signals, 'Vol Contraction (dead)'

def signal_gap_reversal(data, stock_tickers):
    """DEAD solo: gap down >2% with reversal close."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns: continue
        close = data['close'][t].dropna()
        high = data['high'].get(t, close).dropna()
        low = data['low'].get(t, close).dropna()
        idx = close.index.intersection(high.index).intersection(low.index)
        if len(idx) < 5: continue
        close, high, low = close.loc[idx], high.loc[idx], low.loc[idx]
        for i in range(1, len(idx)):
            gap = (low.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
            if gap < -0.02:
                day_range = high.iloc[i] - low.iloc[i]
                if day_range > 0:
                    close_position = (close.iloc[i] - low.iloc[i]) / day_range
                    if close_position > 0.6:
                        signals[(idx[i], t)] = True
    return signals, 'Gap Reversal (dead)'

# ═══════════════════════════════════════════════════════════════════════
# 3. BACKTESTER (identical to v1)
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(signal_entries, data, stock_tickers, label=''):
    close = data['close']
    spy_close = close['SPY']
    bt_start = pd.Timestamp(BACKTEST_START)
    entries = [(d, t) for (d, t) in signal_entries if d >= bt_start]
    entries.sort(key=lambda x: x[0])
    if not entries: return None
    trades = []
    open_positions = []
    for entry_date, ticker in entries:
        open_positions = [p for p in open_positions if p[3] > entry_date]
        if len(open_positions) >= MAX_CONCURRENT: continue
        try: entry_price = close.at[entry_date, ticker]
        except: continue
        if pd.isna(entry_price) or entry_price <= 0: continue
        future_dates = close.index[close.index > entry_date]
        if len(future_dates) == 0: continue
        exit_price = None; exit_date = None; exit_reason = 'hold_expiry'
        for j, fdate in enumerate(future_dates[:HOLD_DAYS]):
            try: price = close.at[fdate, ticker]
            except: continue
            if pd.isna(price): continue
            ret = (price - entry_price) / entry_price
            if ret >= PROFIT_TARGET:
                exit_price = price; exit_date = fdate; exit_reason = 'profit_target'; break
            elif ret <= STOP_LOSS:
                exit_price = price; exit_date = fdate; exit_reason = 'stop_loss'; break
        if exit_price is None:
            hold_end = min(HOLD_DAYS, len(future_dates))
            if hold_end > 0:
                exit_date = future_dates[hold_end - 1]
                try: exit_price = close.at[exit_date, ticker]
                except: continue
                if pd.isna(exit_price): continue
        if exit_price is None: continue
        raw_ret = (exit_price - entry_price) / entry_price
        net_ret = raw_ret - SPREAD_COST_PCT
        pnl = POS_SIZE * net_ret
        try:
            spy_sma = spy_close.rolling(200).mean()
            spy_regime = 'bull' if spy_close.at[entry_date] > spy_sma.at[entry_date] else 'bear'
        except: spy_regime = 'unknown'
        trades.append({
            'entry_date': entry_date, 'exit_date': exit_date, 'ticker': ticker,
            'entry_price': entry_price, 'exit_price': exit_price,
            'return': net_ret, 'pnl': pnl, 'exit_reason': exit_reason,
            'regime': spy_regime, 'hold_days': (exit_date - entry_date).days,
        })
        open_positions.append((entry_date, ticker, entry_price, exit_date))
    if not trades: return None
    return pd.DataFrame(trades)

# ═══════════════════════════════════════════════════════════════════════
# 4. VALIDATION (5-gate + perm test)
# ═══════════════════════════════════════════════════════════════════════

def validate_strategy(trades_df, label='', run_perm=True, all_signal_entries=None, data=None, stock_tickers=None):
    if trades_df is None or len(trades_df) < 10:
        return {
            'label': label, 'n_trades': 0 if trades_df is None else len(trades_df),
            'sharpe': 0, 'wr': 0, 'pf': 0, 'mdd': 0, 'regime_gap': 1.0,
            'perm_p': 1.0, 'passed': False, 'reason': 'insufficient trades'
        }
    rets = trades_df['return'].values
    n = len(rets)
    trades_per_year = n / max(1, (trades_df['entry_date'].max() - trades_df['entry_date'].min()).days / 365.25)
    if trades_per_year < 1: trades_per_year = 1
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1.0
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0
    sortino_denom = np.std(rets[rets < 0], ddof=1) if np.sum(rets < 0) > 1 else std_ret
    sortino = (mean_ret / sortino_denom) * np.sqrt(trades_per_year) if sortino_denom > 0 else 0
    wr = np.mean(rets > 0)
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = np.abs(np.sum(rets[rets < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0
    cum_pnl = np.cumsum(trades_df['pnl'].values)
    peak = np.maximum.accumulate(cum_pnl)
    dd = cum_pnl - peak
    mdd = np.min(dd) if len(dd) > 0 else 0
    bull_trades = trades_df[trades_df['regime'] == 'bull']
    bear_trades = trades_df[trades_df['regime'] == 'bear']
    if len(bull_trades) >= 3 and len(bear_trades) >= 3:
        bull_sharpe_raw = np.mean(bull_trades['return']) / max(np.std(bull_trades['return'], ddof=1), 1e-6)
        bear_sharpe_raw = np.mean(bear_trades['return']) / max(np.std(bear_trades['return'], ddof=1), 1e-6)
        max_abs = max(abs(bull_sharpe_raw), abs(bear_sharpe_raw), 1e-6)
        regime_gap = abs(bull_sharpe_raw - bear_sharpe_raw) / max_abs
    else:
        regime_gap = 0.0
    perm_p = 1.0
    if run_perm and n >= 10 and all_signal_entries is not None and data is not None and stock_tickers is not None:
        actual_sharpe = sharpe
        perm_sharpes = []
        bt_start = pd.Timestamp(BACKTEST_START)
        valid_dates = data['close'].index[data['close'].index >= bt_start]
        valid_dates = valid_dates[:-HOLD_DAYS-5]
        for _ in range(N_PERMS):
            n_raw = len(all_signal_entries)
            rand_dates = np.random.choice(valid_dates, size=min(n_raw, len(valid_dates)), replace=True)
            rand_tickers = np.random.choice(stock_tickers, size=min(n_raw, len(valid_dates)), replace=True)
            rand_signals = {(d, t): True for d, t in zip(rand_dates, rand_tickers)}
            rand_trades = run_backtest(rand_signals, data, stock_tickers)
            if rand_trades is not None and len(rand_trades) >= 5:
                r_rets = rand_trades['return'].values
                r_tpy = len(r_rets) / max(1, (rand_trades['entry_date'].max() - rand_trades['entry_date'].min()).days / 365.25)
                if r_tpy < 1: r_tpy = 1
                r_sharpe = (np.mean(r_rets) / max(np.std(r_rets, ddof=1), 1e-8)) * np.sqrt(r_tpy)
            else:
                r_sharpe = 0.0
            perm_sharpes.append(r_sharpe)
        perm_p = np.mean(np.array(perm_sharpes) >= actual_sharpe)
    gates = {
        'sharpe': sharpe > 0.3, 'wr': wr > 0.45, 'pf': pf > 1.0,
        'mdd': mdd > -POS_SIZE * 5, 'regime_gap': regime_gap < REGIME_GAP_LIMIT,
    }
    perm_pass = perm_p < 0.05
    passed = all(gates.values()) and perm_pass
    failed_gates = [k for k, v in gates.items() if not v]
    if not perm_pass: failed_gates.append('perm_test')
    return {
        'label': label, 'n_trades': n, 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'wr': round(wr, 3), 'pf': round(pf, 3), 'mdd': round(mdd, 2),
        'regime_gap': round(regime_gap, 3), 'perm_p': round(perm_p, 4),
        'passed': passed, 'failed_gates': failed_gates,
        'mean_ret': round(mean_ret * 100, 2),
        'bull_n': len(bull_trades), 'bear_n': len(bear_trades),
        'avg_hold': round(trades_df['hold_days'].mean(), 1),
        'reason': 'PASSED' if passed else f"FAILED: {', '.join(failed_gates)}",
    }

# ═══════════════════════════════════════════════════════════════════════
# 5. CONFLUENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════

def confluence_signals(sig_a, sig_b, window=1):
    if window == 0:
        return {k: True for k in sig_a if k in sig_b}
    from collections import defaultdict
    b_by_ticker = defaultdict(set)
    for (d, t) in sig_b:
        b_by_ticker[t].add(d)
    result = {}
    for (d, t) in sig_a:
        if t not in b_by_ticker: continue
        for offset in range(-window, window + 1):
            check_d = d + pd.Timedelta(days=offset)
            if check_d in b_by_ticker[t]:
                result[(d, t)] = True
                break
    return result

# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN — UNTESTED PAIRS
# ═══════════════════════════════════════════════════════════════════════

def main():
    data = download_data()
    stock_tickers = [t for t in UNIVERSE if t in data['close'].columns]
    print(f"  Stock universe: {len(stock_tickers)} tickers")

    # ── Generate all signals ──
    print("\n[2] Generating signals...")
    signal_generators = {
        'iv_rv_gap':        signal_iv_rv_gap,
        'rsi_divergence':   signal_rsi_divergence,
        'bond_yield':       signal_bond_yield,
        'liquidity':        signal_liquidity,
        'vol_term_str':     signal_vol_term_structure,     # VALIDATED #12
        'consecutive_dip':  signal_consecutive_dip,
        'base_mr':          signal_base_mr,
        'vix_term_dead':    signal_vix_term_structure_dead, # dead solo
        'vol_contraction':  signal_vol_contraction,         # dead solo
        'gap_reversal':     signal_gap_reversal,            # dead solo
    }

    all_signals = {}
    for key, gen_func in signal_generators.items():
        t0 = time.time()
        sigs, name = gen_func(data, stock_tickers)
        elapsed = time.time() - t0
        print(f"  {name:30s} | {len(sigs):6d} raw entries ({elapsed:.1f}s)")
        all_signals[key] = sigs

    # ── Solo backtests ──
    print("\n[3] Running SOLO backtests (baselines)...")
    solo_results = {}
    for key, sigs in all_signals.items():
        trades_df = run_backtest(sigs, data, stock_tickers, label=key)
        result = validate_strategy(trades_df, label=f"SOLO: {key}", run_perm=True,
                                   all_signal_entries=sigs, data=data, stock_tickers=stock_tickers)
        solo_results[key] = result
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"  {key:25s} | n={result['n_trades']:4d} | Sharpe={result['sharpe']:6.3f} | "
              f"WR={result['wr']:.3f} | PF={result['pf']:.3f} | perm_p={result['perm_p']:.4f} | {status}")

    # ── UNTESTED pairs — focus on #9-#12 cross combos + wider windows ──
    # v1 already tested these 12 pairs (w=0,1 only):
    #   iv_rv_gap + rsi_div, iv_rv_gap + liquidity, bond_yield + rsi_div,
    #   iv_rv_gap + bond_yield, rsi_div + liquidity, iv_rv_gap + consec_dip,
    #   iv_rv_gap + vix_term, iv_rv_gap + vol_contract, iv_rv_gap + gap_rev,
    #   base_mr + iv_rv_gap, base_mr + rsi_div, base_mr + liquidity
    #
    # UNTESTED pairs involving validated strategies:
    untested_pairs = [
        # Cross-combos among strongest validated strategies
        ('bond_yield',       'liquidity',        'BondYield + Liquidity'),
        ('bond_yield',       'vol_term_str',     'BondYield + VolTermStr'),
        ('bond_yield',       'consecutive_dip',  'BondYield + ConsecDip'),
        ('bond_yield',       'iv_rv_gap',        'BondYield + IV-RV (wider)'),   # re-test with wider windows
        ('rsi_divergence',   'vol_term_str',     'RSI Div + VolTermStr'),
        ('rsi_divergence',   'consecutive_dip',  'RSI Div + ConsecDip'),
        ('liquidity',        'vol_term_str',     'Liquidity + VolTermStr'),
        ('liquidity',        'consecutive_dip',  'Liquidity + ConsecDip'),
        ('vol_term_str',     'iv_rv_gap',        'VolTermStr + IV-RV'),
        ('vol_term_str',     'consecutive_dip',  'VolTermStr + ConsecDip'),
        ('vol_term_str',     'base_mr',          'VolTermStr + BaseMR'),
        # Dead-solo as filter on other validated signals (not just IV-RV)
        ('bond_yield',       'vix_term_dead',    'BondYield + VIX (dead)'),
        ('bond_yield',       'vol_contraction',  'BondYield + VolContract (dead)'),
        ('rsi_divergence',   'vol_contraction',  'RSI Div + VolContract (dead)'),
        ('liquidity',        'gap_reversal',     'Liquidity + GapRev (dead)'),
        ('base_mr',          'vol_term_str',     'BaseMR + VolTermStr'),
        ('base_mr',          'consecutive_dip',  'BaseMR + ConsecDip'),
        ('base_mr',          'bond_yield',       'BaseMR + BondYield'),
    ]

    # Test windows: 0, 1, 3, 5, 7 days
    WINDOWS = [0, 1, 3, 5, 7]

    print(f"\n[4] Running {len(untested_pairs)} UNTESTED confluence pairs × {len(WINDOWS)} windows = {len(untested_pairs) * len(WINDOWS)} tests...")
    confluence_results = []
    passes = []

    for pair_idx, (sig_a_key, sig_b_key, pair_label) in enumerate(untested_pairs):
        sig_a = all_signals.get(sig_a_key, {})
        sig_b = all_signals.get(sig_b_key, {})
        if not sig_a or not sig_b:
            print(f"  SKIP {pair_label}: missing signal data")
            continue

        for window in WINDOWS:
            window_label = f"{pair_label} (w={window}d)"
            intersected = confluence_signals(sig_a, sig_b, window=window)

            if len(intersected) < 5:
                confluence_results.append({
                    'label': window_label, 'sig_a': sig_a_key, 'sig_b': sig_b_key,
                    'window': window, 'n_intersect': len(intersected),
                    'n_trades': 0, 'sharpe': 0, 'passed': False,
                    'reason': f'insufficient entries ({len(intersected)})',
                })
                continue

            trades_df = run_backtest(intersected, data, stock_tickers, label=window_label)
            result = validate_strategy(trades_df, label=window_label, run_perm=True,
                                       all_signal_entries=intersected, data=data, stock_tickers=stock_tickers)

            solo_a_sharpe = solo_results.get(sig_a_key, {}).get('sharpe', 0)
            solo_b_sharpe = solo_results.get(sig_b_key, {}).get('sharpe', 0)
            pair_sharpe = result['sharpe']
            confluence_alpha = pair_sharpe > max(solo_a_sharpe, solo_b_sharpe)
            sharpe_lift = round(pair_sharpe - max(solo_a_sharpe, solo_b_sharpe), 3)

            result['sig_a'] = sig_a_key
            result['sig_b'] = sig_b_key
            result['window'] = window
            result['n_intersect'] = len(intersected)
            result['solo_a_sharpe'] = solo_a_sharpe
            result['solo_b_sharpe'] = solo_b_sharpe
            result['confluence_alpha'] = confluence_alpha
            result['sharpe_lift'] = sharpe_lift

            confluence_results.append(result)

            status = "✅ PASS" if result['passed'] else "❌"
            alpha = "📈 ALPHA" if confluence_alpha else ""
            print(f"  [{pair_idx+1:2d}/{len(untested_pairs)}] {window_label:45s} | "
                  f"n={result['n_trades']:4d} | Sharpe={pair_sharpe:6.3f} | "
                  f"perm_p={result.get('perm_p', 1.0):.4f} | lift={sharpe_lift:+.3f} | {status} {alpha}")

            if result['passed']:
                passes.append(result)

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(confluence_results)
    n_pass = len([r for r in confluence_results if r.get('passed', False)])
    n_alpha = len([r for r in confluence_results if r.get('confluence_alpha', False)])

    print(f"\nTotal tests: {total}")
    print(f"5-gate PASS: {n_pass}")
    print(f"Confluence alpha (Sharpe > both solos): {n_alpha}")

    if passes:
        print(f"\n{'─'*60}")
        print("PASSING PAIRS:")
        for p in sorted(passes, key=lambda x: x['sharpe'], reverse=True):
            print(f"  🏆 {p['label']:45s} | Sharpe={p['sharpe']:6.3f} | Sortino={p.get('sortino',0):6.3f} | "
                  f"WR={p['wr']:.3f} | PF={p['pf']:.3f} | n={p['n_trades']} | "
                  f"perm_p={p['perm_p']:.4f} | lift={p.get('sharpe_lift',0):+.3f}")
    else:
        print("\n  No pairs passed all 5 gates. Confluence continues to underperform solo signals.")

    # Top confluence alpha (even if not 5-gate pass)
    alpha_results = [r for r in confluence_results if r.get('confluence_alpha', False) and r.get('n_trades', 0) >= 10]
    if alpha_results:
        print(f"\n{'─'*60}")
        print("POSITIVE CONFLUENCE ALPHA (even if didn't pass 5-gate):")
        for r in sorted(alpha_results, key=lambda x: x.get('sharpe_lift', 0), reverse=True)[:10]:
            print(f"  📈 {r['label']:45s} | Sharpe={r['sharpe']:6.3f} | lift={r.get('sharpe_lift',0):+.3f} | "
                  f"n={r['n_trades']} | perm_p={r.get('perm_p', 1.0):.4f}")

    # ── Save results ──
    output_file = OUTPUT_DIR / 'confluence_v2_results.json'
    # Convert timestamps for JSON
    for r in confluence_results:
        for k, v in r.items():
            if isinstance(v, (pd.Timestamp, datetime)):
                r[k] = str(v)
    with open(output_file, 'w') as f:
        json.dump({'solo': solo_results, 'confluence': confluence_results, 'passes': [p['label'] for p in passes],
                   'timestamp': str(datetime.now())}, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    print("\n" + "=" * 80)
    print("DONE — Cross-Signal Confluence v2")
    print("=" * 80)

if __name__ == '__main__':
    main()
