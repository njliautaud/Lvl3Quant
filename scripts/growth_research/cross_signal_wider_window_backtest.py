#!/usr/bin/env python3
"""
Cross-Signal WIDER WINDOW Confluence Backtest
==============================================
Previous test: same-day confluence was too restrictive.
This test: Signal A fires, Signal B must confirm within N days (N=1,3,5,7).
More realistic — signals cascade rather than perfectly align.

Entry logic: Signal A fires on day T. If Signal B fires on any day within
[T, T+N], enter on the day Signal B fires.

5-gate validation: Sharpe>0.5, WR>55%, PF>1.5, regime_gap<0.50, perm_p<0.05
Confluence alpha: pair_sharpe - max(solo_A_sharpe, solo_B_sharpe) > 0
"""

import os, sys, json, warnings, time, functools
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
SPREAD_COST_PCT = 0.001  # 10bps RT
N_PERMS = 200
REGIME_GAP_LIMIT = 0.50

# 5-gate thresholds
SHARPE_GATE = 0.5
WR_GATE = 0.55
PF_GATE = 1.5

CONFLUENCE_WINDOWS = [1, 3, 5, 7]

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/cross_signal_wider_window')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
]

MACRO_TICKERS = ['SPY', '^VIX', 'TLT', '^TNX']

print("=" * 90)
print("CROSS-SIGNAL WIDER WINDOW CONFLUENCE BACKTEST")
print("=" * 90)

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════
def download_data():
    import yfinance as yf
    cache_file = OUTPUT_DIR / '_wider_window_cache.pkl'
    if cache_file.exists():
        import pickle
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded cache: {len(data['close'].columns)} stocks, {len(data['close'])} days")
        return data

    print(f"\n[1] Downloading {len(UNIVERSE)} stocks + {len(MACRO_TICKERS)} macro...")
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
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Downloaded: {len(close.columns)} tickers, {len(close)} days")
    return data


# ═══════════════════════════════════════════════════════════════════════
# 2. SIGNAL GENERATORS — return dict of {(date, ticker): True}
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
    """VIX > SPY 20-day realized vol by 5+ points."""
    spy_ret = data['close']['SPY'].pct_change()
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
                except:
                    pass
    return signals, 'IV-RV Gap'


def signal_rsi_divergence(data, stock_tickers):
    """Price makes lower low but RSI(14) makes higher low (bullish divergence)."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        rsi = _rsi(close, 14)
        for i in range(20, len(close)):
            window_close = close.iloc[i-10:i+1]
            window_rsi = rsi.iloc[i-10:i+1]
            if len(window_close) < 11 or window_rsi.isna().any():
                continue
            # Price lower low, RSI higher low
            if close.iloc[i] < window_close.iloc[0]:
                if rsi.iloc[i] > window_rsi.iloc[0]:
                    if rsi.iloc[i] < 40:
                        signals[(close.index[i], t)] = True
    return signals, 'RSI Divergence'


def signal_bond_yield(data, stock_tickers):
    """10Y yield drops >0.1% over 5 days (flight-to-safety ending)."""
    if '^TNX' in data['close'].columns:
        tnx = data['close']['^TNX']
        yield_change = tnx.diff(5)
        fire_days = yield_change[yield_change < -0.10].index
    else:
        tlt = data['close']['TLT']
        yield_change = tlt.pct_change(5)
        fire_days = yield_change[yield_change > 0.02].index

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
    """High-Low spread < 60-day average (bid-ask narrowing = smart money)."""
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
        narrow = hl_spread < avg_60 * 0.85
        rsi = _rsi(close, 14)
        for i in range(60, len(idx)):
            if narrow.iloc[i] and rsi.iloc[i] < 40:
                signals[(idx[i], t)] = True
    return signals, 'Liquidity Signal'


def signal_consecutive_dip(data, stock_tickers):
    """3+ consecutive red days with each loss bigger than previous."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        ret = close.pct_change()
        for i in range(3, len(close)):
            r1, r2, r3 = ret.iloc[i-2], ret.iloc[i-1], ret.iloc[i]
            if r1 < 0 and r2 < 0 and r3 < 0:
                if r2 < r1 and r3 < r2:
                    signals[(close.index[i], t)] = True
    return signals, 'Consecutive Dip'


def signal_base_mr(data, stock_tickers):
    """RSI(14) < 30 AND price > 7% below 52-week high."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        rsi = _rsi(close, 14)
        high_252 = close.rolling(252, min_periods=50).max()
        drawdown = (close - high_252) / high_252
        for i in range(50, len(close)):
            if rsi.iloc[i] < 30 and drawdown.iloc[i] < -0.07:
                signals[(close.index[i], t)] = True
    return signals, 'Base MR'


# ═══════════════════════════════════════════════════════════════════════
# 3. BACKTESTER
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(signal_entries, data, stock_tickers):
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

        exit_price = exit_date = None
        exit_reason = 'hold_expiry'

        for fdate in future_dates[:HOLD_DAYS]:
            try:
                price = close.at[fdate, ticker]
            except:
                continue
            if pd.isna(price):
                continue
            ret = (price - entry_price) / entry_price
            if ret >= PROFIT_TARGET:
                exit_price, exit_date, exit_reason = price, fdate, 'profit_target'
                break
            elif ret <= STOP_LOSS:
                exit_price, exit_date, exit_reason = price, fdate, 'stop_loss'
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
            spy_sma = spy_close.rolling(200).mean()
            spy_regime = 'bull' if spy_close.at[entry_date] > spy_sma.at[entry_date] else 'bear'
        except:
            spy_regime = 'unknown'

        trades.append({
            'entry_date': entry_date, 'exit_date': exit_date, 'ticker': ticker,
            'entry_price': entry_price, 'exit_price': exit_price,
            'return': net_ret, 'pnl': pnl, 'exit_reason': exit_reason,
            'regime': spy_regime, 'hold_days': (exit_date - entry_date).days,
        })
        open_positions.append((entry_date, ticker, entry_price, exit_date))

    return pd.DataFrame(trades) if trades else None


# ═══════════════════════════════════════════════════════════════════════
# 4. VALIDATION (5-gate)
# ═══════════════════════════════════════════════════════════════════════

def validate_strategy(trades_df, label='', run_perm=True, all_signal_entries=None,
                      data=None, stock_tickers=None):
    if trades_df is None or len(trades_df) < 10:
        return {
            'label': label, 'n_trades': 0 if trades_df is None else len(trades_df),
            'sharpe': 0, 'wr': 0, 'pf': 0, 'mdd': 0, 'regime_gap': 1.0,
            'perm_p': 1.0, 'passed': False, 'failed_gates': ['insufficient trades'],
        }

    rets = trades_df['return'].values
    n = len(rets)

    # Sharpe
    years = max(1, (trades_df['entry_date'].max() - trades_df['entry_date'].min()).days / 365.25)
    tpy = n / years
    if tpy < 1: tpy = 1
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1.0
    sharpe = (mean_ret / std_ret) * np.sqrt(tpy) if std_ret > 0 else 0

    wr = np.mean(rets > 0)

    gross_profit = np.sum(rets[rets > 0])
    gross_loss = np.abs(np.sum(rets[rets < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

    cum_pnl = np.cumsum(trades_df['pnl'].values)
    peak = np.maximum.accumulate(cum_pnl)
    dd = cum_pnl - peak
    mdd = np.min(dd) if len(dd) > 0 else 0

    # Regime gap
    bull_t = trades_df[trades_df['regime'] == 'bull']
    bear_t = trades_df[trades_df['regime'] == 'bear']
    if len(bull_t) >= 3 and len(bear_t) >= 3:
        bull_s = np.mean(bull_t['return']) / max(np.std(bull_t['return'], ddof=1), 1e-6)
        bear_s = np.mean(bear_t['return']) / max(np.std(bear_t['return'], ddof=1), 1e-6)
        regime_gap = abs(bull_s - bear_s) / max(abs(bull_s), abs(bear_s), 1e-6)
    else:
        regime_gap = 0.0

    # Permutation test
    perm_p = 1.0
    if run_perm and n >= 10 and all_signal_entries is not None and data is not None:
        bt_start = pd.Timestamp(BACKTEST_START)
        valid_dates = data['close'].index[data['close'].index >= bt_start]
        valid_dates = valid_dates[:-HOLD_DAYS-5]
        perm_sharpes = []
        n_raw = len(all_signal_entries)

        for _ in range(N_PERMS):
            rand_dates = np.random.choice(valid_dates, size=min(n_raw, len(valid_dates)), replace=True)
            rand_tickers = np.random.choice(stock_tickers, size=min(n_raw, len(valid_dates)), replace=True)
            rand_signals = {(d, t): True for d, t in zip(rand_dates, rand_tickers)}
            rand_trades = run_backtest(rand_signals, data, stock_tickers)
            if rand_trades is not None and len(rand_trades) >= 5:
                r = rand_trades['return'].values
                r_tpy = len(r) / max(1, (rand_trades['entry_date'].max() - rand_trades['entry_date'].min()).days / 365.25)
                if r_tpy < 1: r_tpy = 1
                r_s = (np.mean(r) / max(np.std(r, ddof=1), 1e-9)) * np.sqrt(r_tpy)
            else:
                r_s = 0.0
            perm_sharpes.append(r_s)
        perm_p = np.mean(np.array(perm_sharpes) >= sharpe)

    # 5-gate
    gates = {
        'sharpe': sharpe > SHARPE_GATE,
        'wr': wr > WR_GATE,
        'pf': pf > PF_GATE,
        'regime_gap': regime_gap < REGIME_GAP_LIMIT,
        'perm_test': perm_p < 0.05,
    }
    failed = [k for k, v in gates.items() if not v]
    passed = len(failed) == 0

    return {
        'label': label, 'n_trades': n, 'sharpe': round(sharpe, 3),
        'wr': round(wr, 3), 'pf': round(pf, 3), 'mdd': round(mdd, 2),
        'regime_gap': round(regime_gap, 3), 'perm_p': round(perm_p, 4),
        'mean_ret_pct': round(mean_ret * 100, 2),
        'bull_n': len(bull_t), 'bear_n': len(bear_t),
        'avg_hold': round(trades_df['hold_days'].mean(), 1),
        'passed': passed, 'failed_gates': failed,
        'gates_passed': sum(1 for v in gates.values() if v),
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. WIDER WINDOW CONFLUENCE
# ═══════════════════════════════════════════════════════════════════════

def wider_window_confluence(sig_primary, sig_confirm, window_days):
    """Signal A fires on day T. Signal B must confirm within [T, T+window_days].
    Entry on the day Signal B fires. Both must be for the same ticker."""
    # Index confirming signal by ticker -> sorted dates
    confirm_by_ticker = defaultdict(list)
    for (d, t) in sig_confirm:
        confirm_by_ticker[t].append(d)
    for t in confirm_by_ticker:
        confirm_by_ticker[t].sort()

    entries = {}
    for (d_primary, t) in sig_primary:
        if t not in confirm_by_ticker:
            continue
        # Find confirming signals within [d_primary, d_primary + window_days]
        window_end = d_primary + pd.Timedelta(days=window_days)
        for d_confirm in confirm_by_ticker[t]:
            if d_confirm < d_primary:
                continue
            if d_confirm > window_end:
                break
            # Entry on the confirm day
            entries[(d_confirm, t)] = True
            break  # Take first confirmation only

    return entries


def any_two_of_three_confluence(sig_a, sig_b, sig_c, window_days):
    """Any 2 of 3 signals fire within a window_days window for the same ticker.
    Entry on the day the second signal fires."""
    # Combine all three: build per-ticker timeline of which signals fired when
    ticker_events = defaultdict(list)  # ticker -> [(date, signal_id)]
    for (d, t) in sig_a:
        ticker_events[t].append((d, 'A'))
    for (d, t) in sig_b:
        ticker_events[t].append((d, 'B'))
    for (d, t) in sig_c:
        ticker_events[t].append((d, 'C'))

    entries = {}
    for t, events in ticker_events.items():
        events.sort(key=lambda x: x[0])
        # Sliding window: for each event, look forward within window_days for a different signal
        for i, (d1, s1) in enumerate(events):
            window_end = d1 + pd.Timedelta(days=window_days)
            for j in range(i + 1, len(events)):
                d2, s2 = events[j]
                if d2 > window_end:
                    break
                if s2 != s1:
                    # Two different signals within window — enter on the later date
                    entries[(d2, t)] = True
                    break

    return entries


# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    data = download_data()
    stock_tickers = [t for t in UNIVERSE if t in data['close'].columns]
    print(f"  Stock universe: {len(stock_tickers)} tickers")

    # Generate all signals
    print("\n[2] Generating signals...")
    generators = {
        'iv_rv_gap': signal_iv_rv_gap,
        'rsi_div': signal_rsi_divergence,
        'bond_yield': signal_bond_yield,
        'liquidity': signal_liquidity,
        'consec_dip': signal_consecutive_dip,
        'base_mr': signal_base_mr,
    }

    all_signals = {}
    for key, func in generators.items():
        t0 = time.time()
        sigs, name = func(data, stock_tickers)
        print(f"  {name:20s}: {len(sigs):6d} raw entries ({time.time()-t0:.1f}s)")
        all_signals[key] = sigs

    # Solo baselines
    print("\n[3] Solo baselines...")
    solo = {}
    for key, sigs in all_signals.items():
        trades_df = run_backtest(sigs, data, stock_tickers)
        result = validate_strategy(trades_df, label=key, run_perm=True,
                                   all_signal_entries=sigs, data=data, stock_tickers=stock_tickers)
        solo[key] = result
        tag = "PASS" if result['passed'] else "FAIL"
        print(f"  {key:15s} | n={result['n_trades']:4d} | Sharpe={result['sharpe']:6.3f} | "
              f"WR={result['wr']:.3f} | PF={result['pf']:.3f} | p={result['perm_p']:.4f} | {tag}")

    # Define pair tests
    pair_defs = [
        ('iv_rv_gap', 'rsi_div',    'IVRV+RSI'),
        ('iv_rv_gap', 'liquidity',  'IVRV+Liq'),
        ('iv_rv_gap', 'bond_yield', 'IVRV+Bond'),
        ('rsi_div',   'liquidity',  'RSI+Liq'),
        ('bond_yield','rsi_div',    'Bond+RSI'),
        ('consec_dip','iv_rv_gap',  'Dip+IVRV'),
    ]

    # Run all pairs x windows
    print(f"\n[4] Confluence pairs x windows {CONFLUENCE_WINDOWS}...")
    all_results = []

    for primary_key, confirm_key, pair_name in pair_defs:
        for w in CONFLUENCE_WINDOWS:
            label = f"{pair_name} w={w}d"
            entries = wider_window_confluence(all_signals[primary_key], all_signals[confirm_key], w)

            if len(entries) < 5:
                print(f"  {label:25s} | SKIP ({len(entries)} entries)")
                all_results.append({
                    'label': label, 'primary': primary_key, 'confirm': confirm_key,
                    'window': w, 'n_trades': len(entries), 'sharpe': 0, 'wr': 0,
                    'pf': 0, 'perm_p': 1.0, 'passed': False, 'confluence_alpha': 0,
                    'failed_gates': ['insufficient entries'], 'gates_passed': 0,
                })
                continue

            trades_df = run_backtest(entries, data, stock_tickers)
            result = validate_strategy(trades_df, label=label, run_perm=True,
                                       all_signal_entries=entries, data=data,
                                       stock_tickers=stock_tickers)

            solo_a = solo[primary_key]['sharpe']
            solo_b = solo[confirm_key]['sharpe']
            c_alpha = round(result['sharpe'] - max(solo_a, solo_b), 3)

            result['primary'] = primary_key
            result['confirm'] = confirm_key
            result['window'] = w
            result['solo_a_sharpe'] = solo_a
            result['solo_b_sharpe'] = solo_b
            result['confluence_alpha'] = c_alpha

            alpha_tag = "ALPHA" if c_alpha > 0 else "no-alpha"
            tag = "PASS" if result['passed'] else f"FAIL({result['gates_passed']}/5)"
            print(f"  {label:25s} | n={result['n_trades']:4d} | Sharpe={result['sharpe']:6.3f} "
                  f"(vs {solo_a:.3f}/{solo_b:.3f}) ca={c_alpha:+.3f} | "
                  f"WR={result['wr']:.3f} PF={result['pf']:.3f} p={result['perm_p']:.4f} | {tag} {alpha_tag}")

            all_results.append(result)

    # 3-signal: any 2 of 3 (IV-RV + RSI Div + Liquidity) within 5-day window
    print(f"\n[5] 3-signal confluence: any 2-of-3 (IVRV, RSI, Liq) w=5d...")
    trio_entries = any_two_of_three_confluence(
        all_signals['iv_rv_gap'], all_signals['rsi_div'], all_signals['liquidity'], 5
    )
    trio_label = "2of3(IVRV+RSI+Liq) w=5d"
    if len(trio_entries) >= 5:
        trio_trades = run_backtest(trio_entries, data, stock_tickers)
        trio_result = validate_strategy(trio_trades, label=trio_label, run_perm=True,
                                        all_signal_entries=trio_entries, data=data,
                                        stock_tickers=stock_tickers)
        best_solo = max(solo['iv_rv_gap']['sharpe'], solo['rsi_div']['sharpe'], solo['liquidity']['sharpe'])
        trio_result['confluence_alpha'] = round(trio_result['sharpe'] - best_solo, 3)
        trio_result['primary'] = 'iv_rv_gap+rsi_div+liquidity'
        trio_result['confirm'] = 'any-2-of-3'
        trio_result['window'] = 5

        tag = "PASS" if trio_result['passed'] else f"FAIL({trio_result['gates_passed']}/5)"
        print(f"  {trio_label:25s} | n={trio_result['n_trades']:4d} | Sharpe={trio_result['sharpe']:6.3f} "
              f"ca={trio_result['confluence_alpha']:+.3f} | WR={trio_result['wr']:.3f} "
              f"PF={trio_result['pf']:.3f} p={trio_result['perm_p']:.4f} | {tag}")
        all_results.append(trio_result)
    else:
        print(f"  {trio_label}: SKIP ({len(trio_entries)} entries)")
        all_results.append({
            'label': trio_label, 'primary': 'trio', 'confirm': 'any-2-of-3',
            'window': 5, 'n_trades': len(trio_entries), 'sharpe': 0,
            'passed': False, 'confluence_alpha': 0, 'gates_passed': 0,
            'failed_gates': ['insufficient entries'],
        })

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("RESULTS SUMMARY")
    print("=" * 120)

    print("\n── SOLO BASELINES ──")
    print(f"{'Signal':15s} | {'N':>5s} | {'Sharpe':>7s} | {'WR':>5s} | {'PF':>6s} | {'MDD':>8s} | {'RegGap':>6s} | {'p-val':>6s} | Gate")
    print("-" * 85)
    for key, r in solo.items():
        tag = "PASS" if r['passed'] else f"FAIL"
        print(f"{key:15s} | {r['n_trades']:5d} | {r['sharpe']:7.3f} | {r['wr']:.3f} | {r['pf']:6.3f} | "
              f"{r['mdd']:8.2f} | {r['regime_gap']:6.3f} | {r['perm_p']:.4f} | {tag}")

    print(f"\n── CONFLUENCE PAIRS (sorted by Sharpe) ──")
    header = (f"{'Pair':25s} | {'N':>4s} | {'Sharpe':>7s} | {'SoloA':>6s} | {'SoloB':>6s} | "
              f"{'CAlpha':>7s} | {'WR':>5s} | {'PF':>6s} | {'p-val':>6s} | {'RGap':>5s} | {'Gate':>8s}")
    print(header)
    print("-" * len(header))

    sorted_r = sorted(all_results, key=lambda x: x.get('sharpe', 0), reverse=True)
    for r in sorted_r:
        ca = r.get('confluence_alpha', 0)
        ca_tag = f"{ca:+.3f}" if ca != 0 else " 0.000"
        tag = "5/5 PASS" if r.get('passed', False) else f"{r.get('gates_passed',0)}/5"
        solo_a = r.get('solo_a_sharpe', '-')
        solo_b = r.get('solo_b_sharpe', '-')
        if isinstance(solo_a, (int, float)):
            solo_a = f"{solo_a:6.3f}"
        if isinstance(solo_b, (int, float)):
            solo_b = f"{solo_b:6.3f}"
        print(f"{r['label']:25s} | {r['n_trades']:4d} | {r.get('sharpe',0):7.3f} | {solo_a:>6s} | {solo_b:>6s} | "
              f"{ca_tag:>7s} | {r.get('wr',0):.3f} | {r.get('pf',0):6.3f} | {r.get('perm_p',1):.4f} | "
              f"{r.get('regime_gap',0):5.3f} | {tag:>8s}")

    # Highlight winners
    winners = [r for r in sorted_r if r.get('passed', False) and r.get('confluence_alpha', 0) > 0]
    print(f"\n── WINNERS: 5/5 gates + positive confluence alpha ──")
    if winners:
        for r in winners:
            print(f"  ** {r['label']}: Sharpe={r['sharpe']:.3f}, CAlpha={r['confluence_alpha']:+.3f}, "
                  f"WR={r['wr']:.1%}, PF={r['pf']:.2f}, p={r['perm_p']:.4f}")
    else:
        print("  None found.")
        # Show best near-misses
        near = [r for r in sorted_r if r.get('sharpe', 0) > 0.3 and r.get('n_trades', 0) >= 10]
        if near:
            print("  Near-misses (Sharpe > 0.3, 10+ trades):")
            for r in near[:5]:
                ca = r.get('confluence_alpha', 0)
                print(f"    {r['label']}: Sharpe={r.get('sharpe',0):.3f}, CAlpha={ca:+.3f}, "
                      f"WR={r.get('wr',0):.1%}, gates={r.get('gates_passed',0)}/5, "
                      f"failed={r.get('failed_gates',[])}")

    # Passed 5/5 but no alpha
    passed_no_alpha = [r for r in sorted_r if r.get('passed', False) and r.get('confluence_alpha', 0) <= 0]
    if passed_no_alpha:
        print(f"\n  Passed 5/5 but no confluence alpha (just filtering, not improving):")
        for r in passed_no_alpha:
            print(f"    {r['label']}: Sharpe={r['sharpe']:.3f}, CAlpha={r.get('confluence_alpha',0):+.3f}")

    # Window analysis — does wider window help?
    print(f"\n── WINDOW ANALYSIS (avg Sharpe by window size) ──")
    for w in CONFLUENCE_WINDOWS:
        w_results = [r for r in all_results if r.get('window') == w and r.get('n_trades', 0) >= 10]
        if w_results:
            avg_sharpe = np.mean([r['sharpe'] for r in w_results])
            avg_wr = np.mean([r.get('wr', 0) for r in w_results])
            avg_ca = np.mean([r.get('confluence_alpha', 0) for r in w_results])
            n_pass = sum(1 for r in w_results if r.get('passed', False))
            print(f"  w={w}d: avg_Sharpe={avg_sharpe:.3f}, avg_WR={avg_wr:.3f}, "
                  f"avg_CAlpha={avg_ca:+.3f}, passed={n_pass}/{len(w_results)}")

    # Save
    def _clean(v):
        if isinstance(v, (pd.Timestamp,)):
            return str(v)
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v

    save_data = {
        'run_date': datetime.now().isoformat(),
        'config': {
            'universe': UNIVERSE, 'backtest_start': BACKTEST_START, 'end': END_DATE,
            'pos_size': POS_SIZE, 'max_concurrent': MAX_CONCURRENT, 'hold_days': HOLD_DAYS,
            'windows_tested': CONFLUENCE_WINDOWS, 'n_perms': N_PERMS,
            'gates': {'sharpe': SHARPE_GATE, 'wr': WR_GATE, 'pf': PF_GATE,
                      'regime_gap': REGIME_GAP_LIMIT, 'perm_p': 0.05},
        },
        'solo': {k: {kk: _clean(vv) for kk, vv in v.items()} for k, v in solo.items()},
        'confluence': [{k: _clean(v) for k, v in r.items()} for r in sorted_r],
        'winners': [{k: _clean(v) for k, v in r.items()} for r in winners],
    }
    out_file = OUTPUT_DIR / 'wider_window_results.json'
    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")
    print("\nDONE")


if __name__ == '__main__':
    main()
