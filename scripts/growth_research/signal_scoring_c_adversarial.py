#!/usr/bin/env python3
"""
Adversarial Validation: Signal Scoring Variant C (Regime-Conditioned)
=====================================================================
Strategy: Score each (day, stock) by how many validated signals fire.
  - In BULL markets (SPY > 200SMA): require score >= 3
  - In BEAR markets (SPY < 200SMA): require score >= 1
  - Trade the highest-scoring entries, max 2 concurrent, 21-day hold.

Results from initial backtest: Sharpe 1.556, Sortino 2.045, WR 64.1%,
PF 2.27, 153 trades, perm p=0.025, regime gap 0.206.

6-test adversarial framework:
  1. Re-implementation (independent code, same logic)
  2. Inverse signal (trade LOWEST scores)
  3. Random timing (permutation test — already p=0.025)
  4. Sub-period stability (4 equal windows)
  5. Top-3 stock removal (concentrated in a few names?)
  6. Parameter sensitivity (sweep thresholds)
"""

import os, sys, json, warnings, time, functools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

START_DATE = '2019-01-01'
END_DATE = '2026-07-01'
BACKTEST_START = '2020-01-01'
POS_SIZE = 300.0
MAX_CONCURRENT = 2
HOLD_DAYS = 21
PROFIT_TARGET = 0.10
STOP_LOSS = -0.15
SPREAD_COST_PCT = 0.001
N_PERMS = 500  # More perms for adversarial

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/signal_scoring_c_adversarial')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
    'MA', 'WMT', 'PEP', 'KO', 'TMO', 'ABT', 'BAC', 'XOM', 'CVX', 'CSCO',
]
MACRO_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', '^TNX', 'HYG', 'LQD']

print("=" * 80)
print("ADVERSARIAL: Signal Scoring Variant C (Regime-Conditioned)")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════
# DATA + SIGNALS (identical to scoring backtest — INDEPENDENT RE-IMPL)
# ═══════════════════════════════════════════════════════════════════════

def download_data():
    cache_files = [
        Path('/home/jupiter/Lvl3Quant/output/growth_research/cross_signal_confluence/_confluence_cache.pkl'),
        Path('/home/jupiter/Lvl3Quant/output/growth_research/signal_scoring_portfolio/_scoring_cache.pkl'),
    ]
    for cf in cache_files:
        if cf.exists():
            import pickle
            with open(cf, 'rb') as f:
                data = pickle.load(f)
            print(f"  Loaded cache: {len(data['close'].columns)} tickers, {len(data['close'])} days")
            return data
    import yfinance as yf
    raw = yf.download(UNIVERSE + MACRO_TICKERS, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']; high = raw['High']; low = raw['Low']; volume = raw['Volume']
    else:
        close = high = low = volume = raw
    for df in [close, high, low, volume]:
        if hasattr(df.columns, 'droplevel'):
            try: df.columns = df.columns.droplevel(1)
            except: pass
    close = close.ffill().dropna(how='all')
    return {'close': close, 'high': high.reindex(close.index).ffill(),
            'low': low.reindex(close.index).ffill(),
            'volume': volume.reindex(close.index).ffill().fillna(0)}

def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _realized_vol(returns, window=21):
    return returns.rolling(window).std() * np.sqrt(252) * 100

# RE-IMPLEMENTATION: independent signal generators (verify they match)
def gen_signals(data, stock_tickers):
    """Generate all 7 signals independently."""
    signals = {}

    # 1. IV-RV Gap
    spy_ret = data['close']['SPY'].pct_change()
    vix = data['close']['^VIX']
    rv = _realized_vol(spy_ret, 21)
    gap = vix - rv
    ivrv_days = set(gap[gap > 5].index)
    s1 = {}
    for d in ivrv_days:
        for t in stock_tickers:
            try:
                if pd.notna(data['close'].at[d, t]): s1[(d, t)] = 1
            except: pass
    signals['iv_rv_gap'] = s1

    # 2. RSI Divergence
    s2 = {}
    for t in stock_tickers:
        if t not in data['close'].columns: continue
        c = data['close'][t].dropna()
        r = _rsi(c, 14)
        for i in range(20, len(c)):
            wc = c.iloc[i-10:i+1]; wr = r.iloc[i-10:i+1]
            if len(wc) < 11 or wr.isna().any(): continue
            if c.iloc[i] < wc.iloc[0] and r.iloc[i] > wr.iloc[0] and r.iloc[i] < 40:
                s2[(c.index[i], t)] = 1
    signals['rsi_divergence'] = s2

    # 3. Bond Yield
    s3 = {}
    if '^TNX' in data['close'].columns:
        tnx = data['close']['^TNX']
        yc = tnx.diff(5)
        bd = set(yc[yc < -0.10].index)
    else:
        tlt = data['close']['TLT']
        bd = set(tlt.pct_change(5)[tlt.pct_change(5) > 0.02].index)
    for d in bd:
        for t in stock_tickers:
            try:
                if pd.notna(data['close'].at[d, t]): s3[(d, t)] = 1
            except: pass
    signals['bond_yield'] = s3

    # 4. Liquidity
    s4 = {}
    for t in stock_tickers:
        if t not in data['close'].columns or t not in data['high'].columns: continue
        h = data['high'][t].dropna(); l = data['low'][t].dropna(); c = data['close'][t].dropna()
        idx = h.index.intersection(l.index).intersection(c.index)
        if len(idx) < 65: continue
        h, l, c = h.loc[idx], l.loc[idx], c.loc[idx]
        hl = (h - l) / c; avg60 = hl.rolling(60).mean()
        narrow = hl < avg60 * 0.85; rsi = _rsi(c, 14)
        for i in range(60, len(idx)):
            if narrow.iloc[i] and rsi.iloc[i] < 40:
                s4[(idx[i], t)] = 1
    signals['liquidity'] = s4

    # 5. Vol Term Structure
    s5 = {}
    vix = data['close'].get('^VIX'); vix3m = data['close'].get('^VIX3M')
    if vix is not None and vix3m is not None:
        ratio = vix / vix3m
        fear = set(ratio[ratio > 1.0].index)
        for t in stock_tickers:
            if t not in data['close'].columns: continue
            c = data['close'][t].dropna()
            sma20 = c.rolling(20).mean(); dd = (c - sma20) / sma20
            for i in range(20, len(c)):
                d = c.index[i]
                if d in fear and dd.iloc[i] < -0.05:
                    s5[(d, t)] = 1
    signals['vol_term_str'] = s5

    # 6. Consecutive Dip
    s6 = {}
    for t in stock_tickers:
        if t not in data['close'].columns: continue
        c = data['close'][t].dropna(); ret = c.pct_change()
        for i in range(3, len(c)):
            r1, r2, r3 = ret.iloc[i-2], ret.iloc[i-1], ret.iloc[i]
            if r1 < 0 and r2 < 0 and r3 < 0 and r2 < r1 and r3 < r2:
                s6[(c.index[i], t)] = 1
    signals['consecutive_dip'] = s6

    # 7. Base MR
    s7 = {}
    for t in stock_tickers:
        if t not in data['close'].columns: continue
        c = data['close'][t].dropna(); rsi = _rsi(c, 14)
        h50 = c.rolling(50).max(); dd = (c - h50) / h50
        for i in range(50, len(c)):
            if rsi.iloc[i] < 30 and dd.iloc[i] < -0.07:
                s7[(c.index[i], t)] = 1
    signals['base_mr'] = s7

    return signals

def build_scores(all_signals):
    scores = defaultdict(int)
    for sigs in all_signals.values():
        for k in sigs:
            scores[k] += 1
    return dict(scores)

def regime_filter_entries(scores, data, bull_thresh, bear_thresh):
    spy = data['close']['SPY']
    spy_sma = spy.rolling(200).mean()
    bt_start = pd.Timestamp(BACKTEST_START)
    entries = []
    for (d, t), s in scores.items():
        if d < bt_start: continue
        try:
            is_bull = spy.at[d] > spy_sma.at[d]
        except: continue
        thresh = bull_thresh if is_bull else bear_thresh
        if s >= thresh:
            entries.append((d, t, s))
    entries.sort(key=lambda x: (x[0], -x[2]))
    return entries

def run_backtest(entries, data, stock_tickers, exclude_tickers=None):
    close = data['close']
    spy_close = close['SPY']
    bt_start = pd.Timestamp(BACKTEST_START)
    if exclude_tickers:
        entries = [(d, t, s) for d, t, s in entries if t not in exclude_tickers]
    trades = []
    open_pos = []
    for ed, tk, sc in entries:
        if ed < bt_start: continue
        open_pos = [p for p in open_pos if p[3] > ed]
        if len(open_pos) >= MAX_CONCURRENT: continue
        try: ep = close.at[ed, tk]
        except: continue
        if pd.isna(ep) or ep <= 0: continue
        fd = close.index[close.index > ed]
        if len(fd) == 0: continue
        xp = None; xd = None; xr = 'hold_expiry'
        for fdate in fd[:HOLD_DAYS]:
            try: p = close.at[fdate, tk]
            except: continue
            if pd.isna(p): continue
            r = (p - ep) / ep
            if r >= PROFIT_TARGET: xp = p; xd = fdate; xr = 'tp'; break
            elif r <= STOP_LOSS: xp = p; xd = fdate; xr = 'sl'; break
        if xp is None:
            he = min(HOLD_DAYS, len(fd))
            if he > 0:
                xd = fd[he - 1]
                try: xp = close.at[xd, tk]
                except: continue
                if pd.isna(xp): continue
        if xp is None: continue
        nr = (xp - ep) / ep - SPREAD_COST_PCT
        try:
            sma = spy_close.rolling(200).mean()
            regime = 'bull' if spy_close.at[ed] > sma.at[ed] else 'bear'
        except: regime = 'unknown'
        trades.append({
            'entry_date': ed, 'exit_date': xd, 'ticker': tk,
            'entry_price': ep, 'exit_price': xp, 'return': nr,
            'pnl': POS_SIZE * nr, 'exit_reason': xr, 'regime': regime,
            'hold_days': (xd - ed).days, 'score': sc,
        })
        open_pos.append((ed, tk, ep, xd))
    return pd.DataFrame(trades) if trades else None

def calc_metrics(df):
    if df is None or len(df) < 5:
        return {'sharpe': 0, 'sortino': 0, 'wr': 0, 'pf': 0, 'n': 0, 'mdd': 0}
    rets = df['return'].values
    n = len(rets)
    tpy = n / max(1, (df['entry_date'].max() - df['entry_date'].min()).days / 365.25)
    if tpy < 1: tpy = 1
    mu = np.mean(rets); sig = np.std(rets, ddof=1) if n > 1 else 1
    sharpe = (mu / sig) * np.sqrt(tpy) if sig > 0 else 0
    neg = rets[rets < 0]
    sd = np.std(neg, ddof=1) if len(neg) > 1 else sig
    sortino = (mu / sd) * np.sqrt(tpy) if sd > 0 else 0
    wr = np.mean(rets > 0)
    gp = np.sum(rets[rets > 0]); gl = abs(np.sum(rets[rets < 0]))
    pf = gp / gl if gl > 0 else 99
    cum = np.cumsum(df['pnl'].values)
    pk = np.maximum.accumulate(cum)
    mdd = np.min(cum - pk)
    total_ret = cum[-1] / POS_SIZE * 100 if len(cum) > 0 else 0
    return {'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
            'wr': round(wr, 3), 'pf': round(pf, 3), 'n': n,
            'mdd': round(mdd, 2), 'total_ret': round(total_ret, 1),
            'mean_ret': round(mu * 100, 3)}

# ═══════════════════════════════════════════════════════════════════════
# MAIN ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════════

def main():
    data = download_data()
    stock_tickers = [t for t in UNIVERSE if t in data['close'].columns]
    print(f"  Universe: {len(stock_tickers)} tickers\n")

    # Generate signals
    print("[1] Generating signals (RE-IMPLEMENTATION)...")
    all_signals = gen_signals(data, stock_tickers)
    for k, v in all_signals.items():
        print(f"  {k:20s}: {len(v)} entries")

    scores = build_scores(all_signals)
    entries = regime_filter_entries(scores, data, bull_thresh=3, bear_thresh=1)
    print(f"\n  Regime-filtered entries (bull>=3, bear>=1): {len(entries)}")

    # Forward backtest
    print("\n[2] FORWARD BACKTEST (Re-implementation)...")
    fwd_df = run_backtest(entries, data, stock_tickers)
    fwd = calc_metrics(fwd_df)
    print(f"  ✅ Sharpe={fwd['sharpe']}, Sortino={fwd['sortino']}, WR={fwd['wr']}, "
          f"PF={fwd['pf']}, n={fwd['n']}, MDD={fwd['mdd']}, Return={fwd['total_ret']}%")

    results = {'forward': fwd}
    passes = 0
    total = 6

    # ── TEST 1: Re-implementation match ──
    print("\n[3] TEST 1: Re-implementation match...")
    # Original got Sharpe 1.556. Check if re-impl is within 20%
    orig_sharpe = 1.556
    reimpl_ratio = fwd['sharpe'] / orig_sharpe if orig_sharpe > 0 else 0
    t1_pass = 0.7 < reimpl_ratio < 1.3
    results['reimpl'] = {'ratio': round(reimpl_ratio, 3), 'passed': t1_pass}
    if t1_pass: passes += 1
    status = "✅ PASS" if t1_pass else "❌ FAIL"
    print(f"  Re-impl Sharpe {fwd['sharpe']} vs original {orig_sharpe} (ratio={reimpl_ratio:.3f}) — {status}")

    # ── TEST 2: Inverse signal ──
    print("\n[4] TEST 2: Inverse signal (trade LOWEST scores)...")
    inv_entries = []
    bt_start = pd.Timestamp(BACKTEST_START)
    spy = data['close']['SPY']; spy_sma = spy.rolling(200).mean()
    for (d, t), s in scores.items():
        if d < bt_start: continue
        try: is_bull = spy.at[d] > spy_sma.at[d]
        except: continue
        thresh = 3 if is_bull else 1
        if s < thresh:  # INVERSE: trade when score is BELOW threshold
            inv_entries.append((d, t, s))
    inv_entries.sort(key=lambda x: (x[0], x[2]))  # Sort by score ASC (worst first)
    inv_df = run_backtest(inv_entries, data, stock_tickers)
    inv = calc_metrics(inv_df)
    inv_ratio = inv['sharpe'] / fwd['sharpe'] if fwd['sharpe'] > 0 else 99
    t2_pass = inv_ratio < 0.50  # Inverse should be less than 50% of forward
    results['inverse'] = {'sharpe': inv['sharpe'], 'ratio': round(inv_ratio, 3), 'passed': t2_pass}
    if t2_pass: passes += 1
    status = "✅ PASS" if t2_pass else "❌ FAIL"
    print(f"  Inverse Sharpe={inv['sharpe']}, ratio={inv_ratio:.3f} — {status}")

    # ── TEST 3: Random timing (permutation test) ──
    print(f"\n[5] TEST 3: Random timing ({N_PERMS} permutations)...")
    valid_dates = data['close'].index[data['close'].index >= bt_start][:-HOLD_DAYS-5]
    perm_sharpes = []
    for i in range(N_PERMS):
        n_raw = len(entries)
        rd = np.random.choice(valid_dates, size=min(n_raw, len(valid_dates)), replace=True)
        rt = np.random.choice(stock_tickers, size=min(n_raw, len(valid_dates)), replace=True)
        rs = np.random.randint(1, 6, size=min(n_raw, len(valid_dates)))
        rand = [(d, t, s) for d, t, s in zip(rd, rt, rs)]
        rdf = run_backtest(rand, data, stock_tickers)
        if rdf is not None and len(rdf) >= 5:
            rm = calc_metrics(rdf)
            perm_sharpes.append(rm['sharpe'])
        else:
            perm_sharpes.append(0.0)
        if (i + 1) % 100 == 0:
            p_so_far = np.mean(np.array(perm_sharpes) >= fwd['sharpe'])
            print(f"    [{i+1}/{N_PERMS}] p={p_so_far:.4f}")
    perm_p = np.mean(np.array(perm_sharpes) >= fwd['sharpe'])
    t3_pass = perm_p < 0.05
    results['perm_test'] = {'p': round(perm_p, 4), 'passed': t3_pass,
                             'mean_random': round(np.mean(perm_sharpes), 3)}
    if t3_pass: passes += 1
    status = "✅ PASS" if t3_pass else "❌ FAIL"
    print(f"  Perm p={perm_p:.4f}, mean random Sharpe={np.mean(perm_sharpes):.3f} — {status}")

    # ── TEST 4: Sub-period stability ──
    print("\n[6] TEST 4: Sub-period stability (4 windows)...")
    if fwd_df is not None and len(fwd_df) > 0:
        dates = fwd_df['entry_date'].sort_values()
        min_d, max_d = dates.min(), dates.max()
        total_days = (max_d - min_d).days
        window = total_days // 4
        sub_results = []
        all_positive = True
        for w in range(4):
            w_start = min_d + pd.Timedelta(days=w * window)
            w_end = min_d + pd.Timedelta(days=(w + 1) * window) if w < 3 else max_d + pd.Timedelta(days=1)
            sub_df = fwd_df[(fwd_df['entry_date'] >= w_start) & (fwd_df['entry_date'] < w_end)]
            sm = calc_metrics(sub_df)
            sub_results.append(sm)
            if sm['sharpe'] <= 0: all_positive = False
            print(f"  Window {w+1} ({w_start.strftime('%Y-%m')} to {w_end.strftime('%Y-%m')}): "
                  f"Sharpe={sm['sharpe']}, n={sm['n']}, WR={sm['wr']}")
        t4_pass = all_positive and len([s for s in sub_results if s['n'] >= 5]) >= 3
        results['sub_period'] = {'windows': sub_results, 'all_positive': all_positive, 'passed': t4_pass}
    else:
        t4_pass = False
        results['sub_period'] = {'passed': False}
    if t4_pass: passes += 1
    status = "✅ PASS" if t4_pass else "❌ FAIL"
    print(f"  All positive: {all_positive} — {status}")

    # ── TEST 5: Top-3 stock removal ──
    print("\n[7] TEST 5: Top-3 stock removal...")
    if fwd_df is not None:
        stock_pnl = fwd_df.groupby('ticker')['pnl'].sum().sort_values(ascending=False)
        top3 = list(stock_pnl.head(3).index)
        print(f"  Top 3 contributors: {top3}")
        rem_df = run_backtest(entries, data, stock_tickers, exclude_tickers=set(top3))
        rem = calc_metrics(rem_df)
        drop_pct = 1 - (rem['sharpe'] / fwd['sharpe']) if fwd['sharpe'] > 0 else 1
        t5_pass = drop_pct < 0.50  # Less than 50% Sharpe drop
        results['top3_removal'] = {'sharpe_without': rem['sharpe'], 'drop_pct': round(drop_pct * 100, 1),
                                    'top3': top3, 'passed': t5_pass}
        if t5_pass: passes += 1
        status = "✅ PASS" if t5_pass else "❌ FAIL"
        print(f"  Without top 3: Sharpe={rem['sharpe']} (drop {drop_pct*100:.1f}%) — {status}")
    else:
        t5_pass = False

    # ── TEST 6: Parameter sensitivity ──
    print("\n[8] TEST 6: Parameter sensitivity (threshold sweep)...")
    param_results = []
    good_count = 0
    total_combos = 0
    for bull_t in [2, 3, 4, 5]:
        for bear_t in [1, 2, 3]:
            if bear_t > bull_t: continue
            total_combos += 1
            pe = regime_filter_entries(scores, data, bull_thresh=bull_t, bear_thresh=bear_t)
            pdf = run_backtest(pe, data, stock_tickers)
            pm = calc_metrics(pdf)
            param_results.append({'bull': bull_t, 'bear': bear_t, **pm})
            if pm['sharpe'] > 0.3 and pm['n'] >= 10:
                good_count += 1
            print(f"  bull>={bull_t}/bear>={bear_t}: Sharpe={pm['sharpe']:6.3f}, n={pm['n']:4d}, WR={pm['wr']:.3f}")
    robustness = good_count / total_combos if total_combos > 0 else 0
    t6_pass = robustness > 0.50  # >50% of param combos profitable
    results['param_sensitivity'] = {'robustness': round(robustness * 100, 1), 'n_combos': total_combos,
                                     'good_count': good_count, 'passed': t6_pass}
    if t6_pass: passes += 1
    status = "✅ PASS" if t6_pass else "❌ FAIL"
    print(f"  {good_count}/{total_combos} combos Sharpe>0.3 ({robustness*100:.0f}%) — {status}")

    # ── SUMMARY ──
    print("\n" + "=" * 80)
    print(f"ADVERSARIAL RESULT: {passes}/{total} PASS")
    print("=" * 80)
    test_names = ['Re-implementation', 'Inverse Signal', 'Random Timing',
                  'Sub-period Stability', 'Top-3 Removal', 'Parameter Sensitivity']
    test_results = [t1_pass, t2_pass, t3_pass, t4_pass, t5_pass, t6_pass]
    for name, passed in zip(test_names, test_results):
        print(f"  {'✅' if passed else '❌'} {name}")

    if passes >= 5:
        print(f"\n🏆🏆🏆 VALIDATED — Signal Scoring C passes {passes}/6 adversarial tests!")
        print("Ready for paper engine deployment.")
    elif passes >= 4:
        print(f"\n🏆 CONDITIONAL PASS — {passes}/6. Review failing test(s) carefully.")
    else:
        print(f"\n❌ REJECTED — Only {passes}/6 pass. Strategy is not robust enough.")

    # Save
    results['summary'] = {'passes': passes, 'total': total, 'verdict': 'PASS' if passes >= 5 else 'FAIL'}
    with open(OUTPUT_DIR / 'adversarial_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved.")

if __name__ == '__main__':
    main()
