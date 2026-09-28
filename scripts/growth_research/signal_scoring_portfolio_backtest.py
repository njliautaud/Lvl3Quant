#!/usr/bin/env python3
"""
Signal Scoring Portfolio Backtest (HC #772 — Alternative to Confluence)
=======================================================================
Instead of INTERSECTING signals (which kills trade count), SCORE each day
by how many validated signals are firing. Higher score = bigger position.

Hypothesis: the intersection approach fails because requiring 2+ signals
eliminates most trades. But the times when 3+ signals align should still
be highest conviction. This tests ADDITIVE scoring vs binary confluence.

Variants tested:
  A) Equal-weight scoring (each signal = +1 point, trade when score >= threshold)
  B) Sharpe-weighted scoring (signals weighted by their solo Sharpe)
  C) Regime-conditioned scoring (different thresholds in bull/bear)
  D) Top-N concentration (only trade the highest-scoring stock each day)
  E) Inverse confidence test (trade LOWEST score as control)
  F) Dynamic sizing (position size proportional to score)
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
BASE_POS_SIZE = 300.0
MAX_CONCURRENT = 2
HOLD_DAYS = 21
PROFIT_TARGET = 0.10
STOP_LOSS = -0.15
SPREAD_COST_PCT = 0.001
N_PERMS = 200
REGIME_GAP_LIMIT = 0.50

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/signal_scoring_portfolio')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
    'MA', 'WMT', 'PEP', 'KO', 'TMO', 'ABT', 'BAC', 'XOM', 'CVX', 'CSCO',
]

MACRO_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', '^TNX', 'HYG', 'LQD']

print("=" * 80)
print("SIGNAL SCORING PORTFOLIO BACKTEST (HC #772 — Additive Scoring)")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════════════
def download_data():
    import yfinance as yf
    cache_files = [
        Path('/home/jupiter/Lvl3Quant/output/growth_research/cross_signal_confluence/_confluence_cache.pkl'),
        Path('/home/jupiter/Lvl3Quant/output/growth_research/cross_signal_confluence_v2/_confluence_v2_cache.pkl'),
        OUTPUT_DIR / '_scoring_cache.pkl',
    ]
    for cache_file in cache_files:
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
        close = raw['Close']; high = raw['High']; low = raw['Low']; volume = raw['Volume']
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
    with open(OUTPUT_DIR / '_scoring_cache.pkl', 'wb') as f:
        pickle.dump(data, f)
    return data

# ═══════════════════════════════════════════════════════════════════════
# 2. SIGNAL GENERATORS (all validated + dead solos)
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
    spy_ret = data['close']['SPY'].pct_change()
    vix = data['close']['^VIX']
    rv_21 = _realized_vol(spy_ret, 21)
    gap = vix - rv_21
    fire_days = set(gap[gap > 5].index)
    signals = {}
    for day in fire_days:
        for t in stock_tickers:
            try:
                if pd.notna(data['close'].at[day, t]):
                    signals[(day, t)] = 1.0
            except: pass
    return signals

def signal_rsi_divergence(data, stock_tickers):
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns: continue
        close = data['close'][t].dropna()
        rsi = _rsi(close, 14)
        for i in range(20, len(close)):
            wc = close.iloc[i-10:i+1]; wr = rsi.iloc[i-10:i+1]
            if len(wc) < 11 or wr.isna().any(): continue
            if close.iloc[i] < wc.iloc[0] and rsi.iloc[i] > wr.iloc[0] and rsi.iloc[i] < 40:
                signals[(close.index[i], t)] = 1.0
    return signals

def signal_bond_yield(data, stock_tickers):
    if '^TNX' not in data['close'].columns:
        tlt = data['close']['TLT']
        fire_days = set(tlt.pct_change(5)[tlt.pct_change(5) > 0.02].index)
    else:
        tnx = data['close']['^TNX']
        fire_days = set(tnx.diff(5)[tnx.diff(5) < -0.10].index)
    signals = {}
    for day in fire_days:
        for t in stock_tickers:
            try:
                if pd.notna(data['close'].at[day, t]):
                    signals[(day, t)] = 1.0
            except: pass
    return signals

def signal_liquidity(data, stock_tickers):
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns or t not in data['high'].columns: continue
        high = data['high'][t].dropna(); low = data['low'][t].dropna(); close = data['close'][t].dropna()
        idx = high.index.intersection(low.index).intersection(close.index)
        if len(idx) < 65: continue
        high, low, close = high.loc[idx], low.loc[idx], close.loc[idx]
        hl_spread = (high - low) / close
        avg_60 = hl_spread.rolling(60).mean()
        narrow = hl_spread < avg_60 * 0.85
        rsi = _rsi(close, 14)
        for i in range(60, len(idx)):
            if narrow.iloc[i] and rsi.iloc[i] < 40:
                signals[(idx[i], t)] = 1.0
    return signals

def signal_vol_term_structure(data, stock_tickers):
    vix = data['close'].get('^VIX'); vix3m = data['close'].get('^VIX3M')
    if vix is None or vix3m is None: return {}
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
                signals[(day, t)] = 1.0
    return signals

def signal_consecutive_dip(data, stock_tickers):
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns: continue
        close = data['close'][t].dropna(); ret = close.pct_change()
        for i in range(3, len(close)):
            r1, r2, r3 = ret.iloc[i-2], ret.iloc[i-1], ret.iloc[i]
            if r1 < 0 and r2 < 0 and r3 < 0 and r2 < r1 and r3 < r2:
                signals[(close.index[i], t)] = 1.0
    return signals

def signal_base_mr(data, stock_tickers):
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns: continue
        close = data['close'][t].dropna(); rsi = _rsi(close, 14)
        high_50 = close.rolling(50).max(); drawdown = (close - high_50) / high_50
        for i in range(50, len(close)):
            if rsi.iloc[i] < 30 and drawdown.iloc[i] < -0.07:
                signals[(close.index[i], t)] = 1.0
    return signals

# ═══════════════════════════════════════════════════════════════════════
# 3. SCORE BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_daily_scores(all_signals, weights=None):
    """Build a (date, ticker) -> score mapping by summing signals."""
    scores = defaultdict(float)
    for sig_name, sig_dict in all_signals.items():
        w = weights.get(sig_name, 1.0) if weights else 1.0
        for (day, ticker), val in sig_dict.items():
            scores[(day, ticker)] += w * val
    return dict(scores)

# ═══════════════════════════════════════════════════════════════════════
# 4. BACKTESTER
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(entries_with_scores, data, stock_tickers, pos_size=BASE_POS_SIZE,
                 dynamic_sizing=False, max_score=7.0):
    """Run backtest on scored entries. entries_with_scores = [(date, ticker, score), ...]"""
    close = data['close']
    spy_close = close['SPY']
    bt_start = pd.Timestamp(BACKTEST_START)
    entries = [(d, t, s) for d, t, s in entries_with_scores if d >= bt_start]
    entries.sort(key=lambda x: (x[0], -x[2]))  # Sort by date, then score descending
    if not entries: return None
    trades = []
    open_positions = []
    for entry_date, ticker, score in entries:
        open_positions = [p for p in open_positions if p[3] > entry_date]
        if len(open_positions) >= MAX_CONCURRENT: continue
        try: entry_price = close.at[entry_date, ticker]
        except: continue
        if pd.isna(entry_price) or entry_price <= 0: continue
        future_dates = close.index[close.index > entry_date]
        if len(future_dates) == 0: continue

        # Dynamic sizing: scale position by score
        if dynamic_sizing:
            size_mult = min(score / max_score * 2.0, 2.0)  # 0.5x to 2x
            actual_pos = pos_size * max(0.5, size_mult)
        else:
            actual_pos = pos_size

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
        pnl = actual_pos * net_ret
        try:
            spy_sma = spy_close.rolling(200).mean()
            spy_regime = 'bull' if spy_close.at[entry_date] > spy_sma.at[entry_date] else 'bear'
        except: spy_regime = 'unknown'
        trades.append({
            'entry_date': entry_date, 'exit_date': exit_date, 'ticker': ticker,
            'entry_price': entry_price, 'exit_price': exit_price,
            'return': net_ret, 'pnl': pnl, 'exit_reason': exit_reason,
            'regime': spy_regime, 'hold_days': (exit_date - entry_date).days,
            'score': score, 'pos_size': actual_pos,
        })
        open_positions.append((entry_date, ticker, entry_price, exit_date))
    if not trades: return None
    return pd.DataFrame(trades)

# ═══════════════════════════════════════════════════════════════════════
# 5. VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def validate_strategy(trades_df, label='', run_perm=True, all_entries=None, data=None, stock_tickers=None):
    if trades_df is None or len(trades_df) < 10:
        return {
            'label': label, 'n_trades': 0 if trades_df is None else len(trades_df),
            'sharpe': 0, 'sortino': 0, 'wr': 0, 'pf': 0, 'mdd': 0,
            'regime_gap': 1.0, 'perm_p': 1.0, 'passed': False,
            'reason': 'insufficient trades'
        }
    rets = trades_df['return'].values
    n = len(rets)
    tpy = n / max(1, (trades_df['entry_date'].max() - trades_df['entry_date'].min()).days / 365.25)
    if tpy < 1: tpy = 1
    mean_ret = np.mean(rets); std_ret = np.std(rets, ddof=1) if n > 1 else 1.0
    sharpe = (mean_ret / std_ret) * np.sqrt(tpy) if std_ret > 0 else 0
    neg_rets = rets[rets < 0]
    sortino_denom = np.std(neg_rets, ddof=1) if len(neg_rets) > 1 else std_ret
    sortino = (mean_ret / sortino_denom) * np.sqrt(tpy) if sortino_denom > 0 else 0
    wr = np.mean(rets > 0)
    gp = np.sum(rets[rets > 0]); gl = np.abs(np.sum(rets[rets < 0]))
    pf = gp / gl if gl > 0 else 99.0
    cum_pnl = np.cumsum(trades_df['pnl'].values)
    peak = np.maximum.accumulate(cum_pnl); dd = cum_pnl - peak
    mdd = np.min(dd) if len(dd) > 0 else 0
    bull = trades_df[trades_df['regime'] == 'bull']; bear = trades_df[trades_df['regime'] == 'bear']
    if len(bull) >= 3 and len(bear) >= 3:
        bs = np.mean(bull['return']) / max(np.std(bull['return'], ddof=1), 1e-6)
        brs = np.mean(bear['return']) / max(np.std(bear['return'], ddof=1), 1e-6)
        regime_gap = abs(bs - brs) / max(abs(bs), abs(brs), 1e-6)
    else: regime_gap = 0.0

    perm_p = 1.0
    if run_perm and n >= 10 and all_entries is not None and data is not None:
        bt_start = pd.Timestamp(BACKTEST_START)
        valid_dates = data['close'].index[data['close'].index >= bt_start][:-HOLD_DAYS-5]
        perm_sharpes = []
        for _ in range(N_PERMS):
            n_raw = len(all_entries)
            rd = np.random.choice(valid_dates, size=min(n_raw, len(valid_dates)), replace=True)
            rt = np.random.choice(stock_tickers, size=min(n_raw, len(valid_dates)), replace=True)
            rs = np.random.uniform(1, 5, size=min(n_raw, len(valid_dates)))
            rand_entries = [(d, t, s) for d, t, s in zip(rd, rt, rs)]
            rand_trades = run_backtest(rand_entries, data, stock_tickers)
            if rand_trades is not None and len(rand_trades) >= 5:
                rr = rand_trades['return'].values
                r_tpy = len(rr) / max(1, (rand_trades['entry_date'].max() - rand_trades['entry_date'].min()).days / 365.25)
                if r_tpy < 1: r_tpy = 1
                r_sharpe = (np.mean(rr) / max(np.std(rr, ddof=1), 1e-8)) * np.sqrt(r_tpy)
            else: r_sharpe = 0.0
            perm_sharpes.append(r_sharpe)
        perm_p = np.mean(np.array(perm_sharpes) >= sharpe)

    gates = {'sharpe': sharpe > 0.3, 'wr': wr > 0.45, 'pf': pf > 1.0,
             'mdd': mdd > -BASE_POS_SIZE * 5, 'regime_gap': regime_gap < REGIME_GAP_LIMIT}
    passed = all(gates.values()) and perm_p < 0.05
    failed = [k for k, v in gates.items() if not v]
    if perm_p >= 0.05: failed.append('perm_test')

    return {
        'label': label, 'n_trades': n, 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'wr': round(wr, 3), 'pf': round(pf, 3), 'mdd': round(mdd, 2),
        'regime_gap': round(regime_gap, 3), 'perm_p': round(perm_p, 4),
        'passed': passed, 'failed_gates': failed,
        'mean_ret': round(mean_ret * 100, 2), 'avg_hold': round(trades_df['hold_days'].mean(), 1),
        'bull_n': len(bull), 'bear_n': len(bear),
        'avg_score': round(trades_df['score'].mean(), 2),
        'reason': 'PASSED' if passed else f"FAILED: {', '.join(failed)}",
    }

# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    data = download_data()
    stock_tickers = [t for t in UNIVERSE if t in data['close'].columns]
    print(f"  Stock universe: {len(stock_tickers)} tickers\n")

    # Generate all signals
    print("[2] Generating signals...")
    signal_generators = {
        'iv_rv_gap':       signal_iv_rv_gap,
        'rsi_divergence':  signal_rsi_divergence,
        'bond_yield':      signal_bond_yield,
        'liquidity':       signal_liquidity,
        'vol_term_str':    signal_vol_term_structure,
        'consecutive_dip': signal_consecutive_dip,
        'base_mr':         signal_base_mr,
    }

    # Solo Sharpe ratios (from adversarial-validated results)
    SOLO_SHARPES = {
        'iv_rv_gap': 1.408,       # #10 adversarial Sharpe
        'rsi_divergence': 3.52,   # #8
        'bond_yield': 2.18,       # #9
        'liquidity': 1.798,       # #11
        'vol_term_str': 4.30,     # #12
        'consecutive_dip': 0.50,  # not independently validated
        'base_mr': 1.03,          # baseline
    }

    all_signals = {}
    for key, gen_func in signal_generators.items():
        t0 = time.time()
        sigs = gen_func(data, stock_tickers)
        elapsed = time.time() - t0
        print(f"  {key:25s} | {len(sigs):6d} entries ({elapsed:.1f}s)")
        all_signals[key] = sigs

    # Build daily scores
    print("\n[3] Building daily scores...")

    # Equal weights
    equal_scores = build_daily_scores(all_signals)
    print(f"  Equal-weight: {len(equal_scores)} scored entries")
    print(f"  Score distribution: min={min(equal_scores.values()):.0f}, max={max(equal_scores.values()):.0f}, "
          f"mean={np.mean(list(equal_scores.values())):.2f}")

    # Score histogram
    score_counts = defaultdict(int)
    for s in equal_scores.values():
        score_counts[int(s)] += 1
    print(f"  Histogram: " + " | ".join(f"{k}sig:{v}" for k, v in sorted(score_counts.items())))

    # Sharpe-weighted
    sharpe_weights = {k: v / max(SOLO_SHARPES.values()) for k, v in SOLO_SHARPES.items()}
    sharpe_scores = build_daily_scores(all_signals, weights=sharpe_weights)
    print(f"  Sharpe-weighted: {len(sharpe_scores)} scored entries")

    # ── Run variants ──
    print(f"\n[4] Running {6} scoring variants...")
    results = []

    # Variant A: Equal-weight, threshold sweep (1, 2, 3, 4, 5 signals)
    for threshold in [1, 2, 3, 4, 5]:
        entries = [(d, t, s) for (d, t), s in equal_scores.items() if s >= threshold]
        label = f"A: Equal score >= {threshold}"
        trades_df = run_backtest(entries, data, stock_tickers)
        result = validate_strategy(trades_df, label=label, run_perm=True,
                                   all_entries=entries, data=data, stock_tickers=stock_tickers)
        results.append(result)
        status = "✅" if result['passed'] else "❌"
        print(f"  {label:40s} | n={result['n_trades']:4d} | Sharpe={result['sharpe']:6.3f} | "
              f"WR={result['wr']:.3f} | perm_p={result['perm_p']:.4f} | avg_score={result.get('avg_score',0):.1f} | {status}")

    # Variant B: Sharpe-weighted, threshold sweep
    max_sharpe_score = sum(sharpe_weights.values())
    for threshold_pct in [0.15, 0.30, 0.45, 0.60, 0.75]:
        threshold = threshold_pct * max_sharpe_score
        entries = [(d, t, s) for (d, t), s in sharpe_scores.items() if s >= threshold]
        label = f"B: Sharpe-wt >= {threshold_pct:.0%}"
        trades_df = run_backtest(entries, data, stock_tickers)
        result = validate_strategy(trades_df, label=label, run_perm=True,
                                   all_entries=entries, data=data, stock_tickers=stock_tickers)
        results.append(result)
        status = "✅" if result['passed'] else "❌"
        print(f"  {label:40s} | n={result['n_trades']:4d} | Sharpe={result['sharpe']:6.3f} | "
              f"WR={result['wr']:.3f} | perm_p={result['perm_p']:.4f} | avg_score={result.get('avg_score',0):.1f} | {status}")

    # Variant C: Regime-conditioned (lower threshold in bear markets)
    spy_close = data['close']['SPY']
    spy_sma200 = spy_close.rolling(200).mean()
    for bull_thresh, bear_thresh in [(3, 2), (4, 2), (3, 1), (2, 1)]:
        entries = []
        for (d, t), s in equal_scores.items():
            try:
                is_bull = spy_close.at[d] > spy_sma200.at[d]
            except:
                is_bull = True
            threshold = bull_thresh if is_bull else bear_thresh
            if s >= threshold:
                entries.append((d, t, s))
        label = f"C: Regime bull>={bull_thresh}/bear>={bear_thresh}"
        trades_df = run_backtest(entries, data, stock_tickers)
        result = validate_strategy(trades_df, label=label, run_perm=True,
                                   all_entries=entries, data=data, stock_tickers=stock_tickers)
        results.append(result)
        status = "✅" if result['passed'] else "❌"
        print(f"  {label:40s} | n={result['n_trades']:4d} | Sharpe={result['sharpe']:6.3f} | "
              f"WR={result['wr']:.3f} | perm_p={result['perm_p']:.4f} | {status}")

    # Variant D: Top-N concentration (highest score per day)
    daily_best = defaultdict(list)
    for (d, t), s in equal_scores.items():
        if d >= pd.Timestamp(BACKTEST_START):
            daily_best[d].append((t, s))
    for top_n in [1, 2, 3]:
        entries = []
        for d, stock_scores in daily_best.items():
            stock_scores.sort(key=lambda x: -x[1])
            for t, s in stock_scores[:top_n]:
                if s >= 2:  # at least 2 signals
                    entries.append((d, t, s))
        label = f"D: Top-{top_n} per day (score>=2)"
        trades_df = run_backtest(entries, data, stock_tickers)
        result = validate_strategy(trades_df, label=label, run_perm=True,
                                   all_entries=entries, data=data, stock_tickers=stock_tickers)
        results.append(result)
        status = "✅" if result['passed'] else "❌"
        print(f"  {label:40s} | n={result['n_trades']:4d} | Sharpe={result['sharpe']:6.3f} | "
              f"WR={result['wr']:.3f} | perm_p={result['perm_p']:.4f} | {status}")

    # Variant E: INVERSE — trade lowest scores (control)
    for max_score in [1]:
        entries = [(d, t, s) for (d, t), s in equal_scores.items() if s <= max_score]
        label = f"E: INVERSE (score <= {max_score})"
        trades_df = run_backtest(entries, data, stock_tickers)
        result = validate_strategy(trades_df, label=label, run_perm=True,
                                   all_entries=entries, data=data, stock_tickers=stock_tickers)
        results.append(result)
        status = "✅" if result['passed'] else "❌"
        print(f"  {label:40s} | n={result['n_trades']:4d} | Sharpe={result['sharpe']:6.3f} | "
              f"WR={result['wr']:.3f} | perm_p={result['perm_p']:.4f} | {status} (CONTROL)")

    # Variant F: Dynamic sizing (position size proportional to score)
    for min_score in [2, 3]:
        entries = [(d, t, s) for (d, t), s in equal_scores.items() if s >= min_score]
        label = f"F: Dynamic sizing (score>={min_score})"
        trades_df = run_backtest(entries, data, stock_tickers, dynamic_sizing=True, max_score=7.0)
        result = validate_strategy(trades_df, label=label, run_perm=True,
                                   all_entries=entries, data=data, stock_tickers=stock_tickers)
        results.append(result)
        status = "✅" if result['passed'] else "❌"
        print(f"  {label:40s} | n={result['n_trades']:4d} | Sharpe={result['sharpe']:6.3f} | "
              f"WR={result['wr']:.3f} | perm_p={result['perm_p']:.4f} | {status}")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passes = [r for r in results if r.get('passed', False)]
    print(f"\nTotal variants: {len(results)}")
    print(f"5-gate PASS: {len(passes)}")

    if passes:
        print(f"\n{'─'*60}")
        print("PASSING VARIANTS:")
        for p in sorted(passes, key=lambda x: x['sharpe'], reverse=True):
            print(f"  🏆 {p['label']:45s} | Sharpe={p['sharpe']:6.3f} | Sortino={p['sortino']:6.3f} | "
                  f"WR={p['wr']:.3f} | PF={p['pf']:.3f} | n={p['n_trades']} | "
                  f"perm_p={p['perm_p']:.4f} | gap={p['regime_gap']:.3f}")

    # Score-Sharpe monotonicity check
    score_sharpes = [(r['label'], r.get('avg_score', 0), r['sharpe'])
                     for r in results if r['label'].startswith('A:') and r['n_trades'] >= 10]
    if len(score_sharpes) >= 3:
        print(f"\n{'─'*60}")
        print("MONOTONICITY CHECK (higher score → higher Sharpe?):")
        for label, avg_score, sharpe in score_sharpes:
            bar = "█" * int(max(0, sharpe) * 10)
            print(f"  {label:30s} | avg_score={avg_score:.1f} | Sharpe={sharpe:6.3f} | {bar}")

    # Save results
    output_file = OUTPUT_DIR / 'scoring_results.json'
    with open(output_file, 'w') as f:
        json.dump({'results': results, 'timestamp': str(datetime.now()),
                   'solo_sharpes': SOLO_SHARPES}, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")
    print("=" * 80)

if __name__ == '__main__':
    main()
