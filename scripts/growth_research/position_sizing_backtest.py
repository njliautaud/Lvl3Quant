#!/usr/bin/env python3
"""
Signal-Weighted Position Sizing Backtest
=========================================
Research question: Can varying position SIZE based on signal conviction
improve risk-adjusted returns over the fixed $300 baseline (Strategy #14)?

Variants:
  A) Score-proportional: $150 × score, cap $600
  B) Kelly-inspired: historical WR by score bracket × $1000 bankroll
  C) Inverse-volatility: $300 / realized_vol_ratio (more size when calm)
  D) Regime-scaled: Bull $200 fixed, Bear $400 fixed
  E) Score + Regime combo: Bull $150×score, Bear $300×score
  F) Anti-correlation: size up when new trade is low-corr to current portfolio

Baseline: Fixed $300 (Strategy #14 — Sharpe 1.556)
Entry rule: Regime-conditioned, bull score≥3, bear score≥1
Exit: +10% TP, -15% SL, 21-day max hold
Max 2 concurrent positions, 0.1% round-trip cost
Universe: 30 megacap stocks, 2020-2026
"""

import os, sys, json, warnings, time, functools, pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

# ─── Config ───────────────────────────────────────────────────────────────────
START_DATE       = '2019-01-01'
END_DATE         = '2026-07-01'
BACKTEST_START   = '2020-01-01'
BASE_POS_SIZE    = 300.0
BANKROLL         = 1000.0
MAX_CONCURRENT   = 2
HOLD_DAYS        = 21
PROFIT_TARGET    = 0.10
STOP_LOSS        = -0.15
SPREAD_COST_PCT  = 0.001
N_PERMS          = 1000
REGIME_GAP_LIMIT = 0.50
BASELINE_SHARPE  = 1.556   # Strategy #14 fixed-$300 baseline

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/position_sizing_backtest')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
    'MA', 'WMT', 'PEP', 'KO', 'TMO', 'ABT', 'BAC', 'XOM', 'CVX', 'CSCO',
]

MACRO_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', '^TNX', 'HYG', 'LQD']

print("=" * 80)
print("SIGNAL-WEIGHTED POSITION SIZING BACKTEST")
print(f"Baseline: Fixed ${BASE_POS_SIZE:.0f} | Sharpe {BASELINE_SHARPE}")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════════════════════

def download_data():
    import yfinance as yf
    # Try existing caches first (avoid re-downloading)
    cache_candidates = [
        Path('/home/jupiter/Lvl3Quant/output/growth_research/cross_signal_confluence/_confluence_cache.pkl'),
        Path('/home/jupiter/Lvl3Quant/output/growth_research/cross_signal_confluence_v2/_confluence_v2_cache.pkl'),
        Path('/home/jupiter/Lvl3Quant/output/growth_research/signal_scoring_portfolio/_scoring_cache.pkl'),
        OUTPUT_DIR / '_pos_sizing_cache.pkl',
    ]
    for cf in cache_candidates:
        if cf.exists():
            with open(cf, 'rb') as f:
                data = pickle.load(f)
            # Verify it has what we need
            if 'close' in data and 'SPY' in data['close'].columns:
                print(f"  Loaded cache: {len(data['close'].columns)} tickers, {len(data['close'])} days")
                return data

    print(f"\n[1] Downloading {len(UNIVERSE)} stocks + macro tickers...")
    all_tickers = UNIVERSE + MACRO_TICKERS
    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw['Close']
        high   = raw['High']
        low    = raw['Low']
        volume = raw['Volume']
    else:
        close = high = low = volume = raw

    for df in [close, high, low, volume]:
        if hasattr(df.columns, 'droplevel'):
            try: df.columns = df.columns.droplevel(1)
            except: pass

    close = close.ffill().dropna(how='all')
    data = {
        'close':  close,
        'high':   high.reindex(close.index).ffill(),
        'low':    low.reindex(close.index).ffill(),
        'volume': volume.reindex(close.index).ffill().fillna(0),
    }
    with open(OUTPUT_DIR / '_pos_sizing_cache.pkl', 'wb') as f:
        pickle.dump(data, f)
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIGNAL GENERATORS (same 7 as Strategy #14)
# ═══════════════════════════════════════════════════════════════════════════════

def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _realized_vol(returns, window=21):
    return returns.rolling(window).std() * np.sqrt(252) * 100

def signal_iv_rv_gap(data, tickers):
    spy_ret = data['close']['SPY'].pct_change()
    vix = data['close']['^VIX']
    rv_21 = _realized_vol(spy_ret, 21)
    gap = vix - rv_21
    fire_days = set(gap[gap > 5].index)
    sigs = {}
    for day in fire_days:
        for t in tickers:
            try:
                if pd.notna(data['close'].at[day, t]):
                    sigs[(day, t)] = 1.0
            except: pass
    return sigs

def signal_rsi_divergence(data, tickers):
    sigs = {}
    for t in tickers:
        if t not in data['close'].columns: continue
        close = data['close'][t].dropna()
        rsi = _rsi(close, 14)
        for i in range(20, len(close)):
            wc = close.iloc[i-10:i+1]; wr = rsi.iloc[i-10:i+1]
            if len(wc) < 11 or wr.isna().any(): continue
            if close.iloc[i] < wc.iloc[0] and rsi.iloc[i] > wr.iloc[0] and rsi.iloc[i] < 40:
                sigs[(close.index[i], t)] = 1.0
    return sigs

def signal_bond_yield(data, tickers):
    if '^TNX' in data['close'].columns:
        tnx = data['close']['^TNX']
        fire_days = set(tnx.diff(5)[tnx.diff(5) < -0.10].index)
    else:
        tlt = data['close']['TLT']
        fire_days = set(tlt.pct_change(5)[tlt.pct_change(5) > 0.02].index)
    sigs = {}
    for day in fire_days:
        for t in tickers:
            try:
                if pd.notna(data['close'].at[day, t]):
                    sigs[(day, t)] = 1.0
            except: pass
    return sigs

def signal_liquidity(data, tickers):
    sigs = {}
    for t in tickers:
        if t not in data['close'].columns or t not in data['high'].columns: continue
        hi = data['high'][t].dropna(); lo = data['low'][t].dropna(); cl = data['close'][t].dropna()
        idx = hi.index.intersection(lo.index).intersection(cl.index)
        if len(idx) < 65: continue
        hi, lo, cl = hi.loc[idx], lo.loc[idx], cl.loc[idx]
        hl_spread = (hi - lo) / cl
        avg_60 = hl_spread.rolling(60).mean()
        narrow = hl_spread < avg_60 * 0.85
        rsi = _rsi(cl, 14)
        for i in range(60, len(idx)):
            if narrow.iloc[i] and rsi.iloc[i] < 40:
                sigs[(idx[i], t)] = 1.0
    return sigs

def signal_vol_term_structure(data, tickers):
    vix = data['close'].get('^VIX'); vix3m = data['close'].get('^VIX3M')
    if vix is None or vix3m is None: return {}
    ratio = vix / vix3m
    fear_days = set(ratio[ratio > 1.0].index)
    sigs = {}
    for t in tickers:
        if t not in data['close'].columns: continue
        close = data['close'][t].dropna()
        sma20 = close.rolling(20).mean()
        dd = (close - sma20) / sma20
        for i in range(20, len(close)):
            day = close.index[i]
            if day in fear_days and dd.iloc[i] < -0.05:
                sigs[(day, t)] = 1.0
    return sigs

def signal_consecutive_dip(data, tickers):
    sigs = {}
    for t in tickers:
        if t not in data['close'].columns: continue
        close = data['close'][t].dropna(); ret = close.pct_change()
        for i in range(3, len(close)):
            r1, r2, r3 = ret.iloc[i-2], ret.iloc[i-1], ret.iloc[i]
            if r1 < 0 and r2 < 0 and r3 < 0 and r2 < r1 and r3 < r2:
                sigs[(close.index[i], t)] = 1.0
    return sigs

def signal_base_mr(data, tickers):
    sigs = {}
    for t in tickers:
        if t not in data['close'].columns: continue
        close = data['close'][t].dropna(); rsi = _rsi(close, 14)
        high_50 = close.rolling(50).max(); dd = (close - high_50) / high_50
        for i in range(50, len(close)):
            if rsi.iloc[i] < 30 and dd.iloc[i] < -0.07:
                sigs[(close.index[i], t)] = 1.0
    return sigs


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SCORE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_scores(all_signals):
    scores = defaultdict(float)
    for sig_dict in all_signals.values():
        for (day, t), val in sig_dict.items():
            scores[(day, t)] += val
    return dict(scores)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. REALIZED VOL RATIO (for Variant C)
# ═══════════════════════════════════════════════════════════════════════════════

def build_vol_ratio(data, tickers, window=21, long_window=63):
    """Per-stock realized vol / 63-day average vol. >1 = high vol, <1 = low vol."""
    vol_ratios = {}
    for t in tickers:
        if t not in data['close'].columns: continue
        ret = data['close'][t].pct_change().dropna()
        rv_short = ret.rolling(window).std() * np.sqrt(252)
        rv_long  = ret.rolling(long_window).std() * np.sqrt(252)
        ratio = rv_short / rv_long.replace(0, np.nan)
        vol_ratios[t] = ratio
    return vol_ratios


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BACKTESTER
# ═══════════════════════════════════════════════════════════════════════════════

def regime_at(date, spy_close, spy_sma200):
    try:
        return 'bull' if spy_close.at[date] > spy_sma200.at[date] else 'bear'
    except:
        return 'bull'

def run_backtest(entries_with_meta, data, tickers, sizing_fn, label=''):
    """
    entries_with_meta: list of (date, ticker, score) — pre-filtered for entry rules
    sizing_fn: function(date, ticker, score, open_positions, data, aux_data) -> dollar_size
    Returns DataFrame of trades.
    """
    close     = data['close']
    spy_close = close['SPY']
    spy_sma200 = spy_close.rolling(200).mean()
    bt_start  = pd.Timestamp(BACKTEST_START)

    entries = [(d, t, s) for d, t, s in entries_with_meta if d >= bt_start]
    entries.sort(key=lambda x: (x[0], -x[2]))  # date asc, score desc (prefer highest)

    trades = []
    open_positions = []  # list of (entry_date, ticker, entry_price, exit_date)

    for entry_date, ticker, score in entries:
        # Prune closed positions
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

        # Compute position size
        pos_size = sizing_fn(entry_date, ticker, score, open_positions, data, close)
        if pos_size <= 0:
            continue

        # Simulate trade
        exit_price  = None
        exit_date   = None
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
                exit_price = price; exit_date = fdate; exit_reason = 'profit_target'; break
            elif ret <= STOP_LOSS:
                exit_price = price; exit_date = fdate; exit_reason = 'stop_loss'; break

        if exit_price is None:
            hold_end = min(HOLD_DAYS, len(future_dates))
            if hold_end == 0:
                continue
            exit_date = future_dates[hold_end - 1]
            try:
                exit_price = close.at[exit_date, ticker]
            except:
                continue
            if pd.isna(exit_price):
                continue

        raw_ret = (exit_price - entry_price) / entry_price
        net_ret = raw_ret - SPREAD_COST_PCT
        pnl     = pos_size * net_ret

        trades.append({
            'entry_date':  entry_date,
            'exit_date':   exit_date,
            'ticker':      ticker,
            'entry_price': entry_price,
            'exit_price':  exit_price,
            'return':      net_ret,
            'pnl':         pnl,
            'pos_size':    pos_size,
            'score':       score,
            'exit_reason': exit_reason,
            'regime':      regime_at(entry_date, spy_close, spy_sma200),
            'hold_days':   (exit_date - entry_date).days,
        })
        open_positions.append((entry_date, ticker, entry_price, exit_date))

    return pd.DataFrame(trades) if trades else None


# ═══════════════════════════════════════════════════════════════════════════════
# 6. METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(trades_df, label=''):
    if trades_df is None or len(trades_df) < 5:
        return {'label': label, 'n_trades': 0, 'sharpe': 0, 'sortino': 0,
                'wr': 0, 'pf': 0, 'mdd': 0, 'total_return_pct': 0,
                'avg_pos': 0, 'regime_gap': 1.0, 'perm_p': 1.0, 'passed': False,
                'reason': 'insufficient trades'}

    rets = trades_df['return'].values
    n    = len(rets)
    span_days = max(1, (trades_df['entry_date'].max() - trades_df['entry_date'].min()).days)
    tpy  = n / max(1, span_days / 365.25)
    if tpy < 1: tpy = 1

    mean_ret = np.mean(rets)
    std_ret  = np.std(rets, ddof=1) if n > 1 else 1.0
    sharpe   = (mean_ret / std_ret) * np.sqrt(tpy) if std_ret > 0 else 0

    neg = rets[rets < 0]
    sdenom   = np.std(neg, ddof=1) if len(neg) > 1 else std_ret
    sortino  = (mean_ret / sdenom) * np.sqrt(tpy) if sdenom > 0 else 0

    wr = float(np.mean(rets > 0))
    gp = np.sum(rets[rets > 0])
    gl = abs(np.sum(rets[rets < 0]))
    pf = gp / gl if gl > 0 else 99.0

    cum_pnl = np.cumsum(trades_df['pnl'].values)
    peak    = np.maximum.accumulate(cum_pnl)
    mdd     = float(np.min(cum_pnl - peak)) if len(cum_pnl) > 0 else 0

    total_ret_pct = (cum_pnl[-1] / BANKROLL) * 100 if len(cum_pnl) > 0 else 0

    bull = trades_df[trades_df['regime'] == 'bull']
    bear = trades_df[trades_df['regime'] == 'bear']
    if len(bull) >= 3 and len(bear) >= 3:
        bs  = np.mean(bull['return']) / max(np.std(bull['return'], ddof=1), 1e-6)
        brs = np.mean(bear['return']) / max(np.std(bear['return'], ddof=1), 1e-6)
        regime_gap = abs(bs - brs) / max(abs(bs), abs(brs), 1e-6)
    else:
        regime_gap = 0.0

    # Sub-period positivity (4 equal sub-periods)
    dates = trades_df['entry_date'].sort_values()
    periods = np.array_split(dates.index, 4)
    sub_positive = sum(1 for p in periods if len(p) > 0 and trades_df.loc[p, 'return'].mean() > 0)

    return {
        'label': label,
        'n_trades': n,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'wr': round(wr, 3),
        'pf': round(pf, 3),
        'mdd': round(mdd, 2),
        'total_return_pct': round(total_ret_pct, 1),
        'avg_pos': round(trades_df['pos_size'].mean(), 1),
        'avg_score': round(trades_df['score'].mean(), 2),
        'regime_gap': round(regime_gap, 3),
        'sub_positive': sub_positive,
        'perm_p': 1.0,   # filled in later
        'passed': False,
        'reason': '',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════════════════════

def permutation_test(trades_df, data, tickers, actual_sharpe, n_perms=N_PERMS):
    if trades_df is None or len(trades_df) < 10:
        return 1.0

    close = data['close']
    bt_start = pd.Timestamp(BACKTEST_START)
    valid_dates = close.index[close.index >= bt_start][:-HOLD_DAYS - 5]
    n_raw = len(trades_df)

    perm_sharpes = []
    for _ in range(n_perms):
        rd = np.random.choice(valid_dates, size=n_raw, replace=True)
        rt = np.random.choice(tickers, size=n_raw, replace=True)
        rs = np.random.uniform(1, 5, size=n_raw)
        rand_entries = list(zip(rd, rt, rs))

        # Fixed sizing for perm test (neutral reference)
        def fixed_fn(date, t, score, ops, data, cl): return BASE_POS_SIZE
        rt_df = run_backtest(rand_entries, data, tickers, fixed_fn)

        if rt_df is not None and len(rt_df) >= 5:
            rr = rt_df['return'].values
            r_tpy = len(rr) / max(1, (rt_df['entry_date'].max() - rt_df['entry_date'].min()).days / 365.25)
            if r_tpy < 1: r_tpy = 1
            rs_val = (np.mean(rr) / max(np.std(rr, ddof=1), 1e-8)) * np.sqrt(r_tpy)
        else:
            rs_val = 0.0
        perm_sharpes.append(rs_val)

    return float(np.mean(np.array(perm_sharpes) >= actual_sharpe))


# ═══════════════════════════════════════════════════════════════════════════════
# 8. 5-GATE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_5gate(metrics, perm_p):
    metrics = dict(metrics)
    metrics['perm_p'] = round(perm_p, 4)

    gates = {
        'regime_gap_lt_050': metrics['regime_gap'] < REGIME_GAP_LIMIT,
        'perm_p_lt_005':     perm_p < 0.05,
        'sub_periods_all_pos': metrics['sub_positive'] == 4,
        'mdd_gt_neg50pct':   metrics['mdd'] > -BANKROLL * 0.50,
        'n_ge_20':           metrics['n_trades'] >= 20,
    }

    passed = all(gates.values())
    failed = [k for k, v in gates.items() if not v]
    metrics['passed']      = passed
    metrics['gates']       = gates
    metrics['reason']      = 'PASSED' if passed else 'FAILED: ' + ', '.join(failed)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 9. KELLY WR TABLE (pre-compute from baseline entries)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_kelly_by_score(entries_with_scores, data, tickers):
    """Run a fixed-$300 backtest and compute WR per score bracket."""
    def fixed_fn(date, t, score, ops, data, cl): return BASE_POS_SIZE
    trades_df = run_backtest(entries_with_scores, data, tickers, fixed_fn)
    if trades_df is None or len(trades_df) < 10:
        return {}

    kelly_table = {}
    for score_val in sorted(trades_df['score'].unique()):
        subset = trades_df[trades_df['score'] == score_val]
        if len(subset) < 3:
            continue
        wr = float(np.mean(subset['return'] > 0))
        avg_win  = float(np.mean(subset.loc[subset['return'] > 0, 'return'])) if (subset['return'] > 0).any() else 0.01
        avg_loss = float(abs(np.mean(subset.loc[subset['return'] < 0, 'return']))) if (subset['return'] < 0).any() else 0.01
        # Full Kelly: (wr/loss - (1-wr)/win) ← assumes b = avg_win/avg_loss
        b = avg_win / max(avg_loss, 1e-6)
        full_kelly = (wr * b - (1 - wr)) / max(b, 1e-6)
        half_kelly = max(0.0, full_kelly / 2)  # use half-kelly for safety
        kelly_table[score_val] = {
            'wr': wr, 'n': len(subset),
            'avg_win': avg_win, 'avg_loss': avg_loss,
            'b': b, 'full_kelly': full_kelly, 'half_kelly': half_kelly,
        }
    return kelly_table


# ═══════════════════════════════════════════════════════════════════════════════
# 10. ENTRY FILTER: Strategy #14 regime-conditioned
# ═══════════════════════════════════════════════════════════════════════════════

def filter_entries(scores, spy_close, spy_sma200, bull_min=3, bear_min=1):
    entries = []
    for (day, ticker), score in scores.items():
        try:
            is_bull = spy_close.at[day] > spy_sma200.at[day]
        except:
            is_bull = True
        threshold = bull_min if is_bull else bear_min
        if score >= threshold:
            entries.append((day, ticker, score))
    return entries


# ═══════════════════════════════════════════════════════════════════════════════
# 11. ANTI-CORRELATION SIZING HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def build_return_matrix(data, tickers, window=21):
    """Return matrix: daily returns per ticker, rolling."""
    returns = {}
    for t in tickers:
        if t in data['close'].columns:
            returns[t] = data['close'][t].pct_change().dropna()
    return returns


def portfolio_corr_to_new(open_positions, new_ticker, returns_dict, entry_date, window=60):
    """Avg correlation of new ticker to open positions in recent window."""
    if not open_positions:
        return 0.0   # no positions → no correlation → full size
    if new_ticker not in returns_dict:
        return 0.5

    new_ret = returns_dict[new_ticker]
    # Use recent window ending at entry_date
    end_idx = new_ret.index.searchsorted(entry_date)
    start_idx = max(0, end_idx - window)
    new_slice = new_ret.iloc[start_idx:end_idx]
    if len(new_slice) < 10:
        return 0.5

    corrs = []
    for _, open_ticker, _, _ in open_positions:
        if open_ticker not in returns_dict:
            continue
        ot_ret = returns_dict[open_ticker]
        common = new_slice.index.intersection(ot_ret.index)
        if len(common) < 10:
            continue
        c = np.corrcoef(new_slice.loc[common].values, ot_ret.loc[common].values)[0, 1]
        if not np.isnan(c):
            corrs.append(c)

    return float(np.mean(corrs)) if corrs else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 12. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    data    = download_data()
    tickers = [t for t in UNIVERSE if t in data['close'].columns]
    print(f"  Universe: {len(tickers)} tickers\n")

    spy_close  = data['close']['SPY']
    spy_sma200 = spy_close.rolling(200).mean()

    # ── 2. Generate 7 signals ────────────────────────────────────────────────
    print("[2] Generating 7 validated signals...")
    SIG_FUNCS = {
        'iv_rv_gap':       signal_iv_rv_gap,
        'rsi_divergence':  signal_rsi_divergence,
        'bond_yield':      signal_bond_yield,
        'liquidity':       signal_liquidity,
        'vol_term_str':    signal_vol_term_structure,
        'consecutive_dip': signal_consecutive_dip,
        'base_mr':         signal_base_mr,
    }
    all_signals = {}
    for key, fn in SIG_FUNCS.items():
        t0 = time.time()
        sigs = fn(data, tickers)
        print(f"  {key:25s} | {len(sigs):6d} entries ({time.time()-t0:.1f}s)")
        all_signals[key] = sigs

    # ── 3. Build scores + entries ────────────────────────────────────────────
    print("\n[3] Building scores and applying Strategy #14 entry rules...")
    scores  = build_scores(all_signals)
    entries = filter_entries(scores, spy_close, spy_sma200, bull_min=3, bear_min=1)
    print(f"  Total scored entries: {len(scores)}")
    print(f"  After regime-conditioned filter (bull>=3, bear>=1): {len(entries)}")

    sc_hist = defaultdict(int)
    for _, _, s in entries:
        sc_hist[int(s)] += 1
    print(f"  Score distribution: {dict(sorted(sc_hist.items()))}")

    # ── 4. Pre-compute aux data ───────────────────────────────────────────────
    print("\n[4] Pre-computing vol ratios and Kelly table...")
    vol_ratios = build_vol_ratio(data, tickers)

    # Kelly table from baseline fixed-$300 run
    kelly_table = compute_kelly_by_score(entries, data, tickers)
    print("  Kelly table by score:")
    for sv, kt in sorted(kelly_table.items()):
        print(f"    score={sv:.0f} | WR={kt['wr']:.3f} | half_kelly={kt['half_kelly']:.3f} | n={kt['n']}")

    # Return matrix for anti-corr sizing
    returns_dict = build_return_matrix(data, tickers)

    # ── 5. BASELINE: Fixed $300 ───────────────────────────────────────────────
    print("\n[5] Running baseline (fixed $300)...")
    def sizing_baseline(date, ticker, score, ops, data, cl):
        return BASE_POS_SIZE

    baseline_trades = run_backtest(entries, data, tickers, sizing_baseline, 'Baseline $300')
    baseline_metrics = compute_metrics(baseline_trades, 'Baseline: Fixed $300')
    print(f"  Sharpe={baseline_metrics['sharpe']:.3f} | Sortino={baseline_metrics['sortino']:.3f} | "
          f"WR={baseline_metrics['wr']:.3f} | n={baseline_metrics['n_trades']}")

    # ── 6. VARIANTS ───────────────────────────────────────────────────────────
    variants = []

    def run_variant(label, sizing_fn, run_perm=True):
        print(f"\n  [{label}] Running...")
        trades = run_backtest(entries, data, tickers, sizing_fn, label)
        metrics = compute_metrics(trades, label)

        if run_perm and metrics['n_trades'] >= 20:
            print(f"    Running permutation test ({N_PERMS} perms)...")
            pp = permutation_test(trades, data, tickers, metrics['sharpe'])
        else:
            pp = 1.0 if metrics['n_trades'] < 20 else 1.0

        final = validate_5gate(metrics, pp)
        beat  = '✅ BEATS BASELINE' if final['sharpe'] > BASELINE_SHARPE else '❌ below baseline'
        print(f"    Sharpe={final['sharpe']:.3f} | Sortino={final['sortino']:.3f} | "
              f"WR={final['wr']:.3f} | PF={final['pf']:.3f} | MDD=${final['mdd']:.0f} | "
              f"n={final['n_trades']} | avg_pos=${final['avg_pos']:.0f} | "
              f"perm_p={final['perm_p']:.4f} | {beat}")
        print(f"    Gate result: {final['reason']}")
        return final

    # ──── Variant A: Score-proportional ($150 × score, cap $600) ─────────────
    print("\n[6] VARIANT A: Score-proportional sizing ($150 × score, cap $600)")
    def sizing_A(date, ticker, score, ops, data, cl):
        return min(150.0 * score, 600.0)
    vA = run_variant('A: Score-prop ($150×score, cap $600)', sizing_A)
    variants.append(vA)

    # ──── Variant B: Kelly-inspired ──────────────────────────────────────────
    print("\n[6] VARIANT B: Kelly-inspired (half-Kelly × $1000)")
    def sizing_B(date, ticker, score, ops, data, cl):
        score_key = float(int(score))
        kt = kelly_table.get(score_key)
        if kt is None:
            # Find nearest
            keys = sorted(kelly_table.keys())
            if not keys: return BASE_POS_SIZE
            nearest = min(keys, key=lambda k: abs(k - score_key))
            kt = kelly_table[nearest]
        fraction = max(0.0, min(kt['half_kelly'], 0.40))  # cap at 40% bankroll
        return max(50.0, fraction * BANKROLL)
    vB = run_variant('B: Kelly-inspired (half-Kelly × $1000)', sizing_B)
    variants.append(vB)

    # ──── Variant C: Inverse-volatility ──────────────────────────────────────
    print("\n[6] VARIANT C: Inverse-volatility ($300 / vol_ratio)")
    def sizing_C(date, ticker, score, ops, data, cl):
        if ticker not in vol_ratios:
            return BASE_POS_SIZE
        vr_series = vol_ratios[ticker]
        try:
            ratio = vr_series.at[date]
        except:
            ratio = 1.0
        if pd.isna(ratio) or ratio <= 0:
            return BASE_POS_SIZE
        # Clamp ratio to [0.5, 2.0] to prevent extreme sizes
        ratio = max(0.5, min(ratio, 2.0))
        size  = BASE_POS_SIZE / ratio
        return max(100.0, min(size, 600.0))
    vC = run_variant('C: Inverse-vol ($300/vol_ratio)', sizing_C)
    variants.append(vC)

    # ──── Variant D: Regime-scaled (Bull $200, Bear $400) ────────────────────
    print("\n[6] VARIANT D: Regime-scaled (Bull=$200, Bear=$400)")
    def sizing_D(date, ticker, score, ops, data, cl):
        r = regime_at(date, spy_close, spy_sma200)
        return 400.0 if r == 'bear' else 200.0
    vD = run_variant('D: Regime-scaled (Bull=$200, Bear=$400)', sizing_D)
    variants.append(vD)

    # ──── Variant E: Score + Regime combo ────────────────────────────────────
    print("\n[6] VARIANT E: Score+Regime combo (Bull $150×score, Bear $300×score)")
    def sizing_E(date, ticker, score, ops, data, cl):
        r = regime_at(date, spy_close, spy_sma200)
        if r == 'bear':
            return min(300.0 * score, 1200.0)
        else:
            return min(150.0 * score, 600.0)
    vE = run_variant('E: Score+Regime (Bull $150×s, Bear $300×s)', sizing_E)
    variants.append(vE)

    # ──── Variant F: Anti-correlation sizing ─────────────────────────────────
    print("\n[6] VARIANT F: Anti-correlation sizing")
    def sizing_F(date, ticker, score, ops, data, cl):
        avg_corr = portfolio_corr_to_new(ops, ticker, returns_dict, date)
        # Low correlation → full $300, high correlation → reduce
        # Scale: corr=0 → $450, corr=0.5 → $300, corr=1.0 → $150
        factor = 1.5 - avg_corr  # ranges from 0.5 to 1.5
        return max(100.0, min(BASE_POS_SIZE * factor, 600.0))
    vF = run_variant('F: Anti-correlation sizing', sizing_F)
    variants.append(vF)

    # ── 7. SUMMARY ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY: POSITION SIZING VARIANTS vs BASELINE")
    print(f"Baseline: Fixed $300 | Sharpe = {BASELINE_SHARPE} (Strategy #14)")
    print("=" * 80)

    all_results = [baseline_metrics] + variants

    # Print comparison table
    header = f"{'Variant':<48} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'MDD':>8} {'N':>5} {'AvgPos':>7} {'perm_p':>8} {'5-gate'}"
    print("\n" + header)
    print("-" * len(header))

    for r in all_results:
        gate_str = 'PASS' if r.get('passed', False) else 'fail'
        beat_flag = ' ★' if r.get('sharpe', 0) > BASELINE_SHARPE else ''
        print(f"  {r['label']:<46} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} "
              f"{r['wr']:>6.3f} {r['pf']:>6.3f} {r['mdd']:>8.0f} "
              f"{r['n_trades']:>5} {r.get('avg_pos', BASE_POS_SIZE):>7.0f} "
              f"{r.get('perm_p', 1.0):>8.4f} {gate_str}{beat_flag}")

    # Winners
    winners = [r for r in variants if r.get('sharpe', 0) > BASELINE_SHARPE and r.get('passed', False)]
    print(f"\n{'─'*60}")
    if winners:
        print(f"VARIANTS BEATING BASELINE AND PASSING ALL 5 GATES: {len(winners)}")
        for w in sorted(winners, key=lambda x: x['sharpe'], reverse=True):
            print(f"  ★ {w['label']}")
            print(f"    Sharpe {w['sharpe']:.3f} vs baseline {BASELINE_SHARPE} "
                  f"(+{w['sharpe']-BASELINE_SHARPE:.3f})")
            print(f"    Sortino={w['sortino']:.3f} | WR={w['wr']:.3f} | PF={w['pf']:.3f} | "
                  f"MDD=${w['mdd']:.0f} | n={w['n_trades']} | avg_pos=${w.get('avg_pos',0):.0f}")
            print(f"    Sub-periods positive: {w.get('sub_positive',0)}/4 | "
                  f"Regime gap: {w['regime_gap']:.3f} | perm_p: {w['perm_p']:.4f}")
    else:
        beats = [r for r in variants if r.get('sharpe', 0) > BASELINE_SHARPE]
        passes = [r for r in variants if r.get('passed', False)]
        print(f"No variant beats baseline AND passes all 5 gates.")
        print(f"  Variants beating Sharpe baseline: {len(beats)} — {[r['label'] for r in beats]}")
        print(f"  Variants passing all gates: {len(passes)} — {[r['label'] for r in passes]}")

    # Save results
    out_file = OUTPUT_DIR / 'position_sizing_results.json'
    with open(out_file, 'w') as f:
        json.dump({
            'timestamp': str(datetime.now()),
            'baseline_sharpe': BASELINE_SHARPE,
            'baseline_computed': baseline_metrics,
            'variants': variants,
            'kelly_table': {str(k): v for k, v in kelly_table.items()},
            'config': {
                'start': START_DATE, 'end': END_DATE,
                'backtest_start': BACKTEST_START,
                'universe': UNIVERSE, 'hold_days': HOLD_DAYS,
                'profit_target': PROFIT_TARGET, 'stop_loss': STOP_LOSS,
                'cost_pct': SPREAD_COST_PCT, 'max_concurrent': MAX_CONCURRENT,
                'n_perms': N_PERMS,
            },
        }, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed/60:.1f} min")
    print(f"Results saved.")
    print("=" * 80)


if __name__ == '__main__':
    main()
