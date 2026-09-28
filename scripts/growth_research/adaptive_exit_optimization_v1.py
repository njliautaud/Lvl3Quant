#!/usr/bin/env python3
"""
Adaptive Exit Optimization — Bond Yield Signal B
=================================================
Research question: Can we beat the baseline +10% TP / -15% SL / 21-day hold
(Sharpe 2.18) by using smarter exit logic?

Six exit variants tested:
  A) Trailing stop     — lock in gains with 50% giveback from peak
  B) Signal-based exit — exit when RSI>70 OR 5 consecutive up days
  C) Volatility-scaled — VIX-conditioned TP/SL (wide in fear, tight in calm)
  D) Time-weighted     — TP target shrinks as hold period extends
  E) Regime-conditioned— different TP/SL in bull vs bear
  F) Momentum cont.    — extend hold up to 42d if still accelerating (3d mom)

5-gate validation on any winner (Sharpe > 2.18 AND N >= 30):
  1. Regime gap < 0.50
  2. Permutation test p < 0.05
  3. All 4 sub-periods positive return
  4. MDD > -50%
  5. N >= 20

Universe: 30 megacap stocks, 2020-01-01 to 2026-07-01
Entry: Bond Yield Signal B (10Y yield drops >0.1% over 5d, stock >5% below 20-SMA)
Position: $300 per trade, max 2 concurrent, 0.1% round-trip cost
"""

import os, sys, json, warnings, time, pickle, functools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
    'MA', 'WMT', 'PEP', 'KO', 'TMO', 'ABT', 'BAC', 'XOM', 'CVX', 'CSCO',
]
MACRO_TICKERS = ['SPY', '^VIX', '^TNX', 'TLT']
START_DATE = '2019-01-01'       # extra warmup for indicators
BACKTEST_START = '2020-01-01'
END_DATE = '2026-07-01'

POS_SIZE = 300.0
MAX_CONCURRENT = 2
COST_PCT = 0.001                # 0.1% round-trip
N_PERMS = 1000
REGIME_GAP_LIMIT = 0.50
BASELINE_SHARPE = 2.18
BASELINE_N = 90
MIN_N = 30
SEED = 42

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/adaptive_exit')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)

print("=" * 90)
print("ADAPTIVE EXIT OPTIMIZATION  —  Bond Yield Signal B")
print("Baseline: Sharpe=2.18, N=90, +10% TP / -15% SL / 21-day hold")
print("=" * 90)


# ── 1. DATA ───────────────────────────────────────────────────────────────────
def download_data():
    import yfinance as yf
    cache = OUTPUT_DIR / '_aeopt_cache.pkl'
    if cache.exists():
        with open(cache, 'rb') as f:
            data = pickle.load(f)
        print(f"[DATA] Loaded cache: {len(data['close'].columns)} tickers, {len(data['close'])} days")
        return data

    print(f"[DATA] Downloading {len(UNIVERSE)} stocks + {len(MACRO_TICKERS)} macro...")
    all_tickers = UNIVERSE + MACRO_TICKERS
    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False, threads=True)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close'].copy()
        high  = raw['High'].copy()
        low   = raw['Low'].copy()
        vol   = raw['Volume'].copy()
    else:
        close = high = low = vol = raw.copy()

    close = close.ffill().dropna(how='all')
    data = {
        'close':  close,
        'high':   high.reindex(close.index).ffill(),
        'low':    low.reindex(close.index).ffill(),
        'volume': vol.reindex(close.index).ffill().fillna(0),
    }
    with open(cache, 'wb') as f:
        pickle.dump(data, f)
    print(f"[DATA] Downloaded: {len(close.columns)} tickers, {len(close)} days")
    return data


# ── 2. HELPERS ────────────────────────────────────────────────────────────────
def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _sharpe(returns, rf=0.0):
    rets = np.array(returns)
    if len(rets) < 2 or rets.std() == 0:
        return 0.0
    return (rets.mean() - rf) / rets.std() * np.sqrt(252)


def _sortino(returns, rf=0.0):
    rets = np.array(returns)
    downside = rets[rets < rf]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    return (rets.mean() - rf) / downside.std() * np.sqrt(252)


def _mdd(pnls):
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak)
    return float(dd.min()) if len(dd) else 0.0


def _metrics(trades_df):
    if trades_df is None or len(trades_df) == 0:
        return dict(sharpe=0, sortino=0, wr=0, pf=0, mdd=0, n=0, total_ret=0, avg_hold=0)
    rets = trades_df['ret'].values
    pnls = trades_df['pnl'].values
    wins  = rets[rets > 0]
    losss = rets[rets < 0]
    pf = (wins.sum() / abs(losss.sum())) if len(losss) and losss.sum() != 0 else np.inf
    total_ret = pnls.sum() / (POS_SIZE * MAX_CONCURRENT) * 100
    return dict(
        sharpe=_sharpe(rets),
        sortino=_sortino(rets),
        wr=float((rets > 0).mean()),
        pf=float(pf),
        mdd=_mdd(pnls),
        n=len(trades_df),
        total_ret=round(total_ret, 2),
        avg_hold=round(trades_df['hold_days'].mean(), 1),
    )


# ── 3. BOND YIELD SIGNAL B ────────────────────────────────────────────────────
def generate_entries(data):
    """Bond Yield Signal B: 10Y yield drops >0.1% over 5 days AND stock >5% below 20-SMA."""
    close = data['close']
    bt_start = pd.Timestamp(BACKTEST_START)

    if '^TNX' in close.columns:
        tnx = close['^TNX']
        yield_change = tnx.diff(5)
        fire_mask = yield_change < -0.10
    else:
        tlt = close['TLT']
        yield_change = tlt.pct_change(5)
        fire_mask = yield_change > 0.02

    fire_days = set(fire_mask[fire_mask].index)

    entries = []
    for t in UNIVERSE:
        if t not in close.columns:
            continue
        c = close[t].dropna()
        sma20 = c.rolling(20).mean()
        drawdown = (c - sma20) / sma20

        for i in range(20, len(c)):
            day = c.index[i]
            if day < bt_start:
                continue
            if day not in fire_days:
                continue
            if pd.isna(drawdown.iloc[i]) or drawdown.iloc[i] > -0.05:
                continue
            entries.append((day, t))

    entries.sort(key=lambda x: x[0])
    print(f"[SIGNAL] Bond Yield B: {len(entries)} raw entry signals across {len(UNIVERSE)} stocks")
    return entries


# ── 4. BACKTESTERS ────────────────────────────────────────────────────────────

def _spy_regime_series(data):
    spy = data['close']['SPY']
    spy_sma200 = spy.rolling(200).mean()
    return spy, spy_sma200


def backtest_baseline(entries, data):
    """Baseline: +10% TP, -15% SL, 21-day hold."""
    close = data['close']
    spy, spy_sma200 = _spy_regime_series(data)
    trades = []
    open_pos = []

    for entry_date, ticker in entries:
        open_pos = [p for p in open_pos if p['exit_date'] > entry_date]
        if len(open_pos) >= MAX_CONCURRENT:
            continue
        try:
            ep = close.at[entry_date, ticker]
        except:
            continue
        if pd.isna(ep) or ep <= 0:
            continue

        future = close.index[close.index > entry_date][:21]
        exit_price, exit_date, reason = ep, future[-1] if len(future) else None, 'hold_expiry'

        for fdate in future:
            try:
                p = close.at[fdate, ticker]
            except:
                continue
            if pd.isna(p):
                continue
            ret = (p - ep) / ep
            if ret >= 0.10:
                exit_price, exit_date, reason = p, fdate, 'tp'; break
            elif ret <= -0.15:
                exit_price, exit_date, reason = p, fdate, 'sl'; break
        else:
            if len(future) > 0:
                exit_price = close.at[future[-1], ticker] if future[-1] in close.index else ep

        if exit_date is None:
            continue

        net_ret = (exit_price - ep) / ep - COST_PCT
        try:
            regime = 'bull' if spy.at[entry_date] > spy_sma200.at[entry_date] else 'bear'
        except:
            regime = 'unknown'

        trades.append(dict(
            entry_date=entry_date, exit_date=exit_date, ticker=ticker,
            ret=net_ret, pnl=POS_SIZE * net_ret,
            exit_reason=reason, regime=regime,
            hold_days=(exit_date - entry_date).days,
        ))
        open_pos.append({'exit_date': exit_date})

    return pd.DataFrame(trades) if trades else None


def backtest_A_trailing(entries, data, trail_giveback=0.50):
    """A) Trailing stop: exit when price gives back 50% of peak gain."""
    close = data['close']
    spy, spy_sma200 = _spy_regime_series(data)
    trades = []
    open_pos = []

    for entry_date, ticker in entries:
        open_pos = [p for p in open_pos if p['exit_date'] > entry_date]
        if len(open_pos) >= MAX_CONCURRENT:
            continue
        try:
            ep = close.at[entry_date, ticker]
        except:
            continue
        if pd.isna(ep) or ep <= 0:
            continue

        future = close.index[close.index > entry_date][:21]
        if len(future) == 0:
            continue

        peak = ep
        exit_price = close.at[future[-1], ticker] if future[-1] in close.index else ep
        exit_date = future[-1]
        reason = 'hold_expiry'

        for fdate in future:
            try:
                p = close.at[fdate, ticker]
            except:
                continue
            if pd.isna(p):
                continue
            if p > peak:
                peak = p
            ret = (p - ep) / ep
            # Hard SL
            if ret <= -0.15:
                exit_price, exit_date, reason = p, fdate, 'sl'; break
            # Trailing: if we've peaked above entry, and gave back 50% of gain
            peak_gain = (peak - ep) / ep
            if peak_gain > 0.0:
                current_gain = (p - ep) / ep
                if current_gain <= peak_gain * (1 - trail_giveback):
                    exit_price, exit_date, reason = p, fdate, 'trailing_stop'; break

        net_ret = (exit_price - ep) / ep - COST_PCT
        try:
            regime = 'bull' if spy.at[entry_date] > spy_sma200.at[entry_date] else 'bear'
        except:
            regime = 'unknown'

        trades.append(dict(
            entry_date=entry_date, exit_date=exit_date, ticker=ticker,
            ret=net_ret, pnl=POS_SIZE * net_ret,
            exit_reason=reason, regime=regime,
            hold_days=(exit_date - entry_date).days,
        ))
        open_pos.append({'exit_date': exit_date})

    return pd.DataFrame(trades) if trades else None


def backtest_B_signal_exit(entries, data):
    """B) Signal-based exit: exit when RSI>70 OR 5 consecutive up days."""
    close = data['close']
    spy, spy_sma200 = _spy_regime_series(data)

    # Pre-compute RSI for all stocks
    rsi_cache = {}
    consec_up_cache = {}
    for t in UNIVERSE:
        if t not in close.columns:
            continue
        c = close[t].dropna()
        rsi_cache[t] = _rsi(c, 14)
        # consecutive up days
        up = (c.diff() > 0).astype(int)
        consec_up_cache[t] = up.rolling(5).sum()  # 5 = all 5 up

    trades = []
    open_pos = []

    for entry_date, ticker in entries:
        open_pos = [p for p in open_pos if p['exit_date'] > entry_date]
        if len(open_pos) >= MAX_CONCURRENT:
            continue
        try:
            ep = close.at[entry_date, ticker]
        except:
            continue
        if pd.isna(ep) or ep <= 0:
            continue

        future = close.index[close.index > entry_date][:21]
        if len(future) == 0:
            continue

        exit_price = close.at[future[-1], ticker] if future[-1] in close.index else ep
        exit_date = future[-1]
        reason = 'hold_expiry'

        rsi_s = rsi_cache.get(ticker)
        consec_s = consec_up_cache.get(ticker)

        for fdate in future:
            try:
                p = close.at[fdate, ticker]
            except:
                continue
            if pd.isna(p):
                continue
            ret = (p - ep) / ep
            # TP/SL as backstop
            if ret >= 0.10:
                exit_price, exit_date, reason = p, fdate, 'tp'; break
            if ret <= -0.15:
                exit_price, exit_date, reason = p, fdate, 'sl'; break
            # Signal exit: RSI overbought
            if rsi_s is not None and fdate in rsi_s.index:
                rsi_val = rsi_s.at[fdate]
                if not pd.isna(rsi_val) and rsi_val > 70:
                    exit_price, exit_date, reason = p, fdate, 'rsi_exit'; break
            # Signal exit: 5 consecutive up days
            if consec_s is not None and fdate in consec_s.index:
                cu = consec_s.at[fdate]
                if not pd.isna(cu) and cu >= 5:
                    exit_price, exit_date, reason = p, fdate, 'consec_up_exit'; break

        net_ret = (exit_price - ep) / ep - COST_PCT
        try:
            regime = 'bull' if spy.at[entry_date] > spy_sma200.at[entry_date] else 'bear'
        except:
            regime = 'unknown'

        trades.append(dict(
            entry_date=entry_date, exit_date=exit_date, ticker=ticker,
            ret=net_ret, pnl=POS_SIZE * net_ret,
            exit_reason=reason, regime=regime,
            hold_days=(exit_date - entry_date).days,
        ))
        open_pos.append({'exit_date': exit_date})

    return pd.DataFrame(trades) if trades else None


def backtest_C_vol_scaled(entries, data):
    """C) Volatility-scaled TP/SL: wider when VIX high (>25), tighter when VIX low (<15)."""
    close = data['close']
    spy, spy_sma200 = _spy_regime_series(data)
    vix = close.get('^VIX') if '^VIX' in close.columns else None
    trades = []
    open_pos = []

    for entry_date, ticker in entries:
        open_pos = [p for p in open_pos if p['exit_date'] > entry_date]
        if len(open_pos) >= MAX_CONCURRENT:
            continue
        try:
            ep = close.at[entry_date, ticker]
        except:
            continue
        if pd.isna(ep) or ep <= 0:
            continue

        # Get VIX at entry to set TP/SL
        if vix is not None and entry_date in vix.index:
            vix_val = vix.at[entry_date]
        else:
            vix_val = 20.0  # neutral default

        if vix_val > 25:
            tp, sl = 0.15, -0.20   # wide in fear
        elif vix_val < 15:
            tp, sl = 0.07, -0.10   # tight in calm
        else:
            tp, sl = 0.10, -0.15   # baseline in neutral

        future = close.index[close.index > entry_date][:21]
        if len(future) == 0:
            continue

        exit_price = close.at[future[-1], ticker] if future[-1] in close.index else ep
        exit_date = future[-1]
        reason = 'hold_expiry'

        for fdate in future:
            try:
                p = close.at[fdate, ticker]
            except:
                continue
            if pd.isna(p):
                continue
            ret = (p - ep) / ep
            if ret >= tp:
                exit_price, exit_date, reason = p, fdate, 'tp'; break
            elif ret <= sl:
                exit_price, exit_date, reason = p, fdate, 'sl'; break

        net_ret = (exit_price - ep) / ep - COST_PCT
        try:
            regime = 'bull' if spy.at[entry_date] > spy_sma200.at[entry_date] else 'bear'
        except:
            regime = 'unknown'

        trades.append(dict(
            entry_date=entry_date, exit_date=exit_date, ticker=ticker,
            ret=net_ret, pnl=POS_SIZE * net_ret,
            exit_reason=reason, regime=regime, vix_at_entry=vix_val,
            hold_days=(exit_date - entry_date).days,
        ))
        open_pos.append({'exit_date': exit_date})

    return pd.DataFrame(trades) if trades else None


def backtest_D_time_weighted(entries, data):
    """D) Time-weighted exit: TP target shrinks from 10% to 5% linearly over 21 days."""
    close = data['close']
    spy, spy_sma200 = _spy_regime_series(data)
    trades = []
    open_pos = []

    for entry_date, ticker in entries:
        open_pos = [p for p in open_pos if p['exit_date'] > entry_date]
        if len(open_pos) >= MAX_CONCURRENT:
            continue
        try:
            ep = close.at[entry_date, ticker]
        except:
            continue
        if pd.isna(ep) or ep <= 0:
            continue

        future = close.index[close.index > entry_date][:21]
        if len(future) == 0:
            continue

        exit_price = close.at[future[-1], ticker] if future[-1] in close.index else ep
        exit_date = future[-1]
        reason = 'hold_expiry'
        max_days = len(future)

        for j, fdate in enumerate(future):
            try:
                p = close.at[fdate, ticker]
            except:
                continue
            if pd.isna(p):
                continue
            ret = (p - ep) / ep
            # TP decreases linearly: 10% at day 0 → 5% at day 21
            time_frac = j / max(max_days - 1, 1)
            tp_dynamic = 0.10 - 0.05 * time_frac  # 10% → 5%
            sl = -0.15  # SL stays fixed

            if ret >= tp_dynamic:
                exit_price, exit_date, reason = p, fdate, 'tp'; break
            elif ret <= sl:
                exit_price, exit_date, reason = p, fdate, 'sl'; break

        net_ret = (exit_price - ep) / ep - COST_PCT
        try:
            regime = 'bull' if spy.at[entry_date] > spy_sma200.at[entry_date] else 'bear'
        except:
            regime = 'unknown'

        trades.append(dict(
            entry_date=entry_date, exit_date=exit_date, ticker=ticker,
            ret=net_ret, pnl=POS_SIZE * net_ret,
            exit_reason=reason, regime=regime,
            hold_days=(exit_date - entry_date).days,
        ))
        open_pos.append({'exit_date': exit_date})

    return pd.DataFrame(trades) if trades else None


def backtest_E_regime_conditioned(entries, data):
    """E) Regime-conditioned: bull TP=12%/SL=-12%, bear TP=8%/SL=-20%."""
    close = data['close']
    spy, spy_sma200 = _spy_regime_series(data)
    trades = []
    open_pos = []

    for entry_date, ticker in entries:
        open_pos = [p for p in open_pos if p['exit_date'] > entry_date]
        if len(open_pos) >= MAX_CONCURRENT:
            continue
        try:
            ep = close.at[entry_date, ticker]
        except:
            continue
        if pd.isna(ep) or ep <= 0:
            continue

        try:
            is_bull = spy.at[entry_date] > spy_sma200.at[entry_date]
        except:
            is_bull = True

        # Bull: tighter SL (cut losses fast), higher TP (let it run)
        # Bear: wider SL (more volatile, avoid fake stops), lower TP (take what you can get)
        if is_bull:
            tp, sl = 0.12, -0.12
            regime = 'bull'
        else:
            tp, sl = 0.08, -0.20
            regime = 'bear'

        future = close.index[close.index > entry_date][:21]
        if len(future) == 0:
            continue

        exit_price = close.at[future[-1], ticker] if future[-1] in close.index else ep
        exit_date = future[-1]
        reason = 'hold_expiry'

        for fdate in future:
            try:
                p = close.at[fdate, ticker]
            except:
                continue
            if pd.isna(p):
                continue
            ret = (p - ep) / ep
            if ret >= tp:
                exit_price, exit_date, reason = p, fdate, 'tp'; break
            elif ret <= sl:
                exit_price, exit_date, reason = p, fdate, 'sl'; break

        net_ret = (exit_price - ep) / ep - COST_PCT
        trades.append(dict(
            entry_date=entry_date, exit_date=exit_date, ticker=ticker,
            ret=net_ret, pnl=POS_SIZE * net_ret,
            exit_reason=reason, regime=regime,
            hold_days=(exit_date - entry_date).days,
        ))
        open_pos.append({'exit_date': exit_date})

    return pd.DataFrame(trades) if trades else None


def backtest_F_momentum_cont(entries, data):
    """F) Momentum continuation: extend hold to 42d if 3-day momentum still positive."""
    close = data['close']
    spy, spy_sma200 = _spy_regime_series(data)
    trades = []
    open_pos = []

    for entry_date, ticker in entries:
        open_pos = [p for p in open_pos if p['exit_date'] > entry_date]
        if len(open_pos) >= MAX_CONCURRENT:
            continue
        try:
            ep = close.at[entry_date, ticker]
        except:
            continue
        if pd.isna(ep) or ep <= 0:
            continue

        future = close.index[close.index > entry_date][:42]
        if len(future) == 0:
            continue

        c = close[ticker]
        exit_price = close.at[future[-1], ticker] if future[-1] in close.index else ep
        exit_date = future[-1]
        reason = 'hold_expiry'

        for j, fdate in enumerate(future):
            try:
                p = close.at[fdate, ticker]
            except:
                continue
            if pd.isna(p):
                continue
            ret = (p - ep) / ep

            if ret >= 0.10:
                exit_price, exit_date, reason = p, fdate, 'tp'; break
            elif ret <= -0.15:
                exit_price, exit_date, reason = p, fdate, 'sl'; break

            # After day 21, only continue if momentum is still positive
            if j >= 20:
                # 3-day momentum: is price above where it was 3 days ago?
                idx = c.index.get_loc(fdate)
                if idx >= 3:
                    p3ago = c.iloc[idx - 3]
                    if not pd.isna(p3ago) and p <= p3ago:
                        # Momentum fading — exit
                        exit_price, exit_date, reason = p, fdate, 'mom_fade'; break

        net_ret = (exit_price - ep) / ep - COST_PCT
        try:
            regime = 'bull' if spy.at[entry_date] > spy_sma200.at[entry_date] else 'bear'
        except:
            regime = 'unknown'

        trades.append(dict(
            entry_date=entry_date, exit_date=exit_date, ticker=ticker,
            ret=net_ret, pnl=POS_SIZE * net_ret,
            exit_reason=reason, regime=regime,
            hold_days=(exit_date - entry_date).days,
        ))
        open_pos.append({'exit_date': exit_date})

    return pd.DataFrame(trades) if trades else None


# ── 5. 5-GATE VALIDATION ──────────────────────────────────────────────────────
def five_gate_validation(trades_df, entries, data, label):
    """Full 5-gate adversarial check."""
    close = data['close']
    spy, spy_sma200 = _spy_regime_series(data)
    results = {'label': label, 'gates': {}, 'passed': False, 'n_failed': 0}

    if trades_df is None or len(trades_df) == 0:
        results['gates'] = {g: False for g in ['regime_gap', 'perm_test', 'subperiod', 'mdd', 'min_n']}
        results['n_failed'] = 5
        return results

    rets = trades_df['ret'].values
    pnls = trades_df['pnl'].values

    # Gate 1: Regime gap
    bull_rets = trades_df[trades_df['regime'] == 'bull']['ret'].values
    bear_rets = trades_df[trades_df['regime'] == 'bear']['ret'].values
    sh_bull = _sharpe(bull_rets) if len(bull_rets) >= 5 else 0.0
    sh_bear = _sharpe(bear_rets) if len(bear_rets) >= 5 else 0.0
    denom = max(abs(sh_bull), abs(sh_bear), 1e-6)
    regime_gap = abs(sh_bull - sh_bear) / denom
    gate1 = regime_gap < REGIME_GAP_LIMIT
    results['gates']['regime_gap'] = {'pass': gate1, 'gap': round(regime_gap, 3),
                                       'sharpe_bull': round(sh_bull, 3), 'sharpe_bear': round(sh_bear, 3)}

    # Gate 2: Permutation test (randomize entry dates)
    real_sharpe = _sharpe(rets)
    all_dates = close.index[close.index >= pd.Timestamp(BACKTEST_START)].tolist()
    perm_sharpes = []
    n_signals = len(entries)

    np.random.seed(SEED)
    for _ in range(N_PERMS):
        rand_dates = np.random.choice(all_dates, size=n_signals, replace=False)
        tickers_shuffled = [t for _, t in entries]
        perm_rets = []
        for rd, t in zip(sorted(rand_dates), tickers_shuffled):
            rd = pd.Timestamp(rd)
            if t not in close.columns or rd not in close.index:
                continue
            ep = close.at[rd, t]
            if pd.isna(ep) or ep <= 0:
                continue
            future = close.index[close.index > rd][:21]
            if len(future) == 0:
                continue
            xp = close.at[future[-1], t] if future[-1] in close.index else ep
            for fd in future:
                try:
                    p = close.at[fd, t]
                except:
                    continue
                if pd.isna(p):
                    continue
                r = (p - ep) / ep
                if r >= 0.10:
                    xp = p; break
                elif r <= -0.15:
                    xp = p; break
            nr = (xp - ep) / ep - COST_PCT
            perm_rets.append(nr)
        if perm_rets:
            perm_sharpes.append(_sharpe(perm_rets))

    perm_sharpes = np.array(perm_sharpes)
    p_value = float((perm_sharpes >= real_sharpe).mean()) if len(perm_sharpes) else 1.0
    gate2 = p_value < 0.05
    results['gates']['perm_test'] = {'pass': gate2, 'p_value': round(p_value, 4),
                                      'real_sharpe': round(real_sharpe, 3),
                                      'perm_mean': round(float(perm_sharpes.mean()), 3) if len(perm_sharpes) else 0}

    # Gate 3: All 4 sub-periods positive
    bt_start = pd.Timestamp(BACKTEST_START)
    bt_end = pd.Timestamp(END_DATE)
    total_days = (bt_end - bt_start).days
    sub_size = total_days // 4
    sub_periods = [(bt_start + timedelta(days=i * sub_size),
                    bt_start + timedelta(days=(i + 1) * sub_size)) for i in range(4)]
    sub_returns = []
    for sp_start, sp_end in sub_periods:
        sub = trades_df[(trades_df['entry_date'] >= sp_start) & (trades_df['entry_date'] < sp_end)]
        sub_pnl = sub['pnl'].sum() if len(sub) > 0 else 0.0
        sub_returns.append(sub_pnl)
    all_positive = all(r > 0 for r in sub_returns)
    gate3 = all_positive
    results['gates']['subperiod'] = {'pass': gate3, 'sub_pnls': [round(r, 2) for r in sub_returns]}

    # Gate 4: MDD > -50%
    mdd = _mdd(pnls)
    # MDD in % of starting capital (2 positions * $300 = $600)
    mdd_pct = mdd / (POS_SIZE * MAX_CONCURRENT) * 100
    gate4 = mdd_pct > -50.0
    results['gates']['mdd'] = {'pass': gate4, 'mdd_pct': round(mdd_pct, 2), 'mdd_abs': round(mdd, 2)}

    # Gate 5: N >= 20
    gate5 = len(trades_df) >= 20
    results['gates']['min_n'] = {'pass': gate5, 'n': len(trades_df)}

    passed_gates = sum([gate1, gate2, gate3, gate4, gate5])
    results['n_passed'] = passed_gates
    results['n_failed'] = 5 - passed_gates
    results['passed'] = passed_gates == 5
    return results


# ── 6. MAIN ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # Download data
    data = download_data()

    # Generate entries
    entries = generate_entries(data)

    print(f"\n[BACKTEST] Running baseline + 6 exit variants...")

    # Run all backtests
    print("  [Baseline] +10% TP / -15% SL / 21d...")
    df_base = backtest_baseline(entries, data)

    print("  [A] Trailing stop (50% giveback from peak)...")
    df_A = backtest_A_trailing(entries, data, trail_giveback=0.50)

    print("  [B] Signal-based exit (RSI>70 or 5 consec up days)...")
    df_B = backtest_B_signal_exit(entries, data)

    print("  [C] Volatility-scaled TP/SL (VIX-conditioned)...")
    df_C = backtest_C_vol_scaled(entries, data)

    print("  [D] Time-weighted exit (TP shrinks 10%→5% over 21d)...")
    df_D = backtest_D_time_weighted(entries, data)

    print("  [E] Regime-conditioned exit (bull/bear TP/SL)...")
    df_E = backtest_E_regime_conditioned(entries, data)

    print("  [F] Momentum continuation (extend to 42d if still running)...")
    df_F = backtest_F_momentum_cont(entries, data)

    all_results = [
        ('Baseline (+10%TP/-15%SL/21d)', df_base),
        ('A: Trailing Stop (50% giveback)', df_A),
        ('B: Signal Exit (RSI>70/5up)', df_B),
        ('C: VIX-Scaled TP/SL', df_C),
        ('D: Time-Weighted TP', df_D),
        ('E: Regime-Conditioned', df_E),
        ('F: Momentum Continuation', df_F),
    ]

    # ── Summary table
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    fmt = "{:<35} {:>6} {:>7} {:>7} {:>7} {:>7} {:>9} {:>7} {:>8}"
    print(fmt.format("Strategy", "N", "Sharpe", "Sortino", "WR%", "PF", "MDD%", "Hold", "Beat?"))
    print("-" * 90)

    winners = []
    all_metrics = {}
    for name, df in all_results:
        m = _metrics(df)
        all_metrics[name] = m
        beat = "YES ✓" if (name != 'Baseline (+10%TP/-15%SL/21d)' and
                           m['sharpe'] > BASELINE_SHARPE and m['n'] >= MIN_N) else (
                           "BASELINE" if name == 'Baseline (+10%TP/-15%SL/21d)' else "no")
        print(fmt.format(
            name[:35],
            m['n'],
            f"{m['sharpe']:.3f}",
            f"{m['sortino']:.3f}",
            f"{m['wr']*100:.1f}%",
            f"{min(m['pf'], 99.9):.2f}",
            f"{m['mdd']:.0f}",
            f"{m['avg_hold']:.1f}d",
            beat,
        ))
        if (name != 'Baseline (+10%TP/-15%SL/21d)' and
                m['sharpe'] > BASELINE_SHARPE and m['n'] >= MIN_N):
            winners.append((name, df, m))

    print("-" * 90)
    print(f"\nBaseline Sharpe: {BASELINE_SHARPE} | Threshold: Sharpe > {BASELINE_SHARPE} AND N >= {MIN_N}")
    print(f"Winners found: {len(winners)}")

    # ── 5-Gate validation on winners
    validation_results = []
    if winners:
        print("\n" + "=" * 90)
        print("5-GATE VALIDATION ON WINNERS")
        print("=" * 90)
        for name, df, m in winners:
            print(f"\n  Validating: {name} (Sharpe={m['sharpe']:.3f}, N={m['n']})")
            val = five_gate_validation(df, entries, data, name)
            validation_results.append(val)

            gates = val['gates']
            print(f"    Gate 1 Regime Gap: {'PASS' if gates['regime_gap']['pass'] else 'FAIL'} "
                  f"(gap={gates['regime_gap']['gap']}, bull={gates['regime_gap']['sharpe_bull']}, bear={gates['regime_gap']['sharpe_bear']})")
            print(f"    Gate 2 Perm Test:  {'PASS' if gates['perm_test']['pass'] else 'FAIL'} "
                  f"(p={gates['perm_test']['p_value']}, perm_mean={gates['perm_test']['perm_mean']})")
            print(f"    Gate 3 Sub-periods:{'PASS' if gates['subperiod']['pass'] else 'FAIL'} "
                  f"(pnls={gates['subperiod']['sub_pnls']})")
            print(f"    Gate 4 MDD:        {'PASS' if gates['mdd']['pass'] else 'FAIL'} "
                  f"(MDD={gates['mdd']['mdd_pct']:.1f}%)")
            print(f"    Gate 5 Min N:      {'PASS' if gates['min_n']['pass'] else 'FAIL'} "
                  f"(N={gates['min_n']['n']})")
            print(f"    OVERALL: {val['n_passed']}/5 gates passed — {'APPROVED ✓' if val['passed'] else 'REJECTED ✗'}")
    else:
        print("\n  No winners to validate. All variants at or below baseline.")

    # ── Exit reason breakdown for each variant
    print("\n" + "=" * 90)
    print("EXIT REASON BREAKDOWN")
    print("=" * 90)
    for name, df in all_results:
        if df is not None and len(df) > 0:
            reason_counts = df['exit_reason'].value_counts()
            print(f"  {name[:40]}:")
            for reason, cnt in reason_counts.items():
                print(f"    {reason}: {cnt} ({cnt/len(df)*100:.1f}%)")

    # ── Save results
    elapsed = time.time() - t0
    output = {
        'run_date': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'baseline': {'sharpe': BASELINE_SHARPE, 'n': BASELINE_N},
        'metrics': {k: v for k, v in all_metrics.items()},
        'winners': [{'name': n, 'metrics': m} for n, _, m in winners],
        'validation': validation_results,
        'n_signal_entries': len(entries),
    }
    out_file = OUTPUT_DIR / 'adaptive_exit_results.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[DONE] Results saved to {out_file}")
    print(f"[TIME] Elapsed: {elapsed:.1f}s")

    # ── Final verdict
    print("\n" + "=" * 90)
    print("FINAL VERDICT")
    print("=" * 90)
    fully_validated = [v for v in validation_results if v.get('passed')]
    if fully_validated:
        print(f"  {len(fully_validated)} variant(s) BEAT baseline AND passed all 5 gates:")
        for v in fully_validated:
            m = all_metrics[v['label']]
            print(f"    → {v['label']}: Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
                  f"WR={m['wr']*100:.1f}%, N={m['n']}, MDD={m['mdd']:.0f}")
    elif winners:
        print(f"  {len(winners)} variant(s) beat baseline on Sharpe but failed adversarial gates:")
        for n, _, m in winners:
            print(f"    → {n}: Sharpe={m['sharpe']:.3f} — see gate results above")
    else:
        print("  No exit variant beat the baseline. Baseline exits remain optimal for Bond Yield Signal B.")
        print("  Recommendation: Keep using the simple +10%TP / -15%SL / 21d hold.")


if __name__ == '__main__':
    main()
