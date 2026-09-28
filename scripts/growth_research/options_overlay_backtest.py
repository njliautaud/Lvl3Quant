#!/usr/bin/env python3
"""
Options Overlay Backtest — Signal Scoring Strategy #14 Entries
==============================================================
Tests 6 option structure variants on the same dip-buying signals
that validated megacap equity strategies use. Same entries, different
risk structures.

Variants:
  A) ATM Call, 30 DTE
  B) 5% OTM Call, 30 DTE
  C) Bull Call Spread (ATM long / 5% OTM short), 30 DTE
  D) ATM Call, 7 DTE (weekly)
  E) Cash-Secured Put (ATM put sell), 30 DTE
  F) Bull Put Spread (ATM sell / 5% OTM put buy), 30 DTE

Pricing: Black-Scholes mid, VIX as proxy IV
Sizing: $100 max premium per trade (or $100 max risk for spreads)
Gates: 5-gate validation per variant
"""

import os, sys, warnings, functools, pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy.stats import norm
from collections import defaultdict

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

# ─── Config ───────────────────────────────────────────────────────────
START_DATE       = '2019-01-01'
END_DATE         = '2026-07-01'
BACKTEST_START   = '2020-01-01'
MAX_PREMIUM      = 100.0          # Max premium / risk per trade ($)
MAX_CONCURRENT   = 2
HOLD_DAYS        = 21             # Max hold (option exit rule matches equity)
PROFIT_TARGET    = 0.10           # 10% TP on underlying price (for early exit)
STOP_LOSS        = -0.15          # -15% SL on underlying
RISK_FREE        = 0.045          # 4.5%
COMMISSION       = 0.65           # Per contract per leg
CONTRACT_MULT    = 100            # 1 contract = 100 shares
N_PERMS          = 1000
REGIME_GAP_LIMIT = 0.50

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
    'MA', 'WMT', 'PEP', 'KO', 'TMO', 'ABT', 'BAC', 'XOM', 'CVX', 'CSCO',
]
MACRO_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', '^TNX']

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/options_overlay')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("OPTIONS OVERLAY BACKTEST — Signal Scoring #14 Entries")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD (with cache reuse from existing backtests)
# ═══════════════════════════════════════════════════════════════════════
def download_data():
    import yfinance as yf
    # Try to reuse existing caches
    cache_candidates = [
        Path('/home/jupiter/Lvl3Quant/output/growth_research/cross_signal_confluence/_confluence_cache.pkl'),
        Path('/home/jupiter/Lvl3Quant/output/growth_research/cross_signal_confluence_v2/_confluence_v2_cache.pkl'),
        Path('/home/jupiter/Lvl3Quant/output/growth_research/signal_scoring_portfolio/_scoring_cache.pkl'),
        OUTPUT_DIR / '_options_cache.pkl',
    ]
    for cache_file in cache_candidates:
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            if 'close' in data and len(data['close'].columns) >= 10:
                print(f"  Reused cache: {len(data['close'].columns)} tickers, {len(data['close'])} days")
                return data

    print(f"\n[1] Downloading {len(UNIVERSE)} stocks + macro tickers...")
    all_tickers = UNIVERSE + MACRO_TICKERS
    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']; high = raw['High']; low = raw['Low']; volume = raw['Volume']
    else:
        close = high = low = volume = raw
    # flatten if needed
    for df in [close, high, low, volume]:
        if hasattr(df, 'columns') and hasattr(df.columns, 'droplevel'):
            try:
                if df.columns.nlevels > 1:
                    df.columns = df.columns.droplevel(1)
            except Exception:
                pass
    close = close.ffill().dropna(how='all')
    data = {
        'close': close,
        'high': high.reindex(close.index).ffill(),
        'low': low.reindex(close.index).ffill(),
        'volume': volume.reindex(close.index).ffill().fillna(0),
    }
    with open(OUTPUT_DIR / '_options_cache.pkl', 'wb') as f:
        pickle.dump(data, f)
    return data

# ═══════════════════════════════════════════════════════════════════════
# 2. BLACK-SCHOLES PRICING
# ═══════════════════════════════════════════════════════════════════════
def bs_price(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes option price. T in years."""
    if T <= 0 or sigma <= 0:
        # Intrinsic value only
        if option_type == 'call':
            return max(0, S - K)
        else:
            return max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_price_safe(S, K, T, r, sigma, option_type='call'):
    try:
        return float(bs_price(S, K, T, r, sigma, option_type))
    except Exception:
        return 0.0

# ═══════════════════════════════════════════════════════════════════════
# 3. SIGNAL SCORING (Strategy #14 — regime-conditioned)
# ═══════════════════════════════════════════════════════════════════════
def compute_signals(data):
    """Compute the 7 signals for every stock every day. Return score matrix."""
    close = data['close']
    high  = data['high']
    low   = data['low']

    stocks = [t for t in UNIVERSE if t in close.columns]
    spy    = close['SPY'] if 'SPY' in close.columns else None
    vix    = close['^VIX'] if '^VIX' in close.columns else None

    print(f"\n[2] Computing signals for {len(stocks)} stocks...")

    # Pre-compute macro features
    spy_sma200 = spy.rolling(200).mean() if spy is not None else None
    bull_regime = (spy > spy_sma200) if spy_sma200 is not None else pd.Series(True, index=close.index)

    # Realized vol: 20-day rolling std of daily returns, annualized
    spy_rvol = spy.pct_change().rolling(20).std() * np.sqrt(252) * 100 if spy is not None else None

    # Bond yield approx: TLT-based (inverse proxy)
    tlt = close['TLT'] if 'TLT' in close.columns else None
    tnx = close['^TNX'] if '^TNX' in close.columns else None

    all_scores = {}

    for ticker in stocks:
        px = close[ticker].dropna()
        if len(px) < 60:
            continue

        idx = px.index

        # Stock features
        sma20  = px.rolling(20).mean()
        sma50  = px.rolling(50).mean()
        hl_range = (high[ticker] - low[ticker]).reindex(idx) if ticker in high.columns else pd.Series(np.nan, index=idx)
        hl_avg60 = hl_range.rolling(60).mean()

        # RSI
        delta   = px.diff()
        gain    = delta.clip(lower=0).rolling(14).mean()
        loss    = (-delta.clip(upper=0)).rolling(14).mean()
        rs      = gain / loss.replace(0, np.nan)
        rsi     = 100 - 100 / (1 + rs)

        # Daily returns
        ret     = px.pct_change()
        # 3-day consecutive red: returns negative and deepening
        red1    = ret < 0
        red2    = ret.shift(1) < ret.shift(2)   # each day worse than prior

        # Align vix and spy to this ticker's index
        vix_a   = vix.reindex(idx).ffill() if vix is not None else pd.Series(np.nan, index=idx)
        rvol_a  = spy_rvol.reindex(idx).ffill() if spy_rvol is not None else pd.Series(np.nan, index=idx)
        vix3m_a = close['^VIX3M'].reindex(idx).ffill() if '^VIX3M' in close.columns else pd.Series(np.nan, index=idx)

        # Score components (7 signals):
        s1 = (  # IV-RV Gap
            (vix_a > 20) &
            ((vix_a - rvol_a) >= 5) &
            (px < sma20 * 0.95) &
            (rsi < 40)
        ).astype(int)

        # RSI divergence: price lower low vs 14 days ago, RSI higher low
        price_lower = px < px.shift(14)
        rsi_higher  = rsi > rsi.shift(14)
        s2 = (price_lower & rsi_higher & (rsi < 40)).astype(int)

        # Bond yield signal: TLT-based proxy (if TLT rose 1%+ in 5d → yields fell)
        if tlt is not None:
            tlt_a  = tlt.reindex(idx).ffill()
            tlt_5d = tlt_a.pct_change(5)
            bond_signal = (tlt_5d > 0.01) & (px < sma20 * 0.95)
        elif tnx is not None:
            tnx_a  = tnx.reindex(idx).ffill()
            tnx_5d = tnx_a.diff(5)
            bond_signal = (tnx_5d < -0.10) & (px < sma20 * 0.95)
        else:
            bond_signal = pd.Series(False, index=idx)
        s3 = bond_signal.astype(int)

        s4 = (  # Liquidity
            (hl_range < hl_avg60 * 0.85) &
            (rsi < 40)
        ).astype(int)

        s5 = (  # VIX Term Structure
            ((vix_a / vix3m_a.replace(0, np.nan)) > 1.0) &
            (px < sma20 * 0.95)
        ).astype(int)

        # Consecutive dip: 3+ red days, each more negative
        ret_neg   = ret < 0
        ret_deep  = ret < ret.shift(1)
        s6 = (ret_neg & ret_neg.shift(1) & ret_neg.shift(2) &
              ret_deep & ret_deep.shift(1)).astype(int)

        s7 = (  # Base MR
            (rsi < 30) &
            (px < px.rolling(50).max() * 0.93)
        ).astype(int)

        score = s1 + s2 + s3 + s4 + s5 + s6 + s7
        all_scores[ticker] = score.reindex(close.index).fillna(0)

    scores_df   = pd.DataFrame(all_scores)
    bull_df     = bull_regime.reindex(close.index).fillna(True)
    return scores_df, bull_df, stocks

# ═══════════════════════════════════════════════════════════════════════
# 4. ENTRY SIGNAL — regime conditioned
# ═══════════════════════════════════════════════════════════════════════
def get_entries(scores_df, bull_df):
    """Return DataFrame of (date, ticker) with True if signal fires."""
    bull_thresh = 3
    bear_thresh = 1
    entries = {}
    for ticker in scores_df.columns:
        sc   = scores_df[ticker]
        bull = bull_df
        cond = ((bull) & (sc >= bull_thresh)) | ((~bull) & (sc >= bear_thresh))
        entries[ticker] = cond
    return pd.DataFrame(entries)

# ═══════════════════════════════════════════════════════════════════════
# 5. OPTION STRUCTURE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════
STRUCTURES = {
    'A': {'name': 'ATM Call 30d',       'type': 'long_call',      'dte': 30, 'moneyness': 1.00, 'spread': False},
    'B': {'name': '5%OTM Call 30d',     'type': 'long_call',      'dte': 30, 'moneyness': 1.05, 'spread': False},
    'C': {'name': 'Bull Call Spread 30d','type': 'bull_call_spread','dte': 30, 'moneyness': 1.00, 'spread': True},
    'D': {'name': 'ATM Call 7d',        'type': 'long_call',      'dte': 7,  'moneyness': 1.00, 'spread': False},
    'E': {'name': 'Short ATM Put 30d',  'type': 'short_put',      'dte': 30, 'moneyness': 1.00, 'spread': False},
    'F': {'name': 'Bull Put Spread 30d','type': 'bull_put_spread','dte': 30, 'moneyness': 1.00, 'spread': True},
}

def price_structure(struct_key, S, vix_level, dte, r=RISK_FREE):
    """
    Price an option structure at entry.
    Returns: (entry_cost, max_loss, max_gain, legs) where cost>0=debit, <0=credit
    legs = list of (option_type, K, contracts, sign) where sign=+1 long/-1 short
    """
    cfg   = STRUCTURES[struct_key]
    sigma = max(vix_level / 100.0, 0.10)   # VIX as IV proxy
    T     = dte / 252.0
    K_atm = S                               # ATM
    K_otm = S * 1.05                        # 5% OTM for calls
    K_otp = S * 0.95                        # 5% OTM for puts

    stype = cfg['type']

    if stype == 'long_call':
        K  = S * cfg['moneyness']
        px = bs_price_safe(S, K, T, r, sigma, 'call')
        if px <= 0.01:
            return None
        # Size: how many contracts fit in $MAX_PREMIUM
        n = max(1, int(MAX_PREMIUM / (px * CONTRACT_MULT)))
        cost      = px * n * CONTRACT_MULT
        comm      = COMMISSION * n           # 1 leg
        entry_net = cost + comm
        if entry_net > MAX_PREMIUM * 1.5:    # reject if too expensive
            return None
        return {
            'entry_debit': entry_net,
            'max_loss':    entry_net,         # premium paid
            'max_gain':    float('inf'),       # unlimited
            'legs': [('call', K, n, +1)],
            'sigma': sigma, 'T0': T, 'dte': dte,
        }

    elif stype == 'bull_call_spread':
        K_lo = S * 1.00
        K_hi = S * 1.05
        px_lo = bs_price_safe(S, K_lo, T, r, sigma, 'call')
        px_hi = bs_price_safe(S, K_hi, T, r, sigma, 'call')
        spread_debit = px_lo - px_hi
        if spread_debit <= 0.01:
            return None
        n = max(1, int(MAX_PREMIUM / (spread_debit * CONTRACT_MULT)))
        cost      = spread_debit * n * CONTRACT_MULT
        comm      = COMMISSION * n * 2       # 2 legs
        entry_net = cost + comm
        max_gain  = (K_hi - K_lo) * n * CONTRACT_MULT - entry_net
        return {
            'entry_debit': entry_net,
            'max_loss':    entry_net,
            'max_gain':    max_gain,
            'legs': [('call', K_lo, n, +1), ('call', K_hi, n, -1)],
            'sigma': sigma, 'T0': T, 'dte': dte,
            'K_lo': K_lo, 'K_hi': K_hi,
        }

    elif stype == 'short_put':
        px = bs_price_safe(S, K_atm, T, r, sigma, 'put')
        if px <= 0.01:
            return None
        n = max(1, int(MAX_PREMIUM / (K_atm * n_for_csp(K_atm))))
        # For CSP: risk = K*100 per contract (assignment risk), but we cap at $100 premium-equivalent
        # We size by premium collected = $100 target
        n = max(1, int(MAX_PREMIUM / (px * CONTRACT_MULT)))
        credit    = px * n * CONTRACT_MULT
        comm      = COMMISSION * n
        entry_net = credit - comm             # net credit received
        max_loss  = K_atm * n * CONTRACT_MULT  # full assignment (theoretical)
        return {
            'entry_debit': -entry_net,        # negative = credit
            'max_loss':    max_loss,
            'max_gain':    entry_net,
            'legs': [('put', K_atm, n, -1)],
            'sigma': sigma, 'T0': T, 'dte': dte,
        }

    elif stype == 'bull_put_spread':
        K_sell = S * 1.00     # ATM put sold
        K_buy  = S * 0.95     # 5% OTM put bought (protection)
        px_sell = bs_price_safe(S, K_sell, T, r, sigma, 'put')
        px_buy  = bs_price_safe(S, K_buy, T, r, sigma, 'put')
        credit  = px_sell - px_buy
        if credit <= 0.01:
            return None
        spread_width = K_sell - K_buy         # $, max loss per share
        # Size: max risk = $100
        n = max(1, int(MAX_PREMIUM / (spread_width * CONTRACT_MULT)))
        net_credit = credit * n * CONTRACT_MULT
        comm       = COMMISSION * n * 2
        entry_net  = net_credit - comm
        max_loss   = spread_width * n * CONTRACT_MULT - entry_net
        return {
            'entry_debit': -entry_net,        # negative = credit
            'max_loss':    max_loss,
            'max_gain':    entry_net,
            'legs': [('put', K_sell, n, -1), ('put', K_buy, n, +1)],
            'sigma': sigma, 'T0': T, 'dte': dte,
            'K_lo': K_buy, 'K_hi': K_sell,
        }
    return None

def n_for_csp(K):
    """Dummy helper to avoid NameError in short_put branch."""
    return 1.0

def exit_value(struct_key, pos, S_exit, vix_exit, days_held, r=RISK_FREE):
    """
    Compute exit P&L for a position.
    pos: dict returned by price_structure()
    Returns: pnl (in dollars, net of exit commissions)
    """
    cfg   = STRUCTURES[struct_key]
    stype = cfg['type']
    T_rem = max(0, (pos['dte'] - days_held) / 252.0)
    sigma = max(vix_exit / 100.0 * 0.9, 0.10)  # slight vol mean-revert

    exit_comm = 0.0

    if stype == 'long_call':
        leg_type, K, n, sign = pos['legs'][0]
        px_exit = bs_price_safe(S_exit, K, T_rem, r, sigma, 'call')
        exit_value_total = px_exit * n * CONTRACT_MULT
        exit_comm = COMMISSION * n
        pnl = exit_value_total - pos['entry_debit'] - exit_comm
        return pnl

    elif stype == 'bull_call_spread':
        K_lo = pos['K_lo']; K_hi = pos['K_hi']
        n    = pos['legs'][0][2]
        px_lo = bs_price_safe(S_exit, K_lo, T_rem, r, sigma, 'call')
        px_hi = bs_price_safe(S_exit, K_hi, T_rem, r, sigma, 'call')
        spread_val = (px_lo - px_hi) * n * CONTRACT_MULT
        exit_comm  = COMMISSION * n * 2
        pnl = spread_val - pos['entry_debit'] - exit_comm
        # Cap at max_gain
        return min(pnl, pos['max_gain'])

    elif stype == 'short_put':
        leg_type, K, n, sign = pos['legs'][0]
        px_exit = bs_price_safe(S_exit, K, T_rem, r, sigma, 'put')
        exit_cost = px_exit * n * CONTRACT_MULT  # cost to close the short
        exit_comm = COMMISSION * n
        premium_received = pos['max_gain']
        pnl = premium_received - exit_cost - exit_comm
        return pnl

    elif stype == 'bull_put_spread':
        K_lo = pos['K_lo']; K_hi = pos['K_hi']
        n    = pos['legs'][0][2]
        px_sell = bs_price_safe(S_exit, K_hi, T_rem, r, sigma, 'put')
        px_buy  = bs_price_safe(S_exit, K_lo, T_rem, r, sigma, 'put')
        cost_to_close = (px_sell - px_buy) * n * CONTRACT_MULT
        exit_comm     = COMMISSION * n * 2
        premium_received = pos['max_gain']
        pnl = premium_received - cost_to_close - exit_comm
        # Cap: max profit = premium, max loss = spread width - premium
        return max(pnl, -pos['max_loss'])

    return 0.0

# ═══════════════════════════════════════════════════════════════════════
# 6. BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════
def run_backtest(struct_key, entries_df, close, vix_series, start_date):
    """
    Run backtest for one structure variant.
    Returns list of trade dicts.
    """
    cfg    = STRUCTURES[struct_key]
    dates  = close.index
    dates  = dates[dates >= start_date]
    stocks = [t for t in entries_df.columns if t in close.columns]

    active_positions = {}    # (ticker, entry_date) -> pos dict
    trades = []

    for date in dates:
        # 1. Check exits for active positions
        to_remove = []
        for key, pos in active_positions.items():
            ticker, entry_date = key
            days_held = (date - entry_date).days
            S_entry   = pos['S_entry']
            S_now     = close[ticker].get(date, np.nan)
            vix_now   = float(vix_series.get(date, 20.0))
            if np.isnan(S_now):
                continue

            # Exit conditions
            ret_pct    = (S_now / S_entry) - 1.0
            max_hold   = days_held >= HOLD_DAYS
            tp_hit     = ret_pct >= PROFIT_TARGET
            sl_hit     = ret_pct <= STOP_LOSS
            expiry_hit = days_held >= cfg['dte']

            if max_hold or tp_hit or sl_hit or expiry_hit:
                pnl = exit_value(struct_key, pos, S_now, vix_now, days_held)
                trades.append({
                    'entry_date':  entry_date,
                    'exit_date':   date,
                    'ticker':      ticker,
                    'S_entry':     S_entry,
                    'S_exit':      S_now,
                    'ret_pct':     ret_pct,
                    'days_held':   days_held,
                    'entry_debit': pos['entry_debit'],
                    'pnl':         pnl,
                    'exit_reason': ('TP' if tp_hit else 'SL' if sl_hit else 'EXPIRY' if expiry_hit else 'MAXHOLD'),
                })
                to_remove.append(key)

        for key in to_remove:
            del active_positions[key]

        # 2. Check entries (if capacity)
        if len(active_positions) >= MAX_CONCURRENT:
            continue
        if date not in entries_df.index:
            continue

        today_entries = entries_df.loc[date]
        candidates = [t for t in stocks if today_entries.get(t, False)]

        # Avoid re-entering same ticker if already in position
        already_held = {k[0] for k in active_positions}
        candidates   = [t for t in candidates if t not in already_held]

        for ticker in candidates:
            if len(active_positions) >= MAX_CONCURRENT:
                break
            S = close[ticker].get(date, np.nan)
            if np.isnan(S) or S <= 0:
                continue
            vix_val = float(vix_series.get(date, 20.0))

            pos = price_structure(struct_key, S, vix_val, cfg['dte'])
            if pos is None:
                continue

            pos['S_entry']    = S
            pos['entry_date'] = date
            active_positions[(ticker, date)] = pos

    # Close any remaining at end
    last_date = dates[-1]
    for key, pos in active_positions.items():
        ticker, entry_date = key
        S_last   = close[ticker].get(last_date, pos['S_entry'])
        vix_last = float(vix_series.get(last_date, 20.0))
        days_held = (last_date - entry_date).days
        ret_pct   = (S_last / pos['S_entry']) - 1.0
        pnl = exit_value(struct_key, pos, S_last, vix_last, days_held)
        trades.append({
            'entry_date':  entry_date,
            'exit_date':   last_date,
            'ticker':      ticker,
            'S_entry':     pos['S_entry'],
            'S_exit':      S_last,
            'ret_pct':     ret_pct,
            'days_held':   days_held,
            'entry_debit': pos['entry_debit'],
            'pnl':         pnl,
            'exit_reason': 'EOD',
        })

    return trades

# ═══════════════════════════════════════════════════════════════════════
# 7. PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════
def compute_metrics(trades, close_index, start_date=BACKTEST_START, label=''):
    if not trades:
        return {'label': label, 'N': 0, 'sharpe': np.nan, 'sortino': np.nan,
                'win_rate': np.nan, 'pf': np.nan, 'mdd': np.nan,
                'total_pnl': 0, 'total_ret': 0, 'avg_premium': np.nan, 'avg_pnl': np.nan}

    df = pd.DataFrame(trades)
    df = df[df['entry_date'] >= pd.Timestamp(start_date)]
    if len(df) == 0:
        return {'label': label, 'N': 0, 'sharpe': np.nan, 'sortino': np.nan,
                'win_rate': np.nan, 'pf': np.nan, 'mdd': np.nan,
                'total_pnl': 0, 'total_ret': 0, 'avg_premium': np.nan, 'avg_pnl': np.nan}

    # Daily P&L series (by exit date)
    daily_pnl = df.groupby('exit_date')['pnl'].sum()
    date_range = pd.date_range(start=start_date, end=close_index[-1], freq='B')
    daily_pnl  = daily_pnl.reindex(date_range, fill_value=0.0)

    N       = len(df)
    wins    = df['pnl'] > 0
    losses  = df['pnl'] < 0

    win_rate = wins.mean()
    gross_w  = df.loc[wins, 'pnl'].sum() if wins.any() else 0
    gross_l  = abs(df.loc[losses, 'pnl'].sum()) if losses.any() else 1e-9
    pf       = gross_w / max(gross_l, 1e-9)

    # Sharpe (annualized, based on daily PnL)
    daily_std = daily_pnl.std()
    sharpe    = (daily_pnl.mean() / max(daily_std, 1e-9)) * np.sqrt(252) if daily_std > 0 else np.nan

    # Sortino
    downside   = daily_pnl[daily_pnl < 0]
    down_std   = downside.std() if len(downside) > 1 else daily_std
    sortino    = (daily_pnl.mean() / max(down_std, 1e-9)) * np.sqrt(252) if down_std > 0 else np.nan

    # MDD
    cum = daily_pnl.cumsum()
    roll_max = cum.cummax()
    drawdown  = cum - roll_max
    mdd       = drawdown.min()

    total_pnl = df['pnl'].sum()
    avg_prem  = df['entry_debit'].abs().mean()
    avg_pnl   = df['pnl'].mean()

    # Total return (as % of capital deployed = N * avg_premium)
    capital   = N * avg_prem if avg_prem > 0 else 1
    total_ret = total_pnl / capital * 100

    return {
        'label': label, 'N': N,
        'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'win_rate': round(win_rate * 100, 1), 'pf': round(pf, 3),
        'mdd': round(mdd, 2), 'total_pnl': round(total_pnl, 2),
        'total_ret': round(total_ret, 1),
        'avg_premium': round(avg_prem, 2), 'avg_pnl': round(avg_pnl, 2),
        '_daily_pnl': daily_pnl,
        '_df': df,
    }

# ═══════════════════════════════════════════════════════════════════════
# 8. 5-GATE VALIDATION
# ═══════════════════════════════════════════════════════════════════════
def five_gate_validation(metrics, trades_df, close_index, bull_series, label=''):
    gates = {}

    daily_pnl = metrics.get('_daily_pnl', pd.Series(dtype=float))
    df        = metrics.get('_df', pd.DataFrame())

    if metrics['N'] < 20:
        print(f"    GATE FAIL [N<20]: {metrics['N']} trades")
        return False, {'N_fail': True}

    # Gate 1: Regime gap < 0.50
    if len(df) > 10 and bull_series is not None:
        bull_al = bull_series.reindex(df['entry_date']).fillna(True)
        bull_trades = df[bull_al.values]
        bear_trades = df[~bull_al.values]
        if len(bull_trades) >= 5 and len(bear_trades) >= 5:
            sh_bull = bull_trades['pnl'].mean() / max(bull_trades['pnl'].std(), 1e-9)
            sh_bear = bear_trades['pnl'].mean() / max(bear_trades['pnl'].std(), 1e-9)
            denom   = max(abs(sh_bull), abs(sh_bear), 1e-9)
            gap     = abs(sh_bull - sh_bear) / denom
            gates['regime_gap'] = round(gap, 3)
            if gap > REGIME_GAP_LIMIT:
                print(f"    GATE FAIL [regime_gap={gap:.3f} > {REGIME_GAP_LIMIT}]")
                gates['pass'] = False
                return False, gates
        else:
            gates['regime_gap'] = 'insufficient_data'

    # Gate 2: Permutation p-value < 0.05 (1000 random timings)
    if len(df) >= 20 and len(daily_pnl) > 50:
        observed_sharpe = metrics['sharpe']
        n_days = len(daily_pnl)
        n_trades = len(df)
        perm_sharpes = []
        for _ in range(N_PERMS):
            # Random trade P&Ls assigned to random days
            random_daily = np.zeros(n_days)
            trade_pnls   = df['pnl'].values
            random_idx   = np.random.choice(n_days, size=n_trades, replace=True)
            for i, idx in enumerate(random_idx):
                random_daily[idx] += trade_pnls[i]
            std_r = random_daily.std()
            sh_r  = (random_daily.mean() / max(std_r, 1e-9)) * np.sqrt(252) if std_r > 0 else 0
            perm_sharpes.append(sh_r)
        p_val = np.mean(np.array(perm_sharpes) >= observed_sharpe)
        gates['perm_p'] = round(p_val, 4)
        if p_val >= 0.05:
            print(f"    GATE FAIL [perm_p={p_val:.4f} >= 0.05]")
            gates['pass'] = False
            return False, gates
    else:
        gates['perm_p'] = 'skipped'

    # Gate 3: All 4 sub-periods positive
    if len(df) >= 20:
        year_range = pd.date_range(start=BACKTEST_START, end=END_DATE, freq='YS')
        sub_periods = [
            (str(year_range[i].year), year_range[i], year_range[i+1])
            for i in range(min(4, len(year_range)-1))
        ]
        # Use 4 equal-length sub-periods
        total_days = len(daily_pnl)
        chunk      = total_days // 4
        sp_results = []
        for q in range(4):
            chunk_pnl = daily_pnl.iloc[q*chunk : (q+1)*chunk]
            sp_results.append(chunk_pnl.sum())
        gates['sub_periods'] = [round(v, 2) for v in sp_results]
        if any(v <= 0 for v in sp_results):
            print(f"    GATE FAIL [sub-period negative: {sp_results}]")
            gates['pass'] = False
            return False, gates

    # Gate 4: MDD > -50%
    gates['mdd'] = metrics['mdd']
    # MDD is in $; compare to avg_prem * N as rough capital
    capital = metrics['N'] * max(metrics['avg_premium'], 1)
    mdd_pct = metrics['mdd'] / capital * 100
    gates['mdd_pct'] = round(mdd_pct, 1)
    if mdd_pct < -50:
        print(f"    GATE FAIL [MDD={mdd_pct:.1f}% < -50%]")
        gates['pass'] = False
        return False, gates

    # Gate 5: N >= 20
    gates['N'] = metrics['N']
    if metrics['N'] < 20:
        print(f"    GATE FAIL [N={metrics['N']} < 20]")
        gates['pass'] = False
        return False, gates

    gates['pass'] = True
    print(f"    ALL 5 GATES PASSED")
    return True, gates

# ═══════════════════════════════════════════════════════════════════════
# 9. EQUITY BASELINE (buy stock, $300 position, same signals)
# ═══════════════════════════════════════════════════════════════════════
def run_equity_baseline(entries_df, close, start_date):
    """Run the original equity strategy for comparison."""
    EQUITY_SIZE = 300.0
    dates  = close.index
    dates  = dates[dates >= start_date]
    stocks = [t for t in entries_df.columns if t in close.columns]

    active = {}
    trades = []

    for date in dates:
        to_remove = []
        for key, pos in active.items():
            ticker, entry_date = key
            days_held = (date - entry_date).days
            S_now     = close[ticker].get(date, np.nan)
            if np.isnan(S_now):
                continue
            ret_pct = (S_now / pos['S_entry']) - 1.0
            if days_held >= HOLD_DAYS or ret_pct >= PROFIT_TARGET or ret_pct <= STOP_LOSS:
                pnl = ret_pct * pos['cost'] - pos['cost'] * 0.001  # spread cost
                trades.append({
                    'entry_date': entry_date, 'exit_date': date,
                    'ticker': ticker, 'S_entry': pos['S_entry'], 'S_exit': S_now,
                    'ret_pct': ret_pct, 'days_held': days_held,
                    'entry_debit': pos['cost'], 'pnl': pnl,
                    'exit_reason': 'TP' if ret_pct >= PROFIT_TARGET else 'SL' if ret_pct <= STOP_LOSS else 'MAXHOLD',
                })
                to_remove.append(key)

        for key in to_remove:
            del active[key]

        if len(active) >= MAX_CONCURRENT:
            continue
        if date not in entries_df.index:
            continue

        today_entries = entries_df.loc[date]
        candidates = [t for t in stocks if today_entries.get(t, False)]
        already_held = {k[0] for k in active}
        candidates   = [t for t in candidates if t not in already_held]

        for ticker in candidates:
            if len(active) >= MAX_CONCURRENT:
                break
            S = close[ticker].get(date, np.nan)
            if np.isnan(S) or S <= 0:
                continue
            shares = EQUITY_SIZE / S
            active[(ticker, date)] = {'S_entry': S, 'cost': EQUITY_SIZE}

    # Force close remaining
    last_date = dates[-1]
    for key, pos in active.items():
        ticker, entry_date = key
        S_last  = close[ticker].get(last_date, pos['S_entry'])
        ret_pct = (S_last / pos['S_entry']) - 1.0
        pnl = ret_pct * pos['cost'] - pos['cost'] * 0.001
        trades.append({
            'entry_date': entry_date, 'exit_date': last_date,
            'ticker': ticker, 'S_entry': pos['S_entry'], 'S_exit': S_last,
            'ret_pct': ret_pct, 'days_held': (last_date - entry_date).days,
            'entry_debit': pos['cost'], 'pnl': pnl, 'exit_reason': 'EOD',
        })

    return trades

# ═══════════════════════════════════════════════════════════════════════
# 10. MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Download data
    data      = download_data()
    close     = data['close']
    vix_s     = close['^VIX'].ffill() if '^VIX' in close.columns else pd.Series(20.0, index=close.index)

    # Compute signals
    scores_df, bull_df, stocks = compute_signals(data)

    # Get entries
    entries_df = get_entries(scores_df, bull_df)
    n_signals  = entries_df.sum().sum()
    print(f"\n  Total signal-days: {int(n_signals)}")

    # Filter to backtest period
    bt_start  = pd.Timestamp(BACKTEST_START)
    bt_close  = close[close.index >= bt_start]

    # Run equity baseline
    print("\n[3] Running equity baseline (buy stock, $300, same signals)...")
    eq_trades = run_equity_baseline(entries_df, close, bt_start)
    eq_met    = compute_metrics(eq_trades, close.index, label='EQUITY_BASELINE')
    print(f"    Equity baseline: N={eq_met['N']}, Sharpe={eq_met['sharpe']}, "
          f"WR={eq_met['win_rate']}%, PF={eq_met['pf']}, Total PnL=${eq_met['total_pnl']:,.0f}")

    # Run all 6 option structures
    print("\n[4] Running 6 option structure variants...")
    all_results = []
    gate_results = {}

    for struct_key, cfg in STRUCTURES.items():
        print(f"\n  --- Variant {struct_key}: {cfg['name']} ---")
        trades = run_backtest(struct_key, entries_df, close, vix_s, bt_start)
        met    = compute_metrics(trades, close.index, label=f"{struct_key}_{cfg['name']}")
        print(f"    N={met['N']}, Sharpe={met['sharpe']}, Sortino={met['sortino']}, "
              f"WR={met['win_rate']}%, PF={met['pf']}, "
              f"Avg Premium=${met['avg_premium']}, Avg PnL=${met['avg_pnl']}, "
              f"Total PnL=${met['total_pnl']:,.0f}")

        # 5-gate validation
        print(f"  Validating {struct_key}:")
        bull_aligned = bull_df.reindex(close.index)
        passed, gates = five_gate_validation(met, met.get('_df', pd.DataFrame()),
                                              close.index, bull_aligned, label=struct_key)
        gates['variant'] = struct_key
        gates['name']    = cfg['name']

        result = {**met, 'gates_passed': passed, 'gates': gates}
        result.pop('_daily_pnl', None)
        result.pop('_df', None)
        all_results.append(result)
        gate_results[struct_key] = gates

    # ── Final Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Variant':<6} {'Name':<25} {'N':>5} {'Sharpe':>8} {'Sortino':>8} "
          f"{'WR%':>6} {'PF':>6} {'MDD$':>9} {'TotPnL':>10} {'Pass?':>6}")
    print("-" * 95)

    # Equity baseline
    print(f"{'EQ':<6} {'Equity Baseline (300)':<25} {eq_met['N']:>5} "
          f"{str(eq_met['sharpe']):>8} {str(eq_met['sortino']):>8} "
          f"{str(eq_met['win_rate']):>6} {str(eq_met['pf']):>6} "
          f"{eq_met['mdd']:>9,.0f} {eq_met['total_pnl']:>10,.0f} {'---':>6}")

    best_sharpe = -999
    best_key    = None
    for r in all_results:
        passed_str = 'PASS' if r['gates_passed'] else 'FAIL'
        print(f"{r['label'][:6]:<6} {STRUCTURES[r['label'][0]]['name']:<25} {r['N']:>5} "
              f"{str(r['sharpe']):>8} {str(r['sortino']):>8} "
              f"{str(r['win_rate']):>6} {str(r['pf']):>6} "
              f"{r['mdd']:>9,.0f} {r['total_pnl']:>10,.0f} {passed_str:>6}")
        if r['gates_passed'] and isinstance(r['sharpe'], float) and r['sharpe'] > best_sharpe:
            best_sharpe = r['sharpe']
            best_key    = r['label'][0]

    print("\n" + "=" * 80)
    if best_key:
        best = STRUCTURES[best_key]
        print(f"BEST VALIDATED VARIANT: {best_key} — {best['name']}")
        best_r = next(r for r in all_results if r['label'].startswith(best_key))
        print(f"  Sharpe: {best_r['sharpe']} | Sortino: {best_r['sortino']} | "
              f"WR: {best_r['win_rate']}% | PF: {best_r['pf']}")
        print(f"  N trades: {best_r['N']} | Total PnL: ${best_r['total_pnl']:,.0f}")
        print(f"  Avg premium paid: ${best_r['avg_premium']} | Avg trade PnL: ${best_r['avg_pnl']}")
    else:
        print("NO VARIANT PASSED ALL 5 GATES")

    # Gate summary
    print("\nGATE DETAIL PER VARIANT:")
    for struct_key, gates in gate_results.items():
        passed_str = 'PASS' if gates.get('pass', False) else 'FAIL'
        regime_g   = gates.get('regime_gap', 'N/A')
        perm_p     = gates.get('perm_p', 'N/A')
        mdd_pct    = gates.get('mdd_pct', 'N/A')
        n_g        = gates.get('N', 'N/A')
        sp         = gates.get('sub_periods', 'N/A')
        print(f"  {struct_key} [{passed_str}]: RegimeGap={regime_g}, "
              f"PermP={perm_p}, SubPeriods={sp}, MDD%={mdd_pct}, N={n_g}")

    # Save results
    import json
    save_data = []
    for r in all_results:
        save_row = {k: v for k, v in r.items() if not k.startswith('_')}
        if isinstance(save_row.get('gates'), dict):
            # Make gates JSON-serializable
            g = {}
            for k2, v2 in save_row['gates'].items():
                if isinstance(v2, (np.integer, np.floating)):
                    g[k2] = float(v2)
                elif isinstance(v2, np.ndarray):
                    g[k2] = v2.tolist()
                else:
                    g[k2] = v2
            save_row['gates'] = g
        save_data.append(save_row)

    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump({'equity_baseline': {k: v for k, v in eq_met.items() if not k.startswith('_')},
                   'options_variants': save_data,
                   'best_variant': best_key,
                   'run_date': datetime.now().isoformat()}, f, indent=2, default=str)

    print(f"\nResults saved.")
    print("Done.")
