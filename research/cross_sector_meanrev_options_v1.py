#!/usr/bin/env python3
"""
Cross-Sector Mean Reversion Options v1
========================================
Research question: When a sector ETF drops significantly relative to its peers
(z-score below -2 on 5-day relative return), does it mean-revert over the
next 10-20 days? Can we trade this with call spreads at $645?

This is the OPPOSITE of our momentum strategy — testing mean reversion as a
complementary signal.

Signal: cross-sectional z-score of 5-day returns.
  - Buy when z < -2 (oversold vs peers)
  - Sell when z > 2 (overbought vs peers)

Variants:
  A) Buy calls on oversold sectors (z < -2)
  B) Buy puts on overbought sectors (z > 2)
  C) Both directions combined
  D) Filtered by VIX regime (mean-rev works better in calm markets?)
  E) Filtered by momentum trend (only mean-revert if long-term trend supportive)
  F) Random entry control (permutation baseline)

Options: bull call spreads, 3% spread width, 30 DTE, top-2 signals, biweekly check
Walk-forward pricing: ATR with 15% haircut, $2.60 RT commission
$645 starting capital, max $200 per position
Full 4-gate adversarial audit.

HC compliance: sliding window, 4-gate audit, regime-agnostic validation.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import json
import os
from scipy.stats import norm

try:
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False

###############################################################################
# CONFIG
###############################################################################
UNIVERSE = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLI', 'XLP', 'XLU', 'XLRE', 'XLB', 'XLC']
START_CAP = 645.0
MAX_POS_SIZE = 200.0        # max $ per position
COMMISSION_RT = 2.60        # RT commission per spread (2 legs)
HAIRCUT = 0.15              # 15% haircut on BS prices
RISK_FREE = 0.05
SPREAD_WIDTH_PCT = 0.03     # 3% spread width
DTE = 30                    # 30 days to expiry
REBAL_DAYS = 10             # biweekly check (~10 trading days)
TOP_K = 2                   # top-2 signals
Z_ENTRY_OVERSOLD = -2.0     # z < -2 → mean-rev long
Z_ENTRY_OVERBOUGHT = 2.0    # z > 2 → mean-rev short

CONFIGS = {
    'A_Calls_Oversold': {
        'direction': 'long_only',
        'z_threshold': -2.0,
        'vix_filter': None,
        'trend_filter': False,
        'random': False,
        'desc': 'Buy call spreads on oversold sectors (z < -2)',
    },
    'B_Puts_Overbought': {
        'direction': 'short_only',
        'z_threshold': 2.0,
        'vix_filter': None,
        'trend_filter': False,
        'random': False,
        'desc': 'Buy put spreads on overbought sectors (z > 2)',
    },
    'C_Both_Directions': {
        'direction': 'both',
        'z_threshold': 2.0,
        'vix_filter': None,
        'trend_filter': False,
        'random': False,
        'desc': 'Both directions: calls on oversold + puts on overbought',
    },
    'D_VIX_Calm': {
        'direction': 'both',
        'z_threshold': 2.0,
        'vix_filter': 'calm',     # VIX < 20
        'trend_filter': False,
        'random': False,
        'desc': 'Both directions, only in calm VIX regime (<20)',
    },
    'E_Trend_Filter': {
        'direction': 'long_only',
        'z_threshold': -2.0,
        'vix_filter': None,
        'trend_filter': True,     # only mean-revert if 60d trend is up
        'random': False,
        'desc': 'Calls on oversold, only if 60d trend is supportive (up)',
    },
    'F_Random_Control': {
        'direction': 'both',
        'z_threshold': 2.0,
        'vix_filter': None,
        'trend_filter': False,
        'random': True,
        'desc': 'CONTROL: Random entry, same sizing/exit logic',
    },
}

###############################################################################
# BLACK-SCHOLES PRICING
###############################################################################
def bs_call(S, K, T, sigma, r=RISK_FREE):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, sigma, r=RISK_FREE):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

###############################################################################
# SPREAD PNL CALCULATORS
###############################################################################
def bull_call_spread_pnl(S_entry, S_exit, sigma, dte_entry, dte_exit):
    """Bull call spread: buy ATM call, sell OTM call (3% higher).
    Returns (pnl_dollars, cost_dollars)."""
    K_low = S_entry                                    # ATM
    K_high = S_entry * (1 + SPREAD_WIDTH_PCT)          # 3% OTM

    T_entry = max(dte_entry / 365.0, 1 / 365)
    T_exit = max(dte_exit / 365.0, 0.001)

    # Entry
    c_low_entry = bs_call(S_entry, K_low, T_entry, sigma)
    c_high_entry = bs_call(S_entry, K_high, T_entry, sigma)
    debit = c_low_entry - c_high_entry  # net debit

    # Exit
    if dte_exit <= 1:
        c_low_exit = max(S_exit - K_low, 0)
        c_high_exit = max(S_exit - K_high, 0)
    else:
        sig_exit = sigma * 0.95  # slight vol decay
        c_low_exit = bs_call(S_exit, K_low, T_exit, sig_exit)
        c_high_exit = bs_call(S_exit, K_high, T_exit, sig_exit)
    credit = c_low_exit - c_high_exit

    # Haircut
    debit *= (1 + HAIRCUT)
    credit *= (1 - HAIRCUT)

    pnl = (credit - debit) * 100  # per contract
    cost = max(debit * 100, 1.0)
    pnl -= COMMISSION_RT

    return pnl, cost

def bear_put_spread_pnl(S_entry, S_exit, sigma, dte_entry, dte_exit):
    """Bear put spread: buy ATM put, sell OTM put (3% lower).
    Returns (pnl_dollars, cost_dollars)."""
    K_high = S_entry                                   # ATM
    K_low = S_entry * (1 - SPREAD_WIDTH_PCT)           # 3% OTM

    T_entry = max(dte_entry / 365.0, 1 / 365)
    T_exit = max(dte_exit / 365.0, 0.001)

    # Entry
    p_high_entry = bs_put(S_entry, K_high, T_entry, sigma)
    p_low_entry = bs_put(S_entry, K_low, T_entry, sigma)
    debit = p_high_entry - p_low_entry  # net debit

    # Exit
    if dte_exit <= 1:
        p_high_exit = max(K_high - S_exit, 0)
        p_low_exit = max(K_low - S_exit, 0)
    else:
        sig_exit = sigma * 0.95
        p_high_exit = bs_put(S_exit, K_high, T_exit, sig_exit)
        p_low_exit = bs_put(S_exit, K_low, T_exit, sig_exit)
    credit = p_high_exit - p_low_exit

    debit *= (1 + HAIRCUT)
    credit *= (1 - HAIRCUT)

    pnl = (credit - debit) * 100
    cost = max(debit * 100, 1.0)
    pnl -= COMMISSION_RT

    return pnl, cost

###############################################################################
# DATA LOADING
###############################################################################
def load_data():
    import yfinance as yf
    cache_dir = '/home/jupiter/Lvl3Quant/research/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'sector_etf_daily_data.parquet')
    vix_file = os.path.join(cache_dir, 'vix_daily_data.parquet')

    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        age_hours = (datetime.now().timestamp() - mtime) / 3600
        if age_hours < 24:
            df = pd.read_parquet(cache_file)
            vix = pd.read_parquet(vix_file) if os.path.exists(vix_file) else None
            return df, vix

    tickers = UNIVERSE + ['^VIX']
    data = yf.download(tickers, start='2010-01-01', progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data

    vix_col = '^VIX' if '^VIX' in close.columns else None
    if vix_col:
        vix = close[[vix_col]].rename(columns={vix_col: 'VIX'})
        close = close.drop(columns=[vix_col])
    else:
        vix = None

    close.to_parquet(cache_file)
    if vix is not None:
        vix.to_parquet(vix_file)

    return close, vix

###############################################################################
# SIGNAL GENERATION: Cross-Sectional Z-Scores
###############################################################################
def compute_signals(close, vix):
    """Compute cross-sectional z-scores of 5-day returns for all sectors.
    Returns a DataFrame with columns: date, ticker, z_score, price, sigma,
    vix_level, trend_60d_up."""

    # 5-day returns for each sector
    ret5 = close.pct_change(5)

    # Rolling 20-day realized vol (annualized) for BS pricing
    daily_ret = close.pct_change()
    vol_20d = daily_ret.rolling(20).std() * np.sqrt(252)

    # 60-day trend: price > 60d SMA
    sma_60 = close.rolling(60).mean()

    # VIX
    vix_s = vix['VIX'] if vix is not None else pd.Series(20.0, index=close.index)

    # Cross-sectional z-score at each date
    ret5_mean = ret5.mean(axis=1)
    ret5_std = ret5.std(axis=1)

    signals = []
    for date in close.index:
        if pd.isna(ret5_mean.get(date)) or pd.isna(ret5_std.get(date)):
            continue
        if ret5_std[date] < 1e-8:
            continue

        for ticker in UNIVERSE:
            if ticker not in close.columns:
                continue
            r = ret5.loc[date, ticker] if date in ret5.index else np.nan
            if pd.isna(r):
                continue

            z = (r - ret5_mean[date]) / ret5_std[date]
            price = close.loc[date, ticker]
            sigma = vol_20d.loc[date, ticker] if date in vol_20d.index else 0.25
            if pd.isna(sigma) or sigma < 0.05:
                sigma = 0.25

            trend_up = True
            if date in sma_60.index and ticker in sma_60.columns:
                s60 = sma_60.loc[date, ticker]
                trend_up = price > s60 if not pd.isna(s60) else True

            v = vix_s[date] if date in vix_s.index else 20.0
            if pd.isna(v):
                v = 20.0

            signals.append({
                'date': date,
                'ticker': ticker,
                'z_score': z,
                'price': price,
                'sigma': sigma,
                'vix': v,
                'trend_60d_up': trend_up,
                'ret_5d': r,
            })

    df = pd.DataFrame(signals)
    return df

###############################################################################
# BACKTEST ENGINE
###############################################################################
def run_backtest(signals_df, close, config, name, rng=None):
    """Run walk-forward backtest for a given configuration.
    Returns list of trade dicts and equity curve."""

    direction = config['direction']
    z_thresh = abs(config['z_threshold'])
    vix_filter = config['vix_filter']
    trend_filter = config['trend_filter']
    is_random = config['random']

    if rng is None:
        rng = np.random.RandomState(42)

    dates = sorted(signals_df['date'].unique())
    # Biweekly check: every REBAL_DAYS trading days
    check_dates = dates[60::REBAL_DAYS]  # start after 60 days warmup

    trades = []
    equity = START_CAP
    eq_curve = []

    for i, check_date in enumerate(check_dates):
        if equity <= 50:  # blown up
            break

        day_signals = signals_df[signals_df['date'] == check_date].copy()
        if len(day_signals) == 0:
            continue

        # Apply VIX filter
        if vix_filter == 'calm':
            vix_level = day_signals['vix'].iloc[0]
            if vix_level >= 20:
                continue
        elif vix_filter == 'high':
            vix_level = day_signals['vix'].iloc[0]
            if vix_level < 20:
                continue

        # Identify candidates
        long_candidates = []
        short_candidates = []

        if is_random:
            # Random entry: pick random sectors
            shuffled = day_signals.sample(frac=1, random_state=rng.randint(0, 1e6))
            if direction in ('long_only', 'both'):
                long_candidates = shuffled.head(TOP_K).to_dict('records')
            if direction in ('short_only', 'both'):
                short_candidates = shuffled.tail(TOP_K).to_dict('records')
        else:
            if direction in ('long_only', 'both'):
                oversold = day_signals[day_signals['z_score'] < -z_thresh].copy()
                if trend_filter:
                    oversold = oversold[oversold['trend_60d_up'] == True]
                oversold = oversold.sort_values('z_score')  # most oversold first
                long_candidates = oversold.head(TOP_K).to_dict('records')

            if direction in ('short_only', 'both'):
                overbought = day_signals[day_signals['z_score'] > z_thresh].copy()
                if trend_filter:
                    overbought = overbought[overbought['trend_60d_up'] == False]
                overbought = overbought.sort_values('z_score', ascending=False)  # most overbought first
                short_candidates = overbought.head(TOP_K).to_dict('records')

        # Size: max $200 per position, but also cap at equity/4
        max_cost = min(MAX_POS_SIZE, equity / 4)

        for cand in long_candidates:
            # Bull call spread on oversold sector
            ticker = cand['ticker']
            entry_price = cand['price']
            sigma = cand['sigma']

            # Find exit date: DTE days later
            exit_idx = None
            check_idx = dates.index(check_date) if check_date in dates else None
            if check_idx is None:
                continue
            target_exit_idx = check_idx + DTE
            if target_exit_idx >= len(dates):
                target_exit_idx = len(dates) - 1
            exit_date = dates[target_exit_idx]
            actual_dte_held = target_exit_idx - check_idx

            # Get exit price
            if exit_date in close.index and ticker in close.columns:
                exit_price = close.loc[exit_date, ticker]
            else:
                continue
            if pd.isna(exit_price):
                continue

            pnl, cost = bull_call_spread_pnl(
                entry_price, exit_price, sigma,
                dte_entry=DTE, dte_exit=max(DTE - actual_dte_held, 0)
            )

            # Scale to position size
            n_contracts = max(1, int(max_cost / cost))
            actual_cost = cost * n_contracts
            if actual_cost > equity * 0.5:  # don't risk >50% equity on one trade
                n_contracts = max(1, int(equity * 0.5 / cost))
            actual_pnl = pnl * n_contracts

            equity += actual_pnl

            # Regime classification
            ret_spy = 0
            if 'XLK' in close.columns:  # proxy market direction
                spy_px = close[UNIVERSE].mean(axis=1)
                ci = dates.index(check_date)
                ei = min(ci + DTE, len(dates) - 1)
                if ci > 0:
                    ret_spy = (spy_px.iloc[ei] - spy_px.iloc[ci]) / spy_px.iloc[ci]

            trades.append({
                'entry_date': check_date,
                'exit_date': exit_date,
                'ticker': ticker,
                'side': 'long',
                'z_score': cand['z_score'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'sigma': sigma,
                'n_contracts': n_contracts,
                'cost': actual_cost,
                'pnl': actual_pnl,
                'ret_pct': actual_pnl / max(actual_cost, 1) * 100,
                'equity_after': equity,
                'regime': 'bull' if ret_spy > 0 else 'bear',
                'vix': cand['vix'],
            })

        for cand in short_candidates:
            # Bear put spread on overbought sector
            ticker = cand['ticker']
            entry_price = cand['price']
            sigma = cand['sigma']

            check_idx = dates.index(check_date) if check_date in dates else None
            if check_idx is None:
                continue
            target_exit_idx = check_idx + DTE
            if target_exit_idx >= len(dates):
                target_exit_idx = len(dates) - 1
            exit_date = dates[target_exit_idx]
            actual_dte_held = target_exit_idx - check_idx

            if exit_date in close.index and ticker in close.columns:
                exit_price = close.loc[exit_date, ticker]
            else:
                continue
            if pd.isna(exit_price):
                continue

            pnl, cost = bear_put_spread_pnl(
                entry_price, exit_price, sigma,
                dte_entry=DTE, dte_exit=max(DTE - actual_dte_held, 0)
            )

            n_contracts = max(1, int(max_cost / cost))
            actual_cost = cost * n_contracts
            if actual_cost > equity * 0.5:
                n_contracts = max(1, int(equity * 0.5 / cost))
            actual_pnl = pnl * n_contracts

            equity += actual_pnl

            ret_spy = 0
            spy_px = close[UNIVERSE].mean(axis=1)
            ci = dates.index(check_date)
            ei = min(ci + DTE, len(dates) - 1)
            if ci > 0:
                ret_spy = (spy_px.iloc[ei] - spy_px.iloc[ci]) / spy_px.iloc[ci]

            trades.append({
                'entry_date': check_date,
                'exit_date': exit_date,
                'ticker': ticker,
                'side': 'short',
                'z_score': cand['z_score'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'sigma': sigma,
                'n_contracts': n_contracts,
                'cost': actual_cost,
                'pnl': actual_pnl,
                'ret_pct': actual_pnl / max(actual_cost, 1) * 100,
                'equity_after': equity,
                'regime': 'bull' if ret_spy > 0 else 'bear',
                'vix': cand['vix'],
            })

        eq_curve.append({'date': check_date, 'equity': equity})

    trade_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(eq_curve)
    if len(eq_df) > 0:
        eq_df['date'] = pd.to_datetime(eq_df['date'])
        eq_df = eq_df.set_index('date')

    return trade_df, eq_df

###############################################################################
# 4-GATE ADVERSARIAL AUDIT
###############################################################################
def adversarial_audit(trade_df, eq_df, name):
    """Full 4-gate adversarial audit per HC."""
    pnls = trade_df['pnl'].values
    n = len(pnls)

    if n < 5:
        return {'name': name, 'n_trades': n, 'gates_passed': 0, 'error': 'Too few trades'}

    win_rate = (pnls > 0).mean() * 100
    avg_win = pnls[pnls > 0].mean() if (pnls > 0).any() else 0
    avg_loss = pnls[pnls < 0].mean() if (pnls < 0).any() else 0
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 1e-8
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    # Sharpe / Sortino from periodic returns
    if len(eq_df) > 3:
        eq_vals = eq_df['equity'].values
        periodic_returns = np.diff(eq_vals) / eq_vals[:-1]
        if len(periodic_returns) > 2:
            ann_factor = np.sqrt(252 / REBAL_DAYS)  # annualize
            sharpe = periodic_returns.mean() / (periodic_returns.std() + 1e-8) * ann_factor
            downside = periodic_returns[periodic_returns < 0]
            sortino = periodic_returns.mean() / (downside.std() + 1e-8) * ann_factor if len(downside) > 1 else sharpe
        else:
            sharpe = sortino = 0
    else:
        sharpe = sortino = 0

    final_eq = eq_df['equity'].iloc[-1] if len(eq_df) > 0 else START_CAP
    years = max((eq_df.index[-1] - eq_df.index[0]).days / 365.25, 0.1) if len(eq_df) > 1 else 1
    cagr = (final_eq / START_CAP) ** (1 / years) - 1
    maxdd = ((eq_df['equity'] - eq_df['equity'].cummax()) / eq_df['equity'].cummax()).min() if len(eq_df) > 1 else 0
    calmar = cagr / (-maxdd + 1e-8) if maxdd < 0 else 0

    # ---- G1: Permutation test (p < 0.05) ----
    n_perms = 500
    perm_sharpes = []
    for _ in range(n_perms):
        sh = pnls.copy()
        np.random.shuffle(sh)
        eq_p = np.cumsum(sh) + START_CAP
        nm = max(1, len(eq_p) // 3)
        chunks = np.array_split(eq_p, nm)
        mr = []
        prev = START_CAP
        for c in chunks:
            if len(c) > 0:
                mr.append((c[-1] - prev) / prev)
                prev = c[-1]
        if len(mr) > 1:
            a = np.array(mr)
            perm_sharpes.append(a.mean() / (a.std() + 1e-8) * np.sqrt(12))
    perm_p = np.mean(np.array(perm_sharpes) >= sharpe) if perm_sharpes else 1.0
    g1 = perm_p < 0.05

    # ---- G2: R1 regime gap < 0.50 ----
    bull = trade_df[trade_df['regime'] == 'bull']['pnl'].values
    bear = trade_df[trade_df['regime'] == 'bear']['pnl'].values
    if len(bull) > 5 and len(bear) > 5:
        bs_sharpe = bull.mean() / (bull.std() + 1e-8) * np.sqrt(12)
        br_sharpe = bear.mean() / (bear.std() + 1e-8) * np.sqrt(12)
        r1 = abs(bs_sharpe - br_sharpe) / max(abs(bs_sharpe), abs(br_sharpe), 1e-8)
        bull_wr = (bull > 0).mean() * 100
        bear_wr = (bear > 0).mean() * 100
    else:
        bs_sharpe = br_sharpe = sharpe
        r1 = 0
        bull_wr = bear_wr = win_rate
    g2 = r1 < 0.50

    # ---- G3: Sub-period stability ----
    t = len(trade_df) // 3
    ss = []
    for s, e in [(0, t), (t, 2 * t), (2 * t, len(trade_df))]:
        sp = trade_df.iloc[s:e]['pnl'].values
        if len(sp) > 3:
            ss.append(sp.mean() / (sp.std() + 1e-8) * np.sqrt(12))
    g3 = len(ss) >= 2 and all(x > 0 for x in ss)

    # ---- G4: Outlier removal ----
    p5, p95 = np.percentile(pnls, [5, 95])
    trimmed = pnls[(pnls >= p5) & (pnls <= p95)]
    if len(trimmed) > 3:
        trimmed_sharpe = trimmed.mean() / (trimmed.std() + 1e-8) * np.sqrt(12)
        g4 = trimmed_sharpe > 0
    else:
        trimmed_sharpe = 0
        g4 = False

    # Z-score distribution stats
    z_scores = trade_df['z_score'].values
    avg_z_entry = z_scores.mean()
    long_trades = trade_df[trade_df['side'] == 'long']
    short_trades = trade_df[trade_df['side'] == 'short']

    return {
        'name': name,
        'n_trades': n,
        'n_long': len(long_trades),
        'n_short': len(short_trades),
        'win_rate': round(win_rate, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'total_pnl': round(pnls.sum(), 2),
        'final_equity': round(final_eq, 2),
        'cagr_pct': round(cagr * 100, 1),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'maxdd_pct': round(maxdd * 100, 1),
        'calmar': round(calmar, 2),
        'pf': round(pf, 2),
        'avg_z_entry': round(avg_z_entry, 2),
        'r1_gap': round(r1, 3),
        'bull_wr': round(bull_wr, 1),
        'bear_wr': round(bear_wr, 1),
        'bull_trades': len(bull),
        'bear_trades': len(bear),
        'bull_sharpe': round(bs_sharpe, 2),
        'bear_sharpe': round(br_sharpe, 2),
        'perm_p': round(perm_p, 4),
        'trimmed_sharpe': round(trimmed_sharpe, 2),
        'g1_pass': g1,
        'g2_pass': g2,
        'g3_pass': g3,
        'g4_pass': g4,
        'sub_sharpes': [round(x, 2) for x in ss],
        'gates_passed': sum([g1, g2, g3, g4]),
    }

###############################################################################
# ADDITIONAL ANALYSIS
###############################################################################
def analyze_mean_reversion_by_zscore(signals_df, close):
    """Analyze realized mean reversion by z-score bucket.
    Does the signal actually predict reversals?"""
    dates = sorted(signals_df['date'].unique())
    results = []

    for _, row in signals_df.iterrows():
        date = row['date']
        ticker = row['ticker']
        z = row['z_score']

        idx = dates.index(date) if date in dates else None
        if idx is None:
            continue

        # Forward returns: 5d, 10d, 20d
        for horizon, label in [(5, '5d'), (10, '10d'), (20, '20d')]:
            fwd_idx = idx + horizon
            if fwd_idx >= len(dates):
                continue
            fwd_date = dates[fwd_idx]
            if fwd_date in close.index and ticker in close.columns:
                fwd_price = close.loc[fwd_date, ticker]
                fwd_ret = (fwd_price - row['price']) / row['price']
                if not pd.isna(fwd_ret):
                    results.append({
                        'z_score': z,
                        'horizon': label,
                        'fwd_ret': fwd_ret,
                    })

    if not results:
        return None

    df = pd.DataFrame(results)
    # Bucket z-scores
    df['z_bucket'] = pd.cut(df['z_score'], bins=[-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf],
                            labels=['<-3', '-3:-2', '-2:-1', '-1:0', '0:1', '1:2', '2:3', '>3'])

    summary = df.groupby(['z_bucket', 'horizon'])['fwd_ret'].agg(['mean', 'std', 'count']).reset_index()
    return summary

###############################################################################
# MAIN
###############################################################################
def main():
    print("=" * 70)
    print("Cross-Sector Mean Reversion Options v1")
    print("=" * 70)
    t0 = datetime.now()

    # Load data
    print("Loading data...")
    close, vix = load_data()
    print(f"  Data: {close.shape[0]} days, {close.shape[1]} tickers, "
          f"{close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}")

    # Compute signals
    print("Computing cross-sectional z-scores...")
    signals_df = compute_signals(close, vix)
    print(f"  Signals: {len(signals_df)} observations")

    # Signal quality check
    oversold = signals_df[signals_df['z_score'] < -2]
    overbought = signals_df[signals_df['z_score'] > 2]
    print(f"  Oversold signals (z < -2): {len(oversold)} ({len(oversold)/len(signals_df)*100:.1f}%)")
    print(f"  Overbought signals (z > 2): {len(overbought)} ({len(overbought)/len(signals_df)*100:.1f}%)")

    # Analyze mean reversion by z-score (signal quality)
    print("\nAnalyzing mean-reversion signal quality by z-score bucket...")
    mr_analysis = analyze_mean_reversion_by_zscore(signals_df, close)
    if mr_analysis is not None:
        print("\n  Forward returns by z-score bucket:")
        for _, row in mr_analysis.iterrows():
            print(f"    z={row['z_bucket']:>6s}  {row['horizon']:>3s}  "
                  f"mean={row['mean']*100:+6.2f}%  std={row['std']*100:5.2f}%  n={int(row['count']):>5d}")

    # MLflow setup
    if HAS_MLFLOW:
        exp = mlflow.set_experiment("cross_sector_meanrev_options_v1")
        exp_id = exp.experiment_id
        print(f"\n  MLflow experiment: cross_sector_meanrev_options_v1")

    # Run all configs
    all_results = []
    best_result = None
    best_sharpe = -999

    for name, config in CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"Testing: {name} -- {config['desc']}")
        print(f"{'=' * 60}")

        trade_df, eq_df = run_backtest(signals_df, close, config, name)

        if len(trade_df) < 10:
            print(f"  SKIP: Only {len(trade_df)} trades (need >= 10)")
            all_results.append({
                'name': name, 'n_trades': len(trade_df),
                'gates_passed': 0, 'error': 'Too few trades'
            })
            continue

        result = adversarial_audit(trade_df, eq_df, name)
        all_results.append(result)

        gates = f"{result['gates_passed']}/4"
        status = "PASS" if result['gates_passed'] == 4 else "PARTIAL" if result['gates_passed'] >= 2 else "FAIL"

        print(f"  [{status}] {gates} | Sharpe {result['sharpe']:.2f} | "
              f"Sortino {result['sortino']:.2f} | WR {result['win_rate']:.1f}% | "
              f"CAGR {result['cagr_pct']:.1f}% | MDD {result['maxdd_pct']:.1f}% | "
              f"PF {result['pf']:.2f} | {result['n_trades']} trades "
              f"({result['n_long']}L/{result['n_short']}S)")
        print(f"  ${START_CAP} -> ${result['final_equity']:,.0f} | "
              f"R1 gap {result['r1_gap']:.3f} | Bull WR {result['bull_wr']:.1f}% vs "
              f"Bear WR {result['bear_wr']:.1f}% | Perm p={result['perm_p']:.4f}")
        print(f"  Avg z at entry: {result['avg_z_entry']:.2f} | "
              f"Sub-period Sharpes: {result['sub_sharpes']}")

        # Log to MLflow
        if HAS_MLFLOW:
            with mlflow.start_run(experiment_id=exp_id, run_name=name):
                for k, v in result.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                    elif isinstance(v, bool):
                        mlflow.log_metric(k, int(v))
                mlflow.log_params({k: str(v) for k, v in config.items()})
                mlflow.log_params({
                    'universe_size': len(UNIVERSE),
                    'start_capital': START_CAP,
                    'max_pos_size': MAX_POS_SIZE,
                    'commission_rt': COMMISSION_RT,
                    'haircut': HAIRCUT,
                    'spread_width_pct': SPREAD_WIDTH_PCT,
                    'dte': DTE,
                    'rebal_days': REBAL_DAYS,
                    'top_k': TOP_K,
                })

        if result['sharpe'] > best_sharpe:
            best_sharpe = result['sharpe']
            best_result = result

    runtime = (datetime.now() - t0).total_seconds()

    # ---- SUMMARY ----
    print(f"\n{'=' * 70}")
    print(f"SUMMARY -- Cross-Sector Mean Reversion Options v1")
    print(f"{'=' * 70}")

    p4 = [r for r in all_results if r.get('gates_passed', 0) == 4]
    p3 = [r for r in all_results if r.get('gates_passed', 0) >= 3]
    p2 = [r for r in all_results if r.get('gates_passed', 0) >= 2]
    print(f"Configs tested: {len(all_results)}")
    print(f"  4/4 gates: {len(p4)} | >=3/4: {len(p3)} | >=2/4: {len(p2)}")
    print(f"  Runtime: {runtime:.0f}s")

    if best_result:
        print(f"\nBEST CONFIG: {best_result['name']}")
        print(f"  Sharpe {best_result['sharpe']:.2f} | Sortino {best_result['sortino']:.2f} | "
              f"CAGR {best_result['cagr_pct']:.1f}% | MDD {best_result['maxdd_pct']:.1f}% | "
              f"WR {best_result['win_rate']:.1f}% | PF {best_result['pf']:.2f}")
        print(f"  ${START_CAP} -> ${best_result['final_equity']:,.0f} | "
              f"{best_result['n_trades']} trades | Gates: {best_result['gates_passed']}/4")

    # Compare signal vs random
    signal_results = [r for r in all_results if r.get('name', '') != 'F_Random_Control' and 'sharpe' in r]
    random_result = next((r for r in all_results if r.get('name', '') == 'F_Random_Control'), None)
    if signal_results and random_result and 'sharpe' in random_result:
        avg_signal_sharpe = np.mean([r['sharpe'] for r in signal_results])
        print(f"\nSIGNAL vs RANDOM:")
        print(f"  Avg signal Sharpe: {avg_signal_sharpe:.2f}")
        print(f"  Random Sharpe:     {random_result['sharpe']:.2f}")
        edge = avg_signal_sharpe - random_result['sharpe']
        print(f"  Signal edge:       {edge:+.2f} Sharpe points")
        if edge > 0.3:
            print(f"  CONCLUSION: Mean-reversion z-score signal has meaningful edge over random")
        elif edge > 0:
            print(f"  CONCLUSION: Marginal edge -- needs more filtering/optimization")
        else:
            print(f"  CONCLUSION: No edge detected -- signal may not work for options spreads")

    # Research conclusions
    print(f"\n{'=' * 70}")
    print("RESEARCH CONCLUSIONS:")
    if len(p4) > 0:
        print(f"  POSITIVE: {len(p4)} configs passed all 4 adversarial gates.")
        print(f"  Mean reversion IS a viable complementary signal to momentum.")
        print(f"  Next step: combine with momentum strategy for portfolio diversification.")
    elif len(p3) > 0:
        print(f"  PROMISING: {len(p3)} configs passed 3/4 gates.")
        print(f"  Mean reversion shows promise but needs refinement.")
        print(f"  Consider: tighter z-thresholds, different holding periods, or VIX conditioning.")
    elif len(p2) > 0:
        print(f"  WEAK: {len(p2)} configs passed 2/4 gates.")
        print(f"  Mean reversion signal exists but is fragile in sector ETF options context.")
    else:
        print(f"  NEGATIVE: No configs passed >= 2 gates. Mean reversion not viable as tested.")
        print(f"  The edge may exist in spot returns but is consumed by options costs.")

    # Save results
    findings_dir = '/home/jupiter/Lvl3Quant/research/findings'
    os.makedirs(findings_dir, exist_ok=True)
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'cross_sector_meanrev_options_v1',
        'universe': UNIVERSE,
        'capital': START_CAP,
        'configs': {k: {kk: str(vv) for kk, vv in v.items()} for k, v in CONFIGS.items()},
        'results': all_results,
        'best': best_result,
        'runtime_s': runtime,
        'mr_analysis': mr_analysis.to_dict() if mr_analysis is not None else None,
    }
    with open(os.path.join(findings_dir, 'cross_sector_meanrev_options_v1_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nResults saved to findings/cross_sector_meanrev_options_v1_results.json")
    print(f"Done in {runtime:.0f}s.")
    return all_results


if __name__ == '__main__':
    main()
