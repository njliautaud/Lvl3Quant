#!/usr/bin/env python3
"""
LEAPS Momentum Growth v1 — Leveraged Growth via Long-Dated Options
===================================================================

THESIS: Our LightGBM ETF ranker (validated Sharpe 0.84-4.63) picks winning sectors.
Instead of holding ETFs directly, buy 6-12 month LEAPS calls for ~3-5x leverage
with defined risk (can't lose more than premium paid).

This gives GROWTH exposure (HC #696) using OPTIONS (HC #749) built on our
validated MOMENTUM SIGNAL (sector ETF ranker v2).

Key design:
- Monthly rebalance using LightGBM ranking (same features as sector_etf_momentum_v2)
- Buy ATM or slightly ITM LEAPS calls (0.70 delta) for best leverage ratio
- Position size: equal weight across top N picks, max loss = premium
- Roll when < 90 DTE remaining (avoid theta acceleration)
- Compare vs direct ETF holding baseline

Variants:
A: ATM calls, top 3, monthly
B: 70-delta (slightly ITM), top 3, monthly
C: ATM calls, top 5, monthly
D: 70-delta, top 3, bi-weekly
E: ATM calls, top 3, defensive shift (SMA200 filter)
F: 70-delta, top 3, VIX regime filter (no entry when VIX>25)
G: ATM + protective puts (bull call spread), top 3
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'leaps_momentum_growth_v1_results.json'

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
except:
    pass


# === Black-Scholes for LEAPS pricing ===

def bs_price(S, K, T, sigma, r=0.04, opt='call'):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if opt == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, sigma, r=0.04, opt='call'):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if (opt == 'call' and S > K) else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if opt == 'call':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


def strike_for_delta(S, T, sigma, target_delta=0.50, r=0.04):
    """Find call strike that gives approximately target_delta."""
    lo, hi = S * 0.7, S * 1.3
    for _ in range(50):
        mid = (lo + hi) / 2
        d = bs_delta(S, mid, T, sigma, r, 'call')
        if d > target_delta:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2)


# === LightGBM Ranking (same as validated sector_etf_momentum_v2) ===

def build_features(close_df, lookback_end):
    """Build momentum + risk features for LightGBM ranking."""
    features = {}
    for col in close_df.columns:
        px = close_df[col].loc[:lookback_end].dropna()
        if len(px) < 252:
            continue

        ret_1m = px.iloc[-1] / px.iloc[-21] - 1 if len(px) > 21 else 0
        ret_3m = px.iloc[-1] / px.iloc[-63] - 1 if len(px) > 63 else 0
        ret_6m = px.iloc[-1] / px.iloc[-126] - 1 if len(px) > 126 else 0
        ret_12m = px.iloc[-1] / px.iloc[-252] - 1 if len(px) > 252 else 0
        mom_12_1 = ret_12m - ret_1m  # Classic 12-1 momentum

        vol_20d = px.pct_change().iloc[-20:].std() * np.sqrt(252) if len(px) > 20 else 0.2
        vol_60d = px.pct_change().iloc[-60:].std() * np.sqrt(252) if len(px) > 60 else 0.2

        # Drawdown
        peak = px.iloc[-252:].expanding().max()
        dd = (px.iloc[-252:] - peak) / peak
        maxdd = dd.min()

        # Kurtosis
        rets = px.pct_change().iloc[-60:].dropna()
        kurt = rets.kurtosis() if len(rets) > 10 else 0

        # RSI
        delta = px.diff().iloc[-14:]
        gain = delta.clip(lower=0).mean()
        loss = -delta.clip(upper=0).mean()
        rsi = 100 - 100 / (1 + gain / (loss + 1e-10)) if loss > 0 else 50

        features[col] = {
            'ret_1m': ret_1m, 'ret_3m': ret_3m, 'ret_6m': ret_6m,
            'ret_12m': ret_12m, 'mom_12_1': mom_12_1,
            'vol_20d': vol_20d, 'vol_60d': vol_60d,
            'maxdd': maxdd, 'kurtosis': kurt, 'rsi': rsi,
        }

    return pd.DataFrame(features).T


# Global cache for precomputed features
_FEATURE_CACHE = {}

def precompute_all_features(close_df):
    """Precompute features for all month-ends at once. Called once, cached."""
    global _FEATURE_CACHE
    if _FEATURE_CACHE:
        return _FEATURE_CACHE

    fprint("  Precomputing features for all months...")
    month_ends = close_df.index.to_series().resample('M').last().dropna()

    feature_cols = ['ret_1m', 'ret_3m', 'ret_6m', 'ret_12m', 'mom_12_1',
                    'vol_20d', 'vol_60d', 'maxdd', 'kurtosis', 'rsi']

    # Precompute returns and features using vectorized operations
    pct = close_df.pct_change()

    for me_idx, me_date in enumerate(month_ends):
        if me_idx < 12:  # Need at least 12 months of history
            continue

        feats = {}
        for col in close_df.columns:
            px = close_df[col].loc[:me_date].dropna()
            if len(px) < 252:
                continue

            ret_1m = px.iloc[-1] / px.iloc[-21] - 1 if len(px) > 21 else 0
            ret_3m = px.iloc[-1] / px.iloc[-63] - 1 if len(px) > 63 else 0
            ret_6m = px.iloc[-1] / px.iloc[-126] - 1 if len(px) > 126 else 0
            ret_12m = px.iloc[-1] / px.iloc[-252] - 1 if len(px) > 252 else 0
            mom_12_1 = ret_12m - ret_1m

            vol_20d = px.pct_change().iloc[-20:].std() * np.sqrt(252) if len(px) > 20 else 0.2
            vol_60d = px.pct_change().iloc[-60:].std() * np.sqrt(252) if len(px) > 60 else 0.2

            peak = px.iloc[-252:].expanding().max()
            dd = (px.iloc[-252:] - peak) / peak
            maxdd = dd.min()

            rets = px.pct_change().iloc[-60:].dropna()
            kurt = rets.kurtosis() if len(rets) > 10 else 0

            delta = px.diff().iloc[-14:]
            gain = delta.clip(lower=0).mean()
            loss = -delta.clip(upper=0).mean()
            rsi = 100 - 100 / (1 + gain / (loss + 1e-10)) if loss > 0 else 50

            feats[col] = [ret_1m, ret_3m, ret_6m, ret_12m, mom_12_1,
                          vol_20d, vol_60d, maxdd, kurt, rsi]

        _FEATURE_CACHE[me_date] = feats

    # Precompute forward returns
    for i in range(len(month_ends) - 1):
        me = month_ends.iloc[i]
        next_me = month_ends.iloc[i + 1]
        if me not in _FEATURE_CACHE:
            continue
        fwd = {}
        for col in close_df.columns:
            px_curr = close_df[col].loc[:me].dropna()
            px_next = close_df[col].loc[:next_me].dropna()
            if len(px_curr) > 0 and len(px_next) > 0:
                fwd[col] = px_next.iloc[-1] / px_curr.iloc[-1] - 1
        _FEATURE_CACHE[me]['_fwd_ret'] = fwd

    fprint(f"  Precomputed {len(_FEATURE_CACHE)} month-ends")
    return _FEATURE_CACHE


def lgbm_rank(close_df, rebal_date, train_months=60):
    """Train LightGBM on historical momentum returns, rank ETFs. Uses precomputed cache."""
    try:
        import lightgbm as lgb
    except ImportError:
        return simple_rank(close_df, rebal_date)

    cache = precompute_all_features(close_df)

    feature_cols_n = 10  # Number of feature columns

    # Find closest month-end <= rebal_date
    month_ends = sorted([k for k in cache.keys() if k <= rebal_date and not isinstance(k, str)])
    if len(month_ends) < 24:
        return simple_rank(close_df, rebal_date)

    # Use last train_months months for training
    cutoff = max(0, len(month_ends) - train_months)
    train_months_list = month_ends[cutoff:-1]  # Exclude current month (need fwd ret)

    X_all, y_all = [], []
    for me in train_months_list:
        feats = cache.get(me, {})
        fwd = feats.get('_fwd_ret', {})
        for ticker in feats:
            if ticker == '_fwd_ret':
                continue
            if ticker not in fwd:
                continue
            row = feats[ticker]
            if any(np.isnan(v) for v in row):
                continue
            X_all.append(row)
            y_all.append(fwd[ticker])

    if len(X_all) < 50:
        return simple_rank(close_df, rebal_date)

    X_train = np.array(X_all)
    y_train = np.array(y_all)

    model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, verbose=-1, n_jobs=4
    )
    model.fit(X_train, y_train)

    # Current features (from closest cached month or compute fresh)
    current_me = month_ends[-1]
    current_feats = cache.get(current_me, {})

    scores = {}
    for ticker in current_feats:
        if ticker == '_fwd_ret':
            continue
        row = current_feats[ticker]
        if any(np.isnan(v) for v in row):
            continue
        scores[ticker] = model.predict(np.array([row]))[0]

    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def simple_rank(close_df, rebal_date):
    """Simple 12-1 momentum ranking as fallback."""
    scores = {}
    for col in close_df.columns:
        px = close_df[col].loc[:rebal_date].dropna()
        if len(px) > 252:
            mom = px.iloc[-1] / px.iloc[-252] - 1
            recent = px.iloc[-1] / px.iloc[-21] - 1
            scores[col] = mom - recent  # 12-1 momentum
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


# === LEAPS Simulation ===

def simulate_leaps(close_df, vix_close, spy_close,
                   top_n=3, target_delta=0.50, leaps_dte=365,
                   roll_dte=90, rebal_freq='M', defensive_shift=False,
                   vix_filter=None, spread_width=None,
                   capital=100000, name='base'):
    """
    Simulate LEAPS momentum strategy.

    Buy LEAPS calls on top-ranked ETFs. Monthly rotation.
    When ranking changes, sell old LEAPS and buy new ones.

    Args:
        target_delta: 0.50 = ATM, 0.70 = slightly ITM
        leaps_dte: days to expiration for new LEAPS (365 = 1 year)
        roll_dte: roll when remaining DTE falls below this
        rebal_freq: 'M' monthly, '2W' bi-weekly
        defensive_shift: if True, go to cash when SPY < SMA200
        vix_filter: if set, no new entries when VIX > this level
        spread_width: if set, buy bull call spread (defined risk) instead of naked calls
    """
    fprint(f"\n--- {name} ---")

    spy_sma200 = spy_close.rolling(200).mean()

    # Get rebalance dates
    all_dates = close_df.index[close_df.index >= close_df.index[252]]
    if rebal_freq == 'M':
        rebal_series = all_dates.to_series().resample('M').last().dropna()
    elif rebal_freq == '2W':
        rebal_series = all_dates.to_series().resample('2W').last().dropna()
    else:
        rebal_series = all_dates.to_series().resample('M').last().dropna()

    rebal_dates = rebal_series.values

    # Track positions and equity
    positions = {}  # ticker -> {strike, dte_remaining, entry_price, entry_date, n_contracts, spread_strike}
    equity_curve = []
    equity = capital
    trades = []

    for i, rdate in enumerate(rebal_dates):
        rdate = pd.Timestamp(rdate)
        if rdate not in close_df.index:
            continue

        # Current VIX and regime
        vix_val = float(vix_close.loc[rdate]) if rdate in vix_close.index else 15.0
        sigma = vix_val / 100
        spy_val = float(spy_close.loc[rdate]) if rdate in spy_close.index else 0
        sma_val = float(spy_sma200.loc[rdate]) if rdate in spy_sma200.index else spy_val
        in_bear = spy_val < sma_val
        regime = 'bear' if in_bear else 'bull'

        # === Close existing positions (mark to market) ===
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            if ticker not in close_df.columns:
                continue

            S_now = float(close_df[ticker].loc[rdate])
            days_held = (rdate - pos['entry_date']).days
            dte_remain = max(pos['initial_dte'] - days_held, 1)
            T_now = dte_remain / 365

            # Price the LEAPS now
            current_price = bs_price(S_now, pos['strike'], T_now, sigma)
            if spread_width and 'long_strike' in pos:
                long_price = bs_price(S_now, pos['long_strike'], T_now, sigma)
                current_price = current_price - long_price

            pos['current_price'] = current_price
            pos['dte_remaining'] = dte_remain

        # === Decide what to hold ===
        # Get LightGBM rankings
        rankings = lgbm_rank(close_df, rdate)
        top_picks = rankings[:top_n]

        # Defensive shift: go to cash in bear regime
        if defensive_shift and in_bear:
            top_picks = []

        # VIX filter: no new entries when VIX is elevated
        if vix_filter and vix_val > vix_filter:
            # Keep existing positions but don't open new ones
            top_picks = [t for t in top_picks if t in positions]

        # === Close positions not in top picks or need rolling ===
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            should_close = (ticker not in top_picks) or (pos['dte_remaining'] < roll_dte)

            if should_close:
                S_now = float(close_df[ticker].loc[rdate])
                days_held = (rdate - pos['entry_date']).days
                dte_remain = max(pos['initial_dte'] - days_held, 1)
                T_now = dte_remain / 365

                exit_price = bs_price(S_now, pos['strike'], T_now, sigma)
                if spread_width and 'long_strike' in pos:
                    long_exit = bs_price(S_now, pos['long_strike'], T_now, sigma)
                    exit_price = exit_price - long_exit

                pnl_per_contract = (exit_price - pos['entry_price']) * 100
                total_pnl = pnl_per_contract * pos['n_contracts'] - 2  # $2 commission
                equity += total_pnl

                trades.append({
                    'ticker': ticker, 'entry': str(pos['entry_date']),
                    'exit': str(rdate), 'entry_px': round(pos['entry_price'], 2),
                    'exit_px': round(exit_price, 2),
                    'pnl': round(total_pnl, 2),
                    'win': total_pnl > 0, 'regime': regime,
                    'underlying_entry': round(pos['underlying_entry'], 2),
                    'underlying_exit': round(S_now, 2),
                })

                del positions[ticker]

        # === Open new positions ===
        n_to_open = top_n - len(positions)
        tickers_to_open = [t for t in top_picks if t not in positions][:n_to_open]

        if tickers_to_open and equity > 0:
            # Equal weight allocation
            alloc_per_pos = equity / top_n

            for ticker in tickers_to_open:
                if ticker not in close_df.columns:
                    continue
                S = float(close_df[ticker].loc[rdate])
                if S <= 0 or np.isnan(S):
                    continue

                T = leaps_dte / 365

                # Find strike for target delta
                K = strike_for_delta(S, T, sigma, target_delta)

                # Price the LEAPS
                call_price = bs_price(S, K, T, sigma)

                if spread_width:
                    # Bull call spread: buy call at K, sell call at K + spread_width% of S
                    K_long = K + int(S * spread_width / 100)
                    long_price = bs_price(S, K_long, T, sigma)
                    net_price = call_price - long_price
                else:
                    net_price = call_price
                    K_long = None

                if net_price <= 0.5:  # Skip if too cheap (probably pricing error)
                    continue

                # Number of contracts (each = 100 shares)
                cost_per_contract = net_price * 100
                n_contracts = max(1, int(alloc_per_pos / cost_per_contract))
                n_contracts = min(n_contracts, 10)  # Cap

                actual_cost = cost_per_contract * n_contracts
                if actual_cost > equity * 0.4:  # Max 40% in one position
                    n_contracts = max(1, int(equity * 0.4 / cost_per_contract))

                equity -= 2  # Commission to open

                positions[ticker] = {
                    'strike': K,
                    'entry_price': net_price,
                    'entry_date': rdate,
                    'n_contracts': n_contracts,
                    'initial_dte': leaps_dte,
                    'dte_remaining': leaps_dte,
                    'underlying_entry': S,
                }
                if K_long:
                    positions[ticker]['long_strike'] = K_long

        # Track equity (unrealized)
        unrealized = 0
        for ticker, pos in positions.items():
            if ticker in close_df.columns and rdate in close_df[ticker].index:
                S_now = float(close_df[ticker].loc[rdate])
                days_held = (rdate - pos['entry_date']).days
                dte_remain = max(pos['initial_dte'] - days_held, 1)
                T_now = dte_remain / 365

                current = bs_price(S_now, pos['strike'], T_now, sigma)
                if spread_width and 'long_strike' in pos:
                    current -= bs_price(S_now, pos['long_strike'], T_now, sigma)

                unrealized += (current - pos['entry_price']) * 100 * pos['n_contracts']

        total_equity = equity + unrealized
        equity_curve.append({'date': str(rdate), 'equity': round(total_equity, 2), 'regime': regime})

    # Close remaining positions at end
    last_date = pd.Timestamp(rebal_dates[-1])
    for ticker in list(positions.keys()):
        pos = positions[ticker]
        if ticker in close_df.columns and last_date in close_df[ticker].index:
            S_now = float(close_df[ticker].loc[last_date])
            days_held = (last_date - pos['entry_date']).days
            dte_remain = max(pos['initial_dte'] - days_held, 1)
            T_now = dte_remain / 365
            sigma_final = float(vix_close.iloc[-1]) / 100 if len(vix_close) > 0 else 0.15

            exit_price = bs_price(S_now, pos['strike'], T_now, sigma_final)
            if spread_width and 'long_strike' in pos:
                exit_price -= bs_price(S_now, pos['long_strike'], T_now, sigma_final)

            pnl = (exit_price - pos['entry_price']) * 100 * pos['n_contracts'] - 2
            equity += pnl
            trades.append({
                'ticker': ticker, 'entry': str(pos['entry_date']),
                'exit': str(last_date), 'pnl': round(pnl, 2),
                'win': pnl > 0, 'regime': 'bull',
            })

    if not trades:
        fprint("  No trades!")
        return None

    # === Compute metrics ===
    n_trades = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n_trades * 100

    total_pnl = sum(t['pnl'] for t in trades)
    avg_win = np.mean([t['pnl'] for t in trades if t['win']]) if wins > 0 else 0
    avg_loss = np.mean([t['pnl'] for t in trades if not t['win']]) if wins < n_trades else 0

    # Monthly returns from equity curve
    eq_df = pd.DataFrame(equity_curve)
    eq_df['date'] = pd.to_datetime(eq_df['date'])
    eq_df = eq_df.set_index('date')
    monthly_eq = eq_df['equity'].resample('M').last().dropna()
    monthly_ret = monthly_eq.pct_change().dropna()

    n_years = len(monthly_ret) / 12

    sharpe = (monthly_ret.mean() * 12) / (monthly_ret.std() * np.sqrt(12) + 1e-10) if len(monthly_ret) > 3 else 0

    final_eq = equity_curve[-1]['equity'] if equity_curve else capital
    cagr = (final_eq / capital) ** (1 / max(n_years, 0.5)) - 1

    down = monthly_ret[monthly_ret < 0]
    sortino = (monthly_ret.mean() * 12) / (down.std() * np.sqrt(12) + 1e-10) if len(down) > 0 else 999

    # Max drawdown from equity curve
    eq_vals = np.array([e['equity'] for e in equity_curve])
    peak = np.maximum.accumulate(eq_vals)
    dd = (eq_vals - peak) / (peak + 1e-10)
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd < 0 else 999

    gross_wins = sum(t['pnl'] for t in trades if t['win'])
    gross_losses = abs(sum(t['pnl'] for t in trades if not t['win']))
    pf = gross_wins / (gross_losses + 1e-10)

    # Regime analysis
    bull_t = [t for t in trades if t.get('regime') == 'bull']
    bear_t = [t for t in trades if t.get('regime') == 'bear']

    bull_sharpe = 0
    bear_sharpe = 0
    if bull_t:
        bull_rets = [t['pnl'] / capital for t in bull_t]
        bull_sharpe = np.mean(bull_rets) / (np.std(bull_rets) + 1e-10) * np.sqrt(12)
    if bear_t:
        bear_rets = [t['pnl'] / capital for t in bear_t]
        bear_sharpe = np.mean(bear_rets) / (np.std(bear_rets) + 1e-10) * np.sqrt(12)

    bull_wr = sum(1 for t in bull_t if t['win']) / max(len(bull_t), 1) * 100
    bear_wr = sum(1 for t in bear_t if t['win']) / max(len(bear_t), 1) * 100
    r1_gap = abs(bull_wr - bear_wr) / max(bull_wr, bear_wr, 1)

    # Leverage ratio (how much notional exposure per dollar of premium)
    avg_premium = np.mean([t.get('entry_px', 0) for t in trades if 'entry_px' in t])
    avg_underlying = np.mean([t.get('underlying_entry', 0) for t in trades if 'underlying_entry' in t])
    leverage = avg_underlying / (avg_premium + 1e-10) if avg_premium > 0 else 0

    result = {
        'name': name, 'n_trades': n_trades, 'win_rate': round(wr, 1),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
        'total_pnl': round(total_pnl, 2), 'final_equity': round(final_eq, 2),
        'cagr_pct': round(cagr * 100, 1), 'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2), 'maxdd_pct': round(maxdd * 100, 1),
        'calmar': round(calmar, 2), 'pf': round(pf, 2),
        'r1_gap': round(r1_gap, 3), 'r1_pass': r1_gap <= 0.50,
        'bull_wr': round(bull_wr, 1), 'bear_wr': round(bear_wr, 1),
        'bull_trades': len(bull_t), 'bear_trades': len(bear_t),
        'bull_sharpe': round(bull_sharpe, 2), 'bear_sharpe': round(bear_sharpe, 2),
        'leverage_ratio': round(leverage, 1),
        'monthly_returns': monthly_ret.tolist(),
    }

    fprint(f"  {name}: {n_trades} trades, WR {wr:.1f}%, Sharpe {sharpe:.2f}, "
           f"CAGR {cagr*100:.1f}%, MaxDD {maxdd*100:.1f}%, PF {pf:.2f}, "
           f"Sortino {sortino:.2f}, Calmar {calmar:.2f}")
    fprint(f"    ${capital:,} → ${final_eq:,.0f} | Leverage ~{leverage:.1f}x")
    fprint(f"    Bull: WR {bull_wr:.0f}% ({len(bull_t)} trades) | Bear: WR {bear_wr:.0f}% ({len(bear_t)} trades)")
    fprint(f"    R1 gap: {r1_gap:.3f} {'PASS' if r1_gap <= 0.50 else 'FAIL'}")

    return result


def permutation_test(returns, n_perms=1000):
    """Direction-shuffle permutation test."""
    if len(returns) < 5:
        return 1.0
    real = np.mean(returns) / (np.std(returns) + 1e-10)
    count = 0
    for _ in range(n_perms):
        shuffled = returns * np.random.choice([-1, 1], len(returns))
        fake = np.mean(shuffled) / (np.std(shuffled) + 1e-10)
        if fake >= real:
            count += 1
    return count / n_perms


def main():
    import yfinance as yf

    fprint(f"LEAPS Momentum Growth v1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)

    # Same ETF universe as sector_etf_momentum_v2
    tickers = [
        'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE',
        'XLC', 'SMH', 'XBI', 'XHB', 'XRT',  # Sector ETFs
        'SPY', 'QQQ', 'IWM',  # Broad market
        'GLD', 'TLT', 'EEM', 'EFA',  # Diversifiers
    ]

    fprint("Downloading data...")
    raw = yf.download(tickers + ['^VIX'], start='2008-01-01', end='2026-07-25', progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw

    # Separate VIX
    vix_col = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vix_col].dropna()

    spy = close['SPY'].dropna()

    # ETF close (no VIX)
    etf_close = close[[c for c in tickers if c in close.columns]].dropna(how='all')

    # Align
    common = etf_close.index.intersection(vix.index).intersection(spy.index)
    etf_close = etf_close.loc[common]
    vix = vix.loc[common]
    spy = spy.loc[common]

    fprint(f"Data: {len(etf_close)} days, {len(etf_close.columns)} ETFs")
    fprint(f"Range: {etf_close.index[0].strftime('%Y-%m-%d')} to {etf_close.index[-1].strftime('%Y-%m-%d')}")

    # === Run variants ===
    variants = [
        # (name, top_n, target_delta, leaps_dte, rebal_freq, defensive_shift, vix_filter, spread_width)
        ('A_ATM_Top3_Monthly', 3, 0.50, 365, 'M', False, None, None),
        ('B_70d_Top3_Monthly', 3, 0.70, 365, 'M', False, None, None),
        ('C_ATM_Top5_Monthly', 5, 0.50, 365, 'M', False, None, None),
        ('D_70d_Top3_Biweekly', 3, 0.70, 365, '2W', False, None, None),
        ('E_ATM_Top3_DefShift', 3, 0.50, 365, 'M', True, None, None),
        ('F_70d_Top3_VIXfilt', 3, 0.70, 365, 'M', False, 25, None),
        ('G_Spread10_Top3', 3, 0.50, 365, 'M', False, None, 10),  # 10% OTM spread
    ]

    results = []

    if MLFLOW_OK:
        exp_name = 'leaps_momentum_growth_v1'
        try:
            if not mlflow.get_experiment_by_name(exp_name):
                mlflow.create_experiment(exp_name)
        except:
            pass
        mlflow.set_experiment(exp_name)

    for vname, top_n, delta, dte, freq, def_shift, vix_filt, spread_w in variants:
        try:
            r = simulate_leaps(
                etf_close, vix, spy,
                top_n=top_n, target_delta=delta, leaps_dte=dte,
                rebal_freq=freq, defensive_shift=def_shift,
                vix_filter=vix_filt, spread_width=spread_w,
                capital=100000, name=vname
            )
            if r:
                if MLFLOW_OK:
                    with mlflow.start_run(run_name=vname):
                        mlflow.log_params({
                            'top_n': top_n, 'target_delta': delta,
                            'leaps_dte': dte, 'rebal_freq': freq,
                            'defensive_shift': def_shift, 'vix_filter': vix_filt or 'none',
                            'spread_width': spread_w or 'none',
                        })
                        mlflow.log_metrics({k: v for k, v in r.items()
                                           if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v)})
                results.append(r)
        except Exception as e:
            fprint(f"  ERROR {vname}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        fprint("No results!")
        return

    # === Adversarial Validation ===
    fprint("\n" + "=" * 70)
    fprint("ADVERSARIAL VALIDATION")
    fprint("=" * 70)

    for r in results:
        rets = np.array(r['monthly_returns'])
        if len(rets) < 5:
            r['perm_p'] = 1.0
            r['g1_pass'] = False
            r['g2_pass'] = r['r1_pass']
            r['g3_pass'] = False
            r['g4_pass'] = False
            r['gates_passed'] = 0
            continue

        # G1: Permutation test
        r['perm_p'] = round(permutation_test(rets, 1000), 3)
        r['g1_pass'] = r['perm_p'] < 0.05

        # G2: Regime-agnostic (R1)
        r['g2_pass'] = r['r1_pass']

        # G3: Sub-period stability
        n = len(rets)
        chunk = max(n // 3, 1)
        subs = []
        for j in range(3):
            sub = rets[j * chunk:(j + 1) * chunk]
            if len(sub) > 1:
                subs.append(np.mean(sub) * 12 / (np.std(sub) * np.sqrt(12) + 1e-10))
            else:
                subs.append(0)
        r['g3_pass'] = all(s > 0 for s in subs)
        r['sub_sharpes'] = [round(s, 2) for s in subs]

        # G4: Outlier removal
        if n > 5:
            nt = max(1, int(n * 0.05))
            tr = np.sort(rets)[nt:-nt] if nt < n // 2 else rets
            trimmed_sharpe = np.mean(tr) / (np.std(tr) + 1e-10)
            orig_sharpe = np.mean(rets) / (np.std(rets) + 1e-10)
            r['g4_pass'] = trimmed_sharpe > 0 and trimmed_sharpe / (orig_sharpe + 1e-10) > 0.5
        else:
            r['g4_pass'] = False

        r['gates_passed'] = sum([r['g1_pass'], r['g2_pass'], r['g3_pass'], r['g4_pass']])

        fprint(f"\n{r['name']}: "
               f"G1={'PASS' if r['g1_pass'] else 'FAIL'}(p={r['perm_p']}), "
               f"G2={'PASS' if r['g2_pass'] else 'FAIL'}(gap={r['r1_gap']}), "
               f"G3={'PASS' if r['g3_pass'] else 'FAIL'}(subs={r['sub_sharpes']}), "
               f"G4={'PASS' if r['g4_pass'] else 'FAIL'} "
               f"→ {r['gates_passed']}/4")

    # === Summary ===
    fprint("\n" + "=" * 70)
    fprint("SUMMARY — LEAPS Momentum Growth v1")
    fprint("=" * 70)
    fprint(f"{'Name':<28} {'Trades':>6} {'WR':>6} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} "
           f"{'Sortino':>8} {'PF':>5} {'Lev':>4} {'Gates':>6}")
    fprint("-" * 95)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<28} {r['n_trades']:>6} {r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} "
               f"{r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% {r['sortino']:>8.2f} "
               f"{r['pf']:>5.2f} {r['leverage_ratio']:>3.0f}x {r['gates_passed']:>4}/4")

    # === Compare to direct ETF holding ===
    fprint("\n--- BASELINE: Direct ETF Holding (no options) ---")
    best = max(results, key=lambda x: x['sharpe'])
    fprint(f"Best LEAPS variant: {best['name']} — Sharpe {best['sharpe']}, CAGR {best['cagr_pct']}%, MaxDD {best['maxdd_pct']}%")
    fprint(f"Compare to Sector ETF Momentum v2 best: Sharpe 4.63, CAGR 71.7% (but with B-S pricing)")
    fprint(f"LEAPS leverage ratio: ~{best['leverage_ratio']:.0f}x (pay premium, get notional exposure)")

    # Save
    save = [{k: v for k, v in r.items() if k != 'monthly_returns'} for r in results]
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    fprint(f"\nResults saved. Done — {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
