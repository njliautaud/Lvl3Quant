#!/usr/bin/env python3
"""
Covered Call Overlay on Sector ETF Momentum — v1
=================================================

Combines the validated LightGBM sector ETF momentum strategy (Sharpe ~4.63)
with a covered call overlay to generate additional income.

Approach:
  1. Same LightGBM momentum ranker as v2 (22 ETFs, 504d train, 21d slide, top 3)
     with regime-adaptive defensive shift.
  2. After selecting top ETFs each month, simulate selling covered calls at
     various delta levels.
  3. Premium estimated via Black-Scholes with 63d realized vol.

Variants:
  A) No calls (pure momentum baseline)
  B) 30-delta monthly covered calls
  C) 20-delta monthly covered calls (further OTM)
  D) ATM covered calls (max premium, caps upside)
  E) Regime-adaptive: sell calls in bull only, skip in bear
  F) IV-rank gated: only sell calls when IV rank > 50%

Capital: $100,000. Period: 2019-2026. Costs: 20bps ETF + $0.65/contract.
Walk-forward: 504d train, 21d test, sliding window.

Author: Claude Opus 4.6
Date: 2026-07-25
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

# ─── Helpers ─────────────────────────────────────────────────────────────────

def fprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'covered_call_overlay_v1_results.json'

# MLflow
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
except:
    pass

# Universe (same as v2)
UNIVERSE = [
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE',
    'XLC', 'QQQ', 'IWM', 'MDY', 'EFA', 'EEM', 'GLD', 'TLT', 'HYG', 'IYR',
    'VNQ', 'DBC',
]

DEFENSIVE_ETFS = {'XLP', 'XLU', 'TLT', 'IEF', 'GLD'}
RISK_ON_ETFS = {'XLK', 'QQQ', 'XLY', 'XBI', 'IWM', 'EEM'}

STARTING_CAPITAL = 100_000
OPTION_COST_PER_CONTRACT = 0.65  # dollars per contract
ETF_COST_BPS = 20  # basis points round-trip
RISK_FREE_RATE = 0.04  # approximate, used for BS pricing


# ─── Black-Scholes helpers ───────────────────────────────────────────────────

def bs_d1(S, K, T, r, sigma):
    """d1 in Black-Scholes formula."""
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_call_delta(S, K, T, r, sigma):
    """Black-Scholes call delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.cdf(d1)

def strike_for_delta(S, T, r, sigma, target_delta):
    """Find strike that gives a specific call delta using bisection."""
    if T <= 0 or sigma <= 0:
        return S * 1.05  # fallback OTM
    K_lo, K_hi = S * 0.5, S * 3.0
    for _ in range(100):
        K_mid = (K_lo + K_hi) / 2
        d = bs_call_delta(S, K_mid, T, r, sigma)
        if d > target_delta:
            K_lo = K_mid
        else:
            K_hi = K_mid
        if abs(d - target_delta) < 1e-6:
            break
    return K_mid


# ─── Feature builder (identical to v2) ───────────────────────────────────────

def build_features(close, volume, spy_close=None):
    """Build momentum + quality + regime features for a single ETF."""
    lr = np.log(close / close.shift(1))

    feat = pd.DataFrame({
        # Momentum features
        'ret_5d': close.pct_change(5),
        'ret_10d': close.pct_change(10),
        'ret_21d': close.pct_change(21),
        'ret_63d': close.pct_change(63),
        'ret_126d': close.pct_change(126),
        'ret_252d': close.pct_change(252),
        'mom_12_1': close.pct_change(252) - close.pct_change(21),
        'high_52w_pct': close / close.rolling(252).max(),
        'mom_accel': close.pct_change(63) - close.pct_change(63).shift(63),

        # Volatility/quality features
        'vol_20d': lr.rolling(20).std() * np.sqrt(252),
        'vol_60d': lr.rolling(60).std() * np.sqrt(252),
        'vol_ratio': (lr.rolling(20).std() / lr.rolling(60).std()),
        'sharpe_63d': lr.rolling(63).mean() / lr.rolling(63).std(),
        'sharpe_126d': lr.rolling(126).mean() / lr.rolling(126).std(),
        'maxdd_63d': (close / close.rolling(63).max() - 1).rolling(63).min(),

        # Volume features
        'vol_rel': volume / volume.rolling(20).mean() if volume is not None else 0,

        # Higher moments
        'skew_63d': lr.rolling(63).skew(),
        'kurt_63d': lr.rolling(63).kurt(),

        # Trend strength
        'vol_trend': (lr.rolling(20).std() - lr.rolling(60).std()) / lr.rolling(60).std(),
    }, index=close.index)

    # Regime features
    feat['above_sma50'] = (close > close.rolling(50).mean()).astype(int)
    feat['above_sma200'] = (close > close.rolling(200).mean()).astype(int)
    feat['dist_sma200'] = (close - close.rolling(200).mean()) / close.rolling(200).mean()
    feat['rv_21d'] = lr.rolling(21).std() * np.sqrt(252)
    feat['rv_ratio_short_long'] = feat['rv_21d'] / (feat['vol_60d'] + 1e-8)

    # Cross-asset regime features (from SPY)
    if spy_close is not None:
        spy_lr = np.log(spy_close / spy_close.shift(1))
        feat['spy_ret_21d'] = spy_close.pct_change(21)
        feat['spy_ret_63d'] = spy_close.pct_change(63)
        feat['spy_above_sma200'] = (spy_close > spy_close.rolling(200).mean()).astype(int)
        feat['spy_rv_21d'] = spy_lr.rolling(21).std() * np.sqrt(252)
        feat['spy_dist_sma200'] = (spy_close - spy_close.rolling(200).mean()) / spy_close.rolling(200).mean()
        feat['corr_spy_63d'] = lr.rolling(63).corr(spy_lr)
        feat['beta_spy_63d'] = lr.rolling(63).cov(spy_lr) / (spy_lr.rolling(63).var() + 1e-10)
    else:
        for c in ['spy_ret_21d', 'spy_ret_63d', 'spy_above_sma200', 'spy_rv_21d',
                   'spy_dist_sma200', 'corr_spy_63d', 'beta_spy_63d']:
            feat[c] = 0

    return feat


def detect_regime(spy_close, date):
    """Detect bull/bear regime from SPY at given date."""
    if spy_close is None:
        return 'bull'
    loc = spy_close.index.searchsorted(date)
    if loc < 200:
        return 'bull'
    sma200 = spy_close.iloc[max(0, loc-200):loc].mean()
    return 'bear' if spy_close.iloc[loc-1] < sma200 else 'bull'


# ─── IV rank helper ──────────────────────────────────────────────────────────

def compute_iv_rank(lr_series, date, lookback_252=252, window_63=63):
    """
    Compute IV rank: percentile of current 63d realized vol within past 252d range.
    Returns 0-100. Above 50 means vol is elevated relative to the past year.
    """
    loc = lr_series.index.searchsorted(date)
    if loc < lookback_252:
        return 50.0  # default to neutral
    segment = lr_series.iloc[max(0, loc - lookback_252):loc]
    rolling_vols = segment.rolling(window_63).std() * np.sqrt(252)
    rolling_vols = rolling_vols.dropna()
    if len(rolling_vols) < 10:
        return 50.0
    current_vol = rolling_vols.iloc[-1]
    rank = (rolling_vols < current_vol).sum() / len(rolling_vols) * 100
    return float(rank)


# ─── Covered call return logic ───────────────────────────────────────────────

def compute_covered_call_return(
    entry_price, exit_price, sigma, delta_target, T=21/252,
    r=RISK_FREE_RATE, shares=100, etf_cost_bps=ETF_COST_BPS,
    option_cost=OPTION_COST_PER_CONTRACT, sell_call=True
):
    """
    Compute the return from holding ETF shares + selling a covered call.

    Parameters:
        entry_price: ETF price at entry
        exit_price: ETF price at expiry
        sigma: annualized realized vol (used as IV proxy)
        delta_target: target delta for the call (0.5 for ATM, 0.3 for 30-delta, etc.)
        T: time to expiration in years (21/252 for monthly)
        r: risk-free rate
        shares: number of shares per lot (100 for standard contract)
        etf_cost_bps: round-trip ETF transaction cost in bps
        option_cost: cost per option contract
        sell_call: whether to actually sell the call (False = baseline)

    Returns:
        dict with return components and total return
    """
    # ETF portion
    etf_return_pct = (exit_price - entry_price) / entry_price
    etf_cost_pct = etf_cost_bps / 10000 * 2  # buy + sell

    if not sell_call:
        # Pure momentum, no call overlay
        total_return = etf_return_pct - etf_cost_pct
        return {
            'total_return': float(total_return),
            'etf_return': float(etf_return_pct),
            'premium_pct': 0.0,
            'assignment_cost_pct': 0.0,
            'option_cost_pct': 0.0,
            'capped': False,
        }

    # Find strike for target delta
    strike = strike_for_delta(entry_price, T, r, sigma, delta_target)

    # Premium received (per share)
    premium_per_share = bs_call_price(entry_price, strike, T, r, sigma)
    premium_pct = premium_per_share / entry_price

    # Option transaction cost (per share basis for 1 contract = 100 shares)
    option_cost_pct = option_cost / (entry_price * shares)

    # At expiry: check assignment
    if exit_price > strike:
        # Assigned: gain capped at (strike - entry) + premium
        capped_return = (strike - entry_price) / entry_price + premium_pct
        total_return = capped_return - etf_cost_pct - option_cost_pct
        capped = True
    else:
        # Not assigned: keep full ETF move + premium
        total_return = etf_return_pct + premium_pct - etf_cost_pct - option_cost_pct
        capped = False

    return {
        'total_return': float(total_return),
        'etf_return': float(etf_return_pct),
        'premium_pct': float(premium_pct),
        'assignment_cost_pct': float(option_cost_pct),
        'option_cost_pct': float(option_cost_pct),
        'strike': float(strike),
        'capped': capped,
    }


# ─── Feature matrix builder (called once, reused across variants) ────────────

def build_feature_matrix(all_data, spy_close):
    """Build the feature matrix once for all variants."""
    features_list = []
    labels_list = []
    meta_list = []

    common = None
    for t, df in all_data.items():
        common = df.index if common is None else common.intersection(df.index)

    for t, df in all_data.items():
        c = df['Close'].reindex(common)
        v = df['Volume'].reindex(common) if 'Volume' in df.columns else None
        sc = spy_close.reindex(common)
        feat = build_features(c, v, spy_close=sc)

        fwd = c.pct_change(21).shift(-21)
        valid = feat.dropna().index.intersection(fwd.dropna().index)

        for d in valid:
            row = feat.loc[d].values
            if not np.any(np.isnan(row)) and not np.any(np.isinf(row)):
                features_list.append(row)
                labels_list.append(fwd.loc[d])
                meta_list.append({'date': d, 'ticker': t})

    X = np.array(features_list)
    y = np.array(labels_list)
    meta = pd.DataFrame(meta_list)
    dates = sorted(meta['date'].unique())

    # Pre-compute date-to-index mapping for fast mask creation
    date_to_idx = {d: i for i, d in enumerate(dates)}
    meta_date_idx = meta['date'].map(date_to_idx).values

    fprint(f"  Feature matrix: {X.shape[0]} x {X.shape[1]}, "
           f"dates: {dates[0].date()} to {dates[-1].date()}")

    return X, y, meta, dates, meta_date_idx


# ─── Main walk-forward engine ────────────────────────────────────────────────

def run_momentum_with_overlay(X, y, meta, dates, meta_date_idx,
                               all_data, spy_data, spy_close, all_lr,
                               top_k=3, defensive_shift_factor=1.0,
                               call_config=None, name='base'):
    """
    Run walk-forward LightGBM momentum with covered call overlay.

    call_config: dict with:
        'delta': target delta (0.5=ATM, 0.3=30-delta, 0.2=20-delta, None=no calls)
        'regime_gate': if True, only sell calls in bull regime
        'iv_rank_gate': if float, only sell calls when IV rank > this threshold
    """
    import lightgbm as lgb

    if call_config is None:
        call_config = {'delta': None}

    fprint(f"  [{name}] {X.shape[0]} samples, {len(dates)} dates")

    # Walk-forward: 504d train (2 years), 21d test
    TRAIN_WINDOW = 504
    monthly_returns = []
    monthly_picks = []
    monthly_details = []
    fold = 0
    i = TRAIN_WINDOW
    total_folds = (len(dates) - TRAIN_WINDOW) // 21

    n_calls_sold = 0
    n_assignments = 0
    total_premium_collected = 0.0

    while i + 21 <= len(dates):
        fold += 1
        if fold % 10 == 1:
            fprint(f"    Fold {fold}/{total_folds}...")

        # Use integer index ranges for fast masking
        train_lo, train_hi = i - TRAIN_WINDOW, i
        test_lo, test_hi = i, min(i + 21, len(dates))
        test_dates = dates[test_lo:test_hi]

        train_mask = (meta_date_idx >= train_lo) & (meta_date_idx < train_hi)
        test_mask = (meta_date_idx >= test_lo) & (meta_date_idx < test_hi)

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te = X[test_mask]
        meta_te = meta[test_mask].copy()

        if len(X_tr) < 50 or len(X_te) < 5:
            i += 21
            continue

        model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            subsample=0.8, verbose=-1, n_jobs=4
        )
        model.fit(X_tr, y_tr)

        meta_te = meta_te.copy()
        meta_te['pred'] = model.predict(X_te)

        # Get predictions for rebalance day
        td = test_dates[0]
        dp = meta_te[meta_te['date'] == td].copy()
        if len(dp) < top_k:
            i += 21
            continue

        # Apply defensive shift in bear regime
        regime = detect_regime(spy_close, td)
        if defensive_shift_factor > 0 and regime == 'bear':
            for idx in dp.index:
                ticker = dp.loc[idx, 'ticker']
                if ticker in DEFENSIVE_ETFS:
                    dp.loc[idx, 'pred'] *= (1 + 0.5 * defensive_shift_factor)
                elif ticker in RISK_ON_ETFS:
                    dp.loc[idx, 'pred'] *= (1 - 0.5 * defensive_shift_factor)

        top = dp.nlargest(top_k, 'pred')

        # Calculate returns with optional covered call overlay
        fold_return = 0.0
        picks = []
        fold_details = []

        for _, row in top.iterrows():
            t = row['ticker']
            if t not in all_data:
                continue

            tc = all_data[t]['Close']
            si = tc.index.searchsorted(test_dates[0])
            ei = tc.index.searchsorted(test_dates[-1])
            if si >= len(tc) or ei >= len(tc) or tc.iloc[si] <= 0:
                continue

            entry_price = float(tc.iloc[si])
            exit_price = float(tc.iloc[ei])

            # Get realized vol for this ETF (63d)
            if t in all_lr:
                lr_t = all_lr[t]
                vol_loc = lr_t.index.searchsorted(test_dates[0])
                if vol_loc >= 63:
                    sigma = float(lr_t.iloc[vol_loc - 63:vol_loc].std() * np.sqrt(252))
                else:
                    sigma = 0.20  # default
            else:
                sigma = 0.20

            sigma = max(sigma, 0.05)  # floor

            # Decide whether to sell call
            sell_call = False
            delta_target = call_config.get('delta', None)

            if delta_target is not None:
                sell_call = True

                # Regime gate: only sell in bull
                if call_config.get('regime_gate', False) and regime == 'bear':
                    sell_call = False

                # IV rank gate
                iv_gate = call_config.get('iv_rank_gate', None)
                if iv_gate is not None and t in all_lr:
                    iv_rank = compute_iv_rank(all_lr[t], test_dates[0])
                    if iv_rank < iv_gate:
                        sell_call = False

            cc_result = compute_covered_call_return(
                entry_price=entry_price,
                exit_price=exit_price,
                sigma=sigma,
                delta_target=delta_target if sell_call else 0.3,
                sell_call=sell_call,
            )

            fold_return += cc_result['total_return'] / top_k
            picks.append(t)

            if sell_call:
                n_calls_sold += 1
                total_premium_collected += cc_result['premium_pct']
                if cc_result['capped']:
                    n_assignments += 1

            fold_details.append({
                'ticker': t,
                'entry': entry_price,
                'exit': exit_price,
                'sigma': round(sigma, 4),
                'sell_call': sell_call,
                'premium_pct': round(cc_result['premium_pct'], 5),
                'capped': cc_result['capped'],
                'total_return': round(cc_result['total_return'], 5),
            })

        monthly_returns.append(float(fold_return))
        monthly_picks.append({
            'date': str(td.date()),
            'picks': picks,
            'return': float(fold_return),
            'regime': regime,
        })
        monthly_details.append(fold_details)
        i += 21

    # SPY benchmark
    spy_returns = []
    i = TRAIN_WINDOW
    while i + 21 <= len(dates):
        test_dates_b = dates[i:i + 21]
        sc = spy_data['Close']
        si = sc.index.searchsorted(test_dates_b[0])
        ei = sc.index.searchsorted(test_dates_b[-1])
        if si < len(sc) and ei < len(sc) and sc.iloc[si] > 0:
            spy_ret = sc.iloc[ei] / sc.iloc[si] - 1
            spy_returns.append(float(spy_ret))
        i += 21

    return {
        'monthly_returns': monthly_returns,
        'spy_returns': spy_returns[:len(monthly_returns)],
        'monthly_picks': monthly_picks,
        'monthly_details': monthly_details,
        'n_folds': len(monthly_returns),
        'name': name,
        'call_stats': {
            'n_calls_sold': n_calls_sold,
            'n_assignments': n_assignments,
            'assignment_rate': round(n_assignments / max(n_calls_sold, 1) * 100, 1),
            'avg_premium_pct': round(total_premium_collected / max(n_calls_sold, 1) * 100, 3),
        },
    }


# ─── Metrics and gates (same as v2) ──────────────────────────────────────────

def compute_metrics(returns, name):
    """Compute risk-adjusted metrics."""
    r = np.array(returns)
    if len(r) < 2:
        return {'name': name}

    equity = STARTING_CAPITAL * np.cumprod(1 + r)
    ppy = 12
    sharpe = np.mean(r) / np.std(r) * np.sqrt(ppy) if np.std(r) > 0 else 0
    ds = r[r < 0]
    sortino = np.mean(r) / np.std(ds) * np.sqrt(ppy) if len(ds) > 0 and np.std(ds) > 0 else 0
    years = len(r) / ppy
    cagr = ((equity[-1] / STARTING_CAPITAL) ** (1 / max(years, 0.01)) - 1) * 100
    peak = np.maximum.accumulate(equity)
    maxdd = float(np.min((equity - peak) / peak) * 100)
    wr = len(r[r > 0]) / len(r) * 100
    pf = abs(r[r > 0].sum() / r[r < 0].sum()) if len(ds) > 0 and r[r < 0].sum() != 0 else 999
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0

    return {
        'name': name,
        'sharpe': round(float(sharpe), 2),
        'sortino': round(float(sortino), 2),
        'cagr': round(float(cagr), 1),
        'maxdd': round(float(maxdd), 1),
        'wr': round(float(wr), 1),
        'pf': round(float(pf), 2),
        'calmar': round(float(calmar), 2),
        'n_months': len(r),
        'final_equity': round(float(equity[-1]), 2),
    }


def adversarial_gates(returns, spy_returns, monthly_picks, n_perm=200):
    """Run all 4 adversarial gates."""
    r = np.array(returns)
    sp = np.array(spy_returns[:len(r)])
    gates = {}

    # 1. Permutation test
    fprint("    Permutation test (200 shuffles)...")
    real_sharpe = np.mean(r) / np.std(r) * np.sqrt(12) if np.std(r) > 0 else 0
    perm_sharpes = []
    for _ in range(n_perm):
        signs = np.random.choice([-1, 1], size=len(r))
        perm_r = r * signs
        ps = np.mean(perm_r) / np.std(perm_r) * np.sqrt(12) if np.std(perm_r) > 0 else 0
        perm_sharpes.append(ps)
    perm_p = np.mean(np.array(perm_sharpes) >= real_sharpe)
    gates['permutation'] = {
        'p_value': round(float(perm_p), 3),
        'pass': perm_p < 0.05,
        'real_sharpe': round(float(real_sharpe), 2),
    }
    fprint(f"      p={perm_p:.3f} {'PASS' if perm_p < 0.05 else 'FAIL'}")

    # 2. Regime test (R1)
    fprint("    Regime test...")
    bull_mask = sp > 0
    bear_mask = sp <= 0
    if bull_mask.sum() >= 6 and bear_mask.sum() >= 6:
        bull_r = r[bull_mask]
        bear_r = r[bear_mask]
        bull_sharpe = np.mean(bull_r) / np.std(bull_r) * np.sqrt(12) if np.std(bull_r) > 0 else 0
        bear_sharpe = np.mean(bear_r) / np.std(bear_r) * np.sqrt(12) if np.std(bear_r) > 0 else 0
        max_s = max(abs(bull_sharpe), abs(bear_sharpe))
        gap = abs(bull_sharpe - bear_sharpe) / max_s if max_s > 0 else 0
        gates['regime_r1'] = {
            'bull_sharpe': round(float(bull_sharpe), 2),
            'bear_sharpe': round(float(bear_sharpe), 2),
            'gap': round(float(gap), 3),
            'pass': gap < 0.50,
        }
        fprint(f"      bull={bull_sharpe:.2f}, bear={bear_sharpe:.2f}, gap={gap:.3f} "
               f"{'PASS' if gap < 0.50 else 'FAIL'}")
    else:
        gates['regime_r1'] = {'pass': None, 'note': 'Insufficient regime data'}

    # 3. Sub-period test
    mid = len(r) // 2
    h1, h2 = r[:mid], r[mid:]
    h1_sharpe = np.mean(h1) / np.std(h1) * np.sqrt(12) if np.std(h1) > 0 else 0
    h2_sharpe = np.mean(h2) / np.std(h2) * np.sqrt(12) if np.std(h2) > 0 else 0
    sub_pass = h1_sharpe > 0 and h2_sharpe > 0
    gates['sub_period'] = {
        'h1_sharpe': round(float(h1_sharpe), 2),
        'h2_sharpe': round(float(h2_sharpe), 2),
        'pass': sub_pass,
    }
    fprint(f"      H1={h1_sharpe:.2f}, H2={h2_sharpe:.2f} {'PASS' if sub_pass else 'FAIL'}")

    # 4. Outlier test
    n_remove = max(1, int(len(r) * 0.05))
    sorted_idx = np.argsort(r)[::-1]
    trimmed = np.delete(r, sorted_idx[:n_remove])
    trim_sharpe = np.mean(trimmed) / np.std(trimmed) * np.sqrt(12) if np.std(trimmed) > 0 else 0
    outlier_pass = trim_sharpe > 0
    gates['outlier'] = {
        'trimmed_sharpe': round(float(trim_sharpe), 2),
        'n_removed': n_remove,
        'pass': outlier_pass,
    }
    fprint(f"      Trimmed Sharpe={trim_sharpe:.2f} {'PASS' if outlier_pass else 'FAIL'}")

    return gates


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import yfinance as yf
    try:
        import lightgbm as lgb
    except ImportError:
        fprint("ERROR: LightGBM not available. pip install lightgbm")
        return None

    fprint("=" * 75)
    fprint("COVERED CALL OVERLAY ON SECTOR ETF MOMENTUM — v1")
    fprint("=" * 75)
    fprint(f"Universe: {len(UNIVERSE)} ETFs | Top 3 | 504d train, 21d slide")
    fprint(f"Capital: ${STARTING_CAPITAL:,} | Period: 2019-2026")
    fprint(f"ETF cost: {ETF_COST_BPS}bps RT | Option cost: ${OPTION_COST_PER_CONTRACT}/contract")
    fprint(f"Variants: A=baseline, B=30d, C=20d, D=ATM, E=regime-gate, F=IV-rank-gate")
    fprint()

    # ─── Download data ────────────────────────────────────────────────────
    fprint("Downloading ETF data...")
    all_data = {}
    all_lr = {}  # log returns for vol estimation
    for t in UNIVERSE:
        try:
            df = yf.download(t, start='2016-01-01', end='2026-07-25', progress=False)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 252:
                all_data[t] = df
                all_lr[t] = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        except Exception as e:
            fprint(f"  WARN: Could not download {t}: {e}")

    fprint(f"  Loaded {len(all_data)}/{len(UNIVERSE)} ETFs")

    spy_data = yf.download('SPY', start='2016-01-01', end='2026-07-25', progress=False)
    if spy_data.index.tz is not None:
        spy_data.index = spy_data.index.tz_convert(None)
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.get_level_values(0)
    spy_close = spy_data['Close']

    fprint(f"  SPY: {len(spy_data)} days ({spy_data.index[0].date()} to {spy_data.index[-1].date()})")

    # ─── Build feature matrix once ────────────────────────────────────────
    fprint("\nBuilding feature matrix (one-time)...")
    X, y, meta, dates, meta_date_idx = build_feature_matrix(all_data, spy_close)

    # ─── Define variants ──────────────────────────────────────────────────
    variants = [
        ('A_NoCall_Baseline', {'delta': None}),
        ('B_30delta_Monthly', {'delta': 0.30}),
        ('C_20delta_Monthly', {'delta': 0.20}),
        ('D_ATM_Monthly',     {'delta': 0.50}),
        ('E_RegimeGate_30d',  {'delta': 0.30, 'regime_gate': True}),
        ('F_IVRankGate_30d',  {'delta': 0.30, 'iv_rank_gate': 50.0}),
    ]

    results = []
    gate_results = {}

    for vname, cc_config in variants:
        fprint(f"\n{'─'*60}")
        fprint(f"Running: {vname}")
        fprint(f"  Config: {cc_config}")

        result = run_momentum_with_overlay(
            X, y, meta, dates, meta_date_idx,
            all_data, spy_data, spy_close, all_lr,
            top_k=3, defensive_shift_factor=1.0,
            call_config=cc_config, name=vname,
        )

        if result is None or len(result['monthly_returns']) == 0:
            fprint("  No results — skipping")
            continue

        m = compute_metrics(result['monthly_returns'], vname)
        result['metrics'] = m
        results.append(result)

        cs = result['call_stats']
        fprint(f"  Sharpe: {m['sharpe']} | Sortino: {m['sortino']} | CAGR: {m['cagr']}% | "
               f"MaxDD: {m['maxdd']}% | WR: {m['wr']}% | PF: {m['pf']}")
        if cs['n_calls_sold'] > 0:
            fprint(f"  Calls sold: {cs['n_calls_sold']} | Assigned: {cs['n_assignments']} "
                   f"({cs['assignment_rate']}%) | Avg premium: {cs['avg_premium_pct']}%")

        # Adversarial gates
        fprint(f"  Adversarial gates:")
        gates = adversarial_gates(
            result['monthly_returns'],
            result['spy_returns'],
            result['monthly_picks'],
        )
        gate_results[vname] = gates
        n_pass = sum(1 for g in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                     if gates.get(g, {}).get('pass') is True)
        n_total = sum(1 for g in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                      if gates.get(g, {}).get('pass') is not None)
        fprint(f"  GATES: {n_pass}/{n_total}")

    if not results:
        fprint("\nNO RESULTS — exiting")
        return

    # ─── Comparison table ─────────────────────────────────────────────────
    fprint(f"\n{'='*75}")
    fprint("VARIANT COMPARISON")
    fprint(f"{'='*75}")
    fprint(f"{'Variant':<22} {'Sharpe':>7} {'Sort':>6} {'CAGR':>8} {'MaxDD':>8} "
           f"{'WR':>6} {'PF':>6} {'R1gap':>7} {'Gates':>6}")
    fprint("─" * 75)

    for result in sorted(results, key=lambda x: x['metrics'].get('sharpe', 0), reverse=True):
        m = result['metrics']
        g = gate_results.get(result['name'], {})
        r1_gap = g.get('regime_r1', {}).get('gap', -1)
        n_p = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                  if g.get(k, {}).get('pass') is True)
        n_t = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                  if g.get(k, {}).get('pass') is not None)
        fprint(f"{result['name']:<22} {m.get('sharpe',0):>7.2f} {m.get('sortino',0):>6.2f} "
               f"{m.get('cagr',0):>7.1f}% {m.get('maxdd',0):>7.1f}% "
               f"{m.get('wr',0):>5.1f}% {m.get('pf',0):>6.2f} "
               f"{r1_gap:>7.3f} {n_p}/{n_t}")

    # ─── Call overlay impact analysis ─────────────────────────────────────
    baseline = next((r for r in results if r['name'] == 'A_NoCall_Baseline'), None)
    if baseline:
        bm = baseline['metrics']
        fprint(f"\n{'='*75}")
        fprint("COVERED CALL OVERLAY IMPACT (vs Baseline)")
        fprint(f"{'='*75}")
        fprint(f"{'Variant':<22} {'dSharpe':>8} {'dCAGR':>8} {'dMaxDD':>8} "
               f"{'Calls':>6} {'Assign%':>8} {'AvgPrem':>8}")
        fprint("─" * 75)

        for result in results:
            if result['name'] == 'A_NoCall_Baseline':
                continue
            m = result['metrics']
            cs = result['call_stats']
            ds = m.get('sharpe', 0) - bm.get('sharpe', 0)
            dc = m.get('cagr', 0) - bm.get('cagr', 0)
            dd = m.get('maxdd', 0) - bm.get('maxdd', 0)
            fprint(f"{result['name']:<22} {ds:>+8.2f} {dc:>+7.1f}% {dd:>+7.1f}% "
                   f"{cs['n_calls_sold']:>6} {cs['assignment_rate']:>7.1f}% "
                   f"{cs['avg_premium_pct']:>7.3f}%")

    # ─── Year-by-year for best variant ────────────────────────────────────
    best = max(results, key=lambda x: x['metrics'].get('sharpe', 0))
    fprint(f"\n{'='*75}")
    fprint(f"YEAR-BY-YEAR: {best['name']}")
    fprint(f"{'='*75}")

    picks_by_year = {}
    for p in best['monthly_picks']:
        yr = p['date'][:4]
        if yr not in picks_by_year:
            picks_by_year[yr] = []
        picks_by_year[yr].append(p['return'])

    for yr in sorted(picks_by_year.keys()):
        rets = np.array(picks_by_year[yr])
        yr_ret = (np.prod(1 + rets) - 1) * 100
        yr_sharpe = np.mean(rets) / np.std(rets) * np.sqrt(12) if np.std(rets) > 0 else 0
        fprint(f"  {yr}: {yr_ret:>+8.1f}%  Sharpe {yr_sharpe:>5.2f}  ({len(rets)} months)")

    # ─── SPY comparison ───────────────────────────────────────────────────
    spy_m = compute_metrics(best['spy_returns'][:len(best['monthly_returns'])], 'SPY')
    fprint(f"\n  SPY benchmark: Sharpe {spy_m.get('sharpe',0)}, "
           f"CAGR {spy_m.get('cagr',0)}%, MaxDD {spy_m.get('maxdd',0)}%")

    # ─── Save results ─────────────────────────────────────────────────────
    output = {
        'strategy': 'Covered Call Overlay on Sector ETF Momentum v1',
        'run_date': str(datetime.now()),
        'config': {
            'universe_size': len(UNIVERSE),
            'top_k': 3,
            'train_window': 504,
            'test_window': 21,
            'starting_capital': STARTING_CAPITAL,
            'etf_cost_bps': ETF_COST_BPS,
            'option_cost_per_contract': OPTION_COST_PER_CONTRACT,
        },
        'all_variants': [],
    }

    for result in sorted(results, key=lambda x: x['metrics'].get('sharpe', 0), reverse=True):
        v_out = {
            'name': result['name'],
            'metrics': result['metrics'],
            'call_stats': result['call_stats'],
            'gates': {},
        }
        g = gate_results.get(result['name'], {})
        for gname, gval in g.items():
            v_out['gates'][gname] = {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in gval.items()
            }
        n_pass = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                     if g.get(k, {}).get('pass') is True)
        v_out['gates_passed'] = n_pass
        output['all_variants'].append(v_out)

    # Add monthly picks for best variant
    output['best_variant'] = best['name']
    output['best_metrics'] = best['metrics']
    output['monthly_picks'] = best['monthly_picks']

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # ─── MLflow logging ───────────────────────────────────────────────────
    if MLFLOW_OK:
        try:
            mlflow.set_experiment('covered_call_overlay_momentum')
            for result in results:
                m = result['metrics']
                g = gate_results.get(result['name'], {})
                cs = result['call_stats']
                with mlflow.start_run(run_name=f'cc_overlay_{result["name"]}'):
                    mlflow.log_metrics({
                        'sharpe': m.get('sharpe', 0),
                        'sortino': m.get('sortino', 0),
                        'cagr': m.get('cagr', 0),
                        'maxdd': m.get('maxdd', 0),
                        'wr': m.get('wr', 0),
                        'pf': m.get('pf', 0),
                        'calmar': m.get('calmar', 0),
                        'perm_p': g.get('permutation', {}).get('p_value', -1),
                        'r1_gap': g.get('regime_r1', {}).get('gap', -1),
                        'n_gates_pass': sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                                            if g.get(k, {}).get('pass') is True),
                        'n_calls_sold': cs['n_calls_sold'],
                        'assignment_rate': cs['assignment_rate'],
                        'avg_premium_pct': cs['avg_premium_pct'],
                    })
                    mlflow.log_params({
                        'variant': result['name'],
                        'universe_size': len(UNIVERSE),
                        'top_k': 3,
                        'train_window': 504,
                        'starting_capital': STARTING_CAPITAL,
                    })
            fprint("MLflow logging complete.")
        except Exception as e:
            fprint(f"MLflow error: {e}")

    fprint(f"\n{'='*75}")
    fprint("DONE — Covered Call Overlay v1")
    fprint(f"{'='*75}")

    return output


if __name__ == '__main__':
    main()
