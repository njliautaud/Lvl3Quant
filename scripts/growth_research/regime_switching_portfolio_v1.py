#!/usr/bin/env python3
"""
Regime-Switching Portfolio v1 — Fundamentally Different Approach to R1
======================================================================

Instead of fixing one strategy across regimes, SWITCH between:
  Bull: Sector ETF momentum (LightGBM ranker, proven Sharpe 4.63)
  Bear: Purpose-built defensive strategies

Bear-market components tested:
  A) Treasury momentum — TLT/IEF when SPY < 200d SMA
  B) Gold/commodity rotation — GLD/DBC in uncertainty
  C) Inverse momentum — short worst ETFs (or cash)
  D) Volatility harvesting — buy after VIX spikes
  E) Cash — simple risk-off

Regime detectors:
  1) SPY vs 200d SMA
  2) SPY 63d return < -5%
  3) VIX > 25
  4) Ensemble (2 of 3 agree)

20 core variants (5 bear × 4 detectors) + blends.
Walk-forward 2019-2026, 22 sector ETFs, LightGBM ranker, $100K start.

KEY METRIC: R1 gap < 0.50 while Sharpe > 2.0.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product
import warnings
warnings.filterwarnings('ignore')


def fprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'regime_switching_portfolio_v1_results.json'

# MLflow setup
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
    fprint("MLflow connected")
except Exception:
    fprint("MLflow unavailable — results will be saved locally only")

# ── Universe (same as sector_etf_momentum_v2) ──
UNIVERSE = [
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE',
    'XLC', 'QQQ', 'IWM', 'MDY', 'EFA', 'EEM', 'GLD', 'TLT', 'HYG', 'IYR',
    'VNQ', 'DBC',
]

DEFENSIVE_ETFS = {'XLP', 'XLU', 'TLT', 'GLD'}
RISK_ON_ETFS = {'XLK', 'QQQ', 'XLY', 'IWM', 'EEM'}

# Bear-market specific instruments
TREASURY_ETFS = ['TLT', 'IEF']
COMMODITY_ETFS = ['GLD', 'DBC']
DEFENSIVE_ROTATION = ['XLP', 'XLU', 'XLV']  # Consumer staples, utilities, healthcare


# ═══════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (identical to sector_etf_momentum_v2_regime_lgbm)
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════

def detect_regime_sma200(spy_close, date):
    """SPY < 200d SMA → bear."""
    loc = spy_close.index.searchsorted(date)
    if loc < 200:
        return 'bull'
    sma200 = spy_close.iloc[max(0, loc - 200):loc].mean()
    return 'bear' if spy_close.iloc[loc - 1] < sma200 else 'bull'


def detect_regime_drawdown(spy_close, date):
    """SPY 63d return < -5% → bear."""
    loc = spy_close.index.searchsorted(date)
    if loc < 63:
        return 'bull'
    ret_63d = spy_close.iloc[loc - 1] / spy_close.iloc[loc - 63] - 1
    return 'bear' if ret_63d < -0.05 else 'bull'


def detect_regime_vix(vix_close, date):
    """VIX > 25 → bear."""
    loc = vix_close.index.searchsorted(date)
    if loc < 1:
        return 'bull'
    return 'bear' if vix_close.iloc[loc - 1] > 25 else 'bull'


def detect_regime_ensemble(spy_close, vix_close, date):
    """2 of 3 detectors must agree on bear."""
    votes = [
        detect_regime_sma200(spy_close, date),
        detect_regime_drawdown(spy_close, date),
        detect_regime_vix(vix_close, date),
    ]
    bear_votes = sum(1 for v in votes if v == 'bear')
    return 'bear' if bear_votes >= 2 else 'bull'


REGIME_DETECTORS = {
    'sma200': lambda spy, vix, d: detect_regime_sma200(spy, d),
    'drawdown': lambda spy, vix, d: detect_regime_drawdown(spy, d),
    'vix25': lambda spy, vix, d: detect_regime_vix(vix, d),
    'ensemble': lambda spy, vix, d: detect_regime_ensemble(spy, vix, d),
}


# ═══════════════════════════════════════════════════════════════════════
# BEAR-MARKET STRATEGIES
# ═══════════════════════════════════════════════════════════════════════

def bear_treasury_momentum(all_data, date, test_dates):
    """Rotate to best-performing treasury ETF (TLT vs IEF)."""
    best_ret = -999
    best_ticker = 'TLT'
    for t in TREASURY_ETFS:
        if t in all_data:
            c = all_data[t]['Close']
            loc = c.index.searchsorted(date)
            if loc >= 63:
                ret = c.iloc[loc - 1] / c.iloc[loc - 63] - 1
                if ret > best_ret:
                    best_ret = ret
                    best_ticker = t
    # Return the actual return over the test period
    if best_ticker in all_data:
        c = all_data[best_ticker]['Close']
        si = c.index.searchsorted(test_dates[0])
        ei = c.index.searchsorted(test_dates[-1])
        if si < len(c) and ei < len(c) and c.iloc[si] > 0:
            return c.iloc[ei] / c.iloc[si] - 1, [best_ticker]
    return 0.0, [best_ticker]


def bear_gold_commodity(all_data, date, test_dates):
    """Equal-weight GLD + DBC rotation (pick best 63d momentum of available)."""
    available = [t for t in COMMODITY_ETFS if t in all_data]
    if not available:
        return 0.0, []
    ret_total = 0.0
    picks = []
    for t in available:
        c = all_data[t]['Close']
        si = c.index.searchsorted(test_dates[0])
        ei = c.index.searchsorted(test_dates[-1])
        if si < len(c) and ei < len(c) and c.iloc[si] > 0:
            ret_total += (c.iloc[ei] / c.iloc[si] - 1) / len(available)
            picks.append(t)
    return ret_total, picks


def bear_inverse_momentum(all_data, date, test_dates, universe):
    """Short the 3 worst-momentum ETFs (equal weight). If not possible, go to cash."""
    # Calculate 63d momentum for all
    mom = {}
    for t in universe:
        if t in all_data:
            c = all_data[t]['Close']
            loc = c.index.searchsorted(date)
            if loc >= 63 and c.iloc[loc - 63] > 0:
                mom[t] = c.iloc[loc - 1] / c.iloc[loc - 63] - 1
    if len(mom) < 3:
        return 0.0, ['CASH']
    # Sort by momentum ascending (worst first) — short these
    worst = sorted(mom.items(), key=lambda x: x[1])[:3]
    ret_total = 0.0
    picks = []
    for t, _ in worst:
        c = all_data[t]['Close']
        si = c.index.searchsorted(test_dates[0])
        ei = c.index.searchsorted(test_dates[-1])
        if si < len(c) and ei < len(c) and c.iloc[si] > 0:
            stock_ret = c.iloc[ei] / c.iloc[si] - 1
            ret_total += (-stock_ret) / 3  # SHORT
            picks.append(f'-{t}')
    return ret_total, picks


def bear_vix_harvest(all_data, spy_close, vix_close, date, test_dates):
    """Buy SPY when VIX is elevated (>30) → mean-reversion. Otherwise cash."""
    loc = vix_close.index.searchsorted(date)
    if loc < 1:
        return 0.0, ['CASH']
    current_vix = vix_close.iloc[loc - 1]
    if current_vix > 30:
        # VIX spike — buy SPY for mean-reversion
        si = spy_close.index.searchsorted(test_dates[0])
        ei = spy_close.index.searchsorted(test_dates[-1])
        if si < len(spy_close) and ei < len(spy_close) and spy_close.iloc[si] > 0:
            return spy_close.iloc[ei] / spy_close.iloc[si] - 1, ['SPY(VIX>30)']
    # Also try defensive rotation when VIX is 25-30
    if current_vix > 25:
        available = [t for t in DEFENSIVE_ROTATION if t in all_data]
        if available:
            ret_total = 0.0
            picks = []
            for t in available:
                c = all_data[t]['Close']
                si = c.index.searchsorted(test_dates[0])
                ei = c.index.searchsorted(test_dates[-1])
                if si < len(c) and ei < len(c) and c.iloc[si] > 0:
                    ret_total += (c.iloc[ei] / c.iloc[si] - 1) / len(available)
                    picks.append(t)
            return ret_total, picks
    return 0.0, ['CASH']


def bear_cash(all_data, date, test_dates):
    """100% cash in bear regime."""
    return 0.0, ['CASH']


BEAR_STRATEGIES = {
    'treasury': bear_treasury_momentum,
    'gold_commodity': bear_gold_commodity,
    'inverse_mom': bear_inverse_momentum,
    'vix_harvest': bear_vix_harvest,
    'cash': bear_cash,
}


# ═══════════════════════════════════════════════════════════════════════
# PRECOMPUTE: Feature matrix + LightGBM walk-forward (run ONCE)
# ═══════════════════════════════════════════════════════════════════════

def precompute_features_and_predictions(all_data, spy_close, top_k=3):
    """
    Build features once and run LightGBM walk-forward once.
    Returns per-fold predictions and bull-mode picks/returns for reuse.
    """
    import lightgbm as lgb

    fprint("  Building feature matrix (one-time)...")
    common = None
    for t, df in all_data.items():
        common = df.index if common is None else common.intersection(df.index)

    features_list = []
    labels_list = []
    meta_list = []

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

    start_2019 = pd.Timestamp('2019-01-01')
    test_start_idx = next((i for i, d in enumerate(dates) if d >= start_2019), 252)
    test_start_idx = max(test_start_idx, 252)

    fprint(f"  Features: {X.shape[0]}x{X.shape[1]}, "
           f"test from {dates[test_start_idx].date()}")

    # Walk-forward LightGBM — cache per-fold predictions
    fprint("  Running LightGBM walk-forward (one-time)...")
    fold_results = []  # list of dicts: {date, test_dates, top_picks, bull_return}
    feat_imp_accum = None
    n_folds = 0

    i = test_start_idx
    while i + 21 <= len(dates):
        n_folds += 1
        train_dates = dates[i - 252:i]
        test_dates = dates[i:i + 21]
        td = test_dates[0]

        train_mask = meta['date'].isin(train_dates)
        test_mask = meta['date'].isin([td])

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te = X[test_mask]
        meta_te = meta[test_mask].copy()

        if len(X_tr) < 50 or len(X_te) < 3:
            i += 21
            continue

        model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            subsample=0.8, verbose=-1, n_jobs=-1,
        )
        model.fit(X_tr, y_tr)

        if feat_imp_accum is None:
            feat_imp_accum = model.feature_importances_.astype(float)
        else:
            feat_imp_accum += model.feature_importances_.astype(float)

        meta_te = meta_te.copy()
        meta_te['pred'] = model.predict(X_te)

        if len(meta_te) < top_k:
            i += 21
            continue

        top = meta_te.nlargest(top_k, 'pred')

        # Compute bull return
        bull_ret = 0.0
        bull_picks = []
        for _, row in top.iterrows():
            t = row['ticker']
            if t in all_data:
                tc = all_data[t]['Close']
                si = tc.index.searchsorted(test_dates[0])
                ei = tc.index.searchsorted(test_dates[-1])
                if si < len(tc) and ei < len(tc) and tc.iloc[si] > 0:
                    stock_ret = tc.iloc[ei] / tc.iloc[si] - 1
                    bull_ret += stock_ret / top_k
                    bull_picks.append(t)

        fold_results.append({
            'date': td,
            'test_dates': test_dates,
            'bull_return': bull_ret,
            'bull_picks': bull_picks,
        })

        if n_folds % 20 == 0:
            fprint(f"    Fold {n_folds}...")

        i += 21

    fprint(f"  Walk-forward done: {n_folds} folds, {len(fold_results)} valid")

    return {
        'fold_results': fold_results,
        'feature_importances': feat_imp_accum / max(n_folds, 1) if feat_imp_accum is not None else None,
        'dates': dates,
        'test_start_idx': test_start_idx,
    }


# ═══════════════════════════════════════════════════════════════════════
# REGIME-SWITCHING ASSEMBLER (fast — uses precomputed data)
# ═══════════════════════════════════════════════════════════════════════

def assemble_regime_switching(precomputed, all_data, spy_data, spy_close, vix_close,
                              regime_detector_name, bear_strategy_name,
                              cost_bps=20, name='variant'):
    """
    Assemble regime-switching returns using precomputed bull predictions.
    Only the regime detection + bear strategy logic runs per variant (fast).
    """
    detect_fn = REGIME_DETECTORS[regime_detector_name]
    fold_results = precomputed['fold_results']

    monthly_picks = []
    monthly_returns = []
    regime_log = []

    for fold in fold_results:
        td = fold['date']
        test_dates = fold['test_dates']

        regime = detect_fn(spy_close, vix_close, td)
        regime_log.append(regime)

        if regime == 'bull':
            ret = fold['bull_return']
            picks = fold['bull_picks']
        else:
            # Bear strategy
            if bear_strategy_name == 'treasury':
                ret, picks = bear_treasury_momentum(all_data, td, test_dates)
            elif bear_strategy_name == 'gold_commodity':
                ret, picks = bear_gold_commodity(all_data, td, test_dates)
            elif bear_strategy_name == 'inverse_mom':
                ret, picks = bear_inverse_momentum(all_data, td, test_dates, UNIVERSE)
            elif bear_strategy_name == 'vix_harvest':
                ret, picks = bear_vix_harvest(all_data, spy_close, vix_close, td, test_dates)
            elif bear_strategy_name == 'cash':
                ret, picks = bear_cash(all_data, td, test_dates)
            else:
                ret, picks = 0.0, ['CASH']

        ret -= cost_bps / 10000 * 2  # transition cost

        monthly_picks.append({
            'date': str(td.date()),
            'picks': picks,
            'return': float(ret),
            'regime': regime,
        })
        monthly_returns.append(float(ret))

    # SPY benchmark
    spy_returns = []
    for fold in fold_results:
        test_dates_b = fold['test_dates']
        sc = spy_data['Close']
        si = sc.index.searchsorted(test_dates_b[0])
        ei = sc.index.searchsorted(test_dates_b[-1])
        if si < len(sc) and ei < len(sc) and sc.iloc[si] > 0:
            spy_returns.append(float(sc.iloc[ei] / sc.iloc[si] - 1))
        else:
            spy_returns.append(0.0)

    bull_months = sum(1 for r in regime_log if r == 'bull')
    bear_months = sum(1 for r in regime_log if r == 'bear')

    return {
        'monthly_returns': monthly_returns,
        'spy_returns': spy_returns[:len(monthly_returns)],
        'monthly_picks': monthly_picks,
        'n_folds': len(monthly_returns),
        'regime_detector': regime_detector_name,
        'bear_strategy': bear_strategy_name,
        'bull_months': bull_months,
        'bear_months': bear_months,
        'feature_importances': precomputed['feature_importances'],
        'name': name,
    }


# ═══════════════════════════════════════════════════════════════════════
# METRICS + ADVERSARIAL GATES
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(returns, name):
    """Compute risk-adjusted metrics."""
    r = np.array(returns)
    if len(r) < 2:
        return {'name': name, 'sharpe': 0, 'cagr': 0, 'maxdd': 0, 'wr': 0, 'pf': 0}

    equity = 100000 * np.cumprod(1 + r)
    ppy = 12
    sharpe = np.mean(r) / np.std(r) * np.sqrt(ppy) if np.std(r) > 0 else 0
    ds = r[r < 0]
    sortino = np.mean(r) / np.std(ds) * np.sqrt(ppy) if len(ds) > 0 and np.std(ds) > 0 else 0
    years = len(r) / ppy
    cagr = ((equity[-1] / 100000) ** (1 / max(years, 0.01)) - 1) * 100
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
    """4-gate adversarial audit: permutation, R1 regime, sub-period, outlier."""
    r = np.array(returns)
    sp = np.array(spy_returns[:len(r)])
    gates = {}

    # 1. Permutation test
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
        'pass': bool(perm_p < 0.05),
        'real_sharpe': round(float(real_sharpe), 2),
    }

    # 2. Regime test (R1) — SPY monthly returns as classifier
    bull_mask = sp > 0
    bear_mask = sp <= 0

    # Also use per-month regime labels from picks
    regimes = [p.get('regime', 'bull') if isinstance(p, dict) else 'bull'
               for p in monthly_picks[:len(r)]]
    regime_bull_mask = np.array([reg == 'bull' for reg in regimes])
    regime_bear_mask = np.array([reg == 'bear' for reg in regimes])

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
            'pass': bool(gap < 0.50),
            'bull_months': int(bull_mask.sum()),
            'bear_months': int(bear_mask.sum()),
        }
    else:
        gates['regime_r1'] = {'pass': None, 'note': 'Insufficient regime data'}

    # R1 using strategy's own regime labels
    if regime_bull_mask.sum() >= 3 and regime_bear_mask.sum() >= 3:
        br = r[regime_bull_mask]
        brr = r[regime_bear_mask]
        bs = np.mean(br) / np.std(br) * np.sqrt(12) if np.std(br) > 0 else 0
        brs = np.mean(brr) / np.std(brr) * np.sqrt(12) if np.std(brr) > 0 else 0
        ms = max(abs(bs), abs(brs))
        g2 = abs(bs - brs) / ms if ms > 0 else 0
        gates['regime_own_labels'] = {
            'bull_sharpe': round(float(bs), 2),
            'bear_sharpe': round(float(brs), 2),
            'gap': round(float(g2), 3),
            'pass': bool(g2 < 0.50),
            'bull_months': int(regime_bull_mask.sum()),
            'bear_months': int(regime_bear_mask.sum()),
        }

    # 3. Sub-period test
    mid = len(r) // 2
    h1, h2 = r[:mid], r[mid:]
    h1_sharpe = np.mean(h1) / np.std(h1) * np.sqrt(12) if np.std(h1) > 0 else 0
    h2_sharpe = np.mean(h2) / np.std(h2) * np.sqrt(12) if np.std(h2) > 0 else 0
    sub_pass = h1_sharpe > 0 and h2_sharpe > 0
    gates['sub_period'] = {
        'h1_sharpe': round(float(h1_sharpe), 2),
        'h2_sharpe': round(float(h2_sharpe), 2),
        'pass': bool(sub_pass),
    }

    # 4. Outlier test
    n_remove = max(1, int(len(r) * 0.05))
    sorted_idx = np.argsort(r)[::-1]
    trimmed = np.delete(r, sorted_idx[:n_remove])
    trim_sharpe = np.mean(trimmed) / np.std(trimmed) * np.sqrt(12) if np.std(trimmed) > 0 else 0
    gates['outlier'] = {
        'trimmed_sharpe': round(float(trim_sharpe), 2),
        'n_removed': n_remove,
        'pass': bool(trim_sharpe > 0),
    }

    return gates


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    import yfinance as yf
    try:
        import lightgbm as lgb
    except ImportError:
        fprint("ERROR: LightGBM not available")
        return None

    t0 = time.time()

    fprint("=" * 70)
    fprint("REGIME-SWITCHING PORTFOLIO v1")
    fprint("Bull: LightGBM sector momentum | Bear: purpose-built defense")
    fprint("=" * 70)
    fprint(f"Universe: {len(UNIVERSE)} ETFs | Top 3 in bull | 20bps cost")
    fprint(f"Regime detectors: {list(REGIME_DETECTORS.keys())}")
    fprint(f"Bear strategies: {list(BEAR_STRATEGIES.keys())}")
    fprint(f"Variants: {len(REGIME_DETECTORS) * len(BEAR_STRATEGIES)} core\n")

    # ── Download data ──
    fprint("Downloading ETF data...")
    all_tickers = list(set(UNIVERSE + TREASURY_ETFS + COMMODITY_ETFS +
                           DEFENSIVE_ROTATION + ['SPY', 'IEF', '^VIX']))
    all_data = {}

    for t in all_tickers:
        if t in ('^VIX', 'SPY'):
            continue
        try:
            df = yf.download(t, start='2008-01-01', end='2026-07-25', progress=False)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 252:
                all_data[t] = df
        except Exception as e:
            fprint(f"  Skip {t}: {e}")

    # SPY
    spy_data = yf.download('SPY', start='2008-01-01', end='2026-07-25', progress=False)
    if spy_data.index.tz is not None:
        spy_data.index = spy_data.index.tz_convert(None)
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.get_level_values(0)
    spy_close = spy_data['Close']

    # VIX
    vix_data = yf.download('^VIX', start='2008-01-01', end='2026-07-25', progress=False)
    if vix_data.index.tz is not None:
        vix_data.index = vix_data.index.tz_convert(None)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.get_level_values(0)
    vix_close = vix_data['Close']

    fprint(f"  Loaded {len(all_data)} ETFs + SPY ({len(spy_data)}d) + VIX ({len(vix_data)}d)")

    # ── Precompute features + LightGBM walk-forward ONCE ──
    precomputed = precompute_features_and_predictions(all_data, spy_close, top_k=3)

    # ── Run all variants (fast — only regime detection + bear strategy) ──
    variant_configs = list(product(
        list(BEAR_STRATEGIES.keys()),
        list(REGIME_DETECTORS.keys()),
    ))

    all_results = []
    all_gate_results = {}

    for vi, (bear_name, detector_name) in enumerate(variant_configs):
        name = f'{bear_name}__{detector_name}'
        fprint(f"\n[{vi+1}/{len(variant_configs)}] {name}")

        result = assemble_regime_switching(
            precomputed, all_data, spy_data, spy_close, vix_close,
            regime_detector_name=detector_name,
            bear_strategy_name=bear_name,
            cost_bps=20, name=name,
        )

        if result is None or len(result['monthly_returns']) < 6:
            fprint("  Skipped — insufficient data")
            continue

        m = compute_metrics(result['monthly_returns'], name)
        result['metrics'] = m

        fprint(f"  Sharpe {m['sharpe']:.2f} | CAGR {m['cagr']:.1f}% | "
               f"MaxDD {m['maxdd']:.1f}% | WR {m['wr']:.1f}% | PF {m['pf']:.2f} | "
               f"Bull {result['bull_months']}m Bear {result['bear_months']}m")

        # Run adversarial gates
        gates = adversarial_gates(
            result['monthly_returns'],
            result['spy_returns'],
            result['monthly_picks'],
        )
        all_gate_results[name] = gates

        r1 = gates.get('regime_r1', {})
        n_pass = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                     if gates.get(k, {}).get('pass') is True)
        n_total = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                      if gates.get(k, {}).get('pass') is not None)
        result['n_gates_pass'] = n_pass
        result['n_gates_total'] = n_total

        fprint(f"  R1: bull={r1.get('bull_sharpe','?')}, bear={r1.get('bear_sharpe','?')}, "
               f"gap={r1.get('gap','?')} {'PASS' if r1.get('pass') else 'FAIL'} | "
               f"Gates {n_pass}/{n_total}")

        all_results.append(result)

    if not all_results:
        fprint("NO RESULTS — aborting")
        return

    # ═══════════════════════════════════════════════════════════════════
    # RESULTS SUMMARY
    # ═══════════════════════════════════════════════════════════════════

    fprint(f"\n{'='*90}")
    fprint("ALL VARIANTS — SORTED BY R1 GAP (lower = better)")
    fprint(f"{'='*90}")
    fprint(f"{'Variant':<30} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'WR':>6} "
           f"{'R1gap':>7} {'R1':>5} {'Gates':>6} {'Bull':>5} {'Bear':>5}")
    fprint("-" * 90)

    sorted_results = sorted(all_results,
                            key=lambda x: all_gate_results[x['name']].get('regime_r1', {}).get('gap', 999))

    for result in sorted_results:
        m = result['metrics']
        g = all_gate_results[result['name']]
        r1_gap = g.get('regime_r1', {}).get('gap', -1)
        r1_pass = g.get('regime_r1', {}).get('pass', False)
        r1_str = "PASS" if r1_pass else "FAIL"
        fprint(f"{result['name']:<30} {m['sharpe']:>7.2f} {m['cagr']:>6.1f}% "
               f"{m['maxdd']:>6.1f}% {m['wr']:>5.1f}% "
               f"{r1_gap:>7.3f} {r1_str:>5} "
               f"{result['n_gates_pass']}/{result['n_gates_total']} "
               f"{result['bull_months']:>5} {result['bear_months']:>5}")

    # ── Identify winners ──
    r1_passers = [r for r in all_results
                  if all_gate_results[r['name']].get('regime_r1', {}).get('pass') is True
                  and r['metrics']['sharpe'] >= 2.0]

    if r1_passers:
        # Best R1-passing variant with Sharpe >= 2.0
        winner = max(r1_passers, key=lambda x: x['metrics']['sharpe'])
        fprint(f"\n{'='*70}")
        fprint(f"WINNER (R1 PASS + Sharpe >= 2.0): {winner['name']}")
        fprint(f"{'='*70}")
    else:
        # Fall back to best Sharpe overall
        winner = max(all_results, key=lambda x: x['metrics']['sharpe'])
        fprint(f"\n{'='*70}")
        fprint(f"NO R1 PASSERS with Sharpe >= 2.0 — Best overall: {winner['name']}")
        fprint(f"{'='*70}")

    wm = winner['metrics']
    wg = all_gate_results[winner['name']]
    wr1 = wg.get('regime_r1', {})

    fprint(f"  Sharpe:      {wm['sharpe']}")
    fprint(f"  Sortino:     {wm['sortino']}")
    fprint(f"  CAGR:        {wm['cagr']}%")
    fprint(f"  MaxDD:       {wm['maxdd']}%")
    fprint(f"  WR:          {wm['wr']}%")
    fprint(f"  PF:          {wm['pf']}")
    fprint(f"  Calmar:      {wm['calmar']}")
    fprint(f"  Bull Sharpe: {wr1.get('bull_sharpe', '?')}")
    fprint(f"  Bear Sharpe: {wr1.get('bear_sharpe', '?')}")
    fprint(f"  R1 Gap:      {wr1.get('gap', '?')}")
    fprint(f"  Bull months: {winner['bull_months']}")
    fprint(f"  Bear months: {winner['bear_months']}")
    fprint(f"  Regime det:  {winner['regime_detector']}")
    fprint(f"  Bear strat:  {winner['bear_strategy']}")

    # ── Year-by-year breakdown ──
    fprint(f"\nYear-by-year ({winner['name']}):")
    picks_by_year = {}
    for p in winner['monthly_picks']:
        yr = p['date'][:4]
        if yr not in picks_by_year:
            picks_by_year[yr] = []
        picks_by_year[yr].append(p['return'])

    for yr in sorted(picks_by_year.keys()):
        rets = picks_by_year[yr]
        yr_ret = (np.prod(1 + np.array(rets)) - 1) * 100
        yr_sharpe = np.mean(rets) / np.std(rets) * np.sqrt(12) if np.std(rets) > 0 else 0
        n_bear = sum(1 for p in winner['monthly_picks'] if p['date'].startswith(yr) and p['regime'] == 'bear')
        fprint(f"  {yr}: {yr_ret:>7.1f}%  Sharpe {yr_sharpe:.2f}  ({len(rets)}m, {n_bear} bear)")

    # ── Gate details for winner ──
    fprint(f"\nGate details ({winner['name']}):")
    for gn in ['permutation', 'regime_r1', 'regime_own_labels', 'sub_period', 'outlier']:
        if gn in wg:
            status = 'PASS' if wg[gn].get('pass') else 'FAIL'
            fprint(f"  {gn}: {status} — {wg[gn]}")

    # ── Best by bear strategy ──
    fprint(f"\n{'='*70}")
    fprint("BEST BY BEAR STRATEGY")
    fprint(f"{'='*70}")
    for bs in BEAR_STRATEGIES.keys():
        bs_results = [r for r in all_results if r['bear_strategy'] == bs]
        if bs_results:
            best = max(bs_results, key=lambda x: x['metrics']['sharpe'])
            m = best['metrics']
            g = all_gate_results[best['name']]
            r1 = g.get('regime_r1', {})
            fprint(f"  {bs:<18} Best: {best['regime_detector']:<12} "
                   f"Sharpe {m['sharpe']:.2f}  R1gap {r1.get('gap', '?')}")

    # ── Best by regime detector ──
    fprint(f"\nBEST BY REGIME DETECTOR")
    for rd in REGIME_DETECTORS.keys():
        rd_results = [r for r in all_results if r['regime_detector'] == rd]
        if rd_results:
            best = max(rd_results, key=lambda x: x['metrics']['sharpe'])
            m = best['metrics']
            g = all_gate_results[best['name']]
            r1 = g.get('regime_r1', {})
            fprint(f"  {rd:<12} Best: {best['bear_strategy']:<18} "
                   f"Sharpe {m['sharpe']:.2f}  R1gap {r1.get('gap', '?')}")

    # ═══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════

    def sanitize(obj):
        """Make JSON-serializable."""
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        return obj

    output = {
        'strategy': 'Regime-Switching Portfolio v1',
        'run_date': str(datetime.now()),
        'approach': 'Bull=LightGBM sector momentum, Bear=purpose-built defense, regime switch',
        'cost_bps': 20,
        'top_k_bull': 3,
        'universe_size': len(UNIVERSE),
        'winner': {
            'name': winner['name'],
            'metrics': sanitize(wm),
            'gates': sanitize(wg),
            'regime_detector': winner['regime_detector'],
            'bear_strategy': winner['bear_strategy'],
            'bull_months': winner['bull_months'],
            'bear_months': winner['bear_months'],
        },
        'r1_passers_sharpe_ge_2': [{
            'name': r['name'],
            'sharpe': r['metrics']['sharpe'],
            'r1_gap': all_gate_results[r['name']].get('regime_r1', {}).get('gap', -1),
            'gates_pass': r['n_gates_pass'],
        } for r in r1_passers] if r1_passers else [],
        'all_variants': [{
            'name': r['name'],
            'metrics': sanitize(r['metrics']),
            'gates': sanitize(all_gate_results[r['name']]),
            'regime_detector': r['regime_detector'],
            'bear_strategy': r['bear_strategy'],
            'bull_months': r['bull_months'],
            'bear_months': r['bear_months'],
        } for r in sorted_results],
        'monthly_picks_winner': sanitize(winner['monthly_picks']),
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # ═══════════════════════════════════════════════════════════════════
    # MLFLOW LOGGING
    # ═══════════════════════════════════════════════════════════════════

    if MLFLOW_OK:
        try:
            mlflow.set_experiment('regime_switching_portfolio')

            # Log winner
            with mlflow.start_run(run_name=f'rsw_v1_{winner["name"]}'):
                mlflow.log_metrics({
                    'sharpe': wm.get('sharpe', 0),
                    'sortino': wm.get('sortino', 0),
                    'cagr': wm.get('cagr', 0),
                    'maxdd': wm.get('maxdd', 0),
                    'wr': wm.get('wr', 0),
                    'pf': wm.get('pf', 0),
                    'calmar': wm.get('calmar', 0),
                    'r1_gap': wr1.get('gap', -1),
                    'perm_p': wg.get('permutation', {}).get('p_value', -1),
                    'n_gates_pass': winner['n_gates_pass'],
                    'bull_months': winner['bull_months'],
                    'bear_months': winner['bear_months'],
                })
                mlflow.log_params({
                    'winner_name': winner['name'],
                    'regime_detector': winner['regime_detector'],
                    'bear_strategy': winner['bear_strategy'],
                    'top_k': 3,
                    'cost_bps': 20,
                    'universe_size': len(UNIVERSE),
                    'n_variants': len(all_results),
                    'n_r1_passers': len(r1_passers) if r1_passers else 0,
                })
                mlflow.log_artifact(str(RESULTS_PATH))
            fprint("MLflow: winner logged")

            # Log all variants as child runs
            for result in all_results:
                rn = result['name']
                rm = result['metrics']
                rg = all_gate_results[rn]
                rr1 = rg.get('regime_r1', {})
                with mlflow.start_run(run_name=f'rsw_v1_{rn}'):
                    mlflow.log_metrics({
                        'sharpe': rm.get('sharpe', 0),
                        'sortino': rm.get('sortino', 0),
                        'cagr': rm.get('cagr', 0),
                        'maxdd': rm.get('maxdd', 0),
                        'wr': rm.get('wr', 0),
                        'pf': rm.get('pf', 0),
                        'r1_gap': rr1.get('gap', -1),
                        'r1_pass': 1 if rr1.get('pass') else 0,
                        'n_gates_pass': result['n_gates_pass'],
                    })
                    mlflow.log_params({
                        'regime_detector': result['regime_detector'],
                        'bear_strategy': result['bear_strategy'],
                        'bull_months': result['bull_months'],
                        'bear_months': result['bear_months'],
                    })
            fprint("MLflow: all variants logged")

        except Exception as e:
            fprint(f"MLflow error: {e}")

    elapsed = time.time() - t0
    fprint(f"\n{'='*70}")
    fprint(f"DONE in {elapsed/60:.1f} min")
    fprint(f"{'='*70}")

    return output


if __name__ == '__main__':
    main()
