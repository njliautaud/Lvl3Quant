#!/usr/bin/env python3
"""
Sector ETF Momentum v2 — LightGBM + Regime-Adaptive Defensive Shift
=====================================================================

Takes the validated LightGBM ETF momentum (Sharpe 3.96, 3/4 gates, R1 FAIL 0.784)
and applies the defensive shift regime overlay that fixed R1 in the simple scoring
version (gap 0.784 → 0.085).

Two approaches combined:
1. Add regime features to LightGBM (VIX proxy, QQQ vs SMA200, cross-asset momentum)
2. Post-prediction defensive shift in bear markets (boost defensive, penalize risk-on)

Also tests Top 5 instead of Top 3 (which helped the simple version).

Walk-forward: 252d train, 21d test, sliding window, 2008-2026.
Cost: 20bps round-trip (same as v1).
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
    print(*args, **kwargs)
    sys.stdout.flush()

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'sector_etf_momentum_v2_regime_lgbm_results.json'

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

# Universe: same as v1 — all existed 10+ years
UNIVERSE = [
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE',
    'XLC', 'QQQ', 'IWM', 'MDY', 'EFA', 'EEM', 'GLD', 'TLT', 'HYG', 'IYR',
    'VNQ', 'DBC',
]

DEFENSIVE_ETFS = {'XLP', 'XLU', 'TLT', 'IEF', 'GLD'}
RISK_ON_ETFS = {'XLK', 'QQQ', 'XLY', 'XBI', 'IWM', 'EEM'}


def build_features(close, volume, spy_close=None):
    """Build momentum + quality + regime features for a single ETF."""
    lr = np.log(close / close.shift(1))

    feat = pd.DataFrame({
        # Momentum features (same as v1)
        'ret_5d': close.pct_change(5),
        'ret_10d': close.pct_change(10),
        'ret_21d': close.pct_change(21),
        'ret_63d': close.pct_change(63),
        'ret_126d': close.pct_change(126),
        'ret_252d': close.pct_change(252),
        'mom_12_1': close.pct_change(252) - close.pct_change(21),
        'high_52w_pct': close / close.rolling(252).max(),
        'mom_accel': close.pct_change(63) - close.pct_change(63).shift(63),

        # Volatility/quality features (same as v1)
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

    # NEW: Regime features
    # Relative to SMA
    feat['above_sma50'] = (close > close.rolling(50).mean()).astype(int)
    feat['above_sma200'] = (close > close.rolling(200).mean()).astype(int)
    feat['dist_sma200'] = (close - close.rolling(200).mean()) / close.rolling(200).mean()

    # Own realized vol regime
    feat['rv_21d'] = lr.rolling(21).std() * np.sqrt(252)
    feat['rv_ratio_short_long'] = feat['rv_21d'] / (feat['vol_60d'] + 1e-8)

    # Cross-asset regime features (from SPY if available)
    if spy_close is not None:
        spy_lr = np.log(spy_close / spy_close.shift(1))
        feat['spy_ret_21d'] = spy_close.pct_change(21)
        feat['spy_ret_63d'] = spy_close.pct_change(63)
        feat['spy_above_sma200'] = (spy_close > spy_close.rolling(200).mean()).astype(int)
        feat['spy_rv_21d'] = spy_lr.rolling(21).std() * np.sqrt(252)
        feat['spy_dist_sma200'] = (spy_close - spy_close.rolling(200).mean()) / spy_close.rolling(200).mean()

        # Correlation with SPY (rolling 63d)
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


def run_variant(all_data, spy_data, spy_close, top_k, defensive_shift_factor=0.0,
                cost_bps=10, name='base'):
    """
    Run walk-forward LightGBM with optional post-prediction defensive shift.

    defensive_shift_factor: 0 = no shift, 1.0 = boost defensives by 50% in bear, etc.
    """
    import lightgbm as lgb

    # Build feature matrix with regime features
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

    fprint(f"  [{name}] Features: {X.shape[0]} x {X.shape[1]}, dates: {dates[0].date()} to {dates[-1].date()}")

    # Walk-forward
    monthly_picks = []
    monthly_returns = []
    fold = 0
    i = 252
    feat_imp_accum = None

    while i + 21 <= len(dates):
        fold += 1
        train_dates = dates[i-252:i]
        test_dates = dates[i:i+21]

        train_mask = meta['date'].isin(train_dates)
        test_mask = meta['date'].isin(test_dates)

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te = X[test_mask]
        meta_te = meta[test_mask].copy()

        if len(X_tr) < 50 or len(X_te) < 5:
            i += 21
            continue

        model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            subsample=0.8, verbose=-1, n_jobs=-1
        )
        model.fit(X_tr, y_tr)

        if feat_imp_accum is None:
            feat_imp_accum = model.feature_importances_.astype(float)
        else:
            feat_imp_accum += model.feature_importances_.astype(float)

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

        # Calculate actual returns (equal weight)
        ret = 0
        picks = []
        for _, row in top.iterrows():
            t = row['ticker']
            if t in all_data:
                tc = all_data[t]['Close']
                si = tc.index.searchsorted(test_dates[0])
                ei = tc.index.searchsorted(test_dates[-1])
                if si < len(tc) and ei < len(tc) and tc.iloc[si] > 0:
                    stock_ret = tc.iloc[ei] / tc.iloc[si] - 1
                    ret += stock_ret / top_k
                    picks.append(t)

        ret -= cost_bps / 10000 * 2  # buy + sell cost

        monthly_picks.append({'date': str(td.date()), 'picks': picks, 'return': float(ret), 'regime': regime})
        monthly_returns.append(float(ret))
        i += 21

    # SPY benchmark
    spy_returns = []
    i = 252
    while i + 21 <= len(dates):
        test_dates_b = dates[i:i+21]
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
        'n_folds': len(monthly_returns),
        'feature_importances': feat_imp_accum / max(fold, 1) if feat_imp_accum is not None else None,
        'name': name,
    }


def compute_metrics(returns, name):
    """Compute risk-adjusted metrics."""
    r = np.array(returns)
    if len(r) < 2:
        return {'name': name}

    equity = 100000 * np.cumprod(1 + r)
    ppy = 12
    sharpe = np.mean(r) / np.std(r) * np.sqrt(ppy) if np.std(r) > 0 else 0
    ds = r[r < 0]
    sortino = np.mean(r) / np.std(ds) * np.sqrt(ppy) if len(ds) > 0 and np.std(ds) > 0 else 0
    years = len(r) / ppy
    cagr = ((equity[-1] / 100000) ** (1/max(years, 0.01)) - 1) * 100
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
    """Run all 4 adversarial gates with regime-stratified analysis."""
    r = np.array(returns)
    sp = np.array(spy_returns[:len(r)])

    gates = {}

    # 1. Permutation test
    fprint("  Running permutation test (200 shuffles)...")
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
    fprint(f"    Permutation: p={perm_p:.3f} {'PASS' if perm_p < 0.05 else 'FAIL'}")

    # 2. Regime test (R1) — using SPY monthly returns as regime classifier
    fprint("  Running regime test...")
    bull_mask = sp > 0
    bear_mask = sp <= 0

    # Also compute using the regime labels from picks
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
            'pass': gap < 0.50,
        }
        fprint(f"    R1 (SPY ret): bull={bull_sharpe:.2f}, bear={bear_sharpe:.2f}, gap={gap:.3f} {'PASS' if gap < 0.50 else 'FAIL'}")
    else:
        gates['regime_r1'] = {'pass': None, 'note': 'Insufficient regime data'}

    # Also report SMA200-based regime gap
    if regime_bull_mask.sum() >= 3 and regime_bear_mask.sum() >= 3:
        br = r[regime_bull_mask]
        brr = r[regime_bear_mask]
        bs = np.mean(br) / np.std(br) * np.sqrt(12) if np.std(br) > 0 else 0
        brs = np.mean(brr) / np.std(brr) * np.sqrt(12) if np.std(brr) > 0 else 0
        ms = max(abs(bs), abs(brs))
        g2 = abs(bs - brs) / ms if ms > 0 else 0
        gates['regime_sma200'] = {
            'bull_sharpe': round(float(bs), 2),
            'bear_sharpe': round(float(brs), 2),
            'gap': round(float(g2), 3),
            'pass': g2 < 0.50,
            'bull_months': int(regime_bull_mask.sum()),
            'bear_months': int(regime_bear_mask.sum()),
        }
        fprint(f"    R1 (SMA200): bull={bs:.2f}, bear={brs:.2f}, gap={g2:.3f} {'PASS' if g2 < 0.50 else 'FAIL'}")

    # 3. Sub-period test
    mid = len(r) // 2
    h1 = r[:mid]
    h2 = r[mid:]
    h1_sharpe = np.mean(h1) / np.std(h1) * np.sqrt(12) if np.std(h1) > 0 else 0
    h2_sharpe = np.mean(h2) / np.std(h2) * np.sqrt(12) if np.std(h2) > 0 else 0
    sub_pass = h1_sharpe > 0 and h2_sharpe > 0
    gates['sub_period'] = {
        'h1_sharpe': round(float(h1_sharpe), 2),
        'h2_sharpe': round(float(h2_sharpe), 2),
        'pass': sub_pass,
    }
    fprint(f"    Sub-period: H1={h1_sharpe:.2f}, H2={h2_sharpe:.2f} {'PASS' if sub_pass else 'FAIL'}")

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
    fprint(f"    Outlier: Sharpe after removing top {n_remove} months = {trim_sharpe:.2f} {'PASS' if outlier_pass else 'FAIL'}")

    return gates


def main():
    import yfinance as yf
    try:
        import lightgbm as lgb
    except ImportError:
        fprint("ERROR: LightGBM not available")
        return None

    fprint("=" * 70)
    fprint("SECTOR ETF MOMENTUM v2 — LightGBM + REGIME-ADAPTIVE DEFENSIVE SHIFT")
    fprint("=" * 70)
    fprint(f"Universe: {len(UNIVERSE)} ETFs | Top K: 3 and 5")
    fprint(f"NEW: regime features (SPY-relative) + post-prediction defensive shift")
    fprint(f"Goal: Fix R1 gap (was 0.784) while maintaining Sharpe ~3.96\n")

    # Download data
    fprint("Downloading ETFs...")
    all_data = {}
    for t in UNIVERSE:
        try:
            df = yf.download(t, start='2008-01-01', end='2026-07-25', progress=False)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 252:
                all_data[t] = df
        except:
            pass

    spy_data = yf.download('SPY', start='2008-01-01', end='2026-07-25', progress=False)
    if spy_data.index.tz is not None:
        spy_data.index = spy_data.index.tz_convert(None)
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.get_level_values(0)
    spy_close = spy_data['Close']

    fprint(f"  Loaded {len(all_data)} ETFs + SPY ({len(spy_data)} days)")

    # Variants to test
    variants = [
        # (name, top_k, defensive_shift_factor)
        ('A_Base_Top3', 3, 0.0),            # Baseline (should match v1)
        ('B_Base_Top5', 5, 0.0),            # More diversified
        ('C_DefShift_Top3', 3, 1.0),        # Defensive shift, top 3
        ('D_DefShift_Top5', 5, 1.0),        # Defensive shift, top 5 (regime-adaptive winner)
        ('E_StrongShift_Top3', 3, 1.5),     # Stronger defensive shift
        ('F_StrongShift_Top5', 5, 1.5),     # Strong + diversified
        ('G_MildShift_Top5', 5, 0.5),       # Mild shift
        ('H_VeryStrong_Top5', 5, 2.0),      # Very strong shift (risk of over-correction)
    ]

    results = []
    for name, tk, dsf in variants:
        fprint(f"\n--- {name} (Top {tk}, shift={dsf}) ---")
        result = run_variant(all_data, spy_data, spy_close, top_k=tk,
                           defensive_shift_factor=dsf, cost_bps=10, name=name)
        if result is None or len(result['monthly_returns']) == 0:
            fprint("  No results")
            continue

        m = compute_metrics(result['monthly_returns'], name)
        result['metrics'] = m
        results.append(result)

        fprint(f"  Sharpe: {m['sharpe']}, CAGR: {m['cagr']}%, MaxDD: {m['maxdd']}%, "
               f"WR: {m['wr']}%, PF: {m['pf']}")

    if not results:
        fprint("NO RESULTS")
        return

    # Quick R1 screening for all variants
    fprint(f"\n{'='*70}")
    fprint("VARIANT COMPARISON")
    fprint(f"{'='*70}")
    fprint(f"{'Variant':<22} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'WR':>6} {'PF':>6}")
    fprint("-" * 60)

    for result in sorted(results, key=lambda x: x['metrics']['sharpe'], reverse=True):
        m = result['metrics']
        fprint(f"{result['name']:<22} {m['sharpe']:>7.2f} {m['cagr']:>7.1f}% "
               f"{m['maxdd']:>7.1f}% {m['wr']:>5.1f}% {m['pf']:>6.2f}")

    # Run full adversarial gates on ALL variants
    fprint(f"\n{'='*70}")
    fprint("ADVERSARIAL GATES — ALL VARIANTS")
    fprint(f"{'='*70}")

    best_r1_passing = None
    gate_results = {}

    for result in results:
        name = result['name']
        fprint(f"\n--- {name} ---")
        gates = adversarial_gates(
            result['monthly_returns'],
            result['spy_returns'],
            result['monthly_picks']
        )
        gate_results[name] = gates

        n_pass = sum(1 for g in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                     if gates.get(g, {}).get('pass') is True)
        n_total = sum(1 for g in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                      if gates.get(g, {}).get('pass') is not None)
        fprint(f"  GATES: {n_pass}/{n_total}")

        r1_pass = gates.get('regime_r1', {}).get('pass', False)
        if r1_pass and n_pass >= 3:
            if best_r1_passing is None or result['metrics']['sharpe'] > best_r1_passing['metrics']['sharpe']:
                best_r1_passing = result
                best_r1_passing['gates'] = gates
                best_r1_passing['n_pass'] = n_pass

    # Summary table with gates
    fprint(f"\n{'='*70}")
    fprint("FINAL COMPARISON")
    fprint(f"{'='*70}")
    fprint(f"{'Variant':<22} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'R1gap':>7} {'R1':>5} {'Gates':>6}")
    fprint("-" * 70)

    for result in sorted(results, key=lambda x: x['metrics']['sharpe'], reverse=True):
        m = result['metrics']
        g = gate_results[result['name']]
        r1_gap = g.get('regime_r1', {}).get('gap', -1)
        r1_pass = g.get('regime_r1', {}).get('pass', False)
        n_p = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                  if g.get(k, {}).get('pass') is True)
        n_t = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                  if g.get(k, {}).get('pass') is not None)
        r1_str = "PASS" if r1_pass else "FAIL"
        fprint(f"{result['name']:<22} {m['sharpe']:>7.2f} {m['cagr']:>7.1f}% "
               f"{m['maxdd']:>7.1f}% {r1_gap:>7.3f} {r1_str:>5} {n_p}/{n_t}")

    # Report best
    if best_r1_passing:
        winner = best_r1_passing
        fprint(f"\n{'='*70}")
        fprint(f"WINNER: {winner['name']} — ALL R1-CRITICAL GATES PASS")
        fprint(f"{'='*70}")
    else:
        winner = max(results, key=lambda x: x['metrics']['sharpe'])
        fprint(f"\n{'='*70}")
        fprint(f"NO R1 PASSERS — Best overall: {winner['name']}")
        fprint(f"{'='*70}")

    m = winner['metrics']
    g = gate_results[winner['name']]
    fprint(f"  Sharpe:      {m['sharpe']}")
    fprint(f"  Sortino:     {m['sortino']}")
    fprint(f"  CAGR:        {m['cagr']}%")
    fprint(f"  MaxDD:       {m['maxdd']}%")
    fprint(f"  WR:          {m['wr']}%")
    fprint(f"  PF:          {m['pf']}")
    fprint(f"  Calmar:      {m['calmar']}")
    r1 = g.get('regime_r1', {})
    fprint(f"  Bull Sharpe: {r1.get('bull_sharpe', '?')}")
    fprint(f"  Bear Sharpe: {r1.get('bear_sharpe', '?')}")
    fprint(f"  R1 Gap:      {r1.get('gap', '?')}")

    # Feature importance from winner
    feat_names = [
        'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
        'mom_12_1', 'high_52w_pct', 'mom_accel', 'vol_20d', 'vol_60d',
        'vol_ratio', 'sharpe_63d', 'sharpe_126d', 'maxdd_63d', 'vol_rel',
        'skew_63d', 'kurt_63d', 'vol_trend',
        'above_sma50', 'above_sma200', 'dist_sma200',
        'rv_21d', 'rv_ratio_short_long',
        'spy_ret_21d', 'spy_ret_63d', 'spy_above_sma200', 'spy_rv_21d',
        'spy_dist_sma200', 'corr_spy_63d', 'beta_spy_63d',
    ]
    if winner.get('feature_importances') is not None:
        imp = winner['feature_importances']
        if len(imp) == len(feat_names):
            importance = dict(zip(feat_names, imp.tolist()))
            fprint(f"\nTop features:")
            for fn, iv in sorted(importance.items(), key=lambda x: -x[1])[:10]:
                fprint(f"  {fn}: {iv:.1f}")

            # Separate regime feature importance
            regime_feats = ['above_sma50', 'above_sma200', 'dist_sma200', 'rv_21d',
                           'rv_ratio_short_long', 'spy_ret_21d', 'spy_ret_63d',
                           'spy_above_sma200', 'spy_rv_21d', 'spy_dist_sma200',
                           'corr_spy_63d', 'beta_spy_63d']
            regime_imp = sum(importance.get(f, 0) for f in regime_feats)
            total_imp = sum(importance.values())
            fprint(f"\n  Regime features importance: {regime_imp/total_imp*100:.1f}% of total")

    # Year-by-year breakdown
    fprint(f"\nYear-by-year returns ({winner['name']}):")
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
        fprint(f"  {yr}: {yr_ret:>7.1f}%  Sharpe {yr_sharpe:.2f}  ({len(rets)} months)")

    # Most selected ETFs
    pick_counts = {}
    for p in winner['monthly_picks']:
        for t in p['picks']:
            pick_counts[t] = pick_counts.get(t, 0) + 1
    fprint(f"\nMost selected ETFs:")
    for t, c in sorted(pick_counts.items(), key=lambda x: -x[1])[:8]:
        fprint(f"  {t}: {c} times ({c/winner['n_folds']*100:.0f}%)")

    # Save results
    output = {
        'strategy': 'Sector ETF Momentum v2 — LightGBM + Regime-Adaptive Defensive Shift',
        'run_date': str(datetime.now()),
        'winner': winner['name'],
        'winner_metrics': winner['metrics'],
        'winner_gates': {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                             for kk, vv in v.items()}
                        for k, v in gate_results[winner['name']].items()},
        'all_variants': [{
            'name': r['name'],
            'metrics': r['metrics'],
            'gates': {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                          for kk, vv in v.items()}
                     for k, v in gate_results[r['name']].items()},
        } for r in results],
        'most_selected': dict(sorted(pick_counts.items(), key=lambda x: -x[1])[:10]),
        'monthly_picks': winner['monthly_picks'],
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved.")

    # MLflow
    if MLFLOW_OK:
        try:
            mlflow.set_experiment('sector_etf_momentum_v2_regime')
            with mlflow.start_run(run_name=f'v2_regime_{winner["name"]}'):
                wm = winner['metrics']
                wg = gate_results[winner['name']]
                mlflow.log_metrics({
                    'sharpe': wm.get('sharpe', 0),
                    'sortino': wm.get('sortino', 0),
                    'cagr': wm.get('cagr', 0),
                    'maxdd': wm.get('maxdd', 0),
                    'wr': wm.get('wr', 0),
                    'pf': wm.get('pf', 0),
                    'calmar': wm.get('calmar', 0),
                    'perm_p': wg.get('permutation', {}).get('p_value', -1),
                    'r1_gap': wg.get('regime_r1', {}).get('gap', -1),
                    'n_pass': sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                                  if wg.get(k, {}).get('pass') is True),
                })
                mlflow.log_params({
                    'winner': winner['name'],
                    'universe_size': len(UNIVERSE),
                    'regime_features': True,
                    'n_variants': len(results),
                })
        except Exception as e:
            fprint(f"MLflow error: {e}")

    fprint(f"\n{'='*70}")
    fprint("DONE")
    fprint(f"{'='*70}")

    return output


if __name__ == '__main__':
    main()
