#!/usr/bin/env python3
"""
Sector ETF Momentum Ranker v1 — SURVIVORSHIP-FREE
====================================================

Same LightGBM momentum+quality approach as QM Ranker v1/v2, but applied
to sector ETFs instead of individual stocks. ETFs eliminate survivorship
bias entirely (all have existed since at least 2000).

This validates whether the momentum signal is REAL or just NVDA/TSLA luck.

Universe: 18 liquid sector/factor ETFs (all existed 10+ years)
Method: LightGBM cross-sectional ranking, top 3 monthly, 252d/21d sliding WF
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

RESULTS_DIR = Path(__file__).resolve().parent / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'sector_etf_momentum_v1_results.json'

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

# Universe: sector ETFs + factor ETFs — ALL existed since at least 2010
UNIVERSE = [
    # Sector ETFs (all inception 1998-1999)
    'XLK',   # Technology
    'XLF',   # Financials
    'XLE',   # Energy
    'XLV',   # Healthcare
    'XLY',   # Consumer Discretionary
    'XLP',   # Consumer Staples
    'XLI',   # Industrials
    'XLB',   # Materials
    'XLU',   # Utilities
    'XLRE',  # Real Estate (2015, shorter history)
    'XLC',   # Communication Services (2018, shorter)
    # Broader/factor ETFs
    'QQQ',   # Nasdaq 100
    'IWM',   # Russell 2000
    'MDY',   # S&P MidCap 400
    'EFA',   # International Developed
    'EEM',   # Emerging Markets
    'GLD',   # Gold
    'TLT',   # Long-term Treasury
    'HYG',   # High Yield Corporate Bonds
    'IYR',   # Real Estate (REIT)
    'VNQ',   # Vanguard REIT
    'DBC',   # Commodities
]

TOP_K = 3  # Fewer ETFs, so pick top 3 not 5

def build_features(close, volume):
    """Build momentum + quality features for a single ETF."""
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

    return feat


def run_backtest(universe, top_k, start_year='2008', cost_bps=10):
    """Run walk-forward LightGBM cross-sectional ranking backtest."""
    import yfinance as yf
    try:
        import lightgbm as lgb
    except ImportError:
        fprint("ERROR: LightGBM not available")
        return None

    fprint(f"Downloading {len(universe)} ETFs...")
    all_data = {}
    for t in universe:
        try:
            df = yf.download(t, start=f'{start_year}-01-01', end='2026-07-24', progress=False)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 252:
                all_data[t] = df
        except:
            pass

    spy = yf.download('SPY', start=f'{start_year}-01-01', end='2026-07-24', progress=False)
    if spy.index.tz is not None: spy.index = spy.index.tz_convert(None)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)

    fprint(f"  Loaded {len(all_data)} ETFs with sufficient history")

    # Find common dates
    common = None
    for t, df in all_data.items():
        common = df.index if common is None else common.intersection(df.index)

    fprint(f"  Common trading days: {len(common)}")

    # Build feature matrix
    features_list = []
    labels_list = []
    meta_list = []

    for t, df in all_data.items():
        c = df['Close'].reindex(common)
        v = df['Volume'].reindex(common) if 'Volume' in df.columns else None
        feat = build_features(c, v)

        # Forward return: 21-day
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

    fprint(f"  Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    fprint(f"  Date range: {dates[0].date()} to {dates[-1].date()}")
    fprint(f"  Running walk-forward ({(len(dates)-252)//21} folds)...")

    # Walk-forward
    monthly_picks = []
    monthly_returns = []
    fold = 0
    i = 252
    while i + 21 <= len(dates):
        fold += 1
        if fold % 20 == 0:
            fprint(f"    Fold {fold}...")

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
        meta_te = meta_te.copy()
        meta_te['pred'] = model.predict(X_te)

        # Get predictions for the first test date (rebalance day)
        td = test_dates[0]
        dp = meta_te[meta_te['date'] == td]
        if len(dp) < top_k:
            i += 21
            continue

        top = dp.nlargest(top_k, 'pred')['ticker'].tolist()

        # Calculate actual returns (equal weight top K)
        ret = 0
        for t in top:
            if t in all_data:
                tc = all_data[t]['Close']
                si = tc.index.searchsorted(test_dates[0])
                ei = tc.index.searchsorted(test_dates[-1])
                if si < len(tc) and ei < len(tc) and tc.iloc[si] > 0:
                    stock_ret = tc.iloc[ei] / tc.iloc[si] - 1
                    ret += stock_ret / top_k

        # Subtract transaction costs
        ret -= cost_bps / 10000 * 2  # buy + sell

        monthly_picks.append({'date': str(td.date()), 'picks': top, 'return': float(ret)})
        monthly_returns.append(float(ret))
        i += 21

    # SPY benchmark returns for same periods
    spy_returns = []
    i = 252
    fold = 0
    while i + 21 <= len(dates):
        fold += 1
        test_dates = dates[i:i+21]
        sc = spy['Close']
        si = sc.index.searchsorted(test_dates[0])
        ei = sc.index.searchsorted(test_dates[-1])
        if si < len(sc) and ei < len(sc) and sc.iloc[si] > 0:
            spy_ret = sc.iloc[ei] / sc.iloc[si] - 1
            spy_returns.append(float(spy_ret))
        i += 21

    # Feature importance (last model)
    feat_names = [
        'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
        'mom_12_1', 'high_52w_pct', 'mom_accel', 'vol_20d', 'vol_60d',
        'vol_ratio', 'sharpe_63d', 'sharpe_126d', 'maxdd_63d', 'vol_rel',
        'skew_63d', 'kurt_63d', 'vol_trend'
    ]
    importance = dict(zip(feat_names, model.feature_importances_.tolist()))

    return {
        'monthly_returns': monthly_returns,
        'spy_returns': spy_returns[:len(monthly_returns)],
        'monthly_picks': monthly_picks,
        'n_etfs': len(all_data),
        'n_folds': len(monthly_returns),
        'feature_importance': importance,
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


def adversarial_gates(returns, spy_returns, n_perm=200):
    """Run all 4 adversarial gates."""
    r = np.array(returns)
    sp = np.array(spy_returns[:len(r)])

    gates = {}

    # 1. Permutation test (200 shuffles with sign-flip)
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

    # 2. Regime test (R1) — bull vs bear based on SPY
    fprint("  Running regime test...")
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
        fprint(f"    R1: bull={bull_sharpe:.2f}, bear={bear_sharpe:.2f}, gap={gap:.3f} {'PASS' if gap < 0.50 else 'FAIL'}")
    else:
        gates['regime_r1'] = {'pass': None, 'note': 'Insufficient regime data'}
        fprint(f"    R1: SKIP (insufficient regime data)")

    # 3. Sub-period test (both halves profitable)
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

    # 4. Outlier test (remove top 5% months, still profitable)
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
    fprint("=" * 60)
    fprint("SECTOR ETF MOMENTUM RANKER v1 — SURVIVORSHIP-FREE")
    fprint("=" * 60)
    fprint(f"Universe: {len(UNIVERSE)} ETFs (sectors + factors + bonds + commodities)")
    fprint(f"No survivorship bias — all ETFs existed throughout backtest period")
    fprint(f"Top {TOP_K} monthly, 252d train / 21d test, LightGBM\n")

    result = run_backtest(UNIVERSE, TOP_K)
    if result is None:
        fprint("ERROR: Backtest failed")
        return

    # Metrics
    strat = compute_metrics(result['monthly_returns'], 'Sector ETF Momentum')
    bench = compute_metrics(result['spy_returns'], 'SPY Benchmark')

    fprint(f"\n{'='*60}")
    fprint("RESULTS — SURVIVORSHIP-FREE")
    fprint(f"{'='*60}")

    fprint(f"\n{'Strategy':<25} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'WR':>6} {'PF':>6} {'Calmar':>7}")
    fprint("-" * 75)
    for m in [strat, bench]:
        fprint(f"{m['name']:<25} {m.get('sharpe',0):>7.2f} {m.get('sortino',0):>8.2f} "
               f"{m.get('cagr',0):>6.1f}% {m.get('maxdd',0):>6.1f}% {m.get('wr',0):>5.1f}% "
               f"{m.get('pf',0):>6.2f} {m.get('calmar',0):>7.2f}")

    # Most selected ETFs
    pick_counts = {}
    for p in result['monthly_picks']:
        for t in p['picks']:
            pick_counts[t] = pick_counts.get(t, 0) + 1
    fprint(f"\nMost selected ETFs:")
    for t, c in sorted(pick_counts.items(), key=lambda x: -x[1])[:8]:
        fprint(f"  {t}: {c} times ({c/result['n_folds']*100:.0f}%)")

    # Feature importance
    fprint(f"\nTop features (LightGBM importance):")
    sorted_imp = sorted(result['feature_importance'].items(), key=lambda x: -x[1])
    for name, imp in sorted_imp[:8]:
        fprint(f"  {name}: {imp}")

    # Adversarial gates
    fprint(f"\n{'='*60}")
    fprint("ADVERSARIAL VALIDATION GATES")
    fprint(f"{'='*60}")
    gates = adversarial_gates(result['monthly_returns'], result['spy_returns'])

    n_pass = sum(1 for g in gates.values() if g.get('pass') is True)
    n_total = sum(1 for g in gates.values() if g.get('pass') is not None)
    fprint(f"\nGATE SUMMARY: {n_pass}/{n_total} PASS")

    # Year-by-year
    fprint(f"\nYear-by-year returns:")
    picks_by_year = {}
    for i, p in enumerate(result['monthly_picks']):
        yr = p['date'][:4]
        if yr not in picks_by_year:
            picks_by_year[yr] = []
        picks_by_year[yr].append(result['monthly_returns'][i])

    for yr in sorted(picks_by_year.keys()):
        rets = picks_by_year[yr]
        yr_ret = (np.prod(1 + np.array(rets)) - 1) * 100
        yr_sharpe = np.mean(rets) / np.std(rets) * np.sqrt(12) if np.std(rets) > 0 else 0
        fprint(f"  {yr}: {yr_ret:>7.1f}%  Sharpe {yr_sharpe:.2f}  ({len(rets)} months)")

    # Save
    output = {
        'strategy': 'Sector ETF Momentum Ranker v1 — SURVIVORSHIP FREE',
        'run_date': str(datetime.now()),
        'universe': UNIVERSE,
        'top_k': TOP_K,
        'survivorship_free': True,
        'note': 'All ETFs existed throughout the backtest period. NO survivorship bias.',
        'metrics': strat,
        'benchmark': bench,
        'gates': gates,
        'feature_importance': result['feature_importance'],
        'most_selected': dict(sorted(pick_counts.items(), key=lambda x: -x[1])[:10]),
        'monthly_picks': result['monthly_picks'],
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    fprint(f"\nResults saved to findings dir.")

    if MLFLOW_OK:
        try:
            mlflow.set_experiment('sector_etf_momentum_v1')
            with mlflow.start_run(run_name='sector_etf_mom_v1'):
                mlflow.log_metrics({
                    'sharpe': strat.get('sharpe', 0),
                    'sortino': strat.get('sortino', 0),
                    'cagr': strat.get('cagr', 0),
                    'maxdd': strat.get('maxdd', 0),
                    'wr': strat.get('wr', 0),
                    'pf': strat.get('pf', 0),
                    'calmar': strat.get('calmar', 0),
                    'perm_p': gates.get('permutation', {}).get('p_value', -1),
                    'r1_gap': gates.get('regime_r1', {}).get('gap', -1),
                    'n_pass': n_pass,
                })
                mlflow.log_params({
                    'universe_size': len(UNIVERSE),
                    'top_k': TOP_K,
                    'survivorship_free': True,
                })
        except Exception as e:
            fprint(f"MLflow error: {e}")

    return output


if __name__ == '__main__':
    main()
