#!/usr/bin/env python3
"""
Sector ETF Momentum v2 — REGIME-HEDGED
========================================

v1 failed R1 (bull Sharpe 6.06 vs bear 1.31, gap 0.784).
Fix: add regime detection and defensive rotation during bear markets.

Approach:
- Same LightGBM cross-sectional ranking on 22 ETFs
- Add regime overlay: when SPY is in bear regime (below 200d SMA or
  trailing 63d return < -5%), shift allocation toward defensive assets
- Test multiple hedge variants:
  A) Regime-aware feature addition (let model learn regimes)
  B) Position sizing overlay (reduce risk in bear, increase in bull)
  C) Defensive rotation (force TLT/GLD allocation in bear regimes)
  D) Combined: regime features + defensive overlay
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def fprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

RESULTS_DIR = Path(__file__).resolve().parent / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'sector_etf_momentum_v2_results.json'

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
except:
    pass

UNIVERSE = [
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU',
    'XLRE', 'XLC',
    'QQQ', 'IWM', 'MDY', 'EFA', 'EEM',
    'GLD', 'TLT', 'HYG', 'IYR', 'VNQ', 'DBC',
]

DEFENSIVE_ETFS = {'TLT', 'GLD', 'XLP', 'XLU', 'XLV'}  # Safe havens
TOP_K = 3


def build_features(close, volume, spy_close=None):
    """Build features including regime indicators."""
    lr = np.log(close / close.shift(1))

    feat = pd.DataFrame({
        # Core momentum
        'ret_5d': close.pct_change(5),
        'ret_10d': close.pct_change(10),
        'ret_21d': close.pct_change(21),
        'ret_63d': close.pct_change(63),
        'ret_126d': close.pct_change(126),
        'ret_252d': close.pct_change(252),
        'mom_12_1': close.pct_change(252) - close.pct_change(21),
        'high_52w_pct': close / close.rolling(252).max(),
        'mom_accel': close.pct_change(63) - close.pct_change(63).shift(63),

        # Quality / vol
        'vol_20d': lr.rolling(20).std() * np.sqrt(252),
        'vol_60d': lr.rolling(60).std() * np.sqrt(252),
        'vol_ratio': lr.rolling(20).std() / lr.rolling(60).std(),
        'sharpe_63d': lr.rolling(63).mean() / lr.rolling(63).std(),
        'sharpe_126d': lr.rolling(126).mean() / lr.rolling(126).std(),
        'maxdd_63d': (close / close.rolling(63).max() - 1).rolling(63).min(),
        'vol_rel': volume / volume.rolling(20).mean() if volume is not None else 0,
        'skew_63d': lr.rolling(63).skew(),
        'kurt_63d': lr.rolling(63).kurt(),
        'vol_trend': (lr.rolling(20).std() - lr.rolling(60).std()) / lr.rolling(60).std(),
    }, index=close.index)

    # Regime features (from SPY)
    if spy_close is not None:
        spy_lr = np.log(spy_close / spy_close.shift(1))
        spy_reindexed = spy_close.reindex(close.index, method='ffill')
        spy_lr_reindexed = spy_lr.reindex(close.index, method='ffill')

        feat['spy_above_200sma'] = (spy_reindexed / spy_reindexed.rolling(200).mean() - 1)
        feat['spy_ret_63d'] = spy_reindexed.pct_change(63)
        feat['spy_vol_20d'] = spy_lr_reindexed.rolling(20).std() * np.sqrt(252)
        feat['spy_vol_ratio'] = spy_lr_reindexed.rolling(20).std() / spy_lr_reindexed.rolling(60).std()
        feat['spy_drawdown'] = spy_reindexed / spy_reindexed.rolling(252).max() - 1

        # Relative momentum (ETF vs SPY)
        feat['rel_mom_21d'] = close.pct_change(21) - spy_reindexed.pct_change(21)
        feat['rel_mom_63d'] = close.pct_change(63) - spy_reindexed.pct_change(63)

    return feat


def detect_regime(spy_close, date):
    """Detect market regime at a given date."""
    idx = spy_close.index.searchsorted(date)
    if idx < 200:
        return 'neutral'

    price = spy_close.iloc[idx]
    sma200 = spy_close.iloc[max(0, idx-200):idx+1].mean()
    ret_63d = price / spy_close.iloc[max(0, idx-63)] - 1 if idx >= 63 else 0

    if price < sma200 and ret_63d < -0.05:
        return 'bear'
    elif price > sma200 and ret_63d > 0.05:
        return 'bull'
    else:
        return 'neutral'


def run_variant(all_data, spy, common, variant_name, top_k=3, use_regime_features=True,
                defensive_overlay=False, position_sizing=False):
    """Run a single variant of the strategy."""
    import lightgbm as lgb

    spy_close = spy['Close']

    # Build feature matrix
    features_list = []
    labels_list = []
    meta_list = []
    n_features = None

    for t, df in all_data.items():
        c = df['Close'].reindex(common)
        v = df['Volume'].reindex(common) if 'Volume' in df.columns else None

        if use_regime_features:
            feat = build_features(c, v, spy_close)
        else:
            feat = build_features(c, v, None)

        if n_features is None:
            n_features = len(feat.columns)

        fwd = c.pct_change(21).shift(-21)
        valid = feat.dropna().index.intersection(fwd.dropna().index)

        for d in valid:
            row = feat.loc[d].values
            if len(row) == n_features and not np.any(np.isnan(row)) and not np.any(np.isinf(row)):
                features_list.append(row)
                labels_list.append(fwd.loc[d])
                meta_list.append({'date': d, 'ticker': t})

    X = np.array(features_list)
    y = np.array(labels_list)
    meta = pd.DataFrame(meta_list)
    dates = sorted(meta['date'].unique())

    monthly_returns = []
    monthly_picks_list = []
    spy_returns = []

    fold = 0
    i = 252
    while i + 21 <= len(dates):
        fold += 1
        if fold % 20 == 0:
            fprint(f"    [{variant_name}] Fold {fold}...")

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

        td = test_dates[0]
        dp = meta_te[meta_te['date'] == td]
        if len(dp) < top_k:
            i += 21
            continue

        # Detect regime
        regime = detect_regime(spy_close, td)

        # Select top K
        if defensive_overlay and regime == 'bear':
            # In bear: force at least 1 defensive pick, boost defensive predictions
            dp_copy = dp.copy()
            for idx_row in dp_copy.index:
                if dp_copy.loc[idx_row, 'ticker'] in DEFENSIVE_ETFS:
                    dp_copy.loc[idx_row, 'pred'] += 0.02  # boost defensive by 2%
            top = dp_copy.nlargest(top_k, 'pred')['ticker'].tolist()
        else:
            top = dp.nlargest(top_k, 'pred')['ticker'].tolist()

        # Calculate returns
        ret = 0
        for t in top:
            if t in all_data:
                tc = all_data[t]['Close']
                si = tc.index.searchsorted(test_dates[0])
                ei = tc.index.searchsorted(test_dates[-1])
                if si < len(tc) and ei < len(tc) and tc.iloc[si] > 0:
                    stock_ret = tc.iloc[ei] / tc.iloc[si] - 1
                    ret += stock_ret / top_k

        # Position sizing based on regime
        if position_sizing:
            if regime == 'bear':
                ret *= 0.5  # Half position in bear
            elif regime == 'neutral':
                ret *= 0.75

        # Transaction costs
        ret -= 10 / 10000 * 2  # 10bps buy + sell

        monthly_returns.append(float(ret))
        monthly_picks_list.append({'date': str(td.date()), 'picks': top, 'regime': regime})

        # SPY benchmark
        sc = spy['Close']
        si = sc.index.searchsorted(test_dates[0])
        ei = sc.index.searchsorted(test_dates[-1])
        if si < len(sc) and ei < len(sc) and sc.iloc[si] > 0:
            spy_returns.append(float(sc.iloc[ei] / sc.iloc[si] - 1))

        i += 21

    return monthly_returns, spy_returns, monthly_picks_list


def compute_metrics(returns, name):
    """Compute risk-adjusted metrics."""
    r = np.array(returns)
    if len(r) < 2:
        return {'name': name, 'sharpe': 0}

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
    }


def adversarial_gates(returns, spy_returns, name=""):
    """Run all 4 adversarial gates."""
    r = np.array(returns)
    sp = np.array(spy_returns[:len(r)])
    gates = {}

    # 1. Permutation
    real_sharpe = np.mean(r) / np.std(r) * np.sqrt(12) if np.std(r) > 0 else 0
    perm_count = 0
    for _ in range(200):
        signs = np.random.choice([-1, 1], size=len(r))
        perm_r = r * signs
        ps = np.mean(perm_r) / np.std(perm_r) * np.sqrt(12) if np.std(perm_r) > 0 else 0
        if ps >= real_sharpe:
            perm_count += 1
    perm_p = perm_count / 200
    gates['permutation'] = {'p_value': round(float(perm_p), 3), 'pass': perm_p < 0.05}

    # 2. Regime R1
    bull_mask = sp > 0
    bear_mask = sp <= 0
    if bull_mask.sum() >= 6 and bear_mask.sum() >= 6:
        bull_r, bear_r = r[bull_mask], r[bear_mask]
        bull_s = np.mean(bull_r) / np.std(bull_r) * np.sqrt(12) if np.std(bull_r) > 0 else 0
        bear_s = np.mean(bear_r) / np.std(bear_r) * np.sqrt(12) if np.std(bear_r) > 0 else 0
        max_s = max(abs(bull_s), abs(bear_s))
        gap = abs(bull_s - bear_s) / max_s if max_s > 0 else 0
        gates['regime_r1'] = {
            'bull_sharpe': round(float(bull_s), 2), 'bear_sharpe': round(float(bear_s), 2),
            'gap': round(float(gap), 3), 'pass': gap < 0.50
        }
    else:
        gates['regime_r1'] = {'pass': None, 'note': 'Insufficient data'}

    # 3. Sub-period
    mid = len(r) // 2
    h1_s = np.mean(r[:mid]) / np.std(r[:mid]) * np.sqrt(12) if np.std(r[:mid]) > 0 else 0
    h2_s = np.mean(r[mid:]) / np.std(r[mid:]) * np.sqrt(12) if np.std(r[mid:]) > 0 else 0
    gates['sub_period'] = {
        'h1_sharpe': round(float(h1_s), 2), 'h2_sharpe': round(float(h2_s), 2),
        'pass': h1_s > 0 and h2_s > 0
    }

    # 4. Outlier
    n_remove = max(1, int(len(r) * 0.05))
    sorted_idx = np.argsort(r)[::-1]
    trimmed = np.delete(r, sorted_idx[:n_remove])
    trim_s = np.mean(trimmed) / np.std(trimmed) * np.sqrt(12) if np.std(trimmed) > 0 else 0
    gates['outlier'] = {'trimmed_sharpe': round(float(trim_s), 2), 'pass': trim_s > 0}

    n_pass = sum(1 for g in gates.values() if g.get('pass') is True)
    n_total = sum(1 for g in gates.values() if g.get('pass') is not None)

    return gates, n_pass, n_total


def main():
    import yfinance as yf

    fprint("=" * 60)
    fprint("SECTOR ETF MOMENTUM v2 — REGIME-HEDGED")
    fprint("=" * 60)
    fprint("Goal: Fix R1 regime test failure (v1 gap: 0.784)")
    fprint(f"Testing 4 hedging variants on {len(UNIVERSE)} ETFs\n")

    # Download data
    fprint("Downloading data...")
    all_data = {}
    for t in UNIVERSE:
        try:
            df = yf.download(t, start='2008-01-01', end='2026-07-24', progress=False)
            if df.index.tz is not None: df.index = df.index.tz_convert(None)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if len(df) > 252: all_data[t] = df
        except: pass

    spy = yf.download('SPY', start='2008-01-01', end='2026-07-24', progress=False)
    if spy.index.tz is not None: spy.index = spy.index.tz_convert(None)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)

    fprint(f"  Loaded {len(all_data)} ETFs")

    # Common dates
    common = None
    for t, df in all_data.items():
        common = df.index if common is None else common.intersection(df.index)
    fprint(f"  Common trading days: {len(common)}")

    # Run 4 variants
    variants = {
        'A_baseline': {'use_regime_features': False, 'defensive_overlay': False, 'position_sizing': False},
        'B_regime_features': {'use_regime_features': True, 'defensive_overlay': False, 'position_sizing': False},
        'C_defensive_overlay': {'use_regime_features': False, 'defensive_overlay': True, 'position_sizing': False},
        'D_position_sizing': {'use_regime_features': False, 'defensive_overlay': False, 'position_sizing': True},
        'E_features_plus_overlay': {'use_regime_features': True, 'defensive_overlay': True, 'position_sizing': False},
        'F_full_hedge': {'use_regime_features': True, 'defensive_overlay': True, 'position_sizing': True},
    }

    all_results = {}

    for vname, vparams in variants.items():
        fprint(f"\n--- Variant {vname} ---")
        rets, spy_rets, picks = run_variant(all_data, spy, common, vname, TOP_K, **vparams)
        metrics = compute_metrics(rets, vname)
        gates, n_pass, n_total = adversarial_gates(rets, spy_rets, vname)

        r1 = gates.get('regime_r1', {})
        fprint(f"  Sharpe {metrics['sharpe']:.2f} | CAGR {metrics['cagr']:.1f}% | MaxDD {metrics['maxdd']:.1f}% | "
               f"WR {metrics['wr']:.1f}% | R1 gap {r1.get('gap', '?')} | Gates {n_pass}/{n_total}")

        # Count regime distribution
        regimes = [p['regime'] for p in picks]
        fprint(f"  Regime distribution: bull={regimes.count('bull')}, bear={regimes.count('bear')}, neutral={regimes.count('neutral')}")

        all_results[vname] = {
            'metrics': metrics,
            'gates': gates,
            'n_pass': n_pass,
            'n_total': n_total,
            'picks_sample': picks[:5],
            'regime_dist': {
                'bull': regimes.count('bull'),
                'bear': regimes.count('bear'),
                'neutral': regimes.count('neutral'),
            }
        }

    # Summary
    fprint(f"\n{'='*80}")
    fprint("VARIANT COMPARISON — REGIME HEDGE EFFECTIVENESS")
    fprint(f"{'='*80}")
    fprint(f"\n{'Variant':<28} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'WR':>6} {'R1 Gap':>7} {'Gates':>6}")
    fprint("-" * 75)

    best_variant = None
    best_score = -999

    for vname, vdata in sorted(all_results.items()):
        m = vdata['metrics']
        r1 = vdata['gates'].get('regime_r1', {})
        gap = r1.get('gap', 999)
        r1_status = 'PASS' if r1.get('pass') else 'FAIL'
        fprint(f"{vname:<28} {m['sharpe']:>7.2f} {m['cagr']:>6.1f}% {m['maxdd']:>6.1f}% "
               f"{m['wr']:>5.1f}% {gap:>6.3f} {vdata['n_pass']}/{vdata['n_total']} {r1_status}")

        # Score: maximize gates passed, then Sharpe
        score = vdata['n_pass'] * 100 + m['sharpe']
        if score > best_score:
            best_score = score
            best_variant = vname

    fprint(f"\nBEST VARIANT: {best_variant}")
    best = all_results[best_variant]
    bm = best['metrics']
    bg = best['gates']

    fprint(f"  Sharpe: {bm['sharpe']}")
    fprint(f"  CAGR: {bm['cagr']}%")
    fprint(f"  MaxDD: {bm['maxdd']}%")
    fprint(f"  Gates: {best['n_pass']}/{best['n_total']}")
    for gname, gdata in bg.items():
        fprint(f"    {gname}: {'PASS' if gdata.get('pass') else 'FAIL'} — {gdata}")

    # Save
    output = {
        'strategy': 'Sector ETF Momentum v2 — Regime-Hedged',
        'run_date': str(datetime.now()),
        'goal': 'Fix R1 regime test failure from v1 (gap 0.784)',
        'universe': UNIVERSE,
        'top_k': TOP_K,
        'survivorship_free': True,
        'variants': {k: {
            'metrics': v['metrics'],
            'gates': v['gates'],
            'n_pass': v['n_pass'],
            'regime_dist': v['regime_dist'],
        } for k, v in all_results.items()},
        'best_variant': best_variant,
        'best_metrics': bm,
        'best_gates': bg,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    fprint(f"\nResults saved.")

    if MLFLOW_OK:
        try:
            mlflow.set_experiment('sector_etf_momentum_v2')
            with mlflow.start_run(run_name=f'v2_{best_variant}'):
                mlflow.log_metrics({
                    'sharpe': bm.get('sharpe', 0),
                    'sortino': bm.get('sortino', 0),
                    'cagr': bm.get('cagr', 0),
                    'maxdd': bm.get('maxdd', 0),
                    'wr': bm.get('wr', 0),
                    'pf': bm.get('pf', 0),
                    'r1_gap': bg.get('regime_r1', {}).get('gap', -1),
                    'n_pass': best['n_pass'],
                })
                mlflow.log_params({
                    'variant': best_variant,
                    'universe_size': len(UNIVERSE),
                    'top_k': TOP_K,
                })
                for vname, vdata in all_results.items():
                    vm = vdata['metrics']
                    vg = vdata['gates'].get('regime_r1', {})
                    mlflow.log_metrics({
                        f'{vname}_sharpe': vm.get('sharpe', 0),
                        f'{vname}_r1_gap': vg.get('gap', -1),
                        f'{vname}_cagr': vm.get('cagr', 0),
                    })
        except Exception as e:
            fprint(f"MLflow error: {e}")

    return output


if __name__ == '__main__':
    main()
