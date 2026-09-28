#!/usr/bin/env python3
"""
Sector ETF Momentum v3 — Regime-Switching Ensemble
=====================================================

v2 proved that simple score-boosting can't fix LightGBM's R1 regime gap.
Root cause: LightGBM inherently picks risk-on ETFs, and soft adjustments
can't overcome its predictions.

NEW APPROACH: Use DIFFERENT selection methods per regime:
- BULL (SPY > SMA200): LightGBM momentum (where it has Sharpe 6.4!)
- BEAR (SPY < SMA200): Simple defensive scoring (Sharpe 0.48 in bear, proven regime-safe)

Also tests:
- Adding regime features to LightGBM (VIX proxy, SPY trend, regime flag)
- Hard defensive filter in bear (only pick from defensive ETFs)
- Blend approaches

Walk-forward: 252d train, 21d test, sliding window.
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

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE',
    'XLC', 'QQQ', 'IWM', 'MDY', 'EFA', 'EEM', 'GLD', 'TLT', 'HYG', 'IYR',
    'VNQ', 'DBC',
]

DEFENSIVE_ETFS = {'XLP', 'XLU', 'TLT', 'GLD', 'XLV'}
RISK_ON_ETFS = {'XLK', 'QQQ', 'XLY', 'IWM', 'EEM', 'HYG'}

BASE_FEAT_NAMES = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'mom_12_1', 'high_52w_pct', 'mom_accel', 'vol_20d', 'vol_60d',
    'vol_ratio', 'sharpe_63d', 'sharpe_126d', 'maxdd_63d', 'vol_rel',
    'skew_63d', 'kurt_63d', 'vol_trend'
]

REGIME_FEAT_NAMES = ['spy_sma200_ratio', 'spy_rv21d', 'spy_mom_63d', 'regime_flag']


def build_features(close, volume):
    """Build base momentum + quality features."""
    lr = np.log(close / close.shift(1))
    vol_rel = volume / volume.rolling(20).mean() if volume is not None else pd.Series(0, index=close.index)

    return pd.DataFrame({
        'ret_5d': close.pct_change(5),
        'ret_10d': close.pct_change(10),
        'ret_21d': close.pct_change(21),
        'ret_63d': close.pct_change(63),
        'ret_126d': close.pct_change(126),
        'ret_252d': close.pct_change(252),
        'mom_12_1': close.pct_change(252) - close.pct_change(21),
        'high_52w_pct': close / close.rolling(252).max(),
        'mom_accel': close.pct_change(63) - close.pct_change(63).shift(63),
        'vol_20d': lr.rolling(20).std() * np.sqrt(252),
        'vol_60d': lr.rolling(60).std() * np.sqrt(252),
        'vol_ratio': lr.rolling(20).std() / lr.rolling(60).std(),
        'sharpe_63d': lr.rolling(63).mean() / lr.rolling(63).std(),
        'sharpe_126d': lr.rolling(126).mean() / lr.rolling(126).std(),
        'maxdd_63d': (close / close.rolling(63).max() - 1).rolling(63).min(),
        'vol_rel': vol_rel,
        'skew_63d': lr.rolling(63).skew(),
        'kurt_63d': lr.rolling(63).kurt(),
        'vol_trend': (lr.rolling(20).std() - lr.rolling(60).std()) / lr.rolling(60).std(),
    }, index=close.index)


def build_spy_features(spy_close):
    """Build SPY regime features to merge into each ETF's features."""
    sma200 = spy_close.rolling(200).mean()
    lr = np.log(spy_close / spy_close.shift(1))

    feat = pd.DataFrame({
        'spy_sma200_ratio': spy_close / sma200,
        'spy_rv21d': lr.rolling(21).std() * np.sqrt(252) * 100,  # Vol in %
        'spy_mom_63d': spy_close.pct_change(63),
        'regime_flag': (spy_close < sma200).astype(float),  # 1.0 = bear
    }, index=spy_close.index)

    return feat


def simple_defensive_score(ticker, close_data, date):
    """Simple momentum score with defensive bias for bear markets."""
    c = close_data
    idx = c.index.searchsorted(date)
    if idx < 126 or idx >= len(c):
        return 0

    # Simple momentum (3m)
    mom_3m = c.iloc[idx] / c.iloc[max(0, idx-63)] - 1

    # Defensive boost
    if ticker in DEFENSIVE_ETFS:
        return mom_3m + 0.05  # Slight boost for defensive
    elif ticker in RISK_ON_ETFS:
        return mom_3m - 0.05  # Slight penalty for risk-on
    return mom_3m


def precompute(universe, start='2008-01-01', end='2026-07-25'):
    """Download data, build feature matrices."""
    import yfinance as yf

    fprint("Downloading ETF data...")
    all_data = {}
    for t in universe:
        try:
            df = yf.download(t, start=start, end=end, progress=False)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 252:
                all_data[t] = df
        except:
            pass

    spy = yf.download('SPY', start=start, end=end, progress=False)
    if spy.index.tz is not None: spy.index = spy.index.tz_convert(None)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)

    fprint(f"  Loaded {len(all_data)} ETFs")

    common = None
    for t, df in all_data.items():
        common = df.index if common is None else common.intersection(df.index)
    fprint(f"  Common dates: {len(common)}")

    spy_feats = build_spy_features(spy['Close'].reindex(common))
    spy_sma200 = spy['Close'].rolling(200).mean()
    spy_regime = pd.Series('bull', index=spy['Close'].index)
    spy_regime[spy['Close'] < spy_sma200] = 'bear'

    # Build base + regime feature matrices
    fprint("  Building features...")
    base_dfs = []
    regime_dfs = []

    for t, df in all_data.items():
        c = df['Close'].reindex(common)
        v = df['Volume'].reindex(common) if 'Volume' in df.columns else None
        feat = build_features(c, v)
        fwd = c.pct_change(21).shift(-21)

        # Add regime features
        feat_regime = feat.copy()
        for col in REGIME_FEAT_NAMES:
            if col in spy_feats.columns:
                feat_regime[col] = spy_feats[col].reindex(feat.index)

        feat['fwd_ret'] = fwd
        feat['ticker'] = t
        feat['date'] = feat.index

        feat_regime['fwd_ret'] = fwd
        feat_regime['ticker'] = t
        feat_regime['date'] = feat_regime.index

        # Valid rows
        all_cols = BASE_FEAT_NAMES + ['fwd_ret']
        valid = feat[all_cols].notna().all(axis=1) & np.isfinite(feat[BASE_FEAT_NAMES].values).all(axis=1)
        base_dfs.append(feat[valid])

        all_cols_r = BASE_FEAT_NAMES + REGIME_FEAT_NAMES + ['fwd_ret']
        valid_r = feat_regime[all_cols_r].notna().all(axis=1) & np.isfinite(feat_regime[BASE_FEAT_NAMES + REGIME_FEAT_NAMES].values).all(axis=1)
        regime_dfs.append(feat_regime[valid_r])

    base_combined = pd.concat(base_dfs, ignore_index=True)
    regime_combined = pd.concat(regime_dfs, ignore_index=True)

    fprint(f"  Base matrix: {len(base_combined)} × {len(BASE_FEAT_NAMES)}")
    fprint(f"  Regime matrix: {len(regime_combined)} × {len(BASE_FEAT_NAMES) + len(REGIME_FEAT_NAMES)}")

    return all_data, spy, spy_regime, common, base_combined, regime_combined


def run_lgbm_walk_forward(combined, feat_names, all_data, spy, spy_regime):
    """Run LightGBM walk-forward, return per-fold predictions."""
    import lightgbm as lgb

    X = combined[feat_names].values
    y = combined['fwd_ret'].values
    meta = combined[['date', 'ticker']].reset_index(drop=True)
    dates = sorted(meta['date'].unique())

    # Index lookup
    date_to_idx = {}
    for idx in range(len(meta)):
        d = meta.iloc[idx]['date']
        if d not in date_to_idx:
            date_to_idx[d] = []
        date_to_idx[d].append(idx)

    fold_results = []
    i = 252
    fold = 0
    while i + 21 <= len(dates):
        fold += 1
        if fold % 20 == 0:
            fprint(f"    Fold {fold}...")

        train_dates = set(dates[i-252:i])
        test_dates_list = dates[i:i+21]
        test_dates = set(test_dates_list)

        train_idxs = []
        for d in train_dates:
            train_idxs.extend(date_to_idx.get(d, []))
        test_idxs = []
        for d in test_dates:
            test_idxs.extend(date_to_idx.get(d, []))

        if len(train_idxs) < 50 or len(test_idxs) < 5:
            i += 21
            continue

        model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            subsample=0.8, verbose=-1, n_jobs=-1
        )
        model.fit(X[train_idxs], y[train_idxs])

        meta_te = meta.iloc[test_idxs].copy()
        meta_te['pred'] = model.predict(X[test_idxs])

        td = test_dates_list[0]
        dp = meta_te[meta_te['date'] == td][['ticker', 'pred']].to_dict('records')

        # Regime
        regime = 'bull'
        if td in spy_regime.index:
            regime = spy_regime.loc[td]
        else:
            ni = spy_regime.index.searchsorted(td)
            if ni > 0:
                regime = spy_regime.iloc[ni - 1]

        # SPY return
        sc = spy['Close']
        si = sc.index.searchsorted(test_dates_list[0])
        ei = sc.index.searchsorted(test_dates_list[-1])
        spy_ret = float(sc.iloc[ei] / sc.iloc[si] - 1) if si < len(sc) and ei < len(sc) and sc.iloc[si] > 0 else 0

        # Ticker returns
        ticker_rets = {}
        for t in set(r['ticker'] for r in dp):
            if t in all_data:
                tc = all_data[t]['Close']
                tsi = tc.index.searchsorted(test_dates_list[0])
                tei = tc.index.searchsorted(test_dates_list[-1])
                if tsi < len(tc) and tei < len(tc) and tc.iloc[tsi] > 0:
                    ticker_rets[t] = float(tc.iloc[tei] / tc.iloc[tsi] - 1)

        fold_results.append({
            'date': td, 'predictions': dp, 'regime': regime,
            'spy_ret': spy_ret, 'ticker_returns': ticker_rets,
        })
        i += 21

    fprint(f"    {len(fold_results)} folds complete")
    return fold_results


def apply_selection(fold_results, all_data, method, top_k=3, cost_bps=10):
    """Apply selection method to precomputed predictions."""
    monthly_returns = []
    spy_returns = []
    picks_list = []

    for fold in fold_results:
        preds = pd.DataFrame(fold['predictions'])
        regime = fold['regime']
        ticker_rets = fold['ticker_returns']
        td = fold['date']

        if len(preds) < top_k:
            continue

        if method == 'lgbm_only':
            # Pure LightGBM
            top = preds.nlargest(top_k, 'pred')['ticker'].tolist()

        elif method == 'regime_switch':
            # BULL: LightGBM. BEAR: Simple defensive scoring
            if regime == 'bull':
                top = preds.nlargest(top_k, 'pred')['ticker'].tolist()
            else:
                # Simple defensive: momentum + defensive bias
                scores = {}
                for _, row in preds.iterrows():
                    t = row['ticker']
                    if t in all_data:
                        c = all_data[t]['Close']
                        idx = c.index.searchsorted(td)
                        if idx >= 63 and idx < len(c) and c.iloc[max(0,idx-63)] > 0:
                            mom = c.iloc[idx] / c.iloc[idx-63] - 1
                        else:
                            mom = 0
                        # Defensive bias
                        if t in DEFENSIVE_ETFS:
                            scores[t] = mom + 0.05
                        elif t in RISK_ON_ETFS:
                            scores[t] = mom - 0.05
                        else:
                            scores[t] = mom
                top = sorted(scores, key=scores.get, reverse=True)[:top_k]

        elif method == 'hard_defensive_filter':
            # BULL: LightGBM from all. BEAR: LightGBM from defensive only
            if regime == 'bull':
                top = preds.nlargest(top_k, 'pred')['ticker'].tolist()
            else:
                defensive_preds = preds[preds['ticker'].isin(DEFENSIVE_ETFS)]
                if len(defensive_preds) >= top_k:
                    top = defensive_preds.nlargest(top_k, 'pred')['ticker'].tolist()
                else:
                    top = defensive_preds['ticker'].tolist()
                    # Fill remaining from neutral ETFs
                    neutral = preds[~preds['ticker'].isin(RISK_ON_ETFS)]
                    remaining = neutral[~neutral['ticker'].isin(top)].nlargest(
                        top_k - len(top), 'pred')['ticker'].tolist()
                    top.extend(remaining)

        elif method == 'lgbm_regime_features':
            # LightGBM with regime features (already in predictions from regime model)
            top = preds.nlargest(top_k, 'pred')['ticker'].tolist()

        elif method == 'blend_50_50':
            # 50% LightGBM, 50% defensive scoring in bear. 100% LightGBM in bull.
            if regime == 'bull':
                top = preds.nlargest(top_k, 'pred')['ticker'].tolist()
            else:
                # Blend: normalize LGBM preds, add defensive score, pick top
                preds_copy = preds.copy()
                pred_std = preds_copy['pred'].std()
                if pred_std > 0:
                    preds_copy['pred_norm'] = (preds_copy['pred'] - preds_copy['pred'].mean()) / pred_std
                else:
                    preds_copy['pred_norm'] = 0

                for idx, row in preds_copy.iterrows():
                    t = row['ticker']
                    if t in all_data:
                        c = all_data[t]['Close']
                        ci = c.index.searchsorted(td)
                        if ci >= 63 and ci < len(c) and c.iloc[max(0,ci-63)] > 0:
                            mom = c.iloc[ci] / c.iloc[ci-63] - 1
                        else:
                            mom = 0
                        def_score = 0
                        if t in DEFENSIVE_ETFS:
                            def_score = 1.0
                        elif t in RISK_ON_ETFS:
                            def_score = -1.0
                        preds_copy.at[idx, 'blend'] = 0.5 * row['pred_norm'] + 0.5 * (mom * 10 + def_score)
                    else:
                        preds_copy.at[idx, 'blend'] = row['pred_norm']
                top = preds_copy.nlargest(top_k, 'blend')['ticker'].tolist()

        elif method == 'hedged_lgbm':
            # LightGBM picks + forced TLT allocation in bear
            if regime == 'bull':
                top = preds.nlargest(top_k, 'pred')['ticker'].tolist()
            else:
                # Reserve 1 slot for TLT, rest from LightGBM
                non_tlt = preds[preds['ticker'] != 'TLT']
                lgbm_picks = non_tlt.nlargest(top_k - 1, 'pred')['ticker'].tolist()
                top = lgbm_picks + ['TLT']

        else:
            top = preds.nlargest(top_k, 'pred')['ticker'].tolist()

        # Calculate returns
        ret = 0
        n_valid = 0
        for t in top:
            if t in ticker_rets:
                ret += ticker_rets[t]
                n_valid += 1
        if n_valid > 0:
            ret /= top_k
        ret -= cost_bps / 10000 * 2

        monthly_returns.append(ret)
        spy_returns.append(fold['spy_ret'])
        picks_list.append({
            'date': str(td.date()) if hasattr(td, 'date') else str(td),
            'picks': top, 'regime': regime, 'return': ret,
        })

    return monthly_returns, spy_returns, picks_list


def compute_metrics(returns, name='Strategy'):
    r = np.array(returns)
    if len(r) < 2:
        return {'name': name}
    equity = 100000 * np.cumprod(1 + r)
    ppy = 12
    sharpe = float(np.mean(r) / np.std(r) * np.sqrt(ppy)) if np.std(r) > 0 else 0
    ds = r[r < 0]
    sortino = float(np.mean(r) / np.std(ds) * np.sqrt(ppy)) if len(ds) > 0 and np.std(ds) > 0 else 0
    years = len(r) / ppy
    cagr = float(((equity[-1] / 100000) ** (1/max(years, 0.01)) - 1) * 100)
    peak = np.maximum.accumulate(equity)
    maxdd = float(np.min((equity - peak) / peak) * 100)
    wr = float(len(r[r > 0]) / len(r) * 100)
    pf = float(abs(r[r > 0].sum() / r[r < 0].sum())) if len(ds) > 0 and r[r < 0].sum() != 0 else 999
    calmar = float(cagr / abs(maxdd)) if maxdd != 0 else 0
    return {
        'name': name, 'sharpe': round(sharpe, 2), 'sortino': round(sortino, 2),
        'cagr': round(cagr, 1), 'maxdd': round(maxdd, 1), 'wr': round(wr, 1),
        'pf': round(pf, 2), 'calmar': round(calmar, 2), 'n_months': len(r),
        'final_equity': round(float(equity[-1]), 2),
    }


def adversarial_gates(returns, spy_returns, n_perm=200):
    r = np.array(returns)
    sp = np.array(spy_returns[:len(r)])
    gates = {}

    real_sharpe = float(np.mean(r) / np.std(r) * np.sqrt(12)) if np.std(r) > 0 else 0
    rng = np.random.default_rng(42)
    perm_sharpes = []
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(r))
        perm_r = r * signs
        ps = float(np.mean(perm_r) / np.std(perm_r) * np.sqrt(12)) if np.std(perm_r) > 0 else 0
        perm_sharpes.append(ps)
    perm_p = float(np.mean(np.array(perm_sharpes) >= real_sharpe))
    gates['permutation'] = {'p_value': round(perm_p, 3), 'pass': bool(perm_p < 0.05), 'real_sharpe': round(real_sharpe, 2)}

    bull_mask = sp > 0
    bear_mask = sp <= 0
    if bull_mask.sum() >= 6 and bear_mask.sum() >= 6:
        bull_r, bear_r = r[bull_mask], r[bear_mask]
        bull_s = float(np.mean(bull_r) / np.std(bull_r) * np.sqrt(12)) if np.std(bull_r) > 0 else 0
        bear_s = float(np.mean(bear_r) / np.std(bear_r) * np.sqrt(12)) if np.std(bear_r) > 0 else 0
        max_s = max(abs(bull_s), abs(bear_s))
        gap = float(abs(bull_s - bear_s) / max_s) if max_s > 0 else 0
        gates['regime_r1'] = {'bull_sharpe': round(bull_s, 2), 'bear_sharpe': round(bear_s, 2),
                              'gap': round(gap, 3), 'pass': bool(gap < 0.50)}
    else:
        gates['regime_r1'] = {'pass': None, 'note': 'Insufficient data'}

    mid = len(r) // 2
    h1_s = float(np.mean(r[:mid]) / np.std(r[:mid]) * np.sqrt(12)) if np.std(r[:mid]) > 0 else 0
    h2_s = float(np.mean(r[mid:]) / np.std(r[mid:]) * np.sqrt(12)) if np.std(r[mid:]) > 0 else 0
    gates['sub_period'] = {'h1_sharpe': round(h1_s, 2), 'h2_sharpe': round(h2_s, 2),
                           'pass': bool(h1_s > 0 and h2_s > 0)}

    n_rm = max(1, int(len(r) * 0.05))
    trimmed = np.delete(r, np.argsort(r)[::-1][:n_rm])
    trim_s = float(np.mean(trimmed) / np.std(trimmed) * np.sqrt(12)) if np.std(trimmed) > 0 else 0
    gates['outlier'] = {'trimmed_sharpe': round(trim_s, 2), 'n_removed': n_rm, 'pass': bool(trim_s > 0)}

    return gates


def main():
    fprint("=" * 70)
    fprint("SECTOR ETF MOMENTUM v3 — Regime-Switching Ensemble")
    fprint("=" * 70)
    fprint("v2 proved overlays don't fix R1 (gap INCREASES with boost).")
    fprint("New approach: use DIFFERENT selection methods per regime.\n")

    all_data, spy, spy_regime, common, base_combined, regime_combined = precompute(UNIVERSE)

    # Train two LightGBM models: base (19 features) and regime-aware (23 features)
    fprint("\nTraining BASE LightGBM (19 features)...")
    base_folds = run_lgbm_walk_forward(base_combined, BASE_FEAT_NAMES, all_data, spy, spy_regime)

    fprint("\nTraining REGIME-AWARE LightGBM (23 features)...")
    regime_folds = run_lgbm_walk_forward(regime_combined, BASE_FEAT_NAMES + REGIME_FEAT_NAMES, all_data, spy, spy_regime)

    # Define variants
    variants = [
        # (name, fold_source, method, top_k)
        ('LGBM_Base_Top3', base_folds, 'lgbm_only', 3),
        ('LGBM_Base_Top5', base_folds, 'lgbm_only', 5),
        ('RegimeSwitch_Top3', base_folds, 'regime_switch', 3),
        ('RegimeSwitch_Top5', base_folds, 'regime_switch', 5),
        ('HardDefFilter_Top3', base_folds, 'hard_defensive_filter', 3),
        ('HardDefFilter_Top5', base_folds, 'hard_defensive_filter', 5),
        ('Blend50_Top3', base_folds, 'blend_50_50', 3),
        ('Blend50_Top5', base_folds, 'blend_50_50', 5),
        ('HedgedLGBM_Top3', base_folds, 'hedged_lgbm', 3),
        ('HedgedLGBM_Top5', base_folds, 'hedged_lgbm', 5),
        ('LGBM_Regime_Top3', regime_folds, 'lgbm_only', 3),
        ('LGBM_Regime_Top5', regime_folds, 'lgbm_only', 5),
    ]

    fprint(f"\nTesting {len(variants)} variants...")
    results = []

    for name, folds, method, top_k in variants:
        monthly_rets, spy_rets, picks = apply_selection(folds, all_data, method, top_k)

        if len(monthly_rets) < 12:
            continue

        metrics = compute_metrics(monthly_rets, name)
        gates = adversarial_gates(monthly_rets, spy_rets)
        n_pass = sum(1 for g in gates.values() if g.get('pass') is True)
        n_total = sum(1 for g in gates.values() if g.get('pass') is not None)

        r1 = gates.get('regime_r1', {})
        fprint(f"  {name:<22} Sharpe {metrics['sharpe']:>5.2f} | CAGR {metrics['cagr']:>5.1f}% | "
               f"MaxDD {metrics['maxdd']:>5.1f}% | R1 gap {r1.get('gap', '?'):>5} | Gates {n_pass}/{n_total}")

        results.append({
            'variant': name, 'method': method, 'top_k': top_k,
            'metrics': metrics, 'gates': gates,
            'gates_pass': n_pass, 'gates_total': n_total,
            'monthly_returns': monthly_rets, 'spy_returns': spy_rets,
            'monthly_picks': picks,
        })

    results.sort(key=lambda x: (x['gates_pass'], x['metrics'].get('sharpe', 0)), reverse=True)

    # Summary
    fprint(f"\n{'='*90}")
    fprint("VARIANT RANKING")
    fprint(f"{'='*90}")
    fprint(f"{'Variant':<22} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'WR':>6} {'R1gap':>6} {'Gates':>6}")
    fprint("-" * 75)
    for r in results:
        m = r['metrics']
        g = r['gates'].get('regime_r1', {})
        gap = g.get('gap', 0)
        fprint(f"{r['variant']:<22} {m.get('sharpe',0):>7.2f} {m.get('sortino',0):>8.2f} "
               f"{m.get('cagr',0):>6.1f}% {m.get('maxdd',0):>6.1f}% {m.get('wr',0):>5.1f}% "
               f"{gap:>6.3f} {r['gates_pass']}/{r['gates_total']}")

    # Winner analysis
    best_4 = [r for r in results if r['gates_pass'] >= 4]
    if best_4:
        best = best_4[0]
        m, g = best['metrics'], best['gates']
        fprint(f"\n{'='*70}")
        fprint(f"WINNER (4/4 GATES): {best['variant']}")
        fprint(f"{'='*70}")
        for k in ['sharpe', 'sortino', 'cagr', 'maxdd', 'wr', 'pf', 'calmar']:
            fprint(f"  {k}: {m[k]}")
        fprint(f"\n  GATES:")
        fprint(f"    Perm: p={g['permutation']['p_value']:.3f} {'PASS' if g['permutation']['pass'] else 'FAIL'}")
        r1 = g['regime_r1']
        fprint(f"    R1: bull={r1['bull_sharpe']:.2f}, bear={r1['bear_sharpe']:.2f}, gap={r1['gap']:.3f} {'PASS' if r1['pass'] else 'FAIL'}")
        fprint(f"    Sub: H1={g['sub_period']['h1_sharpe']:.2f}, H2={g['sub_period']['h2_sharpe']:.2f} {'PASS' if g['sub_period']['pass'] else 'FAIL'}")
        fprint(f"    Outlier: {g['outlier']['trimmed_sharpe']:.2f} {'PASS' if g['outlier']['pass'] else 'FAIL'}")

        # Year-by-year
        fprint(f"\n  Year-by-year:")
        by_year = {}
        for i, p in enumerate(best['monthly_picks']):
            yr = p['date'][:4]
            by_year.setdefault(yr, []).append(best['monthly_returns'][i])
        for yr in sorted(by_year):
            rets = by_year[yr]
            yr_ret = (np.prod(1 + np.array(rets)) - 1) * 100
            yr_s = np.mean(rets) / np.std(rets) * np.sqrt(12) if np.std(rets) > 0 else 0
            fprint(f"    {yr}: {yr_ret:>7.1f}%  Sharpe {yr_s:.2f}")

        # Bear picks
        fprint(f"\n  Bear regime picks:")
        bear = [p for p in best['monthly_picks'] if p.get('regime') == 'bear']
        for p in bear[:8]:
            fprint(f"    {p['date']}: {p['picks']} ret={p['return']*100:+.1f}%")
    else:
        fprint("\nNo 4/4 winner.")
        best_3 = [r for r in results if r['gates_pass'] >= 3]
        if best_3:
            fprint(f"Best 3/4: {best_3[0]['variant']} Sharpe {best_3[0]['metrics']['sharpe']:.2f}")

    # Improvement vs baseline
    baseline = next((r for r in results if r['variant'] == 'LGBM_Base_Top3'), None)
    if baseline:
        fprint(f"\n  Baseline (LGBM_Base_Top3): Sharpe {baseline['metrics']['sharpe']:.2f}, R1 gap {baseline['gates']['regime_r1'].get('gap', '?')}")
        best_regime = min(results, key=lambda x: x['gates']['regime_r1'].get('gap', 999))
        fprint(f"  Best R1 gap: {best_regime['variant']} gap={best_regime['gates']['regime_r1'].get('gap', '?')}")

    # Save
    save_data = []
    for r in results:
        sr = {k: v for k, v in r.items() if k not in ['monthly_picks']}
        if r == results[0]:
            sr['monthly_picks'] = r['monthly_picks']
        save_data.append(sr)

    output = {
        'strategy': 'Sector ETF Momentum v3 — Regime-Switching Ensemble',
        'run_date': str(datetime.now()),
        'n_variants': len(results),
        'best_variant': results[0]['variant'] if results else None,
        'results': save_data,
    }
    with open(RESULTS_DIR / 'sector_etf_momentum_v3_ensemble_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved.")

    # MLflow
    if MLFLOW_OK and results:
        try:
            mlflow.set_experiment('sector_etf_momentum_v3_ensemble')
            best = results[0]
            m, g = best['metrics'], best['gates']
            with mlflow.start_run(run_name=f"v3_{best['variant']}"):
                mlflow.log_params({'variant': best['variant'], 'method': best.get('method',''), 'top_k': best['top_k']})
                mlflow.log_metrics({
                    'sharpe': m.get('sharpe', 0), 'sortino': m.get('sortino', 0),
                    'cagr': m.get('cagr', 0), 'maxdd': m.get('maxdd', 0),
                    'wr': m.get('wr', 0), 'pf': m.get('pf', 0),
                    'r1_gap': g.get('regime_r1', {}).get('gap', 0),
                    'perm_p': g.get('permutation', {}).get('p_value', 1),
                    'gates_passed': best['gates_pass'],
                })
            fprint("MLflow logged.")
        except Exception as e:
            fprint(f"MLflow: {e}")

    fprint("\nDone.")


if __name__ == '__main__':
    main()
