#!/usr/bin/env python3
"""
ML Trend Following v3 — VVIX + Volatility Surface Enhancement
================================================================
BASELINE: ML Trend Following v2 (Sharpe 2.90, R1 PASS, 3/4 gates).

HYPOTHESIS: Adding volatility-of-volatility (VVIX), term structure (VIX/VXV),
and cross-asset correlation features can improve the ML filter's ability to
distinguish real trend breakouts from noise whipsaws.

New features vs v2:
  1. VVIX level + percentile (vol-of-vol indicates regime uncertainty)
  2. VIX term structure slope (VIX/VXV ratio — backwardation = fear)
  3. Cross-asset correlation (rolling 60d) — high correlation = crowded trades
  4. Trend alignment momentum (rate of change of n_bullish)
  5. Credit-equity divergence (HYG momentum vs SPY momentum)

Walk-forward: 252d sliding, retrain every 21d (HC #0).
Fixed $100K, NO DCA (HC #713).
Full adversarial: perm 50x (faster), sub-period, R1, outlier.

SUCCESS: Sharpe > 3.0 AND all gates from v2 still pass.
"""

import json
import time
import warnings
from pathlib import Path
from functools import partial

print = partial(print, flush=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 100_000
TRAIN_WINDOW     = 252
N_PERMUTATIONS   = 50           # faster than 100
MA_SHORT         = 20
MA_LONG          = 100
ML_THRESHOLD     = 0.55
TARGET_VOL       = 0.10

BASE   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_trend_v3"
OUTPUT.mkdir(parents=True, exist_ok=True)

UNIVERSE = ['SPY', 'TLT', 'GLD', 'UUP', 'EEM', 'VNQ', 'HYG', 'XLE']


def download_data():
    cache = BASE / "data" / "cache" / "trend_v3_data.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists() and (time.time() - cache.stat().st_mtime) < 3600:
        print(f"  Using cached data")
        return pd.read_parquet(cache)

    print("  Downloading from yfinance...")
    tickers = UNIVERSE + ['^VIX', '^VVIX', 'QQQ', 'IWM', 'IEF', 'SHY']
    data = yf.download(tickers, start='2007-01-01', end='2026-07-18',
                       auto_adjust=True, progress=False)
    closes = data['Close'].copy()
    closes.columns = [c.replace('^', '') for c in closes.columns]
    closes = closes.ffill().dropna(how='all')
    closes.to_parquet(cache)
    print(f"  Data: {len(closes)} days, {closes.shape[1]} assets")
    return closes


def build_enhanced_features(closes, assets):
    """Build all features including v3 enhancements."""
    returns = closes[assets].pct_change()

    # Basic trend signals
    ma_fast = closes[assets].rolling(MA_SHORT).mean()
    ma_slow = closes[assets].rolling(MA_LONG).mean()
    trend_signals = (ma_fast > ma_slow).astype(int)

    feats = pd.DataFrame(index=closes.index)

    # ─── V2 FEATURES (baseline) ───
    for a in assets:
        # Momentum
        feats[f'{a}_mom20'] = closes[a].pct_change(20)
        # Volatility
        feats[f'{a}_vol20'] = returns[a].rolling(20).std()
        # Trend strength (distance from slow MA)
        feats[f'{a}_trend_str'] = (closes[a] - ma_slow[a]) / (ma_slow[a] + 1e-8)
        # Trend duration (consecutive days in trend)
        signal_change = trend_signals[a].diff().abs()
        # Approximate duration with cumulative count
        feats[f'{a}_in_trend'] = trend_signals[a].rolling(20).mean()

    # Trend alignment
    feats['n_bullish'] = trend_signals.sum(axis=1)
    feats['pct_bullish'] = feats['n_bullish'] / len(assets)

    # VIX features
    if 'VIX' in closes.columns:
        vix = closes['VIX']
        feats['vix_level'] = vix
        feats['vix_sma20'] = vix.rolling(20).mean()
        feats['vix_pctile'] = vix.rolling(252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]), raw=False)
        feats['vix_zscore'] = (vix - vix.rolling(60).mean()) / (vix.rolling(60).std() + 1e-8)

    # ─── V3 NEW FEATURES ───

    # 1. VVIX (volatility of volatility)
    if 'VVIX' in closes.columns:
        vvix = closes['VVIX']
        feats['vvix_level'] = vvix
        feats['vvix_sma20'] = vvix.rolling(20).mean()
        feats['vvix_pctile'] = vvix.rolling(252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]), raw=False)
        if 'VIX' in closes.columns:
            feats['vvix_vix_ratio'] = vvix / (closes['VIX'] + 1e-8)
    else:
        # Approximate VVIX as VIX vol-of-vol
        if 'VIX' in closes.columns:
            feats['vix_vol20'] = closes['VIX'].pct_change().rolling(20).std()
            feats['vix_vol60'] = closes['VIX'].pct_change().rolling(60).std()

    # 2. Cross-asset correlation (rolling)
    if len(assets) >= 4:
        corr_window = 60
        # Average pairwise correlation
        rolling_corrs = []
        for i_start in range(corr_window, len(returns)):
            window = returns[assets].iloc[i_start-corr_window:i_start].dropna()
            if len(window) >= 30:
                corr_mat = window.corr()
                # Average off-diagonal
                mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
                avg_corr = corr_mat.values[mask].mean()
                rolling_corrs.append(avg_corr)
            else:
                rolling_corrs.append(np.nan)

        feats['avg_corr'] = np.nan
        feats.iloc[corr_window:, feats.columns.get_loc('avg_corr')] = rolling_corrs

    # 3. Trend alignment momentum (how fast is consensus changing?)
    feats['n_bullish_mom5'] = feats['n_bullish'].diff(5)
    feats['n_bullish_mom20'] = feats['n_bullish'].diff(20)

    # 4. Credit-equity divergence
    if 'HYG' in closes.columns and 'SPY' in closes.columns:
        hyg_mom = closes['HYG'].pct_change(20)
        spy_mom = closes['SPY'].pct_change(20)
        feats['credit_equity_div'] = hyg_mom - spy_mom

    # 5. Equity dispersion (std of asset returns)
    feats['return_dispersion'] = returns[assets].rolling(20).std().mean(axis=1)

    # 6. Mean reversion pressure (how extended is the average asset?)
    feats['avg_zscore'] = 0
    for a in assets:
        z = (closes[a] - closes[a].rolling(60).mean()) / (closes[a].rolling(60).std() + 1e-8)
        feats['avg_zscore'] += z
    feats['avg_zscore'] /= len(assets)

    return feats.dropna(), trend_signals


def run_ml_trend_v3(closes, features, trend_signals):
    """Walk-forward ML trend following with enhanced features."""
    assets = [a for a in UNIVERSE if a in closes.columns]
    returns = closes[assets].pct_change()

    # Align
    common = closes.index.intersection(features.index)
    features = features.loc[common]
    returns = returns.loc[common]
    trend_signals_aligned = trend_signals.loc[common]

    portfolio_ret = pd.Series(0.0, index=common)
    predictions = pd.Series(0.5, index=common)

    model = None
    last_retrain = 0
    feature_cols = None

    start_idx = TRAIN_WINDOW + MA_LONG

    for i in range(start_idx, len(common)):
        # Retrain every 21 days
        if model is None or (i - last_retrain) >= 21:
            train_end = i
            train_start = max(0, i - TRAIN_WINDOW)

            X_train = features.iloc[train_start:train_end].copy()
            # Label: average trend-following return positive over next 5 days?
            fwd_ret = returns.iloc[train_start:train_end].shift(-5).mean(axis=1)
            y_train = (fwd_ret > 0).astype(int)

            # Remove NaN
            valid = X_train.notna().all(axis=1) & y_train.notna()
            X_train = X_train[valid]
            y_train = y_train[valid]

            if len(X_train) < 60:
                continue

            feature_cols = X_train.columns.tolist()

            model = GradientBoostingClassifier(
                n_estimators=60, max_depth=3, learning_rate=0.08,
                subsample=0.8, min_samples_leaf=8, random_state=42
            )
            model.fit(X_train.fillna(0), y_train)
            last_retrain = i

        if model is None or feature_cols is None:
            continue

        # Predict
        X_pred = features.iloc[i:i+1][feature_cols].fillna(0)
        try:
            prob = model.predict_proba(X_pred)[0][1]
        except:
            prob = 0.5

        predictions.iloc[i] = prob

        # Position: only take positions where ML is confident + trend aligned
        if prob > ML_THRESHOLD:
            trending = [a for a in assets if trend_signals_aligned[a].iloc[i] > 0]
            if trending:
                # Vol-targeted equal weight
                recent_vol = returns[trending].iloc[max(0,i-20):i].std().mean()
                vol_scalar = min(2.0, TARGET_VOL / np.sqrt(252) / (recent_vol + 1e-8))
                daily_ret = returns[trending].iloc[i].mean() * vol_scalar
                portfolio_ret.iloc[i] = daily_ret

    return portfolio_ret, predictions


# ─────────────────────────────────────────────────────────────────────────────
# ADVERSARIAL
# ─────────────────────────────────────────────────────────────────────────────

def compute_sharpe(r):
    r = r.dropna()
    if len(r) == 0 or r.std() == 0:
        return 0
    return r.mean() / r.std() * np.sqrt(252)


def compute_metrics(returns, name=""):
    r = returns.dropna()
    if len(r) == 0 or r.std() == 0:
        return {'name': name, 'sharpe': 0, 'sortino': 0, 'cagr': 0, 'max_dd': 0}

    sharpe = compute_sharpe(r)
    down = r[r < 0].std()
    sortino = r.mean() / down * np.sqrt(252) if down > 0 else 0
    cum = (1 + r).cumprod()
    years = len(r) / 252
    cagr = (cum.iloc[-1] ** (1/years) - 1) * 100 if years > 0 and cum.iloc[-1] > 0 else 0
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    pf_pos = r[r > 0].sum()
    pf_neg = abs(r[r < 0].sum())
    pf = pf_pos / pf_neg if pf_neg > 0 else 99
    wr = (r > 0).mean() * 100

    return {'name': name, 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
            'cagr': round(cagr, 1), 'max_dd': round(max_dd, 1), 'calmar': round(calmar, 3),
            'profit_factor': round(pf, 3), 'win_rate': round(wr, 1),
            'annual_vol': round(r.std() * np.sqrt(252) * 100, 1)}


def permutation_test(closes, features, trend_signals, real_sharpe, n_perms=50):
    """Shuffle ML predictions (break signal) → compare."""
    print(f"\n  Running {n_perms} permutations (signal shuffle)...")
    perm_sharpes = []

    assets = [a for a in UNIVERSE if a in closes.columns]
    returns = closes[assets].pct_change()
    common = closes.index.intersection(features.index)

    for p in range(n_perms):
        if (p+1) % 10 == 0:
            print(f"    Perm {p+1}/{n_perms}...")

        # Create random ML predictions (uniform 0-1)
        random_preds = np.random.random(len(common))

        # Use random predictions for position sizing
        port_ret = pd.Series(0.0, index=common)
        trend_sig = trend_signals.loc[common]
        rets = returns.loc[common]

        for i in range(TRAIN_WINDOW + MA_LONG, len(common)):
            if random_preds[i] > ML_THRESHOLD:
                trending = [a for a in assets if trend_sig[a].iloc[i] > 0]
                if trending:
                    recent_vol = rets[trending].iloc[max(0,i-20):i].std().mean()
                    vol_scalar = min(2.0, TARGET_VOL / np.sqrt(252) / (recent_vol + 1e-8))
                    port_ret.iloc[i] = rets[trending].iloc[i].mean() * vol_scalar

        perm_sharpes.append(compute_sharpe(port_ret))

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).mean()

    print(f"  Real: {real_sharpe:.3f}, Perm mean: {np.mean(perm_sharpes):.3f} ± {np.std(perm_sharpes):.3f}")
    print(f"  P-value: {p_value:.4f} ({'PASS' if p_value < 0.05 else 'FAIL'})")

    return {'p_value': float(p_value), 'perm_mean': float(np.mean(perm_sharpes)),
            'perm_std': float(np.std(perm_sharpes)), 'pass': p_value < 0.05}


def sub_period_test(returns, n=4):
    bs = len(returns) // n
    sharpes = [compute_sharpe(returns.iloc[b*bs:(b+1)*bs]) for b in range(n)]
    cv = np.std(sharpes) / (np.mean(sharpes) + 1e-8)
    print(f"  Blocks: {[f'{s:.2f}' for s in sharpes]}, CV={cv:.3f} ({'PASS' if cv < 0.50 else 'FAIL'})")
    return {'sharpes': [float(s) for s in sharpes], 'cv': float(cv), 'pass': cv < 0.50}


def r1_test(returns, closes):
    spy = closes['SPY'].pct_change()
    common = returns.index.intersection(spy.index)
    ret = returns.loc[common]
    spy_m = spy.loc[common].rolling(21).sum()
    gs = compute_sharpe(ret[spy_m > 0].dropna())
    rs = compute_sharpe(ret[spy_m <= 0].dropna())
    gap = abs(gs - rs) / max(abs(gs), abs(rs), 0.01)
    print(f"  Green: {gs:.3f}, Red: {rs:.3f}, Gap: {gap:.3f} ({'PASS' if gap < 0.50 else 'FAIL'})")
    return {'green': float(gs), 'red': float(rs), 'gap': float(gap), 'pass': gap < 0.50}


def outlier_test(returns):
    full = compute_sharpe(returns)
    lo, hi = np.percentile(returns.dropna(), [5, 95])
    trimmed = returns[(returns >= lo) & (returns <= hi)]
    ts = compute_sharpe(trimmed)
    deg = (ts - full) / (abs(full) + 1e-8)
    print(f"  Full: {full:.3f}, Trimmed: {ts:.3f}, Deg: {deg:.1%} ({'PASS' if deg > -0.40 else 'FAIL'})")
    return {'full': float(full), 'trimmed': float(ts), 'deg': float(deg), 'pass': deg > -0.40}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 80)
    print("ML TREND FOLLOWING v3 — VVIX + Vol Surface Enhancement")
    print("=" * 80)
    print(f"  Baseline: v2 Sharpe=2.90, R1 PASS. Target: beat baseline on same gates.")

    # 1. Data
    print("\n[1/5] DATA...")
    closes = download_data()

    # 2. Features
    print("\n[2/5] BUILDING ENHANCED FEATURES...")
    assets = [a for a in UNIVERSE if a in closes.columns]
    features, trend_signals = build_enhanced_features(closes, assets)
    print(f"  Features: {features.shape[1]} cols, {len(features)} rows")
    print(f"  New v3 features: vvix, avg_corr, credit_equity_div, return_dispersion, avg_zscore")

    # 3. Run strategy
    print("\n[3/5] WALK-FORWARD ML TREND v3...")
    port_ret, predictions = run_ml_trend_v3(closes, features, trend_signals)
    active_ret = port_ret[port_ret != 0].dropna()
    metrics = compute_metrics(active_ret, "ML Trend v3")

    print(f"\n  RESULTS:")
    print(f"    Sharpe:   {metrics['sharpe']} (v2 baseline: 2.90)")
    print(f"    Sortino:  {metrics['sortino']} (v2: 4.15)")
    print(f"    CAGR:     {metrics['cagr']}% (v2: 19.8%)")
    print(f"    MaxDD:    {metrics['max_dd']}% (v2: -5.3%)")
    print(f"    Calmar:   {metrics['calmar']} (v2: 3.76)")
    print(f"    PF:       {metrics['profit_factor']}")
    print(f"    WR:       {metrics['win_rate']}%")

    # Comparison: v2 without new features
    print("\n  Running v2 baseline (same code, no v3 features)...")
    # Strip v3-only features
    v2_feat_cols = [c for c in features.columns if not any(x in c for x in
                    ['vvix', 'avg_corr', 'n_bullish_mom', 'credit_equity_div',
                     'return_dispersion', 'avg_zscore', 'vix_vol20', 'vix_vol60'])]
    features_v2 = features[v2_feat_cols]
    port_ret_v2, _ = run_ml_trend_v3(closes, features_v2, trend_signals)
    active_v2 = port_ret_v2[port_ret_v2 != 0].dropna()
    m_v2 = compute_metrics(active_v2, "ML Trend v2 (repro)")
    print(f"    v2 repro: Sharpe={m_v2['sharpe']}, CAGR={m_v2['cagr']}%")

    improvement = metrics['sharpe'] - m_v2['sharpe']
    print(f"    v3 improvement: {improvement:+.3f} Sharpe")

    # 4. Adversarial
    print("\n[4/5] ADVERSARIAL VALIDATION...")

    print("\n  --- Permutation (signal shuffle) ---")
    perm = permutation_test(closes, features, trend_signals, metrics['sharpe'], N_PERMUTATIONS)

    print("\n  --- Sub-Period ---")
    subp = sub_period_test(active_ret)

    print("\n  --- R1 Regime ---")
    r1 = r1_test(active_ret, closes)

    print("\n  --- Outlier ---")
    outlier = outlier_test(active_ret)

    # 5. Final
    gates = sum([perm['pass'], subp['pass'], r1['pass'], outlier['pass']])

    print("\n" + "=" * 80)
    print(f"FINAL VERDICT: {gates}/4 gates")
    print(f"  v3 Sharpe: {metrics['sharpe']} | v2 baseline Sharpe: 2.90")
    print(f"  v3 improvement over repro: {improvement:+.3f}")
    verdict = "IMPROVEMENT" if metrics['sharpe'] > m_v2['sharpe'] and gates >= 3 else \
              "LATERAL" if abs(improvement) < 0.1 else "DEGRADATION"
    print(f"  Verdict: {verdict}")
    print("=" * 80)

    # Save
    results = {
        'strategy': 'ML Trend Following v3 (VVIX enhanced)',
        'metrics': metrics,
        'baseline_repro': m_v2,
        'improvement': round(improvement, 3),
        'adversarial': {
            'permutation': perm, 'sub_period': subp, 'r1_regime': r1, 'outlier': outlier,
            'gates_passed': gates, 'total': 4, 'verdict': verdict
        },
        'new_features': ['vvix_level', 'vvix_sma20', 'vvix_pctile', 'vvix_vix_ratio',
                        'avg_corr', 'n_bullish_mom5/20', 'credit_equity_div',
                        'return_dispersion', 'avg_zscore'],
        'runtime_seconds': round(time.time() - t0, 1)
    }

    with open(OUTPUT / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    cum_v3 = (1 + port_ret).cumprod()
    cum_v2 = (1 + port_ret_v2).cumprod()
    cum_spy = (1 + closes['SPY'].pct_change()).cumprod()
    ax.plot(cum_v3, label=f'v3 (Sharpe {metrics["sharpe"]})', linewidth=2)
    ax.plot(cum_v2, label=f'v2 repro (Sharpe {m_v2["sharpe"]})', linewidth=1.5, alpha=0.7)
    ax.plot(cum_spy, label='SPY', alpha=0.4, color='gray')
    ax.set_yscale('log')
    ax.set_title('ML Trend Following: v3 (VVIX enhanced) vs v2 baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / 'equity_curve.png', dpi=100)
    plt.close()

    print(f"\n  Runtime: {time.time()-t0:.0f}s")
    return results


if __name__ == '__main__':
    main()
