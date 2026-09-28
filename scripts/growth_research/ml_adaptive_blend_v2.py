#!/usr/bin/env python3
"""
ML Adaptive Blend v2 — Dynamic VIX/Trend Allocation (FAST)
============================================================
HYPOTHESIS: ML regime detector allocates dynamically between:
  - Component A: VIX-scaled leverage (UPRO/SPY/SHY based on VIX thresholds)
  - Component B: Multi-asset trend following (8 assets, risk-parity weighted)

ML allocator uses cross-asset features to decide which component to favor.
High regime confidence → VIX leverage. Low → trend following.

FIXES from v1:
  - Proper UPRO simulation (3x daily SPY returns - daily expense)
  - Vectorized operations instead of daily loops
  - Monthly retrain (not daily feature compute in loop)

Walk-forward: 504d sliding, monthly retrain (HC #0).
Fixed $100K, NO DCA (HC #713).
Full adversarial: permutation 100x, sub-period 4-block, outlier, R1 regime.
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
INITIAL_CAPITAL   = 100_000
TRAIN_WINDOW      = 504          # 2 years sliding
N_PERMUTATIONS    = 100
REBAL_COST_BPS    = 10

BASE   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_adaptive_blend"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────

TREND_ASSETS = ['SPY', 'TLT', 'GLD', 'UUP', 'EEM', 'VNQ', 'HYG', 'XLE']


def download_data():
    cache = BASE / "data" / "cache" / "adaptive_blend_v2_data.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists() and (time.time() - cache.stat().st_mtime) < 3600:
        print(f"  Using cached data")
        return pd.read_parquet(cache)

    print("  Downloading from yfinance...")
    tickers = list(set(TREND_ASSETS + ['SPY', 'SHY', 'QQQ', 'IWM', 'IEF', 'USO', 'CPER']))
    data = yf.download(tickers + ['^VIX'], start='2006-01-01', end='2026-07-18',
                       auto_adjust=True, progress=False)
    closes = data['Close'].copy()
    closes.columns = [c.replace('^', '') for c in closes.columns]
    closes = closes.ffill().dropna(how='all')
    closes.to_parquet(cache)
    print(f"  Data: {len(closes)} days, {closes.shape[1]} assets")
    return closes


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT A: VIX LEVERAGE (vectorized, proper 3x simulation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_vix_leverage_returns(closes):
    """
    Ultra-Aggressive VIX leverage: thresholds 12/16/22.
    VIX < 16: 3x SPY (simulated UPRO)
    VIX 16-22: 1x SPY
    VIX > 22: cash (SHY)

    Uses 3x daily SPY return minus 0.01%/day expense for UPRO simulation.
    """
    spy_ret = closes['SPY'].pct_change()
    shy_ret = closes['SHY'].pct_change() if 'SHY' in closes.columns else pd.Series(0.0001, index=closes.index)
    upro_ret = spy_ret * 3 - 0.0001  # Simulated 3x leveraged ETF (daily)

    vix = closes['VIX'].shift(1)  # Use previous day's VIX for signal

    # Allocation based on VIX thresholds
    daily_ret = pd.Series(0.0, index=closes.index)

    mask_lev = vix < 16
    mask_spy = (vix >= 16) & (vix < 22)
    mask_cash = vix >= 22

    daily_ret[mask_lev] = upro_ret[mask_lev]
    daily_ret[mask_spy] = spy_ret[mask_spy]
    daily_ret[mask_cash] = shy_ret[mask_cash]

    # Fill NaN with 0
    daily_ret = daily_ret.fillna(0)

    return daily_ret


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT B: MULTI-ASSET TREND FOLLOWING (vectorized)
# ─────────────────────────────────────────────────────────────────────────────

def compute_trend_returns(closes):
    """
    8-asset CTA trend following with risk-parity sizing.
    Signal: 50/200 MA crossover per asset.
    Sizing: inverse-vol weighted, target 10% portfolio vol.
    """
    assets = [a for a in TREND_ASSETS if a in closes.columns]
    returns = closes[assets].pct_change()

    # MA signals: +1 = long, 0 = flat
    ma_fast = closes[assets].rolling(50).mean()
    ma_slow = closes[assets].rolling(200).mean()
    signals = (ma_fast > ma_slow).astype(float)

    # Inverse-vol weights (rolling 60d realized vol)
    vol = returns.rolling(60).std()
    inv_vol = 1.0 / (vol + 1e-8)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)

    # Target 10% portfolio vol
    target_vol = 0.10 / np.sqrt(252)
    port_vol = (returns * signals * weights).sum(axis=1).rolling(60).std()
    vol_scalar = target_vol / (port_vol + 1e-8)
    vol_scalar = vol_scalar.clip(0.1, 2.0)  # Don't over/under-lever

    # Portfolio return
    raw_ret = (returns * signals * weights).sum(axis=1)
    scaled_ret = raw_ret * vol_scalar

    return scaled_ret.fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# ML REGIME ALLOCATOR (monthly retrain, fast features)
# ─────────────────────────────────────────────────────────────────────────────

def build_features_vectorized(closes):
    """Pre-compute all features as DataFrames (vectorized, fast)."""
    feats = pd.DataFrame(index=closes.index)

    # VIX features
    if 'VIX' in closes.columns:
        vix = closes['VIX']
        feats['vix_level'] = vix
        feats['vix_sma20'] = vix.rolling(20).mean()
        feats['vix_sma50'] = vix.rolling(50).mean()
        feats['vix_pctile'] = vix.rolling(252).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]), raw=False)
        feats['vix_slope5'] = vix.diff(5) / 5
        feats['vix_above_sma'] = (vix > vix.rolling(50).mean()).astype(int)

    # SPY trend features
    spy = closes['SPY']
    feats['spy_above_sma50'] = (spy > spy.rolling(50).mean()).astype(int)
    feats['spy_above_sma200'] = (spy > spy.rolling(200).mean()).astype(int)
    feats['spy_mom20'] = spy.pct_change(20)
    feats['spy_mom60'] = spy.pct_change(60)
    feats['spy_vol20'] = spy.pct_change().rolling(20).std()
    feats['spy_vol60'] = spy.pct_change().rolling(60).std()
    feats['spy_dd'] = spy / spy.rolling(50).max() - 1

    # Cross-asset
    if 'HYG' in closes.columns and 'TLT' in closes.columns:
        spread = closes['HYG'] / closes['TLT']
        feats['credit_z'] = (spread - spread.rolling(60).mean()) / (spread.rolling(60).std() + 1e-8)
        feats['credit_mom20'] = spread.pct_change(20)

    if 'IWM' in closes.columns:
        breadth = closes['IWM'] / closes['SPY']
        feats['breadth_mom20'] = breadth.pct_change(20)

    if 'GLD' in closes.columns:
        feats['gold_mom20'] = closes['GLD'].pct_change(20)

    if 'TLT' in closes.columns:
        feats['tlt_mom20'] = closes['TLT'].pct_change(20)

    # Trend alignment
    assets = [a for a in TREND_ASSETS if a in closes.columns]
    n_bullish = pd.DataFrame(index=closes.index)
    for a in assets:
        n_bullish[a] = (closes[a] > closes[a].rolling(50).mean()).astype(int)
    feats['n_bullish'] = n_bullish.sum(axis=1)
    feats['pct_bullish'] = feats['n_bullish'] / len(assets)

    return feats.dropna()


def run_adaptive_blend(closes, vix_ret, trend_ret, features,
                       threshold_high=0.65, threshold_low=0.40, label_horizon=21):
    """
    Monthly-retrain ML allocator.
    Label: 1 if VIX leverage outperformed trend over next label_horizon days.
    """
    # Align all series
    common_idx = closes.index.intersection(features.index)
    common_idx = common_idx.intersection(vix_ret.index).intersection(trend_ret.index)

    vix_ret = vix_ret.loc[common_idx]
    trend_ret = trend_ret.loc[common_idx]
    features = features.loc[common_idx]

    # Pre-compute forward returns for labeling
    vix_fwd = vix_ret.rolling(label_horizon).sum().shift(-label_horizon)
    trend_fwd = trend_ret.rolling(label_horizon).sum().shift(-label_horizon)
    labels = (vix_fwd > trend_fwd).astype(int)

    # Walk-forward
    portfolio_ret = pd.Series(0.0, index=common_idx)
    probs = pd.Series(0.5, index=common_idx)
    alloc_vix = pd.Series(0.5, index=common_idx)

    model = None
    retrain_dates = []

    start_idx = TRAIN_WINDOW + 200  # need enough data for features

    for i in range(start_idx, len(common_idx)):
        # Retrain monthly
        if model is None or (i - (retrain_dates[-1] if retrain_dates else 0)) >= 21:
            # Training data: [i - train_window, i - label_horizon] to avoid lookahead
            train_end = i - label_horizon
            train_start = max(0, train_end - TRAIN_WINDOW)

            X_train = features.iloc[train_start:train_end]
            y_train = labels.iloc[train_start:train_end]

            # Drop NaN labels
            valid = y_train.notna()
            X_train = X_train[valid]
            y_train = y_train[valid]

            if len(X_train) < 60:
                portfolio_ret.iloc[i] = 0.5 * vix_ret.iloc[i] + 0.5 * trend_ret.iloc[i]
                continue

            model = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10, random_state=42
            )
            model.fit(X_train.fillna(0), y_train.astype(int))
            retrain_dates.append(i)

        # Predict
        X_pred = features.iloc[i:i+1].fillna(0)
        try:
            prob = model.predict_proba(X_pred)[0][1]  # P(VIX better)
        except:
            prob = 0.5

        probs.iloc[i] = prob

        # Allocation
        if prob >= threshold_high:
            vix_wt = 0.80
        elif prob <= threshold_low:
            vix_wt = 0.20
        else:
            frac = (prob - threshold_low) / (threshold_high - threshold_low)
            vix_wt = 0.20 + frac * 0.60

        trend_wt = 1.0 - vix_wt
        alloc_vix.iloc[i] = vix_wt

        portfolio_ret.iloc[i] = vix_wt * vix_ret.iloc[i] + trend_wt * trend_ret.iloc[i]

    return portfolio_ret, probs, alloc_vix


# ─────────────────────────────────────────────────────────────────────────────
# METRICS & ADVERSARIAL
# ─────────────────────────────────────────────────────────────────────────────

def compute_sharpe(returns):
    r = returns.dropna()
    if len(r) == 0 or r.std() == 0:
        return 0
    return r.mean() / r.std() * np.sqrt(252)


def compute_metrics(returns, name="Strategy"):
    r = returns.dropna()
    if len(r) == 0 or r.std() == 0:
        return {'name': name, 'sharpe': 0, 'sortino': 0, 'cagr': 0, 'max_dd': 0, 'calmar': 0}

    sharpe = compute_sharpe(r)
    down = r[r < 0].std()
    sortino = r.mean() / down * np.sqrt(252) if down > 0 else 0

    cum = (1 + r).cumprod()
    years = len(r) / 252
    cagr = (cum.iloc[-1] ** (1/years) - 1) * 100 if years > 0 and cum.iloc[-1] > 0 else 0
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100

    pf_pos = r[r > 0].sum()
    pf_neg = abs(r[r < 0].sum())
    pf = pf_pos / pf_neg if pf_neg > 0 else 99
    wr = (r > 0).mean() * 100

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        'name': name, 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'cagr': round(cagr, 1), 'max_dd': round(max_dd, 1), 'calmar': round(calmar, 3),
        'profit_factor': round(pf, 3), 'win_rate': round(wr, 1),
        'total_return': round((cum.iloc[-1] - 1) * 100, 1),
        'annual_vol': round(r.std() * np.sqrt(252) * 100, 1)
    }


def permutation_test(closes, vix_ret, trend_ret, features, real_sharpe, n_perms=100):
    """Shuffle regime labels → re-run allocator → compare Sharpe."""
    print(f"\n  Running {n_perms} permutations...")
    perm_sharpes = []

    for p in range(n_perms):
        if (p+1) % 25 == 0:
            print(f"    Perm {p+1}/{n_perms}...")

        # Shuffle features (break signal-outcome relationship)
        features_shuffled = features.copy()
        # Block shuffle rows (preserve temporal structure within blocks)
        block_size = 21
        n_blocks = len(features_shuffled) // block_size
        block_indices = list(range(n_blocks))
        np.random.shuffle(block_indices)
        shuffled_rows = []
        for bi in block_indices:
            shuffled_rows.append(features_shuffled.iloc[bi*block_size:(bi+1)*block_size])
        features_shuffled = pd.concat(shuffled_rows).reset_index(drop=True)
        features_shuffled.index = features.index[:len(features_shuffled)]

        perm_ret, _, _ = run_adaptive_blend(closes, vix_ret, trend_ret, features_shuffled)
        active = perm_ret[perm_ret != 0]
        perm_sharpes.append(compute_sharpe(active))

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).mean()

    print(f"  Real Sharpe: {real_sharpe:.3f}")
    print(f"  Perm mean: {np.mean(perm_sharpes):.3f} ± {np.std(perm_sharpes):.3f}")
    print(f"  P-value: {p_value:.4f}")

    return {'p_value': float(p_value), 'perm_mean': float(np.mean(perm_sharpes)),
            'perm_std': float(np.std(perm_sharpes)), 'pass': p_value < 0.05}


def sub_period_test(returns, n_blocks=4):
    block_size = len(returns) // n_blocks
    block_sharpes = [compute_sharpe(returns.iloc[b*block_size:(b+1)*block_size]) for b in range(n_blocks)]
    cv = np.std(block_sharpes) / (np.mean(block_sharpes) + 1e-8)
    print(f"  Sub-period Sharpes: {[f'{s:.2f}' for s in block_sharpes]}")
    print(f"  CV: {cv:.3f} (PASS < 0.50)")
    return {'block_sharpes': [float(s) for s in block_sharpes], 'cv': float(cv), 'pass': cv < 0.50}


def r1_regime_test(returns, closes):
    spy_ret = closes['SPY'].pct_change()
    common = returns.index.intersection(spy_ret.index)
    ret = returns.loc[common]
    spy = spy_ret.loc[common]
    spy_monthly = spy.rolling(21).sum()

    green = ret[spy_monthly > 0].dropna()
    red = ret[spy_monthly <= 0].dropna()

    gs = compute_sharpe(green)
    rs = compute_sharpe(red)
    gap = abs(gs - rs) / max(abs(gs), abs(rs), 0.01)

    print(f"  Green Sharpe: {gs:.3f}, Red Sharpe: {rs:.3f}, Gap: {gap:.3f} (PASS < 0.50)")
    return {'green_sharpe': float(gs), 'red_sharpe': float(rs), 'gap': float(gap), 'pass': gap < 0.50}


def outlier_test(returns, pctile=95):
    full = compute_sharpe(returns)
    lo, hi = np.percentile(returns.dropna(), [100-pctile, pctile])
    trimmed = returns[(returns >= lo) & (returns <= hi)]
    ts = compute_sharpe(trimmed)
    deg = (ts - full) / (abs(full) + 1e-8)
    print(f"  Full: {full:.3f}, Trimmed: {ts:.3f}, Degradation: {deg:.1%} (PASS > -0.40)")
    return {'full_sharpe': float(full), 'trimmed_sharpe': float(ts), 'degradation': float(deg), 'pass': deg > -0.40}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 80)
    print("ML ADAPTIVE BLEND v2 — VIX Leverage + Trend Following")
    print("=" * 80)

    # 1. Data
    print("\n[1/6] DATA...")
    closes = download_data()

    # 2. Component strategies (vectorized)
    print("\n[2/6] COMPONENT STRATEGIES...")
    vix_ret = compute_vix_leverage_returns(closes)
    vix_active = vix_ret[vix_ret.index >= '2007-01-01']  # need enough data
    vm = compute_metrics(vix_active, "VIX Leverage")
    print(f"  VIX Leverage: Sharpe={vm['sharpe']}, CAGR={vm['cagr']}%, MaxDD={vm['max_dd']}%")

    trend_ret = compute_trend_returns(closes)
    trend_active = trend_ret[trend_ret.index >= '2007-06-01']
    tm = compute_metrics(trend_active, "Trend Following")
    print(f"  Trend Following: Sharpe={tm['sharpe']}, CAGR={tm['cagr']}%, MaxDD={tm['max_dd']}%")

    # Naive 50/50
    naive_ret = 0.5 * vix_ret + 0.5 * trend_ret
    naive_active = naive_ret[naive_ret.index >= '2008-01-01']
    nm = compute_metrics(naive_active, "Naive 50/50")
    print(f"  Naive 50/50: Sharpe={nm['sharpe']}")

    # SPY benchmark
    spy_ret = closes['SPY'].pct_change()
    spy_m = compute_metrics(spy_ret.dropna(), "SPY B&H")

    # 3. Features + ML Blend
    print("\n[3/6] ML ADAPTIVE BLEND...")
    features = build_features_vectorized(closes)
    print(f"  Features: {features.shape[1]} columns, {len(features)} rows")

    blend_ret, probs, alloc_vix = run_adaptive_blend(closes, vix_ret, trend_ret, features)
    blend_active = blend_ret[blend_ret != 0].dropna()
    bm = compute_metrics(blend_active, "ML Adaptive Blend")
    print(f"  ML Blend: Sharpe={bm['sharpe']}, CAGR={bm['cagr']}%, MaxDD={bm['max_dd']}%")
    print(f"            Sortino={bm['sortino']}, Calmar={bm['calmar']}")

    # 4. Adversarial
    print("\n[4/6] ADVERSARIAL VALIDATION...")

    print("\n  --- Permutation Test ---")
    perm = permutation_test(closes, vix_ret, trend_ret, features, bm['sharpe'], N_PERMUTATIONS)

    print("\n  --- Sub-Period Consistency ---")
    subp = sub_period_test(blend_active)

    print("\n  --- R1 Regime Test ---")
    r1 = r1_regime_test(blend_active, closes)

    print("\n  --- Outlier Sensitivity ---")
    outlier = outlier_test(blend_active)

    # 5. Summary
    gates = sum([perm['pass'], subp['pass'], r1['pass'], outlier['pass']])

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n  ML Adaptive Blend v2:")
    print(f"    Sharpe:  {bm['sharpe']}")
    print(f"    Sortino: {bm['sortino']}")
    print(f"    CAGR:    {bm['cagr']}%")
    print(f"    MaxDD:   {bm['max_dd']}%")
    print(f"    Calmar:  {bm['calmar']}")
    print(f"    PF:      {bm['profit_factor']}")
    print(f"    WR:      {bm['win_rate']}%")
    print(f"\n  Gates: {gates}/4")
    print(f"    Perm:    {'✅' if perm['pass'] else '❌'} (p={perm['p_value']:.4f})")
    print(f"    SubP:    {'✅' if subp['pass'] else '❌'} (CV={subp['cv']:.3f})")
    print(f"    R1:      {'✅' if r1['pass'] else '❌'} (gap={r1['gap']:.3f})")
    print(f"    Outlier: {'✅' if outlier['pass'] else '❌'} (deg={outlier['degradation']:.1%})")

    print(f"\n  Comparison:")
    print(f"    VIX Leverage:  Sharpe {vm['sharpe']}, CAGR {vm['cagr']}%")
    print(f"    Trend Follow:  Sharpe {tm['sharpe']}, CAGR {tm['cagr']}%")
    print(f"    Naive 50/50:   Sharpe {nm['sharpe']}")
    print(f"    ML Blend:      Sharpe {bm['sharpe']}, CAGR {bm['cagr']}%")
    print(f"    SPY B&H:       Sharpe {spy_m['sharpe']}")

    verdict = "PASS" if gates >= 3 and bm['sharpe'] > 2.5 else "NEEDS WORK" if gates >= 2 else "FAIL"
    print(f"\n  VERDICT: {verdict}")

    # 6. Save
    print("\n[6/6] SAVING...")
    results = {
        'strategy': 'ML Adaptive Blend v2',
        'metrics': {'blend': bm, 'vix_leverage': vm, 'trend_following': tm, 'naive_5050': nm, 'spy': spy_m},
        'adversarial': {
            'permutation': perm, 'sub_period': subp, 'r1_regime': r1, 'outlier': outlier,
            'gates_passed': gates, 'total': 4, 'verdict': verdict
        },
        'parameters': {
            'train_window': TRAIN_WINDOW, 'threshold_high': 0.65, 'threshold_low': 0.40,
            'vix_thresholds': [16, 22], 'trend_ma': [50, 200], 'label_horizon': 21
        },
        'runtime_seconds': round(time.time() - t0, 1)
    }

    with open(OUTPUT / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Equity curves
    cum_blend = (1 + blend_ret).cumprod() * INITIAL_CAPITAL
    cum_vix = (1 + vix_ret).cumprod() * INITIAL_CAPITAL
    cum_trend = (1 + trend_ret).cumprod() * INITIAL_CAPITAL
    cum_spy = (1 + spy_ret).cumprod() * INITIAL_CAPITAL

    ax = axes[0]
    ax.plot(cum_blend, label=f'ML Blend (Sharpe {bm["sharpe"]})', linewidth=2, color='purple')
    ax.plot(cum_vix, label=f'VIX Leverage (Sharpe {vm["sharpe"]})', alpha=0.7, color='green')
    ax.plot(cum_trend, label=f'Trend (Sharpe {tm["sharpe"]})', alpha=0.7, color='blue')
    ax.plot(cum_spy, label=f'SPY (Sharpe {spy_m["sharpe"]})', alpha=0.5, color='gray')
    ax.set_yscale('log')
    ax.set_title('ML Adaptive Blend v2 — Equity Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Allocation weights
    ax = axes[1]
    alloc_clean = alloc_vix[alloc_vix != 0.5].dropna()  # only where ML is active
    if len(alloc_clean) > 100:
        ax.fill_between(alloc_clean.index, 0, alloc_clean.values, alpha=0.6, label='VIX Weight', color='green')
        ax.fill_between(alloc_clean.index, alloc_clean.values, 1.0, alpha=0.6, label='Trend Weight', color='blue')
    ax.set_title('Dynamic Allocation')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # ML probability
    ax = axes[2]
    prob_clean = probs[probs != 0.5].dropna()
    if len(prob_clean) > 100:
        ax.plot(prob_clean.index, prob_clean.values, alpha=0.5, linewidth=0.5)
        ax.axhline(0.65, color='green', linestyle='--', label='VIX threshold')
        ax.axhline(0.40, color='blue', linestyle='--', label='Trend threshold')
    ax.set_title('ML Regime Probability (P(VIX better))')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / 'equity_curve.png', dpi=100, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s")
    return results


if __name__ == '__main__':
    results = main()
