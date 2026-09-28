#!/usr/bin/env python3
"""
v4.4 + ML Volatility Targeting — Combined Strategy
====================================================
Combines two proven strategies:

  1. v4.4 signal: VIX 20d SMA < 200d SMA AND VIX close < 25 → "risk on" (UPRO)
                  else → "risk off" (SPY)

  2. ML Vol Targeting: GBM predicts next-5d realized vol, then sizes UPRO
     allocation inversely to predicted vol (target_vol / pred_vol, capped
     0.3–1.5). When v4.4 says SPY → stay 100% SPY. When v4.4 says UPRO →
     let vol-targeting decide how much UPRO vs SPY.

Walk-forward: SLIDING 252d window (HC #0).
Fixed capital: $100K, NO DCA (HC #713).
Adversarial validation: permutation (100), sub-period (4 blocks), outlier
removal, R1 regime test (HC #705).

Benchmarks compared (5 strategies):
  (a) SPY buy & hold
  (b) v4.4 pure (binary UPRO/SPY)
  (c) ML vol targeting pure (all-UPRO universe, no v4.4 gate)
  (d) v4.4 + ML vol targeting COMBINED  ← main strategy
  (e) v4.4 + simple realized-vol targeting

Output: /home/jupiter/Lvl3Quant/output/v44_vol_targeted/results.json
"""

import os
import json
import time
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL     = 100_000
TARGET_VOL          = 0.15        # 15% annualized — tune-able
TRAIN_WINDOW        = 252         # sliding window (HC #0)
REBAL_COST_BPS      = 5           # one-way switching cost
N_PERMUTATIONS      = 100
UPRO_WEIGHT_MIN     = 0.30        # floor when v4.4 says UPRO
UPRO_WEIGHT_MAX     = 1.50        # cap (allows modest leverage via UPRO)
VIX_THRESHOLD       = 25.0        # v4.4 VIX level gate
VIX_SMA_SHORT       = 20
VIX_SMA_LONG        = 200

BASE   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "v44_vol_targeted"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Top features from completed ml_vol_targeting run
TOP_FEATURES = [
    'spy_rvol_10d', 'spy_rvol_20d', 'vix_level', 'vix_rv_ratio',
    'vol_of_vol_20d', 'uup_mom_20d', 'gld_vol_20d', 'xlf_vol_20d',
    'spy_drawdown', 'spy_kurt_20d', 'spy_skew_20d', 'spy_tlt_corr_20d',
    'spy_abs_ret_5d_avg',
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: DATA
# ─────────────────────────────────────────────────────────────────────────────
def download_data():
    print("=" * 80)
    print("v4.4 + ML VOLATILITY TARGETING — COMBINED STRATEGY")
    print("=" * 80)
    print(f"\nRun started: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Capital: ${INITIAL_CAPITAL:,} fixed, NO DCA | Target vol: {TARGET_VOL:.0%}")

    print("\n[1/9] Downloading data...")
    cache = OUTPUT / "raw_data.parquet"

    tickers_map = {
        'SPY': 'SPY', 'UPRO': 'UPRO',
        'VIX': '^VIX',
        'GLD': 'GLD', 'TLT': 'TLT', 'SHY': 'SHY',
        'UUP': 'UUP', 'XLF': 'XLF', 'XLU': 'XLU',
        'IWM': 'IWM', 'QQQ': 'QQQ',
    }

    if cache.exists():
        prices = pd.read_parquet(cache)
        print(f"  Loaded from cache: {prices.shape}")
    else:
        raw = yf.download(
            list(tickers_map.values()),
            start='2008-01-01', end='2026-07-18',
            auto_adjust=True, progress=False
        )
        prices = raw['Close'].copy()
        inv = {v: k for k, v in tickers_map.items()}
        prices.columns = [inv.get(c, c) for c in prices.columns]
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
        prices.to_parquet(cache)
        print(f"  Downloaded and cached: {prices.shape}")

    prices = prices.ffill()
    prices = prices.dropna(subset=['SPY', 'VIX', 'UPRO'])
    prices = prices[prices.index >= '2010-01-01']
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    print(f"  Period: {prices.index[0].date()} → {prices.index[-1].date()}, {len(prices)} days")
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def build_features(prices):
    print("\n[2/9] Building features...")
    spy_ret = prices['SPY'].pct_change()

    feat = pd.DataFrame(index=prices.index)

    # Realized vol
    for w in [5, 10, 20, 60]:
        feat[f'spy_rvol_{w}d'] = spy_ret.rolling(w).std() * np.sqrt(252)

    # VIX features
    feat['vix_level']    = prices['VIX']
    feat['vix_5d_chg']   = prices['VIX'].pct_change(5)
    feat['vix_20d_chg']  = prices['VIX'].pct_change(20)
    feat['vix_z_20d']    = ((prices['VIX'] - prices['VIX'].rolling(20).mean())
                            / prices['VIX'].rolling(20).std())
    feat['vix_sma_20']   = prices['VIX'].rolling(VIX_SMA_SHORT).mean()
    feat['vix_sma_200']  = prices['VIX'].rolling(VIX_SMA_LONG).mean()

    # VIX vs realized vol ratio (term structure proxy)
    rv20 = spy_ret.rolling(20).std() * np.sqrt(252) * 100
    feat['vix_rv_ratio'] = prices['VIX'] / rv20.replace(0, np.nan)

    # SPY trend / momentum
    for w in [5, 10, 20, 60]:
        feat[f'spy_mom_{w}d'] = prices['SPY'].pct_change(w)
    feat['spy_sma_20']   = prices['SPY'].rolling(20).mean()
    feat['spy_sma_200']  = prices['SPY'].rolling(200).mean()
    feat['spy_above_200'] = (prices['SPY'] > feat['spy_sma_200']).astype(float)

    # Drawdown from all-time high
    spy_peak = prices['SPY'].cummax()
    feat['spy_drawdown'] = (prices['SPY'] - spy_peak) / spy_peak

    # Higher moments
    feat['spy_skew_20d'] = spy_ret.rolling(20).skew()
    feat['spy_kurt_20d'] = spy_ret.rolling(20).kurt()
    feat['spy_abs_ret_5d_avg'] = spy_ret.abs().rolling(5).mean()

    # Vol of vol
    feat['vol_of_vol_20d'] = feat['spy_rvol_20d'].rolling(20).std()

    # Cross-asset
    for asset in ['GLD', 'TLT', 'UUP', 'XLF', 'XLU', 'IWM', 'QQQ']:
        if asset in prices.columns:
            ret = prices[asset].pct_change()
            feat[f'{asset.lower()}_mom_20d'] = prices[asset].pct_change(20)
            feat[f'{asset.lower()}_vol_20d'] = ret.rolling(20).std() * np.sqrt(252)

    # SPY-TLT correlation (flight to quality)
    if 'TLT' in prices.columns:
        feat['spy_tlt_corr_20d'] = spy_ret.rolling(20).corr(prices['TLT'].pct_change())
    if 'GLD' in prices.columns:
        feat['spy_gld_corr_20d'] = spy_ret.rolling(20).corr(prices['GLD'].pct_change())

    feat = feat.replace([np.inf, -np.inf], np.nan)
    print(f"  Features built: {feat.shape[1]} columns")
    return feat, spy_ret


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: v4.4 SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
def compute_v44_signal(prices, feat):
    """
    v4.4 signal (pure binary):
      risk_on  = VIX_SMA_20 < VIX_SMA_200  AND  VIX_close < VIX_THRESHOLD
      risk_off = otherwise → SPY
    Signal is computed on day t, executed on day t+1 (shifted by 1).
    Returns a boolean Series: True = UPRO, False = SPY
    """
    sma_short = feat['vix_sma_20']
    sma_long  = feat['vix_sma_200']
    vix       = prices['VIX']

    risk_on = (sma_short < sma_long) & (vix < VIX_THRESHOLD)
    # Shift by 1: signal from today, execute tomorrow
    return risk_on.shift(1).fillna(False).astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: WALK-FORWARD ML TRAINING (SLIDING 252d — HC #0)
# ─────────────────────────────────────────────────────────────────────────────
def run_walk_forward(prices, feat, spy_ret):
    """
    SLIDING 252d walk-forward for GBM vol predictor.
    Predicts next-5d realized vol of SPY.
    Uses only TOP_FEATURES (from prior run feature importance).
    Returns daily predicted vol series (aligned to trade date).
    """
    print("\n[3/9] Walk-forward ML training (SLIDING 252d)...")

    # Target: next 5-day realized vol
    fwd_rvol = spy_ret.rolling(5).std().shift(-5) * np.sqrt(252)
    fwd_rvol.name = 'fwd_rvol_5d'

    # Keep only features we need (subset intersection)
    avail_feats = [f for f in TOP_FEATURES if f in feat.columns]
    missing = set(TOP_FEATURES) - set(avail_feats)
    if missing:
        print(f"  Warning: missing features {missing} — skipping")
    X_all = feat[avail_feats].copy()

    # Common valid index
    valid_idx = (X_all.notna().all(axis=1) & fwd_rvol.notna())
    X_all  = X_all[valid_idx]
    y_all  = fwd_rvol[valid_idx]

    print(f"  Valid samples: {len(X_all)} | Features: {len(avail_feats)}")
    print(f"  Period: {X_all.index[0].date()} → {X_all.index[-1].date()}")

    predictions = pd.Series(dtype=float, index=X_all.index, name='pred_vol')
    n_folds = 0

    for i in range(TRAIN_WINDOW, len(X_all) - 1):
        train_s = i - TRAIN_WINDOW
        X_tr = X_all.iloc[train_s:i]
        y_tr = y_all.iloc[train_s:i]
        X_te = X_all.iloc[i:i+1]

        # Sanity check
        if y_tr.isna().any() or X_tr.isna().any().any():
            continue
        if len(y_tr) < 100:
            continue

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
        model.fit(X_tr, y_tr)
        predictions.iloc[i] = model.predict(X_te)[0]
        n_folds += 1

        if n_folds % 500 == 0:
            valid_pred = predictions.dropna()
            valid_act  = y_all.reindex(valid_pred.index).dropna()
            idx = valid_pred.index.intersection(valid_act.index)
            r2 = r2_score(valid_act.loc[idx], valid_pred.loc[idx]) if len(idx) > 10 else float('nan')
            print(f"  Fold {n_folds}: R²={r2:.4f}")

    predictions = predictions.dropna()
    actuals     = y_all.reindex(predictions.index).dropna()
    common      = predictions.index.intersection(actuals.index)
    r2   = r2_score(actuals.loc[common], predictions.loc[common])
    rmse = np.sqrt(mean_squared_error(actuals.loc[common], predictions.loc[common]))
    corr = actuals.loc[common].corr(predictions.loc[common])

    print(f"  Folds: {n_folds} | R²={r2:.4f} | RMSE={rmse:.4f} | Corr={corr:.4f}")
    print(f"  Pred vol — min: {predictions.min():.1%}  median: {predictions.median():.1%}  "
          f"max: {predictions.max():.1%}")

    # Feature importance from LAST model
    fi = pd.Series(model.feature_importances_, index=avail_feats).sort_values(ascending=False)
    print("\n  Top features (last fold):")
    for feat_name, imp in fi.head(10).items():
        print(f"    {feat_name}: {imp:.4f}")

    return predictions, {'r2': float(r2), 'rmse': float(rmse), 'corr': float(corr),
                         'n_folds': n_folds,
                         'feature_importance': {k: float(v) for k, v in fi.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: COMPUTE VOL-BASED UPRO WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def compute_vol_weights(predictions, method='ml'):
    """
    Convert predicted (or realized) vol to UPRO allocation weight.
    UPRO vol ≈ 3× SPY vol (leverage factor).
    weight = target_vol / (pred_spy_vol * 3), clipped to [UPRO_WEIGHT_MIN, UPRO_WEIGHT_MAX]
    """
    upro_vol_est = predictions * 3
    weights = (TARGET_VOL / upro_vol_est).clip(UPRO_WEIGHT_MIN, UPRO_WEIGHT_MAX)
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: SIMULATE STRATEGIES (NO DCA, NEXT-DAY EXECUTION)
# ─────────────────────────────────────────────────────────────────────────────
def simulate(prices, spy_ret, v44_signal, upro_weight_series=None,
             mode='v44_pure', warmup=TRAIN_WINDOW + 5):
    """
    Simulate one strategy from prices.

    Modes:
      'spy_bh'          : 100% SPY buy & hold
      'v44_pure'        : v4.4 binary (UPRO or SPY, no sizing)
      'ml_vol_pure'     : ML vol targeting over ALL-UPRO universe (no v4.4 gate)
      'v44_ml_combined' : v4.4 gate + ML vol sizing when risk-on
      'v44_simple_vol'  : v4.4 gate + simple realized-vol sizing when risk-on

    upro_weight_series: precomputed daily UPRO weight Series (for vol-targeting modes)
    NO DCA. Next-day execution (all signals shifted 1 day already).
    """
    upro_ret = prices['UPRO'].pct_change()
    spy_r    = prices['SPY'].pct_change()

    # Align to common index post-warmup
    start_idx = warmup
    idx = prices.index[start_idx:]

    v44   = v44_signal.reindex(idx).fillna(False)
    s_ret = spy_r.reindex(idx).fillna(0)
    u_ret = upro_ret.reindex(idx).fillna(0)

    capital = float(INITIAL_CAPITAL)
    prev_label = None
    n_switches = 0
    vals = []

    # For simple realized vol: use 20d realized, shifted 1d (no lookahead)
    rv20_shifted = (spy_ret.rolling(20).std() * np.sqrt(252)).shift(1).reindex(idx)

    for date in idx:
        # Decide weight
        if mode == 'spy_bh':
            upro_w = 0.0   # 0% UPRO, 100% SPY
            label  = 'SPY'

        elif mode == 'v44_pure':
            upro_w = 1.0 if v44.loc[date] else 0.0
            label  = 'UPRO' if v44.loc[date] else 'SPY'

        elif mode == 'ml_vol_pure':
            if upro_weight_series is not None and date in upro_weight_series.index:
                upro_w = float(upro_weight_series.loc[date])
            else:
                upro_w = 0.0
            label = f'UPRO_{upro_w:.2f}'

        elif mode == 'v44_ml_combined':
            if not v44.loc[date]:
                upro_w = 0.0
                label  = 'SPY'
            else:
                if upro_weight_series is not None and date in upro_weight_series.index:
                    upro_w = float(upro_weight_series.loc[date])
                else:
                    upro_w = 1.0   # fallback: full UPRO if no prediction
                label  = f'UPRO_{upro_w:.2f}'

        elif mode == 'v44_simple_vol':
            if not v44.loc[date]:
                upro_w = 0.0
                label  = 'SPY'
            else:
                rv = rv20_shifted.loc[date] if not pd.isna(rv20_shifted.loc[date]) else 0.20
                upro_vol_est = rv * 3
                upro_w = float(np.clip(TARGET_VOL / max(upro_vol_est, 0.01),
                                       UPRO_WEIGHT_MIN, UPRO_WEIGHT_MAX))
                label = f'UPRO_{upro_w:.2f}'
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Switching cost (simplified: if label changes materially)
        simplified_label = 'SPY' if upro_w == 0 else 'UPRO'
        if prev_label is not None and simplified_label != prev_label:
            n_switches += 1
            capital *= (1 - REBAL_COST_BPS / 10_000)
        prev_label = simplified_label

        # Return
        day_ret = upro_w * u_ret.loc[date] + (1 - upro_w) * s_ret.loc[date]
        capital *= (1 + day_ret)
        vals.append(capital)

    series = pd.Series(vals, index=idx, name=mode)
    return {'values': series, 'n_switches': n_switches}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: METRICS
# ─────────────────────────────────────────────────────────────────────────────
def calc_metrics(result, label):
    vals = result['values'].dropna()
    if len(vals) < 50:
        return {'label': label}

    rets = vals.pct_change().dropna()
    ann_ret = (1 + rets).prod() ** (252 / len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0

    neg = rets[rets < 0]
    downside_vol = neg.std() * np.sqrt(252) if len(neg) > 10 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    peak   = vals.cummax()
    dd     = (vals - peak) / peak
    max_dd = float(dd.min())

    years  = (vals.index[-1] - vals.index[0]).days / 365.25
    calmar = (ann_ret / abs(max_dd)) if max_dd != 0 else 0

    gross_pos = rets[rets > 0].sum()
    gross_neg = abs(rets[rets < 0].sum())
    pf = gross_pos / gross_neg if gross_neg > 0 else float('inf')
    wr = float((rets > 0).mean())

    sw_yr = result['n_switches'] / years if years > 0 else 0

    return {
        'label':        label,
        'sharpe':       float(sharpe),
        'sortino':      float(sortino),
        'cagr':         float(ann_ret),
        'ann_vol':      float(ann_vol),
        'max_dd':       float(max_dd),
        'calmar':       float(calmar),
        'pf':           float(pf),
        'wr':           float(wr),
        'final_value':  float(vals.iloc[-1]),
        'n_switches':   int(result['n_switches']),
        'switches_yr':  float(sw_yr),
    }


def print_metrics_table(metrics_list):
    print(f"\n  {'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} "
          f"{'MaxDD':>8} {'Calmar':>7} {'PF':>6} {'WR':>6} {'Sw/yr':>6}")
    print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*6} {'-'*6} {'-'*6}")
    for m in metrics_list:
        if 'sharpe' not in m:
            continue
        print(f"  {m['label']:<30} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr']:>6.1%} {m['max_dd']:>7.1%} {m['calmar']:>7.3f} "
              f"{m['pf']:>6.2f} {m['wr']:>5.1%} {m['switches_yr']:>6.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: ADVERSARIAL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def adversarial_permutation(rets, upro_w, spy_r, upro_r, label, n_perms=N_PERMUTATIONS):
    """Gate 1: Shuffle vol-targeting weights → does timing matter?"""
    print(f"\n  [Gate 1] Permutation test ({n_perms} iters)...")
    ann_r = (1 + rets).prod() ** (252 / len(rets)) - 1
    ann_v = rets.std() * np.sqrt(252)
    real_sharpe = ann_r / ann_v if ann_v > 0 else 0

    spy_aligned  = spy_r.reindex(rets.index).fillna(0)
    upro_aligned = upro_r.reindex(rets.index).fillna(0)
    w_arr = upro_w.reindex(rets.index).fillna(0).values

    perm_sharpes = []
    for p in range(n_perms):
        np.random.seed(p + 7777)
        shuf_w = np.random.permutation(w_arr)
        pr = shuf_w * upro_aligned.values + (1 - shuf_w) * spy_aligned.values
        pr = pd.Series(pr, index=rets.index)
        p_ann = (1 + pr).prod() ** (252 / len(pr)) - 1
        p_vol = pr.std() * np.sqrt(252)
        perm_sharpes.append(p_ann / p_vol if p_vol > 0 else 0)

    perm_sharpes = np.array(perm_sharpes)
    p_val = float((perm_sharpes >= real_sharpe).mean())
    passed = p_val < 0.05
    print(f"    Real={real_sharpe:.3f}, Perm mean={np.mean(perm_sharpes):.3f}, "
          f"p={p_val:.4f} → {'PASS' if passed else 'FAIL'}")
    return {'real_sharpe': float(real_sharpe), 'perm_mean': float(np.mean(perm_sharpes)),
            'p_value': p_val, 'pass': bool(passed)}


def adversarial_subperiod(rets, spy_r, label, n_blocks=4):
    """Gate 2: 4-block sub-period consistency."""
    print(f"\n  [Gate 2] Sub-period consistency ({n_blocks} blocks)...")
    n = len(rets)
    block_size = n // n_blocks
    block_sharpes = []
    for i in range(n_blocks):
        s = i * block_size
        e = (i + 1) * block_size if i < n_blocks - 1 else n
        sub = rets.iloc[s:e]
        spy_sub = spy_r.reindex(sub.index).dropna()
        sub_ann = (1 + sub).prod() ** (252 / len(sub)) - 1
        sub_vol = sub.std() * np.sqrt(252)
        sh = sub_ann / sub_vol if sub_vol > 0 else 0
        sub_cum = (1 + sub).cumprod()
        sub_dd  = ((sub_cum - sub_cum.cummax()) / sub_cum.cummax()).min()
        block_sharpes.append(sh)
        yr_range = f"{sub.index[0].strftime('%Y-%m')}→{sub.index[-1].strftime('%Y-%m')}"
        print(f"    Block {i+1} ({yr_range}): Sharpe={sh:.3f}, MaxDD={sub_dd:.1%}")

    pos = sum(1 for s in block_sharpes if s > 0)
    passed = pos >= 3
    print(f"    Positive Sharpe blocks: {pos}/{n_blocks} → {'PASS' if passed else 'FAIL'}")
    return {'block_sharpes': [float(s) for s in block_sharpes],
            'positive_blocks': pos, 'total_blocks': n_blocks, 'pass': bool(passed)}


def adversarial_outlier(rets, label):
    """Gate 3: Outlier removal — trim top/bottom 1%."""
    print(f"\n  [Gate 3] Outlier removal...")
    ann_r = (1 + rets).prod() ** (252 / len(rets)) - 1
    ann_v = rets.std() * np.sqrt(252)
    full_sharpe = ann_r / ann_v if ann_v > 0 else 0

    p1, p99 = rets.quantile(0.01), rets.quantile(0.99)
    trim = rets[(rets >= p1) & (rets <= p99)]
    t_ann = (1 + trim).prod() ** (252 / len(trim)) - 1
    t_vol = trim.std() * np.sqrt(252)
    trim_sharpe = t_ann / t_vol if t_vol > 0 else 0

    passed = trim_sharpe > 0
    ratio  = trim_sharpe / full_sharpe if full_sharpe != 0 else 0
    print(f"    Full={full_sharpe:.3f}, Trimmed={trim_sharpe:.3f} "
          f"(ratio={ratio:.2f}) → {'PASS' if passed else 'FAIL'}")
    return {'full_sharpe': float(full_sharpe), 'trimmed_sharpe': float(trim_sharpe),
            'ratio': float(ratio), 'pass': bool(passed)}


def adversarial_regime_r1(rets, spy_r, label):
    """Gate 4: R1 regime test — |Sharpe_green - Sharpe_red| / max < 0.50."""
    print(f"\n  [Gate 4] R1 regime test...")
    spy_aligned = spy_r.reindex(rets.index).dropna()
    common = rets.index.intersection(spy_aligned.index)
    r = rets.loc[common]
    s = spy_aligned.loc[common]

    green = r[s > 0]
    red   = r[s < 0]

    def _sharpe(x):
        if len(x) < 20 or x.std() == 0:
            return 0.0
        ann = (1 + x).prod() ** (252 / len(x)) - 1
        return ann / (x.std() * np.sqrt(252))

    sg = _sharpe(green)
    sr = _sharpe(red)
    denom = max(abs(sg), abs(sr), 0.01)
    delta = abs(sg - sr) / denom
    passed = delta < 0.50
    print(f"    Sharpe green={sg:.3f}, red={sr:.3f}, delta={delta:.3f} "
          f"(threshold 0.50) → {'PASS' if passed else 'FAIL'}")
    return {'sharpe_green': float(sg), 'sharpe_red': float(sr),
            'delta': float(delta), 'pass': bool(passed)}


def run_adversarial(result, v44_signal, predictions, prices, spy_ret, strategy_label,
                    mode, warmup=TRAIN_WINDOW + 5):
    """Run all 4 adversarial gates on the combined strategy."""
    print(f"\n{'#'*70}")
    print(f"# ADVERSARIAL VALIDATION: {strategy_label}")
    print(f"{'#'*70}")

    rets     = result['values'].pct_change().dropna()
    spy_r    = prices['SPY'].pct_change()
    upro_r   = prices['UPRO'].pct_change()

    # Build upro_w for permutation test
    if mode in ('v44_ml_combined', 'ml_vol_pure'):
        upro_w = compute_vol_weights(predictions)
    elif mode == 'v44_simple_vol':
        rv20_s = (spy_ret.rolling(20).std() * np.sqrt(252)).shift(1)
        upro_w = (TARGET_VOL / (rv20_s * 3)).clip(UPRO_WEIGHT_MIN, UPRO_WEIGHT_MAX)
    else:
        upro_w = pd.Series(
            np.where(v44_signal.reindex(rets.index).fillna(False), 1.0, 0.0),
            index=rets.index
        )

    # For combined mode, zero out weight when v4.4 is off
    if mode == 'v44_ml_combined':
        v44_r = v44_signal.reindex(rets.index).fillna(False)
        upro_w = upro_w.reindex(rets.index).fillna(0)
        upro_w[~v44_r] = 0.0

    gates = {}
    gates['permutation'] = adversarial_permutation(
        rets, upro_w.reindex(rets.index).fillna(0), spy_r, upro_r, strategy_label)
    gates['subperiod']   = adversarial_subperiod(rets, spy_r, strategy_label)
    gates['outlier']     = adversarial_outlier(rets, strategy_label)
    gates['regime_r1']   = adversarial_regime_r1(rets, spy_r, strategy_label)

    n_pass = sum(1 for g in gates.values() if g.get('pass', False))
    n_tot  = len(gates)
    overall = n_pass >= 3

    print(f"\n  {'='*50}")
    print(f"  ADVERSARIAL SUMMARY: {strategy_label}")
    print(f"  {'='*50}")
    for gname, g in gates.items():
        print(f"    {gname:<20}: {'PASS' if g.get('pass') else 'FAIL'}")
    print(f"\n    TOTAL: {n_pass}/{n_tot} | OVERALL: {'*** PASS ***' if overall else '--- FAIL ---'}")

    gates['n_pass'] = n_pass
    gates['n_total'] = n_tot
    gates['overall_pass'] = bool(overall)
    return gates


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def create_charts(all_results, all_metrics, prices, v44_signal, predictions, spy_ret):
    print("\n[8/9] Creating charts...")

    fig, axes = plt.subplots(4, 1, figsize=(16, 22))
    colors = {
        '(a) SPY Buy & Hold':           '#888888',
        '(b) v4.4 Pure':                '#2196F3',
        '(c) ML Vol Pure':              '#FF9800',
        '(d) v4.4 + ML Vol Combined':   '#4CAF50',
        '(e) v4.4 + Simple Vol':        '#E91E63',
    }

    # 1. Equity curves
    ax = axes[0]
    for label, res in all_results.items():
        m   = {x['label']: x for x in all_metrics}[label] if any(x['label'] == label for x in all_metrics) else {}
        c   = colors.get(label, 'steelblue')
        lw  = 2.5 if 'Combined' in label else 1.5
        sh  = m.get('sharpe', 0)
        dd  = m.get('max_dd', 0)
        ax.plot(res['values'].index, res['values'],
                label=f"{label} (Sh={sh:.2f}, DD={dd*100:.0f}%)",
                color=c, linewidth=lw, alpha=0.9)
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title('v4.4 + ML Vol Targeting — Equity Curves ($100K, No DCA)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 2. v4.4 signal + predicted vol overlay
    ax = axes[1]
    v44_pct = v44_signal.rolling(20).mean() * 100
    ax.plot(v44_pct.index, v44_pct, color='#2196F3', alpha=0.7, lw=0.8,
            label='v4.4 Risk-On % (20d roll)')
    ax2 = ax.twinx()
    pred_pct = predictions * 100
    ax2.plot(pred_pct.index, pred_pct, color='#FF5722', alpha=0.6, lw=0.8,
             label='ML Pred Vol % (annualized)')
    ax2.axhline(TARGET_VOL * 100, color='green', linestyle='--', lw=1.0,
                label=f'Target vol {TARGET_VOL:.0%}')
    ax.set_ylabel('v4.4 Risk-On %')
    ax2.set_ylabel('ML Predicted SPY Vol (%)')
    ax.set_title('v4.4 Signal vs ML Predicted Volatility')
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. UPRO allocation weight for combined strategy
    ax = axes[2]
    upro_w = compute_vol_weights(predictions)
    v44_r  = v44_signal.reindex(upro_w.index).fillna(False)
    effective_w = upro_w.copy()
    effective_w[~v44_r] = 0.0
    ax.fill_between(effective_w.index, 0, effective_w,
                    where=v44_r.reindex(effective_w.index, fill_value=False),
                    color='#4CAF50', alpha=0.6, label='UPRO weight (v4.4 ON)')
    ax.fill_between(effective_w.index, 0, 0,
                    where=~v44_r.reindex(effective_w.index, fill_value=True),
                    color='#888888', alpha=0.3, label='v4.4 OFF → SPY')
    ax.axhline(1.0, color='blue', linestyle='--', lw=0.8, label='100% UPRO line')
    ax.set_ylim(0, UPRO_WEIGHT_MAX + 0.2)
    ax.set_ylabel('UPRO Allocation Weight')
    ax.set_title('Combined Strategy: Effective UPRO Weight Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Drawdowns
    ax = axes[3]
    key_strats = [
        '(a) SPY Buy & Hold',
        '(b) v4.4 Pure',
        '(d) v4.4 + ML Vol Combined',
    ]
    for label in key_strats:
        if label not in all_results:
            continue
        vals = all_results[label]['values']
        dd = (vals - vals.cummax()) / vals.cummax()
        c  = colors.get(label, 'steelblue')
        ax.fill_between(dd.index, 0, dd * 100, alpha=0.3, color=c, label=label)
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdown Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = OUTPUT / 'strategy_overview.png'
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  Saved strategy_overview.png")

    # Rolling Sharpe chart
    fig, ax = plt.subplots(figsize=(14, 5))
    for label in [
        '(a) SPY Buy & Hold', '(b) v4.4 Pure',
        '(d) v4.4 + ML Vol Combined', '(e) v4.4 + Simple Vol',
    ]:
        if label not in all_results:
            continue
        rets = all_results[label]['values'].pct_change().dropna()
        rs   = rets.rolling(252).mean() / rets.rolling(252).std() * np.sqrt(252)
        ax.plot(rs.index, rs, color=colors.get(label, 'gray'), lw=1.2,
                label=label, alpha=0.85)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_ylabel('Rolling 1yr Sharpe')
    ax.set_title('Rolling 1-Year Sharpe Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / 'rolling_sharpe.png', dpi=150)
    plt.close()
    print(f"  Saved rolling_sharpe.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # 1. Data
    prices = download_data()

    # 2. Features
    feat, spy_ret = build_features(prices)

    # 3. v4.4 signal
    print("\n[4/9] Computing v4.4 signal...")
    v44_signal = compute_v44_signal(prices, feat)
    pct_risk_on = v44_signal.mean()
    print(f"  v4.4 risk-on: {pct_risk_on:.1%} of days")
    print(f"  VIX threshold: {VIX_THRESHOLD}, SMA {VIX_SMA_SHORT}/{VIX_SMA_LONG}")

    # 4. Walk-forward ML training
    predictions, ml_stats = run_walk_forward(prices, feat, spy_ret)

    # 5. Vol targeting weights
    print("\n[5/9] Computing vol-targeting weights...")
    ml_upro_weights     = compute_vol_weights(predictions)
    mean_w_all          = ml_upro_weights.mean()
    v44_on_mask         = v44_signal.reindex(ml_upro_weights.index).fillna(False)
    mean_w_when_on      = ml_upro_weights[v44_on_mask].mean()
    mean_w_when_off     = ml_upro_weights[~v44_on_mask].mean()
    full_upro_pct       = (ml_upro_weights >= 0.99).mean()
    defensive_pct       = (ml_upro_weights <= UPRO_WEIGHT_MIN + 0.01).mean()
    print(f"  ML UPRO weight — overall mean: {mean_w_all:.2f}")
    print(f"    When v4.4 ON:  {mean_w_when_on:.2f}")
    print(f"    When v4.4 OFF: {mean_w_when_off:.2f}  (irrelevant — goes to SPY)")
    print(f"    Full UPRO (≥99%): {full_upro_pct:.1%} | "
          f"Min weight ({UPRO_WEIGHT_MIN:.0%}): {defensive_pct:.1%}")

    # 6. Simulate all 5 strategies
    print("\n[6/9] Simulating all 5 strategies...")
    warmup = TRAIN_WINDOW + 10  # ensure predictions exist

    results = {}
    results['(a) SPY Buy & Hold'] = simulate(
        prices, spy_ret, v44_signal, None, mode='spy_bh', warmup=warmup)

    results['(b) v4.4 Pure'] = simulate(
        prices, spy_ret, v44_signal, None, mode='v44_pure', warmup=warmup)

    results['(c) ML Vol Pure'] = simulate(
        prices, spy_ret, v44_signal, ml_upro_weights, mode='ml_vol_pure', warmup=warmup)

    results['(d) v4.4 + ML Vol Combined'] = simulate(
        prices, spy_ret, v44_signal, ml_upro_weights, mode='v44_ml_combined', warmup=warmup)

    results['(e) v4.4 + Simple Vol'] = simulate(
        prices, spy_ret, v44_signal, None, mode='v44_simple_vol', warmup=warmup)

    # 7. Metrics
    print("\n[7/9] Computing performance metrics...")
    all_metrics = [calc_metrics(res, label) for label, res in results.items()]
    print_metrics_table(all_metrics)

    # 8. Adversarial validation on the combined strategy (primary)
    print("\n[8/9] Adversarial validation...")
    adv_combined = run_adversarial(
        results['(d) v4.4 + ML Vol Combined'],
        v44_signal, ml_upro_weights, prices, spy_ret,
        strategy_label='v4.4 + ML Vol Combined',
        mode='v44_ml_combined', warmup=warmup
    )

    # Also run on v4.4 pure as reference
    adv_v44 = run_adversarial(
        results['(b) v4.4 Pure'],
        v44_signal, ml_upro_weights, prices, spy_ret,
        strategy_label='v4.4 Pure',
        mode='v44_pure', warmup=warmup
    )

    # 9. Charts
    create_charts(results, all_metrics, prices, v44_signal, ml_upro_weights, spy_ret)

    # 10. Save results.json
    print("\n[9/9] Saving results...")
    elapsed = time.time() - t0

    # Delta comparison: combined vs v4.4 pure
    m_dict = {m['label']: m for m in all_metrics}
    m_v44  = m_dict.get('(b) v4.4 Pure', {})
    m_comb = m_dict.get('(d) v4.4 + ML Vol Combined', {})
    m_spy  = m_dict.get('(a) SPY Buy & Hold', {})

    delta_sharpe = m_comb.get('sharpe', 0) - m_v44.get('sharpe', 0)
    delta_dd     = m_comb.get('max_dd', 0) - m_v44.get('max_dd', 0)
    delta_cagr   = m_comb.get('cagr', 0)   - m_v44.get('cagr', 0)

    output_payload = {
        'strategy': 'v4.4 + ML Vol Targeting Combined',
        'timestamp': dt.datetime.now().isoformat(),
        'period': f"{prices.index[0].date()} to {prices.index[-1].date()}",
        'config': {
            'initial_capital': INITIAL_CAPITAL,
            'dca': 'NONE',
            'target_vol': TARGET_VOL,
            'upro_weight_range': [UPRO_WEIGHT_MIN, UPRO_WEIGHT_MAX],
            'train_window': TRAIN_WINDOW,
            'vix_threshold': VIX_THRESHOLD,
            'vix_sma_short': VIX_SMA_SHORT,
            'vix_sma_long': VIX_SMA_LONG,
            'features_used': TOP_FEATURES,
            'rebal_cost_bps': REBAL_COST_BPS,
        },
        'ml_stats': ml_stats,
        'metrics': {m['label']: m for m in all_metrics},
        'adversarial_combined': {
            k: (v if not isinstance(v, dict) else
                {kk: vv for kk, vv in v.items()
                 if not isinstance(vv, (pd.Series, pd.DataFrame))})
            for k, v in adv_combined.items()
        },
        'adversarial_v44_reference': {
            k: (v if not isinstance(v, dict) else
                {kk: vv for kk, vv in v.items()
                 if not isinstance(vv, (pd.Series, pd.DataFrame))})
            for k, v in adv_v44.items()
        },
        'improvement_vs_v44': {
            'sharpe_delta':    float(delta_sharpe),
            'max_dd_delta_pp': float(delta_dd * 100),
            'cagr_delta_pp':   float(delta_cagr * 100),
        },
        'allocation_stats': {
            'pct_v44_risk_on':       float(pct_risk_on),
            'mean_upro_weight':      float(mean_w_all),
            'mean_w_when_v44_on':    float(mean_w_when_on),
            'pct_full_upro':         float(full_upro_pct),
            'pct_min_weight':        float(defensive_pct),
        },
    }

    results_path = OUTPUT / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(output_payload, f, indent=2, default=str)

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print_metrics_table(all_metrics)

    print(f"\n  Combined vs v4.4 Pure:")
    print(f"    Sharpe:  {m_v44.get('sharpe', 0):.3f} → {m_comb.get('sharpe', 0):.3f} "
          f"({delta_sharpe:+.3f})")
    print(f"    MaxDD:   {m_v44.get('max_dd', 0):.1%} → {m_comb.get('max_dd', 0):.1%} "
          f"({delta_dd*100:+.1f}pp)")
    print(f"    CAGR:    {m_v44.get('cagr', 0):.1%} → {m_comb.get('cagr', 0):.1%} "
          f"({delta_cagr*100:+.1f}pp)")

    print(f"\n  Adversarial (Combined): {adv_combined['n_pass']}/{adv_combined['n_total']} "
          f"gates → {'PASS' if adv_combined['overall_pass'] else 'FAIL'}")
    print(f"  Adversarial (v4.4 ref): {adv_v44['n_pass']}/{adv_v44['n_total']} "
          f"gates → {'PASS' if adv_v44['overall_pass'] else 'FAIL'}")

    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output:  {OUTPUT}/")
    print(f"{'='*80}")
    print("DONE")
    print(f"{'='*80}")

    return output_payload


if __name__ == "__main__":
    results = main()
