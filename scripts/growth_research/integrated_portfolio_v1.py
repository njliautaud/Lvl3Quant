#!/usr/bin/env python3
"""
Integrated Portfolio v1 — Combined Strategy System
====================================================
Tests four proven components COMBINED as a single portfolio allocation system.
Critical question: do components work together or cancel each other out?

Components:
  1. v4.4 signal: VIX 20d SMA < 200d SMA AND VIX < 25 → UPRO, else → SPY
     Standalone Sharpe ~1.0
  2. ML vol targeting: GBM predicts next-5d realized vol, sizes UPRO inversely.
     R²=0.65, corr=0.81. Target vol = 15%. With v4.4: Sharpe 1.26.
  3. VIX spike buying (HC #714 R4): VIX > 30 → allocate 10% reserve to SVIX.
     VIX spikes 4-6×/yr, avg +7% return in 1 month, 71% WR.
  4. Macro danger overlay (#472): credit spreads, VIX term structure, gold
     momentum, dollar strength → scale leverage. 2 signals → -30%, 3+ → -60%.

Integrated portfolio logic:
  - Base: 90% growth (v4.4 + ML vol targeting), 10% cash reserve
  - v4.4 risk-off → 100% SPY
  - v4.4 risk-on → ML vol targeting sets UPRO weight, remainder SPY
  - Macro danger overlay scales down UPRO when 2+ danger signals fire
  - VIX spike reserve: VIX > 30 → 10% reserve → SVIX; exit below 22

HC compliance:
  HC #0  : SLIDING 252d walk-forward for ML vol model
  HC #713: Fixed $100K, NO DCA, next-day execution
  HC #705: Full adversarial (permutation 100×, 4 sub-periods, outlier removal, R1 regime)

Comparisons:
  (a) SPY buy & hold
  (b) v4.4 pure
  (c) v4.4 + ML vol targeted
  (d) FULL integrated portfolio (all 4 components)

Output: /home/jupiter/Lvl3Quant/output/integrated_portfolio_v1/results.json
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
TARGET_VOL          = 0.15          # 15% annualized vol target
TRAIN_WINDOW        = 252           # sliding window (HC #0)
REBAL_COST_BPS      = 5             # one-way switching cost (5bps)
N_PERMUTATIONS      = 100
N_SUB_PERIODS       = 4             # sub-period blocks
OUTLIER_REMOVE_N    = 5             # top N days to remove
ANNUALIZE           = 252
R1_GAP_THRESHOLD    = 0.50

# v4.4 signal parameters
VIX_THRESHOLD       = 25.0
VIX_SMA_SHORT       = 20
VIX_SMA_LONG        = 200

# ML vol targeting UPRO weight bounds
UPRO_WEIGHT_MIN     = 0.30
UPRO_WEIGHT_MAX     = 1.50

# VIX spike reserve parameters (HC #714 R4)
VIX_SPIKE_ENTRY     = 30.0          # VIX > 30 → allocate reserve to SVIX
VIX_SPIKE_EXIT      = 22.0          # VIX < 22 → exit SVIX, restore reserve
RESERVE_ALLOC       = 0.10          # 10% reserve for VIX spikes

# Macro danger overlay thresholds
DANGER_REDUCE_2     = 0.30          # 2 signals → reduce UPRO by 30%
DANGER_REDUCE_3     = 0.60          # 3+ signals → reduce UPRO by 60%
CREDIT_Z_THRESHOLD  = 1.5           # HYG/IEF z-score above this = danger
GLD_MOM_SIGMA       = 2.0           # gold momentum > 2σ = danger
DOLLAR_MOM_SIGMA    = 2.0           # dollar momentum > 2σ = danger

# Top features from validated ML vol targeting run (v44_vol_targeted.py)
TOP_FEATURES = [
    'spy_rvol_10d', 'spy_rvol_20d', 'vix_level', 'vix_rv_ratio',
    'vol_of_vol_20d', 'uup_mom_20d', 'gld_vol_20d', 'xlf_vol_20d',
    'spy_drawdown', 'spy_kurt_20d', 'spy_skew_20d', 'spy_tlt_corr_20d',
    'spy_abs_ret_5d_avg',
]

BASE   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "integrated_portfolio_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# METRICS UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def sharpe(returns):
    """Annualized Sharpe ratio."""
    r = pd.Series(returns).dropna()
    if len(r) < 5 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(ANNUALIZE))


def sortino(returns):
    """Annualized Sortino ratio."""
    r = pd.Series(returns).dropna()
    if len(r) < 5:
        return 0.0
    downside = r[r < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float('inf') if r.mean() > 0 else 0.0
    return float(r.mean() / downside.std() * np.sqrt(ANNUALIZE))


def cagr(returns):
    """CAGR from daily returns."""
    r = pd.Series(returns).dropna()
    cum = (1 + r).prod()
    n_years = len(r) / ANNUALIZE
    if n_years <= 0 or cum <= 0:
        return 0.0
    return float(cum ** (1 / n_years) - 1)


def max_drawdown(returns):
    """Maximum drawdown from daily returns."""
    r = pd.Series(returns).dropna()
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def calmar(returns):
    """Calmar ratio = CAGR / abs(MaxDD)."""
    c = cagr(returns)
    md = max_drawdown(returns)
    if md == 0:
        return 0.0
    return float(c / abs(md))


def win_rate(returns):
    """Fraction of positive-return days."""
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def profit_factor(returns):
    """Gross profit / gross loss."""
    r = pd.Series(returns).dropna()
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return float(gains / losses)


def compute_metrics(returns, label=""):
    """Full metrics dict for a daily return series."""
    r = pd.Series(returns).dropna()
    return {
        "label":         label,
        "sharpe":        round(sharpe(r), 3),
        "sortino":       round(sortino(r), 3),
        "cagr_pct":      round(cagr(r) * 100, 2),
        "max_dd_pct":    round(max_drawdown(r) * 100, 2),
        "calmar":        round(calmar(r), 3),
        "win_rate_pct":  round(win_rate(r) * 100, 1),
        "profit_factor": round(profit_factor(r), 3),
        "n_days":        len(r),
    }


# ═════════════════════════════════════════════════════════════════════════════
# R1 REGIME TEST
# ═════════════════════════════════════════════════════════════════════════════

def r1_regime_test(returns, regime_labels, label=""):
    """
    R1 regime-agnostic validation (HC #705).
    regime_labels: Series aligned with returns, values in {'green','red','flat'}.
    Fail if |Sharpe_green - Sharpe_red| / max(|Sg|,|Sr|) > 0.50
    """
    r = pd.Series(returns).dropna()
    rl = regime_labels.reindex(r.index)

    results = {}
    for regime in ['green', 'red', 'flat']:
        mask = rl == regime
        sub = r[mask]
        results[regime] = {
            "sharpe":       round(sharpe(sub), 3),
            "n_days":       int(mask.sum()),
            "win_rate_pct": round(win_rate(sub) * 100, 1),
        }

    s_green = results['green']['sharpe']
    s_red   = results['red']['sharpe']
    denom   = max(abs(s_green), abs(s_red), 0.001)
    gap     = abs(s_green - s_red) / denom

    return {
        "label":       label,
        "per_regime":  results,
        "regime_gap":  round(gap, 4),
        "passes_r1":   gap < R1_GAP_THRESHOLD,
        "threshold":   R1_GAP_THRESHOLD,
    }


# ═════════════════════════════════════════════════════════════════════════════
# ADVERSARIAL CHECKS
# ═════════════════════════════════════════════════════════════════════════════

def permutation_test(returns, n_perms=N_PERMUTATIONS, seed=42):
    """
    Shuffle return dates 100×, test if actual Sharpe is in top 10%.
    Returns p-value (fraction of shuffled Sharpes >= actual).
    """
    r = pd.Series(returns).dropna()
    actual = sharpe(r)
    rng = np.random.RandomState(seed)
    count_better = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(r.values)
        s = sharpe(pd.Series(shuffled))
        if s >= actual:
            count_better += 1
    p_val = count_better / n_perms
    return {
        "actual_sharpe":  round(actual, 3),
        "p_value":        round(p_val, 4),
        "n_permutations": n_perms,
        "passes":         p_val < 0.10,
        "note":           "p<0.10 means timing adds real value vs random",
    }


def sub_period_test(returns, n_blocks=N_SUB_PERIODS):
    """Split into N equal blocks, report Sharpe per block + count positives."""
    r = pd.Series(returns).dropna()
    block_size = len(r) // n_blocks
    results = []
    for i in range(n_blocks):
        start = i * block_size
        end   = (i + 1) * block_size if i < n_blocks - 1 else len(r)
        block = r.iloc[start:end]
        results.append({
            "block":        i + 1,
            "start":        str(block.index[0].date()) if len(block) > 0 else "",
            "end":          str(block.index[-1].date()) if len(block) > 0 else "",
            "sharpe":       round(sharpe(block), 3),
            "cagr_pct":     round(cagr(block) * 100, 2),
            "n_days":       len(block),
        })
    n_pos = sum(1 for b in results if b['sharpe'] > 0)
    return {
        "blocks":              results,
        "n_positive_blocks":   n_pos,
        "total_blocks":        n_blocks,
        "passes":              n_pos >= (n_blocks - 1),  # at worst 1 bad block
        "note":                f"{n_pos}/{n_blocks} blocks positive Sharpe",
    }


def outlier_removal_test(returns, n_remove=OUTLIER_REMOVE_N):
    """Remove top N best days, recheck Sharpe."""
    r = pd.Series(returns).dropna()
    s_full = sharpe(r)
    drop_idx = r.nlargest(n_remove).index
    r_trimmed = r.drop(drop_idx)
    s_trimmed = sharpe(r_trimmed)
    pct_drop = (1 - s_trimmed / max(s_full, 0.001)) * 100
    return {
        "full_sharpe":     round(s_full, 3),
        "trimmed_sharpe":  round(s_trimmed, 3),
        "sharpe_drop_pct": round(pct_drop, 1),
        "n_days_removed":  n_remove,
        "passes":          s_trimmed > 0.50,
        "note":            f"Sharpe after removing {n_remove} best days",
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: DATA DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════

def download_data():
    print("=" * 80)
    print("INTEGRATED PORTFOLIO v1 — COMBINED STRATEGY SYSTEM")
    print("=" * 80)
    print(f"\nRun started: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Capital: ${INITIAL_CAPITAL:,} fixed, NO DCA | Target vol: {TARGET_VOL:.0%}")

    print("\n[1/8] Downloading data...")
    cache = OUTPUT / "raw_data.parquet"

    tickers_map = {
        'SPY':  'SPY',
        'UPRO': 'UPRO',
        'VIX':  '^VIX',
        'GLD':  'GLD',
        'TLT':  'TLT',
        'SHY':  'SHY',
        'UUP':  'UUP',
        'XLF':  'XLF',
        'HYG':  'HYG',
        'IEF':  'IEF',
        'SVIX': 'SVIX',   # Short VIX ETF — may not exist pre-2022
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
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw['Close'].copy()
        else:
            prices = raw.copy()
        inv = {v: k for k, v in tickers_map.items()}
        prices.columns = [inv.get(str(c), str(c)) for c in prices.columns]
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

    # ── SVIX proxy ──────────────────────────────────────────────────────────
    # SVIX launched ~2022. Before that, use inverse VIX index return × 0.5
    # as conservative proxy (captures spike recovery, dampened by 0.5× to
    # reflect ETF decay / tracking error / lower liquidity pre-launch).
    if 'SVIX' not in prices.columns or prices['SVIX'].isna().sum() > len(prices) * 0.5:
        print("  SVIX not available or mostly missing — building proxy")
        prices['SVIX'] = np.nan

    # Extend SVIX backward with proxy where missing
    vix_ret = prices['VIX'].pct_change()
    svix_proxy_ret = (-vix_ret * 0.5).fillna(0.0)          # inverse, 50% scale
    svix_proxy_price = 100 * (1 + svix_proxy_ret).cumprod()

    svix_missing = prices['SVIX'].isna()
    if svix_missing.any():
        prices.loc[svix_missing, 'SVIX'] = svix_proxy_price.loc[svix_missing]
    prices['SVIX_is_proxy'] = svix_missing.astype(int)

    n_real = (~svix_missing).sum()
    n_proxy = svix_missing.sum()
    print(f"  SVIX: {n_real} real days, {n_proxy} proxy days")
    print(f"  Period: {prices.index[0].date()} → {prices.index[-1].date()}, {len(prices)} days")
    return prices


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

def build_features(prices):
    print("\n[2/8] Building features...")
    spy_ret = prices['SPY'].pct_change()

    feat = pd.DataFrame(index=prices.index)

    # ── Realized vol ──────────────────────────────────────────────────────────
    for w in [5, 10, 20, 60]:
        feat[f'spy_rvol_{w}d'] = spy_ret.rolling(w).std() * np.sqrt(252)

    # ── VIX features ──────────────────────────────────────────────────────────
    feat['vix_level']   = prices['VIX']
    feat['vix_5d_chg']  = prices['VIX'].pct_change(5)
    feat['vix_20d_chg'] = prices['VIX'].pct_change(20)
    feat['vix_z_20d']   = ((prices['VIX'] - prices['VIX'].rolling(20).mean())
                           / prices['VIX'].rolling(20).std())
    feat['vix_sma_20']  = prices['VIX'].rolling(VIX_SMA_SHORT).mean()
    feat['vix_sma_200'] = prices['VIX'].rolling(VIX_SMA_LONG).mean()

    # VIX vs realized vol ratio (term structure proxy)
    rv20 = spy_ret.rolling(20).std() * np.sqrt(252) * 100
    feat['vix_rv_ratio'] = prices['VIX'] / rv20.replace(0, np.nan)

    # ── SPY trend / momentum ──────────────────────────────────────────────────
    for w in [5, 10, 20, 60]:
        feat[f'spy_mom_{w}d'] = prices['SPY'].pct_change(w)
    feat['spy_above_200'] = (prices['SPY'] > prices['SPY'].rolling(200).mean()).astype(float)

    # Drawdown from all-time high
    spy_peak = prices['SPY'].cummax()
    feat['spy_drawdown'] = (prices['SPY'] - spy_peak) / spy_peak

    # Higher moments
    feat['spy_skew_20d']      = spy_ret.rolling(20).skew()
    feat['spy_kurt_20d']      = spy_ret.rolling(20).kurt()
    feat['spy_abs_ret_5d_avg'] = spy_ret.abs().rolling(5).mean()

    # Vol of vol
    feat['vol_of_vol_20d'] = feat['spy_rvol_20d'].rolling(20).std()

    # ── Cross-asset ───────────────────────────────────────────────────────────
    for asset in ['GLD', 'TLT', 'UUP', 'XLF']:
        if asset in prices.columns:
            ret = prices[asset].pct_change()
            feat[f'{asset.lower()}_mom_20d'] = prices[asset].pct_change(20)
            feat[f'{asset.lower()}_vol_20d'] = ret.rolling(20).std() * np.sqrt(252)

    # SPY-TLT correlation (flight to quality)
    if 'TLT' in prices.columns:
        feat['spy_tlt_corr_20d'] = spy_ret.rolling(20).corr(prices['TLT'].pct_change())

    # ── Credit spread danger signal ───────────────────────────────────────────
    if 'HYG' in prices.columns and 'IEF' in prices.columns:
        hyg_ief = prices['HYG'] / prices['IEF']
        feat['credit_spread_z'] = ((hyg_ief - hyg_ief.rolling(60).mean())
                                   / hyg_ief.rolling(60).std())
        feat['credit_5d_chg']   = hyg_ief.pct_change(5)
    else:
        feat['credit_spread_z'] = np.nan
        feat['credit_5d_chg']   = np.nan

    # ── Dollar / Gold momentum z-score for danger overlay ────────────────────
    if 'UUP' in prices.columns:
        uup_mom = prices['UUP'].pct_change(20)
        uup_vol = prices['UUP'].pct_change().rolling(60).std()
        feat['dollar_mom_zscore'] = uup_mom / (uup_vol * np.sqrt(20)).replace(0, np.nan)
    else:
        feat['dollar_mom_zscore'] = np.nan

    if 'GLD' in prices.columns:
        gld_mom = prices['GLD'].pct_change(20)
        gld_vol = prices['GLD'].pct_change().rolling(60).std()
        feat['gold_mom_zscore'] = gld_mom / (gld_vol * np.sqrt(20)).replace(0, np.nan)
    else:
        feat['gold_mom_zscore'] = np.nan

    # VIX term structure inversion signal:
    # When VIX 20d SMA > VIX 200d SMA (contango → backwardation flip) = danger
    feat['vix_ts_inverted'] = (feat['vix_sma_20'] > feat['vix_sma_200']).astype(float)

    feat = feat.replace([np.inf, -np.inf], np.nan)
    print(f"  Features built: {feat.shape[1]} columns")
    return feat, spy_ret


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: v4.4 SIGNAL
# ═════════════════════════════════════════════════════════════════════════════

def compute_v44_signal(prices, feat):
    """
    v4.4 signal (binary):
      risk_on  = VIX_SMA_20 < VIX_SMA_200  AND  VIX_close < VIX_THRESHOLD
      risk_off = otherwise
    Signal computed on day t, executed on day t+1 (shifted by 1).
    Returns boolean Series: True = risk-on (→ UPRO domain), False = SPY.
    """
    risk_on = (
        (feat['vix_sma_20'] < feat['vix_sma_200']) &
        (prices['VIX'] < VIX_THRESHOLD)
    )
    # Shift 1: see today's close, trade tomorrow's open
    signal = risk_on.shift(1).fillna(False).astype(bool)
    n_on = signal.sum()
    n_off = (~signal).sum()
    pct_on = 100 * n_on / len(signal)
    print(f"  v4.4 signal: risk-on {n_on}d ({pct_on:.1f}%), risk-off {n_off}d ({100-pct_on:.1f}%)")
    return signal


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: WALK-FORWARD ML VOL TARGETING (SLIDING 252d — HC #0)
# ═════════════════════════════════════════════════════════════════════════════

def run_walk_forward_ml(prices, feat, spy_ret):
    """
    SLIDING 252d walk-forward GBM vol predictor.
    Target: next-5d realized vol of SPY.
    Returns: daily predicted vol Series + diagnostics dict.
    """
    print("\n[4/8] Walk-forward ML vol targeting (SLIDING 252d — HC #0)...")

    # Target: next 5-day realized vol (annualized)
    fwd_rvol = spy_ret.rolling(5).std().shift(-5) * np.sqrt(252)
    fwd_rvol.name = 'fwd_rvol_5d'

    # Keep only available top features
    avail = [f for f in TOP_FEATURES if f in feat.columns]
    missing = set(TOP_FEATURES) - set(avail)
    if missing:
        print(f"  Warning: missing features {missing}")
    X_all = feat[avail].copy()

    # Valid rows only
    valid_mask = X_all.notna().all(axis=1) & fwd_rvol.notna()
    X_all = X_all[valid_mask]
    y_all = fwd_rvol[valid_mask]

    print(f"  Valid samples: {len(X_all)} | Features: {len(avail)}")
    print(f"  Period: {X_all.index[0].date()} → {X_all.index[-1].date()}")

    predictions = pd.Series(dtype=float, index=X_all.index, name='pred_vol')
    last_model  = None
    n_folds     = 0

    for i in range(TRAIN_WINDOW, len(X_all) - 1):
        train_s = i - TRAIN_WINDOW
        X_tr = X_all.iloc[train_s:i]
        y_tr = y_all.iloc[train_s:i]
        X_te = X_all.iloc[i:i+1]

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
        last_model = model
        n_folds += 1

        if n_folds % 500 == 0:
            valid_pred = predictions.dropna()
            valid_act  = y_all.reindex(valid_pred.index).dropna()
            idx = valid_pred.index.intersection(valid_act.index)
            r2 = r2_score(valid_act.loc[idx], valid_pred.loc[idx]) if len(idx) > 10 else float('nan')
            print(f"  Fold {n_folds}: R²={r2:.4f}")

    predictions = predictions.dropna()
    # Clip extreme predictions (can't predict negative vol)
    predictions = predictions.clip(lower=0.02, upper=1.50)

    actuals = y_all.reindex(predictions.index).dropna()
    common  = predictions.index.intersection(actuals.index)
    r2      = r2_score(actuals.loc[common], predictions.loc[common]) if len(common) > 10 else float('nan')
    rmse    = np.sqrt(mean_squared_error(actuals.loc[common], predictions.loc[common])) if len(common) > 10 else float('nan')
    corr    = actuals.loc[common].corr(predictions.loc[common]) if len(common) > 10 else float('nan')

    print(f"  Folds: {n_folds} | R²={r2:.4f} | RMSE={rmse:.4f} | Corr={corr:.4f}")
    print(f"  Pred vol — min:{predictions.min():.1%}  median:{predictions.median():.1%}  max:{predictions.max():.1%}")

    # Feature importance from last model
    fi_dict = {}
    if last_model is not None:
        fi = pd.Series(last_model.feature_importances_, index=avail).sort_values(ascending=False)
        print("  Top features (last fold):")
        for fname, imp in fi.head(8).items():
            print(f"    {fname}: {imp:.4f}")
        fi_dict = {k: float(v) for k, v in fi.items()}

    return predictions, {
        'r2':                float(r2),
        'rmse':              float(rmse),
        'corr':              float(corr),
        'n_folds':           n_folds,
        'feature_importance': fi_dict,
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: COMPUTE COMPONENT SIGNALS (MACRO DANGER, VIX SPIKE)
# ═════════════════════════════════════════════════════════════════════════════

def compute_macro_danger(feat):
    """
    Macro danger overlay (#472).
    Counts how many of 4 signals fire each day (shifted by 1 for no lookahead):
      1. Credit spread widening: HYG/IEF z-score > CREDIT_Z_THRESHOLD
      2. VIX term structure inversion: vix_20d_sma > vix_200d_sma
      3. Gold momentum > GLD_MOM_SIGMA
      4. Dollar momentum > DOLLAR_MOM_SIGMA

    Returns: Series of danger counts (0-4), shifted for execution next day.
    Also returns: reduction multiplier (1.0, 0.70, 0.40) aligned to same index.
    """
    signals = pd.DataFrame(index=feat.index)

    # Signal 1: Credit spread widening
    if 'credit_spread_z' in feat.columns and feat['credit_spread_z'].notna().sum() > 50:
        signals['credit'] = (feat['credit_spread_z'] > CREDIT_Z_THRESHOLD).astype(int)
    else:
        signals['credit'] = 0

    # Signal 2: VIX term structure inversion (short SMA > long SMA = contango → backwardation)
    signals['vix_ts'] = feat['vix_ts_inverted'].fillna(0).astype(int)

    # Signal 3: Gold momentum > 2σ (flight to safety = risk-off)
    if 'gold_mom_zscore' in feat.columns and feat['gold_mom_zscore'].notna().sum() > 50:
        signals['gold'] = (feat['gold_mom_zscore'] > GLD_MOM_SIGMA).astype(int)
    else:
        signals['gold'] = 0

    # Signal 4: Dollar momentum > 2σ (USD strength = global risk-off)
    if 'dollar_mom_zscore' in feat.columns and feat['dollar_mom_zscore'].notna().sum() > 50:
        signals['dollar'] = (feat['dollar_mom_zscore'] > DOLLAR_MOM_SIGMA).astype(int)
    else:
        signals['dollar'] = 0

    danger_count = signals.sum(axis=1)
    # Shift 1: signals computed on day t, executed t+1
    danger_count = danger_count.shift(1).fillna(0)

    # Compute reduction multiplier
    def danger_mult(n):
        if n >= 3:
            return 1.0 - DANGER_REDUCE_3
        elif n >= 2:
            return 1.0 - DANGER_REDUCE_2
        else:
            return 1.0

    multiplier = danger_count.map(danger_mult)

    # Diagnostics
    vc = danger_count.value_counts().sort_index()
    print(f"\n  Macro danger counts: {dict(vc.items())}")
    print(f"  Days with 2+ danger signals: {(danger_count >= 2).sum()} ({100*(danger_count>=2).mean():.1f}%)")
    print(f"  Days with 3+ danger signals: {(danger_count >= 3).sum()} ({100*(danger_count>=3).mean():.1f}%)")

    return danger_count, multiplier


def compute_vix_spike_state(prices):
    """
    VIX spike reserve logic (HC #714 R4):
      Entry: VIX > VIX_SPIKE_ENTRY → allocate RESERVE_ALLOC to SVIX
      Exit:  VIX < VIX_SPIKE_EXIT  → exit SVIX, restore reserve to cash
    Returns: boolean Series (True = SVIX position active), shifted 1 for execution.
    """
    vix = prices['VIX']
    in_svix = pd.Series(False, index=vix.index)
    state   = False

    for date in vix.index:
        v = vix.loc[date]
        if not state and v > VIX_SPIKE_ENTRY:
            state = True
        elif state and v < VIX_SPIKE_EXIT:
            state = False
        in_svix.loc[date] = state

    # Shift 1: VIX level known at close, trade next day
    in_svix = in_svix.shift(1).fillna(False).astype(bool)

    n_active = in_svix.sum()
    pct_active = 100 * n_active / len(in_svix)
    print(f"\n  VIX spike reserve: active {n_active}d ({pct_active:.1f}%)")

    # Count distinct spike episodes
    transitions = in_svix.astype(int).diff().fillna(0)
    n_entries = (transitions == 1).sum()
    print(f"  Distinct VIX spike episodes: {n_entries}")

    return in_svix


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6: SIMULATE STRATEGIES
# ═════════════════════════════════════════════════════════════════════════════

def simulate(prices, spy_ret, v44_signal, ml_vol_preds,
             macro_multiplier, vix_spike_active,
             mode='spy_bh', warmup=TRAIN_WINDOW + 10):
    """
    Simulate one of four strategies.

    Modes:
      'spy_bh'         : 100% SPY buy & hold
      'v44_pure'       : v4.4 binary (UPRO or SPY)
      'v44_vol'        : v4.4 + ML vol targeting
      'integrated'     : v4.4 + ML vol + macro danger + VIX spike reserve

    NO DCA. Next-day execution throughout (all signals shifted 1d already).
    Rebalancing cost: REBAL_COST_BPS per switch.
    """
    upro_ret = prices['UPRO'].pct_change()
    spy_r    = prices['SPY'].pct_change()
    svix_ret = prices['SVIX'].pct_change()

    idx = prices.index[warmup:]

    v44       = v44_signal.reindex(idx).fillna(False)
    s_ret     = spy_r.reindex(idx).fillna(0)
    u_ret     = upro_ret.reindex(idx).fillna(0)
    sv_ret    = svix_ret.reindex(idx).fillna(0)
    macro_m   = macro_multiplier.reindex(idx).fillna(1.0)
    vix_spike = vix_spike_active.reindex(idx).fillna(False)

    # Realized vol (lagged 1d) for fallback if ML pred missing
    rv20_lagged = (spy_ret.rolling(20).std() * np.sqrt(252)).shift(1).reindex(idx)

    # ML preds aligned
    ml_preds = ml_vol_preds.reindex(idx) if ml_vol_preds is not None else None

    # Tracking
    capital       = float(INITIAL_CAPITAL)
    prev_label    = None
    n_switches    = 0
    daily_returns = []
    daily_weights = {
        'spy': [], 'upro': [], 'svix': [], 'cash': [],
        'n_danger_signals': [],
    }
    component_active = {
        'v44_risk_on':    0,
        'ml_vol_reduced': 0,
        'macro_reduced':  0,
        'vix_spike':      0,
    }

    for date in idx:
        # ── Component 1: v4.4 gate ─────────────────────────────────────────────
        risk_on = bool(v44.loc[date])

        # ── Component 2: ML vol targeting weight ─────────────────────────────
        if mode in ('v44_vol', 'integrated') and risk_on:
            if ml_preds is not None and date in ml_preds.index and not pd.isna(ml_preds.loc[date]):
                pred_vol = float(ml_preds.loc[date])
            else:
                rv = rv20_lagged.loc[date]
                pred_vol = float(rv) if not pd.isna(rv) else 0.20
            upro_vol_est = pred_vol * 3.0  # UPRO ≈ 3× SPY vol
            base_upro_w  = float(np.clip(TARGET_VOL / max(upro_vol_est, 0.01),
                                         UPRO_WEIGHT_MIN, UPRO_WEIGHT_MAX))
        elif mode in ('v44_pure',) and risk_on:
            base_upro_w = 1.0
        else:
            base_upro_w = 0.0

        # ── Component 3: Macro danger overlay ────────────────────────────────
        if mode == 'integrated' and risk_on:
            mult = float(macro_m.loc[date])
            upro_w_after_macro = base_upro_w * mult
            danger_n = 0
            # recover danger_n from multiplier
            if mult <= (1.0 - DANGER_REDUCE_3 + 0.01):
                danger_n = 3
            elif mult <= (1.0 - DANGER_REDUCE_2 + 0.01):
                danger_n = 2
        else:
            upro_w_after_macro = base_upro_w
            danger_n = 0

        # ── Component 4: VIX spike reserve ───────────────────────────────────
        spike_active = bool(vix_spike.loc[date])
        if mode == 'integrated' and spike_active:
            # Reserve 10% for SVIX, reduce growth allocation proportionally
            svix_w     = RESERVE_ALLOC
            growth_alloc = 1.0 - svix_w
            upro_w     = min(upro_w_after_macro, growth_alloc)  # cap by remaining
            spy_w      = growth_alloc - upro_w
            cash_w     = 0.0
        else:
            svix_w = 0.0
            upro_w = upro_w_after_macro
            spy_w  = max(0.0, 1.0 - upro_w)
            cash_w = 0.0

        # ── Switching cost ────────────────────────────────────────────────────
        simple_label = f"U{upro_w:.2f}_S{svix_w:.2f}"
        if prev_label is not None and simple_label != prev_label:
            n_switches += 1
            capital    *= (1 - REBAL_COST_BPS / 10_000)
        prev_label = simple_label

        # ── Portfolio return ──────────────────────────────────────────────────
        day_ret = (upro_w * u_ret.loc[date]
                   + spy_w * s_ret.loc[date]
                   + svix_w * sv_ret.loc[date]
                   + cash_w * 0.0)
        capital *= (1 + day_ret)
        daily_returns.append(day_ret)

        # ── Tracking ──────────────────────────────────────────────────────────
        daily_weights['upro'].append(upro_w)
        daily_weights['spy'].append(spy_w)
        daily_weights['svix'].append(svix_w)
        daily_weights['cash'].append(cash_w)
        daily_weights['n_danger_signals'].append(danger_n)

        if risk_on:
            component_active['v44_risk_on'] += 1
        if mode in ('v44_vol', 'integrated') and risk_on and base_upro_w < 1.0:
            component_active['ml_vol_reduced'] += 1
        if mode == 'integrated' and mult < 1.0:
            component_active['macro_reduced'] += 1
        if mode == 'integrated' and spike_active:
            component_active['vix_spike'] += 1

    returns_series = pd.Series(daily_returns, index=idx)
    capital_series = (1 + returns_series).cumprod() * INITIAL_CAPITAL

    n_days = len(idx)
    avg_weights = {
        'avg_upro_pct':  round(100 * np.mean(daily_weights['upro']), 1),
        'avg_spy_pct':   round(100 * np.mean(daily_weights['spy']), 1),
        'avg_svix_pct':  round(100 * np.mean(daily_weights['svix']), 1),
        'avg_cash_pct':  round(100 * np.mean(daily_weights['cash']), 1),
    }
    component_pct = {
        k: round(100 * v / n_days, 1) for k, v in component_active.items()
    }

    return {
        'returns':       returns_series,
        'capital':       capital_series,
        'n_switches':    n_switches,
        'avg_weights':   avg_weights,
        'component_pct': component_pct,
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7: RUN ALL STRATEGIES
# ═════════════════════════════════════════════════════════════════════════════

def run_all_strategies(prices, spy_ret, v44_signal, ml_vol_preds,
                       macro_multiplier, vix_spike_active, warmup):
    print("\n[6/8] Simulating all 4 strategies...")

    results = {}
    for mode, label in [
        ('spy_bh',      'SPY Buy & Hold'),
        ('v44_pure',    'v4.4 Pure'),
        ('v44_vol',     'v4.4 + ML Vol Targeted'),
        ('integrated',  'Full Integrated Portfolio'),
    ]:
        print(f"  Simulating: {label}...")
        res = simulate(
            prices=prices,
            spy_ret=spy_ret,
            v44_signal=v44_signal,
            ml_vol_preds=ml_vol_preds,
            macro_multiplier=macro_multiplier,
            vix_spike_active=vix_spike_active,
            mode=mode,
            warmup=warmup,
        )
        results[mode] = res
        m = compute_metrics(res['returns'], label)
        print(f"    Sharpe={m['sharpe']:.3f} | CAGR={m['cagr_pct']:.1f}% | MaxDD={m['max_dd_pct']:.1f}% | Calmar={m['calmar']:.3f}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8: FULL ADVERSARIAL VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def run_adversarial(strat_results, regime_labels, target_mode='integrated'):
    """
    Full adversarial validation for the integrated portfolio.
    Also tests v4.4_pure and v44_vol for comparison.
    """
    print("\n[7/8] Running adversarial validation...")

    adv = {}
    for mode in ['v44_pure', 'v44_vol', 'integrated']:
        if mode not in strat_results:
            continue
        r = strat_results[mode]['returns']
        label = mode

        print(f"\n  --- {label} ---")

        perm   = permutation_test(r)
        subp   = sub_period_test(r, n_blocks=N_SUB_PERIODS)
        outlr  = outlier_removal_test(r)
        r1     = r1_regime_test(r, regime_labels, label=label)

        adv[mode] = {
            'permutation_test':   perm,
            'sub_period_test':    subp,
            'outlier_removal':    outlr,
            'r1_regime_test':     r1,
        }

        # Print summary
        perm_pass  = "PASS" if perm['passes']  else "FAIL"
        subp_pass  = "PASS" if subp['passes']  else "FAIL"
        outlr_pass = "PASS" if outlr['passes'] else "FAIL"
        r1_pass    = "PASS" if r1['passes_r1'] else "FAIL"

        print(f"    Permutation: p={perm['p_value']:.3f} [{perm_pass}]")
        print(f"    Sub-period:  {subp['note']} [{subp_pass}]")
        print(f"    Outlier:     Sharpe {outlr['full_sharpe']} → {outlr['trimmed_sharpe']} [{outlr_pass}]")
        print(f"    R1 regime:   gap={r1['regime_gap']:.3f} [{r1_pass}]  "
              f"green={r1['per_regime']['green']['sharpe']:.2f}  "
              f"red={r1['per_regime']['red']['sharpe']:.2f}")

    return adv


# ═════════════════════════════════════════════════════════════════════════════
# STEP 9: VIX SPIKE COMPONENT ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def analyze_vix_spikes(prices, vix_spike_active, warmup):
    """
    Analyze VIX spike component standalone:
    For each spike episode, track SVIX performance from entry to exit.
    """
    spike = vix_spike_active.iloc[warmup:]
    transitions = spike.astype(int).diff().fillna(0)
    entries = spike.index[transitions == 1]
    exits   = spike.index[transitions == -1]

    svix_ret = prices['SVIX'].pct_change()
    episodes = []

    for entry in entries:
        # Find exit
        future_exits = exits[exits > entry]
        exit_date    = future_exits[0] if len(future_exits) > 0 else spike.index[-1]
        ep_ret       = svix_ret.loc[entry:exit_date]
        cum_ret      = float((1 + ep_ret).prod() - 1)
        n_days       = len(ep_ret)
        vix_at_entry = float(prices['VIX'].loc[entry]) if entry in prices.index else np.nan
        episodes.append({
            'entry':        str(entry.date()),
            'exit':         str(exit_date.date()),
            'n_days':       n_days,
            'svix_return':  round(cum_ret * 100, 1),
            'vix_at_entry': round(vix_at_entry, 1),
            'win':          cum_ret > 0,
        })

    n_ep  = len(episodes)
    n_win = sum(1 for e in episodes if e['win'])
    avg_r = float(np.mean([e['svix_return'] for e in episodes])) if episodes else 0.0
    wr    = float(n_win / n_ep) if n_ep > 0 else 0.0

    print(f"\n  VIX Spike episodes: {n_ep} | WR: {wr:.1%} | Avg return: {avg_r:.1f}%")

    return {
        'n_episodes': n_ep,
        'win_rate':   round(wr * 100, 1),
        'avg_return': round(avg_r, 1),
        'episodes':   episodes[:20],  # store up to 20
    }


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

def plot_results(strat_results, warmup, output_dir):
    """Generate equity curve and allocation plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Integrated Portfolio v1 — Combined Strategy Results', fontsize=14)

    colors = {
        'spy_bh':     '#808080',
        'v44_pure':   '#2196F3',
        'v44_vol':    '#FF9800',
        'integrated': '#4CAF50',
    }
    labels = {
        'spy_bh':     'SPY B&H',
        'v44_pure':   'v4.4 Pure',
        'v44_vol':    'v4.4 + ML Vol',
        'integrated': 'Full Integrated',
    }

    # ── Equity curves ─────────────────────────────────────────────────────────
    ax = axes[0, 0]
    for mode, res in strat_results.items():
        cap = res['capital'].dropna()
        ax.plot(cap.index, cap / INITIAL_CAPITAL, label=labels.get(mode, mode),
                color=colors.get(mode, 'black'), linewidth=1.5 if mode == 'integrated' else 1.0)
    ax.set_title('Equity Curves (log scale)')
    ax.set_ylabel('Portfolio / Starting Capital')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Drawdowns ─────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    for mode, res in strat_results.items():
        cap = res['capital'].dropna()
        dd  = (cap / cap.cummax() - 1) * 100
        ax.fill_between(dd.index, dd, 0, alpha=0.3, color=colors.get(mode, 'black'),
                        label=labels.get(mode, mode))
    ax.set_title('Drawdowns (%)')
    ax.set_ylabel('Drawdown (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Rolling 252d Sharpe — integrated vs SPY ───────────────────────────────
    ax = axes[1, 0]
    for mode in ['spy_bh', 'integrated']:
        if mode in strat_results:
            r = strat_results[mode]['returns'].dropna()
            roll_sharpe = r.rolling(252).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0,
                raw=True
            )
            ax.plot(roll_sharpe.index, roll_sharpe, label=labels.get(mode, mode),
                    color=colors.get(mode, 'black'))
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axhline(1.0, color='g', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_title('Rolling 252d Sharpe')
    ax.set_ylabel('Sharpe')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Average allocation bar chart ──────────────────────────────────────────
    ax = axes[1, 1]
    modes_with_weights = [(m, r) for m, r in strat_results.items()
                          if 'avg_weights' in r and m != 'spy_bh']
    if modes_with_weights:
        mode_labels = [labels.get(m, m) for m, _ in modes_with_weights]
        upro_vals   = [r['avg_weights']['avg_upro_pct'] for _, r in modes_with_weights]
        spy_vals    = [r['avg_weights']['avg_spy_pct'] for _, r in modes_with_weights]
        svix_vals   = [r['avg_weights']['avg_svix_pct'] for _, r in modes_with_weights]

        x = np.arange(len(mode_labels))
        w = 0.25
        ax.bar(x - w,   upro_vals, w, label='UPRO', color='#2196F3')
        ax.bar(x,       spy_vals,  w, label='SPY',  color='#4CAF50')
        ax.bar(x + w,   svix_vals, w, label='SVIX', color='#FF5722')
        ax.set_xticks(x)
        ax.set_xticklabels(mode_labels, fontsize=8)
        ax.set_title('Average Allocations (%)')
        ax.set_ylabel('%')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = output_dir / 'integrated_portfolio_v1.png'
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  Chart saved: {plot_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # ── Step 1: Data ───────────────────────────────────────────────────────────
    prices = download_data()

    # ── Step 2: Features ──────────────────────────────────────────────────────
    feat, spy_ret = build_features(prices)

    # ── Step 3: v4.4 Signal ───────────────────────────────────────────────────
    print("\n[3/8] Computing v4.4 signal...")
    v44_signal = compute_v44_signal(prices, feat)

    # ── Step 4: Walk-forward ML vol ───────────────────────────────────────────
    ml_vol_preds, ml_diagnostics = run_walk_forward_ml(prices, feat, spy_ret)

    # ── Step 5: Macro danger + VIX spike ──────────────────────────────────────
    print("\n[5/8] Computing macro danger overlay + VIX spike signals...")
    danger_count, macro_multiplier = compute_macro_danger(feat)
    vix_spike_active = compute_vix_spike_state(prices)

    # ── Warmup ────────────────────────────────────────────────────────────────
    warmup = TRAIN_WINDOW + 10

    # ── Step 6: Simulate strategies ───────────────────────────────────────────
    strat_results = run_all_strategies(
        prices, spy_ret, v44_signal, ml_vol_preds,
        macro_multiplier, vix_spike_active, warmup
    )

    # ── Regime labels (for R1 test) ───────────────────────────────────────────
    spy_r = prices['SPY'].pct_change()
    regime = pd.Series('flat', index=prices.index)
    regime[spy_r > 0.001]  = 'green'
    regime[spy_r < -0.001] = 'red'

    # ── Step 7: Adversarial validation ────────────────────────────────────────
    adversarial = run_adversarial(strat_results, regime)

    # ── VIX spike component analysis ──────────────────────────────────────────
    vix_spike_analysis = analyze_vix_spikes(prices, vix_spike_active, warmup)

    # ── Step 8: Compile results ───────────────────────────────────────────────
    print("\n[8/8] Compiling results...")

    strategy_metrics = {}
    for mode, label in [
        ('spy_bh',     'SPY Buy & Hold'),
        ('v44_pure',   'v4.4 Pure'),
        ('v44_vol',    'v4.4 + ML Vol Targeted'),
        ('integrated', 'Full Integrated Portfolio'),
    ]:
        if mode not in strat_results:
            continue
        res = strat_results[mode]
        m   = compute_metrics(res['returns'], label)
        strategy_metrics[mode] = {
            **m,
            'n_switches':    res['n_switches'],
            'avg_weights':   res.get('avg_weights', {}),
            'component_pct': res.get('component_pct', {}),
        }

    # Compare strategies
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    header = f"{'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>8} {'Calmar':>7} {'WR':>6} {'PF':>6}"
    print(header)
    print("-" * 80)
    for mode in ['spy_bh', 'v44_pure', 'v44_vol', 'integrated']:
        if mode not in strategy_metrics:
            continue
        m = strategy_metrics[mode]
        print(f"{m['label']:<30} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr_pct']:>6.1f}% {m['max_dd_pct']:>7.1f}% {m['calmar']:>7.3f} "
              f"{m['win_rate_pct']:>5.1f}% {m['profit_factor']:>6.3f}")

    # ── Adversarial summary ───────────────────────────────────────────────────
    print("\nADVERSARIAL VALIDATION SUMMARY")
    print("-" * 60)
    for mode in ['v44_pure', 'v44_vol', 'integrated']:
        if mode not in adversarial:
            continue
        adv = adversarial[mode]
        perm_p   = adv['permutation_test']['p_value']
        r1_gap   = adv['r1_regime_test']['regime_gap']
        r1_pass  = adv['r1_regime_test']['passes_r1']
        subp_n   = adv['sub_period_test']['n_positive_blocks']
        subp_tot = adv['sub_period_test']['total_blocks']
        print(f"  {mode:<20} perm p={perm_p:.3f}  R1 gap={r1_gap:.3f} {'PASS' if r1_pass else 'FAIL'}  "
              f"sub-periods {subp_n}/{subp_tot} positive")

    # ── Component activity ────────────────────────────────────────────────────
    if 'integrated' in strategy_metrics:
        cp = strategy_metrics['integrated']['component_pct']
        print("\nCOMPONENT ACTIVITY (integrated portfolio):")
        for k, v in cp.items():
            print(f"  {k:<25}: {v:.1f}% of trading days")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(strat_results, warmup, OUTPUT)

    # ── Final JSON output ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    output = {
        "run_metadata": {
            "script":     "integrated_portfolio_v1.py",
            "run_date":   dt.datetime.now().isoformat(),
            "elapsed_s":  round(elapsed, 1),
            "initial_capital": INITIAL_CAPITAL,
            "target_vol":      TARGET_VOL,
            "train_window":    TRAIN_WINDOW,
            "data_start":      str(prices.index[warmup].date()),
            "data_end":        str(prices.index[-1].date()),
            "hc_compliance": {
                "HC0_sliding_wf":  True,
                "HC713_no_dca":    True,
                "HC705_adversarial": True,
                "HC714_vix_spike": True,
            },
        },
        "ml_vol_diagnostics":    ml_diagnostics,
        "strategy_metrics":      strategy_metrics,
        "adversarial_validation": adversarial,
        "vix_spike_analysis":    vix_spike_analysis,
        "component_signals_summary": {
            "v44_signal_pct_risk_on": round(100 * v44_signal.mean(), 1),
            "macro_danger_2plus_pct": round(100 * (danger_count >= 2).mean(), 1),
            "macro_danger_3plus_pct": round(100 * (danger_count >= 3).mean(), 1),
            "vix_spike_active_pct":   round(100 * vix_spike_active.mean(), 1),
        },
    }

    out_path = OUTPUT / "results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved: {out_path}")
    print(f"Total runtime: {elapsed:.1f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
