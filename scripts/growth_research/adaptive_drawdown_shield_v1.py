#!/usr/bin/env python3
"""
Adaptive Drawdown Shield V1
============================

Tests whether an ML-based drawdown predictor can protect our validated strategies
from large drawdowns WITHOUT sacrificing too much upside.

CONCEPT: Train LGBM to predict "will the next N days have a drawdown > X%?"
using cross-asset features. When probability is high, reduce exposure or go to cash.

APPLIED TO 3 VALIDATED STRATEGIES:
  1. Sector ETF Rotation (Sharpe 1.40 baseline)
  2. Factor ETF Rotation (Sharpe 1.13)
  3. SPY buy-and-hold (benchmark)

FEATURES FOR DRAWDOWN PREDICTION (cross-asset regime indicators):
  - VIX level, VIX 5d change, VIX term structure (VIX/VIX3M)
  - Credit spreads proxy: HYG/LQD ratio, HYG 21d return
  - Treasury signal: TLT 21d return, TLT/IEF ratio
  - Breadth: % of sector ETFs above 50d SMA
  - Momentum crash indicator: max sector drawdown last 10d
  - Equity put/call proxy: XLP/XLY ratio (defensive vs cyclical)
  - Volatility regime: realized vol 10d vs 63d

SIX VARIANTS:
  A: Baseline — reduce to 50% exposure when P(DD>10%) > 0.6
  B: Binary — 100% or 0% (cash) based on P(DD>10%) > 0.5
  C: Graduated — scale exposure linearly: exposure = 1 - P(DD)
  D: Trailing stop + ML overlay (stop at 8%, re-enter when ML says safe)
  E: Applied to factor rotation instead of sector rotation
  F: Multi-threshold (reduce at P>0.4, exit at P>0.7)

VALIDATION: Compare protected vs unprotected on each strategy.
Success = better Sharpe OR similar Sharpe with <50% max drawdown.

Output: output/growth_research/adaptive_drawdown_shield_v1/
MLflow: adaptive_drawdown_shield_v1
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

# ── Detect environment ──
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
BASE = _NEPTUNE_BASE if _NEPTUNE_BASE.exists() else _JUPITER_BASE
fprint(f"Running on: {BASE}")
sys.path.insert(0, str(BASE))

OUTPUT_DIR = BASE / "output" / "growth_research" / "adaptive_drawdown_shield_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow ──
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "adaptive_drawdown_shield_v1"
MLFLOW_OK = False
try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    MLFLOW_OK = True
    fprint(f"MLflow OK: {MLFLOW_URI}")
except Exception as e:
    fprint(f"MLflow not available: {e}")

# ── LightGBM ──
try:
    import lightgbm as lgb
    fprint("LightGBM OK")
except ImportError:
    fprint("ERROR: lightgbm required"); sys.exit(1)

# ── yfinance ──
try:
    import yfinance as yf
    fprint("yfinance OK")
except ImportError:
    fprint("ERROR: yfinance required"); sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

SECTOR_ETFS = ['XLK','XLV','XLF','XLE','XLI','XLC','XLY','XLP','XLU','XLRE','XLB']
FACTOR_ETFS = ['MTUM','VLUE','QUAL','SIZE','USMV','VTV','VUG','MOAT','COWZ','NOBL']
REGIME_TICKERS = ['SPY','QQQ','VIX','HYG','LQD','TLT','IEF','GLD']
ALL_TICKERS = list(set(SECTOR_ETFS + FACTOR_ETFS + REGIME_TICKERS))

START_DATE = '2016-01-01'
CAPITAL = 645.0
TRAIN_WINDOW = 252
DD_HORIZON = 21  # predict drawdown over next 21 trading days
DD_THRESHOLD = 0.10  # 10% drawdown threshold


def download_data():
    """Download all required price data."""
    fprint(f"Downloading {len(ALL_TICKERS)} tickers from {START_DATE}...")
    data = {}
    for tk in ALL_TICKERS:
        try:
            df = yf.download(tk, start=START_DATE, progress=False, auto_adjust=True)
            if len(df) > 100:
                data[tk] = df['Close'].squeeze()
                fprint(f"  {tk}: {len(df)} bars")
        except Exception as e:
            fprint(f"  {tk}: FAILED ({e})")
    prices = pd.DataFrame(data).dropna(how='all')
    prices = prices.ffill().dropna()
    fprint(f"Combined: {len(prices)} days, {len(prices.columns)} tickers")
    return prices


def compute_dd_features(prices, idx):
    """Compute cross-asset drawdown prediction features at a given index."""
    if idx < 126:
        return None

    features = {}
    spy = prices['SPY'].iloc[:idx+1]

    # VIX features (if available)
    if 'VIX' in prices.columns:
        vix = prices['VIX'].iloc[:idx+1]
        features['vix_level'] = vix.iloc[-1]
        features['vix_5d_chg'] = (vix.iloc[-1] / vix.iloc[-6] - 1) if len(vix) > 5 else 0
        features['vix_21d_pct'] = vix.iloc[-1] / vix.iloc[-22:].mean() if len(vix) > 21 else 1

    # Credit spread proxy: HYG/LQD
    if 'HYG' in prices.columns and 'LQD' in prices.columns:
        hyg = prices['HYG'].iloc[:idx+1]
        lqd = prices['LQD'].iloc[:idx+1]
        ratio = hyg / lqd
        features['credit_ratio'] = ratio.iloc[-1]
        features['credit_ratio_z'] = (ratio.iloc[-1] - ratio.iloc[-63:].mean()) / (ratio.iloc[-63:].std() + 1e-8)
        features['hyg_ret_21d'] = hyg.iloc[-1] / hyg.iloc[-22] - 1 if len(hyg) > 21 else 0

    # Treasury signal
    if 'TLT' in prices.columns:
        tlt = prices['TLT'].iloc[:idx+1]
        features['tlt_ret_21d'] = tlt.iloc[-1] / tlt.iloc[-22] - 1 if len(tlt) > 21 else 0
        if 'IEF' in prices.columns:
            ief = prices['IEF'].iloc[:idx+1]
            features['tlt_ief_ratio'] = (tlt.iloc[-1] / ief.iloc[-1])

    # SPY features
    features['spy_ret_5d'] = spy.iloc[-1] / spy.iloc[-6] - 1 if len(spy) > 5 else 0
    features['spy_ret_21d'] = spy.iloc[-1] / spy.iloc[-22] - 1 if len(spy) > 21 else 0
    features['spy_vol_10d'] = spy.pct_change().iloc[-10:].std() * np.sqrt(252)
    features['spy_vol_63d'] = spy.pct_change().iloc[-63:].std() * np.sqrt(252)
    features['spy_vol_ratio'] = features['spy_vol_10d'] / (features['spy_vol_63d'] + 1e-8)

    # SPY distance from 50d high
    features['spy_dist_50d_high'] = spy.iloc[-1] / spy.iloc[-50:].max() - 1
    features['spy_dist_200d_sma'] = spy.iloc[-1] / spy.iloc[-200:].mean() - 1 if len(spy) > 200 else 0

    # Breadth: % of sector ETFs above 50d SMA
    above_sma = 0
    n_sectors = 0
    for s in SECTOR_ETFS:
        if s in prices.columns:
            sec = prices[s].iloc[:idx+1]
            if len(sec) > 50:
                n_sectors += 1
                if sec.iloc[-1] > sec.iloc[-50:].mean():
                    above_sma += 1
    features['breadth_pct'] = above_sma / max(n_sectors, 1)

    # Max sector drawdown last 10d
    max_sec_dd = 0
    for s in SECTOR_ETFS:
        if s in prices.columns:
            sec = prices[s].iloc[:idx+1]
            if len(sec) > 10:
                peak = sec.iloc[-10:].max()
                dd = sec.iloc[-1] / peak - 1
                max_sec_dd = min(max_sec_dd, dd)
    features['max_sector_dd_10d'] = max_sec_dd

    # Defensive/cyclical ratio
    if 'XLP' in prices.columns and 'XLY' in prices.columns:
        xlp = prices['XLP'].iloc[:idx+1]
        xly = prices['XLY'].iloc[:idx+1]
        ratio = xlp / xly
        features['def_cyc_ratio'] = ratio.iloc[-1]
        features['def_cyc_z'] = (ratio.iloc[-1] - ratio.iloc[-63:].mean()) / (ratio.iloc[-63:].std() + 1e-8)

    # GLD momentum (flight to safety)
    if 'GLD' in prices.columns:
        gld = prices['GLD'].iloc[:idx+1]
        features['gld_ret_21d'] = gld.iloc[-1] / gld.iloc[-22] - 1 if len(gld) > 21 else 0

    return features


def compute_dd_target(prices, idx, horizon=21, threshold=0.10):
    """Compute whether a drawdown > threshold occurs in next horizon days."""
    spy = prices['SPY']
    if idx + horizon >= len(spy):
        return np.nan

    future = spy.iloc[idx+1:idx+1+horizon]
    peak = spy.iloc[idx]
    max_dd = (future / peak - 1).min()

    return 1.0 if max_dd < -threshold else 0.0


def build_dd_dataset(prices):
    """Build features + target for drawdown prediction."""
    fprint("Building drawdown prediction dataset...")
    rows = []
    spy = prices['SPY']

    for i in range(252, len(prices) - DD_HORIZON, 5):  # every 5 days to speed up
        feats = compute_dd_features(prices, i)
        if feats is None:
            continue
        target = compute_dd_target(prices, i, DD_HORIZON, DD_THRESHOLD)
        if np.isnan(target):
            continue
        feats['target'] = target
        feats['date_idx'] = i
        rows.append(feats)

    df = pd.DataFrame(rows)
    fprint(f"Dataset: {len(df)} samples, {df['target'].sum():.0f} drawdown events ({df['target'].mean()*100:.1f}%)")
    return df


def simulate_sector_rotation(prices, etfs, rebal_days=21):
    """Simple momentum-based sector rotation (baseline without protection)."""
    spy = prices['SPY']
    dates = prices.index
    daily_rets = prices.pct_change()

    equity = CAPITAL
    equity_curve = []
    holdings = None
    last_rebal = 0

    for i in range(252, len(prices)):
        dt = dates[i]

        # Rebalance?
        if i - last_rebal >= rebal_days or holdings is None:
            # Rank by trailing 63d return
            scores = {}
            for tk in etfs:
                if tk in prices.columns:
                    ret = prices[tk].iloc[i] / prices[tk].iloc[i-63] - 1
                    scores[tk] = ret
            ranked = sorted(scores, key=scores.get, reverse=True)
            holdings = ranked[:2]  # top 2
            last_rebal = i

        # Apply returns
        port_ret = np.mean([daily_rets[tk].iloc[i] for tk in holdings if tk in daily_rets.columns])
        equity *= (1 + port_ret)
        equity_curve.append({'date': dt, 'equity': equity})

    return pd.DataFrame(equity_curve).set_index('date')


def apply_dd_shield(prices, base_curve, dd_model, dd_features_df, variant='A'):
    """Apply drawdown protection overlay to a base equity curve."""
    dates = base_curve.index
    base_rets = base_curve['equity'].pct_change().fillna(0)

    protected_equity = CAPITAL
    protected_curve = []

    # Build prediction lookup (date_idx → P(DD))
    predictions = {}
    for _, row in dd_features_df.iterrows():
        idx = int(row['date_idx'])
        feat_cols = [c for c in row.index if c not in ['target', 'date_idx']]
        X = row[feat_cols].values.reshape(1, -1)
        try:
            pred = dd_model.predict(X)[0]
        except:
            pred = 0.0
        dt = prices.index[idx]
        predictions[dt] = pred

    # Interpolate predictions for all dates
    last_pred = 0.0

    for dt in dates:
        if dt in predictions:
            last_pred = predictions[dt]

        base_ret = base_rets.loc[dt]

        if variant == 'A':
            # Reduce to 50% when P(DD) > 0.6
            exposure = 0.5 if last_pred > 0.6 else 1.0
        elif variant == 'B':
            # Binary: all or nothing
            exposure = 0.0 if last_pred > 0.5 else 1.0
        elif variant == 'C':
            # Graduated: scale linearly
            exposure = max(0.0, 1.0 - last_pred)
        elif variant == 'D':
            # Trailing stop + ML re-entry
            peak = max([r['equity'] for r in protected_curve[-63:]] + [protected_equity]) if protected_curve else protected_equity
            dd = protected_equity / peak - 1
            if dd < -0.08:  # 8% trailing stop triggered
                exposure = 0.0 if last_pred > 0.3 else 0.5  # ML decides re-entry
            else:
                exposure = 1.0
        elif variant == 'F':
            # Multi-threshold
            if last_pred > 0.7:
                exposure = 0.0
            elif last_pred > 0.4:
                exposure = 0.5
            else:
                exposure = 1.0
        else:
            exposure = 1.0

        protected_equity *= (1 + base_ret * exposure)
        protected_curve.append({'date': dt, 'equity': protected_equity, 'exposure': exposure, 'dd_prob': last_pred})

    return pd.DataFrame(protected_curve).set_index('date')


def compute_metrics(curve, label=""):
    """Compute risk-adjusted metrics from equity curve."""
    rets = curve['equity'].pct_change().dropna()
    if len(rets) < 20:
        return {'label': label, 'sharpe': 0, 'sortino': 0, 'cagr': 0, 'mdd': -1, 'wr': 0}

    sharpe = rets.mean() / (rets.std() + 1e-8) * np.sqrt(252)
    downside = rets[rets < 0].std()
    sortino = rets.mean() / (downside + 1e-8) * np.sqrt(252)

    days = (curve.index[-1] - curve.index[0]).days
    years = days / 365.25
    cagr = (curve['equity'].iloc[-1] / curve['equity'].iloc[0]) ** (1/max(years, 0.1)) - 1

    rolling_max = curve['equity'].cummax()
    dd = curve['equity'] / rolling_max - 1
    mdd = dd.min()

    monthly = rets.resample('ME').sum()
    wr = (monthly > 0).mean() if len(monthly) > 0 else 0

    final = curve['equity'].iloc[-1]

    return {
        'label': label,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 1),
        'mdd': round(mdd * 100, 1),
        'wr': round(wr * 100, 1),
        'final_equity': round(final, 2),
        'n_days': len(rets),
    }


def run_permutation_test(metrics_real, prices, etfs, n_shuffles=100):
    """Permutation test: shuffle sector rankings, compare Sharpe."""
    fprint(f"  Permutation test ({n_shuffles} shuffles)...")
    random_sharpes = []
    daily_rets = prices.pct_change()

    for s in range(n_shuffles):
        equity = CAPITAL
        holdings = None
        last_rebal = 0
        curve = []

        for i in range(252, len(prices)):
            if i - last_rebal >= 21 or holdings is None:
                available = [tk for tk in etfs if tk in prices.columns]
                holdings = list(np.random.choice(available, size=min(2, len(available)), replace=False))
                last_rebal = i

            port_ret = np.mean([daily_rets[tk].iloc[i] for tk in holdings if tk in daily_rets.columns])
            equity *= (1 + port_ret)
            curve.append(equity)

        rets = pd.Series(curve).pct_change().dropna()
        sh = rets.mean() / (rets.std() + 1e-8) * np.sqrt(252)
        random_sharpes.append(sh)

    p_val = np.mean([rs >= metrics_real['sharpe'] for rs in random_sharpes])
    random_mean = np.mean(random_sharpes)
    return p_val, random_mean


def classify_regime(prices, idx):
    """Green/red day based on SPY close-to-close."""
    if idx < 1:
        return 'flat'
    spy = prices['SPY']
    ret = spy.iloc[idx] / spy.iloc[idx-1] - 1
    if ret > 0.001:
        return 'green'
    elif ret < -0.001:
        return 'red'
    return 'flat'


def regime_stratified_sharpe(curve, prices):
    """Compute Sharpe by regime."""
    rets = curve['equity'].pct_change().dropna()

    green_rets, red_rets = [], []
    for dt in rets.index:
        if dt in prices.index:
            idx = prices.index.get_loc(dt)
            regime = classify_regime(prices, idx)
            if regime == 'green':
                green_rets.append(rets[dt])
            elif regime == 'red':
                red_rets.append(rets[dt])

    green_sh = np.mean(green_rets) / (np.std(green_rets) + 1e-8) * np.sqrt(252) if len(green_rets) > 10 else 0
    red_sh = np.mean(red_rets) / (np.std(red_rets) + 1e-8) * np.sqrt(252) if len(red_rets) > 10 else 0
    gap = abs(green_sh - red_sh) / max(abs(green_sh), abs(red_sh), 0.01)

    return green_sh, red_sh, gap


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    fprint("="*70)
    fprint("ADAPTIVE DRAWDOWN SHIELD V1")
    fprint("="*70)

    run = None
    if MLFLOW_OK:
        run = mlflow.start_run(run_name=f"dd_shield_{datetime.now():%Y%m%d_%H%M}")

    # 1. Download data
    prices = download_data()

    # 2. Build drawdown prediction dataset
    dd_df = build_dd_dataset(prices)

    # 3. Walk-forward train drawdown predictor
    fprint("\n--- Training drawdown predictor (walk-forward) ---")
    feat_cols = [c for c in dd_df.columns if c not in ['target', 'date_idx']]

    # Split: train on first 70%, test on last 30%
    split_idx = int(len(dd_df) * 0.7)
    train_df = dd_df.iloc[:split_idx]
    test_df = dd_df.iloc[split_idx:]

    fprint(f"Train: {len(train_df)}, Test: {len(test_df)}")
    fprint(f"Train DD rate: {train_df['target'].mean()*100:.1f}%, Test DD rate: {test_df['target'].mean()*100:.1f}%")

    model = lgb.LGBMRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        verbose=-1, random_state=42
    )
    model.fit(train_df[feat_cols], train_df['target'])

    # Evaluate prediction quality
    test_preds = model.predict(test_df[feat_cols])
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(test_df['target'], test_preds)
        fprint(f"DD Predictor AUC: {auc:.3f}")
    except:
        auc = 0.5
        fprint("AUC computation failed (likely no DD events in test)")

    # Feature importance
    importances = dict(zip(feat_cols, model.feature_importances_))
    top_feats = sorted(importances, key=importances.get, reverse=True)[:5]
    fprint(f"Top features: {', '.join(f'{f} ({importances[f]})' for f in top_feats)}")

    # 4. Generate base equity curves
    fprint("\n--- Generating base strategies ---")
    sector_curve = simulate_sector_rotation(prices, SECTOR_ETFS, rebal_days=21)
    factor_curve = simulate_sector_rotation(prices, FACTOR_ETFS, rebal_days=21)

    spy_equity = CAPITAL
    spy_data = []
    spy_rets = prices['SPY'].pct_change()
    for i in range(252, len(prices)):
        spy_equity *= (1 + spy_rets.iloc[i])
        spy_data.append({'date': prices.index[i], 'equity': spy_equity})
    spy_curve = pd.DataFrame(spy_data).set_index('date')

    sector_metrics = compute_metrics(sector_curve, "Sector Rotation (unprotected)")
    factor_metrics = compute_metrics(factor_curve, "Factor Rotation (unprotected)")
    spy_metrics = compute_metrics(spy_curve, "SPY B&H (unprotected)")

    fprint(f"\nBase strategies:")
    fprint(f"  Sector: Sharpe {sector_metrics['sharpe']}, MDD {sector_metrics['mdd']}%, CAGR {sector_metrics['cagr']}%")
    fprint(f"  Factor: Sharpe {factor_metrics['sharpe']}, MDD {factor_metrics['mdd']}%, CAGR {factor_metrics['cagr']}%")
    fprint(f"  SPY:    Sharpe {spy_metrics['sharpe']}, MDD {spy_metrics['mdd']}%, CAGR {spy_metrics['cagr']}%")

    # 5. Apply drawdown shield variants
    fprint("\n--- Testing drawdown shield variants ---")

    variants = {
        'A': 'Reduce 50% when P(DD)>0.6',
        'B': 'Binary: cash when P(DD)>0.5',
        'C': 'Graduated: exposure = 1-P(DD)',
        'D': 'Trailing stop 8% + ML re-entry',
        'E': 'Applied to factor rotation',
        'F': 'Multi-threshold (0.4/0.7)',
    }

    results = []
    results.append(sector_metrics)
    results.append(factor_metrics)
    results.append(spy_metrics)

    for var_key, var_desc in variants.items():
        fprint(f"\n  Variant {var_key}: {var_desc}")

        # Choose base curve
        if var_key == 'E':
            base = factor_curve
            base_label = "Factor"
        else:
            base = sector_curve
            base_label = "Sector"

        protected = apply_dd_shield(prices, base, model, test_df, variant=var_key)
        metrics = compute_metrics(protected, f"{var_key}_{var_desc}")

        # Regime analysis
        g_sh, r_sh, gap = regime_stratified_sharpe(protected, prices)
        metrics['green_sharpe'] = round(g_sh, 3)
        metrics['red_sharpe'] = round(r_sh, 3)
        metrics['regime_gap'] = round(gap, 3)

        # Exposure stats
        if 'exposure' in protected.columns:
            avg_exp = protected['exposure'].mean()
            pct_cash = (protected['exposure'] == 0).mean()
            metrics['avg_exposure'] = round(avg_exp, 3)
            metrics['pct_cash'] = round(pct_cash * 100, 1)

        # Compare to unprotected
        if var_key == 'E':
            base_metrics = factor_metrics
        else:
            base_metrics = sector_metrics

        sharpe_delta = metrics['sharpe'] - base_metrics['sharpe']
        mdd_delta = metrics['mdd'] - base_metrics['mdd']  # less negative = better

        metrics['sharpe_delta'] = round(sharpe_delta, 3)
        metrics['mdd_improvement'] = round(-mdd_delta, 1)  # positive = smaller DD

        # 5-gate validation
        gates = 0
        if metrics['sharpe'] > 1.0: gates += 1
        if metrics['wr'] > 40: gates += 1
        if metrics['regime_gap'] < 0.50: gates += 1
        if mdd_delta > 0: gates += 1  # DD shield actually reduced DD
        if sharpe_delta > -0.1: gates += 1  # didn't destroy Sharpe
        metrics['gates'] = f"{gates}/5"

        fprint(f"    Sharpe: {metrics['sharpe']} (Δ{sharpe_delta:+.3f}), MDD: {metrics['mdd']}% (improved {-mdd_delta:.1f}pp)")
        fprint(f"    WR: {metrics['wr']}%, CAGR: {metrics['cagr']}%, Gates: {gates}/5")
        fprint(f"    Regime: green={g_sh:.2f}, red={r_sh:.2f}, gap={gap:.2f}")
        if 'avg_exposure' in metrics:
            fprint(f"    Avg exposure: {metrics['avg_exposure']:.0%}, Cash: {metrics['pct_cash']:.0f}%")

        results.append(metrics)

    # 6. Permutation test on best variant
    fprint("\n--- Permutation test on best protected variant ---")
    best = max([r for r in results if r['label'].startswith(('A_','B_','C_','D_','E_','F_'))],
               key=lambda x: x['sharpe'], default=None)

    if best:
        p_val, rand_mean = run_permutation_test(best, prices, SECTOR_ETFS, n_shuffles=100)
        fprint(f"  Best variant: {best['label']}")
        fprint(f"  Sharpe: {best['sharpe']} vs random mean {rand_mean:.3f}, p={p_val:.4f}")

    # 7. Summary
    fprint("\n" + "="*70)
    fprint("SUMMARY — DRAWDOWN SHIELD RESULTS")
    fprint("="*70)
    fprint(f"{'Variant':<50} {'Sharpe':>7} {'MDD%':>7} {'CAGR%':>7} {'WR%':>6} {'Gates':>6}")
    fprint("-"*90)
    for r in results:
        fprint(f"{r['label']:<50} {r['sharpe']:>7} {r['mdd']:>7} {r['cagr']:>7} {r['wr']:>6} {r.get('gates','N/A'):>6}")

    # Save results
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    fprint(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")

    # MLflow logging
    if MLFLOW_OK and run:
        for r in results:
            prefix = r['label'][:20].replace(' ','_')
            mlflow.log_metric(f"{prefix}_sharpe", r['sharpe'])
            mlflow.log_metric(f"{prefix}_mdd", r['mdd'])
        mlflow.log_metric("dd_predictor_auc", auc)
        mlflow.log_artifact(str(OUTPUT_DIR / 'results.json'))
        mlflow.end_run()

    elapsed = time.time() - t0
    fprint(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
