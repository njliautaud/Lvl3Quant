#!/usr/bin/env python3
"""
Enhanced 2h Directional Model — IC Push Experiment
===================================================

Goal: Push genuine IC (with 5-day purge) from 0.065 → 0.069+ to cross
the market-order profitability threshold.

Approach:
  1. Enhanced features (microprice deviation, vol profile deviation,
     autocorrelation, cross-period interactions)
  2. Test both LightGBM (with stronger regularization) and ElasticNet
  3. 5-day purge gap for honest shuffle-label permutation test
  4. 100 permutations for robust genuine-IC estimate
  5. Strict pass/fail: genuine_IC > 0.069

Based on longer_horizon_v2_regime_gate.py. Sliding 60d train, 1d OOT (HC #0).

Author: Claude (HC #658 hourly productivity, HC #659 tick-level mandatory)
"""

import gc
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
OUTPUT_DIR = ROOT / "output" / "lh_2h_enhanced_ic_push"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [LH-ENH] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(ROOT / "logs" / "lh_2h_enhanced_ic_push.log")),
    ],
)
log = logging.getLogger('LH-ENH')

# ── Constants ──
TRAIN_DAYS = 60
PURGE_DAYS = 5
HORIZON = '2h'
HORIZON_BARS = 2  # 2 hourly bars forward
N_PERMUTATIONS = 100
IC_THRESHOLD = 0.069  # Market order profitability threshold

# ── Cost constants (AMP/Rithmic canonical) ──
ES_TICK_VALUE = 12.50
COST_MARKET_RT_TICKS = 1.376

# =============================================================================
# DATA LOADING
# =============================================================================

def load_minute_bars():
    """Load all minute bar parquets."""
    files = sorted(MINUTE_BAR_DIR.glob("*.parquet"))
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df['date'] = f.stem
            frames.append(df)
        except Exception as e:
            log.warning(f"Skip {f.stem}: {e}")
    combined = pd.concat(frames, ignore_index=True)
    combined['ts_minute'] = pd.to_datetime(combined['ts_minute'], utc=True)
    combined = combined.sort_values('ts_minute').reset_index(drop=True)
    log.info(f"Loaded {len(combined)} bars, {len(frames)} days")
    return combined


# =============================================================================
# ENHANCED FEATURE ENGINEERING
# =============================================================================

def compute_enhanced_hourly(df):
    """Aggregate 1-min → hourly with ENHANCED feature set."""
    df = df.copy()
    df['hour'] = df['ts_minute'].dt.hour
    df['date_str'] = df['date']
    df['return_1m'] = df.groupby('date_str')['close'].pct_change()

    records = []
    for (date_str, hour), g in df.groupby(['date_str', 'hour']):
        if len(g) < 5:
            continue
        c = g['close'].values.astype(float)
        v = g['volume'].values.astype(float)
        ofi = g['ofi_1min'].values.astype(float)
        sv = g['signed_volume'].values.astype(float)
        ret = g['return_1m'].fillna(0).values.astype(float)
        sp = g['spread_mean'].values.astype(float)
        tc = g['trade_count'].values.astype(float)
        vwap = g['vwap'].values.astype(float)
        mp = g['microprice_close'].values.astype(float)

        # --- STANDARD features (from v2) ---
        rec = {
            'date': date_str, 'hour': hour, 'ts': g['ts_minute'].iloc[0],
            'open': c[0], 'high': c.max(), 'low': c.min(), 'close': c[-1],
            'return_1h': (c[-1] / c[0] - 1) if c[0] > 0 else 0,
            'range_ticks': (c.max() - c.min()),
            'close_position': (c[-1] - c.min()) / max(c.max() - c.min(), 1),
            'total_volume': v.sum(),
            'avg_volume': v.mean(),
            'volume_trend': np.polyfit(np.arange(len(v)), v, 1)[0] if len(v) > 1 else 0,
            'volume_concentration': v.max() / max(v.mean(), 1),
            'ofi_sum': ofi.sum(),
            'ofi_mean': ofi.mean(),
            'ofi_trend': np.polyfit(np.arange(len(ofi)), ofi, 1)[0] if len(ofi) > 1 else 0,
            'ofi_consistency': np.mean(np.sign(ofi) == np.sign(ofi.sum())) if ofi.sum() != 0 else 0.5,
            'ofi_late_vs_early': ofi[len(ofi)//2:].sum() - ofi[:len(ofi)//2].sum(),
            'signed_volume_sum': sv.sum(),
            'signed_volume_ratio': sv.sum() / max(v.sum(), 1),
            'buy_volume_fraction': np.sum(sv[sv > 0]) / max(v.sum(), 1),
            'sell_volume_fraction': -np.sum(sv[sv < 0]) / max(v.sum(), 1),
            'sweep_minutes': int(np.sum(np.abs(sv) > 2 * sv.std())) if sv.std() > 0 else 0,
            'spread_mean': sp.mean(),
            'spread_max': sp.max(),
            'trade_count_sum': tc.sum(),
            'trade_intensity': tc.mean(),
            'return_std': ret.std(),
            'return_skew': float(stats.skew(ret)) if len(ret) > 3 else 0,
            'realized_vol': ret.std() * np.sqrt(60),
        }

        # --- ENHANCED features (new) ---

        # 1. Microprice deviation: indicates order book imbalance
        mp_valid = mp[mp > 0]
        c_valid = c[:len(mp_valid)] if len(mp_valid) > 0 else c
        if len(mp_valid) > 0 and len(c_valid) > 0:
            mp_dev = (mp_valid - c_valid[:len(mp_valid)]) / np.maximum(c_valid[:len(mp_valid)], 1)
            rec['microprice_dev_mean'] = mp_dev.mean()
            rec['microprice_dev_trend'] = np.polyfit(np.arange(len(mp_dev)), mp_dev, 1)[0] if len(mp_dev) > 1 else 0
            rec['microprice_dev_late'] = mp_dev[-len(mp_dev)//3:].mean() if len(mp_dev) >= 3 else mp_dev.mean()
        else:
            rec['microprice_dev_mean'] = 0
            rec['microprice_dev_trend'] = 0
            rec['microprice_dev_late'] = 0

        # 2. VWAP deviation: institutional buying/selling pressure
        vwap_valid = vwap[vwap > 0]
        if len(vwap_valid) > 0:
            vwap_dev = (c[:len(vwap_valid)] - vwap_valid) / np.maximum(vwap_valid, 1)
            rec['vwap_dev_final'] = vwap_dev[-1] if len(vwap_dev) > 0 else 0
            rec['vwap_dev_trend'] = np.polyfit(np.arange(len(vwap_dev)), vwap_dev, 1)[0] if len(vwap_dev) > 1 else 0
        else:
            rec['vwap_dev_final'] = 0
            rec['vwap_dev_trend'] = 0

        # 3. Volume-weighted return: direction of large trades
        if v.sum() > 0:
            rec['vw_return'] = np.sum(ret * v[:len(ret)]) / v[:len(ret)].sum() if len(ret) <= len(v) else 0
        else:
            rec['vw_return'] = 0

        # 4. Return autocorrelation: momentum/reversal signature
        if len(ret) > 10:
            rec['return_autocorr_1'] = np.corrcoef(ret[:-1], ret[1:])[0, 1] if ret[:-1].std() > 0 and ret[1:].std() > 0 else 0
            rec['return_autocorr_5'] = np.corrcoef(ret[:-5], ret[5:])[0, 1] if ret[:-5].std() > 0 and ret[5:].std() > 0 else 0
        else:
            rec['return_autocorr_1'] = 0
            rec['return_autocorr_5'] = 0

        # 5. Trade size distribution: large vs small trades
        if tc.sum() > 0 and v.sum() > 0:
            rec['avg_trade_size'] = v.sum() / tc.sum()
            # Volume in top vs bottom half of minute-level volumes
            v_sorted = np.sort(v)
            mid = len(v_sorted) // 2
            rec['volume_top_half_ratio'] = v_sorted[mid:].sum() / max(v.sum(), 1)
        else:
            rec['avg_trade_size'] = 0
            rec['volume_top_half_ratio'] = 0.5

        # 6. OFI acceleration: second derivative of order flow
        if len(ofi) > 5:
            ofi_first_half = ofi[:len(ofi)//2].sum()
            ofi_second_half = ofi[len(ofi)//2:].sum()
            rec['ofi_acceleration'] = ofi_second_half - ofi_first_half
            # Quadratic fit for curvature
            if len(ofi) > 3:
                coeffs = np.polyfit(np.arange(len(ofi)), ofi, 2)
                rec['ofi_curvature'] = coeffs[0]
            else:
                rec['ofi_curvature'] = 0
        else:
            rec['ofi_acceleration'] = 0
            rec['ofi_curvature'] = 0

        # 7. Intraday time features
        rec['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        rec['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        # 8. High-low range position (where did we close in the range?)
        rng = c.max() - c.min()
        rec['hl_range_position'] = (c[-1] - c.min()) / max(rng, 1)  # 0=closed at low, 1=at high

        # 9. Volume-price divergence
        if len(v) > 5 and v.std() > 0 and c.std() > 0:
            price_direction = np.sign(c[-1] - c[0])
            volume_trend_dir = np.sign(np.polyfit(np.arange(len(v)), v, 1)[0])
            rec['vol_price_divergence'] = float(price_direction != volume_trend_dir)
        else:
            rec['vol_price_divergence'] = 0

        # 10. Realized vol of OFI (stability of order flow)
        if ofi.std() > 0:
            rec['ofi_vol'] = ofi.std()
            rec['ofi_vol_normalized'] = ofi.std() / max(abs(ofi.mean()), 1)
        else:
            rec['ofi_vol'] = 0
            rec['ofi_vol_normalized'] = 0

        records.append(rec)

    hourly = pd.DataFrame(records)
    hourly = hourly.sort_values('ts').reset_index(drop=True)
    log.info(f"Computed {len(hourly)} hourly bars with {len(hourly.columns)} columns")
    return hourly


def add_rolling_features(df):
    """Add multi-hour rolling features (these drive most of the signal)."""
    for w in [2, 4, 6]:
        lbl = f'{w}h'
        df[f'ofi_sum_{lbl}'] = df['ofi_sum'].rolling(w, min_periods=1).sum()
        df[f'ofi_trend_{lbl}'] = df['ofi_trend'].rolling(w, min_periods=1).mean()
        df[f'sv_sum_{lbl}'] = df['signed_volume_sum'].rolling(w, min_periods=1).sum()
        df[f'volume_ma_{lbl}'] = df['total_volume'].rolling(w, min_periods=1).mean()
        df[f'volume_vs_ma_{lbl}'] = df['total_volume'] / df[f'volume_ma_{lbl}'].clip(lower=1)
        df[f'vol_trend_{lbl}'] = df['realized_vol'].rolling(w, min_periods=1).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True)

    # Enhanced rolling features
    for w in [2, 4, 6]:
        lbl = f'{w}h'
        # Microprice deviation rolling
        df[f'mpdev_sum_{lbl}'] = df['microprice_dev_mean'].rolling(w, min_periods=1).sum()
        # VWAP deviation rolling
        df[f'vwapdev_sum_{lbl}'] = df['vwap_dev_final'].rolling(w, min_periods=1).sum()
        # OFI acceleration rolling
        df[f'ofi_accel_{lbl}'] = df['ofi_acceleration'].rolling(w, min_periods=1).sum()
        # Autocorrelation rolling mean
        df[f'autocorr_mean_{lbl}'] = df['return_autocorr_1'].rolling(w, min_periods=1).mean()

    # Price momentum
    df['mom_2h'] = df['close'].pct_change(2)
    df['mom_4h'] = df['close'].pct_change(4)
    df['mom_6h'] = df['close'].pct_change(6)

    return df


def add_regime_context(df):
    """Regime features."""
    df['vol_20h'] = df['realized_vol'].rolling(20, min_periods=5).mean()
    df['vol_regime_f'] = pd.qcut(df['vol_20h'].rank(method='first'), 3, labels=[0, 1, 2]).astype(float)
    df['trend_8h'] = df['close'].pct_change(8)
    df['trend_20h'] = df['close'].pct_change(20)
    df['trend_regime'] = np.where(df['trend_20h'] > 0.005, 1, np.where(df['trend_20h'] < -0.005, -1, 0))
    return df


def add_forward_labels(df, horizon_bars=2):
    """2h forward label in ticks (close is already in tick units)."""
    df['fwd_ticks'] = df['close'].shift(-horizon_bars) - df['close']
    df = df.dropna(subset=['fwd_ticks'])
    return df


# =============================================================================
# FEATURE COLUMNS
# =============================================================================

def get_feature_cols(df):
    """Get all feature columns (exclude date, ts, target, identifiers)."""
    exclude = {'date', 'hour', 'ts', 'open', 'high', 'low', 'close', 'fwd_ticks'}
    return [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]


# =============================================================================
# MODELS
# =============================================================================

def train_lgbm(X_train, y_train, X_val, y_val):
    """LightGBM with STRONG regularization to reduce overfitting."""
    import lightgbm as lgb

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'num_leaves': 15,          # Reduced from default 31
        'max_depth': 4,            # Shallow trees
        'learning_rate': 0.02,     # Slow learning
        'feature_fraction': 0.5,   # Only use half the features per tree
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 50,   # High minimum leaf count
        'lambda_l1': 1.0,         # L1 regularization
        'lambda_l2': 5.0,         # L2 regularization
        'n_estimators': 500,
        'early_stopping_rounds': 50,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.log_evaluation(0)])
    return model


def train_elasticnet(X_train, y_train):
    """ElasticNet (ridge + lasso) — can't overfit as easily."""
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        alphas=np.logspace(-4, 1, 20),
        cv=5,
        max_iter=5000,
        n_jobs=-1,
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def train_ridge(X_train, y_train):
    """Pure Ridge — most robust, can't overfit."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
    model.fit(X_scaled, y_train)
    return model, scaler


# =============================================================================
# WALK-FORWARD ENGINE
# =============================================================================

def run_walkforward(hourly, feature_cols, model_type='lgbm', shuffle_labels=False, purge_days=0):
    """Run sliding walk-forward with optional shuffle + purge."""
    dates = sorted(hourly['date'].unique())

    all_preds = []
    all_actuals = []
    all_dates = []

    for i in range(TRAIN_DAYS + purge_days, len(dates)):
        oot_date = dates[i]
        train_end_idx = i - purge_days  # Purge gap
        train_start_idx = max(0, train_end_idx - TRAIN_DAYS)

        train_dates = dates[train_start_idx:train_end_idx]

        train = hourly[hourly['date'].isin(train_dates)]
        oot = hourly[hourly['date'] == oot_date]

        if len(train) < 100 or len(oot) == 0:
            continue

        X_train = train[feature_cols].fillna(0).values
        y_train = train['fwd_ticks'].values
        X_oot = oot[feature_cols].fillna(0).values
        y_oot = oot['fwd_ticks'].values

        if shuffle_labels:
            y_train = np.random.permutation(y_train)

        try:
            if model_type == 'lgbm':
                # Split last 20% of training for validation (early stopping)
                split = int(len(X_train) * 0.8)
                model = train_lgbm(X_train[:split], y_train[:split],
                                   X_train[split:], y_train[split:])
                preds = model.predict(X_oot)
            elif model_type == 'elasticnet':
                model, scaler = train_elasticnet(X_train, y_train)
                preds = model.predict(scaler.transform(X_oot))
            elif model_type == 'ridge':
                model, scaler = train_ridge(X_train, y_train)
                preds = model.predict(scaler.transform(X_oot))
            else:
                raise ValueError(f"Unknown model: {model_type}")

            all_preds.extend(preds)
            all_actuals.extend(y_oot)
            all_dates.extend([oot_date] * len(y_oot))

        except Exception as e:
            log.warning(f"Fold {oot_date} failed: {e}")
            continue

    preds = np.array(all_preds)
    actuals = np.array(all_actuals)

    if len(preds) < 50:
        return 0.0, preds, actuals

    # IC = Spearman rank correlation between predictions and actuals
    ic = float(stats.spearmanr(preds, actuals)[0])
    return ic, preds, actuals


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("Enhanced 2h IC Push Experiment")
    log.info(f"Purge days: {PURGE_DAYS}, Permutations: {N_PERMUTATIONS}")
    log.info(f"IC threshold for profitability: {IC_THRESHOLD}")
    log.info("=" * 60)

    # Load and process data
    log.info("Loading minute bars...")
    minutes = load_minute_bars()

    log.info("Computing enhanced hourly features...")
    hourly = compute_enhanced_hourly(minutes)
    hourly = add_rolling_features(hourly)
    hourly = add_regime_context(hourly)
    hourly = add_forward_labels(hourly, horizon_bars=HORIZON_BARS)

    feature_cols = get_feature_cols(hourly)
    log.info(f"Feature count: {len(feature_cols)}")
    log.info(f"Sample count: {len(hourly)}")
    log.info(f"Date range: {hourly['date'].min()} → {hourly['date'].max()}")

    del minutes
    gc.collect()

    results = {}

    for model_type in ['lgbm', 'elasticnet', 'ridge']:
        log.info(f"\n{'='*40}")
        log.info(f"Model: {model_type.upper()}")
        log.info(f"{'='*40}")

        # Real IC
        t0 = time.time()
        real_ic, preds, actuals = run_walkforward(
            hourly, feature_cols, model_type=model_type,
            shuffle_labels=False, purge_days=PURGE_DAYS
        )
        elapsed = time.time() - t0
        log.info(f"Real IC ({model_type}): {real_ic:.4f} (took {elapsed:.0f}s)")

        # Permutation test
        shuf_ics = []
        log.info(f"Running {N_PERMUTATIONS} permutations...")
        for trial in range(N_PERMUTATIONS):
            shuf_ic, _, _ = run_walkforward(
                hourly, feature_cols, model_type=model_type,
                shuffle_labels=True, purge_days=PURGE_DAYS
            )
            shuf_ics.append(shuf_ic)
            if (trial + 1) % 10 == 0:
                log.info(f"  Permutation {trial+1}/{N_PERMUTATIONS}: mean_shuf_ic={np.mean(shuf_ics):.4f}")

        mean_shuf = np.mean(shuf_ics)
        std_shuf = np.std(shuf_ics)
        genuine_ic = real_ic - mean_shuf
        p_value = np.mean([s >= real_ic for s in shuf_ics])

        # Trade simulation (simple: go long if pred > 0, short if < 0)
        if len(preds) > 0:
            directions = np.sign(preds)
            gross_ticks = directions * actuals
            net_ticks = gross_ticks - COST_MARKET_RT_TICKS

            wr = np.mean(net_ticks > 0)
            avg_net = net_ticks.mean()
            sharpe = net_ticks.mean() / max(net_ticks.std(), 1e-6) * np.sqrt(252 * 4)  # ~4 bars/day
        else:
            wr = 0; avg_net = 0; sharpe = 0

        result = {
            'model': model_type,
            'real_ic': float(real_ic),
            'mean_shuffle_ic': float(mean_shuf),
            'std_shuffle_ic': float(std_shuf),
            'genuine_ic': float(genuine_ic),
            'p_value': float(p_value),
            'n_samples': len(preds),
            'wr': float(wr),
            'avg_net_ticks': float(avg_net),
            'annual_sharpe': float(sharpe),
            'passes_threshold': genuine_ic > IC_THRESHOLD,
            'profitable_with_mkt_orders': avg_net > 0,
        }
        results[model_type] = result

        log.info(f"  Real IC:        {real_ic:.4f}")
        log.info(f"  Shuffle IC:     {mean_shuf:.4f} ± {std_shuf:.4f}")
        log.info(f"  GENUINE IC:     {genuine_ic:.4f}")
        log.info(f"  p-value:        {p_value:.3f}")
        log.info(f"  Passes 0.069:   {'✅ YES' if genuine_ic > IC_THRESHOLD else '❌ NO'}")
        log.info(f"  Avg net ticks:  {avg_net:.3f}")
        log.info(f"  Win rate:       {wr:.1%}")
        log.info(f"  Annual Sharpe:  {sharpe:.2f}")

        gc.collect()

    # ── Feature importance (from best model) ──
    log.info("\n" + "=" * 40)
    log.info("FEATURE IMPORTANCE ANALYSIS")
    log.info("=" * 40)

    # Quick single LGBM fit for feature importance
    dates = sorted(hourly['date'].unique())
    X_all = hourly[feature_cols].fillna(0).values
    y_all = hourly['fwd_ticks'].values

    import lightgbm as lgb
    temp_model = lgb.LGBMRegressor(
        num_leaves=15, max_depth=4, n_estimators=200,
        feature_fraction=0.5, min_child_samples=50,
        lambda_l1=1.0, lambda_l2=5.0, verbosity=-1
    )
    temp_model.fit(X_all, y_all)
    importances = temp_model.feature_importances_
    fi_pairs = sorted(zip(feature_cols, importances), key=lambda x: -x[1])

    log.info("Top 20 features:")
    for name, imp in fi_pairs[:20]:
        log.info(f"  {name:35s} {imp:6d}")

    results['feature_importance_top20'] = [(n, int(i)) for n, i in fi_pairs[:20]]
    results['n_features'] = len(feature_cols)
    results['feature_names'] = feature_cols

    # ── Summary ──
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)

    best_model = max(results.keys() - {'feature_importance_top20', 'n_features', 'feature_names'},
                     key=lambda k: results[k].get('genuine_ic', -999))
    best = results[best_model]

    log.info(f"Best model: {best_model}")
    log.info(f"Genuine IC: {best['genuine_ic']:.4f}")
    log.info(f"Threshold:  {IC_THRESHOLD}")
    log.info(f"VERDICT:    {'✅ PASSES — above cost threshold' if best['genuine_ic'] > IC_THRESHOLD else '❌ FAILS — below cost threshold'}")

    if best['genuine_ic'] > IC_THRESHOLD:
        expected_edge = best['genuine_ic'] * 20  # ~20 tick avg 2h move
        net_edge = expected_edge - COST_MARKET_RT_TICKS
        annual_pnl = net_edge * ES_TICK_VALUE * 2 * 252  # ~2 trades/day
        log.info(f"Expected gross edge: {expected_edge:.2f} ticks/trade")
        log.info(f"Net edge: {net_edge:.2f} ticks/trade")
        log.info(f"Estimated annual PnL (1 contract): ${annual_pnl:,.0f}")

    # Save results
    output_file = OUTPUT_DIR / "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {output_file}")

    return results


if __name__ == '__main__':
    main()
