#!/usr/bin/env python3
"""
Longer-Horizon Directional v2 — Clean Regime-Gated Validation (HC #428 R1)
==========================================================================

Re-runs the LH directional v1 LightGBM model with:
  1. FIXED tick scaling (close is in tick-units = price*4, so fwd_ticks / 0.25 → 4x inflated)
  2. Full HC #428 R1 regime stratification (green/red/flat per ES close-to-close ±0.10%)
  3. Trade simulation with canonical costs (1.376 market RT, 0.376 passive)
  4. Per-day Sharpe, day concentration
  5. Saves OOT predictions per fold for post-hoc analysis

SLIDING walk-forward (HC #0). 60-day train, 1-day OOT.

Author: Claude (HC #637/647/658 research)
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
OUTPUT_DIR = ROOT / "output" / "longer_horizon_v2_regime_gate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [LH-v2] %(levelname)s %(message)s',
    level=logging.INFO,
)
log = logging.getLogger('LH-v2')

# ── Cost constants (HC canonical) ──
ES_TICK_VALUE = 12.50
COST_MARKET_RT_TICKS = 1.376   # commission + 1 spread crossing
COST_PASSIVE_RT_TICKS = 0.376  # commission only

# Minimum edge to justify a trade (in REAL ticks, not inflated)
MIN_EDGE_TICKS = {'1h': 3.0, '2h': 4.0, '4h': 5.0, 'eod': 6.0}

# ──────────────────────────────────────────────
#  DATA LOADING
# ──────────────────────────────────────────────

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
    log.info(f"Loaded {len(combined)} minute bars, {len(frames)} days ({frames[0]['date'].iloc[0]}→{frames[-1]['date'].iloc[0]})")
    return combined


def compute_hourly_features(df):
    """Aggregate 1-min bars → hourly bars with microstructure features."""
    df = df.copy()
    df['hour'] = df['ts_minute'].dt.hour
    df['date_str'] = df['date']
    df['return_1m'] = df.groupby('date_str')['close'].pct_change()
    df['abs_ofi'] = df['ofi_1min'].abs()

    sv_std = df.groupby('date_str')['signed_volume'].transform('std').replace(0, 1)
    df['sv_zscore'] = df['signed_volume'] / sv_std
    df['vwap_dev'] = (df['close'] - df['vwap']) / df['close'].clip(lower=1)

    records = []
    for (date_str, hour), g in df.groupby(['date_str', 'hour']):
        if len(g) < 5:
            continue
        c = g['close'].values
        v = g['volume'].values
        ofi = g['ofi_1min'].values
        sv = g['signed_volume'].values
        ret = g['return_1m'].fillna(0).values
        sp = g['spread_mean'].values
        tc = g['trade_count'].values

        rec = {
            'date': date_str, 'hour': hour, 'ts': g['ts_minute'].iloc[0],
            'open': c[0], 'high': c.max(), 'low': c.min(), 'close': c[-1],
            'return_1h': (c[-1] / c[0] - 1) if c[0] > 0 else 0,
            'range_ticks': (c.max() - c.min()),  # Already in tick units
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
            'sweep_minutes': int(np.sum(np.abs(g['sv_zscore'].values) > 2)),
            'max_sweep_intensity': float(np.abs(g['sv_zscore'].values).max()),
            'spread_mean': sp.mean(),
            'spread_max': sp.max(),
            'trade_count_sum': tc.sum(),
            'trade_intensity': tc.mean(),
            'return_std': ret.std(),
            'return_skew': float(stats.skew(ret)) if len(ret) > 3 else 0,
            'realized_vol': ret.std() * np.sqrt(60),
            'vol_asymmetry': float(np.mean(ret[ret < 0]**2) / max(np.mean(ret[ret > 0]**2), 1e-10)) if (ret < 0).any() and (ret > 0).any() else 1.0,
        }
        records.append(rec)

    hourly = pd.DataFrame(records)
    hourly = hourly.sort_values('ts').reset_index(drop=True)
    log.info(f"Computed {len(hourly)} hourly bars")
    return hourly


def add_rolling_features(df):
    """Add multi-hour rolling features."""
    for w in [2, 4, 6]:
        lbl = f'{w}h'
        df[f'ofi_sum_{lbl}'] = df['ofi_sum'].rolling(w, min_periods=1).sum()
        df[f'ofi_trend_{lbl}'] = df['ofi_trend'].rolling(w, min_periods=1).mean()
        df[f'sv_sum_{lbl}'] = df['signed_volume_sum'].rolling(w, min_periods=1).sum()
        df[f'volume_ma_{lbl}'] = df['total_volume'].rolling(w, min_periods=1).mean()
        df[f'volume_vs_ma_{lbl}'] = df['total_volume'] / df[f'volume_ma_{lbl}'].clip(lower=1)
        df[f'vol_trend_{lbl}'] = df['realized_vol'].rolling(w, min_periods=1).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True)

    # Price momentum
    df['mom_2h'] = df['close'].pct_change(2)
    df['mom_4h'] = df['close'].pct_change(4)
    df['mom_6h'] = df['close'].pct_change(6)

    return df


def add_regime_context(df):
    """Add regime features (VIX proxy from vol, trend from momentum)."""
    # Realized vol regime (proxy for VIX)
    df['vol_20h'] = df['realized_vol'].rolling(20, min_periods=5).mean()
    df['vol_regime'] = pd.qcut(df['vol_20h'].rank(method='first'), 3, labels=[0, 1, 2]).astype(float)

    # Trend regime (rolling return sign)
    df['trend_8h'] = df['close'].pct_change(8)
    df['trend_20h'] = df['close'].pct_change(20)
    df['trend_regime'] = np.where(df['trend_20h'] > 0.005, 1, np.where(df['trend_20h'] < -0.005, -1, 0))

    return df


def add_forward_labels(df, horizons):
    """Add forward labels with CORRECT tick scaling.

    Close is stored in tick units (price * 4). So diff in close = diff in ticks directly.
    DO NOT divide by 0.25 again.
    """
    for label, h in horizons.items():
        if label == 'eod':
            # EOD = last bar of the day
            day_close = df.groupby('date')['close'].transform('last')
            df[f'fwd_ticks_{label}'] = day_close - df['close']  # Already in ticks
            df[f'fwd_return_{label}'] = day_close / df['close'] - 1
        elif label == '4h':
            fwd_close = df['close'].shift(-h).values
            ts_curr = df['ts'].values.astype(np.int64) // 10**9
            ts_fwd = pd.Series(df['ts']).shift(-h).values.astype(np.int64) // 10**9
            ts_diff = ts_fwd - ts_curr
            df[f'fwd_ticks_{label}'] = fwd_close - df['close'].values  # Already in ticks
            df[f'fwd_return_{label}'] = fwd_close / df['close'].values - 1
            # Mask cross-day (>8h gap means overnight)
            df.loc[ts_diff > 8 * 3600, f'fwd_ticks_{label}'] = np.nan
            df.loc[ts_diff > 8 * 3600, f'fwd_return_{label}'] = np.nan
        else:
            df[f'fwd_ticks_{label}'] = df.groupby('date')['close'].shift(-h) - df['close']
            df[f'fwd_return_{label}'] = df.groupby('date')['close'].shift(-h) / df['close'] - 1

    return df


def get_feature_cols(df):
    """Get feature columns (exclude labels, metadata)."""
    exclude = {'date', 'hour', 'ts', 'open', 'high', 'low', 'close'}
    exclude.update(c for c in df.columns if 'fwd_' in c or 'direction' in c)
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]


# ──────────────────────────────────────────────
#  WALK-FORWARD TRAINING
# ──────────────────────────────────────────────

def train_walk_forward(df, target_col, train_days=60, test_days=1):
    """Sliding walk-forward LightGBM. Returns per-fold OOT predictions."""
    import lightgbm as lgb

    feature_cols = get_feature_cols(df)
    dates = sorted(df['date'].unique())
    log.info(f"WF: {len(dates)} dates, {len(feature_cols)} features, target={target_col}")

    if len(dates) < train_days + test_days + 5:
        log.error(f"Not enough dates ({len(dates)})")
        return None

    params = {
        'objective': 'regression', 'metric': 'mse',
        'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': 6,
        'min_data_in_leaf': 50, 'feature_fraction': 0.7,
        'bagging_fraction': 0.8, 'bagging_freq': 5,
        'lambda_l1': 0.1, 'lambda_l2': 1.0,
        'verbose': -1, 'n_jobs': 8, 'seed': 42,
    }

    all_preds = []
    all_actuals = []
    all_dates_list = []
    all_hours = []
    importances = np.zeros(len(feature_cols))
    n_folds = 0

    for fold_start in range(train_days, len(dates) - test_days + 1, test_days):
        train_dates = dates[fold_start - train_days : fold_start]
        test_dates = dates[fold_start : fold_start + test_days]

        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'].isin(test_dates)

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, target_col].values

        tv = ~np.isnan(y_train)
        te = ~np.isnan(y_test)
        if tv.sum() < 50 or te.sum() < 2:
            continue

        X_train, y_train = X_train[tv], y_train[tv]
        X_test, y_test = X_test[te], y_test[te]
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

        td = lgb.Dataset(X_train, label=y_train)
        vd = lgb.Dataset(X_test, label=y_test, reference=td)

        model = lgb.train(params, td, num_boost_round=300,
                          valid_sets=[vd],
                          callbacks=[lgb.early_stopping(30, verbose=False)])

        preds = model.predict(X_test)
        all_preds.extend(preds)
        all_actuals.extend(y_test)
        all_dates_list.extend([test_dates[0]] * len(preds))
        test_hours = df.loc[test_mask, 'hour'].values[te]
        all_hours.extend(test_hours)

        importances += model.feature_importance(importance_type='gain')
        n_folds += 1

        if n_folds % 20 == 0:
            ic = np.corrcoef(all_preds, all_actuals)[0, 1]
            log.info(f"  Fold {n_folds}: running IC={ic:.4f}, {len(all_preds)} samples")

    log.info(f"WF complete: {n_folds} folds, {len(all_preds)} OOT samples")

    # Feature importances
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances / max(n_folds, 1)})
    imp_df = imp_df.sort_values('importance', ascending=False)

    return {
        'predictions': np.array(all_preds),
        'actuals': np.array(all_actuals),
        'dates': all_dates_list,
        'hours': all_hours,
        'feature_importance': imp_df,
        'n_folds': n_folds,
    }


# ──────────────────────────────────────────────
#  REGIME CLASSIFICATION
# ──────────────────────────────────────────────

def classify_days_regime(df):
    """Classify each day as green/red/flat by ES close-to-close (±0.10%)."""
    day_open_close = df.groupby('date').agg(
        day_open=('close', 'first'),
        day_close=('close', 'last')
    ).reset_index()
    day_open_close['day_return'] = day_open_close['day_close'] / day_open_close['day_open'] - 1

    regimes = {}
    for _, row in day_open_close.iterrows():
        r = row['day_return']
        if r > 0.001:
            regimes[row['date']] = 'green'
        elif r < -0.001:
            regimes[row['date']] = 'red'
        else:
            regimes[row['date']] = 'flat'

    return regimes


# ──────────────────────────────────────────────
#  TRADE SIMULATION + REGIME GATE
# ──────────────────────────────────────────────

def simulate_trades(preds, actuals, dates, cost_ticks=COST_MARKET_RT_TICKS, top_pct=0.20):
    """Simulate trades: go long top-pct, short bottom-pct, apply costs."""
    n = len(preds)
    abs_preds = np.abs(preds)
    threshold = np.percentile(abs_preds, (1 - top_pct) * 100)

    trades = []
    for i in range(n):
        if abs_preds[i] < threshold:
            continue
        direction = 1 if preds[i] > 0 else -1
        gross_ticks = direction * actuals[i]
        net_ticks = gross_ticks - cost_ticks
        trades.append({
            'date': dates[i],
            'direction': direction,
            'gross_ticks': gross_ticks,
            'net_ticks': net_ticks,
            'pred': preds[i],
            'actual': actuals[i],
        })

    if not trades:
        return None

    tdf = pd.DataFrame(trades)

    # Per-day PnL
    daily = tdf.groupby('date').agg(
        pnl_ticks=('net_ticks', 'sum'),
        n_trades=('net_ticks', 'count'),
        gross_ticks=('gross_ticks', 'sum'),
        wr=('net_ticks', lambda x: (x > 0).mean()),
    ).reset_index()

    total_pnl = daily['pnl_ticks'].sum()
    n_trades = tdf.shape[0]
    n_days = daily.shape[0]

    # Daily Sharpe (annualized)
    daily_returns = daily['pnl_ticks'].values
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-10
    sortino = daily_returns.mean() / downside_std * np.sqrt(252)

    # PF
    gross_wins = tdf.loc[tdf['net_ticks'] > 0, 'net_ticks'].sum()
    gross_losses = -tdf.loc[tdf['net_ticks'] < 0, 'net_ticks'].sum()
    pf = gross_wins / max(gross_losses, 1e-10)

    # WR
    wr = (tdf['net_ticks'] > 0).mean()

    # Day concentration
    best_day_pnl = daily['pnl_ticks'].max()
    day_conc = best_day_pnl / max(total_pnl, 1e-10) if total_pnl > 0 else 1.0

    # Max drawdown
    cum = daily['pnl_ticks'].cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    max_dd = dd.min()

    return {
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'pf': round(pf, 2),
        'wr': round(wr, 4),
        'n_trades': n_trades,
        'n_days': n_days,
        'total_pnl_ticks': round(total_pnl, 1),
        'total_pnl_dollars': round(total_pnl * ES_TICK_VALUE, 0),
        'day_conc': round(day_conc, 3),
        'max_dd_ticks': round(max_dd, 1),
        'trades_df': tdf,
        'daily_df': daily,
    }


def regime_gate(sim_result, regime_map):
    """HC #428 R1: regime-stratified validation."""
    if sim_result is None:
        return None

    daily = sim_result['daily_df'].copy()
    daily['regime'] = daily['date'].map(regime_map).fillna('unknown')

    regime_stats = {}
    for regime in ['green', 'red', 'flat']:
        rd = daily[daily['regime'] == regime]
        if len(rd) < 3:
            regime_stats[regime] = {'sharpe': np.nan, 'n_days': len(rd), 'pnl': 0}
            continue
        dr = rd['pnl_ticks'].values
        s = (dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0
        regime_stats[regime] = {
            'sharpe': round(s, 2),
            'n_days': len(rd),
            'pnl': round(dr.sum(), 1),
            'wr': round((dr > 0).mean(), 3),
        }

    # Regime gap
    sg = regime_stats.get('green', {}).get('sharpe', np.nan)
    sr = regime_stats.get('red', {}).get('sharpe', np.nan)
    if not np.isnan(sg) and not np.isnan(sr) and max(abs(sg), abs(sr)) > 0:
        gap = abs(sg - sr) / max(abs(sg), abs(sr))
    else:
        gap = np.nan

    return {
        'regime_stats': regime_stats,
        'regime_gap': round(gap, 3) if not np.isnan(gap) else None,
        'gap_pass': gap <= 0.50 if not np.isnan(gap) else False,
    }


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("LONGER-HORIZON v2 — REGIME-GATED VALIDATION")
    log.info("=" * 60)

    horizons = {'2h': 2, '4h': 4, 'eod': 0}

    # Step 1: Load data
    log.info("Step 1: Loading minute bars...")
    minute_df = load_minute_bars()

    # Step 2: Compute hourly features
    log.info("Step 2: Computing hourly features...")
    hourly = compute_hourly_features(minute_df)
    del minute_df; gc.collect()

    # Step 3: Rolling features + regime context
    log.info("Step 3: Rolling features + regime context...")
    hourly = add_rolling_features(hourly)
    hourly = add_regime_context(hourly)

    # Step 4: Forward labels (FIXED tick scaling)
    log.info("Step 4: Computing forward labels (corrected tick scaling)...")
    hourly = add_forward_labels(hourly, horizons)

    # Classify days for regime gate
    regime_map = classify_days_regime(hourly)
    n_green = sum(1 for v in regime_map.values() if v == 'green')
    n_red = sum(1 for v in regime_map.values() if v == 'red')
    n_flat = sum(1 for v in regime_map.values() if v == 'flat')
    log.info(f"Regime split: {n_green} green, {n_red} red, {n_flat} flat days")

    # Save labeled data
    hourly.to_parquet(OUTPUT_DIR / "hourly_features_labeled_v2.parquet", index=False)

    all_results = {}

    for horizon_label in horizons:
        target = f'fwd_ticks_{horizon_label}'
        if target not in hourly.columns:
            continue
        valid = hourly[target].notna().sum()
        log.info(f"\n{'='*50}")
        log.info(f"HORIZON: {horizon_label} ({valid} valid samples)")
        log.info(f"{'='*50}")

        # Train
        results = train_walk_forward(hourly, target, train_days=60, test_days=1)
        if results is None:
            continue

        preds = results['predictions']
        actuals = results['actuals']
        dates = results['dates']

        # IC metrics
        ic = np.corrcoef(preds, actuals)[0, 1]
        rank_ic = stats.spearmanr(preds, actuals)[0]

        # Per-day IC
        daily_ics = []
        for d in sorted(set(dates)):
            mask = [i for i, dd in enumerate(dates) if dd == d]
            if len(mask) < 3:
                continue
            p = preds[mask]
            a = actuals[mask]
            if np.std(p) > 0 and np.std(a) > 0:
                daily_ics.append(np.corrcoef(p, a)[0, 1])
        daily_ic_mean = np.mean(daily_ics) if daily_ics else 0
        daily_ic_sharpe = (np.mean(daily_ics) / np.std(daily_ics)) if daily_ics and np.std(daily_ics) > 0 else 0

        # Directional accuracy
        dir_correct = np.mean(np.sign(preds) == np.sign(actuals))

        log.info(f"  IC={ic:.4f}, RankIC={rank_ic:.4f}, DirAcc={dir_correct:.4f}")
        log.info(f"  Daily IC mean={daily_ic_mean:.4f}, IC Sharpe={daily_ic_sharpe:.3f}")
        log.info(f"  Top 5 features: {results['feature_importance'].head(5)['feature'].tolist()}")

        # Trade simulation at multiple filter levels
        horizon_results = {
            'ic': round(ic, 4),
            'rank_ic': round(rank_ic, 4),
            'daily_ic_mean': round(daily_ic_mean, 4),
            'daily_ic_sharpe': round(daily_ic_sharpe, 3),
            'dir_acc': round(dir_correct, 4),
            'n_oot_samples': len(preds),
            'n_folds': results['n_folds'],
            'top_features': results['feature_importance'].head(10)['feature'].tolist(),
            'simulations': {},
        }

        for top_pct_label, top_pct in [('top10', 0.10), ('top20', 0.20), ('top30', 0.30)]:
            for cost_label, cost in [('market', COST_MARKET_RT_TICKS), ('passive', COST_PASSIVE_RT_TICKS)]:
                sim = simulate_trades(preds, actuals, dates, cost_ticks=cost, top_pct=top_pct)
                if sim is None:
                    continue

                rg = regime_gate(sim, regime_map)
                key = f'{top_pct_label}_{cost_label}'

                log.info(f"\n  {key}: Sharpe={sim['sharpe']}, Sortino={sim['sortino']}, "
                         f"PF={sim['pf']}, WR={sim['wr']:.3f}, N={sim['n_trades']}, "
                         f"PnL={sim['total_pnl_ticks']:.0f}t (${sim['total_pnl_dollars']:.0f}), "
                         f"DayConc={sim['day_conc']:.3f}")

                if rg:
                    rs = rg['regime_stats']
                    log.info(f"    Green: Sharpe={rs.get('green',{}).get('sharpe','N/A')}, "
                             f"n={rs.get('green',{}).get('n_days',0)}")
                    log.info(f"    Red:   Sharpe={rs.get('red',{}).get('sharpe','N/A')}, "
                             f"n={rs.get('red',{}).get('n_days',0)}")
                    log.info(f"    Gap={rg['regime_gap']} (≤0.50 to pass): "
                             f"{'✅ PASS' if rg['gap_pass'] else '❌ FAIL'}")

                    # Full gate check
                    passes_all = (
                        sim['sharpe'] >= 1.0 and
                        sim['pf'] >= 1.2 and
                        sim['wr'] >= 0.45 and
                        rg['gap_pass'] and
                        sim['day_conc'] <= 0.70 and
                        sim['n_days'] >= 20 and
                        sim['n_trades'] >= 50
                    )
                    log.info(f"    FULL GATE: {'✅ PASS' if passes_all else '❌ FAIL'} "
                             f"(Sharpe≥1.0:{sim['sharpe']>=1.0}, PF≥1.2:{sim['pf']>=1.2}, "
                             f"WR≥0.45:{sim['wr']>=0.45}, Gap≤0.50:{rg['gap_pass']}, "
                             f"DayConc≤0.70:{sim['day_conc']<=0.70}, "
                             f"N_days≥20:{sim['n_days']>=20}, N_trades≥50:{sim['n_trades']>=50})")

                sim_summary = {k: v for k, v in sim.items() if k not in ('trades_df', 'daily_df')}
                sim_summary['regime'] = rg
                sim_summary['passes_full_gate'] = passes_all if rg else False
                horizon_results['simulations'][key] = sim_summary

                # Save trades for this config
                sim['trades_df'].to_parquet(OUTPUT_DIR / f"trades_{horizon_label}_{key}.parquet", index=False)

        all_results[horizon_label] = horizon_results

        # Save OOT predictions for this horizon
        pred_df = pd.DataFrame({
            'date': dates,
            'hour': results['hours'],
            'prediction': preds,
            'actual': actuals,
        })
        pred_df.to_parquet(OUTPUT_DIR / f"oot_predictions_{horizon_label}.parquet", index=False)

    # Save summary
    summary = {
        'run_time': datetime.now().isoformat(),
        'config': {'train_days': 60, 'test_days': 1, 'horizons': list(horizons.keys()),
                   'cost_market_rt': COST_MARKET_RT_TICKS, 'cost_passive_rt': COST_PASSIVE_RT_TICKS,
                   'tick_scaling': 'FIXED — close in tick units, no /0.25'},
        'regime_counts': {'green': n_green, 'red': n_red, 'flat': n_flat},
        'results': all_results,
    }
    with open(OUTPUT_DIR / 'summary_v2.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Print final verdict
    log.info("\n" + "=" * 60)
    log.info("FINAL VERDICT")
    log.info("=" * 60)
    any_pass = False
    for hz, res in all_results.items():
        for cfg, sim in res.get('simulations', {}).items():
            if sim.get('passes_full_gate', False):
                log.info(f"  ✅ {hz}/{cfg}: Sharpe={sim['sharpe']}, PF={sim['pf']}, "
                         f"WR={sim['wr']}, Gap={sim['regime']['regime_gap']}")
                any_pass = True
    if not any_pass:
        log.info("  ❌ NO CONFIG PASSES ALL GATES")

    elapsed = time.time() - t0
    log.info(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    log.info(f"Output saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
