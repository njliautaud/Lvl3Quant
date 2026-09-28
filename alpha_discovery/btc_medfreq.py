#!/usr/bin/env python3
"""
BTC/USDT Medium-Frequency Strategy (5min - 4hr horizons)
=========================================================
At longer horizons, costs become manageable:
  5min:  breakeven IC ~0.19 (maker)
  30min: breakeven IC ~0.08
  1hr:   breakeven IC ~0.04
  4hr:   breakeven IC ~0.01

Strategy ideas:
  1. Momentum (trend strength at multiple timeframes)
  2. Volume profile (buy/sell imbalance persistence)
  3. Volatility breakout (vol compression → explosion)
  4. Mean reversion (overextended moves snap back)
  5. Trade-size segmentation (institutional vs retail flow)

Data: 30 days BTCUSDT aggTrades (5.6GB), 90M trades
"""

import gc
import glob
import json
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr

logging.basicConfig(format='%(asctime)s [btc_mf] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger('btc_mf')

DATA_DIR = r"C:\Users\Footb\Documents\Github\Lvl3Quant\data\raw\crypto\aggTrades"
RESULTS_DIR = Path(r"C:\Users\Footb\Documents\Github\Lvl3Quant\alpha_discovery\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Use 1-second bars for medium-frequency (not 100ms)
BAR_MS = 1000  # 1 second bars

# Costs in bps
RT_TAKER_BPS = 10.0
RT_MIXED_BPS = 7.0
RT_MAKER_BPS = 4.0


def load_day_1s(filepath):
    """Load one day of aggTrades and build 1-second bars with medium-freq features."""
    df = pd.read_csv(filepath,
                     usecols=['price', 'quantity', 'transact_time', 'is_buyer_maker'],
                     dtype={'price': 'float64', 'quantity': 'float64',
                            'transact_time': 'int64', 'is_buyer_maker': 'str'})

    is_bm = df['is_buyer_maker'].str.lower() == 'true'
    df['side'] = np.where(is_bm, -1, 1).astype(np.int8)
    df['notional'] = df['price'] * df['quantity']
    df['bar_key'] = (df['transact_time'] // BAR_MS).astype('int64') * BAR_MS

    buy_mask = df['side'] == 1
    sell_mask = df['side'] == -1

    g = df.groupby('bar_key', sort=True)
    bars = g['price'].agg(['first', 'last', 'max', 'min', 'count']).rename(
        columns={'first': 'open', 'last': 'close', 'max': 'high', 'min': 'low', 'count': 'n_trades'})
    bars['volume'] = g['quantity'].sum()
    bars['notional'] = g['notional'].sum()
    bars['buy_vol'] = df[buy_mask].groupby('bar_key', sort=True)['quantity'].sum().reindex(bars.index, fill_value=0)
    bars['sell_vol'] = df[sell_mask].groupby('bar_key', sort=True)['quantity'].sum().reindex(bars.index, fill_value=0)

    # Large trade detection (>1 BTC per bar)
    bars['large_buy'] = df[buy_mask & (df['quantity'] > 1.0)].groupby('bar_key')['quantity'].sum().reindex(bars.index, fill_value=0)
    bars['large_sell'] = df[sell_mask & (df['quantity'] > 1.0)].groupby('bar_key')['quantity'].sum().reindex(bars.index, fill_value=0)

    for col in ['open', 'close', 'high', 'low', 'volume', 'notional', 'buy_vol', 'sell_vol', 'large_buy', 'large_sell']:
        bars[col] = bars[col].astype('float64')

    return bars


def compute_medium_freq_features(bars):
    """Compute medium-frequency features on 1-second bars."""
    safe_vol = bars['volume'].replace(0, np.nan)
    close = bars['close'].values
    N = len(bars)

    # === Returns at multiple horizons ===
    for w in [5, 15, 30, 60, 300, 900, 3600]:  # 5s to 1hr
        label = f'{w}s'
        bars[f'ret_{label}'] = bars['close'].pct_change(w)

    # === 1. MOMENTUM FEATURES ===
    # Trend strength: returns over lookback windows
    for w in [60, 300, 900, 1800, 3600]:  # 1m, 5m, 15m, 30m, 1hr
        label = f'{w}s'
        bars[f'mom_{label}'] = bars['close'].pct_change(w)

    # Momentum acceleration (change in momentum)
    for w in [300, 900, 3600]:
        label = f'{w}s'
        bars[f'mom_accel_{label}'] = bars[f'mom_{label}'].diff(w)

    # Trend consistency: sign(return) persistence
    ret_1s = bars['close'].pct_change()
    for w in [60, 300, 900]:
        label = f'{w}s'
        bars[f'trend_sign_{label}'] = ret_1s.apply(np.sign).rolling(w, min_periods=1).mean()

    # === 2. VOLUME FEATURES ===
    # OFI (order flow imbalance) at multiple windows
    bars['ofi_1s'] = (bars['buy_vol'] - bars['sell_vol']) / safe_vol

    for w in [60, 300, 900, 3600]:
        label = f'{w}s'
        bars[f'ofi_{label}'] = bars['ofi_1s'].rolling(w, min_periods=1).mean()

    # Volume relative to rolling average
    for w in [300, 3600]:
        label = f'{w}s'
        vol_ma = bars['volume'].rolling(w, min_periods=10).mean()
        bars[f'vol_ratio_{label}'] = bars['volume'].rolling(60, min_periods=1).mean() / vol_ma.replace(0, np.nan)

    # Buy/sell volume ratio trend
    buy_ratio = bars['buy_vol'] / safe_vol
    for w in [300, 900, 3600]:
        label = f'{w}s'
        bars[f'buy_ratio_{label}'] = buy_ratio.rolling(w, min_periods=1).mean()

    # === 3. VOLATILITY FEATURES ===
    # Realized volatility at multiple windows
    log_ret = np.log(bars['close'] / bars['close'].shift(1))
    for w in [60, 300, 900, 3600]:
        label = f'{w}s'
        bars[f'rvol_{label}'] = log_ret.rolling(w, min_periods=10).std() * np.sqrt(w)

    # Volatility ratio (short/long) — vol compression indicator
    bars['vol_compression'] = bars.get('rvol_300s', pd.Series(dtype=float)) / bars.get('rvol_3600s', pd.Series(dtype=float)).replace(0, np.nan)

    # High-low range as % of close
    for w in [300, 3600]:
        label = f'{w}s'
        roll_high = bars['high'].rolling(w, min_periods=1).max()
        roll_low = bars['low'].rolling(w, min_periods=1).min()
        bars[f'range_{label}'] = (roll_high - roll_low) / bars['close']

    # === 4. MEAN REVERSION FEATURES ===
    # Z-score of price relative to moving averages
    for w in [300, 900, 3600, 7200]:  # 5m, 15m, 1hr, 2hr
        label = f'{w}s'
        ma = bars['close'].rolling(w, min_periods=10).mean()
        std = bars['close'].rolling(w, min_periods=10).std()
        bars[f'zscore_{label}'] = (bars['close'] - ma) / std.replace(0, np.nan)

    # VWAP deviation
    cum_notional = bars['notional'].cumsum()
    cum_vol = bars['volume'].cumsum()
    bars['vwap_all'] = cum_notional / cum_vol.replace(0, np.nan)
    bars['vwap_dev'] = (bars['close'] - bars['vwap_all']) / bars['close']

    # === 5. INSTITUTIONAL FLOW ===
    # Large trade imbalance (>1 BTC trades)
    large_total = bars['large_buy'] + bars['large_sell']
    bars['large_imbalance'] = (bars['large_buy'] - bars['large_sell']) / large_total.replace(0, np.nan)

    # Rolling large trade imbalance
    for w in [300, 900, 3600]:
        label = f'{w}s'
        large_buy_roll = bars['large_buy'].rolling(w, min_periods=1).sum()
        large_sell_roll = bars['large_sell'].rolling(w, min_periods=1).sum()
        large_total_roll = large_buy_roll + large_sell_roll
        bars[f'large_imb_{label}'] = (large_buy_roll - large_sell_roll) / large_total_roll.replace(0, np.nan)

    # === 6. FORWARD RETURNS (targets) ===
    for w in [300, 900, 1800, 3600, 14400]:  # 5m, 15m, 30m, 1hr, 4hr
        label = f'{w}s'
        bars[f'fwd_{label}'] = bars['close'].pct_change(w).shift(-w)

    return bars


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = RESULTS_DIR / f'btc_medfreq_{timestamp}.log'
    fh = logging.FileHandler(str(log_path), mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s [btc_mf] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("BTC/USDT Medium-Frequency Strategy (5min-4hr)")
    logger.info("=" * 70)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "BTCUSDT_aggTrades_2026-*.csv")))
    logger.info(f"Found {len(files)} days")

    # Feature columns to test
    feature_cols = [
        # Momentum
        'mom_60s', 'mom_300s', 'mom_900s', 'mom_1800s', 'mom_3600s',
        'mom_accel_300s', 'mom_accel_900s', 'mom_accel_3600s',
        'trend_sign_60s', 'trend_sign_300s', 'trend_sign_900s',
        # Volume / OFI
        'ofi_60s', 'ofi_300s', 'ofi_900s', 'ofi_3600s',
        'vol_ratio_300s', 'vol_ratio_3600s',
        'buy_ratio_300s', 'buy_ratio_900s', 'buy_ratio_3600s',
        # Volatility
        'rvol_60s', 'rvol_300s', 'rvol_900s', 'rvol_3600s',
        'vol_compression',
        'range_300s', 'range_3600s',
        # Mean reversion
        'zscore_300s', 'zscore_900s', 'zscore_3600s', 'zscore_7200s',
        'vwap_dev',
        # Institutional flow
        'large_imbalance',
        'large_imb_300s', 'large_imb_900s', 'large_imb_3600s',
    ]

    target_horizons = ['300s', '900s', '1800s', '3600s', '14400s']
    horizon_labels = ['5min', '15min', '30min', '1hr', '4hr']

    # Load data
    day_data = []
    t0 = time.time()
    for i, fpath in enumerate(files):
        date = os.path.basename(fpath).replace('BTCUSDT_aggTrades_', '').replace('.csv', '')
        logger.info(f"Loading [{i+1}/{len(files)}] {date}...")
        try:
            bars = load_day_1s(fpath)
            bars = compute_medium_freq_features(bars)
            day_data.append({'date': date, 'bars': bars, 'n_bars': len(bars)})
            logger.info(f"  -> {len(bars):,} bars (1s)")
        except Exception as e:
            logger.error(f"  -> FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    elapsed = time.time() - t0
    total_bars = sum(d['n_bars'] for d in day_data)
    logger.info(f"\nLoaded {len(day_data)} days, {total_bars:,} bars ({elapsed:.1f}s)")

    # ── Phase 1: Daily IC Analysis ──────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 1: Daily IC Analysis")
    logger.info(f"{'='*70}")

    all_daily_ics = {}
    for feat in feature_cols:
        for h_sec, h_label in zip(target_horizons, horizon_labels):
            key = (feat, h_label)
            all_daily_ics[key] = []

    for d in day_data:
        bars = d['bars']
        for feat in feature_cols:
            if feat not in bars.columns:
                continue
            for h_sec, h_label in zip(target_horizons, horizon_labels):
                ret_col = f'fwd_{h_sec}'
                if ret_col not in bars.columns:
                    continue
                valid = bars[[feat, ret_col]].dropna()
                if len(valid) < 100:
                    continue
                ic = float(valid[feat].corr(valid[ret_col], method='spearman'))
                if np.isfinite(ic):
                    all_daily_ics[(feat, h_label)].append(ic)

    # Print IC table
    logger.info(f"\n{'Feature':<25} " + " ".join(f"{h:>8}" for h in horizon_labels))
    logger.info("-" * (25 + 9 * len(horizon_labels)))

    best_results = []

    for feat in feature_cols:
        vals = []
        for h_label in horizon_labels:
            ics = all_daily_ics.get((feat, h_label), [])
            if len(ics) >= 5:
                mean_ic = np.mean(ics)
                std_ic = np.std(ics, ddof=1)
                t_stat = mean_ic / (std_ic / np.sqrt(len(ics))) if std_ic > 0 else 0
                pct_pos = np.mean(np.array(ics) > 0) * 100
                best_results.append({
                    'feature': feat, 'horizon': h_label,
                    'mean_ic': mean_ic, 't_stat': t_stat, 'pct_pos': pct_pos,
                    'n_days': len(ics)
                })
                vals.append(f"{mean_ic:+.4f}")
            else:
                vals.append("    nan")
        logger.info(f"{feat:<25} " + " ".join(f"{v:>8}" for v in vals))

    # Sort by |IC| and show top 20
    best_results.sort(key=lambda x: abs(x['mean_ic']), reverse=True)
    logger.info(f"\n{'='*70}")
    logger.info("TOP 20 FEATURES BY |IC|")
    logger.info(f"{'='*70}")
    logger.info(f"{'Feature':<25} {'Horizon':<8} {'IC':>8} {'t-stat':>8} {'%pos':>6} {'N':>4}")
    logger.info("-" * 65)
    for r in best_results[:20]:
        logger.info(f"{r['feature']:<25} {r['horizon']:<8} {r['mean_ic']:+.4f}  {r['t_stat']:+6.2f}  {r['pct_pos']:5.1f}  {r['n_days']:4d}")

    # ── Phase 2: Breakeven Analysis ─────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 2: Breakeven IC Analysis")
    logger.info(f"{'='*70}")

    for h_sec, h_label in zip(target_horizons, horizon_labels):
        ret_col = f'fwd_{h_sec}'
        all_rets = []
        for d in day_data:
            r = d['bars'][ret_col].dropna().values
            all_rets.append(r)
        all_rets = np.concatenate(all_rets)
        avg_move_bps = np.mean(np.abs(all_rets)) * 10000

        be_taker = RT_TAKER_BPS / avg_move_bps
        be_mixed = RT_MIXED_BPS / avg_move_bps
        be_maker = RT_MAKER_BPS / avg_move_bps

        logger.info(f"\n  {h_label}: avg |move| = {avg_move_bps:.2f} bps")
        logger.info(f"    Breakeven IC (taker 10bps): {be_taker:.4f}")
        logger.info(f"    Breakeven IC (mixed  7bps): {be_mixed:.4f}")
        logger.info(f"    Breakeven IC (maker  4bps): {be_maker:.4f}")

        # What's our best IC at this horizon?
        best_at_h = [r for r in best_results if r['horizon'] == h_label]
        if best_at_h:
            top = best_at_h[0]
            ratio = abs(top['mean_ic']) / be_maker if be_maker > 0 else 0
            logger.info(f"    Best IC: {top['feature']} IC={top['mean_ic']:+.4f} ({ratio:.2f}x breakeven maker)")

    # ── Phase 3: Walk-Forward LightGBM ──────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 3: Walk-Forward LightGBM")
    logger.info(f"{'='*70}")

    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm not installed")
        return

    # Focus on 15min and 1hr horizons
    for h_sec, h_label in [('900s', '15min'), ('3600s', '1hr')]:
        logger.info(f"\n--- Horizon: {h_label} ---")
        ret_col = f'fwd_{h_sec}'
        h_bars = int(h_sec.replace('s', ''))

        lgbm_params = {
            'objective': 'regression',
            'metric': 'mse',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 5,
            'min_child_samples': 50,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'verbose': -1,
            'n_jobs': 8,
            'seed': 42,
        }

        fold_ics = []
        feature_imp_total = None
        MIN_TRAIN_DAYS = 5

        # Available features (must exist in all days)
        avail_feats = [f for f in feature_cols if all(f in d['bars'].columns for d in day_data)]

        for test_idx in range(MIN_TRAIN_DAYS, len(day_data)):
            # Subsample: use every Nth second to reduce autocorrelation
            spacing = max(h_bars // 2, 30)  # Half the horizon or 30s

            train_frames = []
            for d in day_data[:test_idx]:
                sub = d['bars'][avail_feats + [ret_col]].dropna().iloc[::spacing]
                train_frames.append(sub)

            test_frame = day_data[test_idx]['bars'][avail_feats + [ret_col]].dropna().iloc[::spacing]

            if len(test_frame) < 20:
                fold_ics.append(np.nan)
                continue

            train_df = pd.concat(train_frames, ignore_index=True)
            if len(train_df) < 200:
                fold_ics.append(np.nan)
                continue

            X_train = train_df[avail_feats].values.astype(np.float32)
            y_train = train_df[ret_col].values.astype(np.float32)
            X_test = test_frame[avail_feats].values.astype(np.float32)
            y_test = test_frame[ret_col].values.astype(np.float32)

            X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
            X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
            y_train = np.nan_to_num(y_train, nan=0, posinf=0, neginf=0)

            dtrain = lgb.Dataset(X_train, label=y_train, feature_name=avail_feats)
            model = lgb.train(lgbm_params, dtrain, num_boost_round=200)

            preds = model.predict(X_test)

            # Check if predictions are constant
            if np.std(preds) < 1e-12:
                fold_ics.append(np.nan)
                continue

            ic, _ = spearmanr(preds, y_test)
            fold_ics.append(ic if np.isfinite(ic) else np.nan)

            imp = model.feature_importance(importance_type='gain')
            if feature_imp_total is None:
                feature_imp_total = imp.astype(np.float64)
            else:
                feature_imp_total += imp

            if (test_idx - MIN_TRAIN_DAYS) % 5 == 0 or test_idx == len(day_data) - 1:
                valid_ics = [x for x in fold_ics if np.isfinite(x)]
                mean_ic = np.mean(valid_ics) if valid_ics else 0
                logger.info(f"  [{test_idx+1}/{len(day_data)}] {day_data[test_idx]['date']}  "
                           f"IC={ic:+.4f}  mean_IC={mean_ic:+.4f}  train={len(X_train):,}")

            del X_train, y_train, dtrain, model
            gc.collect()

        valid_ics = [x for x in fold_ics if np.isfinite(x)]
        if valid_ics:
            arr = np.array(valid_ics)
            mean_ic = np.mean(arr)
            std_ic = np.std(arr, ddof=1) if len(arr) > 1 else 0
            t_stat = mean_ic / (std_ic / np.sqrt(len(arr))) if std_ic > 0 else 0
            pct_pos = np.mean(arr > 0) * 100

            logger.info(f"\n  LightGBM @ {h_label}:")
            logger.info(f"    Valid folds: {len(arr)}/{len(fold_ics)}")
            logger.info(f"    Mean IC: {mean_ic:+.4f}")
            logger.info(f"    Std IC:  {std_ic:.4f}")
            logger.info(f"    t-stat:  {t_stat:.2f}")
            logger.info(f"    Pct positive: {pct_pos:.1f}%")

            # Feature importance
            if feature_imp_total is not None:
                top_idx = np.argsort(feature_imp_total)[-10:][::-1]
                logger.info(f"    Top 10 features:")
                for rank, j in enumerate(top_idx, 1):
                    logger.info(f"      {rank:2d}. {avail_feats[j]:<25}: {feature_imp_total[j]:.0f}")

            # Trading simulation
            all_rets = np.concatenate([d['bars'][ret_col].dropna().values for d in day_data[MIN_TRAIN_DAYS:]])
            avg_move_bps = np.mean(np.abs(all_rets)) * 10000
            be_maker = RT_MAKER_BPS / avg_move_bps

            edge_bps = abs(mean_ic) * avg_move_bps
            net_pnl_bps = edge_bps - RT_MAKER_BPS

            logger.info(f"\n    Economics @ {h_label}:")
            logger.info(f"      Avg |move|: {avg_move_bps:.2f} bps")
            logger.info(f"      Model edge: {edge_bps:.4f} bps/trade")
            logger.info(f"      Cost (maker): {RT_MAKER_BPS:.1f} bps")
            logger.info(f"      Net PnL: {net_pnl_bps:+.4f} bps/trade")
            logger.info(f"      IC/breakeven: {abs(mean_ic)/be_maker:.2f}x")

            if net_pnl_bps > 0:
                trades_per_day = 86400 / (int(h_sec.replace('s', '')) / 2)  # conservative
                daily_bps = net_pnl_bps * trades_per_day * 0.5  # assume 50% position rate
                daily_dollars = daily_bps / 10000 * 100000  # 1 BTC position
                logger.info(f"      PROFITABLE! ~{daily_dollars:+.0f}/day per BTC")
            else:
                logger.info(f"      NOT PROFITABLE. Need {be_maker:.4f} IC, have {abs(mean_ic):.4f}")
        else:
            logger.info(f"\n  No valid folds at {h_label}")

    # Save results
    results = {
        'timestamp': timestamp,
        'n_days': len(day_data),
        'total_bars': total_bars,
        'top_features': best_results[:30],
    }
    results_path = RESULTS_DIR / f'btc_medfreq_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")
    logger.info(f"Log saved to {log_path}")
    logger.info("\nDone.")


if __name__ == '__main__':
    main()
