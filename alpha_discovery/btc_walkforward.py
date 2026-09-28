#!/usr/bin/env python3
"""
BTC/USDT Walk-Forward Microstructure Backtest
==============================================
Proper walk-forward test: train on expanding window of past days, test on next day.
Uses aggTrades data to build 100ms bars with microstructure features.

Key advantage over ES:
  BTC vol ~3.5% daily vs ES ~0.1%
  BTC moves are ~35x larger relative to price
  So IC of 0.05-0.08 on BTC ≈ IC of 0.20 on ES in terms of profitability

Costs (Binance futures):
  Taker: 5bps each way = 10bps RT
  Maker: 2bps each way = 4bps RT
  Mixed (maker entry, taker exit): 7bps RT
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

logging.basicConfig(format='%(asctime)s [btc_wf] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger('btc_wf')

DATA_DIR = r"C:\Users\Footb\Documents\Github\Lvl3Quant\data\raw\crypto\aggTrades"
RESULTS_DIR = Path(r"C:\Users\Footb\Documents\Github\Lvl3Quant\alpha_discovery\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BAR_MS = 100  # 100ms bars

# Binance futures costs (basis points)
TAKER_FEE_BPS = 5.0   # each way
MAKER_FEE_BPS = 2.0   # each way
RT_TAKER_BPS = 2 * TAKER_FEE_BPS   # 10 bps
RT_MIXED_BPS = MAKER_FEE_BPS + TAKER_FEE_BPS  # 7 bps
RT_MAKER_BPS = 2 * MAKER_FEE_BPS   # 4 bps

# Feature windows
OFI_WINDOWS = [10, 20, 50, 100]  # bars (1s, 2s, 5s, 10s)
ARRIVAL_WINDOW = 20


def load_day(filepath):
    """Load one day of aggTrades and build 100ms bars with features."""
    df = pd.read_csv(filepath,
                     usecols=['price', 'quantity', 'transact_time', 'is_buyer_maker'],
                     dtype={'price': 'float64', 'quantity': 'float64',
                            'transact_time': 'int64', 'is_buyer_maker': 'str'})

    # Direction: is_buyer_maker='true' -> sell aggressor -> side=-1
    is_bm = df['is_buyer_maker'].str.lower() == 'true'
    df['side'] = np.where(is_bm, -1, 1).astype(np.int8)
    df['signed_qty'] = df['side'] * df['quantity']
    df['notional'] = df['price'] * df['quantity']

    # Build 100ms bars
    df['bar_key'] = (df['transact_time'] // BAR_MS).astype('int64') * BAR_MS

    buy_mask = df['side'] == 1
    sell_mask = df['side'] == -1

    g = df.groupby('bar_key', sort=True)
    bars = g['price'].agg(['first', 'last', 'count']).rename(
        columns={'first': 'open', 'last': 'close', 'count': 'trade_count'})
    bars['volume'] = g['quantity'].sum()
    bars['notional'] = g['notional'].sum()

    buy_trades = df[buy_mask]
    sell_trades = df[sell_mask]
    gb = buy_trades.groupby('bar_key', sort=True)['quantity']
    gs = sell_trades.groupby('bar_key', sort=True)['quantity']
    bars['buy_vol'] = gb.sum().reindex(bars.index, fill_value=0)
    bars['sell_vol'] = gs.sum().reindex(bars.index, fill_value=0)
    bars['buy_count'] = buy_trades.groupby('bar_key', sort=True).size().reindex(bars.index, fill_value=0)
    bars['sell_count'] = sell_trades.groupby('bar_key', sort=True).size().reindex(bars.index, fill_value=0)

    for col in ['open', 'close', 'volume', 'notional', 'buy_vol', 'sell_vol']:
        bars[col] = bars[col].astype('float64')

    safe_vol = bars['volume'].replace(0, np.nan)

    # Features
    # 1. Normalized OFI [-1, +1]
    bars['ofi_norm'] = (bars['buy_vol'] - bars['sell_vol']) / safe_vol

    # 2. VWAP deviation
    bars['vwap'] = bars['notional'] / safe_vol
    bars['vwap_dev'] = (bars['close'] - bars['vwap']) / bars['close']

    # 3. Rolling OFI at multiple windows
    for w in OFI_WINDOWS:
        bars[f'ofi_roll_{w}'] = bars['ofi_norm'].rolling(w, min_periods=1).mean()

    # 4. Trade imbalance (rolling buy fraction - 0.5)
    buy_ratio = bars['buy_vol'] / safe_vol
    bars['trade_imbalance'] = buy_ratio.rolling(20, min_periods=1).mean() - 0.5

    # 5. VPIN (|buy-sell|/total, rolling)
    bars['vpin'] = bars['ofi_norm'].abs().rolling(50, min_periods=10).mean()

    # 6. Arrival rate (trades per second, rolling)
    bars['arrival_rate'] = bars['trade_count'] / (BAR_MS / 1000.0)
    bars['arrival_rate_roll'] = bars['arrival_rate'].rolling(ARRIVAL_WINDOW, min_periods=1).mean()

    # 7. Large trade indicator
    vol_mean = bars['volume'].mean()
    vol_std = bars['volume'].std()
    if vol_std > 0:
        bars['large_trade'] = (bars['volume'] > vol_mean + 2.0 * vol_std).astype(float)
    else:
        bars['large_trade'] = 0.0

    # 8. Volume momentum (current bar vol vs rolling avg)
    bars['vol_ratio'] = bars['volume'] / bars['volume'].rolling(50, min_periods=5).mean()

    # 9. OFI acceleration (change of change)
    bars['ofi_accel'] = bars['ofi_norm'].diff()

    # 10. Buy-sell count imbalance (different from volume imbalance)
    safe_count = bars['trade_count'].replace(0, np.nan)
    bars['count_imbalance'] = (bars['buy_count'] - bars['sell_count']) / safe_count

    # 11. Rolling count imbalance
    bars['count_imb_roll'] = bars['count_imbalance'].rolling(20, min_periods=1).mean()

    # Forward returns at multiple horizons
    horizons = {
        '1s': 10,     # 10 bars = 1 second
        '5s': 50,     # 5 seconds
        '10s': 100,   # 10 seconds
        '30s': 300,   # 30 seconds
        '1min': 600,  # 1 minute
        '5min': 3000, # 5 minutes
    }
    for label, n in horizons.items():
        if n < len(bars):
            bars[f'fwd_ret_{label}'] = bars['close'].pct_change(n).shift(-n)

    return bars


def compute_daily_ics(bars, feature_cols, horizon_labels):
    """Compute per-day IC for all feature×horizon combinations."""
    results = {}
    for feat in feature_cols:
        for h in horizon_labels:
            ret_col = f'fwd_ret_{h}'
            if ret_col not in bars.columns:
                continue
            valid = bars[[feat, ret_col]].dropna()
            if len(valid) < 100:
                results[(feat, h)] = np.nan
            else:
                ic, _ = spearmanr(valid[feat], valid[ret_col])
                results[(feat, h)] = ic
    return results


def run_walkforward_lgbm(day_data_list, feature_cols, horizon='10s', min_train_days=5):
    """Walk-forward LightGBM test across days."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm not installed")
        return None

    ret_col = f'fwd_ret_{horizon}'
    lgbm_params = {
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 5,
        'min_child_samples': 100,
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

    for test_idx in range(min_train_days, len(day_data_list)):
        train_bars = pd.concat([d['bars'] for d in day_data_list[:test_idx]], ignore_index=True)
        test_bars = day_data_list[test_idx]['bars']

        # Subsample training data (every 10th bar to reduce overlap)
        train_valid = train_bars[feature_cols + [ret_col]].dropna()
        test_valid = test_bars[feature_cols + [ret_col]].dropna()

        if len(train_valid) < 1000 or len(test_valid) < 100:
            fold_ics.append(np.nan)
            continue

        # Subsample training to reduce autocorrelation
        spacing = max(1, len(train_valid) // 50000)
        train_sub = train_valid.iloc[::spacing]

        X_train = train_sub[feature_cols].values.astype(np.float32)
        y_train = train_sub[ret_col].values.astype(np.float32)
        X_test = test_valid[feature_cols].values.astype(np.float32)
        y_test = test_valid[ret_col].values.astype(np.float32)

        # Clean
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
        y_train = np.nan_to_num(y_train, nan=0, posinf=0, neginf=0)

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        model = lgb.train(lgbm_params, dtrain, num_boost_round=100)

        preds = model.predict(X_test)
        ic, _ = spearmanr(preds, y_test)
        fold_ics.append(ic if np.isfinite(ic) else np.nan)

        imp = model.feature_importance(importance_type='gain')
        if feature_imp_total is None:
            feature_imp_total = imp.astype(np.float64)
        else:
            feature_imp_total += imp

        if (test_idx - min_train_days) % 5 == 0 or test_idx == len(day_data_list) - 1:
            valid_ics = [x for x in fold_ics if np.isfinite(x)]
            mean_ic = np.mean(valid_ics) if valid_ics else 0
            logger.info(f"  [{test_idx+1}/{len(day_data_list)}] {day_data_list[test_idx]['date']}  "
                       f"IC={ic:+.4f}  mean_IC={mean_ic:+.4f}  train={len(X_train):,}")

        del X_train, y_train, dtrain, model
        gc.collect()

    return {
        'fold_ics': fold_ics,
        'feature_importance': feature_imp_total,
        'feature_names': feature_cols,
    }


def simulate_trading(day_bars, feature_cols, horizon, model_preds=None, cost_bps=RT_TAKER_BPS):
    """
    Simulate trading based on signal quantile.
    If model_preds given, use them. Otherwise use raw best feature.
    Returns daily PnL in basis points.
    """
    ret_col = f'fwd_ret_{horizon}'
    if ret_col not in day_bars.columns:
        return None

    if model_preds is not None:
        signal = model_preds
    else:
        signal = day_bars['ofi_roll_20'].values

    fwd_ret = day_bars[ret_col].values

    valid = np.isfinite(signal) & np.isfinite(fwd_ret)
    signal = signal[valid]
    fwd_ret = fwd_ret[valid]

    if len(signal) < 100:
        return None

    # Threshold: trade when signal > p75 (long) or < p25 (short)
    p25, p75 = np.percentile(signal, [25, 75])

    long_mask = signal > p75
    short_mask = signal < p25

    # PnL in bps
    long_pnl = fwd_ret[long_mask] * 10000 - cost_bps
    short_pnl = -fwd_ret[short_mask] * 10000 - cost_bps

    total_trades = len(long_pnl) + len(short_pnl)
    if total_trades == 0:
        return None

    all_pnl = np.concatenate([long_pnl, short_pnl])
    return {
        'total_pnl_bps': float(np.sum(all_pnl)),
        'mean_pnl_bps': float(np.mean(all_pnl)),
        'n_trades': total_trades,
        'win_rate': float(np.mean(all_pnl > 0)),
        'gross_pnl_bps': float(np.sum(np.concatenate([
            fwd_ret[long_mask] * 10000,
            -fwd_ret[short_mask] * 10000
        ]))),
    }


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = RESULTS_DIR / f'btc_walkforward_{timestamp}.log'
    fh = logging.FileHandler(str(log_path), mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s [btc_wf] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("BTC/USDT Walk-Forward Microstructure Backtest")
    logger.info("=" * 70)

    # Load all days
    files = sorted(glob.glob(os.path.join(DATA_DIR, "BTCUSDT_aggTrades_2026-*.csv")))
    logger.info(f"Found {len(files)} days of data")

    feature_cols = [
        'ofi_norm', 'ofi_roll_10', 'ofi_roll_20', 'ofi_roll_50', 'ofi_roll_100',
        'trade_imbalance', 'vpin', 'vwap_dev', 'arrival_rate_roll',
        'large_trade', 'vol_ratio', 'ofi_accel', 'count_imbalance', 'count_imb_roll',
    ]
    horizon_labels = ['1s', '5s', '10s', '30s', '1min', '5min']

    # Load data day by day
    day_data = []
    t0 = time.time()
    for i, fpath in enumerate(files):
        date = os.path.basename(fpath).replace('BTCUSDT_aggTrades_', '').replace('.csv', '')
        logger.info(f"Loading [{i+1}/{len(files)}] {date}...")
        try:
            bars = load_day(fpath)
            day_data.append({'date': date, 'bars': bars, 'n_bars': len(bars)})
            logger.info(f"  -> {len(bars):,} bars")
        except Exception as e:
            logger.error(f"  -> FAILED: {e}")
            continue

    elapsed = time.time() - t0
    total_bars = sum(d['n_bars'] for d in day_data)
    logger.info(f"\nLoaded {len(day_data)} days, {total_bars:,} bars ({elapsed:.1f}s)")

    # ── Phase 1: Daily IC Analysis ──────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 1: Daily IC Analysis (per-day Spearman correlation)")
    logger.info(f"{'='*70}")

    all_daily_ics = {(f, h): [] for f in feature_cols for h in horizon_labels}

    for d in day_data:
        ics = compute_daily_ics(d['bars'], feature_cols, horizon_labels)
        for key, ic in ics.items():
            if np.isfinite(ic):
                all_daily_ics[key].append(ic)

    # Print IC table
    logger.info(f"\n{'Feature':<25} " + " ".join(f"{h:>8}" for h in horizon_labels))
    logger.info("-" * (25 + 9 * len(horizon_labels)))

    best_ic = 0
    best_pair = None

    for feat in feature_cols:
        vals = []
        for h in horizon_labels:
            ics = all_daily_ics[(feat, h)]
            if ics:
                mean_ic = np.mean(ics)
                if abs(mean_ic) > abs(best_ic):
                    best_ic = mean_ic
                    best_pair = (feat, h)
                vals.append(f"{mean_ic:+.4f}")
            else:
                vals.append("    nan")
        logger.info(f"{feat:<25} " + " ".join(f"{v:>8}" for v in vals))

    # T-statistics for best features
    logger.info(f"\nT-STATISTICS (per-day IC consistency):")
    logger.info(f"{'Feature':<25} " + " ".join(f"{h:>8}" for h in horizon_labels))
    logger.info("-" * (25 + 9 * len(horizon_labels)))

    for feat in feature_cols:
        vals = []
        for h in horizon_labels:
            ics = all_daily_ics[(feat, h)]
            if len(ics) >= 5:
                arr = np.array(ics)
                mean_ic = np.mean(arr)
                std_ic = np.std(arr, ddof=1)
                t_stat = mean_ic / (std_ic / np.sqrt(len(arr))) if std_ic > 0 else 0
                vals.append(f"{t_stat:+.2f}")
            else:
                vals.append("    nan")
        logger.info(f"{feat:<25} " + " ".join(f"{v:>8}" for v in vals))

    logger.info(f"\nBest signal: {best_pair[0]} @ {best_pair[1]}, mean IC = {best_ic:+.4f}")

    # ── Phase 2: Walk-Forward LightGBM ──────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 2: Walk-Forward LightGBM (expanding window)")
    logger.info(f"{'='*70}")

    for horizon in ['10s', '30s', '1min']:
        logger.info(f"\n--- Horizon: {horizon} ---")
        wf_result = run_walkforward_lgbm(day_data, feature_cols, horizon=horizon, min_train_days=5)
        if wf_result is None:
            logger.info(f"  FAILED")
            continue

        valid_ics = [x for x in wf_result['fold_ics'] if np.isfinite(x)]
        if not valid_ics:
            logger.info(f"  No valid folds")
            continue

        arr = np.array(valid_ics)
        mean_ic = np.mean(arr)
        std_ic = np.std(arr, ddof=1)
        t_stat = mean_ic / (std_ic / np.sqrt(len(arr))) if std_ic > 0 else 0
        pct_pos = np.mean(arr > 0) * 100

        logger.info(f"\n  LightGBM @ {horizon}:")
        logger.info(f"    Folds: {len(arr)}")
        logger.info(f"    Mean IC: {mean_ic:+.4f}")
        logger.info(f"    Std IC:  {std_ic:.4f}")
        logger.info(f"    t-stat:  {t_stat:.2f}")
        logger.info(f"    Pct positive: {pct_pos:.1f}%")

        # Feature importance
        if wf_result['feature_importance'] is not None:
            imp = wf_result['feature_importance']
            top_idx = np.argsort(imp)[-5:][::-1]
            logger.info(f"    Top features:")
            for rank, j in enumerate(top_idx, 1):
                logger.info(f"      {rank}. {feature_cols[j]}: {imp[j]:.0f}")

    # ── Phase 3: Trading Simulation ─────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 3: Trading Simulation (quantile long/short)")
    logger.info(f"{'='*70}")

    for horizon in ['1s', '5s', '10s', '30s', '1min']:
        for cost_label, cost_bps in [('taker', RT_TAKER_BPS), ('mixed', RT_MIXED_BPS), ('maker', RT_MAKER_BPS)]:
            daily_pnls = []
            total_trades = 0
            for d in day_data:
                result = simulate_trading(d['bars'], feature_cols, horizon, cost_bps=cost_bps)
                if result:
                    daily_pnls.append(result['total_pnl_bps'])
                    total_trades += result['n_trades']

            if daily_pnls:
                arr = np.array(daily_pnls)
                mean_daily = np.mean(arr)
                std_daily = np.std(arr, ddof=1)
                sharpe = mean_daily / std_daily * np.sqrt(252) if std_daily > 0 else 0
                total_pnl = np.sum(arr)
                avg_trades = total_trades / len(daily_pnls)

                # Convert to dollar terms (assume 1 BTC position at ~$100K)
                dollar_pnl = total_pnl / 10000 * 100000

                logger.info(f"  {horizon:>5} {cost_label:>6}: total={total_pnl:+8.1f}bps  "
                           f"mean={mean_daily:+6.2f}bps/d  Sharpe={sharpe:+.2f}  "
                           f"trades/d={avg_trades:.0f}  ${dollar_pnl:+,.0f}")

    # ── Phase 4: Cost Analysis ──────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 4: Cost Breakeven Analysis")
    logger.info(f"{'='*70}")

    # What IC do we need to be profitable?
    for horizon in ['10s', '30s', '1min']:
        ret_col = f'fwd_ret_{horizon}'
        all_rets = []
        for d in day_data:
            r = d['bars'][ret_col].dropna().values
            all_rets.append(r)
        all_rets = np.concatenate(all_rets)
        avg_move_bps = np.mean(np.abs(all_rets)) * 10000

        breakeven_taker = RT_TAKER_BPS / avg_move_bps
        breakeven_mixed = RT_MIXED_BPS / avg_move_bps
        breakeven_maker = RT_MAKER_BPS / avg_move_bps

        logger.info(f"\n  {horizon}: avg |move| = {avg_move_bps:.2f} bps")
        logger.info(f"    Breakeven IC (taker 10bps): {breakeven_taker:.4f}")
        logger.info(f"    Breakeven IC (mixed  7bps): {breakeven_mixed:.4f}")
        logger.info(f"    Breakeven IC (maker  4bps): {breakeven_maker:.4f}")

    # ── Save Results ────────────────────────────────────────────────────────────
    results = {
        'timestamp': timestamp,
        'n_days': len(day_data),
        'total_bars': total_bars,
        'best_raw_ic': {'feature': best_pair[0] if best_pair else None,
                        'horizon': best_pair[1] if best_pair else None,
                        'mean_ic': float(best_ic)},
        'feature_cols': feature_cols,
        'dates': [d['date'] for d in day_data],
    }
    results_path = RESULTS_DIR / f'btc_walkforward_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")
    logger.info(f"Log saved to {log_path}")
    logger.info("\nDone.")


if __name__ == '__main__':
    main()
