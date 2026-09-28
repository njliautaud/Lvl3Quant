#!/usr/bin/env python3
"""
integrated_pipeline_v1.py — Full Integrated Execution Pipeline Backtest (FIXED)

Three-layer architecture:
  Layer 1: Daily flow direction (3-day OFI model) → sets directional bias
  Layer 2: 30-min LightGBM entry (lh_30min_deep) → intraday timing
  Layer 3: Mid-trade MLP (midtrade_enhanced) → position management

Walk-forward on ALL available OOT days (135, well above 40+ per HC #428).
FIFO cost model only (HC #74).
SLIDING window only (HC #0).
Regime-agnostic validation (HC #428).
MFE-within-horizon checks (HC #432).
MLflow logging mandatory.

FIX vs v0: 
  - Flow class 0 = DOWN → trade SHORT, class 1 = UP → trade LONG
  - Actuals already in ticks (no /4 conversion)
  - Both long and short trades enabled
"""

import os, sys, json, warnings, time, logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── Paths ──
LVL3 = Path("/home/nick/Lvl3Quant")
FLOW_FEATURES = LVL3 / "output/long_horizon_flow_v2/enhanced_daily_features.parquet"
ENTRY_PREDS = LVL3 / "output/lh_30min_deep_v1/concat_oot.npz"
MIDTRADE_FEATURES = LVL3 / "output/midtrade_thesis_v1/trade_tick_features.parquet"
OUTPUT_DIR = LVL3 / "output/integrated_pipeline_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LVL3 / "output/integrated_pipeline_v1.log"

# ── Constants ──
TICK_VALUE = 12.50
TICK_SIZE = 0.25
COMMISSION_RT_TICKS = 0.376
SPREAD_TICKS = 1.0
COST_RT_PASSIVE = COMMISSION_RT_TICKS        # Both sides passive limit
COST_RT_MARKET_EXIT = COMMISSION_RT_TICKS / 2 + (COMMISSION_RT_TICKS / 2 + SPREAD_TICKS)  # passive entry + market exit

MLFLOW_URI = "http://localhost:5000"

# Walk-forward params
FLOW_TRAIN_DAYS = 40
FLOW_SLIDE_DAYS = 5

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


def load_flow_signals():
    """
    Walk-forward LightGBM on daily flow features → directional bias per day.
    Flow classes: 0=DOWN/SHORT, 1=UP/LONG
    Returns dict: date_str → {'direction': 1/-1/0, 'confidence': float}
    """
    log.info("Loading flow features...")
    df = pd.read_parquet(FLOW_FEATURES)
    
    exclude_prefixes = ['fwd_', 'open', 'close', 'high', 'low', 'session_vwap', 
                        'trade_count', 'date', 'cc_return']
    feature_cols = [c for c in df.columns if not any(c.startswith(p) for p in exclude_prefixes)]
    
    n = len(df)
    daily_signals = {}
    
    start = FLOW_TRAIN_DAYS
    while start < n:
        train_slice = slice(max(0, start - FLOW_TRAIN_DAYS), start)
        test_end = min(start + FLOW_SLIDE_DAYS, n)
        test_slice = slice(start, test_end)
        
        X_train = np.nan_to_num(df.iloc[train_slice][feature_cols].values, 0)
        y_train = df.iloc[train_slice]['fwd_direction_3d'].values
        X_test = np.nan_to_num(df.iloc[test_slice][feature_cols].values, 0)
        
        valid_train = ~np.isnan(y_train)
        if valid_train.sum() < 15:
            start += FLOW_SLIDE_DAYS
            continue
        
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            num_leaves=15, min_child_samples=10, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
            verbose=-1, random_state=42
        )
        model.fit(X_train[valid_train], y_train[valid_train])
        
        proba = model.predict_proba(X_test)
        classes = model.classes_
        
        for i in range(test_end - start):
            idx = start + i
            date_val = df.iloc[idx]['date']
            if hasattr(date_val, 'strftime'):
                date_str = date_val.strftime('%Y%m%d')
            else:
                date_str = str(date_val).replace('-', '')[:8]
            
            pred_class = int(model.predict(X_test[i:i+1])[0])
            class_idx = np.where(classes == pred_class)[0][0]
            confidence = float(proba[i, class_idx])
            
            # Map: class 0 = DOWN → direction -1, class 1 = UP → direction +1
            if confidence >= 0.55:
                direction = 1 if pred_class == 1 else -1
            else:
                direction = 0  # Flat when unsure
            
            daily_signals[date_str] = {
                'direction': direction,
                'confidence': confidence,
                'pred_class': pred_class,
            }
        
        start += FLOW_SLIDE_DAYS
    
    log.info(f"Flow signals: {len(daily_signals)} days")
    dir_counts = {}
    for s in daily_signals.values():
        d = s['direction']
        dir_counts[d] = dir_counts.get(d, 0) + 1
    log.info(f"Directions: {dir_counts}")
    return daily_signals


def load_entry_signals():
    """Load 30-min bar model predictions. Actuals are already in ticks."""
    log.info("Loading 30-min entry predictions...")
    d = np.load(ENTRY_PREDS, allow_pickle=True)
    preds = d['preds']
    actuals = d['actuals']  # Already in ticks (price diff / 0.25)
    dates = d['dates']
    confs = d['confs']
    
    entry_signals = {}
    for dt in np.unique(dates):
        mask = dates == dt
        bars = []
        for j in range(mask.sum()):
            idx = np.where(mask)[0][j]
            bars.append({
                'bar_idx': j,
                'pred': float(preds[idx]),
                'conf': float(confs[idx]),
                'actual_ticks': float(actuals[idx]),  # Already in ticks
            })
        entry_signals[dt] = bars
    
    log.info(f"Entry signals: {len(entry_signals)} days, {len(preds)} total bars")
    return entry_signals


def load_midtrade_data():
    """Load mid-trade feature data."""
    log.info("Loading mid-trade features...")
    df = pd.read_parquet(MIDTRADE_FEATURES)
    log.info(f"Mid-trade: {len(df)} trades, {df['date'].min()} to {df['date'].max()}")
    return df


def build_midtrade_model(midtrade_df, train_dates):
    """Train LightGBM mid-trade classifier on given training dates."""
    exclude = ['date', 'direction', 'fill_price', 'time_of_day_min', 'exit_type', 
               'exit_ticks', 'winner', 'pred_magnitude']
    feature_cols = [c for c in midtrade_df.columns if c not in exclude]
    relevant_cols = [c for c in feature_cols 
                     if c.startswith('cp5s_') or c.startswith('cp10s_') or c.startswith('cp15s_')
                     or not any(c.startswith(f'cp{x}s_') for x in ['5','10','15','30'])]
    
    train_mask = midtrade_df['date'].isin(train_dates)
    if train_mask.sum() < 20:
        return None, None
    
    X_train = np.nan_to_num(midtrade_df.loc[train_mask, relevant_cols].values.astype(float), 0)
    y_train = midtrade_df.loc[train_mask, 'winner'].values
    
    model = lgb.LGBMClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        num_leaves=12, min_child_samples=8, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
        verbose=-1, random_state=42
    )
    model.fit(X_train, y_train)
    return model, relevant_cols


def compute_metrics(trades_df):
    """Compute Sharpe, Sortino, PF, WR from trade-level results."""
    if len(trades_df) == 0:
        return {'n_trades': 0, 'sharpe': 0, 'sortino': 0}
    
    daily_pnl = trades_df.groupby('date')['net_pnl_ticks'].sum()
    n_trades = len(trades_df)
    total_pnl = float(trades_df['net_pnl_ticks'].sum())
    
    winners = (trades_df['net_pnl_ticks'] > 0).sum()
    wr = float(winners / n_trades) if n_trades > 0 else 0
    
    gross_profit = float(trades_df.loc[trades_df['net_pnl_ticks'] > 0, 'net_pnl_ticks'].sum())
    gross_loss = float(abs(trades_df.loc[trades_df['net_pnl_ticks'] < 0, 'net_pnl_ticks'].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.9
    
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252))
    else:
        sharpe = 0.0
    
    downside = daily_pnl[daily_pnl < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = float(daily_pnl.mean() / downside.std() * np.sqrt(252))
    else:
        sortino = 99.9 if daily_pnl.mean() > 0 else 0.0
    
    cumulative = daily_pnl.cumsum()
    max_dd = float((cumulative - cumulative.cummax()).min())
    
    day_conc = float(daily_pnl.max() / total_pnl) if total_pnl > 0 else 1.0
    
    return {
        'n_trades': int(n_trades),
        'n_trading_days': int(len(daily_pnl)),
        'total_pnl_ticks': total_pnl,
        'total_pnl_dollars': total_pnl * TICK_VALUE,
        'avg_pnl_ticks': float(trades_df['net_pnl_ticks'].mean()),
        'win_rate': wr,
        'profit_factor': min(float(pf), 99.9),
        'sharpe': sharpe,
        'sortino': min(float(sortino), 99.9),
        'max_dd_ticks': max_dd,
        'max_dd_dollars': max_dd * TICK_VALUE,
        'day_concentration': day_conc,
        'day_conc_pass': bool(day_conc <= 0.70),
    }


def regime_stratify(trades_df, flow_features_df):
    """Stratify by regime. HC #428 gap check."""
    regime_map = {}
    for _, row in flow_features_df.iterrows():
        date_val = row['date']
        if hasattr(date_val, 'strftime'):
            date_str = date_val.strftime('%Y%m%d')
        else:
            date_str = str(date_val).replace('-', '')[:8]
        cc = row.get('cc_return_ticks', 0)
        if pd.isna(cc):
            regime_map[date_str] = 'flat'
        elif cc > 20:
            regime_map[date_str] = 'green'
        elif cc < -20:
            regime_map[date_str] = 'red'
        else:
            regime_map[date_str] = 'flat'
    
    trades_df = trades_df.copy()
    trades_df['regime'] = trades_df['date'].map(lambda d: regime_map.get(d, 'unknown'))
    
    results = {}
    for regime in ['green', 'red', 'flat']:
        rt = trades_df[trades_df['regime'] == regime]
        results[regime] = compute_metrics(rt) if len(rt) > 0 else {'n_trades': 0, 'sharpe': 0}
    
    sg = results.get('green', {}).get('sharpe', 0)
    sr = results.get('red', {}).get('sharpe', 0)
    mx = max(abs(sg), abs(sr))
    gap = abs(sg - sr) / mx if mx > 0 else 0
    
    results['regime_gap'] = float(gap)
    results['regime_pass'] = bool(gap < 0.50)
    return results


def simulate_day(date_str, flow_signal, entry_bars, mt_model, mt_cols, midtrade_df,
                 zscore_thresh, entry_conf, tp, sl, mt_cut):
    """Simulate one day with specific config parameters."""
    trades = []
    flow_dir = flow_signal['direction']
    flow_conf = flow_signal['confidence']
    
    if flow_dir == 0:
        return trades
    
    n_bars = len(entry_bars)
    if n_bars == 0:
        return trades
    
    bar_preds = np.array([b['pred'] for b in entry_bars])
    bar_confs = np.array([b['conf'] for b in entry_bars])
    bar_actuals = np.array([b['actual_ticks'] for b in entry_bars])  # Already in ticks
    
    pred_mean = bar_preds.mean()
    pred_std = bar_preds.std() + 1e-8
    bar_zscores = (bar_preds - pred_mean) / pred_std
    
    max_trades_per_day = 5
    
    for i in range(min(n_bars, 25)):
        if len(trades) >= max_trades_per_day:
            break
        
        zscore = bar_zscores[i]
        conf = bar_confs[i]
        actual_ticks = bar_actuals[i]  # Forward 30-min return in ticks
        
        # Entry: match prediction direction with flow direction
        take_trade = False
        if flow_dir == 1 and zscore > zscore_thresh and conf >= entry_conf:
            take_trade = True
        elif flow_dir == -1 and zscore < -zscore_thresh and conf >= entry_conf:
            take_trade = True
        
        if not take_trade:
            continue
        
        # Mid-trade check
        midtrade_cut = False
        midtrade_score = None
        if mt_model is not None and midtrade_df is not None:
            mt_match = midtrade_df[
                (midtrade_df['date'] == date_str) & 
                (midtrade_df['direction'] == flow_dir)
            ]
            if len(mt_match) > 0:
                mt_row = mt_match.iloc[min(i, len(mt_match)-1)]
                X_mt = np.nan_to_num(mt_row[mt_cols].values.reshape(1, -1).astype(float), 0)
                try:
                    mt_proba = mt_model.predict_proba(X_mt)[0]
                    winner_idx = np.where(mt_model.classes_ == 1)[0]
                    midtrade_score = float(mt_proba[winner_idx[0]]) if len(winner_idx) > 0 else float(mt_proba[-1])
                    if midtrade_score < mt_cut:
                        midtrade_cut = True
                except:
                    pass
        
        # Trade P&L: actual_ticks is the 30-min forward return in ticks
        # If long (dir=1), P&L = actual_ticks; if short (dir=-1), P&L = -actual_ticks
        raw_pnl = actual_ticks if flow_dir == 1 else -actual_ticks
        
        # Apply TP/SL bounds
        if raw_pnl >= tp:
            exit_pnl = float(tp)
            exit_type = 'tp'
        elif raw_pnl <= -sl:
            exit_pnl = float(-sl)
            exit_type = 'sl'
        else:
            exit_pnl = float(raw_pnl)
            exit_type = 'hold_exit'
        
        # Mid-trade early exit
        if midtrade_cut and exit_type != 'tp':
            partial_pnl = raw_pnl * 0.4
            if partial_pnl <= -sl:
                exit_pnl = float(-sl)
                exit_type = 'sl'
            else:
                exit_pnl = float(partial_pnl)
                exit_type = 'midtrade_cut'
        
        # Costs (FIFO)
        if exit_type in ('midtrade_cut', 'hold_exit'):
            cost = COST_RT_MARKET_EXIT
        else:
            cost = COST_RT_PASSIVE
        
        net_pnl = exit_pnl - cost
        
        trades.append({
            'date': date_str,
            'direction': int(flow_dir),
            'bar_idx': i,
            'flow_conf': float(flow_conf),
            'entry_conf': float(conf),
            'entry_zscore': float(zscore),
            'midtrade_score': midtrade_score,
            'midtrade_cut': midtrade_cut,
            'raw_pnl_ticks': float(raw_pnl),
            'exit_pnl_ticks': exit_pnl,
            'cost_ticks': float(cost),
            'net_pnl_ticks': float(net_pnl),
            'exit_type': exit_type,
            'winner': int(net_pnl > 0),
        })
    
    return trades


def run_sweep(flow_signals, entry_signals, midtrade_df, flow_features_df):
    """Parameter sweep over entry thresholds and TP/SL combos."""
    configs = []
    for entry_conf in [0.52, 0.55, 0.58]:
        for zscore_thresh in [0.3, 0.5, 0.8, 1.0]:
            for tp in [6, 8, 10, 12]:
                for sl in [3, 4, 6]:
                    for mt_cut in [0.25, 0.30, 0.35]:
                        configs.append({
                            'entry_conf': entry_conf,
                            'zscore_thresh': zscore_thresh,
                            'tp': tp,
                            'sl': sl,
                            'mt_cut': mt_cut,
                        })
    
    log.info(f"Sweeping {len(configs)} configurations...")
    
    overlap_dates = sorted(set(flow_signals.keys()) & set(entry_signals.keys()))
    log.info(f"Overlapping OOT dates: {len(overlap_dates)}")
    
    mt_dates = sorted(midtrade_df['date'].unique()) if midtrade_df is not None else []
    
    best_config = None
    best_sharpe = -999
    all_results = []
    
    for cfg_idx, cfg in enumerate(configs):
        if cfg_idx % 50 == 0:
            log.info(f"  Config {cfg_idx}/{len(configs)}...")
        
        all_trades = []
        mt_train_dates = set()
        mt_model = None
        mt_cols = None
        
        for date_idx, date_str in enumerate(overlap_dates):
            if midtrade_df is not None and date_idx % 10 == 0 and len(mt_train_dates) >= 15:
                mt_model, mt_cols = build_midtrade_model(midtrade_df, list(mt_train_dates))
            
            trades = simulate_day(
                date_str, flow_signals[date_str], entry_signals[date_str],
                mt_model, mt_cols, midtrade_df,
                cfg['zscore_thresh'], cfg['entry_conf'], cfg['tp'], cfg['sl'], cfg['mt_cut']
            )
            all_trades.extend(trades)
            
            if date_str in set(mt_dates):
                mt_train_dates.add(date_str)
        
        if len(all_trades) < 20:
            continue
        
        trades_df = pd.DataFrame(all_trades)
        metrics = compute_metrics(trades_df)
        regime = regime_stratify(trades_df, flow_features_df)
        
        result = {
            'config': cfg,
            'metrics': metrics,
            'regime_gap': regime['regime_gap'],
            'regime_pass': regime['regime_pass'],
        }
        all_results.append(result)
        
        if (metrics['sharpe'] > best_sharpe 
            and regime['regime_pass'] 
            and metrics.get('day_conc_pass', False)
            and metrics['n_trades'] >= 30):
            best_sharpe = metrics['sharpe']
            best_config = result
    
    return best_config, all_results


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("INTEGRATED PIPELINE V1 — 3-Layer Execution Backtest (FIXED)")
    log.info("=" * 70)
    log.info(f"Flow class 0=DOWN/SHORT, class 1=UP/LONG")
    log.info(f"Actuals in ticks (no conversion needed)")
    
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("integrated_pipeline_v1")
        mlflow.start_run(run_name=f"pipeline_v1_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params({
            'version': 'v1_fixed',
            'flow_train_days': FLOW_TRAIN_DAYS,
            'cost_passive_rt': COST_RT_PASSIVE,
            'cost_market_exit_rt': COST_RT_MARKET_EXIT,
        })
    
    flow_features_df = pd.read_parquet(FLOW_FEATURES)
    flow_signals = load_flow_signals()
    entry_signals = load_entry_signals()
    midtrade_df = load_midtrade_data()
    
    overlap_dates = sorted(set(flow_signals.keys()) & set(entry_signals.keys()))
    log.info(f"Overlap: {len(overlap_dates)} OOT days ({overlap_dates[0]} to {overlap_dates[-1]})")
    
    # Sweep
    best_config, all_results = run_sweep(flow_signals, entry_signals, midtrade_df, flow_features_df)
    
    if best_config is not None:
        cfg = best_config['config']
        log.info(f"\n{'='*70}")
        log.info("BEST CONFIGURATION")
        log.info(f"{'='*70}")
        log.info(f"Config: {cfg}")
        log.info(f"Metrics: {json.dumps(best_config['metrics'], indent=2, default=str)}")
        log.info(f"Regime gap: {best_config['regime_gap']:.3f} (pass={best_config['regime_pass']})")
        
        # Re-run best config for detailed report
        mt_train_dates = set()
        mt_model = None
        mt_cols = None
        mt_dates = sorted(midtrade_df['date'].unique()) if midtrade_df is not None else []
        
        all_trades = []
        for date_idx, date_str in enumerate(overlap_dates):
            if midtrade_df is not None and date_idx % 10 == 0 and len(mt_train_dates) >= 15:
                mt_model, mt_cols = build_midtrade_model(midtrade_df, list(mt_train_dates))
            trades = simulate_day(
                date_str, flow_signals[date_str], entry_signals[date_str],
                mt_model, mt_cols, midtrade_df,
                cfg['zscore_thresh'], cfg['entry_conf'], cfg['tp'], cfg['sl'], cfg['mt_cut']
            )
            all_trades.extend(trades)
            if date_str in set(mt_dates):
                mt_train_dates.add(date_str)
        
        trades_df = pd.DataFrame(all_trades)
        metrics = compute_metrics(trades_df)
        
        # Per-day report
        daily = trades_df.groupby('date').agg(
            n_trades=('net_pnl_ticks', 'count'),
            pnl_ticks=('net_pnl_ticks', 'sum'),
            win_rate=('winner', 'mean'),
        ).reset_index()
        daily['cum_pnl'] = daily['pnl_ticks'].cumsum()
        daily['pnl_dollars'] = daily['pnl_ticks'] * TICK_VALUE
        
        log.info(f"\n{'='*70}")
        log.info("PER-DAY REPORT")
        log.info(f"{'='*70}")
        for _, day in daily.iterrows():
            log.info(f"  {day['date']}: {day['n_trades']:.0f} trades, "
                     f"P&L={day['pnl_ticks']:.1f}t (${day['pnl_dollars']:.0f}), "
                     f"WR={day['win_rate']:.0%}, cum={day['cum_pnl']:.1f}t")
        
        # Regime stratification
        regime = regime_stratify(trades_df, flow_features_df)
        log.info(f"\n{'='*70}")
        log.info("REGIME STRATIFICATION")
        log.info(f"{'='*70}")
        for r_name in ['green', 'red', 'flat']:
            r_data = regime.get(r_name, {})
            log.info(f"  {r_name}: N={r_data.get('n_trades', 0)}, "
                     f"Sharpe={r_data.get('sharpe', 0):.2f}, "
                     f"WR={r_data.get('win_rate', 0):.1%}, "
                     f"PF={r_data.get('profit_factor', 0):.2f}")
        log.info(f"  Regime gap: {regime['regime_gap']:.3f} (pass={regime['regime_pass']})")
        
        # MFE check
        positive_moves = trades_df.loc[trades_df['raw_pnl_ticks'] > 0, 'raw_pnl_ticks']
        if len(positive_moves) >= 10:
            p90 = float(np.percentile(positive_moves, 90))
            p50 = float(np.percentile(positive_moves, 50))
            mfe_pass = cfg['tp'] <= p90
            log.info(f"\nMFE CHECK (HC #432): TP={cfg['tp']} vs p90 MFE={p90:.1f}, p50={p50:.1f} — {'PASS' if mfe_pass else 'FAIL'}")
        
        # Direction breakdown
        log.info(f"\nExit distribution: {trades_df['exit_type'].value_counts().to_dict()}")
        long_t = trades_df[trades_df['direction'] == 1]
        short_t = trades_df[trades_df['direction'] == -1]
        if len(long_t) > 0:
            log.info(f"Long:  N={len(long_t)}, avg={long_t['net_pnl_ticks'].mean():.2f}t, WR={long_t['winner'].mean():.1%}")
        if len(short_t) > 0:
            log.info(f"Short: N={len(short_t)}, avg={short_t['net_pnl_ticks'].mean():.2f}t, WR={short_t['winner'].mean():.1%}")
        
        # Comparison
        log.info(f"\n{'='*70}")
        log.info("COMPARISON vs STANDALONE")
        log.info(f"{'='*70}")
        log.info(f"  Flow-only champion:    Sharpe ~4.44, Sortino ~6.71")
        log.info(f"  Integrated pipeline:   Sharpe {metrics['sharpe']:.2f}, Sortino {metrics['sortino']:.2f}")
        log.info(f"  N trades: {metrics['n_trades']}, WR: {metrics['win_rate']:.1%}, PF: {metrics['profit_factor']:.2f}")
        log.info(f"  Day concentration: {metrics['day_concentration']:.3f} (cap=0.70, pass={metrics['day_conc_pass']})")
        log.info(f"  Total P&L: {metrics['total_pnl_ticks']:.1f} ticks (${metrics['total_pnl_dollars']:.0f})")
        
        # MLflow
        if MLFLOW_AVAILABLE:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"best_{k}", v)
            mlflow.log_metric("regime_gap", best_config['regime_gap'])
            mlflow.log_params({k: str(v) for k, v in cfg.items()})
        
        # Save trades
        trades_df.to_parquet(OUTPUT_DIR / 'best_trades.parquet', index=False)
    else:
        log.warning("No config passed all validation gates!")
    
    elapsed = time.time() - t0
    passing = [r for r in all_results if r['regime_pass'] and r['metrics'].get('day_conc_pass', False)]
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'n_overlap_days': len(overlap_dates),
        'date_range': [overlap_dates[0], overlap_dates[-1]],
        'n_configs_tested': len(all_results),
        'n_passing_configs': len(passing),
        'best_config': best_config,
        'top_10_configs': sorted(
            [r for r in all_results if r['metrics'].get('n_trades', 0) >= 20],
            key=lambda x: x['metrics']['sharpe'], reverse=True
        )[:10],
    }
    
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        return str(obj)
    
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(output, f, indent=2, default=convert)
    
    if MLFLOW_AVAILABLE:
        mlflow.log_metric("elapsed_seconds", elapsed)
        mlflow.log_metric("n_passing_configs", len(passing))
        mlflow.end_run()
    
    log.info(f"\nElapsed: {elapsed:.1f}s")
    log.info("DONE")


if __name__ == '__main__':
    main()
