#!/usr/bin/env python3
"""
integrated_pipeline_v3_combined.py — Combined SL/TP + MLP Early Exit

Combines the two best V2 approaches:
  - Approach B (SL15/TP30): Sharpe 9.64, best risk-adjusted, tiny MaxDD
  - Approach C (MLP cut): Sharpe 2.89, best PnL ($108K), best regime balance

Strategy: SL=15/TP=30 as structural stops, MLP classifier as early-exit override.
  1. Enter per model signal (460 trades from best_trades.parquet)
  2. Place SL=15 and TP=30 tick stops
  3. Run MLP mid-trade classifier continuously
  4. If MLP confidence drops below threshold → exit early (market order) BEFORE SL
  5. If MLP stays confident → let TP/SL play out normally
  6. Adaptive TP variant: high confidence → extend TP to 50

HC #432: exits bounded by model prediction horizon
HC #649: prefer dynamic (classifier) over static (fixed stops)
HC #74:  FIFO cost model only
HC #428: regime-agnostic validation, day-conc ≤ 0.70
"""

import os, sys, json, time, logging, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── Paths ──
LVL3 = Path("/home/nick/Lvl3Quant")
TRADES_FILE = LVL3 / "output/integrated_pipeline_v1/best_trades.parquet"
MINUTE_BARS_DIR = LVL3 / "data/processed/mbo_minute_bars_v1"
ENTRY_PREDS = LVL3 / "output/lh_30min_deep_v1/concat_oot.npz"
FLOW_FEATURES = LVL3 / "output/long_horizon_flow_v2/enhanced_daily_features.parquet"
MIDTRADE_FEATURES = LVL3 / "output/midtrade_thesis_v1/trade_tick_features.parquet"
OUTPUT_DIR = LVL3 / "output/integrated_pipeline_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants (canonical per CLAUDE.md) ──
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376
SPREAD_TICKS = 1.0

# Cost model per task spec:
COST_PASSIVE_ENTRY = COMMISSION_RT_TICKS   # 0.376 ticks (passive limit entry)
COST_MARKET_EXIT = COMMISSION_RT_TICKS + SPREAD_TICKS  # 1.376 ticks (market order exit: SL/classifier cut)
COST_PASSIVE_EXIT = COMMISSION_RT_TICKS    # 0.376 ticks (passive limit exit: TP hit)

MLFLOW_URI = "http://localhost:5000"

LOG_FILE = OUTPUT_DIR / "pipeline_v3.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# SHARED UTILITIES (from V2)
# ═══════════════════════════════════════════════════════════════

def get_unique_bar_count(date_str, pred_dates):
    """Figure out how many UNIQUE bars this date has in predictions."""
    mask = pred_dates == date_str
    n_total = mask.sum()
    if n_total <= 15:
        return n_total
    if n_total == 28: return 14
    elif n_total == 21: return 7
    elif n_total == 16: return 8
    elif n_total == 18: return 9
    else: return min(n_total, 15)


def build_30min_bars_and_minutes(date_str):
    """Build 30-min bars from minute data."""
    fpath = MINUTE_BARS_DIR / f"{date_str}.parquet"
    if not fpath.exists():
        return None, None

    mbars = pd.read_parquet(fpath)
    mbars['ts_minute'] = pd.to_datetime(mbars['ts_minute'], utc=True)
    mbars = mbars.sort_values('ts_minute').reset_index(drop=True)
    mbars['bar_key'] = mbars['ts_minute'].dt.floor('30min')

    bars_30 = []
    for bar_key, grp in mbars.groupby('bar_key'):
        if len(grp) < 3:
            continue
        bars_30.append({
            'bar_key': bar_key,
            'open': grp['open'].iloc[0],
            'high': grp['high'].max(),
            'low': grp['low'].min(),
            'close': grp['close'].iloc[-1],
            'volume': grp['volume'].sum(),
        })

    bars_30_df = pd.DataFrame(bars_30).sort_values('bar_key').reset_index(drop=True)
    return bars_30_df, mbars


def compute_metrics(trades_df, label=""):
    """Compute Sharpe, Sortino, PF, WR, max DD, day-concentration."""
    if len(trades_df) == 0:
        return {'label': label, 'n_trades': 0, 'sharpe': 0, 'sortino': 0,
                'win_rate': 0, 'profit_factor': 0, 'total_pnl_ticks': 0,
                'total_pnl_dollars': 0, 'max_dd_ticks': 0, 'max_dd_dollars': 0,
                'day_concentration': 1.0, 'day_conc_pass': False}

    daily_pnl = trades_df.groupby('date')['net_pnl_ticks'].sum()
    n_trades = len(trades_df)
    total_pnl = float(trades_df['net_pnl_ticks'].sum())

    winners = (trades_df['net_pnl_ticks'] > 0).sum()
    wr = float(winners / n_trades)

    gross_profit = float(trades_df.loc[trades_df['net_pnl_ticks'] > 0, 'net_pnl_ticks'].sum())
    gross_loss = float(abs(trades_df.loc[trades_df['net_pnl_ticks'] < 0, 'net_pnl_ticks'].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.9

    sharpe = float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if len(daily_pnl) > 1 and daily_pnl.std() > 0 else 0.0

    downside = daily_pnl[daily_pnl < 0]
    sortino = float(daily_pnl.mean() / downside.std() * np.sqrt(252)) if len(downside) > 1 and downside.std() > 0 else (99.9 if daily_pnl.mean() > 0 else 0.0)

    cumulative = daily_pnl.cumsum()
    max_dd = float((cumulative - cumulative.cummax()).min())

    day_conc = float(daily_pnl.max() / total_pnl) if total_pnl > 0 else 1.0

    return {
        'label': label,
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


def regime_stratify(trades_df):
    """Regime stratification with gap check (HC #428)."""
    flow_df = pd.read_parquet(FLOW_FEATURES)
    regime_map = {}
    for _, row in flow_df.iterrows():
        date_val = row['date']
        date_str = date_val.strftime('%Y%m%d') if hasattr(date_val, 'strftime') else str(date_val).replace('-', '')[:8]
        cc = row.get('cc_return_ticks', 0)
        if pd.isna(cc): regime_map[date_str] = 'flat'
        elif cc > 20: regime_map[date_str] = 'green'
        elif cc < -20: regime_map[date_str] = 'red'
        else: regime_map[date_str] = 'flat'

    trades_df = trades_df.copy()
    trades_df['regime'] = trades_df['date'].map(lambda d: regime_map.get(d, 'unknown'))

    results = {}
    for regime in ['green', 'red', 'flat']:
        rt = trades_df[trades_df['regime'] == regime]
        results[regime] = compute_metrics(rt, regime) if len(rt) > 0 else {'n_trades': 0, 'sharpe': 0}

    sg = results.get('green', {}).get('sharpe', 0)
    sr = results.get('red', {}).get('sharpe', 0)
    mx = max(abs(sg), abs(sr))
    gap = abs(sg - sr) / mx if mx > 0 else 0
    results['regime_gap'] = float(gap)
    results['regime_pass'] = bool(gap < 0.50)
    return results


def get_next_bar_minutes(date_str, actual_bar_idx, bar_cache):
    """Get minute bars for the next 30-min period after a trade entry."""
    if date_str not in bar_cache:
        bars_30, mbars = build_30min_bars_and_minutes(date_str)
        bar_cache[date_str] = (bars_30, mbars)
    bars_30, mbars = bar_cache[date_str]

    if bars_30 is None or actual_bar_idx >= len(bars_30):
        return None, None, None

    entry_price = bars_30.iloc[actual_bar_idx]['close']
    bar_key = bars_30.iloc[actual_bar_idx]['bar_key']

    next_start = bar_key + pd.Timedelta(minutes=30)
    next_end = bar_key + pd.Timedelta(minutes=59)
    mbars_ts = mbars.set_index('ts_minute')
    next_minutes = mbars_ts.loc[
        (mbars_ts.index >= next_start) & (mbars_ts.index < next_end + pd.Timedelta(minutes=1))
    ]

    return entry_price, next_minutes, bar_key


# ═══════════════════════════════════════════════════════════════
# MLP CLASSIFIER TRAINING (walk-forward)
# ═══════════════════════════════════════════════════════════════

def train_mlp_classifier(mt_df, feature_cols, mt_dates, current_date):
    """Train MLP on all midtrade data before current_date."""
    available_dates = [d for d in mt_dates if d < current_date]
    if len(available_dates) < 15:
        return None, None

    train_mask = mt_df['date'].isin(available_dates)
    X_train = np.nan_to_num(mt_df.loc[train_mask, feature_cols].values.astype(float), 0)
    y_train = mt_df.loc[train_mask, 'winner'].values

    if len(np.unique(y_train)) < 2:
        return None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        learning_rate_init=0.001,
        alpha=0.01,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
    )
    mlp.fit(X_scaled, y_train)
    return mlp, scaler


def get_mlp_confidence(mlp_model, scaler, mt_df, feature_cols, date_str, direction, bar_idx):
    """Get MLP win probability for a specific trade."""
    if mlp_model is None or scaler is None:
        return None

    mt_match = mt_df[
        (mt_df['date'] == date_str) &
        (mt_df['direction'] == direction)
    ]
    if len(mt_match) == 0:
        return None

    mt_row = mt_match.iloc[min(bar_idx, len(mt_match) - 1)]
    X_mt = np.nan_to_num(mt_row[feature_cols].values.reshape(1, -1).astype(float), 0)
    X_mt_scaled = scaler.transform(X_mt)

    try:
        proba = mlp_model.predict_proba(X_mt_scaled)[0]
        winner_idx = np.where(mlp_model.classes_ == 1)[0]
        return float(proba[winner_idx[0]]) if len(winner_idx) > 0 else float(proba[-1])
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# TRADE REPLAY ENGINE
# ═══════════════════════════════════════════════════════════════

def replay_combined(entry_price, direction, minute_bars_next, tp_ticks, sl_ticks,
                    mlp_confidence, mlp_threshold, early_exit_minute=4):
    """
    Replay a trade with combined SL/TP stops + MLP early exit.

    Logic:
    - Check SL/TP at each minute bar (structural stops)
    - If MLP confidence < threshold: exit early at specified minute (market order)
    - Otherwise let SL/TP play out, or exit at bar close

    Returns dict with exit details.
    """
    if len(minute_bars_next) == 0:
        return None

    mfe = 0.0
    mae = 0.0

    # Determine if MLP triggers early exit
    mlp_cuts = (mlp_confidence is not None and mlp_confidence < mlp_threshold)

    for i, (idx, row) in enumerate(minute_bars_next.iterrows()):
        hi = row['high']
        lo = row['low']

        if direction == 1:
            fav = (hi - entry_price) / TICK_SIZE
            adv = (entry_price - lo) / TICK_SIZE
            tp_hit = (hi - entry_price) >= tp_ticks * TICK_SIZE
            sl_hit = (entry_price - lo) >= sl_ticks * TICK_SIZE
        else:
            fav = (entry_price - lo) / TICK_SIZE
            adv = (hi - entry_price) / TICK_SIZE
            tp_hit = (entry_price - lo) >= tp_ticks * TICK_SIZE
            sl_hit = (hi - entry_price) >= sl_ticks * TICK_SIZE

        mfe = max(mfe, fav)
        mae = max(mae, adv)

        # MLP early exit: check BEFORE SL so we can potentially save ticks
        # Exit at ~5 min in (minute index 4) if MLP says cut
        if mlp_cuts and i == early_exit_minute:
            exit_price = row['close']
            if direction == 1:
                exit_pnl = (exit_price - entry_price) / TICK_SIZE
            else:
                exit_pnl = (entry_price - exit_price) / TICK_SIZE
            return {
                'exit_type': 'mlp_early_cut',
                'exit_pnl_ticks': exit_pnl,
                'minutes_held': i + 1,
                'mfe_ticks': mfe,
                'mae_ticks': mae,
            }

        # Check structural stops
        if tp_hit and sl_hit:
            remaining_tp = max(tp_ticks - mfe, 0.1)
            remaining_sl = max(sl_ticks - mae, 0.1)
            p_sl_first = remaining_tp / (remaining_tp + remaining_sl)
            if p_sl_first > 0.5:
                return {
                    'exit_type': 'sl',
                    'exit_pnl_ticks': -sl_ticks,
                    'minutes_held': i + 1,
                    'mfe_ticks': mfe,
                    'mae_ticks': mae,
                }
            else:
                return {
                    'exit_type': 'tp',
                    'exit_pnl_ticks': tp_ticks,
                    'minutes_held': i + 1,
                    'mfe_ticks': mfe,
                    'mae_ticks': mae,
                }
        elif sl_hit:
            # If MLP would have cut before this minute, the cut already happened above
            return {
                'exit_type': 'sl',
                'exit_pnl_ticks': -sl_ticks,
                'minutes_held': i + 1,
                'mfe_ticks': mfe,
                'mae_ticks': mae,
            }
        elif tp_hit:
            return {
                'exit_type': 'tp',
                'exit_pnl_ticks': tp_ticks,
                'minutes_held': i + 1,
                'mfe_ticks': mfe,
                'mae_ticks': mae,
            }

    # Neither hit — exit at bar close (passive)
    exit_price = minute_bars_next.iloc[-1]['close']
    if direction == 1:
        exit_pnl = (exit_price - entry_price) / TICK_SIZE
    else:
        exit_pnl = (entry_price - exit_price) / TICK_SIZE

    return {
        'exit_type': 'bar_close',
        'exit_pnl_ticks': exit_pnl,
        'minutes_held': len(minute_bars_next),
        'mfe_ticks': mfe,
        'mae_ticks': mae,
    }


def replay_adaptive_tp(entry_price, direction, minute_bars_next, sl_ticks,
                       mlp_confidence, early_exit_minute=4):
    """
    Adaptive TP based on MLP confidence:
    - confidence > 0.7: TP=50 (let winners run)
    - confidence 0.5-0.7: TP=30 (standard)
    - confidence < 0.5: cut immediately (market order at early_exit_minute)

    SL=15 always active as backstop.
    """
    if len(minute_bars_next) == 0:
        return None

    # Determine adaptive TP
    if mlp_confidence is not None:
        if mlp_confidence > 0.7:
            tp_ticks = 50
            tp_label = 'tp50_high_conf'
        elif mlp_confidence >= 0.5:
            tp_ticks = 30
            tp_label = 'tp30_mod_conf'
        else:
            tp_ticks = None  # Will cut early
            tp_label = 'cut_low_conf'
    else:
        tp_ticks = 30  # Default if no MLP
        tp_label = 'tp30_no_mlp'

    mfe = 0.0
    mae = 0.0

    for i, (idx, row) in enumerate(minute_bars_next.iterrows()):
        hi = row['high']
        lo = row['low']

        if direction == 1:
            fav = (hi - entry_price) / TICK_SIZE
            adv = (entry_price - lo) / TICK_SIZE
            sl_hit = (entry_price - lo) >= sl_ticks * TICK_SIZE
            tp_hit = (tp_ticks is not None) and ((hi - entry_price) >= tp_ticks * TICK_SIZE)
        else:
            fav = (entry_price - lo) / TICK_SIZE
            adv = (hi - entry_price) / TICK_SIZE
            sl_hit = (hi - entry_price) >= sl_ticks * TICK_SIZE
            tp_hit = (tp_ticks is not None) and ((entry_price - lo) >= tp_ticks * TICK_SIZE)

        mfe = max(mfe, fav)
        mae = max(mae, adv)

        # Low confidence: cut early
        if tp_ticks is None and i == early_exit_minute:
            exit_price = row['close']
            if direction == 1:
                exit_pnl = (exit_price - entry_price) / TICK_SIZE
            else:
                exit_pnl = (entry_price - exit_price) / TICK_SIZE
            return {
                'exit_type': 'adaptive_cut',
                'exit_pnl_ticks': exit_pnl,
                'minutes_held': i + 1,
                'mfe_ticks': mfe,
                'mae_ticks': mae,
                'tp_label': tp_label,
            }

        # Structural stops
        if tp_hit and sl_hit:
            remaining_tp = max(tp_ticks - mfe, 0.1)
            remaining_sl = max(sl_ticks - mae, 0.1)
            p_sl_first = remaining_tp / (remaining_tp + remaining_sl)
            if p_sl_first > 0.5:
                return {
                    'exit_type': 'sl',
                    'exit_pnl_ticks': -sl_ticks,
                    'minutes_held': i + 1,
                    'mfe_ticks': mfe,
                    'mae_ticks': mae,
                    'tp_label': tp_label,
                }
            else:
                return {
                    'exit_type': 'tp',
                    'exit_pnl_ticks': tp_ticks,
                    'minutes_held': i + 1,
                    'mfe_ticks': mfe,
                    'mae_ticks': mae,
                    'tp_label': tp_label,
                }
        elif sl_hit:
            return {
                'exit_type': 'sl',
                'exit_pnl_ticks': -sl_ticks,
                'minutes_held': i + 1,
                'mfe_ticks': mfe,
                'mae_ticks': mae,
                'tp_label': tp_label,
            }
        elif tp_hit:
            return {
                'exit_type': 'tp',
                'exit_pnl_ticks': tp_ticks,
                'minutes_held': i + 1,
                'mfe_ticks': mfe,
                'mae_ticks': mae,
                'tp_label': tp_label,
            }

    # Neither hit — bar close
    exit_price = minute_bars_next.iloc[-1]['close']
    if direction == 1:
        exit_pnl = (exit_price - entry_price) / TICK_SIZE
    else:
        exit_pnl = (entry_price - exit_price) / TICK_SIZE

    return {
        'exit_type': 'bar_close',
        'exit_pnl_ticks': exit_pnl,
        'minutes_held': len(minute_bars_next),
        'mfe_ticks': mfe,
        'mae_ticks': mae,
        'tp_label': tp_label,
    }


# ═══════════════════════════════════════════════════════════════
# VARIANT 1: FIXED SL/TP + MLP EARLY CUT
# ═══════════════════════════════════════════════════════════════

def run_fixed_tp_mlp_cut(trades_orig, pred_dates, bar_cache,
                         mt_df, feature_cols, mt_dates, mlp_threshold):
    """
    SL=15 / TP=30 structural stops + MLP early-exit override.
    If MLP confidence < threshold → exit at ~5min (market order), saving some SL losses.
    """
    label = f"Combined_SL15_TP30_MLP{mlp_threshold}"
    log.info(f"\n{'='*70}")
    log.info(f"VARIANT: {label}")
    log.info(f"{'='*70}")

    unique_bar_counts = {}
    mlp_model = None
    scaler = None
    last_train_date = None
    results = []

    for _, trade in trades_orig.iterrows():
        date_str = trade['date']
        bar_idx = trade['bar_idx']
        direction = trade['direction']

        if date_str not in unique_bar_counts:
            unique_bar_counts[date_str] = get_unique_bar_count(date_str, pred_dates)
        actual_bar_idx = bar_idx % unique_bar_counts[date_str]

        # Retrain MLP periodically (walk-forward)
        available_mt_dates = [d for d in mt_dates if d < date_str]
        if len(available_mt_dates) >= 15 and (mlp_model is None or
            (last_train_date is not None and
             len([d for d in mt_dates if last_train_date < d < date_str]) >= 5)):
            mlp_model, scaler = train_mlp_classifier(mt_df, feature_cols, mt_dates, date_str)
            last_train_date = date_str

        # Get minute bars
        entry_price, next_minutes, bar_key = get_next_bar_minutes(
            date_str, actual_bar_idx, bar_cache)

        if entry_price is None or next_minutes is None or len(next_minutes) == 0:
            continue

        # Get MLP confidence
        mlp_conf = get_mlp_confidence(
            mlp_model, scaler, mt_df, feature_cols, date_str, direction, bar_idx)

        # Replay with combined exits
        r = replay_combined(entry_price, direction, next_minutes,
                           tp_ticks=30, sl_ticks=15,
                           mlp_confidence=mlp_conf,
                           mlp_threshold=mlp_threshold)
        if r is None:
            continue

        # Cost model
        if r['exit_type'] in ('sl', 'mlp_early_cut'):
            # Passive entry + market exit
            cost = COST_PASSIVE_ENTRY + COST_MARKET_EXIT
            # But that double-counts commission. Canonical:
            # Entry passive: 0.376 ticks commission (no spread, we're on the book)
            # Exit market: 0.376 ticks commission + 1.0 tick spread
            # But COMMISSION_RT_TICKS = 0.376 is ROUND TRIP. So each leg = 0.188.
            # Actually per V2 convention: COST_MARKET_EXIT = 1.376 = full RT cost
            cost = COST_MARKET_EXIT  # 1.376 ticks
        elif r['exit_type'] == 'tp':
            # Passive entry + passive exit (TP is a resting limit)
            cost = COST_PASSIVE_EXIT  # 0.376 ticks
        else:  # bar_close
            cost = COST_PASSIVE_EXIT  # 0.376 ticks (passive both sides, but RT = 0.376)

        net_pnl = r['exit_pnl_ticks'] - cost

        # Also compute what standalone B would have done (for comparison)
        r_standalone = replay_combined(entry_price, direction, next_minutes,
                                       tp_ticks=30, sl_ticks=15,
                                       mlp_confidence=None,
                                       mlp_threshold=1.0)  # Never cuts
        standalone_pnl = r_standalone['exit_pnl_ticks'] if r_standalone else r['exit_pnl_ticks']
        standalone_exit = r_standalone['exit_type'] if r_standalone else r['exit_type']

        results.append({
            'date': date_str,
            'direction': direction,
            'bar_idx': bar_idx,
            'entry_price': entry_price,
            'mlp_confidence': mlp_conf,
            'exit_type': r['exit_type'],
            'raw_pnl_ticks': r['exit_pnl_ticks'],
            'cost_ticks': cost,
            'net_pnl_ticks': net_pnl,
            'mfe_ticks': r['mfe_ticks'],
            'mae_ticks': r['mae_ticks'],
            'minutes_held': r['minutes_held'],
            'standalone_b_exit': standalone_exit,
            'standalone_b_pnl': standalone_pnl,
        })

    df = pd.DataFrame(results)
    metrics = compute_metrics(df, label)
    regime = regime_stratify(df)

    # Log results
    log.info(f"N trades: {metrics['n_trades']}")
    log.info(f"Sharpe: {metrics['sharpe']:.2f}, Sortino: {metrics['sortino']:.2f}")
    log.info(f"WR: {metrics['win_rate']:.1%}, PF: {metrics['profit_factor']:.2f}")
    log.info(f"Total PnL: {metrics['total_pnl_ticks']:.1f} ticks (${metrics['total_pnl_dollars']:.0f})")
    log.info(f"Max DD: {metrics['max_dd_ticks']:.1f} ticks (${metrics['max_dd_dollars']:.0f})")
    log.info(f"Day concentration: {metrics['day_concentration']:.3f} (pass={metrics['day_conc_pass']})")
    log.info(f"Regime gap: {regime['regime_gap']:.3f} (pass={regime['regime_pass']})")

    # Exit distribution
    exit_dist = df['exit_type'].value_counts().to_dict()
    log.info(f"Exit distribution: {exit_dist}")

    # MLP value-add analysis
    cut_trades = df[df['exit_type'] == 'mlp_early_cut']
    if len(cut_trades) > 0:
        # Compare: what MLP-cut trades got vs what standalone B would have given
        mlp_avg = cut_trades['net_pnl_ticks'].mean()
        standalone_avg = cut_trades['standalone_b_pnl'].mean()
        saved_per_trade = mlp_avg - standalone_avg
        n_would_hit_sl = (cut_trades['standalone_b_exit'] == 'sl').sum()
        log.info(f"\nMLP Value-Add Analysis:")
        log.info(f"  MLP cut {len(cut_trades)} trades early")
        log.info(f"  Avg PnL on cut trades: {mlp_avg:.2f}t (vs standalone B: {standalone_avg:.2f}t)")
        log.info(f"  Savings per cut trade: {saved_per_trade:+.2f}t")
        log.info(f"  Of {len(cut_trades)} cuts, {n_would_hit_sl} would have hit SL in standalone B")

    for r_name in ['green', 'red', 'flat']:
        rd = regime.get(r_name, {})
        log.info(f"  {r_name}: N={rd.get('n_trades', 0)}, Sharpe={rd.get('sharpe', 0):.2f}, "
                 f"WR={rd.get('win_rate', 0):.1%}, PF={rd.get('profit_factor', 0):.2f}")

    return df, metrics, regime


# ═══════════════════════════════════════════════════════════════
# VARIANT 2: ADAPTIVE TP BASED ON MLP CONFIDENCE
# ═══════════════════════════════════════════════════════════════

def run_adaptive_tp(trades_orig, pred_dates, bar_cache,
                    mt_df, feature_cols, mt_dates):
    """
    SL=15 always. TP adapts to MLP confidence:
    - conf > 0.7: TP=50 (let winners run)
    - conf 0.5-0.7: TP=30 (standard)
    - conf < 0.5: cut immediately at ~5min
    """
    label = "Adaptive_TP_SL15"
    log.info(f"\n{'='*70}")
    log.info(f"VARIANT: {label}")
    log.info(f"{'='*70}")

    unique_bar_counts = {}
    mlp_model = None
    scaler = None
    last_train_date = None
    results = []

    for _, trade in trades_orig.iterrows():
        date_str = trade['date']
        bar_idx = trade['bar_idx']
        direction = trade['direction']

        if date_str not in unique_bar_counts:
            unique_bar_counts[date_str] = get_unique_bar_count(date_str, pred_dates)
        actual_bar_idx = bar_idx % unique_bar_counts[date_str]

        # Retrain MLP periodically
        available_mt_dates = [d for d in mt_dates if d < date_str]
        if len(available_mt_dates) >= 15 and (mlp_model is None or
            (last_train_date is not None and
             len([d for d in mt_dates if last_train_date < d < date_str]) >= 5)):
            mlp_model, scaler = train_mlp_classifier(mt_df, feature_cols, mt_dates, date_str)
            last_train_date = date_str

        entry_price, next_minutes, bar_key = get_next_bar_minutes(
            date_str, actual_bar_idx, bar_cache)

        if entry_price is None or next_minutes is None or len(next_minutes) == 0:
            continue

        mlp_conf = get_mlp_confidence(
            mlp_model, scaler, mt_df, feature_cols, date_str, direction, bar_idx)

        r = replay_adaptive_tp(entry_price, direction, next_minutes,
                               sl_ticks=15, mlp_confidence=mlp_conf)
        if r is None:
            continue

        # Cost model
        if r['exit_type'] in ('sl', 'adaptive_cut'):
            cost = COST_MARKET_EXIT  # 1.376 ticks
        elif r['exit_type'] == 'tp':
            cost = COST_PASSIVE_EXIT  # 0.376 ticks
        else:  # bar_close
            cost = COST_PASSIVE_EXIT  # 0.376 ticks

        net_pnl = r['exit_pnl_ticks'] - cost

        results.append({
            'date': date_str,
            'direction': direction,
            'bar_idx': bar_idx,
            'entry_price': entry_price,
            'mlp_confidence': mlp_conf,
            'exit_type': r['exit_type'],
            'tp_label': r.get('tp_label', ''),
            'raw_pnl_ticks': r['exit_pnl_ticks'],
            'cost_ticks': cost,
            'net_pnl_ticks': net_pnl,
            'mfe_ticks': r['mfe_ticks'],
            'mae_ticks': r['mae_ticks'],
            'minutes_held': r['minutes_held'],
        })

    df = pd.DataFrame(results)
    metrics = compute_metrics(df, label)
    regime = regime_stratify(df)

    log.info(f"N trades: {metrics['n_trades']}")
    log.info(f"Sharpe: {metrics['sharpe']:.2f}, Sortino: {metrics['sortino']:.2f}")
    log.info(f"WR: {metrics['win_rate']:.1%}, PF: {metrics['profit_factor']:.2f}")
    log.info(f"Total PnL: {metrics['total_pnl_ticks']:.1f} ticks (${metrics['total_pnl_dollars']:.0f})")
    log.info(f"Max DD: {metrics['max_dd_ticks']:.1f} ticks (${metrics['max_dd_dollars']:.0f})")
    log.info(f"Day concentration: {metrics['day_concentration']:.3f} (pass={metrics['day_conc_pass']})")
    log.info(f"Regime gap: {regime['regime_gap']:.3f} (pass={regime['regime_pass']})")

    exit_dist = df['exit_type'].value_counts().to_dict()
    log.info(f"Exit distribution: {exit_dist}")

    if 'tp_label' in df.columns:
        tp_dist = df['tp_label'].value_counts().to_dict()
        log.info(f"TP tier distribution: {tp_dist}")

    # Analyze by confidence tier
    for tier_name, lo, hi in [('High (>0.7)', 0.7, 1.01), ('Moderate (0.5-0.7)', 0.5, 0.7), ('Low (<0.5)', 0, 0.5)]:
        tier = df[(df['mlp_confidence'] >= lo) & (df['mlp_confidence'] < hi)] if df['mlp_confidence'].notna().any() else pd.DataFrame()
        if len(tier) > 0:
            tier_m = compute_metrics(tier, tier_name)
            log.info(f"  {tier_name}: N={tier_m['n_trades']}, Sharpe={tier_m['sharpe']:.2f}, "
                     f"WR={tier_m['win_rate']:.1%}, PnL={tier_m['total_pnl_ticks']:.1f}t")

    for r_name in ['green', 'red', 'flat']:
        rd = regime.get(r_name, {})
        log.info(f"  {r_name}: N={rd.get('n_trades', 0)}, Sharpe={rd.get('sharpe', 0):.2f}, "
                 f"WR={rd.get('win_rate', 0):.1%}, PF={rd.get('profit_factor', 0):.2f}")

    return df, metrics, regime


# ═══════════════════════════════════════════════════════════════
# STANDALONE B BASELINE (for comparison)
# ═══════════════════════════════════════════════════════════════

def run_standalone_b(trades_orig, pred_dates, bar_cache):
    """Standalone SL15/TP30 without any MLP — the baseline to beat."""
    label = "Standalone_B_SL15_TP30"
    log.info(f"\n{'='*70}")
    log.info(f"BASELINE: {label}")
    log.info(f"{'='*70}")

    unique_bar_counts = {}
    results = []

    for _, trade in trades_orig.iterrows():
        date_str = trade['date']
        bar_idx = trade['bar_idx']
        direction = trade['direction']

        if date_str not in unique_bar_counts:
            unique_bar_counts[date_str] = get_unique_bar_count(date_str, pred_dates)
        actual_bar_idx = bar_idx % unique_bar_counts[date_str]

        entry_price, next_minutes, bar_key = get_next_bar_minutes(
            date_str, actual_bar_idx, bar_cache)

        if entry_price is None or next_minutes is None or len(next_minutes) == 0:
            continue

        r = replay_combined(entry_price, direction, next_minutes,
                           tp_ticks=30, sl_ticks=15,
                           mlp_confidence=None, mlp_threshold=1.0)
        if r is None:
            continue

        if r['exit_type'] == 'sl':
            cost = COST_MARKET_EXIT
        elif r['exit_type'] == 'tp':
            cost = COST_PASSIVE_EXIT
        else:
            cost = COST_PASSIVE_EXIT

        net_pnl = r['exit_pnl_ticks'] - cost

        results.append({
            'date': date_str,
            'direction': direction,
            'bar_idx': bar_idx,
            'entry_price': entry_price,
            'exit_type': r['exit_type'],
            'raw_pnl_ticks': r['exit_pnl_ticks'],
            'cost_ticks': cost,
            'net_pnl_ticks': net_pnl,
            'mfe_ticks': r['mfe_ticks'],
            'mae_ticks': r['mae_ticks'],
            'minutes_held': r['minutes_held'],
        })

    df = pd.DataFrame(results)
    metrics = compute_metrics(df, label)
    regime = regime_stratify(df)

    log.info(f"N trades: {metrics['n_trades']}")
    log.info(f"Sharpe: {metrics['sharpe']:.2f}, Sortino: {metrics['sortino']:.2f}")
    log.info(f"WR: {metrics['win_rate']:.1%}, PF: {metrics['profit_factor']:.2f}")
    log.info(f"Total PnL: {metrics['total_pnl_ticks']:.1f} ticks (${metrics['total_pnl_dollars']:.0f})")
    log.info(f"Max DD: {metrics['max_dd_ticks']:.1f} ticks (${metrics['max_dd_dollars']:.0f})")
    log.info(f"Day concentration: {metrics['day_concentration']:.3f} (pass={metrics['day_conc_pass']})")
    log.info(f"Regime gap: {regime['regime_gap']:.3f} (pass={regime['regime_pass']})")

    exit_dist = df['exit_type'].value_counts().to_dict()
    log.info(f"Exit distribution: {exit_dist}")

    for r_name in ['green', 'red', 'flat']:
        rd = regime.get(r_name, {})
        log.info(f"  {r_name}: N={rd.get('n_trades', 0)}, Sharpe={rd.get('sharpe', 0):.2f}, "
                 f"WR={rd.get('win_rate', 0):.1%}, PF={rd.get('profit_factor', 0):.2f}")

    return df, metrics, regime


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("INTEGRATED PIPELINE V3 — COMBINED SL/TP + MLP EARLY EXIT")
    log.info("=" * 70)
    log.info("Combining best of V2: SL15/TP30 structure + MLP classifier override")
    log.info("")

    # MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("integrated_pipeline_v3_combined")
        mlflow.start_run(run_name=f"combined_exit_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Load data
    trades_orig = pd.read_parquet(TRADES_FILE)
    pred_data = np.load(ENTRY_PREDS, allow_pickle=True)
    pred_dates = pred_data['dates']

    log.info(f"Loaded {len(trades_orig)} trades from integrated_pipeline_v1")
    log.info(f"Date range: {trades_orig['date'].min()} to {trades_orig['date'].max()}")

    # Load midtrade features for MLP
    mt_df = pd.read_parquet(MIDTRADE_FEATURES)
    mt_dates = sorted(mt_df['date'].unique())
    exclude = ['date', 'direction', 'fill_price', 'time_of_day_min', 'exit_type',
               'exit_ticks', 'winner', 'pred_magnitude']
    feature_cols = [c for c in mt_df.columns if c not in exclude]
    log.info(f"Midtrade features: {len(mt_df)} trades, {len(feature_cols)} features, {len(mt_dates)} dates")

    bar_cache = {}

    # ── Baseline: Standalone B ──
    b_df, b_metrics, b_regime = run_standalone_b(trades_orig, pred_dates, bar_cache)

    # ── Variant 1: Fixed TP + MLP cut (3 thresholds) ──
    combined_results = {}
    for thresh in [0.3, 0.4, 0.5]:
        c_df, c_metrics, c_regime = run_fixed_tp_mlp_cut(
            trades_orig, pred_dates, bar_cache,
            mt_df, feature_cols, mt_dates, thresh)
        combined_results[f"Combined_MLP{thresh}"] = {
            'trades_df': c_df, 'metrics': c_metrics, 'regime': c_regime}

    # ── Variant 2: Adaptive TP ──
    a_df, a_metrics, a_regime = run_adaptive_tp(
        trades_orig, pred_dates, bar_cache,
        mt_df, feature_cols, mt_dates)

    # ═══════════════════════════════════════════════════════════
    # COMPARATIVE SUMMARY
    # ═══════════════════════════════════════════════════════════
    log.info(f"\n{'='*70}")
    log.info("COMPARATIVE SUMMARY — V3 COMBINED vs STANDALONE BASELINES")
    log.info(f"{'='*70}")

    all_approaches = {}
    all_approaches['Standalone_B (SL15/TP30)'] = {'metrics': b_metrics, 'regime': b_regime}
    for k, v in combined_results.items():
        all_approaches[k] = {'metrics': v['metrics'], 'regime': v['regime']}
    all_approaches['Adaptive_TP'] = {'metrics': a_metrics, 'regime': a_regime}

    header = (f"{'Approach':<30} {'N':>5} {'Sharpe':>8} {'Sortino':>8} "
              f"{'WR':>7} {'PF':>7} {'PnL_t':>8} {'PnL$':>8} {'MaxDD_t':>8} "
              f"{'DayConc':>8} {'RegGap':>7} {'Pass':>5}")
    log.info(header)
    log.info("-" * 120)

    for name, data in all_approaches.items():
        m = data['metrics']
        r = data['regime']
        regime_pass = r.get('regime_pass', False)
        day_conc_pass = m.get('day_conc_pass', False)
        overall_pass = regime_pass and day_conc_pass and m['sharpe'] > 0

        log.info(f"{name:<30} {m['n_trades']:>5} {m['sharpe']:>8.2f} {m['sortino']:>8.2f} "
                 f"{m['win_rate']:>6.1%} {m['profit_factor']:>7.2f} {m['total_pnl_ticks']:>8.1f} "
                 f"{m['total_pnl_dollars']:>7.0f} {m['max_dd_ticks']:>8.1f} "
                 f"{m['day_concentration']:>8.3f} "
                 f"{r.get('regime_gap', 0):>7.3f} {'YES' if overall_pass else 'NO':>5}")

    # ── Delta analysis vs standalone B ──
    log.info(f"\n{'='*70}")
    log.info("DELTA vs STANDALONE B BASELINE")
    log.info(f"{'='*70}")
    b_sharpe = b_metrics['sharpe']
    b_pnl = b_metrics['total_pnl_ticks']

    for name, data in all_approaches.items():
        if name == 'Standalone_B (SL15/TP30)':
            continue
        m = data['metrics']
        d_sharpe = m['sharpe'] - b_sharpe
        d_pnl = m['total_pnl_ticks'] - b_pnl
        log.info(f"  {name}: Sharpe {d_sharpe:+.2f}, PnL {d_pnl:+.1f}t (${d_pnl * TICK_VALUE:+.0f})")

    # ── Verdict ──
    log.info(f"\n{'='*70}")
    log.info("VERDICT")
    log.info(f"{'='*70}")

    best_name = None
    best_sharpe = -999
    for name, data in all_approaches.items():
        m = data['metrics']
        r = data['regime']
        if (r.get('regime_pass', False) and m.get('day_conc_pass', False)
            and m['sharpe'] > best_sharpe and m['n_trades'] >= 20):
            best_sharpe = m['sharpe']
            best_name = name

    if best_name:
        bm = all_approaches[best_name]['metrics']
        br = all_approaches[best_name]['regime']
        log.info(f"BEST: {best_name}")
        log.info(f"  Sharpe={bm['sharpe']:.2f}, Sortino={bm['sortino']:.2f}, "
                 f"WR={bm['win_rate']:.1%}, PF={bm['profit_factor']:.2f}")
        log.info(f"  PnL={bm['total_pnl_ticks']:.1f}t (${bm['total_pnl_dollars']:.0f}), "
                 f"MaxDD={bm['max_dd_ticks']:.1f}t")
        log.info(f"  Regime gap={br['regime_gap']:.3f}, Day conc={bm['day_concentration']:.3f}")

        if best_name != 'Standalone_B (SL15/TP30)':
            log.info(f"\n  IMPROVEMENT over standalone B:")
            log.info(f"    Sharpe: {bm['sharpe'] - b_sharpe:+.2f}")
            log.info(f"    PnL: {bm['total_pnl_ticks'] - b_pnl:+.1f}t "
                     f"(${(bm['total_pnl_ticks'] - b_pnl) * TICK_VALUE:+.0f})")
        else:
            log.info(f"\n  Standalone B remains the best — MLP overlay did not improve risk-adjusted returns.")
    else:
        log.info("No approach passes all validation gates.")

    # ── MLflow logging ──
    if MLFLOW_AVAILABLE:
        for name, data in all_approaches.items():
            prefix = name.replace(' ', '_').replace(':', '').replace('/', '_').replace('(', '').replace(')', '').lower()
            for k, v in data['metrics'].items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{prefix}_{k}", v)
            mlflow.log_metric(f"{prefix}_regime_gap", data['regime'].get('regime_gap', 0))

        if best_name:
            mlflow.log_param("best_approach", best_name)
            mlflow.log_metric("best_sharpe", best_sharpe)

        mlflow.log_param("sl_ticks", 15)
        mlflow.log_param("tp_ticks_base", 30)
        mlflow.log_param("tp_ticks_extended", 50)
        mlflow.log_param("mlp_thresholds", "0.3,0.4,0.5")
        mlflow.log_param("cost_passive", COST_PASSIVE_ENTRY)
        mlflow.log_param("cost_market_exit", COST_MARKET_EXIT)

        mlflow.end_run()

    # ── Save results ──
    summary = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': time.time() - t0,
        'n_trades_input': len(trades_orig),
        'approaches': {},
    }

    for name, data in all_approaches.items():
        summary['approaches'][name] = {
            'metrics': {k: v for k, v in data['metrics'].items() if k != 'label'},
            'regime_gap': data['regime'].get('regime_gap', 0),
            'regime_pass': data['regime'].get('regime_pass', False),
            'regime_details': {r: {k: v for k, v in data['regime'].get(r, {}).items() if k != 'label'}
                              for r in ['green', 'red', 'flat']},
        }

    if best_name:
        summary['best_approach'] = best_name
        summary['best_sharpe'] = best_sharpe

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        return str(obj)

    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=convert)

    # Save trade-level results
    b_df.to_parquet(OUTPUT_DIR / 'trades_standalone_b.parquet', index=False)
    for k, v in combined_results.items():
        v['trades_df'].to_parquet(OUTPUT_DIR / f'trades_{k.lower()}.parquet', index=False)
    a_df.to_parquet(OUTPUT_DIR / 'trades_adaptive_tp.parquet', index=False)

    elapsed = time.time() - t0
    log.info(f"\nElapsed: {elapsed:.1f}s")
    log.info(f"Results saved to {OUTPUT_DIR}")
    log.info("DONE")


if __name__ == '__main__':
    main()
