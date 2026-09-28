"""
OOS Holdout Test: ask_orders Signal Predictions
================================================

Tests the 100-day ask_orders predictions through a Python-based market
order simulator (Rust MBO sim blocked on Windows by Smart App Control).

Signal source: data/processed/signal_predictions/ask_orders_*.npz
Mid prices:    data/processed/mbo_features_cache/*_mbo_features.npz (column 0)

Costs:
  $3.00 RT commission + $12.50 half-spread each side = $15.50 total
  (= 1 tick commission + 1 tick spread = 1.24 ticks total)

Hold periods tested: 10s, 30s, 1min, 5min
Signal thresholds: rolling causal quantiles (0.80, 0.90, 0.95)
Execution: market orders (most conservative / realistic)

Usage:
    python alpha_discovery/run_ask_orders_oos_holdout.py
    python alpha_discovery/run_ask_orders_oos_holdout.py --threshold 0.90 --hold-sec 30
"""

import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / 'alpha_discovery' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR = ROOT / 'data' / 'processed' / 'signal_predictions'
FEAT_DIR = ROOT / 'data' / 'processed' / 'mbo_features_cache'

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(RESULTS_DIR / f'ask_orders_oos_{_ts}.log'), mode='w'),
    ]
)
log = logging.getLogger('ask_orders_oos')

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE       = 0.25     # ES min increment
TICK_VALUE      = 12.50    # $/tick
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)     # AMP + Rithmic + CME round-trip
SPREAD_COST     = 12.50    # Half-spread each side = 1 tick = $12.50
TOTAL_COST_PER_RT = COMMISSION_RT + SPREAD_COST  # $15.50
TOTAL_COST_TICKS  = TOTAL_COST_PER_RT / TICK_VALUE   # 1.24 ticks
BARS_PER_SEC    = 10       # 100ms bars


# ============================================================================
# DATA LOADING
# ============================================================================
def load_day(date_str: str):
    """Load predictions and mid prices for a single date. Returns (mid, preds) or None."""
    pred_file = PRED_DIR / f'ask_orders_{date_str}.npz'
    feat_file = FEAT_DIR / f'{date_str}_mbo_features.npz'

    if not pred_file.exists() or not feat_file.exists():
        return None

    preds = np.load(pred_file)['predictions'].astype(np.float64)
    feats = np.load(feat_file)['mbo_features']
    mid   = feats[:, 0].astype(np.float64)  # column 0 = 'mid'

    if len(preds) != len(mid):
        log.warning(f"  {date_str}: preds={len(preds)} != mid={len(mid)}, trimming")
        n = min(len(preds), len(mid))
        preds, mid = preds[:n], mid[:n]

    return mid, preds


def get_all_dates():
    """Return sorted list of dates that have BOTH prediction and feature files."""
    pred_dates = set()
    for f in PRED_DIR.glob('ask_orders_*.npz'):
        date_str = f.stem.replace('ask_orders_', '')
        pred_dates.add(date_str)

    feat_dates = set()
    for f in FEAT_DIR.glob('*_mbo_features.npz'):
        date_str = f.stem.replace('_mbo_features', '')
        feat_dates.add(date_str)

    available = sorted(pred_dates & feat_dates)
    return available


# ============================================================================
# IC ANALYSIS
# ============================================================================
def compute_ic_per_day(dates, hold_periods_bars):
    """
    Compute Spearman IC between ask_orders predictions and forward returns
    at each hold period, per day.

    Returns:
        dict: {hold_name: {'daily_ics': [...], 'mean_ic': float, 'consistency': float, ...}}
    """
    results = {hp: {'daily_ics': [], 'dates': []} for hp in hold_periods_bars}

    for date_str in dates:
        data = load_day(date_str)
        if data is None:
            continue
        mid, preds = data
        N = len(mid)

        for hp_bars, hp_name in hold_periods_bars.items():
            # Forward return in ticks
            if N <= hp_bars:
                continue
            fwd_ret = np.zeros(N, dtype=np.float64)
            fwd_ret[:N - hp_bars] = (mid[hp_bars:] - mid[:N - hp_bars]) / TICK_SIZE

            # Only use non-zero predictions and valid forward returns
            valid = (preds != 0) & np.isfinite(preds) & np.isfinite(fwd_ret)
            valid[:100] = False  # skip warmup
            valid[N - hp_bars:] = False  # skip tail

            if valid.sum() < 200:
                continue

            ic, _ = spearmanr(preds[valid], fwd_ret[valid])
            results[hp_bars]['daily_ics'].append(float(ic))
            results[hp_bars]['dates'].append(date_str)

    for hp_bars, hp_name in hold_periods_bars.items():
        ics = np.array(results[hp_bars]['daily_ics'])
        if len(ics) == 0:
            results[hp_bars].update({'mean_ic': 0, 'ic_std': 0, 'icir': 0,
                                     'consistency': 0, 'n_days': 0})
            continue
        mean_ic = float(np.mean(ics))
        ic_std  = float(np.std(ics))
        icir    = mean_ic / ic_std if ic_std > 0 else 0
        consistency = float((ics > 0).mean())
        results[hp_bars].update({
            'mean_ic': mean_ic,
            'ic_std': ic_std,
            'icir': icir,
            'consistency': consistency,
            'n_days': len(ics),
        })

    return results


# ============================================================================
# MARKET ORDER SIMULATION (conservative baseline)
# ============================================================================
def simulate_market_orders(
    dates,
    hold_bars: int,
    signal_quantile: float = 0.90,
    min_trade_spacing_bars: int = 50,
    stop_loss_ticks: float = 8.0,
    label: str = '',
) -> dict:
    """
    Market order simulation — most conservative and realistic execution.

    Entry: signal exceeds causal rolling quantile threshold
    Execution: market order (pay half spread = 0.5 ticks each side)
    Exit: time-based exit after hold_bars (or stop loss)
    Total cost: $15.50 per RT (1.24 ticks: 1.0 tick spread + 0.24 tick commission)

    Rolling causal thresholds: computed from PREVIOUS days only (no look-ahead).
    """
    # ---- Pass 1: Collect ALL predictions across all days in order ----
    all_preds = []
    day_data = []
    for date_str in dates:
        data = load_day(date_str)
        if data is None:
            continue
        mid, preds = data
        day_data.append((date_str, mid, preds))
        valid_mask = (preds != 0) & np.isfinite(preds)
        all_preds.extend(preds[valid_mask].tolist())

    if not day_data:
        return {'n_trades': 0, 'error': 'No data'}

    trades = []
    daily_pnls = []

    # Rolling threshold state (expanding, causal)
    pred_history = []

    for day_idx, (date_str, mid, preds) in enumerate(day_data):
        N = len(mid)
        day_trades = []

        # Compute threshold from ALL PRIOR days (strictly causal)
        if len(pred_history) >= 200:
            hist = np.array(pred_history)
            thresh_pos = float(np.percentile(hist, signal_quantile * 100))
            thresh_neg = float(np.percentile(hist, (1 - signal_quantile) * 100))
        else:
            # Not enough history: skip this day but collect data
            valid_day = preds[(preds != 0) & np.isfinite(preds)]
            pred_history.extend(valid_day.tolist())
            daily_pnls.append({'date': date_str, 'pnl': 0.0, 'n_trades': 0,
                               'reason': 'insufficient_history'})
            continue

        max_start = N - hold_bars - 5
        last_exit = -1

        i = 100  # warmup
        while i < max_start:
            sig = preds[i]
            if sig == 0 or not np.isfinite(sig) or not np.isfinite(mid[i]):
                i += 1
                continue

            if i <= last_exit + min_trade_spacing_bars:
                i += 1
                continue

            # Determine direction
            if sig > thresh_pos:
                direction = 1
            elif sig < thresh_neg:
                direction = -1
            else:
                i += 1
                continue

            entry_bar = i + 1  # 1 bar latency (100ms)
            if entry_bar >= max_start or not np.isfinite(mid[entry_bar]):
                i += 1
                continue

            entry_mid = mid[entry_bar]

            # --- Find exit ---
            exit_bar = min(entry_bar + hold_bars, N - 1)
            exit_type = 'timeout'

            for j in range(entry_bar + 1, exit_bar + 1):
                if not np.isfinite(mid[j]):
                    continue
                unrealized_ticks = (mid[j] - entry_mid) / TICK_SIZE * direction
                if stop_loss_ticks and unrealized_ticks <= -stop_loss_ticks:
                    exit_bar = j
                    exit_type = 'stop_loss'
                    break

            if not np.isfinite(mid[exit_bar]):
                i += 1
                continue

            # --- PnL calculation ---
            # Market entry: pay half spread (0.5 ticks)
            # Market exit: pay half spread (0.5 ticks)
            # Commission: 0.24 ticks
            # Total deduction: 1.24 ticks
            raw_pnl_ticks = (mid[exit_bar] - entry_mid) / TICK_SIZE * direction
            net_pnl_ticks = raw_pnl_ticks - TOTAL_COST_TICKS
            net_pnl_dollars = net_pnl_ticks * TICK_VALUE

            trade = {
                'date': date_str,
                'day': day_idx,
                'direction': direction,
                'entry_bar': entry_bar,
                'exit_bar': exit_bar,
                'signal': float(sig),
                'entry_mid': float(entry_mid),
                'exit_mid': float(mid[exit_bar]),
                'raw_pnl_ticks': float(raw_pnl_ticks),
                'net_pnl_ticks': float(net_pnl_ticks),
                'net_pnl_dollars': float(net_pnl_dollars),
                'exit_type': exit_type,
                'hold_bars_actual': exit_bar - entry_bar,
            }
            trades.append(trade)
            day_trades.append(net_pnl_dollars)
            last_exit = exit_bar
            i = exit_bar + 1  # move past trade

        day_pnl = sum(day_trades)
        daily_pnls.append({
            'date': date_str,
            'pnl': day_pnl,
            'n_trades': len(day_trades),
        })

        # Collect this day's predictions for FUTURE threshold computation
        valid_day = preds[(preds != 0) & np.isfinite(preds)]
        pred_history.extend(valid_day.tolist())

    # ---- Aggregate stats ----
    if not trades:
        return {'n_trades': 0, 'error': 'No trades generated', 'label': label}

    pnl_arr = np.array([t['net_pnl_dollars'] for t in trades])
    raw_ticks = np.array([t['raw_pnl_ticks'] for t in trades])

    active_daily = [d for d in daily_pnls if d.get('n_trades', 0) > 0]
    daily_pnl_arr = np.array([d['pnl'] for d in active_daily]) if active_daily else np.array([0.0])

    # Sharpe (annualized from daily)
    if len(daily_pnl_arr) > 1 and np.std(daily_pnl_arr) > 0:
        daily_sharpe = float(np.mean(daily_pnl_arr) / np.std(daily_pnl_arr) * np.sqrt(252))
    else:
        daily_sharpe = 0.0

    # Max drawdown
    cum_pnl = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cum_pnl)
    max_dd = float((cum_pnl - peak).min())

    n_pos_days = int(sum(1 for d in active_daily if d['pnl'] > 0))
    n_neg_days = int(sum(1 for d in active_daily if d['pnl'] < 0))

    exit_breakdown = {}
    for t in trades:
        et = t['exit_type']
        if et not in exit_breakdown:
            exit_breakdown[et] = {'count': 0, 'total_pnl': 0}
        exit_breakdown[et]['count'] += 1
        exit_breakdown[et]['total_pnl'] += t['net_pnl_dollars']

    return {
        'label': label,
        'n_trades': len(trades),
        'n_active_days': len(active_daily),
        'trades_per_day': float(len(trades) / max(len(active_daily), 1)),
        'mean_raw_pnl_ticks': float(raw_ticks.mean()),
        'mean_net_pnl_ticks': float(pnl_arr.mean() / TICK_VALUE),
        'mean_net_pnl_dollars': float(pnl_arr.mean()),
        'total_pnl_dollars': float(pnl_arr.sum()),
        'daily_pnl_mean': float(daily_pnl_arr.mean()),
        'daily_pnl_std': float(daily_pnl_arr.std()),
        'daily_sharpe': daily_sharpe,
        'win_rate': float((pnl_arr > 0).mean()),
        'n_positive_days': n_pos_days,
        'n_negative_days': n_neg_days,
        'pct_win_days': float(n_pos_days / max(n_pos_days + n_neg_days, 1)),
        'max_drawdown': max_dd,
        'profit_factor': float(pnl_arr[pnl_arr > 0].sum() / max(abs(pnl_arr[pnl_arr < 0].sum()), 0.01)),
        'exit_breakdown': {
            k: {'count': v['count'], 'avg_pnl': v['total_pnl'] / v['count']}
            for k, v in exit_breakdown.items()
        },
        'daily_pnls': daily_pnls,
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='ask_orders OOS holdout test')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Signal quantile threshold (default: sweep 0.80, 0.90, 0.95)')
    parser.add_argument('--hold-sec', type=float, default=None,
                        help='Hold period in seconds (default: sweep 10, 30, 60, 300)')
    parser.add_argument('--stop-loss', type=float, default=8.0,
                        help='Stop loss in ticks (default: 8 = 2 points)')
    args = parser.parse_args()

    log.info('=' * 70)
    log.info('ask_orders OOS HOLDOUT TEST — Market Orders, Causal Thresholds')
    log.info('=' * 70)
    log.info(f'Costs: ${TOTAL_COST_PER_RT:.2f} RT = {TOTAL_COST_TICKS:.2f} ticks '
             f'(${COMMISSION_RT} comm + ${SPREAD_COST} spread)')

    t0 = time.time()

    # Get all dates
    all_dates = get_all_dates()
    log.info(f'Found {len(all_dates)} days with matching prediction + feature files')
    log.info(f'Date range: {all_dates[0]} to {all_dates[-1]}')

    # --- IC Analysis ---
    log.info('\n--- IC ANALYSIS ---')
    hold_periods = {
        100:   '10s',
        300:   '30s',
        600:   '1min',
        3000:  '5min',
    }
    ic_results = compute_ic_per_day(all_dates, hold_periods)

    log.info(f'\n{"Hold":>8} {"Mean IC":>10} {"IC Std":>8} {"ICIR":>7} {"Consistency":>13} {"N Days":>8}')
    log.info('-' * 60)
    for hp_bars in sorted(hold_periods.keys()):
        hp_name = hold_periods[hp_bars]
        r = ic_results[hp_bars]
        log.info(f'{hp_name:>8} {r["mean_ic"]:>10.4f} {r["ic_std"]:>8.4f} '
                 f'{r["icir"]:>7.2f} {r["consistency"]:>12.1%} {r["n_days"]:>8}')

    # --- Market Order Simulation ---
    log.info('\n--- MARKET ORDER SIMULATION ---')

    thresholds = [args.threshold] if args.threshold else [0.90, 0.95]
    hold_secs  = [args.hold_sec]  if args.hold_sec  else [10.0, 30.0, 60.0, 300.0]

    all_sim_results = []

    for q in thresholds:
        for hs in hold_secs:
            hold_bars = int(hs * BARS_PER_SEC)
            label = f'q{int(q*100)}_hold{int(hs)}s'
            log.info(f'\n  Config: threshold={q:.0%}, hold={hs:.0f}s, stop={args.stop_loss}t')

            res = simulate_market_orders(
                all_dates,
                hold_bars=hold_bars,
                signal_quantile=q,
                stop_loss_ticks=args.stop_loss,
                label=label,
            )
            all_sim_results.append(res)

            if res.get('n_trades', 0) == 0:
                log.info(f'    No trades — {res.get("error", "unknown")}')
                continue

            log.info(f'    Trades: {res["n_trades"]} over {res["n_active_days"]} days '
                     f'({res["trades_per_day"]:.1f}/day)')
            log.info(f'    Raw dir PnL: {res["mean_raw_pnl_ticks"]:+.3f} ticks/trade')
            log.info(f'    Net PnL:     {res["mean_net_pnl_ticks"]:+.3f} ticks/trade = '
                     f'${res["mean_net_pnl_dollars"]:+.2f}/trade')
            log.info(f'    Total PnL:   ${res["total_pnl_dollars"]:+,.0f}')
            log.info(f'    Daily mean:  ${res["daily_pnl_mean"]:+.0f}/day  '
                     f'std=${res["daily_pnl_std"]:.0f}')
            log.info(f'    Sharpe:      {res["daily_sharpe"]:+.2f}  '
                     f'WinRate={res["win_rate"]:.1%}')
            log.info(f'    Days +/-:    {res["n_positive_days"]}/{res["n_negative_days"]}  '
                     f'({res["pct_win_days"]:.1%} win)')
            log.info(f'    Max DD:      ${res["max_drawdown"]:,.0f}')
            log.info(f'    PF:          {res["profit_factor"]:.2f}')
            exits = res.get('exit_breakdown', {})
            for et, ev in exits.items():
                log.info(f'    Exit [{et}]: {ev["count"]} trades, avg ${ev["avg_pnl"]:+.2f}')

    # --- Summary Table ---
    log.info('\n' + '=' * 90)
    log.info('SUMMARY TABLE — ask_orders OOS HOLDOUT')
    log.info('=' * 90)
    log.info(f'{"Config":>20} {"Trades":>8} {"Days":>6} {"Dir(t)":>8} '
             f'{"Net(t)":>8} {"$/day":>9} {"Sharpe":>8} {"WR":>7} {"W/L days":>10}')
    log.info('-' * 90)

    for res in all_sim_results:
        if res.get('n_trades', 0) == 0:
            log.info(f'{res.get("label",""):>20} {"NO TRADES":>8}')
            continue
        log.info(
            f'{res["label"]:>20} {res["n_trades"]:>8} {res["n_active_days"]:>6} '
            f'{res["mean_raw_pnl_ticks"]:>+8.3f} {res["mean_net_pnl_ticks"]:>+8.3f} '
            f'{res["daily_pnl_mean"]:>+9.0f} {res["daily_sharpe"]:>+8.2f} '
            f'{res["win_rate"]:>7.1%} '
            f'{res["n_positive_days"]}/{res["n_negative_days"]} days'
        )

    # --- IC Summary ---
    log.info('\nIC SUMMARY:')
    for hp_bars in sorted(hold_periods.keys()):
        hp_name = hold_periods[hp_bars]
        r = ic_results[hp_bars]
        log.info(f'  {hp_name}: IC={r["mean_ic"]:+.4f}, ICIR={r["icir"]:.2f}, '
                 f'consistency={r["consistency"]:.1%}')

    # --- Save results ---
    output = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_sec': round(time.time() - t0, 1),
        'n_days': len(all_dates),
        'date_range': [all_dates[0], all_dates[-1]],
        'costs': {
            'commission_rt': COMMISSION_RT,
            'spread_per_side_ticks': 0.5,
            'total_cost_ticks': TOTAL_COST_TICKS,
            'total_cost_dollars': TOTAL_COST_PER_RT,
        },
        'ic_analysis': {
            hold_periods[k]: {
                'mean_ic': v['mean_ic'],
                'ic_std': v.get('ic_std', 0),
                'icir': v.get('icir', 0),
                'consistency': v.get('consistency', 0),
                'n_days': v.get('n_days', 0),
                'daily_ics': v.get('daily_ics', []),
            }
            for k, v in ic_results.items()
        },
        'sim_results': all_sim_results,
    }

    out_file = RESULTS_DIR / f'ask_orders_oos_holdout_{_ts}.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    log.info(f'\nResults saved: {out_file}')
    log.info(f'Elapsed: {time.time() - t0:.1f}s')

    return output


if __name__ == '__main__':
    main()
