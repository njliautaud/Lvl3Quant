"""
Limit Order Execution Study -- ES Futures MBO Alpha

Investigates whether the confirmed directional signal (IC=0.08-0.11) becomes
profitable using limit orders instead of market orders.

Parts:
  1. Spread statistics (distribution, time-of-day, vol regime)
  2. Fill probability (mid touching limit price within N bars)
  3. Adverse selection per fill (price path after fill)
  4. Combined strategy simulation (direction + limit entry/exit)
  5. Signal-conditioned fill study (does confidence reduce adverse selection?)

Usage:
    python alpha_discovery/run_limit_order_study.py
    python alpha_discovery/run_limit_order_study.py --fast
"""
import sys, gc, json, time, logging, argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR
from alpha_discovery.run_return_multihorizon import (
    load_feature_cache, compute_return_targets, EXCLUDE_FEATURES_DIRECTION,
)
from alpha_discovery.run_model_refinement import walk_forward_evaluate

# ============================================================================
# LOGGING (ASCII-only for Windows cp1252)
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(RESULTS_DIR / 'limit_order_study.log'), mode='a', encoding='utf-8'),
    ]
)
log = logging.getLogger("limit_order_study")

TICK_SIZE    = 0.25
TICK_VALUE   = 12.50
BARS_PER_SEC = 10      # 100ms bars

_last_log = [0.0]
def progress(msg, interval=30.0):
    if time.time() - _last_log[0] >= interval:
        log.info(msg); _last_log[0] = time.time()


# ============================================================================
# PART 1: SPREAD STATISTICS
# ============================================================================
def analyze_spread_stats(spread, hour_of_day, mid_prices, day_boundaries):
    log.info("Part 1: Spread Statistics")
    ticks = spread / TICK_SIZE
    ok = np.isfinite(ticks) & (ticks > 0)
    s = ticks[ok]

    dist = {
        'pct_1tick':  float((s <= 1.01).mean()),
        'pct_2ticks': float(((s > 1.01) & (s <= 2.01)).mean()),
        'pct_3plus':  float((s > 2.01).mean()),
        'mean_ticks': float(np.mean(s)),
        'median_ticks': float(np.median(s)),
        'p95_ticks':  float(np.percentile(s, 95)),
        'n_valid':    int(ok.sum()),
    }
    log.info(f"  1-tick={dist['pct_1tick']:.1%}  2-tick={dist['pct_2ticks']:.1%}  "
             f"3+={dist['pct_3plus']:.1%}  mean={dist['mean_ticks']:.3f}t")

    # Time-of-day
    tod = {}
    for h in range(9, 17):
        m = (hour_of_day >= h) & (hour_of_day < h + 1) & ok
        if m.sum() > 50:
            tod[str(h)] = {'mean_ticks': float(ticks[m].mean()),
                           'pct_1tick': float((ticks[m] <= 1.01).mean()),
                           'n': int(m.sum())}

    # Vol regime via 50-bar realized vol
    lr = np.diff(np.log(np.maximum(mid_prices, 1.0)), prepend=0.0)
    cs2 = np.cumsum((lr ** 2).astype(np.float64))
    cs2p = np.concatenate([[0.0], cs2])  # length N+1
    rvol = np.full(len(lr), np.nan)
    W = 50
    N_lr = len(lr)
    if N_lr > W:
        # rvol[W:] has shape (N_lr - W,)
        # cs2p[W:N_lr] - cs2p[0:N_lr-W] both have shape (N_lr - W,)
        rvol[W:] = np.sqrt((cs2p[W:N_lr] - cs2p[0:N_lr - W]) / W)
    p33, p67 = np.nanpercentile(rvol, 33), np.nanpercentile(rvol, 67)
    regimes = {'low_vol': rvol <= p33, 'mid_vol': (rvol > p33) & (rvol <= p67), 'high_vol': rvol > p67}
    vol_regime = {}
    for name, mask in regimes.items():
        m2 = mask & ok
        if m2.sum() > 50:
            vol_regime[name] = {'mean_ticks': float(ticks[m2].mean()),
                                'pct_1tick': float((ticks[m2] <= 1.01).mean()),
                                'n': int(m2.sum())}

    edge = dist['mean_ticks'] * 0.5 * TICK_VALUE
    log.info(f"  Mean edge per passive fill: ${edge:.2f} (= 0.5 * mean_spread)")
    return {'distribution': dist, 'time_of_day': tod, 'vol_regime': vol_regime,
            'mean_edge_per_fill_dollars': edge}


# ============================================================================
# PART 2: FILL PROBABILITY
# ============================================================================
def analyze_fill_probability(mid_prices, best_bid, best_ask, day_boundaries,
                              hour_of_day, horizons_bars=None, offsets_ticks=None):
    log.info("Part 2: Fill Probability Analysis")
    if horizons_bars is None:
        horizons_bars = [10, 30, 50, 100, 300]   # 1s, 3s, 5s, 10s, 30s
    if offsets_ticks is None:
        offsets_ticks = [0, -1, -2]              # at best, 1 behind, 2 behind

    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    results = {}

    for offset in offsets_ticks:
        buy_lim = best_bid + offset * TICK_SIZE   # passive buy (behind best_bid)
        sell_lim = best_ask - offset * TICK_SIZE  # passive sell

        results[offset] = {}
        for hz in horizons_bars:
            # Build forward-looking min/max of mid within each day
            fwd_min = np.full(N, np.inf)
            fwd_max = np.full(N, -np.inf)
            valid_mask = np.zeros(N, dtype=bool)

            for d in range(n_days):
                ds, de = day_boundaries[d], day_boundaries[d + 1]
                if de - ds < 2:
                    continue
                m = mid_prices[ds:de]
                L = len(m)
                valid_mask[ds:de - 1] = True

                # Sliding window min/max of size hz over m, shifted by 1 bar forward
                if L > hz:
                    # Use stride_tricks for vectorized sliding window
                    windows = np.lib.stride_tricks.sliding_window_view(m, hz)
                    # windows[j] = m[j:j+hz], so fwd[i] = min(m[i+1:i+1+hz]) = windows[i+1].min()
                    w_min = windows.min(axis=1)   # shape (L-hz+1,)
                    w_max = windows.max(axis=1)
                    n_assign = min(L - hz, len(w_min) - 1)
                    if n_assign > 0:
                        fwd_min[ds:ds + n_assign] = w_min[1:n_assign + 1]
                        fwd_max[ds:ds + n_assign] = w_max[1:n_assign + 1]
                # Tail bars (truncated window to end of day)
                tail = max(0, L - hz)
                for i in range(tail, L - 1):
                    fwd_min[ds + i] = m[i + 1:].min()
                    fwd_max[ds + i] = m[i + 1:].max()

            vb = valid_mask & np.isfinite(buy_lim) & np.isfinite(fwd_min) & (buy_lim > 0)
            vs = valid_mask & np.isfinite(sell_lim) & np.isfinite(fwd_max) & (sell_lim > 0)
            fr_buy  = float((fwd_min[vb] <= buy_lim[vb]).mean()) if vb.sum() > 0 else 0.0
            fr_sell = float((fwd_max[vs] >= sell_lim[vs]).mean()) if vs.sum() > 0 else 0.0
            hz_sec = hz / BARS_PER_SEC
            results[offset][hz_sec] = {
                'hz_sec': hz_sec, 'offset_ticks': offset,
                'fill_rate_buy': fr_buy, 'fill_rate_sell': fr_sell,
                'fill_rate_avg': (fr_buy + fr_sell) / 2,
            }
            log.info(f"  offset={offset:+d}t hz={hz_sec:.0f}s  buy={fr_buy:.1%}  sell={fr_sell:.1%}")

    return {'fill_probs': results}


# ============================================================================
# PART 3: ADVERSE SELECTION
# ============================================================================
def analyze_adverse_selection(mid_prices, best_bid, best_ask, day_boundaries,
                               predictions, pred_indices,
                               fill_horizon_bars=50, post_fill_horizons=None):
    log.info("Part 3: Adverse Selection Analysis")
    if post_fill_horizons is None:
        post_fill_horizons = [10, 30, 50, 100, 300]  # 1s,3s,5s,10s,30s

    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    pred_signal = np.full(N, np.nan)
    if len(predictions) > 0 and len(pred_indices) > 0:
        valid_idx = (pred_indices >= 0) & (pred_indices < N)
        pred_signal[pred_indices[valid_idx]] = predictions[valid_idx]

    buy_fills, sell_fills = [], []

    for d in range(n_days):
        ds, de = day_boundaries[d], day_boundaries[d + 1]
        if de - ds < fill_horizon_bars + max(post_fill_horizons) + 5:
            continue
        m, bb, ba, ps = mid_prices[ds:de], best_bid[ds:de], best_ask[ds:de], pred_signal[ds:de]
        L = len(m)

        for i in range(L - fill_horizon_bars - max(post_fill_horizons) - 2):
            if not (np.isfinite(bb[i]) and np.isfinite(ba[i])):
                continue
            sig = ps[i]
            future = m[i + 1:i + fill_horizon_bars + 1]

            # Buy fill: mid dips to best_bid
            bt = np.where(future <= bb[i])[0]
            if len(bt) > 0:
                fbar = i + bt[0] + 1
                if fbar + max(post_fill_horizons) < L:
                    adv = [float((m[fbar + k] - m[fbar]) / TICK_SIZE) for k in post_fill_horizons]
                    buy_fills.append({'sig': float(sig) if np.isfinite(sig) else None, 'adv': adv})

            # Sell fill: mid rises to best_ask
            st = np.where(future >= ba[i])[0]
            if len(st) > 0:
                fbar = i + st[0] + 1
                if fbar + max(post_fill_horizons) < L:
                    adv = [float(-(m[fbar + k] - m[fbar]) / TICK_SIZE) for k in post_fill_horizons]
                    sell_fills.append({'sig': float(sig) if np.isfinite(sig) else None, 'adv': adv})

        progress(f"  Adv-sel: day {d+1}/{n_days} buy={len(buy_fills)} sell={len(sell_fills)}")

    log.info(f"  Total fills: buy={len(buy_fills)} sell={len(sell_fills)}")

    def summarize(fill_list, side):
        if not fill_list:
            return {'n': 0}
        n_hz = len(post_fill_horizons)
        has_sig = [f for f in fill_list if f['sig'] is not None]
        med_sig = np.median([f['sig'] for f in has_sig]) if has_sig else 0.0
        aligned = [f for f in has_sig if (side == 'buy' and f['sig'] > med_sig) or
                   (side == 'sell' and f['sig'] < med_sig)]
        opposed = [f for f in has_sig if f not in aligned and f in has_sig]

        out = {'n': len(fill_list), 'n_with_signal': len(has_sig)}
        for k_idx, k in enumerate(post_fill_horizons):
            k_sec = k / BARS_PER_SEC
            vals = np.array([f['adv'][k_idx] for f in fill_list])
            out[f'k_{k_sec:.0f}s'] = {
                'mean_adv_ticks': float(np.nanmean(vals)),
                'pct_adverse': float((vals < 0).mean()),
                'n': len(vals),
                'aligned_adv_ticks': float(np.nanmean([f['adv'][k_idx] for f in aligned])) if aligned else None,
                'opposed_adv_ticks': float(np.nanmean([f['adv'][k_idx] for f in opposed])) if opposed else None,
            }
        return out

    buy_sum  = summarize(buy_fills, 'buy')
    sell_sum = summarize(sell_fills, 'sell')

    log.info("  Adverse selection (mean ticks after fill, positive=good):")
    for k in post_fill_horizons:
        k_sec = k / BARS_PER_SEC
        lbl = f'k_{k_sec:.0f}s'
        bs, ss = buy_sum.get(lbl, {}), sell_sum.get(lbl, {})
        log.info(f"    {k_sec:>4.0f}s: buy={bs.get('mean_adv_ticks', float('nan')):>+.3f}t "
                 f"({bs.get('pct_adverse', 0):.1%} adverse)  "
                 f"sell={ss.get('mean_adv_ticks', float('nan')):>+.3f}t "
                 f"({ss.get('pct_adverse', 0):.1%} adverse)")

    return {
        'buy': buy_sum, 'sell': sell_sum,
        'post_fill_horizons_sec': [k / BARS_PER_SEC for k in post_fill_horizons],
        'fill_horizon_sec': fill_horizon_bars / BARS_PER_SEC,
    }


# ============================================================================
# PART 4: COMBINED STRATEGY SIMULATION
# ============================================================================
def simulate_limit_strategy(mid_prices, best_bid, best_ask, spread, day_boundaries,
                             predictions, pred_indices,
                             fill_horizon_bars=50, hold_horizon_bars=300,
                             min_signal_quantile=0.6, use_limit_exit=False,
                             label=''):
    mode = 'limit_exit' if use_limit_exit else 'market_exit'
    log.info(f"Part 4: Strategy Simulation [{label or mode}] "
             f"fill={fill_horizon_bars/BARS_PER_SEC:.0f}s "
             f"hold={hold_horizon_bars/BARS_PER_SEC:.0f}s "
             f"q={min_signal_quantile:.0%}")

    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    pred_signal = np.full(N, np.nan)
    if len(predictions) > 0 and len(pred_indices) > 0:
        ok = (pred_indices >= 0) & (pred_indices < N)
        pred_signal[pred_indices[ok]] = predictions[ok]

    valid_p = predictions[np.isfinite(predictions)]
    if len(valid_p) == 0:
        return {'error': 'No valid predictions'}
    thresh_pos = float(np.percentile(valid_p, min_signal_quantile * 100))
    thresh_neg = float(np.percentile(valid_p, (1 - min_signal_quantile) * 100))

    trades = []
    n_posted = 0

    for d in range(n_days):
        ds, de = day_boundaries[d], day_boundaries[d + 1]
        if de - ds < fill_horizon_bars + hold_horizon_bars + 5:
            continue
        m, bb, ba, sp, ps = (mid_prices[ds:de], best_bid[ds:de], best_ask[ds:de],
                               spread[ds:de], pred_signal[ds:de])
        L = len(m)
        last_exit = -1

        for i in range(L - fill_horizon_bars - hold_horizon_bars - 2):
            if i <= last_exit:
                continue
            sig, s_i = ps[i], sp[i]
            if not np.isfinite(sig) or not (np.isfinite(bb[i]) and np.isfinite(ba[i])):
                continue
            if sig > thresh_pos:
                direction = 1
            elif sig < thresh_neg:
                direction = -1
            else:
                continue

            n_posted += 1
            entry_lim = bb[i] if direction == 1 else ba[i]
            future = m[i + 1:min(i + fill_horizon_bars + 1, L)]
            touch = np.where(future <= entry_lim)[0] if direction == 1 else np.where(future >= entry_lim)[0]
            if len(touch) == 0:
                continue

            fill_bar = i + touch[0] + 1
            exit_bar = min(fill_bar + hold_horizon_bars, L - 1)

            # Early exit on signal reversal
            for j in range(fill_bar + 1, exit_bar):
                p_j = ps[j]
                if np.isfinite(p_j):
                    if direction == 1 and p_j < thresh_neg: exit_bar = j; break
                    elif direction == -1 and p_j > thresh_pos: exit_bar = j; break

            sp_entry = sp[fill_bar] if np.isfinite(sp[fill_bar]) else s_i
            sp_exit  = sp[exit_bar] if np.isfinite(sp[exit_bar]) else sp_entry

            # PnL components (all in ticks)
            entry_edge    = (0.5 * sp_entry / TICK_SIZE) * direction   # passive fill earns half-spread vs mid
            dir_pnl_ticks = (m[exit_bar] - m[fill_bar]) / TICK_SIZE * direction
            exit_cost     = 0.0 if use_limit_exit else 0.5 * sp_exit / TICK_SIZE
            # limit exit earns no extra edge vs market exit in terms of spread,
            # but avoids crossing. Approximate: limit exit saves the exit_cost.
            exit_edge     = (0.5 * sp_exit / TICK_SIZE) if use_limit_exit else 0.0
            net_ticks     = entry_edge + dir_pnl_ticks - exit_cost + exit_edge

            trades.append({
                'direction': direction, 'signal': float(sig),
                'fill_offset': touch[0] + 1, 'bars_held': exit_bar - fill_bar,
                'entry_edge_ticks': float(entry_edge),
                'dir_pnl_ticks': float(dir_pnl_ticks),
                'exit_cost_ticks': float(exit_cost),
                'exit_edge_ticks': float(exit_edge),
                'net_ticks': float(net_ticks),
                'net_dollars': float(net_ticks * TICK_VALUE),
                'spread_entry_ticks': float(sp_entry / TICK_SIZE),
            })
            last_exit = exit_bar

        progress(f"  Sim: day {d+1}/{n_days} trades={len(trades)}")

    if not trades:
        return {'error': 'No trades generated', 'n_posted': n_posted}

    pnl = np.array([t['net_ticks'] for t in trades])
    dir_pnl = np.array([t['dir_pnl_ticks'] for t in trades])
    entry_edge = np.array([t['entry_edge_ticks'] for t in trades])
    exit_cost = np.array([t['exit_cost_ticks'] for t in trades])
    exit_edge = np.array([t['exit_edge_ticks'] for t in trades])
    sharpe = (float(np.mean(pnl) / np.std(pnl) * np.sqrt(252 * 6.5 * 3600 / (hold_horizon_bars / BARS_PER_SEC)))
              if np.std(pnl) > 0 else 0.0)

    result = {
        'n_posted': n_posted, 'n_filled': len(trades),
        'fill_rate': len(trades) / n_posted if n_posted > 0 else 0.0,
        'use_limit_exit': use_limit_exit,
        'mean_pnl_ticks': float(np.mean(pnl)),
        'std_pnl_ticks': float(np.std(pnl)),
        'mean_pnl_dollars': float(np.mean(pnl) * TICK_VALUE),
        'total_pnl_dollars': float(np.sum(pnl) * TICK_VALUE),
        'win_rate': float((pnl > 0).mean()),
        'sharpe_approx': sharpe,
        'mean_entry_edge_ticks': float(np.mean(entry_edge)),
        'mean_dir_pnl_ticks': float(np.mean(dir_pnl)),
        'mean_exit_cost_ticks': float(np.mean(exit_cost)),
        'mean_exit_edge_ticks': float(np.mean(exit_edge)),
        'mean_fill_bars': float(np.mean([t['fill_offset'] for t in trades])),
        'min_signal_quantile': min_signal_quantile,
        'fill_horizon_sec': fill_horizon_bars / BARS_PER_SEC,
        'hold_horizon_sec': hold_horizon_bars / BARS_PER_SEC,
    }
    log.info(f"  fill={result['fill_rate']:.1%}  mean_pnl={result['mean_pnl_ticks']:+.4f}t "
             f"(entry={result['mean_entry_edge_ticks']:+.4f}t "
             f"dir={result['mean_dir_pnl_ticks']:+.4f}t "
             f"exit_cost={result['mean_exit_cost_ticks']:+.4f}t "
             f"exit_edge={result['mean_exit_edge_ticks']:+.4f}t)  "
             f"win={result['win_rate']:.1%}  sharpe={sharpe:.2f}  "
             f"total=${result['total_pnl_dollars']:+.2f}")
    return result


# ============================================================================
# PART 5: SIGNAL-CONDITIONED FILL STUDY
# ============================================================================
def analyze_signal_conditioned(mid_prices, best_bid, best_ask, day_boundaries,
                                predictions, pred_indices,
                                fill_horizon_bars=50, post_fill_bars=100,
                                n_quintiles=5):
    log.info("Part 5: Signal-Conditioned Fill Study")
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    pred_signal = np.full(N, np.nan)
    if len(predictions) > 0 and len(pred_indices) > 0:
        ok = (pred_indices >= 0) & (pred_indices < N)
        pred_signal[pred_indices[ok]] = predictions[ok]

    valid_p = predictions[np.isfinite(predictions)]
    qedges = np.percentile(np.abs(valid_p), np.linspace(0, 100, n_quintiles + 1))
    qdata = {q: {'fills': [], 'total': 0} for q in range(n_quintiles)}

    for d in range(n_days):
        ds, de = day_boundaries[d], day_boundaries[d + 1]
        if de - ds < fill_horizon_bars + post_fill_bars + 5:
            continue
        m, bb, ba, ps = mid_prices[ds:de], best_bid[ds:de], best_ask[ds:de], pred_signal[ds:de]
        L = len(m)

        for i in range(L - fill_horizon_bars - post_fill_bars - 2):
            sig = ps[i]
            if not np.isfinite(sig) or not (np.isfinite(bb[i]) and np.isfinite(ba[i])):
                continue
            direction = int(np.sign(sig))
            if direction == 0:
                continue
            q = min(int(np.searchsorted(qedges[1:], abs(sig), side='right')), n_quintiles - 1)
            qdata[q]['total'] += 1

            lim = bb[i] if direction == 1 else ba[i]
            future = m[i + 1:min(i + fill_horizon_bars + 1, L)]
            touch = np.where(future <= lim)[0] if direction == 1 else np.where(future >= lim)[0]
            if len(touch) == 0:
                continue
            fbar = i + touch[0] + 1
            if fbar + post_fill_bars >= L:
                continue
            price_chg_ticks = (m[fbar + post_fill_bars] - m[fbar]) / TICK_SIZE * direction
            qdata[q]['fills'].append(float(price_chg_ticks))

        progress(f"  Signal-conditioned: day {d+1}/{n_days}")

    summary = {}
    log.info(f"  {'Q':>3s} {'SigRange':>18s} {'FillRate':>10s} {'AdvSel(t)':>10s} {'%Adverse':>9s}")
    for q in range(n_quintiles):
        d = qdata[q]
        fills = d['fills']
        total = d['total']
        fill_rate = len(fills) / total if total > 0 else 0.0
        if fills:
            pc = np.array(fills)
            mean_adv = float(-np.mean(pc))  # adv_sel = price moved against us
            pct_adv = float((pc < 0).mean())
        else:
            mean_adv, pct_adv = float('nan'), float('nan')

        qlo, qhi = float(qedges[q]), float(qedges[q + 1])
        summary[f'q{q+1}'] = {
            'signal_range': [qlo, qhi], 'n_total': total,
            'n_filled': len(fills), 'fill_rate': fill_rate,
            'mean_adv_sel_ticks': mean_adv, 'pct_adverse': pct_adv,
        }
        log.info(f"  {q+1:>3d} [{qlo:>6.4f},{qhi:>6.4f}] {fill_rate:>10.1%} "
                 f"{mean_adv:>10.4f}t {pct_adv:>9.1%}")

    return {'quintile_summary': summary, 'n_quintiles': n_quintiles,
            'fill_horizon_sec': fill_horizon_bars / BARS_PER_SEC,
            'post_fill_horizon_sec': post_fill_bars / BARS_PER_SEC}


# ============================================================================
# DIRECTION MODEL
# ============================================================================
def get_direction_model(scanner, target, min_train_days=3, fast=False):
    log.info("Running walk-forward direction model (ret_3s)...")
    feature_names = scanner.feature_names
    keep_mask = np.array([fn not in EXCLUDE_FEATURES_DIRECTION for fn in feature_names])
    features_use = scanner.features[:, keep_mask]
    feature_names_use = [fn for fn in feature_names if fn not in EXCLUDE_FEATURES_DIRECTION]

    params = {'n_estimators': 200 if fast else 500, 'max_depth': 6,
               'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7,
               'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 100,
               'verbose': -1, 'n_jobs': -1}

    result = walk_forward_evaluate(
        features=features_use, target=target,
        day_boundaries=scanner.day_boundaries,
        feature_names=feature_names_use,
        model_type='lgbm', min_train_days=min_train_days,
        hour_of_day=scanner.hour_of_day, lgbm_params=params,
    )
    if 'error' in result:
        log.error(f"Walk-forward failed: {result['error']}")
        return np.array([]), np.array([], dtype=np.int64), result

    log.info(f"  IC={result['ic']:.4f}  ICIR={result['icir']:.2f}  "
             f"t={result['tstat']:.2f}  n={result.get('n_preds', 0):,}")
    return result['predictions'], result['pred_indices'], result


# ============================================================================
# SUMMARY FORMATTER
# ============================================================================
def format_summary(spread_stats, fill_probs, adv_sel, strat_mkt, strat_lmt,
                   sig_cond, dir_result):
    d = spread_stats['distribution']
    lines = [
        "", "=" * 65, "LIMIT ORDER EXECUTION STUDY -- SUMMARY", "=" * 65, "",
        "DIRECTION MODEL (ret_3s walk-forward):",
        f"  IC={dir_result.get('ic', float('nan')):.4f}  "
        f"ICIR={dir_result.get('icir', float('nan')):.2f}  "
        f"t={dir_result.get('tstat', float('nan')):.2f}",
        "",
        "PART 1 -- SPREAD STATISTICS:",
        f"  1-tick: {d['pct_1tick']:.1%}  2-tick: {d['pct_2ticks']:.1%}  "
        f"3+tick: {d['pct_3plus']:.1%}",
        f"  Mean: {d['mean_ticks']:.3f}t  Median: {d['median_ticks']:.3f}t",
        f"  Passive fill earns ~${spread_stats['mean_edge_per_fill_dollars']:.2f}/trade (0.5*mean_spread)",
        "",
        "PART 2 -- FILL PROBABILITY (at best bid/ask, offset=0):",
    ]
    for hz_sec, s in sorted(fill_probs.get('fill_probs', {}).get(0, {}).items()):
        lines.append(f"  {hz_sec:>4.0f}s: buy={s['fill_rate_buy']:.1%}  "
                     f"sell={s['fill_rate_sell']:.1%}  avg={s['fill_rate_avg']:.1%}")

    lines += ["", "PART 3 -- ADVERSE SELECTION (ticks after fill, positive=good):"]
    for side in ['buy', 'sell']:
        sd = adv_sel.get(side, {})
        lines.append(f"  {side.upper()} fills (n={sd.get('n', 0)}):")
        for k_sec in adv_sel.get('post_fill_horizons_sec', []):
            lbl = f'k_{k_sec:.0f}s'
            s = sd.get(lbl, {})
            a_al = s.get('aligned_adv_ticks')
            a_op = s.get('opposed_adv_ticks')
            al_str = f"aligned={a_al:>+.3f}t" if a_al is not None else "aligned=n/a"
            op_str = f"opposed={a_op:>+.3f}t" if a_op is not None else ""
            lines.append(f"    {k_sec:>4.0f}s: adv={s.get('mean_adv_ticks', float('nan')):>+.4f}t "
                         f"({s.get('pct_adverse', 0):.1%} adverse)  {al_str}  {op_str}")

    lines += ["", "PART 4 -- STRATEGY SIMULATION:"]
    for label, strat in [("Limit entry + Market exit", strat_mkt),
                          ("Limit entry + Limit exit", strat_lmt)]:
        if 'error' in strat:
            lines.append(f"  {label}: ERROR -- {strat['error']}"); continue
        lines += [
            f"  {label}:",
            f"    Fill rate: {strat['fill_rate']:.1%}  ({strat['n_filled']}/{strat['n_posted']})",
            f"    Net PnL/trade: {strat['mean_pnl_ticks']:+.4f}t = ${strat['mean_pnl_dollars']:+.2f}",
            f"      Entry edge: {strat['mean_entry_edge_ticks']:+.4f}t",
            f"      Dir PnL:    {strat['mean_dir_pnl_ticks']:+.4f}t",
            f"      Exit cost:  {strat['mean_exit_cost_ticks']:+.4f}t",
            f"      Exit edge:  {strat['mean_exit_edge_ticks']:+.4f}t",
            f"    Win rate: {strat['win_rate']:.1%}  Approx Sharpe: {strat['sharpe_approx']:.2f}",
            f"    Total PnL: ${strat['total_pnl_dollars']:+.2f} over {strat['n_filled']} trades",
        ]

    lines += ["", "PART 5 -- SIGNAL-CONDITIONED FILLS:"]
    lines.append(f"  {'Q':>3s} {'SigHi':>10s} {'FillRate':>9s} {'AdvSel(t)':>10s} {'%Adverse':>9s}")
    for q_lbl, qs in sig_cond.get('quintile_summary', {}).items():
        lines.append(f"  {q_lbl:>3s} {qs['signal_range'][1]:>10.5f} "
                     f"{qs['fill_rate']:>9.1%} "
                     f"{qs['mean_adv_sel_ticks']:>10.4f}t "
                     f"{qs['pct_adverse']:>9.1%}")

    # Verdict
    lines += ["", "=" * 65, "VERDICT:"]
    if 'error' not in strat_mkt:
        pnl = strat_mkt['mean_pnl_ticks']
        sh = strat_mkt['sharpe_approx']
        fr = strat_mkt['fill_rate']
        if pnl > 0.05 and sh > 1.0:
            verdict = f"POSITIVE EDGE: {pnl:+.4f}t/trade, Sharpe~{sh:.1f}, fill={fr:.1%}"
        elif pnl > 0:
            verdict = f"MARGINAL: {pnl:+.4f}t/trade positive but Sharpe={sh:.2f} weak. More data needed."
        else:
            verdict = f"NO EDGE: {pnl:+.4f}t/trade negative. Adverse selection dominates."
        lines.append(f"  {verdict}")
    else:
        lines.append("  INCONCLUSIVE: Simulation failed.")

    lines += [
        "",
        "PROFITABILITY THRESHOLDS:",
        "  With 1-tick spread: earn $6.25 passive. Need win_rate > ~45% for breakeven.",
        "  Market exit costs $6.25 (0.5t). Both legs limit avoids exit cost.",
        "  Key risk: adverse selection after fill > passive edge earned.",
    ]
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Limit order execution study')
    parser.add_argument('--fast', action='store_true', help='Fewer LightGBM iters')
    parser.add_argument('--min-train-days', type=int, default=3)
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("LIMIT ORDER EXECUTION STUDY")
    log.info(f"  Mode: {'fast' if args.fast else 'full'}  TICK=${TICK_VALUE}")
    log.info("=" * 65)
    t_start = time.time()

    # --- Load data ---
    scanner = MBOAlphaScanner(sample_interval_ms=100)
    stats = load_feature_cache(scanner)
    if stats is None:
        log.info("No cache -- computing from scratch...")
        stats = scanner.load_from_cache()

    N = len(scanner.mid_prices)
    n_days = len(scanner.day_boundaries) - 1
    log.info(f"Loaded: {N:,} bars, {n_days} days, {len(scanner.feature_names)} features")

    fn = scanner.feature_names
    def gcol(name): return scanner.features[:, fn.index(name)]

    mid_prices = scanner.mid_prices
    spread   = gcol('spread')
    best_bid = gcol('best_bid')
    best_ask = gcol('best_ask')

    log.info(f"Sanity: mid={np.nanmean(mid_prices):.2f}  "
             f"spread_mean={np.nanmean(spread)/TICK_SIZE:.3f}t")

    # --- ret_3s target ---
    log.info("Computing ret_3s...")
    targets = compute_return_targets(
        mid_prices=mid_prices, day_boundaries=scanner.day_boundaries,
        sample_interval_ms=100, horizons_sec={'3s': 3}, include_flow_target=False,
    )
    ret_3s = targets['ret_3s']

    # --- Direction model ---
    predictions, pred_indices, dir_result = get_direction_model(
        scanner=scanner, target=ret_3s,
        min_train_days=args.min_train_days, fast=args.fast,
    )

    # --- Part 1 ---
    spread_stats = analyze_spread_stats(spread, scanner.hour_of_day, mid_prices,
                                        scanner.day_boundaries)
    gc.collect()

    # --- Part 2 ---
    hz_bars = ([30, 100, 300] if args.fast
               else [10, 30, 50, 100, 300])
    offsets = [0, -1] if args.fast else [0, -1, -2]
    fill_probs = analyze_fill_probability(
        mid_prices, best_bid, best_ask, scanner.day_boundaries,
        scanner.hour_of_day, horizons_bars=hz_bars, offsets_ticks=offsets,
    )
    gc.collect()

    # --- Part 3 ---
    adv_sel = analyze_adverse_selection(
        mid_prices, best_bid, best_ask, scanner.day_boundaries,
        predictions, pred_indices,
        fill_horizon_bars=5 * BARS_PER_SEC,
        post_fill_horizons=[10, 30, 50, 100, 300],
    )
    gc.collect()

    # --- Part 4 ---
    strat_mkt = simulate_limit_strategy(
        mid_prices, best_bid, best_ask, spread, scanner.day_boundaries,
        predictions, pred_indices,
        fill_horizon_bars=5 * BARS_PER_SEC, hold_horizon_bars=30 * BARS_PER_SEC,
        min_signal_quantile=0.6, use_limit_exit=False, label='MktExit',
    )
    strat_lmt = simulate_limit_strategy(
        mid_prices, best_bid, best_ask, spread, scanner.day_boundaries,
        predictions, pred_indices,
        fill_horizon_bars=5 * BARS_PER_SEC, hold_horizon_bars=30 * BARS_PER_SEC,
        min_signal_quantile=0.6, use_limit_exit=True, label='LmtExit',
    )
    gc.collect()

    # --- Part 5 ---
    sig_cond = analyze_signal_conditioned(
        mid_prices, best_bid, best_ask, scanner.day_boundaries,
        predictions, pred_indices,
        fill_horizon_bars=5 * BARS_PER_SEC, post_fill_bars=10 * BARS_PER_SEC,
    )
    gc.collect()

    # --- Summary ---
    summary = format_summary(spread_stats, fill_probs, adv_sel,
                              strat_mkt, strat_lmt, sig_cond, dir_result)
    log.info("\n" + summary)

    # --- Save ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f'limit_order_study_{timestamp}.json'

    def to_safe(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: to_safe(v) for k, v in obj.items()}
        if isinstance(obj, list): return [to_safe(v) for v in obj]
        return obj

    output = {
        'timestamp': timestamp,
        'data_stats': stats,
        'n_bars': int(N), 'n_days': int(n_days),
        'fast_mode': args.fast,
        'direction_model': {k: dir_result.get(k) for k in
                             ['ic', 'icir', 'tstat', 'n_preds', 'fold_ics', 'fold_con']},
        'spread_stats':           to_safe(spread_stats),
        'fill_probability':       to_safe(fill_probs),
        'adverse_selection':      to_safe(adv_sel),
        'strategy_market_exit':   to_safe(strat_mkt),
        'strategy_limit_exit':    to_safe(strat_lmt),
        'signal_conditioned':     to_safe(sig_cond),
        'summary': summary,
        'total_elapsed_sec': time.time() - t_start,
    }
    with open(str(out_file), 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    log.info(f"\nResults saved: {out_file}")
    log.info(f"Total elapsed: {(time.time()-t_start)/60:.1f} min")
    print("\n" + "=" * 65)
    print(summary)
    print(f"\nResults: {out_file}")
    return output


if __name__ == '__main__':
    main()
