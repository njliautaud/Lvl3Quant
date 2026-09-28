"""
Strategy 3: Hold Until Prediction Flips — CORRECTED Re-run
===========================================================
The previous agent run produced all zeros. This version has debug output.
"""

import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

TICK = 0.25
TICK_VAL = 12.50
BARS_PER_SEC = 10
TRAIN_DAYS = 40

COST_PROFILES = [
    (1.0, 0.24, 'ES_futures'),       # 1 tick spread + $3 RT comm
    (0.40, 0.0, 'SPY_500sh'),        # $5/RT for 500sh equiv
    (0.0, 0.0, 'zero_cost'),
]

LVL3_ROOT = Path(__file__).parent.parent
FEAT_CACHE = LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'
PRED_FILE = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oos_predictions_event_20260228_123832.npz'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'results'


def load_data():
    print("Loading predictions...")
    pred_data = np.load(str(PRED_FILE), allow_pickle=True)
    pred_dates = sorted(set(
        k.replace('_preds', '').replace('_targets', '')
        for k in pred_data.keys()
    ))
    print(f"  {len(pred_dates)} dates")

    days = {}
    for date in pred_dates:
        mbo_path = FEAT_CACHE / f'{date}_mbo_features.npz'
        if not mbo_path.exists():
            continue

        preds = pred_data[f'{date}_preds']
        targets = pred_data[f'{date}_targets']
        mbo = np.load(str(mbo_path))
        feats = mbo['mbo_features']
        mid = feats[:, 0].astype(np.float32)
        spread = feats[:, 1].astype(np.float32)

        n = min(len(mid), len(preds))
        raw_preds = preds[:n].copy()

        pred_std = np.std(raw_preds)
        pred_mean = np.mean(raw_preds)
        if pred_std > 0:
            z_preds = (raw_preds - pred_mean) / pred_std
        else:
            z_preds = np.zeros_like(raw_preds)

        days[date] = {
            'mid': mid[:n].copy(),
            'spread': spread[:n].copy(),
            'preds': raw_preds,
            'z_preds': z_preds,
            'targets': targets[:n].copy(),
            'n_bars': n,
        }
        del feats, mbo

    all_dates = sorted(days.keys())
    print(f"  Loaded {len(all_dates)} days")

    # Verify z-scores
    all_z = np.concatenate([days[d]['z_preds'] for d in all_dates])
    for t in [1.0, 1.5, 2.0, 2.5, 3.0]:
        pct = np.mean(np.abs(all_z) > t) * 100
        print(f"  |z| > {t}: {pct:.2f}%")

    return days, all_dates


def sim_hold_until_flip(mid, spread, preds, z_preds, thresh,
                        min_hold=100, max_hold=18000, cooldown=100,
                        spread_cost_ticks=0.0, comm_ticks=0.24):
    # HC #231(A): spread cost deleted — fill price already encodes side
    n = len(mid)
    abs_z = np.abs(z_preds)
    cost_per_trade = spread_cost_ticks + comm_ticks

    total_pnl = 0.0
    trades = 0
    wins = 0
    trade_pnls = []
    hold_times = []
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0

    for i in range(n):
        if in_pos:
            elapsed = i - entry_bar
            do_exit = False

            if elapsed >= max_hold:
                do_exit = True
            elif elapsed >= min_hold:
                if direction == 1 and preds[i] < 0:
                    do_exit = True
                elif direction == -1 and preds[i] > 0:
                    do_exit = True

            if do_exit:
                curr_pnl = direction * (mid[i] - entry_price) / TICK
                pnl = curr_pnl - cost_per_trade
                total_pnl += pnl
                trade_pnls.append(pnl)
                hold_times.append(elapsed)
                trades += 1
                if pnl > 0:
                    wins += 1
                in_pos = False
                last_exit = i

        elif i - last_exit >= cooldown and abs_z[i] > thresh:
            entry_price = mid[i]
            direction = 1 if preds[i] > 0 else -1
            entry_bar = i
            in_pos = True
            entry_price += direction * spread[i] / 2

    avg_hold = np.mean(hold_times) if hold_times else 0
    edge = np.mean(trade_pnls) if trade_pnls else 0
    return total_pnl * TICK_VAL, trades, wins, trade_pnls, edge, avg_hold, hold_times


def sim_vol_gated(mid, spread, preds, z_preds, thresh, vol_top_pct,
                  hold_bars=600, cooldown=100,
                  spread_cost_ticks=0.0, comm_ticks=0.24):
    # HC #231(A): spread cost deleted — fill price already encodes side
    """Vol-gated: trade when |z_pred|>thresh AND trailing rvol is in top N%."""
    n = len(mid)
    abs_z = np.abs(z_preds)
    cost_per_trade = spread_cost_ticks + comm_ticks

    # Compute trailing rvol (500 bar = 50s window)
    window = 500
    returns = np.diff(mid)
    rvol = np.zeros(n, dtype=np.float32)
    if len(returns) >= window:
        ret2 = returns ** 2
        cs = np.cumsum(returns)
        cs2 = np.cumsum(ret2)
        for i in range(window, n):
            s = cs[i-1] - (cs[i-1-window] if i-1-window >= 0 else 0)
            s2 = cs2[i-1] - (cs2[i-1-window] if i-1-window >= 0 else 0)
            var = s2/window - (s/window)**2
            rvol[i] = np.sqrt(max(var, 0))

    # Vol gate: top N% of nonzero rvol
    nonzero = rvol[rvol > 0]
    if len(nonzero) == 0:
        return 0.0, 0, 0, [], 0.0
    vol_cutoff = np.percentile(nonzero, (1.0 - vol_top_pct) * 100)

    total_pnl = 0.0
    trades = 0
    wins = 0
    trade_pnls = []
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0

    for i in range(n):
        if in_pos:
            elapsed = i - entry_bar
            if elapsed >= hold_bars:
                curr_pnl = direction * (mid[i] - entry_price) / TICK
                pnl = curr_pnl - cost_per_trade
                total_pnl += pnl
                trade_pnls.append(pnl)
                trades += 1
                if pnl > 0:
                    wins += 1
                in_pos = False
                last_exit = i
        elif (i - last_exit >= cooldown
              and abs_z[i] > thresh
              and rvol[i] >= vol_cutoff):
            entry_price = mid[i]
            direction = 1 if preds[i] > 0 else -1
            entry_bar = i
            in_pos = True
            entry_price += direction * spread[i] / 2

    edge = np.mean(trade_pnls) if trade_pnls else 0
    return total_pnl * TICK_VAL, trades, wins, trade_pnls, edge


def main():
    t0 = time.time()
    print("=" * 70)
    print("ADVANCED STRATEGY TEST (CORRECTED RE-RUN)")
    print("=" * 70)

    days, all_dates = load_data()
    train_dates = all_dates[:TRAIN_DAYS]
    test_dates = all_dates[TRAIN_DAYS:]
    print(f"\nTRAIN: {len(train_dates)} days | TEST: {len(test_dates)} days")

    # ── STRATEGY 3: Hold Until Flip ──
    print("\n" + "=" * 70)
    print("STRATEGY 3: HOLD UNTIL PREDICTION FLIPS")
    print("=" * 70)

    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    min_hold = 100
    max_hold = 18000

    s3_results = {}

    for cost_spread, cost_comm, cost_label in COST_PROFILES:
        cost_total = cost_spread + cost_comm
        print(f"\n--- {cost_label} ({cost_total:.2f} ticks RT) ---")

        best_train_pnl = -np.inf
        best_thresh = None
        config_results = []

        for thresh in thresholds:
            train_pnl = 0.0
            train_trades = 0
            train_day_pnls = []
            all_hold_times = []

            for date in train_dates:
                d = days[date]
                pnl, trades, wins, tpnls, edge, ah, ht = sim_hold_until_flip(
                    d['mid'], d['spread'], d['preds'], d['z_preds'], thresh,
                    min_hold=min_hold, max_hold=max_hold, cooldown=100,
                    spread_cost_ticks=cost_spread, comm_ticks=cost_comm
                )
                train_pnl += pnl
                train_trades += trades
                train_day_pnls.append(pnl)
                all_hold_times.extend(ht)

            tpd = train_trades / len(train_dates) if train_dates else 0
            avg_hold_s = np.mean(all_hold_times) / BARS_PER_SEC if all_hold_times else 0
            med_hold_s = np.median(all_hold_times) / BARS_PER_SEC if all_hold_times else 0
            avg_daily = np.mean(train_day_pnls) if train_day_pnls else 0
            std_daily = np.std(train_day_pnls) if train_day_pnls else 1
            sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0

            print(f"  TRAIN t={thresh:.1f}: ${train_pnl:+9,.0f} | {tpd:5.1f} t/d | "
                  f"Sharpe={sharpe:+.2f} | avg_hold={avg_hold_s:.0f}s med={med_hold_s:.0f}s")

            config_results.append({
                'thresh': thresh, 'train_pnl': round(train_pnl),
                'train_trades': train_trades, 'train_sharpe': round(sharpe, 2),
                'trades_per_day': round(tpd, 1),
                'avg_hold_s': round(avg_hold_s, 1), 'med_hold_s': round(med_hold_s, 1),
            })

            if train_pnl > best_train_pnl:
                best_train_pnl = train_pnl
                best_thresh = thresh

        print(f"\n  BEST TRAIN: t={best_thresh} (${best_train_pnl:+,.0f})")

        # OOS ALL thresholds
        oos_by_thresh = {}
        for thresh in thresholds:
            oos_pnl = 0.0
            oos_trades = 0
            oos_wins = 0
            oos_day_pnls = []
            all_trade_pnls = []
            all_hold_times = []

            for date in test_dates:
                d = days[date]
                pnl, trades, wins, tpnls, edge, ah, ht = sim_hold_until_flip(
                    d['mid'], d['spread'], d['preds'], d['z_preds'], thresh,
                    min_hold=min_hold, max_hold=max_hold, cooldown=100,
                    spread_cost_ticks=cost_spread, comm_ticks=cost_comm
                )
                oos_pnl += pnl
                oos_trades += trades
                oos_wins += wins
                oos_day_pnls.append(pnl)
                all_trade_pnls.extend(tpnls)
                all_hold_times.extend(ht)

            avg_oos = np.mean(oos_day_pnls) if oos_day_pnls else 0
            std_oos = np.std(oos_day_pnls) if oos_day_pnls else 1
            sharpe_oos = (avg_oos / std_oos * np.sqrt(252)) if std_oos > 0 else 0
            wr = oos_wins / oos_trades if oos_trades > 0 else 0
            avg_edge = np.mean(all_trade_pnls) if all_trade_pnls else 0
            tpd = oos_trades / len(test_dates) if test_dates else 0
            avg_hold_s = np.mean(all_hold_times) / BARS_PER_SEC if all_hold_times else 0
            med_hold_s = np.median(all_hold_times) / BARS_PER_SEC if all_hold_times else 0

            monthly = defaultdict(lambda: {'pnl': 0.0, 'days': 0})
            for date, dp in zip(test_dates, oos_day_pnls):
                m = date[:7]
                monthly[m]['pnl'] += dp
                monthly[m]['days'] += 1

            is_best = (thresh == best_thresh)
            marker = " <<<< BEST" if is_best else ""
            print(f"\n  OOS t={thresh:.1f}:{marker}")
            print(f"    PnL: ${oos_pnl:+,.0f} | Sharpe: {sharpe_oos:+.2f} | {oos_trades} trades ({tpd:.1f}/d)")
            print(f"    WR: {wr:.1%} | Edge: {avg_edge:+.3f} ticks | Hold: avg={avg_hold_s:.0f}s med={med_hold_s:.0f}s")
            for m in sorted(monthly.keys()):
                print(f"      {m}: ${monthly[m]['pnl']:+8,.0f}")

            oos_by_thresh[str(thresh)] = {
                'oos_pnl': round(oos_pnl),
                'oos_sharpe': round(sharpe_oos, 2),
                'oos_trades': oos_trades,
                'trades_per_day': round(tpd, 1),
                'win_rate': round(wr, 3),
                'avg_edge_ticks': round(avg_edge, 3),
                'avg_hold_s': round(avg_hold_s, 1),
                'med_hold_s': round(med_hold_s, 1),
                'monthly': {m: {'pnl': round(v['pnl']), 'days': v['days']} for m, v in monthly.items()},
                'is_best_train': is_best,
            }

        s3_results[cost_label] = {
            'best_train_thresh': best_thresh,
            'best_train_pnl': round(best_train_pnl),
            'oos_by_threshold': oos_by_thresh,
            'config_results': config_results,
        }

    # ── STRATEGY 2: Vol-Gated ──
    print("\n" + "=" * 70)
    print("STRATEGY 2: VOL-GATED TRANSFORMER")
    print("=" * 70)

    s2_thresholds = [1.0, 1.5, 2.0]
    vol_gates = [0.10, 0.20, 0.30, 0.50]
    s2_results = {}

    for cost_spread, cost_comm, cost_label in COST_PROFILES:
        cost_total = cost_spread + cost_comm
        print(f"\n--- {cost_label} ({cost_total:.2f} ticks RT) ---")

        best_train_pnl = -np.inf
        best_config = None
        config_results = []

        for thresh in s2_thresholds:
            for vg in vol_gates:
                train_pnl = 0.0
                train_trades = 0

                for date in train_dates:
                    d = days[date]
                    pnl, trades, wins, _, edge = sim_vol_gated(
                        d['mid'], d['spread'], d['preds'], d['z_preds'],
                        thresh, vg, hold_bars=600, cooldown=100,
                        spread_cost_ticks=cost_spread, comm_ticks=cost_comm
                    )
                    train_pnl += pnl
                    train_trades += trades

                tpd = train_trades / len(train_dates) if train_dates else 0
                print(f"  TRAIN t={thresh:.1f} vol_top={vg*100:.0f}%: ${train_pnl:+9,.0f} | {tpd:5.1f} t/d")

                config_results.append({
                    'thresh': thresh, 'vol_gate': vg,
                    'train_pnl': round(train_pnl), 'train_trades': train_trades,
                    'trades_per_day': round(tpd, 1),
                })

                if train_pnl > best_train_pnl:
                    best_train_pnl = train_pnl
                    best_config = (thresh, vg)

        b_t, b_v = best_config
        print(f"\n  BEST TRAIN: t={b_t}, vol_top={b_v*100:.0f}% (${best_train_pnl:+,.0f})")

        # OOS
        oos_pnl = 0.0
        oos_trades = 0
        oos_wins = 0
        oos_day_pnls = []

        for date in test_dates:
            d = days[date]
            pnl, trades, wins, _, edge = sim_vol_gated(
                d['mid'], d['spread'], d['preds'], d['z_preds'],
                b_t, b_v, hold_bars=600, cooldown=100,
                spread_cost_ticks=cost_spread, comm_ticks=cost_comm
            )
            oos_pnl += pnl
            oos_trades += trades
            oos_wins += wins
            oos_day_pnls.append(pnl)

        avg_oos = np.mean(oos_day_pnls) if oos_day_pnls else 0
        std_oos = np.std(oos_day_pnls) if oos_day_pnls else 1
        sharpe_oos = (avg_oos / std_oos * np.sqrt(252)) if std_oos > 0 else 0
        wr = oos_wins / oos_trades if oos_trades > 0 else 0
        tpd = oos_trades / len(test_dates) if test_dates else 0

        monthly = defaultdict(lambda: {'pnl': 0.0, 'days': 0})
        for date, dp in zip(test_dates, oos_day_pnls):
            m = date[:7]
            monthly[m]['pnl'] += dp
            monthly[m]['days'] += 1

        print(f"\n  OOS ({cost_label}): ${oos_pnl:+,.0f} | Sharpe={sharpe_oos:+.2f} | {oos_trades} trades ({tpd:.1f}/d) | WR={wr:.1%}")
        for m in sorted(monthly.keys()):
            print(f"    {m}: ${monthly[m]['pnl']:+8,.0f}")

        s2_results[cost_label] = {
            'best_thresh': b_t, 'best_vol_gate': b_v,
            'best_train_pnl': round(best_train_pnl),
            'oos_pnl': round(oos_pnl),
            'oos_sharpe': round(sharpe_oos, 2),
            'oos_trades': oos_trades,
            'trades_per_day': round(tpd, 1),
            'win_rate': round(wr, 3),
            'monthly': {m: {'pnl': round(v['pnl']), 'days': v['days']} for m, v in monthly.items()},
            'config_results': config_results,
        }

    # ── FINAL COMPARISON ──
    print("\n" + "=" * 70)
    print("COMPARISON TABLE (OOS)")
    print("=" * 70)
    print(f"{'Strategy':<40} {'ES PnL':>10} {'SPY PnL':>10} {'Zero PnL':>10} {'ES Sharpe':>10}")
    print("-" * 82)

    # S2
    es2 = s2_results.get('ES_futures', {})
    spy2 = s2_results.get('SPY_500sh', {})
    zc2 = s2_results.get('zero_cost', {})
    print(f"{'S2: Vol-Gated':<40} ${es2.get('oos_pnl',0):>+9,} ${spy2.get('oos_pnl',0):>+9,} ${zc2.get('oos_pnl',0):>+9,} {es2.get('oos_sharpe',0):>+10.2f}")

    # S3 all thresholds
    for t_str in sorted(s3_results.get('ES_futures', {}).get('oos_by_threshold', {}).keys(), key=float):
        es3 = s3_results.get('ES_futures', {}).get('oos_by_threshold', {}).get(t_str, {})
        spy3 = s3_results.get('SPY_500sh', {}).get('oos_by_threshold', {}).get(t_str, {})
        zc3 = s3_results.get('zero_cost', {}).get('oos_by_threshold', {}).get(t_str, {})
        best = " *" if es3.get('is_best_train') else ""
        print(f"{'S3: Flip t=' + t_str + best:<40} ${es3.get('oos_pnl',0):>+9,} ${spy3.get('oos_pnl',0):>+9,} ${zc3.get('oos_pnl',0):>+9,} {es3.get('oos_sharpe',0):>+10.2f}")

    # Check profitability
    print(f"\n--- PROFITABLE AT ES OR SPY COSTS? ---")
    found = False
    for t_str in s3_results.get('ES_futures', {}).get('oos_by_threshold', {}).keys():
        es_pnl = s3_results['ES_futures']['oos_by_threshold'][t_str]['oos_pnl']
        spy_pnl = s3_results['SPY_500sh']['oos_by_threshold'][t_str]['oos_pnl']
        if es_pnl > 0:
            print(f"  >> S3 t={t_str} PROFITABLE at ES: ${es_pnl:+,}")
            found = True
        if spy_pnl > 0:
            print(f"  >> S3 t={t_str} PROFITABLE at SPY: ${spy_pnl:+,}")
            found = True

    es2_pnl = s2_results.get('ES_futures', {}).get('oos_pnl', 0)
    spy2_pnl = s2_results.get('SPY_500sh', {}).get('oos_pnl', 0)
    if es2_pnl > 0:
        print(f"  >> S2 PROFITABLE at ES: ${es2_pnl:+,}")
        found = True
    if spy2_pnl > 0:
        print(f"  >> S2 PROFITABLE at SPY: ${spy2_pnl:+,}")
        found = True

    if not found:
        print("  NO PROFITABLE CONFIGS at ES or SPY costs.")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.0f}s")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = {
        'timestamp': ts,
        'strategy_2_vol_gated': s2_results,
        'strategy_3_hold_until_flip': s3_results,
        'elapsed_s': round(elapsed, 1),
    }
    out_path = RESULTS_DIR / f'strategy23_corrected_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
