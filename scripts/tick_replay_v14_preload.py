#!/usr/bin/env python3
"""
Tick Replay v14-preload — Preload MBO + Sweep Configs
======================================================

Uses TickReplayEngine.preload_mbo() + run_day_from_arrays() to parse each
MBO day ONCE, then re-run the event loop for each config. Each config still
iterates 15M events (order book must be re-simulated), but avoids the 2-3 min
parse overhead.

Expected: ~60-90s per config-day (pure Python event loop).
With 3 days × 12 configs × 3 heads × 3 quantiles = 324 config-days
→ ~5-8 hours total.

For faster screening, we test 3 days first, then expand winners to all 32 days.

HC #659: tick-level FIFO replay, permutation test.

Author: Claude
Date: 2026-07-10
"""

import sys, os, time, json
import numpy as np
import glob
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/jupiter/Lvl3Quant/engines')
from tick_replay_engine import TickReplayEngine, compute_metrics

MBO_DIR = '/home/jupiter/Lvl3Quant/data/raw/mbo'
PRED_DIR = '/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/tick_replay_v14_preload'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIGNAL_HEADS = ['pred_log_ret_1s', 'pred_fifo_tp4sl3_net', 'pred_pred_mfe_30s_ticks']
N_SCREEN_DAYS = 3
N_PERMS = 30

def load_predictions(pred_dir):
    preds = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, 'oot_*.npz'))):
        d = np.load(f, allow_pickle=True)
        date_str = os.path.basename(f).replace('oot_', '').replace('.npz', '')
        heads = {}
        for head in SIGNAL_HEADS:
            if head in d:
                heads[head] = d[head].astype(np.float32)
        if heads:
            preds[date_str] = heads
    return preds

def compute_quantile_threshold(preds_dict, head, q):
    vals = []
    for d, heads in preds_dict.items():
        if head in heads:
            vals.append(np.abs(heads[head]))
    if not vals:
        return 999.0
    return float(np.percentile(np.concatenate(vals), (1 - q) * 100))


def main():
    t0 = time.time()
    print("=" * 70)
    print("TICK REPLAY v14-PRELOAD — PARSE ONCE, SWEEP MANY")
    print("=" * 70)

    # Load predictions
    print("\nLoading predictions...")
    preds_dict = load_predictions(PRED_DIR)
    print(f"  {len(preds_dict)} dates")

    # Match MBO
    mbo_files = sorted(glob.glob(os.path.join(MBO_DIR, 'glbx-mdp3-*.mbo.dbn.zst')))
    all_matched = []
    for mbo_path in mbo_files:
        date8 = os.path.basename(mbo_path).split('-')[2].split('.')[0]
        if date8 in preds_dict:
            all_matched.append((mbo_path, date8))

    # Select screening days (spread out)
    indices = np.linspace(0, len(all_matched)-1, N_SCREEN_DAYS, dtype=int)
    screen_matched = [all_matched[i] for i in indices]
    print(f"  Screening: {[d for _,d in screen_matched]}")

    # Preload MBO data
    print(f"\nPreloading {len(screen_matched)} MBO days...")
    preloaded = {}
    for mbo_path, date_key in screen_matched:
        t1 = time.time()
        data = TickReplayEngine.preload_mbo(mbo_path)
        if data:
            preloaded[date_key] = data
            print(f"  {date_key}: {data['n_events']:,} events ({time.time()-t1:.0f}s)")

    # Focused test configs
    QUANTILES = [0.05, 0.10, 0.20]

    CONFIGS = [
        # (hold_s, tp, sl, cancel_s)
        (5,  3, 3, 10),
        (5,  4, 3, 10),
        (10, 4, 3, 15),
        (10, 3, 5, 15),
        (15, 5, 3, 15),
        (5, 99, 99, 10),    # pure hold 5s
        (10, 99, 99, 15),   # pure hold 10s
        (15, 99, 99, 15),   # pure hold 15s
        (30, 99, 99, 20),   # pure hold 30s
        (3,  2, 2, 8),      # very tight
        (7,  3, 4, 12),
        (5,  2, 3, 10),
    ]

    print(f"\n  {len(SIGNAL_HEADS)} heads × {len(QUANTILES)} quantiles × {len(CONFIGS)} configs "
          f"= {len(SIGNAL_HEADS) * len(QUANTILES) * len(CONFIGS)} combos")
    print(f"  × {len(preloaded)} days = {len(SIGNAL_HEADS) * len(QUANTILES) * len(CONFIGS) * len(preloaded)} engine runs")

    results = []
    promising = []
    config_count = 0

    for head in SIGNAL_HEADS:
        has_head = any(head in preds_dict[d] for d in preloaded if d in preds_dict)
        if not has_head:
            print(f"\n  Skipping {head}")
            continue

        print(f"\n{'='*60}")
        print(f"HEAD: {head}")
        print(f"{'='*60}")

        for q in QUANTILES:
            thresh = compute_quantile_threshold(preds_dict, head, q)
            print(f"\n  Q={q*100:.0f}% → threshold={thresh:.6f}")

            for hold_s, tp, sl, cancel_s in CONFIGS:
                config_count += 1
                label = f"{head.split('_',1)[1]}|q{q*100:.0f}|h{hold_s}tp{tp}sl{sl}"

                all_trades = []
                day_pnls = []
                t1 = time.time()

                for date_key, data in preloaded.items():
                    if date_key not in preds_dict or head not in preds_dict[date_key]:
                        continue

                    engine = TickReplayEngine(
                        tp_ticks=tp, sl_ticks=sl,
                        hold_seconds=hold_s,
                        signal_threshold=thresh,
                        cancel_seconds=cancel_s,
                    )
                    trades = engine.run_day_from_arrays(
                        data['ts_event'], data['actions'], data['sides'],
                        data['prices'], data['sizes'], data['order_ids'],
                        preds_dict[date_key][head]
                    )
                    all_trades.extend(trades)
                    day_pnls.append(sum(t.pnl_ticks for t in trades))

                elapsed = time.time() - t1

                if not all_trades:
                    if config_count <= 5:  # Only print first few NO TRADES
                        print(f"    [{config_count:3d}] {label}: NO TRADES ({elapsed:.0f}s)")
                    continue

                m = compute_metrics(all_trades, label)
                n = m.get('n_trades', 0)
                net = m.get('net_pnl_ticks', 0)
                wr = m.get('win_rate', 0)
                sharpe = m.get('sharpe', 0)
                avg = net / max(n, 1)
                green = sum(1 for p in day_pnls if p > 0)
                red = sum(1 for p in day_pnls if p < 0)
                sign = '+' if net > 0 else ''

                result = {
                    'label': label, 'head': head, 'quantile': q,
                    'threshold': thresh, 'hold_s': hold_s,
                    'tp': tp, 'sl': sl,
                    'n_trades': n, 'trades_per_day': round(n / max(len(day_pnls), 1), 1),
                    'net_ticks': round(float(net), 1),
                    'per_trade': round(float(avg), 4),
                    'win_rate': round(float(wr), 4),
                    'sharpe': round(float(sharpe), 3),
                    'green_days': green, 'red_days': red,
                    'n_days': len(day_pnls),
                    'exit_reasons': m.get('exit_reasons', {}),
                    'avg_mfe': round(float(np.mean([t.mfe_ticks for t in all_trades])), 2),
                    'avg_mae': round(float(np.mean([t.mae_ticks for t in all_trades])), 2),
                    'elapsed_s': round(elapsed, 1),
                }
                results.append(result)

                marker = "✅" if sharpe > 0.5 else ("➕" if sharpe > 0 else "  ")
                print(f"    [{config_count:3d}] {marker} {label}: {sign}{net:.0f}t "
                      f"({sign}{avg:.3f}t/tr) n={n} WR={wr:.1%} Sh={sharpe:.2f} "
                      f"G/R={green}/{red} MFE={result['avg_mfe']:.1f} MAE={result['avg_mae']:.1f} "
                      f"({elapsed:.0f}s)")

                if sharpe > 0.3 and n >= 8:
                    promising.append(result)

    # Sort and print summary
    print(f"\n{'='*60}")
    print(f"SCREENING COMPLETE — {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
    print(f"{'='*60}")
    print(f"Configs tested: {len(results)}")
    print(f"Promising (Sharpe>0.3): {len(promising)}")

    # Run permutation on top 5 promising
    if promising:
        print(f"\n{'='*60}")
        print(f"PERMUTATION TESTS (top 5 of {len(promising)})")
        print(f"{'='*60}")

        validated = []
        for cfg in sorted(promising, key=lambda x: -x['sharpe'])[:5]:
            head = cfg['head']
            thresh = cfg['threshold']
            tp, sl = cfg['tp'], cfg['sl']
            hold_s = cfg['hold_s']
            real_sharpe = cfg['sharpe']

            print(f"\n  {cfg['label']} (Sh={real_sharpe:.2f}, n={cfg['n_trades']})")
            t1 = time.time()

            perm_sharpes = []
            for pi in range(N_PERMS):
                perm_trades = []
                for date_key, data in preloaded.items():
                    if date_key not in preds_dict or head not in preds_dict[date_key]:
                        continue
                    preds = preds_dict[date_key][head].copy()
                    signs = np.random.choice([-1, 1], size=len(preds))
                    preds_shuffled = np.abs(preds) * signs

                    engine = TickReplayEngine(
                        tp_ticks=tp, sl_ticks=sl,
                        hold_seconds=hold_s,
                        signal_threshold=thresh,
                        cancel_seconds=cfg.get('cancel_s', 15.0) if 'cancel_s' in cfg else 15.0,
                    )
                    trades = engine.run_day_from_arrays(
                        data['ts_event'], data['actions'], data['sides'],
                        data['prices'], data['sizes'], data['order_ids'],
                        preds_shuffled
                    )
                    perm_trades.extend(trades)

                pm = compute_metrics(perm_trades, f'perm_{pi}')
                perm_sharpes.append(pm.get('sharpe', 0))

            p_val = np.mean([s >= real_sharpe for s in perm_sharpes])
            cfg['perm_p'] = round(p_val, 4)
            cfg['rand_sharpe_mean'] = round(float(np.mean(perm_sharpes)), 3)
            cfg['rand_sharpe_std'] = round(float(np.std(perm_sharpes)), 3)

            status = "✅ PASS" if p_val < 0.05 else "❌ FAIL"
            print(f"    {status}: p={p_val:.3f} (random: {cfg['rand_sharpe_mean']:.2f} ± "
                  f"{cfg['rand_sharpe_std']:.2f}) ({time.time()-t1:.0f}s)")

            if p_val < 0.05:
                validated.append(cfg)
    else:
        validated = []

    # Top 20
    print(f"\nTOP 20 BY SHARPE:")
    for r in sorted(results, key=lambda x: -x.get('sharpe', -999))[:20]:
        perm = f" p={r['perm_p']}" if 'perm_p' in r else ""
        print(f"  {r['label']}: Sh={r['sharpe']:.2f} WR={r['win_rate']:.1%} "
              f"n={r['n_trades']} {r['per_trade']:+.4f}t/tr "
              f"MFE={r['avg_mfe']:.1f} MAE={r['avg_mae']:.1f} "
              f"G/R={r['green_days']}/{r['red_days']}{perm}")

    # Diagnostic: MFE/MAE by hold time
    print(f"\nDIAGNOSTIC — Pure hold (no TP/SL) by hold time:")
    for h in [5, 10, 15, 30]:
        h_res = [r for r in results if r.get('hold_s') == h
                 and r.get('tp') == 99 and r.get('sl') == 99 and r['n_trades'] > 0]
        if h_res:
            avg_per = np.mean([r['per_trade'] for r in h_res])
            avg_wr = np.mean([r['win_rate'] for r in h_res])
            avg_mfe = np.mean([r['avg_mfe'] for r in h_res])
            avg_mae = np.mean([r['avg_mae'] for r in h_res])
            print(f"  hold={h}s: avg {avg_per:+.3f}t/tr WR={avg_wr:.1%} "
                  f"MFE={avg_mfe:.1f} MAE={avg_mae:.1f}")

    total_elapsed = time.time() - t0
    print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'v14_preload_results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'generated': time.strftime('%Y-%m-%d %H:%M'),
            'elapsed_s': round(total_elapsed, 1),
            'n_configs': len(results),
            'n_promising': len(promising),
            'n_validated': len(validated),
            'validated': validated,
            'all_results': results,
            'top_20': sorted(results, key=lambda x: -x.get('sharpe', -999))[:20],
        }, f, indent=2, default=str)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
