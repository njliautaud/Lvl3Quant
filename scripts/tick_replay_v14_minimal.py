#!/usr/bin/env python3
"""
Tick Replay v14-minimal — Speed-first config screening
========================================================

Strategy: Use the existing TickReplayEngine.run_day() for 3 days only,
but test ONE config at a time. Each day takes ~3 min, so 3 days × N configs.

To keep total time <60 min, test 6 configs per head × 3 heads = 18 configs.
Focus on the most promising parameter combinations.

HC #659: tick-level, permutation test.

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
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/tick_replay_v14_minimal'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIGNAL_HEADS = ['pred_log_ret_1s', 'pred_fifo_tp4sl3_net', 'pred_pred_mfe_30s_ticks']
N_DAYS = 5  # 5 spread-out days
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

def compute_quantile_thresholds(preds_dict, head, quantiles):
    all_vals = []
    for d, heads in preds_dict.items():
        if head in heads:
            all_vals.append(np.abs(heads[head]))
    if not all_vals:
        return {}
    combined = np.concatenate(all_vals)
    return {q: float(np.percentile(combined, (1 - q) * 100)) for q in quantiles}

def run_config_on_days(mbo_matched, preds_dict, head, threshold, tp, sl, hold_s, cancel_s=15.0):
    """Run single config across matched days. Returns trades + day_pnls."""
    all_trades = []
    day_pnls = []

    for mbo_path, date_key in mbo_matched:
        if head not in preds_dict[date_key]:
            continue

        engine = TickReplayEngine(
            tp_ticks=tp, sl_ticks=sl,
            hold_seconds=hold_s,
            signal_threshold=threshold,
            cancel_seconds=cancel_s,
        )
        trades = engine.run_day(mbo_path, preds_dict[date_key][head])
        all_trades.extend(trades)
        day_pnls.append(sum(t.pnl_ticks for t in trades))

    return all_trades, day_pnls

def run_permutation(mbo_matched, preds_dict, head, threshold, tp, sl, hold_s,
                    real_sharpe, n_perms=30):
    """Permutation test: shuffle signal signs, re-run."""
    perm_sharpes = []
    for _ in range(n_perms):
        perm_trades = []
        for mbo_path, date_key in mbo_matched:
            if head not in preds_dict[date_key]:
                continue
            preds = preds_dict[date_key][head].copy()
            signs = np.random.choice([-1, 1], size=len(preds))
            preds_shuffled = np.abs(preds) * signs

            engine = TickReplayEngine(
                tp_ticks=tp, sl_ticks=sl,
                hold_seconds=hold_s,
                signal_threshold=threshold,
                cancel_seconds=15.0,
            )
            trades = engine.run_day(mbo_path, preds_shuffled)
            perm_trades.extend(trades)

        if perm_trades:
            m = compute_metrics(perm_trades, 'perm')
            perm_sharpes.append(m.get('sharpe', 0))
        else:
            perm_sharpes.append(0)

    p_val = np.mean([s >= real_sharpe for s in perm_sharpes])
    return p_val, perm_sharpes

def main():
    t0 = time.time()
    print("=" * 70)
    print("TICK REPLAY v14-MINIMAL — FOCUSED SCREENING")
    print("=" * 70)

    # Load predictions
    print("\nLoading predictions...")
    preds_dict = load_predictions(PRED_DIR)
    print(f"  {len(preds_dict)} dates")

    # Match MBO
    mbo_files = sorted(glob.glob(os.path.join(MBO_DIR, 'glbx-mdp3-*.mbo.dbn.zst')))
    matched = []
    for mbo_path in mbo_files:
        date8 = os.path.basename(mbo_path).split('-')[2].split('.')[0]
        if date8 in preds_dict:
            matched.append((mbo_path, date8))

    # Select N spread-out days
    indices = np.linspace(0, len(matched)-1, N_DAYS, dtype=int)
    matched = [matched[i] for i in indices]
    print(f"  {len(matched)} days: {[d for _,d in matched]}")

    # Focused configs: based on signal characteristics
    # IC_1s = 0.22, signal decays by 30s, short side stronger
    # Key insight: most predictions are near 0 (mean 0.057, std 0.187)
    # Useful quantiles: top 5%, 10%, 20%
    FOCUSED_QUANTILES = [0.05, 0.10, 0.20]

    # Based on prior decay analysis: 1s-5s horizon is best
    # TP/SL should match 1-5 tick MFE within that horizon
    FOCUSED_CONFIGS = [
        # (hold_s, tp, sl, cancel_s, label_suffix)
        (5,  3, 3, 10, 'sym3_h5'),
        (5,  4, 3, 10, 'asym_h5'),
        (10, 4, 3, 15, 'asym_h10'),
        (10, 3, 5, 15, 'wideSL_h10'),
        (15, 5, 3, 15, 'wideTP_h15'),
        (5, 99, 99, 10, 'pure_hold5'),
        (10, 99, 99, 15, 'pure_hold10'),
        (15, 99, 99, 15, 'pure_hold15'),
        (30, 99, 99, 20, 'pure_hold30'),
        (3,  2, 2, 8, 'tight_h3'),
        (5,  2, 3, 10, 'smallTP_h5'),
        (7,  3, 4, 12, 'asym_h7'),
    ]

    results = []
    promising = []
    config_count = 0

    for head in SIGNAL_HEADS:
        has_head = any(head in preds_dict[d] for d in preds_dict)
        if not has_head:
            print(f"\n  Skipping {head} — not in predictions")
            continue

        # Compute quantile thresholds
        thresholds = compute_quantile_thresholds(preds_dict, head, FOCUSED_QUANTILES)
        print(f"\n{'='*60}")
        print(f"HEAD: {head}")
        for q, t in thresholds.items():
            print(f"  top-{q*100:.0f}%: threshold={t:.6f}")
        print(f"{'='*60}")

        for q in FOCUSED_QUANTILES:
            thresh = thresholds[q]

            for hold_s, tp, sl, cancel_s, label_suffix in FOCUSED_CONFIGS:
                config_count += 1
                label = f"{head.split('_',1)[1]}|q{q*100:.0f}|{label_suffix}"

                t1 = time.time()
                trades, day_pnls = run_config_on_days(
                    matched, preds_dict, head, thresh, tp, sl, hold_s, cancel_s
                )
                elapsed = time.time() - t1

                if not trades:
                    print(f"  [{config_count:3d}] {label}: NO TRADES ({elapsed:.0f}s)")
                    continue

                m = compute_metrics(trades, label)
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
                }
                results.append(result)

                marker = "✅" if sharpe > 0.5 else ("➕" if sharpe > 0 else "  ")
                print(f"  [{config_count:3d}] {marker} {label}: {sign}{net:.0f}t "
                      f"({sign}{avg:.3f}t/tr) n={n} WR={wr:.1%} Sh={sharpe:.2f} "
                      f"G/R={green}/{red} exits={m.get('exit_reasons',{})} ({elapsed:.0f}s)")

                if sharpe > 0.3 and n >= 10:
                    promising.append(result)

    # Permutation tests
    print(f"\n{'='*60}")
    print(f"PERMUTATION TESTS ({len(promising)} promising configs)")
    print(f"{'='*60}")

    validated = []
    for cfg in sorted(promising, key=lambda x: -x['sharpe'])[:8]:
        head = cfg['head']
        thresh = cfg['threshold']
        tp, sl = cfg['tp'], cfg['sl']
        hold_s = cfg['hold_s']
        real_sharpe = cfg['sharpe']

        print(f"\n  {cfg['label']} (Sh={real_sharpe:.2f}, n={cfg['n_trades']})")
        t1 = time.time()

        p_val, rand_sharpes = run_permutation(
            matched, preds_dict, head, thresh, tp, sl, hold_s,
            real_sharpe, n_perms=N_PERMS
        )

        cfg['perm_p'] = round(p_val, 4)
        cfg['rand_sharpe_mean'] = round(float(np.mean(rand_sharpes)), 3)
        cfg['rand_sharpe_std'] = round(float(np.std(rand_sharpes)), 3)

        status = "✅ PASS" if p_val < 0.05 else "❌ FAIL"
        elapsed = time.time() - t1
        print(f"    {status}: p={p_val:.3f} (random: {cfg['rand_sharpe_mean']:.2f} ± "
              f"{cfg['rand_sharpe_std']:.2f}) ({elapsed:.0f}s)")

        if p_val < 0.05:
            validated.append(cfg)

    # Summary
    total_elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE — {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"Configs tested: {len(results)}")
    print(f"Promising (Sharpe>0.3): {len(promising)}")
    print(f"Permutation-validated: {len(validated)}")

    if validated:
        print(f"\n✅ VALIDATED CONFIGS:")
        for v in validated:
            print(f"  {v['label']}: Sh={v['sharpe']:.2f} WR={v['win_rate']:.1%} "
                  f"n={v['n_trades']} {v['per_trade']:+.4f}t/tr perm_p={v['perm_p']:.3f}")
    else:
        print(f"\n⚠️  NO CONFIGS PASSED PERMUTATION TEST")
        # Still useful: print diagnostic info
        if results:
            positive = [r for r in results if r['sharpe'] > 0]
            print(f"\n  {len(positive)}/{len(results)} configs had positive Sharpe")
            if positive:
                best = max(positive, key=lambda x: x['sharpe'])
                print(f"  Best: {best['label']} Sh={best['sharpe']:.2f} "
                      f"WR={best['win_rate']:.1%} n={best['n_trades']}")
            print(f"\n  Diagnostic: Average MFE/MAE at different holds:")
            for h in [5, 10, 15, 30]:
                h_results = [r for r in results if r.get('hold_s') == h and
                            r.get('tp') == 99 and r.get('sl') == 99 and r['n_trades'] > 0]
                if h_results:
                    avg_per = np.mean([r['per_trade'] for r in h_results])
                    avg_wr = np.mean([r['win_rate'] for r in h_results])
                    print(f"    hold={h}s: avg {avg_per:+.3f}t/tr, WR={avg_wr:.1%}")

    # Top 20
    print(f"\nTOP 20 BY SHARPE:")
    for r in sorted(results, key=lambda x: -x.get('sharpe', -999))[:20]:
        perm = f" p={r['perm_p']}" if 'perm_p' in r else ""
        print(f"  {r['label']}: Sh={r['sharpe']:.2f} WR={r['win_rate']:.1%} "
              f"n={r['n_trades']} {r['per_trade']:+.4f}t/tr "
              f"G/R={r['green_days']}/{r['red_days']}{perm}")

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'v14_minimal_results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'generated': time.strftime('%Y-%m-%d %H:%M'),
            'elapsed_s': round(total_elapsed, 1),
            'n_configs': len(results),
            'n_validated': len(validated),
            'validated': validated,
            'all_results': results,
        }, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
