#!/usr/bin/env python3
"""
Tick Replay v14 — Calibrated Thresholds + Multiple Signal Heads
================================================================

Fixes from v13 disaster (-34 ticks/trade):
  - Problem: threshold=0.3 on log_ret_1s means "expect 30% move" → only noise triggers
  - Fix: use QUANTILE thresholds (top N% of |signal|) instead of absolute
  - Also test execution-specific heads: pred_fifo_tp4sl3_net, pred_pred_mfe_30s_ticks

HC #659 compliant: tick-level replay, permutation test on any profitable config.

Author: Claude
Date: 2026-07-10
"""

import sys, os, time, json
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/jupiter/Lvl3Quant/engines')
from tick_replay_engine import TickReplayEngine, compute_metrics, Trade

# =============================================================================
# Config
# =============================================================================

MBO_DIR = '/home/jupiter/Lvl3Quant/data/raw/mbo'
PRED_DIR = '/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/tick_replay_v14_calibrated'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Signal heads to test
SIGNAL_HEADS = [
    'pred_log_ret_1s',        # Primary 1s signal (IC=0.22)
    'pred_fifo_tp4sl3_net',   # Direct FIFO execution prediction
    'pred_pred_mfe_30s_ticks', # Max favorable excursion prediction
]

# Quantile thresholds (fraction of signals to trade)
QUANTILE_CONFIGS = [0.01, 0.03, 0.05, 0.10, 0.20]  # top 1%, 3%, 5%, 10%, 20%

# Hold times to test
HOLD_CONFIGS = [3, 5, 10, 15, 30]  # seconds

# TP/SL configs
TPSL_CONFIGS = [
    (2, 3),   # tight TP, moderate SL
    (3, 3),   # symmetric 3
    (4, 3),   # asymmetric favorable
    (99, 99), # no TP/SL (pure time stop)
    (3, 5),   # wide SL
    (5, 3),   # wide TP
]

N_PERMUTATIONS = 100  # For permutation test
MAX_DAYS = 15  # Use more days for reliability, but not all 34 to save time

# =============================================================================
# Load predictions with multiple heads
# =============================================================================

def load_multi_head_predictions(pred_dir):
    """Load predictions with all signal heads."""
    import glob
    preds = {}  # date -> {head_name: array}

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


def find_mbo_files(mbo_dir):
    """Find all MBO files."""
    import glob
    return sorted(glob.glob(os.path.join(mbo_dir, 'glbx-mdp3-*.mbo.dbn.zst')))


def compute_quantile_threshold(preds_dict, head, quantile_frac):
    """
    Compute the absolute threshold that selects the top `quantile_frac`
    fraction of signals across ALL days.
    """
    all_vals = []
    for date_key, heads in preds_dict.items():
        if head in heads:
            all_vals.append(np.abs(heads[head]))

    if not all_vals:
        return 999.0

    combined = np.concatenate(all_vals)
    # We want the top quantile_frac, so threshold is at (1 - quantile_frac) percentile
    threshold = np.percentile(combined, (1 - quantile_frac) * 100)
    return float(threshold)


def run_single_config(mbo_files, preds_dict, head, threshold, tp, sl, hold_s,
                      cancel_s=15.0, max_days=None):
    """Run a single config across all matching days. Returns list of trades + per-day pnls."""
    matched = []
    for mbo_path in mbo_files:
        date8 = os.path.basename(mbo_path).split('-')[2].split('.')[0]
        if date8 in preds_dict and head in preds_dict[date8]:
            matched.append((mbo_path, date8))

    if max_days and len(matched) > max_days:
        # Spread evenly across available days
        indices = np.linspace(0, len(matched)-1, max_days, dtype=int)
        matched = [matched[i] for i in indices]

    all_trades = []
    day_pnls = []

    for mbo_path, date_key in matched:
        engine = TickReplayEngine(
            tp_ticks=tp, sl_ticks=sl,
            hold_seconds=hold_s,
            signal_threshold=threshold,
            cancel_seconds=cancel_s,
        )
        # Override the engine to use the right prediction head
        trades = engine.run_day(mbo_path, preds_dict[date_key][head])
        all_trades.extend(trades)
        day_pnl = sum(t.pnl_ticks for t in trades)
        day_pnls.append(day_pnl)

    return all_trades, day_pnls, len(matched)


def run_permutation_test(mbo_files, preds_dict, head, threshold, tp, sl, hold_s,
                         cancel_s, n_perms, max_days, real_sharpe):
    """
    Permutation test: shuffle signal SIGNS randomly, re-run, check if random
    achieves similar Sharpe. p-value = fraction of random runs >= real Sharpe.
    """
    random_sharpes = []

    matched = []
    for mbo_path in mbo_files:
        date8 = os.path.basename(mbo_path).split('-')[2].split('.')[0]
        if date8 in preds_dict and head in preds_dict[date8]:
            matched.append((mbo_path, date8))

    if max_days and len(matched) > max_days:
        indices = np.linspace(0, len(matched)-1, max_days, dtype=int)
        matched = [matched[i] for i in indices]

    for perm_i in range(n_perms):
        perm_trades = []
        for mbo_path, date_key in matched:
            preds = preds_dict[date_key][head].copy()
            # Randomly flip signs
            signs = np.random.choice([-1, 1], size=len(preds))
            preds_shuffled = np.abs(preds) * signs

            engine = TickReplayEngine(
                tp_ticks=tp, sl_ticks=sl,
                hold_seconds=hold_s,
                signal_threshold=threshold,
                cancel_seconds=cancel_s,
            )
            trades = engine.run_day(mbo_path, preds_shuffled)
            perm_trades.extend(trades)

        if perm_trades:
            m = compute_metrics(perm_trades, f'perm_{perm_i}')
            random_sharpes.append(m.get('sharpe', 0))
        else:
            random_sharpes.append(0)

    p_value = np.mean([s >= real_sharpe for s in random_sharpes])
    return p_value, random_sharpes


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("TICK REPLAY v14 — CALIBRATED THRESHOLDS")
    print("=" * 70)

    # Load data
    print("\nLoading predictions...")
    preds_dict = load_multi_head_predictions(PRED_DIR)
    print(f"  {len(preds_dict)} dates with predictions")

    mbo_files = find_mbo_files(MBO_DIR)
    print(f"  {len(mbo_files)} MBO files found")

    # Match days
    matched_dates = []
    for mbo_path in mbo_files:
        date8 = os.path.basename(mbo_path).split('-')[2].split('.')[0]
        if date8 in preds_dict:
            matched_dates.append(date8)
    print(f"  {len(matched_dates)} matching days")

    # Phase 1: Quick screening with log_ret_1s across quantiles and holds
    # Use fewer days for screening
    screen_days = min(8, MAX_DAYS)

    results = []
    best_configs = []

    for head in SIGNAL_HEADS:
        # Check if this head exists in predictions
        has_head = any(head in preds_dict[d] for d in preds_dict)
        if not has_head:
            print(f"\n  Skipping {head} — not in predictions")
            continue

        print(f"\n{'='*50}")
        print(f"SIGNAL HEAD: {head}")
        print(f"{'='*50}")

        # Compute quantile thresholds
        for q_frac in QUANTILE_CONFIGS:
            threshold = compute_quantile_threshold(preds_dict, head, q_frac)
            print(f"\n  Quantile top-{q_frac*100:.0f}%: threshold = {threshold:.6f}")

            for hold_s in [5, 10, 15]:  # Start with 3 hold times for screening
                for tp, sl in [(4, 3), (99, 99), (3, 5)]:  # 3 TP/SL for screening
                    label = f"{head.replace('pred_', '')}|q{q_frac*100:.0f}|h{hold_s}|tp{tp}sl{sl}"

                    trades, day_pnls, n_days = run_single_config(
                        mbo_files, preds_dict, head, threshold,
                        tp, sl, hold_s, cancel_s=15.0, max_days=screen_days
                    )

                    if not trades:
                        print(f"    {label}: NO TRADES")
                        continue

                    m = compute_metrics(trades, label)
                    net = m.get('net_pnl_ticks', 0)
                    wr = m.get('win_rate', 0)
                    n = m.get('n_trades', 0)
                    sharpe = m.get('sharpe', 0)
                    green = sum(1 for p in day_pnls if p > 0)
                    red = sum(1 for p in day_pnls if p < 0)
                    tpd = n / max(n_days, 1)
                    sign = '+' if net > 0 else ''

                    result = {
                        'label': label, 'head': head, 'quantile': q_frac,
                        'threshold': threshold, 'hold_s': hold_s,
                        'tp': tp, 'sl': sl,
                        'n_trades': n, 'trades_per_day': tpd,
                        'net_ticks': net, 'per_trade': net / max(n, 1),
                        'win_rate': wr, 'sharpe': sharpe,
                        'green_days': green, 'red_days': red,
                        'n_days': n_days,
                        'exit_reasons': m.get('exit_reasons', {}),
                    }
                    results.append(result)

                    print(f"    {label}: {sign}{net:.0f}t ({sign}{net/max(n,1):.3f}t/tr) "
                          f"n={n} ({tpd:.0f}/d) WR={wr:.1%} Sh={sharpe:.2f} "
                          f"G/R={green}/{red}")

                    # Track promising configs (positive Sharpe)
                    if sharpe > 0.5 and n >= 20:
                        best_configs.append(result)

    # Phase 2: Run permutation test on best configs
    print(f"\n{'='*70}")
    print(f"PHASE 2: PERMUTATION TESTS ON {len(best_configs)} PROMISING CONFIGS")
    print(f"{'='*70}")

    validated_configs = []

    for cfg in sorted(best_configs, key=lambda x: -x['sharpe'])[:10]:  # Top 10 only
        print(f"\n  Testing {cfg['label']}  (Sharpe={cfg['sharpe']:.2f}, n={cfg['n_trades']})")

        p_val, rand_sharpes = run_permutation_test(
            mbo_files, preds_dict, cfg['head'], cfg['threshold'],
            cfg['tp'], cfg['sl'], cfg['hold_s'],
            cancel_s=15.0, n_perms=N_PERMUTATIONS,
            max_days=screen_days, real_sharpe=cfg['sharpe']
        )

        cfg['perm_p_value'] = p_val
        cfg['rand_sharpe_mean'] = np.mean(rand_sharpes)
        cfg['rand_sharpe_std'] = np.std(rand_sharpes)

        status = "✅ PASS" if p_val < 0.05 else "❌ FAIL"
        print(f"    {status}: p={p_val:.3f} (random Sharpe: "
              f"{cfg['rand_sharpe_mean']:.2f} ± {cfg['rand_sharpe_std']:.2f})")

        if p_val < 0.05:
            validated_configs.append(cfg)

    # Phase 3: Expand validated configs to all days
    if validated_configs:
        print(f"\n{'='*70}")
        print(f"PHASE 3: FULL-DAY VALIDATION ON {len(validated_configs)} PASSING CONFIGS")
        print(f"{'='*70}")

        for cfg in validated_configs:
            trades, day_pnls, n_days = run_single_config(
                mbo_files, preds_dict, cfg['head'], cfg['threshold'],
                cfg['tp'], cfg['sl'], cfg['hold_s'],
                cancel_s=15.0, max_days=None  # ALL days
            )

            if not trades:
                continue

            m = compute_metrics(trades, cfg['label'])
            net = m.get('net_pnl_ticks', 0)
            sharpe = m.get('sharpe', 0)
            wr = m.get('win_rate', 0)
            n = len(trades)
            green = sum(1 for p in day_pnls if p > 0)
            red = sum(1 for p in day_pnls if p < 0)

            # Run permutation on full data
            p_val, _ = run_permutation_test(
                mbo_files, preds_dict, cfg['head'], cfg['threshold'],
                cfg['tp'], cfg['sl'], cfg['hold_s'],
                cancel_s=15.0, n_perms=N_PERMUTATIONS,
                max_days=None, real_sharpe=sharpe
            )

            cfg['full_n_days'] = n_days
            cfg['full_n_trades'] = n
            cfg['full_sharpe'] = sharpe
            cfg['full_wr'] = wr
            cfg['full_net_ticks'] = net
            cfg['full_perm_p'] = p_val
            cfg['full_green'] = green
            cfg['full_red'] = red

            sign = '+' if net > 0 else ''
            status = "✅" if p_val < 0.05 and sharpe > 0 else "❌"
            print(f"  {status} {cfg['label']}: {sign}{net:.0f}t Sh={sharpe:.2f} "
                  f"WR={wr:.1%} n={n} ({n/max(n_days,1):.0f}/d) "
                  f"G/R={green}/{red} perm_p={p_val:.3f} [{n_days}d]")

    # Save results
    elapsed = time.time() - t0
    output = {
        'generated': time.strftime('%Y-%m-%d %H:%M'),
        'elapsed_seconds': elapsed,
        'screening_results': len(results),
        'promising_configs': len(best_configs),
        'permutation_passing': len(validated_configs),
        'all_results': results,
        'validated': validated_configs,
    }

    output_path = os.path.join(OUTPUT_DIR, 'v14_calibrated_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Screened: {len(results)} configs")
    print(f"Promising (Sharpe>0.5): {len(best_configs)}")
    print(f"Permutation-validated: {len(validated_configs)}")
    print(f"Results saved to {output_path}")

    if not validated_configs:
        print("\n⚠️  NO CONFIGS PASSED PERMUTATION TEST.")
        print("   This means v3.4.2 predictions cannot generate alpha via")
        print("   passive limit scalping at these horizons. The signal may")
        print("   need a different execution approach (longer hold, confluence, etc).")


if __name__ == '__main__':
    main()
