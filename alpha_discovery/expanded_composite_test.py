"""
Expanded Composite Test — Run 14-signal composite (9 original + 5 new) through MBO fill sim.

Original 9: causal_chain, cancel_asym_chain, ask_orders, depth_ratio_z,
            meta_global_3s/10s/30s, slow_decay_combo, volgated
New 5: stacked_depth_asym, dwfi_20, l1_ratio_shift_5, dwfi_5, ofi_l3_5

Generates composite predictions, uploads to Jupiter, runs fill_sim_cli.

Usage:
    python alpha_discovery/expanded_composite_test.py
"""

import sys
import time
import json
import logging
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))
sys.path.insert(0, str(Path('C:/Users/Footb/Documents/Github/teleclaude-main')))

# Logging
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
_log_file = RESULTS_DIR / f"expanded_composite_{_ts}.log"
_root = logging.getLogger()
_root.setLevel(logging.INFO)
for _h in _root.handlers[:]:
    _root.removeHandler(_h)
_fmt = logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S')
_fh = logging.FileHandler(str(_log_file), mode='w')
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
_root.addHandler(_fh)
_root.addHandler(_sh)
logger = logging.getLogger("expanded")

TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)

SIG_DIR = LVL3_ROOT / "data" / "processed" / "signal_predictions"

# Signal sets to test
ORIGINAL_9 = [
    'causal_chain', 'cancel_asym_chain', 'ask_orders', 'depth_ratio_z',
    'meta_global_3s', 'meta_global_10s', 'meta_global_30s',
    'slow_decay_combo', 'volgated',
]

NEW_5 = [
    'stacked_depth_asym', 'dwfi_20', 'l1_ratio_shift_5', 'dwfi_5', 'ofi_l3_5',
]

COMPOSITES = {
    'expanded_14sig': ORIGINAL_9 + NEW_5,
    'new_5only': NEW_5,
    'deep_ofi_3': ['dwfi_5', 'dwfi_20', 'ofi_l3_5'],  # Group R trio
    'original_9_plus_depth': ORIGINAL_9 + ['stacked_depth_asym', 'l1_ratio_shift_5'],  # add best 2 non-R
}

# MBO fill sim configs to test
CONFIGS = [
    # (name, threshold, hold_ms, trailing_ticks, extra_args)
    ('t3.5_h900s', 3.5, 900000, 0, ''),
    ('t3.5_h600s', 3.5, 600000, 0, ''),
    ('t3.5_h1200s', 3.5, 1200000, 0, ''),
    ('t3.0_h900s', 3.0, 900000, 0, ''),
    ('t4.0_h900s', 4.0, 900000, 0, ''),
]


def discover_days() -> List[str]:
    """Find dates where ALL signals have predictions."""
    dates = set()
    # Get dates from first signal
    for f in SIG_DIR.glob(f"{ORIGINAL_9[0]}_*.npz"):
        date = f.stem.replace(f"{ORIGINAL_9[0]}_", "")
        dates.add(date)
    return sorted(dates)


def generate_composite(date: str, signal_names: List[str]) -> np.ndarray:
    """Load signals for a date and compute mean z-score composite."""
    signals = []
    for sig_name in signal_names:
        fpath = SIG_DIR / f"{sig_name}_{date}.npz"
        if not fpath.exists():
            return None
        data = np.load(str(fpath))
        pred = data['predictions']
        signals.append(pred)

    # Find minimum length (they should all be same, but be safe)
    min_len = min(len(s) for s in signals)
    signals = [s[:min_len] for s in signals]

    # Mean z-score, then re-normalize to std=1.0
    # Averaging N z-scores compresses variance to ~1/sqrt(N).
    # The fill_sim threshold expects std~1.0, so we re-standardize.
    stacked = np.stack(signals)
    composite = np.mean(stacked, axis=0)
    std = np.std(composite)
    if std > 1e-8:
        composite = composite / std  # Now std=1.0, preserving signal direction
    return composite.astype(np.float32)


def upload_and_run(composite_name: str, predictions_by_date: Dict[str, np.ndarray]):
    """Upload predictions to Jupiter and run fill_sim_cli."""
    from utils.ssh_exec import connect_jupiter, sftp_upload
    import tempfile

    logger.info(f"  Setting up Jupiter directories...")

    # Create prediction directory on Jupiter
    remote_pred_dir = f"/home/jupiter/lvl3quant/data/predictions_{composite_name}"
    remote_result_dir = f"/home/jupiter/lvl3quant/results/{composite_name}"
    connect_jupiter(f"mkdir -p {remote_pred_dir} {remote_result_dir}", prefer='ethernet', timeout=10)

    # Upload predictions via SFTP
    logger.info(f"  Uploading {len(predictions_by_date)} prediction files...")
    tmp_dir = Path(tempfile.mkdtemp())
    uploaded = 0
    for date, pred in predictions_by_date.items():
        local_path = str(tmp_dir / f"composite_mean_{date}.npz")
        remote_path = f"{remote_pred_dir}/composite_mean_{date}.npz"
        np.savez_compressed(local_path, predictions=pred)
        res = sftp_upload(local_path, remote_path)
        if res['success']:
            uploaded += 1
        Path(local_path).unlink()
        if uploaded % 20 == 0:
            logger.info(f"    Uploaded {uploaded}/{len(predictions_by_date)}")
    # Clean up temp dir
    try:
        tmp_dir.rmdir()
    except Exception:
        pass
    logger.info(f"  Upload complete: {uploaded} files")

    # Get list of MBO files
    res = connect_jupiter('ls /home/jupiter/lvl3quant/data/mbo/ | sort', prefer='ethernet', timeout=10)
    if not res['success']:
        logger.error(f"  Failed to list MBO files: {res.get('error')}")
        return {}
    mbo_files = [f.strip() for f in res['stdout'].strip().split('\n') if f.strip()]
    logger.info(f"  Found {len(mbo_files)} MBO files on Jupiter")

    # Run fill_sim_cli for each config
    fill_sim = "/home/jupiter/lvl3quant/rust_cache_builder/target/release/fill_sim_cli"
    results = {}

    for cfg_name, threshold, hold_ms, trail, extra in CONFIGS:
        full_name = f"{composite_name}_{cfg_name}"
        logger.info(f"\n  Running config: {full_name}")

        day_pnls = []
        total_trades = 0
        total_wins = 0

        # Get list of available prediction files
        pred_list_res = connect_jupiter(f"ls {remote_pred_dir}/ 2>/dev/null", prefer='ethernet', timeout=10)
        available_pred_dates = set()
        if pred_list_res['success']:
            for fname in pred_list_res['stdout'].strip().split('\n'):
                # composite_mean_2025-08-11.npz -> 2025-08-11
                import re as re_mod
                m2 = re_mod.search(r'(\d{4}-\d{2}-\d{2})', fname)
                if m2:
                    available_pred_dates.add(m2.group(1))
        logger.info(f"    Available prediction dates: {len(available_pred_dates)}")

        for mbo_file in mbo_files:
            # MBO files: glbx-mdp3-20250714.mbo.dbn -> date = 2025-07-14
            import re
            m = re.search(r'(\d{4})(\d{2})(\d{2})', mbo_file)
            if not m:
                continue
            date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

            # Skip if no prediction for this date
            if date not in available_pred_dates:
                continue

            pred_file = f"{remote_pred_dir}/composite_mean_{date}.npz"
            result_file = f"{remote_result_dir}/{full_name}_{date}.json"

            cmd = (f"{fill_sim} "
                   f"--mbo-file /home/jupiter/lvl3quant/data/mbo/{mbo_file} "
                   f"--predictions {pred_file} "
                   f"--output {result_file} "
                   f"--signal-threshold {threshold} "
                   f"--hold-ms {hold_ms}")
            if trail > 0:
                cmd += f" --trailing-ticks {trail}"
            if extra:
                cmd += f" {extra}"

            res = connect_jupiter(cmd, prefer='ethernet', timeout=120)
            if res['success'] and res['exit_code'] == 0:
                # Read result
                cat_res = connect_jupiter(f"cat {result_file}", prefer='ethernet', timeout=30)
                if cat_res['success']:
                    try:
                        data = json.loads(cat_res['stdout'])
                        pnl = data.get('total_pnl_dollars', 0)
                        n_trades = data.get('total_trades', 0)
                        wr = data.get('win_rate', 0)
                        day_pnls.append(pnl)
                        total_trades += n_trades
                        total_wins += int(round(wr * n_trades))
                    except Exception as e:
                        logger.warning(f"      Parse error for {date}: {e}")
                        day_pnls.append(0)
                else:
                    day_pnls.append(0)
            else:
                day_pnls.append(0)

        # Aggregate
        total_pnl = sum(day_pnls)
        n_days = len(day_pnls)
        h1_pnl = sum(day_pnls[:n_days//2])
        h2_pnl = sum(day_pnls[n_days//2:])
        win_rate = total_wins / max(total_trades, 1)

        # Sharpe (daily)
        pnl_arr = np.array(day_pnls)
        if len(pnl_arr) > 1 and np.std(pnl_arr) > 0:
            sharpe = np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252)
        else:
            sharpe = 0.0

        both_plus = "BOTH+" if h1_pnl > 0 and h2_pnl > 0 else "FAIL"

        logger.info(f"    PnL: ${total_pnl:+,.0f}  H1=${h1_pnl:+,.0f}  H2=${h2_pnl:+,.0f}  "
                   f"Trades={total_trades}  WR={win_rate:.0%}  Sharpe={sharpe:.2f}  {both_plus}")

        results[full_name] = {
            'composite': composite_name,
            'config': cfg_name,
            'total_pnl': total_pnl,
            'h1_pnl': h1_pnl,
            'h2_pnl': h2_pnl,
            'n_days': n_days,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'both_plus': both_plus == "BOTH+",
            'per_day_pnl': day_pnls,
        }

    return results


def main():
    logger.info("Expanded Composite Test")
    logger.info(f"  Original signals: {ORIGINAL_9}")
    logger.info(f"  New signals: {NEW_5}")

    all_dates = discover_days()
    logger.info(f"  Available dates: {len(all_dates)}")

    all_results = {}

    for comp_name, signal_list in COMPOSITES.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"Composite: {comp_name} ({len(signal_list)} signals)")
        logger.info(f"  Signals: {signal_list}")
        logger.info(f"{'='*70}")

        # Generate predictions for each date
        predictions = {}
        skipped = 0
        for date in all_dates:
            pred = generate_composite(date, signal_list)
            if pred is not None:
                predictions[date] = pred
            else:
                skipped += 1

        logger.info(f"  Generated {len(predictions)} days, skipped {skipped}")

        if len(predictions) < 20:
            logger.warning(f"  Not enough days, skipping")
            continue

        # Upload and run on Jupiter
        results = upload_and_run(comp_name, predictions)
        all_results.update(results)

        # Clean up predictions dict
        del predictions

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Config':40s} {'PnL':>10s} {'H1':>10s} {'H2':>10s} {'Trades':>7s} {'WR':>5s} {'Sharpe':>7s} {'Both+':>6s}")
    logger.info("-" * 90)

    sorted_res = sorted(all_results.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
    for name, r in sorted_res:
        bp = "YES" if r['both_plus'] else "no"
        logger.info(f"{name:40s} ${r['total_pnl']:>+9,.0f} ${r['h1_pnl']:>+9,.0f} ${r['h2_pnl']:>+9,.0f} "
                   f"{r['total_trades']:>7d} {r['win_rate']:>4.0%} {r['sharpe']:>7.2f} {bp:>6s}")

    # Save
    out_file = RESULTS_DIR / f"expanded_composite_{_ts}.json"
    with open(str(out_file), 'w') as f:
        json.dump({
            'timestamp': _ts,
            'composites': {k: v for k, v in COMPOSITES.items()},
            'configs': [(n, t, h, tr, e) for n, t, h, tr, e in CONFIGS],
            'results': all_results,
        }, f, indent=2)
    logger.info(f"\nResults saved: {out_file}")


if __name__ == '__main__':
    main()
