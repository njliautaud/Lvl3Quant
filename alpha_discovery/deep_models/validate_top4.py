#!/usr/bin/env python3
"""
Validate the top 4 strategy candidates on ALL available WF OOT dates.
Uses existing stacked prediction files + fill_sim_cli.
"""
import json, glob, os, sys, subprocess, logging, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

LVL3 = Path(__file__).resolve().parent.parent.parent
PRED_DIR = LVL3 / 'data' / 'processed' / 'cnn_wf_stacked_predictions'
MBO_DIR = LVL3 / 'data' / 'raw' / 'mbo'
BINARY = LVL3 / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
OUT_DIR = LVL3 / 'data' / 'processed' / 'top4_validation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger('top4')
log.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
log.addHandler(h)

# The 4 strategies with their exact prediction file patterns and fill_sim params
STRATEGIES = [
    {
        "name": "1_BookConfirmed",
        "pred_pattern": "book_predstdExit_conv2.5_vol70",
        "tp": 5, "sl": None,
        "hold_ms": 3600000, "threshold": 0.1,
        "chase_ticks": 1, "chase_reprices": 3,
    },
    {
        "name": "2_EMABookExit",
        "pred_pattern": "ema_bookExit_conv1.5_vol70",
        "tp": 10, "sl": 25,
        "hold_ms": 3600000, "threshold": 0.1,
        "chase_ticks": 1, "chase_reprices": 3,
    },
    {
        "name": "3_MomentumSniper",
        "pred_pattern": "mom_emaExit_conv0.3_ethr0.5_vol70",
        "tp": 15, "sl": None,
        "hold_ms": 3600000, "threshold": 0.1,
        "chase_ticks": 1, "chase_reprices": 3,
    },
    {
        "name": "4_SmoothTrailing",
        "pred_pattern": "smooth_smoothExit_conv1.5_ethr0.5_vol70",
        "tp": 20, "sl": 25,
        "hold_ms": 3600000, "threshold": 0.1,
        "chase_ticks": 1, "chase_reprices": 3,
    },
]


def run_sim(strategy, pred_file, mbo_file, date):
    """Run one fill_sim job."""
    tp_str = f"tp{strategy['tp']}" if strategy['tp'] else "tpN"
    sl_str = f"sl{strategy['sl']}" if strategy['sl'] else "slN"
    out_file = OUT_DIR / f"{strategy['name']}_{tp_str}_{sl_str}_{date}.json"

    if out_file.exists():
        return str(out_file), True  # already done

    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(out_file),
        '--hold-ms', str(strategy['hold_ms']),
        '--signal-threshold', str(strategy['threshold']),
        '--latency-ms', '0',
        '--quiet',
        '--chase-entry',
        '--chase-max-ticks', str(strategy['chase_ticks']),
        '--chase-max-reprices', str(strategy['chase_reprices']),
    ]

    if strategy['tp']:
        cmd.extend(['--take-profit-ticks', str(strategy['tp'])])
    if strategy['sl']:
        cmd.extend(['--trailing-ticks', str(strategy['sl'])])

    try:
        subprocess.run(cmd, timeout=300, capture_output=True)
        return str(out_file), out_file.exists()
    except Exception as e:
        return str(out_file), False


def main():
    log.info("=" * 60)
    log.info("TOP 4 STRATEGY VALIDATION")
    log.info("=" * 60)

    # Find all available dates with both predictions and MBO data
    jobs = []
    for strat in STRATEGIES:
        pred_files = sorted(PRED_DIR.glob(f"*_{strat['pred_pattern']}.npz"))
        for pf in pred_files:
            date = pf.stem[:10]
            nodash = date.replace('-', '')
            mbo = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
            if not mbo.exists():
                continue
            jobs.append((strat, str(pf), str(mbo), date))

    log.info(f"Total jobs: {len(jobs)} ({len(jobs)//4} dates × 4 strategies)")

    # Run with 8 workers (leave resources for other tasks)
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(run_sim, s, p, m, d): (s['name'], d) for s, p, m, d in jobs}
        for fut in as_completed(futures):
            name, date = futures[fut]
            out_file, success = fut.result()
            done += 1
            if not success:
                failed += 1
            if done % 20 == 0:
                log.info(f"  [{done}/{len(jobs)}] {done/len(jobs)*100:.0f}%, {failed} failed")

    log.info(f"\nDone: {done} jobs, {failed} failed")

    # Aggregate results per strategy
    log.info("\n" + "=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)

    for strat in STRATEGIES:
        tp_str = f"tp{strat['tp']}" if strat['tp'] else "tpN"
        sl_str = f"sl{strat['sl']}" if strat['sl'] else "slN"
        pattern = f"{strat['name']}_{tp_str}_{sl_str}_*.json"
        files = sorted(OUT_DIR.glob(pattern))

        total_pnl = 0
        total_trades = 0
        total_wins = 0
        total_signals = 0
        total_fills = 0
        daily_pnls = []
        days = 0

        for f in files:
            try:
                with open(f) as fh:
                    d = json.load(fh)
                pnl = d.get('total_pnl_dollars', 0)
                t = d.get('total_trades', 0)
                total_pnl += pnl
                total_trades += t
                daily_pnls.append(pnl)
                days += 1
                if t > 0:
                    total_wins += round(t * d.get('win_rate', 0))
                total_signals += d.get('total_signals', 0)
                total_fills += d.get('total_filled', 0)
            except:
                pass

        if days == 0:
            log.info(f"\n{strat['name']}: NO DATA")
            continue

        wr = total_wins / max(total_trades, 1) * 100
        fill_rate = total_fills / max(total_signals, 1) * 100
        mean_daily = total_pnl / days
        import numpy as np
        std_daily = np.std(daily_pnls) if daily_pnls else 0
        sharpe = mean_daily / max(std_daily, 1) * (252 ** 0.5)
        active_days = sum(1 for p in daily_pnls if p != 0 or True)
        profitable_days = sum(1 for p in daily_pnls if p > 0)

        # Max drawdown
        cum = 0
        peak = 0
        max_dd = 0
        for p in daily_pnls:
            cum += p
            if cum > peak: peak = cum
            dd = peak - cum
            if dd > max_dd: max_dd = dd

        log.info(f"\n{'='*50}")
        log.info(f"{strat['name']}")
        log.info(f"  Config: {strat['pred_pattern']} | TP={strat['tp']} SL={strat['sl']}")
        log.info(f"  Days: {days} | Active: {active_days} | Profitable: {profitable_days} ({profitable_days/days*100:.0f}%)")
        log.info(f"  Total P&L: ${total_pnl:,.0f} | $/day: ${mean_daily:,.0f} | Sharpe: {sharpe:.2f}")
        log.info(f"  Trades: {total_trades} | WR: {wr:.1f}% | Fill: {fill_rate:.1f}%")
        log.info(f"  Max DD: ${max_dd:,.0f} | Worst day: ${min(daily_pnls):,.0f} | Best day: ${max(daily_pnls):,.0f}")
        if total_trades > 0:
            log.info(f"  $/trade: ${total_pnl/total_trades:.1f} | Trades/day: {total_trades/days:.1f}")

    log.info(f"\nResults saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
