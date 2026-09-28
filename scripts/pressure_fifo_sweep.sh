#!/bin/bash
# Pressure Exit FIFO Sweep: Uses fill_sim_cli's conviction-exit feature
# which exits when prediction flips opposite for N consecutive bars.
# This is the REAL FIFO-validated version of the pressure exit concept.
#
# Setup: buy-only afternoon (14:00-16:00 ET), TP 8, SL 16, 30min hold
# Sweep: conviction_bars × conviction_mag combinations

set -euo pipefail

FILL_SIM="/home/nick/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR="/home/nick/Lvl3Quant/data/raw/mbo"
PRED_DIR="/home/nick/Lvl3Quant/output/extended_oot_validation/pred_npzs"
OUT_BASE="/home/nick/Lvl3Quant/output/pressure_fifo_sweep"
mkdir -p "$OUT_BASE"

# Get all available prediction dates
DATES=$(ls "$PRED_DIR"/*_unfiltered.npz 2>/dev/null | sed 's/.*\///' | sed 's/_unfiltered.npz//' | sort)
echo "Found $(echo "$DATES" | wc -l) prediction dates"

# Configs to sweep
# conviction_bars: 0 (baseline), 20 (2s), 40 (4s), 80 (8s), 160 (16s)
# conviction_mag: 0.0 (any flip), 0.1, 0.3, 0.5
CONV_BARS="0 20 40 80 160"
CONV_MAGS="0.0 0.1 0.3 0.5"

# Also test signal thresholds
THRESHOLDS="0.3"  # same as extended OOT validation used

SUMMARY_FILE="$OUT_BASE/sweep_summary.json"
echo "[" > "$SUMMARY_FILE"
FIRST=true

for CONV_B in $CONV_BARS; do
    for CONV_M in $CONV_MAGS; do
        # Skip mag sweep when bars=0 (baseline — no conviction exit)
        if [ "$CONV_B" = "0" ] && [ "$CONV_M" != "0.0" ]; then
            continue
        fi

        CONFIG_NAME="conv${CONV_B}_mag${CONV_M}"
        CONFIG_DIR="$OUT_BASE/$CONFIG_NAME"
        mkdir -p "$CONFIG_DIR"

        echo "=== $CONFIG_NAME ==="

        TOTAL_PNL=0
        TOTAL_TRADES=0
        TOTAL_WINS=0
        GROSS_WIN=0
        GROSS_LOSS=0

        for DATE in $DATES; do
            MBO_FILE="$MBO_DIR/glbx-mdp3-${DATE}.mbo.dbn.zst"
            PRED_FILE="$PRED_DIR/${DATE}_unfiltered.npz"
            OUT_FILE="$CONFIG_DIR/${DATE}.json"

            if [ ! -f "$MBO_FILE" ]; then
                continue
            fi

            # Build fill_sim command
            CMD="$FILL_SIM \
                --mbo-file $MBO_FILE \
                --predictions $PRED_FILE \
                --output $OUT_FILE \
                --signal-threshold $THRESHOLDS \
                --hold-ms 1800000 \
                --stop-loss-ticks 16 \
                --take-profit-ticks 8 \
                --max-wait-bars 100 \
                --time-window-start 14:00 \
                --time-window-end 16:00 \
                --quiet"

            # Add conviction exit if bars > 0
            if [ "$CONV_B" != "0" ]; then
                CMD="$CMD --conviction-exit-bars $CONV_B --conviction-exit-mag $CONV_M"
            fi

            # Run
            eval $CMD 2>/dev/null || true
        done

        # Aggregate results
        python3 -c "
import json, glob, sys
import numpy as np

files = sorted(glob.glob('$CONFIG_DIR/*.json'))
total_pnl = 0
total_trades = 0
total_wins = 0
gross_win = 0
gross_loss = 0
daily_pnl = {}
exit_reasons = {}

for f in files:
    try:
        with open(f) as fh:
            d = json.load(fh)
    except:
        continue
    date = f.split('/')[-1].replace('.json','')
    trades = d.get('trades', [])
    day_pnl = 0
    for t in trades:
        pnl = t['pnl_ticks']
        total_pnl += pnl
        total_trades += 1
        day_pnl += pnl
        if pnl > 0:
            total_wins += 1
            gross_win += pnl
        else:
            gross_loss += abs(pnl)
        reason = t.get('exit_reason', 'Unknown')
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    daily_pnl[date] = day_pnl

pf = gross_win / gross_loss if gross_loss > 0 else 999
wr = total_wins / total_trades if total_trades > 0 else 0
daily_vals = list(daily_pnl.values())
sharpe = np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(252) if len(daily_vals) > 1 and np.std(daily_vals) > 0 else 0
neg = [v for v in daily_vals if v < 0]
sortino = np.mean(daily_vals) / np.std(neg) * np.sqrt(252) if neg and np.std(neg) > 0 else 0

march = sum(v for k,v in daily_pnl.items() if k.startswith('202603'))
april = sum(v for k,v in daily_pnl.items() if k.startswith('202604'))
green = sum(1 for v in daily_vals if v > 0)
red = sum(1 for v in daily_vals if v < 0)

result = {
    'config': '$CONFIG_NAME',
    'conv_bars': $CONV_B,
    'conv_mag': $CONV_M,
    'pf': round(pf, 4),
    'wr': round(wr, 4),
    'net_ticks': round(total_pnl, 1),
    'sharpe': round(sharpe, 2),
    'sortino': round(sortino, 2),
    'trades': total_trades,
    'green_days': green,
    'red_days': red,
    'march_ticks': round(march, 1),
    'april_ticks': round(april, 1),
    'exit_reasons': exit_reasons,
}
print(f'  PF={pf:.3f} WR={wr:.1%} Net={total_pnl:.0f}t Sharpe={sharpe:.1f} Trades={total_trades} Green={green}/Red={red} March={march:.0f}/April={april:.0f}')
print(f'  Exits: {exit_reasons}')

# Append to summary
with open('$SUMMARY_FILE', 'a') as sf:
    if not $FIRST:
        sf.write(',\n')
    json.dump(result, sf, indent=2)
" 2>&1 || echo "  ERROR in aggregation"

        FIRST=false
    done
done

echo "]" >> "$SUMMARY_FILE"
echo ""
echo "=== SWEEP COMPLETE ==="
echo "Summary: $SUMMARY_FILE"
