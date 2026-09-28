#!/usr/bin/env bash
# Run the full HC #413 / #415 / #411 gauntlet on a v3.4.2 full-OOT NPZ.
# Mirrors what agent a43dfaf1 did for the 5-date ep-1 NPZ at 03:48 ET on 5/18,
# but parameterised on NPZ path and output dir.
#
# Usage:
#   bash run_full_oot_gauntlet.sh <npz_path> <output_dir> [<label>]
#
# Example (inference #1, temporal-only):
#   bash run_full_oot_gauntlet.sh \
#     /home/jupiter/Lvl3Quant/output/hc417_v342_full_oot.npz \
#     /home/jupiter/Lvl3Quant/output/hc417_v342_full_oot_eval \
#     v342_full_oot_run1_temporalonly
#
# Example (inference #2, full book features):
#   bash run_full_oot_gauntlet.sh \
#     /home/jupiter/Lvl3Quant/output/hc417_v342_full_oot_v2.npz \
#     /home/jupiter/Lvl3Quant/output/hc417_v342_full_oot_v2_eval \
#     v342_full_oot_run2_fullbook
set -euo pipefail

NPZ="${1:?need NPZ path}"
OUTDIR="${2:?need output dir}"
LABEL="${3:-v342_full_oot}"

LVL3="/home/jupiter/Lvl3Quant"
PYTHON="/home/jupiter/miniconda3/envs/py311-train/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

echo "[$(date -Iseconds)] gauntlet start npz=$NPZ outdir=$OUTDIR label=$LABEL"
mkdir -p "$OUTDIR"

# 1) wrap NPZ (handles v3.4.2 multi-head schema -> per-sample dates + canonical column names)
echo "[$(date -Iseconds)] step 1/4: wrap NPZ"
"$PYTHON" "$LVL3/scripts/v3_4_research/hc417_v342_ep1/wrap_v342_ep1_npz.py" \
    --npz "$NPZ" \
    --output "$OUTDIR/wrapped.npz" 2>&1 | tee "$OUTDIR/01_wrap.log"

# 2) Concat IC across all heads
echo "[$(date -Iseconds)] step 2/4: concat IC summary"
"$PYTHON" "$LVL3/scripts/v3_4_research/hc417_v342_ep1/compute_ic.py" \
    --npz "$OUTDIR/wrapped.npz" \
    --output "$OUTDIR/ic_summary.csv" 2>&1 | tee "$OUTDIR/02_ic.log"

# 3) HC #413 scalping backtester (TP/SL with MFE config from hc411)
echo "[$(date -Iseconds)] step 3/4: HC #413 scalping backtester"
mkdir -p "$OUTDIR/hc413_backtest"
"$PYTHON" "$LVL3/scripts/hc413_scalping_backtester/backtester.py" \
    --npz "$OUTDIR/wrapped.npz" \
    --mfe-config "$LVL3/output/hc411_regime_agnostic_20260517_215211/mfe_at_confidence_matrix.csv" \
    --output-dir "$OUTDIR/hc413_backtest" \
    --model v3.4.2 \
    --order-type passive_at_touch 2>&1 | tee "$OUTDIR/03_hc413.log"

# 4) HC #411 sub-window stability (N=3, N=4, N=6 — full-OOT supports finer N now)
echo "[$(date -Iseconds)] step 4/4: HC #411 sub-window N=3,4,6"
"$PYTHON" "$LVL3/scripts/v3_4_research/hc417_v342_ep1/subwindow_stability.py" \
    --backtest-csv "$OUTDIR/hc413_backtest/scalping_backtest_results.csv" \
    --wrapped-npz "$OUTDIR/wrapped.npz" \
    --output-dir "$OUTDIR/hc411_subwindow" \
    --n-windows 3 4 6 2>&1 | tee "$OUTDIR/04_subwindow.log"

echo "[$(date -Iseconds)] gauntlet DONE — review $OUTDIR/"
ls -la "$OUTDIR/" | head -20
