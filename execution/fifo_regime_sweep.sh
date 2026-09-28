#!/bin/bash
# FIFO Regime Sweep — Jupiter CPU
# Runs fill_sim_cli across all 10 OOT dates with multiple config regimes
# This creates GROUND TRUTH profitability data per regime
#
# Configs tested:
#   1. Tight_TP:  TP=4, SL=3, hold=10s  (scalping, tight spread regime)
#   2. Medium_TP: TP=6, SL=5, hold=30s  (balanced)
#   3. Wide_TP:   TP=10, SL=8, hold=60s (wider, higher conviction only)
#   4. High_conv: threshold=0.02, TP=6, SL=4, hold=20s (high selectivity)
#   5. Passive:   TP=4, SL=3, hold=10s, max_wait=50 (patient limit)
#   6. Aggressive: TP=8, SL=5, hold=30s, max_wait=10 (chase entry)

set -e

FILL_SIM="/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR="/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR="/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar"
OUT_DIR="/home/jupiter/Lvl3Quant/output/fifo_regime_sweep"
mkdir -p "$OUT_DIR"

DATES=(20260223 20260224 20260225 20260226 20260227 20260302 20260303 20260304 20260305)
# Skip 20260301 (only 758 predictions, too small)

# Config definitions: name|threshold|tp|sl|hold_ms|max_wait|trailing
CONFIGS=(
    "tight_scalp|0.01|4|3|10000|30|0"
    "medium_balanced|0.01|6|5|30000|30|0"
    "wide_conviction|0.015|10|8|60000|30|0"
    "high_selectivity|0.02|6|4|20000|30|0"
    "passive_patient|0.01|4|3|10000|50|0"
    "aggressive_chase|0.01|8|5|30000|10|0"
    "trailing_4|0.01|8|5|30000|30|4"
    "tight_trailing|0.01|6|3|15000|30|3"
)

echo "═══ FIFO Regime Sweep — $(date) ═══"
echo "  Dates: ${#DATES[@]}"
echo "  Configs: ${#CONFIGS[@]}"
echo "  Total runs: $((${#DATES[@]} * ${#CONFIGS[@]}))"
echo ""

LOG="$OUT_DIR/sweep.log"
echo "Starting sweep at $(date)" > "$LOG"

# Run all configs for all dates
TOTAL=0
SUCCESS=0
FAIL=0

for config_str in "${CONFIGS[@]}"; do
    IFS='|' read -r name threshold tp sl hold_ms max_wait trailing <<< "$config_str"
    echo "Config: $name (thresh=$threshold TP=$tp SL=$sl hold=${hold_ms}ms)"

    for date in "${DATES[@]}"; do
        FOLD_IDX=-1
        # Map date to fold
        case $date in
            20260223) FOLD_IDX=0 ;;
            20260224) FOLD_IDX=1 ;;
            20260225) FOLD_IDX=2 ;;
            20260226) FOLD_IDX=3 ;;
            20260227) FOLD_IDX=4 ;;
            20260302) FOLD_IDX=6 ;;
            20260303) FOLD_IDX=7 ;;
            20260304) FOLD_IDX=8 ;;
            20260305) FOLD_IDX=9 ;;
        esac

        MBO_FILE="$MBO_DIR/glbx-mdp3-${date}.mbo.dbn.zst"
        PRED_FILE="$PRED_DIR/fold_$(printf '%02d' $FOLD_IDX)_oot_predictions.npz"
        OUT_FILE="$OUT_DIR/${name}_${date}.json"

        if [ ! -f "$MBO_FILE" ]; then
            echo "  ❌ $date: MBO file missing" | tee -a "$LOG"
            FAIL=$((FAIL+1))
            continue
        fi

        if [ ! -f "$PRED_FILE" ]; then
            echo "  ❌ $date: prediction file missing (fold $FOLD_IDX)" | tee -a "$LOG"
            FAIL=$((FAIL+1))
            continue
        fi

        if [ -f "$OUT_FILE" ]; then
            echo "  ⏭️  $date/$name: already done"
            SUCCESS=$((SUCCESS+1))
            TOTAL=$((TOTAL+1))
            continue
        fi

        TRAILING_ARG=""
        if [ "$trailing" != "0" ]; then
            TRAILING_ARG="--trailing-ticks $trailing"
        fi

        echo -n "  $date: "
        if $FILL_SIM \
            --mbo-file "$MBO_FILE" \
            --predictions "$PRED_FILE" \
            --output "$OUT_FILE" \
            --signal-threshold "$threshold" \
            --take-profit-ticks "$tp" \
            --stop-loss-ticks "$sl" \
            --hold-ms "$hold_ms" \
            --max-wait-bars "$max_wait" \
            $TRAILING_ARG \
            --quiet 2>>"$LOG"; then
            # Extract key metrics from output
            if [ -f "$OUT_FILE" ]; then
                trades=$(python3 -c "import json; d=json.load(open('$OUT_FILE')); print(d.get('total_trades', d.get('n_trades', '?')))" 2>/dev/null || echo "?")
                pnl=$(python3 -c "import json; d=json.load(open('$OUT_FILE')); print(f\"{d.get('net_pnl_ticks', d.get('pnl_ticks', 0)):.1f}t\")" 2>/dev/null || echo "?")
                echo "✅ trades=$trades pnl=$pnl"
                SUCCESS=$((SUCCESS+1))
            else
                echo "❌ no output"
                FAIL=$((FAIL+1))
            fi
        else
            echo "❌ fill_sim failed"
            FAIL=$((FAIL+1))
        fi

        TOTAL=$((TOTAL+1))
    done
    echo ""
done

echo "═══ SWEEP COMPLETE ═══"
echo "  Total: $TOTAL | Success: $SUCCESS | Failed: $FAIL"
echo "  Results in: $OUT_DIR"
echo "Completed at $(date)" >> "$LOG"
