#!/bin/bash
# HC #270 — Phase 3 → Phase 4 auto-driver.
# Polls for Phase 2 completion (all enriched parquets present), then runs
# Phase 3 LGBM training (both, short-only, long-only) and Phase 4 validation.
# Posts each phase completion to Discord #general via the existing bot token
# (same mechanism as sweep_result_watcher.sh — wakes Claude through the bridge).
set -u
cd /home/jupiter/Lvl3Quant
mkdir -p logs

LABEL_DIR=output/meta_lgbm_labels
FEAT_DIR=output/meta_lgbm_features
LOG=logs/meta_lgbm_pipeline_driver.log
GENERAL_CHANNEL_ID="${DISCORD_GENERAL_CHANNEL_ID:-}"
TOKEN=$(python3 -c "import json; print(json.load(open('/home/jupiter/teleclaude-main/config.json'))['discordToken'])" 2>/dev/null)

post_to_discord() {
  local msg="$1"
  if [ -z "${TOKEN:-}" ]; then
    return 0
  fi
  local truncated
  truncated=$(printf "%s" "$msg" | head -c 1900)
  local payload
  payload=$(python3 -c "import json,sys; print(json.dumps({'content': sys.stdin.read()}))" <<< "$truncated")
  curl -sS -X POST \
    -H "Authorization: Bot $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "https://discord.com/api/v10/channels/${GENERAL_CHANNEL_ID}/messages" >/dev/null 2>&1
}

echo "[$(date +%H:%M:%S)] DRIVER START" >> $LOG

# ----- Wait for Phase 1+2 to complete -----
echo "[$(date +%H:%M:%S)] Waiting for 46 enriched parquets..." >> $LOG
LAST_REPORTED=0
while true; do
  N_ENRICHED=$(ls $FEAT_DIR/2026*_signals_enriched.parquet 2>/dev/null | wc -l)
  N_LABELED=$(ls $LABEL_DIR/2026*_signals_labeled.parquet 2>/dev/null | wc -l)
  if [ "$N_ENRICHED" -ge 46 ]; then
    echo "[$(date +%H:%M:%S)] Phase1+2 COMPLETE: $N_ENRICHED enriched parquets" >> $LOG
    break
  fi
  if ! pgrep -f "extract_fifo_labels_for_lgbm.py" >/dev/null; then
    if [ "$N_LABELED" -lt 46 ]; then
      echo "[$(date +%H:%M:%S)] Phase 1 PROCESS DIED with only $N_LABELED labeled — abort" >> $LOG
      post_to_discord "❌ meta-LGBM pipeline driver: Phase 1 process died at $N_LABELED/46 dates. Check logs/meta_lgbm_label_full_extraction.log"
      exit 1
    fi
  fi
  if [ "$N_ENRICHED" -gt "$LAST_REPORTED" ] && [ $((N_ENRICHED % 10)) -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] progress: $N_ENRICHED/46 enriched, $N_LABELED/46 labeled" >> $LOG
    post_to_discord "⏳ meta-LGBM Phase 1+2 progress: $N_ENRICHED/46 enriched"
    LAST_REPORTED=$N_ENRICHED
  fi
  sleep 30
done

post_to_discord "✅ meta-LGBM pipeline: **Phase 1+2 complete** (46/46 enriched parquets). Launching Phase 3 LGBM walk-forward training on **Neptune** (HC #263 — keep all nodes productive; HC #271(E))."

# ----- Phase 3: Train three LGBM gates ON NEPTUNE (HC #263, #271(E)) -----
# Strategy: rsync enriched parquets to Neptune, run train script over SSH, rsync results back.
NEPTUNE_HOST=nick@neptune
NEPTUNE_KEY=/home/jupiter/.ssh/id_ed25519
NEPTUNE_LVL3=/home/nick/Lvl3Quant
NEPTUNE_PY=$NEPTUNE_LVL3/venv_training/bin/python3

echo "[$(date +%H:%M:%S)] === Phase 3 prep: rsync enriched parquets to Neptune ===" >> $LOG
rsync -avz -e "ssh -i $NEPTUNE_KEY -o StrictHostKeyChecking=no" \
  $FEAT_DIR/2026*_signals_enriched.parquet \
  ${NEPTUNE_HOST}:${NEPTUNE_LVL3}/output/meta_lgbm_features/ \
  >> $LOG 2>&1
RC=$?
echo "[$(date +%H:%M:%S)] rsync features → Neptune exit=$RC" >> $LOG
if [ $RC -ne 0 ]; then
  post_to_discord "❌ meta-LGBM Phase 3 prep: rsync to Neptune FAILED exit=$RC — falling back to local training"
  NEPTUNE_OK=0
else
  NEPTUNE_OK=1
fi

# Sync the latest training script too
ssh -i $NEPTUNE_KEY -o StrictHostKeyChecking=no $NEPTUNE_HOST "mkdir -p $NEPTUNE_LVL3/scripts $NEPTUNE_LVL3/output/meta_lgbm_features $NEPTUNE_LVL3/logs" >/dev/null 2>&1
scp -i $NEPTUNE_KEY -o StrictHostKeyChecking=no scripts/train_meta_lgbm_gate.py $NEPTUNE_HOST:$NEPTUNE_LVL3/scripts/ >> $LOG 2>&1

for SIDE in both short long; do
  TAG=meta_lgbm_gate_v1_$SIDE
  OUTDIR=output/$TAG
  LOG3=logs/${TAG}_train.log
  echo "[$(date +%H:%M:%S)] === Phase 3: training $SIDE on Neptune → $OUTDIR ===" >> $LOG

  if [ $NEPTUNE_OK -eq 1 ]; then
    REMOTE_OUT=$NEPTUNE_LVL3/$OUTDIR
    ssh -i $NEPTUNE_KEY -o StrictHostKeyChecking=no $NEPTUNE_HOST \
      "mkdir -p $REMOTE_OUT && cd $NEPTUNE_LVL3 && \
       MLFLOW_TRACKING_URI=http://neptune-win:5000 \
       $NEPTUNE_PY -u scripts/train_meta_lgbm_gate.py \
         --out-dir $OUTDIR \
         --filter-direction $SIDE \
         --mlflow-experiment meta_lgbm_gate" \
      > $LOG3 2>&1
    EXIT=$?
    if [ $EXIT -eq 0 ]; then
      mkdir -p $OUTDIR
      rsync -avz -e "ssh -i $NEPTUNE_KEY -o StrictHostKeyChecking=no" \
        $NEPTUNE_HOST:$REMOTE_OUT/ $OUTDIR/ >> $LOG 2>&1
    fi
  else
    # Fallback: local training
    python3 -u scripts/train_meta_lgbm_gate.py \
      --out-dir $OUTDIR \
      --filter-direction $SIDE \
      --mlflow-experiment meta_lgbm_gate \
      > $LOG3 2>&1
    EXIT=$?
  fi

  if [ $EXIT -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] $SIDE training FAILED exit=$EXIT" >> $LOG
    post_to_discord "❌ meta-LGBM Phase 3 ($SIDE): training FAILED exit=$EXIT — check $LOG3"
    continue
  fi
  AUC=$(python3 -c "import json; d=json.load(open('$OUTDIR/concat_oot_metrics.json')); print(f\"{d['overall_auc']:.4f}\")" 2>/dev/null || echo "N/A")
  PCT=$(python3 -c "import json; d=json.load(open('$OUTDIR/concat_oot_metrics.json')); print(f\"{100*d['pct_folds_positive_top10']:.1f}%\")" 2>/dev/null || echo "N/A")
  echo "[$(date +%H:%M:%S)] $SIDE done: AUC=$AUC folds_top10_pos=$PCT" >> $LOG
  post_to_discord "📊 Phase 3 ($SIDE) on Neptune: AUC=$AUC, $PCT folds with positive top-10% NET. Artifacts pulled to $OUTDIR/"
done

# ----- Phase 4: Validate the short-only gate (HC #267 deploy alignment) -----
echo "[$(date +%H:%M:%S)] === Phase 4: short-side validation ===" >> $LOG
LOG4=logs/meta_lgbm_gate_v1_short_validate.log
python3 -u scripts/validate_meta_lgbm_gate.py \
  --gate-dir output/meta_lgbm_gate_v1_short \
  --side short \
  --thresholds "0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95" \
  --top-pcts "0.001,0.005,0.01,0.02,0.05,0.10" \
  > $LOG4 2>&1
EXIT=$?
if [ $EXIT -ne 0 ]; then
  post_to_discord "❌ meta-LGBM Phase 4 (short): validation FAILED exit=$EXIT — check $LOG4"
else
  if [ -f output/meta_lgbm_gate_v1_short/best_threshold.json ]; then
    BT=$(python3 -c "
import json
d=json.load(open('output/meta_lgbm_gate_v1_short/best_threshold.json'))
print(f\"thresh={d['threshold']:.3f} NET={d['sum_net_ticks']:+.1f}t trades={d['n_trades']} folds+={d['pct_folds_positive']:.1%} Sortino={d['sortino']:.3f} max_date_share={d['max_date_share_of_net']:.1%}\")
" 2>/dev/null)
    post_to_discord "🎯 **Phase 4 SHORT — BEST OPERATING POINT FOUND**: $BT — see output/meta_lgbm_gate_v1_short/meta_lgbm_gate_validation.csv for full sweep."
  else
    SUMMARY=$(tail -40 $LOG4 | grep -E "THRESHOLD SWEEP|threshold|NO THRESHOLD" | head -20 | tr '\n' '|')
    post_to_discord "⚠️ Phase 4 SHORT — no threshold passed HC #254 + #258 gates. Sweep tail: $SUMMARY"
  fi
fi

echo "[$(date +%H:%M:%S)] DRIVER END" >> $LOG
post_to_discord "🏁 meta-LGBM pipeline driver finished. Review $LOG and Phase 3/4 outputs for next decision."
