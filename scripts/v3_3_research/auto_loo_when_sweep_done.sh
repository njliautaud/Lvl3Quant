#!/bin/bash
# auto_loo_when_sweep_done.sh
# Polls v3.4.2 sweep dir; when it reaches 3000 trials in study.db, runs LOO validation.
# Posts result location to Discord (best-effort).
# Used overnight 2026-05-18→05-19 so the morning has v3.4.2 LOO-robust configs ready.

set -e
PROJ=/home/jupiter/Lvl3Quant
SWEEP_DIR=$PROJ/output/v342_execution_optuna_20260518
PREDS=$PROJ/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_predictions.npz
LOG=$SWEEP_DIR/auto_loo.log
TARGET_TRIALS=3000

mkdir -p "$SWEEP_DIR"

while true; do
  if [ -f "$SWEEP_DIR/study.db" ]; then
    n=$(sqlite3 "$SWEEP_DIR/study.db" "SELECT COUNT(*) FROM trials WHERE state='COMPLETE'" 2>/dev/null || echo 0)
  else
    n=0
  fi
  ts=$(date -Iseconds)
  echo "[$ts] sweep trials_complete=$n / $TARGET_TRIALS" >> "$LOG"
  if [ "$n" -ge "$TARGET_TRIALS" ]; then
    echo "[$ts] SWEEP COMPLETE — running LOO validation" >> "$LOG"
    if [ ! -f "$SWEEP_DIR/best_configs.json" ]; then
      # Some sweeps auto-emit best_configs at end. If missing, run a finalizer.
      cd "$PROJ"
      python3 - <<PY >> "$LOG" 2>&1
import json, sqlite3
from pathlib import Path
sd = Path("$SWEEP_DIR")
db = sd / "study.db"
con = sqlite3.connect(str(db))
cur = con.cursor()
# Read top-K by Optuna value (assumed Sharpe in single-objective).
cur.execute("SELECT trial_id FROM trials WHERE state='COMPLETE'")
trials = cur.fetchall()
# Pull params + user_attrs (metrics) per trial. progress.jsonl is the canonical source.
records = []
pj = sd / "progress.jsonl"
if pj.exists():
    for line in pj.read_text().splitlines():
        try:
            r = json.loads(line)
            if r.get("hc344_pass") and isinstance(r.get("metrics"), dict):
                records.append(r)
        except Exception:
            continue
records.sort(key=lambda r: r["metrics"].get("sharpe", 0), reverse=True)
top = records[:30]
out = [{"trial": r["trial"], "params": r["params"], "metrics": r["metrics"]} for r in top]
(sd / "best_configs.json").write_text(json.dumps(out, indent=2))
print(f"[finalize] wrote {len(out)} configs to best_configs.json")
PY
    fi
    cd "$PROJ"
    python3 scripts/v3_3_research/oot_loo_validate_top_configs.py \
      --sweep-dir "$SWEEP_DIR" \
      --preds "$PREDS" \
      --top-k 30 >> "$LOG" 2>&1
    echo "[$ts] LOO complete — see $SWEEP_DIR/loo_robust_configs.json" >> "$LOG"
    break
  fi
  sleep 600  # 10 min poll
done
