#!/usr/bin/env bash
# HC #432 — Auto-rerun trigger.
#
# Polls Neptune every 10 min for the per-date NPZ count. Once the count hits
# TARGET_N (default 47), runs the full pipeline (concat → FIFO → validate) and
# sends a plain-English Discord summary (HC #433). Idempotent — fires once.
#
# Run:   nohup bash scripts/v3_4_research/hc432_autorun_when_ready.sh > \
#        output/hc432_v342_47day_validation/autorun.log 2>&1 &
#
# Stops on: ready run completed, max-poll exhausted (default 18h), kill signal.

set -uo pipefail
LVL3="/home/jupiter/Lvl3Quant"
OUT="$LVL3/output/hc432_v342_47day_validation"
NEPTUNE_HOST="${NEPTUNE_HOST:-nick@neptune}"
NEPTUNE_DIR="${NEPTUNE_DIR:-/home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate}"
TARGET_N="${TARGET_N:-47}"
POLL_S="${POLL_S:-600}"     # 10 min
MAX_POLLS="${MAX_POLLS:-108}"  # 18h
DONE_FLAG="$OUT/.autorun_done.flag"
LOCK="$OUT/.autorun.lock"
LOG="$OUT/autorun.log"

mkdir -p "$OUT"

if [[ -f "$DONE_FLAG" ]]; then
  echo "[$(date -Iseconds)] autorun already completed (flag present). Exiting." >> "$LOG"
  exit 0
fi
if [[ -f "$LOCK" ]]; then
  pid=$(cat "$LOCK" 2>/dev/null || echo "")
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[$(date -Iseconds)] another autorun pid=$pid is active; exiting." >> "$LOG"
    exit 0
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

CFG_NAME="v342_long_1s_top0.5"
CFG_FILLS="$OUT/${CFG_NAME}_tp1.0_sl0.5_h1.5_c1.0_passive_at_touch_fifo_fills.csv"

# HC #432 extended candidate list (all 4 FIFO-untested per task spec).
# Format: cfg_name|horizon|side|conf_band|tp|sl|hold_s|cancel_s|order_type
EXTENDED_CFGS=(
  "v342_long_1s_top0.5|1|long|top0.5|1|0.5|1.5|1|passive_at_touch"
  "v342_short_5s_top0.5_t1422_R2fix|5|short|top0.5|1|1|1.1|5|passive_at_touch"
  "v342_short_10s_top0.5_t2831_R2fix|10|short|top0.5|2|1.5|2.4|10|passive_at_touch"
  "v342_long_5s_top0.5_for_ensemble|5|long|top0.5|1|0.5|1.5|5|passive_at_touch"
)
ENSEMBLE_LONG_CFG="v342_long_5s_top0.5_for_ensemble"
ENSEMBLE_SHORT_CFG="v342_short_5s_top0.5_t1422_R2fix"
ENSEMBLE_NAME="v342_lshort_5s_ensemble_50_50"
# v2 sanity baseline (lives in separate v2 harness — see hc432_v2_baseline_runner.py)
V2_BASELINE_CFG="v2_short_1s_top0.5_baseline"

run_one_config() {
  # args: cfg_name horizon side conf_band tp sl hold cancel order_type
  local cfg=$1 horizon=$2 side=$3 conf=$4 tp=$5 sl=$6 hold=$7 cancel=$8 order=$9
  echo "[$(date -Iseconds)] [$cfg] FIFO replay..." >> "$LOG"
  python3 "$LVL3/scripts/v3_4_research/hc432_fifo_full_market_replay.py" \
      --horizon "$horizon" --side "$side" --conf-band "$conf" \
      --tp-ticks "$tp" --sl-ticks "$sl" --hold-s "$hold" --cancel-s "$cancel" \
      --order-type "$order" --workers 12 \
      --config-name "$cfg" >> "$LOG" 2>&1 || {
    echo "[$(date -Iseconds)] [$cfg] FIFO replay failed" >> "$LOG"
    return 1
  }
  local fills_csv="$OUT/${cfg}_fifo_fills.csv"
  if [[ ! -s "$fills_csv" ]]; then
    echo "[$(date -Iseconds)] [$cfg] no/empty fills CSV: $fills_csv" >> "$LOG"
    return 1
  fi
  python3 "$LVL3/scripts/v3_4_research/hc432_validate_full.py" \
      --fills-csv "$fills_csv" \
      --config-name "$cfg" \
      --horizon "$horizon" --side "$side" --conf-band "$conf" \
      --tp-ticks "$tp" --sl-ticks "$sl" --hold-s "$hold" --cancel-s "$cancel" \
      --order-type "$order" \
      --total-oot-dates "$TARGET_N" >> "$LOG" 2>&1 || {
    echo "[$(date -Iseconds)] [$cfg] validation failed" >> "$LOG"
    return 1
  }
  return 0
}

run_ensemble() {
  # Merge long + short fills 50/50 and validate
  echo "[$(date -Iseconds)] [$ENSEMBLE_NAME] building 50/50 ensemble..." >> "$LOG"
  local long_csv="$OUT/${ENSEMBLE_LONG_CFG}_fifo_fills.csv"
  local short_csv="$OUT/${ENSEMBLE_SHORT_CFG}_fifo_fills.csv"
  local merged_csv="$OUT/${ENSEMBLE_NAME}_fifo_fills.csv"
  python3 "$LVL3/scripts/v3_4_research/hc432_ensemble_fills.py" \
      --long-csv "$long_csv" --short-csv "$short_csv" \
      --weight 0.5 --out-csv "$merged_csv" >> "$LOG" 2>&1 || {
    echo "[$(date -Iseconds)] [$ENSEMBLE_NAME] merge failed" >> "$LOG"
    return 1
  }
  # Validate the merged ensemble. Use horizon=5 (both legs at 5s).
  # R2 checks use the more restrictive (short) leg's TP/SL/hold/cancel since
  # both legs share horizon. We pick TP=1, SL=1 (short leg), hold=1.5, cancel=5
  # since validator only uses these for R2 sanity checks vs MFE-p90.
  python3 "$LVL3/scripts/v3_4_research/hc432_validate_full.py" \
      --fills-csv "$merged_csv" \
      --config-name "$ENSEMBLE_NAME" \
      --horizon 5 --side long --conf-band top0.5 \
      --tp-ticks 1 --sl-ticks 1 --hold-s 1.5 --cancel-s 5 \
      --order-type passive_at_touch \
      --total-oot-dates "$TARGET_N" >> "$LOG" 2>&1 || {
    echo "[$(date -Iseconds)] [$ENSEMBLE_NAME] validation failed" >> "$LOG"
    return 1
  }
  return 0
}

run_v2_baseline() {
  # v2 sanity baseline (HC #413 expected ~+0.274 tk/fill). Runs against the v2
  # per-date predictions in cnn_mamba_v2_all_oot/. Best-effort — does not block
  # the main pipeline if missing.
  local runner="$LVL3/scripts/v3_4_research/hc432_v2_baseline_runner.py"
  if [[ ! -f "$runner" ]]; then
    echo "[$(date -Iseconds)] [$V2_BASELINE_CFG] runner not present — skipping (gap documented in final verdict)" >> "$LOG"
    return 0
  fi
  echo "[$(date -Iseconds)] [$V2_BASELINE_CFG] running v2 sanity baseline..." >> "$LOG"
  python3 "$runner" \
      --horizon 1 --side short --conf-band top0.5 \
      --tp-ticks 1 --sl-ticks 0.5 --hold-s 1.5 --cancel-s 1 \
      --order-type passive_at_touch \
      --config-name "$V2_BASELINE_CFG" >> "$LOG" 2>&1 || {
    echo "[$(date -Iseconds)] [$V2_BASELINE_CFG] v2 baseline failed — non-fatal" >> "$LOG"
    return 0  # do not abort main pipeline
  }
  return 0
}

build_final_verdict() {
  python3 "$LVL3/scripts/v3_4_research/hc432_build_final_verdict.py" \
      --output-dir "$OUT" >> "$LOG" 2>&1 || {
    echo "[$(date -Iseconds)] final verdict build failed" >> "$LOG"
    return 1
  }
  # HC #436 — also publish under HC436 name per task spec (same content).
  if [[ -f "$OUT/HC432_FINAL_VERDICT.md" ]]; then
    cp "$OUT/HC432_FINAL_VERDICT.md" "$OUT/HC436_FINAL_VERDICT.md"
  fi
  return 0
}

run_pipeline() {
  echo "[$(date -Iseconds)] running full pipeline (concat → FIFO×4 → ensemble → v2 baseline → verdict)..." >> "$LOG"

  python3 "$LVL3/scripts/v3_4_research/hc432_incremental_concat.py" \
      --require "$TARGET_N" >> "$LOG" 2>&1 || {
    echo "[$(date -Iseconds)] concat failed" >> "$LOG"
    return 1
  }

  # Run all 4 single-leg configs
  local any_ok=0
  for entry in "${EXTENDED_CFGS[@]}"; do
    IFS='|' read -r cfg horizon side conf tp sl hold cancel order <<< "$entry"
    if run_one_config "$cfg" "$horizon" "$side" "$conf" "$tp" "$sl" "$hold" "$cancel" "$order"; then
      any_ok=1
    fi
  done

  # Ensemble (depends on the two 5s legs above)
  run_ensemble || echo "[$(date -Iseconds)] ensemble step did not complete cleanly" >> "$LOG"

  # v2 baseline (best effort)
  run_v2_baseline

  # Combined leaderboard
  build_final_verdict || echo "[$(date -Iseconds)] verdict build had errors" >> "$LOG"

  if (( any_ok == 0 )); then
    echo "[$(date -Iseconds)] no configs succeeded" >> "$LOG"
    return 1
  fi
  touch "$DONE_FLAG"
  return 0
}

send_discord_summary() {
  # Build a single plain-English summary covering all 5 candidates (HC #433).
  # Uses the combined verdict's leaderboard JSON written by hc432_build_final_verdict.py.
  python3 - <<'PY' >> "$LOG" 2>&1
import json, os
from pathlib import Path
out = Path("/home/jupiter/Lvl3Quant/output/hc432_v342_47day_validation")
board_p = out / "HC432_FINAL_LEADERBOARD.json"
stash = out / "discord_pending.txt"
lines = []
if not board_p.exists():
    # Fall back to the single long-1s summary if leaderboard missing
    p = out / "v342_long_1s_top0.5_summary.json"
    if p.exists():
        s = json.load(open(p))
        o = s["overall"]
        verdict = "PASS" if s["combined_pass"] else "FAIL"
        lines = [
            f"v3.4.2 1-second long top-0.5% — 47-day FIFO: {verdict}",
            f"Net per fill {o['mean_tk']:+.3f} tk  Sharpe(sqrt-N) {o['Sharpe_sqrtN']:+.2f}  PF {o['PF']:.2f}  WR {o['WR']:.1f}%  ({o['n']:,} fills, {s['n_dates_present']} days)",
            f"Regime gap green-vs-red Sharpe ratio: {s['r1']['ratio']:.2f} (cap 0.50)",
        ]
    else:
        lines = ["HC 432 pipeline finished but no summaries were produced. Check logs."]
else:
    board = json.load(open(board_p))
    rows = board.get("rows", [])
    n_pass = sum(1 for r in rows if r.get("overall_pass"))
    lines.append(f"HC 432 — 47-day FIFO validation of 5 candidates: {n_pass} of {len(rows)} PASS the regime + MFE gates.")
    for r in rows[:5]:
        name = r.get("display_name", r.get("config_name", "?"))
        verdict = "PASS" if r.get("overall_pass") else "FAIL"
        mean = r.get("mean_tk", 0.0)
        sharpe = r.get("Sharpe_sqrtN", 0.0)
        pf = r.get("PF", 0.0)
        wr = r.get("WR", 0.0)
        lines.append(f"- {name}: {verdict}  net/fill {mean:+.3f} tk  Sharpe {sharpe:+.2f}  PF {pf:.2f}  WR {wr:.1f}%")
    if board.get("v2_baseline_note"):
        lines.append(board["v2_baseline_note"])
msg = "\n".join(lines)
open(stash, "w").write(msg)
print("STASHED Discord message:")
print(msg)
PY
}

n=0
while (( n < MAX_POLLS )); do
  count=$(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$NEPTUNE_HOST" \
          "ls $NEPTUNE_DIR/oot_*.npz 2>/dev/null | wc -l" 2>/dev/null || echo "0")
  echo "[$(date -Iseconds)] poll $n: Neptune NPZ count = $count / $TARGET_N" >> "$LOG"
  if [[ "$count" -ge "$TARGET_N" ]]; then
    if run_pipeline; then
      send_discord_summary
      echo "[$(date -Iseconds)] autorun COMPLETE." >> "$LOG"
      exit 0
    else
      echo "[$(date -Iseconds)] pipeline failure; will retry in $POLL_S s" >> "$LOG"
    fi
  fi
  n=$((n+1))
  sleep "$POLL_S"
done

echo "[$(date -Iseconds)] autorun MAX_POLLS exhausted without completion" >> "$LOG"
exit 1
