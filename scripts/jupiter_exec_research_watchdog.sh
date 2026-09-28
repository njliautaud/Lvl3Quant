#!/bin/bash
# HC #30 watchdog: DISABLED per HC #61 — simple fillsim sweeps are pointless.
# Jupiter exec research should be SMART execution (adaptive TP/SL, RL, MLP gates),
# NOT dumb 3-factor grid sweeps.
# Re-enable only when real smart exec research scripts exist.
echo "[$(date)] DISABLED per HC #61 — no simple fillsim sweeps" >> /home/jupiter/Lvl3Quant/logs/jupiter_exec_research/watchdog.log
exit 0
# ORIGINAL CODE BELOW (kept for reference):
# HC #30 watchdog: ensure Jupiter is always running CNN Mamba v2 execution research
# during weekday market+post-market hours (9 AM - 8 PM ET, Mon-Fri).
# If no fillsim/exec process detected → auto-launch the sweep.

LOG=/home/jupiter/Lvl3Quant/logs/jupiter_exec_research/watchdog.log
mkdir -p $(dirname "$LOG")

DOW=$(date +%u)  # 1=Mon, 7=Sun
HOUR=$(date +%H)

# Only enforce during weekday market + post-market window (9-20 ET)
if [ "$DOW" -gt 5 ]; then
  echo "[$(date)] Weekend — skipping watchdog" >> "$LOG"
  exit 0
fi
if [ "$HOUR" -lt 9 ] || [ "$HOUR" -gt 20 ]; then
  echo "[$(date)] Outside market+post-market window (hour=$HOUR) — skipping" >> "$LOG"
  exit 0
fi

# Check for any active exec-research process
ACTIVE=$(pgrep -fa "deep_pred_to_fillsim|fill_sim_cli|cnn_mamba_v2_fillsim_sweep|event_conviction_fillsim_sweep|execution_strategy_sweep" | grep -v grep | wc -l)

if [ "$ACTIVE" -gt 0 ]; then
  echo "[$(date)] OK — $ACTIVE exec-research process(es) active" >> "$LOG"
  exit 0
fi

echo "[$(date)] HC #30 VIOLATION: no exec-research process — launching sweep" >> "$LOG"

# Auto-launch the sweep
nohup /home/jupiter/Lvl3Quant/scripts/cnn_mamba_v2_fillsim_sweep.sh \
  >> /home/jupiter/Lvl3Quant/logs/jupiter_exec_research/sweep_launch.log 2>&1 &
NEW_PID=$!
echo "[$(date)] Launched fillsim sweep PID=$NEW_PID" >> "$LOG"

# Push Discord alert via inject endpoint
curl -s -X POST http://127.0.0.1:7731/inject \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"HC #30 watchdog auto-launched CNN Mamba v2 fillsim sweep on Jupiter (was idle). PID=$NEW_PID. Push status to #general after first config completes.\"}" \
  >> "$LOG" 2>&1
