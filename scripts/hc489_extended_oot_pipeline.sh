#!/bin/bash
# HC #489 R2 + HC #428 R1: ship 20 OOT-candidate smart_v3 days to Razer + trigger inference batch.
# Designed to survive Claude session restart / sub-agent exit (use setsid).
set -u
export SSHPASS="${CLUSTER_SSH_PASSWORD:-}"
RAZER=claude@razer
SRC=/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3
LOG=/home/jupiter/Lvl3Quant/logs/hc489_ext_oot_pipeline.log
DEST="C:/Users/claude/Lvl3Quant/data/processed/mbo_events_smart_v3/"
DATES="20260301 20260302 20260303 20260304 20260305 20260306 20260308 20260309 20260310 20260311 20260312 20260313 20260315 20260316 20260317 20260318 20260319 20260320 20260322 20260323"

echo "[$(date)] PIPELINE START" >> "$LOG"

# Build a single scp invocation with all files (re-uses one SSH connection)
FILES=""
for d in $DATES; do
    f="$SRC/${d}_mbo_events.npz"
    if [[ -f "$f" ]]; then
        FILES="$FILES $f"
    else
        echo "[$(date)] MISSING source: $f" >> "$LOG"
    fi
done

echo "[$(date)] Starting scp of $(echo $FILES | wc -w) files (~28 GB) -> Razer" >> "$LOG"
sshpass -e scp -o StrictHostKeyChecking=no -o ServerAliveInterval=30 $FILES "$RAZER:$DEST" >> "$LOG" 2>&1
RC=$?
echo "[$(date)] scp exit code: $RC" >> "$LOG"

if [[ $RC -ne 0 ]]; then
    echo "[$(date)] SCP FAILED — aborting batch trigger." >> "$LOG"
    exit 1
fi

# Trigger Razer batch via SSH
echo "[$(date)] SCP done. Triggering Razer inference batch..." >> "$LOG"
sshpass -e ssh -o StrictHostKeyChecking=no "$RAZER" \
    "schtasks /create /tn hc489_ext_oot /sc once /st 00:01 /sd 01/01/2030 /tr \"cmd /c C:\\Users\\claude\\Lvl3Quant\\scripts\\launch_hc489_extended_oot.bat > C:\\Users\\claude\\Lvl3Quant\\logs\\hc489_ext_oot.log 2>&1\" /f && schtasks /run /tn hc489_ext_oot" \
    >> "$LOG" 2>&1
echo "[$(date)] PIPELINE LAUNCH COMPLETE (inference runs autonomously on Razer)" >> "$LOG"
