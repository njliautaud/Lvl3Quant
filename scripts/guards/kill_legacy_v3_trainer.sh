#!/bin/bash
# HC #389 GUARD — kills any unauthorized legacy v3 trainer (train_cnn_mamba_v3.py
# matching v3_smart_v3_fifo output OR v2 warmstart). Runs every 5 min via cron.
# Authorized v3.3 (uncertainty_weighted) and v3.4.2 (fixed_mtl) do NOT match.

LOG=/home/nick/Lvl3Quant/logs/guards/legacy_v3_killer.log
mkdir -p /home/nick/Lvl3Quant/logs/guards

TARGETS=$(ps -eo pid,cmd | grep -E "train_cnn_mamba_v3\.py" | grep -vE "v3_2|v3_3|v3_4|grep" | grep -E "cnn_mamba_v3_smart_v3_fifo|cnn_mamba_v2_smart_v3_mar" | awk '{print $1}')

if [ -n "$TARGETS" ]; then
    TS=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$TS] LEGACY V3 TRAINER DETECTED, killing: $TARGETS" >> "$LOG"
    for pid in $TARGETS; do
        CMDLINE=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')
        echo "[$TS]   PID $pid cmdline: $CMDLINE" >> "$LOG"
        CHILDREN=$(pgrep -P $pid 2>/dev/null)
        echo "[$TS]   PID $pid children: $CHILDREN" >> "$LOG"
        for c in $CHILDREN; do kill -9 $c 2>>"$LOG"; done
        kill -9 $pid 2>>"$LOG"
        echo "[$TS]   PID $pid killed" >> "$LOG"
    done
    echo "[$TS] LEGACY_V3_KILLED pids=$TARGETS" >> /home/nick/Lvl3Quant/logs/guards/discord_alerts.log
fi
