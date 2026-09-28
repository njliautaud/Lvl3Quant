#!/bin/bash
# check_node_actively_producing.sh
# Chunk-transition guard for idle-watchdogs (HC #469 R6(c)/(d), WEAKNESSES.md 2026-05-21 12:10 ET).
#
# Returns exit 0 if node is "actively producing" (output files mtime within window).
# Returns exit 1 if node is genuinely idle (no output activity in window).
#
# Usage: check_node_actively_producing.sh <node> [window_minutes]
#   node: neptune | razer | jupiter
#   window_minutes: how many minutes back to count as "active" (default 5)
#
# Intended use: idle-watchdog calls this BEFORE firing an idle alert.
#   If exit 0 → suppress the alert (it's a chunk-transition dip, not real idle).
#   If exit 1 → fire the alert (genuine idle).

NODE="${1:-}"
WINDOW="${2:-5}"

if [[ -z "$NODE" ]]; then
    echo "ERROR: missing node argument" >&2
    echo "Usage: $0 <neptune|razer|jupiter> [window_minutes]" >&2
    exit 2
fi

case "$NODE" in
    neptune)
        # Neptune: check output/ AND logs/ — training writes log lines every batch
        COUNT=$(ssh -o ConnectTimeout=5 -i ~/.ssh/id_ed25519 nick@neptune \
            "{ find /home/nick/Lvl3Quant/output /home/nick/Lvl3Quant/logs -type f -mmin -${WINDOW} 2>/dev/null | head -10 | wc -l; }" \
            2>/dev/null)
        ;;
    razer)
        # Razer: check output\\ AND logs\\ — Powershell on Windows
        COUNT=$(sshpass -p "${CLUSTER_SSH_PASSWORD}" ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no \
            claude@razer \
            "powershell -Command \"(Get-ChildItem C:\\Users\\claude\\Lvl3Quant\\output,C:\\Users\\claude\\Lvl3Quant\\logs -Recurse -EA SilentlyContinue | Where-Object {\$_.LastWriteTime -gt (Get-Date).AddMinutes(-${WINDOW})}).Count\"" \
            2>/dev/null | tr -d '\r' | tail -1)
        ;;
    jupiter)
        # Jupiter local: check output/ AND logs/ — FIFO replay writes log lines every ~90s
        COUNT=$(find /home/jupiter/Lvl3Quant/output /home/jupiter/Lvl3Quant/logs -type f -mmin -${WINDOW} 2>/dev/null | head -10 | wc -l)
        ;;
    *)
        echo "ERROR: unknown node '$NODE'" >&2
        exit 2
        ;;
esac

COUNT="${COUNT:-0}"

if [[ "$COUNT" -gt 0 ]]; then
    echo "ACTIVE: $NODE has $COUNT file(s) written in last ${WINDOW}min — suppress idle alert"
    exit 0
else
    echo "IDLE: $NODE has no output activity in last ${WINDOW}min — alert is genuine"
    exit 1
fi
