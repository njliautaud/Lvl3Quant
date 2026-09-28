#!/bin/bash
# Sync Razer's daily MBO NPZ files back to Jupiter (HC #33)
# Runs nightly after market close
# Syncs only new/changed files to avoid redundant transfers

RAZER_HOST="claude@razer"
RAZER_DIR="C:/Users/claude/Lvl3Quant/data/processed/mbo_events/"
JUPITER_DIR="/home/jupiter/Lvl3Quant/data/processed/mbo_events/"
LOG="/home/jupiter/Lvl3Quant/logs/razer_npz_sync.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting Razer NPZ sync..." >> "$LOG"

# Use scp with sshpass for the latest day's file
TODAY_UTC=$(date -u +%Y%m%d)
YESTERDAY_UTC=$(date -u -d "yesterday" +%Y%m%d 2>/dev/null || date -u -v-1d +%Y%m%d)

for DATE in "$TODAY_UTC" "$YESTERDAY_UTC"; do
    FILE="${DATE}_mbo_events.npz"
    sshpass -p "${CLUSTER_SSH_PASSWORD}" scp -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        "${RAZER_HOST}:${RAZER_DIR}${FILE}" "${JUPITER_DIR}${FILE}.tmp" 2>/dev/null
    if [ $? -eq 0 ] && [ -f "${JUPITER_DIR}${FILE}.tmp" ]; then
        mv "${JUPITER_DIR}${FILE}.tmp" "${JUPITER_DIR}${FILE}"
        echo "$(date '+%Y-%m-%d %H:%M:%S') Synced ${FILE} ($(stat -c%s "${JUPITER_DIR}${FILE}") bytes)" >> "$LOG"
    fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S') Sync complete." >> "$LOG"
