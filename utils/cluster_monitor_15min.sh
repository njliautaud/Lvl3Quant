#!/bin/bash
# 15-minute cluster monitoring - runs locally on Jupiter
# Fixed: reads QCC SQLite DB directly instead of broken HTTP endpoint
# Also runs infra_sync.py to keep all documentation up to date

LOG="/home/jupiter/Lvl3Quant/logs/cluster_monitor.log"
DB="/home/jupiter/teleclaude-main/data/qcc.db"
SYNC_SCRIPT="/home/jupiter/Lvl3Quant/scripts/infra_sync.py"

echo "[$(date)] === 15-min Cluster Check ===" >> "$LOG"

# Get node status from QCC SQLite DB directly
python3 -c "
import sqlite3, json
conn = sqlite3.connect('file:${DB}?mode=ro', uri=True)
conn.row_factory = sqlite3.Row
nodes = conn.execute('SELECT name, status, last_gpu_util, last_gpu_mem_mb, last_heartbeat FROM compute_nodes').fetchall()
for n in nodes:
    gpu = n['last_gpu_util'] or 0
    mem = n['last_gpu_mem_mb'] or 0
    print(f'  {n[\"name\"]:>10}: status={n[\"status\"]:>8} gpu={gpu:>3.0f}% mem={mem:>5.0f}MB hb={n[\"last_heartbeat\"] or \"never\"}')
conn.close()
" >> "$LOG" 2>&1

# Check local training processes
echo "  Local processes:" >> "$LOG"
ps aux | grep -E "train_|lgbm_|mamba" | grep -v grep | awk '{printf "    PID=%s CPU=%s MEM=%s CMD=%s\n", $2, $3, $4, $11}' >> "$LOG"

# Run infrastructure sync (directives + queue + docs)
python3 "$SYNC_SCRIPT" --directives >> "$LOG" 2>&1

# Trim log if too big (>5MB)
if [ -f "$LOG" ] && [ $(stat -f%z "$LOG" 2>/dev/null || stat -c%s "$LOG" 2>/dev/null) -gt 5242880 ]; then
    tail -1000 "$LOG" > "${LOG}.tmp" && mv "${LOG}.tmp" "$LOG"
fi
