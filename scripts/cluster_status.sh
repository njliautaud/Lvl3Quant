#!/bin/bash
# Compact cluster status check — outputs max ~20 lines
# Designed to minimize context window usage in Claude sessions
# Usage: bash scripts/cluster_status.sh

echo "=== CLUSTER $(date '+%H:%M ET %b %d') ==="

# Jupiter (local)
JUP_PROCS=$(ps aux | grep -E "python.*train" | grep -v grep | awk '{print $NF}' | sed 's|.*/||' | tr '\n' ', ' | sed 's/,$//')
JUP_MEM=$(free -g | awk '/Mem/{printf "%d/%dGB", $3, $2}')
echo "JUP: ${JUP_PROCS:-idle} | RAM $JUP_MEM"

# Razer
RAZER=$(timeout 10 sshpass -p "${CLUSTER_SSH_PASSWORD}" ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no claude@razer 'powershell -Command "
$procs = Get-Process python* -EA SilentlyContinue | Select-Object Id
$gpu = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
if ($procs) { Write-Host \"PIDs: $($procs.Id -join \",\") | GPU: $gpu\" } else { Write-Host \"idle | GPU: $gpu\" }
"' 2>/dev/null)
echo "RAZ: ${RAZER:-SSH_FAIL}"

# Neptune
NEPTUNE=$(timeout 5 ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no nick@neptune '
procs=$(ps aux | grep "python.*train" | grep -v grep | awk "{print \$NF}" | sed "s|.*/||" | tr "\n" "," | sed "s/,$//" )
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
echo "${procs:-idle} | GPU: ${gpu:-N/A}"
' 2>/dev/null)
echo "NEP: ${NEPTUNE:-SSH_FAIL}"

# Saturn
SATURN=$(timeout 5 ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no saturn@saturn 'echo "up | RAM: $(free -g | awk "/Mem/{printf \"%d/%dGB\", \$3, \$2}")"' 2>/dev/null)
echo "SAT: ${SATURN:-SSH_FAIL}"

# Recent log tails (last line only)
for log in /home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results/lgbm_v2_sliding.log /home/jupiter/Lvl3Quant/results/lgbm_book30.log; do
    if [ -f "$log" ]; then
        LAST=$(tail -1 "$log" 2>/dev/null | head -c 120)
        echo "LOG $(basename $log): $LAST"
    fi
done
echo "=== END ==="
