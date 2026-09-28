#!/bin/bash
# Wait for GNN sweep to finish, then launch card optimization
echo "Waiting for GNN sweep (PID 1305376) to finish..."
while kill -0 1305376 2>/dev/null; do
    count=$(ps aux | grep fill_sim | grep -v grep | wc -l)
    echo "  fill_sim workers: $count (waiting...)"
    sleep 30
done
echo "GNN sweep done. Waiting 10s for cleanup..."
sleep 10
echo "Launching card optimization sweep..."
python3 /home/jupiter/card_optimization_sweep.py 2>&1 | tee /home/jupiter/card_optimization.log
echo "Card optimization complete!"

