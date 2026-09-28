#\!/bin/bash
set -e

RESULTS_DIR=/home/jupiter/lvl3quant/alpha_discovery/deep_models/results
TS=60224_223253

# Kill existing sessions if they exist (clean start)
tmux kill-session -t event_70d 2>/dev/null || true
tmux kill-session -t lstm_70d 2>/dev/null || true
tmux kill-session -t hybrid_70d 2>/dev/null || true

# Deploy EventTransformer 70d
tmux new-session -d -s event_70d
tmux send-keys -t event_70d "cd /home/jupiter/lvl3quant && ./venv/bin/python3 alpha_discovery/deep_models/train_walkforward.py --model event --days 70 --epochs 3 --batch-size 128 --subsample-train 5 --device cpu --output-dir alpha_discovery/deep_models/results/ 2>&1 | tee alpha_discovery/deep_models/results/server_event_70d_\.log" Enter
echo "event_70d deployed"

sleep 2

# Deploy LSTM 70d
tmux new-session -d -s lstm_70d
tmux send-keys -t lstm_70d "cd /home/jupiter/lvl3quant && ./venv/bin/python3 alpha_discovery/deep_models/train_walkforward.py --model lstm --days 70 --epochs 3 --batch-size 128 --subsample-train 5 --device cpu --output-dir alpha_discovery/deep_models/results/ 2>&1 | tee alpha_discovery/deep_models/results/server_lstm_70d_\.log" Enter
echo "lstm_70d deployed"

sleep 2

# Deploy Hybrid 70d
tmux new-session -d -s hybrid_70d
tmux send-keys -t hybrid_70d "cd /home/jupiter/lvl3quant && ./venv/bin/python3 alpha_discovery/deep_models/train_walkforward.py --model hybrid --days 70 --epochs 3 --batch-size 128 --subsample-train 5 --device cpu --output-dir alpha_discovery/deep_models/results/ 2>&1 | tee alpha_discovery/deep_models/results/server_hybrid_70d_\.log" Enter
echo "hybrid_70d deployed"

sleep 1

# Verify sessions
echo "--- TMUX SESSIONS ---"
tmux list-sessions

echo "--- LOAD ---"
uptime
