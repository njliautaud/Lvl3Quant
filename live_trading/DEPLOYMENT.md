# Deployment Guide — Live Paper Trading Engine

## Pre-Flight Checklist

### Saturday Testing (April 19)

#### 1. Verify Data Availability
```bash
ls -lh /home/jupiter/Lvl3Quant/data/processed/mbo_events/*.npz | tail -20
# Confirm Jan 2026 data available for testing
```

#### 2. Verify Model Files
```bash
# EventCNN1D
find /home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results/event_cnn_1d -name "model_final.pt"

# CNN-Jamba (if training complete)
find /home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results/cnn_jamba -name "model_final.pt"

# LGBM (on Saturn - use QCC to check)
```

#### 3. Test Replay Mode
```bash
cd /home/jupiter/Lvl3Quant

# Quick test (1 day, unlimited speed)
python -m live_trading.main \
  --config live_trading/configs/paper_test.json \
  --log-level INFO

# Check output
tail -100 live_trading/logs/engine_*.log
cat live_trading/logs/final_report.json
```

Expected output:
- Events processed: ~3-5M per day
- Predictions: 10k-50k per day
- Orders: 50-200 per day
- PnL: Should match backtest expectations (~$200-500/day for baseline)

#### 4. Test Multi-Model Mode
```bash
# Requires LGBM model + StreamingFeatures working
python -m live_trading.main \
  --config live_trading/configs/multi_model.json \
  --log-level DEBUG

# Verify all 3 models load and generate predictions
grep "Loaded.*model" live_trading/logs/engine_*.log
```

#### 5. Stress Test (Full Month Replay)
```bash
# Edit paper_test.json: set start_date="20260101", end_date="20260131"
python -m live_trading.main \
  --config live_trading/configs/paper_test.json

# Monitor for memory leaks, crashes
watch -n 5 'ps aux | grep live_trading | grep -v grep'
```

#### 6. Validate PnL Calculations
```python
# Manual verification
import json
with open('live_trading/logs/trades.jsonl') as f:
    trades = [json.loads(line) for line in f]

# Check:
# - PnL calculations match (entry-exit) * tick_value
# - No duplicate trades
# - Timestamps monotonically increasing
# - Fill prices within reasonable spread

total_pnl = sum(t['pnl_dollars'] for t in trades)
print(f"Total PnL: ${total_pnl:.2f}")
```

---

### Sunday Prep (April 20)

#### 1. Deploy Models to Production Paths
```bash
# Create production model directory
mkdir -p /home/jupiter/Lvl3Quant/models/production

# Copy best fold EventCNN1D
cp /home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results/event_cnn_1d/fold_X/model_final.pt \
   /home/jupiter/Lvl3Quant/models/production/event_cnn_baseline_v1.pt

# Copy CNN-Jamba if ready
# cp .../cnn_jamba/fold_X/model_final.pt \
#    /home/jupiter/Lvl3Quant/models/production/cnn_jamba_v1.pt

# Sync LGBM from Saturn
scp saturn:/home/saturn/Lvl3Quant/lgbm_prod_wf_60_5_output/fold_11/labels_10s_lgbm.pkl \
    /home/jupiter/Lvl3Quant/models/production/lgbm_prod_v1.pkl
```

#### 2. Create Live Configuration
```bash
cp live_trading/configs/paper_test.json live_trading/configs/live.json

# Edit live.json:
# - feed.mode = "live"
# - Update model paths to /models/production/
# - Set conservative position limits
# - Enable only proven cards
# - Set realistic slippage (0.5-1.0 ticks)
```

#### 3. Configure Rithmic Credentials
```bash
# Add to ~/.bashrc or systemd service file
export RITHMIC_SYSTEM="Rithmic Paper Trading"
export RITHMIC_USER="<AMP_USER>"
export RITHMIC_PASSWORD="<AMP_PASSWORD>"
export RITHMIC_URI="wss://rituz00100.rithmic.com:443"

# Test connection
cd /home/jupiter/teleclaude-main/downloads/rithmic_api/0.89.0.0/samples/samples.py
python SampleMD.py
# Should connect and list available systems
```

#### 4. Setup Monitoring
```bash
# Discord webhook for alerts
export DISCORD_WEBHOOK_URL="<webhook_url>"

# QCC integration
# Verify QCC daemon running
curl http://localhost:3456/health

# Test Discord notification
python -c "
import requests
requests.post('$DISCORD_WEBHOOK_URL', json={'content': 'Paper trading engine ready for Monday'})
"
```

#### 5. Create systemd Service (optional)
```bash
sudo nano /etc/systemd/system/paper-trading.service
```

```ini
[Unit]
Description=Paper Trading Engine
After=network.target

[Service]
Type=simple
User=jupiter
WorkingDirectory=/home/jupiter/Lvl3Quant
Environment=RITHMIC_SYSTEM=Rithmic Paper Trading
Environment=RITHMIC_USER=<user>
Environment=RITHMIC_PASSWORD=<pass>
Environment=RITHMIC_URI=wss://rituz00100.rithmic.com:443
ExecStart=/usr/bin/python3 -m live_trading.main --config live_trading/configs/live.json
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable paper-trading.service
```

---

### Monday Go-Live (April 21)

#### Pre-Market (8:00 AM ET)

```bash
# 1. Verify markets open
# ES regular trading hours: 9:30 AM - 4:00 PM ET

# 2. Final config check
cat live_trading/configs/live.json | jq .

# 3. Clear old logs
rm live_trading/logs/engine_*.log
rm live_trading/logs/trades.jsonl

# 4. Start engine in screen/tmux
screen -S paper_trading
cd /home/jupiter/Lvl3Quant
python -m live_trading.main --config live_trading/configs/live.json --log-level INFO

# Detach: Ctrl+A, D
```

#### Market Open (9:30 AM ET)

Monitor first 30 minutes closely:

```bash
# Watch live logs
tail -f live_trading/logs/engine_*.log

# Check for:
# - Data feed connected
# - Models loaded
# - Predictions generating
# - Orders submitting
# - Fills executing
# - PnL updating

# Status check every 5 minutes
watch -n 300 'tail -20 live_trading/logs/engine_*.log'
```

#### Intraday Monitoring

```bash
# Check positions every 15 minutes
python -c "
import json
with open('live_trading/logs/final_report.json') as f:
    report = json.load(f)
pnl = report['pnl']
print(f\"Open: {pnl['open_positions']} | Exposure: {pnl['total_exposure']}\")
print(f\"PnL: \${pnl['total_pnl_dollars']:.2f} ({pnl['total_pnl_ticks']:.1f}t)\")
print(f\"Trades: {pnl['total_trades']} | WR: {pnl['win_rate']:.1%}\")
"

# Check for alerts
curl http://localhost:3456/alerts | jq .
```

#### Market Close (4:00 PM ET)

```bash
# Graceful shutdown
screen -r paper_trading
# Press Ctrl+C

# Review final report
cat live_trading/logs/final_report.json | jq .

# Archive logs
tar -czf logs/paper_trading_$(date +%Y%m%d).tar.gz live_trading/logs/

# Post results to Discord
python -c "
import json, requests, os
with open('live_trading/logs/final_report.json') as f:
    r = json.load(f)
pnl = r['pnl']
msg = f\"\"\"**Paper Trading Day 1 Results**
Total PnL: \${pnl['total_pnl_dollars']:+,.2f} ({pnl['total_pnl_ticks']:+.1f} ticks)
Trades: {pnl['total_trades']} (W/L: {pnl['winning_trades']}/{pnl['losing_trades']}, WR: {pnl['win_rate']:.1%})
Cards: {len(r['cards'])}
\"\"\"
requests.post(os.environ['DISCORD_WEBHOOK_URL'], json={'content': msg})
"
```

---

## Emergency Procedures

### Kill Switch
```bash
# Immediate shutdown
pkill -9 -f "live_trading.main"

# Or via systemd
sudo systemctl stop paper-trading.service
```

### Model Hot-Swap
```bash
# 1. Stop engine
# 2. Replace model file in /models/production/
# 3. Update config if needed
# 4. Restart engine

# Engine will reload models on startup
```

### Position Reconciliation
```bash
# If positions get out of sync:
# 1. Check Rithmic account positions
# 2. Compare with trades.jsonl
# 3. Manually adjust if needed (rare — should auto-reconcile)
```

### Data Feed Failure
```bash
# If Rithmic disconnects:
# 1. Check network
# 2. Verify Rithmic credentials
# 3. Restart engine (will auto-reconnect)

# Fallback: switch to replay mode temporarily
# Edit live.json: feed.mode = "replay"
```

---

## Performance Monitoring

### Key Metrics

1. **Inference Latency**: Should be <10ms per prediction
   - Check: `grep "latency_ms" live_trading/logs/engine_*.log`

2. **Order Fill Rate**: Should be >90% for market orders
   - Check: `orders_submitted` vs `fills_received` in status logs

3. **Model IC**: Track rolling IC over live predictions
   - Compare with backtest IC expectations

4. **Slippage**: Track actual fill prices vs expected
   - Should average <1 tick for market orders

5. **Win Rate**: Should match backtest ±5%
   - EventCNN1D: expect 55-60%
   - LGBM: expect 50-55%

6. **Sharpe**: Track intraday Sharpe
   - Should be positive by end of day

### Logging

All important events logged to:
- `engine_*.log`: Main event log
- `trades.jsonl`: Trade-by-trade details
- `final_report.json`: Summary statistics

Parse with:
```bash
# Find all fills today
grep "FILLED" live_trading/logs/engine_*.log | tail -50

# Count predictions by model
grep "PREDICTION" live_trading/logs/engine_*.log | cut -d' ' -f5 | sort | uniq -c

# Track PnL over time
grep "Total PnL:" live_trading/logs/engine_*.log | tail -20
```

---

## Troubleshooting

### "Model file not found"
- Check path in config
- Verify file exists: `ls -lh /path/to/model.pt`
- Check permissions: `chmod 644 /path/to/model.pt`

### "StreamingFeatures not available"
- LGBM models require `live_trading_linux.streaming_features`
- Verify module exists: `python -c "from live_trading_linux.streaming_features import StreamingFeatures"`
- If missing, copy from existing system

### "No predictions generated"
- Check warmup_events reached (default 500)
- Verify model inference not erroring: `grep "ERROR.*inference" logs/engine_*.log`
- Check input features shape matches model expectations

### "Orders not filling"
- Verify fill simulator receiving market updates
- Check limit prices (may be too aggressive)
- Review chase logic parameters

### High slippage
- Increase `market_slippage_ticks` in config to match reality
- Consider switching to limit orders with chase

### Memory leak
- Monitor with: `watch -n 60 'ps aux | grep live_trading'`
- If growing unbounded, may be accumulating predictions/events
- Add periodic cleanup in inference_engine.py

---

## Success Criteria

### Saturday Testing
- ✅ Replay mode works end-to-end
- ✅ All models load successfully
- ✅ Predictions generate at expected rate
- ✅ Orders submit and fill
- ✅ PnL calculations accurate
- ✅ No crashes on full month replay

### Monday Live
- ✅ Connect to Rithmic successfully
- ✅ Receive live data
- ✅ Generate first prediction within 5 minutes
- ✅ Execute first trade within 30 minutes
- ✅ Survive full RTH session without crashes
- ✅ Final PnL matches expectations (within 2σ)
- ✅ No position limit violations
- ✅ All fills logged correctly

---

## Next Steps (Week 2+)

1. **Optimize fill simulation**
   - Integrate Rust fill simulator for higher fidelity
   - Add queue position tracking
   - Model adverse selection

2. **Enhanced risk management**
   - Volatility-based position sizing
   - Correlation limits across cards
   - Drawdown circuit breakers

3. **Multi-symbol support**
   - Add NQ (Nasdaq futures)
   - Cross-market correlation
   - Symbol-specific cards

4. **Model improvements**
   - Deploy CNN-Jamba when validated
   - Add Event Transformer
   - Ensemble predictions

5. **Live → Real transition**
   - Validate paper results vs backtest
   - Get regulatory approval if needed
   - Start with minimum size (1 contract)
   - Scale gradually based on performance
