# Live Paper Trading Engine

Production-grade paper trading infrastructure for real-time model deployment.

## Architecture

```
MBO Feed → Feature Engine → Models → Cards → Orders → Fill Sim → Positions → PnL
```

### Components

1. **data_feed.py** - Real-time MBO data ingestion
   - Replay mode: Stream from preprocessed .npz files
   - Live mode: Rithmic WebSocket integration
   - Rate-controlled replay for testing

2. **model_registry.py** - Model loading and management
   - EventCNN1D (PyTorch)
   - CNN-Jamba (PyTorch)
   - LGBM (joblib)
   - Hot-reload support
   - Version tracking

3. **inference_engine.py** - Real-time inference
   - Event-based models: windowed event sequences
   - Bar-based models: StreamingFeatures integration
   - Multi-model parallel inference
   - Feature caching

4. **card_engine.py** - Trading strategy execution
   - Multiple concurrent cards
   - Entry rules (threshold, tier, filters)
   - Exit rules (TP/SL, time, conviction decay)
   - Card-level PnL tracking

5. **fill_simulator.py** - Realistic fill modeling
   - Market orders: immediate fill with slippage
   - Limit orders: queue position simulation
   - Chase logic for repricing
   - Latency simulation

6. **position_manager.py** - Position and PnL tracking
   - Real-time mark-to-market
   - Closed trade history
   - Position limits enforcement
   - JSONL trade log

7. **main.py** - Main orchestrator
   - Event loop coordination
   - Status monitoring
   - Graceful shutdown
   - Final report generation

## Quick Start

### 1. Test with Replay Data

```bash
cd /home/jupiter/Lvl3Quant
python -m live_trading.main --config live_trading/configs/paper_test.json --log-level INFO
```

This will:
- Replay January 2026 MBO data at unlimited speed
- Run EventCNN1D baseline model
- Execute 2 trading cards (aggressive + conservative)
- Generate full PnL report

### 2. Multi-Model Test

```bash
python -m live_trading.main --config live_trading/configs/multi_model.json
```

Runs 3 models simultaneously:
- EventCNN1D (primary)
- CNN-Jamba (experimental, disabled)
- LGBM (secondary)

### 3. Live Deployment (Monday)

```bash
# Edit configs/live.json with:
# - feed.mode = "live"
# - Rithmic credentials in env vars
# - Production model paths

python -m live_trading.main --config live_trading/configs/live.json
```

## Configuration

Config JSON structure:

```json
{
  "feed": {
    "mode": "replay" | "live",
    "data_dir": "<path to .npz files>",
    "replay_speed": 0.0,  // 0 = unlimited, 1.0 = real-time
    "start_date": "YYYYMMDD",
    "end_date": "YYYYMMDD",
    "symbol": "ES",
    "exchange": "CME"
  },
  "event_window_size": 500,
  "warmup_events": 500,
  "market_slippage_ticks": 0.5,
  "limit_fill_probability": 0.7,
  "latency_ms": 10.0,
  "max_total_position": 5,
  "max_card_position": 1,
  "models": [ ... ],
  "cards": [ ... ]
}
```

### Model Configuration

```json
{
  "name": "model_id",
  "path": "/path/to/model.pt|.pkl",
  "type": "pytorch" | "lightgbm",
  "architecture": "event_cnn_1d" | "cnn_jamba" | "lgbm",
  "fold": 0,
  "label_horizon": "10s",
  "ic_score": 0.132
}
```

### Card Configuration

```json
{
  "name": "card_name",
  "model_name": "model_id",
  "threshold": 2.0,
  "min_tier": "all" | "top50" | "top25" | "top10",
  "max_position_size": 1,
  "take_profit_ticks": 3.0,
  "stop_loss_ticks": 1.5,
  "max_hold_seconds": 1800,
  "conviction_decay": true,
  "time_filter": "morning" | "afternoon" | "morning_afternoon",
  "chase_max_ticks": 1,
  "chase_max_reprices": 3,
  "enabled": true
}
```

## Output Files

All logs and reports stored in `live_trading/logs/`:

- `engine_<timestamp>.log` - Main engine log
- `trades.jsonl` - Trade-by-trade JSONL log
- `final_report.json` - Final PnL summary

## Testing Checklist (Saturday)

- [ ] Test replay with single model (EventCNN1D)
- [ ] Verify predictions generated correctly
- [ ] Confirm orders submitted and filled
- [ ] Check PnL calculations
- [ ] Test multiple concurrent cards
- [ ] Verify position limits enforced
- [ ] Test TP/SL exit logic
- [ ] Test conviction decay exits
- [ ] Test time filters (morning/afternoon)
- [ ] Load LGBM model with StreamingFeatures
- [ ] Run multi-model config
- [ ] Verify no memory leaks on long replay
- [ ] Test graceful shutdown (Ctrl+C)
- [ ] Review final_report.json format

## Monday Deployment Checklist

- [ ] Rithmic credentials configured (env vars)
- [ ] Production model paths confirmed
- [ ] Live config file created
- [ ] Max position limits set conservatively
- [ ] Discord webhook configured for alerts
- [ ] QCC monitoring integration tested
- [ ] Kill switch procedure documented
- [ ] Backup/restore procedure documented

## Model Paths (UPDATE BEFORE LIVE)

### EventCNN1D (baseline)
```
# Neptune training complete - check latest fold
/mnt/neptune_models/event_cnn_1d/fold_X/model_final.pt
```

### CNN-Jamba (currently training)
```
# Neptune - check MLflow for best fold
/mnt/neptune_models/cnn_jamba/fold_X/model_final.pt
```

### LGBM (Saturn)
```
# Latest production fold
/home/saturn/Lvl3Quant/lgbm_prod_wf_60_5_output/fold_11/labels_10s_lgbm.pkl
```

## Performance Expectations

Based on backtests:

### EventCNN1D (IC_10s = 0.132)
- Expected Sharpe: 2.5-3.5
- Win rate: ~55%
- Avg win: 2-3 ticks
- Avg loss: 1-2 ticks

### LGBM (IC_10s = 0.085)
- Expected Sharpe: 1.5-2.5
- Win rate: ~52%
- Faster signals (shorter hold time)

## Known Issues / TODOs

1. **MBO feed quality**: Replay mode uses synthesized add/cancel events from BBO updates (lossier than full MBO). Live Rithmic feed has same limitation until we subscribe to full order-by-order stream.

2. **Fill simulation**: Current model is simplified. Real fills depend on:
   - Queue position dynamics
   - Adverse selection
   - Market impact
   Consider integrating existing Rust fill simulator for more realism.

3. **Feature parity**: LGBM model requires exact parity with offline training features. Verify StreamingFeatures output matches `compute_derived()` batch version.

4. **Latency modeling**: Fixed 10ms latency. Real latency is variable (5-50ms). Consider stochastic latency model.

5. **Risk management**: Basic position limits implemented. Need:
   - Volatility-based sizing
   - Drawdown circuit breakers
   - Correlation limits across cards

6. **Monitoring**: Add Discord webhook notifications for:
   - Large drawdowns
   - Position limit violations
   - Model inference errors
   - Fill rejections

## Support Files

- `/home/jupiter/Lvl3Quant/live_trading_linux/` - Original LGBM live system (reference)
- `/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/` - Model training scripts
- `/home/jupiter/Lvl3Quant/scripts/mbo_event_pipeline.py` - Data preprocessing

## Contact

Issues or questions:
- Discord: #system-status channel
- QCC alerts: http://localhost:3456
- MLflow: http://localhost:5000
