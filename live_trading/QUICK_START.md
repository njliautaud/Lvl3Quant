# Quick Start Guide

## 5-Minute Test Run

```bash
cd /home/jupiter/Lvl3Quant

# 1. Verify system components
python3 live_trading/test_system.py

# 2. Run quick test (1 day replay, unlimited speed)
python3 -m live_trading.main \
  --config live_trading/configs/paper_test.json \
  --log-level INFO

# 3. Check results
cat live_trading/logs/final_report.json | python3 -m json.tool
```

## What to Expect

The test will:
1. Load EventCNN1D baseline model
2. Replay 1 day of MBO data (~3-5M events)
3. Generate 10k-50k predictions
4. Execute 50-200 trades via 2 cards
5. Calculate final PnL

**Expected runtime**: 2-5 minutes (unlimited replay speed)

**Expected output**:
```
Events processed: ~4,000,000
Predictions: ~30,000
Orders: ~150
Fills: ~150
Total PnL: $200-500 (baseline model)
Win Rate: 55-60%
```

## Sample Output

```json
{
  "pnl": {
    "open_positions": 0,
    "total_pnl_ticks": 32.4,
    "total_pnl_dollars": 405.00,
    "total_trades": 142,
    "winning_trades": 82,
    "losing_trades": 60,
    "win_rate": 0.577
  },
  "cards": {
    "cnn_baseline_aggressive": {
      "total_trades": 89,
      "total_pnl_ticks": 18.2,
      "win_rate": 0.573
    },
    "cnn_baseline_conservative": {
      "total_trades": 53,
      "total_pnl_ticks": 14.2,
      "win_rate": 0.585
    }
  }
}
```

## Log Files

All output saved to `live_trading/logs/`:
- `engine_*.log` - Main event log
- `trades.jsonl` - Trade details (one per line)
- `final_report.json` - Summary statistics

## Troubleshooting

### "Model file not found"
The test config points to a placeholder model path. Update the path in `configs/paper_test.json`:

```json
{
  "models": [
    {
      "name": "event_cnn_baseline",
      "path": "/actual/path/to/your/model.pt",
      ...
    }
  ]
}
```

To find trained models:
```bash
find /home/jupiter/Lvl3Quant -name "model_final.pt" 2>/dev/null
```

### "No data files found"
Verify MBO event data exists:
```bash
ls -lh /home/jupiter/Lvl3Quant/data/processed/mbo_events/*.npz | head -10
```

If missing, run data preprocessing:
```bash
python3 /home/jupiter/Lvl3Quant/scripts/mbo_event_pipeline.py
```

### "StreamingFeatures not available"
LGBM models require the StreamingFeatures module. Either:
1. Use only PyTorch models (EventCNN1D, CNN-Jamba)
2. Copy StreamingFeatures from `/home/jupiter/Lvl3Quant/live_trading_linux/streaming_features.py`

## Next Steps

Once the quick test passes:

1. **Run multi-model test**:
   ```bash
   python3 -m live_trading.main --config live_trading/configs/multi_model.json
   ```

2. **Full month backtest**:
   Edit `paper_test.json`: set `start_date="20260101"`, `end_date="20260131"`
   ```bash
   python3 -m live_trading.main --config live_trading/configs/paper_test.json
   ```

3. **Review full documentation**:
   - `README.md` - Complete system overview
   - `DEPLOYMENT.md` - Saturday testing + Monday go-live checklist

## Support

Questions? Check:
- System status: http://localhost:3456
- MLflow experiments: http://localhost:5000
- Discord: #system-status channel
