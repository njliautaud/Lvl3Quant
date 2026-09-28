# Saturday Testing Plan — April 19, 2026

**Goal**: Full validation of paper trading engine with real trained models

**Current Status** (as of Friday night 1:00 AM):
- ✅ Infrastructure 100% complete and validated
- ✅ All 8 core modules working correctly  
- ✅ Data feed processing 2.7M events/day
- ✅ Inference generating 20K+ predictions/day
- ⏳ Need real trained model for order/fill testing

---

## Morning Session (9:00 AM - 12:00 PM)

### 1. Locate EventCNN1D Champion Model
- **Target**: IC_10s = 0.132 baseline (10+ folds proven)
- **Search locations**:
  - Jupiter: `/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results/`
  - Neptune: `/home/nick/Lvl3Quant/alpha_discovery/deep_models/results/`
  - MLflow artifacts: Check experiment "EventDriven_CNN1D"
- **Fallback**: Use best fold from current Razer training (if IC > 0.10)

### 2. Single Day Validation Test
```bash
cd /home/jupiter/Lvl3Quant
python3 -m live_trading.main --config live_trading/configs/paper_test.json
```

**Success Criteria**:
- Predictions generated: >10,000
- Orders submitted: >100 (with reasonable thresholds)
- Fills received: >50
- PnL tracked correctly
- No crashes or memory leaks

### 3. Verify IC Matches Backtest
- Compare predictions to actual label movements
- Calculate Spearman IC_10s on replay data
- Should be within 20% of backtest IC (noise is expected)

---

## Afternoon Session (1:00 PM - 5:00 PM)

### 4. Multi-Day Stress Test
- Config: `start_date`: "20260102", `end_date`: "20260131"
- Full month replay (~70M events)
- Monitor: CPU, RAM, execution time
- Check for memory leaks (RSS growth over time)

### 5. Risk Management Validation
- Test position limits (max 1 contract per card)
- Test stop loss triggers
- Test take profit triggers  
- Test max hold time
- Test daily loss limits

### 6. Multi-Model Testing
- Load both EventCNN1D and LGBM
- Run 3-card configuration
- Verify models don't interfere
- Check position allocation across cards

---

## Evening Session (6:00 PM - 9:00 PM)

### 7. Production Configuration
Create `live_trading/configs/live.json`:
```json
{
  "feed": {
    "mode": "live",
    "rithmic_template": 299
  },
  "models": [
    {
      "name": "event_cnn_production",
      "path": "/models/production/event_cnn1d_fold5.pt",
      "type": "pytorch",
      "architecture": "event_cnn_1d"
    }
  ],
  "cards": [
    {
      "name": "cnn_conservative_live",
      "threshold": 2.5,
      "min_tier": "top10",
      "max_position_size": 1,
      "take_profit_ticks": 5.0,
      "stop_loss_ticks": 2.0
    }
  ],
  "max_total_position": 1,
  "max_daily_loss_dollars": 500,
  "max_daily_trades": 200
}
```

### 8. Documentation Review
- [ ] README.md complete
- [ ] DEPLOYMENT.md checklist ready
- [ ] QUICK_START.md tested
- [ ] Emergency procedures documented

### 9. Final Checklist
- [ ] Model files copied to production directory
- [ ] Configs validated (no typos, correct paths)
- [ ] Discord webhooks configured
- [ ] Backup strategy in place
- [ ] Rollback plan documented

---

## Issues to Watch For

1. **Model Loading**: Ensure state_dict keys match architecture
2. **Memory Leaks**: Monitor RSS growth during long replays
3. **Prediction Latency**: Should be <10ms per prediction with stride=100
4. **Fill Quality**: Check slippage vs configuration
5. **PnL Tracking**: Verify against manual calculation

---

## Sunday Plan Preview

- **Morning**: Rithmic live connection test
- **Afternoon**: Dry-run with live feed (no orders)
- **Evening**: Final pre-flight check

---

## Monday Go-Live

- **8:00 AM**: Start engine in tmux/screen
- **9:30 AM**: Market open - first trades
- **Throughout day**: 15-min status checks
- **4:00 PM**: EOD report and analysis

---

**Notes**:
- Keep conservative for Week 1 (1 contract max)
- Start with single model (EventCNN1D only)
- Monitor first 50 trades very closely
- Don't hesitate to shut down if something looks wrong

**Emergency Stop Criteria**:
- Unrealized loss > $500
- >5 consecutive losing trades
- Unusual fill behavior (consistent adverse selection)
- Any system errors or crashes
