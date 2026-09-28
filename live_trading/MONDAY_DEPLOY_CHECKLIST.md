# Monday Stacked Confluence — Paper Trading Deploy Checklist

Config: `live_trading/configs/monday_stacked_v1.json`
Strategy: CNN-Mamba v2 (1s) + meta-model (shorts) + OFI gate, PASSIVE-ONLY
Default preset: **balanced** (sig10% + meta30% + OFI)

---

## Pre-Market Checks (before 9:25 ET)

- [ ] **CNN-Mamba v2 weights present on Razer**: `C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_smart_v3_mar\fold_10_best.pt`
- [ ] **Feature stats present**: `fold_09_feature_stats.npz` same directory
- [ ] **Meta-model weights synced to Razer**: `C:\Users\claude\Lvl3Quant\output\meta_production_v1\weights\` (15 fold files)
- [ ] **MBO data feed alive**: Rithmic connected, streaming ES quotes, no stale data
- [ ] **Config loaded**: paper trader reads `monday_stacked_v1.json`, confirm active_preset = "balanced"
- [ ] **Contract rolled**: Confirm ESU6 (Sep 2026) is front month
- [ ] **GPU healthy**: `nvidia-smi` on Razer shows RTX 3070 available, no zombie processes
- [ ] **Disk space**: >10 GB free on Razer for logs + MBO recording

## Go-Live Sequence

1. Start MBO recorder on Razer (if not already running via scheduled task)
2. Start inference daemon with config: `--config configs/monday_stacked_v1.json`
3. Verify first predictions appear within 30s of market data flowing
4. Confirm OFI gate is receiving book data and computing imbalance
5. Confirm meta-model weights loaded (log should show "15 folds loaded")
6. Wait for 9:40 ET (TOD block clears) — first trades should appear shortly after
7. Verify first paper trade logged with full attribution (signal pct, meta score, OFI direction)

## What to Monitor

- **Fill rate**: Passive limit orders should fill 60-80% of the time in normal conditions. Below 40% = something wrong with queue position or pricing
- **Signal frequency**: Balanced preset expects ~527 trades/day (~1.4/min during RTH). If <200 by noon, investigate thresholds
- **Per-trade P&L distribution**: Mean should be near +0.46 ticks. If negative after 50+ trades, pause and investigate
- **Win rate**: Target 58-59%. Below 52% after 100+ trades = concern
- **Hold time**: Should cluster around 5s. If many trades hitting stop or timing out at cancel window, check signal quality
- **Meta-model gate-out rate**: Should reject ~70% of raw signals. If rejecting >90% or <50%, check meta weights
- **OFI agreement rate**: Expect OFI to agree with ~60-70% of signals. If <30%, book microstructure may have shifted

## Kill Criteria (stop trading immediately)

- **Daily loss exceeds $500** (automatic safety net)
- **5 consecutive losses** (automatic cooldown, but manually review if it happens twice)
- **Win rate below 48% after 200+ trades** — strategy edge not present today
- **Fill rate below 30%** — passive orders not getting filled, execution broken
- **Zero trades by 11:00 ET** — something fundamentally wrong with pipeline
- **MBO feed stale >10 seconds** — data integrity compromised
- **Inference latency >100ms** — model too slow, predictions stale before acting
- **Sharpe intraday below -1.0 after 100+ trades** — negative edge, shut it down

## Fallback Plan

If balanced preset underperforms, switch to **simpler** preset (sig5% + OFI, no meta):
- Edit config: set `active_preset` to `"simpler"`
- Restart paper trader
- Higher trade count (~665/day) but slightly lower per-trade edge
- Removes meta-model dependency (fewer failure points)

## Post-Session

- [ ] Export trade log and compute daily metrics (Sharpe, Sortino, PF, WR)
- [ ] Compare actual vs backtest: expected +0.46 ticks/trade, Sharpe ~26.8
- [ ] Check regime: was today green/red/flat? Compare to regime-stratified backtest
- [ ] Log results to MLflow experiment `monday_stacked_v1`
- [ ] Report summary to Discord
