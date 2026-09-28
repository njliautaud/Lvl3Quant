# LH 2H Paper Engine — Monday Deployment Readiness Checklist

**Last updated**: 2026-07-12 (HC #664 adversarial audit fixes applied)

---

## Status: READY — Waiting on Rithmic data feed fix

---

## What Needs to Happen on Razer

- [ ] Rithmic data feed reconnected (user fixing)
- [ ] MBO recorder running (`start_mbo_recorder.sh` or PM2 equivalent)
- [ ] Verify MBO `.npz` files are being written to Razer's data directory
- [ ] Confirm Tailscale connectivity (Razer @ razer)

## What Needs to Happen on Jupiter

- [ ] `razer_auto_sync.sh` cron is active (already in crontab, runs every 30 min)
- [ ] After sync: `run_minute_bars_bulk.sh` converts MBO to minute bar parquets
- [ ] Verify new parquet files appear in `/home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1/`
- [ ] Paper engine (PM2: `lh-2h-paper`) is running and will auto-detect fresh data on next hourly reload

## Expected Timeline: Rithmic Fixed -> First Paper Trade

1. **T+0**: Rithmic reconnects, MBO recording starts on Razer
2. **T+30min**: `razer_auto_sync.sh` cron detects Razer online, syncs new MBO files
3. **T+35min**: `run_minute_bars_bulk.sh` converts to minute bar parquets
4. **T+60min**: Paper engine reloads minute bars (hourly reload interval)
5. **T+60min**: Stale data guard clears once latest bar date is within 2 trading days
6. **T+60min**: Engine retrains on fresh data (daily retrain trigger)
7. **Next signal hour (14-18 UTC)**: First live paper trade generated

**Minimum requirement**: Need at least 1 day of fresh minute bars PLUS the existing 197 days already on Jupiter. The engine will retrain on the most recent 60 days once the stale data guard passes.

## Safeguards In Place (HC #664)

| Safeguard | Threshold | Behavior |
|-----------|-----------|----------|
| Stale Data Guard | > 2 trading days old | REFUSE to trade, log warning |
| Confidence Filter | \|pred\| < 16 ticks | Skip trade (bottom 20% loses money) |
| Max Daily Loss | > 100 ticks ($1,250) | Stop trading for the day |
| Constant Prediction | Same value 3+ hours | Flag model as broken, stop |
| Mid-Trade Invalidation | Prediction flips sign with confidence | Exit early (HC #648) |
| Vol Regime Feature | Expanding rank (past-only) | No look-ahead bias |

## Methodology Verification (Confirmed Correct)

- Train/val split: 80/20 temporal split for early stopping
- Purge gap: 5 days between train end and prediction day
- Sliding window: 60 days (not expanding, per HC #0)
- Cost: 1.376 ticks RT (market order + AMP commission)
- Labels: forward 2-hour tick move, overnight gaps nulled
- Intraday-clean: hours 19-20 UTC excluded from entry

## Key Files

- Engine: `live_trading_linux/lh_2h_paper_engine.py`
- PM2 config: `live_trading_linux/lh_2h_paper_ecosystem.config.js`
- State: `live_trading_linux/lh_2h_paper_state/state.json`
- Trades log: `output/lh_2h_paper/trades.csv`
- Minute bars: `data/processed/mbo_minute_bars_v1/`
- Sync script: `scripts/razer_auto_sync.sh`
- Walkforward results: `output/lh_2h_full_walkforward_results.json`
- Predictions (real): `output/lh_2h_intraday_clean/predictions.parquet`

## Model Performance (Walkforward, 132 OOT Days)

- 387 trades, WR 73.9%
- Sharpe 16.6, Sortino 62.3, PF 47.7
- Max DD: -191 ticks
- Permutation p-value: 0.000 (signal is real)
- Regime-agnostic: |GREEN-RED Sharpe gap| = 9.2% (passes < 50% gate)
