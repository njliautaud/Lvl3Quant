# Safety Nets — Per-Gate Rationale

**Authorized**: HC #344 (LIVE-READINESS), HC #368 (this package).

Each gate below is one watchdog in `safety_nets.py`. All 8 must be enabled at
startup; ANY failure halts new entries (open positions are managed by
per-trade stop loss + time exit, not by watchdog).

## 1. `MaxDDCircuitBreaker` — intraday drawdown

| Field | Default | Rationale |
|---|---|---|
| `max_intraday_drawdown_ticks` | 50 | ES 50 ticks = $625. Sized at ~5× avg-daily-loss from v2 history. |
| `max_intraday_loss_dollars` | 625 | Dollar mirror of tick threshold; primary driver under contract changes. |
| `action_on_trip` | halt + force-exit-flat | Most aggressive. The day is over once you hit this. |

Why this big: ES is wide. A 30-tick threshold would trip on normal session noise.
50 ticks is "something is structurally wrong" territory.

## 2. `PerTradeStopLoss`

| Field | Default | Rationale |
|---|---|---|
| `per_trade_stop_loss_ticks` | 3 | HC #361 MAE distribution: P95 MAE for SHORT Top0.1% is ~2.5 ticks. 3 ticks is "this trade is just wrong" line. |
| `action_on_trip` | advisory only (paper_trader.py owns the close) | Watchdog doesn't issue orders — it flags the breach for paper_trader to act on. |

## 3. `TradeCountCap`

| Field | Default | Rationale |
|---|---|---|
| `max_trades_per_day` | 30 (ML mode) / 10 (static fallback) | Throughput cap. P99-percentile entries fire ~5-15× per day on historical data; 30 is 2× ceiling. |
| `action_on_trip` | halt entries (let current trade close) | Don't unwind, just stop opening new. |

## 4. `ModelStaleWatchdog`

| Field | Default | Rationale |
|---|---|---|
| `model_stale_seconds` | 30 | Inference daemon emits heartbeat every prediction (~250ms stride). 30s = ~120 missed strides. By then we know it's dead. |
| `action_on_trip` | halt entries, switch to static fallback if available | Don't trade blind. |

## 5. `MBOFeedWatchdog`

| Field | Default | Rationale |
|---|---|---|
| `mbo_feed_stale_seconds` | 10 | MBO events arrive at >10 Hz during RTH. 10s of silence = recorder dead or market halted. |
| `action_on_trip` | halt entries + force-exit-flat | If we can't see the book we can't manage exits. |

## 6. `SigmaSpikeHalt` — model uncertainty blow-up

| Field | Default | Rationale |
|---|---|---|
| `sigma_spike_multiplier_halt` | 5.0 | Each head has learned σ from MTL training. If live σ > 5× OOT baseline, model is operating out-of-distribution. |
| `oot_sigma_baseline_path` | `<ckpt>/oot_sigma_baseline.json` | Saved at training EOD. Per-head mean σ over OOT. |
| `action_on_trip` | halt entries | The model itself is telling you it's confused. |

## 7. `ConfluenceDisagreementHalt`

| Field | Default | Rationale |
|---|---|---|
| `min_agreeing_heads` | 2 | HC #358 confluence matrix: solo heads ≤ 0.5 Sharpe, two-head agreement reaches 1.80. |
| `agreement_horizon_set` | `["log_ret_1s", "log_ret_5s"]` | Closest-to-entry heads where signal is strongest (HC #363 decay analysis). |
| `action_on_trip` | block entry, but don't halt session | This is a per-trade decision, not a kill switch. |

## 8. `InferenceLatencyWatchdog`

| Field | Default | Rationale |
|---|---|---|
| `inference_latency_max_ms` | 100 | Razer RTX 3070 typical fwd pass < 30ms. 100ms = something stalled. |
| `consecutive_slow_predictions_to_halt` | 5 | Singles can be GC pauses; 5 in a row = real problem. |
| `action_on_trip` | halt entries | Stale prediction = wrong action. |

## 9. `PositionReconcileWatchdog`

| Field | Default | Rationale |
|---|---|---|
| `position_reconcile_interval_seconds` | 60 | Once a minute compare broker position vs internal accounting. |
| `tolerance_contracts` | 0 | Must match exactly. Even 1-contract drift is a partial-fill bug. |
| `action_on_trip` | halt entries + alert Discord | Human-in-the-loop required. Worse: do NOT auto-flatten — could compound the bug. |

---

## Why these gates and not others

**Excluded by design**:
- "Spread too wide" — handled by passive-limit entry inherently (we don't cross).
- "Bid-ask flip rate" — useful diagnostic, doesn't gate trades. Logged for EOD.
- "News blackout" — handled by `time_of_day_gate` + the static fallback's `within_seconds_of_news_release` block. Not a runtime watchdog.

**Static fallback is the 10th gate**: not in `safety_nets.py` because it's a
*replacement* engine, not a halt. When ML-stack watchdogs trip, paper_trader
swaps to `static_fallback_rules.yaml` so trades don't stop entirely. See HC #368.

## Self-test

```
python safety_nets.py --self-test
```

Six scenarios: healthy, max_dd_breach, model_stale, confluence_disagree,
trade_cap, position_mismatch. Each is asserted to produce the expected verdict.

Run as final pre-deploy check in `deploy_v33_to_razer.ps1` step 3.
