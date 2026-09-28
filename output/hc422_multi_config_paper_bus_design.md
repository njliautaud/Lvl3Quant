# HC #422 Rule 6 — Multi-Config Paper Bus — Design Document

**Status**: DESIGN (no code shipped yet — scaffolding plan below)
**Date**: 2026-05-18
**Author**: Claude (HC #422 deliverable)
**User mandate** (verbatim): *"We should be able to deploy multiple configs to paper to test different setups at the same time."*

---

## TL;DR

The current Razer live stack already has a publisher–subscriber pattern by accident:
`mbo_recorder.py` writes `live_events.jsonl` → each paper trader tails it independently. We're 70 % of the way there. The redesign is:

1. **Promote `live_events.jsonl` from "side effect" to "first-class MBO bus"** with a documented schema, retention policy, and offset-tracking discipline.
2. **Replace per-config standalone scripts with one generic `paper_runner.py` driven by a YAML config.** N configs = N runner processes reading the same bus, writing to their own `<config>/heartbeat.json` + `<config>/trades.jsonl`.
3. **A manifest file (`configs/paper_manifest.yaml`)** that the watchdog and launcher auto-discover.

No new infra dependencies (Redis / Kafka / etc.) — flat-file JSONL is already proven and survives reboots cleanly.

---

## Current state — what we have

```
┌─────────────────────┐
│  mbo_recorder.py    │  PID 15720 on Razer
│  → live_events.jsonl│  (Databento MBO → flat JSONL, flush every 5 min)
└──────────┬──────────┘
           │ tail-follow (file offset tracked in memory)
           │
   ┌───────┴───────┬─────────────────────────────┐
   │               │                             │
   ▼               ▼                             ▼
paper_trading_  paper_trading_v2_       (future paper config N)
mamba_v2.py     1s_short_top05.py
(legacy Top5%)  (shadow, gate-starved)
PID 25512       PID 29600

Each subscriber:
  - hardcodes its own weights path, gate floor, percentile thresholds
  - writes its own log file (different naming convention)
  - writes its own heartbeat JSON (different schema)
  - has its own argparse, its own broker glue, its own Discord notifier
```

**Pain points** that have already burned us:
- The shadow's "gate-starved 5.5h silent failure" required a custom watchdog because each subscriber's heartbeat schema is bespoke.
- Adding a new config = copying a 1242-line `paper_trading_*.py` and tweaking constants → fork drift inevitable.
- No way to A/B test parameter sweeps in paper without spawning a script per cell.

---

## Target state — Multi-Config Bus v1

```
                       ┌──────────────────────┐
                       │   mbo_recorder.py    │
                       │  → live_events.jsonl │ (canonical MBO bus)
                       │  + .offset_idx       │ (durable byte offset index, optional)
                       └──────────┬───────────┘
                                  │
                  ┌───────────────┼────────────────┬─────────────────┐
                  ▼               ▼                ▼                 ▼
            paper_runner    paper_runner     paper_runner       paper_runner
            --config        --config         --config           --config
            legacy_top05    short_top05      long_top10         long_top20_passive
            .yaml           .yaml            .yaml              .yaml

            Each runner:
              - identical script: paper_runner.py
              - all behavior in the YAML config
              - writes:  logs/<config_name>/heartbeat.json
                         logs/<config_name>/trades.jsonl
                         logs/<config_name>/decisions.jsonl  (every prediction, gated or not)
              - uniform stop/restart signal: touch logs/<config_name>/STOP
              - uniform schema watchdog can inspect generically
```

### YAML config schema (`configs/paper_<name>.yaml`)

```yaml
name: short_top05_v2
description: "HC #421 deployment candidate, 1s short, top 5% confidence gate"

# Model
weights: C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_smart_v3_mar\fold_10_best.pt
stats:   C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_smart_v3_mar\fold_09_feature_stats.npz
sha256:  300e338d3c16137fc587b10cce92204e8fe0a486fc3c7aaf53856fdd8c21281e
device:  cuda

# Strategy gate
direction:        short          # short | long | both
gate_floor_pred:  -0.6926        # pred_1s threshold
percentile_gate:  Top05          # Top05 | Top10 | Top20
min_confidence:   0.85

# Position
size_contracts:   1
exec_type:        passive_at_touch   # passive | market | sniper
hold_evals_max:   45

# Broker
account:          paper             # paper | live (live requires manual flag flip)
broker_glue:      jsonl_only        # jsonl_only | rithmic_paper | rithmic_live

# Heartbeat / health
heartbeat_path:   logs/short_top05_v2/heartbeat.json
heartbeat_every_s: 5
counters_to_track: [n_signals_total, n_signals_passed_gate, n_orders_sent, n_fills, n_pnl_realized]

# Kill-switch
kill_switch_max_drawdown_ticks: 30
kill_switch_max_consecutive_losses: 5
```

### Unified heartbeat schema (every config writes the same shape)

```json
{
  "ts": "2026-05-18T18:48:08+00:00",
  "config_name": "short_top05_v2",
  "pid": 29600,
  "rth": true,
  "events_consumed": 6831,
  "predictions_made": 6831,
  "n_signals_total": 6831,
  "n_signals_passed_gate": 0,
  "n_orders_sent": 0,
  "n_fills": 0,
  "open_position": null,
  "pnl_ticks_today": 0.0,
  "kill_switch_tripped": false,
  "last_event_ts": "2026-05-18T18:48:06+00:00",
  "last_event_lag_ms": 2100
}
```

### Manifest (`configs/paper_manifest.yaml`)

```yaml
# Watchdog + launcher both read this
active_configs:
  - configs/paper_legacy_top05.yaml
  - configs/paper_short_top05_v2.yaml
  - configs/paper_long_top10_passive.yaml   # NEW — multi-config A/B
inactive_configs:
  - configs/paper_short_top20_market.yaml   # killed by user
```

The watchdog (`live_stack_watchdog.py`) reads the manifest each cycle and auto-discovers what to track. No hard-coded process names.

---

## Migration path (incremental — does NOT break legacy +$45 winner)

**Stage 1** (1–2 hours, no live disruption): Build `paper_runner.py` as a thin wrapper that just instantiates the existing `V2_1S_Short_Top05_PaperTrader` class with constants pulled from YAML. The legacy `paper_trading_mamba_v2.py` keeps running unchanged.

**Stage 2** (1 day): Create a `paper_legacy_top05.yaml` that exactly reproduces the legacy script's params. Run it side-by-side with the legacy script for 1 day. Verify outputs match within rounding. Then retire the legacy script.

**Stage 3** (1 day): Add `paper_long_top10_passive.yaml`, `paper_short_top20_market.yaml`, etc. — the A/B parameter sweep the user actually wants.

**Stage 4** (later): Refactor `paper_runner.py` internals to remove the `V2_1S_Short_Top05_PaperTrader` class import and inline only what's needed. Drop the legacy script.

---

## What the watchdog has to learn

`live_stack_watchdog.py` v3 will read `configs/paper_manifest.yaml` instead of the hard-coded `PROCESS_REGISTRY` list. For each active config:
- Match process by `cmdline contains paper_runner.py AND --config <yaml_path>`
- Verify `heartbeat_path` is fresh (config-declared threshold, default 10 min)
- Verify `n_signals_passed_gate` > 0 within RTH (GATE-STARVED alarm — already implemented for shadow)
- Verify `kill_switch_tripped == false`
- Generic alarm format → uniform Discord output

---

## What NOT to build (yet)

- ❌ Redis / Kafka / RabbitMQ — flat JSONL is fine, we are nowhere near the throughput where it matters
- ❌ A web dashboard — the Discord + JSONL workflow is the dashboard for now
- ❌ Backtest reproduction inside paper_runner — keep that as `simulate_trades()` pure function (already exists)
- ❌ Multi-symbol — single instrument (ESM6) is plenty for the next 3 months

---

## Open questions (escalate to user before Stage 3)

1. **Capital allocation across configs** — do all N configs trade as if they each have full capital, or do they share a pool? Stage 2 says "each runs independent paper account". Stage 3+ needs a decision.
2. **Live promotion path** — when a config A/B-wins in paper, who flips the `account: paper → live` switch? Manual user approval per HC #393 escalation list ("PRODUCTION DEPLOYMENT of new live-trading strategy").
3. **Kill-switch interaction** — if config A trips kill-switch, does B keep running? Default proposal: yes, kill-switch is per-config.

---

## Effort estimate

| Stage | Hours | Risk to legacy +$45 winner |
|-------|-------|----------------------------|
| 1: paper_runner.py wrapper | 2 | None — wrapper only |
| 2: legacy parity YAML + 1-day verify | 4 | LOW — runs side-by-side |
| 3: 2–3 new A/B configs | 6 | NONE — additive |
| 4: refactor inline + retire legacy | 8 | LOW — done after verified parity |
| **Total** | **20 h** | Containable |

---

## Next concrete actions

When the autonomy task-queue (HC #422 Rule 4 sub-agent deliverable) lands, enqueue:
- `T1`: "Implement paper_runner.py wrapper" (priority 5, Razer-only)
- `T2`: "Generate paper_legacy_top05.yaml + diff against current legacy params" (priority 6, Jupiter)
- `T3`: "Add manifest support to watchdog v3" (priority 7, Razer + Jupiter)

Do NOT touch the +$45 legacy winner during Stages 1–2. It stays running unchanged.
