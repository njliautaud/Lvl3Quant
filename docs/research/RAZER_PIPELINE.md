# RAZER LIVE STACK — PIPELINE DOC (HC #210)

**Process**: `paper_trading_mamba_v2_patched.py` (PID resolved via `Get-CimInstance` filter on cmdline)
**Source**: `C:\Users\claude\Lvl3Quant\paper_trading_mamba_v2_patched.py` (846 lines)
**Log**: `C:\Users\claude\Lvl3Quant\logs\paper_trader_live.out.log`
**Heartbeat**: `C:\Users\claude\Lvl3Quant\status\razer_heartbeat.json` (schema=`v2_per_stage`, written every 60s by Windows scheduled task `RazerHeartbeat`)
**Watchdog**: Windows scheduled task `RazerWatchdog` (60s, 5min cooldown, 3-fail alert threshold)
**Single-command query** (from Jupiter): `python3 /home/jupiter/Lvl3Quant/cluster_status.py --razer`

The pipeline is 10 stages. Each emits a heartbeat entry under `stages.<name>` with `{status, last_activity_ts, age_s, throughput, notes}`. A stage is **GREEN** iff its last activity is within its expected cadence and no recent error.

---

## Stage 1 — `ingest`
**What it does**: Maintains Rithmic MD WebSocket (wss://rprotocol.rithmic.com:443), receives ESU6@CME tick stream, plus separate ORDER socket for paper account XXXXXX.
**Code**: `live_trading/rithmic_client.py` (handshake, heartbeat, MBO subscription).
**Heartbeat green when**: `rithmic_md=ok` AND log mtime <600s (STATUS line every 5min so >10min = stale).
**Throughput**: events_per_min (computed from STATUS-line delta vs previous heartbeat).
**Failure modes**: socket disconnect (rp_code logged), license stolen by another instance (rp_code 1067), heartbeat timeout.
**Recovery**: watchdog re-launches paper_trader if pid missing; Rithmic license is single-instance, so duplicate connection attempts fail by design (HC #204).

## Stage 2 — `preprocess`
**What it does**: Raw Rithmic MBO event → smart_v3 25-feature window. Builds `FeatureBuilder` rolling buffer of 5000 events for warmup before any inference.
**Code**: paper_trader `self.features.update(event)` → smart_v3 features.
**Heartbeat green when**: warmup_pct=100% AND log mtime <600s.
**Heartbeat warming when**: events_seen < warmup_target (5000).
**Throughput**: events_seen (cumulative), events_per_min, warmup_pct.
**Notes**: warmup typically ~5-15 min depending on market activity. Empty market → slow warmup.

## Stage 3 — `signal_cnn_mamba`
**What it does**: Primary signal model. `CNNMambaV2Inference.update(features)` runs on RTX 3070 CUDA, produces `{tier, confidence, signal}` at every stride (250ms predictions).
**Weights**: `fold_10_best.pt` (CNN-Mamba v2, IC_1s=0.222, no decay, signal edge ~30s).
**Heartbeat green when**: post-warmup AND `preds > 0` AND log fresh.
**Heartbeat idle when**: warmup not done OR no prediction yet.
**Throughput**: preds_per_min, preds_total, last_signal `{side, conf, tier}`.

## Stage 4 — `signal_patchtst`
**What it does**: Secondary confluence model (PatchTST). Currently DISABLED due to head shape mismatch (128 vs 256) on Razer weights.
**Heartbeat status**: always `disabled` until weights fixed.
**Notes**: Single-model mode is acceptable per directive — known issue tracked, not blocking.

## Stage 5 — `confluence`
**What it does**: Combines signal models into final SIGNAL events. With PatchTST disabled, this is equivalent to CNN-Mamba's tier output passed through min-tier filter.
**Heartbeat green when**: producing signals at expected ratio (signals_total > 0).
**Heartbeat idle when**: no preds yet.
**Throughput**: signals_per_min, signal_pct_of_preds (e.g., Top5% min-tier → ~5% pass-through).

## Stage 6 — `exec_rl_entry` (NOT LOADED)
**What it does (when deployed)**: DQN entry Q-network decides WHEN to enter given a confluence SIGNAL. Trained on Neptune.
**Current state**: `not_loaded` — DQN is still training on Neptune (split_dqn_v1_r22). Current paper trader uses **rules-based entry**: tier filter (≥min_tier) + risk gates only.
**Heartbeat status**: always `not_loaded` until `--rl-entry-weights` flag added on next paper_trader launch.

## Stage 7 — `exec_rl_exit` (NOT LOADED)
**What it does (when deployed)**: DQN exit Q-network decides WHEN to exit. Currently rules-based (max_hold_time + risk thresholds).
**Heartbeat status**: always `not_loaded` until DQN training completes + weights deployed to Razer.

## Stage 8 — `gates_filters`
**What it does**: Final filtering before order submission. Applies:
  - **Tier filter**: `_tier_meets_min` (e.g., min_tier=Top5% means only top 5% confidence signals pass).
  - **Risk gates**: `_risk_exit` triggers, max_spread cap (default 1.5 ticks), daily P&L cap, drawdown.
  - **Re-entry blocks**: time-since-last-exit, flip cool-down.
**Heartbeat green when**: post-warmup AND log fresh.
**Throughput**: blocked_total (counter of blocked entries), min_tier, last_block (kind + reason).

## Stage 9 — `order_submission`
**What it does**: PAPER mode — `PaperPosition.enter/exit` writes ENTRY/EXIT lines to log and signals JSONL. Rithmic ORDER socket is connected (account XXXXXX, route=eurex) but **no real orders sent**.
**Heartbeat green when**: order_socket=ok AND signals leading to fills (or signals_total=0 and stage=idle is also acceptable until first signal).
**Throughput**: trades_per_min, trades_total, win_rate_pct, last_entry, last_exit.
**Failure modes**: order_socket disconnect (would auto-recover via watchdog); paper-mode mismatch on Rithmic (route=eurex on ES — known paper-only quirk, doesn't matter for paper).

## Stage 10 — `risk_monitor`
**What it does**: Background risk supervisor. Each loop iteration checks max_hold_time (default 60s), daily P&L cap, drawdown threshold. Triggers `_risk_exit` when breached.
**Heartbeat green when**: log fresh AND pid alive.
**Throughput**: risk_exits_total, blocked_total, pnl_usd, daily_pnl_usd.
**Notes**: HC #46 risk gates. risk_exits is a counter, not a fail signal — exiting on max_hold_time is normal.

---

## Status Vocabulary

| Status | Icon | Meaning |
|--------|------|---------|
| `green` | 🟢 | Stage active and within expected cadence |
| `warming` | 🟡 | Stage is starting up (e.g., preprocess accumulating events) |
| `idle` | 🟡 | Stage is ready but no work yet (e.g., warmup not done) |
| `disabled` | ⚫ | Stage intentionally off (PatchTST head mismatch known) |
| `not_loaded` | ⚫ | Component not deployed in current paper_trader (DQN entry/exit) |
| `stale` | 🟠 | Log frozen, no heartbeat data |
| `down` | 🔴 | Stage crashed or pid missing |
| `error` | 🔴 | Recent traceback in log |

## Overall Health Aggregation

`health` field in heartbeat:
- `ok` if all relevant stages green (disabled/not_loaded ignored)
- `warming` if any user-facing stage is warming or idle
- `degraded` if any stage is stale/down/error
- `down` if pid missing

## Operational Procedures

**Query state**: `python3 /home/jupiter/Lvl3Quant/cluster_status.py --razer`
**Raw heartbeat**: `ssh claude@razer 'type C:\Users\claude\Lvl3Quant\status\razer_heartbeat.json'`
**Watchdog log**: `ssh claude@razer 'type C:\Users\claude\Lvl3Quant\status\watchdog.log'`
**Force re-launch**: paper_trader exits → RazerWatchdog re-launches within 60s. Manual: `cmd /c launch_paper_trader.bat`.
**Tail live log**: `ssh claude@razer 'powershell Get-Content C:\Users\claude\Lvl3Quant\logs\paper_trader_live.out.log -Tail 30 -Wait'`

## Known Acceptable States

- `signal_patchtst=disabled` — head shape mismatch, signal flow continues with cnn_mamba alone
- `exec_rl_entry=not_loaded` and `exec_rl_exit=not_loaded` — DQN still training on Neptune, rules-based execution active
- `route=eurex` on ESU6 — Rithmic paper account quirk, no real-order impact
- `last_error` containing the PatchTST load failure — single occurrence at startup, non-fatal

## When to Alert User

The watchdog writes `status\razer_alert.json` only after **3 consecutive failures**. Crons read this file and DM the user. Healthy state = silent.
