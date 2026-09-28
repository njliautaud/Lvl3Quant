# SPY Subsecond Pivot — Architecture Survey & Plan

**Date**: 2026-06-04
**Author**: SPY-pivot sub-agent
**Trigger**: FINRA killed the $25k PDT rule today. ES taker is parked. User order: stand up SPY subsecond research and trade it.

---

## 1. Existing ES Stack (what we are re-pointing)

### 1.1 Live recorder
- **Component**: `live_trading_linux/mbo_recorder.py`
- **Inputs**: Rithmic R|Protocol WebSocket via `rithmic_client.py`. Subscribes to BBO + last-trade for a single CME instrument (e.g. `ESM6`).
- **Outputs**:
  1. Daily NPZ `data/processed/mbo_events/<YYYYMMDD>_mbo_events.npz` — 6-column float32 `events`, int64 `timestamps` (ns), placeholder NaN labels at 1s/5s/10s/30s, metadata blob.
  2. Fan-out JSONL `logs/live_events.jsonl` — one JSON record per event with `{timestamp_ns, side, action, price_ticks, size, order_id, bbo}`. Consumed by paper trader.
- **Singleton lock** on PID to prevent dual Rithmic MD sockets.
- **Flush cadence**: every `--flush-minutes` (default 5).

### 1.2 Data feed abstraction
- **Component**: `live_trading/data_feed.py`
- Defines `DataFeedBase` (`connect`, `subscribe`, `stream`, `disconnect`), `MBOEvent` dataclass, plus concrete `ReplayFeed` (NPZ replay) and `LiveFeed` (Rithmic wrapper). This is the single seam where SPY adapters plug in.

### 1.3 Inference + paper trader
- **Components**:
  - `live_trading_linux/cnn_mamba_v2_inference.py` — runs CNN-Mamba v2 on live JSONL stream
  - `live_trading_linux/paper_engine.py` (721 lines) — generic paper trader, consumes predictions
  - `live_trading_linux/paper_trading_mamba_v2.py` (1143 lines) — Mamba-specific paper trader
- Config: `configs/live_paper_trading.yaml`

### 1.4 Walk-forward training harness
- `alpha_discovery/deep_models/walkforward_oot_lean.py` — primary SLIDING window WF (HC #0)
- Variants: `walkforward_oot_lean_fast.py`, `walkforward_oot_lean_sg.py`
- All ingest the 6-col event format above (or richer 29-col `mbo_events_smart_v4` derivative)

### 1.5 Canonical 6-column event schema (the contract)
| idx | name              | meaning                                       |
| --- | ----------------- | --------------------------------------------- |
| 0   | time_delta_log    | log1p(microseconds since previous event)      |
| 1   | event_type_id     | 0=A(add) 1=C(cancel) 2=M(modify) 3=T(trade) 4=F(fill) |
| 2   | side_id           | 0=Bid 1=Ask 2=None                            |
| 3   | price_rel_ticks   | (price - mid) / tick_size                     |
| 4   | qty_log           | log(max(1, qty))                              |
| 5   | spread_ticks      | (ask - bid) / tick_size                       |

Companion `timestamps` int64 ns. Labels `labels_1s/5s/10s/30s` (float32, NaN until offline labeller runs).

### 1.6 Credentials
- `live_trading_linux/.env` — `RITHMIC_USER`, `RITHMIC_PASSWORD`, `RITHMIC_SYSTEM`, `RITHMIC_URI`. No DATABENTO / ALPACA / POLYGON keys yet.

### 1.7 Cost constants
- `constants.py` — canonical ES costs (TICK_SIZE 0.25, TICK_VALUE $12.50, COMMISSION_RT $4.70 ⇒ 0.376 ticks). Has explicit "DO NOT USE" list. **Has NO equities section.** Added by this pivot — see `cost_constants_spy.py`.

---

## 2. The pivot (what changes for SPY)

### 2.1 Instrument differences (Rithmic ES → Databento SPY)
| Field          | ES (CME)            | SPY (XNAS/ARCA equities)    |
| -------------- | ------------------- | --------------------------- |
| Tick size      | 0.25                | 0.01                        |
| Tick value     | $12.50              | $0.01 (per share)           |
| Quote unit     | contract            | share                       |
| Commissions    | $4.70 RT (AMP)      | $0.00 retail (Alpaca, etc.) |
| Spread (RTH)   | 1 tick typical      | 1 cent typical liquid       |
| Session        | 23h CME             | 09:30–16:00 ET regular      |
| Aggressor flag | Rithmic `aggressor` | Databento `side`/`flags`    |

Implication for `price_rel_ticks`: SAME math, different denominator (0.01 vs 0.25). The downstream model is scale-invariant on this axis; the cost model is NOT and must be re-derived per `cost_constants_spy.py`.

### 2.2 Schema mapping Rithmic → Databento (MBO schema)
Databento `mbo` record fields:
- `ts_event` (ns since epoch) → maps to our `timestamps`
- `action` chars `A/C/M/T/F` → maps to `event_type_id` via existing encoding `{A:0,C:1,M:2,T:3,F:4}` — IDENTICAL to ours
- `side` chars `B/A/N` → `side_id` via `{B:0,A:1,N:2}` — IDENTICAL
- `price` int64 in fixed precision (1e-9 scaling) → divide by 1e9 then by tick_size
- `size` uint32 → directly into qty_log
- `flags`, `depth`, `channel_id`, `order_id`, `ts_in_delta`, `sequence` — extras (not in our 6-col, ignored)

Conclusion: the action/side encodings already match. The only adapter work is unit conversion (price scaling, tick size) and BBO maintenance for `price_rel_ticks` / `spread_ticks`. We maintain a running best-bid/best-ask exactly like the Rithmic recorder.

### 2.3 Components changed vs. unchanged
| Component               | Action                                 |
| ----------------------- | -------------------------------------- |
| `rithmic_client.py`     | UNCHANGED (still used for ES)          |
| `mbo_recorder.py`       | UNCHANGED — SPY uses parallel script    |
| `data_feed.py`          | EXTEND — add `DatabentoLiveFeed`        |
| 6-col schema            | UNCHANGED (this is the contract)       |
| `constants.py`          | UNCHANGED. New file: `cost_constants_spy.py` |
| `walkforward_oot_lean.py`| UNCHANGED — feed it SPY-shaped NPZs   |
| `paper_engine.py`        | EXTEND — accept SPY instrument config; new Alpaca paper bridge |

---

## 3. Free-tier feed decision

**Pick: Databento free trial credits.** Reasoning:
1. SDK already installed (`databento 0.71.0`).
2. Full MBO schema identical to what we need (`A/C/M/T/F` actions, `B/A/N` sides, ns timestamps). Zero impedance.
3. New users get free trial credits (typically a few GB) — enough for multi-day SPY captures.
4. Eventually we pay the $199/mo anyway; no wasted integration work.
5. Polygon free tier is delayed and 1-min bars (useless for subsecond). IEX sandbox lacks MBO. Alpaca SIP-derived feed is BBO-only, no full book depth.

**Polygon** as fallback if Databento trial credits are exhausted before we ship paper. **Alpaca SIP** retained ONLY for paper-trading order routing (it's free and instant-approval — not used as the research feed).

---

## 4. Deliverables in this pivot

1. `docs/SPY_PIVOT_PLAN.md` — this file
2. `feeds/spy_databento_trial.py` — Databento live + historical SPY → 6-col NPZ + JSONL fan-out (mirrors `mbo_recorder.py`)
3. `feeds/schema_adapter.py` — Rithmic ↔ Databento MBO field translator + BBO tracker
4. `feeds/spy_walkforward_smoke.py` — ingest 1 day of SPY NPZ into existing WF harness shape, verify no errors. NO training launch.
5. `feeds/spy_paper_alpaca.py` — paper trader stub talking to Alpaca paper API
6. `cost_constants_spy.py` — equities cost model

---

## 5. Hard rules respected

- HC #0 SLIDING window only. Smoke test uses sliding window.
- HC #420 codebase authorization acknowledged.
- HC #433 Discord messages stay plain-English (this doc lives in `docs/`, not Discord).
- NO training launches (deliverable 4 is an INGEST smoke test only).
- MLflow logging hook included in feed scripts (`MLFLOW_TRACKING_URI` env var, lazy import).

---

## 6. Open items / blockers (escalate)

- **DATABENTO_API_KEY** not in `.env` — user must paste their trial key. Script gracefully skips live capture if absent and falls back to historical-trial test fixture.
- **ALPACA_API_KEY / ALPACA_SECRET** not in `.env` — required for paper-trade routing. Stub runs in dry-mode without them.
- **No SIP entitlement plan yet** for production. Trial-only for now.
- `alpaca_trade_api` and `polygon` Python packages NOT installed — listed in feed scripts but tolerated via lazy import.
