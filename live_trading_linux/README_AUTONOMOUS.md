# Live Rithmic Pipeline — Autonomous Build Report

## What this is

A Linux-native, async Python pipeline that connects to Rithmic (AMP paper),
streams BBO + last-trade data for a single futures contract, computes the
exact 21-feature vector our LGBM model was trained on, runs streaming
predictions, and routes directional market orders to paper trading.

Directory: `/home/jupiter/Lvl3Quant/live_trading_linux/`

## Architecture

```
           +--------------------+
           |  Rithmic WS (AMP)  |   TICKER_PLANT + ORDER_PLANT
           +---------+----------+
                     |
                     v
           +--------------------+        protobuf/base_pb2
           |  RithmicClient     |  ---+  wss + ssl cert auth
           |  rithmic_client.py |     |  heartbeat loop (asyncio task)
           +---------+----------+     |  logs -> logs/rithmic.log
                     |
       MD events     |   OrderEvents
                     v
           +--------------------+
           |  SignalEngine      |  signal_engine.py
           |  - BBO -> add evt  |  Orchestrator.  Converts top-of-book updates
           |  - trade -> trade  |  and trades into (event_type, side, price,
           |  - synth td/spread |  qty, spread, time_delta) rows matching the
           +---------+----------+  training data schema.
                     |
                     v
           +--------------------+
           |  StreamingFeatures |  streaming_features.py
           |  (21 features,     |  deque-based rolling sums that mirror
           |   causal rolling)  |  compute_derived() from
           +---------+----------+  lgbm_prod_wf_60_5.py (batch version),
                     |              verified by a parity self-test.
                     v
           +--------------------+
           |  LGBMInference     |  lgbm_inference.py
           |  .pkl -> pred      |  Plus percentile-based tier classifier
           |  .predict_tier()   |  (all / top50 / top25 / top10).
           +---------+----------+
                     |
                     v
           +--------------------+
           |  Position mgr      |  max 1 contract long/short
           |  flip / 30s timeout|  fires MARKET orders via RithmicClient
           +--------------------+
```

## Files written tonight

| File | Purpose |
|------|---------|
| `__init__.py` | Marks the directory as a Python package |
| `rithmic_client.py` | Async WebSocket client wrapping R|Protocol API (login, MD subscribe, submit_order, heartbeat, disconnect). Callbacks for MD + order events. |
| `streaming_features.py` | `StreamingFeatures` class: 21-dim feature vector updated incrementally. Includes a batch-parity self-test (`python -m live_trading_linux.streaming_features`). |
| `lgbm_inference.py` | Joblib loader + tier classifier based on |pred| percentiles. |
| `signal_engine.py` | Orchestrator + CLI. Wires everything together, writes signals to JSONL, routes orders. |
| `SETUP.md` | Step-by-step setup: env vars, model copy, calibration, run. |
| `README_AUTONOMOUS.md` | This file. |
| `logs/` | Empty dir for runtime logs. |

## What the user must provide before live trading

These are the only things NOT automated — you need real AMP account data:

1. **Rithmic paper credentials from AMP** (check your AMP onboarding email):
   * `RITHMIC_SYSTEM` — the exact system name string (e.g. "Rithmic Paper Trading",
     "Rithmic 01", etc.). Run the `SampleMD.py` with only the URI to list
     available systems; the correct name is whichever one grants paper access.
   * `RITHMIC_USER` — your AMP user id
   * `RITHMIC_PASSWORD` — your AMP password
   * `RITHMIC_URI` — almost certainly `wss://rituz00100.rithmic.com:443` for
     paper trading; verify with AMP if different.

2. **Confirm the symbol to trade.** The training data was continuous ES futures;
   for live you need the specific contract month (e.g. `ESZ5` for Dec 2025,
   `ESH6` for Mar 2026). Use the CME exchange code `CME`. Note that AMP
   requires the literal Rithmic symbol — not `ES` or `/ES`.

3. **A trained LGBM `.pkl` model from Saturn.** Specifically a `labels_10s`
   PKL from the latest fold of `lgbm_prod_wf_60_5_output/`. See SETUP.md §2
   for the scp command.

4. **(Optional but recommended) A calibration JSON.** Built in 10 seconds from
   the matching `*_preds.npz`. See SETUP.md §3.

## What was NOT done (deliberately out of scope tonight)

These are tracked as TODOs in `SETUP.md`:

* **Full MBO subscription (template 318).** We only subscribe to BBO + trades;
  synthesised add/cancel events will be lossier than the Databento MBO stream
  used in training. This is the single biggest source of potential live-vs-
  offline IC drift. Plan: after first paper run, add `MARKET_BY_ORDER`
  subscription and rewire `_handle_bbo` / add a dedicated MBO handler.
* **Reconnect logic.** Signal engine stops on WS disconnect. Add an outer
  retry loop once we've validated a clean paper run.
* **Fill reconciliation.** We track position optimistically (assume the
  MARKET order fills). Fine for paper, dangerous for live. To fix, update
  `_on_order` to mutate `self.position` based on `ExchangeOrderNotification`
  fills.
* **Multi-symbol support.** Current design is one engine per symbol; scale up
  by running N `signal_engine` processes or refactoring `StreamingFeatures`
  to key on (symbol,).

## How to validate before trading real money

1. **Run the feature parity self-test:**
   ```bash
   python3 -m live_trading_linux.streaming_features
   ```
   Every feature should report `OK` with tiny diffs. `cum_delta` may need the
   looser tolerance path — this is expected due to the `sqrt(var - mean^2)`
   precision gap between f4 convolution and Python-float running sums.

2. **Dry-run the signal engine against paper:**
   ```bash
   python3 -m live_trading_linux.signal_engine \
       --model .../labels_10s_lgbm.pkl \
       --calibration .../labels_10s_calibration.json \
       --symbol ESZ5 --exchange CME \
       --dry-run --paper
   ```
   Watch `logs/signals_ESZ5.jsonl` grow — one line per emitted signal.
   Confirm the signal rate is sane (we expect a few per minute on ES).

3. **Remove `--dry-run` only once you've verified orders in AMP's UI.**

## Files referenced but not modified

* Rithmic samples + protobufs:
  `/home/jupiter/teleclaude-main/downloads/rithmic_api/0.89.0.0/samples/samples.py/`
* Training reference (for `compute_derived` parity):
  `/home/saturn/Lvl3Quant/lgbm_prod_wf_60_5.py`
