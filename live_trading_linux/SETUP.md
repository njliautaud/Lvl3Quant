# Live Trading on Linux — Setup Guide

This directory contains a Linux-native Rithmic paper-trading pipeline:

    Rithmic WS  →  streaming features  →  LGBM model  →  paper orders

## 1. Environment variables (Rithmic credentials)

```bash
# AMP paper trading (verify with your AMP onboarding email)
export RITHMIC_URI="wss://rituz00100.rithmic.com:443"
export RITHMIC_SYSTEM="Rithmic Paper Trading"   # or whatever AMP told you
export RITHMIC_USER="your_amp_user_id"
export RITHMIC_PASSWORD="your_amp_password"

# Optional: override where we find *_pb2.py modules + SSL cert
export RITHMIC_PB_DIR="/home/jupiter/teleclaude-main/downloads/rithmic_api/0.89.0.0/samples/samples.py"
```

To *list* available systems (first-time sanity check), you can run the
stock sample with only the URI:

```bash
cd /home/jupiter/teleclaude-main/downloads/rithmic_api/0.89.0.0/samples/samples.py
python3 SampleMD.py wss://rituz00100.rithmic.com:443
```

## 2. Copy the trained LGBM model from Saturn

The training pipeline writes per-fold, per-horizon PKLs:

```
/home/saturn/lgbm_prod_wf_60_5_output/fold{NN}/{horizon}_lgbm.pkl
/home/saturn/lgbm_prod_wf_60_5_output/fold{NN}/{horizon}_preds.npz
```

Pick the LATEST fold (highest NN) and the `labels_10s` horizon, since IC_10s
is our default reporting timeframe.

```bash
# From Jupiter:
mkdir -p /home/jupiter/Lvl3Quant/live_trading_linux/models
scp saturn:/home/saturn/lgbm_prod_wf_60_5_output/fold11/labels_10s_lgbm.pkl \
    /home/jupiter/Lvl3Quant/live_trading_linux/models/
scp saturn:/home/saturn/lgbm_prod_wf_60_5_output/fold11/labels_10s_preds.npz \
    /home/jupiter/Lvl3Quant/live_trading_linux/models/
```

## 3. Build a calibration file (one-time)

Tier gating depends on percentile thresholds of |pred|.  Use the preds.npz
from the chosen fold to compute them:

```python
import numpy as np, json
from live_trading_linux.lgbm_inference import LGBMInference

preds = np.load('models/labels_10s_preds.npz')['preds']
cal   = LGBMInference.build_calibration_from_preds(preds)
with open('models/labels_10s_calibration.json', 'w') as f:
    json.dump(cal, f, indent=2)
print(cal)
```

## 4. Run the feature-parity self-test (CRITICAL)

Before trading a dollar, confirm the streaming features match the batch
computation used in training:

```bash
cd /home/jupiter/Lvl3Quant
python3 -m live_trading_linux.streaming_features
```

Expected: every feature reports `[OK ]` with max|diff| below tolerance.
`cum_delta` may need the looser tolerance path; that's normal due to float
precision in the std-dev subtraction.

## 5. Start the signal engine

```bash
cd /home/jupiter/Lvl3Quant
python3 -m live_trading_linux.signal_engine \
    --model       live_trading_linux/models/labels_10s_lgbm.pkl \
    --calibration live_trading_linux/models/labels_10s_calibration.json \
    --symbol      ESZ5 \
    --exchange    CME \
    --threshold   0.00 \
    --min-tier    top25 \
    --timeout     30 \
    --paper
```

Add `--dry-run` to compute signals without sending any orders — useful for a
first live test.

Logs:

* `logs/rithmic.log` — WS + protobuf activity
* `logs/signal_engine.log` — engine decisions
* `logs/signals_ESZ5.jsonl` — one JSON line per emitted signal

## Known gaps / v1 limitations

1. **MBO (template 318) is NOT wired.** We only subscribe to BBO + last-trade.
   BBO updates only fire when the top of book changes, so the synthetic "add"
   events we feed into the feature extractor are lossy vs. training data
   (which was built from Databento MBO events). Expect IC to be lower live
   than offline. To close this gap, subscribe to `MARKET_BY_ORDER` and map
   its add/cancel/trade notifications into the feature stream.

2. **No depth-of-book.** Spread is computed as `ask - bid`; book imbalance
   features (rolling_ofi, qty_add_imbalance, etc.) are approximated from top
   of book only.

3. **Aggressor inference fallback** — when Rithmic leaves the aggressor flag
   unset on a trade, we guess by comparing trade_price to the last known BBO.

4. **Position sizing is hard-capped at 1 contract.** Exits fire on either a
   flipped signal or a 30-second timeout; no stop-loss, no take-profit.

5. **Trade-route selection is naive.** The client uses the first trade_route
   returned by Rithmic for the account's FCM/IB. For exotic exchanges you may
   need to pass `trade_route=` explicitly to `submit_order`.

6. **Reconnection is not yet implemented** — if the WebSocket drops, the
   engine currently stops. Wrap `engine.run()` in an outer retry loop if
   you need overnight reliability.
