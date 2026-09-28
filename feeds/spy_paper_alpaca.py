#!/usr/bin/env python3
"""spy_paper_alpaca.py — SPY paper-trading bridge to Alpaca paper API.

Consumes prediction JSONL emitted by the inference daemon, decides on
paper orders against SPY cost rules, and routes them to Alpaca's paper
endpoint via the REST/WS API.

If ALPACA_API_KEY / ALPACA_SECRET are absent, runs in DRY-RUN mode:
prints intended orders to stdout + a paper-trade log; never hits the network.

This is a STUB — the heavy execution logic still lives in
`live_trading_linux/paper_engine.py` and `paper_trading_mamba_v2.py`. Once we
have one full SPY trading day captured + a SPY-trained model, we will wire
those engines to read this paper trader's order/fill stream instead of the
Rithmic-specific one. For now, this stub validates:

  * the prediction → decision → Alpaca-order JSON path,
  * the SPY cost model is applied correctly,
  * the order log shape is compatible with the existing position_manager.

HC compliance:
  - NO live money. ALPACA_PAPER_ENDPOINT only.
  - Per-trade size capped to PAPER_MAX_SHARES.
  - All decisions logged to live_trading_linux/logs/paper_trades_spy.jsonl.

Usage:
  python feeds/spy_paper_alpaca.py --dry-run --signal-source fixture
  python feeds/spy_paper_alpaca.py \
      --pred-jsonl live_trading_linux/logs/live_predictions_spy.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

LVL3 = Path("/home/jupiter/Lvl3Quant")
if str(LVL3) not in sys.path:
    sys.path.insert(0, str(LVL3))
import cost_constants_spy as costs

LOG_DIR = LVL3 / "live_trading_linux" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRADE_LOG = LOG_DIR / "paper_trades_spy.jsonl"

ALPACA_PAPER_ENDPOINT = "https://paper-api.alpaca.markets"

log = logging.getLogger("spy_paper")
log.setLevel(logging.INFO)
for _h in (logging.FileHandler(LOG_DIR / "spy_paper_alpaca.log"),
           logging.StreamHandler(sys.stdout)):
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(_h)

# Risk caps (paper)
PAPER_MAX_SHARES = 200          # max 200 SPY/order = ~$116k notional
PAPER_MIN_CONFIDENCE = 0.7      # only act on top-decile signals
PAPER_MAX_POSITION = 1000       # absolute share limit


@dataclass
class PaperOrder:
    ts_ns: int
    symbol: str
    side: str         # "buy" or "sell"
    qty: int
    type: str         # "limit" / "market"
    limit_price: Optional[float]
    time_in_force: str
    client_order_id: str
    note: str


def decide_order(pred: dict, current_position: int) -> Optional[PaperOrder]:
    """Translate a prediction record into a paper order. Returns None for no-op.

    Expected `pred` shape (compatible with existing inference daemon output):
        {
          "ts_ns": int,
          "symbol": "SPY",
          "horizon": "1s" | "5s" | "10s",
          "score": float (+ long bias, - short bias),
          "confidence": float in [0,1],
          "bbo": {"bid_price": float, "ask_price": float}
        }
    """
    conf = float(pred.get("confidence", 0.0))
    if conf < PAPER_MIN_CONFIDENCE:
        return None

    score = float(pred.get("score", 0.0))
    bbo = pred.get("bbo", {})
    bid = float(bbo.get("bid_price", 0.0))
    ask = float(bbo.get("ask_price", 0.0))
    if bid <= 0 or ask <= 0:
        return None

    # Position-aware sizing: scale shares by confidence
    base_qty = int(round(PAPER_MAX_SHARES * conf))
    if score > 0:  # long signal
        if current_position >= PAPER_MAX_POSITION:
            return None
        qty = min(base_qty, PAPER_MAX_POSITION - current_position)
        side = "buy"
        # Passive limit at bid (free per cost model)
        limit_price = bid
    else:          # short signal
        if current_position <= -PAPER_MAX_POSITION:
            return None
        qty = min(base_qty, PAPER_MAX_POSITION + current_position)
        side = "sell"
        limit_price = ask
    if qty <= 0:
        return None

    return PaperOrder(
        ts_ns=int(pred["ts_ns"]),
        symbol=pred.get("symbol", "SPY"),
        side=side,
        qty=qty,
        type="limit",
        limit_price=round(limit_price, 2),
        time_in_force="day",
        client_order_id=f"spy-{int(pred['ts_ns'])}-{side}",
        note=f"conf={conf:.2f} score={score:+.3f} h={pred.get('horizon','?')}",
    )


def submit_to_alpaca(order: PaperOrder, dry: bool) -> dict:
    """Submit to Alpaca paper. Returns response dict (or stub if dry)."""
    if dry:
        return {"status": "dry_run_accepted", "order": asdict(order)}

    key = os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("ALPACA_SECRET")
    if not (key and sec):
        log.warning("Alpaca creds missing — falling back to dry_run")
        return {"status": "dry_run_no_creds", "order": asdict(order)}

    try:
        import requests
    except ImportError:
        log.warning("requests not installed — falling back to dry_run")
        return {"status": "dry_run_no_requests", "order": asdict(order)}

    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": sec,
        "Content-Type": "application/json",
    }
    payload = {
        "symbol": order.symbol,
        "qty": order.qty,
        "side": order.side,
        "type": order.type,
        "limit_price": order.limit_price,
        "time_in_force": order.time_in_force,
        "client_order_id": order.client_order_id,
    }
    try:
        r = requests.post(f"{ALPACA_PAPER_ENDPOINT}/v2/orders",
                          headers=headers, json=payload, timeout=5)
        return {"status_code": r.status_code, "body": r.json()}
    except Exception as e:
        log.error("Alpaca submit failed: %s", e)
        return {"status": "error", "error": str(e), "order": asdict(order)}


def expected_cost_ticks(order_type: str = "limit") -> float:
    """Use the SPY cost model for accounting on each order."""
    if order_type == "limit":
        return costs.round_trip_cost_ticks("passive")
    return costs.round_trip_cost_ticks("aggressive")


def log_trade(order: PaperOrder, response: dict):
    rec = {"order": asdict(order), "response": response,
           "expected_rt_cost_ticks": expected_cost_ticks(order.type)}
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# -----------------------------------------------------------------------
# Fixture predictions — for smoke-testing without the inference daemon
# -----------------------------------------------------------------------
def fixture_predictions(n: int = 5):
    base_ts = int(time.time() * 1e9)
    samples = [
        {"ts_ns": base_ts + 1_000_000_000, "symbol": "SPY",
         "horizon": "1s", "score": +0.42, "confidence": 0.88,
         "bbo": {"bid_price": 580.10, "ask_price": 580.11}},
        {"ts_ns": base_ts + 2_000_000_000, "symbol": "SPY",
         "horizon": "1s", "score": -0.31, "confidence": 0.75,
         "bbo": {"bid_price": 580.09, "ask_price": 580.10}},
        {"ts_ns": base_ts + 3_000_000_000, "symbol": "SPY",
         "horizon": "5s", "score": +0.05, "confidence": 0.55,  # below thresh
         "bbo": {"bid_price": 580.09, "ask_price": 580.10}},
        {"ts_ns": base_ts + 4_000_000_000, "symbol": "SPY",
         "horizon": "1s", "score": -0.48, "confidence": 0.92,
         "bbo": {"bid_price": 580.08, "ask_price": 580.09}},
        {"ts_ns": base_ts + 5_000_000_000, "symbol": "SPY",
         "horizon": "10s", "score": +0.22, "confidence": 0.81,
         "bbo": {"bid_price": 580.10, "ask_price": 580.11}},
    ]
    yield from samples[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip Alpaca network calls (default if no creds)")
    ap.add_argument("--signal-source", choices=["fixture", "jsonl"],
                    default="fixture")
    ap.add_argument("--pred-jsonl", default=None,
                    help="Path to predictions JSONL (when signal-source=jsonl)")
    args = ap.parse_args()

    dry = args.dry_run or not (os.environ.get("ALPACA_API_KEY")
                               and os.environ.get("ALPACA_SECRET"))
    if dry:
        log.info("DRY-RUN mode (no Alpaca network calls)")
    else:
        log.info("LIVE PAPER mode (Alpaca paper endpoint)")

    if args.signal_source == "fixture":
        preds = list(fixture_predictions(5))
    else:
        if not args.pred_jsonl:
            ap.error("--pred-jsonl required when signal-source=jsonl")
        preds = []
        with open(args.pred_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    preds.append(json.loads(line))

    position = 0
    n_decided = n_skipped = 0
    for p in preds:
        order = decide_order(p, position)
        if order is None:
            n_skipped += 1
            continue
        n_decided += 1
        position += order.qty if order.side == "buy" else -order.qty
        resp = submit_to_alpaca(order, dry)
        log_trade(order, resp)
        log.info("ORDER %s %s %d @%s  conf-driven  resp=%s",
                 order.side.upper(), order.symbol, order.qty,
                 order.limit_price, resp.get("status") or resp.get("status_code"))

    summary = {
        "mode": "dry_run" if dry else "alpaca_paper",
        "predictions_in": len(preds),
        "orders_submitted": n_decided,
        "predictions_skipped": n_skipped,
        "ending_position": position,
        "trade_log": str(TRADE_LOG),
    }
    log.info("SUMMARY: %s", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
