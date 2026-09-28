"""
HC #413 — fill simulation for TP/SL scalping backtester.

Canonical FIFO market replay (HC #397B):
  - NO midpoint shortcuts.
  - Passive limit fills decided by FIFO labels (real MBO bid/ask replay).
  - Market orders fill instantly, paying ES spread crossing.

Cost constants (CANONICAL per CLAUDE.md):
  ES_TICK_VALUE        = $12.50
  ES_RT_COMMISSION     = $4.70  (= 0.376 ticks)
  passive limit total  ~ 0.376 ticks  (commission only)
  market order total   ~ 1.376 ticks  (commission + 1.0 tick spread crossing)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

LVL3 = Path("/home/jupiter/Lvl3Quant")
if str(LVL3) not in sys.path:
    sys.path.insert(0, str(LVL3))

# Reuse the canonical loader from full_market_replay (HC #397B canonical infra)
from scripts.v3_3_research.full_market_replay import _load_fifo_labels  # noqa: E402

# --- Canonical cost constants (CLAUDE.md COST CONSTANTS section) -------------
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION_DOLLARS = 4.70
ES_RT_COMMISSION_TICKS = ES_RT_COMMISSION_DOLLARS / ES_TICK_VALUE  # 0.376
ES_SPREAD_TICKS_RTH = 1.0  # book is 1 tick wide in RTH

PASSIVE_LIMIT_COST_TICKS = ES_RT_COMMISSION_TICKS              # ~0.376
MARKET_ORDER_COST_TICKS = ES_RT_COMMISSION_TICKS + ES_SPREAD_TICKS_RTH  # ~1.376

EVAL_STRIDE_SEC = 0.25
DEFAULT_CANCEL_EVAL_WINDOW = 40  # 40 * 250ms = 10s — typical of v3.3 work

# Horizons we have realized-tick fields for
HORIZONS = ("1s", "5s", "10s", "30s")
HORIZON_SEC = {"1s": 1.0, "5s": 5.0, "10s": 10.0, "30s": 30.0}


@dataclass
class FillSimConfig:
    """Configuration for entry-fill simulation.

    order_type:
        "passive_at_touch" -> FIFO-modeled passive fill, pay commission only
        "market" / "ioc_market" -> always fills, pay commission + spread
    cancel_eval_window: number of 250ms evals before cancelling a passive order.
    side: "long" or "short" — picks which FIFO bracket label to consult.
    """
    order_type: str = "passive_at_touch"
    cancel_eval_window: int = DEFAULT_CANCEL_EVAL_WINDOW
    side: str = "long"


def entry_cost_ticks(order_type: str) -> float:
    """Round-trip entry cost (commission + crossing if market)."""
    if order_type in ("market", "ioc_market", "mkt"):
        return MARKET_ORDER_COST_TICKS
    if order_type in ("passive_at_touch", "passive", "limit"):
        return PASSIVE_LIMIT_COST_TICKS
    raise ValueError(f"unknown order_type: {order_type}")


def load_fifo_for_dates(labels_dir: Path, dates: Sequence[str]) -> dict:
    """Thin wrapper around the canonical FIFO loader (HC #397B)."""
    return _load_fifo_labels(Path(labels_dir), dates)


def passive_fill_mask(
    fifo: dict,
    side: str,
    cancel_eval_window: int = DEFAULT_CANCEL_EVAL_WINDOW,
    seed: int = 42,
) -> np.ndarray:
    """Return boolean array (N,) — True where a passive_at_touch order
    would have filled within cancel_eval_window per FIFO labels.

    Model (matches canonical replay in full_market_replay._queue_position_model):
      - label_filled must be True (real MBO replay said price level traded)
      - label_hold_time_ns / 1e9 <= 4 * cancel_eval_window * EVAL_STRIDE_SEC
        (proxy for "fill happened early enough in our window")
      - deflate by 0.5 (mean-of-queue heuristic) using deterministic RNG with
        slow-exit (max_hold) penalised to 0.5 * 0.25 = 0.125
    """
    side_key = side.lower()
    if side_key not in ("long", "short"):
        raise ValueError(f"side must be long/short, got {side}")
    filled_lbl = fifo[f"tp4sl3_{side_key}_filled"].astype(bool)
    exit_reason = fifo[f"tp4sl3_{side_key}_exit_reason"]
    hold_time_ns = fifo[f"tp4sl3_{side_key}_hold_time_ns"].astype(np.int64)

    n = filled_lbl.shape[0]
    cancel_sec = cancel_eval_window * EVAL_STRIDE_SEC
    hold_sec = hold_time_ns / 1e9
    base = filled_lbl & (hold_sec <= 4 * cancel_sec)

    rng = np.random.default_rng(seed)
    coin = rng.random(n)
    slow = (exit_reason == "max_hold")
    deflator = 0.5
    effective = np.where(slow, deflator * 0.25, deflator)
    return base & (coin < effective)


def market_fill_mask(n: int) -> np.ndarray:
    """Market orders always fill."""
    return np.ones(n, dtype=bool)


def entry_filled_mask(
    fifo: dict,
    cfg: FillSimConfig,
    seed: int = 42,
) -> np.ndarray:
    """Top-level dispatch for entry fill given FillSimConfig."""
    ot = cfg.order_type
    if ot in ("market", "ioc_market", "mkt"):
        return market_fill_mask(fifo["tp4sl3_long_filled"].shape[0])
    if ot in ("passive_at_touch", "passive", "limit"):
        return passive_fill_mask(fifo, cfg.side, cfg.cancel_eval_window, seed=seed)
    raise ValueError(f"unknown order_type: {ot}")
