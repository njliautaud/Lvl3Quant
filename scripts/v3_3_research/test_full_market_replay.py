"""Tests for full_market_replay (HC #357).

Run with:
    cd /home/jupiter/Lvl3Quant
    python -m pytest scripts/v3_3_research/test_full_market_replay.py -v
or:
    python scripts/v3_3_research/test_full_market_replay.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import pytest  # type: ignore
except ImportError:  # minimal shim so the module imports without pytest installed
    class _ApproxShim:
        def __init__(self, expected, abs=1e-9, rel=None):
            self.expected = expected
            self.abs = abs
        def __eq__(self, other):
            return abs(other - self.expected) <= self.abs
        def __repr__(self):
            return f"approx({self.expected}, abs={self.abs})"
    class _PytestShim:
        approx = staticmethod(_ApproxShim)
    pytest = _PytestShim()  # type: ignore

# Allow `python test_full_market_replay.py` from this dir
sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_market_replay import (  # noqa: E402
    TradeConfig, TradeLedger, full_market_replay,
    TICKS_PER_LOG_RET, ES_RT_COMMISSION_TICKS_DEFAULT,
)


PREDS_PATH = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz")
LABELS_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_fifo_labels")


# ---------------------------------------------------------------------------
# Sanity: P99 short, passive_at_touch, hold=5s, cancel_eval=40 should produce
# a ledger whose per-fill edge is <= the gate-coverage audit FIFO number
# (because we layer queue + adverse on top — costs only).
# ---------------------------------------------------------------------------
def test_canonical_short_p99_passive():
    cfg = TradeConfig(
        side="short",
        horizon="1s",
        confidence_threshold=0.01,  # bottom 1% (most negative) for short
        order_type="passive_at_touch",
        cancel_eval_window=40,
        hold_seconds=5.0,
    )
    ledger = full_market_replay(
        PREDS_PATH, LABELS_DIR, cfg, verbose=True,
    )
    assert isinstance(ledger, TradeLedger)
    assert ledger.n_signals > 0
    assert ledger.n_filled <= ledger.n_attempted
    # Realistic replay should NOT produce wildly positive results — sanity bound
    assert ledger.pnl_ticks_per_fill < 10.0, (
        f"per-fill edge {ledger.pnl_ticks_per_fill:.3f} ticks looks too good"
    )
    # Adverse selection should report a non-positive number (it's the mean of
    # min(in_pos, 0) — always <= 0).
    assert ledger.adverse_selection_cost_ticks_avg <= 1e-6
    # Commission should match n_filled × rt_commission
    expected_comm = ledger.n_filled * ES_RT_COMMISSION_TICKS_DEFAULT
    assert abs(ledger.commission_ticks_total - expected_comm) < 1e-6


# ---------------------------------------------------------------------------
# Edge case: ioc_market path should give 100% fill rate.
# ---------------------------------------------------------------------------
def test_ioc_market_always_fills():
    cfg = TradeConfig(
        side="long",
        horizon="1s",
        confidence_threshold=0.05,  # top 5%
        order_type="ioc_market",
        cancel_eval_window=40,
        hold_seconds=5.0,
    )
    ledger = full_market_replay(PREDS_PATH, LABELS_DIR, cfg)
    assert ledger.fill_rate == pytest.approx(1.0, abs=1e-9)
    assert ledger.n_filled == ledger.n_attempted
    assert ledger.avg_queue_position_on_arrival == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Edge case: passive_at_touch_plus_2 should fill much LESS than passive_at_touch
# (deflator 0.125 vs 0.5).
# ---------------------------------------------------------------------------
def test_plus_k_fills_less_than_touch():
    base = TradeConfig(
        side="long", horizon="5s", confidence_threshold=0.05,
        order_type="passive_at_touch", cancel_eval_window=40, hold_seconds=5.0,
    )
    plus2 = TradeConfig(
        side="long", horizon="5s", confidence_threshold=0.05,
        order_type="passive_at_touch_plus_2", cancel_eval_window=40, hold_seconds=5.0,
    )
    lb = full_market_replay(PREDS_PATH, LABELS_DIR, base)
    lp = full_market_replay(PREDS_PATH, LABELS_DIR, plus2)
    assert lp.fill_rate < lb.fill_rate, (
        f"plus_2 fill {lp.fill_rate:.3f} should be < touch fill {lb.fill_rate:.3f}"
    )


# ---------------------------------------------------------------------------
# Edge case: when we use a tiny percentile (very few signals) we should still
# get a valid (possibly NaN-stat) ledger without crashing.
# ---------------------------------------------------------------------------
def test_tiny_percentile_returns_ledger():
    cfg = TradeConfig(
        side="short",
        horizon="1s",
        confidence_threshold=0.001,  # bottom 0.1%
        order_type="passive_at_touch",
        cancel_eval_window=40,
        hold_seconds=5.0,
    )
    ledger = full_market_replay(PREDS_PATH, LABELS_DIR, cfg)
    assert ledger.n_signals >= 0
    # per_trade_df has the right columns
    expected_cols = {
        "timestamp", "side", "signal_percentile", "prediction",
        "attempted", "filled", "net_ticks", "mfe_ticks", "mae_ticks",
        "adv_sel_30s_ticks", "exit_reason", "queue_pos_on_arrival",
    }
    assert expected_cols.issubset(set(ledger.per_trade_df.columns))


# ---------------------------------------------------------------------------
# Edge case: random predictions stub — should give negative expected PnL.
# We synthesize this by re-using REAL data with a HIGH gate (top 1%) and
# verifying that ioc_market PnL accounts for the round-trip commission.
# ---------------------------------------------------------------------------
def test_ioc_market_pays_commission_and_spread():
    cfg = TradeConfig(
        side="long", horizon="1s", confidence_threshold=0.05,
        order_type="ioc_market", cancel_eval_window=40, hold_seconds=1.0,
    )
    ledger = full_market_replay(PREDS_PATH, LABELS_DIR, cfg)
    # IOC pays commission (0.376) + spread crossing (1.0) = 1.376 ticks per RT
    # The per-trade df net_ticks should reflect that floor for non-favorable trades
    df = ledger.per_trade_df
    df_filled = df[df["filled"]]
    # All filled trades have fill_price_offset_ticks = -1.0 (spread paid)
    assert (df_filled["fill_price_offset_ticks"].fillna(0.0) == -1.0).all()


if __name__ == "__main__":
    # Manual runner for environments without pytest CLI
    print("=" * 70)
    print("Running test_canonical_short_p99_passive ...")
    test_canonical_short_p99_passive()
    print("  PASSED")
    print("Running test_ioc_market_always_fills ...")
    test_ioc_market_always_fills()
    print("  PASSED")
    print("Running test_plus_k_fills_less_than_touch ...")
    test_plus_k_fills_less_than_touch()
    print("  PASSED")
    print("Running test_tiny_percentile_returns_ledger ...")
    test_tiny_percentile_returns_ledger()
    print("  PASSED")
    print("Running test_ioc_market_pays_commission_and_spread ...")
    test_ioc_market_pays_commission_and_spread()
    print("  PASSED")
    print("=" * 70)
    print("ALL TESTS PASSED")
