#!/usr/bin/env python3
"""
HC #437 Bug 2 — unit test for the HC #413 bracket-exit resolver added to
FIFOReplayEngine.

Tests three synthetic positions with known label values and verifies:
  1. SL-first ordering (SL hit beats TP at same checkpoint)
  2. TP2 preferred over TP1 when both met
  3. Time-stop emitted when no horizon hits
  4. Long vs short sign handling

Run:  python3 scripts/v3_4_research/hc437_test_bracket_resolver.py
Exit: 0 on success, 1 on any assertion failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))

from alpha_discovery.deep_models.fifo_market_replay import _resolve_hc413_bracket


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def test_short_sl_first_at_1s():
    """Short signal, 1s label = +2.0 ticks (adverse for short). SL=0.5.
    -> inpos(1s) = -2.0 <= -0.5 -> SL hit, gross = -0.5, hold_sec = 1."""
    reason, gross, hold = _resolve_hc413_bracket(
        direction='short',
        labels_by_h={1: +2.0, 5: -3.0, 10: -3.0},  # would TP2 at 5s if not for SL
        tp1=0.5, tp2=1.0, sl=0.5,
    )
    assert reason == 'sl', f"expected sl, got {reason}"
    assert approx(gross, -0.5), f"gross={gross}"
    assert approx(hold, 1.0), f"hold={hold}"
    print(f"  [OK] short_sl_first_at_1s: reason={reason} gross={gross:+.3f} hold={hold}s")


def test_long_tp2_at_5s():
    """Long signal, 1s = +0.4 (below TP1), 5s = +1.2 (TP2 hit since >=1.0).
    Expected: tp2 at 5s, gross=+1.0."""
    reason, gross, hold = _resolve_hc413_bracket(
        direction='long',
        labels_by_h={1: +0.4, 5: +1.2, 10: +1.5},
        tp1=0.5, tp2=1.0, sl=0.5,
    )
    assert reason == 'tp2', f"expected tp2, got {reason}"
    assert approx(gross, +1.0), f"gross={gross}"
    assert approx(hold, 5.0), f"hold={hold}"
    print(f"  [OK] long_tp2_at_5s: reason={reason} gross={gross:+.3f} hold={hold}s")


def test_long_tp1_at_1s_when_tp2_not_met():
    """Long, 1s = +0.6 (TP1 met, not TP2). Should TP1 at 1s, not wait for 5s."""
    reason, gross, hold = _resolve_hc413_bracket(
        direction='long',
        labels_by_h={1: +0.6, 5: +1.5, 10: +1.5},
        tp1=0.5, tp2=1.0, sl=0.5,
    )
    assert reason == 'tp1', f"expected tp1, got {reason}"
    assert approx(gross, +0.5), f"gross={gross}"
    assert approx(hold, 1.0), f"hold={hold}"
    print(f"  [OK] long_tp1_at_1s_when_tp2_not_met: reason={reason} gross={gross:+.3f} hold={hold}s")


def test_short_time_stop():
    """Short, all labels between -SL and +TP1. -> time_stop at 10s with
    gross = inpos(10s) = -0.2 (small favorable for short = +0.2)."""
    reason, gross, hold = _resolve_hc413_bracket(
        direction='short',
        labels_by_h={1: +0.1, 5: +0.0, 10: -0.2},
        tp1=0.5, tp2=1.0, sl=0.5,
    )
    # short sign-flip: inpos(10s) = -(-0.2) = +0.2 (favorable, but below TP1=0.5)
    assert reason == 'time_stop', f"expected time_stop, got {reason}"
    assert approx(gross, +0.2), f"gross={gross} (expected +0.2 favorable)"
    assert approx(hold, 10.0), f"hold={hold}"
    print(f"  [OK] short_time_stop: reason={reason} gross={gross:+.3f} hold={hold}s")


def test_no_label_returns_special_code():
    """All labels NaN/missing -> no_label, drop the trade."""
    reason, gross, hold = _resolve_hc413_bracket(
        direction='long',
        labels_by_h={},
        tp1=0.5, tp2=1.0, sl=0.5,
    )
    assert reason == 'no_label', f"expected no_label, got {reason}"
    print(f"  [OK] no_label_returns_special_code: reason={reason}")


def test_sl_first_within_horizon_priority():
    """3-row synthetic table demonstrating SL-first across all 3 horizons."""
    # Row 1: SHORT, 1s label +0.3 (small adverse, under SL), 5s +0.8 (still under SL),
    #         10s +1.0 -> -1.0 inpos -> SL hit at 10s.
    cases = [
        # (direction, labels, expected_reason, expected_gross, expected_hold)
        # First SL-hit checkpoint (5s for row 0; 10s for row 1 because |label|<SL at 5s)
        ('short', {1: +0.3, 5: +0.8, 10: +1.0}, 'sl',  -0.5, 5.0),
        ('long',  {1: -0.3, 5: -0.4, 10: -0.6}, 'sl',  -0.5, 10.0),
        ('short', {1: +0.0, 5: -1.5, 10: -2.0}, 'tp2', +1.0, 5.0),
    ]
    for i, (dirn, lbls, exp_r, exp_g, exp_h) in enumerate(cases):
        r, g, h = _resolve_hc413_bracket(direction=dirn, labels_by_h=lbls,
                                          tp1=0.5, tp2=1.0, sl=0.5)
        assert r == exp_r and approx(g, exp_g) and approx(h, exp_h), \
            f"row {i}: dir={dirn} lbls={lbls} -> ({r},{g},{h}), expected ({exp_r},{exp_g},{exp_h})"
        print(f"  [OK] sl_first row{i}: dir={dirn} reason={r} gross={g:+.3f}")


def main() -> int:
    print("HC #437 Bug 2 — bracket resolver unit tests")
    print("=" * 60)
    tests = [
        test_short_sl_first_at_1s,
        test_long_tp2_at_5s,
        test_long_tp1_at_1s_when_tp2_not_met,
        test_short_time_stop,
        test_no_label_returns_special_code,
        test_sl_first_within_horizon_priority,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failures += 1
        except Exception as e:
            print(f"  [ERROR] {t.__name__}: {e!r}")
            failures += 1
    print("=" * 60)
    if failures:
        print(f"RESULT: {failures} FAILED")
        return 1
    print(f"RESULT: {len(tests)}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
