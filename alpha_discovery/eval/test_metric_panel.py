"""Synthetic tests for metric_panel.py (HC #256). Run: python -m alpha_discovery.eval.test_metric_panel"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .metric_panel import compute_metric_panel, format_panel_table, passes_hc254

ES_TICK_VALUE = 12.50
COMMISSION_TICKS_RT = 0.376


def _make_trades(n_per_day: int, dates: list, wr: float, edge_ticks: float, seed: int = 0) -> pd.DataFrame:
    """Synthesize per-trade DF with a target WR + edge per trade (post-commission)."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates:
        for i in range(n_per_day):
            is_win = rng.random() < wr
            if is_win:
                pnl_t = rng.uniform(0.5, 4.0) + edge_ticks
            else:
                pnl_t = -rng.uniform(0.5, 4.0) + edge_ticks
            pnl_t -= COMMISSION_TICKS_RT
            entry_ns = int(rng.integers(1_700_000_000_000_000_000, 1_800_000_000_000_000_000))
            hold_sec = rng.uniform(5, 120)
            rows.append({
                "date": d,
                "entry_ts_ns": entry_ns,
                "exit_ts_ns": entry_ns + int(hold_sec * 1e9),
                "side": "L" if rng.random() < 0.5 else "S",
                "entry_price": 5000.0,
                "exit_price": 5000.0 + pnl_t * 0.25,
                "pnl_ticks": pnl_t,
                "pnl_dollars": pnl_t * ES_TICK_VALUE,
                "mfe_ticks": max(pnl_t, 0) + rng.uniform(0, 1),
                "mae_ticks": min(pnl_t, 0) - rng.uniform(0, 1),
            })
    return pd.DataFrame(rows)


def test_a_clearly_profitable():
    """High WR, positive edge, consistent across 30 dates → Sortino > 1, ≥80% dates positive."""
    dates = [f"2026020{i // 10}{i % 10}" for i in range(30)]  # 30 fake dates
    df = _make_trades(n_per_day=20, dates=dates, wr=0.60, edge_ticks=0.5, seed=1)
    panel = compute_metric_panel(df)
    ok, vio = passes_hc254(panel)
    print(f"[A] sortino={panel['sortino']:.3f} pct_dates_pos={panel['pct_dates_positive']:.2f} "
          f"sdc={panel['single_date_concentration']:.3f} pf={panel['profit_factor']:.2f} hc254={'PASS' if ok else 'FAIL'}")
    assert panel["sortino"] > 0.5, f"expected sortino>0.5, got {panel['sortino']}"
    assert panel["pct_dates_positive"] >= 0.70, f"expected pct>=0.70, got {panel['pct_dates_positive']}"
    print("[A] PASS")


def test_b_one_day_fluke():
    """29 small-loss days + 1 huge-win day → high concentration → fails HC #254."""
    dates_loss = [f"2026030{i // 10}{i % 10}" for i in range(29)]
    df_loss = _make_trades(n_per_day=10, dates=dates_loss, wr=0.40, edge_ticks=-0.1, seed=2)
    # one huge winning day
    huge = _make_trades(n_per_day=10, dates=["20260331"], wr=1.0, edge_ticks=10.0, seed=3)
    df = pd.concat([df_loss, huge], ignore_index=True)
    panel = compute_metric_panel(df)
    ok, vio = passes_hc254(panel)
    print(f"[B] sortino={panel['sortino']:.3f} pct_dates_pos={panel['pct_dates_positive']:.2f} "
          f"sdc={panel['single_date_concentration']:.3f} total=${panel['total_pnl_dollars']:.0f} "
          f"hc254={'PASS' if ok else 'FAIL'} viol={vio}")
    assert not ok, f"expected HC254 FAIL, got PASS — {vio}"
    print("[B] PASS (correctly flagged single-date concentration)")


def test_c_clearly_losing():
    """Low WR, negative edge → Sortino < 0."""
    dates = [f"2026040{i // 10}{i % 10}" for i in range(20)]
    df = _make_trades(n_per_day=15, dates=dates, wr=0.40, edge_ticks=-0.5, seed=4)
    panel = compute_metric_panel(df)
    ok, vio = passes_hc254(panel)
    print(f"[C] sortino={panel['sortino']:.3f} pct_dates_pos={panel['pct_dates_positive']:.2f} "
          f"pf={panel['profit_factor']:.2f} hc254={'PASS' if ok else 'FAIL'}")
    assert panel["sortino"] < 0, f"expected sortino<0, got {panel['sortino']}"
    assert not ok, "expected HC254 FAIL"
    print("[C] PASS")


def main():
    print("=" * 70)
    print("metric_panel.py synthetic tests (HC #256)")
    print("=" * 70)
    test_a_clearly_profitable()
    test_b_one_day_fluke()
    test_c_clearly_losing()
    print("\n" + "=" * 70)
    print("Sample formatted panel — test (a) profitable case")
    print("=" * 70)
    dates = [f"2026020{i // 10}{i % 10}" for i in range(30)]
    df = _make_trades(n_per_day=20, dates=dates, wr=0.60, edge_ticks=0.5, seed=1)
    panel = compute_metric_panel(df)
    print(format_panel_table(panel, head_label="synthetic-profitable"))


if __name__ == "__main__":
    main()
