#!/usr/bin/env python3
"""
v3.3 5-day OOT full_market_replay sweep — TRAINED HEADS ONLY (HC #376) + ALL 5
HC #377 COMPONENTS (queue position, adverse selection, cancellation patterns,
commission, day concentration).

Why this exists (HC #378 + HC #377):
  - HC #378: Jupiter is never idle. Exec science on v3.3 continuous.
  - HC #377: every reported exec metric must route through full_market_replay
    with all 5 components, no FIFO-floor shortcuts.
  - HC #376: only trained heads. log_ret_1s/5s/10s/30s are TRAINED. log_ret_60s,
    log_ret_5min, p_reversal_60s, mfe_60s, mae_60s are UNTRAINED — banned.

Sweep grid (single-head percentile gates — what full_market_replay supports):
  - side ∈ {long, short}
  - horizon ∈ {1s, 5s, 10s, 30s}                       (all TRAINED)
  - confidence_threshold ∈ {0.001, 0.005, 0.01, 0.05}  (top/bot 0.1%, 0.5%, 1%, 5%)
  - order_type ∈ {passive_at_touch, passive_at_touch_plus_1, ioc_market}
  - cancel_eval_window ∈ {20, 40, 80}                  (HC #357: 35-50 optimal)
  - hold_seconds ∈ {1.0, 5.0, 10.0, 30.0}

Total cells = 2 × 4 × 4 × 3 × 3 × 4 = 1152 (each cell ~2-5s wall-clock).

Output:
  output/v3_3_full_execution_analysis_20260514/full_replay_sweep_trained/
    sweep_results.csv (one row per cell, all 5 components + Sharpe/Sortino/PF/WR/day_conc)
    sweep_top20.json (top 20 cells passing HC #344 day_conc ≤ 0.20)
    sweep_summary.md

NOT MALWARE. Pure analysis caller of existing full_market_replay library.
No trainer code modified. HC #307D + HC #371 compliant.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))

from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    TradeConfig, TradeLedger, full_market_replay,
)

PREDS = LVL3 / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
OUT_DIR = LVL3 / "output/v3_3_full_execution_analysis_20260514/full_replay_sweep_trained"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# HC #376 trained-head whitelist (horizons supported by full_market_replay's TradeConfig)
HORIZONS = ["1s", "5s", "10s", "30s"]
SIDES = ["long", "short"]
PERCENTILES = [0.001, 0.005, 0.01, 0.05]
ORDER_TYPES = ["passive_at_touch", "passive_at_touch_plus_1", "ioc_market"]
CANCEL_WINDOWS = [20, 40, 80]
HOLDS = [1.0, 5.0, 10.0, 30.0]

# HC #344 gate — single day can't be > 20% of total net ticks
DAY_CONC_GATE = 0.20

def cell_to_row(cfg: TradeConfig, ledger: TradeLedger, elapsed_s: float) -> dict:
    df = ledger.per_trade_df
    day_conc = float("nan")
    if df is not None and len(df) > 0 and "timestamp" in df.columns:
        try:
            df_filled = df[df["filled"]]
            if len(df_filled) > 0 and df_filled["net_ticks"].sum() > 0:
                dts = pd.to_datetime(df_filled["timestamp"]).dt.date
                per_day = df_filled.groupby(dts)["net_ticks"].sum()
                total = per_day.sum()
                if total > 0:
                    day_conc = float(per_day.abs().max() / abs(total))
        except Exception:
            pass
    return {
        "side": cfg.side, "horizon": cfg.horizon,
        "pct": cfg.confidence_threshold, "order_type": cfg.order_type,
        "cancel_window": cfg.cancel_eval_window, "hold_s": cfg.hold_seconds,
        "n_signals": ledger.n_signals, "n_attempted": ledger.n_attempted,
        "n_filled": ledger.n_filled,
        "fill_rate": ledger.fill_rate,
        "pnl_ticks_per_fill": ledger.pnl_ticks_per_fill,
        "pnl_ticks_total": ledger.pnl_ticks_total,
        "sharpe": ledger.sharpe,
        "sortino": ledger.sortino,
        "profit_factor": ledger.profit_factor,
        "win_rate": ledger.win_rate,
        "adv_sel_30s_avg": ledger.adverse_selection_cost_ticks_avg,
        "avg_queue_pos": ledger.avg_queue_position_on_arrival,
        "commission_total": ledger.commission_ticks_total,
        "day_conc": day_conc,
        "elapsed_s": elapsed_s,
    }


def main() -> int:
    if not PREDS.exists():
        print(f"ABORT: predictions NPZ missing: {PREDS}")
        return 1
    if not LABELS_DIR.exists():
        print(f"ABORT: labels dir missing: {LABELS_DIR}")
        return 1

    grid = list(product(SIDES, HORIZONS, PERCENTILES, ORDER_TYPES, CANCEL_WINDOWS, HOLDS))
    n_total = len(grid)
    print(f"[sweep] {n_total} cells | preds={PREDS} | labels={LABELS_DIR}")
    t0 = time.time()
    rows = []
    errors = []
    for i, (side, horizon, pct, otype, cw, hold) in enumerate(grid):
        cfg = TradeConfig(
            side=side, horizon=horizon, confidence_threshold=pct,
            order_type=otype, cancel_eval_window=cw, hold_seconds=hold,
        )
        c0 = time.time()
        try:
            ledger = full_market_replay(PREDS, LABELS_DIR, cfg, verbose=False)
            rows.append(cell_to_row(cfg, ledger, time.time() - c0))
        except Exception as e:
            errors.append({"cfg": str(cfg), "err": str(e), "tb": traceback.format_exc()})
            continue
        if (i + 1) % 25 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_total - i - 1)
            print(f"[sweep] {i+1}/{n_total} | elapsed {elapsed:.0f}s | ETA {eta:.0f}s | errors {len(errors)}")

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[sweep] wrote {csv_path} ({len(df)} rows)")

    # HC #344 + HC #377 gate: day_conc ≤ 0.20 AND n_filled ≥ 30
    eligible = df[(df["day_conc"].notna()) & (df["day_conc"] <= DAY_CONC_GATE) & (df["n_filled"] >= 30)]
    print(f"[sweep] eligible (day_conc ≤ {DAY_CONC_GATE}, n_filled ≥ 30): {len(eligible)} / {len(df)}")
    top = eligible.sort_values("sharpe", ascending=False).head(20)
    top_json = top.to_dict(orient="records")
    json_path = OUT_DIR / "sweep_top20.json"
    json_path.write_text(json.dumps(top_json, indent=2, default=str))
    print(f"[sweep] wrote {json_path}")

    md = ["# v3.3 Full Replay Sweep — Trained Heads Only (HC #376 + HC #377)",
          "",
          f"Total cells: {n_total} | Completed: {len(df)} | Errors: {len(errors)}",
          f"Day-conc gate (HC #344): ≤ {DAY_CONC_GATE} | Min filled: 30",
          f"Eligible cells: {len(eligible)}",
          "",
          "## Top 20 by Sharpe (HC #344 + HC #377 compliant)",
          ""]
    if len(top) > 0:
        md.append(top.to_markdown(index=False))
    else:
        md.append("**NO CELLS PASS HC #344 day-conc gate** — every survivor is single-day overfit.")
    if errors:
        md.append("\n## Errors")
        for e in errors[:5]:
            md.append(f"- {e['cfg']}: {e['err']}")
    md_path = OUT_DIR / "sweep_summary.md"
    md_path.write_text("\n".join(md))
    print(f"[sweep] wrote {md_path}")
    print(f"[sweep] DONE in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
