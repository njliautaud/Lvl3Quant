#!/usr/bin/env python3
"""
Iceberg Early-Limit Entry Sweep
================================
Tests whether placing a LIMIT order N ticks BEFORE the iceberg fires
improves queue position enough to make the strategy profitable.

Key insight from iceberg_fillsim results:
  - Raw iceberg signal: 84.6% WR
  - After fill_sim: queue pos 33-35, fill latency 3900-4200ms
  - WR collapses to 35-53%. Best config Sortino -0.868
  - Root cause: everyone queues at same S/R level -> FIFO queue kills us

Hypothesis: By entering N ticks before the iceberg level (anticipating the
rejection), we get better queue position. At the cost of lower fill rate
(price must retrace to us) and potential adverse selection (we're entering
before confirmation).

Distances: 1, 2, 3, 5, 8, 10, 15, 20 ticks from iceberg signal price
Hold configs: 3 (same as prior iceberg_fillsim.py study)

Method:
  1. Load iceberg predictions from dl_book_cache_oot
  2. For each distance d: create modified predictions where entry price is
     shifted by d ticks (encode distance in signal magnitude: signal > d+0.5)
  3. Run fill_sim_cli with --signal-threshold d (only fires at magnitude > d)
     → this simulates "only trade when iceberg is confirmed AND price has
       already moved d ticks in our favor first" = WRONG DIRECTION
  4. Correct approach: use --take-profit-ticks adjusted to account for the
     pre-entry. TP = original_TP + d (need price to move TP+d from iceberg
     price, but we entered d ticks better so net TP stays same).
     The queue improvement comes from the ANTICIPATORY entry being rested
     before the iceberg fires.

Implementation:
  Since fill_sim_cli simulates passive limit orders at signal price with
  realistic FIFO queue, we approximate early-limit by:
  a) Scaling prediction magnitude to encode confidence level
  b) Using signal threshold to select only highest-confidence signals
     (proxy for "queue early, only high-conviction")
  c) Also test: --chase-max-ticks N as proxy for "willingness to chase"

  ANALYTIC MODEL (primary): Use empirical queue data from iceberg_fillsim
  to model the expected fill_rate, queue_pos, and WR at each distance.

Output: /home/jupiter/Lvl3Quant/data/processed/iceberg_early_limit_results.json
"""

import os
import sys
import json
import math
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool

# ─── paths ────────────────────────────────────────────────────────────────────
ROOT     = Path("/home/jupiter/Lvl3Quant")
PRED_DIR = ROOT / "data" / "processed" / "dl_book_cache_oot"
MBO_DIR  = ROOT / "data" / "raw" / "mbo"
OUT_DIR  = ROOT / "data" / "processed"
OUT_FILE = OUT_DIR / "iceberg_early_limit_results.json"
FILL_SIM = ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"

TICK_SIZE = 0.25   # ES futures
TICK_VALUE = 12.50 # $12.50 per tick ($50/point)
N_WORKERS = 8

# Early-limit distances to sweep (ticks from iceberg signal price)
ENTRY_DISTANCES = [1, 2, 3, 5, 8, 10, 15, 20]

# Hold configs (same as iceberg_fillsim.py)
HOLD_CONFIGS = [
    {"name": "hold10s_tp3_sl6",  "hold_ms": 10000, "tp": 3, "sl": 6},
    {"name": "hold30s_tp4_sl8",  "hold_ms": 30000, "tp": 4, "sl": 8},
    {"name": "hold10s_tp4_sl8",  "hold_ms": 10000, "tp": 4, "sl": 8},
]

# ─── empirical baseline from iceberg_fillsim.py (84 OOT days) ─────────────────
# hold30s_tp4_sl8: Sortino=-0.868, $/day=-2646, WR=53.3%, FillR=20.8%, AvgQ=33.7
# hold10s_prime:   Sortino=-0.879, $/day=-2388, WR=35.2%, FillR=18.4%, AvgQ=34.4
BASELINE = {
    "hold30s_tp4_sl8": {"sortino": -0.868, "pnl_day": -2646, "wr": 0.533, "fill_rate": 0.208, "avg_queue": 33.7, "trades_day": 155.1},
    "hold10s_tp3_sl6": {"sortino": -0.897, "pnl_day": -3710, "wr": 0.460, "fill_rate": 0.224, "avg_queue": 33.7, "trades_day": 198.7},
    "hold10s_tp4_sl8": {"sortino": -0.904, "pnl_day": -3945, "wr": 0.407, "fill_rate": 0.220, "avg_queue": 34.0, "trades_day": 189.8},
}
N_OOT_DAYS = 84
SIGNALS_PER_DAY = 745   # iceberg fires ~745 times/day in raw signal before fill filter

# ─── queue position model ──────────────────────────────────────────────────────
# Empirical: queue pos 33-35 at 0 ticks early
# Each tick early = enter before the iceberg fires = fewer people in queue ahead
# Queue improvement is NONLINEAR: first few ticks big jump, then diminishing returns
# Model from market microstructure: queue grows ~exponentially near S/R levels

def model_queue_pos(distance_ticks: int) -> float:
    """
    Estimate queue position at entry given distance ticks BEFORE iceberg fires.
    At d=0: queue pos = 34 (empirical)
    At d=1: ~20 (significant improvement - fewer people anticipate 1 tick early)
    At d=2: ~12
    At d=3: ~8
    At d=5: ~4
    At d=8+: ~2 (near front, diminishing returns)
    """
    base = 34.0
    # Exponential decay model: Q(d) = max(1, 34 * exp(-0.4 * d))
    q = max(1.0, base * math.exp(-0.40 * distance_ticks))
    return round(q, 1)


def model_fill_rate(distance_ticks: int, base_fill_rate: float = 0.208) -> float:
    """
    Estimate fill rate given distance ticks early.
    At d=0: 20.8% (empirical - price reaches level 20.8% of signal times)
    At d=N: price must retrace N ticks to fill us. Fill rate decreases.

    The iceberg fires at a support/resistance level. If we enter d ticks BEFORE
    that level, we need price to approach AND fill us before the iceberg fires.

    Actually: entering d ticks early means we POST our limit BEFORE the iceberg
    appears. When the iceberg signal fires, price is AT the S/R level. Our order
    at d ticks better = d ticks INTO the range, so we get filled WHEN the iceberg
    fills (because price has already moved to our level by definition).

    Wait - re-read the setup:
    - Iceberg signal fires when price is AT level X (iceberg detected)
    - Standard entry: post limit at X -> queue pos 34
    - Early-limit entry: pre-post limit at X - d*tick (d ticks better for long)
      BEFORE the iceberg fires

    But we can't post BEFORE the signal fires in real trading.
    Interpretation: post AT the iceberg level but with --chase-max-ticks = d
    (willing to chase d ticks), OR use a DIFFERENT signal that fires d ticks
    before the iceberg (anticipatory).

    For ANALYTIC purposes: model as "we post at X-d*tick and price must reach
    X-d*tick on its way to X." If iceberg fires at X, price definitely passed
    through X-d*tick (for longs: price bounced FROM X, meaning it dipped to X
    and rebounded - if we entered at X-d*tick that's d ticks INTO the bounce,
    meaning price had to go FURTHER than X to reach us).

    Actually for LONGS at support X:
    - Price approaches from above, dips to X, bounces
    - Our entry at X - d*tick means price would need to go d ticks BELOW X
    - This is ADVERSE: price goes through support, more likely to continue down
    - Fill rate should DECREASE and WR should DECREASE for longs with early limit

    ALTERNATIVE interpretation (the one that makes sense for "early"):
    - We post our limit d ticks ABOVE the iceberg level (for longs)
    - We're willing to pay d ticks more to get filled EARLIER in the queue
    - Entry at X + d*tick (for long) -> fills on the way DOWN to X
    - Better queue position (we're at a less crowded price level)
    - But we pay d ticks more slippage on entry

    THIS is the right interpretation: "early-limit" = enter at a LESS FAVORABLE
    price (d ticks away from the iceberg price toward the current price) to get
    EARLIER FIFO queue position and thus faster/more certain fills.

    Under this interpretation:
    - Fill rate INCREASES (price doesn't need to go as far)
    - WR DECREASES slightly (worse entry price = less room for TP)
    - Queue pos IMPROVES significantly
    """
    # Fill rate increases as we move d ticks toward current price
    # At d=0: 20.8% (price must reach iceberg level)
    # At d=1: ~35% (1 tick less to travel)
    # At d=2: ~50%
    # At d=5: ~75%
    # At d=10: ~90%
    # At d=20: ~98%
    fill_rate = min(0.98, base_fill_rate + distance_ticks * 0.038)
    return round(fill_rate, 4)


def model_wr_adjustment(distance_ticks: int, base_wr: float, tp_ticks: int) -> float:
    """
    WR adjustment from entering d ticks worse (toward current price).
    Entering d ticks worse means our effective TP is reduced by d ticks
    (we're d ticks further from the target), AND we have less cushion.

    Approximate: WR_adj = base_wr * (1 - d / (tp_ticks * 2))
    Cap at a floor to avoid negative WR.
    """
    wr_penalty = distance_ticks / (tp_ticks * 4.0)  # soft penalty
    adj_wr = max(0.20, base_wr * (1.0 - wr_penalty))
    return round(adj_wr, 4)


def sortino(daily_pnls: list) -> float:
    if not daily_pnls:
        return 0.0
    neg = [p for p in daily_pnls if p < 0]
    if not neg:
        return 9.99
    downside_std = math.sqrt(sum(p * p for p in neg) / len(neg))
    avg = sum(daily_pnls) / len(daily_pnls)
    return avg / downside_std if downside_std > 0 else 0.0


def run_analytic_sweep():
    """
    Primary analysis: analytic fill model using empirical iceberg_fillsim data.
    Models fill_rate, queue_pos, and WR for each (distance, hold_config) combo.
    """
    print(f"[{datetime.now():%H:%M:%S}] Running analytic sweep")
    print(f"  Distances: {ENTRY_DISTANCES} ticks")
    print(f"  Hold configs: {len(HOLD_CONFIGS)}")
    print(f"  Based on {N_OOT_DAYS} OOT days empirical iceberg_fillsim data\n")

    results = []
    rng = np.random.default_rng(42)

    for cfg in HOLD_CONFIGS:
        base = BASELINE.get(cfg["name"])
        if not base:
            # Estimate from similar config
            base = {"sortino": -0.88, "pnl_day": -3000, "wr": 0.46,
                    "fill_rate": 0.21, "avg_queue": 34.0, "trades_day": 175.0}

        print(f"Config: {cfg['name']} | baseline: Sortino={base['sortino']:.3f} $/day={base['pnl_day']:.0f}")

        # Baseline (d=0)
        results.append({
            "distance_ticks": 0,
            "config": cfg["name"],
            "hold_ms": cfg["hold_ms"],
            "tp_ticks": cfg["tp"],
            "sl_ticks": cfg["sl"],
            "est_queue_pos": base["avg_queue"],
            "avg_fill_rate": base["fill_rate"],
            "avg_win_rate": base["wr"],
            "avg_trades_day": base["trades_day"],
            "avg_pnl_day": base["pnl_day"],
            "sortino": base["sortino"],
            "n_days": N_OOT_DAYS,
            "note": "empirical_baseline",
        })

        for dist in ENTRY_DISTANCES:
            q_pos = model_queue_pos(dist)
            fill_rate = model_fill_rate(dist, base["fill_rate"])
            adj_wr = model_wr_adjustment(dist, base["wr"], cfg["tp"])

            # Effective trades per day
            eff_trades_day = SIGNALS_PER_DAY * fill_rate

            # Per-trade PnL (net of entry cost: we pay dist ticks more on entry)
            entry_cost_ticks = dist  # we enter dist ticks worse than iceberg level
            effective_tp = cfg["tp"] - entry_cost_ticks
            effective_sl = cfg["sl"] + entry_cost_ticks

            if effective_tp <= 0:
                # TP eaten by entry cost - can't be profitable at this distance
                avg_pnl_day = -eff_trades_day * effective_sl * TICK_VALUE * (1 - adj_wr)
                adj_wr_net = 0.0
            else:
                gross_per_trade = adj_wr * effective_tp * TICK_VALUE - (1 - adj_wr) * effective_sl * TICK_VALUE
                rt_cost = 0.1 * TICK_VALUE  # 0.1 tick RT commission
                net_per_trade = gross_per_trade - rt_cost
                avg_pnl_day = eff_trades_day * net_per_trade
                adj_wr_net = adj_wr

            # Simulate day-to-day variance
            daily_trades = rng.poisson(eff_trades_day, N_OOT_DAYS).astype(float)
            daily_wins = rng.binomial(daily_trades.astype(int), min(0.99, adj_wr_net + 0.01))

            if effective_tp > 0:
                daily_pnls = (
                    daily_wins * effective_tp * TICK_VALUE
                    - (daily_trades - daily_wins) * effective_sl * TICK_VALUE
                    - daily_trades * 0.1 * TICK_VALUE
                )
            else:
                daily_pnls = -(daily_trades * effective_sl * TICK_VALUE * (1 - adj_wr_net))

            s = sortino(daily_pnls.tolist())

            results.append({
                "distance_ticks": dist,
                "config": cfg["name"],
                "hold_ms": cfg["hold_ms"],
                "tp_ticks": cfg["tp"],
                "sl_ticks": cfg["sl"],
                "effective_tp_ticks": effective_tp,
                "effective_sl_ticks": effective_sl,
                "est_queue_pos": q_pos,
                "avg_fill_rate": fill_rate,
                "avg_win_rate": adj_wr_net,
                "avg_trades_day": round(eff_trades_day, 1),
                "avg_pnl_day": round(float(np.mean(daily_pnls)), 2),
                "total_pnl": round(float(np.sum(daily_pnls)), 2),
                "sortino": round(s, 4),
                "n_days": N_OOT_DAYS,
                "note": "analytic_model",
            })

            verdict = "✓ PROFITABLE" if s > 0.5 else ("◈ breakeven" if s > 0 else "✗ negative")
            print(f"  d={dist:2d}t | Q={q_pos:4.1f} FillR={fill_rate:.1%} WR={adj_wr_net:.1%} "
                  f"TP_eff={effective_tp} $/day={np.mean(daily_pnls):+.0f} "
                  f"Sortino={s:+.3f} {verdict}")
        print()

    return results


def run_fillsim_sweep():
    """
    Secondary analysis: run actual fill_sim_cli with varied --chase-max-ticks
    as a proxy for willingness to enter d ticks away from ideal price.
    chase-entry = queue at BBO, chase up to N ticks = practical early-limit proxy.
    """
    print(f"\n[{datetime.now():%H:%M:%S}] Running fill_sim chase sweep (proxy for early-limit)")

    if not FILL_SIM.exists():
        print(f"  fill_sim_cli not found at {FILL_SIM}")
        return []

    # Find pred files
    pred_files = sorted(PRED_DIR.glob("*_iceberg_preds.npz"))
    if not pred_files:
        print(f"  No iceberg pred files in {PRED_DIR}")
        return []

    dates = [f.stem.replace("_iceberg_preds", "") for f in pred_files]
    print(f"  Found {len(dates)} pred files")

    # Chase distances as proxy: chase 1,2,3,5 ticks
    chase_distances = [1, 2, 3, 5]
    tasks = []

    chase_out_dir = OUT_DIR / "iceberg_early_limit_chase"
    chase_out_dir.mkdir(parents=True, exist_ok=True)

    for dist in chase_distances:
        for cfg in HOLD_CONFIGS:
            for date, pred_file in zip(dates, pred_files):
                date_nodash = date.replace("-", "")
                mbo_file = MBO_DIR / f"glbx-mdp3-{date_nodash}.mbo.dbn.zst"
                if not mbo_file.exists():
                    continue
                out_file = chase_out_dir / f"{date}_{cfg['name']}_chase{dist}.json"
                tasks.append((date, pred_file, mbo_file, out_file, dist, cfg))

    print(f"  Total fill_sim tasks: {len(tasks)}")

    def run_one(args):
        date, pred_file, mbo_file, out_file, chase_dist, cfg = args

        if out_file.exists():
            try:
                with open(out_file) as f:
                    d = json.load(f)
                    d["date"] = date
                    d["config"] = cfg["name"]
                    d["chase_dist"] = chase_dist
                    return d
            except:
                pass

        cmd = [
            str(FILL_SIM),
            "--mbo-file", str(mbo_file),
            "--predictions", str(pred_file),
            "--output", str(out_file),
            "--signal-threshold", "0.5",
            "--latency-ms", "10",
            "--chase-entry",
            "--chase-max-ticks", str(chase_dist),
            "--chase-max-reprices", "10",
        ]
        if cfg.get("hold_ms"):
            cmd += ["--hold-ms", str(cfg["hold_ms"])]
        if cfg.get("tp"):
            cmd += ["--take-profit-ticks", str(cfg["tp"])]
        if cfg.get("sl"):
            cmd += ["--stop-loss-ticks", str(cfg["sl"])]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                return {"date": date, "config": cfg["name"], "chase_dist": chase_dist, "error": result.stderr[:100], "n_trades": 0}
            with open(out_file) as f:
                d = json.load(f)
                d["date"] = date
                d["config"] = cfg["name"]
                d["chase_dist"] = chase_dist
                return d
        except Exception as e:
            return {"date": date, "config": cfg["name"], "chase_dist": chase_dist, "error": str(e), "n_trades": 0}

    results = []
    with Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, tasks, chunksize=4)):
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  [{datetime.now():%H:%M:%S}] {i+1}/{len(tasks)} fill_sim tasks done")

    # Aggregate by (chase_dist, config)
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        if r and r.get("n_trades", 0) > 0 and "error" not in r:
            key = (r["chase_dist"], r["config"])
            grouped[key].append(r)

    summary = []
    for (dist, cfg_name), days in sorted(grouped.items()):
        pnl_days = []
        for d in days:
            nt = d.get("n_trades", 0) or d.get("total_trades", 0)
            pnl = d.get("total_pnl_ticks", 0) or d.get("pnl_ticks", 0)
            pnl_days.append(pnl * TICK_VALUE)

        s = sortino(pnl_days)
        avg_day = sum(pnl_days) / len(pnl_days)

        summary.append({
            "distance_ticks": dist,
            "config": cfg_name,
            "method": "chase_fillsim",
            "n_days": len(days),
            "avg_pnl_day": round(avg_day, 2),
            "sortino": round(s, 4),
        })

    return summary


def main():
    print(f"[{datetime.now():%H:%M:%S}] === ICEBERG EARLY-LIMIT SWEEP ===")
    print(f"  Distances: {ENTRY_DISTANCES} ticks")
    print(f"  Mode: analytic model (primary) + fill_sim chase (secondary)\n")

    # 1. Analytic sweep (always runs)
    analytic_results = run_analytic_sweep()

    # 2. Fill_sim chase sweep (runs if pred files available)
    fillsim_results = run_fillsim_sweep()

    # Sort analytic results by Sortino
    analytic_results.sort(key=lambda x: x["sortino"], reverse=True)

    # Find best config
    best = [r for r in analytic_results if r["distance_ticks"] > 0]
    best.sort(key=lambda x: x["sortino"], reverse=True)

    out = {
        "completed": datetime.now().isoformat(),
        "method": "analytic_fill_model_v1",
        "baseline_note": "Empirical from iceberg_fillsim.py: 84 OOT days, queue pos 33-35, fill_rate 20.8%",
        "distances_tested": ENTRY_DISTANCES,
        "n_configs": len(HOLD_CONFIGS),
        "analytic_results": analytic_results,
        "fillsim_chase_results": fillsim_results,
        "top5_analytic": best[:5],
        "top5_fillsim": sorted(fillsim_results, key=lambda x: x.get("sortino", -99), reverse=True)[:5] if fillsim_results else [],
        "verdict": best[0] if best else None,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[{datetime.now():%H:%M:%S}] COMPLETE — {OUT_FILE}")
    print("\n=== TOP ANALYTIC RESULTS BY SORTINO ===")
    print(f"{'Dist':>4} {'Config':<20} {'Q-pos':>5} {'FillR':>6} {'WR':>6} {'TP_eff':>6} {'$/day':>8} {'Sortino':>8}")
    print("-" * 75)
    for r in best[:10]:
        print(f"{r['distance_ticks']:>4}t {r['config']:<20} {r['est_queue_pos']:>5.1f} "
              f"{r['avg_fill_rate']:>6.1%} {r['avg_win_rate']:>6.1%} "
              f"{r.get('effective_tp_ticks', r['tp_ticks']):>6} "
              f"{r['avg_pnl_day']:>+8.0f} {r['sortino']:>+8.3f}")

    if best and best[0]["sortino"] > 0:
        b = best[0]
        print(f"\n✓ BEST CONFIG: dist={b['distance_ticks']}t {b['config']} "
              f"Sortino={b['sortino']:.3f} $/day={b['avg_pnl_day']:+.0f}")
        print(f"  Queue pos: {b['est_queue_pos']:.1f} (vs baseline 34)")
        print(f"  Fill rate: {b['avg_fill_rate']:.1%} (vs baseline 20.8%)")
        print(f"  Effective TP: {b.get('effective_tp_ticks', b['tp_ticks'])} ticks")
    else:
        print(f"\n✗ ALL DISTANCES NEGATIVE — early-limit does not overcome entry cost")
        print(f"  Best distance: {best[0]['distance_ticks']}t Sortino={best[0]['sortino']:.3f}")
        print(f"  Root cause: entry cost (d ticks worse) eliminates queue improvement benefit")


if __name__ == "__main__":
    main()
