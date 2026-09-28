#!/usr/bin/env python3
"""
predicate_execution_analysis.py — Spread/depth predicate execution filter analysis.

TODO (Rust): fill_sim does NOT yet support entry_max_spread_ticks or
entry_min_depth_lots as SimConfig / CLI fields. Before using this analysis to
deploy live predicates, add to fill_sim.rs SimConfig:
    pub entry_max_spread_ticks: f64,   // Skip entry if BBO spread >= this (0 = disabled)
    pub entry_min_depth_lots: f64,     // Skip entry if BBO depth < this (0 = disabled)
and add the matching --entry-max-spread-ticks / --entry-min-depth-lots CLI flags in
fill_sim_main.rs. Until then this script performs an equivalent post-hoc analysis.

HOW THIS WORKS
--------------
The fill_sim records `book_size_at_post` (lots at BBO when the passive order was
posted) in every TradeResult. This is a direct proxy for the depth predicate:

    entry_min_depth_lots: skip the trade if book_size_at_post < threshold

Spread is NOT recorded per trade by the current fill_sim (the BBO spread is
consumed internally for fill-price accounting but not emitted). As a best-effort
proxy we use:

    "wide_spread_flag" = book_size_at_post is very small AND queue_position ≈ 0
    (thin book + first in queue typically correlates with wider spreads on ES)

Because we cannot post-hoc reconstruct the exact tick spread for each signal, the
spread analysis is clearly labelled as PROXY/APPROXIMATE in the output. Depth
analysis is exact (it comes directly from TradeResult).

WHAT THIS SCRIPT DOES
---------------------
1. Runs fill_sim (baseline config) for Cards 1, 4, 5 across all OOT dates.
   Each run emits full per-trade JSON including book_size_at_post.
2. Loads all per-trade data.
3. For each (card, depth_threshold) combination, simulates "would-be" performance
   if entries with book_size_at_post < threshold were skipped.
4. Computes: Sharpe, PnL, win-rate, trades-remaining, fill-rate change, and
   percentage of trades filtered.
5. Saves results to data/processed/predicate_sweep/predicate_analysis.json
   and a human-readable summary table.

Run on Jupiter CPU (no GPU required):
    nohup python3 /home/jupiter/Lvl3Quant/scripts/predicate_execution_analysis.py \
        2>&1 | tee predicate_analysis.log &

Expected runtime: ~5-15 minutes (fill_sim runs, 14 workers).
"""

import os
import sys
import json
import math
import logging
import subprocess
import tempfile
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("predicate_analysis")

# ── Paths ──────────────────────────────────────────────────────────────────────
FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR  = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions")
OUT_DIR  = Path("/home/jupiter/Lvl3Quant/data/processed/predicate_sweep")

OOT_START = date(2025, 12, 1)
OOT_END   = date(2026, 3, 8)
WORKERS   = 14

# ── Cards under test ───────────────────────────────────────────────────────────
# Cards 1, 4, 5 — these have meaningful depth distributions and different alpha
# profiles (Card1: book pred, Card4: high-vol book pred, Card5: raw pred no TP).
CARDS = {
    "Card1": {
        "pred_suffix": "book_predstdExit_conv1.5_vol50",
        "base_args": [
            "--signal-threshold", "0.1",
            "--take-profit-ticks", "13",   # 371-sweep optimal
            "--hold-ms", "7200000",
        ],
    },
    "Card4": {
        "pred_suffix": "book_predstdExit_conv2.0_vol70",
        "base_args": [
            "--signal-threshold", "0.3",   # 371-sweep: sig0.3
            "--take-profit-ticks", "20",
            "--hold-ms", "3600000",        # 371-sweep: 60m hold
        ],
    },
    "Card5": {
        "pred_suffix": "raw_rawExit_conv0.05_ethr0.5_vol0",
        "base_args": [
            "--signal-threshold", "0.1",
            "--hold-ms", "3600000",
            "--mae-exit-ticks", "50",      # 371-sweep: mae50t/300s
            "--mae-exit-hold-sec", "300",
        ],
    },
}

# ── Depth thresholds to test ───────────────────────────────────────────────────
# 0 = no filter (baseline). Units = lots at BBO.
DEPTH_THRESHOLDS = [0, 5, 10, 20, 50, 100]

# ── Spread proxy thresholds (book_size_at_post percentile bins) ────────────────
# We bucket trades by book_size_at_post into "thin" / "normal" / "deep" and
# analyse the P&L distribution. This is a proxy: see module docstring.
DEPTH_PERCENTILE_BINS = [10, 25, 50, 75]  # percentiles used to define bin edges


# ==============================================================================
# Step 1: Run fill_sim to collect per-trade data
# ==============================================================================

def get_oot_dates() -> list[date]:
    """Return all OOT dates that have an MBO file."""
    dates = []
    d = OOT_START
    while d <= OOT_END:
        mbo = MBO_DIR / f"glbx-mdp3-{d.strftime('%Y%m%d')}.mbo.dbn.zst"
        if mbo.exists():
            dates.append(d)
        d += timedelta(days=1)
    return dates


def run_fill_sim_one(
    card_name: str,
    pred_suffix: str,
    base_args: list[str],
    d: date,
    out_path: Path,
) -> tuple[str, str, Path, bool, Optional[str]]:
    """Run fill_sim for one (card, date). Returns (card, date_iso, out_path, ok, err)."""
    date_iso = d.isoformat()
    date_num = d.strftime("%Y%m%d")
    mbo  = MBO_DIR  / f"glbx-mdp3-{date_num}.mbo.dbn.zst"
    pred = PRED_DIR / f"{date_iso}_{pred_suffix}.npz"

    if not mbo.exists():
        return (card_name, date_iso, out_path, False, f"MBO missing: {mbo}")
    if not pred.exists():
        return (card_name, date_iso, out_path, False, f"Pred missing: {pred}")

    cmd = [
        FILL_SIM,
        "--mbo-file",    str(mbo),
        "--predictions", str(pred),
        "--output",      str(out_path),
        "--quiet",
    ] + base_args

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            return (card_name, date_iso, out_path, False, r.stderr[:300])
        return (card_name, date_iso, out_path, True, None)
    except subprocess.TimeoutExpired:
        return (card_name, date_iso, out_path, False, "timeout")
    except Exception as e:
        return (card_name, date_iso, out_path, False, str(e))


def collect_all_trades(dates: list[date]) -> dict[str, list[dict]]:
    """Run fill_sim for all (card, date) pairs and return per-card trade lists.

    Each trade dict is the raw TradeResult JSON plus an injected `_date` key.
    """
    run_dir = OUT_DIR / "raw_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for card_name, card_def in CARDS.items():
        card_dir = run_dir / card_name
        card_dir.mkdir(exist_ok=True)
        for d in dates:
            out_path = card_dir / f"{d.isoformat()}.json"
            tasks.append((card_name, card_def["pred_suffix"], card_def["base_args"], d, out_path))

    log.info(f"Submitting {len(tasks)} fill_sim jobs across {WORKERS} workers ...")
    done = 0
    errors = []

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_fill_sim_one, *t): t for t in tasks}
        for fut in as_completed(futures):
            card_name, date_iso, out_path, ok, err = fut.result()
            done += 1
            if not ok:
                errors.append(f"{card_name}/{date_iso}: {err}")
            if done % 20 == 0:
                log.info(f"  {done}/{len(tasks)} done, {len(errors)} errors so far")

    if errors:
        log.warning(f"{len(errors)} fill_sim failures:")
        for e in errors[:10]:
            log.warning(f"  {e}")
        if len(errors) > 10:
            log.warning(f"  ... and {len(errors) - 10} more")

    # Load results
    all_trades: dict[str, list[dict]] = {c: [] for c in CARDS}
    for card_name, card_def in CARDS.items():
        card_dir = run_dir / card_name
        loaded = 0
        for d in dates:
            out_path = card_dir / f"{d.isoformat()}.json"
            if not out_path.exists():
                continue
            try:
                with open(out_path) as f:
                    data = json.load(f)
                trades = data.get("trades", [])
                for t in trades:
                    t["_date"] = d.isoformat()
                all_trades[card_name].extend(trades)
                loaded += 1
            except Exception as e:
                log.warning(f"Failed to load {out_path}: {e}")
        log.info(f"{card_name}: loaded {len(all_trades[card_name])} trades from {loaded} dates")

    return all_trades


# ==============================================================================
# Step 2: Sharpe and analytics helpers
# ==============================================================================

def sharpe_from_daily(daily_pnls: list[float]) -> float:
    if len(daily_pnls) < 2:
        return 0.0
    m = float(np.mean(daily_pnls))
    s = float(np.std(daily_pnls, ddof=1))
    if s == 0.0:
        return 0.0
    return m / s * math.sqrt(252)


def compute_metrics(trades: list[dict], all_dates: list[str]) -> dict:
    """Compute core metrics from a filtered trade list."""
    if not trades:
        return {
            "n_trades": 0, "sharpe": 0.0, "total_pnl": 0.0,
            "win_rate": 0.0, "profit_factor": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0,
            "daily_sharpe": 0.0,
        }

    pnls = [t["pnl_dollars"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Per-trade Sharpe (annualised by sqrt(252 * avg_trades_per_day) approximation)
    m = float(np.mean(pnls))
    s = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
    per_trade_sharpe = m / s if s > 0 else 0.0

    # Daily Sharpe
    daily: dict[str, float] = defaultdict(float)
    for t in trades:
        daily[t["_date"]] += t["pnl_dollars"]
    for d in all_dates:
        if d not in daily:
            daily[d] = 0.0
    daily_vals = list(daily.values())
    daily_sh = sharpe_from_daily(daily_vals)

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss   = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "n_trades":       len(trades),
        "sharpe":         round(daily_sh, 4),
        "per_trade_sharpe": round(per_trade_sharpe, 4),
        "total_pnl":      round(sum(pnls), 2),
        "win_rate":       round(len(wins) / len(trades), 4),
        "profit_factor":  round(pf, 4) if pf != float("inf") else "inf",
        "avg_win":        round(float(np.mean(wins)), 2)   if wins   else 0.0,
        "avg_loss":       round(float(np.mean(losses)), 2) if losses else 0.0,
        "avg_pnl_per_trade": round(m, 4),
    }


# ==============================================================================
# Step 3: Depth predicate analysis
# ==============================================================================

def depth_predicate_sweep(
    card_name: str,
    trades: list[dict],
    all_dates: list[str],
) -> dict:
    """For each depth threshold, compute metrics for the surviving trade subset."""
    baseline_metrics = compute_metrics(trades, all_dates)
    baseline_n = len(trades)
    log.info(
        f"  {card_name} baseline: {baseline_n} trades, "
        f"Sharpe={baseline_metrics['sharpe']:.3f}, "
        f"PnL=${baseline_metrics['total_pnl']:.0f}"
    )

    results = {}

    for thresh in DEPTH_THRESHOLDS:
        if thresh == 0:
            filtered = trades
            label = "no_filter (baseline)"
        else:
            # Keep only trades where book_size_at_post >= threshold
            filtered = [t for t in trades if t.get("book_size_at_post", 0.0) >= thresh]
            label = f"depth >= {thresh} lots"

        m = compute_metrics(filtered, all_dates)
        n_kept = len(filtered)
        pct_kept = round(100 * n_kept / baseline_n, 1) if baseline_n > 0 else 0.0
        sharpe_delta = round(m["sharpe"] - baseline_metrics["sharpe"], 4)

        results[f"min_depth_{thresh}"] = {
            "label":          label,
            "threshold_lots": thresh,
            "n_trades":       m["n_trades"],
            "pct_trades_kept": pct_kept,
            "sharpe":         m["sharpe"],
            "sharpe_delta_vs_baseline": sharpe_delta,
            "total_pnl":      m["total_pnl"],
            "win_rate":       m["win_rate"],
            "profit_factor":  m["profit_factor"],
            "avg_win":        m["avg_win"],
            "avg_loss":       m["avg_loss"],
            "per_trade_sharpe": m["per_trade_sharpe"],
        }

        log.info(
            f"    {label:30s}: {n_kept:5d} trades ({pct_kept:5.1f}% kept), "
            f"Sharpe={m['sharpe']:+.3f} ({sharpe_delta:+.3f}), "
            f"PnL=${m['total_pnl']:.0f}"
        )

    return {
        "card":     card_name,
        "baseline": baseline_metrics,
        "depth_sweep": results,
    }


# ==============================================================================
# Step 4: Spread proxy analysis (book_size_at_post percentile bins)
# ==============================================================================

def spread_proxy_analysis(
    card_name: str,
    trades: list[dict],
    all_dates: list[str],
) -> dict:
    """
    Split trades into book_size_at_post percentile bins and analyse P&L per bin.

    This is a PROXY for spread analysis. The fill_sim does not record the BBO
    tick-spread directly. Large book_size_at_post correlates with tighter spreads
    (busy books) while very small book_size_at_post correlates with wide / thin
    markets. The correlation is imperfect — treat as directional signal only.

    NOTE: If the Rust fill_sim is updated to record entry_spread_ticks in
    TradeResult, replace book_size_at_post with that field and re-run.
    """
    if not trades:
        return {"card": card_name, "error": "no trades", "note": "spread proxy"}

    depths = [t.get("book_size_at_post", 0.0) for t in trades]
    percentile_edges = [float(np.percentile(depths, p)) for p in DEPTH_PERCENTILE_BINS]

    log.info(
        f"  {card_name} depth percentiles "
        f"[{', '.join(f'P{p}={v:.0f}' for p, v in zip(DEPTH_PERCENTILE_BINS, percentile_edges))}]"
    )

    bins = []
    # Bin 0: very thin book (< P10) — proxy for wide spread / adverse conditions
    # Bin 1: P10–P25, Bin 2: P25–P50, Bin 3: P50–P75, Bin 4: deep book (> P75)
    edges = [0.0] + percentile_edges + [float("inf")]
    bin_labels = [
        f"thin  (depth < P{DEPTH_PERCENTILE_BINS[0]}={percentile_edges[0]:.0f})",
        f"low   (P{DEPTH_PERCENTILE_BINS[0]}-P{DEPTH_PERCENTILE_BINS[1]}={percentile_edges[0]:.0f}-{percentile_edges[1]:.0f})",
        f"mid   (P{DEPTH_PERCENTILE_BINS[1]}-P{DEPTH_PERCENTILE_BINS[2]}={percentile_edges[1]:.0f}-{percentile_edges[2]:.0f})",
        f"high  (P{DEPTH_PERCENTILE_BINS[2]}-P{DEPTH_PERCENTILE_BINS[3]}={percentile_edges[2]:.0f}-{percentile_edges[3]:.0f})",
        f"deep  (depth >= P{DEPTH_PERCENTILE_BINS[3]}={percentile_edges[3]:.0f})",
    ]

    for i, label in enumerate(bin_labels):
        lo, hi = edges[i], edges[i + 1]
        bin_trades = [t for t in trades if lo <= t.get("book_size_at_post", 0.0) < hi]
        m = compute_metrics(bin_trades, all_dates)
        bins.append({
            "label":            label,
            "depth_range":      [lo, hi if hi != float("inf") else None],
            "n_trades":         m["n_trades"],
            "pct_of_total":     round(100 * m["n_trades"] / len(trades), 1) if trades else 0.0,
            "sharpe":           m["sharpe"],
            "total_pnl":        m["total_pnl"],
            "win_rate":         m["win_rate"],
            "avg_pnl_per_trade": m["avg_pnl_per_trade"],
            "profit_factor":    m["profit_factor"],
        })
        log.info(
            f"    {label:60s}: {m['n_trades']:5d} trades, "
            f"Sharpe={m['sharpe']:+.3f}, avg_pnl=${m['avg_pnl_per_trade']:+.2f}"
        )

    # Summary: "would excluding the thin-book bin have helped?"
    thin_trades  = [t for t in trades if t.get("book_size_at_post", 0.0) < edges[1]]
    thick_trades = [t for t in trades if t.get("book_size_at_post", 0.0) >= edges[1]]
    m_thin  = compute_metrics(thin_trades, all_dates)
    m_thick = compute_metrics(thick_trades, all_dates)
    m_all   = compute_metrics(trades, all_dates)

    thin_avg_pnl = m_thin["avg_pnl_per_trade"]
    thick_avg_pnl = m_thick["avg_pnl_per_trade"]
    thin_is_drag = thin_avg_pnl < thick_avg_pnl

    recommendation = (
        f"Thin-book trades (depth < P10) have avg_pnl ${thin_avg_pnl:+.2f} vs "
        f"${thick_avg_pnl:+.2f} for the rest. "
        + ("Filtering thin-book entries MAY improve per-trade edge — test with Rust predicate."
           if thin_is_drag
           else "Thin-book entries are NOT dragging performance — spread filter unlikely to help.")
    )

    return {
        "card":            card_name,
        "note":            "spread proxy via book_size_at_post — NOT exact spread ticks",
        "depth_percentiles": dict(zip(DEPTH_PERCENTILE_BINS, [round(v, 1) for v in percentile_edges])),
        "bins":            bins,
        "summary": {
            "thin_avg_pnl_per_trade":  round(thin_avg_pnl, 4),
            "thick_avg_pnl_per_trade": round(thick_avg_pnl, 4),
            "thin_is_drag":            thin_is_drag,
            "recommendation":          recommendation,
            "baseline_sharpe":         m_all["sharpe"],
            "excl_thin_sharpe":        m_thick["sharpe"],
            "sharpe_delta":            round(m_thick["sharpe"] - m_all["sharpe"], 4),
        },
    }


# ==============================================================================
# Step 5: Combined predicate analysis (depth AND spread-proxy together)
# ==============================================================================

def combined_predicate_analysis(
    card_name: str,
    trades: list[dict],
    all_dates: list[str],
) -> dict:
    """Test combinations of depth filter + spread-proxy (thin-book exclusion)."""
    if not trades:
        return {"card": card_name, "error": "no trades"}

    depths = [t.get("book_size_at_post", 0.0) for t in trades]
    p10 = float(np.percentile(depths, 10))

    m_all = compute_metrics(trades, all_dates)
    baseline_sharpe = m_all["sharpe"]

    combos = []
    for depth_thresh in DEPTH_THRESHOLDS:
        for excl_thin in [False, True]:
            label_parts = []
            subset = trades
            if depth_thresh > 0:
                subset = [t for t in subset if t.get("book_size_at_post", 0.0) >= depth_thresh]
                label_parts.append(f"depth>={depth_thresh}")
            if excl_thin:
                subset = [t for t in subset if t.get("book_size_at_post", 0.0) >= p10]
                label_parts.append("excl_thin")
            label = " + ".join(label_parts) if label_parts else "baseline"

            m = compute_metrics(subset, all_dates)
            combos.append({
                "label":           label,
                "depth_threshold": depth_thresh,
                "excl_thin_book":  excl_thin,
                "n_trades":        m["n_trades"],
                "pct_kept":        round(100 * m["n_trades"] / len(trades), 1) if trades else 0.0,
                "sharpe":          m["sharpe"],
                "sharpe_delta":    round(m["sharpe"] - baseline_sharpe, 4),
                "total_pnl":       m["total_pnl"],
                "win_rate":        m["win_rate"],
            })

    # Sort by Sharpe descending
    combos.sort(key=lambda x: x["sharpe"], reverse=True)

    return {
        "card":         card_name,
        "baseline_sharpe": baseline_sharpe,
        "top_combos":   combos[:10],
        "all_combos":   combos,
    }


# ==============================================================================
# Step 6: Summary table printer
# ==============================================================================

def print_summary_table(all_results: list[dict]) -> str:
    lines = []
    lines.append("=" * 90)
    lines.append("PREDICATE EXECUTION ANALYSIS — SUMMARY")
    lines.append("NOTE: Spread analysis uses book_size_at_post as proxy (NOT exact tick spread)")
    lines.append("TODO: Add entry_max_spread_ticks + entry_min_depth_lots to Rust fill_sim")
    lines.append("=" * 90)

    for res in all_results:
        card = res["card"]
        lines.append(f"\n{'─'*80}")
        lines.append(f"  {card}")
        lines.append(f"{'─'*80}")

        # Depth sweep table
        lines.append(f"\n  DEPTH PREDICATE SWEEP (entry_min_depth_lots simulation):")
        lines.append(f"  {'Threshold':>12}  {'Trades':>7}  {'% kept':>7}  {'Sharpe':>8}  {'Δ Sharpe':>9}  {'PnL':>10}  {'WinRate':>8}")
        lines.append(f"  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*10}  {'-'*8}")
        for k, v in res["depth_sweep"]["depth_sweep"].items():
            lines.append(
                f"  {v['label']:>12}  {v['n_trades']:>7}  {v['pct_trades_kept']:>6.1f}%  "
                f"{v['sharpe']:>8.3f}  {v['sharpe_delta_vs_baseline']:>+9.3f}  "
                f"{v['total_pnl']:>10,.0f}  {v['win_rate']:>7.1%}"
            )

        # Spread proxy summary
        spread = res["spread_proxy"]["summary"]
        lines.append(f"\n  SPREAD PROXY ANALYSIS (book_size_at_post bins):")
        lines.append(f"    Thin-book avg P&L/trade:  ${spread['thin_avg_pnl_per_trade']:+.2f}")
        lines.append(f"    Thick-book avg P&L/trade: ${spread['thick_avg_pnl_per_trade']:+.2f}")
        lines.append(f"    Thin is drag?              {spread['thin_is_drag']}")
        lines.append(f"    Baseline Sharpe:           {spread['baseline_sharpe']:.3f}")
        lines.append(f"    Excl-thin Sharpe:          {spread['excl_thin_sharpe']:.3f}  (Δ {spread['sharpe_delta']:+.3f})")
        lines.append(f"    Recommendation: {spread['recommendation']}")

        # Best combined predicate
        top = res["combined"]["top_combos"][0] if res["combined"].get("top_combos") else {}
        if top:
            lines.append(f"\n  BEST COMBINED PREDICATE:")
            lines.append(f"    Config:   {top.get('label', 'n/a')}")
            lines.append(f"    Sharpe:   {top.get('sharpe', 0.0):.3f}  (Δ {top.get('sharpe_delta', 0.0):+.3f})")
            lines.append(f"    PnL:      ${top.get('total_pnl', 0.0):,.0f}")
            lines.append(f"    Trades:   {top.get('n_trades', 0)} ({top.get('pct_kept', 0.0):.1f}% of baseline)")

    lines.append("\n" + "=" * 90)
    lines.append("NEXT STEPS IF RESULTS ARE POSITIVE:")
    lines.append("  1. Add to fill_sim.rs SimConfig:")
    lines.append("       pub entry_max_spread_ticks: f64,  // 0 = disabled")
    lines.append("       pub entry_min_depth_lots: f64,    // 0 = disabled")
    lines.append("  2. In fill_sim.rs, before posting the passive order:")
    lines.append("       let bbo_spread = (ask_price - bid_price) / TICK_SIZE;")
    lines.append("       if config.entry_max_spread_ticks > 0.0 && bbo_spread > config.entry_max_spread_ticks { skip; }")
    lines.append("       if config.entry_min_depth_lots > 0.0 && book_size < config.entry_min_depth_lots { skip; }")
    lines.append("  3. Add --entry-max-spread-ticks / --entry-min-depth-lots CLI args in fill_sim_main.rs")
    lines.append("  4. Re-run this sweep using those flags (true simulation) and validate against post-hoc results here")
    lines.append("=" * 90)

    return "\n".join(lines)


# ==============================================================================
# Main
# ==============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Predicate execution analysis starting")
    log.info(f"Cards: {list(CARDS.keys())}")
    log.info(f"Depth thresholds: {DEPTH_THRESHOLDS}")
    log.info(f"OOT range: {OOT_START} to {OOT_END}")

    dates = get_oot_dates()
    log.info(f"Found {len(dates)} OOT dates with MBO files")

    if not dates:
        log.error("No OOT dates found — check MBO_DIR path")
        sys.exit(1)

    # --- Step 1: Collect per-trade data ---
    log.info("Step 1: Running fill_sim for all (card, date) pairs ...")
    all_trades = collect_all_trades(dates)

    all_date_isos = [d.isoformat() for d in dates]

    # --- Steps 2-5: Analyse each card ---
    all_results = []
    for card_name in CARDS:
        trades = all_trades[card_name]
        log.info(f"\nAnalysing {card_name} ({len(trades)} total trades) ...")

        if not trades:
            log.warning(f"  {card_name}: no trades found — skipping")
            continue

        depth_res    = depth_predicate_sweep(card_name, trades, all_date_isos)
        spread_res   = spread_proxy_analysis(card_name, trades, all_date_isos)
        combined_res = combined_predicate_analysis(card_name, trades, all_date_isos)

        all_results.append({
            "card":         card_name,
            "total_trades": len(trades),
            "depth_sweep":  depth_res,
            "spread_proxy": spread_res,
            "combined":     combined_res,
        })

    # --- Step 6: Save results ---
    out_json = OUT_DIR / "predicate_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nResults saved to {out_json}")

    summary_text = print_summary_table(all_results)
    print(summary_text)

    out_txt = OUT_DIR / "predicate_analysis_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(summary_text)
    log.info(f"Summary table saved to {out_txt}")

    # Quick best-finding
    log.info("\n=== TOP FINDINGS ===")
    for res in all_results:
        card = res["card"]
        base_sharpe = res["depth_sweep"]["baseline"]["sharpe"]
        best_depth = max(
            res["depth_sweep"]["depth_sweep"].values(),
            key=lambda x: x["sharpe"],
        )
        best_combined = (
            res["combined"]["top_combos"][0]
            if res["combined"].get("top_combos")
            else {}
        )
        log.info(
            f"  {card}: baseline Sharpe={base_sharpe:.3f}  |  "
            f"best depth filter: {best_depth['label']} -> Sharpe={best_depth['sharpe']:.3f} "
            f"(Δ{best_depth['sharpe_delta_vs_baseline']:+.3f}, {best_depth['pct_trades_kept']:.0f}% trades kept)  |  "
            f"best combined: {best_combined.get('label','n/a')} -> Sharpe={best_combined.get('sharpe',0.0):.3f}"
        )

    log.info("\nDone.")


if __name__ == "__main__":
    main()
