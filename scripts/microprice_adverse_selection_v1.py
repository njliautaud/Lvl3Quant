#!/usr/bin/env python3
"""
microprice_adverse_selection_v1.py — HC #488 creativity-mandate axis #1.

Hypothesis: Filter passive-limit fills on microprice pressure alignment.
Microprice = (bid*ask_size + ask*bid_size) / (bid_size + ask_size). It is a
known leading indicator of next-tick mid drift. If at signal time the
microprice already drifts AGAINST the intended trade direction, the limit
will likely be filled adversely → SKIP.

Pipeline:
  1. Load fills from output/hc475_ab/symmetric_gate_fills.parquet
  2. For each OOT date, load preprocessed L1 (bid/ask price + size) from
     data/processed/mbo_book_features/{date}_book_features.npz
  3. At each fill's entry_time, snapshot microprice and queue imbalance.
     Also snapshot at entry_time - 1s and entry_time - 5s to get drift.
  4. Build per-fill feature parquet.
  5. Sweep (config × alignment × drift_threshold) and score net_ticks,
     Sharpe, WR, PF, day-coverage. Apply HC #428 gates.

Acceptance gates (HC #428):
  - net > +0.10 t/trade
  - Sharpe > 0.3
  - PF > 1.1
  - Profitable on >= 60% of OOT days

NOTE on price encoding: `mbo_book_features` stores prices in ticks relative
to a session-anchor. Since microprice and drift are differences (or weighted
combinations with sizes), the anchor cancels out — we only need consistent
relative units within a single day. The anchor does NOT change intraday.

Causality: snapshot at (entry - 50 ms) to ensure features use only data
strictly before the fill execution. No lookahead.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO = Path("/home/jupiter/Lvl3Quant")
FILLS_PARQUET = REPO / "output/hc475_ab/symmetric_gate_fills.parquet"
BOOK_DIR = REPO / "data/processed/mbo_book_features"
OUT_DIR = REPO / "output/microprice_adverse_selection_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_SIZE = 0.25
COMMISSION_TICKS = 0.376  # HC #428 R2 passive-limit cost (already in fills.net_ticks)

# Snapshot lookback windows (causal)
DRIFT_LOOKBACK_NS = {
    "1s": 1_000_000_000,
    "5s": 5_000_000_000,
}
PRE_ENTRY_BUFFER_NS = 50_000_000  # 50 ms causal buffer

# Sweep grid
ALIGNMENT_BUCKETS = ["aligned", "against", "neutral"]
DRIFT_THRESHOLDS_TICKS = [0.0, 0.05, 0.10, 0.20, 0.50]
QUEUE_IMB_THRESHOLDS = [0.10, 0.30, 0.50]

# HC #428 gates
GATE_NET_TICKS = 0.10
GATE_SHARPE = 0.30
GATE_PF = 1.10
GATE_DAY_COVERAGE = 0.60
GATE_MIN_TRADES = 50


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -------------------------------------------------------------------------
# Step 1: Load fills
# -------------------------------------------------------------------------
def load_fills() -> pd.DataFrame:
    df = pd.read_parquet(FILLS_PARQUET)
    log(f"loaded fills: {df.shape}, configs={df['config'].nunique()}, dates={df['date'].nunique()}")

    # The fills parquet's order_type values
    log(f"  order_type values: {df['order_type'].value_counts().to_dict()}")
    log(f"  fill_type values: {df['fill_type'].value_counts().to_dict()}")

    # Filter to passive-limit fills if such a label exists.
    # Inspecting the HC475 spec: order_type is typically 'limit' for passive.
    if "order_type" in df.columns:
        n_before = len(df)
        mask = df["order_type"].astype(str).str.lower().str.contains("limit", na=False)
        if mask.sum() > 0:
            df = df[mask].copy()
            log(f"  filtered to passive-limit by order_type: {len(df)} / {n_before}")
    return df


# -------------------------------------------------------------------------
# Step 2: Load L1 + compute microprice
# -------------------------------------------------------------------------
def load_day_l1(date_yyyymmdd: str) -> pd.DataFrame:
    path = BOOK_DIR / f"{date_yyyymmdd}_book_features.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True)
    feat = d["features"]  # (N, 30) float32
    ts = d["timestamps"]  # (N,) int64

    # feature indices (from feature_names):
    #   0: bid_price_1 (ticks, relative to session anchor)
    #   5: ask_price_1 (ticks, relative)
    #   10: bid_size_1
    #   15: ask_size_1
    bid = feat[:, 0].astype(np.float64)
    ask = feat[:, 5].astype(np.float64)
    bsz = feat[:, 10].astype(np.float64)
    asz = feat[:, 15].astype(np.float64)

    # Filter to rows with valid two-sided L1 (non-zero sizes, ask > bid)
    valid = (bsz > 0) & (asz > 0) & (ask > bid)
    if not valid.any():
        raise ValueError(f"no valid L1 rows on {date_yyyymmdd}")

    ts_v = ts[valid]
    bid = bid[valid]
    ask = ask[valid]
    bsz = bsz[valid]
    asz = asz[valid]

    total = bsz + asz
    mp = (bid * asz + ask * bsz) / total  # ticks
    mid = (bid + ask) / 2.0
    imb = (bsz - asz) / total

    df = pd.DataFrame({
        "ts_ns": ts_v,
        "bid_ticks": bid,
        "ask_ticks": ask,
        "bid_size": bsz,
        "ask_size": asz,
        "microprice_ticks": mp,
        "mid_ticks": mid,
        "queue_imb": imb,
    })
    # ensure monotonic by ts
    if not (np.diff(df["ts_ns"].values) >= 0).all():
        df.sort_values("ts_ns", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


# -------------------------------------------------------------------------
# Step 3: Snapshot features at fill entry times
# -------------------------------------------------------------------------
def compute_features_for_date(fills_day: pd.DataFrame, l1: pd.DataFrame) -> pd.DataFrame:
    tob_ts = l1["ts_ns"].values
    mp = l1["microprice_ticks"].values
    mid = l1["mid_ticks"].values
    imb = l1["queue_imb"].values

    entry_ns = fills_day["ts_entry_ns"].values.astype(np.int64)
    t_entry = entry_ns - PRE_ENTRY_BUFFER_NS
    t_1s = entry_ns - DRIFT_LOOKBACK_NS["1s"]
    t_5s = entry_ns - DRIFT_LOOKBACK_NS["5s"]

    def snapshot_at(times: np.ndarray):
        # last observation strictly before `times`
        idx = np.searchsorted(tob_ts, times, side="right") - 1
        idx_clip = np.clip(idx, 0, len(tob_ts) - 1)
        valid = idx >= 0  # True if there is at least one earlier obs
        return mp[idx_clip], mid[idx_clip], imb[idx_clip], valid

    mp_e, mid_e, imb_e, v_e = snapshot_at(t_entry)
    mp_1s, _, _, v_1 = snapshot_at(t_1s)
    mp_5s, _, _, v_5 = snapshot_at(t_5s)

    drift_1s = mp_e - mp_1s
    drift_5s = mp_e - mp_5s

    out = fills_day.copy()
    out["microprice_entry"] = mp_e
    out["mid_entry"] = mid_e
    out["queue_imb_at_entry"] = imb_e
    out["microprice_drift_1s"] = drift_1s
    out["microprice_drift_5s"] = drift_5s
    dir_sign = np.where(out["direction"].values == "long", 1, -1)
    out["microprice_pressure_aligned"] = (np.sign(drift_1s) * dir_sign) > 0
    # Track validity of snapshots — drop rows where 5s lookback is before session start
    out["snapshot_valid"] = v_e & v_1 & v_5
    return out


# -------------------------------------------------------------------------
# Step 4: Sweep + score
# -------------------------------------------------------------------------
def sharpe(net: np.ndarray) -> float:
    if len(net) < 5:
        return 0.0
    s = np.std(net, ddof=1)
    if s <= 0:
        return 0.0
    return float(np.sqrt(252) * np.mean(net) / s)


def profit_factor(net: np.ndarray) -> float:
    pos = float(net[net > 0].sum())
    neg = float(-net[net < 0].sum())
    if neg <= 0:
        return float("inf") if pos > 0 else 0.0
    return pos / neg


def day_coverage(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    by_day = df.groupby("date")["net_ticks"].mean()
    if len(by_day) == 0:
        return 0.0
    return float((by_day > 0).mean())


def score_cell(df: pd.DataFrame) -> Dict:
    if len(df) == 0:
        return dict(n_trades=0, net_ticks_per_trade=0.0, sharpe=0.0, wr=0.0,
                    pf=0.0, day_coverage=0.0, n_days=0)
    net = df["net_ticks"].values.astype(np.float64)
    return dict(
        n_trades=int(len(net)),
        net_ticks_per_trade=float(np.mean(net)),
        sharpe=sharpe(net),
        wr=float((net > 0).mean()),
        pf=profit_factor(net),
        day_coverage=day_coverage(df),
        n_days=int(df["date"].nunique()),
    )


def sweep(feats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cfg in feats["config"].unique().tolist():
        sub = feats[feats["config"] == cfg]
        dir_sign = np.where(sub["direction"].values == "long", 1, -1)

        # Baseline (no filter)
        base = score_cell(sub)
        base.update(config=cfg, alignment="any", drift_thresh_ticks=0.0,
                    filter_type="baseline", lookback="-")
        rows.append(base)

        # Microprice drift filters
        for lookback in ["1s", "5s"]:
            drift_col = f"microprice_drift_{lookback}"
            drift = sub[drift_col].values
            drift_signed = drift * dir_sign  # >0 means microprice moving WITH trade direction

            for align in ALIGNMENT_BUCKETS:
                for thr in DRIFT_THRESHOLDS_TICKS:
                    if align == "aligned":
                        mask = drift_signed >= thr
                    elif align == "against":
                        mask = drift_signed <= -thr
                    else:  # neutral
                        mask = np.abs(drift_signed) < max(thr, 1e-6)
                    cell = score_cell(sub[mask])
                    cell.update(
                        config=cfg, alignment=align, drift_thresh_ticks=thr,
                        filter_type=f"drift_{lookback}", lookback=lookback,
                    )
                    rows.append(cell)

        # Queue imbalance filter (at entry)
        imb_signed = sub["queue_imb_at_entry"].values * dir_sign
        for align in ["aligned", "against"]:
            for thr in QUEUE_IMB_THRESHOLDS:
                if align == "aligned":
                    mask = imb_signed >= thr
                else:
                    mask = imb_signed <= -thr
                cell = score_cell(sub[mask])
                cell.update(
                    config=cfg, alignment=align, drift_thresh_ticks=thr,
                    filter_type="queue_imb", lookback="-",
                )
                rows.append(cell)

        # Combined: drift_1s aligned AND queue_imb aligned
        drift1 = sub["microprice_drift_1s"].values * dir_sign
        for d_thr in [0.0, 0.05, 0.10]:
            for q_thr in [0.10, 0.30]:
                mask = (drift1 >= d_thr) & (imb_signed >= q_thr)
                cell = score_cell(sub[mask])
                cell.update(
                    config=cfg, alignment=f"drift>={d_thr}&imb>={q_thr}",
                    drift_thresh_ticks=d_thr, filter_type="combo", lookback="1s",
                )
                rows.append(cell)

    return pd.DataFrame(rows)


def apply_gates(sw: pd.DataFrame) -> pd.DataFrame:
    g = (
        (sw["net_ticks_per_trade"] > GATE_NET_TICKS)
        & (sw["sharpe"] > GATE_SHARPE)
        & (sw["pf"] > GATE_PF)
        & (sw["day_coverage"] >= GATE_DAY_COVERAGE)
        & (sw["n_trades"] >= GATE_MIN_TRADES)
    )
    return sw[g].copy()


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    log("=== microprice_adverse_selection_v1 ===")

    fills = load_fills()
    dates = sorted(fills["date"].unique().tolist())
    log(f"OOT dates: {len(dates)}")

    per_day_feats: List[pd.DataFrame] = []
    failures: List[Dict] = []
    for d in dates:
        t_d = time.time()
        try:
            l1 = load_day_l1(d)
            day_fills = fills[fills["date"] == d].copy()
            feats = compute_features_for_date(day_fills, l1)
            per_day_feats.append(feats)
            log(f"  {d}: L1 rows={len(l1):,}, fills={len(feats)}, dt={time.time()-t_d:.1f}s")
            del l1
        except Exception as e:
            log(f"  {d}: FAILED — {type(e).__name__}: {e}")
            failures.append({"date": d, "error": f"{type(e).__name__}: {e}"})

    if not per_day_feats:
        log("FATAL: no days processed")
        return 1

    feats_all = pd.concat(per_day_feats, ignore_index=True)

    # Drop rows with invalid snapshots (e.g. lookback before session start)
    n_pre = len(feats_all)
    feats_all = feats_all[feats_all["snapshot_valid"]].copy()
    log(f"valid feature rows: {len(feats_all)} / {n_pre}")

    feats_out = OUT_DIR / "microprice_features.parquet"
    feats_all.to_parquet(feats_out, index=False)
    log(f"wrote features: {feats_out}")

    # Quick sanity diagnostics
    log("--- diagnostics ---")
    log(f"  microprice_drift_1s mean={feats_all['microprice_drift_1s'].mean():.4f} "
        f"std={feats_all['microprice_drift_1s'].std():.4f}")
    log(f"  microprice_drift_5s mean={feats_all['microprice_drift_5s'].mean():.4f} "
        f"std={feats_all['microprice_drift_5s'].std():.4f}")
    log(f"  pressure_aligned rate: {feats_all['microprice_pressure_aligned'].mean():.3f}")
    log(f"  baseline net_ticks/trade: {feats_all['net_ticks'].mean():+.3f}")

    # Sweep
    log("sweeping...")
    sw = sweep(feats_all)
    sw_out = OUT_DIR / "filter_sweep_summary.csv"
    sw.to_csv(sw_out, index=False)
    log(f"wrote sweep: {sw_out} ({len(sw)} cells)")

    winners = apply_gates(sw)
    win_path = OUT_DIR / "winning_cells.txt"
    if len(winners):
        winners_sorted = winners.sort_values("sharpe", ascending=False)
        with open(win_path, "w") as f:
            f.write(winners_sorted.to_string(index=False))
        log(f"WINNERS: {len(winners)}")
    else:
        with open(win_path, "w") as f:
            f.write("NO CELLS PASSED HC #428 GATES\n")
        log("no winning cells")

    # Closest-miss analysis
    cand = sw[sw["n_trades"] >= GATE_MIN_TRADES].copy()
    cand["gate_score"] = (
        (cand["net_ticks_per_trade"] / GATE_NET_TICKS).clip(upper=2)
        + (cand["sharpe"] / GATE_SHARPE).clip(upper=2)
        + (cand["pf"] / GATE_PF).clip(upper=2)
        + (cand["day_coverage"] / GATE_DAY_COVERAGE).clip(upper=2)
    )
    top_misses = cand.sort_values("gate_score", ascending=False).head(10)

    verdict = "ACCEPT" if len(winners) > 0 else "REJECT"
    baseline = sw[sw["filter_type"] == "baseline"].copy()

    # ---- REPORT ----
    R = []
    R.append("# Microprice Adverse Selection v1 — REPORT")
    R.append("")
    R.append(f"**Verdict: {verdict}**")
    R.append("")
    R.append(f"- elapsed: {time.time()-t0:.1f}s")
    R.append(f"- OOT dates processed: {len(per_day_feats)} / {len(dates)}")
    R.append(f"- failures: {len(failures)}")
    for f in failures:
        R.append(f"  - {f['date']}: {f['error']}")
    R.append(f"- fills with valid microprice features: {len(feats_all)}")
    R.append(f"- baseline net_ticks/trade (no filter, pooled): {feats_all['net_ticks'].mean():+.3f}")
    R.append(f"- baseline pressure-aligned rate: {feats_all['microprice_pressure_aligned'].mean():.3f}")
    R.append(f"- total sweep cells: {len(sw)}")
    R.append(f"- winning cells (HC #428 gates): {len(winners)}")
    R.append("")
    R.append("## HC #428 gates")
    R.append(f"- net_ticks > +{GATE_NET_TICKS}")
    R.append(f"- Sharpe > {GATE_SHARPE}")
    R.append(f"- PF > {GATE_PF}")
    R.append(f"- day_coverage >= {GATE_DAY_COVERAGE*100:.0f}% of OOT days profitable")
    R.append(f"- n_trades >= {GATE_MIN_TRADES}")
    R.append("")
    R.append("## Baseline (no filter) per config")
    R.append("```")
    R.append(baseline[["config","n_trades","net_ticks_per_trade","sharpe","wr","pf","day_coverage"]]
             .to_string(index=False))
    R.append("```")
    R.append("")
    if len(winners) > 0:
        R.append("## Winning cells (top 10 by Sharpe)")
        R.append("```")
        cols = ["config","filter_type","alignment","drift_thresh_ticks","n_trades",
                "net_ticks_per_trade","sharpe","wr","pf","day_coverage","n_days"]
        R.append(winners.sort_values("sharpe", ascending=False).head(10)[cols].to_string(index=False))
        R.append("```")
        R.append("")
        best = winners.sort_values("sharpe", ascending=False).iloc[0]
        R.append("### Best cell")
        R.append(f"- config: `{best['config']}`")
        R.append(f"- filter: `{best['filter_type']}` / alignment=`{best['alignment']}` / thr={best['drift_thresh_ticks']}")
        R.append(f"- n_trades: {int(best['n_trades'])}, n_days: {int(best['n_days'])}")
        R.append(f"- net_ticks/trade: **{best['net_ticks_per_trade']:+.3f}**")
        R.append(f"- Sharpe: **{best['sharpe']:.3f}**, WR: {best['wr']:.3f}, PF: {best['pf']:.3f}")
        R.append(f"- day_coverage: {best['day_coverage']:.2f}")
    else:
        R.append("## Closest misses (top 10 by composite gate-score)")
        R.append("```")
        cols = ["config","filter_type","alignment","drift_thresh_ticks","n_trades",
                "net_ticks_per_trade","sharpe","wr","pf","day_coverage","gate_score"]
        R.append(top_misses[cols].to_string(index=False))
        R.append("```")
        R.append("")
    R.append("")
    R.append("## Interpretation")
    R.append("")
    # Per-config baseline-vs-best-filter comparison
    for cfg in feats_all["config"].unique():
        b = baseline[baseline["config"] == cfg]
        if len(b) == 0:
            continue
        b_net = b["net_ticks_per_trade"].iloc[0]
        b_sh = b["sharpe"].iloc[0]
        nb = sw[(sw["config"] == cfg) & (sw["filter_type"] != "baseline") & (sw["n_trades"] >= GATE_MIN_TRADES)]
        if len(nb) == 0:
            continue
        best_cfg = nb.sort_values("sharpe", ascending=False).iloc[0]
        R.append(
            f"- **{cfg}**: baseline Sharpe={b_sh:.2f}/net={b_net:+.3f}t → "
            f"best filtered Sharpe={best_cfg['sharpe']:.2f}/net={best_cfg['net_ticks_per_trade']:+.3f}t "
            f"({best_cfg['filter_type']}/{best_cfg['alignment']}/thr={best_cfg['drift_thresh_ticks']}, "
            f"n={int(best_cfg['n_trades'])}, days={int(best_cfg['n_days'])}, "
            f"day_cov={best_cfg['day_coverage']:.2f})"
        )
    R.append("")
    R.append("## What to try next if REJECT")
    R.append("")
    R.append("- Combine microprice drift with regime classifier (green/red day per HC #428 R1)")
    R.append("- Trade-side asymmetry: short signals had better historical edge — restrict filter to short side only")
    R.append("- Replace static threshold with rolling-percentile (e.g. drift >= p70 of last 1000 entries)")
    R.append("- Try shorter lookback (250ms-500ms) — signal predictive horizon is sub-second")
    R.append("- Stack: microprice + signed-volume + spread state — 3-feature gate")
    R.append("- Re-examine fillsim — if adverse selection comes from queue wait (not signal), filter may need queue-ahead bucket")
    R.append("")
    R.append("## Methodology notes & caveats")
    R.append("")
    R.append("- L1 from `data/processed/mbo_book_features/{date}_book_features.npz` (raw bid/ask price + size, 30-feature schema).")
    R.append("- Prices encoded as ticks RELATIVE to session anchor; microprice and drift are differences so anchor cancels.")
    R.append("- Microprice snapshot taken at (entry - 50 ms): strictly causal.")
    R.append("- 5s lookback rows that pre-date session start are dropped.")
    R.append("- Pooled OOT analysis only (v1 spec). Walk-forward deferred.")
    R.append(f"- Cost {COMMISSION_TICKS} t passive-limit already in fills.net_ticks.")
    R.append("- Sharpe annualized at sqrt(252) on per-trade returns (so it's a trade-Sharpe, not a daily-Sharpe — use for cell-comparison only).")
    R.append("")

    report_path = OUT_DIR / "REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(R))
    log(f"wrote report: {report_path}")

    regen = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_days_evaluated": int(len(per_day_feats)),
        "n_cells": int(len(sw)),
        "n_winning_cells": int(len(winners)),
        "elapsed_sec": round(time.time() - t0, 1),
        "script": "scripts/microprice_adverse_selection_v1.py",
        "verdict": verdict,
    }
    with open(OUT_DIR / ".regen_complete.json", "w") as f:
        json.dump(regen, f, indent=2)
    log(f"=== DONE in {time.time()-t0:.1f}s — verdict={verdict} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
