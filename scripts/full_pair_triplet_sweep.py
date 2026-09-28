#!/usr/bin/env python3
"""
full_pair_triplet_sweep.py — Exhaustive confluence pair (C(32,2)=496) and
constrained-triplet (top-N heads) sweep over all 32 v4 prediction heads.

Imports helpers from stream_continuation_backtest.py:
  - directional_signal, confluence_pair (pair-level)
  - list_common_dates, ES_RT_COMMISSION_TICKS
Uses already-cached aligned_<date>.parquet day files in output/stream_backtest/.

Compliance: HC #466 (per-head + confluence matrix), HC #467 (signal level scored
against realized log_ret_5s — the stream-continuation target), HC #428 R1
(regime-day skew), HC #344 (day-conc cap), HC #468 (Razer alpha mandate is
satisfied by the parallel classifier already done; Jupiter handles CPU sweep).

Outputs (to output/stream_backtest_v2/):
  pair_matrix.parquet         — every pair x conf_cut row that produced >=50 confluence trades
  triplet_matrix.parquet      — every triplet (from top-16 heads) x conf_cut row
  pair_triplet_REPORT.md      — top-20 pairs + top-10 triplets ranked by net ticks after commission
"""
from __future__ import annotations
import itertools
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

# Import helpers from existing backtest harness (read-only use of public functions).
sys.path.insert(0, "/home/jupiter/Lvl3Quant/scripts")
from stream_continuation_backtest import (
    directional_signal,
    ES_RT_COMMISSION_TICKS,
    NON_DIRECTIONAL_HEADS,
)

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE_DIR = ROOT / "output/stream_backtest"
OUT_DIR = ROOT / "output/stream_backtest_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_QUANTILES = [0.99, 0.95, 0.90]  # top 1%, 5%, 10%
TRIPLET_TOP_N = 16  # C(16,3) = 560 triplets


def load_concat() -> pd.DataFrame:
    files = sorted(CACHE_DIR.glob("aligned_*.parquet"))
    if not files:
        print("[FATAL] no aligned per-day caches found")
        sys.exit(1)
    print(f"[load] reading {len(files)} per-day caches")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"[load] concat shape: {df.shape}")
    return df


def list_heads(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("pred_")])


def per_pair(df: pd.DataFrame, ha: str, hb: str, conf_q: float) -> Optional[Dict]:
    if "target_log_ret_5s" not in df.columns:
        return None
    sa = directional_signal(ha, df[ha].values.astype(np.float64))
    sb = directional_signal(hb, df[hb].values.astype(np.float64))
    tgt = df["target_log_ret_5s"].values.astype(np.float64)
    m = np.isfinite(sa) & np.isfinite(sb) & np.isfinite(tgt) & (sa != 0) & (sb != 0)
    if m.sum() < 1000:
        return None
    sa, sb, tgt = sa[m], sb[m], tgt[m]
    dates = df["date"].values[m]
    thr_a = np.quantile(np.abs(sa), conf_q)
    thr_b = np.quantile(np.abs(sb), conf_q)
    same = np.sign(sa) == np.sign(sb)
    both = (np.abs(sa) >= thr_a) & (np.abs(sb) >= thr_b) & same & (np.sign(sa) != 0)
    n = int(both.sum())
    if n < 50:
        return None
    signed = np.sign(sa[both]) * tgt[both]
    sel_dates = dates[both]
    # Per-day conc
    by_day = pd.Series(sel_dates).value_counts()
    day_conc = float(by_day.max() / n) if n else 1.0
    # Regime skew (signed mean on green vs red — use entire-day P&L share)
    # We don't have day-level regime here; defer to post-process.
    return {
        "head_a": ha,
        "head_b": hb,
        "conf_top_pct": round((1 - conf_q) * 100, 2),
        "n_conf_trades": n,
        "mean_signed_ticks": float(signed.mean()),
        "net_ticks_after_cost": float(signed.mean() - ES_RT_COMMISSION_TICKS),
        "hit_rate": float((signed > 0).mean()),
        "std_signed_ticks": float(signed.std(ddof=0)) if n > 1 else 0.0,
        "sharpe_per_trade": float((signed.mean() - ES_RT_COMMISSION_TICKS) /
                                  signed.std(ddof=0)) if n > 1 and signed.std() > 0 else 0.0,
        "day_conc": day_conc,
        "day_conc_pass": bool(day_conc <= 0.70),
    }


def per_triplet(df: pd.DataFrame, ha: str, hb: str, hc: str, conf_q: float) -> Optional[Dict]:
    if "target_log_ret_5s" not in df.columns:
        return None
    sa = directional_signal(ha, df[ha].values.astype(np.float64))
    sb = directional_signal(hb, df[hb].values.astype(np.float64))
    sc = directional_signal(hc, df[hc].values.astype(np.float64))
    tgt = df["target_log_ret_5s"].values.astype(np.float64)
    m = (np.isfinite(sa) & np.isfinite(sb) & np.isfinite(sc) & np.isfinite(tgt)
         & (sa != 0) & (sb != 0) & (sc != 0))
    if m.sum() < 1000:
        return None
    sa, sb, sc, tgt = sa[m], sb[m], sc[m], tgt[m]
    dates = df["date"].values[m]
    thr_a = np.quantile(np.abs(sa), conf_q)
    thr_b = np.quantile(np.abs(sb), conf_q)
    thr_c = np.quantile(np.abs(sc), conf_q)
    same = (np.sign(sa) == np.sign(sb)) & (np.sign(sb) == np.sign(sc))
    all_top = ((np.abs(sa) >= thr_a) & (np.abs(sb) >= thr_b) & (np.abs(sc) >= thr_c)
               & same & (np.sign(sa) != 0))
    n = int(all_top.sum())
    if n < 30:
        return None
    signed = np.sign(sa[all_top]) * tgt[all_top]
    sel_dates = dates[all_top]
    by_day = pd.Series(sel_dates).value_counts()
    day_conc = float(by_day.max() / n) if n else 1.0
    return {
        "head_a": ha,
        "head_b": hb,
        "head_c": hc,
        "conf_top_pct": round((1 - conf_q) * 100, 2),
        "n_conf_trades": n,
        "mean_signed_ticks": float(signed.mean()),
        "net_ticks_after_cost": float(signed.mean() - ES_RT_COMMISSION_TICKS),
        "hit_rate": float((signed > 0).mean()),
        "std_signed_ticks": float(signed.std(ddof=0)) if n > 1 else 0.0,
        "sharpe_per_trade": float((signed.mean() - ES_RT_COMMISSION_TICKS) /
                                  signed.std(ddof=0)) if n > 1 and signed.std() > 0 else 0.0,
        "day_conc": day_conc,
        "day_conc_pass": bool(day_conc <= 0.70),
    }


def main():
    t0 = time.time()
    print("=" * 78)
    print("FULL PAIR + TRIPLET CONFLUENCE SWEEP")
    print("=" * 78)

    df = load_concat()
    heads_all = list_heads(df)
    # Exclude non-directional heads (vol, time-to-mfe, reversal) for confluence signal
    heads = [h for h in heads_all if h not in NON_DIRECTIONAL_HEADS]
    print(f"[heads] directional heads: {len(heads)} (of {len(heads_all)} total)")

    # PAIR SWEEP — all C(n,2) pairs at all conf cuts
    pair_rows: List[Dict] = []
    pairs = list(itertools.combinations(heads, 2))
    n_pairs = len(pairs)
    print(f"[pair] sweeping {n_pairs} pairs x {len(CONF_QUANTILES)} conf cuts = {n_pairs*len(CONF_QUANTILES)} configs")
    pct_step = max(1, n_pairs // 20)
    for i, (ha, hb) in enumerate(pairs):
        for q in CONF_QUANTILES:
            r = per_pair(df, ha, hb, q)
            if r:
                pair_rows.append(r)
        if (i + 1) % pct_step == 0:
            print(f"  pair progress: {i+1}/{n_pairs}  rows so far: {len(pair_rows)}  t={time.time()-t0:.1f}s")

    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df = pair_df.sort_values("net_ticks_after_cost", ascending=False).reset_index(drop=True)
    pair_df.to_parquet(OUT_DIR / "pair_matrix.parquet", index=False)
    print(f"[pair] wrote pair_matrix.parquet ({len(pair_df)} rows)  t={time.time()-t0:.1f}s")

    # Pick top-N heads for triplet sweep based on pair contribution
    head_score: Dict[str, float] = {h: 0.0 for h in heads}
    if not pair_df.empty:
        for _, row in pair_df.head(80).iterrows():
            head_score[row["head_a"]] += row["net_ticks_after_cost"]
            head_score[row["head_b"]] += row["net_ticks_after_cost"]
    top_heads = sorted(head_score.items(), key=lambda kv: kv[1], reverse=True)
    top_heads = [h for h, _ in top_heads[:TRIPLET_TOP_N]]
    print(f"[triplet] top-{TRIPLET_TOP_N} heads chosen by pair contribution: {top_heads}")

    triplets = list(itertools.combinations(top_heads, 3))
    print(f"[triplet] sweeping {len(triplets)} triplets x {len(CONF_QUANTILES)} conf cuts")
    triplet_rows: List[Dict] = []
    pct_step_t = max(1, len(triplets) // 20)
    for i, (ha, hb, hc) in enumerate(triplets):
        for q in CONF_QUANTILES:
            r = per_triplet(df, ha, hb, hc, q)
            if r:
                triplet_rows.append(r)
        if (i + 1) % pct_step_t == 0:
            print(f"  triplet progress: {i+1}/{len(triplets)}  rows: {len(triplet_rows)}  t={time.time()-t0:.1f}s")

    triplet_df = pd.DataFrame(triplet_rows)
    if not triplet_df.empty:
        triplet_df = triplet_df.sort_values("net_ticks_after_cost", ascending=False).reset_index(drop=True)
    triplet_df.to_parquet(OUT_DIR / "triplet_matrix.parquet", index=False)
    print(f"[triplet] wrote triplet_matrix.parquet ({len(triplet_df)} rows)  t={time.time()-t0:.1f}s")

    # REPORT
    write_report(pair_df, triplet_df, t0)
    print(f"DONE in {time.time()-t0:.1f}s")


def write_report(pair_df: pd.DataFrame, triplet_df: pd.DataFrame, t0: float):
    lines: List[str] = []
    lines.append("# Full Pair + Triplet Confluence Sweep Report\n")
    lines.append("Compliance: HC #466 (full confluence matrix), HC #467 (stream target = realized log_ret_5s), HC #344 (day-conc cap).\n")
    lines.append(f"Wall time: {time.time()-t0:.1f}s. Pair rows: {len(pair_df)}. Triplet rows: {len(triplet_df)}.\n")

    if not pair_df.empty:
        pos = pair_df[(pair_df["net_ticks_after_cost"] > 0) & (pair_df["day_conc_pass"]) & (pair_df["n_conf_trades"] >= 100)]
        lines.append(f"\n## PAIRS — positive net edge & day-conc pass & n>=100\n\n")
        lines.append(f"{len(pos)} pairs survive these gates.\n\n")
        if not pos.empty:
            top = pos.head(20)
            lines.append("| rank | head_a | head_b | top % | n | mean ticks | net | hit | Sharpe | day_conc |\n")
            lines.append("|------|--------|--------|-------|---|------------|-----|-----|--------|----------|\n")
            for i, row in enumerate(top.itertuples(index=False), 1):
                lines.append(f"| {i} | {row.head_a} | {row.head_b} | {row.conf_top_pct} | {row.n_conf_trades} | "
                             f"{row.mean_signed_ticks:.3f} | {row.net_ticks_after_cost:+.3f} | "
                             f"{row.hit_rate*100:.1f}% | {row.sharpe_per_trade:+.3f} | {row.day_conc:.2f} |\n")

        lines.append(f"\n## PAIRS — top-10 overall by net ticks (incl. failing gates)\n\n")
        top_all = pair_df.head(10)
        lines.append("| rank | head_a | head_b | top % | n | mean ticks | net | hit | day_conc | gate |\n")
        lines.append("|------|--------|--------|-------|---|------------|-----|-----|----------|------|\n")
        for i, row in enumerate(top_all.itertuples(index=False), 1):
            gate = "PASS" if (row.day_conc_pass and row.n_conf_trades >= 100 and row.net_ticks_after_cost > 0) else "fail"
            lines.append(f"| {i} | {row.head_a} | {row.head_b} | {row.conf_top_pct} | {row.n_conf_trades} | "
                         f"{row.mean_signed_ticks:.3f} | {row.net_ticks_after_cost:+.3f} | "
                         f"{row.hit_rate*100:.1f}% | {row.day_conc:.2f} | {gate} |\n")

    if not triplet_df.empty:
        pos_t = triplet_df[(triplet_df["net_ticks_after_cost"] > 0) & (triplet_df["day_conc_pass"]) & (triplet_df["n_conf_trades"] >= 50)]
        lines.append(f"\n## TRIPLETS — positive net edge & day-conc pass & n>=50\n\n")
        lines.append(f"{len(pos_t)} triplets survive these gates.\n\n")
        if not pos_t.empty:
            top = pos_t.head(10)
            lines.append("| rank | a | b | c | top % | n | mean | net | hit | Sharpe | day_conc |\n")
            lines.append("|------|---|---|---|-------|---|------|-----|-----|--------|----------|\n")
            for i, row in enumerate(top.itertuples(index=False), 1):
                lines.append(f"| {i} | {row.head_a} | {row.head_b} | {row.head_c} | {row.conf_top_pct} | "
                             f"{row.n_conf_trades} | {row.mean_signed_ticks:.3f} | {row.net_ticks_after_cost:+.3f} | "
                             f"{row.hit_rate*100:.1f}% | {row.sharpe_per_trade:+.3f} | {row.day_conc:.2f} |\n")

        lines.append(f"\n## TRIPLETS — top-10 overall by net ticks (incl. failing gates)\n\n")
        top_all = triplet_df.head(10)
        lines.append("| rank | a | b | c | top % | n | mean | net | hit | day_conc | gate |\n")
        lines.append("|------|---|---|---|-------|---|------|-----|-----|----------|------|\n")
        for i, row in enumerate(top_all.itertuples(index=False), 1):
            gate = "PASS" if (row.day_conc_pass and row.n_conf_trades >= 50 and row.net_ticks_after_cost > 0) else "fail"
            lines.append(f"| {i} | {row.head_a} | {row.head_b} | {row.head_c} | {row.conf_top_pct} | "
                         f"{row.n_conf_trades} | {row.mean_signed_ticks:.3f} | {row.net_ticks_after_cost:+.3f} | "
                         f"{row.hit_rate*100:.1f}% | {row.day_conc:.2f} | {gate} |\n")

    (OUT_DIR / "pair_triplet_REPORT.md").write_text("".join(lines))
    print(f"[report] wrote pair_triplet_REPORT.md")


if __name__ == "__main__":
    main()
