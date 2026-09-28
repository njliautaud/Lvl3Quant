#!/usr/bin/env python3
"""
surviving_confluence_canonical_fifo.py — HC #469 compliance run.

Replaces the simplified-cost stability report with the CANONICAL FIFO market
replay on the 9 confluence configs that survived the HC #428 R1 regime gate
and the HC #344 day-conc gate (3 pairs + 6 triplets).

Cost model: per HC #74 + CLAUDE.md cost constants — FIFOReplayEngine with
real MBO queue position, partial fills, cancellations, real spread + depth.

Data scope: 16 days of v3.4.2 multi-head OOT predictions from
output/hc432_v342_47day_validation/fold_00_ep1_oot_inference_47day_hc432.npz.
The user wants 40+ days (HC #469 R2) — extending OOT inference to the
remaining ~24 dates is a separate GPU job queued behind Razer's current
meta-model training.

Bracket: TP=4 ticks, SL=3 ticks, hold=30s, cancel=10s, passive_at_touch.
This is a v0 reference bracket. Per HC #469 R4, time-based exits are
baselines — the adaptive-exit follow-up will replace this with the learned
stream-continuation exit policy.

Output:
  output/stream_backtest_v2/surviving_canonical_fifo_REPORT.md
  output/stream_backtest_v2/surviving_canonical_fifo_per_day.parquet
  output/stream_backtest_v2/surviving_canonical_fifo_fills.parquet
  output/stream_backtest_v2/surviving_canonical_fifo.DONE   (auto-followup marker)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))
sys.path.insert(0, str(LVL3 / "scripts"))
sys.path.insert(0, str(LVL3 / "scripts" / "v3_4_research"))

from stream_continuation_backtest import (
    directional_signal,
    NON_DIRECTIONAL_HEADS,
    ES_RT_COMMISSION_TICKS,
)
from hc432_fifo_full_market_replay import (
    map_signals_to_timestamps,
    run_one_date,
)

NPZ = LVL3 / "output" / "hc432_v342_47day_validation" / "fold_00_ep1_oot_inference_47day_hc432.npz"
OUT = LVL3 / "output" / "stream_backtest_v2"
OUT.mkdir(parents=True, exist_ok=True)

# Surviving configs (PASS day-conc <=0.70 AND regime-skew <=0.50 in stability report).
# Format: ("type", "name", [head_a, head_b, ...], conf_top_pct)
SURVIVING_CONFIGS = [
    # PAIRS that pass both gates
    ("pair", "pair01_logret1s+pup5s",       ["pred_log_ret_1s", "pred_p_up_5s"],            0.05),
    ("pair", "pair07_logret10s+logret60sq50", ["pred_log_ret_10s", "pred_log_ret_60s_q50"], 0.05),
    ("pair", "pair08_logret5s+pup5s",       ["pred_log_ret_5s", "pred_p_up_5s"],            0.05),
    # TRIPLETS that pass both gates
    ("triplet", "trip01_pup5s+logret1s+logret10s",        ["pred_p_up_5s", "pred_log_ret_1s", "pred_log_ret_10s"], 0.05),
    ("triplet", "trip03_logret5s+pup5s+logret1s",         ["pred_log_ret_5s", "pred_p_up_5s", "pred_log_ret_1s"], 0.05),
    ("triplet", "trip04_pup5s+pup10s+logret1s",           ["pred_p_up_5s", "pred_p_up_10s", "pred_log_ret_1s"], 0.05),
    ("triplet", "trip07_logret60s+logret30sq50+fifotp8sl5", ["pred_log_ret_60s", "pred_log_ret_30s_q50", "pred_fifo_tp8sl5_net"], 0.05),
    ("triplet", "trip09_logret60s+logret10sq50+fifotp8sl5", ["pred_log_ret_60s", "pred_log_ret_10s_q50", "pred_fifo_tp8sl5_net"], 0.05),
    ("triplet", "trip10_logret60s+logret30sq50+fifotp8sl5_top10", ["pred_log_ret_60s", "pred_log_ret_30s_q50", "pred_fifo_tp8sl5_net"], 0.10),
]

# Reference bracket per HC #469 R4 (this is the BASELINE; adaptive exits will replace it).
TP_TICKS = 4.0
SL_TICKS = 3.0
HOLD_S = 30.0
CANCEL_S = 10.0
ORDER_TYPE = "passive_at_touch"


def regime_classify(date_str: str) -> str:
    """Stub regime classification — uses cached labels if present, else 'unknown'.
    The 11:17 ET stability stratifier already classified the 32-day window;
    for the 16-day NPZ scope here we use a lookup from that cache.
    """
    cache = LVL3 / "output" / "stream_backtest_v2" / "top10_per_day_pair.parquet"
    if not cache.exists():
        return "unknown"
    df = pd.read_parquet(cache)
    if "date" not in df.columns or "regime" not in df.columns:
        return "unknown"
    df["date"] = df["date"].astype(str)
    row = df[df["date"] == date_str]
    if row.empty:
        return "unknown"
    return str(row["regime"].iloc[0])


def confluence_mask(arrs: Dict[str, np.ndarray], heads: List[str], conf_top_pct: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (long_mask, short_mask): boolean arrays of length n_total marking
    the events where ALL heads agree in direction AND each head is in its
    top-conf_top_pct of its signed magnitude (independently per head per direction).
    """
    n = arrs[heads[0]].shape[0]
    long_mask = np.ones(n, dtype=bool)
    short_mask = np.ones(n, dtype=bool)

    for h in heads:
        if h not in arrs:
            raise KeyError(f"NPZ missing head {h}")
        raw = arrs[h].astype(np.float64)
        signal = directional_signal(h, raw)

        if h in NON_DIRECTIONAL_HEADS:
            # Non-directional head used as confluence-only: requires |signal| in top conf_top_pct
            mag = np.abs(signal)
            finite = np.isfinite(mag)
            if finite.sum() == 0:
                return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
            thr = np.quantile(mag[finite], 1.0 - conf_top_pct)
            keep = finite & (mag >= thr)
            long_mask &= keep
            short_mask &= keep
        else:
            # Directional head: separate top quantile per side
            positive = signal > 0
            negative = signal < 0
            if positive.any():
                thr_pos = np.quantile(signal[positive], 1.0 - conf_top_pct)
                long_mask &= positive & (signal >= thr_pos)
            else:
                long_mask &= False
            if negative.any():
                thr_neg = np.quantile(-signal[negative], 1.0 - conf_top_pct)
                short_mask &= negative & (-signal >= thr_neg)
            else:
                short_mask &= False

    return long_mask, short_mask


def run_config(arrs: Dict[str, np.ndarray], cfg_type: str, cfg_name: str, heads: List[str], conf_top_pct: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    print(f"[{cfg_name}] computing confluence mask...")
    long_mask, short_mask = confluence_mask(arrs, heads, conf_top_pct)
    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())
    print(f"[{cfg_name}] long_triggers={n_long}, short_triggers={n_short}")

    sample_dates = arrs["sample_dates"]
    unique_dates = sorted(set(sample_dates.tolist()))
    all_fills: List[dict] = []
    per_day_rows: List[dict] = []

    for date_str in unique_dates:
        day_mask = sample_dates == date_str
        for side, mask in [("long", long_mask), ("short", short_mask)]:
            sel = day_mask & mask
            if not sel.any():
                continue
            idx_in_day = np.flatnonzero(sel[day_mask])  # local indices within this day
            strength = np.ones(idx_in_day.size, dtype=np.float64)
            fills = run_one_date(
                date_str=date_str,
                idx_in_day=idx_in_day,
                direction=side,
                strength=strength,
                tp_ticks=TP_TICKS,
                sl_ticks=SL_TICKS,
                hold_s=HOLD_S,
                cancel_s=CANCEL_S,
                order_type=ORDER_TYPE,
            )
            for f in fills:
                if "error" in f:
                    print(f"[{cfg_name}][{date_str}][{side}] ERROR: {f['error']}")
                    continue
                f["config"] = cfg_name
                f["config_type"] = cfg_type
                f["conf_top_pct"] = conf_top_pct
                all_fills.append(f)

    fills_df = pd.DataFrame(all_fills)
    if fills_df.empty:
        print(f"[{cfg_name}] NO FILLS")
        return pd.DataFrame(), pd.DataFrame()

    # Per-day aggregation: separately for the long-side + short-side AND combined
    for date_str in unique_dates:
        day_fills = fills_df[fills_df["date"] == date_str]
        if day_fills.empty:
            continue
        net = day_fills["net_ticks"].values
        per_day_rows.append({
            "config": cfg_name,
            "config_type": cfg_type,
            "date": date_str,
            "regime": regime_classify(date_str),
            "n_fills": int(len(net)),
            "net_ticks_total": float(net.sum()),
            "net_ticks_mean": float(net.mean()),
            "net_ticks_std": float(net.std(ddof=1)) if len(net) > 1 else 0.0,
            "Sharpe": float(net.mean() / net.std(ddof=1)) if len(net) > 1 and net.std(ddof=1) > 0 else 0.0,
            "WR": float((net > 0).mean()),
            "filled_pct": float((day_fills["fill_type"] != "no_fill").mean()) if "fill_type" in day_fills.columns else float("nan"),
        })

    per_day_df = pd.DataFrame(per_day_rows)
    elapsed = time.time() - t0
    print(f"[{cfg_name}] done in {elapsed:.1f}s — fills={len(fills_df)}, days_with_fills={len(per_day_df)}")
    return fills_df, per_day_df


def aggregate_summary(per_day_all: pd.DataFrame) -> pd.DataFrame:
    if per_day_all.empty:
        return pd.DataFrame()
    rows = []
    for cfg, sub in per_day_all.groupby("config"):
        net = sub["net_ticks_total"].values
        total = float(net.sum())
        day_conc = float(net.max() / net.sum()) if net.sum() > 0 else float("nan")

        # Regime split
        green = sub[sub["regime"] == "green"]["Sharpe"].mean() if (sub["regime"] == "green").any() else float("nan")
        red = sub[sub["regime"] == "red"]["Sharpe"].mean() if (sub["regime"] == "red").any() else float("nan")
        if not (np.isnan(green) or np.isnan(red)):
            mx = max(abs(green), abs(red))
            skew = abs(green - red) / mx if mx > 0 else 0.0
        else:
            skew = float("nan")

        # Across-day Sharpe (treating each day as one observation of total net)
        if len(net) > 1 and net.std(ddof=1) > 0:
            agg_sharpe = float(net.mean() / net.std(ddof=1))
        else:
            agg_sharpe = 0.0

        rows.append({
            "config": cfg,
            "config_type": sub["config_type"].iloc[0],
            "n_days_with_fills": int(len(sub)),
            "total_fills": int(sub["n_fills"].sum()),
            "net_ticks_total": total,
            "net_ticks_per_trade": float(total / sub["n_fills"].sum()) if sub["n_fills"].sum() > 0 else 0.0,
            "Sharpe_across_days": agg_sharpe,
            "WR_overall": float((per_day_all[per_day_all["config"] == cfg]["WR"] * per_day_all[per_day_all["config"] == cfg]["n_fills"]).sum()
                                 / per_day_all[per_day_all["config"] == cfg]["n_fills"].sum()) if sub["n_fills"].sum() > 0 else float("nan"),
            "day_conc": day_conc,
            "Sharpe_green": green,
            "Sharpe_red": red,
            "regime_skew": skew,
            "pass_dayconc": day_conc <= 0.70 if not np.isnan(day_conc) else False,
            "pass_regime": skew <= 0.50 if not np.isnan(skew) else False,
        })
    return pd.DataFrame(rows).sort_values("Sharpe_across_days", ascending=False).reset_index(drop=True)


def write_report(summary: pd.DataFrame, per_day_all: pd.DataFrame, wall_s: float) -> None:
    lines = []
    lines.append("# Surviving Configs — Canonical FIFO Market Replay")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M ET')}. Wall: {wall_s:.1f}s.")
    lines.append("")
    lines.append("**Compliance**: HC #469 R3 (canonical FIFO replay only), HC #74 (no midpoint).")
    lines.append("**Data scope**: 16 days of v3.4.2 multi-head OOT (the available NPZ).")
    lines.append("**Bracket**: TP=4t, SL=3t, hold=30s, cancel=10s, passive_at_touch (baseline per HC #469 R4).")
    lines.append("")
    lines.append("⚠️ **HC #469 R2 caveat**: user requires 40+ OOT days. Only 16 days exist in the multi-head NPZ.")
    lines.append("Extending OOT inference to the remaining ~24 dates is queued behind Razer's current meta-model training.")
    lines.append("")
    lines.append("## Summary (sorted by across-day Sharpe)")
    lines.append("")
    lines.append("| Config | Type | Days | Trades | Net (t) | Net/trade | Sharpe | WR | Day-conc | Sharpe_g | Sharpe_r | Skew | Gates |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        gates = "PASS" if (r["pass_dayconc"] and r["pass_regime"]) else "FAIL"
        lines.append(
            f"| {r['config']} | {r['config_type']} | {r['n_days_with_fills']} | "
            f"{r['total_fills']} | {r['net_ticks_total']:+.2f} | "
            f"{r['net_ticks_per_trade']:+.3f} | {r['Sharpe_across_days']:+.3f} | "
            f"{r['WR_overall']*100 if pd.notnull(r['WR_overall']) else 0:.1f}% | "
            f"{r['day_conc']:.2f} | "
            f"{r['Sharpe_green']:+.3f} | {r['Sharpe_red']:+.3f} | "
            f"{r['regime_skew']:.2f} | {gates} |"
        )
    lines.append("")
    lines.append("## Per-day breakdown")
    lines.append("")
    for cfg, sub in per_day_all.groupby("config"):
        lines.append(f"### {cfg}")
        lines.append("")
        lines.append("| Date | Regime | Trades | Net (t) | Net/trade | Sharpe | WR |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['date']} | {r['regime']} | {r['n_fills']} | "
                f"{r['net_ticks_total']:+.2f} | {r['net_ticks_mean']:+.3f} | "
                f"{r['Sharpe']:+.3f} | {r['WR']*100:.1f}% |"
            )
        lines.append("")

    report_path = OUT / "surviving_canonical_fifo_REPORT.md"
    report_path.write_text("\n".join(lines))
    print(f"[report] wrote {report_path}")


def main():
    t_start = time.time()
    print(f"[load] {NPZ}")
    arrs = {k: v for k, v in np.load(NPZ, allow_pickle=False).items()}
    print(f"[load] n_samples={arrs['sample_dates'].shape[0]:,} n_dates={len(set(arrs['sample_dates'].tolist()))}")

    all_fills: List[pd.DataFrame] = []
    all_per_day: List[pd.DataFrame] = []
    for cfg_type, cfg_name, heads, conf_top_pct in SURVIVING_CONFIGS:
        fills_df, per_day_df = run_config(arrs, cfg_type, cfg_name, heads, conf_top_pct)
        if not fills_df.empty:
            all_fills.append(fills_df)
        if not per_day_df.empty:
            all_per_day.append(per_day_df)

    if all_fills:
        fills_all = pd.concat(all_fills, ignore_index=True)
        fills_all.to_parquet(OUT / "surviving_canonical_fifo_fills.parquet", index=False)
        print(f"[write] {OUT}/surviving_canonical_fifo_fills.parquet n={len(fills_all)}")
    if all_per_day:
        per_day_all = pd.concat(all_per_day, ignore_index=True)
        per_day_all.to_parquet(OUT / "surviving_canonical_fifo_per_day.parquet", index=False)
        summary = aggregate_summary(per_day_all)
        summary.to_parquet(OUT / "surviving_canonical_fifo_summary.parquet", index=False)
        wall_s = time.time() - t_start
        write_report(summary, per_day_all, wall_s)

    # auto-followup marker for HC #469 R6(a)
    done_marker = OUT / "surviving_canonical_fifo.DONE"
    done_marker.write_text(f"completed: {time.strftime('%Y-%m-%d %H:%M:%S ET')}\nwall_s: {time.time() - t_start:.1f}\n")
    print(f"[DONE] {time.time() - t_start:.1f}s total")


if __name__ == "__main__":
    main()
